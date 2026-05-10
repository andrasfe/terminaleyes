"""OcrAgent — extract text from the screen using an OCR-specialized
vision model (default: ``nanonets-ocr-s`` on LM Studio).

Tier-1 atomic primitive. Sister of :class:`ReadAgent`:

  * **ReadAgent** — multimodal LLM Q&A over the screen.
  * **OcrAgent** — pure OCR. Pass an image through an OCR-trained
    vision model and get back verbatim text + legibility signals.
    Never interprets intent — never decides which line "answers" a
    question. That's the controller / a higher-tier agent's job.

The model is taken from ``ctx.ocr_model`` (set from
``settings.commander.lmstudio_ocr_model``, default
``nanonets-ocr-s``). The same OpenAI-compatible client used for
``vision_model`` is reused; no new endpoint.

Inputs:

  * ``region`` — explicit named region from :data:`REGION_PRESETS`.
  * ``crop`` — explicit ``(x0, y0, x1, y1)`` fractions when no
    preset fits.
  * ``target`` — natural-language hint used **only** to auto-pick a
    sensible region preset (e.g. "the URL bar" → ``url_bar``). The
    text content is NOT filtered against it.
  * ``image`` — pre-captured frame; otherwise the agent captures.

Outputs (:class:`OcrOutcome`):

  * ``data['text']`` — joined plain text the model returned.
  * ``data['lines']`` — non-empty stripped lines, in OCR order.
  * ``data['region']`` — final region label used.
  * ``data['legibility']`` — ``{low_confidence, sparse,
    edge_clipped}`` flags so the caller can decide whether the
    text is trustworthy or possibly cut off.

Failure ``reason`` codes (semantic, small set):

  * ``ocr unavailable: <why>`` — no vision client / no model name.
  * ``no capture`` — no capture device available in context.
  * ``capture failed: <e>`` — webcam read errored.
  * ``frame too dark`` — mean brightness below threshold.
  * ``no readable text in region <X>`` — model returned nothing.
  * ``model call failed: <e>`` — request to the OCR model errored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.utils.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)

logger = logging.getLogger(__name__)


REGION_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "full":     (0.00, 0.00, 1.00, 1.00),
    "top_bar":  (0.00, 0.00, 1.00, 0.06),
    "title":    (0.00, 0.00, 1.00, 0.05),
    "url_bar":  (0.00, 0.00, 1.00, 0.10),
    "browser_chrome": (0.00, 0.00, 1.00, 0.18),
    "header":   (0.00, 0.00, 1.00, 0.20),
    "footer":   (0.00, 0.92, 1.00, 1.00),
    "left":     (0.00, 0.00, 0.30, 1.00),
    "right":    (0.70, 0.00, 1.00, 1.00),
    "center":   (0.20, 0.15, 0.80, 0.85),
    "body":     (0.00, 0.10, 1.00, 0.92),
}


# Heuristic mapping from natural-language target hints to a region
# preset. Used for region selection ONLY — never for filtering the
# returned text. Order matters; first match wins.
TARGET_REGION_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(url|address)\s*bar\b", re.I), "url_bar"),
    (re.compile(r"\burl\b|\baddress\b|\bweb\s*address\b", re.I),
     "url_bar"),
    (re.compile(r"\btab(s)?\s*strip\b|\btab\s*bar\b", re.I),
     "browser_chrome"),
    (re.compile(r"\b(top|menu)\s*bar\b|\bmenubar\b", re.I), "top_bar"),
    (re.compile(r"\b(window\s+)?title\b", re.I), "title"),
    (re.compile(r"\b(footer|status\s*bar|taskbar)\b", re.I), "footer"),
    (re.compile(r"\b(sidebar|left\s*pane)\b", re.I), "left"),
    (re.compile(r"\b(right\s*pane)\b", re.I), "right"),
    (re.compile(r"\b(header|page\s*header)\b", re.I), "header"),
    (re.compile(r"\b(body|main\s*content|page\s*content)\b", re.I),
     "body"),
]


# Word/char thresholds for legibility signals.
SPARSE_WORD_THRESHOLD = 3
LOW_CHAR_THRESHOLD = 8
# Marker the model is asked to use for unreadable text. Counted in
# the response to set the ``low_confidence`` flag.
ILLEGIBLE_MARKER = "[?]"

OCR_PROMPT = (
    "Extract every visible piece of text from this image, verbatim, "
    "preserving line breaks. Output rules:\n"
    "  * Output text only — no commentary, no markdown fences, no "
    "headers, no JSON.\n"
    "  * Preserve the order text appears on screen, top to bottom, "
    "left to right.\n"
    "  * Preserve line breaks where they exist visually.\n"
    f"  * If a word is unreadable, blurry, or cut off, replace it "
    f"with the literal token {ILLEGIBLE_MARKER} — don't guess.\n"
    "  * If the image contains no readable text at all, reply with "
    "exactly: NOTEXT"
)


@dataclass
class OcrOutcome(Outcome):
    """``data['text']`` carries the extracted plain text on success."""


class OcrAgent(Agent):
    """Read text from the screen via an OCR-specialised vision model.

    Sends the (cropped) frame to the model named in
    ``ctx.ocr_model`` (falling back to ``ctx.vision_model``) and
    returns the verbatim text plus a small set of legibility flags.
    """

    name = "ocr"

    async def run(
        self,
        *,
        target: str | None = None,
        region: str | None = None,
        crop: tuple[float, float, float, float] | None = None,
        image: np.ndarray | None = None,
        darkness_threshold: float = 0.04,
        max_tokens: int = 1500,
        record_label: str = "ocr",
    ) -> OcrOutcome:
        if self.ctx.vision_client is None:
            return OcrOutcome(
                success=False,
                reason="ocr unavailable: no vision client in context",
                data=_empty_data(),
            )
        model_name = self.ctx.ocr_model or self.ctx.vision_model
        if not model_name:
            return OcrOutcome(
                success=False,
                reason="ocr unavailable: no ocr_model / vision_model set",
                data=_empty_data(),
            )

        if image is None:
            if self.ctx.capture is None:
                return OcrOutcome(
                    success=False, reason="no capture in context",
                    data=_empty_data(),
                )
            try:
                frame = await self.ctx.capture.capture_frame()
                image = frame.image
            except Exception as e:
                return OcrOutcome(
                    success=False, reason=f"capture failed: {e}",
                    data=_empty_data(),
                )
            self.ctx.record_frame(image, label=record_label)

        # Frame-too-dark guard — common when the target machine is
        # asleep or the screen is locked.
        try:
            mean_norm = float(image.mean()) / 255.0
        except Exception:
            mean_norm = 1.0
        if mean_norm < darkness_threshold:
            return OcrOutcome(
                success=False,
                reason=(
                    "frame too dark — target screen may be asleep "
                    f"(mean brightness {mean_norm:.3f})"
                ),
                data={**_empty_data(), "brightness": mean_norm},
            )

        region_label, crop_xywh = self._pick_region(
            target=target, region=region, crop=crop,
        )
        h, w = image.shape[:2]
        x0 = max(0, int(crop_xywh[0] * w))
        y0 = max(0, int(crop_xywh[1] * h))
        x1 = min(w, int(crop_xywh[2] * w))
        y1 = min(h, int(crop_xywh[3] * h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return OcrOutcome(
                success=False,
                reason=f"crop {region_label!r} is empty",
                data=_empty_data(region=region_label),
            )
        region_img = image[y0:y1, x0:x1]
        # Save the crop so the UI can replay what got fed to the
        # model — useful when callers ask why OCR failed.
        self.ctx.record_frame(
            region_img, label=f"{record_label}_region_{region_label}",
        )

        # ── Call the OCR model ────────────────────────────────────
        try:
            b64 = numpy_to_base64_png(
                resize_for_mllm(
                    enhance_for_screen(region_img),
                    max_dimension=1280, min_dimension=512,
                )
            )
        except Exception as e:
            return OcrOutcome(
                success=False,
                reason=f"image encoding failed: {e}",
                data=_empty_data(region=region_label),
            )

        messages = [
            {"role": "system", "content": OCR_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "Transcribe."},
                ],
            },
        ]
        try:
            resp = await self.ctx.vision_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            return OcrOutcome(
                success=False,
                reason=f"model call failed: {e}",
                data=_empty_data(region=region_label),
            )

        raw = self._best_text_from_response(resp) or ""
        text = raw.strip()
        # Strip a markdown fence if the model added one despite the
        # rules.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        # nanonets-ocr-s is a document-OCR model that wraps its
        # output in HTML-like structural tags (``<img>image
        # caption</img>``, ``<watermark>UI hint</watermark>``,
        # hallucinated ``<div style=...>`` blocks). Reduce to plain
        # text — keep meaningful content, drop image captions and
        # tag noise.
        text = self._strip_structural_tags(text)

        if not text or text.upper().startswith("NOTEXT"):
            return OcrOutcome(
                success=False,
                reason=f"no readable text in region {region_label!r}",
                data=_empty_data(region=region_label),
            )

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        legibility = self._legibility_flags(text=text, lines=lines)
        flags_str = (
            ",".join(k for k, v in legibility.items() if v) or "clean"
        )
        first_line = lines[0] if lines else ""
        print(
            f"   OcrAgent: model={model_name} region={region_label} "
            f"chars={len(text)} lines={len(lines)} flags={flags_str}"
        )
        if first_line:
            print(f"     first line: {first_line[:160]}")

        return OcrOutcome(
            success=True,
            reason=(
                f"ocr ok ({len(lines)} line(s), flags={flags_str})"
            ),
            data={
                "text": text,
                "lines": lines,
                "region": region_label,
                "model": model_name,
                "word_count": len(text.split()),
                "legibility": legibility,
            },
        )

    # ───────────────────── region picking ─────────────────────

    def _pick_region(
        self,
        *,
        target: str | None,
        region: str | None,
        crop: tuple[float, float, float, float] | None,
    ) -> tuple[str, tuple[float, float, float, float]]:
        """Resolve an explicit crop > explicit region > target hint
        > full image. ``target`` is consulted only to map a name
        like 'the URL bar' onto a region preset — it never affects
        which lines are returned."""
        if crop is not None:
            return "custom", crop
        if region:
            r = region.lower()
            if r in REGION_PRESETS:
                return r, REGION_PRESETS[r]
            logger.debug("Unknown region preset %r, falling back", region)
        if target:
            for pat, label in TARGET_REGION_HINTS:
                if pat.search(target):
                    return label, REGION_PRESETS[label]
        return "full", REGION_PRESETS["full"]

    # ───────────────────── legibility ─────────────────────

    def _legibility_flags(
        self, *, text: str, lines: list[str],
    ) -> dict[str, bool]:
        """Heuristic feedback on whether the OCR'd text is trustworthy.

        Without per-word confidences from a black-box vision model,
        we infer:

          * ``low_confidence`` — the model emitted at least one
            :data:`ILLEGIBLE_MARKER` token, signalling it could
            not read part of the image.
          * ``sparse`` — total word count below
            :data:`SPARSE_WORD_THRESHOLD`.
          * ``edge_clipped`` — text suspiciously contains an
            ellipsis or starts/ends with a partial word (mid-token
            on the very first/last line).
        """
        word_count = len(text.split())
        flags = {
            "low_confidence": ILLEGIBLE_MARKER in text,
            "sparse": word_count < SPARSE_WORD_THRESHOLD,
            "edge_clipped": False,
        }
        if not lines:
            return flags
        # Soft "looks truncated" heuristic.
        first, last = lines[0], lines[-1]
        if (
            "…" in text
            or "..." in text
            or last.endswith(("-", "—"))
            or len(text) < LOW_CHAR_THRESHOLD
        ):
            flags["edge_clipped"] = True
        return flags

    # ───────────────────── tag cleanup ─────────────────────

    def _strip_structural_tags(self, text: str) -> str:
        """Reduce nanonets's HTML-like markup to plain text.

        Rules:

          * ``<img>...</img>`` — drop entirely. The model puts a
            caption of the image in there ("Firefox logo."), which
            is meta-commentary, not screen text.
          * ``<watermark>X</watermark>`` — keep ``X``. Despite the
            name, nanonets uses ``watermark`` for UI text overlays
            (search-bar placeholders, button labels). Often real
            screen text lives here.
          * ``<page_number>``, ``<signature>`` — keep inner content.
          * ``<div ...>``, ``<span ...>``, any tag with attributes —
            drop the whole tag-and-content. These are hallucinated
            DOM reconstructions.
          * Other paired tags — unwrap (keep content, drop tags).
          * Stray opening/closing tags — strip.
        """
        # 1) Drop image descriptions.
        text = re.sub(
            r"<img\b[^>]*>.*?</img>", "", text, flags=re.DOTALL,
        )
        # 2) Unwrap watermark content (real UI text). Drop the whole
        #    block when the inner content is the model's own
        #    "no-text-here" marker.
        def _watermark(m: re.Match) -> str:
            inner = m.group(1).strip()
            if inner.upper() in {"NOTEXT", "[?]", ""}:
                return ""
            return inner
        text = re.sub(
            r"<watermark>(.*?)</watermark>",
            _watermark,
            text, flags=re.DOTALL,
        )
        # 3) Drop tags-with-attributes entirely (DOM hallucinations
        #    like <div style="...">). These are paired with their
        #    closing tag if any, but most are stray openers.
        text = re.sub(
            r"<[a-zA-Z][a-zA-Z0-9]*\s+[^>]*>.*?</[a-zA-Z][a-zA-Z0-9]*>",
            "",
            text, flags=re.DOTALL,
        )
        text = re.sub(
            r"<[a-zA-Z][a-zA-Z0-9]*\s+[^>]*/?>", "", text,
        )
        # 4) Self-closing tags.
        text = re.sub(r"<[a-zA-Z][a-zA-Z0-9_-]*\s*/>", "", text)
        # 5) Unwrap remaining paired tags.
        prev = None
        # Repeat to handle nested unwrapping; stop when stable.
        while prev != text:
            prev = text
            text = re.sub(
                r"<([a-zA-Z][a-zA-Z0-9_-]*)>(.*?)</\1>",
                lambda m: m.group(2),
                text, flags=re.DOTALL,
            )
        # 6) Stray closers / openers.
        text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9_-]*>", "", text)
        # Collapse runs of blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ───────────────────── helpers ─────────────────────

    def _best_text_from_response(self, resp) -> str:
        if self.ctx.evaluator is not None:
            try:
                return self.ctx.evaluator._best_text_from_response(resp) or ""
            except Exception:
                pass
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""


def _empty_data(
    *, region: str = "", confidence: float = 0.0,
) -> dict[str, Any]:
    return {
        "text": "",
        "lines": [],
        "region": region,
        "word_count": 0,
        "legibility": {
            "low_confidence": False,
            "sparse": True,
            "edge_clipped": False,
        },
    }
