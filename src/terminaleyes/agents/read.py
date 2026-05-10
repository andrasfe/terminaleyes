"""ReadAgent — extract text/answers from the current screen.

Pattern: capture a frame, ask the multimodal model a free-form
question about it, return the answer as text.

Used by:
  * Controller intents like "give me the top post titles" /
    "what's on screen" / "tell me the URL".
  * NavigateAgent's post-flight when it wants to confirm something
    specific beyond the URL bar.

Compared to :class:`VerifyAgent` (which returns a yes/no), ReadAgent
returns free-text answers so callers can present them to a user.
"""

from __future__ import annotations

import logging
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


@dataclass
class ReadOutcome(Outcome):
    """``data['answer']`` holds the extracted text on success."""


class ReadAgent(Agent):
    """Ask the multimodal model an open question about the screen."""

    name = "read"

    async def run(
        self,
        *,
        question: str,
        image: np.ndarray | None = None,
        max_tokens: int = 800,
        record_label: str = "read",
        scroll_collect: int = 0,
        scroll_amount: int = 4,
        max_scrolls: int = 5,
    ) -> ReadOutcome:
        if not question or not question.strip():
            return ReadOutcome(success=False, reason="empty question")
        if self.ctx.vision_client is None:
            return ReadOutcome(
                success=False, reason="no vision client in context",
            )

        # ── Scroll-and-collect mode ──
        # When ``scroll_collect`` > 0, we read the current frame,
        # scroll the page, read again, dedupe, and repeat until we
        # have at least that many items or hit ``max_scrolls``. The
        # accumulated answer is returned as a single numbered list.
        if scroll_collect and image is None:
            return await self._run_scroll_collect(
                question=question,
                target_n=scroll_collect,
                scroll_amount=scroll_amount,
                max_scrolls=max_scrolls,
                max_tokens=max_tokens,
                record_label=record_label,
            )

        if image is None:
            if self.ctx.capture is None:
                return ReadOutcome(
                    success=False, reason="no capture in context",
                )
            try:
                frame = await self.ctx.capture.capture_frame()
                image = frame.image
            except Exception as e:
                return ReadOutcome(
                    success=False, reason=f"capture failed: {e}",
                )
            self.ctx.record_frame(image, label=record_label)

        b64 = numpy_to_base64_png(
            resize_for_mllm(
                enhance_for_screen(image),
                max_dimension=1280, min_dimension=768,
            )
        )

        # OCR the frame as ground-truth text. Vision models on cheap
        # tiers (Nemotron-Nano, Gemini Flash) routinely mis-read
        # specific tokens — e.g. they call "IBM Quantum" → "Bill
        # Quantum". Feeding them the literal OCR output as
        # authoritative reference dramatically cuts that class of
        # hallucination. We deliberately use the OCR-specialised
        # model (nanonets-ocr-s by default) which reads UI text
        # cleanly.
        ocr_text = await self._ocr_for_grounding(image)

        prompt = (
            "You are a JSON API. Look at the screen and answer the "
            f"following question:\n\n    {question}\n\n"
            "Output rules — read carefully:\n"
            "  * Reply with EXACTLY one JSON object. No reasoning, "
            "no narration, no 'wait, let me re-evaluate'.\n"
            "  * The first character of your reply must be '{'. The "
            "last character must be '}'.\n"
            "  * NO markdown fence. NO 'json' label. NO preamble.\n"
            "  * If the answer is a list (titles, items, headlines, "
            "etc.), join them with '\\n' inside the answer string. "
            "Number each line as '1. <title>' / '2. <title>' / etc.\n"
            "  * Quote the visible text VERBATIM — do NOT paraphrase, "
            "do NOT add commentary, do NOT explain why something is "
            "or isn't a title.\n"
            "  * If the information is not on screen, set answer='' "
            "and put a one-sentence explanation in 'reason'.\n\n"
            'Schema: {"answer": "<text>", "reason": "<short>"}'
        )
        messages = [
            {"role": "system", "content": prompt},
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
                    {"type": "text", "text": "Reply JSON only."},
                ],
            },
        ]
        async def _ask(msgs, *, json_mode: bool, mt: int):
            kwargs: dict[str, Any] = dict(
                model=self.ctx.vision_model,
                max_tokens=mt,
                temperature=0.0,
                messages=msgs,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return await self.ctx.vision_client.chat.completions.create(
                **kwargs,
            )

        # Attempt 1 — plain-text "numbered list, no reasoning" prompt.
        # Vision models on OpenRouter (especially Gemini Flash) tend
        # to chain-of-thought when JSON mode is on, polluting the
        # answer. The plain prompt + strict "no commentary" gets a
        # clean numbered list directly.
        # Tight token cap (~300) deters runaway chain-of-thought; the
        # JSON fallback below gets the original budget when needed.
        plain_messages = self._plain_text_retry_messages(
            question=question, b64=b64, ocr_text=ocr_text,
        )
        plain_mt = min(max_tokens, 300)
        try:
            resp = await _ask(
                plain_messages, json_mode=False, mt=plain_mt,
            )
        except Exception as e:
            return ReadOutcome(
                success=False, reason=f"model call failed: {e}",
            )

        raw = self._best_text_from_response(resp) or ""
        answer = ""
        why = ""

        if raw.strip().upper().startswith("NOT_FOUND"):
            why = "model says info not on screen"
        else:
            # Strategy that works best with chain-of-thought heavy
            # models (Gemini Flash Lite et al): the ACTUAL titles
            # appear inside double-quotes in the model's reasoning
            # — that's how it identifies them ('the title is "Foo"').
            # Quoted-string salvage gives the cleanest result.
            answer = self._extract_quoted_titles(
                raw, top_n=self._guess_top_n(question),
            )
            if not answer:
                answer = self._extract_numbered_list(raw)
                if self._looks_messy(answer, question):
                    refined = await self._refine_to_list(
                        raw, question=question, ask=_ask,
                    )
                    if refined:
                        answer = refined
                        why = why or "refined from chain-of-thought"

        # Attempt 2 — the JSON path, used as a fallback. Salvage
        # quoted titles out of any reasoning the model emits.
        if not answer:
            try:
                resp2 = await _ask(messages, json_mode=True, mt=max_tokens)
                raw2 = self._best_text_from_response(resp2) or ""
                parsed = self._extract_json(raw2) or {}
                a2 = str(parsed.get("answer", "")).strip()
                w2 = str(parsed.get("reason", "")).strip()
                if a2:
                    answer = a2
                    why = w2 or "json fallback"
                elif raw2.strip():
                    salvaged2 = self._salvage_plain_text(raw2)
                    if salvaged2:
                        answer = salvaged2
                        why = "salvaged from json fallback"
                        raw = raw + "\n---json---\n" + raw2
            except Exception as e:
                logger.debug("json fallback failed: %s", e)

        # Last-resort salvage on the original plain reply.
        if not answer and raw.strip():
            salvaged = self._salvage_plain_text(raw)
            if salvaged:
                answer = salvaged
                why = why or "last-resort salvage"

        if not answer:
            return ReadOutcome(
                success=False,
                reason=(
                    why or
                    "model returned no answer (raw="
                    + raw[:160].replace("\n", " ") + ")"
                ),
                data={"raw": raw, "answer": "", "reason": why},
            )
        # Print the answer so SSE-stream subscribers (UI logs, CLI
        # auto-route stdout) see it without needing to dig into
        # outcome.data.
        first_line = answer.split("\n", 1)[0]
        print(f"   Read answered: {first_line[:160]}")
        if "\n" in answer:
            for ln in answer.split("\n")[1:][:20]:
                if ln.strip():
                    print(f"     · {ln.strip()[:160]}")
        return ReadOutcome(
            success=True,
            reason=why or "ok",
            data={"answer": answer, "reason": why, "raw": raw},
        )

    async def _run_scroll_collect(
        self,
        *,
        question: str,
        target_n: int,
        scroll_amount: int,
        max_scrolls: int,
        max_tokens: int,
        record_label: str,
    ) -> ReadOutcome:
        """Read → scroll → read → dedupe loop until we have target_n
        unique items or run out of scroll attempts.

        Each iteration calls ``self.run(...)`` with a single-frame
        image so the model gets a fresh context per look. Items
        across iterations are deduped case-insensitively. The
        scroll itself is delegated to :class:`ScrollAgent`.
        """
        from terminaleyes.agents.scroll import ScrollAgent
        if self.ctx.capture is None:
            return ReadOutcome(
                success=False, reason="no capture in context",
            )

        accumulated: list[str] = []
        seen: set[str] = set()
        # Per-iteration question doesn't include the global "top N"
        # so the model doesn't insist on padding to N. We dedupe
        # ourselves across iterations.
        per_q = (
            "List every visible main-feed post title on this page, "
            "one per line. Skip ads, sidebar widgets, navigation, "
            "and the subreddit description."
        )

        scroller = ScrollAgent(self.ctx)
        last_outcome: ReadOutcome | None = None
        for it in range(max_scrolls + 1):
            try:
                frame = await self.ctx.capture.capture_frame()
            except Exception as e:
                return ReadOutcome(
                    success=False, reason=f"capture failed: {e}",
                )
            self.ctx.record_frame(
                frame.image, label=f"{record_label}_iter_{it:02d}",
            )
            outcome = await self.run(
                question=per_q,
                image=frame.image,
                max_tokens=max_tokens,
                record_label=f"{record_label}_iter_{it:02d}",
            )
            last_outcome = outcome
            if outcome.success:
                ans = (outcome.data or {}).get("answer", "") or ""
                for ln in ans.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    # Strip "1. " / "2. " prefix added by extractors.
                    import re as _re
                    m = _re.match(r"^(?:\d+[.):]|[-*•])\s+(.+)$", ln)
                    title = m.group(1).strip() if m else ln
                    # Strip markdown bold/italic so '**Foo**' and
                    # 'Foo' dedupe to one entry.
                    title = _re.sub(r"\*+", "", title).strip()
                    title = title.rstrip('.,;:').strip()
                    # Dedupe key: lowercased, alphanumerics only —
                    # also collapses 'Quantum Gates' vs 'Quantum
                    # gates' and ignores trailing punctuation.
                    key = _re.sub(r"[^a-z0-9]", "", title.lower())
                    if title and key not in seen and len(title) >= 4:
                        seen.add(key)
                        accumulated.append(title)
                if len(accumulated) >= target_n:
                    break
            if it >= max_scrolls:
                break
            print(
                f"   scroll-collect: {len(accumulated)}/{target_n} so "
                f"far; scrolling down {scroll_amount}"
            )
            # Belt-and-braces scroll: keyboard Page Down (universal
            # browser shortcut, never falls into a sidebar) PLUS
            # mouse-wheel ticks on the main feed (helps when keyboard
            # focus is in an embedded input).
            kb = self.ctx.keyboard
            try:
                if kb is not None:
                    for _ in range(max(1, scroll_amount // 3)):
                        await kb.send_keystroke("PageDown")
                        import asyncio as _aio
                        await _aio.sleep(0.15)
            except Exception as e:
                logger.debug("Page_Down failed: %s", e)
            try:
                await scroller.run(
                    direction="down", amount=scroll_amount,
                    hover_at=(0.5, 0.55),
                )
            except Exception as e:
                logger.warning("scroll failed: %s", e)
                break

        accumulated = accumulated[:target_n]
        if not accumulated:
            return ReadOutcome(
                success=False,
                reason=(
                    last_outcome.reason if last_outcome
                    else "no titles found across scroll iterations"
                ),
                data={"answer": "", "iterations": it + 1},
            )
        answer = "\n".join(
            f"{i}. {t}" for i, t in enumerate(accumulated, 1)
        )
        print(f"   scroll-collect: collected {len(accumulated)} titles")
        for ln in accumulated:
            print(f"     · {ln[:160]}")
        return ReadOutcome(
            success=True,
            reason=(
                f"collected {len(accumulated)} titles over "
                f"{it + 1} iteration(s)"
            ),
            data={
                "answer": answer,
                "iterations": it + 1,
                "titles": accumulated,
            },
        )

    def _guess_top_n(self, question: str) -> int:
        """Pull a 'top N' count out of the question. Defaults to 5."""
        import re as _re
        m = _re.search(r"top\s+(\d+)", question, _re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 50:
                    return n
            except ValueError:
                pass
        return 5

    def _extract_quoted_titles(self, raw: str, *, top_n: int) -> str:
        """Pull title-shaped double-quoted strings out of the reply.

        Models that reason out loud about a screen typically write
        the actual page text in quotes ('the title is "Foo"').
        Quoted-string extraction filters out the surrounding prose
        far more cleanly than line-based heuristics.

        Filters: ≥ 8 chars, contains a space, doesn't begin with
        a body-text marker (first-person pronouns, narration verbs,
        etc.), starts with a capital or a digit.
        """
        import re as _re
        body_markers = (
            "the user", "you are", "respond with",
            "schema:", "answer:", "reason:",
            "i am ", "i have ", "i was ", "i'm ", "i've ",
            "i would ", "i will ", "i need ", "i want ",
            "we are ", "we have ", "we will ",
            "let me", "wait,", "let's", "let us",
            "looking at", "based on", "if i",
            "subreddit for", "discussion of", "questions about",
            "the following", "here is", "here are",
            "this is", "that is",
        )
        seen: set[str] = set()
        cleaned: list[str] = []
        for q in _re.findall(r'"([^"\n]{8,})"', raw):
            raw_q = q.strip()
            # Reject ellipsis quotes (truncated body excerpts)
            # before stripping trailing punctuation.
            if "..." in raw_q or raw_q.endswith(("…", ",")):
                continue
            t = raw_q.rstrip('.,;:').strip()
            tl = t.lower()
            if not (10 <= len(t) <= 200):
                continue
            # Real titles have at least 2 spaces (3+ words) — drops
            # body fragments like "Gate decomposition".
            if t.count(" ") < 2:
                continue
            if tl in seen:
                continue
            if any(tl.startswith(p) for p in body_markers):
                continue
            first = t[0]
            if first.islower() and not first.isdigit():
                continue
            seen.add(tl)
            cleaned.append(t)
        if not cleaned:
            return ""
        # Cap to requested top N.
        cleaned = cleaned[:top_n]
        return "\n".join(
            f"{i}. {t}" for i, t in enumerate(cleaned, 1)
        )

    def _looks_messy(self, answer: str, question: str) -> bool:
        """Heuristic: does the extracted answer look like chain-of-
        thought leaked into the numbered list?"""
        if not answer:
            return False
        lines = [ln for ln in answer.splitlines() if ln.strip()]
        if not lines:
            return False
        messy_markers = (
            "**", "identify", "locate", "extract",
            "the user", "the screen shows", "i see", "i need",
            "looking at", "let me", "let's", "wait,",
            "post:", "title:", "next ", "below", "above",
            "first post", "second post", "third post",
        )
        bad = 0
        for ln in lines:
            tl = ln.lower()
            if any(m in tl for m in messy_markers):
                bad += 1
        # If more than 1/3 of lines look messy, refine.
        return bad >= max(1, len(lines) // 3)

    async def _refine_to_list(
        self, raw: str, *, question: str, ask,
    ) -> str:
        """Text-only second call: turn the model's chain-of-thought
        reply into a clean numbered list of the requested items.

        Cheap (no image); runs only when the first reply looks
        messy. Returns "" on any failure so the caller can fall
        back to the noisy answer.
        """
        sys_prompt = (
            "You are a strict text extractor. The user previously "
            "asked a vision model the question below. The reply "
            "contained reasoning AND the actual answer mixed together. "
            "Your job: produce ONLY the requested items as a numbered "
            "list. Discard all reasoning, narration, and meta "
            "commentary.\n\n"
            "Output format (no exceptions):\n"
            "  1. <item>\n"
            "  2. <item>\n"
            "  3. <item>\n"
            "Reply with the numbered lines only — nothing else."
        )
        user = (
            f"Original question:\n{question}\n\n"
            f"Vision model reply (messy):\n{raw}\n\n"
            "Extract the answer as a clean numbered list."
        )
        try:
            resp = await ask(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
                json_mode=False, mt=400,
            )
        except Exception as e:
            logger.debug("refine call failed: %s", e)
            return ""
        text = self._best_text_from_response(resp) or ""
        if text.strip().upper().startswith("NOT_FOUND"):
            return ""
        return self._extract_numbered_list(text)

    def _extract_numbered_list(self, raw: str) -> str:
        """Pull a numbered list (``1. ...`` / ``2. ...``) out of a
        free-form reply. Drops items that look like reasoning
        fragments (start with 'At top', 'Then below', 'Possibly',
        speculative pronouns, etc.). Returns "" if nothing of
        usable quality is found."""
        import re as _re
        body_text_starts = (
            "at top", "at bottom", "then below", "then there",
            "further down", "looking at", "looks like",
            "possibly", "maybe", "not sure",
            "let me", "wait,", "let's", "let us",
            "i am", "i have", "i was", "i'm",
            "we are", "we have",
            "the user", "you are",
            "there is", "there are",
            "above that", "below that",
            "based on",
        )
        out: list[str] = []
        for ln in raw.splitlines():
            ln = ln.strip()
            m = _re.match(r"^(?:\d+[.):]|[-*•])\s+(.+)$", ln)
            if not m:
                continue
            t = m.group(1).strip().rstrip('.,;:').strip()
            tl = t.lower()
            if len(t) < 3 or len(t) > 200:
                continue
            if any(tl.startswith(p) for p in body_text_starts):
                continue
            # Strip surrounding quotes — many responses wrap titles
            # like '"Foo"' which is fine, but the bare quoted form
            # reads better.
            if (
                len(t) >= 2
                and t[0] in '"“”‘’\''
                and t[-1] in '"“”‘’\''
            ):
                t = t[1:-1].strip()
                if len(t) < 3:
                    continue
            # Skip if it looks like a sentence (ends with period AND
            # contains common-sentence words). Real titles don't
            # usually contain "is", "are", "was" as full words —
            # they're typically noun phrases. This is a soft filter
            # so don't drop too aggressively.
            if t and t.lower() not in {x.lower() for x in out}:
                out.append(t)
        if out:
            return "\n".join(f"{i}. {t}" for i, t in enumerate(out, 1))
        return ""

    async def _ocr_for_grounding(self, image) -> str:
        """Best-effort OCR pass to give the multimodal model literal
        text as authoritative ground truth. Uses :class:`OcrAgent`
        (which defaults to the OCR-specialised ``nanonets-ocr-s``
        model when configured). Failure swallowed — returns ``""``
        so the model just operates without grounding."""
        # Late import to avoid a cycle with the controller registry.
        from terminaleyes.agents.ocr import OcrAgent
        try:
            outcome = await OcrAgent(self.ctx).run(
                region="full", image=image, record_label="read_ocr",
            )
        except Exception as e:
            logger.debug("read-grounding OCR failed: %s", e)
            return ""
        if not (outcome.success and outcome.data):
            return ""
        return (outcome.data.get("text") or "").strip()

    def _plain_text_retry_messages(
        self, *, question: str, b64: str, ocr_text: str = "",
    ) -> list[dict]:
        """Stripped-down prompt for the retry path. Asks for a plain
        newline-separated list with no JSON wrapping. Some models
        (notably Gemini Flash via OpenRouter) chain-of-thought
        whenever JSON mode is requested; bypassing it cleans up the
        output.

        When ``ocr_text`` is supplied, the OCR'd screen contents are
        included as authoritative ground truth. Vision models on
        cheap tiers misread specific tokens (we've seen "IBM Quantum"
        → "Bill Quantum"); giving them the literal OCR'd characters
        eliminates that class of error.
        """
        sys_prompt = (
            "Look at the screen and answer the user's question.\n\n"
            "OUTPUT FORMAT — follow exactly:\n"
            "  1. <item exactly as it appears on the SCREEN>\n"
            "  2. <item exactly as it appears on the SCREEN>\n"
            "  3. <item exactly as it appears on the SCREEN>\n\n"
            "RULES (strict):\n"
            "  * Reply ONLY with the numbered lines. Nothing else.\n"
            "  * Do NOT explain. Do NOT narrate. Do NOT speculate.\n"
            "  * Do NOT include reasoning words like 'At top', "
            "'Then', 'Looking at', 'Possibly', 'Maybe', 'I think'.\n"
            "  * Do NOT use markdown (no '**bold**', no '*italic*').\n"
            "  * Quote only what is VISIBLY ON THIS SCREEN. Do not "
            "invent or fill in items from memory if you cannot see "
            "enough on screen.\n"
            "  * If the page does not show the requested info, "
            "reply with EXACTLY one line: NOT_FOUND"
        )
        # Truncate OCR text aggressively for the prompt — most useful
        # for finding specific tokens; full content would blow tokens.
        ocr_block = ""
        if ocr_text:
            snip = ocr_text.strip()
            if len(snip) > 3000:
                snip = snip[:3000] + "\n…(truncated)…"
            ocr_block = (
                "\n\nOCR-extracted screen text (AUTHORITATIVE — use "
                "this for the exact spelling of any text you cite; "
                "if your visual reading disagrees with the OCR, "
                "trust the OCR):\n-----\n"
                f"{snip}\n-----"
            )
        return [
            {"role": "system", "content": sys_prompt},
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
                    {"type": "text", "text": question + ocr_block},
                ],
            },
        ]

    # ───────────────────── helpers ─────────────────────

    def _best_text_from_response(self, resp) -> str:
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._best_text_from_response(resp) or ""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def _salvage_plain_text(self, raw: str) -> str:
        """Pull a useful answer out of a non-JSON model reply.

        Three-pass salvage, in order of preference:

          1. **Quoted strings** — when the model reasons out loud
             ("the title is 'Foo'..."), the actual titles end up in
             quotes. Extract every double-quoted substring ≥ 8 chars
             and de-dupe. This is the cleanest output.
          2. **Numbered / bulleted list items**.
          3. **Plain title-shaped lines** — fallback for free-form
             prose where the model just listed titles on lines.

        Returns "" if nothing useful could be salvaged.
        """
        import re as _re
        # Pass 1 — quoted strings. The model frequently writes the
        # exact titles inside ".." while reasoning. These are
        # near-perfect quality.
        quoted = _re.findall(r'"([^"\n]{8,})"', raw)
        # Filter: drop fragments ending with comma+space (mid-sentence
        # quotes) and trim trailing punctuation.
        cleaned: list[str] = []
        seen: set[str] = set()
        body_text_markers = (
            "the user", "you are", "respond with",
            "schema:", "answer:", "reason:",
            # First-person body text the model quotes from posts:
            "i am ", "i have ", "i was ", "i'm ", "i've ",
            "i would ", "i will ", "i need ", "i want ",
            "we are ", "we have ", "we will ",
            # Speculative reasoning about page state:
            "let me", "wait,", "let's", "let us",
            "looking at", "based on", "if i",
            # Subreddit description boilerplate:
            "subreddit for", "discussion of", "questions about",
        )
        for q in quoted:
            t = q.strip().rstrip('.,;:').strip()
            tl = t.lower()
            if not (8 <= len(t) <= 200):
                continue
            # Real titles have at least one space — drops bare
            # identifiers like "r/Qiskit" or "json" that the
            # model quoted while reasoning.
            if " " not in t:
                continue
            if tl in seen:
                continue
            if any(tl.startswith(p) for p in body_text_markers):
                continue
            # Skip mid-sentence fragments: lowercase first letter,
            # ends with a comma, or contains "..." (model truncated
            # a quote).
            first = t[0]
            if first.islower() and not t[0].isdigit():
                continue
            if t.endswith(",") or "..." in t:
                continue
            seen.add(tl)
            cleaned.append(t)
        if cleaned:
            return "\n".join(
                f"{i}. {t}" for i, t in enumerate(cleaned, 1)
            )

        # Pass 2 — numbered/bulleted lines.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bullet_re = _re.compile(r"^(?:\d+[.):]|[-*•])\s+(.+)")
        out: list[str] = []
        for ln in lines:
            m = bullet_re.match(ln)
            if m:
                t = m.group(1).strip().rstrip('.,;:')
                if len(t) >= 4 and t.lower() not in {x.lower() for x in out}:
                    out.append(t)
        if out:
            return "\n".join(out)

        # Pass 3 — title-shaped lines (last resort).
        skip_starts = (
            "the user", "i need to", "i will", "let me",
            "okay", "sure", "here are", "here is", "here's",
            "based on", "looking at", "to answer",
            "```", "json", "answer:", "reason:",
            "wait,", "let's", "are there", "below",
            "subreddit", "description", "body text",
        )
        for ln in lines:
            low = ln.lower()
            if any(low.startswith(p) for p in skip_starts):
                continue
            if (
                len(ln) >= 6
                and any(c.isupper() for c in ln)
                and any(c.islower() for c in ln)
                and not ln.endswith(":")
            ):
                if ln.lower() not in {x.lower() for x in out}:
                    out.append(ln)
        return "\n".join(out)

    def _extract_json(self, raw: str) -> dict | None:
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._extract_json(raw)
        import json
        import re
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
