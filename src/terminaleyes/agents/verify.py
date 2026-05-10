"""Visual yes/no oracle.

Wraps a single multimodal model call: capture-or-take a frame, ask a
focused question, parse a strict JSON answer. Used by FocusAgent,
LoginAgent, and the post-click navigation oracle.
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
class VerifyOutcome(Outcome):
    """Outcome where ``success`` mirrors the model's yes/no verdict."""


class VerifyAgent(Agent):
    """Ask the multimodal model a yes/no question about a frame.

    Args to :meth:`run`:
      - ``question``: short imperative description of what to check.
      - ``image``: optional ndarray; if omitted, captures one.
      - ``visual_only``: if True, the prompt steers the model AWAY
        from text-based shortcuts (e.g. the literal word "password")
        so the answer reflects visual structure only.
      - ``crop``: optional ``(x0, y0, x1, y1)`` fractions to focus
        the question on a region of the frame.

    Returns a VerifyOutcome with ``data`` containing ``raw`` (the
    model's full text response) and ``parsed`` (the JSON dict).
    """

    name = "verify"

    async def run(
        self,
        *,
        question: str,
        image: np.ndarray | None = None,
        visual_only: bool = True,
        crop: tuple[float, float, float, float] | None = None,
        max_tokens: int = 800,
        record_label: str = "verify",
    ) -> VerifyOutcome:
        if self.ctx.vision_client is None:
            return VerifyOutcome(
                success=False, reason="no vision client in context",
            )
        if image is None:
            if self.ctx.capture is None:
                return VerifyOutcome(
                    success=False, reason="no capture in context",
                )
            try:
                frame = await self.ctx.capture.capture_frame()
                image = frame.image
            except Exception as e:
                return VerifyOutcome(
                    success=False, reason=f"capture failed: {e}",
                )
            self.ctx.record_frame(image, label=record_label)
        if crop is not None:
            h, w = image.shape[:2]
            x0 = max(0, int(crop[0] * w))
            y0 = max(0, int(crop[1] * h))
            x1 = min(w, int(crop[2] * w))
            y1 = min(h, int(crop[3] * h))
            image = image[y0:y1, x0:x1]

        b64 = numpy_to_base64_png(
            resize_for_mllm(
                enhance_for_screen(image),
                max_dimension=1280, min_dimension=768,
            )
        )

        steer = ""
        if visual_only:
            steer = (
                "\n\nIMPORTANT: judge by visual structure only. Do "
                "NOT base your answer on whether any specific word "
                "appears as text in the image."
            )

        prompt = (
            "You are a JSON API. Look at the screen and answer the "
            f"question:\n\n    {question}\n"
            f"{steer}\n\n"
            "Respond with ONLY a JSON object — no preamble, no "
            "markdown.\n\n"
            'Schema: {"answer": true|false, '
            '"reason": "<one short sentence>"}'
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
        # Try JSON mode first (LM Studio supports response_format on
        # most models); fall back to free-form on the second pass if
        # the model rejects it.
        for attempt in range(2):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.ctx.vision_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    messages=messages,
                )
                if attempt == 0:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self.ctx.vision_client.chat.completions.create(
                    **kwargs,
                )
                break
            except Exception as e:
                if attempt == 0:
                    logger.debug(
                        "Verify call with json_object format failed (%s) — retrying free-form",
                        e,
                    )
                    continue
                logger.warning("VerifyAgent model call failed: %s", e)
                return VerifyOutcome(
                    success=False, reason=f"model call failed: {e}",
                )

        raw = self._best_text_from_response(resp) or ""
        parsed = self._extract_json(raw) or {}
        verdict = bool(parsed.get("answer", False))
        reason = str(parsed.get("reason", ""))[:200]
        return VerifyOutcome(
            success=verdict, reason=reason,
            data={"raw": raw, "parsed": parsed},
        )

    # ───────────────────── helpers ─────────────────────

    def _best_text_from_response(self, resp: Any) -> str:
        """Use the evaluator's helper if available; fall back to dumb str."""
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._best_text_from_response(resp) or ""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def _extract_json(self, raw: str) -> dict | None:
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._extract_json(raw)
        # Minimal fallback.
        import json
        import re
        if not raw:
            return None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
