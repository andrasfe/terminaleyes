"""TargetAgent — locate a click target on screen by description.

Cascade of locators (first hit wins):

  1. **OCR** (tesseract). When the user supplied a quoted token in
     the description, only that quoted text is matched — generic
     descriptors like "subreddit"/"entry" are kept out so they don't
     match the first generic occurrence on the page.
  2. **Scene-map + ShowUI grounding**. Multimodal model enumerates
     clickable elements; a keyword-scored best match is then grounded
     to pixel coordinates by ShowUI.
  3. **ShowUI on focused crops** (left sidebar, footer strip). Helps
     when the target is small text that ShowUI on the full image
     would miss.

Returns ``(x_pct, y_pct)`` in image-fraction coordinates. Reuses the
existing helpers in :mod:`commander.visual_servo_homer` and
:mod:`commander.ocr_finder` so this agent is a thin wrapper — no
behavioural change to the proven homer logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.commander.closed_loop_homer import ClosedLoopHomer
from terminaleyes.commander.ocr_finder import (
    annotate_ocr_hit,
    find_text as ocr_find_text,
    have_ocr,
)
from terminaleyes.utils.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetOutcome(Outcome):
    """``data['position'] = (x_pct, y_pct)`` and ``data['method']``."""


class TargetAgent(Agent):
    """Locate a target element by free-form description."""

    name = "target"

    async def run(
        self,
        *,
        description: str,
        image: np.ndarray | None = None,
        run_dir: Path | None = None,
    ) -> TargetOutcome:
        if not description:
            return TargetOutcome(
                success=False, reason="empty description",
            )
        if image is None:
            if self.ctx.capture is None:
                return TargetOutcome(
                    success=False, reason="no capture in context",
                )
            frame = await self.ctx.capture.capture_frame()
            image = frame.image

        # 1. OCR — quoted-token primary search.
        if have_ocr():
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", description)
            if quoted:
                primary = [q.lower() for q in quoted]
            else:
                primary = ClosedLoopHomer._target_keywords(description)
            hits = ocr_find_text(image, primary)
            if hits:
                top = hits[0]
                if run_dir is not None:
                    try:
                        cv2.imwrite(
                            str(run_dir / "target_ocr_hit.png"),
                            annotate_ocr_hit(image, top),
                        )
                    except Exception:
                        pass
                return TargetOutcome(
                    success=True,
                    reason=(
                        f"OCR matched {top.text!r} "
                        f"(conf={top.confidence:.0f})"
                    ),
                    data={
                        "position": (top.x_pct, top.y_pct),
                        "method": "ocr",
                        "matched_text": top.text,
                        "confidence": top.confidence,
                    },
                )

        # 2. Scene-map + ShowUI grounding.
        b64 = await self._encode(image)
        helper = ClosedLoopHomer(session=_SessionAdapter(self.ctx))
        try:
            scene = await helper._scene_map(b64, run_dir)
            match = helper._best_scene_match(scene, description)
        except Exception as e:
            logger.warning("Scene-map failed: %s", e)
            match = None
        if match is not None:
            label = match["label"]
            stripped = label.lstrip("/").strip()
            for prefix in ("r/", "/r/", "u/"):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            ground_prompts = [
                f"Click on {label}",
                f"Click on the {label} link",
                f"Click on the {label} button",
                f"Click on {stripped}",
                f"Click on the {stripped} link",
            ]
            for p in ground_prompts:
                pos = await self._showui_query(b64, p)
                if pos is not None:
                    return TargetOutcome(
                        success=True,
                        reason=(
                            f"scene-map matched {label!r}, "
                            f"ShowUI grounded via {p!r}"
                        ),
                        data={
                            "position": pos,
                            "method": "scene_map_showui",
                            "matched_label": label,
                        },
                    )

        # 3. ShowUI fallback — direct prompts on user description.
        for p in ClosedLoopHomer._showui_prompt_variants(description):
            pos = await self._showui_query(b64, p)
            if pos is not None:
                return TargetOutcome(
                    success=True,
                    reason=f"ShowUI direct grounded via {p!r}",
                    data={"position": pos, "method": "showui_direct"},
                )

        # 4. Cropped ShowUI for small / sidebar text.
        crop_regions = [
            ("sidebar_full",   0.0, 0.0,  0.30, 1.0),
            ("sidebar_bottom", 0.0, 0.55, 0.32, 1.0),
            ("footer_strip",   0.0, 0.75, 1.0,  1.0),
        ]
        ih, iw = image.shape[:2]
        quoted_for_crop = re.findall(r"['\"]([^'\"]+)['\"]", description)
        token = (
            quoted_for_crop[0]
            if quoted_for_crop else description.split()[-1]
        )
        for name, x0f, y0f, x1f, y1f in crop_regions:
            x0, y0 = int(x0f * iw), int(y0f * ih)
            x1, y1 = int(x1f * iw), int(y1f * ih)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_b64 = await self._encode(crop)
            for cp in (
                f"Click on {token}",
                f"Click on the {token} link",
                f"Click on r/{token}",
            ):
                pos = await self._showui_query(crop_b64, cp)
                if pos is not None:
                    fx = (pos[0] * (x1 - x0) + x0) / iw
                    fy = (pos[1] * (y1 - y0) + y0) / ih
                    return TargetOutcome(
                        success=True,
                        reason=(
                            f"ShowUI grounded {cp!r} on crop {name!r}"
                        ),
                        data={
                            "position": (fx, fy),
                            "method": "cropped_showui",
                            "crop": name,
                        },
                    )
            if have_ocr():
                hits_crop = ocr_find_text(
                    crop, [token.lower()],
                    crops=[(0.0, 0.0, 1.0, 1.0)],
                )
                if hits_crop:
                    top = hits_crop[0]
                    fx = (top.x_pct * (x1 - x0) + x0) / iw
                    fy = (top.y_pct * (y1 - y0) + y0) / ih
                    return TargetOutcome(
                        success=True,
                        reason=(
                            f"OCR on crop {name!r} matched "
                            f"{top.text!r}"
                        ),
                        data={
                            "position": (fx, fy),
                            "method": "cropped_ocr",
                            "crop": name,
                        },
                    )

        return TargetOutcome(
            success=False,
            reason="OCR + scene-map + ShowUI all missed",
        )

    # ───────────────────── helpers ─────────────────────

    async def _showui_query(self, b64: str, prompt: str):
        if self.ctx.showui_query is None:
            return None
        try:
            return await self.ctx.showui_query(b64, prompt)
        except Exception as e:
            logger.debug("ShowUI query failed: %s", e)
            return None

    @staticmethod
    async def _encode(image: np.ndarray) -> str:
        resized = resize_for_mllm(
            enhance_for_screen(image),
            max_dimension=1280, min_dimension=768,
        )
        return numpy_to_base64_png(resized)


# Tiny adapter so the helper :class:`ClosedLoopHomer` can run its
# scene-map call from an :class:`AgentContext` (it expects a session-
# like object with private attributes).
class _SessionAdapter:
    def __init__(self, ctx) -> None:
        self._capture = ctx.capture
        self._client = ctx.vision_client
        self._model = ctx.vision_model
        self._evaluator = ctx.evaluator

    async def _ensure_client(self) -> None:
        return None
