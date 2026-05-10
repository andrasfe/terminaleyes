"""Shared resources injected into every agent."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from terminaleyes.agents.vault import Vault
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.commander.evaluator import ConditionEvaluator
    from terminaleyes.keyboard.base import KeyboardOutput
    from terminaleyes.mouse.base import MouseOutput

logger = logging.getLogger(__name__)


_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class AgentContext:
    """Bag of shared infrastructure passed to every agent.

    Most fields are optional so agents that don't need them (e.g. the
    Vault used standalone in a CLI) can construct a minimal context.
    Construct with whatever you have; agents document which fields
    they require.
    """

    # I/O — Pi-side HID
    mouse: "MouseOutput | None" = None
    keyboard: "KeyboardOutput | None" = None

    # Vision input
    capture: "WebcamCapture | None" = None

    # LLM clients — OpenAI-compatible (LM Studio) for multimodal calls
    vision_client: Any = None        # an openai.AsyncClient or similar
    vision_model: str = ""
    # OCR-specialised model served by the same vision_client. When
    # set, OcrAgent uses this model name instead of vision_model so
    # general-purpose Q&A and pure OCR can use different specialists.
    ocr_model: str = ""

    # ShowUI grounding helper (callable: b64, prompt -> (x, y) | None)
    showui_query: Any = None

    # Misc helpers from the older commander stack — present so we
    # don't reimplement parsers/JSON extractors during the migration.
    evaluator: "ConditionEvaluator | None" = None

    # Storage
    vault: "Vault | None" = None

    # Per-session output directory. When set, every agent that
    # captures a frame should save the raw image into this folder
    # via :meth:`record_frame`. Sequential filename prefixes make the
    # capture order easy to inspect after a run.
    output_dir: Path | None = None

    # Internal counter for sequential frame filenames. Mutated by
    # :meth:`record_frame`; not part of the public API.
    _frame_counter: int = 0

    # Free-form scratchpad for cross-agent state during a run.
    scratch: dict[str, Any] = field(default_factory=dict)

    # ───────────────────── frame recording ─────────────────────

    def record_frame(
        self,
        image: np.ndarray,
        *,
        label: str = "frame",
        ext: str = "png",
    ) -> Path | None:
        """Save a captured frame to ``output_dir`` if one is configured.

        Returns the path written, or ``None`` when no ``output_dir``
        was set (a no-op so agents can record unconditionally).

        Filename shape: ``{seq:04d}_{HHMMSS}_{label}.{ext}``. Labels
        are sanitised to safe filename chars.
        """
        if self.output_dir is None or image is None:
            return None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create output_dir %s: %s",
                self.output_dir, e,
            )
            return None
        self._frame_counter += 1
        seq = f"{self._frame_counter:04d}"
        ts = datetime.now().strftime("%H%M%S")
        safe = _LABEL_SAFE.sub("_", label)[:80] or "frame"
        path = self.output_dir / f"{seq}_{ts}_{safe}.{ext}"
        try:
            ok = cv2.imwrite(str(path), image)
            if not ok:
                logger.warning("cv2.imwrite returned False for %s", path)
                return None
        except Exception as e:
            logger.warning("Failed to write frame to %s: %s", path, e)
            return None
        return path

    async def capture_and_record(
        self, *, label: str = "frame",
    ) -> "np.ndarray | None":
        """Convenience: capture a frame from ``self.capture`` and record
        it in one call. Returns the raw image or ``None`` on failure."""
        if self.capture is None:
            return None
        try:
            frame = await self.capture.capture_frame()
        except Exception as e:
            logger.warning("capture failed for %r: %s", label, e)
            return None
        self.record_frame(frame.image, label=label)
        return frame.image

    def subdir(self, name: str) -> Path | None:
        """Return a subdirectory of ``output_dir`` (creating it lazily).

        Useful for agents that want their own scratch space alongside
        the main session frames (e.g. the visual servo homer dumping
        per-step annotated images).
        """
        if self.output_dir is None:
            return None
        safe = _LABEL_SAFE.sub("_", name) or "sub"
        path = self.output_dir / safe
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create subdir %s: %s", path, e)
            return None
        return path
