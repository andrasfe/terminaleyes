"""Default :class:`ContextFactory` for the command-center runner.

A factory is invoked at the start of every run to build a fresh
:class:`AgentContext` (mouse + keyboard + capture + vision client +
output dir). The runner closes those resources when the run ends, so
the webcam is held only while a run is in flight — exactly matching
the lifecycle of the ``terminaleyes do`` CLI.

Usage from server bootstrap::

    from terminaleyes.commandcenter.factory import (
        make_default_context_factory,
    )
    factory = make_default_context_factory(
        settings, base_dir=store.watch_dir, bus=bus,
    )
    app = create_app(factory, frame_store=store, bus=bus)

The per-run output dir is named after the runner's ``run_id`` (read
from the bus's ``current_run_id()``) so the UI can correlate frames
to run records: ``<watch_dir>/<run_id>/0001_..._navigate_check.png``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def make_default_context_factory(
    settings,
    *,
    base_dir: Path,
    bus=None,
) -> Callable[[], Awaitable[tuple[Any, Any, Any, Any]]]:
    """Build a runner-compatible ``ContextFactory``.

    ``base_dir`` is the directory under which per-run subdirectories
    are created (same as :class:`FrameStore.watch_dir` so the UI sees
    everything). ``bus`` is the optional :class:`LogBus`; when set,
    the factory uses ``bus.current_run_id()`` to name the per-run
    subdir for clean frame ↔ run correlation in the UI.
    """
    base_dir = Path(base_dir).expanduser().resolve()

    async def factory():
        from openai import AsyncOpenAI
        from terminaleyes.agents.context import AgentContext
        from terminaleyes.capture.webcam import WebcamCapture
        from terminaleyes.commander.evaluator import ConditionEvaluator
        from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
        from terminaleyes.mouse.http_backend import HttpMouseOutput

        cfg = settings.commander

        # Per-run output subdir. Use the bus's current run id when
        # available so the UI can map frame.run_id → runner record.
        run_id = None
        if bus is not None:
            try:
                run_id = bus.current_run_id()
            except Exception:
                run_id = None
        sub = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = base_dir / sub
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Could not create per-run output dir %s: %s",
                output_dir, e,
            )
            output_dir = None

        keyboard = HttpKeyboardOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        mouse = HttpMouseOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        await keyboard.connect()
        await mouse.connect()

        resolution = None
        if (settings.capture.resolution_width
                and settings.capture.resolution_height):
            resolution = (
                settings.capture.resolution_width,
                settings.capture.resolution_height,
            )
        capture = WebcamCapture(
            device_index=settings.capture.device_index,
            resolution=resolution,
        )
        await capture.open()

        client = AsyncOpenAI(
            base_url=cfg.lmstudio_base_url, api_key="not-needed",
        )
        evaluator = ConditionEvaluator(
            model=cfg.lmstudio_model,
            base_url=cfg.lmstudio_base_url,
            max_tokens=cfg.lmstudio_max_tokens,
        )

        ctx = AgentContext(
            mouse=mouse,
            keyboard=keyboard,
            capture=capture,
            vision_client=client,
            vision_model=cfg.lmstudio_model,
            ocr_model=cfg.lmstudio_ocr_model,
            evaluator=evaluator,
            output_dir=output_dir,
        )
        return ctx, keyboard, mouse, capture

    return factory
