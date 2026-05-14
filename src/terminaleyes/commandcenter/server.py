"""FastAPI app for the command center.

Endpoints:
  GET  /                         -> static index.html
  GET  /api/frames               -> list newest-first {limit, before}
  GET  /api/frames/latest        -> meta of newest frame  (optional ?wait=1)
  GET  /api/frames/{id}          -> JPEG/PNG bytes
  GET  /api/frames/{id}/neighbours
  POST /api/run                  -> start a ControllerAgent run
  GET  /api/runs                 -> list recent runs
  GET  /api/runs/{id}            -> single run record
  GET  /api/runs/{id}/logs       -> SSE stream of LogEvents
  GET  /api/logs                 -> SSE stream of all logs
  GET  /api/state                -> {busy, latest_id, run?}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from terminaleyes.commandcenter.frame_store import FrameStore
from terminaleyes.commandcenter.log_bus import LogBus, install_logging
from terminaleyes.commandcenter.runner import (
    ContextFactory, Runner, RunnerBusy,
)

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    intent: str = Field(min_length=1)
    no_focus: bool = False
    vault: str | None = None
    platform: str = "linux"
    dry_run: bool = False
    allow_llm_fallback: bool = True
    planner: str = "auto"           # "auto" | "ml" | "rules"
    ml_adapter: str | None = None   # required when planner == "ml"


class MouseClickAtRequest(BaseModel):
    x_pct: float = Field(ge=0.0, le=1.0)
    y_pct: float = Field(ge=0.0, le=1.0)
    button: str = Field(default="left", pattern="^(left|right|middle)$")
    # Optional overrides; defaults come from settings.commander.
    screen_width: int | None = Field(default=None, gt=0)
    screen_height: int | None = Field(default=None, gt=0)


class MouseClickRequest(BaseModel):
    button: str = Field(default="left", pattern="^(left|right|middle)$")


class MouseMoveRequest(BaseModel):
    dx: int = Field(ge=-127, le=127)
    dy: int = Field(ge=-127, le=127)


def _content_type_for(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


def _sse(event: str | None, data: Any) -> bytes:
    payload = json.dumps(data) if not isinstance(data, str) else data
    out = []
    if event:
        out.append(f"event: {event}")
    for line in payload.splitlines() or [""]:
        out.append(f"data: {line}")
    out.append("")
    out.append("")
    return ("\n".join(out)).encode()


def create_app(
    context_factory: ContextFactory,
    *,
    frame_store: FrameStore | None = None,
    bus: LogBus | None = None,
    settings: Any = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``context_factory`` is awaited at the start of each run to build a
    fresh AgentContext (mouse/keyboard/capture/etc). The runner closes
    those resources when the run ends — so the webcam is only held for
    the duration of an actual run, not while the server is idle.
    """
    store = frame_store or FrameStore()
    bus = bus or LogBus()
    install_logging(bus)
    runner = Runner(context_factory, bus)

    app = FastAPI(title="terminaleyes command center")
    app.state.store = store
    app.state.bus = bus
    app.state.runner = runner

    # Serializes manual-control webcam captures so a click and a
    # background follow-up snapshot don't fight over the device.
    _manual_capture_lock = asyncio.Lock()

    @app.on_event("startup")
    async def _on_startup() -> None:
        bus.bind_loop(asyncio.get_event_loop())
        await store.start()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await store.stop()

    # ── static / index ───────────────────────────────────────────
    # No-cache headers so iterating on the SPA doesn't require the
    # user to hard-refresh after every server restart. Static files
    # are tiny — caching savings aren't worth the dev friction.
    NO_CACHE = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    if STATIC_DIR.exists():
        class _NoCacheStatic(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                for k, v in NO_CACHE.items():
                    resp.headers[k] = v
                return resp

        app.mount(
            "/static",
            _NoCacheStatic(directory=str(STATIC_DIR)),
            name="static",
        )

    @app.get("/")
    def index() -> FileResponse:
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(404, "index.html not found")
        return FileResponse(str(idx), headers=NO_CACHE)

    # ── scripts download ─────────────────────────────────────────
    # Convenience endpoint so the operator can fetch helper shell
    # scripts from the target machine with a single curl, avoiding
    # scp / USB / etc. round-trips. Only ``.sh`` files in the
    # repo's ``scripts/`` directory are exposed; path traversal is
    # blocked.
    _SCRIPTS_DIR = (
        Path(__file__).parents[3] / "scripts"
    ).resolve()

    @app.get("/scripts/{name}")
    def download_script(name: str) -> FileResponse:
        if "/" in name or ".." in name or not name.endswith(".sh"):
            raise HTTPException(400, "only .sh filenames are allowed")
        path = (_SCRIPTS_DIR / name).resolve()
        try:
            path.relative_to(_SCRIPTS_DIR)
        except ValueError:
            raise HTTPException(400, "path traversal blocked")
        if not path.is_file():
            raise HTTPException(404, f"{name} not found")
        return FileResponse(
            str(path),
            media_type="text/x-shellscript",
            filename=name,
        )

    # ── frames ────────────────────────────────────────────────────
    @app.get("/api/frames")
    def list_frames(
        limit: int = Query(50, ge=1, le=500),
        before: int | None = None,
    ) -> JSONResponse:
        items = store.list(limit=limit, before=before)
        return JSONResponse({
            "count": store.count(),
            "items": [m.public() for m in items],
        })

    @app.get("/api/frames/latest")
    async def latest_frame(
        wait: int = Query(0, ge=0, le=1),
        since: int | None = Query(None),
    ) -> JSONResponse:
        if wait:
            meta = await store.wait_for_update(since)
        else:
            meta = store.latest()
        if meta is None:
            return JSONResponse({"item": None})
        return JSONResponse({"item": meta.public()})

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: int) -> Response:
        meta = store.get(frame_id)
        if meta is None:
            # The id is unknown — most likely a stale id from a
            # previous cc instance (FrameStore rebuilt with fresh
            # mtimes) or evicted from the ring buffer. Tell the
            # client what's actually available so it can resync.
            latest = store.latest()
            raise HTTPException(
                404,
                f"frame id {frame_id} not in store "
                f"(have {store.count()} frames; latest id="
                f"{latest.id if latest else 'none'})",
            )
        try:
            data = Path(meta.path).read_bytes()
        except FileNotFoundError:
            raise HTTPException(410, "frame file gone")
        return Response(content=data, media_type=_content_type_for(meta.path))

    @app.get("/api/frames/{frame_id}/neighbours")
    def frame_neighbours(frame_id: int) -> JSONResponse:
        prev_id, next_id = store.neighbours(frame_id)
        return JSONResponse({"prev": prev_id, "next": next_id})

    # ── runs ──────────────────────────────────────────────────────
    @app.post("/api/run")
    async def start_run(req: RunRequest) -> JSONResponse:
        try:
            record = await runner.start(
                intent=req.intent,
                no_focus=req.no_focus,
                vault=req.vault,
                platform=req.platform,
                dry_run=req.dry_run,
                allow_llm_fallback=req.allow_llm_fallback,
                planner=req.planner,
                ml_adapter=req.ml_adapter,
            )
        except RunnerBusy as e:
            raise HTTPException(409, str(e))
        return JSONResponse(record.public())

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse({
            "items": [r.public() for r in runner.list(limit=limit)],
        })

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        r = runner.get(run_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return JSONResponse(r.public())

    @app.get("/api/runs/{run_id}/logs")
    async def run_logs(
        run_id: str, request: Request,
    ) -> StreamingResponse:
        if runner.get(run_id) is None:
            raise HTTPException(404, "run not found")

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for ev in bus.subscribe_run(run_id, replay=True):
                    if await request.is_disconnected():
                        break
                    yield _sse(None, ev.public())
                yield _sse("done", {"run_id": run_id})
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/logs")
    async def logs_global(
        request: Request, tail: int = Query(200, ge=0, le=2000),
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[bytes]:
            try:
                async for ev in bus.subscribe_global(replay_tail=tail):
                    if await request.is_disconnected():
                        break
                    yield _sse(None, ev.public())
            except asyncio.CancelledError:
                return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── manual mouse control ─────────────────────────────────────
    # Lets the UI drive the host cursor directly: click a point on the
    # screenshot or fire a button. Refuses while a run is in flight
    # because the runner owns the mouse for that window.
    async def _snapshot_after_manual_action(label: str) -> None:
        """Grab one webcam frame and drop it in ``watch_dir/manual/``.

        Without this, manual mouse actions never produce a fresh frame
        in the watch dir, so the UI's long-poll sits on the stale
        last-run screenshot and the operator can't see the cursor
        actually moved. Capture is serialized; the webcam is opened
        and closed per call so a real run can still claim the device.
        """
        if settings is None:
            return
        if runner.is_busy():
            # Capture would race with the run's own webcam handle.
            return
        async with _manual_capture_lock:
            from datetime import datetime
            import cv2
            from terminaleyes.capture.webcam import WebcamCapture
            resolution = None
            if (settings.capture.resolution_width
                    and settings.capture.resolution_height):
                resolution = (
                    settings.capture.resolution_width,
                    settings.capture.resolution_height,
                )
            cap = WebcamCapture(
                device_index=settings.capture.device_index,
                resolution=resolution,
            )
            try:
                await cap.open()
                # Let the cursor settle visually before grabbing.
                await asyncio.sleep(0.25)
                frame = await cap.capture_frame()
            except Exception as e:
                logger.warning("manual snapshot failed: %s", e)
                try:
                    await cap.close()
                except Exception:
                    pass
                return
            try:
                await cap.close()
            except Exception:
                pass
            out_dir = store.watch_dir / "manual"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
            seq = int(datetime.now().timestamp() * 1000) % 10_000
            ts = datetime.now().strftime("%H%M%S")
            path = out_dir / f"{seq:04d}_{ts}_{label}.png"
            try:
                ok = cv2.imwrite(str(path), frame.image)
                if not ok:
                    logger.warning("imwrite returned False for %s", path)
            except Exception as e:
                logger.warning("imwrite failed for %s: %s", path, e)

    def _commander_cfg():
        if settings is None:
            class _Default:
                pi_base_url = "http://10.0.0.2:8080"
                transport = "bt"
                screen_width = 1920
                screen_height = 1080
            return _Default()
        return settings.commander

    async def _with_mouse(action):
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        from terminaleyes.mouse.http_backend import HttpMouseOutput
        cfg = _commander_cfg()
        mouse = HttpMouseOutput(
            base_url=cfg.pi_base_url,
            timeout=10.0,
            transport=cfg.transport,
        )
        try:
            await mouse.connect()
        except Exception as e:
            raise HTTPException(502, f"mouse connect failed: {e}")
        try:
            return await action(mouse)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"mouse action failed: {e}")
        finally:
            try:
                await mouse.disconnect()
            except Exception:
                pass

    def _schedule_snapshot(label: str) -> None:
        asyncio.create_task(_snapshot_after_manual_action(label))

    @app.post("/api/mouse/click_at")
    async def mouse_click_at(req: MouseClickAtRequest) -> JSONResponse:
        """Closed-loop visual-servo click at a webcam-image pixel.

        Routes through ``VisualServoHomer.home_to_pixel`` so the cursor
        is actually homed to the supplied pixel using the same CV that
        the controller uses. Open-loop ``MouseOutput.click_at`` was
        wrong on macOS because BT HID relative moves are subject to
        non-linear pointer acceleration.
        """
        if runner.is_busy():
            raise HTTPException(409, "a run is currently in progress")
        # Late import: pulls heavy CV deps only when actually used.
        from terminaleyes.agents.login import _SessionAdapter
        from terminaleyes.commander.visual_servo_homer import (
            VisualServoHomer,
        )

        ctx = keyboard = mouse = capture = None
        try:
            ctx, keyboard, mouse, capture = await context_factory()
        except Exception as e:
            if capture is not None:
                try: await capture.close()
                except Exception: pass
            if keyboard is not None:
                try: await keyboard.disconnect()
                except Exception: pass
            if mouse is not None:
                try: await mouse.disconnect()
                except Exception: pass
            raise HTTPException(502, f"context_factory failed: {e}")

        try:
            adapter = _SessionAdapter(ctx)
            homer = VisualServoHomer(session=adapter)
            outcome = await homer.home_to_pixel(
                req.x_pct, req.y_pct, button=req.button,
            )
            return JSONResponse({
                "ok": bool(outcome.clicked),
                "reason": outcome.reason,
                "steps": outcome.steps,
                "x_pct": req.x_pct, "y_pct": req.y_pct,
                "button": req.button,
            })
        except Exception as e:
            logger.exception("home_to_pixel failed")
            raise HTTPException(502, f"home_to_pixel failed: {e}")
        finally:
            if capture is not None:
                try: await capture.close()
                except Exception: pass
            if keyboard is not None:
                try: await keyboard.disconnect()
                except Exception: pass
            if mouse is not None:
                try: await mouse.disconnect()
                except Exception: pass

    @app.post("/api/mouse/click")
    async def mouse_click(req: MouseClickRequest) -> JSONResponse:
        async def go(mouse):
            await mouse.click(req.button)
            return JSONResponse({"ok": True, "button": req.button})

        try:
            return await _with_mouse(go)
        finally:
            _schedule_snapshot(f"manual_click_{req.button}")

    @app.post("/api/mouse/move")
    async def mouse_move(req: MouseMoveRequest) -> JSONResponse:
        async def go(mouse):
            await mouse.move(req.dx, req.dy)
            return JSONResponse({"ok": True, "dx": req.dx, "dy": req.dy})

        try:
            return await _with_mouse(go)
        finally:
            _schedule_snapshot("manual_move")

    @app.post("/api/snapshot")
    async def manual_snapshot() -> JSONResponse:
        """Capture a fresh webcam frame on demand (no mouse action)."""
        await _snapshot_after_manual_action("manual_snapshot")
        return JSONResponse({"ok": True})

    @app.get("/api/state")
    def state() -> JSONResponse:
        latest = store.latest()
        active = runner.active()
        cfg = _commander_cfg()
        return JSONResponse({
            "busy": runner.is_busy(),
            "latest_id": latest.id if latest else None,
            "frame_count": store.count(),
            "active_run": active.public() if active else None,
            "screen_width": cfg.screen_width,
            "screen_height": cfg.screen_height,
        })

    return app
