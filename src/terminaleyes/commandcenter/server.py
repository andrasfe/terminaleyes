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
            raise HTTPException(404, "frame not found")
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

    @app.get("/api/state")
    def state() -> JSONResponse:
        latest = store.latest()
        active = runner.active()
        return JSONResponse({
            "busy": runner.is_busy(),
            "latest_id": latest.id if latest else None,
            "frame_count": store.count(),
            "active_run": active.public() if active else None,
        })

    return app
