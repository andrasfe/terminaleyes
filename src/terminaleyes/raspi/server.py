"""REST API server that runs on the Raspberry Pi.

Accepts keyboard commands over HTTP and translates them into USB HID
reports via HidWriter. This is the bridge between the terminaleyes
agent (running on another machine) and the target machine the Pi is
plugged into.

Endpoints match the same contract as endpoint/server.py so the
existing HttpKeyboardOutput client works unchanged:

    GET  /health        -> {"status": "ok", ...}
    POST /keystroke     <- {"key": "Enter"}
    POST /key-combo     <- {"modifiers": ["ctrl"], "key": "c"}
    POST /text          <- {"text": "ls -la"}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from terminaleyes.raspi.hid_writer import HidWriteError, HidWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class KeystrokeRequest(BaseModel):
    key: str = Field(description="Key name (e.g., 'Enter', 'Tab', 'a')")


class KeyComboRequest(BaseModel):
    modifiers: list[str] = Field(description="Modifier keys (e.g., ['ctrl'])")
    key: str = Field(description="Main key in the combination")


class TextInputRequest(BaseModel):
    text: str = Field(description="Text to type")


class HealthResponse(BaseModel):
    status: str = "ok"
    hid_device: str = "/dev/hidg0"
    hid_open: bool = False


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    hid_device: str = "/dev/hidg0",
    keypress_delay: float = 0.02,
    inter_char_delay: float = 0.01,
    writer: HidWriter | None = None,
) -> FastAPI:
    """Create the Pi REST API application.

    Args:
        hid_device: Path to the HID gadget device.
        keypress_delay: Seconds between key press and release.
        inter_char_delay: Seconds between characters when typing text.
        writer: Optional pre-configured HidWriter (for testing).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        w = app.state.writer
        if w is None:
            w = HidWriter(
                device_path=hid_device,
                keypress_delay=keypress_delay,
                inter_char_delay=inter_char_delay,
            )
            app.state.writer = w
        await w.open()
        logger.info("Pi keyboard server started (device=%s)", hid_device)
        yield
        await w.close()
        logger.info("Pi keyboard server stopped")

    app = FastAPI(
        title="terminaleyes Pi Keyboard",
        description="Raspberry Pi USB HID keyboard REST API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.writer = writer

    @app.get("/health")
    async def health_check() -> HealthResponse:
        w: HidWriter = app.state.writer
        return HealthResponse(
            status="ok",
            hid_device=hid_device,
            hid_open=w.is_open if w else False,
        )

    @app.post("/keystroke")
    async def receive_keystroke(request: KeystrokeRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_keystroke(request.key)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "key": request.key}

    @app.post("/key-combo")
    async def receive_key_combo(request: KeyComboRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_key_combo(request.modifiers, request.key)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "combo": f"{'+'.join(request.modifiers)}+{request.key}"}

    @app.post("/text")
    async def receive_text(request: TextInputRequest) -> dict[str, str]:
        w: HidWriter = app.state.writer
        try:
            await w.send_text(request.text)
        except (ValueError, HidWriteError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"status": "ok", "length": str(len(request.text))}

    return app


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    hid_device: str = "/dev/hidg0",
) -> None:
    """Run the Pi keyboard server."""
    app = create_app(hid_device=hid_device)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
