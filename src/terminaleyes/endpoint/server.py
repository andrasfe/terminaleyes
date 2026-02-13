"""FastAPI HTTP server for the local command endpoint.

Receives keyboard action requests via HTTP and forwards them to
the persistent shell session. Also manages the terminal display
window that the webcam captures.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from terminaleyes.endpoint.display import TerminalDisplay
from terminaleyes.endpoint.shell import PersistentShell

logger = logging.getLogger(__name__)


class KeystrokeRequest(BaseModel):
    key: str = Field(description="Key name (e.g., 'Enter', 'Tab', 'a')")


class KeyComboRequest(BaseModel):
    modifiers: list[str] = Field(description="Modifier keys (e.g., ['ctrl'])")
    key: str = Field(description="Main key in the combination")


class TextInputRequest(BaseModel):
    text: str = Field(description="Text to type")


class EndpointStatus(BaseModel):
    status: str = "ok"
    shell_alive: bool = True
    display_active: bool = True


# Key name to character mapping
KEY_MAP = {
    "Enter": "\n",
    "Return": "\n",
    "Tab": "\t",
    "Space": " ",
    "Backspace": "\x7f",
    "Delete": "\x1b[3~",
    "Escape": "\x1b",
    "Up": "\x1b[A",
    "Down": "\x1b[B",
    "Right": "\x1b[C",
    "Left": "\x1b[D",
    "Home": "\x1b[H",
    "End": "\x1b[F",
}

# Ctrl key combos to control characters / signals
CTRL_MAP = {
    "c": "SIGINT",
    "d": "EOF",
    "z": "SIGTSTP",
    "l": "\x0c",  # Form feed (clear)
}


def create_app(
    shell: PersistentShell | None = None,
    display: TerminalDisplay | None = None,
    shell_command: str = "/bin/bash",
    rows: int = 24,
    cols: int = 80,
    font_size: int = 24,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    fg_color: tuple[int, int, int] = (192, 192, 192),
    fullscreen: bool = True,
    window_x: int | None = None,
    window_y: int | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        s = app.state.shell
        d = app.state.display
        if s is None:
            s = PersistentShell(shell_command=shell_command, rows=rows, cols=cols)
            app.state.shell = s
        if d is None:
            d = TerminalDisplay(
                rows=s.rows, cols=s.cols,
                font_size=font_size, bg_color=bg_color, fg_color=fg_color,
                fullscreen=fullscreen,
                window_x=window_x, window_y=window_y,
            )
            app.state.display = d
        await s.start()
        d.start()
        # Start display refresh task
        app.state.refresh_task = asyncio.create_task(_refresh_display(s, d))
        logger.info("Endpoint started (shell + display)")
        yield
        # Shutdown
        app.state.refresh_task.cancel()
        try:
            await app.state.refresh_task
        except asyncio.CancelledError:
            pass
        d.stop()
        await s.stop()
        logger.info("Endpoint stopped")

    app = FastAPI(
        title="terminaleyes Endpoint",
        description="Local HTTP command endpoint for the terminaleyes agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.shell = shell
    app.state.display = display

    @app.get("/health")
    async def health_check() -> EndpointStatus:
        return EndpointStatus(
            status="ok",
            shell_alive=app.state.shell.is_alive if app.state.shell else False,
            display_active=app.state.display.is_active if app.state.display else False,
        )

    @app.post("/keystroke")
    async def receive_keystroke(request: KeystrokeRequest) -> dict[str, str]:
        s: PersistentShell = app.state.shell
        key = request.key
        char = KEY_MAP.get(key, key if len(key) == 1 else None)
        if char is None:
            return {"status": "ignored", "reason": f"Unknown key: {key}"}
        await s.send_input(char)
        return {"status": "ok", "key": key}

    @app.post("/key-combo")
    async def receive_key_combo(request: KeyComboRequest) -> dict[str, str]:
        s: PersistentShell = app.state.shell
        if "ctrl" in [m.lower() for m in request.modifiers]:
            key_lower = request.key.lower()
            signal_or_char = CTRL_MAP.get(key_lower)
            if signal_or_char:
                if signal_or_char in ("SIGINT", "SIGTSTP", "EOF"):
                    await s.send_signal(signal_or_char)
                else:
                    await s.send_input(signal_or_char)
                return {"status": "ok", "combo": f"ctrl+{request.key}"}
            # Generic ctrl+key: send control character
            if len(key_lower) == 1 and key_lower.isalpha():
                ctrl_char = chr(ord(key_lower) - ord("a") + 1)
                await s.send_input(ctrl_char)
                return {"status": "ok", "combo": f"ctrl+{request.key}"}
        return {"status": "ignored", "reason": "Unsupported combo"}

    @app.post("/text")
    async def receive_text(request: TextInputRequest) -> dict[str, str]:
        s: PersistentShell = app.state.shell
        await s.send_input(request.text)
        return {"status": "ok", "length": str(len(request.text))}

    @app.get("/screen")
    async def get_screen_content() -> dict[str, str]:
        s: PersistentShell = app.state.shell
        return {"content": s.get_screen_content()}

    return app


async def _refresh_display(shell: PersistentShell, display: TerminalDisplay) -> None:
    """Periodically update the display with shell content."""
    while True:
        try:
            content = shell.get_screen_content()
            display.update_content(content)
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("Display refresh error: %s", e)
            await asyncio.sleep(0.5)


def main() -> None:
    """Entry point for running the endpoint server standalone."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
