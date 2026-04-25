"""MCP server for screen control via webcam + Raspberry Pi BT HID.

Gives Claude Code direct access to:
- Screenshots from the webcam (Claude sees the screen itself)
- Mouse control via Pi's Bluetooth HID
- Keyboard control via Pi's Bluetooth HID

No LLM in the MCP — Claude Code IS the intelligence.

Usage:
    python -m terminaleyes.mcp_server

Configure in Claude Code settings:
    {
        "mcpServers": {
            "terminaleyes": {
                "command": "python",
                "args": ["-m", "terminaleyes.mcp_server"]
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import base64
import os
import time

import cv2
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("terminaleyes")

# Configuration from environment
PI_BASE_URL = os.environ.get("PI_BASE_URL", "http://10.0.0.2:8080")
PI_TRANSPORT = os.environ.get("PI_TRANSPORT", "bt")
WEBCAM_DEVICE = int(os.environ.get("WEBCAM_DEVICE", "0"))

# Endpoint prefixes
_mouse_prefix = "/bt/mouse" if PI_TRANSPORT == "bt" else "/mouse"
_kb_prefix = "/bt" if PI_TRANSPORT == "bt" else ""

# Lazy-initialized resources
_http_client: httpx.AsyncClient | None = None
_webcam: cv2.VideoCapture | None = None


async def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(base_url=PI_BASE_URL, timeout=10.0)
    return _http_client


def _get_webcam() -> cv2.VideoCapture:
    global _webcam
    if _webcam is None or not _webcam.isOpened():
        _webcam = cv2.VideoCapture(WEBCAM_DEVICE)
        # Warmup for autofocus
        for _ in range(20):
            _webcam.read()
            time.sleep(0.03)
    return _webcam


@mcp.tool()
async def screenshot() -> str:
    """Capture a photo of the target screen via webcam.

    Returns base64-encoded PNG image. Use this to see what's on the
    target screen before deciding where to click or what to type.
    """
    cap = _get_webcam()
    ret, frame = cap.read()
    if not ret or frame is None:
        return "ERROR: Failed to capture frame from webcam"

    # Encode as PNG
    success, buf = cv2.imencode(".png", frame)
    if not success:
        return "ERROR: Failed to encode frame"

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    h, w = frame.shape[:2]
    return f"data:image/png;base64,{b64}"


@mcp.tool()
async def mouse_move(dx: int, dy: int) -> str:
    """Move the mouse cursor by (dx, dy) relative HID units.

    Each value ranges from -127 to 127. Positive dx = right, positive dy = down.
    For larger movements, call multiple times. With macOS mouse acceleration,
    small values (1-5) move ~1:1 with pixels. Larger values get amplified.

    Typical usage: slam to corner first, then move in small steps.
    """
    dx = max(-127, min(127, dx))
    dy = max(-127, min(127, dy))
    client = await _get_client()
    resp = await client.post(f"{_mouse_prefix}/move", json={"x": dx, "y": dy})
    resp.raise_for_status()
    return "OK"


@mcp.tool()
async def mouse_move_smooth(dx_total: int, dy_total: int, step: int = 3) -> str:
    """Move mouse smoothly by (dx_total, dy_total) HID units in small steps.

    Splits the movement into steps of `step` units each with 3ms delay,
    staying in macOS's linear acceleration zone. Use this for calibrated
    movements (e.g., after slamming to corner).

    Example: mouse_move_smooth(500, 300) moves right 500 and down 300.
    """
    client = await _get_client()
    rem_x, rem_y = dx_total, dy_total
    while rem_x != 0 or rem_y != 0:
        sx = max(-step, min(step, rem_x))
        sy = max(-step, min(step, rem_y))
        if sx != 0 or sy != 0:
            resp = await client.post(f"{_mouse_prefix}/move", json={"x": sx, "y": sy})
            resp.raise_for_status()
        rem_x -= sx
        rem_y -= sy
        await asyncio.sleep(0.003)
    return f"OK: moved ({dx_total}, {dy_total})"


@mcp.tool()
async def mouse_click(button: str = "left") -> str:
    """Click a mouse button: left, right, or middle."""
    client = await _get_client()
    resp = await client.post(f"{_mouse_prefix}/click", json={"button": button})
    resp.raise_for_status()
    return f"OK: clicked {button}"


@mcp.tool()
async def mouse_slam_corner() -> str:
    """Slam the cursor to the top-left corner of the screen.

    Sends 200 fast moves of (-20, -20) to force the cursor to (0,0),
    regardless of current position. Use this as a known starting point
    before calibrated movements.
    """
    client = await _get_client()
    for _ in range(200):
        await client.post(f"{_mouse_prefix}/move", json={"x": -20, "y": -20})
        await asyncio.sleep(0.001)
    return "OK: cursor at top-left corner"


@mcp.tool()
async def mouse_scroll(amount: int) -> str:
    """Scroll the mouse wheel. Positive = up, negative = down."""
    client = await _get_client()
    resp = await client.post(f"{_mouse_prefix}/scroll", json={"amount": amount})
    resp.raise_for_status()
    return f"OK: scrolled {amount}"


@mcp.tool()
async def type_text(text: str) -> str:
    """Type text on the target machine."""
    client = await _get_client()
    resp = await client.post(f"{_kb_prefix}/text", json={"text": text})
    resp.raise_for_status()
    return f"OK: typed {len(text)} chars"


@mcp.tool()
async def press_key(key: str) -> str:
    """Press a single key: Enter, Tab, Escape, Space, Backspace, Delete,
    Up, Down, Left, Right, Home, End, PageUp, PageDown, F1-F12, etc."""
    client = await _get_client()
    resp = await client.post(f"{_kb_prefix}/keystroke", json={"key": key})
    resp.raise_for_status()
    return f"OK: pressed {key}"


@mcp.tool()
async def key_combo(modifiers: list[str], key: str) -> str:
    """Press a key combination. Modifiers: ctrl, shift, alt, meta.
    Examples: key_combo(["meta"], "c") for Cmd+C, key_combo(["ctrl", "shift"], "z")."""
    client = await _get_client()
    resp = await client.post(
        f"{_kb_prefix}/key-combo",
        json={"modifiers": modifiers, "key": key},
    )
    resp.raise_for_status()
    return f"OK: pressed {'+'.join(modifiers)}+{key}"


@mcp.tool()
async def pi_health() -> str:
    """Check the Pi's connection status and HID device state."""
    client = await _get_client()
    resp = await client.get("/health")
    return resp.text


if __name__ == "__main__":
    mcp.run()
