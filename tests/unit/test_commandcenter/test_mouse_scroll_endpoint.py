"""Tests for the cc ``POST /api/mouse/scroll`` endpoint.

These cover both the server-side wiring (request validation + Pi
forwarding) and the JS-side wheel handler (loaded via a headless
browser fixture, see :mod:`tests.unit.test_commandcenter.test_mouse_scroll_ui`).

What's being validated here:

  1. **Happy path** — positive ``amount`` reaches the mocked
     ``HttpMouseOutput.scroll``. Response carries echo of inputs.
  2. **Negative direction** — works the same; sign preserved end to
     end (browsers send a negative ``deltaY`` for scroll-up, the
     Pi-side convention is "negative = up").
  3. **Position passthrough** — ``x_pct`` / ``y_pct`` are accepted
     and round-trip in the response. They're informational today
     (snapshot label only), but the contract has to be stable so
     a future home-then-scroll path can use them.
  4. **Validation** — amounts outside ``[-30, 30]`` reject with
     422 from pydantic; same for ``x_pct`` / ``y_pct`` out of
     ``[0, 1]``.
  5. **Runner-busy guard** — POST returns 409 when a controller
     run is in progress (same contract as click / move).
  6. **Pi error surfacing** — when ``HttpMouseOutput.scroll``
     raises, the endpoint maps it to 502.

The webcam isn't actually used by ``/api/mouse/scroll`` — we
mock it out anyway so the post-scroll snapshot doesn't error
trying to talk to a real camera.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from terminaleyes.commandcenter.frame_store import FrameStore
from terminaleyes.commandcenter.log_bus import LogBus
from terminaleyes.commandcenter.server import create_app


# ── settings stub ───────────────────────────────────────────────────
class _CaptureCfg:
    device_index = 0
    resolution_width = None
    resolution_height = None


class _CommanderCfg:
    pi_base_url = "http://127.0.0.1:9"
    transport = "usb"
    screen_width = 1920
    screen_height = 1080


class _Settings:
    capture = _CaptureCfg()
    commander = _CommanderCfg()


# ── fixtures ────────────────────────────────────────────────────────
@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / "watch"
    d.mkdir()
    return d


@pytest.fixture
def store(watch_dir):
    return FrameStore(watch_dir=watch_dir, max_frames=10)


@pytest.fixture
def bus():
    return LogBus()


@pytest.fixture
def mouse_log() -> list[tuple[str, dict]]:
    """Captures every call into the mocked mouse backend."""
    return []


@pytest.fixture
def mock_mouse_cls(mouse_log, request):
    """Recording HttpMouseOutput drop-in. The class itself is what
    gets patched onto the module-level import; instantiating it
    yields an object whose ``scroll`` (and other methods) append
    one entry per call to ``mouse_log``.

    Pass ``scroll_raises`` via indirect param to make the next
    ``scroll`` call raise a chosen exception (validates the 502
    error path)."""
    scroll_raises: Exception | None = getattr(request, "param", None)

    class _RecordingMouse:
        def __init__(self, *a, **kw):
            mouse_log.append(("init", {"args": a, "kwargs": kw}))

        async def connect(self):
            mouse_log.append(("connect", {}))

        async def disconnect(self):
            mouse_log.append(("disconnect", {}))

        async def scroll(self, amount: int):
            mouse_log.append(("scroll", {"amount": amount}))
            if scroll_raises is not None:
                raise scroll_raises

        async def click(self, button: str = "left"):
            mouse_log.append(("click", {"button": button}))

        async def move(self, dx: int, dy: int):
            mouse_log.append(("move", {"dx": dx, "dy": dy}))

    return _RecordingMouse


@pytest.fixture
def mock_capture_cls():
    class _ConstantCapture:
        def __init__(self, *a, **kw):
            pass

        async def open(self):
            return None

        async def close(self):
            return None

        async def capture_frame(self):
            from terminaleyes.domain.models import CapturedFrame
            return CapturedFrame(
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                frame_number=1,
            )

    return _ConstantCapture


@contextmanager
def patched_runtime(mock_mouse_cls, mock_capture_cls):
    async def _instant_sleep(_secs, *a, **kw):
        return None

    with patch(
        "terminaleyes.mouse.http_backend.HttpMouseOutput",
        mock_mouse_cls,
    ), patch(
        "terminaleyes.capture.webcam.WebcamCapture",
        mock_capture_cls,
    ), patch(
        "terminaleyes.commandcenter.server.asyncio.sleep",
        side_effect=_instant_sleep,
    ):
        yield


def _build_client(store, bus):
    async def _factory():
        return AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()

    app = create_app(
        _factory, frame_store=store, bus=bus, settings=_Settings(),
    )
    return TestClient(app)


# ── tests ───────────────────────────────────────────────────────────
def test_scroll_positive_amount(
    store, bus, mouse_log, mock_mouse_cls, mock_capture_cls,
):
    """Fast path: amount=3 fans out to 3 × scroll(+1) on the Pi.
    macOS HID stacks treat sequential single-tick reports as a
    gesture and apply real scroll acceleration; one big
    scroll(N) report it caps to a tiny visual scroll instead."""
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        r = client.post(
            "/api/mouse/scroll", json={"amount": 3},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["amount"] == 3
    assert body["ticks_sent"] == 3
    assert body["homed"] is False
    scroll_calls = [c for c in mouse_log if c[0] == "scroll"]
    # Three sequential single-tick reports, not one scroll(3).
    assert scroll_calls == [
        ("scroll", {"amount": 1}),
        ("scroll", {"amount": 1}),
        ("scroll", {"amount": 1}),
    ]


def test_scroll_with_cached_position_skips_home(
    store, bus, mouse_log, mock_mouse_cls, mock_capture_cls,
):
    """When the operator hovers within tolerance of the cached
    last-home position, the slow homer path is skipped."""
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        # Prime the cache so the position lookup is a hit.
        client.app.state.last_scroll_home_xy = (0.5, 0.5)
        r = client.post(
            "/api/mouse/scroll",
            json={"amount": 2, "x_pct": 0.51, "y_pct": 0.49},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["homed"] is False
    assert body["ticks_sent"] == 2
    scroll_calls = [c for c in mouse_log if c[0] == "scroll"]
    assert scroll_calls == [
        ("scroll", {"amount": 1}),
        ("scroll", {"amount": 1}),
    ]


def test_scroll_negative_amount(
    store, bus, mouse_log, mock_mouse_cls, mock_capture_cls,
):
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        r = client.post(
            "/api/mouse/scroll", json={"amount": -5},
        )
    assert r.status_code == 200
    assert r.json()["amount"] == -5
    assert r.json()["ticks_sent"] == 5
    scroll_calls = [c for c in mouse_log if c[0] == "scroll"]
    # Five single-tick reports, negative direction.
    assert scroll_calls == [("scroll", {"amount": -1})] * 5


def test_scroll_position_optional(
    store, bus, mouse_log, mock_mouse_cls, mock_capture_cls,
):
    """The endpoint must accept a scroll with no position fields.
    No position → no homing → fast path, no x_pct/y_pct echoed."""
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        r = client.post("/api/mouse/scroll", json={"amount": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["x_pct"] is None
    assert body["y_pct"] is None
    assert body["homed"] is False


@pytest.mark.parametrize("bad", [
    {"amount": 1, "x_pct": 1.5},          # x_pct > 1
    {"amount": 1, "y_pct": -0.1},         # y_pct < 0
    {"amount": 99},                       # amount > 30
    {"amount": -99},                      # amount < -30
    {},                                   # missing required amount
    {"amount": "two"},                    # non-int
])
def test_scroll_validation_rejects(
    bad, store, bus, mock_mouse_cls, mock_capture_cls,
):
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        r = client.post("/api/mouse/scroll", json=bad)
    assert r.status_code == 422, (bad, r.text)


def test_scroll_blocked_when_runner_busy(
    store, bus, mock_mouse_cls, mock_capture_cls,
):
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        # Force the runner into a "busy" state via the same surface the
        # endpoint checks (Runner.is_busy() is True iff _active is set).
        client.app.state.runner._active = MagicMock(run_id="abc")
        r = client.post("/api/mouse/scroll", json={"amount": 1})
    assert r.status_code == 409
    assert "in progress" in r.text


@pytest.mark.parametrize(
    "mock_mouse_cls", [RuntimeError("usb cable yanked")], indirect=True,
)
def test_scroll_pi_error_returns_502(
    store, bus, mouse_log, mock_mouse_cls, mock_capture_cls,
):
    with patched_runtime(mock_mouse_cls, mock_capture_cls):
        client = _build_client(store, bus)
        r = client.post("/api/mouse/scroll", json={"amount": 1})
    assert r.status_code == 502
    assert "usb cable yanked" in r.text
    # Still attempted the scroll on the mocked Pi, and still tore down
    # the mouse connection so it's not leaked on failure.
    assert any(c[0] == "scroll" for c in mouse_log)
    assert any(c[0] == "disconnect" for c in mouse_log)
