"""Playwright integration test for the wheel → /api/mouse/scroll path.

Spins up the real cc FastAPI app on a free port (uvicorn) with a
stubbed ``HttpMouseOutput`` so no Pi is required, opens the UI in
Chromium, dispatches synthetic ``wheel`` events on the screenshot
element, and asserts the right POST body arrives at
``/api/mouse/scroll`` with correctly-mapped ticks + normalised
hover position.

Run with::

    pip install pytest-playwright playwright
    playwright install chromium
    pytest tests/integration/test_mouse_scroll_browser.py -v

The test is in ``tests/integration/`` (not ``tests/unit/``) because
it requires Chromium and a real network port. CI gates skip it
when ``playwright`` isn't installed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright

import numpy as np
import uvicorn

from terminaleyes.commandcenter.frame_store import FrameStore
from terminaleyes.commandcenter.log_bus import LogBus
from terminaleyes.commandcenter.server import create_app


class _CommanderCfg:
    pi_base_url = "http://127.0.0.1:9"
    transport = "usb"
    screen_width = 1920
    screen_height = 1080


class _CaptureCfg:
    device_index = 0
    resolution_width = None
    resolution_height = None


class _Settings:
    capture = _CaptureCfg()
    commander = _CommanderCfg()


def _make_recording_mouse_cls(log: list[tuple[str, dict]]):
    class _RecordingMouse:
        def __init__(self, *a, **kw):
            log.append(("init", {}))

        async def connect(self):
            log.append(("connect", {}))

        async def disconnect(self):
            log.append(("disconnect", {}))

        async def scroll(self, amount: int):
            log.append(("scroll", {"amount": amount}))

        async def click(self, button="left"):
            log.append(("click", {"button": button}))

        async def move(self, dx, dy):
            log.append(("move", {"dx": dx, "dy": dy}))

    return _RecordingMouse


def _make_constant_capture_cls():
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
def _serve_cc(tmp_path: Path, mouse_log: list, port: int):
    """Run the cc app on a real port in a daemon thread for the
    lifetime of the with-block. Mocks the Pi-side mouse so no
    hardware is needed."""
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    store = FrameStore(watch_dir=watch_dir, max_frames=10)
    bus = LogBus()

    async def _factory():
        from unittest.mock import AsyncMock
        return AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()

    mouse_cls = _make_recording_mouse_cls(mouse_log)
    capture_cls = _make_constant_capture_cls()

    with patch(
        "terminaleyes.mouse.http_backend.HttpMouseOutput", mouse_cls,
    ), patch(
        "terminaleyes.capture.webcam.WebcamCapture", capture_cls,
    ):
        app = create_app(
            _factory, frame_store=store, bus=bus, settings=_Settings(),
        )
        # Pre-prime the scroll-home cache so the position-aware
        # wheel POSTs in these tests hit the fast path (just send
        # mouse.scroll without invoking the visual-servo homer,
        # which would require real webcam + opencv). The slow
        # home-then-scroll path is exercised separately in the
        # unit tests via mocked factories.
        app.state.last_scroll_home_xy = (0.5, 0.5)

        config = uvicorn.Config(
            app, host="127.0.0.1", port=port,
            log_level="warning", loop="asyncio",
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(
            target=server.run, daemon=True, name="cc-test-uvicorn",
        )
        thread.start()

        # Wait until the server is actually accepting connections.
        deadline = time.time() + 8.0
        import socket
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("cc did not come up within 8s")

        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5.0)


def _pick_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_wheel_over_screenshot_posts_scroll(tmp_path):
    """End-to-end: synthetic wheel events on #frame produce
    /api/mouse/scroll POSTs whose `amount` and (x_pct, y_pct)
    match what the JS coalescer derived from deltaY + element
    geometry."""
    mouse_log: list[tuple[str, dict]] = []
    port = _pick_free_port()
    with _serve_cc(tmp_path, mouse_log, port) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                )
                page = ctx.new_page()

                # Capture scroll POST bodies via route interception so
                # we can assert the request shape without touching
                # disk / FrameStore plumbing.
                seen: list[dict] = []

                def _intercept(route):
                    try:
                        body = route.request.post_data_json or {}
                        seen.append(body)
                    except Exception:
                        pass
                    route.continue_()

                page.route("**/api/mouse/scroll", _intercept)

                page.goto(base_url, wait_until="domcontentloaded")
                # Ensure the wheel handler is wired and the test
                # hook is attached.
                page.wait_for_function(
                    "typeof window.__teTest === 'object' "
                    "&& typeof window.__teTest.flushScroll === 'function'",
                    timeout=5_000,
                )

                # ── Stage the page so wheel events on #frame are
                # routed to /api/mouse/scroll (gated on the toggle
                # the click-handler also uses; we tick it on, and
                # we fake a non-empty frame so the handler doesn't
                # early-return on the .empty class).
                # Stage the page: tick click-to-move on, attach a
                # tiny inline PNG so the <img> has a real
                # naturalWidth/Height (imageRect returns null without
                # them), and force the element to a known size.
                # 160×90 PNG (16:9), aspect-matched to the 800×450
                # element so imageRect() fills the box and offsets
                # are zero. With a square tiny image, imageRect would
                # report a centred-in-element rect and our clientX
                # offsets would land OUTSIDE the rendered image, so
                # the handler wouldn't update _wheelLastPos.
                _TINY_PNG = (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAKAAAABaCAIAAACwpMoFAAAA60lEQVR4"
                    "nO3RAQkAIADAMDWNIeyfyxQinC3B4XPvM+havwN4y+A4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjN4tF2eMgFA/IQ9PAAAAABJRU5ErkJggg=="
                )
                page.evaluate(
                    f"""(src) => new Promise((resolve) => {{
                        const cb = document.getElementById('opt-click-to-move');
                        if (cb && !cb.checked) cb.click();
                        const f = document.getElementById('frame');
                        f.classList.remove('empty');
                        f.style.width  = '800px';
                        f.style.height = '450px';
                        f.style.display = 'block';
                        f.onload = () => resolve(true);
                        f.src = src;
                        // Some browsers may already have it cached
                        if (f.complete && f.naturalWidth > 0) resolve(true);
                    }})""",
                    _TINY_PNG,
                )
                # Sanity: naturalWidth must now be > 0 so imageRect
                # resolves to a real rect.
                page.wait_for_function(
                    "() => {"
                    "  const f = document.getElementById('frame');"
                    "  return f && f.naturalWidth > 0;"
                    "}",
                    timeout=4_000,
                )

                # ── Dispatch three wheel events that should coalesce
                # into one POST of amount=+3 at (~0.5, ~0.5).
                # Position coords are computed from the IMAGE's
                # rendered rect inside the element (imageRect, the
                # same function the wheel handler uses), not from the
                # element rect — when image aspect doesn't fill the
                # element, the rendered image is centred with
                # offsets and a naive "centre of element" coordinate
                # can land outside the image area.
                _DISPATCH_RECT_JS = """() => {
                    const f = document.getElementById('frame');
                    const w = f.naturalWidth, h = f.naturalHeight;
                    const box = f.getBoundingClientRect();
                    const scale = Math.min(box.width / w, box.height / h);
                    const renderedW = w * scale;
                    const renderedH = h * scale;
                    return {
                        left: box.left + (box.width - renderedW) / 2,
                        top:  box.top  + (box.height - renderedH) / 2,
                        width: renderedW,
                        height: renderedH,
                    };
                }"""
                page.evaluate(
                    f"""() => {{
                        const rect = ({_DISPATCH_RECT_JS})();
                        const cx = rect.left + rect.width  / 2;
                        const cy = rect.top  + rect.height / 2;
                        const f = document.getElementById('frame');
                        for (let i = 0; i < 3; i++) {{
                            f.dispatchEvent(new WheelEvent('wheel', {{
                                bubbles: true, cancelable: true,
                                deltaY: 100,
                                clientX: cx, clientY: cy,
                            }}));
                        }}
                    }}"""
                )

                # The handler coalesces with a 100 ms timer; give it
                # well over that to land + finish the POST.
                page.wait_for_function(
                    "() => window.__teTest "
                    "&& !window.__teTest.peekScrollState().flushing "
                    "&& window.__teTest.peekScrollState().px === 0",
                    timeout=4_000,
                )
                # Cooperative final flush in case the request hasn't
                # rendered into route.continue_ yet.
                page.evaluate(
                    "() => window.__teTest.flushScroll()"
                )
                page.wait_for_timeout(200)

                assert len(seen) >= 1, "expected at least one /api/mouse/scroll POST"
                # 3 wheel events × deltaY=100 = 300 px. At 30 px per
                # tick that's 10 ticks, which also happens to be the
                # WHEEL_MAX_TICKS_PER_POST clamp — so we expect one
                # POST with amount = 10.
                first = seen[0]
                assert first["amount"] == 10, first
                # Position: should be near the middle of the element.
                assert 0.45 <= first["x_pct"] <= 0.55, first
                assert 0.45 <= first["y_pct"] <= 0.55, first

                # And the request actually reached the mocked Pi.
                scrolls = [c for c in mouse_log if c[0] == "scroll"]
                assert scrolls, "no scroll() call reached the Pi mock"
                assert sum(c[1]["amount"] for c in scrolls) == 10

                # ── Negative direction works too. Dispatch at a
                # point we know is inside the rendered image — pick
                # 10% in from the top-left so the expected
                # normalised coords are ~0.10 regardless of layout.
                # Confirm the wheel handler actually updated
                # _wheelLastPos via the test hook before the flush
                # consumes the accumulated deltaY.
                seen.clear()
                # Use a deltaY that divides cleanly by
                # WHEEL_PX_PER_TICK (30) so the wait below can
                # require px === 0 with no leftover remainder.
                # -180 = -6 ticks, remainder 0.
                page.evaluate(
                    f"""() => {{
                        const rect = ({_DISPATCH_RECT_JS})();
                        const cx = rect.left + rect.width * 0.10;
                        const cy = rect.top  + rect.height * 0.10;
                        document.getElementById('frame').dispatchEvent(
                            new WheelEvent('wheel', {{
                                bubbles: true, cancelable: true,
                                deltaY: -180,
                                clientX: cx, clientY: cy,
                            }})
                        );
                    }}"""
                )
                page.wait_for_function(
                    "() => window.__teTest.peekScrollState().px === 0 "
                    "&& !window.__teTest.peekScrollState().flushing",
                    timeout=4_000,
                )
                page.evaluate("() => window.__teTest.flushScroll()")
                page.wait_for_timeout(200)

                assert seen, "negative wheel didn't fire a POST"
                # -180 px ÷ 30 px/tick = -6 ticks (negative direction).
                neg = seen[0]
                assert neg["amount"] == -6, neg
                # Position should be ~0.10 (we dispatched 10 % in).
                assert 0.05 <= neg["x_pct"] <= 0.20, neg
                assert 0.05 <= neg["y_pct"] <= 0.20, neg

            finally:
                browser.close()


def test_wheel_fires_regardless_of_click_to_move(tmp_path):
    """Scroll-over-screenshot is unambiguous intent, so unlike
    ``click_at`` we don't gate on the ``click-to-move`` toggle.
    A wheel event fires the POST whether the toggle is on or off."""
    mouse_log: list[tuple[str, dict]] = []
    port = _pick_free_port()
    with _serve_cc(tmp_path, mouse_log, port) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                )
                page = ctx.new_page()
                seen = []
                page.route(
                    "**/api/mouse/scroll",
                    lambda route: (seen.append(
                        route.request.post_data_json or {}
                    ), route.continue_()),
                )

                page.goto(base_url, wait_until="domcontentloaded")
                page.wait_for_function(
                    "typeof window.__teTest === 'object'",
                    timeout=5_000,
                )

                # Ensure toggle is OFF; set up #frame visually.
                # 160×90 PNG (16:9), aspect-matched to the 800×450
                # element so imageRect() fills the box and offsets
                # are zero. With a square tiny image, imageRect would
                # report a centred-in-element rect and our clientX
                # offsets would land OUTSIDE the rendered image, so
                # the handler wouldn't update _wheelLastPos.
                _TINY_PNG = (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAKAAAABaCAIAAACwpMoFAAAA60lEQVR4"
                    "nO3RAQkAIADAMDWNIeyfyxQinC3B4XPvM+havwN4y+A4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjN4tF2eMgFA/IQ9PAAAAABJRU5ErkJggg=="
                )
                page.evaluate(
                    f"""(src) => new Promise((resolve) => {{
                        const cb = document.getElementById('opt-click-to-move');
                        if (cb && cb.checked) cb.click();
                        const f = document.getElementById('frame');
                        f.classList.remove('empty');
                        f.style.width  = '800px';
                        f.style.height = '450px';
                        f.style.display = 'block';
                        f.onload = () => resolve(true);
                        f.src = src;
                        if (f.complete && f.naturalWidth > 0) resolve(true);
                    }})""",
                    _TINY_PNG,
                )
                page.wait_for_function(
                    "() => document.getElementById('frame').naturalWidth > 0",
                    timeout=4_000,
                )
                # Compute rect from the rendered image, not the
                # element box, so the dispatched point is inside the
                # active hit-region after parent-flex squashing.
                _DISPATCH_RECT_JS = """() => {
                    const f = document.getElementById('frame');
                    const w = f.naturalWidth, h = f.naturalHeight;
                    const box = f.getBoundingClientRect();
                    const scale = Math.min(box.width / w, box.height / h);
                    return {
                        left: box.left + (box.width - w * scale) / 2,
                        top:  box.top  + (box.height - h * scale) / 2,
                        width: w * scale, height: h * scale,
                    };
                }"""
                page.evaluate(
                    f"""() => {{
                        const rect = ({_DISPATCH_RECT_JS})();
                        const f = document.getElementById('frame');
                        f.dispatchEvent(new WheelEvent('wheel', {{
                            bubbles: true, cancelable: true,
                            deltaY: 90,
                            clientX: rect.left + rect.width / 2,
                            clientY: rect.top + rect.height / 2,
                        }}));
                    }}"""
                )
                page.wait_for_function(
                    "() => window.__teTest && "
                    "  !window.__teTest.peekScrollState().flushing && "
                    "  window.__teTest.peekScrollState().px === 0",
                    timeout=4_000,
                )
                page.wait_for_timeout(200)

                # New semantics: scroll fires even with the toggle
                # off. The POST should have landed and the busy
                # hourglass should NOT still be showing post-flush.
                assert seen, (
                    "wheel handler should have fired even with the "
                    "click-to-move toggle off; no POST was seen."
                )
                assert seen[0]["amount"] >= 1
                # Pi mock saw the scroll.
                assert [c for c in mouse_log if c[0] == "scroll"]
            finally:
                browser.close()


def test_wheel_deltamode_line_normalised(tmp_path):
    """Real mouse-wheel events arrive with ``deltaMode=1``
    (DOM_DELTA_LINE) and small ``deltaY`` values like ±3 (lines,
    not pixels). The handler must scale those up so a normal
    wheel-notch produces a /api/mouse/scroll POST instead of
    silently piling under the 30-px threshold."""
    mouse_log: list[tuple[str, dict]] = []
    port = _pick_free_port()
    with _serve_cc(tmp_path, mouse_log, port) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                )
                page = ctx.new_page()
                seen = []
                page.route(
                    "**/api/mouse/scroll",
                    lambda route: (seen.append(
                        route.request.post_data_json or {}
                    ), route.continue_()),
                )

                page.goto(base_url, wait_until="domcontentloaded")
                page.wait_for_function(
                    "typeof window.__teTest === 'object'",
                    timeout=5_000,
                )
                _TINY_PNG = (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAKAAAABaCAIAAACwpMoFAAAA60lEQVR4"
                    "nO3RAQkAIADAMDWNIeyfyxQinC3B4XPvM+havwN4y+A4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjN4tF2eMgFA/IQ9PAAAAABJRU5ErkJggg=="
                )
                page.evaluate(
                    f"""(src) => new Promise((resolve) => {{
                        const f = document.getElementById('frame');
                        f.classList.remove('empty');
                        f.style.width  = '800px';
                        f.style.height = '450px';
                        f.style.display = 'block';
                        f.onload = () => resolve(true);
                        f.src = src;
                        if (f.complete && f.naturalWidth > 0) resolve(true);
                    }})""",
                    _TINY_PNG,
                )
                page.wait_for_function(
                    "() => document.getElementById('frame').naturalWidth > 0",
                    timeout=4_000,
                )
                # One mouse-wheel-notch in DOM_DELTA_LINE mode:
                # deltaMode=1, deltaY=3. Without normalisation this
                # would add 3 px to the accumulator and never flush.
                _DISPATCH_RECT_JS = """() => {
                    const f = document.getElementById('frame');
                    const w = f.naturalWidth, h = f.naturalHeight;
                    const box = f.getBoundingClientRect();
                    const scale = Math.min(box.width / w, box.height / h);
                    return {
                        left: box.left + (box.width - w * scale) / 2,
                        top:  box.top  + (box.height - h * scale) / 2,
                        width: w * scale, height: h * scale,
                    };
                }"""
                page.evaluate(
                    f"""() => {{
                        const rect = ({_DISPATCH_RECT_JS})();
                        document.getElementById('frame').dispatchEvent(
                            new WheelEvent('wheel', {{
                                bubbles: true, cancelable: true,
                                deltaMode: 1,    // DOM_DELTA_LINE
                                deltaY: 3,       // one notch
                                clientX: rect.left + rect.width / 2,
                                clientY: rect.top + rect.height / 2,
                            }})
                        );
                    }}"""
                )
                page.wait_for_function(
                    "() => window.__teTest "
                    "  && !window.__teTest.peekScrollState().flushing "
                    "  && window.__teTest.peekScrollState().px < 30",
                    timeout=4_000,
                )
                page.wait_for_timeout(200)
                assert seen, (
                    "deltaMode=1, deltaY=3 must fire at least one "
                    "/api/mouse/scroll POST after normalisation; "
                    "raw 3 px would never cross the 30 px gate."
                )
                # 3 lines × 38 px/line = 114 px → 3 ticks
                # (114/30 floored = 3, remainder 24 stays in accum).
                assert seen[0]["amount"] >= 3, seen[0]
            finally:
                browser.close()


def test_wheel_shows_busy_indicator(tmp_path):
    """Per the cc UX contract the hourglass must show for every
    manual mouse action — including scroll. We verify ``#frame-busy``
    becomes visible during a flush and hides after it completes."""
    mouse_log: list[tuple[str, dict]] = []
    port = _pick_free_port()
    with _serve_cc(tmp_path, mouse_log, port) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                )
                page = ctx.new_page()
                # Slow the cc /api/mouse/scroll roundtrip so the
                # hourglass is observably "on" during the in-flight
                # POST. Without a delay the flush completes in
                # milliseconds and we miss the visible-state window.
                def _slow_route(route):
                    import time as _t
                    _t.sleep(0.4)
                    route.continue_()
                page.route("**/api/mouse/scroll", _slow_route)
                page.goto(base_url, wait_until="domcontentloaded")
                page.wait_for_function(
                    "typeof window.__teTest === 'object'", timeout=5_000,
                )
                _TINY_PNG = (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAKAAAABaCAIAAACwpMoFAAAA60lEQVR4"
                    "nO3RAQkAIADAMDWNIeyfyxQinC3B4XPvM+havwN4y+A4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MMjjM4zuA4g+MM"
                    "jjM4zuA4g+MMjjN4tF2eMgFA/IQ9PAAAAABJRU5ErkJggg=="
                )
                page.evaluate(
                    f"""(src) => new Promise((resolve) => {{
                        const f = document.getElementById('frame');
                        f.classList.remove('empty');
                        f.style.width  = '800px';
                        f.style.height = '450px';
                        f.style.display = 'block';
                        f.onload = () => resolve(true);
                        f.src = src;
                        if (f.complete && f.naturalWidth > 0) resolve(true);
                    }})""",
                    _TINY_PNG,
                )
                page.wait_for_function(
                    "() => document.getElementById('frame').naturalWidth > 0",
                    timeout=4_000,
                )
                # Pre-flush: hourglass hidden.
                hidden_before = page.evaluate(
                    "() => document.getElementById('frame-busy')."
                    "classList.contains('hidden')"
                )
                assert hidden_before is True
                _DISPATCH_RECT_JS = """() => {
                    const f = document.getElementById('frame');
                    const w = f.naturalWidth, h = f.naturalHeight;
                    const box = f.getBoundingClientRect();
                    const scale = Math.min(box.width / w, box.height / h);
                    return {
                        left: box.left + (box.width - w * scale) / 2,
                        top:  box.top  + (box.height - h * scale) / 2,
                        width: w * scale, height: h * scale,
                    };
                }"""
                page.evaluate(
                    f"""() => {{
                        const rect = ({_DISPATCH_RECT_JS})();
                        document.getElementById('frame').dispatchEvent(
                            new WheelEvent('wheel', {{
                                bubbles: true, cancelable: true,
                                deltaY: 120,
                                clientX: rect.left + rect.width / 2,
                                clientY: rect.top + rect.height / 2,
                            }})
                        );
                    }}"""
                )
                # During the slow POST the hourglass should flip on.
                page.wait_for_function(
                    "() => !document.getElementById('frame-busy')."
                    "classList.contains('hidden')",
                    timeout=2_000,
                )
                # After the POST settles, it must hide again.
                page.wait_for_function(
                    "() => document.getElementById('frame-busy')."
                    "classList.contains('hidden')",
                    timeout=4_000,
                )
            finally:
                browser.close()
