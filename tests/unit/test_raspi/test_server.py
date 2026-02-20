"""Tests for the Raspberry Pi REST API server."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from terminaleyes.raspi.hid_writer import HidWriteError, HidWriter, MouseHidWriter
from terminaleyes.raspi.server import create_app


@pytest.fixture
def mock_writer() -> AsyncMock:
    """A mock HidWriter with all async methods stubbed."""
    writer = AsyncMock(spec=HidWriter)
    writer.is_open = True
    return writer


@pytest.fixture
def mock_mouse_writer() -> AsyncMock:
    """A mock MouseHidWriter with all async methods stubbed."""
    writer = AsyncMock(spec=MouseHidWriter)
    writer.is_open = True
    return writer


@pytest.fixture
def mock_bt_hid() -> AsyncMock:
    """A mock BluetoothHidServer with keyboard + mouse methods."""
    bt = AsyncMock()
    bt.is_connected = True
    return bt


@pytest.fixture
def client(mock_writer: AsyncMock, mock_mouse_writer: AsyncMock) -> TestClient:
    """A test client with mock HidWriter + MouseHidWriter injected (no BT)."""
    app = create_app(writer=mock_writer, mouse_writer=mock_mouse_writer, enable_bt_hid=False)
    app.state.writer = mock_writer
    app.state.mouse_writer = mock_mouse_writer
    return TestClient(app)


@pytest.fixture
def client_with_bt(
    mock_writer: AsyncMock, mock_mouse_writer: AsyncMock, mock_bt_hid: AsyncMock
) -> TestClient:
    """A test client with mock HidWriter, MouseHidWriter, and BT HID."""
    app = create_app(
        writer=mock_writer,
        mouse_writer=mock_mouse_writer,
        bt_hid=mock_bt_hid,
        enable_bt_hid=False,
    )
    app.state.writer = mock_writer
    app.state.mouse_writer = mock_mouse_writer
    app.state.bt_hid = mock_bt_hid
    return TestClient(app)


# ===================================================================
# USB HID keyboard endpoints
# ===================================================================

class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["hid_open"] is True
        assert data["mouse_hid_open"] is True


class TestKeystrokeEndpoint:
    def test_keystroke_success(self, client: TestClient, mock_writer: AsyncMock) -> None:
        resp = client.post("/keystroke", json={"key": "Enter"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_writer.send_keystroke.assert_called_once_with("Enter")

    def test_keystroke_unknown_key(self, client: TestClient, mock_writer: AsyncMock) -> None:
        mock_writer.send_keystroke.side_effect = ValueError("Unknown key")
        resp = client.post("/keystroke", json={"key": "BadKey"})
        assert resp.status_code == 400

    def test_keystroke_missing_field(self, client: TestClient) -> None:
        resp = client.post("/keystroke", json={})
        assert resp.status_code == 422


class TestKeyComboEndpoint:
    def test_key_combo_success(self, client: TestClient, mock_writer: AsyncMock) -> None:
        resp = client.post("/key-combo", json={"modifiers": ["ctrl"], "key": "c"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_writer.send_key_combo.assert_called_once_with(["ctrl"], "c")

    def test_key_combo_bad_modifier(self, client: TestClient, mock_writer: AsyncMock) -> None:
        mock_writer.send_key_combo.side_effect = ValueError("Unknown modifier")
        resp = client.post("/key-combo", json={"modifiers": ["banana"], "key": "c"})
        assert resp.status_code == 400


class TestTextEndpoint:
    def test_text_success(self, client: TestClient, mock_writer: AsyncMock) -> None:
        resp = client.post("/text", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["length"] == "11"
        mock_writer.send_text.assert_called_once_with("hello world")

    def test_text_empty(self, client: TestClient, mock_writer: AsyncMock) -> None:
        resp = client.post("/text", json={"text": ""})
        assert resp.status_code == 200
        mock_writer.send_text.assert_called_once_with("")


# ===================================================================
# USB HID mouse endpoints
# ===================================================================

class TestMouseMoveEndpoint:
    def test_mouse_move_success(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        resp = client.post("/mouse/move", json={"x": 10, "y": -5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["x"] == "10"
        assert data["y"] == "-5"
        mock_mouse_writer.move.assert_called_once_with(10, -5)

    def test_mouse_move_error(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        mock_mouse_writer.move.side_effect = HidWriteError("I/O error")
        resp = client.post("/mouse/move", json={"x": 10, "y": 0})
        assert resp.status_code == 400


class TestMouseClickEndpoint:
    def test_mouse_click_success(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        resp = client.post("/mouse/click", json={"button": "left"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_mouse_writer.click.assert_called_once_with("left")

    def test_mouse_click_bad_button(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        mock_mouse_writer.click.side_effect = ValueError("Unknown button")
        resp = client.post("/mouse/click", json={"button": "banana"})
        assert resp.status_code == 400

    def test_mouse_click_error(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        mock_mouse_writer.click.side_effect = HidWriteError("I/O error")
        resp = client.post("/mouse/click", json={"button": "right"})
        assert resp.status_code == 400


class TestMouseScrollEndpoint:
    def test_mouse_scroll_success(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        resp = client.post("/mouse/scroll", json={"amount": -3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["amount"] == "-3"
        mock_mouse_writer.scroll.assert_called_once_with(-3)

    def test_mouse_scroll_error(
        self, client: TestClient, mock_mouse_writer: AsyncMock
    ) -> None:
        mock_mouse_writer.scroll.side_effect = HidWriteError("I/O error")
        resp = client.post("/mouse/scroll", json={"amount": 5})
        assert resp.status_code == 400


# ===================================================================
# Bluetooth keyboard endpoints
# ===================================================================

class TestBtKeystrokeEndpoint:
    def test_bt_keystroke_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/keystroke", json={"key": "Enter"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["key"] == "Enter"
        assert data["transport"] == "bluetooth"
        mock_bt_hid.send_keystroke.assert_called_once_with("Enter")

    def test_bt_keystroke_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/keystroke", json={"key": "Enter"})
        assert resp.status_code == 503

    def test_bt_keystroke_error(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        mock_bt_hid.send_keystroke.side_effect = ValueError("Unknown key")
        resp = client_with_bt.post("/bt/keystroke", json={"key": "BadKey"})
        assert resp.status_code == 400


class TestBtKeyComboEndpoint:
    def test_bt_key_combo_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/key-combo", json={"modifiers": ["ctrl"], "key": "c"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["transport"] == "bluetooth"
        mock_bt_hid.send_key_combo.assert_called_once_with(["ctrl"], "c")

    def test_bt_key_combo_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/key-combo", json={"modifiers": ["ctrl"], "key": "c"})
        assert resp.status_code == 503

    def test_bt_key_combo_error(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        mock_bt_hid.send_key_combo.side_effect = ValueError("Unknown modifier")
        resp = client_with_bt.post("/bt/key-combo", json={"modifiers": ["banana"], "key": "c"})
        assert resp.status_code == 400


class TestBtTextEndpoint:
    def test_bt_text_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/text", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["length"] == "11"
        assert data["transport"] == "bluetooth"
        mock_bt_hid.send_text.assert_called_once_with("hello world")

    def test_bt_text_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/text", json={"text": "hello"})
        assert resp.status_code == 503


# ===================================================================
# Bluetooth mouse endpoints
# ===================================================================

class TestBtMouseMoveEndpoint:
    def test_bt_mouse_move_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/mouse/move", json={"x": 10, "y": -5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["x"] == "10"
        assert data["y"] == "-5"
        mock_bt_hid.move.assert_called_once_with(10, -5)

    def test_bt_mouse_move_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/mouse/move", json={"x": 10, "y": 0})
        assert resp.status_code == 503

    def test_bt_mouse_move_error(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        mock_bt_hid.move.side_effect = Exception("BT disconnected")
        resp = client_with_bt.post("/bt/mouse/move", json={"x": 10, "y": 0})
        assert resp.status_code == 400


class TestBtMouseClickEndpoint:
    def test_bt_mouse_click_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/mouse/click", json={"button": "left"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_bt_hid.click.assert_called_once_with("left")

    def test_bt_mouse_click_right(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/mouse/click", json={"button": "right"})
        assert resp.status_code == 200
        mock_bt_hid.click.assert_called_once_with("right")

    def test_bt_mouse_click_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/mouse/click", json={"button": "left"})
        assert resp.status_code == 503

    def test_bt_mouse_click_bad_button(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        mock_bt_hid.click.side_effect = ValueError("Unknown button")
        resp = client_with_bt.post("/bt/mouse/click", json={"button": "banana"})
        assert resp.status_code == 400


class TestBtMouseScrollEndpoint:
    def test_bt_mouse_scroll_success(
        self, client_with_bt: TestClient, mock_bt_hid: AsyncMock
    ) -> None:
        resp = client_with_bt.post("/bt/mouse/scroll", json={"amount": -3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["amount"] == "-3"
        mock_bt_hid.scroll.assert_called_once_with(-3)

    def test_bt_mouse_scroll_no_bt(self, client: TestClient) -> None:
        resp = client.post("/bt/mouse/scroll", json={"amount": 5})
        assert resp.status_code == 503


# ===================================================================
# Health with BT
# ===================================================================

class TestHealthWithBt:
    def test_health_shows_bt_connected(self, client_with_bt: TestClient) -> None:
        resp = client_with_bt.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bt_hid_connected"] is True

    def test_health_no_bt(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bt_hid_connected"] is False
