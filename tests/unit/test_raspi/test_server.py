"""Tests for the Raspberry Pi REST API server."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from terminaleyes.raspi.hid_writer import HidWriter
from terminaleyes.raspi.server import create_app


@pytest.fixture
def mock_writer() -> AsyncMock:
    """A mock HidWriter with all async methods stubbed."""
    writer = AsyncMock(spec=HidWriter)
    writer.is_open = True
    return writer


@pytest.fixture
def client(mock_writer: AsyncMock) -> TestClient:
    """A test client with a mock HidWriter injected."""
    app = create_app(writer=mock_writer)
    # Manually set writer on state so lifespan doesn't try to open real device
    app.state.writer = mock_writer
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["hid_open"] is True


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
