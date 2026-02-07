"""Tests for the HttpKeyboardOutput backend."""

from __future__ import annotations

import pytest

from terminaleyes.keyboard.http_backend import HttpKeyboardOutput


class TestHttpKeyboardOutput:
    """Test the HTTP keyboard output backend.

    TODO: Add tests for:
        - connect() creates httpx client and pings /health
        - connect() raises KeyboardOutputError on unreachable endpoint
        - disconnect() closes the httpx client
        - send_keystroke() POSTs to /keystroke with correct payload
        - send_key_combo() POSTs to /key-combo with correct payload
        - send_text() POSTs to /text with correct payload
        - send_line() calls send_text and send_keystroke('Enter')
        - HTTP errors are wrapped in KeyboardOutputError
        - Async context manager connects and disconnects
    """

    def test_init_defaults(self) -> None:
        """HttpKeyboardOutput should accept default initialization."""
        kb = HttpKeyboardOutput()
        assert kb._base_url == "http://localhost:8080"
        assert kb._timeout == 10.0

    def test_init_custom_url(self) -> None:
        """HttpKeyboardOutput should accept custom URL."""
        kb = HttpKeyboardOutput(base_url="http://192.168.1.100:9090/")
        assert kb._base_url == "http://192.168.1.100:9090"
