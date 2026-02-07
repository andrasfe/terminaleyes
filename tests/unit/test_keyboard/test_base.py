"""Tests for the KeyboardOutput abstract base class."""

from __future__ import annotations

import pytest

from terminaleyes.keyboard.base import KeyboardOutput, KeyboardOutputError


class TestKeyboardOutputInterface:
    """Test that KeyboardOutput defines the expected interface.

    TODO: Add tests for:
        - Cannot instantiate KeyboardOutput directly (abstract)
        - Concrete implementations must implement all abstract methods
        - send_line() calls send_text() then send_keystroke('Enter')
        - Async context manager calls connect() and disconnect()
        - KeyboardOutputError carries backend metadata
    """

    def test_cannot_instantiate_abstract_class(self) -> None:
        """KeyboardOutput should not be instantiable directly."""
        with pytest.raises(TypeError):
            KeyboardOutput()  # type: ignore[abstract]

    def test_keyboard_output_error(self) -> None:
        """KeyboardOutputError should store backend info."""
        error = KeyboardOutputError("connection failed", backend="http")
        assert str(error) == "connection failed"
        assert error.backend == "http"
