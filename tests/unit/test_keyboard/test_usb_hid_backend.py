"""Tests for the UsbHidKeyboardOutput backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from terminaleyes.keyboard.base import KeyboardOutputError
from terminaleyes.keyboard.usb_hid_backend import UsbHidKeyboardOutput
from terminaleyes.raspi.hid_writer import HidWriteError


@pytest.fixture
def mock_writer() -> AsyncMock:
    mock = AsyncMock()
    mock.is_open = False
    return mock


@pytest.fixture
def backend(mock_writer: AsyncMock) -> UsbHidKeyboardOutput:
    kb = UsbHidKeyboardOutput()
    kb._writer = mock_writer
    return kb


class TestUsbHidInit:
    def test_default_device(self) -> None:
        kb = UsbHidKeyboardOutput()
        assert str(kb._writer._device_path) == "/dev/hidg0"

    def test_custom_device(self) -> None:
        kb = UsbHidKeyboardOutput(device_path="/dev/hidg1")
        assert str(kb._writer._device_path) == "/dev/hidg1"


class TestUsbHidConnect:
    @pytest.mark.asyncio
    async def test_connect_calls_open(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        await backend.connect()
        mock_writer.open.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_wraps_error(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        mock_writer.open.side_effect = HidWriteError("No device")
        with pytest.raises(KeyboardOutputError, match="No device"):
            await backend.connect()


class TestUsbHidDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_calls_close(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        await backend.disconnect()
        mock_writer.close.assert_called_once()


class TestUsbHidSendKeystroke:
    @pytest.mark.asyncio
    async def test_sends_keystroke(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        await backend.send_keystroke("Enter")
        mock_writer.send_keystroke.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_wraps_value_error(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        mock_writer.send_keystroke.side_effect = ValueError("Unknown key")
        with pytest.raises(KeyboardOutputError):
            await backend.send_keystroke("BadKey")


class TestUsbHidSendKeyCombo:
    @pytest.mark.asyncio
    async def test_sends_combo(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        await backend.send_key_combo(["ctrl"], "c")
        mock_writer.send_key_combo.assert_called_once_with(["ctrl"], "c")

    @pytest.mark.asyncio
    async def test_wraps_error(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        mock_writer.send_key_combo.side_effect = HidWriteError("I/O error")
        with pytest.raises(KeyboardOutputError):
            await backend.send_key_combo(["ctrl"], "c")


class TestUsbHidSendText:
    @pytest.mark.asyncio
    async def test_sends_text(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        await backend.send_text("hello")
        mock_writer.send_text.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_wraps_error(self, backend: UsbHidKeyboardOutput, mock_writer: AsyncMock) -> None:
        mock_writer.send_text.side_effect = ValueError("No mapping")
        with pytest.raises(KeyboardOutputError):
            await backend.send_text("\x00")
