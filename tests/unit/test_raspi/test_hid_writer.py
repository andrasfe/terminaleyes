"""Tests for the HID report writer (mocked /dev/hidg0)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from terminaleyes.raspi.hid_writer import (
    DEFAULT_INTER_CHAR_DELAY,
    DEFAULT_KEYPRESS_DELAY,
    MOUSE_RELEASE_REPORT,
    RELEASE_REPORT,
    HidWriteError,
    HidWriter,
    MouseHidWriter,
)


@pytest.fixture
def writer() -> HidWriter:
    return HidWriter(device_path="/dev/hidg0")


class TestHidWriterInit:
    def test_defaults(self, writer: HidWriter) -> None:
        assert writer._keypress_delay == DEFAULT_KEYPRESS_DELAY
        assert writer._inter_char_delay == DEFAULT_INTER_CHAR_DELAY
        assert not writer.is_open

    def test_custom_params(self) -> None:
        w = HidWriter(device_path="/dev/hidg1", keypress_delay=0.05, inter_char_delay=0.03)
        assert str(w._device_path) == "/dev/hidg1"
        assert w._keypress_delay == 0.05


class TestHidWriterOpen:
    @pytest.mark.asyncio
    async def test_open_sets_fd(self) -> None:
        w = HidWriter()
        with patch("os.open", return_value=42):
            await w.open()
        assert w._fd == 42
        assert w.is_open
        # Clean up without real close
        w._fd = None

    @pytest.mark.asyncio
    async def test_open_failure_raises(self) -> None:
        w = HidWriter(device_path="/dev/nonexistent")
        with patch("os.open", side_effect=OSError("No such device")):
            with pytest.raises(HidWriteError, match="Cannot open"):
                await w.open()
        assert not w.is_open


class TestHidWriterWrite:
    @pytest.mark.asyncio
    async def test_write_report(self) -> None:
        w = HidWriter()
        w._fd = 42
        with patch("os.write") as mock_write:
            await w._write_report(b"\x00" * 8)
            mock_write.assert_called_once_with(42, b"\x00" * 8)

    @pytest.mark.asyncio
    async def test_write_report_wrong_length(self) -> None:
        w = HidWriter()
        w._fd = 42
        with pytest.raises(HidWriteError, match="must be 8 bytes"):
            await w._write_report(b"\x00" * 5)

    @pytest.mark.asyncio
    async def test_write_report_not_open(self) -> None:
        w = HidWriter()
        with pytest.raises(HidWriteError, match="not open"):
            await w._write_report(b"\x00" * 8)

    @pytest.mark.asyncio
    async def test_write_os_error(self) -> None:
        w = HidWriter()
        w._fd = 42
        with patch("os.write", side_effect=OSError("I/O error")):
            with pytest.raises(HidWriteError, match="Failed to write"):
                await w._write_report(b"\x00" * 8)


class TestHidWriterKeys:
    @pytest.mark.asyncio
    async def test_press_key(self) -> None:
        w = HidWriter()
        w._fd = 42
        with patch("os.write") as mock_write:
            await w.press_key(0x00, 0x28)  # Enter
            expected = bytes([0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00])
            mock_write.assert_called_once_with(42, expected)

    @pytest.mark.asyncio
    async def test_release_keys(self) -> None:
        w = HidWriter()
        w._fd = 42
        with patch("os.write") as mock_write:
            await w.release_keys()
            mock_write.assert_called_once_with(42, RELEASE_REPORT)

    @pytest.mark.asyncio
    async def test_tap_key_press_then_release(self) -> None:
        w = HidWriter(keypress_delay=0.0)
        w._fd = 42
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.tap_key(0x02, 0x04)  # Shift + a
        assert len(reports) == 2
        assert reports[0] == bytes([0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00])
        assert reports[1] == RELEASE_REPORT

    @pytest.mark.asyncio
    async def test_send_keystroke_enter(self) -> None:
        w = HidWriter(keypress_delay=0.0)
        w._fd = 42
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.send_keystroke("Enter")
        assert len(reports) == 2
        # First report should have Enter scan code (0x28)
        assert reports[0][2] == 0x28

    @pytest.mark.asyncio
    async def test_send_keystroke_uppercase(self) -> None:
        w = HidWriter(keypress_delay=0.0)
        w._fd = 42
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.send_keystroke("A")
        assert len(reports) == 2
        # Shift modifier should be set
        assert reports[0][0] == 0x02
        # Scan code for 'a'
        assert reports[0][2] == 0x04

    @pytest.mark.asyncio
    async def test_send_key_combo_ctrl_c(self) -> None:
        w = HidWriter(keypress_delay=0.0)
        w._fd = 42
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.send_key_combo(["ctrl"], "c")
        assert len(reports) == 2
        assert reports[0][0] == 0x01  # Left Ctrl
        assert reports[0][2] == 0x06  # 'c' scan code

    @pytest.mark.asyncio
    async def test_send_text(self) -> None:
        w = HidWriter(keypress_delay=0.0, inter_char_delay=0.0)
        w._fd = 42
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.send_text("hi")
        # 2 chars * (press + release) = 4 reports
        assert len(reports) == 4


class TestHidWriterContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        w = HidWriter()
        with patch("os.open", return_value=42), \
             patch("os.write"), \
             patch("os.close"):
            async with w:
                assert w.is_open
            assert not w.is_open


# ===================================================================
# MouseHidWriter tests
# ===================================================================

@pytest.fixture
def mouse_writer() -> MouseHidWriter:
    return MouseHidWriter(device_path="/dev/hidg1")


class TestMouseHidWriterInit:
    def test_defaults(self, mouse_writer: MouseHidWriter) -> None:
        assert str(mouse_writer._device_path) == "/dev/hidg1"
        assert not mouse_writer.is_open
        assert mouse_writer._buttons == 0

    def test_custom_device_path(self) -> None:
        w = MouseHidWriter(device_path="/dev/hidg5")
        assert str(w._device_path) == "/dev/hidg5"


class TestMouseHidWriterOpen:
    @pytest.mark.asyncio
    async def test_open_sets_fd(self) -> None:
        w = MouseHidWriter()
        with patch("os.open", return_value=99):
            await w.open()
        assert w._fd == 99
        assert w.is_open
        w._fd = None

    @pytest.mark.asyncio
    async def test_open_failure_raises(self) -> None:
        w = MouseHidWriter(device_path="/dev/nonexistent")
        with patch("os.open", side_effect=OSError("No such device")):
            with pytest.raises(HidWriteError, match="Cannot open mouse"):
                await w.open()
        assert not w.is_open


class TestMouseHidWriterWrite:
    @pytest.mark.asyncio
    async def test_write_report(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        with patch("os.write") as mock_write:
            await w._write_report(b"\x00" * 4)
            mock_write.assert_called_once_with(99, b"\x00" * 4)

    @pytest.mark.asyncio
    async def test_write_report_wrong_length(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        with pytest.raises(HidWriteError, match="must be 4 bytes"):
            await w._write_report(b"\x00" * 8)

    @pytest.mark.asyncio
    async def test_write_report_not_open(self) -> None:
        w = MouseHidWriter()
        with pytest.raises(HidWriteError, match="not open"):
            await w._write_report(b"\x00" * 4)

    @pytest.mark.asyncio
    async def test_write_os_error(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        with patch("os.write", side_effect=OSError("I/O error")):
            with pytest.raises(HidWriteError, match="Failed to write"):
                await w._write_report(b"\x00" * 4)


class TestMouseHidWriterMove:
    @pytest.mark.asyncio
    async def test_move(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.move(10, -5)
        assert len(reports) == 1
        # [buttons=0, x=10, y=-5, wheel=0]
        assert reports[0] == b"\x00\x0a\xfb\x00"

    @pytest.mark.asyncio
    async def test_move_clamps(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.move(200, -200)
        assert len(reports) == 1
        # Clamped to 127 and -127
        assert reports[0] == b"\x00\x7f\x81\x00"


class TestMouseHidWriterClick:
    @pytest.mark.asyncio
    async def test_click_left(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.click("left")
        assert len(reports) == 2
        # Press: buttons=LEFT(0x01)
        assert reports[0][0] == 0x01
        # Release: buttons=0
        assert reports[1][0] == 0x00

    @pytest.mark.asyncio
    async def test_click_right(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.click("right")
        assert len(reports) == 2
        assert reports[0][0] == 0x02

    @pytest.mark.asyncio
    async def test_click_unknown_button(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        with pytest.raises(ValueError, match="Unknown button"):
            await w.click("banana")


class TestMouseHidWriterScroll:
    @pytest.mark.asyncio
    async def test_scroll(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.scroll(-3)
        assert len(reports) == 1
        # [buttons=0, x=0, y=0, wheel=-3]
        assert reports[0] == b"\x00\x00\x00\xfd"

    @pytest.mark.asyncio
    async def test_scroll_clamps(self) -> None:
        w = MouseHidWriter()
        w._fd = 99
        reports: list[bytes] = []
        with patch("os.write", side_effect=lambda fd, data: reports.append(data)):
            await w.scroll(200)
        assert len(reports) == 1
        # Clamped to 127
        assert reports[0] == b"\x00\x00\x00\x7f"


class TestMouseHidWriterContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        w = MouseHidWriter()
        with patch("os.open", return_value=99), \
             patch("os.write"), \
             patch("os.close"):
            async with w:
                assert w.is_open
            assert not w.is_open
