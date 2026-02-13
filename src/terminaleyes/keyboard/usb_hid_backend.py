"""USB HID keyboard output backend.

Sends keyboard actions by writing USB HID reports to /dev/hidg0 on a
Raspberry Pi configured as a USB gadget. This is the production backend
that replaces HttpKeyboardOutput when running on real hardware.

Use this backend when the terminaleyes agent runs directly on the Pi,
or use the raspi.server REST API when the agent runs on a separate machine.
"""

from __future__ import annotations

import logging

from terminaleyes.keyboard.base import KeyboardOutput, KeyboardOutputError
from terminaleyes.raspi.hid_writer import HidWriteError, HidWriter

logger = logging.getLogger(__name__)


class UsbHidKeyboardOutput(KeyboardOutput):
    """Sends keyboard actions via USB HID on a Raspberry Pi.

    Wraps HidWriter to conform to the KeyboardOutput ABC, so the agent
    loop can swap between HTTP and USB HID backends transparently.

    Requires:
        - Raspberry Pi with USB OTG (Pi Zero, Pi 4, etc.)
        - USB HID gadget configured (run scripts/setup_usb_gadget.sh)
        - /dev/hidg0 device present and writable
    """

    def __init__(
        self,
        device_path: str = "/dev/hidg0",
        keypress_delay: float = 0.02,
        inter_char_delay: float = 0.01,
    ) -> None:
        self._writer = HidWriter(
            device_path=device_path,
            keypress_delay=keypress_delay,
            inter_char_delay=inter_char_delay,
        )

    async def connect(self) -> None:
        """Open the USB HID gadget device."""
        try:
            await self._writer.open()
            logger.info("Connected to USB HID device")
        except HidWriteError as e:
            raise KeyboardOutputError(str(e), backend="usb_hid") from e

    async def disconnect(self) -> None:
        """Close the USB HID gadget device."""
        await self._writer.close()
        logger.info("Disconnected from USB HID device")

    async def send_keystroke(self, key: str) -> None:
        """Send a keystroke via USB HID report."""
        try:
            await self._writer.send_keystroke(key)
        except (ValueError, HidWriteError) as e:
            raise KeyboardOutputError(
                f"Failed to send keystroke '{key}': {e}", backend="usb_hid"
            ) from e

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination via USB HID report."""
        try:
            await self._writer.send_key_combo(modifiers, key)
        except (ValueError, HidWriteError) as e:
            raise KeyboardOutputError(
                f"Failed to send combo {'+'.join(modifiers)}+{key}: {e}",
                backend="usb_hid",
            ) from e

    async def send_text(self, text: str) -> None:
        """Type text character by character via USB HID reports."""
        try:
            await self._writer.send_text(text)
        except (ValueError, HidWriteError) as e:
            raise KeyboardOutputError(
                f"Failed to send text: {e}", backend="usb_hid"
            ) from e
