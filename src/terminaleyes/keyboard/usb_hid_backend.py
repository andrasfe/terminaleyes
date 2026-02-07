"""USB HID keyboard output backend (future implementation).

Will send keyboard actions via a Raspberry Pi acting as a USB HID
keyboard device. This module is a placeholder for the future
hardware-based backend.

NOTE: This backend is not yet implemented. It exists to demonstrate
the pluggable architecture and to serve as a starting point when
Raspberry Pi hardware integration begins.
"""

from __future__ import annotations

import logging

from terminaleyes.keyboard.base import KeyboardOutput, KeyboardOutputError

logger = logging.getLogger(__name__)


class UsbHidKeyboardOutput(KeyboardOutput):
    """Sends keyboard actions via USB HID on a Raspberry Pi.

    This backend will communicate with a Raspberry Pi configured as a
    USB HID gadget, sending keyboard scan codes that appear as real
    keyboard input to the connected machine.

    NOTE: This is a placeholder. Full implementation requires:
        - Raspberry Pi with USB OTG support (Pi Zero, Pi 4, etc.)
        - Configured USB HID gadget mode (/dev/hidg0)
        - USB HID scan code mapping
        - Serial or network connection to the Pi

    TODO: Full implementation roadmap:
        1. Decide on communication protocol (serial, SSH, local socket)
        2. Implement USB HID report descriptor building
        3. Map key names to USB HID scan codes
        4. Handle modifier key state management
        5. Implement timing for key press/release cycles
        6. Add error recovery for disconnected USB
    """

    def __init__(
        self,
        device_path: str = "/dev/hidg0",
        host: str | None = None,
    ) -> None:
        """Initialize the USB HID backend.

        Args:
            device_path: Path to the HID gadget device on the Pi.
            host: If controlling the Pi remotely, the SSH host.
                  If None, assumes running directly on the Pi.
        """
        self._device_path = device_path
        self._host = host

    async def connect(self) -> None:
        """Open the USB HID device.

        TODO: Implement when hardware is available.
        """
        raise NotImplementedError(
            "USB HID backend is not yet implemented. "
            "Use HttpKeyboardOutput for development."
        )

    async def disconnect(self) -> None:
        """Close the USB HID device.

        TODO: Implement when hardware is available.
        """
        raise NotImplementedError("USB HID backend is not yet implemented.")

    async def send_keystroke(self, key: str) -> None:
        """Send a keystroke via USB HID report.

        TODO: Implement USB HID scan code lookup and report sending.
        """
        raise NotImplementedError("USB HID backend is not yet implemented.")

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination via USB HID report.

        TODO: Implement modifier bitmask + scan code in a single report.
        """
        raise NotImplementedError("USB HID backend is not yet implemented.")

    async def send_text(self, text: str) -> None:
        """Send text via sequential USB HID keystroke reports.

        TODO: Implement character-to-scan-code mapping and sequential sending.
        """
        raise NotImplementedError("USB HID backend is not yet implemented.")
