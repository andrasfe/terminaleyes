"""Abstract base class for keyboard action output.

All keyboard output backends must conform to this interface, enabling
the system to swap between the HTTP backend (for the local endpoint)
and the future USB HID backend (for Raspberry Pi) without changing
any other code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class KeyboardOutput(ABC):
    """Abstract interface for sending keyboard actions to a target.

    Implementations translate logical keyboard actions (keystrokes,
    key combinations, text input) into the appropriate protocol for
    their target (HTTP requests, USB HID reports, etc.).

    Example usage::

        async with HttpKeyboardOutput(base_url="http://localhost:8080") as kb:
            await kb.send_text("ls -la")
            await kb.send_keystroke("Enter")
            await kb.send_key_combo(["ctrl"], "c")
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the keyboard output target.

        Must be called before sending any actions. Implementations
        should verify connectivity (e.g., ping the HTTP endpoint,
        open the USB HID device).

        Raises:
            KeyboardOutputError: If connection cannot be established.

        TODO: Implement connection logic for each backend.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection to the output target.

        Should be safe to call multiple times. Always call this when
        done, or use the async context manager.

        TODO: Implement disconnection and resource cleanup.
        """
        ...

    @abstractmethod
    async def send_keystroke(self, key: str) -> None:
        """Send a single key press.

        Args:
            key: The key to press. Standard key names include:
                 - Printable characters: 'a', 'A', '1', '!', etc.
                 - Special keys: 'Enter', 'Tab', 'Escape', 'Backspace',
                   'Delete', 'Space', 'Up', 'Down', 'Left', 'Right',
                   'Home', 'End', 'PageUp', 'PageDown'
                 - Function keys: 'F1' through 'F12'

        Raises:
            KeyboardOutputError: If the keystroke cannot be sent.

        TODO: Implement keystroke transmission for each backend.
        """
        ...

    @abstractmethod
    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination (modifier keys + main key).

        Args:
            modifiers: List of modifier keys to hold. Valid modifiers:
                       'ctrl', 'alt', 'shift', 'meta'/'super'/'win'
            key: The main key to press while modifiers are held.

        Example::

            await kb.send_key_combo(['ctrl'], 'c')       # Ctrl+C
            await kb.send_key_combo(['ctrl', 'shift'], 'z')  # Ctrl+Shift+Z

        Raises:
            KeyboardOutputError: If the combo cannot be sent.

        TODO: Implement key combination transmission for each backend.
        """
        ...

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Type a string of text character by character.

        This method should handle typing each character in sequence,
        respecting any inter-character delay requirements of the target.

        Args:
            text: The text string to type. May contain any printable
                  characters. Does NOT automatically press Enter at
                  the end.

        Raises:
            KeyboardOutputError: If text input fails.

        TODO: Implement text input for each backend. Consider:
              - Inter-character delays for reliability
              - Character encoding (ASCII vs Unicode)
              - Special character handling
        """
        ...

    async def send_line(self, text: str) -> None:
        """Type a string of text and press Enter.

        Convenience method that combines send_text() and send_keystroke('Enter').

        Args:
            text: The text to type before pressing Enter.
        """
        await self.send_text(text)
        await self.send_keystroke("Enter")

    async def __aenter__(self) -> KeyboardOutput:
        """Async context manager entry -- connects to the output target."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Async context manager exit -- disconnects from the output target."""
        await self.disconnect()


class KeyboardOutputError(Exception):
    """Raised when keyboard output fails."""

    def __init__(self, message: str, backend: str = "") -> None:
        super().__init__(message)
        self.backend = backend
