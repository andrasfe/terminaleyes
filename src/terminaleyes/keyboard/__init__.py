"""Keyboard Action output module for terminaleyes.

Translates agent decisions into keyboard actions via pluggable backends.
The abstract interface supports both the HTTP backend (for the local
endpoint) and the future USB HID backend (for Raspberry Pi).

Public API:
    KeyboardOutput -- Abstract base class
    HttpKeyboardOutput -- HTTP backend for the local endpoint
"""

from terminaleyes.keyboard.base import KeyboardOutput, KeyboardOutputError

__all__ = ["KeyboardOutput", "KeyboardOutputError", "HttpKeyboardOutput"]


def __getattr__(name: str) -> type:
    """Lazy import for concrete implementations that require external deps."""
    if name == "HttpKeyboardOutput":
        from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
        return HttpKeyboardOutput
    if name == "UsbHidKeyboardOutput":
        from terminaleyes.keyboard.usb_hid_backend import UsbHidKeyboardOutput
        return UsbHidKeyboardOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
