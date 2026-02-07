"""Vision Capture module for terminaleyes.

Provides webcam frame capture with configurable intervals, cropping,
and preprocessing. The abstract base class allows alternative capture
implementations (e.g., screen capture, file-based testing).

Public API:
    CaptureSource -- Abstract base class
    WebcamCapture -- OpenCV webcam implementation
"""

from terminaleyes.capture.base import CaptureSource, CaptureError

__all__ = ["CaptureSource", "CaptureError", "WebcamCapture"]


def __getattr__(name: str) -> type:
    """Lazy import for concrete implementations that require external deps."""
    if name == "WebcamCapture":
        from terminaleyes.capture.webcam import WebcamCapture
        return WebcamCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
