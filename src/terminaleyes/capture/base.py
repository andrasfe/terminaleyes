"""Abstract base class for vision capture sources.

All capture implementations must conform to this interface, enabling
the system to swap between webcam capture, screen capture, or file-based
test sources without changing the rest of the pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

from terminaleyes.domain.models import CapturedFrame, CropRegion

logger = logging.getLogger(__name__)


class CaptureSource(ABC):
    """Abstract interface for capturing frames from a visual source.

    Implementations should handle device initialization, frame capture,
    optional cropping, and cleanup. The async iterator pattern allows
    the agent loop to consume frames at its own pace.

    Example usage::

        async with WebcamCapture(device_index=0) as capture:
            async for frame in capture.stream(interval=1.0):
                process(frame)
    """

    def __init__(self, crop_region: CropRegion | None = None) -> None:
        """Initialize the capture source.

        Args:
            crop_region: Optional region to crop from each captured frame.
                         If None, the full frame is used.
        """
        self._crop_region = crop_region
        self._frame_counter: int = 0
        self._is_open: bool = False

    @property
    def is_open(self) -> bool:
        """Whether the capture device is currently open and ready."""
        return self._is_open

    @abstractmethod
    async def open(self) -> None:
        """Open and initialize the capture device.

        Must be called before capturing frames. Implementations should
        acquire any necessary hardware resources here.

        Raises:
            CaptureError: If the device cannot be opened.

        TODO: Implement device initialization for each concrete source.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release the capture device and free resources.

        Should be safe to call multiple times. Always call this when
        done capturing, or use the async context manager.

        TODO: Implement resource cleanup for each concrete source.
        """
        ...

    @abstractmethod
    async def capture_frame(self) -> CapturedFrame:
        """Capture a single frame from the source.

        Returns:
            A CapturedFrame with the image data and metadata.

        Raises:
            CaptureError: If frame capture fails.

        TODO: Implement frame capture, applying crop_region if set.
        """
        ...

    async def stream(self, interval: float = 1.0) -> AsyncIterator[CapturedFrame]:
        """Yield frames at the specified interval.

        This is a convenience method that captures frames in a loop
        with a configurable delay between captures.

        Args:
            interval: Seconds between captures. Must be > 0.

        Yields:
            CapturedFrame instances at approximately the requested interval.

        TODO: Implement with asyncio.sleep for timing control.
              Consider drift compensation for consistent intervals.
        """
        import asyncio

        if not self._is_open:
            raise RuntimeError("Capture source is not open. Call open() first.")

        while self._is_open:
            frame = await self.capture_frame()
            yield frame
            await asyncio.sleep(interval)

    async def __aenter__(self) -> CaptureSource:
        """Async context manager entry -- opens the capture device."""
        await self.open()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Async context manager exit -- closes the capture device."""
        await self.close()


class CaptureError(Exception):
    """Raised when frame capture fails."""
