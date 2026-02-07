"""Tests for the WebcamCapture implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from terminaleyes.capture.webcam import WebcamCapture
from terminaleyes.domain.models import CropRegion


class TestWebcamCapture:
    """Test the WebcamCapture concrete implementation.

    TODO: Add tests for:
        - open() initializes cv2.VideoCapture with correct device index
        - open() sets resolution if specified
        - open() raises CaptureError if device not available
        - close() releases the VideoCapture
        - capture_frame() returns a CapturedFrame with valid image
        - capture_frame() applies crop when crop_region is set
        - capture_frame() increments frame_counter
        - _apply_crop() correctly slices the numpy array
        - Async context manager opens and closes device
    """

    def test_init_defaults(self) -> None:
        """WebcamCapture should accept default initialization."""
        capture = WebcamCapture()
        assert capture._device_index == 0
        assert capture._crop_region is None
        assert capture.is_open is False

    def test_init_with_options(self) -> None:
        """WebcamCapture should accept custom device and crop region."""
        crop = CropRegion(x=10, y=20, width=640, height=480)
        capture = WebcamCapture(device_index=1, crop_region=crop, resolution=(1920, 1080))
        assert capture._device_index == 1
        assert capture._crop_region == crop
        assert capture._resolution == (1920, 1080)
