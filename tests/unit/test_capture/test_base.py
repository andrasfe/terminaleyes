"""Tests for the CaptureSource abstract base class."""

from __future__ import annotations

import pytest

from terminaleyes.capture.base import CaptureSource, CaptureError
from terminaleyes.domain.models import CapturedFrame


class TestCaptureSourceInterface:
    """Test that CaptureSource defines the expected interface.

    TODO: Add tests for:
        - Cannot instantiate CaptureSource directly (abstract)
        - Concrete implementations must implement open/close/capture_frame
        - The stream() method yields frames at intervals
        - Async context manager calls open() and close()
        - CaptureError is raised appropriately
    """

    def test_cannot_instantiate_abstract_class(self) -> None:
        """CaptureSource should not be instantiable directly."""
        with pytest.raises(TypeError):
            CaptureSource()  # type: ignore[abstract]
