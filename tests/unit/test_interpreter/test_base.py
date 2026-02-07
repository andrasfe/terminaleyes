"""Tests for the MLLMProvider abstract base class."""

from __future__ import annotations

import pytest

from terminaleyes.interpreter.base import MLLMProvider, MLLMError


class TestMLLMProviderInterface:
    """Test that MLLMProvider defines the expected interface.

    TODO: Add tests for:
        - Cannot instantiate MLLMProvider directly (abstract)
        - The model property returns the configured model name
        - _parse_response correctly parses valid JSON into TerminalState
        - _parse_response raises MLLMError on invalid JSON
        - _encode_frame_to_base64 produces valid base64 output
        - MLLMError carries provider and raw_response metadata
    """

    def test_cannot_instantiate_abstract_class(self) -> None:
        """MLLMProvider should not be instantiable directly."""
        with pytest.raises(TypeError):
            MLLMProvider(model="test")  # type: ignore[abstract]

    def test_mllm_error_carries_metadata(self) -> None:
        """MLLMError should store provider and raw_response."""
        error = MLLMError("test error", provider="anthropic", raw_response='{"bad": "json"}')
        assert str(error) == "test error"
        assert error.provider == "anthropic"
        assert error.raw_response == '{"bad": "json"}'
