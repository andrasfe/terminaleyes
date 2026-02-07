"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from terminaleyes.config.settings import (
    CaptureConfig,
    EndpointConfig,
    MLLMConfig,
    Settings,
    load_settings,
)


class TestSettings:
    """Test configuration models and loading.

    TODO: Add tests for:
        - Default Settings() creates valid config with all defaults
        - YAML file loading populates settings correctly
        - Environment variable overrides work for API keys
        - Invalid config values raise validation errors
        - Missing config file falls back to defaults with warning
        - All config sections have sensible defaults
        - CropRegion validation rejects negative values
    """

    def test_default_settings(self) -> None:
        """Default Settings should be valid."""
        settings = Settings()
        assert settings.capture.device_index == 0
        assert settings.mllm.provider == "anthropic"
        assert settings.endpoint.port == 8080
        assert settings.keyboard.backend == "http"

    def test_capture_config_defaults(self) -> None:
        """CaptureConfig should have sensible defaults."""
        config = CaptureConfig()
        assert config.capture_interval == 2.0
        assert config.crop_enabled is False

    def test_mllm_config_defaults(self) -> None:
        """MLLMConfig should have sensible defaults."""
        config = MLLMConfig()
        assert config.provider == "anthropic"
        assert config.max_tokens == 1024

    def test_load_settings_missing_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """load_settings with missing file should return defaults."""
        settings = load_settings(tmp_path / "nonexistent.yaml")  # type: ignore[operator]
        assert settings.capture.device_index == 0
