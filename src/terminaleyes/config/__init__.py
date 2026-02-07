"""Configuration management for terminaleyes.

Loads and validates YAML-based configuration with Pydantic models.
Supports environment variable overrides for sensitive values like
API keys.
"""

from terminaleyes.config.settings import Settings, load_settings

__all__ = ["Settings", "load_settings"]
