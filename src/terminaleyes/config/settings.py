"""Configuration management for terminaleyes.

Loads settings from a YAML configuration file with environment variable
overrides for sensitive values (API keys). Supports .env files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/terminaleyes.yaml")


class CaptureConfig(BaseModel):
    device_index: int = Field(default=0, description="OpenCV camera device index")
    capture_interval: float = Field(default=2.0, gt=0)
    crop_enabled: bool = Field(default=False)
    crop_x: int = Field(default=0, ge=0)
    crop_y: int = Field(default=0, ge=0)
    crop_width: int = Field(default=800, gt=0)
    crop_height: int = Field(default=600, gt=0)
    resolution_width: int | None = Field(default=None)
    resolution_height: int | None = Field(default=None)


class MLLMConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = Field(default="openai")
    model: str = Field(default="gpt-4o")
    base_url: str | None = Field(default=None)
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt_override: str | None = Field(default=None)


class EndpointConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    shell_command: str = Field(default="/bin/bash")
    terminal_rows: int = Field(default=24, gt=0)
    terminal_cols: int = Field(default=80, gt=0)
    font_size: int = Field(default=24, gt=0)
    bg_color: tuple[int, int, int] = Field(default=(30, 30, 30))
    fg_color: tuple[int, int, int] = Field(default=(192, 192, 192))


class KeyboardConfig(BaseModel):
    backend: Literal["http", "usb_hid"] = Field(default="http")
    http_base_url: str = Field(default="http://localhost:8080")
    http_timeout: float = Field(default=10.0, gt=0)
    usb_hid_device: str = Field(default="/dev/hidg0")


class AgentConfig(BaseModel):
    action_delay: float = Field(default=1.0, gt=0)
    max_consecutive_errors: int = Field(default=5, gt=0)
    default_max_steps: int = Field(default=100, gt=0)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    file: str | None = Field(default=None)


class Settings(BaseSettings):
    """Root configuration for the terminaleyes system.

    Loads from YAML file and supports environment variable overrides.
    Reads .env files automatically.
    """

    model_config = {
        "env_prefix": "TERMINALEYES_",
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # API Keys
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openrouter_api_key: SecretStr = Field(default=SecretStr(""))

    # Configuration sections
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    mllm: MLLMConfig = Field(default_factory=MLLMConfig)
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    keyboard: KeyboardConfig = Field(default_factory=KeyboardConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML + .env + environment variables.

    Priority: env vars > .env file > YAML file > defaults
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    # Load .env file manually for non-prefixed vars
    _load_dotenv()

    yaml_data = {}
    if path.exists():
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}
        logger.info("Loaded configuration from %s", path)
    else:
        logger.warning("Config file %s not found, using defaults + env vars", path)

    # Apply .env overrides into yaml_data for non-prefixed env vars
    _apply_env_overrides(yaml_data)

    return Settings(**yaml_data)


def _load_dotenv() -> None:
    """Load .env file into os.environ if it exists."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if not os.environ.get(key):
                    os.environ[key] = value


def _apply_env_overrides(yaml_data: dict) -> None:
    """Apply environment variable overrides for non-prefixed vars."""
    # Map common env vars to settings structure
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    or_base_url = os.environ.get("OPENROUTER_BASE_URL", "")
    vision_model = os.environ.get("VISION_MODEL", "")

    if or_key:
        yaml_data["openrouter_api_key"] = or_key

    if "mllm" not in yaml_data:
        yaml_data["mllm"] = {}

    if or_key and not yaml_data["mllm"].get("provider"):
        yaml_data["mllm"]["provider"] = "openai"

    if or_base_url and not yaml_data["mllm"].get("base_url"):
        yaml_data["mllm"]["base_url"] = or_base_url

    if vision_model and not yaml_data["mllm"].get("model"):
        yaml_data["mllm"]["model"] = vision_model
