"""Logging setup utilities for terminaleyes.

Configures structured logging for the entire application based on
the logging configuration settings.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from terminaleyes.config.settings import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure logging for the terminaleyes application.

    Sets up the root logger with the specified level, format, and
    optional file handler.

    Args:
        config: Logging configuration. If None, uses defaults
                (INFO level, stderr output).

    TODO: Implement the following:
        1. Create a LoggingConfig if None was provided
        2. Set up the root 'terminaleyes' logger
        3. Create a StreamHandler for stderr
        4. If config.file is set, create a FileHandler
        5. Apply the format string to all handlers
        6. Set the log level from config.level
        7. Add handlers to the logger
    """
    if config is None:
        config = LoggingConfig()

    root_logger = logging.getLogger("terminaleyes")
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if config.file:
        file_handler = logging.FileHandler(config.file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized at %s level", config.level)
