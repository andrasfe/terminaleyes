"""Abstract base class for MLLM (Multimodal LLM) providers.

All MLLM provider implementations must conform to this interface,
enabling the system to swap between providers without changing the
rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime

from terminaleyes.domain.models import (
    CapturedFrame,
    TerminalContent,
    TerminalReadiness,
    TerminalState,
)
from terminaleyes.utils.imaging import numpy_to_base64_png, resize_for_mllm

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a terminal screen interpreter. You are given a screenshot of a terminal window.

Your task is to analyze the screenshot and provide a structured interpretation of what is visible on the terminal.

You must determine:
1. The full text visible on the screen
2. The most recent command that was executed (if visible)
3. The output of that command (if visible)
4. The current shell prompt (if visible)
5. Any error messages
6. The current working directory (if discernible from the prompt)
7. Whether the terminal is ready for input (showing a prompt), busy (command running), or in an error state

Respond ONLY with valid JSON in the following format (no markdown, no explanation):
{
    "visible_text": "...",
    "last_command": "..." or null,
    "last_output": "..." or null,
    "prompt_text": "..." or null,
    "error_messages": ["..."],
    "working_directory": "..." or null,
    "readiness": "ready" | "busy" | "error" | "unknown",
    "confidence": 0.0 to 1.0
}
"""


class MLLMProvider(ABC):
    """Abstract interface for multimodal LLM providers."""

    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        self._model = model
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    @property
    def model(self) -> str:
        return self._model

    @abstractmethod
    async def interpret(self, frame: CapturedFrame) -> TerminalState:
        """Interpret a captured terminal frame using the MLLM."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the MLLM provider is reachable and authenticated."""
        ...

    def _encode_frame_to_base64(self, frame: CapturedFrame) -> str:
        """Encode a captured frame's image to a base64 PNG string."""
        resized = resize_for_mllm(frame.image)
        return numpy_to_base64_png(resized)

    def _parse_response(self, raw_response: str, frame: CapturedFrame) -> TerminalState:
        """Parse a raw MLLM response string into a TerminalState."""
        # Try to extract JSON from markdown code blocks or raw text
        json_str = raw_response.strip()

        # Remove markdown code block if present
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

        # Try to find JSON object in the text
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MLLMError(
                f"Failed to parse MLLM response as JSON: {e}",
                provider=type(self).__name__,
                raw_response=raw_response,
            ) from e

        try:
            content = TerminalContent(
                visible_text=data.get("visible_text", ""),
                last_command=data.get("last_command"),
                last_output=data.get("last_output"),
                prompt_text=data.get("prompt_text"),
                error_messages=data.get("error_messages", []),
                working_directory=data.get("working_directory"),
            )
            readiness_str = data.get("readiness", "unknown")
            try:
                readiness = TerminalReadiness(readiness_str)
            except ValueError:
                readiness = TerminalReadiness.UNKNOWN

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return TerminalState(
                content=content,
                readiness=readiness,
                confidence=confidence,
                raw_interpretation=raw_response,
                timestamp=datetime.now(),
                frame_number=frame.frame_number,
            )
        except Exception as e:
            raise MLLMError(
                f"Failed to build TerminalState from parsed data: {e}",
                provider=type(self).__name__,
                raw_response=raw_response,
            ) from e


class MLLMError(Exception):
    """Raised when MLLM interpretation fails."""

    def __init__(self, message: str, provider: str = "", raw_response: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.raw_response = raw_response
