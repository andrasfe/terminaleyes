"""Anthropic Claude MLLM provider implementation.

Uses the Anthropic Python SDK to send terminal screenshots to Claude
models with vision capability and receive structured interpretations.
"""

from __future__ import annotations

import logging

from terminaleyes.domain.models import CapturedFrame, TerminalState
from terminaleyes.interpreter.base import MLLMError, MLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(MLLMProvider):
    """MLLM provider using Anthropic's Claude API.

    Sends terminal screenshots to Claude's vision API and parses the
    structured response into a TerminalState.

    Example usage::

        provider = AnthropicProvider(
            api_key="sk-ant-...",
            model="claude-sonnet-4-20250514",
        )
        state = await provider.interpret(frame)

    TODO: Complete implementation requires:
        - anthropic SDK installed
        - Valid API key with vision model access
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier (must support vision).
            system_prompt: Custom system prompt override.
            max_tokens: Maximum tokens in the response.
        """
        super().__init__(model=model, system_prompt=system_prompt)
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client: object | None = None  # Will be anthropic.AsyncAnthropic

    async def _ensure_client(self) -> None:
        """Lazily initialize the Anthropic async client.

        TODO: Implement the following:
            1. Import anthropic
            2. Create anthropic.AsyncAnthropic(api_key=self._api_key)
            3. Store in self._client
        """
        raise NotImplementedError("TODO: Initialize Anthropic client")

    async def interpret(self, frame: CapturedFrame) -> TerminalState:
        """Interpret a terminal screenshot using Claude's vision API.

        Args:
            frame: The captured frame to interpret.

        Returns:
            TerminalState with the structured interpretation.

        Raises:
            MLLMError: If the API call fails.

        TODO: Implement the following:
            1. Call self._ensure_client()
            2. Encode frame to base64 using self._encode_frame_to_base64()
            3. Build the messages list with image content block:
               - type: "image", source: {type: "base64", media_type: "image/png", data: ...}
               - type: "text", text: "Interpret this terminal screenshot."
            4. Call self._client.messages.create(
                   model=self._model,
                   max_tokens=self._max_tokens,
                   system=self._system_prompt,
                   messages=[{"role": "user", "content": content}]
               )
            5. Extract text from response.content[0].text
            6. Parse using self._parse_response()
            7. Handle anthropic.APIError, anthropic.RateLimitError
        """
        raise NotImplementedError("TODO: Implement Anthropic interpret()")

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        TODO: Implement a lightweight API call (e.g., a simple
              text-only message) to verify the API key and connectivity.
        """
        raise NotImplementedError("TODO: Implement Anthropic health check")
