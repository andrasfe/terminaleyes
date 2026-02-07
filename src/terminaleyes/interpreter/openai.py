"""OpenAI-compatible MLLM provider implementation.

Works with OpenAI, OpenRouter, and any OpenAI-compatible API
by setting a custom base_url.
"""

from __future__ import annotations

import logging

from terminaleyes.domain.models import CapturedFrame, TerminalState
from terminaleyes.interpreter.base import MLLMError, MLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(MLLMProvider):
    """MLLM provider using OpenAI's chat completions API.

    Also works with OpenRouter and other OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(model=model, system_prompt=system_prompt)
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = None

    async def _ensure_client(self) -> None:
        """Lazily initialize the OpenAI async client."""
        if self._client is not None:
            return
        from openai import AsyncOpenAI
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)
        logger.info("Initialized OpenAI client (model=%s, base_url=%s)", self._model, self._base_url)

    async def interpret(self, frame: CapturedFrame) -> TerminalState:
        """Interpret a terminal screenshot using the vision API."""
        await self._ensure_client()
        b64_image = self._encode_frame_to_base64(frame)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Interpret this terminal screenshot.",
                    },
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
            )
            raw_text = response.choices[0].message.content
            logger.debug("MLLM raw response: %s", raw_text[:200])
            return self._parse_response(raw_text, frame)
        except Exception as e:
            if "MLLMError" in type(e).__name__:
                raise
            raise MLLMError(
                f"OpenAI API call failed: {e}",
                provider="openai",
            ) from e

    async def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            await self._ensure_client()
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False
