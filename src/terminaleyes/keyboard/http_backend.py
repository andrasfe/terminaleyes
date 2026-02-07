"""HTTP keyboard output backend.

Sends keyboard actions as HTTP requests to the local command endpoint.
"""

from __future__ import annotations

import logging

import httpx

from terminaleyes.keyboard.base import KeyboardOutput, KeyboardOutputError

logger = logging.getLogger(__name__)


class HttpKeyboardOutput(KeyboardOutput):
    """Sends keyboard actions to the local HTTP command endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Create the HTTP client and verify endpoint connectivity."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            logger.info("Connected to endpoint at %s", self._base_url)
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise KeyboardOutputError(
                f"Failed to connect to endpoint: {e}", backend="http"
            ) from e

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from endpoint")

    async def send_keystroke(self, key: str) -> None:
        """Send a keystroke via HTTP POST."""
        await self._post("/keystroke", {"key": key})
        logger.debug("Sent keystroke: %s", key)

    async def send_key_combo(self, modifiers: list[str], key: str) -> None:
        """Send a key combination via HTTP POST."""
        await self._post("/key-combo", {"modifiers": modifiers, "key": key})
        logger.debug("Sent key combo: %s+%s", "+".join(modifiers), key)

    async def send_text(self, text: str) -> None:
        """Send text input via HTTP POST."""
        await self._post("/text", {"text": text})
        logger.debug("Sent text: %s", text[:50])

    async def _post(self, path: str, payload: dict) -> httpx.Response:
        """Send a POST request to the endpoint."""
        if self._client is None:
            raise KeyboardOutputError("Not connected to endpoint", backend="http")
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            raise KeyboardOutputError(
                f"HTTP request to {path} failed: {e}", backend="http"
            ) from e
