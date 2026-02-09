"""Screen reader using MLLM vision for arbitrary screen content.

Unlike the terminal-specific interpreter, this reads any screen content
(browsers, editors, documents) without OCR binarization to preserve colors.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

import numpy as np

from terminaleyes.utils.imaging import numpy_to_base64_png, resize_for_mllm
from terminaleyes.watcher.models import ScreenObservation

logger = logging.getLogger(__name__)

SCREEN_SYSTEM_PROMPT = """You are observing a computer screen through a webcam photograph.
Read and describe what is visible.

1. Identify the application(s) visible (browser, editor, terminal, etc.)
2. Read all text you can clearly see, left to right, top to bottom
3. For text too small or blurry to read, say "[unreadable: description]"
4. Note if the screen edges are cut off or if there are visibility issues

Respond ONLY with JSON:
{
    "content_type": "web_browser|code_editor|text_document|terminal|desktop|other",
    "application_context": "app name or null",
    "visible_text": "all readable text",
    "unreadable_notes": "description of what couldn't be read",
    "positioning_notes": "screen edge clipping, glare, distance issues, or none",
    "confidence": 0.0 to 1.0
}"""


class ScreenReader:
    """Reads arbitrary screen content via MLLM vision."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)
        logger.info(
            "ScreenReader initialized (model=%s, base_url=%s)",
            self._model,
            self._base_url,
        )

    async def read_screen(
        self, image: np.ndarray, frame_number: int
    ) -> ScreenObservation:
        """Read screen content from a raw camera frame.

        No OCR enhancement is applied — the raw color image is resized
        and sent directly to preserve color information.
        """
        await self._ensure_client()

        resized = resize_for_mllm(image)
        b64_image = numpy_to_base64_png(resized)

        messages = [
            {"role": "system", "content": SCREEN_SYSTEM_PROMPT},
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
                        "text": "Read and describe what is visible on this screen.",
                    },
                ],
            },
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        raw_text = response.choices[0].message.content
        logger.debug("ScreenReader raw response: %s", raw_text[:200])
        return self._parse_response(raw_text, frame_number)

    def _parse_response(
        self, raw_response: str, frame_number: int
    ) -> ScreenObservation:
        """Parse MLLM response into a ScreenObservation."""
        json_str = raw_response.strip()

        # Remove markdown code block if present
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

        # Find JSON object in text
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

        data = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fix invalid escape sequences
            try:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", json_str)
                data = json.loads(fixed)
            except json.JSONDecodeError:
                pass

        if data is None:
            logger.warning("Failed to parse screen reader response as JSON")
            data = {
                "content_type": "unknown",
                "visible_text": raw_response[:500],
                "confidence": 0.3,
            }

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return ScreenObservation(
            timestamp=datetime.now(),
            frame_number=frame_number,
            content_type=data.get("content_type", "unknown"),
            application_context=data.get("application_context"),
            visible_text=data.get("visible_text", ""),
            unreadable_notes=data.get("unreadable_notes", ""),
            positioning_notes=data.get("positioning_notes", "none"),
            confidence=confidence,
            raw_response=raw_response,
        )
