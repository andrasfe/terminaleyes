"""Condition evaluator — checks if a visual condition is met in a webcam frame.

Sends the frame + condition description to the MLLM vision API and parses
the response to determine if the condition is satisfied and where the
target element is located on screen.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from terminaleyes.commander.models import (
    ConditionResult,
    ConditionSpec,
    CursorLocateResult,
    ScreenCheckResult,
    ScreenLocation,
)
from terminaleyes.utils.imaging import enhance_for_screen, numpy_to_base64_png, resize_for_mllm

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are a visual screen analyzer. You are given a photograph of a computer screen taken by a webcam, and a condition to check for.

CONDITION TO CHECK:
- Looking for: {description}
- Element type: {element_type}
- Element text: {element_text}
- Visual cues: {visual_cues}
- Spatial context: {spatial_context}

Analyze the screen image carefully. Determine:
1. Is the described condition currently visible on screen?
2. If yes, where on the screen is the element? Estimate position as a fraction from 0.0 to 1.0 where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right.
3. How confident are you in your assessment?

Respond ONLY with JSON (no other text):
{{
    "condition_met": true or false,
    "confidence": 0.0 to 1.0,
    "location_x_pct": 0.0 to 1.0 or null (if not found),
    "location_y_pct": 0.0 to 1.0 or null (if not found),
    "reasoning": "brief explanation of what you see"
}}

IMPORTANT — COLOR MATCHING:
This is a webcam photograph of a monitor. Colors will be shifted, washed out, or distorted by the camera sensor, white balance, ambient lighting, and monitor color profile. When the condition mentions a color (e.g. "lightblue", "red", "green"):
- Treat it as an APPROXIMATE hint, not an exact match
- A "lightblue" button might appear as light blue, cyan, teal, pale blue, gray-blue, or even greenish-blue through the webcam
- A "green" icon might appear as teal, yellow-green, or bright green
- Focus primarily on the TEXT CONTENT, SHAPE, and POSITION of elements rather than exact colors
- If an element matches by text and shape but the color is only roughly similar, still report condition_met=true

OTHER:
- The mouse cursor may be visible and could partially cover UI elements. If you can see an element that matches even though the cursor is on or near it, still report condition_met=true."""

CURSOR_LOCATE_PROMPT = """You are a visual screen analyzer. You are given a photograph of a computer screen taken by a webcam. You need to locate TWO things:

1. The mouse cursor (pointer/arrow) — look for the typical arrow-shaped cursor
2. A target UI element: {target_description}

Estimate positions as fractions from 0.0 to 1.0 where (0.0, 0.0) is the top-left corner and (1.0, 1.0) is the bottom-right corner of the SCREEN area (not the full photograph — focus on the monitor content).

Respond ONLY with JSON (no other text):
{{
    "cursor_found": true or false,
    "cursor_x_pct": 0.0 to 1.0 or null,
    "cursor_y_pct": 0.0 to 1.0 or null,
    "target_found": true or false,
    "target_x_pct": 0.0 to 1.0 or null,
    "target_y_pct": 0.0 to 1.0 or null,
    "cursor_on_target": true or false,
    "reasoning": "brief description of what you see and where"
}}

cursor_on_target should be true ONLY if the cursor tip is visually overlapping or very close to (within ~2% of screen dimensions) the target element.

NOTE: The mouse cursor may partially cover the target element. If you can see the cursor is on top of where the target should be (even if the target text is partially hidden by the cursor), that counts as cursor_on_target=true. Use surrounding context (colors, layout, nearby elements) to confirm the target is under the cursor."""

SCREEN_CHECK_PROMPT = """You are looking at a photograph taken by a webcam pointed at a computer monitor.
Determine whether the ENTIRE monitor screen is visible in the image.

Check all four edges:
- Can you see the top edge/bezel of the monitor?
- Can you see the bottom edge/bezel of the monitor?
- Can you see the left edge/bezel of the monitor?
- Can you see the right edge/bezel of the monitor?

If any part of the screen content is cut off or extends beyond the photograph, report which edges are missing.

Respond ONLY with JSON (no other text):
{{
    "full_screen_visible": true or false,
    "edges_cut_off": ["top", "bottom", "left", "right"],
    "suggestion": "how to adjust the camera (e.g. 'move camera back and to the left')",
    "reasoning": "what you see"
}}

edges_cut_off should be an empty list if full_screen_visible is true.
Only include edges that are actually cut off."""


class ConditionEvaluator:
    """Evaluates visual conditions against webcam frames via MLLM vision."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str = "not-needed",
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI

        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)
        logger.info("ConditionEvaluator initialized (model=%s)", self._model)

    async def check_full_screen(self, image: np.ndarray) -> ScreenCheckResult:
        """Check whether the entire monitor is visible in the webcam frame.

        Should be called at startup before the command loop to ensure
        the camera is positioned correctly.

        Args:
            image: Raw BGR webcam frame (numpy array).

        Returns:
            ScreenCheckResult indicating whether the full screen is visible.
        """
        await self._ensure_client()

        resized = resize_for_mllm(enhance_for_screen(image), max_dimension=768, min_dimension=512)
        b64_image = numpy_to_base64_png(resized)

        messages = [
            {"role": "system", "content": SCREEN_CHECK_PROMPT},
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
                        "text": "Is the entire monitor screen visible in this webcam photo?",
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
            raw_text = self._best_text_from_response(response)

            if not raw_text:
                return ScreenCheckResult(reasoning="Empty MLLM response")

            data = self._extract_json(raw_text)
            if data is None:
                return ScreenCheckResult(
                    reasoning=f"Failed to parse: {raw_text[:200]}",
                    raw_response=raw_text,
                )

            return ScreenCheckResult(
                full_screen_visible=bool(data.get("full_screen_visible", False)),
                edges_cut_off=data.get("edges_cut_off") or [],
                suggestion=data.get("suggestion", ""),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_text,
            )

        except Exception as e:
            logger.error("Screen check failed: %s", e)
            return ScreenCheckResult(
                reasoning=f"Check error: {e}",
            )

    async def evaluate(
        self, image: np.ndarray, condition: ConditionSpec
    ) -> ConditionResult:
        """Evaluate whether the condition is met in the given frame.

        Uses resize_for_mllm() but NOT enhance_for_ocr() — colors must
        be preserved for detecting colored UI elements.

        Args:
            image: Raw BGR webcam frame (numpy array).
            condition: What to look for.

        Returns:
            ConditionResult with match status, confidence, and location.
        """
        await self._ensure_client()

        # Resize smaller for faster inference — 768px max is enough for UI detection
        resized = resize_for_mllm(enhance_for_screen(image), max_dimension=768, min_dimension=512)
        b64_image = numpy_to_base64_png(resized)

        system_prompt = EVALUATOR_SYSTEM_PROMPT.format(
            description=condition.description,
            element_type=condition.element_type or "any",
            element_text=condition.element_text or "any",
            visual_cues=", ".join(condition.visual_cues) if condition.visual_cues else "none",
            spatial_context=condition.spatial_context or "none",
        )

        messages = [
            {"role": "system", "content": system_prompt},
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
                        "text": "Check if the described condition is visible in this screen image.",
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
            raw_text = self._best_text_from_response(response)

            if not raw_text:
                logger.warning("Empty response from MLLM evaluator")
                return ConditionResult(
                    condition_met=False,
                    confidence=0.0,
                    reasoning="Empty MLLM response",
                    raw_response="",
                )

            logger.debug("Evaluator raw response: %s", raw_text[:300])
            return self._parse_response(raw_text)

        except Exception as e:
            logger.error("Condition evaluation failed: %s", e)
            return ConditionResult(
                condition_met=False,
                confidence=0.0,
                reasoning=f"Evaluation error: {e}",
                raw_response="",
            )

    async def locate_cursor(
        self, image: np.ndarray, target_description: str
    ) -> CursorLocateResult:
        """Locate the cursor and a target element in the given frame.

        Used by the visual homing loop to guide the cursor toward a
        target element by taking screenshots and correcting course.

        Args:
            image: Raw BGR webcam frame (numpy array).
            target_description: Human-readable description of the target element.

        Returns:
            CursorLocateResult with positions of cursor and target.
        """
        await self._ensure_client()

        resized = resize_for_mllm(enhance_for_screen(image))
        b64_image = numpy_to_base64_png(resized)

        system_prompt = CURSOR_LOCATE_PROMPT.format(
            target_description=target_description,
        )

        messages = [
            {"role": "system", "content": system_prompt},
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
                        "text": "Locate the mouse cursor and the target element in this screen image.",
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
            raw_text = self._best_text_from_response(response)

            if not raw_text:
                return CursorLocateResult(reasoning="Empty MLLM response")

            logger.debug("Cursor locate response: %s", raw_text[:300])
            return self._parse_cursor_response(raw_text)

        except Exception as e:
            logger.error("Cursor locate failed: %s", e)
            return CursorLocateResult(
                reasoning=f"Locate error: {e}",
                raw_response="",
            )

    def _parse_cursor_response(self, raw: str) -> CursorLocateResult:
        """Parse MLLM response into a CursorLocateResult."""
        data = self._extract_json(raw)

        if data is None:
            return CursorLocateResult(
                reasoning=f"Failed to parse: {raw[:200]}",
                raw_response=raw,
            )

        cursor_loc = None
        cx = data.get("cursor_x_pct")
        cy = data.get("cursor_y_pct")
        if cx is not None and cy is not None:
            cursor_loc = ScreenLocation(
                x_pct=max(0.0, min(1.0, float(cx))),
                y_pct=max(0.0, min(1.0, float(cy))),
            )

        target_loc = None
        tx = data.get("target_x_pct")
        ty = data.get("target_y_pct")
        if tx is not None and ty is not None:
            target_loc = ScreenLocation(
                x_pct=max(0.0, min(1.0, float(tx))),
                y_pct=max(0.0, min(1.0, float(ty))),
            )

        return CursorLocateResult(
            cursor_found=bool(data.get("cursor_found", False)),
            cursor_location=cursor_loc,
            target_found=bool(data.get("target_found", False)),
            target_location=target_loc,
            cursor_on_target=bool(data.get("cursor_on_target", False)),
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )

    @staticmethod
    def _best_text_from_response(response) -> str:
        """Extract the best text from a chat completion response.

        Gemma-4-31b often puts a truncated JSON fragment in `content`
        and the complete reasoning (with full JSON) in `reasoning_content`.
        This method tries `content` first; if it doesn't contain valid JSON,
        falls back to `reasoning_content`.
        """
        content = response.choices[0].message.content or ""
        reasoning = getattr(
            response.choices[0].message, "reasoning_content", None
        ) or ""

        # If content has extractable JSON, use it
        if content.strip():
            data = ConditionEvaluator._extract_json(content)
            if data is not None:
                return content

        # Content empty or has no valid JSON — try reasoning_content
        if reasoning.strip():
            return reasoning

        # Return whatever we have
        return content

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Extract a JSON object from raw MLLM response text."""
        json_str = raw.strip()

        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", json_str)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    def _parse_response(self, raw: str) -> ConditionResult:
        """Parse MLLM response into a ConditionResult."""
        data = self._extract_json(raw)

        if data is None:
            logger.warning("Failed to parse evaluator response as JSON")
            # Try to infer from text
            met = any(
                word in raw.lower()
                for word in ["yes", "visible", "found", "detected", "true"]
            )
            return ConditionResult(
                condition_met=met,
                confidence=0.3 if met else 0.2,
                reasoning=f"Fallback text parse: {raw[:200]}",
                raw_response=raw,
            )

        condition_met = bool(data.get("condition_met", False))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        location = None
        x_pct = data.get("location_x_pct")
        y_pct = data.get("location_y_pct")
        if x_pct is not None and y_pct is not None:
            location = ScreenLocation(
                x_pct=max(0.0, min(1.0, float(x_pct))),
                y_pct=max(0.0, min(1.0, float(y_pct))),
                confidence=confidence,
            )

        return ConditionResult(
            condition_met=condition_met,
            confidence=confidence,
            location=location,
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )
