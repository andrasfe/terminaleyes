"""Command parser — decomposes natural language instructions into CommandSpec.

Uses the MLLM (text-only, no image) to parse free-form instructions like
"when you see a blue Run button, click it" into structured condition + action specs.
"""

from __future__ import annotations

import json
import logging
import re

from terminaleyes.commander.models import ActionSpec, CommandSpec, ConditionSpec

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are a command parser for a screen automation system.
Given a natural language instruction, extract the structured components.

The system can:
- Capture a computer screen via webcam at configurable intervals
- Detect visual elements (buttons, text, icons, colors, UI components)
- Click mouse buttons (left, right, middle) on detected elements
- Move mouse to detected elements
- Type keystrokes and text
- Send key combinations (e.g. Ctrl+C)

Parse the instruction into JSON:
{
    "condition": {
        "description": "human-readable description of what to look for",
        "element_type": "button|text|icon|dialog|field|link|menu|null",
        "element_text": "text on the element, or null if not text-specific",
        "visual_cues": ["color", "shape", "size", "position"],
        "spatial_context": "relative position info (e.g. 'next to X', 'below Y'), or null"
    },
    "action": {
        "action_type": "mouse_click|keystroke|key_combo|text_input",
        "button": "left|right|middle (for mouse_click, default left)",
        "key": "key name (for keystroke/key_combo, e.g. Enter, Tab)",
        "modifiers": ["ctrl", "shift", "alt", "meta"],
        "text": "text to type (for text_input)",
        "target": "element (click the detected element) or current (click in place)"
    },
    "interval_seconds": 180,
    "one_shot": true,
    "max_attempts": 0
}

Rules:
- interval_seconds: extract from instruction if mentioned (e.g. "every 3 minutes" = 180), default 180
- one_shot: true if the instruction implies a single action, false if it should keep watching
- max_attempts: 0 means unlimited, set to a number if the instruction implies a limit
- For "click" without specifying which button, default to "left"
- target should be "element" when clicking on a specific UI element the system detects
- Colors are approximate — the system uses a webcam pointed at a monitor, so colors will be shifted. Put colors in visual_cues but the description should emphasize text content and element type over exact color"""


class CommandParser:
    """Parses natural language instructions into structured CommandSpec."""

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
        logger.info("CommandParser initialized (model=%s)", self._model)

    async def parse(self, instruction: str) -> CommandSpec:
        """Parse a natural language instruction into a CommandSpec.

        Args:
            instruction: Free-form command like "when you see a blue Run button, click it".

        Returns:
            Structured CommandSpec ready for the command loop.

        Raises:
            CommandParseError: If the instruction cannot be parsed.
        """
        await self._ensure_client()

        messages = [
            {"role": "system", "content": PARSER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Parse this instruction:\n\n{instruction}",
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
            )
            # Gemma-4-31b: content may be empty or have truncated JSON,
            # full answer is often in reasoning_content
            content = response.choices[0].message.content or ""
            reasoning = getattr(
                response.choices[0].message, "reasoning_content", None
            ) or ""

            # Try content first; if it has no valid JSON, use reasoning
            raw_text = ""
            if content.strip():
                from terminaleyes.commander.evaluator import ConditionEvaluator
                if ConditionEvaluator._extract_json(content) is not None:
                    raw_text = content
            if not raw_text and reasoning.strip():
                raw_text = reasoning
            if not raw_text:
                raw_text = content  # last resort

            if not raw_text.strip():
                raise CommandParseError(
                    "Empty response from MLLM", raw_response=""
                )

            logger.debug("Parser raw response: %s", raw_text[:300])
            return self._parse_response(raw_text, instruction)

        except CommandParseError:
            raise
        except Exception as e:
            raise CommandParseError(
                f"MLLM call failed: {e}", raw_response=""
            ) from e

    def _parse_response(self, raw: str, instruction: str) -> CommandSpec:
        """Extract CommandSpec from MLLM response text."""
        json_str = raw.strip()

        # Extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()

        # Find JSON object
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing invalid escapes
            try:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", json_str)
                data = json.loads(fixed)
            except json.JSONDecodeError:
                raise CommandParseError(
                    f"Failed to parse JSON from response: {raw[:200]}",
                    raw_response=raw,
                )

        cond_data = data.get("condition", {})
        action_data = data.get("action", {})

        condition = ConditionSpec(
            description=cond_data.get("description", instruction),
            element_type=cond_data.get("element_type"),
            element_text=cond_data.get("element_text"),
            visual_cues=cond_data.get("visual_cues") or [],
            spatial_context=cond_data.get("spatial_context"),
        )

        action = ActionSpec(
            action_type=action_data.get("action_type", "mouse_click"),
            button=action_data.get("button", "left"),
            key=action_data.get("key"),
            modifiers=action_data.get("modifiers") or [],
            text=action_data.get("text"),
            target=action_data.get("target", "element"),
        )

        return CommandSpec(
            raw_instruction=instruction,
            condition=condition,
            action=action,
            interval_seconds=float(data.get("interval_seconds", 180)),
            max_attempts=int(data.get("max_attempts", 0)),
            one_shot=bool(data.get("one_shot", True)),
        )


class CommandParseError(Exception):
    """Raised when command parsing fails."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
