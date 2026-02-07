"""MLLM-driven agent strategy.

Uses the same MLLM provider to decide what action to take next,
given the current terminal state, goal, and history. This is the
most general strategy -- the LLM itself decides what to do.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Union

from terminaleyes.agent.base import AgentStrategy
from terminaleyes.domain.models import (
    AgentContext,
    KeyCombo,
    Keystroke,
    TaskStatus,
    TerminalReadiness,
    TerminalState,
    TextInput,
)
from terminaleyes.interpreter.base import MLLMProvider

logger = logging.getLogger(__name__)


DECISION_PROMPT = """You are an autonomous terminal agent. You can see a terminal screen and must decide what keyboard action to take next.

CURRENT GOAL: {goal_description}
SUCCESS CRITERIA: {success_criteria}

TERMINAL STATE:
- Readiness: {readiness}
- Visible text: {visible_text}
- Last command: {last_command}
- Last output: {last_output}
- Prompt: {prompt_text}
- Errors: {errors}
- Working directory: {working_directory}

STEP {step_number} of maximum {max_steps}

PREVIOUS ACTIONS (last 5):
{recent_actions}

Decide the SINGLE next action. Respond ONLY with valid JSON:

For typing a command (text followed by Enter):
{{"action_type": "text_input", "text": "your command here\\n", "reasoning": "why"}}

For a single keystroke:
{{"action_type": "keystroke", "key": "Enter", "reasoning": "why"}}

For a key combination:
{{"action_type": "key_combo", "modifiers": ["ctrl"], "key": "c", "reasoning": "why"}}

To indicate the goal is COMPLETE:
{{"action_type": "done", "status": "completed", "reasoning": "why it's done"}}

To indicate the goal FAILED:
{{"action_type": "done", "status": "failed", "reasoning": "why it failed"}}

If the terminal is busy, wait:
{{"action_type": "wait", "reasoning": "waiting for command to finish"}}
"""


class MLLMDrivenStrategy(AgentStrategy):
    """Strategy that uses an MLLM to decide every action.

    The LLM receives the current terminal state, goal, and history,
    and decides what action to take next. This is the most flexible
    strategy but consumes the most API calls.
    """

    def __init__(self, mllm: MLLMProvider) -> None:
        self._mllm = mllm
        self._last_status: TaskStatus | None = None

    @property
    def name(self) -> str:
        return "mllm-driven"

    async def decide_action(
        self,
        context: AgentContext,
        observation: TerminalState,
    ) -> tuple[Keystroke | KeyCombo | TextInput | None, str]:
        """Ask the MLLM what action to take next."""
        # Build recent actions summary
        recent = context.action_history[-5:]
        actions_text = "\n".join(
            f"  Step {a.step_number}: {a.action.action_type} - {a.reasoning}"
            for a in recent
        ) or "  (none yet)"

        prompt = DECISION_PROMPT.format(
            goal_description=context.current_goal.description,
            success_criteria=context.current_goal.success_criteria,
            readiness=observation.readiness.value,
            visible_text=observation.content.visible_text[:500],
            last_command=observation.content.last_command or "(none)",
            last_output=(observation.content.last_output or "(none)")[:300],
            prompt_text=observation.content.prompt_text or "(none)",
            errors=", ".join(observation.content.error_messages) or "(none)",
            working_directory=observation.content.working_directory or "(unknown)",
            step_number=context.step_count + 1,
            max_steps=context.current_goal.max_steps,
            recent_actions=actions_text,
        )

        try:
            # Use the MLLM to decide (text-only, no image needed)
            from openai import AsyncOpenAI
            # Reuse the same client configuration from the provider
            client = self._mllm._client
            if client is None:
                await self._mllm._ensure_client()
                client = self._mllm._client

            response = await client.chat.completions.create(
                model=self._mllm.model,
                max_tokens=512,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            logger.debug("Strategy MLLM response: %s", raw[:200])

            return self._parse_decision(raw)
        except Exception as e:
            logger.error("Strategy decision failed: %s", e)
            return None, f"Decision error: {e}"

    def _parse_decision(
        self, raw: str
    ) -> tuple[Keystroke | KeyCombo | TextInput | None, str]:
        """Parse the MLLM's decision response."""
        # Extract JSON
        json_str = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None, f"Failed to parse decision: {raw[:100]}"

        action_type = data.get("action_type", "")
        reasoning = data.get("reasoning", "")

        if action_type == "text_input":
            return TextInput(text=data["text"]), reasoning
        elif action_type == "keystroke":
            return Keystroke(key=data["key"]), reasoning
        elif action_type == "key_combo":
            return KeyCombo(
                modifiers=data.get("modifiers", []),
                key=data["key"],
            ), reasoning
        elif action_type == "done":
            status = data.get("status", "completed")
            self._last_status = (
                TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED
            )
            return None, reasoning
        elif action_type == "wait":
            return None, reasoning
        else:
            return None, f"Unknown action type: {action_type}"

    async def evaluate_completion(
        self,
        context: AgentContext,
        observation: TerminalState,
    ) -> TaskStatus:
        """Check if the MLLM signaled completion."""
        if self._last_status is not None:
            status = self._last_status
            self._last_status = None
            return status
        if context.is_over_limit:
            return TaskStatus.FAILED
        return TaskStatus.IN_PROGRESS
