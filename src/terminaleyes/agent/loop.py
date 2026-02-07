"""The central agent loop that orchestrates the entire system.

Ties together vision capture, MLLM interpretation, strategy-based
decision making, and keyboard action output.
"""

from __future__ import annotations

import asyncio
import logging

from terminaleyes.agent.base import AgentStrategy
from terminaleyes.capture.base import CaptureSource
from terminaleyes.domain.models import (
    AgentAction,
    AgentContext,
    AgentGoal,
    KeyCombo,
    Keystroke,
    TaskStatus,
    TerminalState,
    TextInput,
)
from terminaleyes.interpreter.base import MLLMProvider
from terminaleyes.keyboard.base import KeyboardOutput

logger = logging.getLogger(__name__)


class AgentLoop:
    """The central orchestrator for the vision-action agent loop.

    Coordinates: capture -> interpret -> decide -> act -> repeat
    """

    def __init__(
        self,
        capture: CaptureSource,
        interpreter: MLLMProvider,
        keyboard: KeyboardOutput,
        strategy: AgentStrategy,
        capture_interval: float = 2.0,
        action_delay: float = 1.0,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._capture = capture
        self._interpreter = interpreter
        self._keyboard = keyboard
        self._strategy = strategy
        self._capture_interval = capture_interval
        self._action_delay = action_delay
        self._max_consecutive_errors = max_consecutive_errors
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def run(self, goal: AgentGoal) -> AgentContext:
        """Execute the agent loop for the given goal."""
        self._running = True
        goal.status = TaskStatus.IN_PROGRESS
        context = AgentContext(current_goal=goal)
        consecutive_errors = 0

        logger.info("Agent loop starting: %s", goal.description)

        try:
            async with self._capture:
                async with self._keyboard:
                    while self._running:
                        try:
                            # 1. Capture and interpret
                            observation = await self._capture_and_interpret()
                            context.observation_history.append(observation)
                            consecutive_errors = 0

                            logger.info(
                                "Step %d | Terminal: %s | Confidence: %.2f",
                                context.step_count + 1,
                                observation.readiness.value,
                                observation.confidence,
                            )

                            # 2. Decide action
                            action, reasoning = await self._strategy.decide_action(
                                context, observation
                            )
                            logger.info("Decision: %s | %s",
                                       action.action_type if action else "wait",
                                       reasoning[:100])

                            # 3. Execute action
                            if action is not None:
                                await self._execute_action(action)
                                agent_action = AgentAction(
                                    step_number=context.step_count,
                                    action=action,
                                    reasoning=reasoning,
                                    terminal_state_before=observation,
                                )
                                context.action_history.append(agent_action)
                                await asyncio.sleep(self._action_delay)

                            # 4. Evaluate completion
                            status = await self._strategy.evaluate_completion(
                                context, observation
                            )
                            if status == TaskStatus.COMPLETED:
                                goal.status = TaskStatus.COMPLETED
                                logger.info("Goal completed: %s", reasoning)
                                break
                            elif status == TaskStatus.FAILED:
                                goal.status = TaskStatus.FAILED
                                logger.warning("Goal failed: %s", reasoning)
                                break

                            # 5. Check step limit
                            if context.is_over_limit:
                                goal.status = TaskStatus.FAILED
                                logger.warning(
                                    "Step limit reached (%d)", goal.max_steps
                                )
                                break

                            # Wait before next capture
                            if action is None:
                                await asyncio.sleep(self._capture_interval)

                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(
                                "Error in agent loop (attempt %d/%d): %s",
                                consecutive_errors,
                                self._max_consecutive_errors,
                                e,
                            )
                            if consecutive_errors >= self._max_consecutive_errors:
                                goal.status = TaskStatus.FAILED
                                logger.error("Too many consecutive errors, aborting")
                                break
                            await asyncio.sleep(self._capture_interval)

        except Exception as e:
            goal.status = TaskStatus.FAILED
            logger.error("Agent loop fatal error: %s", e)

        self._running = False
        logger.info("Agent loop finished: status=%s, steps=%d",
                    goal.status.value, context.step_count)
        return context

    async def stop(self) -> None:
        """Signal the agent loop to stop gracefully."""
        self._running = False
        logger.info("Agent loop stop requested")

    async def _capture_and_interpret(self) -> TerminalState:
        """Capture a frame and interpret it via the MLLM."""
        frame = await self._capture.capture_frame()
        logger.debug("Captured frame %d", frame.frame_number)
        state = await self._interpreter.interpret(frame)
        return state

    async def _execute_action(self, action: Keystroke | KeyCombo | TextInput) -> None:
        """Execute a keyboard action via the output backend."""
        if isinstance(action, Keystroke):
            await self._keyboard.send_keystroke(action.key)
        elif isinstance(action, KeyCombo):
            await self._keyboard.send_key_combo(action.modifiers, action.key)
        elif isinstance(action, TextInput):
            await self._keyboard.send_text(action.text)
        else:
            logger.warning("Unknown action type: %s", type(action))
