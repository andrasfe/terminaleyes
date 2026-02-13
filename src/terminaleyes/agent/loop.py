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
        self._stale_count = 0
        self._dup_suppress_count = 0

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

                            # Suppress duplicate consecutive text commands
                            if (
                                action is not None
                                and isinstance(action, TextInput)
                                and context.action_history
                            ):
                                last = context.action_history[-1]
                                if (
                                    isinstance(last.action, TextInput)
                                    and last.action.text == action.text
                                ):
                                    self._dup_suppress_count += 1
                                    logger.info(
                                        "Suppressing duplicate command (%d): %s",
                                        self._dup_suppress_count,
                                        action.text.strip()[:50],
                                    )
                                    # Tell strategy about suppression so MLLM can adjust
                                    context.metadata["suppressed_command"] = action.text.strip()
                                    context.metadata["suppress_count"] = str(self._dup_suppress_count)

                                    if self._dup_suppress_count >= 3:
                                        logger.info("Duplicate suppression limit reached — assuming command output is visible and task progressed")
                                        self._dup_suppress_count = 0
                                        # Remove suppression metadata and let MLLM re-evaluate
                                        context.metadata.pop("suppressed_command", None)
                                        context.metadata.pop("suppress_count", None)
                                        # Force MLLM to re-evaluate with explicit hint
                                        context.metadata["force_reevaluate"] = "true"
                                        action = None
                                        reasoning = "Re-evaluating after repeated duplicate suppression"
                                    else:
                                        action = None
                                        reasoning = "Waiting for previous command to produce output"
                                else:
                                    self._dup_suppress_count = 0
                                    context.metadata.pop("suppressed_command", None)
                                    context.metadata.pop("suppress_count", None)
                                    context.metadata.pop("force_reevaluate", None)

                            # Detect stale repeated keystrokes (same key + same screen = no effect)
                            if (
                                action is not None
                                and isinstance(action, Keystroke)
                                and len(context.action_history) >= 3
                            ):
                                recent_3 = context.action_history[-3:]
                                all_same_key = all(
                                    isinstance(a.action, Keystroke) and a.action.key == action.key
                                    for a in recent_3
                                )
                                if all_same_key:
                                    # Check if terminal content changed
                                    recent_texts = [
                                        a.terminal_state_before.content.visible_text.strip()
                                        for a in recent_3
                                    ]
                                    current_text = observation.content.visible_text.strip()
                                    if all(t == current_text for t in recent_texts):
                                        self._stale_count += 1
                                        logger.info(
                                            "Stale keystroke detected (%d): '%s' has no effect",
                                            self._stale_count, action.key,
                                        )
                                        if self._stale_count >= 3:
                                            logger.info("Stale limit reached — assuming task is complete")
                                            goal.status = TaskStatus.COMPLETED
                                            break
                                        action = None
                                        reasoning = "Keystroke has no effect — screen unchanged"
                                    else:
                                        self._stale_count = 0
                                else:
                                    self._stale_count = 0

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
                                # Text commands that include newline trigger shell
                                # execution -- wait longer for output to render
                                if isinstance(action, TextInput) and "\n" in action.text:
                                    await asyncio.sleep(self._action_delay + 2.0)
                                else:
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
        """Capture a frame and interpret it via the MLLM.

        Uses double-read validation: captures and interprets twice (0.5s apart),
        proceeding with the better read. If both reads agree (>= 80% word overlap),
        confidence is boosted. This catches transient misreads without rejecting
        legitimate low-confidence content.
        """
        frame1 = await self._capture.capture_frame()
        state1 = await self._interpreter.interpret(frame1)

        await asyncio.sleep(0.5)

        frame2 = await self._capture.capture_frame()
        state2 = await self._interpreter.interpret(frame2)

        text1 = state1.content.visible_text.strip()
        text2 = state2.content.visible_text.strip()

        words1 = set(text1.split())
        words2 = set(text2.split())

        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
        elif not words1 and not words2:
            overlap = 1.0
        else:
            overlap = 0.0

        if overlap >= 0.8:
            logger.debug("Double-read validated (%.0f%% overlap)", overlap * 100)
            # Use whichever read has more content (longer visible_text)
            best = state1 if len(text1) >= len(text2) else state2
            return best
        else:
            logger.debug("Double-read diverged (%.0f%% overlap), using longer read", overlap * 100)
            return state1 if len(text1) >= len(text2) else state2

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
