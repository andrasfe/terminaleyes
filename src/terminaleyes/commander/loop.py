"""Command loop — periodic capture, condition evaluation, and action execution.

Combines the passive observation pattern from watcher/ (periodic capture,
change detection) with active execution from agent/ (action dispatch),
but uses general-purpose screen condition evaluation instead of
terminal-specific interpretation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

import cv2

from terminaleyes.capture.base import CaptureSource
from terminaleyes.commander.evaluator import ConditionEvaluator
from terminaleyes.commander.executor import ActionExecutor
from terminaleyes.commander.models import CommandSession, CommandSpec
from terminaleyes.watcher.change import has_frame_changed, is_frame_usable

logger = logging.getLogger(__name__)


class CommandLoop:
    """Orchestrates periodic capture → evaluate → act loop."""

    def __init__(
        self,
        capture: CaptureSource,
        evaluator: ConditionEvaluator,
        executor: ActionExecutor,
        confidence_threshold: float = 0.7,
        change_threshold: float = 0.02,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._capture = capture
        self._evaluator = evaluator
        self._executor = executor
        self._confidence_threshold = confidence_threshold
        self._change_threshold = change_threshold
        self._max_consecutive_errors = max_consecutive_errors
        self._stopped = False

    def stop(self) -> None:
        """Signal the loop to stop gracefully."""
        self._stopped = True

    async def _verify_full_screen(self) -> None:
        """Check that the entire monitor is visible in the webcam frame.

        Loops until the MLLM confirms the full screen is visible,
        printing instructions for the user to adjust the camera.
        """
        print("  Checking camera view...")
        while not self._stopped:
            frame = await self._capture.capture_frame()
            result = await self._evaluator.check_full_screen(frame.image)

            if result.full_screen_visible:
                print("  Full screen visible — camera position OK")
                print()
                return

            edges = ", ".join(result.edges_cut_off) if result.edges_cut_off else "unknown"
            print(f"\n  WARNING: Not all of the monitor is visible!")
            print(f"  Edges cut off: {edges}")
            if result.suggestion:
                print(f"  Suggestion: {result.suggestion}")
            if result.reasoning:
                print(f"  Details: {result.reasoning[:150]}")

            print(f"\n  Please adjust the camera so the entire screen is visible.")
            print(f"  Rechecking in 10 seconds... (Ctrl-C to skip)")

            # Wait 10s but check stop flag
            for _ in range(100):
                if self._stopped:
                    return
                await asyncio.sleep(0.1)

    async def run(self, command: CommandSpec) -> CommandSession:
        """Execute the command loop: capture → evaluate → act → repeat.

        Args:
            command: The parsed command specification.

        Returns:
            CommandSession with execution results.
        """
        session = CommandSession(
            session_id=uuid.uuid4().hex[:12],
            command=command,
            started_at=datetime.now(),
        )

        prev_gray = None
        consecutive_errors = 0

        print(f"\nCommand loop started (session {session.session_id})")
        print(f"  Watching for: {command.condition.description}")
        print(f"  Action: {command.action.action_type} ({command.action.button or command.action.key or command.action.text or ''})")
        print(f"  Interval: {command.interval_seconds}s")
        print(f"  Mode: {'one-shot' if command.one_shot else 'continuous'}")
        print()

        async with self._capture:
            # Pre-check: verify the full monitor is visible
            await self._verify_full_screen()

            while not self._stopped and session.status == "running":
                try:
                    # 1. Capture frame
                    frame = await self._capture.capture_frame()
                    curr_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

                    # 2. Quality gate
                    usable, reason = is_frame_usable(curr_gray)
                    if not usable:
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] Skipped: {reason}")
                        await asyncio.sleep(command.interval_seconds)
                        continue

                    # 3. Change detection gate (skip if screen unchanged)
                    if prev_gray is not None and not has_frame_changed(
                        prev_gray, curr_gray, self._change_threshold
                    ):
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] No change detected, skipping evaluation")
                        await asyncio.sleep(command.interval_seconds)
                        continue

                    prev_gray = curr_gray

                    # 4. Evaluate condition via MLLM vision
                    session.frames_checked += 1
                    session.last_check_at = datetime.now()
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] Evaluating frame #{session.frames_checked}...")

                    result = await self._evaluator.evaluate(
                        frame.image, command.condition
                    )

                    print(
                        f"  → condition_met={result.condition_met}, "
                        f"confidence={result.confidence:.2f}"
                    )
                    if result.reasoning:
                        print(f"  → {result.reasoning[:120]}")

                    # 5. Act if condition met with sufficient confidence
                    if (
                        result.condition_met
                        and result.confidence >= self._confidence_threshold
                    ):
                        session.condition_met_count += 1
                        print(f"  ✓ Condition met! Executing action...")

                        await self._executor.execute(
                            command.action,
                            result.location,
                            target_description=command.condition.description,
                        )
                        session.actions_executed += 1
                        print(f"  ✓ Action executed successfully")

                        if command.one_shot:
                            session.status = "triggered"
                            break
                    elif result.condition_met:
                        print(
                            f"  → Condition detected but confidence "
                            f"({result.confidence:.2f}) below threshold "
                            f"({self._confidence_threshold:.2f})"
                        )

                    # 6. Check attempt limits
                    if (
                        command.max_attempts > 0
                        and session.frames_checked >= command.max_attempts
                    ):
                        print(f"\nMax attempts ({command.max_attempts}) reached")
                        session.status = "completed"
                        break

                    consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    logger.error("Loop error (%d/%d): %s",
                                 consecutive_errors,
                                 self._max_consecutive_errors, e)
                    print(f"  Error: {e}")
                    if consecutive_errors >= self._max_consecutive_errors:
                        print(f"\nToo many consecutive errors, stopping")
                        session.status = "failed"
                        break

                # 7. Wait for next interval
                if session.status == "running":
                    await asyncio.sleep(command.interval_seconds)

        if self._stopped and session.status == "running":
            session.status = "cancelled"

        print(f"\nSession {session.session_id} finished: {session.status}")
        print(f"  Frames checked: {session.frames_checked}")
        print(f"  Conditions met: {session.condition_met_count}")
        print(f"  Actions executed: {session.actions_executed}")

        return session
