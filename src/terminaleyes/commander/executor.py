"""Action executor — dispatches mouse and keyboard actions.

For mouse clicks on elements, uses a pure visual homing loop:
1. Screenshot → MLLM reports cursor and target positions
2. Compute correction → send small calibrated move
3. Screenshot → verify → repeat until cursor is on target
4. Click

No large blind moves. Each step is visually verified.
Corrections refine the calibration over time.
"""

from __future__ import annotations

import asyncio
import logging

from terminaleyes.capture.base import CaptureSource
from terminaleyes.commander.calibration import (
    MOVE_DELAY,
    MOVE_STEP_SIZE,
    CalibrationResult,
    DEFAULT_CALIBRATION,
    MouseCalibrator,
)
from terminaleyes.commander.evaluator import ConditionEvaluator
from terminaleyes.commander.models import ActionSpec, ScreenLocation
from terminaleyes.keyboard.base import KeyboardOutput
from terminaleyes.mouse.base import MouseOutput

logger = logging.getLogger(__name__)

CURSOR_TOLERANCE = 0.03  # 3% of screen


class ActionExecutor:
    """Dispatches actions to mouse and keyboard output backends.

    Mouse clicks use a pure visual homing loop — every move is followed
    by a screenshot and MLLM verification. No blind large moves.
    """

    def __init__(
        self,
        keyboard: KeyboardOutput,
        mouse: MouseOutput,
        screen_width: int = 1920,
        screen_height: int = 1080,
        capture: CaptureSource | None = None,
        evaluator: ConditionEvaluator | None = None,
        max_homing_steps: int = 10,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._capture = capture
        self._evaluator = evaluator
        self._max_homing_steps = max_homing_steps
        self._calibration: CalibrationResult | None = None

    async def execute(
        self,
        action: ActionSpec,
        location: ScreenLocation | None = None,
        target_description: str = "",
    ) -> None:
        """Execute the specified action, optionally at the given location."""
        self._target_description = target_description
        try:
            if action.action_type == "mouse_click":
                await self._execute_mouse_click(action, location)
            elif action.action_type == "keystroke":
                await self._execute_keystroke(action)
            elif action.action_type == "key_combo":
                await self._execute_key_combo(action)
            elif action.action_type == "text_input":
                await self._execute_text_input(action)
            else:
                raise ActionExecutionError(
                    f"Unknown action type: {action.action_type}"
                )
        except ActionExecutionError:
            raise
        except Exception as e:
            raise ActionExecutionError(
                f"Action execution failed: {e}"
            ) from e

    async def _execute_mouse_click(
        self,
        action: ActionSpec,
        location: ScreenLocation | None,
    ) -> None:
        button = action.button or "left"

        if action.target == "element" and location is not None:
            if self._capture is not None and self._evaluator is not None:
                await self._visual_homing_click(action, location, button)
            else:
                await self._mouse.click_at(
                    x_pct=location.x_pct, y_pct=location.y_pct,
                    button=button,
                    screen_width=self._screen_width,
                    screen_height=self._screen_height,
                )
        else:
            await self._mouse.click(button)

    async def _visual_homing_click(
        self,
        action: ActionSpec,
        initial_location: ScreenLocation,
        button: str,
    ) -> None:
        """Move cursor to target using pure visual feedback — no blind moves.

        Every step: screenshot → MLLM locates cursor+target → small correction.
        """
        cal = self._calibration or DEFAULT_CALIBRATION
        target_desc = self._target_description or action.text or "the target element"

        print(f"    Homing to target...")

        # Cap each correction to 20% of screen to prevent wild jumps
        MAX_CORRECTION_PCT = 0.20

        for step in range(self._max_homing_steps):
            # Screenshot and ask MLLM where everything is
            frame = await self._capture.capture_frame()
            result = await self._evaluator.locate_cursor(
                frame.image, target_desc
            )

            reasoning = result.reasoning[:120] if result.reasoning else ""
            print(f"    Step {step + 1}/{self._max_homing_steps}: {reasoning}")

            # Cursor is on target → click
            if result.cursor_on_target:
                print(f"    ON TARGET — clicking {button}")
                await self._mouse.click(button)
                return

            # Target not visible → give up
            if not result.target_found or result.target_location is None:
                print(f"    Target not found — cannot click")
                return

            target_loc = result.target_location

            if result.cursor_found and result.cursor_location is not None:
                cursor_loc = result.cursor_location
                print(f"    Cursor: ({cursor_loc.x_pct:.1%}, {cursor_loc.y_pct:.1%})"
                      f"  Target: ({target_loc.x_pct:.1%}, {target_loc.y_pct:.1%})")

                dx_pct = target_loc.x_pct - cursor_loc.x_pct
                dy_pct = target_loc.y_pct - cursor_loc.y_pct

                # Close enough → click
                if abs(dx_pct) < CURSOR_TOLERANCE and abs(dy_pct) < CURSOR_TOLERANCE:
                    print(f"    Close enough — clicking {button}")
                    await self._mouse.click(button)
                    return

                # Cap each step to prevent overshoot
                dx_pct = max(-MAX_CORRECTION_PCT, min(MAX_CORRECTION_PCT, dx_pct))
                dy_pct = max(-MAX_CORRECTION_PCT, min(MAX_CORRECTION_PCT, dy_pct))

                dx_hid, dy_hid = cal.hid_units_for_pct(dx_pct, dy_pct)
                print(f"    Moving ({dx_pct:+.1%}, {dy_pct:+.1%}) → ({dx_hid:+d}, {dy_hid:+d}) HID")
                await self._send_calibrated_moves(dx_hid, dy_hid)

                # Refine calibration from this observation
                self._refine_calibration(cal, dx_pct, dy_pct, dx_hid, dy_hid)

            else:
                print(f"    Cursor not visible (target at {target_loc.x_pct:.1%}, {target_loc.y_pct:.1%})")
                # Small nudge toward target from assumed center
                dx_pct = max(-MAX_CORRECTION_PCT, min(MAX_CORRECTION_PCT, target_loc.x_pct - 0.5))
                dy_pct = max(-MAX_CORRECTION_PCT, min(MAX_CORRECTION_PCT, target_loc.y_pct - 0.5))
                dx_hid, dy_hid = cal.hid_units_for_pct(dx_pct, dy_pct)
                print(f"    Nudging ({dx_hid:+d}, {dy_hid:+d}) HID")
                await self._send_calibrated_moves(dx_hid, dy_hid)

            await asyncio.sleep(0.3)

        print(f"    Max steps reached — could not reach target")

    def _refine_calibration(
        self,
        cal: CalibrationResult,
        dx_pct: float, dy_pct: float,
        dx_hid: int, dy_hid: int,
    ) -> None:
        """Nudge calibration based on observed corrections (10% EMA)."""
        if abs(dx_pct) < 0.05 and abs(dy_pct) < 0.05:
            return

        alpha = 0.1

        if abs(dx_pct) >= 0.05 and dx_hid != 0:
            observed_x = abs(dx_hid) / abs(dx_pct)
            cal.hid_units_per_full_x = (
                (1 - alpha) * cal.hid_units_per_full_x + alpha * observed_x
            )

        if abs(dy_pct) >= 0.05 and dy_hid != 0:
            observed_y = abs(dy_hid) / abs(dy_pct)
            cal.hid_units_per_full_y = (
                (1 - alpha) * cal.hid_units_per_full_y + alpha * observed_y
            )

        cal.save()
        logger.debug(
            "Calibration refined: x=%.0f, y=%.0f",
            cal.hid_units_per_full_x, cal.hid_units_per_full_y,
        )

    async def _send_calibrated_moves(self, dx_hid: int, dy_hid: int) -> None:
        """Send HID moves 1 unit at a time with slow delay for linearity."""
        remaining_x = dx_hid
        remaining_y = dy_hid

        while remaining_x != 0 or remaining_y != 0:
            step_x = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, remaining_x))
            step_y = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, remaining_y))
            if step_x != 0 or step_y != 0:
                await self._mouse.move(step_x, step_y)
            remaining_x -= step_x
            remaining_y -= step_y
            await asyncio.sleep(MOVE_DELAY)

    async def _execute_keystroke(self, action: ActionSpec) -> None:
        key = action.key or "Enter"
        logger.info("Sending keystroke: %s", key)
        await self._keyboard.send_keystroke(key)

    async def _execute_key_combo(self, action: ActionSpec) -> None:
        key = action.key or ""
        modifiers = action.modifiers or []
        logger.info("Sending key combo: %s+%s", "+".join(modifiers), key)
        await self._keyboard.send_key_combo(modifiers, key)

    async def _execute_text_input(self, action: ActionSpec) -> None:
        text = action.text or ""
        logger.info("Sending text: %s", text[:50])
        await self._keyboard.send_text(text)


class ActionExecutionError(Exception):
    """Raised when action execution fails."""
