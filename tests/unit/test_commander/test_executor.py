"""Tests for ActionExecutor."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import numpy as np
import pytest

from terminaleyes.commander.calibration import CalibrationResult
from terminaleyes.commander.executor import ActionExecutor, ActionExecutionError
from terminaleyes.commander.models import (
    ActionSpec,
    CursorLocateResult,
    ScreenLocation,
)


@pytest.fixture
def mock_keyboard():
    kb = AsyncMock()
    return kb


@pytest.fixture
def mock_mouse():
    mouse = AsyncMock()
    return mouse


@pytest.fixture
def executor(mock_keyboard, mock_mouse):
    """Executor without capture/evaluator — uses blind click fallback."""
    return ActionExecutor(
        keyboard=mock_keyboard,
        mouse=mock_mouse,
        screen_width=1920,
        screen_height=1080,
    )


@pytest.fixture
def mock_capture():
    capture = AsyncMock()
    frame = MagicMock()
    frame.image = np.zeros((100, 100, 3), dtype=np.uint8)
    capture.capture_frame = AsyncMock(return_value=frame)
    return capture


@pytest.fixture
def mock_evaluator():
    return AsyncMock()


@pytest.fixture
def fake_calibration():
    """Pre-set calibration so tests don't run the real calibration routine."""
    return CalibrationResult(
        hid_units_per_full_x=1920.0,
        hid_units_per_full_y=1080.0,
    )


@pytest.fixture
def homing_executor(mock_keyboard, mock_mouse, mock_capture, mock_evaluator, fake_calibration):
    """Executor with capture + evaluator — uses visual homing."""
    ex = ActionExecutor(
        keyboard=mock_keyboard,
        mouse=mock_mouse,
        screen_width=1920,
        screen_height=1080,
        capture=mock_capture,
        evaluator=mock_evaluator,
        max_homing_steps=3,
    )
    # Inject pre-set calibration to skip the calibration routine
    ex._calibration = fake_calibration
    return ex


class TestMouseClickBlind:
    @pytest.mark.asyncio
    async def test_click_at_element_blind_fallback(self, executor, mock_mouse):
        """Without capture/evaluator, falls back to blind click_at."""
        action = ActionSpec(action_type="mouse_click", button="left", target="element")
        location = ScreenLocation(x_pct=0.5, y_pct=0.3)

        await executor.execute(action, location)
        mock_mouse.click_at.assert_called_once_with(
            x_pct=0.5, y_pct=0.3, button="left",
            screen_width=1920, screen_height=1080,
        )

    @pytest.mark.asyncio
    async def test_click_current_position(self, executor, mock_mouse):
        action = ActionSpec(action_type="mouse_click", button="right", target="current")

        await executor.execute(action, None)
        mock_mouse.click.assert_called_once_with("right")

    @pytest.mark.asyncio
    async def test_click_element_no_location(self, executor, mock_mouse):
        action = ActionSpec(action_type="mouse_click", button="left", target="element")

        await executor.execute(action, None)
        mock_mouse.click.assert_called_once_with("left")


class TestVisualHoming:
    @pytest.mark.asyncio
    async def test_cursor_on_target_immediately(
        self, homing_executor, mock_mouse, mock_evaluator
    ):
        """If cursor is already on target, click immediately."""
        mock_evaluator.locate_cursor = AsyncMock(
            return_value=CursorLocateResult(
                cursor_found=True,
                cursor_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                target_found=True,
                target_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                cursor_on_target=True,
                reasoning="Cursor is on the Run button",
            )
        )

        action = ActionSpec(action_type="mouse_click", button="left", target="element")
        location = ScreenLocation(x_pct=0.5, y_pct=0.3)

        await homing_executor.execute(action, location, target_description="Run button")

        assert mock_evaluator.locate_cursor.call_count == 1
        mock_mouse.click.assert_called_once_with("left")

    @pytest.mark.asyncio
    async def test_cursor_corrects_toward_target(
        self, homing_executor, mock_mouse, mock_evaluator
    ):
        """Should correct cursor position step by step."""
        mock_evaluator.locate_cursor = AsyncMock(
            side_effect=[
                CursorLocateResult(  # step 1: off target
                    cursor_found=True,
                    cursor_location=ScreenLocation(x_pct=0.4, y_pct=0.2),
                    target_found=True,
                    target_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                    cursor_on_target=False,
                    reasoning="Cursor is above and left of button",
                ),
                CursorLocateResult(  # step 2: on target
                    cursor_found=True,
                    cursor_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                    target_found=True,
                    target_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                    cursor_on_target=True,
                    reasoning="Cursor is now on the button",
                ),
            ]
        )

        action = ActionSpec(action_type="mouse_click", button="left", target="element")
        location = ScreenLocation(x_pct=0.5, y_pct=0.3)

        await homing_executor.execute(action, location, target_description="Run button")

        assert mock_evaluator.locate_cursor.call_count == 2
        mock_mouse.click.assert_called_once_with("left")

    @pytest.mark.asyncio
    async def test_target_not_found_does_not_click(
        self, homing_executor, mock_mouse, mock_evaluator
    ):
        """If target not found, do not click."""
        mock_evaluator.locate_cursor = AsyncMock(
            return_value=CursorLocateResult(
                cursor_found=True,
                cursor_location=ScreenLocation(x_pct=0.5, y_pct=0.3),
                target_found=False,
                target_location=None,
                cursor_on_target=False,
                reasoning="Cannot see the target element",
            )
        )

        action = ActionSpec(action_type="mouse_click", button="left", target="element")
        location = ScreenLocation(x_pct=0.5, y_pct=0.3)

        await homing_executor.execute(action, location)
        mock_mouse.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_homing_steps_exhausted(
        self, homing_executor, mock_mouse, mock_evaluator
    ):
        """After max steps without reaching target, does not click."""
        mock_evaluator.locate_cursor = AsyncMock(
            return_value=CursorLocateResult(
                cursor_found=True,
                cursor_location=ScreenLocation(x_pct=0.3, y_pct=0.2),
                target_found=True,
                target_location=ScreenLocation(x_pct=0.8, y_pct=0.8),
                cursor_on_target=False,
                reasoning="Cursor is far from target",
            )
        )

        action = ActionSpec(action_type="mouse_click", button="left", target="element")
        location = ScreenLocation(x_pct=0.8, y_pct=0.8)

        await homing_executor.execute(action, location)

        assert mock_evaluator.locate_cursor.call_count == 3  # max_homing_steps=3
        mock_mouse.click.assert_not_called()


class TestKeystroke:
    @pytest.mark.asyncio
    async def test_send_keystroke(self, executor, mock_keyboard):
        action = ActionSpec(action_type="keystroke", key="Enter")
        await executor.execute(action, None)
        mock_keyboard.send_keystroke.assert_called_once_with("Enter")


class TestKeyCombo:
    @pytest.mark.asyncio
    async def test_send_key_combo(self, executor, mock_keyboard):
        action = ActionSpec(
            action_type="key_combo", modifiers=["ctrl"], key="c"
        )
        await executor.execute(action, None)
        mock_keyboard.send_key_combo.assert_called_once_with(["ctrl"], "c")


class TestTextInput:
    @pytest.mark.asyncio
    async def test_send_text(self, executor, mock_keyboard):
        action = ActionSpec(action_type="text_input", text="hello world")
        await executor.execute(action, None)
        mock_keyboard.send_text.assert_called_once_with("hello world")


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_raises(self, executor):
        action = ActionSpec(action_type="unknown_action")
        with pytest.raises(ActionExecutionError, match="Unknown action type"):
            await executor.execute(action, None)
