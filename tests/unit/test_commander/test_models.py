"""Tests for commander domain models."""

from datetime import datetime

from terminaleyes.commander.models import (
    ActionSpec,
    CommandSession,
    CommandSpec,
    ConditionResult,
    ConditionSpec,
    CursorLocateResult,
    ScreenLocation,
)


class TestConditionSpec:
    def test_minimal(self):
        spec = ConditionSpec(description="blue button")
        assert spec.description == "blue button"
        assert spec.element_type is None
        assert spec.visual_cues == []

    def test_full(self):
        spec = ConditionSpec(
            description="a lightblue button with Run written on it",
            element_type="button",
            element_text="Run",
            visual_cues=["lightblue", "button shape"],
            spatial_context="with a return key after it",
        )
        assert spec.element_text == "Run"
        assert "lightblue" in spec.visual_cues
        assert spec.spatial_context == "with a return key after it"


class TestActionSpec:
    def test_mouse_click(self):
        action = ActionSpec(action_type="mouse_click", button="left")
        assert action.target == "element"

    def test_keystroke(self):
        action = ActionSpec(action_type="keystroke", key="Enter")
        assert action.modifiers == []


class TestCommandSpec:
    def test_defaults(self):
        spec = CommandSpec(
            raw_instruction="click run",
            condition=ConditionSpec(description="run button"),
            action=ActionSpec(action_type="mouse_click", button="left"),
        )
        assert spec.interval_seconds == 180.0
        assert spec.one_shot is True
        assert spec.max_attempts == 0


class TestScreenLocation:
    def test_valid(self):
        loc = ScreenLocation(x_pct=0.5, y_pct=0.3, confidence=0.9)
        assert loc.x_pct == 0.5

    def test_bounds(self):
        import pytest
        with pytest.raises(Exception):
            ScreenLocation(x_pct=1.5, y_pct=0.5)


class TestConditionResult:
    def test_not_met(self):
        result = ConditionResult(condition_met=False, confidence=0.1)
        assert result.location is None

    def test_met_with_location(self):
        result = ConditionResult(
            condition_met=True,
            confidence=0.95,
            location=ScreenLocation(x_pct=0.45, y_pct=0.62),
            reasoning="Found blue Run button",
        )
        assert result.location.x_pct == 0.45


class TestCursorLocateResult:
    def test_defaults(self):
        result = CursorLocateResult()
        assert result.cursor_found is False
        assert result.cursor_on_target is False
        assert result.cursor_location is None
        assert result.target_location is None

    def test_with_locations(self):
        result = CursorLocateResult(
            cursor_found=True,
            cursor_location=ScreenLocation(x_pct=0.2, y_pct=0.3),
            target_found=True,
            target_location=ScreenLocation(x_pct=0.5, y_pct=0.6),
            cursor_on_target=False,
            reasoning="Cursor is above target",
        )
        assert result.cursor_location.x_pct == 0.2
        assert result.target_location.x_pct == 0.5


class TestCommandSession:
    def test_defaults(self):
        session = CommandSession(
            session_id="abc123",
            command=CommandSpec(
                raw_instruction="test",
                condition=ConditionSpec(description="test"),
                action=ActionSpec(action_type="mouse_click"),
            ),
            started_at=datetime.now(),
        )
        assert session.status == "running"
        assert session.frames_checked == 0
