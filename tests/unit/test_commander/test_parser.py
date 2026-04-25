"""Tests for CommandParser response parsing."""

from terminaleyes.commander.parser import CommandParser, CommandParseError

import pytest


class TestParseResponse:
    def setup_method(self):
        self.parser = CommandParser(model="test-model")

    def test_parse_valid_json(self):
        raw = """{
            "condition": {
                "description": "a lightblue button with Run written on it",
                "element_type": "button",
                "element_text": "Run",
                "visual_cues": ["lightblue", "button shape"],
                "spatial_context": "with a return key after it"
            },
            "action": {
                "action_type": "mouse_click",
                "button": "left",
                "key": null,
                "modifiers": [],
                "text": null,
                "target": "element"
            },
            "interval_seconds": 180,
            "one_shot": true,
            "max_attempts": 0
        }"""
        result = self.parser._parse_response(raw, "when you see a lightblue button with Run written on it, click it")
        assert result.condition.element_text == "Run"
        assert result.condition.description == "a lightblue button with Run written on it"
        assert "lightblue" in result.condition.visual_cues
        assert result.action.action_type == "mouse_click"
        assert result.action.button == "left"
        assert result.interval_seconds == 180.0
        assert result.one_shot is True

    def test_parse_json_in_markdown(self):
        raw = """Here is the parsed command:
```json
{
    "condition": {"description": "error dialog"},
    "action": {"action_type": "keystroke", "key": "Escape"},
    "interval_seconds": 60
}
```"""
        result = self.parser._parse_response(raw, "press escape on error")
        assert result.condition.description == "error dialog"
        assert result.action.action_type == "keystroke"
        assert result.action.key == "Escape"
        assert result.interval_seconds == 60.0

    def test_parse_minimal_json(self):
        raw = '{"condition": {"description": "test"}, "action": {"action_type": "mouse_click"}}'
        result = self.parser._parse_response(raw, "test")
        assert result.condition.description == "test"
        assert result.action.button == "left"  # default

    def test_parse_invalid_json_raises(self):
        raw = "This is not valid JSON at all"
        with pytest.raises(CommandParseError, match="Failed to parse"):
            self.parser._parse_response(raw, "test")

    def test_parse_with_interval_override(self):
        raw = '{"condition": {"description": "test"}, "action": {"action_type": "mouse_click"}, "interval_seconds": 30}'
        result = self.parser._parse_response(raw, "test")
        assert result.interval_seconds == 30.0

    def test_parse_continuous_mode(self):
        raw = '{"condition": {"description": "test"}, "action": {"action_type": "mouse_click"}, "one_shot": false}'
        result = self.parser._parse_response(raw, "test")
        assert result.one_shot is False
