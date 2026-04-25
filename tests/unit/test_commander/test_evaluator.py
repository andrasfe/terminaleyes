"""Tests for ConditionEvaluator."""

from terminaleyes.commander.evaluator import ConditionEvaluator


class TestParseResponse:
    def setup_method(self):
        self.evaluator = ConditionEvaluator(model="test-model")

    def test_parse_valid_json(self):
        raw = '{"condition_met": true, "confidence": 0.95, "location_x_pct": 0.45, "location_y_pct": 0.62, "reasoning": "Found a lightblue button with Run written on it"}'
        result = self.evaluator._parse_response(raw)
        assert result.condition_met is True
        assert result.confidence == 0.95
        assert result.location is not None
        assert result.location.x_pct == 0.45
        assert result.location.y_pct == 0.62
        assert "lightblue" in result.reasoning

    def test_parse_json_in_markdown(self):
        raw = """Here is my analysis:
```json
{"condition_met": false, "confidence": 0.2, "location_x_pct": null, "location_y_pct": null, "reasoning": "No button visible"}
```"""
        result = self.evaluator._parse_response(raw)
        assert result.condition_met is False
        assert result.location is None

    def test_parse_not_met(self):
        raw = '{"condition_met": false, "confidence": 0.1, "reasoning": "Screen is blank"}'
        result = self.evaluator._parse_response(raw)
        assert result.condition_met is False
        assert result.confidence == 0.1

    def test_parse_invalid_json_fallback(self):
        raw = "I can see a button that says Run"
        result = self.evaluator._parse_response(raw)
        # Should fallback to text analysis
        assert result.confidence <= 0.3

    def test_parse_clamps_confidence(self):
        raw = '{"condition_met": true, "confidence": 1.5}'
        result = self.evaluator._parse_response(raw)
        assert result.confidence == 1.0

    def test_parse_location_clamped(self):
        raw = '{"condition_met": true, "confidence": 0.8, "location_x_pct": 1.5, "location_y_pct": -0.1}'
        result = self.evaluator._parse_response(raw)
        assert result.location.x_pct == 1.0
        assert result.location.y_pct == 0.0


class TestParseCursorResponse:
    def setup_method(self):
        self.evaluator = ConditionEvaluator(model="test-model")

    def test_cursor_on_target(self):
        raw = '{"cursor_found": true, "cursor_x_pct": 0.5, "cursor_y_pct": 0.3, "target_found": true, "target_x_pct": 0.5, "target_y_pct": 0.3, "cursor_on_target": true, "reasoning": "Cursor is on the Run button"}'
        result = self.evaluator._parse_cursor_response(raw)
        assert result.cursor_found is True
        assert result.cursor_on_target is True
        assert result.cursor_location is not None
        assert result.cursor_location.x_pct == 0.5
        assert result.target_location is not None
        assert result.target_location.x_pct == 0.5

    def test_cursor_off_target(self):
        raw = '{"cursor_found": true, "cursor_x_pct": 0.2, "cursor_y_pct": 0.1, "target_found": true, "target_x_pct": 0.7, "target_y_pct": 0.6, "cursor_on_target": false, "reasoning": "Cursor is in top-left, target is center-right"}'
        result = self.evaluator._parse_cursor_response(raw)
        assert result.cursor_found is True
        assert result.cursor_on_target is False
        assert result.cursor_location.x_pct == 0.2
        assert result.target_location.x_pct == 0.7

    def test_cursor_not_found(self):
        raw = '{"cursor_found": false, "cursor_x_pct": null, "cursor_y_pct": null, "target_found": true, "target_x_pct": 0.5, "target_y_pct": 0.5, "cursor_on_target": false, "reasoning": "Cannot see cursor"}'
        result = self.evaluator._parse_cursor_response(raw)
        assert result.cursor_found is False
        assert result.cursor_location is None
        assert result.target_found is True
        assert result.target_location is not None

    def test_target_not_found(self):
        raw = '{"cursor_found": true, "cursor_x_pct": 0.3, "cursor_y_pct": 0.3, "target_found": false, "target_x_pct": null, "target_y_pct": null, "cursor_on_target": false, "reasoning": "Target element not visible"}'
        result = self.evaluator._parse_cursor_response(raw)
        assert result.target_found is False
        assert result.target_location is None

    def test_invalid_json_fallback(self):
        raw = "I cannot parse this at all"
        result = self.evaluator._parse_cursor_response(raw)
        assert result.cursor_found is False
        assert result.cursor_on_target is False

    def test_json_in_markdown(self):
        raw = """```json
{"cursor_found": true, "cursor_x_pct": 0.5, "cursor_y_pct": 0.5, "target_found": true, "target_x_pct": 0.5, "target_y_pct": 0.5, "cursor_on_target": true, "reasoning": "Both at center"}
```"""
        result = self.evaluator._parse_cursor_response(raw)
        assert result.cursor_on_target is True
