"""Tests for edge analyzer (unit tests, no external server required)."""

import json

import pytest

from edgerouter.inference.edge_analyzer import _build_user_prompt, _parse_response
from edgerouter.core.schema import VisionOutput


class TestBuildUserPrompt:
    def test_basic_prompt(self):
        vo = VisionOutput(
            anomaly_level=50.0,
            measurement_confidence=0.9,
            secondary_metric=0.1,
            anomaly_score=0.2,
            anomaly_confidence=0.85,
        )
        prompt = _build_user_prompt(vo)
        assert "50.0" in prompt
        assert "Secondary metric" in prompt

    def test_prompt_with_history(self):
        vo = VisionOutput(anomaly_level=50.0)
        history = [
            VisionOutput(anomaly_level=48.0 + i, secondary_metric=0.1)
            for i in range(5)
        ]
        prompt = _build_user_prompt(vo, recent_history=history)
        assert "Recent 5 frames" in prompt


class TestParseResponse:
    def test_plain_json(self):
        text = '{"judgment": "normal", "confidence": 0.9, "action": "maintain", "reasoning": "ok"}'
        result = _parse_response(text)
        assert result["judgment"] == "normal"
        assert result["confidence"] == 0.9

    def test_markdown_fenced_json(self):
        text = '```json\n{"judgment": "warning", "confidence": 0.6, "action": "adjust_flow", "reasoning": "secondary metric rising"}\n```'
        result = _parse_response(text)
        assert result["judgment"] == "warning"

    def test_json_with_surrounding_text(self):
        text = 'Here is the analysis:\n{"judgment": "alarm", "confidence": 0.3, "action": "emergency_stop", "reasoning": "overflow"}\nDone.'
        result = _parse_response(text)
        assert result["judgment"] == "alarm"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _parse_response("no json here")
