"""Tests for cloud analyzer (unit tests, no vLLM/API required)."""

from edgerouter.inference.cloud_analyzer import _build_user_prompt, _parse_response
from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput


class TestBuildCloudPrompt:
    def test_basic_prompt(self):
        vo = VisionOutput(anomaly_level=40.0, secondary_metric=0.5, anomaly_score=0.6)
        prompt = _build_user_prompt(vo)
        assert "40.0" in prompt
        assert "Secondary metric" in prompt

    def test_prompt_with_edge_draft(self):
        vo = VisionOutput(anomaly_level=40.0)
        draft = AnalysisResult(
            judgment=Judgment.WARNING,
            confidence=0.6,
            suggested_action="adjust_flow",
            reasoning="secondary metric increasing",
        )
        prompt = _build_user_prompt(vo, edge_draft=draft)
        assert "Edge model preliminary judgment" in prompt
        assert "warning" in prompt

    def test_prompt_with_history(self):
        vo = VisionOutput(anomaly_level=40.0)
        history = [VisionOutput(anomaly_level=40.0 + i) for i in range(10)]
        prompt = _build_user_prompt(vo, recent_history=history)
        assert "Recent 10 frames" in prompt


class TestParseCloudResponse:
    def test_full_response(self):
        text = '{"judgment": "alarm", "confidence": 0.85, "action": "reduce_input", "reasoning": "multi-indicator", "root_cause": "reaction runaway"}'
        result = _parse_response(text)
        assert result["judgment"] == "alarm"
        assert result["root_cause"] == "reaction runaway"
