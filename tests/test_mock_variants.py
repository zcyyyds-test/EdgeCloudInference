"""Tests for mock analyzer variants: sized edge analyzers and WAN delay wrapper."""

import pytest

from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput
from edgerouter.inference.mock import (
    MockCloudAnalyzer,
    MockEdgeAnalyzer,
    SizedMockEdgeAnalyzer,
    WANDelayCloudAnalyzer,
)


# -----------------------------------------------------------------------
# SizedMockEdgeAnalyzer
# -----------------------------------------------------------------------

class TestSizedMockEdgeAnalyzer:
    @pytest.mark.asyncio
    async def test_all_sizes_valid(self):
        for size in ["0.6B", "1.7B", "4B", "8B"]:
            analyzer = SizedMockEdgeAnalyzer(model_size=size)
            vo = VisionOutput(anomaly_score=0.5, anomaly_level=30.0, secondary_metric=0.4)
            result = await analyzer.analyze(vo)
            assert isinstance(result, AnalysisResult)
            assert result.source == "edge"

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            SizedMockEdgeAnalyzer(model_size="100B")

    @pytest.mark.asyncio
    async def test_larger_model_lower_latency_than_8b(self):
        """Smaller models should have lower latency."""
        vo = VisionOutput(anomaly_score=0.3, anomaly_level=50.0)
        small = await SizedMockEdgeAnalyzer("0.6B").analyze(vo)
        large = await SizedMockEdgeAnalyzer("8B").analyze(vo)
        assert small.latency_ms < large.latency_ms

    @pytest.mark.asyncio
    async def test_4b_matches_default(self):
        """4B sized analyzer should produce same judgment as default MockEdgeAnalyzer
        for clear cases (thresholds are the same)."""
        # Test with a clearly normal case
        vo = VisionOutput(anomaly_score=0.05, anomaly_level=50.0, secondary_metric=0.05)
        sized = await SizedMockEdgeAnalyzer("4B").analyze(vo)
        default = await MockEdgeAnalyzer().analyze(vo)
        assert sized.judgment == default.judgment

    @pytest.mark.asyncio
    async def test_smaller_model_misses_more(self):
        """Smaller models with higher thresholds should miss more marginal cases."""
        # Marginal case: moderate composite score
        vo = VisionOutput(
            anomaly_score=0.3, anomaly_level=60.0, secondary_metric=0.3,
            measurement_confidence=0.7,
        )
        small = await SizedMockEdgeAnalyzer("0.6B").analyze(vo)
        large = await SizedMockEdgeAnalyzer("8B").analyze(vo)
        # The larger model should be at least as sensitive (same or higher severity)
        severity = {Judgment.NORMAL: 0, Judgment.WARNING: 1, Judgment.ALARM: 2}
        assert severity[large.judgment] >= severity[small.judgment]

    @pytest.mark.asyncio
    async def test_health_check(self):
        analyzer = SizedMockEdgeAnalyzer("4B")
        assert await analyzer.health_check()


# -----------------------------------------------------------------------
# WANDelayCloudAnalyzer
# -----------------------------------------------------------------------

class TestWANDelayCloudAnalyzer:
    @pytest.mark.asyncio
    async def test_adds_delay_to_latency(self):
        inner = MockCloudAnalyzer()
        vo = VisionOutput(anomaly_score=0.5)
        base_result = await inner.analyze(vo)
        base_latency = base_result.latency_ms

        wrapped = WANDelayCloudAnalyzer(inner, wan_delay_ms=200.0)
        result = await wrapped.analyze(vo)
        assert result.latency_ms >= base_latency + 200.0

    @pytest.mark.asyncio
    async def test_judgment_unchanged(self):
        """WAN delay should not affect judgment quality."""
        inner = MockCloudAnalyzer()
        vo = VisionOutput(anomaly_score=0.6, anomaly_level=80.0, secondary_metric=0.7)

        base = await inner.analyze(vo)
        wrapped = WANDelayCloudAnalyzer(inner, wan_delay_ms=500.0)
        delayed = await wrapped.analyze(vo)

        assert base.judgment == delayed.judgment
        assert base.confidence == delayed.confidence

    @pytest.mark.asyncio
    async def test_health_check_delegates(self):
        wrapped = WANDelayCloudAnalyzer(MockCloudAnalyzer(), wan_delay_ms=50.0)
        assert await wrapped.health_check()

    @pytest.mark.asyncio
    async def test_various_delays(self):
        vo = VisionOutput()
        for delay in [10, 50, 200, 500]:
            wrapped = WANDelayCloudAnalyzer(MockCloudAnalyzer(), wan_delay_ms=delay)
            result = await wrapped.analyze(vo)
            assert result.latency_ms >= delay


# -----------------------------------------------------------------------
# Extended workload tests
# -----------------------------------------------------------------------

class TestExtendedWorkloads:
    def test_extended_workload_size(self):
        from edgerouter.eval.workloads import build_extended_workload
        wl = build_extended_workload()
        assert wl.size == 1000

    def test_security_workload_size(self):
        from edgerouter.eval.workloads import build_security_workload
        wl = build_security_workload()
        assert wl.size == 600

    def test_security_workload_has_50_sensitive(self):
        from edgerouter.eval.workloads import build_security_workload
        wl = build_security_workload()
        assert len(wl.sensitive_only()) == 50

    def test_extended_covers_all_difficulties(self):
        from edgerouter.core.schema import Difficulty
        from edgerouter.eval.workloads import build_extended_workload
        wl = build_extended_workload()
        diffs = {s.difficulty for s in wl.scenarios}
        assert Difficulty.NORMAL in diffs
        assert Difficulty.MARGINAL in diffs
        assert Difficulty.ANOMALOUS in diffs
        assert Difficulty.CRITICAL in diffs
