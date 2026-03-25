"""Tests for cascade executor and control engine."""

import pytest

from edgerouter.core.schema import (
    AnalysisResult,
    ControlAction,
    Judgment,
    ProcessContext,
    RoutingDecision,
    RoutingTier,
    VisionOutput,
)
from edgerouter.control.engine import ControlEngine
from edgerouter.core.config import RouterConfig
from edgerouter.inference.base import AnalyzerBackend
from edgerouter.inference.mock import MockCloudAnalyzer, MockEdgeAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.prefetch import PredictivePrefetcher


# -----------------------------------------------------------------------
# Fixed-output mocks for precise test assertions
# -----------------------------------------------------------------------

class FixedEdgeAnalyzer(AnalyzerBackend):
    """Returns a fixed result for testing."""

    def __init__(self, judgment: Judgment = Judgment.NORMAL, confidence: float = 0.8):
        self._judgment = judgment
        self._confidence = confidence

    async def analyze(self, vision_output, recent_history=None, edge_draft=None):
        return AnalysisResult(
            judgment=self._judgment,
            confidence=self._confidence,
            suggested_action="maintain" if self._judgment == Judgment.NORMAL else "adjust_flow",
            reasoning="mock edge analysis",
            latency_ms=45.0,
            source="edge",
        )

    async def health_check(self):
        return True


class FixedCloudAnalyzer(AnalyzerBackend):
    """Returns a fixed result for testing."""

    def __init__(self, judgment: Judgment = Judgment.WARNING, confidence: float = 0.9):
        self._judgment = judgment
        self._confidence = confidence

    async def analyze(self, vision_output, recent_history=None, edge_draft=None):
        return AnalysisResult(
            judgment=self._judgment,
            confidence=self._confidence,
            suggested_action="reduce_input",
            reasoning="mock cloud analysis",
            root_cause="mock root cause",
            latency_ms=500.0,
            source="cloud",
        )

    async def health_check(self):
        return True


# -----------------------------------------------------------------------
# CascadeExecutor tests
# -----------------------------------------------------------------------

class TestCascadeExecutor:
    @pytest.fixture
    def executor(self):
        return CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(),
            cloud_analyzer=FixedCloudAnalyzer(),
        )

    @pytest.mark.asyncio
    async def test_emergency_tier(self, executor):
        vo = VisionOutput(anomaly_level=96.0)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.EDGE_EMERGENCY, reason="overflow")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.final_judgment == Judgment.ALARM
        assert outcome.edge_analysis is not None

    @pytest.mark.asyncio
    async def test_edge_only_tier(self, executor):
        vo = VisionOutput(anomaly_level=50.0)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.EDGE, reason="clearly_normal")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.final_judgment == Judgment.NORMAL
        assert outcome.cloud_analysis is None

    @pytest.mark.asyncio
    async def test_cloud_tier(self, executor):
        vo = VisionOutput(anomaly_level=50.0)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CLOUD, reason="complex_anomaly")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.cloud_analysis is not None
        assert outcome.final_judgment == Judgment.WARNING  # fixed cloud returns WARNING

    @pytest.mark.asyncio
    async def test_cascade_confident_edge(self):
        """Cascade with high-confidence edge → no cloud call."""
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.NORMAL, confidence=0.95),
            cloud_analyzer=FixedCloudAnalyzer(),
        )
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.9)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        outcome = await executor.execute(vo, ctx, decision)
        # High combined confidence → edge result accepted
        assert outcome.final_judgment == Judgment.NORMAL
        assert outcome.cloud_analysis is None

    @pytest.mark.asyncio
    async def test_cascade_low_confidence_escalates(self):
        """Cascade with low-confidence edge → escalate to cloud."""
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.WARNING, confidence=0.3),
            cloud_analyzer=FixedCloudAnalyzer(Judgment.ALARM, confidence=0.9),
        )
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.3)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.cloud_analysis is not None
        assert outcome.final_judgment == Judgment.ALARM

    @pytest.mark.asyncio
    async def test_outcome_has_latency(self, executor):
        vo = VisionOutput(anomaly_level=50.0)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.EDGE, reason="test")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_cascade_speculative_edge_confident(self):
        """Speculative prefetch: edge confident → cloud discarded."""
        config = RouterConfig(
            enable_speculative_prefetch=True,
            speculative_anomaly_threshold=0.1,  # low threshold to trigger
        )
        prefetcher = PredictivePrefetcher()
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.NORMAL, confidence=0.95),
            cloud_analyzer=FixedCloudAnalyzer(),
            config=config,
            prefetcher=prefetcher,
        )
        # anomaly_score=0.5 > threshold=0.1 → speculative path
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.5, anomaly_confidence=0.9)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.final_judgment == Judgment.NORMAL
        # Cloud should have been cancelled/discarded
        assert outcome.cloud_analysis is None

    @pytest.mark.asyncio
    async def test_cascade_speculative_edge_uncertain(self):
        """Speculative prefetch: edge uncertain → use cloud result."""
        config = RouterConfig(
            enable_speculative_prefetch=True,
            speculative_anomaly_threshold=0.1,
        )
        prefetcher = PredictivePrefetcher()
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.WARNING, confidence=0.3),
            cloud_analyzer=FixedCloudAnalyzer(Judgment.ALARM, confidence=0.9),
            config=config,
            prefetcher=prefetcher,
        )
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.5, anomaly_confidence=0.3)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.cloud_analysis is not None
        assert outcome.final_judgment == Judgment.ALARM

    @pytest.mark.asyncio
    async def test_cascade_no_prefetcher_unchanged(self):
        """Without prefetcher, cascade behaves identically to original."""
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.NORMAL, confidence=0.95),
            cloud_analyzer=FixedCloudAnalyzer(),
        )
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.9)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        outcome = await executor.execute(vo, ctx, decision)
        assert outcome.final_judgment == Judgment.NORMAL
        assert outcome.cloud_analysis is None

    @pytest.mark.asyncio
    async def test_prefetcher_stats_tracking(self):
        """Prefetcher tracks useful cascade events."""
        config = RouterConfig(
            enable_speculative_prefetch=True,
            speculative_anomaly_threshold=0.1,
        )
        prefetcher = PredictivePrefetcher()
        executor = CascadeExecutor(
            edge_analyzer=FixedEdgeAnalyzer(Judgment.WARNING, confidence=0.3),
            cloud_analyzer=FixedCloudAnalyzer(Judgment.ALARM, confidence=0.9),
            config=config,
            prefetcher=prefetcher,
        )
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.5, anomaly_confidence=0.3)
        ctx = ProcessContext()
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        await executor.execute(vo, ctx, decision)
        stats = prefetcher.get_stats()
        assert stats["prefetch_useful"] >= 1


# -----------------------------------------------------------------------
# ControlEngine tests
# -----------------------------------------------------------------------

class TestControlEngine:
    def setup_method(self):
        self.control = ControlEngine()

    def test_normal_maintains(self):
        analysis = AnalysisResult(judgment=Judgment.NORMAL)
        decision = RoutingDecision(tier=RoutingTier.EDGE, reason="test")
        action = self.control.execute(analysis, decision)
        assert action.type == "maintain"

    def test_warning_adjusts(self):
        analysis = AnalysisResult(
            judgment=Judgment.WARNING, suggested_action="adjust_flow",
        )
        decision = RoutingDecision(tier=RoutingTier.EDGE, reason="test")
        action = self.control.execute(analysis, decision)
        assert action.type == "adjust_flow"

    def test_alarm_emergency(self):
        analysis = AnalysisResult(judgment=Judgment.ALARM)
        decision = RoutingDecision(tier=RoutingTier.EDGE, reason="test")
        action = self.control.execute(analysis, decision)
        assert action.type == "emergency_stop"

    def test_emergency_tier_action(self):
        analysis = AnalysisResult(judgment=Judgment.ALARM)
        decision = RoutingDecision(tier=RoutingTier.EDGE_EMERGENCY, reason="overflow")
        action = self.control.execute(analysis, decision)
        assert action.type == "emergency_stop"

    def test_cascade_warning_pending_revision(self):
        analysis = AnalysisResult(
            judgment=Judgment.WARNING, suggested_action="adjust_flow",
        )
        decision = RoutingDecision(tier=RoutingTier.CASCADE, reason="uncertain")
        action = self.control.execute(analysis, decision)
        assert action.pending_cloud_revision is True

    def test_revise_action_cloud_confirms(self):
        action = ControlAction(
            type="adjust_flow",
            based_on_judgment=Judgment.WARNING,
            pending_cloud_revision=True,
        )
        cloud = AnalysisResult(judgment=Judgment.WARNING, confidence=0.9)
        revised = self.control.revise_action(action, cloud)
        assert revised.type == "adjust_flow"
        assert revised.pending_cloud_revision is False

    def test_revise_action_cloud_overrides(self):
        action = ControlAction(
            type="adjust_flow",
            based_on_judgment=Judgment.WARNING,
            pending_cloud_revision=True,
        )
        cloud = AnalysisResult(judgment=Judgment.ALARM, confidence=0.9)
        revised = self.control.revise_action(action, cloud)
        assert revised.type == "emergency_stop"

    def test_revise_action_cloud_downgrades(self):
        action = ControlAction(
            type="adjust_flow",
            based_on_judgment=Judgment.WARNING,
            pending_cloud_revision=True,
        )
        cloud = AnalysisResult(judgment=Judgment.NORMAL, confidence=0.9)
        revised = self.control.revise_action(action, cloud)
        assert revised.type == "maintain"

    def test_no_revision_if_not_pending(self):
        action = ControlAction(
            type="adjust_flow",
            based_on_judgment=Judgment.WARNING,
            pending_cloud_revision=False,
        )
        cloud = AnalysisResult(judgment=Judgment.ALARM, confidence=0.9)
        result = self.control.revise_action(action, cloud)
        # Should return unchanged
        assert result.type == "adjust_flow"
