"""Tests for evaluation and online learning modules."""

import pytest

from edgerouter.core.config import RouterConfig
from edgerouter.core.schema import (
    AnalysisResult,
    Difficulty,
    Judgment,
    RoutingDecision,
    RoutingOutcome,
    RoutingTier,
    VisionOutput,
)
from edgerouter.eval.benchmarks import BenchmarkMetrics, BenchmarkRunner
from edgerouter.eval.workloads import build_small_workload, build_standard_workload
from edgerouter.learning.feedback import FeedbackCollector
from edgerouter.learning.online_learner import OnlineRouterLearner
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel
from edgerouter.inference.mock import MockCloudAnalyzer, MockEdgeAnalyzer


# -----------------------------------------------------------------------
# Workload tests
# -----------------------------------------------------------------------

class TestWorkloads:
    def test_standard_workload_size(self):
        wl = build_standard_workload()
        assert wl.size == 600

    def test_small_workload_size(self):
        wl = build_small_workload()
        assert wl.size == 100

    def test_standard_covers_all_difficulties(self):
        wl = build_standard_workload()
        diffs = {s.difficulty for s in wl.scenarios}
        assert Difficulty.NORMAL in diffs
        assert Difficulty.MARGINAL in diffs
        assert Difficulty.ANOMALOUS in diffs
        assert Difficulty.CRITICAL in diffs

    def test_standard_has_sensitive(self):
        wl = build_standard_workload()
        assert len(wl.sensitive_only()) == 50

    def test_by_difficulty(self):
        wl = build_standard_workload()
        normals = wl.by_difficulty(Difficulty.NORMAL)
        assert len(normals) >= 200  # includes sensitive_normal


# -----------------------------------------------------------------------
# Benchmark runner tests (with mock analyzers)
# -----------------------------------------------------------------------

class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_run_small_workload(self):
        config = RouterConfig()
        router = RouterEngine(config)
        vision = VisionModel(seed=42)
        cascade = CascadeExecutor(
            MockEdgeAnalyzer(), MockCloudAnalyzer(), config,
        )
        runner = BenchmarkRunner(router, cascade, vision)
        wl = build_small_workload()
        metrics = await runner.run(wl)

        assert metrics.total_scenarios == 100
        assert metrics.accuracy >= 0.0
        assert metrics.miss_rate >= 0.0
        assert 0.0 <= metrics.data_security_compliance <= 1.0

    @pytest.mark.asyncio
    async def test_summary_keys(self):
        config = RouterConfig()
        router = RouterEngine(config)
        vision = VisionModel(seed=42)
        cascade = CascadeExecutor(
            MockEdgeAnalyzer(), MockCloudAnalyzer(), config,
        )
        runner = BenchmarkRunner(router, cascade, vision)
        wl = build_small_workload()
        metrics = await runner.run(wl)
        summary = metrics.summary()

        expected_keys = {
            "total_scenarios", "accuracy", "miss_rate", "false_alarm_rate",
            "cloud_saving_rate", "upgrade_rate", "p50_latency_ms", "p99_latency_ms",
            "p50_emergency_latency_ms", "p99_emergency_latency_ms",
            "p50_routing_overhead_ms", "p99_routing_overhead_ms",
            "data_security_compliance", "edge_only", "cloud_direct", "cascade",
            "emergency", "sensitive_leaked",
        }
        assert expected_keys.issubset(set(summary.keys()))


# -----------------------------------------------------------------------
# Metrics tests
# -----------------------------------------------------------------------

class TestBenchmarkMetrics:
    def test_empty_metrics(self):
        m = BenchmarkMetrics()
        assert m.accuracy == 0.0
        assert m.p50_latency_ms == 0.0

    def test_cloud_saving_rate(self):
        m = BenchmarkMetrics(total_scenarios=100, cloud_count=10, cascade_count=20)
        assert m.cloud_saving_rate == 0.7


# -----------------------------------------------------------------------
# Online learner tests
# -----------------------------------------------------------------------

class TestOnlineRouterLearner:
    def test_initial_threshold(self):
        learner = OnlineRouterLearner(initial_threshold=0.6)
        assert learner.threshold == 0.6

    def test_confirmed_lowers_threshold(self):
        learner = OnlineRouterLearner(initial_threshold=0.7, learning_rate=0.05)
        outcome = RoutingOutcome(
            routing_decision=RoutingDecision(tier=RoutingTier.CASCADE, reason="test"),
            edge_analysis=AnalysisResult(judgment=Judgment.WARNING, source="edge"),
            cloud_analysis=AnalysisResult(judgment=Judgment.WARNING, source="cloud"),
        )
        new_t = learner.update(outcome)
        assert new_t < 0.7

    def test_overridden_raises_threshold(self):
        learner = OnlineRouterLearner(initial_threshold=0.7, learning_rate=0.05)
        outcome = RoutingOutcome(
            routing_decision=RoutingDecision(tier=RoutingTier.CASCADE, reason="test"),
            edge_analysis=AnalysisResult(judgment=Judgment.NORMAL, source="edge"),
            cloud_analysis=AnalysisResult(judgment=Judgment.ALARM, source="cloud"),
        )
        new_t = learner.update(outcome)
        assert new_t > 0.7

    def test_threshold_stays_in_bounds(self):
        learner = OnlineRouterLearner(
            initial_threshold=0.95, learning_rate=0.1, max_threshold=0.95,
        )
        outcome = RoutingOutcome(
            routing_decision=RoutingDecision(tier=RoutingTier.CASCADE, reason="test"),
            edge_analysis=AnalysisResult(judgment=Judgment.NORMAL, source="edge"),
            cloud_analysis=AnalysisResult(judgment=Judgment.ALARM, source="cloud"),
        )
        new_t = learner.update(outcome)
        assert new_t <= 0.95

    def test_non_cascade_ignored(self):
        learner = OnlineRouterLearner(initial_threshold=0.7)
        outcome = RoutingOutcome(
            routing_decision=RoutingDecision(tier=RoutingTier.EDGE, reason="test"),
        )
        new_t = learner.update(outcome)
        assert new_t == 0.7

    def test_stats(self):
        learner = OnlineRouterLearner()
        stats = learner.get_stats()
        assert "current_threshold" in stats
        assert "total_updates" in stats


# -----------------------------------------------------------------------
# Feedback collector tests
# -----------------------------------------------------------------------

class TestFeedbackCollector:
    def test_record_and_stats(self):
        fc = FeedbackCollector()
        outcome = RoutingOutcome(
            scenario_id="s1",
            edge_analysis=AnalysisResult(judgment=Judgment.WARNING, source="edge"),
            cloud_analysis=AnalysisResult(judgment=Judgment.WARNING, source="cloud"),
            edge_confidence=0.6,
        )
        fc.record(outcome, difficulty=Difficulty.MARGINAL)
        assert fc.total == 1
        assert fc.confirmation_rate == 1.0

    def test_override_tracked(self):
        fc = FeedbackCollector()
        outcome = RoutingOutcome(
            scenario_id="s1",
            edge_analysis=AnalysisResult(judgment=Judgment.NORMAL, source="edge"),
            cloud_analysis=AnalysisResult(judgment=Judgment.ALARM, source="cloud"),
            edge_confidence=0.4,
        )
        fc.record(outcome)
        assert fc.confirmation_rate == 0.0

    def test_calibration_bins(self):
        fc = FeedbackCollector()
        for i in range(20):
            conf = 0.3 + i * 0.03
            confirmed = i % 2 == 0
            j_edge = Judgment.WARNING
            j_cloud = Judgment.WARNING if confirmed else Judgment.ALARM
            outcome = RoutingOutcome(
                scenario_id=f"s{i}",
                edge_analysis=AnalysisResult(judgment=j_edge, source="edge"),
                cloud_analysis=AnalysisResult(judgment=j_cloud, source="cloud"),
                edge_confidence=conf,
            )
            fc.record(outcome)
        cal = fc.confidence_calibration(bins=5)
        assert len(cal) > 0
        assert all("confirmation_rate" in row for row in cal)
