"""Multi-dimensional evaluation benchmarks for EdgeRouter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from edgerouter.core.schema import (
    Difficulty,
    Judgment,
    ProcessContext,
    RoutingOutcome,
    RoutingTier,
    ScenarioProfile,
    VisionOutput,
)
from edgerouter.eval.workloads import EvalWorkload
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    """Aggregated evaluation metrics."""

    total_scenarios: int = 0
    correct_judgments: int = 0
    false_negatives: int = 0       # real anomaly judged as normal (miss)
    false_positives: int = 0       # normal judged as anomaly (false alarm)

    edge_only_count: int = 0
    cloud_count: int = 0
    cascade_count: int = 0
    emergency_count: int = 0

    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    emergency_latencies: list[float] = field(default_factory=list)

    sensitive_leaked: int = 0      # sensitive data sent to cloud (must be 0)
    sensitive_total: int = 0

    routing_overhead_ms: list[float] = field(default_factory=list)  # routing decision time

    outcomes: list[RoutingOutcome] = field(default_factory=list)

    # -- Derived metrics --

    @property
    def accuracy(self) -> float:
        return self.correct_judgments / max(1, self.total_scenarios)

    @property
    def miss_rate(self) -> float:
        """False negative rate: real anomalies missed. Target ≤ 2%."""
        anomaly_total = sum(
            1 for o in self.outcomes
            if o.ground_truth_judgment in (Judgment.WARNING, Judgment.ALARM)
        )
        if anomaly_total == 0:
            return 0.0
        return self.false_negatives / anomaly_total

    @property
    def false_alarm_rate(self) -> float:
        normal_total = sum(
            1 for o in self.outcomes
            if o.ground_truth_judgment == Judgment.NORMAL
        )
        if normal_total == 0:
            return 0.0
        return self.false_positives / normal_total

    @property
    def cloud_saving_rate(self) -> float:
        """1 - (cloud calls / total). Higher = more savings."""
        cloud_calls = self.cloud_count + self.cascade_count  # cascade may call cloud
        return 1.0 - (cloud_calls / max(1, self.total_scenarios))

    @property
    def upgrade_rate(self) -> float:
        return (self.cloud_count + self.cascade_count) / max(1, self.total_scenarios)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.99)]

    @property
    def p50_emergency_latency_ms(self) -> float:
        if not self.emergency_latencies:
            return 0.0
        s = sorted(self.emergency_latencies)
        return s[len(s) // 2]

    @property
    def p99_emergency_latency_ms(self) -> float:
        if not self.emergency_latencies:
            return 0.0
        s = sorted(self.emergency_latencies)
        return s[int(len(s) * 0.99)]

    @property
    def p50_routing_overhead_ms(self) -> float:
        if not self.routing_overhead_ms:
            return 0.0
        s = sorted(self.routing_overhead_ms)
        return s[len(s) // 2]

    @property
    def p99_routing_overhead_ms(self) -> float:
        if not self.routing_overhead_ms:
            return 0.0
        s = sorted(self.routing_overhead_ms)
        return s[int(len(s) * 0.99)]

    @property
    def data_security_compliance(self) -> float:
        if self.sensitive_total == 0:
            return 1.0
        return 1.0 - (self.sensitive_leaked / self.sensitive_total)

    def summary(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "accuracy": round(self.accuracy, 4),
            "miss_rate": round(self.miss_rate, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "cloud_saving_rate": round(self.cloud_saving_rate, 4),
            "upgrade_rate": round(self.upgrade_rate, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "p50_emergency_latency_ms": round(self.p50_emergency_latency_ms, 2),
            "p99_emergency_latency_ms": round(self.p99_emergency_latency_ms, 2),
            "p50_routing_overhead_ms": round(self.p50_routing_overhead_ms, 4),
            "p99_routing_overhead_ms": round(self.p99_routing_overhead_ms, 4),
            "data_security_compliance": round(self.data_security_compliance, 4),
            "edge_only": self.edge_only_count,
            "cloud_direct": self.cloud_count,
            "cascade": self.cascade_count,
            "emergency": self.emergency_count,
            "sensitive_leaked": self.sensitive_leaked,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run a full evaluation over a workload."""

    def __init__(
        self,
        router: RouterEngine,
        cascade_executor: CascadeExecutor,
        vision_model: VisionModel,
    ):
        self.router = router
        self.cascade = cascade_executor
        self.vision = vision_model

    async def run(
        self,
        workload: EvalWorkload,
        ground_truth_map: dict[str, Judgment] | None = None,
    ) -> BenchmarkMetrics:
        """Evaluate on every scenario in the workload."""
        metrics = BenchmarkMetrics()

        for i, scenario in enumerate(workload.scenarios):
            vision_output = self.vision.detect(scenario)
            context = self._scenario_to_context(scenario, idx=i)

            # Route
            decision = self.router.route(vision_output, context)

            # Execute
            outcome = await self.cascade.execute(
                vision_output, context, decision,
            )
            outcome.ground_truth_judgment = scenario.ground_truth_judgment

            # Record
            metrics.total_scenarios += 1
            metrics.outcomes.append(outcome)
            metrics.latencies.append(outcome.total_latency_ms)
            metrics.routing_overhead_ms.append(decision.latency_ms)

            # Tier counts
            if decision.tier == RoutingTier.EDGE_EMERGENCY:
                metrics.emergency_count += 1
                metrics.emergency_latencies.append(outcome.total_latency_ms)
            elif decision.tier == RoutingTier.EDGE:
                metrics.edge_only_count += 1
            elif decision.tier == RoutingTier.CLOUD:
                metrics.cloud_count += 1
            elif decision.tier == RoutingTier.CASCADE:
                metrics.cascade_count += 1

            # Judgment correctness
            gt = scenario.ground_truth_judgment
            predicted = outcome.final_judgment

            if predicted == gt:
                metrics.correct_judgments += 1
            elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
                metrics.false_negatives += 1
            elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
                metrics.false_positives += 1

            # Data security check
            if scenario.contains_process_params:
                metrics.sensitive_total += 1
                if decision.tier in (RoutingTier.CLOUD, RoutingTier.CASCADE):
                    # This would be a leak if cloud was actually called
                    if outcome.cloud_analysis is not None:
                        metrics.sensitive_leaked += 1

        return metrics

    @staticmethod
    def _scenario_to_context(scenario: ScenarioProfile, idx: int = 0) -> ProcessContext:
        return ProcessContext(
            scenario_id=f"eval_{idx}",
            has_recipe_params=scenario.has_recipe_params,
            has_customer_info=scenario.has_customer_info,
            has_reaction_params=scenario.has_reaction_params,
            num_correlated_anomalies=scenario.num_correlated_anomalies,
        )
