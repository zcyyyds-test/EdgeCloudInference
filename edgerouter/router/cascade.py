"""Cascade executor: edge-first analysis with optional cloud escalation."""

from __future__ import annotations

import logging
import time

from edgerouter.core.config import RouterConfig
from edgerouter.core.metrics import (
    CASCADE_EDGE_CONFIRMED,
    CASCADE_EDGE_OVERRIDDEN,
    CASCADE_TOTAL,
)
from edgerouter.core.schema import (
    AnalysisResult,
    Judgment,
    ProcessContext,
    RoutingDecision,
    RoutingOutcome,
    RoutingTier,
    VisionOutput,
)
from edgerouter.inference.base import AnalyzerBackend
from edgerouter.router.confidence import (
    estimate_combined,
    estimate_from_output,
    estimate_from_self_verification,
    estimate_from_temporal,
)
from edgerouter.router.data_security import DataSecurityChecker

logger = logging.getLogger(__name__)


class CascadeExecutor:
    """Execute the cascade routing pipeline.

    Flow:
    1. Edge analyzer produces a draft judgment + confidence
    2. If combined confidence >= threshold → accept edge result
    3. If combined confidence < threshold → escalate to cloud
    4. Record outcome for online learning
    """

    def __init__(
        self,
        edge_analyzer: AnalyzerBackend,
        cloud_analyzer: AnalyzerBackend,
        config: RouterConfig | None = None,
    ):
        self.edge = edge_analyzer
        self.cloud = cloud_analyzer
        self.config = config or RouterConfig()
        self.data_security = DataSecurityChecker()

    async def execute(
        self,
        vision_output: VisionOutput,
        context: ProcessContext,
        routing_decision: RoutingDecision,
        recent_history: list[VisionOutput] | None = None,
    ) -> RoutingOutcome:
        """Run the full analysis pipeline according to the routing decision."""
        t_start = time.perf_counter()

        outcome = RoutingOutcome(
            scenario_id=context.scenario_id,
            vision_output=vision_output,
            routing_decision=routing_decision,
        )

        if routing_decision.tier == RoutingTier.EDGE_EMERGENCY:
            # Emergency: skip analysis, use vision output directly
            outcome.edge_analysis = AnalysisResult(
                judgment=Judgment.ALARM,
                confidence=0.95,
                suggested_action="emergency_stop",
                reasoning="Critical safety event — immediate action",
                source="edge",
            )
            outcome.final_judgment = Judgment.ALARM
            outcome.edge_confidence = 0.95

        elif routing_decision.tier == RoutingTier.EDGE:
            # Edge-only: run edge analyzer, but escalate if post-execution
            # confidence drops below threshold (safety net for misrouting)
            edge_result = await self.edge.analyze(vision_output, recent_history)
            outcome.edge_analysis = edge_result
            outcome.edge_confidence = edge_result.confidence

            if edge_result.confidence < self.config.confidence_threshold:
                # Edge isn't confident — escalate to cloud despite initial routing
                logger.info(
                    "EDGE→CLOUD escalation: edge conf=%.2f < threshold=%.2f",
                    edge_result.confidence, self.config.confidence_threshold,
                )
                cloud_result = await self.cloud.analyze(
                    vision_output, recent_history, edge_draft=edge_result,
                )
                outcome.cloud_analysis = cloud_result
                outcome.final_judgment = cloud_result.judgment
                self._record_cascade(edge_result, cloud_result)
            else:
                outcome.final_judgment = edge_result.judgment

        elif routing_decision.tier == RoutingTier.CLOUD:
            # Direct cloud: run both for comparison but use cloud result
            edge_result = await self.edge.analyze(vision_output, recent_history)
            outcome.edge_analysis = edge_result

            cloud_result = await self.cloud.analyze(
                vision_output, recent_history, edge_draft=edge_result,
            )
            outcome.cloud_analysis = cloud_result
            outcome.final_judgment = cloud_result.judgment
            outcome.edge_confidence = edge_result.confidence
            self._record_cascade(edge_result, cloud_result)

        elif routing_decision.tier == RoutingTier.CASCADE:
            # Cascade: edge first, then decide whether to escalate
            edge_result = await self.edge.analyze(vision_output, recent_history)
            outcome.edge_analysis = edge_result

            combined_conf = self._estimate_confidence(
                vision_output, edge_result, recent_history,
            )
            outcome.edge_confidence = combined_conf

            if combined_conf >= self.config.confidence_threshold:
                # Edge is confident enough
                outcome.final_judgment = edge_result.judgment
            else:
                # Escalate to cloud
                cloud_result = await self.cloud.analyze(
                    vision_output, recent_history, edge_draft=edge_result,
                )
                outcome.cloud_analysis = cloud_result
                outcome.final_judgment = cloud_result.judgment
                self._record_cascade(edge_result, cloud_result)

        # Total latency: sum of analyzer latencies
        # Real wall-clock time may differ from reported latency
        total_ms = 0.0
        if outcome.edge_analysis:
            total_ms += outcome.edge_analysis.latency_ms
        if outcome.cloud_analysis:
            total_ms += outcome.cloud_analysis.latency_ms
        # Use the larger of real elapsed time or accumulated latency
        wall_ms = (time.perf_counter() - t_start) * 1000
        outcome.total_latency_ms = max(wall_ms, total_ms)
        return outcome

    # -------------------------------------------------------------------

    def _estimate_confidence(
        self,
        vision_output: VisionOutput,
        edge_result: AnalysisResult,
        recent_history: list[VisionOutput] | None,
    ) -> float:
        """Select confidence estimation method based on config."""
        method = self.config.confidence_method
        if method == "output_prob":
            return estimate_from_output(vision_output, self.config)
        elif method == "self_verify":
            return estimate_from_self_verification(vision_output, edge_result)
        elif method == "temporal":
            if recent_history and len(recent_history) >= 3:
                return estimate_from_temporal(recent_history)
            return estimate_from_output(vision_output, self.config)
        else:  # "combined" or default
            return estimate_combined(
                vision_output, edge_result, recent_history, self.config,
            )

    @staticmethod
    def _record_cascade(edge: AnalysisResult, cloud: AnalysisResult) -> None:
        CASCADE_TOTAL.inc()
        if edge.judgment == cloud.judgment:
            CASCADE_EDGE_CONFIRMED.inc()
        else:
            CASCADE_EDGE_OVERRIDDEN.inc()
