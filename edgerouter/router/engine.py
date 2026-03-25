"""Router engine: 5-tier hierarchical routing decision logic."""

from __future__ import annotations

import time

from edgerouter.core.config import RouterConfig
from edgerouter.core.metrics import ROUTING_DECISIONS, ROUTING_LATENCY
from edgerouter.core.schema import (
    ProcessContext,
    RoutingDecision,
    RoutingTier,
    VisionOutput,
)
from edgerouter.router.confidence import (
    estimate_combined,
    estimate_from_output,
    estimate_from_self_verification,
    estimate_from_temporal,
)
from edgerouter.router.data_security import DataSecurityChecker
from edgerouter.router.safety import SafetyClassifier


class RouterEngine:
    """5-tier hierarchical routing engine.

    Tier 0: edge_emergency — critical safety, bypass everything
    Tier 1: edge (data_security) — sensitive data must stay on edge
    Tier 2: edge (clearly_normal) — all indicators normal + high confidence
    Tier 3: cloud (complex_anomaly) — multi-indicator or novel anomaly
    Tier 4: cascade — grey zone, edge first then optionally cloud
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.safety = SafetyClassifier(self.config)
        self.data_security = DataSecurityChecker()

    def route(
        self,
        vision_output: VisionOutput,
        context: ProcessContext,
        prev_output: VisionOutput | None = None,
    ) -> RoutingDecision:
        """Make a routing decision for the current frame."""
        t_start = time.perf_counter()

        # --- Tier 0: Safety emergency ---
        if self.safety.is_critical(vision_output, prev_output):
            reason = self.safety.get_reason(vision_output, prev_output)
            decision = RoutingDecision(
                tier=RoutingTier.EDGE_EMERGENCY,
                reason=f"critical_safety: {reason}",
                action="immediate_control",
            )
            return self._finalise(decision, t_start)

        # --- Tier 1: Data security ---
        if self.data_security.contains_sensitive(context):
            decision = RoutingDecision(
                tier=RoutingTier.EDGE,
                reason="data_security",
            )
            return self._finalise(decision, t_start)

        # --- Tier 2: Clearly normal ---
        if (
            vision_output.anomaly_score < self.config.clearly_normal_anomaly_max
            and vision_output.anomaly_confidence > self.config.clearly_normal_confidence_min
        ):
            decision = RoutingDecision(
                tier=RoutingTier.EDGE,
                reason="clearly_normal",
            )
            return self._finalise(decision, t_start)

        # --- Tier 3: Clearly complex ---
        if (
            vision_output.anomaly_score > self.config.complex_anomaly_score_min
            or context.num_correlated_anomalies > self.config.complex_correlated_anomalies_min
        ):
            decision = RoutingDecision(
                tier=RoutingTier.CLOUD,
                reason="complex_anomaly",
            )
            return self._finalise(decision, t_start)

        # --- Tier 4: Grey zone → cascade ---
        confidence = self._estimate_tier4_confidence(vision_output)
        if confidence >= self.config.confidence_threshold:
            decision = RoutingDecision(
                tier=RoutingTier.EDGE,
                reason=f"confident_edge (conf={confidence:.3f})",
            )
        else:
            decision = RoutingDecision(
                tier=RoutingTier.CASCADE,
                reason=f"uncertain (conf={confidence:.3f})",
            )

        return self._finalise(decision, t_start)

    # -------------------------------------------------------------------

    def _estimate_tier4_confidence(self, vision_output: VisionOutput) -> float:
        """Use the configured confidence method for Tier 4 decision.

        Note: self_verify and temporal methods require data not available at
        single-frame routing time (edge analysis result and history window
        respectively). They fall back to output_prob here. The full methods
        are exercised in CascadeExecutor where edge results and history exist.
        """
        # All methods converge to output_prob at routing time because
        # self_verify needs an edge result and temporal needs a history window.
        # The config option still matters for ablation logging and future
        # integration with streaming/batch routing.
        return estimate_from_output(vision_output, self.config)

    def _finalise(self, decision: RoutingDecision, t_start: float) -> RoutingDecision:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        decision.latency_ms = elapsed_ms
        ROUTING_DECISIONS.labels(tier=decision.tier.value, reason=decision.reason.split(":")[0].strip()).inc()
        ROUTING_LATENCY.observe(elapsed_ms / 1000)
        return decision
