"""Tests for the router engine and sub-components."""

import time

import pytest

from edgerouter.core.config import RouterConfig
from edgerouter.core.schema import (
    AnalysisResult,
    Judgment,
    ProcessContext,
    RoutingTier,
    VisionOutput,
)
from edgerouter.router.engine import RouterEngine
from edgerouter.router.safety import SafetyClassifier
from edgerouter.router.data_security import DataSecurityChecker


# -----------------------------------------------------------------------
# SafetyClassifier
# -----------------------------------------------------------------------

class TestSafetyClassifier:
    def setup_method(self):
        self.clf = SafetyClassifier()

    def test_overflow_is_critical(self):
        vo = VisionOutput(anomaly_level=96.0)
        assert self.clf.is_critical(vo) is True

    def test_empty_is_critical(self):
        vo = VisionOutput(anomaly_level=3.0)
        assert self.clf.is_critical(vo) is True

    def test_normal_level_not_critical(self):
        vo = VisionOutput(anomaly_level=50.0)
        assert self.clf.is_critical(vo) is False

    def test_sudden_rate_is_critical(self):
        t = time.time()
        prev = VisionOutput(anomaly_level=50.0, timestamp=t)
        curr = VisionOutput(anomaly_level=62.0, timestamp=t + 1.0)  # 12 cm/s
        assert self.clf.is_critical(curr, prev) is True

    def test_slow_rate_not_critical(self):
        t = time.time()
        prev = VisionOutput(anomaly_level=50.0, timestamp=t)
        curr = VisionOutput(anomaly_level=51.0, timestamp=t + 1.0)  # 1 cm/s
        assert self.clf.is_critical(curr, prev) is False

    def test_get_reason_overflow(self):
        vo = VisionOutput(anomaly_level=96.0)
        reason = self.clf.get_reason(vo)
        assert "upper_bound_breach" in reason


# -----------------------------------------------------------------------
# DataSecurityChecker
# -----------------------------------------------------------------------

class TestDataSecurityChecker:
    def setup_method(self):
        self.checker = DataSecurityChecker()

    def test_recipe_params_is_sensitive(self):
        ctx = ProcessContext(has_recipe_params=True)
        assert self.checker.contains_sensitive(ctx) is True

    def test_customer_info_is_sensitive(self):
        ctx = ProcessContext(has_customer_info=True)
        assert self.checker.contains_sensitive(ctx) is True

    def test_reaction_params_is_sensitive(self):
        ctx = ProcessContext(has_reaction_params=True)
        assert self.checker.contains_sensitive(ctx) is True

    def test_empty_context_not_sensitive(self):
        ctx = ProcessContext()
        assert self.checker.contains_sensitive(ctx) is False

    def test_sanitize_excludes_sensitive_fields(self):
        vo = VisionOutput(anomaly_level=50.0)
        ctx = ProcessContext(equipment_id="EQ-001", batch_id="B-123")
        sanitized = self.checker.sanitize_for_cloud(vo, ctx)
        assert "anomaly_level" in sanitized
        assert "equipment_id" not in sanitized
        assert "batch_id" not in sanitized


# -----------------------------------------------------------------------
# RouterEngine
# -----------------------------------------------------------------------

class TestRouterEngine:
    def setup_method(self):
        self.engine = RouterEngine()

    def test_critical_overflow_routes_emergency(self):
        vo = VisionOutput(anomaly_level=96.0)
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.EDGE_EMERGENCY

    def test_critical_empty_routes_emergency(self):
        vo = VisionOutput(anomaly_level=3.0)
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.EDGE_EMERGENCY

    def test_sensitive_data_routes_edge(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.5)
        ctx = ProcessContext(has_recipe_params=True)
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.EDGE
        assert "data_security" in decision.reason

    def test_clearly_normal_routes_edge(self):
        vo = VisionOutput(
            anomaly_level=50.0,
            anomaly_score=0.1,
            anomaly_confidence=0.92,
        )
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.EDGE
        assert "clearly_normal" in decision.reason

    def test_complex_anomaly_routes_cloud(self):
        vo = VisionOutput(
            anomaly_level=50.0,
            anomaly_score=0.85,
            anomaly_confidence=0.5,
        )
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.CLOUD

    def test_correlated_anomalies_route_cloud(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.4)
        ctx = ProcessContext(num_correlated_anomalies=3)
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.CLOUD

    def test_grey_zone_high_confidence_routes_edge(self):
        vo = VisionOutput(
            anomaly_level=50.0,
            anomaly_score=0.3,
            anomaly_confidence=0.80,
        )
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        # With confidence ~0.80 + margin bonus, should be above 0.7 threshold
        assert decision.tier == RoutingTier.EDGE
        assert "confident_edge" in decision.reason

    def test_grey_zone_low_confidence_cascades(self):
        vo = VisionOutput(
            anomaly_level=50.0,
            anomaly_score=0.4,
            anomaly_confidence=0.45,
        )
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.tier == RoutingTier.CASCADE

    def test_routing_decision_has_latency(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_score=0.1, anomaly_confidence=0.92)
        ctx = ProcessContext()
        decision = self.engine.route(vo, ctx)
        assert decision.latency_ms >= 0

    def test_tier_priority_safety_over_data_security(self):
        """Safety check comes before data security check."""
        vo = VisionOutput(anomaly_level=96.0)
        ctx = ProcessContext(has_recipe_params=True)
        decision = self.engine.route(vo, ctx)
        # Safety should win over data security
        assert decision.tier == RoutingTier.EDGE_EMERGENCY
