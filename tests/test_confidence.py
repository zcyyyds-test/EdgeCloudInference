"""Tests for confidence estimation methods."""

import numpy as np
import pytest

from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput
from edgerouter.router.confidence import (
    estimate_combined,
    estimate_from_output,
    estimate_from_self_verification,
    estimate_from_temporal,
)


class TestOutputProbability:
    def test_high_confidence_normal(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.9)
        conf = estimate_from_output(vo)
        assert conf > 0.85

    def test_low_confidence_anomaly(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.3)
        conf = estimate_from_output(vo)
        assert conf < 0.5

    def test_margin_bonus_at_center(self):
        vo_center = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.8)
        vo_edge = VisionOutput(anomaly_level=85.0, anomaly_confidence=0.8)
        conf_center = estimate_from_output(vo_center)
        conf_edge = estimate_from_output(vo_edge)
        # Center of safe range should get a bonus
        assert conf_center > conf_edge

    def test_clipped_to_01(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.99)
        conf = estimate_from_output(vo)
        assert 0.0 <= conf <= 1.0


class TestSelfVerification:
    def test_agreement_boosts_confidence(self):
        vo = VisionOutput(anomaly_score=0.6, anomaly_confidence=0.6)
        result = AnalysisResult(judgment=Judgment.WARNING, confidence=0.65)
        # Both say anomaly → consistency bonus
        conf = estimate_from_self_verification(vo, result)
        assert conf > 0.5

    def test_disagreement_lowers_confidence(self):
        vo = VisionOutput(anomaly_score=0.6, anomaly_confidence=0.6)
        result = AnalysisResult(judgment=Judgment.NORMAL, confidence=0.7)
        # Vision says anomaly, analyzer says normal → penalty
        conf = estimate_from_self_verification(vo, result)
        assert conf < 0.7

    def test_output_in_range(self):
        vo = VisionOutput(anomaly_score=0.3, anomaly_confidence=0.8)
        result = AnalysisResult(judgment=Judgment.NORMAL, confidence=0.85)
        conf = estimate_from_self_verification(vo, result)
        assert 0.0 <= conf <= 1.0


class TestTemporalConsistency:
    def test_stable_readings_high_confidence(self):
        outputs = [
            VisionOutput(anomaly_level=50.0 + np.random.normal(0, 0.3), secondary_metric=0.1)
            for _ in range(10)
        ]
        conf = estimate_from_temporal(outputs)
        assert conf > 0.7

    def test_noisy_readings_low_confidence(self):
        outputs = [
            VisionOutput(anomaly_level=50.0 + np.random.normal(0, 8.0), secondary_metric=0.1 + np.random.normal(0, 0.2))
            for _ in range(10)
        ]
        conf = estimate_from_temporal(outputs)
        assert conf < 0.6

    def test_insufficient_data_returns_conservative(self):
        outputs = [VisionOutput(anomaly_level=50.0)]
        conf = estimate_from_temporal(outputs)
        assert conf == 0.5

    def test_output_in_range(self):
        outputs = [VisionOutput(anomaly_level=40.0 + i) for i in range(15)]
        conf = estimate_from_temporal(outputs)
        assert 0.0 <= conf <= 1.0


class TestCombinedEstimator:
    def test_all_methods_available(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.8, anomaly_score=0.3)
        edge = AnalysisResult(judgment=Judgment.NORMAL, confidence=0.85)
        history = [VisionOutput(anomaly_level=50.0, secondary_metric=0.1) for _ in range(10)]
        conf = estimate_combined(vo, edge, history)
        assert 0.0 <= conf <= 1.0

    def test_only_output_method(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.8)
        conf = estimate_combined(vo)
        assert 0.0 <= conf <= 1.0

    def test_with_edge_no_history(self):
        vo = VisionOutput(anomaly_level=50.0, anomaly_confidence=0.8)
        edge = AnalysisResult(judgment=Judgment.NORMAL, confidence=0.85)
        conf = estimate_combined(vo, edge)
        assert 0.0 <= conf <= 1.0
