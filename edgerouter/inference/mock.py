"""Mock analyzers for testing and offline evaluation.

These rule-based analyzers replicate what real LLMs produce:
- Edge (small model): uses anomaly_score + level deviation + secondary_metric,
  but has limited ability to detect marginal/gradual anomalies.
- Cloud (large model): uses all indicators + level position + confidence
  cross-checking, more sensitive to subtle patterns.
"""

from __future__ import annotations

from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput
from edgerouter.inference.base import AnalyzerBackend


class MockEdgeAnalyzer(AnalyzerBackend):
    """Deterministic mock edge analyzer (Qwen3.5-4B level).

    Capabilities:
    - Good at obvious anomalies (high anomaly_score)
    - Decent at level deviation detection
    - Weak on multi-indicator correlation and gradual trends
    """

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        score = vision_output.anomaly_score
        level = vision_output.anomaly_level
        secondary = vision_output.secondary_metric
        meas_conf = vision_output.measurement_confidence

        # Level deviation from center (0-50 range → 0-1)
        level_dev = abs(level - 50.0) / 50.0

        # Edge composite: weighted anomaly signal
        # Edge model is decent at anomaly_score and level, weak on subtle secondary metric
        composite = 0.40 * score + 0.35 * level_dev + 0.15 * secondary + 0.10 * (1.0 - meas_conf)

        if composite > 0.45:
            judgment = Judgment.ALARM
            action = "emergency_stop"
            confidence = 0.5 + composite * 0.35
        elif composite > 0.22:
            judgment = Judgment.WARNING
            action = "adjust_flow"
            confidence = 0.45 + (1.0 - composite) * 0.3
        else:
            judgment = Judgment.NORMAL
            action = "maintain"
            confidence = 0.7 + (1.0 - composite) * 0.25

        return AnalysisResult(
            judgment=judgment,
            confidence=min(0.99, confidence),
            suggested_action=action,
            reasoning=f"Mock edge: composite={composite:.3f} (score={score:.3f}, level_dev={level_dev:.3f})",
            latency_ms=45.0,
            source="edge",
        )

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Model size ablation: edge analyzers with different capability levels
# ---------------------------------------------------------------------------

# Profiles: (alarm_threshold, warning_threshold, weights, latency_ms, indicators_used)
_MODEL_SIZE_PROFILES = {
    "0.6B": {
        "alarm_threshold": 0.55,
        "warning_threshold": 0.30,
        "weights": (0.55, 0.30, 0.10, 0.05),  # score, level_dev, secondary, conf
        "latency_ms": 25.0,
        "description": "Very weak: relies heavily on anomaly_score, misses subtle patterns",
    },
    "1.7B": {
        "alarm_threshold": 0.50,
        "warning_threshold": 0.26,
        "weights": (0.45, 0.30, 0.15, 0.10),
        "latency_ms": 35.0,
        "description": "Weak: slightly better at secondary metric, still poor on correlation",
    },
    "4B": {
        "alarm_threshold": 0.45,
        "warning_threshold": 0.22,
        "weights": (0.40, 0.35, 0.15, 0.10),
        "latency_ms": 45.0,
        "description": "Default: decent at anomaly_score + level, weak on subtle",
    },
    "8B": {
        "alarm_threshold": 0.42,
        "warning_threshold": 0.20,
        "weights": (0.30, 0.25, 0.20, 0.25),
        "latency_ms": 80.0,
        "description": "Strong: uses all 4 indicators more evenly, closer to cloud",
    },
}


class SizedMockEdgeAnalyzer(AnalyzerBackend):
    """Mock edge analyzer parameterized by model size.

    Models capability differences between edge model sizes:
    - Smaller models: higher thresholds (miss more), fewer indicators, lower latency
    - Larger models: lower thresholds (catch more), better multi-indicator use, higher latency
    """

    def __init__(self, model_size: str = "4B"):
        if model_size not in _MODEL_SIZE_PROFILES:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(_MODEL_SIZE_PROFILES.keys())}")
        self.model_size = model_size
        self._profile = _MODEL_SIZE_PROFILES[model_size]

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        score = vision_output.anomaly_score
        level_dev = abs(vision_output.anomaly_level - 50.0) / 50.0
        secondary = vision_output.secondary_metric
        conf_factor = 1.0 - vision_output.measurement_confidence

        w = self._profile["weights"]
        composite = w[0] * score + w[1] * level_dev + w[2] * secondary + w[3] * conf_factor

        alarm_t = self._profile["alarm_threshold"]
        warn_t = self._profile["warning_threshold"]

        if composite > alarm_t:
            judgment = Judgment.ALARM
            action = "emergency_stop"
            confidence = 0.5 + composite * 0.35
        elif composite > warn_t:
            judgment = Judgment.WARNING
            action = "adjust_flow"
            confidence = 0.45 + (1.0 - composite) * 0.3
        else:
            judgment = Judgment.NORMAL
            action = "maintain"
            confidence = 0.7 + (1.0 - composite) * 0.25

        return AnalysisResult(
            judgment=judgment,
            confidence=min(0.99, confidence),
            suggested_action=action,
            reasoning=f"Mock edge ({self.model_size}): composite={composite:.3f}",
            latency_ms=self._profile["latency_ms"],
            source="edge",
        )

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# WAN delay wrapper (Experiment 4)
# ---------------------------------------------------------------------------

class WANDelayCloudAnalyzer(AnalyzerBackend):
    """Wraps a cloud analyzer and adds WAN round-trip delay."""

    def __init__(self, inner: AnalyzerBackend, wan_delay_ms: float = 50.0):
        self.inner = inner
        self.wan_delay_ms = wan_delay_ms

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        result = await self.inner.analyze(vision_output, recent_history, edge_draft)
        # Add WAN delay to the reported latency (no actual sleep for benchmark speed)
        result.latency_ms += self.wan_delay_ms
        return result

    async def health_check(self) -> bool:
        return await self.inner.health_check()


class MockCloudAnalyzer(AnalyzerBackend):
    """Deterministic mock cloud analyzer (Qwen3.5-27B level).

    Capabilities:
    - Excellent multi-indicator correlation
    - Sensitive to subtle secondary metric and uniformity changes
    - Uses level position + vision confidence for cross-checking
    - Can detect gradual degradation patterns
    """

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        score = vision_output.anomaly_score
        level = vision_output.anomaly_level
        secondary = vision_output.secondary_metric
        texture = vision_output.texture_irregularity
        uniformity = vision_output.surface_uniformity
        meas_conf = vision_output.measurement_confidence
        anomaly_conf = vision_output.anomaly_confidence

        # Level deviation
        level_dev = abs(level - 50.0) / 50.0

        # Cloud uses all indicators with better weighting
        composite = (
            0.25 * score
            + 0.20 * level_dev
            + 0.20 * secondary
            + 0.15 * texture
            + 0.10 * (1.0 - uniformity)
            + 0.10 * (1.0 - anomaly_conf)   # low model confidence = suspect
        )

        # Cross-check: if vision model itself is uncertain, cloud is more cautious
        if anomaly_conf < 0.6 and score > 0.15:
            composite += 0.08

        if composite > 0.40:
            judgment = Judgment.ALARM
            action = "emergency_stop"
            confidence = 0.65 + composite * 0.3
            root_cause = f"Multi-indicator anomaly: composite={composite:.3f}"
        elif composite > 0.18:
            judgment = Judgment.WARNING
            action = "reduce_input"
            confidence = 0.55 + composite * 0.35
            root_cause = f"Elevated indicators: composite={composite:.3f}"
        else:
            judgment = Judgment.NORMAL
            action = "maintain"
            confidence = 0.85 + (1.0 - composite) * 0.1
            root_cause = ""

        return AnalysisResult(
            judgment=judgment,
            confidence=min(0.99, confidence),
            suggested_action=action,
            reasoning=f"Mock cloud: composite={composite:.3f}",
            root_cause=root_cause,
            latency_ms=500.0,
            source="cloud",
        )

    async def health_check(self) -> bool:
        return True
