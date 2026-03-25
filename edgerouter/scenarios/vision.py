"""Vision model for edge detection scenarios."""

from __future__ import annotations

import time

import numpy as np

from edgerouter.core.config import VisionConfig
from edgerouter.core.schema import Difficulty, ScenarioProfile, VisionOutput


class VisionModel:
    """Edge vision model (MobileNetV3-Small proxy).

    Generates realistic detection outputs based on scenario ground truth
    with difficulty-dependent noise.  No real inference happens — this
    exists to drive the rest of the pipeline.
    """

    def __init__(self, config: VisionConfig | None = None, seed: int | None = None):
        self.config = config or VisionConfig()
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def detect(self, scenario: ScenarioProfile) -> VisionOutput:
        """Produce a detection result for the given scenario."""
        t_start = time.perf_counter()

        noise = self._noise_scale(scenario.difficulty)

        # Anomaly level: ground truth + Gaussian noise
        level = scenario.true_anomaly_level + self.rng.normal(0, noise)
        level = float(np.clip(level, 0.0, 100.0))

        # Measurement confidence: higher for easy scenarios
        meas_conf = self._compute_confidence(scenario, "measurement")

        # Secondary metric with noise
        secondary = scenario.true_secondary_metric + self.rng.normal(0, noise * 0.02)
        secondary = float(np.clip(secondary, 0.0, 1.0))

        # Color with jitter
        jitter = int(noise * 3)
        r, g, b = scenario.true_color_rgb
        color_rgb = (
            int(np.clip(r + self.rng.integers(-jitter, jitter + 1), 0, 255)),
            int(np.clip(g + self.rng.integers(-jitter, jitter + 1), 0, 255)),
            int(np.clip(b + self.rng.integers(-jitter, jitter + 1), 0, 255)),
        )

        # Texture irregularity
        texture = scenario.true_texture_irregularity + self.rng.normal(0, noise * 0.01)
        texture = float(np.clip(texture, 0.0, 1.0))

        # Surface uniformity
        uniformity = scenario.true_surface_uniformity + self.rng.normal(0, noise * 0.02)
        uniformity = float(np.clip(uniformity, 0.0, 1.0))

        # Anomaly score: derived from how far indicators deviate from normal
        anomaly_score = self._compute_anomaly_score(
            level, secondary, texture, uniformity, scenario,
        )

        # Anomaly confidence: model's confidence in its anomaly classification
        anomaly_conf = self._compute_confidence(scenario, "anomaly")

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        # Inference latency model
        sim_latency = self.config.inference_latency_ms + self.rng.normal(0, 1.0)

        return VisionOutput(
            timestamp=time.time(),
            anomaly_level=level,
            measurement_confidence=meas_conf,
            color_rgb=color_rgb,
            secondary_metric=secondary,
            texture_irregularity=texture,
            surface_uniformity=uniformity,
            anomaly_score=anomaly_score,
            anomaly_confidence=anomaly_conf,
            inference_latency_ms=max(1.0, sim_latency),
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _noise_scale(self, difficulty: Difficulty) -> float:
        """Return noise magnitude for a given difficulty."""
        mapping = {
            Difficulty.NORMAL: self.config.noise_scale_normal,
            Difficulty.MARGINAL: self.config.noise_scale_marginal,
            Difficulty.ANOMALOUS: self.config.noise_scale_anomalous,
            Difficulty.CRITICAL: self.config.noise_scale_critical,
        }
        return mapping.get(difficulty, 1.0)

    def _compute_confidence(self, scenario: ScenarioProfile, aspect: str) -> float:
        """Compute model confidence — higher for easier scenarios.

        This replicates the behaviour where a small model is
        confident on routine patterns and uncertain on novel / complex ones.
        """
        base = {
            Difficulty.NORMAL: 0.92,
            Difficulty.MARGINAL: 0.72,
            Difficulty.ANOMALOUS: 0.55,
            Difficulty.CRITICAL: 0.65,  # extreme values are somewhat obvious
        }[scenario.difficulty]

        # Novel scenarios: model has never seen this → extra uncertainty
        if scenario.is_novel:
            base -= 0.15

        # Multi-factor correlation: harder for small model
        base -= 0.05 * scenario.num_correlated_anomalies

        # Small random perturbation
        base += self.rng.normal(0, 0.03)

        return float(np.clip(base, 0.1, 0.99))

    def _compute_anomaly_score(
        self,
        level: float,
        secondary: float,
        texture: float,
        uniformity: float,
        scenario: ScenarioProfile,
    ) -> float:
        """Compute an anomaly score in [0, 1] from indicators."""
        # Distance from safe center (normalised)
        level_deviation = abs(level - 50.0) / 50.0

        # Secondary metric: higher = more anomalous
        secondary_score = secondary

        # Texture: higher = more anomalous
        texture_score = texture

        # Surface non-uniformity
        surface_score = 1.0 - uniformity

        raw = 0.35 * level_deviation + 0.25 * secondary_score + 0.20 * texture_score + 0.20 * surface_score

        # Add noise (model imprecision)
        noise = self._noise_scale(scenario.difficulty) * 0.02
        raw += self.rng.normal(0, noise)

        return float(np.clip(raw, 0.0, 1.0))
