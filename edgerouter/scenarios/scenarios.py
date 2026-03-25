"""Scenario profiles and generator for visual anomaly detection conditions."""

from __future__ import annotations

import random
from copy import deepcopy

from edgerouter.core.schema import (
    Difficulty,
    Judgment,
    ScenarioProfile,
    Trend,
)

# ---------------------------------------------------------------------------
# Pre-defined scenario templates
# ---------------------------------------------------------------------------

SCENARIO_TEMPLATES: dict[str, ScenarioProfile] = {
    # --- Normal (200 in eval set) ---
    "normal_stable": ScenarioProfile(
        name="Normal Steady State",
        difficulty=Difficulty.NORMAL,
        true_anomaly_level=50.0,
        true_secondary_metric=0.08,
        true_color_rgb=(180, 180, 180),
        true_texture_irregularity=0.02,
        true_surface_uniformity=0.95,
        trend=Trend.STABLE,
        ground_truth_judgment=Judgment.NORMAL,
        description="All indicators within normal range, stable operation",
    ),
    "normal_slight_wave": ScenarioProfile(
        name="Normal Minor Fluctuation",
        difficulty=Difficulty.NORMAL,
        true_anomaly_level=48.0,
        true_secondary_metric=0.12,
        true_color_rgb=(175, 175, 180),
        true_texture_irregularity=0.05,
        true_surface_uniformity=0.88,
        trend=Trend.OSCILLATING,
        ground_truth_judgment=Judgment.NORMAL,
        description="Minor natural fluctuation in indicators, all within safe range",
    ),
    "normal_rising_safe": ScenarioProfile(
        name="Normal Gradual Rise (Safe)",
        difficulty=Difficulty.NORMAL,
        true_anomaly_level=55.0,
        true_secondary_metric=0.10,
        trend=Trend.RISING,
        ground_truth_judgment=Judgment.NORMAL,
        description="Primary indicator slowly rising but still within safe range",
    ),
    # --- Marginal (grey zone) ---
    "marginal_rising": ScenarioProfile(
        name="Indicator Approaching Upper Threshold",
        difficulty=Difficulty.MARGINAL,
        true_anomaly_level=68.0,
        true_secondary_metric=0.15,
        trend=Trend.RISING,
        ground_truth_judgment=Judgment.WARNING,
        description="Primary indicator rising, approaching upper safety threshold",
    ),
    "marginal_color": ScenarioProfile(
        name="Subtle Color Shift",
        difficulty=Difficulty.MARGINAL,
        true_anomaly_level=50.0,
        true_secondary_metric=0.30,
        true_color_rgb=(200, 150, 100),
        trend=Trend.STABLE,
        ground_truth_judgment=Judgment.WARNING,
        description="Color shift detected, possible early-stage contamination",
    ),
    "marginal_turbidity": ScenarioProfile(
        name="Secondary Metric Elevated",
        difficulty=Difficulty.MARGINAL,
        true_anomaly_level=52.0,
        true_secondary_metric=0.45,
        true_color_rgb=(170, 170, 160),
        true_texture_irregularity=0.10,
        trend=Trend.STABLE,
        ground_truth_judgment=Judgment.WARNING,
        description="Secondary quality metric elevated, requires monitoring",
    ),
    "marginal_falling": ScenarioProfile(
        name="Indicator Approaching Lower Threshold",
        difficulty=Difficulty.MARGINAL,
        true_anomaly_level=32.0,
        true_secondary_metric=0.10,
        trend=Trend.FALLING,
        ground_truth_judgment=Judgment.WARNING,
        description="Primary indicator falling, approaching lower safety threshold",
    ),
    # --- Anomalous ---
    "anomaly_sudden_drop": ScenarioProfile(
        name="Sudden Indicator Drop",
        difficulty=Difficulty.ANOMALOUS,
        true_anomaly_level=25.0,
        true_secondary_metric=0.20,
        trend=Trend.FALLING,
        num_correlated_anomalies=1,
        ground_truth_judgment=Judgment.ALARM,
        description="Sudden drop in primary indicator, possible equipment failure",
    ),
    "anomaly_multi": ScenarioProfile(
        name="Multi-Indicator Correlated Anomaly",
        difficulty=Difficulty.ANOMALOUS,
        true_anomaly_level=40.0,
        true_secondary_metric=0.65,
        true_color_rgb=(220, 100, 60),
        true_texture_irregularity=0.40,
        true_surface_uniformity=0.50,
        trend=Trend.OSCILLATING,
        num_correlated_anomalies=3,
        ground_truth_judgment=Judgment.ALARM,
        description="Multiple indicators simultaneously anomalous, correlated defect pattern",
    ),
    "anomaly_gradual_degradation": ScenarioProfile(
        name="Gradual Degradation",
        difficulty=Difficulty.ANOMALOUS,
        true_anomaly_level=48.0,
        true_secondary_metric=0.55,
        true_color_rgb=(190, 140, 90),
        trend=Trend.STABLE,
        num_correlated_anomalies=2,
        ground_truth_judgment=Judgment.ALARM,
        description="Slow degradation across metrics, requires trend analysis to detect",
    ),
    "novel_foam": ScenarioProfile(
        name="Novel Texture Pattern",
        difficulty=Difficulty.ANOMALOUS,
        true_anomaly_level=53.0,
        true_secondary_metric=0.35,
        true_texture_irregularity=0.80,
        true_surface_uniformity=0.30,
        trend=Trend.RISING,
        num_correlated_anomalies=1,
        is_novel=True,
        ground_truth_judgment=Judgment.ALARM,
        description="Previously unseen texture irregularity pattern",
    ),
    # --- Critical ---
    "critical_overflow": ScenarioProfile(
        name="Critical Upper Breach",
        difficulty=Difficulty.CRITICAL,
        true_anomaly_level=93.0,
        true_secondary_metric=0.20,
        trend=Trend.RISING,
        num_correlated_anomalies=1,
        ground_truth_judgment=Judgment.ALARM,
        description="Primary indicator at extreme high, imminent safety breach",
    ),
    "critical_empty": ScenarioProfile(
        name="Critical Lower Breach",
        difficulty=Difficulty.CRITICAL,
        true_anomaly_level=6.0,
        true_secondary_metric=0.05,
        trend=Trend.FALLING,
        ground_truth_judgment=Judgment.ALARM,
        description="Primary indicator at extreme low, equipment damage risk",
    ),
    "critical_sudden_surge": ScenarioProfile(
        name="Critical Surge with Texture Anomaly",
        difficulty=Difficulty.CRITICAL,
        true_anomaly_level=88.0,
        true_secondary_metric=0.40,
        true_texture_irregularity=0.50,
        trend=Trend.RISING,
        num_correlated_anomalies=2,
        ground_truth_judgment=Judgment.ALARM,
        description="Rapid indicator surge with texture anomaly, possible process runaway",
    ),
    # --- Sensitive data scenarios ---
    "sensitive_normal": ScenarioProfile(
        name="Normal with Sensitive Context",
        difficulty=Difficulty.NORMAL,
        true_anomaly_level=50.0,
        true_secondary_metric=0.10,
        contains_process_params=True,
        has_recipe_params=True,
        ground_truth_judgment=Judgment.NORMAL,
        description="Normal operation but context contains confidential process parameters",
    ),
    "sensitive_anomaly": ScenarioProfile(
        name="Anomaly with Sensitive Context",
        difficulty=Difficulty.ANOMALOUS,
        true_anomaly_level=42.0,
        true_secondary_metric=0.50,
        true_color_rgb=(210, 120, 70),
        num_correlated_anomalies=2,
        contains_process_params=True,
        has_recipe_params=True,
        has_reaction_params=True,
        ground_truth_judgment=Judgment.ALARM,
        description="Anomaly detected but contains sensitive data, must stay on edge",
    ),
}


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generate scenario instances from templates with randomisation."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate_one(
        self,
        template_key: str | None = None,
        difficulty: Difficulty | None = None,
    ) -> ScenarioProfile:
        """Generate a single scenario instance.

        If *template_key* is given, use that template as base.
        If *difficulty* is given, randomly pick a template of that difficulty.
        If neither, pick a random template.
        """
        if template_key:
            base = SCENARIO_TEMPLATES[template_key]
        elif difficulty:
            candidates = [
                k for k, v in SCENARIO_TEMPLATES.items()
                if v.difficulty == difficulty
            ]
            template_key = self.rng.choice(candidates)
            base = SCENARIO_TEMPLATES[template_key]
        else:
            template_key = self.rng.choice(list(SCENARIO_TEMPLATES.keys()))
            base = SCENARIO_TEMPLATES[template_key]

        scenario = deepcopy(base)
        self._add_noise(scenario)
        return scenario

    def generate_batch(
        self,
        distribution: dict[Difficulty, int] | None = None,
        total: int = 100,
    ) -> list[ScenarioProfile]:
        """Generate a batch following a difficulty distribution.

        Default distribution follows the evaluation plan:
        normal=50%, marginal=23%, anomalous=17%, critical=5%, sensitive=5%
        """
        if distribution is None:
            distribution = {
                Difficulty.NORMAL: int(total * 0.50),
                Difficulty.MARGINAL: int(total * 0.23),
                Difficulty.ANOMALOUS: int(total * 0.17),
                Difficulty.CRITICAL: max(1, int(total * 0.05)),
            }
            # Fill remainder with sensitive scenarios
            assigned = sum(distribution.values())
            remaining = total - assigned
            # Add sensitive scenarios as normal difficulty with sensitive flags
            distribution[Difficulty.NORMAL] += remaining

        scenarios: list[ScenarioProfile] = []
        for diff, count in distribution.items():
            for _ in range(count):
                scenarios.append(self.generate_one(difficulty=diff))

        self.rng.shuffle(scenarios)
        return scenarios

    # -----------------------------------------------------------------------

    def _add_noise(self, scenario: ScenarioProfile) -> None:
        """Add small random perturbations to make each instance unique."""
        noise_map = {
            Difficulty.NORMAL: 1.0,
            Difficulty.MARGINAL: 2.0,
            Difficulty.ANOMALOUS: 3.0,
            Difficulty.CRITICAL: 4.0,
        }
        scale = noise_map.get(scenario.difficulty, 1.0)

        scenario.true_anomaly_level += self.rng.gauss(0, scale * 0.5)
        scenario.true_anomaly_level = max(0.0, min(100.0, scenario.true_anomaly_level))

        scenario.true_secondary_metric += self.rng.gauss(0, scale * 0.02)
        scenario.true_secondary_metric = max(0.0, min(1.0, scenario.true_secondary_metric))

        r, g, b = scenario.true_color_rgb
        jitter = int(scale * 3)
        scenario.true_color_rgb = (
            max(0, min(255, r + self.rng.randint(-jitter, jitter))),
            max(0, min(255, g + self.rng.randint(-jitter, jitter))),
            max(0, min(255, b + self.rng.randint(-jitter, jitter))),
        )
