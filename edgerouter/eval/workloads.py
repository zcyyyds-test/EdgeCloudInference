"""Evaluation workload management: generate and manage benchmark scenario sets."""

from __future__ import annotations

from dataclasses import dataclass, field

from edgerouter.core.schema import Difficulty, Judgment, ScenarioProfile
from edgerouter.scenarios.scenarios import ScenarioGenerator


@dataclass
class EvalWorkload:
    """A labelled evaluation workload with ground-truth annotations."""

    name: str
    scenarios: list[ScenarioProfile] = field(default_factory=list)
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.scenarios)

    def by_difficulty(self, difficulty: Difficulty) -> list[ScenarioProfile]:
        return [s for s in self.scenarios if s.difficulty == difficulty]

    def by_judgment(self, judgment: Judgment) -> list[ScenarioProfile]:
        return [s for s in self.scenarios if s.ground_truth_judgment == judgment]

    def sensitive_only(self) -> list[ScenarioProfile]:
        return [s for s in self.scenarios if s.contains_process_params]


def build_standard_workload(seed: int = 42) -> EvalWorkload:
    """Build the standard 600-scenario evaluation workload.

    Distribution follows the evaluation plan:
    - Normal stable:           200
    - Normal fluctuation:      100
    - Marginal (level):         80
    - Marginal (property):      60
    - Anomalous (sudden):       30
    - Anomalous (multi):        30
    - Anomalous (gradual):      20
    - Critical:                 10
    - Novel:                    20
    - Sensitive data:           50
    """
    gen = ScenarioGenerator(seed=seed)
    scenarios: list[ScenarioProfile] = []

    def _gen_n(key: str, n: int):
        for _ in range(n):
            scenarios.append(gen.generate_one(template_key=key))

    # Normal (300 total)
    _gen_n("normal_stable", 200)
    _gen_n("normal_slight_wave", 80)
    _gen_n("normal_rising_safe", 20)

    # Marginal (140 total)
    _gen_n("marginal_rising", 40)
    _gen_n("marginal_color", 30)
    _gen_n("marginal_turbidity", 40)
    _gen_n("marginal_falling", 30)

    # Anomalous (100 total)
    _gen_n("anomaly_sudden_drop", 30)
    _gen_n("anomaly_multi", 30)
    _gen_n("anomaly_gradual_degradation", 20)
    _gen_n("novel_foam", 20)

    # Critical (10)
    _gen_n("critical_overflow", 4)
    _gen_n("critical_empty", 3)
    _gen_n("critical_sudden_surge", 3)

    # Sensitive data (50)
    _gen_n("sensitive_normal", 25)
    _gen_n("sensitive_anomaly", 25)

    return EvalWorkload(
        name="standard_600",
        scenarios=scenarios,
        description="Standard 600-scenario evaluation workload",
    )


def build_extended_workload(seed: int = 42) -> EvalWorkload:
    """Build a 1000-scenario workload for online learning convergence (Experiment 5).

    Proportionally scaled from the 600-scenario workload with more marginal
    and anomalous cases for richer learning signal.
    """
    gen = ScenarioGenerator(seed=seed)
    scenarios: list[ScenarioProfile] = []

    def _gen_n(key: str, n: int):
        for _ in range(n):
            scenarios.append(gen.generate_one(template_key=key))

    # Normal (480)
    _gen_n("normal_stable", 320)
    _gen_n("normal_slight_wave", 120)
    _gen_n("normal_rising_safe", 40)

    # Marginal (250)
    _gen_n("marginal_rising", 70)
    _gen_n("marginal_color", 50)
    _gen_n("marginal_turbidity", 70)
    _gen_n("marginal_falling", 60)

    # Anomalous (180)
    _gen_n("anomaly_sudden_drop", 55)
    _gen_n("anomaly_multi", 55)
    _gen_n("anomaly_gradual_degradation", 35)
    _gen_n("novel_foam", 35)

    # Critical (20)
    _gen_n("critical_overflow", 8)
    _gen_n("critical_empty", 6)
    _gen_n("critical_sudden_surge", 6)

    # Sensitive data (70)
    _gen_n("sensitive_normal", 35)
    _gen_n("sensitive_anomaly", 35)

    return EvalWorkload(
        name="extended_1000",
        scenarios=scenarios,
        description="Extended 1000-scenario workload for online learning convergence",
    )


def build_security_workload(seed: int = 42) -> EvalWorkload:
    """Build a 600-scenario workload focused on data security compliance (Experiment 6).

    50 sensitive + 550 non-sensitive. Sensitive scenarios include both normal
    and anomalous cases to test that security routing works regardless of
    anomaly status.
    """
    gen = ScenarioGenerator(seed=seed)
    scenarios: list[ScenarioProfile] = []

    def _gen_n(key: str, n: int):
        for _ in range(n):
            scenarios.append(gen.generate_one(template_key=key))

    # Non-sensitive (550)
    _gen_n("normal_stable", 200)
    _gen_n("normal_slight_wave", 80)
    _gen_n("normal_rising_safe", 20)
    _gen_n("marginal_rising", 40)
    _gen_n("marginal_color", 30)
    _gen_n("marginal_turbidity", 30)
    _gen_n("anomaly_sudden_drop", 30)
    _gen_n("anomaly_multi", 30)
    _gen_n("anomaly_gradual_degradation", 20)
    _gen_n("novel_foam", 20)
    _gen_n("critical_overflow", 4)
    _gen_n("critical_empty", 3)
    _gen_n("critical_sudden_surge", 3)
    _gen_n("marginal_falling", 40)

    # Sensitive (50) — mix of normal and anomalous
    _gen_n("sensitive_normal", 25)
    _gen_n("sensitive_anomaly", 25)

    return EvalWorkload(
        name="security_600",
        scenarios=scenarios,
        description="Security-focused 600-scenario workload (50 sensitive + 550 non-sensitive)",
    )


def build_small_workload(seed: int = 42) -> EvalWorkload:
    """Build a smaller 100-scenario workload for quick iteration."""
    gen = ScenarioGenerator(seed=seed)
    scenarios = gen.generate_batch(total=100)
    return EvalWorkload(
        name="quick_100",
        scenarios=scenarios,
        description="Quick 100-scenario workload for development",
    )
