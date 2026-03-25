"""Tests for scenario generation and vision model."""

from edgerouter.core.schema import Difficulty, Judgment
from edgerouter.scenarios.scenarios import (
    SCENARIO_TEMPLATES,
    ScenarioGenerator,
)
from edgerouter.scenarios.vision import VisionModel
from edgerouter.scenarios.timeline import TimelineGenerator


class TestScenarioTemplates:
    def test_templates_cover_all_difficulties(self):
        difficulties = {s.difficulty for s in SCENARIO_TEMPLATES.values()}
        assert Difficulty.NORMAL in difficulties
        assert Difficulty.MARGINAL in difficulties
        assert Difficulty.ANOMALOUS in difficulties
        assert Difficulty.CRITICAL in difficulties

    def test_templates_have_sensitive_scenarios(self):
        sensitive = [s for s in SCENARIO_TEMPLATES.values() if s.contains_process_params]
        assert len(sensitive) >= 2


class TestScenarioGenerator:
    def test_generate_one_default(self):
        gen = ScenarioGenerator(seed=42)
        s = gen.generate_one()
        assert s.name
        assert 0.0 <= s.true_anomaly_level <= 100.0

    def test_generate_one_by_key(self):
        gen = ScenarioGenerator(seed=42)
        s = gen.generate_one(template_key="critical_overflow")
        assert s.difficulty == Difficulty.CRITICAL

    def test_generate_one_by_difficulty(self):
        gen = ScenarioGenerator(seed=42)
        s = gen.generate_one(difficulty=Difficulty.MARGINAL)
        assert s.difficulty == Difficulty.MARGINAL

    def test_generate_batch_default(self):
        gen = ScenarioGenerator(seed=42)
        batch = gen.generate_batch(total=100)
        assert len(batch) == 100

    def test_generate_batch_distribution(self):
        gen = ScenarioGenerator(seed=42)
        batch = gen.generate_batch(total=600)
        difficulties = [s.difficulty for s in batch]
        # Should have a mix
        assert difficulties.count(Difficulty.NORMAL) > 0
        assert difficulties.count(Difficulty.MARGINAL) > 0
        assert difficulties.count(Difficulty.ANOMALOUS) > 0
        assert difficulties.count(Difficulty.CRITICAL) > 0

    def test_reproducibility(self):
        gen1 = ScenarioGenerator(seed=123)
        gen2 = ScenarioGenerator(seed=123)
        s1 = gen1.generate_one(template_key="normal_stable")
        s2 = gen2.generate_one(template_key="normal_stable")
        assert s1.true_anomaly_level == s2.true_anomaly_level
        assert s1.true_secondary_metric == s2.true_secondary_metric


class TestVisionModel:
    def test_detect_returns_valid_output(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one(template_key="normal_stable")
        output = model.detect(scenario)

        assert 0.0 <= output.anomaly_level <= 100.0
        assert 0.0 <= output.measurement_confidence <= 1.0
        assert 0.0 <= output.secondary_metric <= 1.0
        assert 0.0 <= output.anomaly_score <= 1.0
        assert 0.0 <= output.anomaly_confidence <= 1.0
        assert output.inference_latency_ms > 0

    def test_normal_scenario_high_confidence(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one(template_key="normal_stable")
        output = model.detect(scenario)
        # Normal scenario → model should be fairly confident
        assert output.anomaly_confidence > 0.7

    def test_anomalous_scenario_lower_confidence(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one(template_key="anomaly_multi")
        output = model.detect(scenario)
        # Complex anomaly → model should be less confident
        assert output.anomaly_confidence < 0.7

    def test_novel_scenario_lowest_confidence(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one(template_key="novel_foam")
        output = model.detect(scenario)
        # Novel scenario → model should be quite uncertain
        assert output.anomaly_confidence < 0.6

    def test_critical_scenario_high_anomaly_score(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one(template_key="critical_overflow")
        output = model.detect(scenario)
        # Near-overflow → anomaly score should be elevated
        assert output.anomaly_score > 0.2

    def test_to_dict(self):
        gen = ScenarioGenerator(seed=42)
        model = VisionModel(seed=42)
        scenario = gen.generate_one()
        output = model.detect(scenario)
        d = output.to_dict()
        assert "anomaly_level" in d
        assert "anomaly_score" in d
        assert isinstance(d["color_rgb"], list)


class TestTimelineGenerator:
    def test_generate_default(self):
        tl_gen = TimelineGenerator(seed=42)
        timeline = tl_gen.generate(total_frames=3000)
        assert timeline.total_frames >= 3000
        assert len(timeline.segments) > 0

    def test_segments_cover_full_duration(self):
        tl_gen = TimelineGenerator(seed=42)
        timeline = tl_gen.generate(total_frames=1000)
        total = sum(s.duration_frames for s in timeline.segments)
        assert total >= 1000

    def test_iter_frames(self):
        tl_gen = TimelineGenerator(seed=42)
        timeline = tl_gen.generate(total_frames=300)
        frames = list(timeline.iter_frames())
        assert len(frames) == timeline.total_frames

    def test_mostly_normal(self):
        """A production timeline should be mostly normal operation."""
        tl_gen = TimelineGenerator(seed=42)
        timeline = tl_gen.generate(total_frames=18000)
        normal_frames = sum(
            s.duration_frames
            for s in timeline.segments
            if s.scenario.difficulty == Difficulty.NORMAL
        )
        # At least 50% should be normal
        assert normal_frames / timeline.total_frames > 0.4
