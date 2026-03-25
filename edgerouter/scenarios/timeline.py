"""Timeline generator: models a continuous production run."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from edgerouter.core.schema import Difficulty, ScenarioProfile, Trend
from edgerouter.scenarios.scenarios import ScenarioGenerator, SCENARIO_TEMPLATES


@dataclass
class TimelineSegment:
    """A contiguous segment of the production timeline."""

    scenario: ScenarioProfile
    duration_frames: int          # how many frames this segment lasts
    start_frame: int = 0


@dataclass
class ProductionTimeline:
    """Full timeline of a production run."""

    segments: list[TimelineSegment] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 30.0

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps

    def iter_frames(self):
        """Yield (frame_index, scenario) for every frame."""
        for seg in self.segments:
            for i in range(seg.duration_frames):
                yield seg.start_frame + i, seg.scenario


class TimelineGenerator:
    """Generate realistic production timelines.

    A production day consists mostly of normal operation, with
    occasional transitions to marginal / anomalous / critical states.
    This models a Markov-like process with configurable transition
    probabilities.
    """

    # Transition probabilities from each state (to normal/marginal/anomalous/critical)
    TRANSITION_PROBS: dict[Difficulty, dict[Difficulty, float]] = {
        Difficulty.NORMAL: {
            Difficulty.NORMAL: 0.90,
            Difficulty.MARGINAL: 0.08,
            Difficulty.ANOMALOUS: 0.015,
            Difficulty.CRITICAL: 0.005,
        },
        Difficulty.MARGINAL: {
            Difficulty.NORMAL: 0.50,
            Difficulty.MARGINAL: 0.35,
            Difficulty.ANOMALOUS: 0.12,
            Difficulty.CRITICAL: 0.03,
        },
        Difficulty.ANOMALOUS: {
            Difficulty.NORMAL: 0.30,
            Difficulty.MARGINAL: 0.25,
            Difficulty.ANOMALOUS: 0.30,
            Difficulty.CRITICAL: 0.15,
        },
        Difficulty.CRITICAL: {
            Difficulty.NORMAL: 0.20,
            Difficulty.MARGINAL: 0.30,
            Difficulty.ANOMALOUS: 0.30,
            Difficulty.CRITICAL: 0.20,
        },
    }

    # Duration ranges (in frames) for each difficulty
    DURATION_RANGES: dict[Difficulty, tuple[int, int]] = {
        Difficulty.NORMAL: (300, 1800),     # 10-60s at 30fps
        Difficulty.MARGINAL: (150, 600),    # 5-20s
        Difficulty.ANOMALOUS: (90, 300),    # 3-10s
        Difficulty.CRITICAL: (30, 150),     # 1-5s
    }

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.scenario_gen = ScenarioGenerator(seed=seed)

    def generate(
        self,
        total_frames: int = 18000,   # 10 minutes at 30fps
        fps: float = 30.0,
        initial_difficulty: Difficulty = Difficulty.NORMAL,
    ) -> ProductionTimeline:
        """Generate a production timeline with Markov transitions."""
        segments: list[TimelineSegment] = []
        current_frame = 0
        current_difficulty = initial_difficulty

        while current_frame < total_frames:
            # Generate scenario for this segment
            scenario = self.scenario_gen.generate_one(difficulty=current_difficulty)

            # Determine segment duration
            lo, hi = self.DURATION_RANGES[current_difficulty]
            duration = self.rng.randint(lo, hi)
            duration = min(duration, total_frames - current_frame)

            segments.append(TimelineSegment(
                scenario=scenario,
                duration_frames=duration,
                start_frame=current_frame,
            ))

            current_frame += duration

            # Transition to next state
            current_difficulty = self._next_state(current_difficulty)

        return ProductionTimeline(
            segments=segments,
            total_frames=current_frame,
            fps=fps,
        )

    def _next_state(self, current: Difficulty) -> Difficulty:
        probs = self.TRANSITION_PROBS[current]
        states = list(probs.keys())
        weights = list(probs.values())
        return self.rng.choices(states, weights=weights, k=1)[0]
