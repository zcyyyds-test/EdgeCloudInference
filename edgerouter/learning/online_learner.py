"""Online router learning: update thresholds from cascade outcomes."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from edgerouter.core.schema import RoutingOutcome, RoutingTier

logger = logging.getLogger(__name__)


@dataclass
class LearnerState:
    """Persistent state for the online learner."""

    threshold: float = 0.7
    update_count: int = 0
    confirmed_count: int = 0
    overridden_count: int = 0
    threshold_history: list[float] = field(default_factory=list)


class OnlineRouterLearner:
    """Update confidence threshold based on cascade outcomes.

    Key insight: every cascade produces a free label.
    - Cloud confirms edge → similar scenarios can stay on edge next time
    - Cloud overrides edge → similar scenarios should escalate next time

    Uses a simple exponential moving average approach.
    """

    def __init__(
        self,
        initial_threshold: float = 0.7,
        learning_rate: float = 0.01,
        min_threshold: float = 0.3,
        max_threshold: float = 0.95,
        window_size: int = 100,
    ):
        self.state = LearnerState(threshold=initial_threshold)
        self.lr = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_outcomes: deque[RoutingOutcome] = deque(maxlen=window_size)

    @property
    def threshold(self) -> float:
        return self.state.threshold

    def update(self, outcome: RoutingOutcome) -> float:
        """Process one routing outcome and update threshold.

        Returns the new threshold.
        """
        self.recent_outcomes.append(outcome)

        # Only learn from cascade events that reached cloud
        if outcome.routing_decision is None:
            return self.state.threshold

        if outcome.routing_decision.tier != RoutingTier.CASCADE:
            return self.state.threshold

        if outcome.edge_analysis is None or outcome.cloud_analysis is None:
            # Cascade resolved at edge (confident) — no new signal
            return self.state.threshold

        self.state.update_count += 1
        edge_confirmed = (
            outcome.edge_analysis.judgment == outcome.cloud_analysis.judgment
        )

        if edge_confirmed:
            # Edge was right → lower threshold (less cloud needed)
            self.state.confirmed_count += 1
            delta = -self.lr
        else:
            # Edge was wrong → raise threshold (more cloud needed)
            self.state.overridden_count += 1
            delta = +self.lr

        self.state.threshold = float(np.clip(
            self.state.threshold + delta,
            self.min_threshold,
            self.max_threshold,
        ))
        self.state.threshold_history.append(self.state.threshold)

        return self.state.threshold

    def get_stats(self) -> dict:
        total = self.state.confirmed_count + self.state.overridden_count
        return {
            "current_threshold": round(self.state.threshold, 4),
            "total_updates": self.state.update_count,
            "confirmed": self.state.confirmed_count,
            "overridden": self.state.overridden_count,
            "confirmation_rate": round(
                self.state.confirmed_count / max(1, total), 3
            ),
            "threshold_history_len": len(self.state.threshold_history),
        }
