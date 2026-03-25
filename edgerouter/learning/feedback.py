"""Feedback collector: aggregate cascade outcomes for analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from edgerouter.core.schema import Difficulty, Judgment, RoutingOutcome, RoutingTier


@dataclass
class FeedbackRecord:
    """A single feedback entry from a cascade event."""

    scenario_id: str
    edge_judgment: Judgment
    cloud_judgment: Judgment
    edge_confidence: float
    edge_confirmed: bool
    difficulty: Difficulty | None = None


class FeedbackCollector:
    """Collect and aggregate feedback from cascade routing outcomes."""

    def __init__(self):
        self.records: list[FeedbackRecord] = []
        self._by_difficulty: dict[str, list[FeedbackRecord]] = defaultdict(list)

    def record(self, outcome: RoutingOutcome, difficulty: Difficulty | None = None):
        """Record a cascade outcome."""
        if outcome.edge_analysis is None or outcome.cloud_analysis is None:
            return

        confirmed = outcome.edge_analysis.judgment == outcome.cloud_analysis.judgment
        rec = FeedbackRecord(
            scenario_id=outcome.scenario_id,
            edge_judgment=outcome.edge_analysis.judgment,
            cloud_judgment=outcome.cloud_analysis.judgment,
            edge_confidence=outcome.edge_confidence,
            edge_confirmed=confirmed,
            difficulty=difficulty,
        )
        self.records.append(rec)
        if difficulty:
            self._by_difficulty[difficulty.value].append(rec)

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def confirmation_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.edge_confirmed) / len(self.records)

    def stats_by_difficulty(self) -> dict[str, dict]:
        result = {}
        for diff, recs in self._by_difficulty.items():
            total = len(recs)
            confirmed = sum(1 for r in recs if r.edge_confirmed)
            avg_conf = sum(r.edge_confidence for r in recs) / max(1, total)
            result[diff] = {
                "total": total,
                "confirmed": confirmed,
                "overridden": total - confirmed,
                "confirmation_rate": round(confirmed / max(1, total), 3),
                "avg_edge_confidence": round(avg_conf, 3),
            }
        return result

    def confidence_calibration(self, bins: int = 10) -> list[dict]:
        """Check if confidence is well-calibrated.

        For each confidence bin, compute the actual confirmation rate.
        Ideal: confidence 0.8 → 80% confirmed.
        """
        if not self.records:
            return []

        bin_width = 1.0 / bins
        result = []
        for i in range(bins):
            lo = i * bin_width
            hi = lo + bin_width
            in_bin = [r for r in self.records if lo <= r.edge_confidence < hi]
            if not in_bin:
                continue
            actual_rate = sum(1 for r in in_bin if r.edge_confirmed) / len(in_bin)
            result.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "count": len(in_bin),
                "confirmation_rate": round(actual_rate, 3),
                "expected_mid": round((lo + hi) / 2, 2),
            })
        return result
