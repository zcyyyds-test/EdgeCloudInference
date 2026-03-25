"""Result analysis utilities: Pareto curves, comparison tables, etc."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgerouter.eval.benchmarks import BenchmarkMetrics


@dataclass
class ThresholdSweepPoint:
    """One point on the confidence-threshold sweep."""

    threshold: float
    metrics: BenchmarkMetrics

    def to_row(self) -> dict:
        s = self.metrics.summary()
        s["threshold"] = self.threshold
        return s


class AnalysisReport:
    """Collects and analyses benchmark results."""

    def __init__(self):
        self.sweep_points: list[ThresholdSweepPoint] = []
        self.comparison: dict[str, BenchmarkMetrics] = {}  # config_name → metrics

    # -------------------------------------------------------------------
    # Threshold sweep analysis
    # -------------------------------------------------------------------

    def add_sweep_point(self, threshold: float, metrics: BenchmarkMetrics):
        self.sweep_points.append(ThresholdSweepPoint(threshold, metrics))

    def sweep_table(self) -> list[dict]:
        """Return sweep results as a list of dicts (for DataFrame or JSON)."""
        return [p.to_row() for p in sorted(self.sweep_points, key=lambda p: p.threshold)]

    def find_optimal_threshold(self, max_miss_rate: float = 0.02) -> float | None:
        """Find the threshold with highest cloud savings while miss_rate ≤ max."""
        valid = [
            p for p in self.sweep_points
            if p.metrics.miss_rate <= max_miss_rate
        ]
        if not valid:
            return None
        best = max(valid, key=lambda p: p.metrics.cloud_saving_rate)
        return best.threshold

    # -------------------------------------------------------------------
    # Comparison (all-edge / all-cloud / static / EdgeRouter)
    # -------------------------------------------------------------------

    def add_comparison(self, name: str, metrics: BenchmarkMetrics):
        self.comparison[name] = metrics

    def comparison_table(self) -> list[dict]:
        rows = []
        for name, m in self.comparison.items():
            row = m.summary()
            row["config"] = name
            rows.append(row)
        return rows

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def save_json(self, path: str | Path):
        data = {
            "sweep": self.sweep_table(),
            "comparison": self.comparison_table(),
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def print_summary(self):
        print("=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        for row in self.comparison_table():
            print(f"\n--- {row['config']} ---")
            for k, v in row.items():
                if k != "config":
                    print(f"  {k}: {v}")

        if self.sweep_points:
            opt = self.find_optimal_threshold()
            print(f"\n{'=' * 70}")
            print(f"OPTIMAL THRESHOLD (miss_rate ≤ 2%): {opt}")
            print("=" * 70)
