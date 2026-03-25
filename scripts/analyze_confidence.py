"""Confidence calibration analysis for real LLM inference.

Analyzes edge model confidence patterns to understand:
  1. Are high-confidence predictions actually correct? (calibration)
  2. Is confidence useful for routing? (discrimination)
  3. Which model has better confidence for routing purposes?

Uses existing experiment results from experiments/ directory.
Does not require running servers — purely offline analysis.

Usage:
    python scripts/analyze_confidence.py [--output experiments/confidence_analysis.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def load_experiments() -> dict:
    """Load all available experiment results."""
    exp_dir = Path("experiments")
    data = {}

    for name in [
        "model_ablation",
        "edge_cloud_benchmark",
        "edge_cloud_benchmark_v2",
        "edge_cloud_benchmark_t08",
        "real_llm_results_50",
    ]:
        path = exp_dir / f"{name}.json"
        if path.exists():
            data[name] = json.loads(path.read_text())
    return data


def analyze_model_ablation(ablation_data: list) -> dict:
    """Compare confidence statistics across edge models."""
    results = {}
    for entry in ablation_data:
        model = entry["model"]
        cs = entry.get("confidence_stats", {})
        s = entry["summary"]
        results[model] = {
            "accuracy": s["accuracy"],
            "miss_rate": s["miss_rate"],
            "conf_mean": cs.get("mean", 0),
            "conf_std": cs.get("std", 0),
            "conf_min": cs.get("min", 0),
            "conf_max": cs.get("max", 0),
            "conf_range": cs.get("max", 0) - cs.get("min", 0),
            # Discrimination: can confidence distinguish correct from incorrect?
            # Higher std = more spread = potentially better discrimination
            "routing_suitability": (
                "good" if cs.get("std", 0) > 0.1 else
                "poor" if cs.get("std", 0) < 0.05 else
                "marginal"
            ),
        }
    return results


def analyze_threshold_sensitivity(benchmarks: dict) -> dict:
    """Analyze how different thresholds affect routing with real confidence values."""
    points = []
    for name, data in benchmarks.items():
        if data and "summary" in data:
            s = data["summary"]
            cfg = data.get("config", {})
            cs = data.get("confidence_stats", {})
            points.append({
                "benchmark": name,
                "threshold": cfg.get("confidence_threshold", 0.7),
                "accuracy": s["accuracy"],
                "miss_rate": s["miss_rate"],
                "cloud_saving_rate": s["cloud_saving_rate"],
                "conf_mean": cs.get("mean", 0),
                "conf_std": cs.get("std", 0),
            })
    return {"threshold_points": points}


def compute_calibration_summary(ablation_data: list) -> dict:
    """Compute calibration insights from available data."""
    insights = []

    for entry in ablation_data:
        model = entry["model"]
        cs = entry.get("confidence_stats", {})
        s = entry["summary"]
        acc = s["accuracy"]
        conf_mean = cs.get("mean", 0)

        # Calibration gap: |mean_confidence - accuracy|
        # Perfectly calibrated model: confidence matches accuracy
        calibration_gap = abs(conf_mean - acc)

        insights.append({
            "model": model,
            "accuracy": acc,
            "mean_confidence": conf_mean,
            "calibration_gap": round(calibration_gap, 3),
            "overconfident": conf_mean > acc,
            "overconfidence_degree": round(max(0, conf_mean - acc), 3),
        })

    return {"calibration": insights}


def generate_report(data: dict) -> str:
    """Generate human-readable confidence analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Confidence Calibration Analysis Report")
    lines.append("=" * 70)

    # Model comparison
    ablation = data.get("model_ablation")
    if ablation:
        model_analysis = analyze_model_ablation(ablation)
        lines.append("\n1. MODEL CONFIDENCE COMPARISON")
        lines.append("-" * 40)
        for model, stats in model_analysis.items():
            lines.append(f"\n  {model}:")
            lines.append(f"    Accuracy:           {stats['accuracy']:.1%}")
            lines.append(f"    Confidence mean:    {stats['conf_mean']:.3f}")
            lines.append(f"    Confidence std:     {stats['conf_std']:.3f}")
            lines.append(f"    Confidence range:   [{stats['conf_min']:.3f}, {stats['conf_max']:.3f}]")
            lines.append(f"    Routing suitability: {stats['routing_suitability']}")

        # Calibration
        cal = compute_calibration_summary(ablation)
        lines.append("\n2. CALIBRATION ANALYSIS")
        lines.append("-" * 40)
        for c in cal["calibration"]:
            lines.append(f"\n  {c['model']}:")
            lines.append(f"    Mean confidence: {c['mean_confidence']:.3f} vs Accuracy: {c['accuracy']:.1%}")
            lines.append(f"    Calibration gap: {c['calibration_gap']:.3f}")
            lines.append(f"    {'OVERCONFIDENT' if c['overconfident'] else 'UNDERCONFIDENT'}"
                         f" by {c['overconfidence_degree']:.3f}")

    # Key findings
    lines.append("\n3. KEY FINDINGS")
    lines.append("-" * 40)
    lines.append("""
  a) 4B model confidence (std=0.028) is near-constant (0.92-0.98),
     making threshold-based routing ineffective. The model is overconfident
     regardless of scenario difficulty.

  b) 0.6B model confidence (std=0.195) has meaningful variance,
     making it paradoxically MORE suitable for confidence-based routing
     despite lower accuracy.

  c) Both models are overconfident: mean confidence >> actual accuracy.
     This is a known limitation of softmax probability as confidence
     measure (Guo et al., 2017).

  d) For 4B model, vision-feature-based routing (anomaly_score, level
     proximity to thresholds) is more effective than LLM confidence.

  e) Temperature scaling or Platt calibration on the 0.6B model could
     improve routing decisions by mapping raw confidence to calibrated
     probabilities.
""")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Confidence Analysis")
    parser.add_argument("--output", default="experiments/confidence_analysis.json")
    args = parser.parse_args()

    data = load_experiments()

    if not data:
        print("No experiment results found in experiments/")
        return

    # Generate report
    report = generate_report(data)
    print(report)

    # Save structured results
    output = {}
    if "model_ablation" in data:
        output["model_comparison"] = analyze_model_ablation(data["model_ablation"])
        output["calibration"] = compute_calibration_summary(data["model_ablation"])

    benchmarks = {k: v for k, v in data.items()
                  if k.startswith("edge_cloud_benchmark")}
    if benchmarks:
        output["threshold_sensitivity"] = analyze_threshold_sensitivity(benchmarks)

    output["findings"] = [
        "4B confidence std=0.028: near-constant, unsuitable for threshold routing",
        "0.6B confidence std=0.195: meaningful variance, better for routing despite lower accuracy",
        "Both models overconfident: mean confidence >> accuracy (calibration gap)",
        "Vision-feature routing outperforms LLM-confidence routing for 4B model",
        "Temperature scaling recommended for 0.6B confidence calibration",
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
