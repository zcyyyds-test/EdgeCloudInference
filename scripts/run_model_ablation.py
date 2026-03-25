"""Model size ablation: compare edge model sizes with real LLM inference.

Tests multiple edge model sizes against the same cloud backend.

Usage:
    python scripts/run_model_ablation.py [--scenarios 30] [--models 0.6b,4b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgerouter.core.config import CloudAnalyzerConfig, EdgeAnalyzerConfig, RouterConfig
from edgerouter.core.schema import Judgment, RoutingTier
from edgerouter.eval.benchmarks import BenchmarkMetrics, BenchmarkRunner
from edgerouter.eval.workloads import build_small_workload
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


async def run_one_model(
    model_name: str,
    scenarios: int,
    edge_url: str,
    cloud_url: str,
    cloud_model: str,
) -> dict:
    """Run benchmark for a single edge model size."""
    print(f"\n{'─' * 50}")
    print(f"Testing edge model: {model_name}")
    print(f"{'─' * 50}")

    edge_config = EdgeAnalyzerConfig(
        base_url=edge_url, model=model_name, timeout=120.0,
    )
    cloud_config = CloudAnalyzerConfig(
        base_url=cloud_url, model=cloud_model, api_key="EMPTY",
    )
    edge = EdgeAnalyzer(edge_config)
    cloud = CloudAnalyzer(cloud_config)

    router_config = RouterConfig(confidence_threshold=0.7)
    router = RouterEngine(router_config)
    cascade = CascadeExecutor(edge, cloud, router_config)
    vision = VisionModel(seed=42)
    runner = BenchmarkRunner(router, cascade, vision)

    workload = build_small_workload(42)
    if scenarios < workload.size:
        workload.scenarios = workload.scenarios[:scenarios]

    metrics = BenchmarkMetrics()
    t_start = time.perf_counter()
    confidences = []

    for i, scenario in enumerate(workload.scenarios):
        vision_output = vision.detect(scenario)
        context = runner._scenario_to_context(scenario, idx=i)
        decision = router.route(vision_output, context)

        try:
            outcome = await cascade.execute(vision_output, context, decision)
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            continue

        outcome.ground_truth_judgment = scenario.ground_truth_judgment
        metrics.total_scenarios += 1
        metrics.outcomes.append(outcome)
        metrics.latencies.append(outcome.total_latency_ms)
        metrics.routing_overhead_ms.append(decision.latency_ms)

        gt = scenario.ground_truth_judgment
        predicted = outcome.final_judgment
        if predicted == gt:
            metrics.correct_judgments += 1
        elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
            metrics.false_negatives += 1
        elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
            metrics.false_positives += 1

        if decision.tier == RoutingTier.EDGE_EMERGENCY:
            metrics.emergency_count += 1
        elif decision.tier == RoutingTier.EDGE:
            metrics.edge_only_count += 1
        elif decision.tier == RoutingTier.CLOUD:
            metrics.cloud_count += 1
        elif decision.tier == RoutingTier.CASCADE:
            metrics.cascade_count += 1

        conf = outcome.edge_analysis.confidence if outcome.edge_analysis else 0
        confidences.append(conf)

        tier_str = decision.tier.value[:5]
        j_str = outcome.final_judgment.value[:4]
        gt_str = gt.value[:4]
        match = "✓" if predicted == gt else "✗"
        lat = outcome.total_latency_ms
        print(f"  [{i+1:3d}/{scenarios}] tier={tier_str:5s} pred={j_str} gt={gt_str} "
              f"{match} conf={conf:.2f} lat={lat:.0f}ms")

    elapsed = time.perf_counter() - t_start
    summary = metrics.summary()

    import numpy as np
    conf_arr = np.array(confidences)

    result = {
        "model": model_name,
        "summary": summary,
        "confidence_stats": {
            "mean": round(float(conf_arr.mean()), 3),
            "std": round(float(conf_arr.std()), 3),
            "min": round(float(conf_arr.min()), 3),
            "max": round(float(conf_arr.max()), 3),
        },
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"\n  Accuracy:     {summary['accuracy']:.3f}")
    print(f"  Miss rate:    {summary['miss_rate']:.3f}")
    print(f"  False alarm:  {summary['false_alarm_rate']:.3f}")
    print(f"  Cloud savings:{summary['cloud_saving_rate']:.3f}")
    print(f"  p50 latency:  {summary['p50_latency_ms']:.1f}ms")
    print(f"  Conf range:   [{conf_arr.min():.2f}, {conf_arr.max():.2f}] mean={conf_arr.mean():.2f}")
    print(f"  Time:         {elapsed:.1f}s")

    await edge.close()
    await cloud.close()

    return result


async def main():
    parser = argparse.ArgumentParser(description="Model Size Ablation")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--models", default="qwen3.5:0.8b,qwen3.5:4b",
                        help="Comma-separated edge model names")
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3.5-27B")
    parser.add_argument("--output", default="experiments/model_ablation.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    print("=" * 60)
    print("EdgeRouter Model Size Ablation")
    print("=" * 60)
    print(f"  Edge models: {models}")
    print(f"  Cloud model: {args.cloud_model}")
    print(f"  Scenarios:   {args.scenarios}")

    all_results = []
    for model in models:
        result = await run_one_model(
            model, args.scenarios,
            args.edge_url, args.cloud_url, args.cloud_model,
        )
        all_results.append(result)

    # Comparison table
    print(f"\n{'=' * 70}")
    print("Model Ablation Summary")
    print(f"{'=' * 70}")
    print(f"{'Model':15s} {'Acc':>6s} {'Miss':>6s} {'FAlarm':>6s} {'Save':>6s} "
          f"{'p50ms':>7s} {'ConfMean':>8s} {'ConfStd':>8s}")
    print("-" * 70)
    for r in all_results:
        s = r["summary"]
        c = r["confidence_stats"]
        print(f"{r['model']:15s} {s['accuracy']:6.3f} {s['miss_rate']:6.3f} "
              f"{s['false_alarm_rate']:6.3f} {s['cloud_saving_rate']:6.3f} "
              f"{s['p50_latency_ms']:7.1f} {c['mean']:8.3f} {c['std']:8.3f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
