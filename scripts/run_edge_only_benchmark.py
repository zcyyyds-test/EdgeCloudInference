"""Edge-only benchmark: test 0.8B routing model without cloud.

Runs all scenarios through the 5-tier router + edge LLM only.
For CASCADE scenarios, accepts the edge result directly.
Reports: routing distribution, edge accuracy, confidence stats, latency.

Usage:
    python scripts/run_edge_only_benchmark.py [--scenarios 30]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgerouter.core.config import EdgeAnalyzerConfig, RouterConfig
from edgerouter.core.schema import Judgment, ProcessContext, RoutingTier
from edgerouter.eval.workloads import build_small_workload
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


async def main():
    parser = argparse.ArgumentParser(description="Edge-Only Benchmark (no cloud)")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-model", default="qwen3.5:0.8b")
    parser.add_argument("--output", default="experiments/edge_only_08b_benchmark.json")
    args = parser.parse_args()

    print("=" * 70)
    print("EdgeRouter Edge-Only Benchmark (0.8B, no cloud)")
    print("=" * 70)

    # Setup
    edge_config = EdgeAnalyzerConfig(
        base_url=args.edge_url, model=args.edge_model, timeout=120.0,
    )
    edge = EdgeAnalyzer(edge_config)

    print("\nHealth check...")
    if await edge.health_check():
        print(f"  Edge ({args.edge_model}): OK")
    else:
        print(f"  Edge ({args.edge_model}): FAILED — is Ollama running?")
        return

    router_config = RouterConfig(confidence_threshold=0.7)
    router = RouterEngine(router_config)
    vision = VisionModel(seed=42)
    workload = build_small_workload(42)

    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]

    print(f"  Scenarios: {len(workload.scenarios)}")
    print(f"  Model: {args.edge_model}")
    print()

    # Counters
    tier_counts = {t: 0 for t in RoutingTier}
    correct = 0
    false_pos = 0
    false_neg = 0
    total = 0
    latencies = []
    confidences = []
    routing_overheads = []
    per_scenario = []

    t_total = time.perf_counter()

    for i, scenario in enumerate(workload.scenarios):
        vo = vision.detect(scenario)
        ctx = ProcessContext(num_correlated_anomalies=0)
        decision = router.route(vo, ctx)
        tier_counts[decision.tier] += 1
        routing_overheads.append(decision.latency_ms)

        gt = scenario.ground_truth_judgment

        # Tiers that don't need LLM
        if decision.tier == RoutingTier.EDGE_EMERGENCY:
            predicted = Judgment.ALARM
            conf = 1.0
            lat = decision.latency_ms
        elif decision.tier == RoutingTier.EDGE and "clearly_normal" in decision.reason:
            predicted = Judgment.NORMAL
            conf = vo.anomaly_confidence
            lat = decision.latency_ms
        else:
            # EDGE, CASCADE, CLOUD — all go through edge LLM
            t_start = time.perf_counter()
            try:
                result = await edge.analyze(vo)
                predicted = result.judgment
                conf = result.confidence
                lat = (time.perf_counter() - t_start) * 1000
            except Exception as e:
                print(f"  [{i+1:3d}] ERROR: {e}")
                continue

        total += 1
        latencies.append(lat)
        confidences.append(conf)

        if predicted == gt:
            correct += 1
        elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
            false_neg += 1
        elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
            false_pos += 1

        match = "✓" if predicted == gt else "✗"
        tier_str = decision.tier.value[:5]
        print(f"  [{i+1:3d}/{len(workload.scenarios)}] tier={tier_str:8s} "
              f"pred={predicted.value[:4]:4s} gt={gt.value[:4]:4s} "
              f"{match} conf={conf:.2f} lat={lat:.0f}ms")

        per_scenario.append({
            "idx": i + 1,
            "name": scenario.name,
            "tier": decision.tier.value,
            "reason": decision.reason,
            "predicted": predicted.value,
            "ground_truth": gt.value,
            "correct": predicted == gt,
            "confidence": round(conf, 3),
            "latency_ms": round(lat, 1),
        })

    elapsed = time.perf_counter() - t_total

    # Stats
    lat_arr = np.array(latencies)
    conf_arr = np.array(confidences)
    accuracy = correct / max(1, total)
    miss_rate = false_neg / max(1, sum(1 for s in workload.scenarios
                                        if s.ground_truth_judgment in (Judgment.WARNING, Judgment.ALARM)))
    false_alarm_rate = false_pos / max(1, sum(1 for s in workload.scenarios
                                               if s.ground_truth_judgment == Judgment.NORMAL))

    # Routing distribution
    edge_only = tier_counts[RoutingTier.EDGE] + tier_counts[RoutingTier.EDGE_EMERGENCY]
    cloud_direct = tier_counts[RoutingTier.CLOUD]
    cascade = tier_counts[RoutingTier.CASCADE]

    print(f"\n{'=' * 70}")
    print("Results (Edge-Only, 0.8B)")
    print(f"{'=' * 70}")
    print(f"  Scenarios:        {total}")
    print(f"  Accuracy:         {accuracy:.3f} ({correct}/{total})")
    print(f"  Miss rate:        {miss_rate:.3f}")
    print(f"  False alarm rate: {false_alarm_rate:.3f}")
    print()
    print(f"  Routing distribution:")
    print(f"    Emergency:   {tier_counts[RoutingTier.EDGE_EMERGENCY]}")
    print(f"    Edge:        {tier_counts[RoutingTier.EDGE]}")
    print(f"    Cloud:       {tier_counts[RoutingTier.CLOUD]}")
    print(f"    Cascade:     {tier_counts[RoutingTier.CASCADE]}")
    print(f"    Cloud saving: {(edge_only + cascade) / max(1, total) * 100:.1f}%")
    print()
    print(f"  Latency (ms):")
    print(f"    p50:  {np.percentile(lat_arr, 50):.0f}")
    print(f"    p95:  {np.percentile(lat_arr, 95):.0f}")
    print(f"    p99:  {np.percentile(lat_arr, 99):.0f}")
    print(f"    mean: {lat_arr.mean():.0f}")
    print()
    print(f"  Confidence:")
    print(f"    mean: {conf_arr.mean():.3f}")
    print(f"    std:  {conf_arr.std():.3f}")
    print(f"    min:  {conf_arr.min():.3f}")
    print(f"    max:  {conf_arr.max():.3f}")
    print()
    print(f"  Routing overhead: {np.mean(routing_overheads):.3f}ms")
    print(f"  Total time:       {elapsed:.1f}s ({elapsed/max(1,total):.1f}s/scenario)")
    print(f"{'=' * 70}")

    # Save
    results = {
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": "none (edge-only)",
            "scenarios": total,
            "confidence_threshold": router_config.confidence_threshold,
        },
        "summary": {
            "accuracy": round(accuracy, 3),
            "miss_rate": round(miss_rate, 3),
            "false_alarm_rate": round(false_alarm_rate, 3),
            "correct": correct,
            "false_positives": false_pos,
            "false_negatives": false_neg,
        },
        "routing": {
            "emergency": tier_counts[RoutingTier.EDGE_EMERGENCY],
            "edge": tier_counts[RoutingTier.EDGE],
            "cloud": tier_counts[RoutingTier.CLOUD],
            "cascade": tier_counts[RoutingTier.CASCADE],
            "cloud_saving_pct": round((edge_only + cascade) / max(1, total) * 100, 1),
        },
        "latency": {
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 1),
            "mean_ms": round(float(lat_arr.mean()), 1),
        },
        "confidence": {
            "mean": round(float(conf_arr.mean()), 3),
            "std": round(float(conf_arr.std()), 3),
            "min": round(float(conf_arr.min()), 3),
            "max": round(float(conf_arr.max()), 3),
        },
        "routing_overhead_ms": round(float(np.mean(routing_overheads)), 3),
        "elapsed_seconds": round(elapsed, 2),
        "per_scenario": per_scenario,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")

    await edge.close()


if __name__ == "__main__":
    asyncio.run(main())
