"""WAN Latency sweep: measure impact of different network delays.

Runs the same scenarios with different WAN delays (10/50/200/500ms) to
show EdgeRouter's latency advantage over all-cloud under varying network conditions.

Prerequisites:
    1. Edge model server running with target model loaded
    2. Cloud server: python scripts/serve_cloud.py --model ... --port 8000 --gpu 1

Usage:
    python scripts/run_wan_latency_sweep.py [--scenarios 30]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgerouter.core.config import CloudAnalyzerConfig, EdgeAnalyzerConfig, RouterConfig
from edgerouter.core.schema import Judgment, ProcessContext, RoutingTier
from edgerouter.eval.workloads import build_small_workload
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


class DelayedEdgeAnalyzer(EdgeAnalyzer):
    """EdgeAnalyzer wrapper that adds WAN delay."""

    def __init__(self, config, wan_delay_ms: float = 0, jitter_ms: float = 0):
        super().__init__(config)
        self._wan_delay_ms = wan_delay_ms
        self._jitter_ms = jitter_ms

    async def analyze(self, *args, **kwargs):
        if self._wan_delay_ms > 0:
            delay = self._wan_delay_ms + random.gauss(0, self._jitter_ms)
            delay = max(0, delay) / 1000.0
            await asyncio.sleep(delay)
        return await super().analyze(*args, **kwargs)


async def run_one_delay(
    edge_config, cloud: CloudAnalyzer, workload, vision,
    wan_delay_ms: float, jitter_ms: float, threshold: float,
) -> dict:
    """Run benchmark with a specific WAN delay setting."""
    edge = DelayedEdgeAnalyzer(edge_config, wan_delay_ms=wan_delay_ms, jitter_ms=jitter_ms)
    router_config = RouterConfig(confidence_threshold=threshold)
    router = RouterEngine(router_config)
    cascade = CascadeExecutor(edge, cloud, router_config)

    correct = false_neg = false_pos = 0
    edge_only = cloud_count = cascade_count = emergency_count = 0
    latencies = []

    for i, scenario in enumerate(workload.scenarios):
        vo = vision.detect(scenario)
        ctx = ProcessContext(
            scenario_id=f"wan_{i}",
            has_recipe_params=scenario.has_recipe_params,
            has_customer_info=scenario.has_customer_info,
            has_reaction_params=scenario.has_reaction_params,
            num_correlated_anomalies=scenario.num_correlated_anomalies,
        )
        decision = router.route(vo, ctx)

        try:
            outcome = await cascade.execute(vo, ctx, decision)
        except Exception as e:
            print(f"    [{i+1}] ERROR: {e}")
            continue

        gt = scenario.ground_truth_judgment
        predicted = outcome.final_judgment
        latencies.append(outcome.total_latency_ms)

        if predicted == gt:
            correct += 1
        elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
            false_neg += 1
        elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
            false_pos += 1

        if decision.tier == RoutingTier.EDGE_EMERGENCY:
            emergency_count += 1
        elif decision.tier == RoutingTier.EDGE:
            edge_only += 1
        elif decision.tier == RoutingTier.CLOUD:
            cloud_count += 1
        elif decision.tier == RoutingTier.CASCADE:
            cascade_count += 1

    await edge.close()

    import numpy as np
    total = correct + false_neg + false_pos
    anomaly_total = sum(1 for s in workload.scenarios
                        if s.ground_truth_judgment in (Judgment.WARNING, Judgment.ALARM))
    normal_total = sum(1 for s in workload.scenarios
                       if s.ground_truth_judgment == Judgment.NORMAL)
    lat_arr = np.array(latencies) if latencies else np.array([0])

    return {
        "wan_delay_ms": wan_delay_ms,
        "jitter_ms": jitter_ms,
        "accuracy": round(correct / max(total, 1), 4),
        "miss_rate": round(false_neg / max(anomaly_total, 1), 4),
        "false_alarm_rate": round(false_pos / max(normal_total, 1), 4),
        "cloud_saving_rate": round((edge_only + emergency_count) / max(total, 1), 4),
        "p50_latency_ms": round(float(np.median(lat_arr)), 1),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 1),
        "edge_only": edge_only,
        "cloud_direct": cloud_count,
        "cascade": cascade_count,
        "emergency": emergency_count,
        "total": total,
    }


async def main():
    parser = argparse.ArgumentParser(description="WAN Latency Sweep")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--output", default="experiments/wan_latency_sweep.json")
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-model", default="qwen3.5:4b")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3.5-27B")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--delays", type=str, default="10,50,200,500",
                        help="WAN delays to test (ms, comma-separated)")
    args = parser.parse_args()

    delays = [float(d) for d in args.delays.split(",")]

    print("=" * 70)
    print("WAN Latency Sweep")
    print("=" * 70)
    print(f"  Edge:  {args.edge_model} @ {args.edge_url}")
    print(f"  Cloud: {args.cloud_model} @ {args.cloud_url}")
    print(f"  Delays: {delays} ms")
    print(f"  Threshold: {args.threshold}")
    print(f"  Scenarios: {args.scenarios}")

    edge_config = EdgeAnalyzerConfig(
        base_url=args.edge_url, model=args.edge_model, timeout=120.0,
    )
    cloud_config = CloudAnalyzerConfig(
        base_url=args.cloud_url, model=args.cloud_model, api_key="EMPTY",
    )
    cloud = CloudAnalyzer(cloud_config)
    vision = VisionModel(seed=42)

    workload = build_small_workload(42)
    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]

    results = []
    for delay in delays:
        jitter = delay * 0.2  # 20% jitter
        print(f"\n--- WAN delay: {delay}ms +/- {jitter:.0f}ms ---")
        point = await run_one_delay(
            edge_config, cloud, workload, vision,
            wan_delay_ms=delay, jitter_ms=jitter, threshold=args.threshold,
        )
        results.append(point)
        print(f"  acc={point['accuracy']:.3f} miss={point['miss_rate']:.3f} "
              f"p50={point['p50_latency_ms']:.0f}ms p99={point['p99_latency_ms']:.0f}ms")

    # Also compute "all-cloud" baseline latency for each delay
    print("\n--- All-Cloud baseline ---")
    for point in results:
        # All-cloud: every scenario goes through cloud with WAN delay
        all_cloud_p50 = point["p50_latency_ms"] * (
            point["total"] / max(point["cloud_direct"] + point["cascade"], 1)
        )
        point["all_cloud_p50_estimate_ms"] = round(all_cloud_p50, 0)

    output = {
        "sweep_points": results,
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": args.cloud_model,
            "scenarios": len(workload.scenarios),
            "threshold": args.threshold,
            "delays_ms": delays,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")

    await cloud.close()


if __name__ == "__main__":
    asyncio.run(main())
