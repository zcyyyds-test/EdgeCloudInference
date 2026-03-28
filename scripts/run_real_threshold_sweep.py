"""Real LLM threshold sweep: find optimal confidence threshold.

Two-phase strategy to minimize cloud calls:
  Phase 1: Run ALL scenarios through edge only, record edge result + confidence
  Phase 2: For each threshold, evaluate routing and only call cloud for
           scenarios that need escalation (confidence < threshold)

This avoids re-running edge inference for each threshold setting.

Prerequisites:
    1. Edge model server running with target model loaded
    2. Cloud server: python scripts/serve_cloud.py --model ... --port 8000 --gpu 1

Usage:
    python scripts/run_real_threshold_sweep.py [--scenarios 30]
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
from edgerouter.core.schema import Judgment, RoutingTier
from edgerouter.eval.workloads import build_small_workload
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


async def check_services(edge_url: str, cloud_url: str) -> tuple[bool, bool]:
    import httpx
    edge_ok = cloud_ok = False
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            r = await c.get(f"{edge_url}/api/tags")
            if r.status_code == 200:
                edge_ok = True
                print(f"  Edge: OK")
        except Exception as e:
            print(f"  Edge: FAILED ({e})")
        try:
            base = cloud_url.replace("/v1", "")
            r = await c.get(f"{base}/health")
            if r.status_code == 200:
                cloud_ok = True
                print(f"  Cloud: OK")
        except Exception as e:
            print(f"  Cloud: FAILED ({e})")
    return edge_ok, cloud_ok


async def main():
    parser = argparse.ArgumentParser(description="Real LLM Threshold Sweep")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--output", default="experiments/real_llm_threshold_sweep.json")
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-model", default="qwen3.5:0.8b")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3.5-27B")
    parser.add_argument("--wan-delay-ms", type=float, default=50.0)
    parser.add_argument("--thresholds", type=str,
                        default="0.3,0.5,0.6,0.7,0.8,0.9,0.95")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    print("=" * 70)
    print("Real LLM Threshold Sweep")
    print("=" * 70)
    print(f"  Edge:  {args.edge_model} @ {args.edge_url}")
    print(f"  Cloud: {args.cloud_model} @ {args.cloud_url}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Scenarios: {args.scenarios}")

    print("\nChecking services...")
    edge_ok, cloud_ok = await check_services(args.edge_url, args.cloud_url)
    if not edge_ok or not cloud_ok:
        print("\nERROR: Services not ready.")
        return

    # Setup
    edge_config = EdgeAnalyzerConfig(
        base_url=args.edge_url, model=args.edge_model, timeout=120.0,
    )
    cloud_config = CloudAnalyzerConfig(
        base_url=args.cloud_url, model=args.cloud_model, api_key="EMPTY",
    )
    edge = EdgeAnalyzer(edge_config)
    cloud = CloudAnalyzer(cloud_config)
    vision = VisionModel(seed=42)

    workload = build_small_workload(42)
    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]

    # ===================================================================
    # Phase 1: Run all scenarios through edge + routing
    # ===================================================================
    print(f"\nPhase 1: Running {len(workload.scenarios)} scenarios through edge...")

    edge_results = []  # list of {scenario, vision_output, edge_result, routing_decision}
    for i, scenario in enumerate(workload.scenarios):
        vision_output = vision.detect(scenario)

        # Get routing decision for each threshold later; for now just run edge
        try:
            edge_result = await edge.analyze(vision_output)
        except Exception as e:
            print(f"  [{i+1}] Edge ERROR: {e}")
            continue

        edge_results.append({
            "scenario": scenario,
            "vision_output": vision_output,
            "edge_result": edge_result,
            "edge_confidence": edge_result.confidence,
            "gt": scenario.ground_truth_judgment,
        })
        print(f"  [{i+1:3d}/{len(workload.scenarios)}] "
              f"edge={edge_result.judgment.value} conf={edge_result.confidence:.2f} "
              f"gt={scenario.ground_truth_judgment.value} "
              f"lat={edge_result.latency_ms:.0f}ms")

    print(f"\nPhase 1 complete: {len(edge_results)} scenarios processed")

    # ===================================================================
    # Phase 2: For each threshold, evaluate routing decisions
    # ===================================================================
    print(f"\nPhase 2: Sweeping {len(thresholds)} thresholds...")

    # Cache cloud results to avoid duplicate calls
    cloud_cache = {}  # scenario_index -> cloud_result

    sweep_points = []
    for t in thresholds:
        config = RouterConfig(confidence_threshold=t)
        router = RouterEngine(config)

        correct = 0
        false_neg = 0
        false_pos = 0
        edge_only = 0
        cloud_used = 0
        latencies = []

        for idx, item in enumerate(edge_results):
            scenario = item["scenario"]
            vo = item["vision_output"]
            er = item["edge_result"]
            gt = item["gt"]

            # Get routing decision with this threshold
            from edgerouter.core.schema import ProcessContext
            context = ProcessContext(
                scenario_id=f"sweep_{idx}",
                has_recipe_params=scenario.has_recipe_params,
                has_customer_info=scenario.has_customer_info,
                has_reaction_params=scenario.has_reaction_params,
                num_correlated_anomalies=scenario.num_correlated_anomalies,
            )
            decision = router.route(vo, context)

            # Determine if we need cloud
            need_cloud = False
            if decision.tier == RoutingTier.CLOUD:
                need_cloud = True
            elif decision.tier == RoutingTier.CASCADE:
                # Cascade: check combined confidence
                need_cloud = er.confidence < t
            elif decision.tier == RoutingTier.EDGE:
                # Post-execution check (the bug fix)
                need_cloud = er.confidence < t

            if need_cloud:
                # Call cloud (or use cache)
                if idx not in cloud_cache:
                    try:
                        cloud_result = await cloud.analyze(
                            vo, edge_draft=er,
                        )
                        cloud_cache[idx] = cloud_result
                    except Exception as e:
                        print(f"  Cloud ERROR at scenario {idx}: {e}")
                        cloud_cache[idx] = None

                cr = cloud_cache[idx]
                if cr:
                    predicted = cr.judgment
                    lat = er.latency_ms + cr.latency_ms + args.wan_delay_ms
                else:
                    predicted = er.judgment
                    lat = er.latency_ms
                cloud_used += 1
            else:
                predicted = er.judgment
                lat = er.latency_ms
                edge_only += 1

            latencies.append(lat)

            if predicted == gt:
                correct += 1
            elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
                false_neg += 1
            elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
                false_pos += 1

        total = len(edge_results)
        accuracy = correct / total if total > 0 else 0
        miss_rate = false_neg / max(1, sum(1 for r in edge_results if r["gt"] in (Judgment.WARNING, Judgment.ALARM)))
        false_alarm_rate = false_pos / max(1, sum(1 for r in edge_results if r["gt"] == Judgment.NORMAL))
        cloud_saving = edge_only / total if total > 0 else 0

        import numpy as np
        lat_arr = np.array(latencies)

        point = {
            "threshold": t,
            "accuracy": round(accuracy, 4),
            "miss_rate": round(miss_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "cloud_saving_rate": round(cloud_saving, 4),
            "edge_only": edge_only,
            "cloud_used": cloud_used,
            "p50_latency_ms": round(float(np.median(lat_arr)), 1),
            "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 1),
        }
        sweep_points.append(point)

        print(f"  t={t:.2f}: acc={accuracy:.3f} miss={miss_rate:.3f} "
              f"cloud_save={cloud_saving:.3f} edge={edge_only} cloud={cloud_used}")

    # Find optimal threshold (max cloud savings where miss_rate <= 2%)
    safe_points = [p for p in sweep_points if p["miss_rate"] <= 0.02]
    optimal = None
    if safe_points:
        optimal = max(safe_points, key=lambda p: p["cloud_saving_rate"])
        print(f"\n  Optimal threshold (miss_rate<=2%): {optimal['threshold']}")
    else:
        print("\n  No threshold achieves miss_rate <= 2% (model capability limit)")

    # Save results
    results = {
        "sweep_points": sweep_points,
        "optimal_threshold": optimal["threshold"] if optimal else None,
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": args.cloud_model,
            "scenarios": len(edge_results),
            "wan_delay_ms": args.wan_delay_ms,
            "thresholds": thresholds,
        },
        "cloud_calls_total": len(cloud_cache),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")

    await edge.close()
    await cloud.close()


if __name__ == "__main__":
    asyncio.run(main())
