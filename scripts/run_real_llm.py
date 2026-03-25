"""Run EdgeRouter benchmark with real LLMs (edge + cloud).

Prerequisites:
    1. Edge model serving runtime running with target model loaded.
    2. Cloud model server running:
       python scripts/serve_cloud.py --model Qwen/Qwen3.5-27B --port 8000

Usage:
    python scripts/run_real_llm.py [--scenarios 20] [--output real_llm_results.json]
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
from edgerouter.eval.workloads import build_small_workload, build_standard_workload
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


async def check_services() -> tuple[bool, bool]:
    """Check if edge and cloud model servers are reachable."""
    import httpx

    edge_ok = False
    cloud_ok = False

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get("http://127.0.0.1:11434/api/tags")
            edge_ok = r.status_code == 200
            models = r.json().get("models", [])
            print(f"  Edge: OK — {len(models)} model(s) loaded")
            for m in models:
                print(f"    - {m.get('name', '?')} ({m.get('size', '?')} bytes)")
        except Exception as e:
            print(f"  Edge: FAILED — {e}")

        try:
            r = await client.get("http://127.0.0.1:8000/health")
            cloud_ok = r.status_code == 200
            info = r.json()
            print(f"  Cloud server: OK — model={info.get('model', '?')}")
        except Exception as e:
            print(f"  Cloud server: FAILED — {e}")

    return edge_ok, cloud_ok


async def main():
    parser = argparse.ArgumentParser(description="EdgeRouter Real LLM Benchmark")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of scenarios")
    parser.add_argument("--output", default="real_llm_results.json")
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-model", default="qwen3.5:4b")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3.5-27B")
    args = parser.parse_args()

    print("=" * 70)
    print("EdgeRouter Real LLM Benchmark")
    print("=" * 70)

    # Check services
    print("\nChecking services...")
    edge_ok, cloud_ok = await check_services()
    if not edge_ok:
        print("\nERROR: Edge model server not running.")
        return
    if not cloud_ok:
        print("\nERROR: Cloud server not running. Start with:")
        print("  python scripts/serve_cloud.py --model Qwen/Qwen3.5-27B --port 8000")
        return

    # Create real analyzers
    edge_config = EdgeAnalyzerConfig(base_url=args.edge_url, model=args.edge_model)
    cloud_config = CloudAnalyzerConfig(
        base_url=args.cloud_url, model=args.cloud_model, api_key="EMPTY",
    )

    edge = EdgeAnalyzer(edge_config)
    cloud = CloudAnalyzer(cloud_config)

    # Verify analyzers
    print("\nHealth check...")
    if await edge.health_check():
        print("  Edge analyzer: OK")
    else:
        print("  Edge analyzer: FAILED")
        return

    if await cloud.health_check():
        print("  Cloud analyzer: OK")
    else:
        print("  Cloud analyzer: FAILED")
        return

    # Build workload
    seed = 42
    vision = VisionModel(seed=seed)
    workload = build_small_workload(seed)

    # Trim to requested size
    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]
    print(f"\nWorkload: {args.scenarios} scenarios")

    # Run benchmark with real LLMs
    router_config = RouterConfig(confidence_threshold=0.7)
    router = RouterEngine(router_config)
    cascade = CascadeExecutor(edge, cloud, router_config)
    runner = BenchmarkRunner(router, cascade, vision)

    t_start = time.perf_counter()
    print("\nRunning benchmark with real LLMs...")
    print("(This will be slow — each scenario requires LLM inference)\n")

    metrics = BenchmarkMetrics()
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

        tier_str = decision.tier.value[:5]
        j_str = outcome.final_judgment.value[:4]
        gt_str = gt.value[:4]
        match = "✓" if predicted == gt else "✗"
        lat = outcome.total_latency_ms
        conf = outcome.edge_analysis.confidence if outcome.edge_analysis else outcome.edge_confidence
        print(f"  [{i+1:3d}/{args.scenarios}] tier={tier_str:5s} pred={j_str} gt={gt_str} {match} conf={conf:.2f} lat={lat:.0f}ms")

    elapsed = time.perf_counter() - t_start
    summary = metrics.summary()

    print(f"\n{'=' * 70}")
    print("Results (Real LLM)")
    print(f"{'=' * 70}")
    print(f"  Scenarios:     {summary['total_scenarios']}")
    print(f"  Accuracy:      {summary['accuracy']:.3f}")
    print(f"  Miss rate:     {summary['miss_rate']:.3f}")
    print(f"  False alarm:   {summary['false_alarm_rate']:.3f}")
    print(f"  Cloud savings: {summary['cloud_saving_rate']:.3f}")
    print(f"  Upgrade rate:  {summary['upgrade_rate']:.3f}")
    print(f"  p50 latency:   {summary['p50_latency_ms']:.1f}ms")
    print(f"  p99 latency:   {summary['p99_latency_ms']:.1f}ms")
    print(f"  Edge only:     {summary['edge_only']}")
    print(f"  Cloud direct:  {summary['cloud_direct']}")
    print(f"  Cascade:       {summary['cascade']}")
    print(f"  Emergency:     {summary['emergency']}")
    print(f"  Total time:    {elapsed:.1f}s ({elapsed/max(1,metrics.total_scenarios):.1f}s/scenario)")
    print(f"{'=' * 70}")

    # Save
    results = {
        "summary": summary,
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": args.cloud_model,
            "scenarios": args.scenarios,
            "confidence_threshold": 0.7,
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
