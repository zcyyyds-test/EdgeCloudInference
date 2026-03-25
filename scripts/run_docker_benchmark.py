"""Benchmark with Docker edge (0.6B CPU) + native cloud (14B GPU).

This script demonstrates real edge-cloud separation:
  - Edge: Docker container (2 CPU, 2GB RAM, no GPU) running Ollama qwen3:0.6b
  - Cloud: Native GPU process running Qwen3-14B via serve_cloud.py
  - Network: Docker bridge + optional tc netem WAN delay

Prerequisites:
    1. Docker edge container running:
       docker compose -f docker/docker-compose.yml up -d
    2. Cloud server running:
       python scripts/serve_cloud.py --model D:/zcy/models/Qwen/Qwen3-14B --port 8000 --gpu 1

Usage:
    python scripts/run_docker_benchmark.py [--scenarios 20] [--edge-url http://172.20.0.2:11434]
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
from edgerouter.eval.benchmarks import BenchmarkMetrics
from edgerouter.eval.workloads import build_small_workload
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.vision import VisionModel


async def check_edge(url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{url}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                print(f"  Edge (Docker): OK — {len(models)} model(s)")
                for m in models:
                    print(f"    - {m.get('name', '?')}")
                return True
    except Exception as e:
        print(f"  Edge (Docker): FAILED — {e}")
    return False


async def check_cloud(url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            base = url.replace("/v1", "")
            r = await c.get(f"{base}/health")
            if r.status_code == 200:
                info = r.json()
                print(f"  Cloud (Native GPU): OK — model={info.get('model', '?')}")
                return True
    except Exception as e:
        print(f"  Cloud (Native GPU): FAILED — {e}")
    return False


async def measure_rtt(edge_url: str) -> float:
    """Measure round-trip time to edge container."""
    import httpx
    times = []
    async with httpx.AsyncClient(timeout=10.0) as c:
        for _ in range(5):
            t0 = time.perf_counter()
            await c.get(f"{edge_url}/api/tags")
            times.append((time.perf_counter() - t0) * 1000)
    return float(sum(times) / len(times))


async def main():
    parser = argparse.ArgumentParser(description="Docker Edge-Cloud Benchmark")
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--output", default="experiments/docker_benchmark.json")
    parser.add_argument("--edge-url", default="http://172.20.0.2:11434",
                        help="Docker edge Ollama URL")
    parser.add_argument("--edge-model", default="qwen3:0.6b")
    parser.add_argument("--cloud-url", default="http://localhost:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3-14B")
    args = parser.parse_args()

    print("=" * 70)
    print("EdgeRouter Docker Edge-Cloud Benchmark")
    print("=" * 70)
    print(f"  Edge: {args.edge_url} ({args.edge_model}) — Docker, CPU-only")
    print(f"  Cloud: {args.cloud_url} ({args.cloud_model}) — Native GPU")

    print("\nChecking services...")
    edge_ok = await check_edge(args.edge_url)
    cloud_ok = await check_cloud(args.cloud_url)
    if not edge_ok or not cloud_ok:
        print("\nERROR: Services not ready. See prerequisites above.")
        return

    # Measure network RTT
    rtt = await measure_rtt(args.edge_url)
    print(f"\n  Edge RTT: {rtt:.1f}ms (avg of 5 pings)")

    # Setup
    edge_config = EdgeAnalyzerConfig(
        base_url=args.edge_url, model=args.edge_model, timeout=120.0,
    )
    cloud_config = CloudAnalyzerConfig(
        base_url=args.cloud_url, model=args.cloud_model, api_key="EMPTY",
    )
    edge = EdgeAnalyzer(edge_config)
    cloud = CloudAnalyzer(cloud_config)

    router_config = RouterConfig(confidence_threshold=0.7)
    router = RouterEngine(router_config)
    cascade = CascadeExecutor(edge, cloud, router_config)

    vision = VisionModel(seed=42)
    workload = build_small_workload(42)
    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]

    print(f"\nWorkload: {args.scenarios} scenarios")
    print("Running benchmark...\n")

    from edgerouter.eval.benchmarks import BenchmarkRunner
    runner = BenchmarkRunner(router, cascade, vision)
    metrics = BenchmarkMetrics()
    t_start = time.perf_counter()

    for i, scenario in enumerate(workload.scenarios):
        vision_output = vision.detect(scenario)
        context = runner._scenario_to_context(scenario, idx=i)
        decision = router.route(vision_output, context)

        try:
            outcome = await cascade.execute(vision_output, context, decision)
        except Exception as e:
            print(f"  [{i+1:3d}] ERROR: {e}")
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
        conf = outcome.edge_analysis.confidence if outcome.edge_analysis else 0
        lat = outcome.total_latency_ms
        print(f"  [{i+1:3d}/{args.scenarios}] tier={tier_str:5s} pred={j_str} gt={gt_str} "
              f"{match} conf={conf:.2f} lat={lat:.0f}ms")

    elapsed = time.perf_counter() - t_start
    summary = metrics.summary()

    print(f"\n{'=' * 70}")
    print("Results (Docker Edge + Native Cloud)")
    print(f"{'=' * 70}")
    print(f"  Scenarios:     {summary['total_scenarios']}")
    print(f"  Accuracy:      {summary['accuracy']:.3f}")
    print(f"  Miss rate:     {summary['miss_rate']:.3f}")
    print(f"  False alarm:   {summary['false_alarm_rate']:.3f}")
    print(f"  Cloud savings: {summary['cloud_saving_rate']:.3f}")
    print(f"  Upgrade rate:  {summary['upgrade_rate']:.3f}")
    print(f"  p50 latency:   {summary['p50_latency_ms']:.1f}ms")
    print(f"  p99 latency:   {summary['p99_latency_ms']:.1f}ms")
    print(f"  Edge RTT:      {rtt:.1f}ms")
    print(f"  Edge only:     {summary['edge_only']}")
    print(f"  Cloud direct:  {summary['cloud_direct']}")
    print(f"  Cascade:       {summary['cascade']}")
    print(f"  Emergency:     {summary['emergency']}")
    print(f"  Total time:    {elapsed:.1f}s ({elapsed/max(1,metrics.total_scenarios):.1f}s/scenario)")
    print(f"{'=' * 70}")

    results = {
        "summary": summary,
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": args.cloud_model,
            "edge_url": args.edge_url,
            "cloud_url": args.cloud_url,
            "scenarios": args.scenarios,
            "edge_rtt_ms": round(rtt, 1),
            "deployment": "docker_edge + native_cloud",
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
