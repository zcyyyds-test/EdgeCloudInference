"""Edge-Cloud benchmark: 0.6B edge (Ollama) + 14B cloud (serve_cloud.py).

Demonstrates real edge-cloud routing with different model sizes:
  - Edge: Ollama qwen3:0.6b (lightweight, fast, lower accuracy)
  - Cloud: Qwen3-14B via serve_cloud.py (heavy, slower, higher accuracy)
  - Optional: WAN delay on edge requests

Resource isolation note:
    On the Intel server, edge deployment uses resource-constrained inference.
    WSL2 limitations (no netem, process non-persistence, sudo requirements)
    made this impractical. The key differentiator — model quality and routing
    logic — is demonstrated here without physical resource isolation.

Prerequisites:
    1. Ollama running with qwen3:0.6b loaded
    2. Cloud server: python scripts/serve_cloud.py --model ... --port 8000 --gpu 1

Usage:
    python scripts/run_edge_cloud_benchmark.py [--scenarios 30] [--wan-delay-ms 50]
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
from edgerouter.eval.benchmarks import BenchmarkMetrics, BenchmarkRunner
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
        # Add WAN delay before each edge call (network hop)
        if self._wan_delay_ms > 0:
            delay = self._wan_delay_ms + random.gauss(0, self._jitter_ms)
            delay = max(0, delay) / 1000.0  # ms → seconds
            await asyncio.sleep(delay)
        return await super().analyze(*args, **kwargs)


async def check_services(edge_url: str, cloud_url: str) -> tuple[bool, bool]:
    import httpx

    edge_ok = False
    cloud_ok = False

    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            r = await c.get(f"{edge_url}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                print(f"  Edge (Ollama):  OK — {len(models)} model(s)")
                for m in models:
                    print(f"    - {m.get('name', '?')}")
                edge_ok = True
        except Exception as e:
            print(f"  Edge (Ollama):  FAILED — {e}")

        try:
            # Try vLLM health endpoint first (returns empty 200), then /v1/models
            base = cloud_url.replace("/v1", "")
            r = await c.get(f"{base}/health")
            if r.status_code == 200:
                # vLLM health returns empty body; get model info from /v1/models
                r2 = await c.get(f"{cloud_url}/models")
                if r2.status_code == 200:
                    models = r2.json().get("data", [])
                    model_name = models[0]["id"] if models else "?"
                    print(f"  Cloud (GPU):   OK — model={model_name}")
                else:
                    print(f"  Cloud (GPU):   OK — health check passed")
                cloud_ok = True
        except Exception as e:
            print(f"  Cloud (GPU):   FAILED — {e}")

    return edge_ok, cloud_ok


async def measure_rtt(url: str, n: int = 5) -> float:
    import httpx
    times = []
    async with httpx.AsyncClient(timeout=10.0) as c:
        for _ in range(n):
            t0 = time.perf_counter()
            await c.get(f"{url}/api/tags")
            times.append((time.perf_counter() - t0) * 1000)
    return float(sum(times) / len(times))


async def main():
    parser = argparse.ArgumentParser(description="Edge-Cloud Benchmark")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--output", default="experiments/edge_cloud_benchmark.json")
    parser.add_argument("--edge-url", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-model", default="qwen3:0.6b")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cloud-model", default="Qwen3-14B")
    parser.add_argument("--wan-delay-ms", type=float, default=50.0,
                        help="WAN delay for edge requests (ms)")
    parser.add_argument("--wan-jitter-ms", type=float, default=10.0,
                        help="Jitter for WAN delay (ms, normal distribution)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--num-gpu", type=int, default=-1,
                        help="Ollama GPU layers: -1=auto, 0=CPU-only")
    args = parser.parse_args()

    print("=" * 70)
    print("EdgeRouter Edge-Cloud Benchmark")
    print("=" * 70)
    gpu_mode = "CPU-only" if args.num_gpu == 0 else f"GPU layers={args.num_gpu}" if args.num_gpu > 0 else "GPU auto"
    print(f"  Edge:  {args.edge_url} ({args.edge_model}, {gpu_mode})")
    print(f"  Cloud: {args.cloud_url} ({args.cloud_model})")
    print(f"  WAN delay: {args.wan_delay_ms}ms ± {args.wan_jitter_ms}ms")
    print(f"  Confidence threshold: {args.confidence_threshold}")

    print("\nChecking services...")
    edge_ok, cloud_ok = await check_services(args.edge_url, args.cloud_url)
    if not edge_ok or not cloud_ok:
        print("\nERROR: Services not ready.")
        return

    # Measure baseline RTT
    rtt = await measure_rtt(args.edge_url)
    print(f"\n  Baseline edge RTT: {rtt:.1f}ms")
    print(f"  Effective edge RTT: {rtt + args.wan_delay_ms:.1f}ms (with WAN delay)")

    # Setup edge with delay
    edge_config = EdgeAnalyzerConfig(
        base_url=args.edge_url, model=args.edge_model, timeout=120.0,
        num_gpu=args.num_gpu,
    )
    cloud_config = CloudAnalyzerConfig(
        base_url=args.cloud_url, model=args.cloud_model, api_key="EMPTY",
    )
    edge = DelayedEdgeAnalyzer(
        edge_config, wan_delay_ms=args.wan_delay_ms, jitter_ms=args.wan_jitter_ms,
    )
    cloud = CloudAnalyzer(cloud_config)

    router_config = RouterConfig(confidence_threshold=args.confidence_threshold)
    router = RouterEngine(router_config)
    cascade = CascadeExecutor(edge, cloud, router_config)

    vision = VisionModel(seed=42)
    runner = BenchmarkRunner(router, cascade, vision)

    workload = build_small_workload(42)
    if args.scenarios < workload.size:
        workload.scenarios = workload.scenarios[:args.scenarios]

    print(f"\nWorkload: {len(workload.scenarios)} scenarios")
    print("Running benchmark...\n")

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

        conf = outcome.edge_analysis.confidence if outcome.edge_analysis else 0
        confidences.append(conf)

        tier_str = decision.tier.value[:5]
        j_str = outcome.final_judgment.value[:4]
        gt_str = gt.value[:4]
        match = "✓" if predicted == gt else "✗"
        lat = outcome.total_latency_ms
        print(f"  [{i+1:3d}/{len(workload.scenarios)}] tier={tier_str:5s} "
              f"pred={j_str} gt={gt_str} {match} conf={conf:.2f} lat={lat:.0f}ms")

    elapsed = time.perf_counter() - t_start
    summary = metrics.summary()

    import numpy as np
    conf_arr = np.array(confidences) if confidences else np.array([0])

    print(f"\n{'=' * 70}")
    print(f"Results (Edge {args.edge_model} + Cloud {args.cloud_model})")
    print(f"{'=' * 70}")
    print(f"  Scenarios:     {summary['total_scenarios']}")
    print(f"  Accuracy:      {summary['accuracy']:.3f}")
    print(f"  Miss rate:     {summary['miss_rate']:.3f}")
    print(f"  False alarm:   {summary['false_alarm_rate']:.3f}")
    print(f"  Cloud savings: {summary['cloud_saving_rate']:.3f}")
    print(f"  Upgrade rate:  {summary['upgrade_rate']:.3f}")
    print(f"  p50 latency:   {summary['p50_latency_ms']:.1f}ms")
    print(f"  p99 latency:   {summary['p99_latency_ms']:.1f}ms")
    print(f"  Baseline RTT:  {rtt:.1f}ms")
    print(f"  WAN delay:     {args.wan_delay_ms}ms ± {args.wan_jitter_ms}ms")
    print(f"  Edge only:     {summary['edge_only']}")
    print(f"  Cloud direct:  {summary['cloud_direct']}")
    print(f"  Cascade:       {summary['cascade']}")
    print(f"  Emergency:     {summary['emergency']}")
    print(f"  Conf mean/std: {conf_arr.mean():.3f} / {conf_arr.std():.3f}")
    print(f"  Total time:    {elapsed:.1f}s "
          f"({elapsed/max(1,metrics.total_scenarios):.1f}s/scenario)")
    print(f"{'=' * 70}")

    results = {
        "summary": summary,
        "config": {
            "edge_model": args.edge_model,
            "cloud_model": args.cloud_model,
            "edge_url": args.edge_url,
            "cloud_url": args.cloud_url,
            "scenarios": len(workload.scenarios),
            "wan_delay_ms": args.wan_delay_ms,
            "wan_jitter_ms": args.wan_jitter_ms,
            "confidence_threshold": args.confidence_threshold,
            "baseline_rtt_ms": round(rtt, 1),
            "edge_num_gpu": args.num_gpu,
            "deployment": f"ollama_{args.edge_model}_edge({gpu_mode}) + {args.cloud_model}_cloud",
        },
        "confidence_stats": {
            "mean": round(float(conf_arr.mean()), 3),
            "std": round(float(conf_arr.std()), 3),
            "min": round(float(conf_arr.min()), 3),
            "max": round(float(conf_arr.max()), 3),
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {args.output}")

    await edge.close()
    await cloud.close()


if __name__ == "__main__":
    asyncio.run(main())
