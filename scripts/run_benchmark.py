"""Run the full EdgeRouter benchmark suite (offline, no LLM needed).

Usage:
    python scripts/run_benchmark.py [--output results.json] [--quick]

Implements all 6 experiments from the technical proposal:
  1. Configuration Comparison (All-Edge/All-Cloud/Static/Random-Split/EdgeRouter)
  2. Confidence Method Comparison (output_prob/self_verify/temporal/combined)
  3. Edge Model Size Ablation (0.6B/1.7B/4B/8B)
  4. WAN Latency Sweep (10/50/200/500ms)
  5. Online Learning Convergence (1000 scenarios)
  6. Data Security Compliance (50 sensitive + 550 non-sensitive)
Plus: Confidence Threshold Sweep (Pareto curve)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgerouter.core.config import RouterConfig
from edgerouter.core.schema import Judgment, RoutingDecision, RoutingTier
from edgerouter.eval.analysis import AnalysisReport
from edgerouter.eval.benchmarks import BenchmarkMetrics, BenchmarkRunner
from edgerouter.eval.workloads import (
    build_extended_workload,
    build_security_workload,
    build_small_workload,
    build_standard_workload,
)
from edgerouter.inference.mock import (
    MockCloudAnalyzer,
    MockEdgeAnalyzer,
    SizedMockEdgeAnalyzer,
    WANDelayCloudAnalyzer,
)
from edgerouter.learning.feedback import FeedbackCollector
from edgerouter.learning.online_learner import OnlineRouterLearner
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.router.prefetch import PredictivePrefetcher
from edgerouter.scenarios.vision import VisionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_single(
    config: RouterConfig,
    workload,
    vision: VisionModel,
    name: str,
    edge_analyzer=None,
    cloud_analyzer=None,
) -> tuple[dict, BenchmarkMetrics]:
    """Run one benchmark configuration and return (summary_dict, metrics)."""
    router = RouterEngine(config)
    edge = edge_analyzer or MockEdgeAnalyzer()
    cloud = cloud_analyzer or MockCloudAnalyzer()
    cascade = CascadeExecutor(edge, cloud, config)
    runner = BenchmarkRunner(router, cascade, vision)
    metrics = await runner.run(workload)
    summary = metrics.summary()
    summary["config"] = name
    return summary, metrics


def print_row(name: str, s: dict) -> None:
    """Print a compact result row."""
    print(f"  {name:20s}: acc={s['accuracy']:.3f}  miss={s['miss_rate']:.3f}  "
          f"cloud_save={s['cloud_saving_rate']:.3f}  upgrade={s['upgrade_rate']:.3f}  "
          f"sec={s['data_security_compliance']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="EdgeRouter Benchmark Suite")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--quick", action="store_true", help="Use smaller workloads")
    args = parser.parse_args()

    t_total = time.perf_counter()
    seed = 42
    vision = VisionModel(seed=seed)
    results: dict = {}

    print("=" * 70)
    print("EdgeRouter Benchmark Suite — Full 6-Experiment Evaluation")
    print("=" * 70)

    # Build workloads
    if args.quick:
        workload = build_small_workload(seed)
        ext_workload = build_small_workload(seed + 1)
        sec_workload = build_small_workload(seed + 2)
    else:
        workload = build_standard_workload(seed)
        ext_workload = build_extended_workload(seed)
        sec_workload = build_security_workload(seed)

    print(f"\nPrimary workload: {workload.name} ({workload.size} scenarios)")

    report = AnalysisReport()

    # ===================================================================
    # Experiment 1: Configuration Comparison
    #   All-Edge / All-Cloud / Static-0.5 / Random-Split-40% / EdgeRouter-0.7
    # ===================================================================
    print("\n" + "=" * 70)
    print("Experiment 1: Configuration Comparison")
    print("=" * 70)

    configs = {
        "All-Edge": RouterConfig(
            confidence_threshold=0.0,
            clearly_normal_anomaly_max=999,
            clearly_normal_confidence_min=-1,
        ),
        "All-Cloud": RouterConfig(
            confidence_threshold=1.0,
            clearly_normal_anomaly_max=-1,
            clearly_normal_confidence_min=999,
        ),
        "Static-0.5": RouterConfig(confidence_threshold=0.5),
        "Random-40%": RouterConfig(confidence_threshold=0.7),  # will be overridden below
        "EdgeRouter-0.7": RouterConfig(confidence_threshold=0.7),
    }

    exp1_results = []
    for name, cfg in configs.items():
        if name == "Random-40%":
            # Random baseline: randomly route 40% to cloud
            # Implement by running all through edge, then randomly escalating 40%
            summary, metrics = await _run_random_split(
                cfg, workload, vision, cloud_rate=0.4, seed=seed,
            )
            summary["config"] = name
        else:
            summary, metrics = await run_single(cfg, workload, vision, name)
        report.add_comparison(name, metrics)
        print_row(name, summary)
        exp1_results.append(summary)

    results["exp1_comparison"] = exp1_results

    # ===================================================================
    # Experiment 1b: Confidence Threshold Sweep (Pareto curve)
    # ===================================================================
    print("\n" + "=" * 70)
    print("Experiment 1b: Confidence Threshold Sweep")
    print("=" * 70)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for t in thresholds:
        cfg = RouterConfig(confidence_threshold=t)
        summary, metrics = await run_single(cfg, workload, vision, f"thresh={t}")
        report.add_sweep_point(t, metrics)
        print(f"  threshold={t:.2f}: acc={summary['accuracy']:.3f}  "
              f"miss={summary['miss_rate']:.3f}  upgrade={summary['upgrade_rate']:.3f}")

    optimal = report.find_optimal_threshold(max_miss_rate=0.02)
    print(f"\n  Optimal threshold (miss_rate <= 2%): {optimal}")
    results["exp1b_sweep"] = report.sweep_table()
    results["exp1b_optimal_threshold"] = optimal

    # ===================================================================
    # Experiment 2: Confidence Method Comparison
    #   Output Probability / Self-Verification / Temporal / Combined
    # ===================================================================
    print("\n" + "=" * 70)
    print("Experiment 2: Confidence Method Comparison")
    print("=" * 70)

    methods = ["output_prob", "self_verify", "temporal", "combined"]
    exp2_results = []
    for method in methods:
        cfg = RouterConfig(confidence_threshold=0.7, confidence_method=method)
        summary, metrics = await run_single(cfg, workload, vision, f"conf_{method}")
        print_row(f"conf_{method}", summary)

        # Confidence calibration for this method
        fb = FeedbackCollector()
        for o in metrics.outcomes:
            if o.edge_analysis and o.cloud_analysis:
                fb.record(o)
        calibration = fb.confidence_calibration(bins=5)

        row = summary.copy()
        row["method"] = method
        row["calibration"] = calibration
        row["cascade_confirmation_rate"] = fb.confirmation_rate
        exp2_results.append(row)

    results["exp2_confidence_methods"] = exp2_results

    # ===================================================================
    # Experiment 3: Edge Model Size Ablation
    #   0.6B / 1.7B / 4B / 8B with fixed cloud (72B)
    # ===================================================================
    print("\n" + "=" * 70)
    print("Experiment 3: Edge Model Size Ablation")
    print("=" * 70)

    model_sizes = ["0.6B", "1.7B", "4B", "8B"]
    exp3_results = []
    for size in model_sizes:
        cfg = RouterConfig(confidence_threshold=0.7)
        edge = SizedMockEdgeAnalyzer(model_size=size)
        summary, metrics = await run_single(
            cfg, workload, vision, f"edge_{size}", edge_analyzer=edge,
        )
        print(f"  edge={size:4s}: acc={summary['accuracy']:.3f}  miss={summary['miss_rate']:.3f}  "
              f"upgrade={summary['upgrade_rate']:.3f}  p50_lat={summary['p50_latency_ms']:.2f}ms")
        row = summary.copy()
        row["model_size"] = size
        exp3_results.append(row)

    results["exp3_model_size_ablation"] = exp3_results

    # ===================================================================
    # Experiment 4. WAN Latency Sweep
    #   10ms / 50ms / 200ms / 500ms added to cloud RTT
    # ===================================================================
    print("\n" + "=" * 70)
    print("Experiment 4. WAN Latency Sweep")
    print("=" * 70)

    wan_delays = [10, 50, 200, 500]
    exp4_results = []
    for delay in wan_delays:
        # All-Cloud with WAN delay
        cfg_cloud = RouterConfig(
            confidence_threshold=1.0,
            clearly_normal_anomaly_max=-1,
            clearly_normal_confidence_min=999,
        )
        cloud_delayed = WANDelayCloudAnalyzer(MockCloudAnalyzer(), wan_delay_ms=delay)
        s_cloud, _ = await run_single(
            cfg_cloud, workload, vision, f"all_cloud_wan{delay}",
            cloud_analyzer=cloud_delayed,
        )

        # EdgeRouter cascade with WAN delay
        cfg_er = RouterConfig(confidence_threshold=0.7)
        cloud_delayed2 = WANDelayCloudAnalyzer(MockCloudAnalyzer(), wan_delay_ms=delay)
        s_er, _ = await run_single(
            cfg_er, workload, vision, f"edgerouter_wan{delay}",
            cloud_analyzer=cloud_delayed2,
        )

        print(f"  WAN={delay:3d}ms: all_cloud_p50={s_cloud['p50_latency_ms']:.1f}ms  "
              f"edgerouter_p50={s_er['p50_latency_ms']:.1f}ms  "
              f"er_miss={s_er['miss_rate']:.3f}")

        exp4_results.append({
            "wan_delay_ms": delay,
            "all_cloud": s_cloud,
            "edgerouter": s_er,
        })

    results["exp4_wan_latency"] = exp4_results

    # ===================================================================
    # Experiment 5: Online Learning Convergence (1000 scenarios)
    #   Per-scenario threshold update — the router adapts in real time
    # ===================================================================
    print("\n" + "=" * 70)
    print(f"Experiment 5: Online Learning Convergence ({ext_workload.size} scenarios)")
    print("=" * 70)

    learner = OnlineRouterLearner(initial_threshold=0.5, learning_rate=0.01)
    feedback = FeedbackCollector()
    prefetcher = PredictivePrefetcher(window_size=5, decline_threshold=-0.02)

    edge_ol = MockEdgeAnalyzer()
    cloud_ol = MockCloudAnalyzer()
    metrics_ol = BenchmarkMetrics()
    threshold_over_time = []
    accuracy_over_time = []
    correct_so_far = 0

    for i, scenario in enumerate(ext_workload.scenarios):
        # Update router config with learned threshold
        cfg_ol = RouterConfig(confidence_threshold=learner.threshold)
        router_ol = RouterEngine(cfg_ol)
        cascade_ol = CascadeExecutor(edge_ol, cloud_ol, cfg_ol)

        vision_output = vision.detect(scenario)
        context = BenchmarkRunner._scenario_to_context(scenario, idx=i)

        # Prefetch check
        if prefetcher.should_prefetch():
            pass  # In production: preemptively upload context to cloud

        # Route and execute
        decision = router_ol.route(vision_output, context)
        outcome = await cascade_ol.execute(vision_output, context, decision)
        outcome.ground_truth_judgment = scenario.ground_truth_judgment

        # Online learning update
        new_t = learner.update(outcome)
        threshold_over_time.append({"scenario": i, "threshold": round(new_t, 4)})

        # Prefetch tracking
        prefetcher.update(outcome.edge_confidence)
        if decision.tier == RoutingTier.CASCADE and outcome.cloud_analysis:
            prefetcher.mark_cascade_happened()

        if outcome.edge_analysis and outcome.cloud_analysis:
            feedback.record(outcome)

        # Record metrics
        metrics_ol.total_scenarios += 1
        metrics_ol.outcomes.append(outcome)
        metrics_ol.latencies.append(outcome.total_latency_ms)
        metrics_ol.routing_overhead_ms.append(decision.latency_ms)

        gt = scenario.ground_truth_judgment
        predicted = outcome.final_judgment
        if predicted == gt:
            metrics_ol.correct_judgments += 1
            correct_so_far += 1
        elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
            metrics_ol.false_negatives += 1
        elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
            metrics_ol.false_positives += 1

        if decision.tier == RoutingTier.EDGE_EMERGENCY:
            metrics_ol.emergency_count += 1
        elif decision.tier == RoutingTier.EDGE:
            metrics_ol.edge_only_count += 1
        elif decision.tier == RoutingTier.CLOUD:
            metrics_ol.cloud_count += 1
        elif decision.tier == RoutingTier.CASCADE:
            metrics_ol.cascade_count += 1

        # Track running accuracy every 100 scenarios
        if (i + 1) % 100 == 0:
            accuracy_over_time.append({
                "scenario": i + 1,
                "accuracy": round(correct_so_far / (i + 1), 4),
                "threshold": round(new_t, 4),
            })

    # Static baseline for comparison
    cfg_static = RouterConfig(confidence_threshold=0.5)
    s_static, m_static = await run_single(cfg_static, ext_workload, vision, "static_0.5")

    learner_stats = learner.get_stats()
    prefetch_stats = prefetcher.get_stats()
    fb_stats = feedback.stats_by_difficulty()
    ol_summary = metrics_ol.summary()

    print(f"  Online learning: acc={ol_summary['accuracy']:.3f}  miss={ol_summary['miss_rate']:.3f}  "
          f"upgrade={ol_summary['upgrade_rate']:.3f}")
    print(f"  Final threshold: {learner_stats['current_threshold']:.4f}")
    print(f"  Updates: {learner_stats['total_updates']}")
    print(f"  Confirmed: {learner_stats['confirmed']}  Overridden: {learner_stats['overridden']}")
    print(f"  Confirmation rate: {learner_stats['confirmation_rate']:.3f}")
    print(f"  Prefetch stats: triggered={prefetch_stats['prefetch_triggered']} "
          f"useful={prefetch_stats['prefetch_useful']} precision={prefetch_stats['precision']:.3f}")
    print(f"  Static-0.5 baseline: acc={s_static['accuracy']:.3f}  miss={s_static['miss_rate']:.3f}")

    for diff, s in fb_stats.items():
        print(f"  [{diff}] confirm_rate={s['confirmation_rate']:.3f} avg_conf={s['avg_edge_confidence']:.3f}")

    results["exp5_online_learning"] = {
        "online_learning_summary": ol_summary,
        "learner_stats": learner_stats,
        "prefetch_stats": prefetch_stats,
        "feedback_by_difficulty": fb_stats,
        "threshold_trajectory": threshold_over_time[::50],  # sample every 50
        "accuracy_over_time": accuracy_over_time,
        "static_baseline": s_static,
        "calibration": feedback.confidence_calibration(bins=10),
    }

    # ===================================================================
    # Experiment 6: Data Security Compliance
    # ===================================================================
    print("\n" + "=" * 70)
    print(f"Experiment 6: Data Security Compliance ({sec_workload.size} scenarios)")
    print("=" * 70)

    cfg_sec = RouterConfig(confidence_threshold=0.7)
    s_sec, m_sec = await run_single(cfg_sec, sec_workload, vision, "security_test")

    sensitive_scenarios = sec_workload.sensitive_only()
    sensitive_count = len(sensitive_scenarios)
    print(f"  Total scenarios: {s_sec['total_scenarios']}")
    print(f"  Sensitive scenarios: {sensitive_count}")
    print(f"  Sensitive leaked to cloud: {s_sec['sensitive_leaked']}")
    print(f"  Data security compliance: {s_sec['data_security_compliance']:.4f}")
    print(f"  Accuracy (overall): {s_sec['accuracy']:.3f}")
    print(f"  Miss rate: {s_sec['miss_rate']:.3f}")

    # Verify zero leaks
    if s_sec['sensitive_leaked'] == 0:
        print("  ✓ PASS: Zero sensitive data leaks (recall = 100%)")
    else:
        print(f"  ✗ FAIL: {s_sec['sensitive_leaked']} sensitive records leaked!")

    results["exp6_data_security"] = {
        "summary": s_sec,
        "sensitive_total": sensitive_count,
        "sensitive_leaked": s_sec["sensitive_leaked"],
        "compliance": s_sec["data_security_compliance"],
        "pass": s_sec["sensitive_leaked"] == 0,
    }

    # ===================================================================
    # Save all results
    # ===================================================================
    elapsed = time.perf_counter() - t_total

    results["metadata"] = {
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "primary_workload_size": workload.size,
        "extended_workload_size": ext_workload.size,
        "security_workload_size": sec_workload.size,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 70}")
    print(f"All 6 experiments completed in {elapsed:.1f}s")
    print(f"Results saved to {output_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Random-Split baseline helper
# ---------------------------------------------------------------------------

async def _run_random_split(
    config: RouterConfig,
    workload,
    vision: VisionModel,
    cloud_rate: float = 0.4,
    seed: int = 42,
) -> tuple[dict, BenchmarkMetrics]:
    """Random baseline: route cloud_rate fraction randomly to cloud."""
    rng = random.Random(seed)
    router = RouterEngine(config)
    edge = MockEdgeAnalyzer()
    cloud = MockCloudAnalyzer()
    cascade = CascadeExecutor(edge, cloud, config)
    runner = BenchmarkRunner(router, cascade, vision)

    metrics = BenchmarkMetrics()

    for i, scenario in enumerate(workload.scenarios):
        vision_output = vision.detect(scenario)
        context = runner._scenario_to_context(scenario, idx=i)

        # Random routing: ignore normal routing logic
        if rng.random() < cloud_rate:
            decision = RoutingDecision(tier=RoutingTier.CLOUD, reason="random_split")
        else:
            decision = RoutingDecision(tier=RoutingTier.EDGE, reason="random_split")

        outcome = await cascade.execute(vision_output, context, decision)
        outcome.ground_truth_judgment = scenario.ground_truth_judgment

        metrics.total_scenarios += 1
        metrics.outcomes.append(outcome)
        metrics.latencies.append(outcome.total_latency_ms)
        metrics.routing_overhead_ms.append(0.0)

        # Tier counts
        if decision.tier == RoutingTier.CLOUD:
            metrics.cloud_count += 1
        else:
            metrics.edge_only_count += 1

        # Judgment correctness
        gt = scenario.ground_truth_judgment
        predicted = outcome.final_judgment
        if predicted == gt:
            metrics.correct_judgments += 1
        elif gt in (Judgment.WARNING, Judgment.ALARM) and predicted == Judgment.NORMAL:
            metrics.false_negatives += 1
        elif gt == Judgment.NORMAL and predicted in (Judgment.WARNING, Judgment.ALARM):
            metrics.false_positives += 1

        # Data security
        if scenario.contains_process_params:
            metrics.sensitive_total += 1
            if decision.tier == RoutingTier.CLOUD and outcome.cloud_analysis is not None:
                metrics.sensitive_leaked += 1

    summary = metrics.summary()
    return summary, metrics


if __name__ == "__main__":
    asyncio.run(main())
