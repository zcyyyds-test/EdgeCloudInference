"""Run control signal analysis and generate comparison plots.

Usage:
    python scripts/run_control.py [--output-dir experiments/control]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from edgerouter.scenarios.control import (
    SimResult,
    TankConfig,
    run_all_strategies,
)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLORS = {
    "Ideal Controller": ("#222222", "--", 1.5),
    "Edge-Only (0.8B)": ("#2196F3", "-", 1.3),
    "Cloud-Only (27B)": ("#F44336", "-", 1.3),
    "EdgeRouter (Cascade)": ("#4CAF50", "-", 2.0),
    "No Control": ("#9E9E9E", ":", 1.0),
}


def _style(name: str):
    for key, (color, ls, lw) in COLORS.items():
        if key in name:
            return color, ls, lw
    return "#000000", "-", 1.0


def plot_scenario(results: list[SimResult], title: str, out_path: Path):
    """Generate a 3-panel plot: level, valve, disturbance."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    ax_level, ax_valve, ax_dist = axes
    setpoint = results[0].setpoint
    time_s = results[0].time_s

    # --- Level ---
    ax_level.axhline(setpoint, color="#888", linewidth=0.8, label="Setpoint")
    ax_level.axhspan(30, 70, alpha=0.06, color="green", label="Safe range (30-70cm)")
    ax_level.axhline(70, color="orange", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_level.axhline(30, color="orange", linewidth=0.5, linestyle="--", alpha=0.5)

    for r in results:
        c, ls, lw = _style(r.name)
        ax_level.plot(time_s, r.level_cm, color=c, linestyle=ls, linewidth=lw, label=r.name)

    ax_level.set_ylabel("Process Variable (cm)")  # Tank level as canonical example
    ax_level.set_ylim(0, 100)
    ax_level.legend(loc="upper right", fontsize=8)
    ax_level.set_title(title, fontsize=13, fontweight="bold")
    ax_level.grid(True, alpha=0.3)

    # --- Valve ---
    for r in results:
        c, ls, lw = _style(r.name)
        ax_valve.plot(time_s, r.valve, color=c, linestyle=ls, linewidth=lw, label=r.name)

    ax_valve.set_ylabel("Valve Opening")
    ax_valve.set_ylim(-0.05, 1.05)
    ax_valve.grid(True, alpha=0.3)

    # --- Disturbance ---
    Q = results[0].disturbance
    ax_dist.fill_between(time_s, Q * 1e3, alpha=0.3, color="#FF9800")
    ax_dist.plot(time_s, Q * 1e3, color="#FF9800", linewidth=1.0)
    ax_dist.set_ylabel("Inflow (L/s)")
    ax_dist.set_xlabel("Time (s)")
    ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metrics_summary(all_metrics: dict[str, dict[str, dict]], out_path: Path):
    """Bar chart comparing ISE, max deviation, settling time across strategies."""
    scenarios = list(all_metrics.keys())
    strategies = list(next(iter(all_metrics.values())).keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ["ise", "max_deviation_cm", "settling_time_s"]
    metric_labels = ["Integral Squared Error", "Max Deviation (cm)", "Settling Time (s)"]

    x = np.arange(len(scenarios))
    width = 0.15
    offsets = np.arange(len(strategies)) - (len(strategies) - 1) / 2

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        for i, strat in enumerate(strategies):
            vals = [all_metrics[sc][strat][metric] for sc in scenarios]
            c, _, _ = _style(strat)
            ax.bar(x + offsets[i] * width, vals, width * 0.9, label=strat, color=c, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, fontsize=8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2, axis="y")
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Control Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Control Signal Analysis")
    parser.add_argument("--output-dir", default="experiments/control")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TankConfig(total_steps=args.steps)

    disturbances = ["step", "ramp", "oscillation", "multi_phase"]
    all_metrics: dict[str, dict[str, dict]] = {}

    print("=" * 60)
    print("EdgeRouter Control Signal Analysis")
    print("=" * 60)
    print(f"  Steps: {cfg.total_steps} ({cfg.total_steps * cfg.dt_s:.0f}s)")
    print(f"  Edge delay: {cfg.edge_delay} step ({cfg.edge_delay * cfg.dt_s:.1f}s)")
    print(f"  Cloud delay: {cfg.cloud_delay} steps ({cfg.cloud_delay * cfg.dt_s:.1f}s)")
    print(f"  Edge deadband: {cfg.edge_deadband_cm}cm / Cloud deadband: {cfg.cloud_deadband_cm}cm")
    print()

    for dist_name in disturbances:
        print(f"Running: {dist_name}...")
        results = run_all_strategies(dist_name, cfg, seed=42)

        # Plot
        title = f"Control Response — {dist_name.replace('_', ' ').title()} Disturbance"
        plot_scenario(results, title, out_dir / f"{dist_name}.png")

        # Collect metrics
        scenario_metrics = {}
        for r in results:
            scenario_metrics[r.name] = {
                "ise": round(r.ise, 1),
                "max_deviation_cm": round(r.max_deviation, 2),
                "mae_cm": round(r.mean_abs_error, 2),
                "settling_time_s": round(r.settling_idx * cfg.dt_s, 1),
            }
        all_metrics[dist_name] = scenario_metrics

        # Print per-scenario summary
        for r in results:
            print(f"    {r.name:25s}  ISE={r.ise:10.1f}  MaxDev={r.max_deviation:6.2f}cm  "
                  f"MAE={r.mean_abs_error:5.2f}cm  Settle={r.settling_idx * cfg.dt_s:5.1f}s")

    # Summary bar chart
    print("\nGenerating summary chart...")
    plot_metrics_summary(all_metrics, out_dir / "summary_metrics.png")

    # Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
