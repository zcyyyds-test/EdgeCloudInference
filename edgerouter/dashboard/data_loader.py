"""Load experiment results from experiments/ directory for dashboard display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

# Default experiments directory: project_root/experiments/
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


@st.cache_data
def load_json(path: str | Path) -> dict | list | None:
    """Load a JSON file, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def experiments_dir() -> Path:
    return _DEFAULT_DIR


# --- Benchmark results (single-run files) ---

@st.cache_data
def load_real_llm_50() -> dict | None:
    return load_json(_DEFAULT_DIR / "real_llm_results_50.json")


@st.cache_data
def load_edge_cloud_v1() -> dict | None:
    return load_json(_DEFAULT_DIR / "edge_cloud_benchmark.json")


@st.cache_data
def load_edge_cloud_v2() -> dict | None:
    return load_json(_DEFAULT_DIR / "edge_cloud_benchmark_v2.json")


@st.cache_data
def load_edge_cloud_t08() -> dict | None:
    return load_json(_DEFAULT_DIR / "edge_cloud_benchmark_t08.json")


@st.cache_data
def load_edge_cloud_32b() -> dict | None:
    return load_json(_DEFAULT_DIR / "edge_cloud_benchmark_32b.json")


# --- Ablation (array of model results) ---

@st.cache_data
def load_model_ablation() -> list | None:
    return load_json(_DEFAULT_DIR / "model_ablation.json")


@st.cache_data
def load_model_ablation_with_32b() -> list | None:
    return load_json(_DEFAULT_DIR / "model_ablation_with_32b.json")


# --- Control analysis ---

@st.cache_data
def load_control_metrics() -> dict | None:
    return load_json(_DEFAULT_DIR / "control" / "metrics.json")


def control_images() -> dict[str, Path]:
    """Return {scenario_name: image_path} for existing control sim plots."""
    d = _DEFAULT_DIR / "control"
    images = {}
    for name in ("step", "ramp", "oscillation", "multi_phase", "summary_metrics"):
        p = d / f"{name}.png"
        if p.exists():
            images[name] = p
    return images


# --- Threshold sweep (real LLM, if exists) ---

@st.cache_data
def load_threshold_sweep() -> dict | None:
    return load_json(_DEFAULT_DIR / "real_llm_threshold_sweep.json")


# --- Helpers ---

def format_pct(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def format_ms(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}ms"


def summary_to_row(summary: dict[str, Any], label: str = "") -> dict[str, Any]:
    """Extract key metrics from a summary dict into a display-friendly row."""
    row = {}
    if label:
        row["Config"] = label
    row["Accuracy"] = format_pct(summary.get("accuracy"))
    row["Miss Rate"] = format_pct(summary.get("miss_rate"))
    row["False Alarm"] = format_pct(summary.get("false_alarm_rate"))
    row["Cloud Savings"] = format_pct(summary.get("cloud_saving_rate"))
    row["p50 Latency"] = format_ms(summary.get("p50_latency_ms"))
    row["p99 Latency"] = format_ms(summary.get("p99_latency_ms"))
    row["Edge Only"] = summary.get("edge_only", 0)
    row["Cloud"] = summary.get("cloud_direct", 0)
    row["Cascade"] = summary.get("cascade", 0)
    row["Emergency"] = summary.get("emergency", 0)
    return row
