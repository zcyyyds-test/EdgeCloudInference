"""Core data models for EdgeRouter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    NORMAL = "normal"
    MARGINAL = "marginal"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"


class Trend(str, Enum):
    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    OSCILLATING = "oscillating"


class RoutingTier(str, Enum):
    EDGE_EMERGENCY = "edge_emergency"
    EDGE = "edge"
    CASCADE = "cascade"
    CLOUD = "cloud"


class Judgment(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"


class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


# ---------------------------------------------------------------------------
# Scenario (input to the pipeline)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioProfile:
    """Description of an industrial operating condition for visual anomaly detection."""

    name: str
    difficulty: Difficulty
    true_anomaly_level: float             # ground truth anomaly level (0-100)
    true_secondary_metric: float          # secondary quality metric [0, 1]
    true_color_rgb: tuple[int, int, int] = (180, 180, 180)
    true_texture_irregularity: float = 0.0  # [0, 1]
    true_surface_uniformity: float = 1.0  # [0, 1]
    trend: Trend = Trend.STABLE
    num_correlated_anomalies: int = 0
    is_novel: bool = False
    contains_process_params: bool = False
    has_recipe_params: bool = False
    has_customer_info: bool = False
    has_reaction_params: bool = False
    ground_truth_judgment: Judgment = Judgment.NORMAL
    description: str = ""


# ---------------------------------------------------------------------------
# Vision model output
# ---------------------------------------------------------------------------

@dataclass
class VisionOutput:
    """Structured output from the vision model."""

    timestamp: float = field(default_factory=time.time)
    anomaly_level: float = 50.0           # primary anomaly indicator (0-100)
    measurement_confidence: float = 0.9   # confidence in primary measurement [0, 1]
    color_rgb: tuple[int, int, int] = (180, 180, 180)
    secondary_metric: float = 0.1        # secondary quality metric [0, 1]
    texture_irregularity: float = 0.0    # texture irregularity [0, 1]
    surface_uniformity: float = 1.0      # [0, 1]
    anomaly_score: float = 0.0           # composite anomaly score [0, 1]
    anomaly_confidence: float = 0.9      # confidence in anomaly classification [0, 1]
    inference_latency_ms: float = 18.0
    image_path: str | None = None        # path to source image (multimodal)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "anomaly_level": round(self.anomaly_level, 2),
            "measurement_confidence": round(self.measurement_confidence, 3),
            "color_rgb": list(self.color_rgb),
            "secondary_metric": round(self.secondary_metric, 3),
            "texture_irregularity": round(self.texture_irregularity, 3),
            "surface_uniformity": round(self.surface_uniformity, 3),
            "anomaly_score": round(self.anomaly_score, 3),
            "anomaly_confidence": round(self.anomaly_confidence, 3),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
        }
        if self.image_path:
            d["image_path"] = self.image_path
        return d


# ---------------------------------------------------------------------------
# Process context (metadata that travels with a frame)
# ---------------------------------------------------------------------------

@dataclass
class ProcessContext:
    """Contextual metadata accompanying a detection frame."""

    scenario_id: str = ""
    has_recipe_params: bool = False
    has_customer_info: bool = False
    has_reaction_params: bool = False
    num_correlated_anomalies: int = 0
    equipment_id: str = ""
    batch_id: str = ""

    def get_trend_summary(self) -> str:
        return "aggregated_trend_placeholder"


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Output of the router engine."""

    tier: RoutingTier
    reason: str
    action: str = ""                      # optional immediate action hint
    latency_ms: float = 0.0              # router decision time


# ---------------------------------------------------------------------------
# Analysis result (shared by edge and cloud analyzers)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Output of an analyzer (edge or cloud)."""

    judgment: Judgment = Judgment.NORMAL
    confidence: float = 0.9
    suggested_action: str = "maintain"
    reasoning: str = ""
    root_cause: str = ""                  # cloud-only: root cause analysis
    latency_ms: float = 0.0
    source: str = "edge"                  # "edge" or "cloud"

    def to_dict(self) -> dict[str, Any]:
        return {
            "judgment": self.judgment.value,
            "confidence": round(self.confidence, 3),
            "suggested_action": self.suggested_action,
            "reasoning": self.reasoning,
            "root_cause": self.root_cause,
            "latency_ms": round(self.latency_ms, 2),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Control action
# ---------------------------------------------------------------------------

@dataclass
class ControlAction:
    """Output of the control engine."""

    type: str = "maintain"                # maintain / adjust / emergency_stop
    params: dict[str, Any] = field(default_factory=dict)
    based_on_judgment: Judgment = Judgment.NORMAL
    pending_cloud_revision: bool = False


# ---------------------------------------------------------------------------
# Routing outcome (for online learning)
# ---------------------------------------------------------------------------

@dataclass
class RoutingOutcome:
    """Full record of a routing cycle, used for online learning."""

    scenario_id: str = ""
    vision_output: VisionOutput | None = None
    routing_decision: RoutingDecision | None = None
    edge_analysis: AnalysisResult | None = None
    cloud_analysis: AnalysisResult | None = None
    final_judgment: Judgment = Judgment.NORMAL
    ground_truth_judgment: Judgment | None = None  # scenario ground truth
    actual_outcome: str | None = None     # post-hoc ground truth
    edge_confidence: float = 0.0
    total_latency_ms: float = 0.0
