"""Three confidence estimation methods for routing decisions."""

from __future__ import annotations

import numpy as np

from edgerouter.core.config import RouterConfig
from edgerouter.core.schema import AnalysisResult, VisionOutput


# ---------------------------------------------------------------------------
# Method A: Output Probability (zero extra cost)
# ---------------------------------------------------------------------------

def estimate_from_output(
    vision_output: VisionOutput,
    config: RouterConfig | None = None,
) -> float:
    """Use the vision model's own softmax probability as confidence.

    Lightweight: no extra inference. But softmax can be overconfident,
    so we add a margin bonus when the level is well within the safe zone.
    """
    cfg = config or RouterConfig()
    base = vision_output.anomaly_confidence

    level_margin = abs(vision_output.anomaly_level - cfg.safe_anomaly_center) / cfg.safe_anomaly_range
    margin_bonus = max(0.0, 0.1 * (1.0 - level_margin))

    return float(np.clip(base + margin_bonus, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Method B: Self-Verification (one extra edge inference)
# ---------------------------------------------------------------------------

def estimate_from_self_verification(
    vision_output: VisionOutput,
    edge_result: AnalysisResult,
) -> float:
    """Let the edge analyzer evaluate its own judgment's reliability.

    Heuristic proxy (no real second inference at routing time):
    cross-check the analyzer's confidence with the vision model's anomaly
    confidence.  Large disagreement → lower trust.
    """
    # Agreement between vision anomaly confidence and analyzer confidence
    agreement = 1.0 - abs(vision_output.anomaly_confidence - edge_result.confidence)

    # If vision says anomaly but analyzer says normal (or vice versa), penalise
    vision_says_anomaly = vision_output.anomaly_score > 0.5
    analyzer_says_anomaly = edge_result.judgment.value != "normal"
    consistency_bonus = 0.1 if (vision_says_anomaly == analyzer_says_anomaly) else -0.15

    score = 0.5 * edge_result.confidence + 0.3 * agreement + 0.2 * vision_output.anomaly_confidence + consistency_bonus

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Method C: Temporal Consistency Check (most reliable)
# ---------------------------------------------------------------------------

def estimate_from_temporal(
    recent_outputs: list[VisionOutput],
    window_size: int = 10,
) -> float:
    """Check consistency of detections across a sliding window.

    Exploits the temporal continuity of sensor data: single-frame
    anomalies may be noise, multi-frame consistency is a real signal.
    """
    if len(recent_outputs) < 3:
        return 0.5  # not enough data — conservative

    window = recent_outputs[-window_size:]

    levels = np.array([o.anomaly_level for o in window])
    secondaries = np.array([o.secondary_metric for o in window])

    # Stability: lower std → higher confidence
    level_std_thresh = 5.0
    secondary_std_thresh = 0.15

    level_stability = 1.0 - min(1.0, float(np.std(levels)) / level_std_thresh)
    secondary_stability = 1.0 - min(1.0, float(np.std(secondaries)) / secondary_std_thresh)

    # Trend smoothness: fit a line, check residuals
    if len(window) >= 3:
        x = np.arange(len(levels), dtype=float)
        coeffs = np.polyfit(x, levels, 1)
        fitted = np.polyval(coeffs, x)
        residuals = np.abs(levels - fitted)
        mean_residual = float(np.mean(residuals))
        trend_smoothness = 1.0 - min(1.0, mean_residual / level_std_thresh)
    else:
        trend_smoothness = 0.5

    score = 0.4 * level_stability + 0.3 * secondary_stability + 0.3 * trend_smoothness
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Combined estimator
# ---------------------------------------------------------------------------

def estimate_combined(
    vision_output: VisionOutput,
    edge_result: AnalysisResult | None = None,
    recent_outputs: list[VisionOutput] | None = None,
    config: RouterConfig | None = None,
    weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> float:
    """Weighted combination of all three methods.

    Falls back gracefully when not all inputs are available.
    """
    scores: list[float] = []
    w: list[float] = []

    # Method A: always available
    scores.append(estimate_from_output(vision_output, config))
    w.append(weights[0])

    # Method B: only if edge result exists
    if edge_result is not None:
        scores.append(estimate_from_self_verification(vision_output, edge_result))
        w.append(weights[1])

    # Method C: only if temporal history exists
    if recent_outputs and len(recent_outputs) >= 3:
        scores.append(estimate_from_temporal(recent_outputs))
        w.append(weights[2])

    # Normalise weights
    total_w = sum(w)
    return float(np.clip(sum(s * wi / total_w for s, wi in zip(scores, w)), 0.0, 1.0))
