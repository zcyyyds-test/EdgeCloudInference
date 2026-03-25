"""Converters between protobuf messages and core dataclasses."""

from __future__ import annotations

from edgerouter.core.schema import (
    AnalysisResult,
    Judgment,
    ProcessContext,
    RoutingDecision,
    RoutingTier,
    VisionOutput,
)


# ---------------------------------------------------------------------------
# VisionOutput
# ---------------------------------------------------------------------------

def vision_to_proto(vo: VisionOutput, pb_cls):
    """Convert VisionOutput dataclass → proto message."""
    msg = pb_cls()
    msg.timestamp = vo.timestamp
    msg.anomaly_level = vo.anomaly_level
    msg.measurement_confidence = vo.measurement_confidence
    msg.color_rgb.r = vo.color_rgb[0]
    msg.color_rgb.g = vo.color_rgb[1]
    msg.color_rgb.b = vo.color_rgb[2]
    msg.secondary_metric = vo.secondary_metric
    msg.texture_irregularity = vo.texture_irregularity
    msg.surface_uniformity = vo.surface_uniformity
    msg.anomaly_score = vo.anomaly_score
    msg.anomaly_confidence = vo.anomaly_confidence
    msg.inference_latency_ms = vo.inference_latency_ms
    return msg


def vision_from_proto(msg) -> VisionOutput:
    """Convert proto VisionOutput → dataclass."""
    return VisionOutput(
        timestamp=msg.timestamp,
        anomaly_level=msg.anomaly_level,
        measurement_confidence=msg.measurement_confidence,
        color_rgb=(msg.color_rgb.r, msg.color_rgb.g, msg.color_rgb.b),
        secondary_metric=msg.secondary_metric,
        texture_irregularity=msg.texture_irregularity,
        surface_uniformity=msg.surface_uniformity,
        anomaly_score=msg.anomaly_score,
        anomaly_confidence=msg.anomaly_confidence,
        inference_latency_ms=msg.inference_latency_ms,
    )


# ---------------------------------------------------------------------------
# ProcessContext
# ---------------------------------------------------------------------------

def context_from_proto(msg) -> ProcessContext:
    return ProcessContext(
        scenario_id=msg.scenario_id,
        has_recipe_params=msg.has_recipe_params,
        has_customer_info=msg.has_customer_info,
        has_reaction_params=msg.has_reaction_params,
        num_correlated_anomalies=msg.num_correlated_anomalies,
        equipment_id=msg.equipment_id,
        batch_id=msg.batch_id,
    )


def context_to_proto(ctx: ProcessContext, pb_cls):
    msg = pb_cls()
    msg.scenario_id = ctx.scenario_id
    msg.has_recipe_params = ctx.has_recipe_params
    msg.has_customer_info = ctx.has_customer_info
    msg.has_reaction_params = ctx.has_reaction_params
    msg.num_correlated_anomalies = ctx.num_correlated_anomalies
    msg.equipment_id = ctx.equipment_id
    msg.batch_id = ctx.batch_id
    return msg


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

def analysis_to_proto(ar: AnalysisResult, pb_cls):
    msg = pb_cls()
    msg.judgment = ar.judgment.value
    msg.confidence = ar.confidence
    msg.suggested_action = ar.suggested_action
    msg.reasoning = ar.reasoning
    msg.root_cause = ar.root_cause
    msg.latency_ms = ar.latency_ms
    msg.source = ar.source
    return msg


def analysis_from_proto(msg) -> AnalysisResult | None:
    if not msg.judgment:
        return None
    return AnalysisResult(
        judgment=Judgment(msg.judgment),
        confidence=msg.confidence,
        suggested_action=msg.suggested_action,
        reasoning=msg.reasoning,
        root_cause=msg.root_cause,
        latency_ms=msg.latency_ms,
        source=msg.source,
    )


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------

def decision_to_proto(rd: RoutingDecision, pb_cls):
    msg = pb_cls()
    msg.tier = rd.tier.value
    msg.reason = rd.reason
    msg.action = rd.action
    msg.latency_ms = rd.latency_ms
    return msg


def decision_from_proto(msg) -> RoutingDecision:
    return RoutingDecision(
        tier=RoutingTier(msg.tier),
        reason=msg.reason,
        action=msg.action,
        latency_ms=msg.latency_ms,
    )
