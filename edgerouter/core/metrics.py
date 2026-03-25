"""Prometheus metrics for EdgeRouter."""

from prometheus_client import Counter, Histogram, Gauge

# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------

ROUTING_DECISIONS = Counter(
    "edgerouter_routing_decisions_total",
    "Total routing decisions by tier and reason",
    ["tier", "reason"],
)

ROUTING_LATENCY = Histogram(
    "edgerouter_routing_latency_seconds",
    "Router engine decision latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.05],
)

# ---------------------------------------------------------------------------
# Analyzer metrics
# ---------------------------------------------------------------------------

ANALYSIS_LATENCY = Histogram(
    "edgerouter_analysis_latency_seconds",
    "Analyzer inference latency",
    ["source"],  # edge / cloud
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

ANALYSIS_JUDGMENTS = Counter(
    "edgerouter_analysis_judgments_total",
    "Judgment counts by source and judgment type",
    ["source", "judgment"],
)

# ---------------------------------------------------------------------------
# Cascade metrics
# ---------------------------------------------------------------------------

CASCADE_TOTAL = Counter(
    "edgerouter_cascade_total",
    "Total cascade events",
)

CASCADE_EDGE_CONFIRMED = Counter(
    "edgerouter_cascade_edge_confirmed_total",
    "Cascade events where cloud confirmed edge judgment",
)

CASCADE_EDGE_OVERRIDDEN = Counter(
    "edgerouter_cascade_edge_overridden_total",
    "Cascade events where cloud overrode edge judgment",
)

# ---------------------------------------------------------------------------
# Vision metrics
# ---------------------------------------------------------------------------

VISION_LATENCY = Histogram(
    "edgerouter_vision_latency_seconds",
    "Vision model latency",
    buckets=[0.005, 0.01, 0.02, 0.05],
)

# ---------------------------------------------------------------------------
# System gauges
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = Gauge(
    "edgerouter_confidence_threshold",
    "Current confidence threshold for cascade routing",
)

CLOUD_UPGRADE_RATE = Gauge(
    "edgerouter_cloud_upgrade_rate",
    "Rolling rate of cloud upgrades (cascade + direct cloud)",
)
