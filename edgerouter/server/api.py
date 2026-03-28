"""FastAPI entry point for EdgeRouter."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from edgerouter.core.config import AppConfig
from edgerouter.core.schema import Difficulty, ProcessContext, RoutingTier, VisionOutput
from edgerouter.router.engine import RouterEngine
from edgerouter.scenarios.scenarios import ScenarioGenerator
from edgerouter.scenarios.vision import VisionModel
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.inference.cloud_analyzer import CloudAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

config = AppConfig()
scenario_gen = ScenarioGenerator(seed=42)
vision_model = VisionModel(config=config.vision_config, seed=42)
router_engine = RouterEngine(config=config.router)
edge_analyzer: EdgeAnalyzer | None = None
cloud_analyzer: CloudAnalyzer | None = None

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global edge_analyzer, cloud_analyzer
    edge_analyzer = EdgeAnalyzer(config=config.edge_analyzer)
    cloud_analyzer = CloudAnalyzer(config=config.cloud_analyzer)
    logger.info("EdgeRouter API started")
    yield
    if edge_analyzer:
        await edge_analyzer.close()
    if cloud_analyzer:
        await cloud_analyzer.close()
    logger.info("EdgeRouter API stopped")


app = FastAPI(
    title="EdgeCloudInference",
    description="Confidence-Driven Cloud-Edge Inference Routing",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    scenario_key: str | None = None
    difficulty: str | None = None


class DetectResponse(BaseModel):
    vision_output: dict
    scenario_name: str
    scenario_difficulty: str


class AnalyzeRequest(BaseModel):
    vision_output: dict
    source: str = "edge"    # "edge" or "cloud"


class AnalyzeResponse(BaseModel):
    judgment: str
    confidence: float
    suggested_action: str
    reasoning: str
    root_cause: str
    latency_ms: float
    source: str


class HealthResponse(BaseModel):
    status: str
    edge_available: bool
    cloud_available: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
async def health():
    edge_ok = await edge_analyzer.health_check() if edge_analyzer else False
    cloud_ok = await cloud_analyzer.health_check() if cloud_analyzer else False
    return HealthResponse(
        status="ok",
        edge_available=edge_ok,
        cloud_available=cloud_ok,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """Run vision detection on a generated scenario."""
    diff = Difficulty(req.difficulty) if req.difficulty else None
    scenario = scenario_gen.generate_one(
        template_key=req.scenario_key,
        difficulty=diff,
    )
    output = vision_model.detect(scenario)
    return DetectResponse(
        vision_output=output.to_dict(),
        scenario_name=scenario.name,
        scenario_difficulty=scenario.difficulty.value,
    )


@app.post("/analyze/edge", response_model=AnalyzeResponse)
async def analyze_edge(req: AnalyzeRequest):
    """Send vision output to edge analyzer."""
    vo = _dict_to_vision_output(req.vision_output)
    result = await edge_analyzer.analyze(vo)
    return AnalyzeResponse(
        judgment=result.judgment.value,
        confidence=result.confidence,
        suggested_action=result.suggested_action,
        reasoning=result.reasoning,
        root_cause=result.root_cause,
        latency_ms=result.latency_ms,
        source=result.source,
    )


@app.post("/analyze/cloud", response_model=AnalyzeResponse)
async def analyze_cloud(req: AnalyzeRequest):
    """Send vision output to cloud analyzer."""
    vo = _dict_to_vision_output(req.vision_output)
    result = await cloud_analyzer.analyze(vo)
    return AnalyzeResponse(
        judgment=result.judgment.value,
        confidence=result.confidence,
        suggested_action=result.suggested_action,
        reasoning=result.reasoning,
        root_cause=result.root_cause,
        latency_ms=result.latency_ms,
        source=result.source,
    )


# ---------------------------------------------------------------------------
# Frontend API: experiments, system info, demo route
# ---------------------------------------------------------------------------

@app.get("/api/experiments")
async def list_experiments():
    """List all experiment JSON files."""
    if not _EXPERIMENTS_DIR.exists():
        return []
    results = []
    for f in sorted(_EXPERIMENTS_DIR.glob("*.json")):
        results.append({
            "name": f.stem,
            "file": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    return results


@app.get("/api/experiments/{name}")
async def get_experiment(name: str):
    """Return a single experiment JSON by name."""
    path = _EXPERIMENTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/system/config")
async def get_config():
    """Return current system configuration."""
    return {
        "router": config.router.model_dump(),
        "edge_analyzer": config.edge_analyzer.model_dump(),
        "cloud_analyzer": {
            k: v for k, v in config.cloud_analyzer.model_dump().items()
            if k not in ("api_key", "openai_api_key")  # hide secrets
        },
        "vision_config": config.vision_config.model_dump(),
    }


@app.get("/api/system/models")
async def get_models():
    """Return loaded model info."""
    return {
        "edge": {
            "model": config.edge_analyzer.model,
            "base_url": config.edge_analyzer.base_url,
            "available": await edge_analyzer.health_check() if edge_analyzer else False,
        },
        "cloud": {
            "model": config.cloud_analyzer.model,
            "base_url": config.cloud_analyzer.base_url,
            "available": await cloud_analyzer.health_check() if cloud_analyzer else False,
        },
    }


class RouteRequest(BaseModel):
    vision_output: dict
    num_correlated_anomalies: int = 0


@app.post("/api/demo/route")
async def demo_route(req: RouteRequest):
    """Run a single routing decision (no actual LLM call)."""
    vo = _dict_to_vision_output(req.vision_output)
    ctx = ProcessContext(num_correlated_anomalies=req.num_correlated_anomalies)
    decision = router_engine.route(vo, ctx)

    tier_labels = {
        RoutingTier.EDGE_EMERGENCY: {"tier_index": 0, "tier_name": "Safety Check"},
        RoutingTier.EDGE: {"tier_index": 1, "tier_name": "Edge"},
        RoutingTier.CLOUD: {"tier_index": 3, "tier_name": "Complex → Cloud"},
        RoutingTier.CASCADE: {"tier_index": 4, "tier_name": "Cascade"},
    }
    info = tier_labels.get(decision.tier, {"tier_index": -1, "tier_name": "Unknown"})

    # Determine which tiers were evaluated
    tiers_evaluated = []
    reason = decision.reason

    tiers_evaluated.append({
        "tier": "safety", "triggered": "critical_safety" in reason,
        "detail": f"anomaly_level={vo.anomaly_level:.1f}",
    })
    tiers_evaluated.append({
        "tier": "data_security", "triggered": "data_security" in reason,
        "detail": "No sensitive data",
    })
    tiers_evaluated.append({
        "tier": "clearly_normal", "triggered": "clearly_normal" in reason,
        "detail": f"score={vo.anomaly_score:.3f}, conf={vo.anomaly_confidence:.3f}",
    })
    tiers_evaluated.append({
        "tier": "complex", "triggered": "complex_anomaly" in reason,
        "detail": f"score={vo.anomaly_score:.3f}, correlated={req.num_correlated_anomalies}",
    })
    tiers_evaluated.append({
        "tier": "grey_zone", "triggered": "grey_zone" in reason,
        "detail": reason,
    })

    return {
        "tier": decision.tier.value,
        "reason": decision.reason,
        "action": decision.action,
        "latency_ms": decision.latency_ms,
        **info,
        "tiers": tiers_evaluated,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_vision_output(d: dict) -> VisionOutput:
    return VisionOutput(
        timestamp=d.get("timestamp", 0.0),
        anomaly_level=d.get("anomaly_level", 50.0),
        measurement_confidence=d.get("measurement_confidence", 0.9),
        color_rgb=tuple(d.get("color_rgb", [180, 180, 180])),
        secondary_metric=d.get("secondary_metric", 0.1),
        texture_irregularity=d.get("texture_irregularity", 0.0),
        surface_uniformity=d.get("surface_uniformity", 1.0),
        anomaly_score=d.get("anomaly_score", 0.0),
        anomaly_confidence=d.get("anomaly_confidence", 0.9),
        inference_latency_ms=d.get("inference_latency_ms", 18.0),
        image_path=d.get("image_path"),
    )


# ---------------------------------------------------------------------------
# Static file serving (SPA fallback) — must be last
# ---------------------------------------------------------------------------

if _FRONTEND_DIST.exists():
    # Serve built frontend assets at root
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
