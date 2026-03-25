"""gRPC server for EdgeRouter edge-cloud communication.

Usage:
    python -m edgerouter.server.grpc_server [--port 50051]

The gRPC server exposes the same routing and analysis capabilities as the
REST API (api.py), but with lower serialization overhead via protobuf —
suitable for the edge-to-cloud data path.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import deque

import grpc

from edgerouter.core.config import AppConfig
from edgerouter.core.schema import Difficulty, RoutingTier
from edgerouter.inference.edge_analyzer import EdgeAnalyzer
from edgerouter.inference.cloud_analyzer import CloudAnalyzer
from edgerouter.router.cascade import CascadeExecutor
from edgerouter.router.engine import RouterEngine
from edgerouter.server.converter import (
    analysis_from_proto,
    analysis_to_proto,
    context_from_proto,
    decision_from_proto,
    decision_to_proto,
    vision_from_proto,
    vision_to_proto,
)
from edgerouter.scenarios.scenarios import ScenarioGenerator
from edgerouter.scenarios.vision import VisionModel

# Lazy import — generated stubs may not exist yet
try:
    from edgerouter.proto import edgerouter_pb2, edgerouter_pb2_grpc
except ImportError:
    edgerouter_pb2 = None
    edgerouter_pb2_grpc = None

logger = logging.getLogger(__name__)


class EdgeRouterServicer:
    """Implements the EdgeRouterService gRPC interface."""

    def __init__(
        self,
        router: RouterEngine,
        cascade: CascadeExecutor,
        edge_analyzer: EdgeAnalyzer,
        cloud_analyzer: CloudAnalyzer,
        vision: VisionModel | None = None,
        scenario_gen: ScenarioGenerator | None = None,
    ):
        self.router = router
        self.cascade = cascade
        self.edge = edge_analyzer
        self.cloud = cloud_analyzer
        self.vision = vision or VisionModel(seed=42)
        self.scenario_gen = scenario_gen or ScenarioGenerator(seed=42)
        # Sliding window for streaming
        self._stream_history: deque = deque(maxlen=20)

    async def HealthCheck(self, request, context):
        edge_ok = await self.edge.health_check()
        cloud_ok = await self.cloud.health_check()
        return edgerouter_pb2.HealthResponse(
            status="ok", edge_available=edge_ok, cloud_available=cloud_ok,
        )

    async def Detect(self, request, context):
        diff = Difficulty(request.difficulty) if request.difficulty else None
        scenario = self.scenario_gen.generate_one(
            template_key=request.scenario_key or None,
            difficulty=diff,
        )
        vo = self.vision.detect(scenario)
        return edgerouter_pb2.DetectResponse(
            vision_output=vision_to_proto(vo, edgerouter_pb2.VisionOutput),
            scenario_name=scenario.name,
            scenario_difficulty=scenario.difficulty.value,
        )

    async def Route(self, request, context):
        vo = vision_from_proto(request.vision_output)
        ctx = context_from_proto(request.context)
        decision = self.router.route(vo, ctx)
        return edgerouter_pb2.RouteResponse(
            decision=decision_to_proto(decision, edgerouter_pb2.RoutingDecision),
        )

    async def Cascade(self, request, context):
        vo = vision_from_proto(request.vision_output)
        ctx = context_from_proto(request.context)
        rd = decision_from_proto(request.routing_decision)
        history = [vision_from_proto(h) for h in request.recent_history] or None

        outcome = await self.cascade.execute(vo, ctx, rd, history)

        resp = edgerouter_pb2.CascadeResponse(
            final_judgment=outcome.final_judgment.value,
            edge_confidence=outcome.edge_confidence,
            total_latency_ms=outcome.total_latency_ms,
        )
        if outcome.edge_analysis:
            resp.edge_analysis.CopyFrom(
                analysis_to_proto(outcome.edge_analysis, edgerouter_pb2.AnalysisResult)
            )
        if outcome.cloud_analysis:
            resp.cloud_analysis.CopyFrom(
                analysis_to_proto(outcome.cloud_analysis, edgerouter_pb2.AnalysisResult)
            )
        return resp

    async def StreamAnalyze(self, request_iterator, context):
        """Bidirectional streaming: client sends frames, server returns routing results."""
        history = deque(maxlen=20)

        async for request in request_iterator:
            vo = vision_from_proto(request.vision_output)
            ctx = context_from_proto(request.context)
            history.append(vo)

            decision = self.router.route(vo, ctx)
            outcome = await self.cascade.execute(
                vo, ctx, decision, list(history) if len(history) >= 3 else None,
            )

            resp = edgerouter_pb2.CascadeResponse(
                final_judgment=outcome.final_judgment.value,
                edge_confidence=outcome.edge_confidence,
                total_latency_ms=outcome.total_latency_ms,
            )
            if outcome.edge_analysis:
                resp.edge_analysis.CopyFrom(
                    analysis_to_proto(outcome.edge_analysis, edgerouter_pb2.AnalysisResult)
                )
            if outcome.cloud_analysis:
                resp.cloud_analysis.CopyFrom(
                    analysis_to_proto(outcome.cloud_analysis, edgerouter_pb2.AnalysisResult)
                )
            yield resp


class AnalyzerServicer:
    """Implements the AnalyzerService gRPC interface (direct analyzer access)."""

    def __init__(self, edge: EdgeAnalyzer, cloud: CloudAnalyzer):
        self.edge = edge
        self.cloud = cloud

    async def AnalyzeEdge(self, request, context):
        vo = vision_from_proto(request.vision_output)
        history = [vision_from_proto(h) for h in request.recent_history] or None
        result = await self.edge.analyze(vo, history)
        return edgerouter_pb2.AnalyzeResponse(
            result=analysis_to_proto(result, edgerouter_pb2.AnalysisResult),
        )

    async def AnalyzeCloud(self, request, context):
        vo = vision_from_proto(request.vision_output)
        history = [vision_from_proto(h) for h in request.recent_history] or None
        edge_draft = analysis_from_proto(request.edge_draft) if request.HasField("edge_draft") else None
        result = await self.cloud.analyze(vo, history, edge_draft=edge_draft)
        return edgerouter_pb2.AnalyzeResponse(
            result=analysis_to_proto(result, edgerouter_pb2.AnalysisResult),
        )


async def serve(port: int = 50051):
    """Start the gRPC server."""
    if edgerouter_pb2 is None:
        raise RuntimeError(
            "Proto stubs not found. Run: bash scripts/compile_proto.sh"
        )

    config = AppConfig()
    edge = EdgeAnalyzer(config=config.edge_analyzer)
    cloud = CloudAnalyzer(config=config.cloud_analyzer)
    router = RouterEngine(config.router)
    cascade = CascadeExecutor(edge, cloud, config.router)

    server = grpc.aio.server()
    edgerouter_pb2_grpc.add_EdgeRouterServiceServicer_to_server(
        EdgeRouterServicer(router, cascade, edge, cloud), server,
    )
    edgerouter_pb2_grpc.add_AnalyzerServiceServicer_to_server(
        AnalyzerServicer(edge, cloud), server,
    )

    addr = f"[::]:{port}"
    server.add_insecure_port(addr)
    logger.info("gRPC server starting on %s", addr)
    await server.start()
    logger.info("gRPC server ready")

    try:
        await server.wait_for_termination()
    finally:
        await edge.close()
        await cloud.close()


def main():
    parser = argparse.ArgumentParser(description="EdgeRouter gRPC Server")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.port))


if __name__ == "__main__":
    main()
