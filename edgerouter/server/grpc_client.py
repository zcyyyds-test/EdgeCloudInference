"""gRPC client wrapper for EdgeRouter services.

Usage as library:
    client = EdgeRouterClient("localhost:50051")
    health = await client.health_check()
    decision = await client.route(vision_output, context)
    outcome = await client.cascade(vision_output, context, decision)
    await client.close()
"""

from __future__ import annotations

import grpc

from edgerouter.core.schema import (
    AnalysisResult,
    ProcessContext,
    RoutingDecision,
    VisionOutput,
)
from edgerouter.server.converter import (
    analysis_from_proto,
    context_to_proto,
    decision_from_proto,
    decision_to_proto,
    vision_to_proto,
)

try:
    from edgerouter.proto import edgerouter_pb2, edgerouter_pb2_grpc
except ImportError:
    edgerouter_pb2 = None
    edgerouter_pb2_grpc = None


class EdgeRouterClient:
    """Async gRPC client for the EdgeRouter service."""

    def __init__(self, target: str = "localhost:50051"):
        if edgerouter_pb2 is None:
            raise RuntimeError("Proto stubs not found. Run: bash scripts/compile_proto.sh")
        self.channel = grpc.aio.insecure_channel(target)
        self.router_stub = edgerouter_pb2_grpc.EdgeRouterServiceStub(self.channel)
        self.analyzer_stub = edgerouter_pb2_grpc.AnalyzerServiceStub(self.channel)

    async def close(self):
        await self.channel.close()

    async def health_check(self) -> dict:
        resp = await self.router_stub.HealthCheck(edgerouter_pb2.Empty())
        return {
            "status": resp.status,
            "edge_available": resp.edge_available,
            "cloud_available": resp.cloud_available,
        }

    async def detect(self, scenario_key: str = "", difficulty: str = "") -> dict:
        resp = await self.router_stub.Detect(edgerouter_pb2.DetectRequest(
            scenario_key=scenario_key, difficulty=difficulty,
        ))
        return {
            "scenario_name": resp.scenario_name,
            "scenario_difficulty": resp.scenario_difficulty,
        }

    async def route(self, vo: VisionOutput, ctx: ProcessContext) -> RoutingDecision:
        req = edgerouter_pb2.RouteRequest(
            vision_output=vision_to_proto(vo, edgerouter_pb2.VisionOutput),
            context=context_to_proto(ctx, edgerouter_pb2.ProcessContext),
        )
        resp = await self.router_stub.Route(req)
        return decision_from_proto(resp.decision)

    async def cascade(
        self,
        vo: VisionOutput,
        ctx: ProcessContext,
        decision: RoutingDecision,
        history: list[VisionOutput] | None = None,
    ) -> dict:
        """Run full cascade analysis. Returns dict with results."""
        req = edgerouter_pb2.CascadeRequest(
            vision_output=vision_to_proto(vo, edgerouter_pb2.VisionOutput),
            context=context_to_proto(ctx, edgerouter_pb2.ProcessContext),
            routing_decision=decision_to_proto(decision, edgerouter_pb2.RoutingDecision),
        )
        if history:
            for h in history:
                req.recent_history.append(
                    vision_to_proto(h, edgerouter_pb2.VisionOutput)
                )
        resp = await self.router_stub.Cascade(req)
        return {
            "final_judgment": resp.final_judgment,
            "edge_confidence": resp.edge_confidence,
            "total_latency_ms": resp.total_latency_ms,
            "edge_analysis": analysis_from_proto(resp.edge_analysis),
            "cloud_analysis": analysis_from_proto(resp.cloud_analysis),
        }

    async def analyze_edge(self, vo: VisionOutput) -> AnalysisResult:
        req = edgerouter_pb2.AnalyzeRequest(
            vision_output=vision_to_proto(vo, edgerouter_pb2.VisionOutput),
        )
        resp = await self.analyzer_stub.AnalyzeEdge(req)
        return analysis_from_proto(resp.result)

    async def analyze_cloud(
        self, vo: VisionOutput, edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        req = edgerouter_pb2.AnalyzeRequest(
            vision_output=vision_to_proto(vo, edgerouter_pb2.VisionOutput),
        )
        if edge_draft:
            from edgerouter.server.converter import analysis_to_proto
            req.edge_draft.CopyFrom(
                analysis_to_proto(edge_draft, edgerouter_pb2.AnalysisResult)
            )
        resp = await self.analyzer_stub.AnalyzeCloud(req)
        return analysis_from_proto(resp.result)
