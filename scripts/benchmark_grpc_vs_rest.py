"""Benchmark gRPC vs REST serialization and round-trip latency.

This script compares:
  1. Protobuf vs JSON serialization overhead (offline, no network)
  2. gRPC vs REST round-trip latency (requires both servers running)

Usage:
    # Serialization benchmark only (no servers needed):
    python scripts/benchmark_grpc_vs_rest.py --serialization-only

    # Full benchmark (start servers first):
    #   Terminal 1: python -m edgerouter.server.grpc_server --port 50051
    #   Terminal 2: uvicorn edgerouter.server.api:app --port 8080
    python scripts/benchmark_grpc_vs_rest.py --requests 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgerouter.core.schema import VisionOutput


def benchmark_serialization(n: int = 1000):
    """Compare JSON vs protobuf serialization speed and size."""
    print(f"\n{'='*60}")
    print(f"Serialization Benchmark ({n} iterations)")
    print(f"{'='*60}")

    # Create sample VisionOutput
    vo = VisionOutput(
        timestamp=1711234567.89,
        anomaly_level=52.3,
        measurement_confidence=0.87,
        color_rgb=(180, 120, 80),
        secondary_metric=0.42,
        texture_irregularity=0.15,
        surface_uniformity=0.78,
        anomaly_score=0.35,
        anomaly_confidence=0.82,
        inference_latency_ms=18.5,
    )

    # JSON serialization
    json_data = vo.to_dict()
    t0 = time.perf_counter()
    for _ in range(n):
        encoded = json.dumps(json_data).encode("utf-8")
    json_encode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(n):
        json.loads(encoded)
    json_decode_ms = (time.perf_counter() - t0) * 1000

    json_size = len(encoded)

    # Protobuf serialization
    try:
        from edgerouter.proto import edgerouter_pb2
        from edgerouter.server.converter import vision_to_proto, vision_from_proto

        pb_msg = vision_to_proto(vo, edgerouter_pb2.VisionOutput)

        t0 = time.perf_counter()
        for _ in range(n):
            serialized = pb_msg.SerializeToString()
        proto_encode_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            msg = edgerouter_pb2.VisionOutput()
            msg.ParseFromString(serialized)
        proto_decode_ms = (time.perf_counter() - t0) * 1000

        proto_size = len(serialized)
    except ImportError:
        print("  Proto stubs not compiled. Run: bash scripts/compile_proto.sh")
        print("  Skipping protobuf benchmark.")
        proto_encode_ms = proto_decode_ms = 0
        proto_size = 0

    print(f"\n  {'Metric':<25s} {'JSON':>12s} {'Protobuf':>12s} {'Ratio':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Encode (ms/' + str(n) + ')':<25s} {json_encode_ms:>11.2f}  {proto_encode_ms:>11.2f}  "
          f"{json_encode_ms/max(proto_encode_ms, 0.001):>9.1f}x")
    print(f"  {'Decode (ms/' + str(n) + ')':<25s} {json_decode_ms:>11.2f}  {proto_decode_ms:>11.2f}  "
          f"{json_decode_ms/max(proto_decode_ms, 0.001):>9.1f}x")
    print(f"  {'Message size (bytes)':<25s} {json_size:>12d}  {proto_size:>12d}  "
          f"{json_size/max(proto_size, 1):>9.1f}x")

    return {
        "json_encode_ms": round(json_encode_ms, 2),
        "json_decode_ms": round(json_decode_ms, 2),
        "json_size_bytes": json_size,
        "proto_encode_ms": round(proto_encode_ms, 2),
        "proto_decode_ms": round(proto_decode_ms, 2),
        "proto_size_bytes": proto_size,
    }


async def benchmark_roundtrip(n: int = 50, grpc_addr: str = "localhost:50051",
                               rest_url: str = "http://localhost:8080"):
    """Compare gRPC vs REST round-trip latency (requires running servers)."""
    print(f"\n{'='*60}")
    print(f"Round-Trip Benchmark ({n} requests)")
    print(f"{'='*60}")

    import httpx

    vo = VisionOutput(
        timestamp=time.time(), anomaly_level=52.3, measurement_confidence=0.87,
        color_rgb=(180, 120, 80), secondary_metric=0.42, texture_irregularity=0.15,
        surface_uniformity=0.78, anomaly_score=0.35, anomaly_confidence=0.82,
    )

    # REST benchmark
    rest_times = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(n):
                t0 = time.perf_counter()
                resp = await client.post(
                    f"{rest_url}/analyze/edge",
                    json={"vision_output": vo.to_dict(), "source": "edge"},
                )
                rest_times.append((time.perf_counter() - t0) * 1000)
                if resp.status_code != 200:
                    print(f"  REST error: {resp.status_code}")
                    break
        rest_ok = True
    except Exception as e:
        print(f"  REST server not available: {e}")
        rest_ok = False

    # gRPC benchmark
    grpc_times = []
    try:
        from edgerouter.server.grpc_client import EdgeRouterClient
        client = EdgeRouterClient(grpc_addr)
        for _ in range(n):
            t0 = time.perf_counter()
            await client.analyze_edge(vo)
            grpc_times.append((time.perf_counter() - t0) * 1000)
        await client.close()
        grpc_ok = True
    except Exception as e:
        print(f"  gRPC server not available: {e}")
        grpc_ok = False

    if rest_ok and grpc_ok:
        import numpy as np
        rest_arr = np.array(rest_times)
        grpc_arr = np.array(grpc_times)

        print(f"\n  {'Metric':<25s} {'REST':>12s} {'gRPC':>12s} {'Speedup':>10s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'p50 latency (ms)':<25s} {np.median(rest_arr):>11.2f}  {np.median(grpc_arr):>11.2f}  "
              f"{np.median(rest_arr)/max(np.median(grpc_arr), 0.001):>9.1f}x")
        print(f"  {'p99 latency (ms)':<25s} {np.percentile(rest_arr,99):>11.2f}  {np.percentile(grpc_arr,99):>11.2f}  "
              f"{np.percentile(rest_arr,99)/max(np.percentile(grpc_arr,99), 0.001):>9.1f}x")
        print(f"  {'mean latency (ms)':<25s} {rest_arr.mean():>11.2f}  {grpc_arr.mean():>11.2f}  "
              f"{rest_arr.mean()/max(grpc_arr.mean(), 0.001):>9.1f}x")
    elif rest_ok:
        print("  (gRPC server not running — REST-only results)")
    elif grpc_ok:
        print("  (REST server not running — gRPC-only results)")
    else:
        print("  Neither server is running. Start servers to run round-trip benchmark.")


def main():
    parser = argparse.ArgumentParser(description="gRPC vs REST Benchmark")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Iterations for serialization benchmark")
    parser.add_argument("--serialization-only", action="store_true",
                        help="Only run serialization benchmark (no servers needed)")
    parser.add_argument("--grpc-addr", default="localhost:50051")
    parser.add_argument("--rest-url", default="http://localhost:8080")
    args = parser.parse_args()

    results = benchmark_serialization(args.iterations)

    if not args.serialization_only:
        asyncio.run(benchmark_roundtrip(args.requests, args.grpc_addr, args.rest_url))

    print(f"\n{'='*60}")
    print("Summary: gRPC/protobuf provides lower serialization overhead")
    print("and connection multiplexing, critical for high-frequency")
    print("edge-to-cloud communication in industrial monitoring.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
