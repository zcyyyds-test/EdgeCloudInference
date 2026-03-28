# EdgeCloudInference

**Edge-cloud collaborative inference system for industrial visual anomaly detection.**

EdgeCloudInference dynamically distributes visual anomaly detection queries between a lightweight **multimodal edge LLM** and a powerful **cloud LLM** based on confidence estimation, vision features, and safety constraints. A 5-tier hierarchical decision engine with cascade execution reduces cloud API costs by **86.7%** while maintaining **100% accuracy on safety-critical escalations**.

The system is domain-agnostic — while validated on industrial inspection (defect detection, surface quality, process monitoring), the routing architecture generalizes to any visual anomaly detection task where edge-cloud cost-accuracy trade-offs matter.

> **Background**: This project originated from research during my Master's program in industrial IoT. Proprietary deployment data and sensitive configurations have been removed; the codebase has been restructured and optimized for public release. Benchmark scenarios use parameterized anomaly templates (16 patterns + Markov state transitions) in place of production data due to data privacy constraints.

| | |
|---|---|
| **Edge Model** | Qwen3.5-0.8B (multimodal, quantized) |
| **Cloud Model** | Qwen3.5-27B (vLLM, tensor parallel) |
| **Tests** | 132 passing across 9 test files |
| **Code** | ~11,000 lines (Python + TypeScript) |

## Architecture

> Full Mermaid diagrams: [`docs/architecture.md`](docs/architecture.md)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (React)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  FastAPI / gRPC Server                           │
│       REST :8080  ·  gRPC :50051  ·  /metrics (Prometheus)      │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────────┐
│ Vision Model│  │ Router Engine  │  │ Cascade Executor  │
│             │  │   (5-tier)     │  │ edge → cloud      │
└──────┬──────┘  └───┬────────────┘  └──┬─────────┬──────┘
       │             │                  │         │
       │    ┌────────┴─────────┐        │         │
       │    │ T0: Emergency     │        ▼         ▼
       │    │ T1: Data Security │  ┌──────────┐ ┌───────────────┐
       │    │ T2: Clearly Normal│  │  Edge    │ │   Cloud       │
       │    │ T3: Complex→Cloud │  │  Qwen3.5 │ │   vLLM        │
       │    │ T4: Grey Zone     │  │  0.8B    │ │   Qwen3.5-27B │
       │    └───────────────────┘  │(quantized)│ │   TP=2        │
       │                           └──────────┘ └───────────────┘
       ▼
┌──────────────────────────────────┐
│ 16 Anomaly Pattern Templates     │
│ MVTec AD Image Loader (optional) │
│ Markov State Timeline Generator  │
│ Control Signal Analysis          │
└──────────────────────────────────┘
```

### 5-Tier Routing

| Tier | Name | Condition | Latency |
|------|------|-----------|---------|
| T0 | **Emergency** | Safety-critical anomaly (out of bounds) | <1ms |
| T1 | **Data Security** | Sensitive data → keep on-edge | <1ms |
| T2 | **Clearly Normal** | Low anomaly score + high confidence | ~0ms |
| T3 | **Complex → Cloud** | Multi-indicator anomaly pattern | ~0ms |
| T4 | **Edge LLM Gate** | Edge 0.8B runs first; its confidence decides escalation | edge + cloud |

### Multimodal Inference

EdgeCloudInference uses **Qwen3.5** multimodal models that natively process both images and structured metrics:

- **Edge**: Qwen3.5-0.8B (quantized) served with `format="json"` for deterministic structured output. When an `image_path` is available, the model receives the raw image alongside structured detection data. Its self-reported confidence drives cascade escalation.
- **Cloud**: vLLM serves Qwen3.5-27B via OpenAI-compatible API with tensor parallelism. Achieves ~27 tok/s per analysis — a 3.9× speedup over transformers baseline.

## Key Results

### Real LLM Benchmark (30 scenarios, Qwen3.5-0.8B edge + Qwen3.5-27B cloud)

| Metric | Value |
|--------|------:|
| Overall accuracy | **80.0%** |
| Miss rate | 37.5% |
| False alarm rate | **0%** |
| Cloud cost savings | **86.7%** |
| p50 latency | 2,296 ms |
| Confidence mean / std | 0.81 / 0.16 |

### Edge Model Ablation

| Model | Params | Accuracy | P50 Latency | Cloud Savings |
|-------|--------|----------|-------------|---------------|
| **Qwen3.5-0.8B** | **0.8B** | **62.3%** | **280ms** | **82%** |
| Qwen3.5-4B | 4B | 84.5% | 1500ms | 70% |

Qwen3.5-0.8B is the default edge model: sub-300ms latency enables real-time edge deployment, and its self-reported confidence naturally drives cascade escalation to the 27B cloud model.

### Throughput Optimization (async concurrent inference + speculative cloud prefetch)

| Configuration | Throughput (scen/s) | Speedup | Cascade Latency (ms) |
|---------------|--------------------:|--------:|---------------------:|
| Baseline (sequential) | 0.23 | 1.0× | 8,567 |
| Concurrent (×4) | 0.66 | **2.9×** | 11,740 |
| Speculative prefetch only | 0.30 | 1.3× | 7,550 (**−12%**) |
| Concurrent + speculative | 0.58 | 2.5× | 12,059 |

Three cross-platform optimizations (pure Python asyncio, no hardware dependency):
- **vLLM prefix caching**: `--enable-prefix-caching` shares system prompt KV cache across requests
- **Concurrent edge inference**: `asyncio.gather` + `asyncio.Semaphore` for parallel scenario processing
- **Speculative cloud prefetch**: fires edge + cloud in parallel when anomaly is predicted; cancels cloud if edge is confident

> Concurrent processing provides the dominant throughput gain (2.9×). Speculative prefetch reduces cascade latency by 12% but its benefit is bounded by the edge/cloud latency ratio (~2s/~7s).

### WAN Latency Resilience

| WAN Delay | p50 Latency | Cloud Savings |
|-----------|-------------|---------------|
| 10ms | 628ms | 92.9% |
| 50ms | 681ms | 86.7% |
| 200ms | 849ms | 86.7% |
| 500ms | 1124ms | 86.7% |

Cloud savings remain above 86% even at 500ms WAN delay.

## Project Structure

```
edgerouter/                    # Core Python package (~9,400 lines)
├── core/                      # Schema, config, Prometheus metrics
├── control/                   # PID/deadband control for process feedback
├── dashboard/                 # Monitoring dashboard (Streamlit)
├── eval/                      # Benchmark framework, workloads, analysis
├── inference/                 # Edge + Cloud LLM analyzers
├── learning/                  # Online threshold learning + feedback collector
├── proto/                     # gRPC protobuf definitions (2 services, 12 messages)
├── router/                    # 5-tier engine, cascade, confidence, safety, prefetch
├── server/                    # FastAPI REST + gRPC async server + static serving
└── scenarios/                 # 16 anomaly templates, Markov timeline, vision model

frontend/                      # React dashboard for visualization

scripts/
├── serve_cloud.py             # Cloud server (vLLM recommended, transformers fallback)
├── download_mvtec.py          # Download MVTec AD dataset for real image testing
├── run_model_ablation.py      # Edge model size comparison
├── run_edge_cloud_benchmark.py# Full edge-cloud benchmark with WAN delay
├── run_real_threshold_sweep.py# Two-phase threshold optimization
├── run_wan_latency_sweep.py   # WAN delay impact measurement
├── run_control.py             # Physics-based control analysis
├── analyze_confidence.py      # Offline confidence calibration
└── benchmark_grpc_vs_rest.py  # gRPC vs REST benchmark

monitoring/                    # Prometheus config + 6-panel Grafana dashboard
experiments/                   # All experiment results (JSON + PNG)
tests/                         # 132 tests across 9 files
```

## Quick Start

### Prerequisites

- Python 3.11+
- Edge LLM serving runtime (e.g., llama.cpp, TensorRT-LLM)
- GPU with CUDA 12+ (for cloud model)

### Installation

```bash
git clone https://github.com/zcyyyds-test/EdgeCloudInference.git
cd EdgeCloudInference
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/ -v    # 132 tests, ~3s
```

### Start the Server (Standalone Mode)

```bash
# Serves both API and React frontend on :8080
uvicorn edgerouter.server.api:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` for the web dashboard.

### Run with Real LLMs

```bash
# 1. Start edge model (quantized Qwen3.5-0.8B on edge device)
# Adapt to your edge runtime (llama.cpp, Ollama, TensorRT-LLM, etc.)

# 2. Start cloud model
python scripts/serve_cloud.py --mode vllm --model Qwen/Qwen3.5-27B --tp 2

# 3. Start EdgeCloudInference server
uvicorn edgerouter.server.api:app --host 0.0.0.0 --port 8080

# 4. Run experiments
python scripts/run_edge_cloud_benchmark.py --scenarios 50
python scripts/run_model_ablation.py
python scripts/run_wan_latency_sweep.py --scenarios 30 --delays 10,50,200,500
```

### Monitoring

```bash
# Prometheus + Grafana
docker-compose -f monitoring/docker-compose.yml up -d

# Metrics available at :8080/metrics
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Edge Inference | Qwen3.5-0.8B (quantized) | Lightweight multimodal on-device analysis |
| Cloud Inference | vLLM + Qwen3.5-27B | High-throughput cloud analysis with tensor parallelism |
| API Server | FastAPI + gRPC | REST + RPC with Prometheus metrics |
| Testing | pytest (132 tests) | Full coverage of routing logic |
| Scenarios | NumPy + PID control | 16 anomaly templates + physics model |

## Known Limitations

- Edge accuracy (80%) is limited by base model capability — addressable via LoRA fine-tuning on domain data
- Edge p50 latency (~280ms) depends on edge hardware; TensorRT-LLM or further quantization can improve this
- Cloud-escalated scenarios achieve 100% accuracy, confirming the routing architecture is sound

## License

MIT
