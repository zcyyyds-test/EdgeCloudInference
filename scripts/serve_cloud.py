"""Cloud model server with vLLM (recommended) or transformers fallback.

Modes:
    vLLM (default): High-throughput, OpenAI-compatible API with continuous batching.
    transformers: Fallback for environments where vLLM is not available.

Usage:
    # vLLM mode (recommended, requires vLLM >= 0.18.0, Linux/WSL only):
    python scripts/serve_cloud.py --mode vllm --model Qwen/Qwen3.5-27B --tp 2

    # On Windows server — run vLLM inside WSL:
    wsl -d Ubuntu-24.04 -- bash -c "source ~/vllm-env/bin/activate && \\
        VLLM_USE_DEEP_GEMM=0 python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.9 --trust-remote-code"

    # Transformers mode (fallback, Windows native):
    python scripts/serve_cloud.py --mode transformers --model D:/zcy/models/Qwen/Qwen3-14B --gpu 1

    # Check GPU/CUDA availability:
    python scripts/serve_cloud.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="EdgeRouter Cloud Model Server")

# Global model/tokenizer — loaded once at startup
_model = None
_tokenizer = None
_model_name = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 1024
    stream: bool = False


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "owned_by": "local"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    t0 = time.perf_counter()

    # Build prompt from messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # Disable Qwen3 thinking mode for faster JSON output
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(0.01, request.temperature),
            do_sample=request.temperature > 0,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip any residual <think>...</think> blocks
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
        },
    }


def _check_gpu():
    """Check CUDA and GPU availability."""
    print("Checking GPU environment...")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    try:
        import vllm
        print(f"  vLLM: {vllm.__version__}")
    except ImportError:
        print("  vLLM: not installed")


def _serve_vllm(model: str, port: int, tp: int, gpu_util: float):
    """Launch vLLM OpenAI-compatible API server."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    # Workaround for Blackwell GPUs (RTX PRO 6000 / SM120)
    env["VLLM_USE_DEEP_GEMM"] = "0"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server:")
    print(f"  Model: {model}")
    print(f"  Port: {port}")
    print(f"  Tensor parallel: {tp}")
    print(f"  GPU utilization: {gpu_util}")
    print(f"  VLLM_USE_DEEP_GEMM=0 (Blackwell workaround)")
    print(f"  Endpoint: http://localhost:{port}/v1/chat/completions")
    print()

    subprocess.run(cmd, env=env)


def _serve_transformers(model: str, port: int, gpu: int, dtype_str: str):
    """Launch transformers-based server (original fallback mode)."""
    global _model, _tokenizer, _model_name

    _model_name = model.rstrip("/\\").split("/")[-1].split("\\")[-1]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    print(f"Loading model: {model} on GPU {gpu} ({dtype_str})...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype,
        device_map=f"cuda:{gpu}",
        trust_remote_code=True,
    )
    _model.eval()
    print(f"Model loaded. Serving on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Cloud Model Server")
    parser.add_argument("--mode", choices=["vllm", "transformers"], default="vllm",
                        help="Server mode (default: vllm)")
    parser.add_argument("--model", required=False, default="Qwen/Qwen3.5-27B",
                        help="Model name or path (default: Qwen/Qwen3.5-27B)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=1, help="GPU device index (transformers mode)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size (vLLM mode)")
    parser.add_argument("--gpu-util", type=float, default=0.9, help="GPU memory utilization (vLLM)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--check", action="store_true", help="Check GPU availability and exit")
    args = parser.parse_args()

    if args.check:
        _check_gpu()
        return

    if args.mode == "vllm":
        _serve_vllm(args.model, args.port, args.tp, args.gpu_util)
    else:
        _serve_transformers(args.model, args.port, args.gpu, args.dtype)


if __name__ == "__main__":
    main()
