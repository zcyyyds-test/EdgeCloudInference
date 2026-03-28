"""Global configuration for EdgeRouter."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class RouterConfig(BaseSettings):
    """Router engine configuration."""

    # Confidence thresholds
    confidence_threshold: float = Field(0.7, description="Below this → cascade to cloud")
    clearly_normal_anomaly_max: float = 0.2
    clearly_normal_confidence_min: float = 0.85
    complex_anomaly_score_min: float = 0.8
    complex_correlated_anomalies_min: int = 2

    # Safety thresholds
    critical_anomaly_upper: float = 95.0   # upper critical threshold
    critical_anomaly_lower: float = 5.0    # lower critical threshold
    critical_anomaly_rate: float = 10.0    # max change rate per second

    # Safe operating range
    safe_anomaly_center: float = 50.0      # center of safe range
    safe_anomaly_range: float = 40.0       # safe range width (±20 from center)

    # Confidence estimation method: "output_prob", "self_verify", "temporal", "combined"
    confidence_method: str = "edge_llm"   # use edge model's own confidence for cascade

    # Speculative prefetch: fire edge+cloud in parallel for cascade tier
    enable_speculative_prefetch: bool = False
    speculative_anomaly_threshold: float = Field(
        0.3, description="Anomaly score fallback trigger when no trend history available",
    )

    model_config = {"env_prefix": "EDGEROUTER_"}


class EdgeAnalyzerConfig(BaseSettings):
    """Edge analyzer configuration — supports Qwen3.5 multimodal."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:0.8b"             # Qwen3.5 0.8B multimodal — lightweight edge model
    timeout: float = 30.0
    temperature: float = 0.1
    max_tokens: int = 1024
    num_gpu: int = -1                       # GPU layers: -1=auto, 0=CPU-only

    model_config = {"env_prefix": "EDGE_ANALYZER_"}


class CloudAnalyzerConfig(BaseSettings):
    """Cloud analyzer configuration (vLLM or OpenAI-compatible API)."""

    base_url: str = "http://localhost:8000/v1"   # vLLM default
    api_key: str = "EMPTY"                        # vLLM doesn't require a key
    model: str = "Qwen/Qwen3.5-27B"              # Qwen3.5 multimodal cloud model
    timeout: float = 60.0
    temperature: float = 0.1
    max_tokens: int = 1024

    # Fallback: use OpenAI GPT-4o if vLLM unavailable
    use_openai_fallback: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    model_config = {"env_prefix": "CLOUD_ANALYZER_"}


class VisionConfig(BaseSettings):
    """Vision model configuration."""

    model_params: int = 2_500_000              # MobileNetV3-Small level
    inference_latency_ms: float = 18.0
    noise_scale_normal: float = 0.3            # cm noise for normal scenarios
    noise_scale_marginal: float = 1.0
    noise_scale_anomalous: float = 2.0
    noise_scale_critical: float = 3.0

    model_config = {"env_prefix": "VISION_"}


class ServerConfig(BaseSettings):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    model_config = {"env_prefix": "SERVER_"}


class AppConfig(BaseSettings):
    """Top-level application config, composes sub-configs."""

    router: RouterConfig = Field(default_factory=RouterConfig)
    edge_analyzer: EdgeAnalyzerConfig = Field(default_factory=EdgeAnalyzerConfig)
    cloud_analyzer: CloudAnalyzerConfig = Field(default_factory=CloudAnalyzerConfig)
    vision_config: VisionConfig = Field(default_factory=VisionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
