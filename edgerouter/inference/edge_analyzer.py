"""Edge analyzer: lightweight LLM backend for on-device inference."""

from __future__ import annotations

import json
import logging
import time

import httpx

from edgerouter.core.config import EdgeAnalyzerConfig
from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput
from edgerouter.inference.base import AnalyzerBackend

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an industrial visual quality inspection model deployed on an edge device.
Analyze the provided image (if available) and structured detection data to assess
whether the product/process is normal, warning, or alarm. Output JSON only.

Safe ranges:
- Anomaly level: 30-70 normal, <30 or >70 anomalous, <5 or >95 critical
- Secondary metric: <0.15 normal, 0.15-0.35 warning, >0.35 anomalous
- Texture irregularity: <0.08 normal, 0.08-0.20 warning, >0.20 anomalous
- Surface uniformity: >0.85 normal, 0.65-0.85 warning, <0.65 anomalous
- Anomaly score: <0.2 normal, 0.2-0.5 warning, >0.5 anomalous
- Color: grey tone (R≈G≈B≈180) normal, red/yellow shift anomalous

Judgment rules:
- "normal": no visible defects, all metrics within normal range
- "warning": minor anomaly detected, or metrics approaching threshold
- "alarm": clear defect/anomaly, or multiple metrics in abnormal range

Output strict JSON (pick one value, no alternatives):
{"judgment":"normal/warning/alarm","confidence":0-1,"action":"maintain/adjust_flow/reduce_input/emergency_stop","reasoning":"brief reason"}
"""


def _build_user_prompt(
    vision_output: VisionOutput,
    recent_history: list[VisionOutput] | None = None,
) -> str:
    """Build the user prompt from vision data."""
    lines = []

    if vision_output.image_path:
        lines.append("An image has been provided for visual inspection.")
        lines.append("")

    lines.extend([
        "Current detection data:",
        f"- Anomaly level: {vision_output.anomaly_level:.1f} (confidence: {vision_output.measurement_confidence:.2f})",
        f"- Secondary metric: {vision_output.secondary_metric:.3f}",
        f"- Color RGB: {vision_output.color_rgb}",
        f"- Texture irregularity: {vision_output.texture_irregularity:.3f}",
        f"- Surface uniformity: {vision_output.surface_uniformity:.3f}",
        f"- Anomaly score: {vision_output.anomaly_score:.3f}",
        f"- Anomaly confidence: {vision_output.anomaly_confidence:.3f}",
    ])

    if recent_history:
        levels = [o.anomaly_level for o in recent_history[-5:]]
        secondaries = [o.secondary_metric for o in recent_history[-5:]]
        lines.append(f"\nRecent 5 frames anomaly level: {[round(l, 1) for l in levels]}")
        lines.append(f"Recent 5 frames secondary metric: {[round(t, 3) for t in secondaries]}")

    lines.append("\nAnalyze current condition and output JSON. /no_think")
    return "\n".join(lines)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_response(text: str) -> dict:
    """Extract JSON from model response, handling thinking blocks and code fences."""
    import re

    text = _strip_thinking(text)

    # Strip markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try all JSON objects in the text, return the first valid one with required keys
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            obj = json.loads(match.group())
            if "judgment" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    # Fallback: try first { to last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from response: {text[:200]}")


class EdgeAnalyzer(AnalyzerBackend):
    """Edge analyzer backed by local LLM server (Qwen3.5 multimodal)."""

    def __init__(self, config: EdgeAnalyzerConfig | None = None):
        self.config = config or EdgeAnalyzerConfig()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        t_start = time.perf_counter()

        user_prompt = _build_user_prompt(vision_output, recent_history)

        # Build user message — add image if available (multimodal format)
        user_message: dict = {"role": "user", "content": user_prompt}
        if vision_output.image_path:
            try:
                from edgerouter.scenarios.image_loader import encode_image_base64
                image_b64 = encode_image_base64(vision_output.image_path)
                user_message["images"] = [image_b64]
            except Exception as e:
                logger.warning("Failed to load image %s: %s", vision_output.image_path, e)

        try:
            resp = await self._client.post(
                "/api/chat",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        user_message,
                    ],
                    "stream": False,
                    "format": "json",
                    "think": False,  # Disable Qwen3.5 thinking mode for direct JSON output
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        **({"num_gpu": self.config.num_gpu} if self.config.num_gpu >= 0 else {}),
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            parsed = _parse_response(content)

            # Normalize judgment value (handle "normal|warning|alarm" template leak)
            raw_judgment = parsed.get("judgment", "normal").lower().strip()
            if raw_judgment not in ("normal", "warning", "alarm"):
                raw_judgment = "warning"  # Conservative fallback

            latency_ms = (time.perf_counter() - t_start) * 1000
            return AnalysisResult(
                judgment=Judgment(raw_judgment),
                confidence=float(parsed.get("confidence", 0.5)),
                suggested_action=parsed.get("action", "maintain"),
                reasoning=parsed.get("reasoning", ""),
                latency_ms=latency_ms,
                source="edge",
            )

        except Exception as e:
            logger.error("Edge analyzer failed: %s", e)
            latency_ms = (time.perf_counter() - t_start) * 1000
            # Fail-safe: return conservative warning
            return AnalysisResult(
                judgment=Judgment.WARNING,
                confidence=0.3,
                suggested_action="maintain",
                reasoning=f"Edge analyzer error: {e}",
                latency_ms=latency_ms,
                source="edge",
            )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
