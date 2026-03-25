"""Cloud analyzer: vLLM / OpenAI-compatible API backend."""

from __future__ import annotations

import json
import logging
import time

import httpx
from openai import AsyncOpenAI

from edgerouter.core.config import CloudAnalyzerConfig
from edgerouter.core.schema import AnalysisResult, Judgment, VisionOutput
from edgerouter.inference.base import AnalyzerBackend

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are an advanced industrial visual quality inspection model for complex analysis.
Analyze provided images (if available) and structured data to detect defects,
perform root cause analysis, and identify subtle multi-indicator anomaly patterns.
Output JSON only.

Safe ranges:
- Anomaly level: 30-70 normal, <30 or >70 anomalous, <5 or >95 critical
- Secondary metric: <0.15 normal, 0.15-0.35 warning, >0.35 anomalous
- Texture irregularity: <0.08 normal, 0.08-0.20 warning, >0.20 anomalous
- Surface uniformity: >0.85 normal, 0.65-0.85 warning, <0.65 anomalous
- Anomaly score: <0.2 normal, 0.2-0.5 warning, >0.5 anomalous
- Color: grey tone (R≈G≈B≈180) normal, red/yellow shift anomalous

Judgment:
- "normal": all indicators within normal range
- "warning": any indicator in warning range, or anomaly level 30-35/65-70
- "alarm": any indicator anomalous, or multiple warnings, or anomaly level <25/>75

You may receive an edge_draft for reference, but make your own independent judgment.

Output strict JSON (pick one value, no alternatives):
{"judgment":"normal/warning/alarm","confidence":0-1,"action":"maintain/adjust_flow/reduce_input/increase_cooling/emergency_stop","reasoning":"analysis reason","root_cause":"root cause (when anomalous)"}
"""


def _build_user_prompt(
    vision_output: VisionOutput,
    recent_history: list[VisionOutput] | None = None,
    edge_draft: AnalysisResult | None = None,
) -> str:
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
        levels = [o.anomaly_level for o in recent_history[-10:]]
        secondaries = [o.secondary_metric for o in recent_history[-10:]]
        lines.append(f"\nRecent 10 frames anomaly level trend: {[round(l, 1) for l in levels]}")
        lines.append(f"Recent 10 frames secondary metric trend: {[round(t, 3) for t in secondaries]}")

    if edge_draft:
        lines.append(f"\nEdge model preliminary judgment:")
        lines.append(f"- Judgment: {edge_draft.judgment.value}")
        lines.append(f"- Confidence: {edge_draft.confidence:.3f}")
        lines.append(f"- Suggested action: {edge_draft.suggested_action}")
        lines.append(f"- Reasoning: {edge_draft.reasoning}")

    lines.append("\nPerform deep analysis and output JSON. /no_think")
    return "\n".join(lines)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_response(text: str) -> dict:
    import re

    text = _strip_thinking(text)
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


class CloudAnalyzer(AnalyzerBackend):
    """Cloud analyzer backed by vLLM or OpenAI API."""

    def __init__(self, config: CloudAnalyzerConfig | None = None):
        self.config = config or CloudAnalyzerConfig()
        self._init_client()

    def _init_client(self):
        if self.config.use_openai_fallback and self.config.openai_api_key:
            self._client = AsyncOpenAI(
                api_key=self.config.openai_api_key,
            )
            self._model = self.config.openai_model
        else:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
            self._model = self.config.model

    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        t_start = time.perf_counter()

        user_prompt = _build_user_prompt(vision_output, recent_history, edge_draft)

        # Build user content — multimodal if image is available (OpenAI format)
        user_content: str | list = user_prompt
        if vision_output.image_path:
            try:
                from edgerouter.scenarios.image_loader import encode_image_base64
                image_b64 = encode_image_base64(vision_output.image_path)
                user_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": user_prompt},
                ]
            except Exception as e:
                logger.warning("Failed to load image %s: %s", vision_output.image_path, e)

        try:
            # Disable Qwen3.5 thinking mode for faster inference on vLLM
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body=extra_body,
            )
            content = completion.choices[0].message.content or ""
            parsed = _parse_response(content)

            raw_judgment = parsed.get("judgment", "normal").lower().strip()
            if raw_judgment not in ("normal", "warning", "alarm"):
                raw_judgment = "warning"

            latency_ms = (time.perf_counter() - t_start) * 1000
            return AnalysisResult(
                judgment=Judgment(raw_judgment),
                confidence=float(parsed.get("confidence", 0.5)),
                suggested_action=parsed.get("action", "maintain"),
                reasoning=parsed.get("reasoning", ""),
                root_cause=parsed.get("root_cause", ""),
                latency_ms=latency_ms,
                source="cloud",
            )

        except Exception as e:
            logger.error("Cloud analyzer failed: %s", e)
            latency_ms = (time.perf_counter() - t_start) * 1000
            # Cloud failure: if we have an edge draft, trust it
            if edge_draft:
                return AnalysisResult(
                    judgment=edge_draft.judgment,
                    confidence=edge_draft.confidence * 0.8,
                    suggested_action=edge_draft.suggested_action,
                    reasoning=f"Cloud fallback to edge draft: {e}",
                    latency_ms=latency_ms,
                    source="cloud",
                )
            return AnalysisResult(
                judgment=Judgment.WARNING,
                confidence=0.3,
                suggested_action="maintain",
                reasoning=f"Cloud analyzer error: {e}",
                latency_ms=latency_ms,
                source="cloud",
            )

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    async def close(self):
        await self._client.close()
