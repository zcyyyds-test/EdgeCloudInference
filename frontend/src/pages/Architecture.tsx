export default function Architecture() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">System Architecture</h1>
        <p className="text-sm text-slate-500 mt-1">
          5-tier confidence-driven edge-cloud inference routing
        </p>
      </div>

      {/* Architecture diagram (ASCII-style in terminal card) */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
        <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
          DATA FLOW
        </div>
        <pre className="text-xs font-mono text-slate-400 leading-relaxed overflow-x-auto">
{`
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         VISION INPUT                               │
 │  Camera / MVTec AD Image → VisionModel / Real Detector    │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                      5-TIER ROUTING ENGINE                          │
 │                                                                      │
 │  T0 ─ Safety Check ──────── anomaly_level < 5 or > 95 → ALARM     │
 │  │                                                                    │
 │  T1 ─ Clearly Normal ───── score < 0.2 & conf > 0.85 → EDGE ONLY │
 │  │                                                                    │
 │  T2 ─ Complex Pattern ──── score > 0.8 & multi-anomaly → CLOUD    │
 │  │                                                                    │
 │  T3 ─ Confidence Gate ──── conf < threshold(0.7) → CASCADE         │
 │  │                                                                    │
 │  T4 ─ Default ─────────── remaining → EDGE ONLY                    │
 └──────────┬──────────────────────────┬───────────────────────────────┘
            │                          │
     ┌──────▼──────┐          ┌────────▼────────┐
     │  EDGE TIER  │          │   CLOUD TIER    │
     │             │          │                 │
     │ Qwen3.5-4B │──drift──▶│ Qwen3.5-27B    │
     │ quantized   │ cascade  │ vLLM (TP=2)    │
     │ ~92ms P50   │          │ ~800ms P50      │
     │             │          │                 │
     │ Multimodal  │          │ Root cause      │
     │ (vision+    │          │ analysis +      │
     │  text)      │          │ deep reasoning  │
     └──────┬──────┘          └────────┬────────┘
            │                          │
            └────────────┬─────────────┘
                         ▼
          ┌──────────────────────────────┐
          │      ANALYSIS RESULT         │
          │  judgment / confidence /      │
          │  action / reasoning          │
          │  → Control Loop / Dashboard  │
          └──────────────────────────────┘
`}
        </pre>
      </div>

      {/* Component details */}
      <div className="grid grid-cols-2 gap-4">
        {/* Router tiers */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
          <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
            ROUTING TIERS
          </div>
          <div className="space-y-4">
            {[
              {
                tier: 'T0',
                name: 'Safety Check',
                color: 'text-rose-400',
                border: 'border-rose-500/20',
                desc: 'Critical anomaly detection (level <5 or >95, rate >10/s). Immediate alarm, bypasses all analysis.',
              },
              {
                tier: 'T1',
                name: 'Clearly Normal',
                color: 'text-emerald-400',
                border: 'border-emerald-500/20',
                desc: 'Low anomaly score (<0.2) with high confidence (>0.85). Handles ~40% of traffic. Edge-only, no cloud needed.',
              },
              {
                tier: 'T2',
                name: 'Complex Pattern',
                color: 'text-amber-400',
                border: 'border-amber-500/20',
                desc: 'High anomaly score (>0.8) with multiple correlated anomalies (≥2). Routes directly to cloud for deep analysis.',
              },
              {
                tier: 'T3',
                name: 'Confidence Gate',
                color: 'text-blue-400',
                border: 'border-blue-500/20',
                desc: 'Edge confidence below threshold (0.7). Cascades to cloud — edge draft sent as reference for cloud to refine.',
              },
              {
                tier: 'T4',
                name: 'Default Edge',
                color: 'text-cyan-400',
                border: 'border-cyan-500/20',
                desc: 'All remaining requests handled by edge. Covers moderate-confidence scenarios where edge is sufficient.',
              },
            ].map((t) => (
              <div key={t.tier} className={`border ${t.border} rounded-lg p-3 bg-slate-950/30`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs font-mono font-bold ${t.color}`}>{t.tier}</span>
                  <span className="text-sm text-slate-300">{t.name}</span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{t.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Tech stack */}
        <div className="space-y-4">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
            <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
              TECHNOLOGY STACK
            </div>
            <div className="space-y-3">
              {[
                { label: 'Edge Model', value: 'Qwen3.5-4B (quantized)', detail: 'Multimodal (vision + text), ~3.4GB VRAM' },
                { label: 'Cloud Model', value: 'Qwen3.5-27B via vLLM', detail: 'Tensor parallel, continuous batching, OpenAI-compatible API' },
                { label: 'Framework', value: 'Python 3.11+ / FastAPI', detail: 'Async, Prometheus metrics, gRPC support' },
                { label: 'Dataset', value: 'MVTec AD', detail: '15 categories, 5000+ images, public benchmark' },
                { label: 'Hardware', value: 'Edge device + GPU server', detail: 'Edge: ARM/x86 (8GB+) / Cloud: GPU with CUDA' },
              ].map((item) => (
                <div key={item.label} className="flex items-start gap-3">
                  <div className="text-xs text-slate-600 font-mono w-24 shrink-0 pt-0.5">{item.label}</div>
                  <div>
                    <div className="text-sm text-slate-300">{item.value}</div>
                    <div className="text-[10px] text-slate-600">{item.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
            <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
              CONFIDENCE ESTIMATION
            </div>
            <div className="space-y-2 text-xs text-slate-400">
              <p>Combined method using three signals:</p>
              <div className="space-y-1.5 mt-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan-500" />
                  <span><strong className="text-slate-300">Output probability</strong> — model's self-reported confidence</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span><strong className="text-slate-300">Self-verification</strong> — anomaly score proximity to safe range center</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span><strong className="text-slate-300">Temporal smoothing</strong> — stability across recent frames (σ-based)</span>
                </div>
              </div>
              <p className="mt-2 text-slate-600">
                Final confidence = 0.4 × output + 0.3 × self_verify + 0.3 × temporal
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
