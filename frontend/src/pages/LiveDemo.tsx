import { useState } from 'react'
import { Play, RotateCcw, AlertTriangle, CheckCircle, AlertOctagon } from 'lucide-react'
import RoutingFlow from '../components/routing/RoutingFlow'
import type { AnalysisResult, DetectResponse, RouteResponse } from '../api/client'
import { api } from '../api/client'

const SCENARIOS = [
  { key: 'normal_stable', label: 'Normal Stable' },
  { key: 'normal_slight_wave', label: 'Normal Slight Wave' },
  { key: 'normal_rising_safe', label: 'Normal Rising (Safe)' },
  { key: 'marginal_rising', label: 'Marginal — Rising' },
  { key: 'marginal_falling', label: 'Marginal — Falling' },
  { key: 'marginal_color', label: 'Marginal — Color Shift' },
  { key: 'marginal_turbidity', label: 'Marginal — Secondary Metric' },
  { key: 'anomaly_sudden_drop', label: 'Anomaly — Sudden Drop' },
  { key: 'anomaly_multi', label: 'Anomaly — Multi-Indicator' },
  { key: 'anomaly_gradual_degradation', label: 'Anomaly — Gradual Degradation' },
  { key: 'novel_foam', label: 'Novel Pattern' },
  { key: 'critical_overflow', label: 'Critical — Overflow' },
  { key: 'critical_empty', label: 'Critical — Empty' },
  { key: 'critical_sudden_surge', label: 'Critical — Sudden Surge' },
  { key: 'sensitive_normal', label: 'Sensitive — Normal' },
  { key: 'sensitive_anomaly', label: 'Sensitive — Anomaly' },
]

const JUDGMENT_STYLE: Record<string, { bg: string; text: string; icon: typeof CheckCircle }> = {
  normal: { bg: 'bg-emerald-500/10 border-emerald-500/30', text: 'text-emerald-400', icon: CheckCircle },
  warning: { bg: 'bg-amber-500/10 border-amber-500/30', text: 'text-amber-400', icon: AlertTriangle },
  alarm: { bg: 'bg-rose-500/10 border-rose-500/30', text: 'text-rose-400', icon: AlertOctagon },
}

function ResultPanel({ title, result }: { title: string; result: AnalysisResult | null }) {
  if (!result) return null
  const style = JUDGMENT_STYLE[result.judgment] || JUDGMENT_STYLE.warning
  const Icon = style.icon

  return (
    <div className={`border rounded-xl p-4 ${style.bg}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] text-slate-500 font-mono tracking-wider uppercase">{title}</span>
        <span className={`text-xs font-mono ${style.text}`}>{result.latency_ms.toFixed(0)}ms</span>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <Icon size={20} className={style.text} />
        <span className={`text-lg font-semibold font-mono uppercase ${style.text}`}>
          {result.judgment}
        </span>
        <span className="text-xs font-mono text-slate-500 ml-auto">
          conf: {result.confidence.toFixed(3)}
        </span>
      </div>
      <div className="space-y-1.5 text-xs text-slate-400">
        <div><span className="text-slate-600">Action:</span> {result.suggested_action}</div>
        <div><span className="text-slate-600">Reasoning:</span> {result.reasoning}</div>
        {result.root_cause && (
          <div><span className="text-slate-600">Root cause:</span> {result.root_cause}</div>
        )}
      </div>
    </div>
  )
}

export default function LiveDemo() {
  const [scenario, setScenario] = useState(SCENARIOS[0].key)
  const [loading, setLoading] = useState(false)
  const [detection, setDetection] = useState<DetectResponse | null>(null)
  const [routeResult, setRouteResult] = useState<RouteResponse | null>(null)
  const [edgeResult, setEdgeResult] = useState<AnalysisResult | null>(null)
  const [cloudResult, setCloudResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    setEdgeResult(null)
    setCloudResult(null)
    setRouteResult(null)

    try {
      // Step 1: Generate detection data
      const det = await api.detect(scenario)
      setDetection(det)

      // Step 2: Run routing decision
      const route = await api.route(det.vision_output)
      setRouteResult(route)

      // Step 3: Run edge and cloud analysis in parallel
      const [edge, cloud] = await Promise.allSettled([
        api.analyzeEdge(det.vision_output),
        api.analyzeCloud(det.vision_output),
      ])

      if (edge.status === 'fulfilled') setEdgeResult(edge.value)
      if (cloud.status === 'fulfilled') setCloudResult(cloud.value)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setDetection(null)
    setRouteResult(null)
    setEdgeResult(null)
    setCloudResult(null)
    setError(null)
  }

  // Determine which tier was the final match for the routing flow
  const finalTier = routeResult
    ? routeResult.tiers.find((t) => t.triggered)?.tier
    : undefined

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Live Demo</h1>
        <p className="text-sm text-slate-500 mt-1">
          Run a scenario through the 5-tier routing pipeline
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <select
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300
                     font-mono focus:outline-none focus:border-cyan-500/50"
        >
          {SCENARIOS.map((s) => (
            <option key={s.key} value={s.key}>{s.label}</option>
          ))}
        </select>
        <button
          onClick={handleRun}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                     bg-cyan-500/10 text-cyan-400 border border-cyan-500/30
                     hover:bg-cyan-500/20 transition-all disabled:opacity-40"
        >
          <Play size={14} />
          {loading ? 'Running...' : 'Run Scenario'}
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm
                     text-slate-500 hover:text-slate-300 transition-colors"
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 text-sm text-rose-400">
          {error}
        </div>
      )}

      {/* Routing decision banner */}
      {routeResult && (
        <div className="bg-slate-900/60 border border-cyan-500/20 rounded-xl px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-slate-500 font-mono tracking-wider">ROUTE</span>
            <span className="text-sm font-mono text-cyan-400 font-semibold">{routeResult.tier_name}</span>
            <span className="text-xs text-slate-500">({routeResult.tier})</span>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
            <span>Reason: <span className="text-slate-400">{routeResult.reason}</span></span>
            <span>{routeResult.latency_ms.toFixed(2)}ms</span>
          </div>
        </div>
      )}

      {detection && (
        <div className="grid grid-cols-3 gap-4">
          {/* Left: detection data */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
            <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
              VISION OUTPUT
            </div>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-500">Scenario</span>
                <span className="text-slate-300">{detection.scenario_name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Difficulty</span>
                <span className="text-slate-300">{detection.scenario_difficulty}</span>
              </div>
              <div className="h-px bg-slate-800 my-2" />
              {Object.entries(detection.vision_output)
                .filter(([k]) => k !== 'timestamp' && k !== 'image_path')
                .map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-slate-500">{k}</span>
                    <span className="text-cyan-400">
                      {typeof v === 'number' ? v.toFixed(3) : Array.isArray(v) ? v.join(', ') : String(v)}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* Center: routing flow */}
          <RoutingFlow
            tiers={routeResult?.tiers ?? []}
            finalTier={finalTier}
          />

          {/* Right: results */}
          <div className="space-y-4">
            <ResultPanel title="EDGE ANALYSIS" result={edgeResult} />
            <ResultPanel title="CLOUD ANALYSIS" result={cloudResult} />
            {!edgeResult && !cloudResult && !loading && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs text-slate-600 text-center">
                Analyzers not connected — routing decision is still shown above
              </div>
            )}
          </div>
        </div>
      )}

      {!detection && !loading && (
        <div className="flex items-center justify-center h-64 text-slate-600 text-sm">
          Select a scenario and click "Run Scenario" to begin
        </div>
      )}
    </div>
  )
}
