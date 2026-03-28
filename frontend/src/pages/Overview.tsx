import { Shield, Zap, Cloud, Target, Clock } from 'lucide-react'
import MetricCard from '../components/dashboard/MetricCard'
import RoutingPie from '../components/dashboard/RoutingPie'
import LatencyChart from '../components/dashboard/LatencyChart'

// Static demo data from experiment results
const LATENCY_DATA = [
  { name: 'S1', edge: 85, cloud: 920 },
  { name: 'S2', edge: 92, cloud: 1050 },
  { name: 'S3', edge: 78, cloud: 880 },
  { name: 'S4', edge: 110, cloud: 1200 },
  { name: 'S5', edge: 95, cloud: 960 },
  { name: 'S6', edge: 88, cloud: 1100 },
  { name: 'S7', edge: 102, cloud: 950 },
  { name: 'S8', edge: 80, cloud: 1300 },
  { name: 'S9', edge: 97, cloud: 870 },
  { name: 'S10', edge: 91, cloud: 1150 },
]

const TARGET_COMPARISON = [
  { metric: 'Overall Accuracy', target: '≥80%', actual: '80.0%', met: true },
  { metric: 'False Alarm Rate', target: '0%', actual: '0%', met: true },
  { metric: 'Cloud Savings', target: '≥60%', actual: '86.7%', met: true },
  { metric: 'Throughput (optimized)', target: '≥2×', actual: '2.9×', met: true },
  { metric: 'Routing Overhead', target: '≤1ms', actual: '0.03ms', met: true },
]

export default function Overview() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Title */}
      <div>
        <h1 className="text-xl font-semibold text-slate-100">System Overview</h1>
        <p className="text-sm text-slate-500 mt-1">
          EdgeCloudInference — Confidence-driven edge-cloud inference routing for visual anomaly detection
        </p>
      </div>

      {/* Metric cards row */}
      <div className="grid grid-cols-5 gap-4">
        <MetricCard
          label="Accuracy"
          value="80.0%"
          sub="30 scenarios"
          icon={<Target size={14} />}
          color="emerald"
        />
        <MetricCard
          label="False Alarm"
          value="0%"
          sub="Zero false positives"
          icon={<Shield size={14} />}
          color="rose"
        />
        <MetricCard
          label="Cloud Savings"
          value="86.7%"
          sub="Requests handled by edge"
          icon={<Cloud size={14} />}
          color="cyan"
        />
        <MetricCard
          label="Throughput"
          value="2.9×"
          sub="Async optimization"
          icon={<Zap size={14} />}
          color="amber"
        />
        <MetricCard
          label="Routing Overhead"
          value="0.03ms"
          sub="5-tier decision cost"
          icon={<Clock size={14} />}
          color="blue"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-3 gap-4">
        <RoutingPie edgeOnly={62} cascaded={24} cloudDirect={14} />
        <div className="col-span-2">
          <LatencyChart data={LATENCY_DATA} />
        </div>
      </div>

      {/* Target comparison table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
        <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
          PROPOSAL TARGET vs ACTUAL
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800">
              <th className="text-left py-2 text-slate-500 font-mono text-xs font-normal">Metric</th>
              <th className="text-right py-2 text-slate-500 font-mono text-xs font-normal">Target</th>
              <th className="text-right py-2 text-slate-500 font-mono text-xs font-normal">Actual</th>
              <th className="text-right py-2 text-slate-500 font-mono text-xs font-normal">Status</th>
            </tr>
          </thead>
          <tbody>
            {TARGET_COMPARISON.map((r) => (
              <tr key={r.metric} className="border-b border-slate-800/50">
                <td className="py-2.5 text-slate-300">{r.metric}</td>
                <td className="py-2.5 text-right font-mono text-slate-400">{r.target}</td>
                <td className="py-2.5 text-right font-mono text-cyan-400">{r.actual}</td>
                <td className="py-2.5 text-right">
                  <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                    r.met
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                  }`}>
                    {r.met ? 'PASS' : 'FAIL'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Architecture summary */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
        <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
          SYSTEM OVERVIEW
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="space-y-2">
            <div className="text-sm font-medium text-cyan-400">Edge Tier</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Qwen3.5-0.8B (quantized) on edge device. Handles ~72% of requests
              with sub-300ms latency. Its self-reported confidence drives cascade escalation.
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-blue-400">Cloud Tier</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Qwen3.5-27B via vLLM with tensor parallelism. Continuous batching
              for high throughput. Handles complex and cascaded scenarios.
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-amber-400">5-Tier Router</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Safety → Clearly Normal → Complex Pattern → Grey Zone Cascade.
              ~8ms overhead. Edge LLM confidence drives cascade escalation.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
