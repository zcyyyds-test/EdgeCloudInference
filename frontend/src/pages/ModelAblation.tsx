import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
} from 'recharts'

// Data from model ablation experiments
const MODEL_DATA = [
  {
    model: 'Qwen3.5-0.8B',
    params: '0.8B',
    accuracy: 62.3,
    latency_p50: 280,
    normal_acc: 73.0,
    anomaly_acc: 40.0,
    cloud_savings: 82,
  },
  {
    model: 'Qwen3.5-4B',
    params: '4B',
    accuracy: 84.5,
    latency_p50: 1500,
    normal_acc: 95.0,
    anomaly_acc: 60.0,
    cloud_savings: 70,
  },
]

const PARETO_DATA = MODEL_DATA.map((m) => ({
  x: m.latency_p50,
  y: m.accuracy,
  name: m.model,
  z: parseFloat(m.params),
}))

export default function ModelAblation() {
  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Model Ablation</h1>
        <p className="text-sm text-slate-500 mt-1">
          Comparing edge model variants across accuracy, latency, and cost
        </p>
      </div>

      {/* Model comparison table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
        <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
          MODEL COMPARISON
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800">
              {['Model', 'Params', 'Accuracy', 'Normal Acc', 'Anomaly Acc', 'P50 Latency', 'Cloud Savings'].map((h) => (
                <th key={h} className="text-left py-2 text-slate-500 font-mono text-xs font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODEL_DATA.map((m) => (
              <tr key={m.model} className="border-b border-slate-800/50">
                <td className="py-2.5 font-mono text-cyan-400 text-xs">{m.model}</td>
                <td className="py-2.5 font-mono text-slate-400 text-xs">{m.params}</td>
                <td className="py-2.5 font-mono text-slate-300 text-xs">{m.accuracy}%</td>
                <td className="py-2.5 font-mono text-emerald-400 text-xs">{m.normal_acc}%</td>
                <td className="py-2.5 font-mono text-amber-400 text-xs">{m.anomaly_acc}%</td>
                <td className="py-2.5 font-mono text-slate-400 text-xs">{m.latency_p50}ms</td>
                <td className="py-2.5 font-mono text-blue-400 text-xs">{m.cloud_savings}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Accuracy bar chart */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
          <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
            ACCURACY BY MODEL
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MODEL_DATA} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="model"
                  tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  angle={-20}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: 8,
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono',
                  }}
                />
                <Bar dataKey="normal_acc" name="Normal" fill="#10b981" opacity={0.8} radius={[3, 3, 0, 0]} />
                <Bar dataKey="anomaly_acc" name="Anomaly" fill="#f59e0b" opacity={0.8} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pareto curve */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
          <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
            LATENCY vs ACCURACY (PARETO)
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Latency"
                  unit="ms"
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  scale="log"
                  domain={['auto', 'auto']}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Accuracy"
                  unit="%"
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                  domain={[40, 100]}
                />
                <ZAxis type="number" dataKey="z" range={[60, 400]} />
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: 8,
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono',
                  }}
                  formatter={(val, name) =>
                    [name === 'Latency' ? `${val}ms` : `${val}%`, name]
                  }
                />
                <Legend
                  wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
                <Scatter
                  name="Models"
                  data={PARETO_DATA}
                  fill="#06b6d4"
                  opacity={0.9}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Key findings */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
        <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
          KEY FINDINGS
        </div>
        <div className="grid grid-cols-2 gap-4 text-xs text-slate-400">
          <div className="space-y-2">
            <div className="text-cyan-400 font-medium text-sm">Qwen3.5-4B (Recommended Edge)</div>
            <ul className="space-y-1 list-disc list-inside">
              <li>Best accuracy-latency trade-off for edge deployment</li>
              <li>Multimodal vision capability enables direct image analysis</li>
              <li>~70% cloud savings while maintaining 84.5% accuracy</li>
            </ul>
          </div>
          <div className="space-y-2">
            <div className="text-blue-400 font-medium text-sm">Qwen3.5-27B (Cloud)</div>
            <ul className="space-y-1 list-disc list-inside">
              <li>vLLM continuous batching: 19.7s → &lt;1s latency</li>
              <li>Handles complex multi-indicator patterns</li>
              <li>Root cause analysis for anomalous scenarios</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
