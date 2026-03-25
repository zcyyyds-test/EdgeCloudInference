import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

interface DataPoint {
  name: string
  edge: number
  cloud: number
}

interface Props {
  data: DataPoint[]
}

export default function LatencyChart({ data }: Props) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
      <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
        LATENCY TIMELINE (ms)
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="edgeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="cloudGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: 8,
                fontSize: 12,
                fontFamily: 'JetBrains Mono',
              }}
              itemStyle={{ color: '#e2e8f0' }}
            />
            <Area
              type="monotone"
              dataKey="edge"
              stroke="#06b6d4"
              strokeWidth={2}
              fill="url(#edgeGrad)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="cloud"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#cloudGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-center gap-4 mt-2">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-cyan-500 rounded" />
          <span className="text-[10px] text-slate-400 font-mono">Edge</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-0.5 bg-blue-500 rounded" />
          <span className="text-[10px] text-slate-400 font-mono">Cloud</span>
        </div>
      </div>
    </div>
  )
}
