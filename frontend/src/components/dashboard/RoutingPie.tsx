import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

interface Props {
  edgeOnly: number
  cascaded: number
  cloudDirect: number
}

const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6']

export default function RoutingPie({ edgeOnly, cascaded, cloudDirect }: Props) {
  const data = [
    { name: 'Edge Only', value: edgeOnly },
    { name: 'Cascaded', value: cascaded },
    { name: 'Cloud Direct', value: cloudDirect },
  ]

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 backdrop-blur">
      <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-3">
        ROUTING DISTRIBUTION
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={45}
              outerRadius={75}
              dataKey="value"
              stroke="none"
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i]} opacity={0.85} />
              ))}
            </Pie>
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
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-center gap-4 mt-2">
        {data.map((d, i) => (
          <div key={d.name} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: COLORS[i] }} />
            <span className="text-[10px] text-slate-400 font-mono">{d.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
