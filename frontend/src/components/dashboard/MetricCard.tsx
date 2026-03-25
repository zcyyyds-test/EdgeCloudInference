import { type ReactNode } from 'react'

interface Props {
  label: string
  value: string | number
  sub?: string
  icon?: ReactNode
  color?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'blue'
}

const ring: Record<string, string> = {
  cyan: 'border-cyan-500/30',
  emerald: 'border-emerald-500/30',
  amber: 'border-amber-500/30',
  rose: 'border-rose-500/30',
  blue: 'border-blue-500/30',
}

const text: Record<string, string> = {
  cyan: 'text-cyan-400',
  emerald: 'text-emerald-400',
  amber: 'text-amber-400',
  rose: 'text-rose-400',
  blue: 'text-blue-400',
}

export default function MetricCard({ label, value, sub, icon, color = 'cyan' }: Props) {
  return (
    <div className={`bg-slate-900/60 border ${ring[color]} rounded-xl p-4 backdrop-blur glow-cyan`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] text-slate-500 font-mono tracking-wider uppercase">{label}</span>
        {icon && <span className="text-slate-600">{icon}</span>}
      </div>
      <div className={`text-2xl font-semibold font-mono ${text[color]}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}
