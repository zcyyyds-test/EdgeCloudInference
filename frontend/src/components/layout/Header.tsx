import { useEffect, useState } from 'react'
import type { HealthStatus } from '../../api/client'
import { api } from '../../api/client'

function StatusDot({ ok, label }: { ok: boolean | null; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={`w-2 h-2 rounded-full ${
          ok === null
            ? 'bg-slate-600'
            : ok
            ? 'bg-emerald-400 animate-pulse-slow'
            : 'bg-rose-500'
        }`}
      />
      <span className="text-xs text-slate-400 font-mono">{label}</span>
    </div>
  )
}

export default function Header() {
  const [health, setHealth] = useState<HealthStatus | null>(null)

  useEffect(() => {
    const check = () =>
      api.health().then(setHealth).catch(() => setHealth(null))
    check()
    const t = setInterval(check, 10000)
    return () => clearInterval(t)
  }, [])

  return (
    <header className="h-12 bg-slate-900/60 backdrop-blur border-b border-slate-800 flex items-center justify-between px-6 shrink-0">
      <div className="flex items-center gap-4">
        <StatusDot ok={health?.edge_available ?? null} label="EDGE" />
        <StatusDot ok={health?.cloud_available ?? null} label="CLOUD" />
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-slate-600 font-mono tracking-wider">
          VISUAL ANOMALY DETECTION
        </span>
      </div>
    </header>
  )
}
