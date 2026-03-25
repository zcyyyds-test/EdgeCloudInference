import { motion } from 'framer-motion'

interface TierResult {
  tier: string
  triggered: boolean
  detail: string
}

interface Props {
  tiers: TierResult[]
  finalTier?: string
}

const tierMeta: Record<string, { label: string; color: string }> = {
  safety: { label: 'T0 Safety', color: '#f43f5e' },
  clearly_normal: { label: 'T1 Clearly Normal', color: '#10b981' },
  complex: { label: 'T2 Complex Pattern', color: '#f59e0b' },
  confidence: { label: 'T3 Confidence Gate', color: '#3b82f6' },
  default: { label: 'T4 Default Edge', color: '#06b6d4' },
}

export default function RoutingFlow({ tiers, finalTier }: Props) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 backdrop-blur">
      <div className="text-[11px] text-slate-500 font-mono tracking-wider uppercase mb-4">
        5-TIER ROUTING CASCADE
      </div>
      <div className="space-y-2">
        {tiers.map((t, i) => {
          const meta = tierMeta[t.tier] || tierMeta.default
          const isTriggered = t.triggered
          const isFinal = t.tier === finalTier

          return (
            <motion.div
              key={t.tier}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.15, duration: 0.3 }}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-all ${
                isFinal
                  ? 'border-cyan-500/40 bg-cyan-500/5'
                  : isTriggered
                  ? 'border-slate-700 bg-slate-800/40'
                  : 'border-slate-800/50 bg-slate-900/30 opacity-40'
              }`}
            >
              {/* Dot */}
              <div
                className="w-3 h-3 rounded-full shrink-0"
                style={{
                  background: isTriggered ? meta.color : '#334155',
                  boxShadow: isFinal ? `0 0 8px ${meta.color}60` : 'none',
                }}
              />

              {/* Label */}
              <div className="flex-1 min-w-0">
                <div className="text-xs font-mono font-medium" style={{ color: isTriggered ? meta.color : '#64748b' }}>
                  {meta.label}
                </div>
                {t.detail && (
                  <div className="text-[10px] text-slate-500 truncate mt-0.5">{t.detail}</div>
                )}
              </div>

              {/* Arrow or stop */}
              {isFinal ? (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  MATCHED
                </span>
              ) : isTriggered ? (
                <span className="text-slate-600 text-xs">↓</span>
              ) : null}
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
