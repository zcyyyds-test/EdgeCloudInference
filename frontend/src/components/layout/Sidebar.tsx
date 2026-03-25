import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Play,
  FlaskConical,
  BarChart3,
  Network,
} from 'lucide-react'

const links = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/demo', icon: Play, label: 'Live Demo' },
  { to: '/ablation', icon: BarChart3, label: 'Ablation' },
  { to: '/experiments', icon: FlaskConical, label: 'Experiments' },
  { to: '/architecture', icon: Network, label: 'Architecture' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 bg-slate-900/80 border-r border-slate-800 flex flex-col shrink-0">
      {/* Logo */}
      <div className="h-16 flex items-center gap-2.5 px-5 border-b border-slate-800">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 8L8 4L12 8L8 12Z" stroke="white" strokeWidth="1.5" fill="none" />
            <circle cx="8" cy="8" r="1.5" fill="white" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-100 leading-tight">EdgeCloudInference</div>
          <div className="text-[10px] text-cyan-500 font-mono tracking-wider">v2.0</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
              }`
            }
          >
            <Icon size={18} strokeWidth={1.5} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800">
        <div className="text-[10px] text-slate-600 font-mono">
          EDGE-CLOUD INFERENCE ROUTER
        </div>
        <div className="text-[10px] text-slate-700 font-mono mt-0.5">
          Qwen3.5 + vLLM
        </div>
      </div>
    </aside>
  )
}
