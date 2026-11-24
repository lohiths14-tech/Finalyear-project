import React from 'react';
import { motion } from 'framer-motion';
import {
  Leaf, LayoutDashboard, Activity, Map as MapIcon,
  Database, FlaskConical, WifiOff, Cpu, X, Home,
  ChevronRight, ArrowLeft
} from 'lucide-react';

interface Tab {
  id: string;
  icon: React.ComponentType<{ size?: number; className?: string; strokeWidth?: number }>;
  label: string;
  desc?: string;
  section?: string;
}

const TABS: Tab[] = [
  { id: 'field',    icon: MapIcon,         label: 'Field Analysis',   desc: 'Drone GIS & spatial grid',  section: 'DIAGNOSTICS' },
  { id: 'leaf',     icon: FlaskConical,    label: 'Leaf Scanner',     desc: 'Targeted single-leaf',      section: 'DIAGNOSTICS' },
  { id: 'overview', icon: LayoutDashboard, label: 'System Overview',  desc: 'Macro health & telemetry',  section: 'INTELLIGENCE' },
  { id: 'spectral', icon: Activity,        label: 'Spectral Intel',   desc: 'NGVC canopy radar',         section: 'INTELLIGENCE' },
  { id: 'history',  icon: Database,        label: 'Audit History',    desc: 'Temporal scan records',     section: 'RECORDS' },
];

interface SidebarProps {
  activeTab: string;
  setActiveTab: (id: string) => void;
  backendOnline: boolean | null;
  isOpen?: boolean;
  onClose?: () => void;
  onGoHome?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  backendOnline,
  isOpen,
  onClose,
  onGoHome,
}) => {
  const sections = Array.from(new Set(TABS.map((t) => t.section)));

  return (
    <>
      {/* Mobile backdrop overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-slate-950/95 border-r border-white/10 backdrop-blur-2xl flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="p-5 border-b border-white/10 flex items-center justify-between">
          <div
            onClick={onGoHome}
            className="flex items-center gap-3 cursor-pointer group"
            title="Return to Landing Page"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-105 transition-transform">
              <Leaf size={19} className="text-white" strokeWidth={2.5} />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sm text-white tracking-tight">AgriPulse</span>
                <span className="px-1.5 py-0.2 text-[9px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 rounded-full font-mono">
                  v2.0
                </span>
              </div>
              <span className="text-[11px] text-slate-400 flex items-center gap-1 group-hover:text-emerald-400 transition-colors">
                <ArrowLeft size={10} /> Back to Home
              </span>
            </div>
          </div>

          {isOpen && (
            <button
              onClick={onClose}
              className="lg:hidden p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-900"
              aria-label="Close menu"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Navigation list */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {sections.map((section) => (
            <div key={section} className="space-y-1">
              <div className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500 font-mono">
                {section}
              </div>

              {TABS.filter((t) => t.section === section).map((tab) => {
                const IconComponent = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left text-xs font-medium transition-all relative group cursor-pointer ${
                      active
                        ? 'bg-emerald-500/15 text-white font-semibold border border-emerald-500/30 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded-lg transition-colors ${
                        active
                          ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/30'
                          : 'bg-slate-900 text-slate-400 group-hover:text-emerald-400 group-hover:bg-slate-850'
                      }`}
                    >
                      <IconComponent size={16} strokeWidth={active ? 2.5 : 2} />
                    </div>

                    <div className="flex-1 truncate">
                      <div className="truncate text-slate-200">{tab.label}</div>
                      {tab.desc && <div className="text-[10px] text-slate-500 truncate">{tab.desc}</div>}
                    </div>

                    {active && (
                      <motion.div
                        layoutId="active-nav-dot"
                        className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400"
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Footer System Status */}
        <div className="p-4 border-t border-white/10 bg-slate-950/80">
          <div className="p-3 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline === null
                    ? 'bg-amber-400 animate-pulse'
                    : backendOnline
                    ? 'bg-emerald-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-[11px] font-mono text-slate-300">
                {backendOnline === null
                  ? 'Connecting…'
                  : backendOnline
                  ? 'Engine Online (INT8)'
                  : 'Engine Offline'}
              </span>
            </div>

            <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-mono">
              <Cpu size={12} className="text-emerald-400" />
              <span>DINOv2 + Graph Transformer</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
