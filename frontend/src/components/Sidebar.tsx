import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Leaf, LayoutDashboard, Activity, Map as MapIcon,
  Database, FlaskConical, Cpu, X, ArrowLeft, type LucideIcon
} from 'lucide-react';

interface Tab {
  id: string;
  icon: LucideIcon;
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
  const asideRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  const [isDesktop, setIsDesktop] = useState(false);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const media = window.matchMedia?.('(min-width: 1024px)');
    if (!media) return;
    const updateDesktop = () => setIsDesktop(media.matches);
    updateDesktop();
    media.addEventListener?.('change', updateDesktop);
    return () => media.removeEventListener?.('change', updateDesktop);
  }, []);

  useEffect(() => {
    asideRef.current?.toggleAttribute('inert', !isOpen && !isDesktop);
  }, [isOpen, isDesktop]);

  useEffect(() => {
    if (!isOpen || isDesktop) return;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current?.();
        return;
      }
      if (event.key !== 'Tab' || !asideRef.current) return;
      const focusable = Array.from(asideRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isDesktop]);

  return (
    <>
      {/* Mobile backdrop overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-950/75 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        id="primary-navigation"
        ref={asideRef}
        role={isOpen && !isDesktop ? 'dialog' : undefined}
        aria-label="Main navigation"
        aria-modal={isOpen && !isDesktop ? true : undefined}
        aria-hidden={!isOpen && !isDesktop}
        className={`fixed top-0 bottom-0 left-0 z-50 w-[260px] bg-[var(--bg-surface)] border-r border-[var(--border)] backdrop-blur-2xl flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="p-5 border-b border-[var(--border)] flex items-center justify-between">
          <button
            type="button"
            onClick={onGoHome}
            className="flex items-center gap-3 cursor-pointer group border-0 bg-transparent p-0 text-left"
            title="Return to Landing Page"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-105 transition-transform">
              <Leaf size={19} className="text-white" strokeWidth={2.5} />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sm text-[var(--text-primary)] tracking-tight">AgriPulse</span>
                <span className="px-1.5 py-0.5 text-[9px] font-semibold bg-emerald-500/15 text-emerald-500 dark:text-emerald-400 border border-emerald-500/30 rounded-full font-mono">
                  AI
                </span>
              </div>
              <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1 group-hover:text-emerald-500 transition-colors">
                <ArrowLeft size={10} /> Back to Home
              </span>
            </div>
          </button>

          {isOpen && (
            <button
              ref={closeButtonRef}
              type="button"
              onClick={onClose}
              className="lg:hidden p-1.5 text-[var(--text-muted)] hover:text-[var(--text-primary)] rounded-lg hover:bg-[var(--bg-hover)]"
              aria-label="Close menu"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Navigation list */}
        <nav aria-label="Dashboard sections" className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {sections.map((section) => (
            <div key={section} className="space-y-1.5">
              <div className="px-3 text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)] font-mono">
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
                    aria-current={active ? 'page' : undefined}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left text-xs font-medium transition-all relative group cursor-pointer border ${
                      active
                        ? 'bg-emerald-500/12 text-[var(--text-primary)] font-semibold border-emerald-500/30 shadow-sm'
                        : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded-lg transition-colors shrink-0 ${
                        active
                          ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/30'
                          : 'bg-[var(--bg-subtle)] text-[var(--text-muted)] group-hover:text-emerald-500'
                      }`}
                    >
                      <IconComponent size={16} strokeWidth={active ? 2.5 : 2} />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="truncate font-semibold">{tab.label}</div>
                      {tab.desc && <div className="text-[10px] text-[var(--text-muted)] truncate">{tab.desc}</div>}
                    </div>

                    {active && (
                      <motion.div
                        layoutId="active-nav-dot"
                        className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400 shrink-0"
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
        <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-surface)]">
          <div className="p-3 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border)] space-y-2">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  backendOnline === null
                    ? 'bg-amber-400 animate-pulse'
                    : backendOnline
                    ? 'bg-emerald-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-[11px] font-mono text-[var(--text-secondary)]">
                {backendOnline === null
                  ? 'Connecting…'
                  : backendOnline
                  ? 'Core Engine Online'
                  : 'Backend Offline'}
              </span>
            </div>

            <div className="flex items-center gap-1.5 text-[10px] text-[var(--text-muted)] font-mono">
              <Cpu size={12} className="text-emerald-500 shrink-0" />
              <span className="truncate">DINOv2 + Graph Transformer</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
