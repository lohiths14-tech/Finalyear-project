import { motion } from 'framer-motion';
import {
  Leaf, LayoutDashboard, Activity, Map as MapIcon,
  Database, FlaskConical, WifiOff, Cpu, X,
} from 'lucide-react';

interface Tab {
  id: string;
  icon: typeof LayoutDashboard;
  label: string;
  desc?: string;
  section?: string;
}

const TABS: Tab[] = [
  { id: 'field',    icon: MapIcon,         label: 'Field Analysis',   desc: 'Spatial heatmap & grid',    section: 'DIAGNOSTICS' },
  { id: 'leaf',     icon: FlaskConical,    label: 'Leaf Scanner',     desc: 'Micro-diagnostic',          section: 'DIAGNOSTICS' },
  { id: 'overview', icon: LayoutDashboard, label: 'System Overview',  desc: 'Health & analytics',        section: 'INTELLIGENCE' },
  { id: 'spectral', icon: Activity,        label: 'Spectral Intel',   desc: 'NGVC feature radar',        section: 'INTELLIGENCE' },
  { id: 'history',  icon: Database,        label: 'Audit History',    desc: 'Scan logs & records',       section: 'DATA' },
];

interface SidebarProps {
  activeTab: string;
  setActiveTab: (id: string) => void;
  backendOnline: boolean | null;
  isOpen?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ activeTab, setActiveTab, backendOnline, isOpen, onClose }: SidebarProps) {
  // Group tabs by section
  const sections = Array.from(new Set(TABS.map(t => t.section)));

  return (
    <>
      {/* Mobile backdrop overlay */}
      <div
        className={`sidebar-overlay ${isOpen ? 'open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        {/* Brand Header */}
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Leaf size={22} color="#ffffff" strokeWidth={2.5} />
          </div>
          <div className="sidebar-brand-text">
            <span className="sidebar-title">AgriPulse</span>
            <span className="sidebar-badge">v2.0 · Enterprise</span>
          </div>
          {/* Mobile close button */}
          {isOpen && (
            <button
              onClick={onClose}
              style={{
                marginLeft: 'auto',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                padding: 4,
                borderRadius: 6,
                transition: 'var(--transition)',
              }}
              aria-label="Close menu"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Navigation Links — grouped by section */}
        <nav className="nav-list">
          {sections.map((section) => (
            <div key={section} style={{ marginBottom: 6 }}>
              <div className="nav-section-label">{section}</div>
              {TABS.filter(t => t.section === section).map((tab) => {
                const IconComponent = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`sidebar-item ${active ? 'active' : ''}`}
                    title={tab.label}
                  >
                    <div className={`sidebar-item-icon ${active ? 'active' : ''}`}>
                      <IconComponent size={18} strokeWidth={active ? 2.5 : 2} />
                    </div>
                    <div className="sidebar-item-content">
                      <span className="sidebar-item-label">{tab.label}</span>
                      {tab.desc && <span className="sidebar-item-desc">{tab.desc}</span>}
                    </div>
                    {active && (
                      <motion.div
                        layoutId="active-pill"
                        className="active-pill"
                        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Sidebar Footer Status */}
        <div className="sidebar-footer">
          <div className="system-status-card">
            <div className="system-status-header">
              <div className={`status-dot ${backendOnline === null ? 'yellow' : backendOnline ? 'green' : 'red'}`} />
              <span className="system-status-text">
                {backendOnline === null
                  ? 'Connecting engine…'
                  : backendOnline
                    ? 'Core Engine Online'
                    : 'Backend Offline'}
              </span>
            </div>

            {!backendOnline && backendOnline !== null && (
              <div className="offline-command-box">
                <WifiOff size={12} />
                <code>uvicorn app.main:app</code>
              </div>
            )}

            <div className="engine-version-tag">
              <Cpu size={12} />
              <span>DINOv2 + Adaptive Grid</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
