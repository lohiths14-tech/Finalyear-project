import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Leaf, Sprout, ArrowLeftRight, Sparkles, Moon, Sun, Menu, Smartphone, Home } from 'lucide-react';
import LandingPage from './components/LandingPage';
import Sidebar from './components/Sidebar';
import ToastContainer from './components/ToastContainer';
import ErrorBoundary from './components/ErrorBoundary';
import OverviewTab from './components/tabs/OverviewTab';
import FieldMapTab from './components/tabs/FieldMapTab';
import LeafScanTab from './components/tabs/LeafScanTab';
import SpectralTab from './components/tabs/SpectralTab';
import HistoryTab from './components/tabs/HistoryTab';
import FarmerMode from './components/FarmerMode';
import QrCodeModal from './components/ui/QrCodeModal';
import HealthSentinel from './components/ui/HealthSentinel';
import { useAuth } from './context/AuthContext';
import apiClient from './services/api';
import type { FieldResult, LeafResult } from './types';

function SplashScreen() {
  return (
    <div className="splash-screen">
      <motion.div
        className="splash-logo"
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'backOut' }}
      >
        <Leaf size={32} color="#ffffff" strokeWidth={2.5} />
      </motion.div>
      <motion.div
        style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 6 }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        <div className="splash-title">AgriPulse 2.0</div>
        <div className="splash-subtitle">Field-Level Crop Disease Intelligence Platform</div>
      </motion.div>
      <motion.div
        className="spinner"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      />
    </div>
  );
}

interface LoginScreenProps {
  apiKeyInput: string;
  setApiKeyInput: (v: string) => void;
  onLogin: () => void;
  onGuest: () => void;
}

function LoginScreen({ apiKeyInput, setApiKeyInput, onLogin, onGuest }: LoginScreenProps) {
  return (
    <div className="login-screen">
      <motion.div
        className="login-card"
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        {/* Header */}
        <div className="login-header">
          <div className="login-logo">
            <Leaf size={26} color="#ffffff" strokeWidth={2.5} />
          </div>
          <div>
            <div className="login-title">AgriPulse 2.0</div>
            <div className="login-subtitle">Enter your API key or continue as Guest</div>
          </div>
        </div>

        {/* Form */}
        <div className="login-form">
          <div className="form-group">
            <label className="form-label">X-API-Key (Optional)</label>
            <input
              id="api-key-input"
              type="text"
              className="form-input"
              placeholder="e.g. demo (or leave blank for guest)…"
              value={apiKeyInput}
              onChange={(e) => setApiKeyInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') onLogin(); }}
              autoFocus
            />
          </div>
          <button
            id="login-btn"
            className="btn-login"
            onClick={onLogin}
          >
            Authenticate &amp; Enter
          </button>
          <button
            type="button"
            className="btn-guest"
            onClick={onGuest}
          >
            Enter as Demo / Guest
          </button>
        </div>

        <p style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Self-hosted instance · Model weights: DINOv2 + Adaptive Grid
        </p>
      </motion.div>
    </div>
  );
}

const TAB_TITLES: Record<string, { title: string; subtitle: string }> = {
  field: { title: 'Field Intelligence', subtitle: 'Adaptive grid analysis, disease clusters & AI heatmaps' },
  overview: { title: 'System Overview', subtitle: 'Macro health scores, risk distributions & telemetry' },
  spectral: { title: 'Spectral Intelligence', subtitle: 'Non-destructive NGVC vegetation canopy indices' },
  leaf: { title: 'Micro Leaf Scanner', subtitle: 'Targeted single-leaf high resolution pathology' },
  history: { title: 'Audit History', subtitle: 'Temporal scan tracking and historical progress' },
};

let toastIdCounter = 0;

function App() {
  const [viewMode, setViewMode] = useState<'landing' | 'app'>('landing');
  const [activeTab, setActiveTab] = useState('field');
  const [toasts, setToasts] = useState<{ id: number; message: string; type: 'success' | 'error' | 'warning' | 'info' }[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [lastScan, setLastScan] = useState<FieldResult | null>(null);
  const [lastLeafScan, setLastLeafScan] = useState<LeafResult | null>(null);
  const { apiKey, login } = useAuth();
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [isFarmerMode, setIsFarmerMode] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [qrOpen, setQrOpen] = useState(false);
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('agripulse-theme');
    return saved === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('agripulse-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  // Auto-login when backend has no API key requirement
  useEffect(() => {
    if (!apiKey) {
      apiClient.get('/health').then(() => {
        login('__no_auth__');
      }).catch(() => {
        // Cold start or network glitch: auto-login to demo
        login('__no_auth__');
      }).finally(() => {
        setAuthChecked(true);
      });
    } else {
      setAuthChecked(true);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addToast = useCallback((message: string, type: 'success' | 'error' | 'warning' | 'info' = 'success', duration = 5000) => {
    const id = Date.now() * 1000 + (++toastIdCounter % 1000);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, duration);
  }, []);

  const removeToast = (id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  useEffect(() => {
    let wasOnline: boolean | null = null;
    const checkHealth = async () => {
      try {
        const resp = await apiClient.get('/health');
        const online = resp.data.status === 'healthy' || resp.data.status === 'degraded';
        setBackendOnline(online);
        wasOnline = online;
      } catch {
        if (wasOnline !== false) {
          addToast('Cannot connect to AgriVision Core Engine', 'error');
        }
        setBackendOnline(false);
        wasOnline = false;
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 60000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleScanComplete = (results: FieldResult) => {
    setLastScan(results);
    addToast(`Analysis complete: ${results.disease_type}`, 'success');
  };

  const handleLeafScanComplete = (results: LeafResult) => {
    setLastLeafScan(results);
    addToast(`Leaf scan: ${results.disease_type}`, 'success');
  };

  const renderTab = () => {
    switch (activeTab) {
      case 'overview':  return <OverviewTab lastScan={lastScan} />;
      case 'field':     return <FieldMapTab addToast={addToast} backendOnline={backendOnline} onScanComplete={handleScanComplete} />;
      case 'leaf':      return <LeafScanTab addToast={addToast} backendOnline={backendOnline} onScanComplete={handleLeafScanComplete} lastLeafScan={lastLeafScan} />;
      case 'spectral':  return <SpectralTab lastScan={lastScan} />;
      case 'history':   return <HistoryTab addToast={addToast} />;
      default:          return <OverviewTab />;
    }
  };

  if (!authChecked) return <SplashScreen />;

  if (viewMode === 'landing') {
    return (
      <ErrorBoundary>
        <LandingPage
          onEnterExpert={() => {
            setViewMode('app');
            setIsFarmerMode(false);
          }}
          onEnterFarmer={() => {
            setViewMode('app');
            setIsFarmerMode(true);
          }}
        />
        <ToastContainer toasts={toasts} removeToast={removeToast} />
      </ErrorBoundary>
    );
  }

  if (!apiKey) {
    return (
      <ErrorBoundary>
        <LoginScreen
          apiKeyInput={apiKeyInput}
          setApiKeyInput={setApiKeyInput}
          onLogin={() => { login(apiKeyInput.trim() || '__no_auth__'); }}
          onGuest={() => { login('__no_auth__'); }}
        />
        <QrCodeModal isOpen={qrOpen} onClose={() => setQrOpen(false)} />
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      </ErrorBoundary>
    );
  }

  const currentTabMeta = TAB_TITLES[activeTab] || { title: 'AgriPulse AI', subtitle: 'Field Intelligence' };

  return (
    <ErrorBoundary>
      <AnimatePresence mode="wait">
        {isFarmerMode ? (
          <motion.div
            key="farmer"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="farmer-mode-wrapper"
          >
            {/* Top Bar for Farmer Mode */}
            <header className="top-navbar-farmer">
              <div className="top-nav-brand">
                <div className="sidebar-logo mini">
                  <Leaf size={18} color="#ffffff" strokeWidth={2.5} />
                </div>
                <div>
                  <span className="brand-title-small">AgriPulse</span>
                  <span className="farmer-badge-pill">Farmer Edition</span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button
                  onClick={() => setViewMode('landing')}
                  className="mode-toggle-btn"
                  title="Return to Landing Page"
                >
                  <Home size={14} />
                  <span>Home</span>
                </button>
                <HealthSentinel onStatusChange={(online) => setBackendOnline(online)} />
                <button
                  onClick={() => setQrOpen(true)}
                  className="mode-toggle-btn"
                  style={{ padding: '6px 12px' }}
                  title="Open on Mobile Phone"
                >
                  <Smartphone size={14} />
                  <span>Mobile QR</span>
                </button>
                <button
                  id="mode-toggle-btn"
                  className="mode-toggle-btn active"
                  onClick={() => setIsFarmerMode(false)}
                >
                  <ArrowLeftRight size={14} />
                  <span>Switch to Expert Mode</span>
                </button>
              </div>
            </header>

            <FarmerMode addToast={addToast} backendOnline={backendOnline} />
          </motion.div>
        ) : (
          <div className="app-container">
            {/* Left Sidebar */}
            <Sidebar
              activeTab={activeTab}
              setActiveTab={(id) => { setActiveTab(id); setSidebarOpen(false); }}
              backendOnline={backendOnline}
              isOpen={sidebarOpen}
              onClose={() => setSidebarOpen(false)}
              onGoHome={() => setViewMode('landing')}
            />

            {/* Main Area */}
            <div className="main-wrapper">
              {/* Top Navigation Bar */}
              <header className="top-navbar">
                <div className="top-nav-left" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <button
                    className="hamburger-btn"
                    onClick={() => setSidebarOpen(true)}
                    aria-label="Open navigation"
                  >
                    <Menu size={20} />
                  </button>
                  <div>
                    <h1 className="page-heading">{currentTabMeta.title}</h1>
                    <p className="page-subheading">{currentTabMeta.subtitle}</p>
                  </div>
                </div>

                <div className="top-nav-right">
                  <button
                    onClick={() => setViewMode('landing')}
                    className="mode-toggle-btn"
                    title="Return to Landing Page"
                  >
                    <Home size={14} />
                    <span>Home</span>
                  </button>
                  <HealthSentinel onStatusChange={(online) => setBackendOnline(online)} />

                  <div className="ai-badge">
                    <Sparkles size={13} color="var(--primary)" />
                    <span>DINOv2 Engine</span>
                  </div>

                  <button
                    onClick={() => setQrOpen(true)}
                    className="theme-toggle-btn"
                    aria-label="Open on Mobile Phone"
                    title="Open on Mobile Phone via QR"
                  >
                    <Smartphone size={16} />
                  </button>

                  <button
                    className="theme-toggle-btn"
                    onClick={() => setIsDark(d => !d)}
                    aria-label="Toggle dark mode"
                    title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
                  >
                    {isDark ? <Sun size={16} /> : <Moon size={16} />}
                  </button>

                  <button
                    id="mode-toggle-btn"
                    className="mode-toggle-btn"
                    onClick={() => setIsFarmerMode(true)}
                  >
                    <Sprout size={15} />
                    <span>Farmer Mode</span>
                  </button>
                </div>
              </header>

              {/* Tab Viewport */}
              <main className="main-content">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    className="tab-viewport"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.18, ease: 'easeOut' }}
                  >
                    {renderTab()}
                  </motion.div>
                </AnimatePresence>
              </main>
            </div>
          </div>
        )}
      </AnimatePresence>

      <QrCodeModal isOpen={qrOpen} onClose={() => setQrOpen(false)} />
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ErrorBoundary>
  );
}

export default App;
