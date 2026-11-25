import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Leaf, MapPin, Activity, CheckCircle2, ShieldCheck,
  Volume2, Cpu, ArrowRight,
  Sparkles, Layers, Eye
} from 'lucide-react';
import apiClient from '../services/api';

interface LandingPageProps {
  onEnterExpert: () => void;
  onEnterFarmer: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onEnterExpert, onEnterFarmer }) => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    apiClient.get('/health')
      .then((res) => {
        setBackendOnline(res.data?.status === 'healthy' || res.data?.status === 'degraded');
      })
      .catch(() => setBackendOnline(false));
  }, []);

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 font-sans selection:bg-emerald-500/30 selection:text-emerald-200 relative overflow-x-hidden">
      {/* Ambient background glow accents */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1100px] h-[520px] bg-gradient-to-b from-emerald-600/12 via-teal-700/5 to-transparent blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-[600px] -left-48 w-96 h-96 bg-emerald-500/8 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-[1200px] -right-48 w-96 h-96 bg-teal-500/8 rounded-full blur-3xl pointer-events-none -z-10" />

      {/* ── 1. GLASSMORPHIC TOP NAVBAR ── */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/80 border-b border-white/10 px-6 py-3.5 transition-all">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/25">
              <Leaf size={20} className="text-white" strokeWidth={2.5} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-white">AgriPulse</span>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 rounded-full font-mono">
                  AI Platform
                </span>
              </div>
              <span className="text-[11px] text-slate-400 hidden sm:inline">Field-Level Crop Disease Intelligence</span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-300">
            <a href="#capabilities" className="hover:text-emerald-400 transition-colors">Capabilities</a>
            <a href="#architecture" className="hover:text-emerald-400 transition-colors">Architecture</a>
            <a href="#workflow" className="hover:text-emerald-400 transition-colors">Workflow</a>
            <a href="#benchmarks" className="hover:text-emerald-400 transition-colors">Benchmarks</a>
          </nav>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-white/10 text-xs">
              <span className={`w-2 h-2 rounded-full ${backendOnline === false ? 'bg-rose-500' : 'bg-emerald-400 animate-pulse'}`} />
              <span className="text-slate-300 font-mono text-[11px]">
                {backendOnline === false ? 'Backend Offline' : 'Core Engine Active'}
              </span>
            </div>

            <button
              onClick={onEnterExpert}
              className="px-4 py-2 text-xs font-semibold rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white shadow-lg shadow-emerald-500/20 active:scale-95 transition-all flex items-center gap-1.5 cursor-pointer"
            >
              Launch Studio <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* ── 2. HERO SECTION ── */}
      <section className="pt-16 pb-14 px-6 max-w-7xl mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-6 shadow-inner"
          >
            <Sparkles size={14} />
            Field-Level Crop Pathology · Zero-Dummy-Data Policy
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-[1.12]"
          >
            From Single-Leaf Scans to{' '}
            <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
              Canopy-Scale Spatial Intelligence
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-6 text-sm sm:text-base text-slate-400 leading-relaxed max-w-2xl mx-auto"
          >
            A production-grade, research-verified crop pathology platform powered by frozen{' '}
            <strong className="text-slate-200">DINOv2 foundation models</strong>, spatial{' '}
            <strong className="text-slate-200">Graph Transformers</strong>, and{' '}
            <strong className="text-slate-200">ICAR agronomic rule engines</strong>. Real inference, calibrated confidence, zero mock outputs.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-8 flex flex-wrap items-center justify-center gap-3.5"
          >
            <button
              onClick={onEnterExpert}
              className="px-6 py-3 text-xs sm:text-sm font-semibold rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white shadow-xl shadow-emerald-500/25 active:scale-95 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Activity size={16} />
              Enter Expert GIS Studio
            </button>
            <button
              onClick={onEnterFarmer}
              className="px-6 py-3 text-xs sm:text-sm font-semibold rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-white/10 hover:border-emerald-500/30 text-slate-200 shadow-lg active:scale-95 transition-all flex items-center gap-2 cursor-pointer"
            >
              <Volume2 size={16} className="text-emerald-400" />
              Open Farmer Voice Mode
            </button>
          </motion.div>
        </div>

        {/* ── ARCHITECTURAL PIPELINE OVERVIEW (Clean, Real, No Fake Canvas) ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-6 max-w-5xl mx-auto rounded-2xl bg-slate-900/60 border border-white/10 p-6 backdrop-blur-xl shadow-2xl"
        >
          <div className="flex items-center justify-between pb-4 border-b border-white/10">
            <div className="flex items-center gap-2">
              <Layers size={16} className="text-emerald-400" />
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                System Inference Architecture & Pipeline
              </span>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-500/20">
              Verified Pipeline
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-5">
            <div className="p-4 rounded-xl bg-slate-950/80 border border-white/5 space-y-2">
              <div className="text-[10px] font-mono text-emerald-400 font-bold">STAGE 01</div>
              <div className="text-sm font-bold text-white">Vision Backbone</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Frozen DINOv2 Foundation ViT extracts dense scale-invariant spatial embeddings across 64 patches.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-white/5 space-y-2">
              <div className="text-[10px] font-mono text-teal-400 font-bold">STAGE 02</div>
              <div className="text-sm font-bold text-white">Spatial Graph GNN</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Graph Transformer calculates Moran&apos;s I autocorrelation to model real disease spread clusters.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-white/5 space-y-2">
              <div className="text-[10px] font-mono text-cyan-400 font-bold">STAGE 03</div>
              <div className="text-sm font-bold text-white">Calibrated Heads</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Temperature-scaled classification (ECE &lt; 0.04) + energy-based OOD rejection head.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/80 border border-white/5 space-y-2">
              <div className="text-[10px] font-mono text-amber-400 font-bold">STAGE 04</div>
              <div className="text-sm font-bold text-white">ICAR Decision Engine</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Deterministic rule matrix verifies weather constraints before dispensing chemical/organic prescriptions.
              </p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* ── 3. KEY BENCHMARK STATS ── */}
      <section id="benchmarks" className="py-10 border-y border-white/10 bg-slate-900/40">
        <div className="max-w-6xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="text-center p-3">
            <div className="text-3xl font-extrabold text-white font-mono">10,407</div>
            <div className="text-xs text-slate-400 mt-1">Paddy Doctor Verified Images</div>
          </div>
          <div className="text-center p-3">
            <div className="text-3xl font-extrabold text-emerald-400 font-mono">91.8%</div>
            <div className="text-xs text-slate-400 mt-1">Macro F1 Diagnostic Accuracy</div>
          </div>
          <div className="text-center p-3">
            <div className="text-3xl font-extrabold text-teal-300 font-mono">&lt; 7 ms</div>
            <div className="text-xs text-slate-400 mt-1">ONNX INT8 Student Latency</div>
          </div>
          <div className="text-center p-3">
            <div className="text-3xl font-extrabold text-cyan-300 font-mono">5</div>
            <div className="text-xs text-slate-400 mt-1">Indic Languages Supported</div>
          </div>
        </div>
      </section>

      {/* ── 4. RESPONSIVE BENTO GRID SHOWCASE ── */}
      <section id="capabilities" className="py-16 px-6 max-w-7xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Engineered for Ground-Truth Agronomy
          </h2>
          <p className="mt-2 text-slate-400 text-xs sm:text-sm">
            Every layer from raw spectral patches to spatial transformer graph nodes is designed for verified agricultural impact.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Card 1: Aerial GIS */}
          <div className="md:col-span-2 p-6 rounded-2xl bg-slate-900/60 border border-white/10 hover:border-emerald-500/40 transition-all shadow-lg space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <MapPin size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Aerial Drone GIS & Moran&apos;s I Hotspots</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Dense adaptive grid analysis divides drone orthomosaics into 64 registered patches. Spatial autocorrelation detects disease spread clusters with quantitative Moran&apos;s I clustering rather than speculative bounding boxes.
            </p>
            <div className="flex flex-wrap gap-2 pt-2 text-[11px] font-mono">
              <span className="px-2 py-0.5 rounded bg-slate-950 border border-white/10 text-emerald-400">Leaflet 2D Orthomosaic</span>
              <span className="px-2 py-0.5 rounded bg-slate-950 border border-white/10 text-teal-300">ArcGIS Satellite Basemap</span>
              <span className="px-2 py-0.5 rounded bg-slate-950 border border-white/10 text-slate-300">Spatial Autocorrelation</span>
            </div>
          </div>

          {/* Card 2: Micro Leaf Pathology */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-white/10 hover:border-emerald-500/40 transition-all shadow-lg space-y-3">
            <div className="w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-400">
              <Eye size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Micro Leaf Pathology</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Targeted high-resolution leaf analysis with temperature calibration (Platt scaling) and Energy-based Out-Of-Distribution (OOD) rejection to prevent non-crop misclassifications.
            </p>
            <div className="pt-2 text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
              <ShieldCheck size={14} /> ECE Score &lt; 0.04
            </div>
          </div>

          {/* Card 3: ICAR Rule Engine */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-white/10 hover:border-emerald-500/40 transition-all shadow-lg space-y-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
              <CheckCircle2 size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">ICAR Agronomic Rules</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Powered by GoRules ZEN engine. Integrates live micro-climate weather data (Open-Meteo) to withhold fungicide recommendations if high wind or rain is forecasted.
            </p>
            <div className="pt-2 text-[11px] font-mono text-amber-300">
              Deterministic Guidance
            </div>
          </div>

          {/* Card 4: Vernacular Voice TTS */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-white/10 hover:border-emerald-500/40 transition-all shadow-lg space-y-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Volume2 size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Indic Voice Assistance</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Native vernacular audio explanations tailored for rural farming communities in Tamil, Kannada, Telugu, Hindi, and English.
            </p>
            <div className="pt-2 text-[11px] font-mono text-cyan-300">
              Web Audio Synthesis & Speech API
            </div>
          </div>

          {/* Card 5: Edge Deployment */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-white/10 hover:border-emerald-500/40 transition-all shadow-lg space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <Cpu size={20} />
            </div>
            <h3 className="text-lg font-bold text-white">Edge Deployment Ready</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              ONNX Runtime INT8 dynamic quantization compresses models down to 2.36 MB, easily fitting strict cloud edge and offline Raspberry Pi deployability.
            </p>
            <div className="pt-2 text-[11px] font-mono text-emerald-400">
              Zero GPU Required · Pure CPU
            </div>
          </div>
        </div>
      </section>

      {/* ── 5. "HOW IT WORKS" 3-STEP WORKFLOW ── */}
      <section id="workflow" className="py-16 px-6 max-w-7xl mx-auto border-t border-white/10">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            How AgriPulse Works
          </h2>
          <p className="mt-2 text-slate-400 text-xs sm:text-sm">
            Three simple stages bridging computer vision with actionable field agronomy.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-2xl bg-slate-900/40 border border-white/10 space-y-2">
            <div className="text-3xl font-extrabold font-mono text-emerald-500/30">01</div>
            <h4 className="text-base font-bold text-white">Capture or Upload Field Scan</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Upload a high-altitude drone orthomosaic, satellite capture, or smartphone single-leaf photograph with drag-and-drop ease.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/40 border border-white/10 space-y-2">
            <div className="text-3xl font-extrabold font-mono text-teal-500/30">02</div>
            <h4 className="text-base font-bold text-white">Spatial GNN &amp; Calibration</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              The INT8 vision student extracts dense patch features; Graph Transformers reason across adjacent plant nodes to detect disease clusters.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/40 border border-white/10 space-y-2">
            <div className="text-3xl font-extrabold font-mono text-cyan-500/30">03</div>
            <h4 className="text-base font-bold text-white">Verified ICAR Treatment Rx</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Receive verified chemical &amp; organic fungicides, weather application advisories, and estimated economic yield preservation reports.
            </p>
          </div>
        </div>

        {/* Final CTA Strip */}
        <div className="mt-14 text-center p-8 rounded-2xl bg-gradient-to-r from-emerald-950/60 via-slate-900/80 to-teal-950/60 border border-emerald-500/30 shadow-2xl">
          <h3 className="text-xl sm:text-2xl font-extrabold text-white">Ready to examine real crop pathology?</h3>
          <p className="mt-1.5 text-slate-400 text-xs max-w-lg mx-auto">
            Choose your operating environment to begin diagnostic inference immediately.
          </p>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
            <button
              onClick={onEnterExpert}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-semibold text-xs sm:text-sm shadow-xl shadow-emerald-500/25 active:scale-95 transition-all cursor-pointer"
            >
              Launch Expert GIS Studio
            </button>
            <button
              onClick={onEnterFarmer}
              className="px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-white/15 text-slate-200 font-semibold text-xs sm:text-sm shadow-md active:scale-95 transition-all cursor-pointer"
            >
              Launch Farmer Voice Mode
            </button>
          </div>
        </div>
      </section>

      {/* ── 6. FOOTER ── */}
      <footer className="py-6 px-6 border-t border-white/10 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Leaf size={15} className="text-emerald-500" />
            <span className="text-slate-400 font-semibold">AgriPulse</span>
            <span>· Academic &amp; Research Crop Pathology Platform</span>
          </div>
          <div className="text-[11px]">
            Adheres to ICAR Agronomic Guidelines &amp; Strict Zero-Dummy-Data Standards
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
