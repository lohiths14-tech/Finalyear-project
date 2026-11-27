import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Upload, Map as MapIcon,
  Activity, Zap, TrendingUp,
  MapPin, Eye, Layers, CheckCircle2, Sparkles,
  Printer, Cpu, Image as ImageIcon,
  Bot, Send, MessageSquare, ShieldCheck,
  Award, ChevronRight
} from 'lucide-react';
import { analyzeFieldStream, type FieldStreamProgress, consultField, getAiStatus } from '../../services/api';
import { optimizeImageForUpload } from '../../utils/compress';
import { buildThermalFallbackDataUrl } from '../../utils/thermalHeatmap';
import EconomicImpactWidget from '../ui/EconomicImpactWidget';
import ProgressionSimulator from '../ui/ProgressionSimulator';
import ClinicalCertificateModal from '../ui/ClinicalCertificateModal';
import FieldGisMap from '../ui/FieldGisMap';
import type { FieldResult, GridCell, AiProviderStatus } from '../../types';

// Inference stage labels shown during the 2-3 minute CPU run
const INFERENCE_STAGES = [
  { label: 'Preprocessing field image…',         pct: 8,  icon: ImageIcon },
  { label: 'DINOv2 feature extraction…',          pct: 22, icon: Cpu },
  { label: 'Adaptive grid analysis (64 patches)…', pct: 44, icon: Layers },
  { label: 'Graph Transformer spatial reasoning…', pct: 62, icon: Activity },
  { label: 'Disease cluster detection…',           pct: 76, icon: MapPin },
  { label: 'Severity estimation & calibration…',   pct: 88, icon: Zap },
  { label: 'Generating ICAR recommendations…',     pct: 96, icon: CheckCircle2 },
];

function InferenceProgress({
  imageUrl,
  streamProgress,
}: {
  imageUrl: string | null;
  streamProgress: FieldStreamProgress | null;
}) {
  const [stageIdx, setStageIdx] = useState(0);
  const [estimatedProgress, setEstimatedProgress] = useState(4);

  const livePct = streamProgress?.percent ?? null;
  const liveMessage = streamProgress?.message ?? null;
  const displayProgress = livePct !== null ? livePct : estimatedProgress;

  useEffect(() => {
    if (streamProgress) return;
    const timings = [3000, 12000, 18000, 14000, 12000, 10000, 8000];
    let idx = 0;
    let timer: ReturnType<typeof setTimeout>;
    const advance = () => {
      if (idx >= INFERENCE_STAGES.length - 1) return;
      idx += 1;
      setStageIdx(idx);
      setEstimatedProgress(INFERENCE_STAGES[idx].pct);
      timer = setTimeout(advance, timings[idx] || 10000);
    };
    setEstimatedProgress(INFERENCE_STAGES[0].pct);
    timer = setTimeout(advance, timings[0]);
    return () => clearTimeout(timer);
  }, [streamProgress]);

  const stage = INFERENCE_STAGES[stageIdx];
  const StageIcon = stage.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="inference-progress-wrapper"
    >
      {imageUrl && (
        <div style={{ width: '100%', maxWidth: 480, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', marginBottom: 8 }}>
          <img src={imageUrl} alt="Uploaded field" style={{ width: '100%', maxHeight: 160, objectFit: 'cover', display: 'block', opacity: 0.7 }} />
        </div>
      )}
      <div className="inference-spinner" />
      <div>
        <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
          {streamProgress ? 'AgriPulse Live Stream Active' : 'AgriPulse AI Analysis Running'}
        </div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          {streamProgress ? 'Receiving progress events from the backend' : 'Analysis is running; progress below is a timer-based estimate, not backend telemetry.'}
        </div>
      </div>
      <div className="inference-progress-bar-track" style={{ width: '100%', maxWidth: 480 }}>
        <div className="inference-progress-bar-fill" style={{ width: `${displayProgress}%` }} />
      </div>
      <div className="inference-stage-label">
        <StageIcon size={14} color="var(--primary)" />
        <span>{liveMessage || stage.label}</span>
        <span style={{ marginLeft: 'auto', fontWeight: 700, color: 'var(--primary)', fontSize: '0.78rem' }}>
          {livePct !== null ? `${displayProgress}% · backend` : `~${displayProgress}% · estimate`}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', marginTop: 4 }}>
        {INFERENCE_STAGES.map((_, i) => (
          <div
            key={i}
            style={{
              width: 8, height: 8, borderRadius: '50%',
              background: i <= (livePct !== null ? Math.floor((livePct / 100) * INFERENCE_STAGES.length) : stageIdx) ? 'var(--primary)' : 'var(--border)',
              transition: 'background 0.4s ease',
            }}
          />
        ))}
      </div>
    </motion.div>
  );
}

interface FieldMapTabProps {
  onScanComplete?: (results: FieldResult) => void;
  backendOnline: boolean | null;
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
}

export default function FieldMapTab({ onScanComplete, backendOnline, addToast }: FieldMapTabProps) {
  const [scanning, setScanning] = useState(false);
  const [localResults, setLocalResults] = useState<FieldResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [hoveredCell, setHoveredCell] = useState<GridCell | null>(null);
  const [showCellLabels, setShowCellLabels] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.85);
  const [viewMode, setViewMode] = useState<'ai_overlay' | 'heatmap' | 'image' | 'gis_map'>('ai_overlay');
  const [heatmapImgFailed, setHeatmapImgFailed] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [isCertOpen, setIsCertOpen] = useState(false);
  const [streamProgress, setStreamProgress] = useState<FieldStreamProgress | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Field Intelligence Decision Engine State ──
  const [copilotLoading, setCopilotLoading] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [copilotQuestion, setCopilotQuestion] = useState('');
  const [copilotMessages, setCopilotMessages] = useState<Array<{
    role: 'user' | 'assistant';
    text: string;
    model?: string;
    provider?: string;
    timestamp?: string;
  }>>([]);
  const [aiStatus, setAiStatus] = useState<AiProviderStatus | null>(null);

  useEffect(() => {
    getAiStatus().then(status => setAiStatus(status)).catch(() => {});
  }, []);

  const handleCopilotAsk = useCallback(async (questionText: string) => {
    if (!localResults || !questionText.trim()) return;
    setCopilotLoading(true);
    const userMsg = {
      role: 'user' as const,
      text: questionText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setCopilotMessages(prev => [...prev, userMsg]);
    setCopilotQuestion('');

    try {
      const res = await consultField({
        question: questionText,
        disease_type: localResults.disease_type,
        average_severity: localResults.average_severity,
        infection_rate: localResults.infection_rate,
        spread_risk: localResults.spread_risk || 'MODERATE',
        clusters: localResults.clusters || [],
        vegetation_coverage: localResults.veg_coverage || 85.0,
        land_area: 1.0,
        area_unit: 'acre',
        history: copilotMessages.map(m => ({ role: m.role, content: m.text })),
      });

      const assistantMsg = {
        role: 'assistant' as const,
        text: res.answer,
        model: res.model,
        provider: res.provider,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setCopilotMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      addToast(`Field Copilot request failed: ${err.message}`, 'error');
    } finally {
      setCopilotLoading(false);
    }
  }, [localResults, copilotMessages, addToast]);

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      addToast('Please upload a valid field image (JPG, PNG, WEBP).', 'error');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      addToast('Image too large. Please use an image smaller than 20 MB.', 'error');
      return;
    }
    setScanning(true);
    setStreamProgress(null);
    setLocalResults(null);
    setHeatmapImgFailed(false);
    setSelectedCell(null);
    setHoveredCell(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const objectUrl = URL.createObjectURL(file);
    setImageUrl(objectUrl);

    try {
      addToast('Preparing image & starting AI analysis…', 'info', 4000);
      const optimizedFile = await optimizeImageForUpload(file, 1024);
      const fd = new FormData();
      fd.append('file', optimizedFile);

      const data = await analyzeFieldStream(fd, (prog) => {
        setStreamProgress(prog);
      });
      setLocalResults(data);
      if (data.grid_data?.length) setSelectedCell(data.grid_data[0]);
      if (onScanComplete) onScanComplete(data);
      addToast(`Field Scan Complete: ${data.disease_type}`, 'success');
    } catch (err) {
      const axiosErr = err as { response?: { data?: { detail?: string } }; message?: string; code?: string };
      const detail = axiosErr?.response?.data?.detail;
      let msg = detail || axiosErr?.message || 'Unknown error';
      if (axiosErr?.code === 'ECONNRESET' || msg.includes('ECONNRESET')) {
        msg = 'Connection reset during upload. Please try with a smaller image.';
      } else if (axiosErr?.code === 'ECONNABORTED' || msg.includes('timeout')) {
        msg = 'Analysis timed out. Please try again.';
      }
      addToast(`Field scan failed: ${msg}`, 'error', 8000);
    } finally {
      setScanning(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [addToast, imageUrl, onScanComplete]);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFile(file);
  }, [processFile]);

  const handleLoadDemo = useCallback(async () => {
    try {
      setScanning(true);
      const res = await fetch('/samples/sample_blast.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'rice_blast_benchmark.jpg', { type: 'image/jpeg' });
      await processFile(file);
    } catch (e) {
      console.error('Failed to load demo sample:', e);
      setScanning(false);
    }
  }, [processFile]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await processFile(file);
  }, [processFile]);

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);



  const handlePrintReport = () => {
    window.print();
  };

  const gridData = localResults?.grid_data || [];
  const fallbackHeatmapUrl = useMemo(
    () => buildThermalFallbackDataUrl(gridData),
    [gridData]
  );
  const effectiveHeatmapUrl =
    !heatmapImgFailed && localResults?.heatmap_base64
      ? `data:image/png;base64,${localResults.heatmap_base64}`
      : fallbackHeatmapUrl;
  const infectedPct =
    localResults?.percentage_infected_area ??
    localResults?.infection_rate ??
    0;
  const healthyPct =
    localResults?.percentage_healthy_area ??
    (localResults ? Math.max(0, 100 - infectedPct) : 100);
  const clusters = localResults?.clusters || [];
  const delta = localResults?.disease_increasing_pct;

  return (
    <div className="dashboard-container field-layout">
      {/* ── HEADER ACTION CARD ── */}
      <div className="card field-header-card">
        <div className="field-header-card__left">
          <h2 className="field-header-card__title">
            <Activity color="var(--primary)" size={22} /> Field-Level Spatial Diagnosis
          </h2>
          <p className="field-header-card__sub">
            Adaptive grid-based canopy pathology & ICAR-grounded decision support.
          </p>
        </div>
        <div className="field-header-card__actions">
        {localResults && (
            <>
              <button
                className="btn-icon btn-icon--primary"
                onClick={() => setIsCertOpen(true)}
              >
                <Award size={15} /> Certificate
              </button>
              <button className="btn-icon" onClick={handlePrintReport}>
                <Printer size={15} /> Export Report
              </button>
            </>
          )}
          <button
            type="button"
            className="btn-icon btn-icon--primary"
            onClick={handleLoadDemo}
            disabled={scanning || backendOnline === false}
            title="Load verified Paddy Doctor Drone Benchmark"
          >
            <Sparkles size={15} /> Load Demo Scan
          </button>
          <label className={`btn-primary ${scanning || backendOnline === false ? 'btn-disabled' : ''}`}>
            <Upload size={16} /> Upload Field Image
            <input
              ref={fileInputRef}
              type="file"
              hidden
              accept="image/*"
              disabled={scanning || backendOnline === false}
              onChange={handleUpload}
            />
          </label>
          {backendOnline === false && (
            <span className="field-offline-warning">Backend offline</span>
          )}
        </div>
      </div>

      {scanning ? (
        <InferenceProgress imageUrl={imageUrl} streamProgress={streamProgress} />
      ) : !localResults ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div
            className={`empty-state ${dragOver ? 'drag-over' : ''}`}
            style={{ padding: '48px 24px', textAlign: 'center', cursor: 'pointer' }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="empty-state-icon-wrapper" style={{ margin: '0 auto 16px' }}>
              <MapIcon size={36} color="var(--primary)" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800, margin: '0 0 8px 0' }}>Ready for Field-Level Spatial Diagnostics</h3>
            <p style={{ maxWidth: 520, margin: '0 auto 20px', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.5 }}>
              Upload high-altitude drone orthomosaics, satellite crops, or field photographs — or instantly examine our verified benchmark scan.
            </p>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
              <label 
                className={`btn-primary ${scanning || backendOnline === false ? 'btn-disabled' : ''}`} 
                style={{ cursor: scanning || backendOnline === false ? 'not-allowed' : 'pointer' }}
              >
                <Upload size={16} /> Choose Field Image
                <input 
                  type="file" 
                  hidden 
                  accept="image/*" 
                  disabled={scanning || backendOnline === false} 
                  onChange={handleUpload} 
                />
              </label>
              <button
                type="button"
                className="btn-icon"
                onClick={handleLoadDemo}
                disabled={scanning || backendOnline === false}
                style={{ padding: '8px 18px', borderColor: 'var(--primary)', color: 'var(--primary)', background: 'var(--primary-glow)', display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}
              >
                <Sparkles size={16} /> Load Verified Drone Benchmark
              </button>
            </div>
            <div style={{ marginTop: 24, display: 'flex', gap: 16, justifyContent: 'center', fontSize: '0.78rem', color: 'var(--text-muted)', flexWrap: 'wrap' }}>
              <span>✓ 64-Patch Adaptive Grid</span>
              <span>✓ Moran&apos;s I Autocorrelation</span>
              <span>✓ ICAR Treatment Rules</span>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* ── PHASE 6: DECISION ENGINE PANEL ── */}
          <div 
            className={`card ${localResults.disease_type === 'Healthy' ? 'decision-banner-healthy' : 'decision-banner-warning'}`}
            style={{ padding: 26 }}
          >
            <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'space-between' }}>
              <div style={{ flex: '1 1 400px' }}>
                <div style={{ 
                  fontSize: '0.74rem', 
                  color: localResults.disease_type === 'Healthy' ? 'var(--primary)' : 'var(--warning)', 
                  textTransform: 'uppercase', 
                  letterSpacing: '0.08em', 
                  marginBottom: 6, 
                  fontWeight: 800,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}>
                  <Sparkles size={14} /> AI Decision Engine Recommendation
                </div>
                <h3 style={{ margin: '0 0 8px 0', fontSize: '1.4rem', color: 'var(--text-primary)', fontWeight: 800, letterSpacing: '-0.025em' }}>
                  {localResults.disease_type === 'Healthy' ? 'Field is Healthy' : `Action Required: ${localResults.disease_type}`}
                </h3>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.92rem', lineHeight: 1.6 }}>
                  {localResults.treatment}
                </p>
                {localResults.farmer_friendly && (
                  <div className="decision-summary-box">
                    <p style={{ color: 'var(--primary)', fontSize: '0.78rem', fontWeight: 800, margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Farmer Summary</p>
                    <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: '0 0 4px 0' }}>
                      <strong>What is it?</strong> {localResults.farmer_friendly.what_is_it}
                    </p>
                    <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', margin: 0 }}>
                      <strong>Status:</strong> {localResults.farmer_friendly.field_health_message}
                    </p>
                  </div>
                )}
              </div>

              <div className="decision-stat-card">
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Severity</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: localResults.average_severity > 50 ? 'var(--danger)' : localResults.average_severity > 20 ? 'var(--warning)' : 'var(--success)' }}>
                    {localResults.average_severity > 50 ? 'Severe' : localResults.average_severity > 20 ? 'Moderate' : 'Low'}
                  </div>
                </div>
                <div style={{ width: 1, height: 36, background: 'var(--border)' }} />
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Infected Area</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {infectedPct.toFixed(1)}%
                  </div>
                </div>
                <div style={{ width: 1, height: 36, background: 'var(--border)' }} />
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Spread Risk</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: localResults.spread_risk === 'High' ? 'var(--danger)' : 'var(--success)' }}>
                    {localResults.spread_risk}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Economic ROI Impact & 14-Day Progression Simulator */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 20 }}>
            <EconomicImpactWidget
              diseaseType={localResults.disease_type}
              severity={localResults.average_severity}
              confidence={localResults.confidence}
            />
            <ProgressionSimulator
              initialSeverity={localResults.average_severity}
              diseaseType={localResults.disease_type}
            />

            {/* ── COLLAPSIBLE SCIENTIFIC DIAGNOSTICS DRAWER ── */}
            <div className="scientific-drawer">
              <div 
                className="scientific-drawer-header"
                onClick={() => setShowDiagnostics(!showDiagnostics)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Cpu size={14} color="var(--primary)" />
                  <strong>Scientific Diagnostics &amp; Vision Telemetry</strong>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    (DINOv2 Embeddings · Moran&apos;s I · Spatial Graph)
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>{showDiagnostics ? 'Collapse' : 'Expand Details'}</span>
                  <ChevronRight size={14} style={{ transform: showDiagnostics ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }} />
                </div>
              </div>

              {showDiagnostics && (
                <div className="scientific-drawer-content">
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Spatial Moran&apos;s I Index:</span><br/>
                      <strong style={{ color: 'var(--text-primary)' }}>
                        {typeof localResults.spatial_morans_i === 'number'
                          ? `${localResults.spatial_morans_i >= 0 ? '+' : ''}${localResults.spatial_morans_i.toFixed(4)} (${localResults.spatial_morans_i > 0.15 ? 'Spatial Clustering' : 'Uniform / Scattered'})`
                          : 'Unavailable (insufficient grid nodes)'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Adaptive Grid Analysis:</span><br/>
                      <strong style={{ color: 'var(--text-primary)' }}>{gridData.length} dense patches (8×8 grid)</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Confidence Calibration:</span><br/>
                      <strong style={{ color: 'var(--primary)' }}>{localResults.confidence.toFixed(1)}% (Temperature Scaled)</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Inference Architecture:</span><br/>
                      <strong style={{ color: 'var(--text-primary)' }}>MobileNetV2 ONNX INT8 (2.42 MB)</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="field-workspace-grid">
            
            {/* ── PHASE 3: INTERACTIVE DISEASE HEATMAP ── */}
            <div className="card" style={{ padding: 20, background: 'var(--bg-card)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
                <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '1rem' }}>
                  <MapPin size={18} color="var(--primary)" /> Spatial Disease Map
                </h4>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button className={`btn-icon ${viewMode === 'ai_overlay' ? 'active' : ''}`} onClick={() => setViewMode('ai_overlay')}>
                    <Zap size={14}/> AI Thermal Map
                  </button>
                  <button className={`btn-icon ${viewMode === 'heatmap' ? 'active' : ''}`} onClick={() => setViewMode('heatmap')}>
                    <Layers size={14}/> Grid Map
                  </button>
                  <button className={`btn-icon ${viewMode === 'image' ? 'active' : ''}`} onClick={() => setViewMode('image')}>
                    <Eye size={14}/> Original
                  </button>
                  <button className={`btn-icon ${viewMode === 'gis_map' ? 'active' : ''}`} onClick={() => setViewMode('gis_map')}>
                    <MapIcon size={14}/> GIS Satellite
                  </button>
                </div>
              </div>

              {viewMode === 'gis_map' ? (
                <FieldGisMap
                  imageUrl={imageUrl}
                  heatmapDataUrl={effectiveHeatmapUrl}
                  gridData={gridData}
                  diseaseType={localResults?.disease_type}
                  averageSeverity={localResults?.average_severity}
                  onSelectCell={setSelectedCell}
                />
              ) : (
                <div className="spatial-stage">
                  {imageUrl ? (
                    <div className="spatial-image-frame">
                      <img
                        src={imageUrl}
                        alt="Field canopy"
                        className="spatial-base-img"
                        style={{ opacity: viewMode === 'image' ? 1 : 0.9 }}
                      />

                      {/* Mode 1: AI Thermal Map (Gaussian thermal overlay + interactive patch inspection) */}
                      {viewMode === 'ai_overlay' && effectiveHeatmapUrl && (
                        <img
                          src={effectiveHeatmapUrl}
                          alt="AI Disease Heatmap"
                          className="spatial-heatmap-img"
                          style={{ opacity: heatmapOpacity }}
                          onError={() => {
                            if (!heatmapImgFailed) setHeatmapImgFailed(true);
                          }}
                        />
                      )}

                      {viewMode === 'ai_overlay' && gridData.length > 0 && (
                        <div className="spatial-overlay-layer">
                          {gridData.map((cell, idx) => {
                            const isSelected = selectedCell === cell;
                            const isHovered = hoveredCell === cell;
                            return (
                              <div
                                key={idx}
                                className={`spatial-hit-cell ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`}
                                style={{
                                  left: `${cell.x_percent}%`,
                                  top: `${cell.y_percent}%`,
                                  width: `${cell.w_percent}%`,
                                  height: `${cell.h_percent}%`,
                                }}
                                onMouseEnter={() => setHoveredCell(cell)}
                                onMouseLeave={() => setHoveredCell(null)}
                                onClick={() => setSelectedCell(cell)}
                              />
                            );
                          })}
                        </div>
                      )}

                      {/* Mode 2: Grid Map (Color-coded adaptive grid cells aligned 1:1 with image) */}
                      {viewMode === 'heatmap' && gridData.length > 0 && (
                        <div className="spatial-overlay-layer">
                          {gridData.map((cell, idx) => {
                            const sev = cell.severity || 0;
                            const isSelected = selectedCell === cell;
                            const isHovered = hoveredCell === cell;
                            const bg = sev > 50 ? 'rgba(220,38,38,0.38)' : sev > 20 ? 'rgba(217,119,6,0.38)' : 'rgba(16,185,129,0.26)';
                            const border = sev > 50 ? '#dc2626' : sev > 20 ? '#d97706' : '#10b981';

                            return (
                              <div
                                key={idx}
                                style={{
                                  position: 'absolute',
                                  left: `${cell.x_percent}%`,
                                  top: `${cell.y_percent}%`,
                                  width: `${cell.w_percent}%`,
                                  height: `${cell.h_percent}%`,
                                  background: isHovered ? (sev > 50 ? 'rgba(220,38,38,0.62)' : sev > 20 ? 'rgba(217,119,6,0.58)' : 'rgba(16,185,129,0.5)') : bg,
                                  border: isSelected ? '2px solid #ffffff' : `1px solid ${border}`,
                                  boxShadow: isSelected ? '0 0 0 2px #0f172a, 0 0 8px rgba(0,0,0,0.6)' : isHovered ? `0 0 6px ${border}` : 'none',
                                  transition: 'all 0.15s ease-out',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontSize: '0.68rem',
                                  fontWeight: 700,
                                  color: '#ffffff',
                                  textShadow: '0 1px 2px rgba(0,0,0,0.9)',
                                  zIndex: isSelected ? 10 : isHovered ? 5 : 1,
                                }}
                                onMouseEnter={() => setHoveredCell(cell)}
                                onMouseLeave={() => setHoveredCell(null)}
                                onClick={() => setSelectedCell(cell)}
                              >
                                {showCellLabels && sev > 15 && `${sev.toFixed(0)}%`}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="spatial-no-image">Field image preview unavailable</div>
                  )}

                  {/* Floating Hover Tooltip */}
                  {hoveredCell && viewMode !== 'image' && (
                    <div className="spatial-hover-tooltip">
                      <span>📍 Patch ({hoveredCell.x_percent.toFixed(0)}%, {hoveredCell.y_percent.toFixed(0)}%)</span>
                      <span style={{ color: hoveredCell.severity > 50 ? '#f87171' : hoveredCell.severity > 20 ? '#fbbf24' : '#4ade80', fontWeight: 700 }}>
                        Severity: {hoveredCell.severity.toFixed(1)}%
                      </span>
                      <span style={{ color: '#cbd5e1' }}>{hoveredCell.disease}</span>
                    </div>
                  )}
                </div>
              )}

              {/* View Controls & Legend Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 14, flexWrap: 'wrap', gap: 12 }}>
                <div style={{ display: 'flex', gap: 16, fontSize: '0.8rem', background: 'var(--bg-subtle)', padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--success)', borderRadius: 2 }} />
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Healthy (&lt;20%)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--warning)', borderRadius: 2 }} />
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Moderate (20–50%)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--danger)', borderRadius: 2 }} />
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Severe (&gt;50%)</span>
                  </div>
                </div>

                {viewMode === 'heatmap' && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                    <input 
                      type="checkbox" 
                      checked={showCellLabels} 
                      onChange={(e) => setShowCellLabels(e.target.checked)} 
                    />
                    Show percentage labels
                  </label>
                )}

                {viewMode === 'ai_overlay' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <span>Overlay Opacity:</span>
                    <input 
                      type="range" 
                      min="0.3" 
                      max="1.0" 
                      step="0.05"
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      style={{ width: 90, accentColor: 'var(--primary)' }}
                    />
                    <span>{Math.round(heatmapOpacity * 100)}%</span>
                  </div>
                )}

                {selectedCell && (
                  <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    Selected Cell: ({selectedCell.x_percent.toFixed(0)}%, {selectedCell.y_percent.toFixed(0)}%) · Severity: <span style={{ color: 'var(--danger)' }}>{selectedCell.severity.toFixed(1)}%</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Statistics Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              
              {/* ── PHASE 2: DETECTION STATISTICS ── */}
              <div className="card" style={{ padding: 18, background: 'var(--bg-card)' }}>
                <h4 style={{ margin: '0 0 14px 0', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
                  <Activity size={16} color="var(--primary)" /> Field Health Distribution
                </h4>
                
                <div style={{ display: 'grid', gap: 12 }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Healthy Biomass</span>
                      <strong style={{ color: 'var(--success)' }}>{healthyPct.toFixed(1)}%</strong>
                    </div>
                    <div style={{ width: '100%', height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${healthyPct}%`, height: '100%', background: 'var(--success)', borderRadius: 3 }} />
                    </div>
                  </div>

                  {clusters.length === 0 ? (
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>No disease clusters detected.</div>
                  ) : (
                    clusters.map(c => (
                      <div key={c.id}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{c.disease}</span>
                          <strong style={{ color: 'var(--danger)' }}>{c.coverage_pct.toFixed(1)}%</strong>
                        </div>
                        <div style={{ width: '100%', height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ width: `${c.coverage_pct}%`, height: '100%', background: 'var(--danger)', borderRadius: 3 }} />
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* ── PHASE 4: SPATIAL CLUSTERS ── */}
              <div className="card" style={{ padding: 18, background: 'var(--bg-card)' }}>
                <h4 style={{ margin: '0 0 14px 0', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
                  <Layers size={16} color="var(--primary)" /> Spatial Clusters
                </h4>
                
                {clusters.length === 0 ? (
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                    <CheckCircle2 size={16} color="var(--success)" /> No infected clusters found.
                  </div>
                ) : (
                  <div style={{ display: 'grid', gap: 10 }}>
                    {clusters.map((c, i) => (
                      <div key={i} style={{ background: 'var(--bg-subtle)', padding: '10px 12px', borderRadius: 8, borderLeft: '3px solid var(--danger)' }}>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 2 }}>Cluster #{c.id}</div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>{c.disease}</div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--danger)', marginTop: 2, fontWeight: 600 }}>Coverage: {c.coverage_pct.toFixed(1)}% of field</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* ── PHASE 5: TEMPORAL INTELLIGENCE ── */}
              <div className="card" style={{ padding: 18, background: 'var(--bg-card)' }}>
                <h4 style={{ margin: '0 0 14px 0', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
                  <TrendingUp size={16} color="var(--primary)" /> Temporal Progression
                </h4>
                
                {delta !== undefined && delta !== null ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    <div style={{ 
                      width: 44, height: 44, borderRadius: 10, 
                      background: delta > 0 ? 'var(--danger-glow)' : 'var(--success-glow)',
                      color: delta > 0 ? 'var(--danger)' : 'var(--success)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                      {delta > 0 ? <TrendingUp size={22} /> : <TrendingUp size={22} style={{ transform: 'scaleY(-1)' }}/>}
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Compared to previous observation</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 700, color: delta > 0 ? 'var(--danger)' : 'var(--success)' }}>
                        Disease {delta > 0 ? 'increasing' : 'decreasing'} ({delta > 0 ? '+' : ''}{delta}%)
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    Not enough historical observations to estimate progression.
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* ── PHASE 6: FIELD INTELLIGENCE DECISION ENGINE ── */}
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
            style={{
              marginTop: 24,
              padding: 24,
              background: 'linear-gradient(180deg, var(--bg-card) 0%, rgba(16, 185, 129, 0.03) 100%)',
              border: '1px solid rgba(16, 185, 129, 0.25)',
              borderRadius: 16,
              boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 18, borderBottom: '1px solid var(--border)', paddingBottom: 16 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: 8,
                    background: 'rgba(16, 185, 129, 0.15)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Bot size={20} color="var(--primary)" />
                  </div>
                  <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                    Dr. AgriPulse — Field Spatial Reasoning Engine
                  </h3>
                </div>
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Advanced epidemiological reasoning grounded in 64-patch DINOv2 vision telemetry and ICAR-IRRI precision protocols.
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 99,
                  background: 'rgba(16, 185, 129, 0.12)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  fontSize: '0.78rem', fontWeight: 700, color: 'var(--primary)',
                }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--primary)', display: 'inline-block', boxShadow: '0 0 8px var(--primary)' }} />
                  {aiStatus?.display_name || 'ICAR Deterministic Decision Engine'}
                </div>

                <div style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 99,
                  background: 'var(--bg-subtle)',
                  border: '1px solid var(--border)',
                  fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)',
                }}>
                  <ShieldCheck size={14} color="var(--primary)" />
                  ICAR / IRRI Telemetry-Grounded
                </div>
              </div>
            </div>

            {/* Quick Action Suggestion Pills */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                Recommended Agronomic Inquiries for this Field Scan
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {[
                  {
                    icon: '🎯',
                    label: 'Spot-Spray Containment',
                    q: `How should I set up a barrier buffer to isolate the ${clusters.length} detected clusters and contain ${localResults.disease_type} without spraying the whole field?`,
                  },
                  {
                    icon: '🧪',
                    label: 'Knapsack Tank Dilution',
                    q: `Calculate the exact 16L knapsack sprayer tanks, chemical dosage, and water required for the ${localResults.infection_rate.toFixed(1)}% infected acreage.`,
                  },
                  {
                    icon: '🌧️',
                    label: 'Weather & Spread Vectors',
                    q: `Given the ${localResults.spread_risk || 'MODERATE'} spread risk, what weather window or wind speed should I look for before spraying?`,
                  },
                  {
                    icon: '📉',
                    label: 'Economic Loss Risk',
                    q: `What is the economic loss and yield penalty risk at ${localResults.average_severity.toFixed(1)}% average canopy severity if treatment is delayed 48 hours?`,
                  },
                ].map((pill, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="btn-icon"
                    disabled={copilotLoading}
                    onClick={() => handleCopilotAsk(pill.q)}
                    style={{
                      padding: '7px 14px',
                      fontSize: '0.78rem',
                      borderRadius: 20,
                      background: 'var(--bg-subtle)',
                      border: '1px solid var(--border)',
                      display: 'flex', alignItems: 'center', gap: 6,
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <span>{pill.icon}</span>
                    <span style={{ fontWeight: 600 }}>{pill.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Chat Thread Messages */}
            <div style={{
              minHeight: 120,
              maxHeight: 380,
              overflowY: 'auto',
              background: 'var(--bg-base)',
              borderRadius: 12,
              border: '1px solid var(--border)',
              padding: 16,
              marginBottom: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 14,
            }}>
              {copilotMessages.length === 0 ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '30px 16px',
                  color: 'var(--text-muted)',
                  textAlign: 'center',
                }}>
                  <MessageSquare size={32} style={{ opacity: 0.4, marginBottom: 8 }} />
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Field Intelligence Copilot Ready
                  </div>
                  <div style={{ fontSize: '0.8rem', maxWidth: 440, marginTop: 4 }}>
                    Ask questions about spatial containment, spot spraying, barrier perimeters, knapsack pumps, or economic thresholds.
                  </div>
                </div>
              ) : (
                copilotMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      marginBottom: 4,
                      fontSize: '0.72rem',
                      color: 'var(--text-muted)',
                    }}>
                      {msg.role === 'assistant' ? (
                        <>
                          <Sparkles size={12} color="var(--primary)" />
                          <strong style={{ color: 'var(--primary)' }}>Dr. AgriPulse ({msg.model || 'ICAR Grounded Engine'})</strong>
                        </>
                      ) : (
                        <span>You (Agronomist)</span>
                      )}
                      {msg.timestamp && <span>· {msg.timestamp}</span>}
                    </div>

                    <div style={{
                      maxWidth: '92%',
                      padding: '12px 16px',
                      borderRadius: 12,
                      fontSize: '0.84rem',
                      lineHeight: 1.6,
                      background: msg.role === 'user' ? 'var(--primary)' : 'var(--bg-card)',
                      color: msg.role === 'user' ? '#ffffff' : 'var(--text-primary)',
                      border: msg.role === 'user' ? 'none' : '1px solid var(--border)',
                      whiteSpace: 'pre-wrap',
                    }}>
                      {msg.text}
                    </div>
                  </div>
                ))
              )}

              {copilotLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', background: 'var(--bg-card)', borderRadius: 10, border: '1px solid var(--border)', width: 'fit-content' }}>
                  <div className="inference-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Analyzing spatial canopy telemetry against ICAR protocols…
                  </span>
                </div>
              )}
            </div>

            {/* Input Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (copilotQuestion.trim() && !copilotLoading) {
                  handleCopilotAsk(copilotQuestion);
                }
              }}
              style={{ display: 'flex', gap: 10 }}
            >
              <input
                type="text"
                value={copilotQuestion}
                onChange={(e) => setCopilotQuestion(e.target.value)}
                placeholder={`Ask a question about this ${localResults.disease_type} field scan (e.g. spray buffer size, ULV drone spray, dosage)...`}
                disabled={copilotLoading}
                style={{
                  flex: 1,
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  padding: '10px 14px',
                  fontSize: '0.84rem',
                  color: 'var(--text-primary)',
                  outline: 'none',
                }}
              />
              <button
                type="submit"
                className="btn-primary"
                disabled={copilotLoading || !copilotQuestion.trim()}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '0 20px',
                  borderRadius: 10,
                  fontSize: '0.84rem',
                  fontWeight: 700,
                }}
              >
                <Send size={15} /> Consult
              </button>
            </form>
          </motion.div>
        </>
      )}

      {/* Clinical Pathology Certificate Modal */}
      <ClinicalCertificateModal
        isOpen={isCertOpen}
        onClose={() => setIsCertOpen(false)}
        scanResult={localResults}
      />
    </div>
  );
}
