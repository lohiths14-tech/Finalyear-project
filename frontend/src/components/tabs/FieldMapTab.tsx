import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { motion, MotionConfig } from 'framer-motion';
import {
  Upload, Map as MapIcon,
  Activity, Zap, TrendingUp,
  MapPin, Eye, Layers, CheckCircle2, Sparkles,
  Printer, Cpu,
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

export function InferenceProgress({
  imageUrl,
  streamProgress,
}: {
  imageUrl: string | null;
  streamProgress: FieldStreamProgress | null;
}) {
  const livePct = typeof streamProgress?.percent === 'number'
    ? Math.max(0, Math.min(100, streamProgress.percent))
    : null;
  const liveMessage = streamProgress?.message;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="inference-progress-wrapper"
      role="status"
      aria-live="polite"
    >
      {imageUrl && (
        <div style={{ width: '100%', maxWidth: 480, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', marginBottom: 8 }}>
          <img src={imageUrl} alt="Uploaded field" style={{ width: '100%', maxHeight: 160, objectFit: 'cover', display: 'block', opacity: 0.7 }} />
        </div>
      )}
      <div className="inference-spinner" aria-hidden="true" />
      <div>
        <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
          Field analysis in progress
        </div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          {liveMessage || 'Processing the submitted image. Progress percentage is unavailable.'}
        </div>
      </div>
      <div className="inference-progress-bar-track" style={{ width: '100%', maxWidth: 480 }}>
        {livePct !== null ? (
          <div
            className="inference-progress-bar-fill"
            role="progressbar"
            aria-label="Backend-reported field analysis progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={livePct}
            style={{ width: `${livePct}%` }}
          />
        ) : <div className="inference-progress-bar-indeterminate" aria-hidden="true" />}
      </div>
      {livePct !== null && <p className="inference-stage-label">{livePct}% complete · backend reported</p>}
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
    grounding_source?: string;
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
        spread_risk: localResults.spread_risk?.trim() || 'UNAVAILABLE',
        clusters: localResults.clusters || [],
        ...(typeof localResults.veg_coverage === 'number'
          ? { vegetation_coverage: localResults.veg_coverage }
          : {}),
        history: copilotMessages.map(m => ({ role: m.role, content: m.text })),
      });

      const assistantMsg = {
        role: 'assistant' as const,
        text: res.answer,
        model: res.model,
        provider: res.provider,
        grounding_source: res.grounding_source,
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
  const infectedPct = localResults
    ? localResults.percentage_infected_area ?? localResults.infection_rate ?? null
    : null;
  const healthyPct = localResults
    ? localResults.percentage_healthy_area ?? (infectedPct !== null ? Math.max(0, 100 - infectedPct) : null)
    : null;
  const clusters = localResults?.clusters || [];
  const delta = localResults?.disease_increasing_pct;

  return (
    <MotionConfig reducedMotion="user">
    <div className="dashboard-container field-layout">
      {/* ── HEADER ACTION CARD ── */}
      <div className="card field-header-card">
        <div className="field-header-card__left">
          <h2 className="field-header-card__title">
            <Activity color="var(--primary)" size={22} /> Field-Level Spatial Diagnosis
          </h2>
          <p className="field-header-card__sub">
            Adaptive grid-based field analysis with scan-specific results and clearly labeled context.
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
                  <span>Adaptive grid analysis</span>
                  <span>Image-based scan output</span>
                  <span>Backend results shown as returned</span>
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
                  <Sparkles size={14} /> Scan result
                </div>
                <h3 style={{ margin: '0 0 8px 0', fontSize: '1.4rem', color: 'var(--text-primary)', fontWeight: 800, letterSpacing: '-0.025em' }}>
                  {localResults.disease_type === 'Healthy' ? 'Field is Healthy' : `Action Required: ${localResults.disease_type}`}
                </h3>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.92rem', lineHeight: 1.6 }}>
                  {localResults.treatment}
                </p>
                <p className="overview-validation-desc">Guidance is shown as returned by the service; verify treatment decisions with a qualified local agronomist.</p>
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
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {typeof localResults.average_severity === 'number' ? `${localResults.average_severity.toFixed(1)}%` : 'Unavailable'}
                  </div>
                </div>
                <div style={{ width: 1, height: 36, background: 'var(--border)' }} />
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Infected Area</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {infectedPct !== null ? `${infectedPct.toFixed(1)}%` : 'Unavailable'}
                  </div>
                </div>
                <div style={{ width: 1, height: 36, background: 'var(--border)' }} />
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Spread Risk</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: localResults.spread_risk === 'High' ? 'var(--danger)' : 'var(--success)' }}>
                    {localResults.spread_risk || 'Not returned'}
                  </div>
                </div>
              </div>
            </div>
          </div>

            {/* Illustrative scenario tools; assumptions are disclosed in each widget. */}
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
              <button
                type="button"
                className="scientific-drawer-header"
                aria-expanded={showDiagnostics}
                onClick={() => setShowDiagnostics((open) => !open)}
              >
                <span className="scientific-drawer-header__label">
                  <Cpu size={14} color="var(--primary)" />
                  <strong>Scan diagnostics</strong>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Moran&apos;s I and adaptive grid values</span>
                </span>
                <span className="scientific-drawer-header__action">
                  <span>{showDiagnostics ? 'Hide details' : 'Show details'}</span>
                  <ChevronRight size={14} aria-hidden="true" />
                </span>
              </button>

              {showDiagnostics && (
                <div className="scientific-drawer-content">
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Moran&apos;s I (backend-reported):</span><br/>
                      <strong style={{ color: 'var(--text-primary)' }}>
                        {typeof localResults.spatial_morans_i === 'number'
                          ? `${localResults.spatial_morans_i >= 0 ? '+' : ''}${localResults.spatial_morans_i.toFixed(4)}`
                          : 'Unavailable'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Adaptive Grid Analysis:</span><br/>
                      <strong style={{ color: 'var(--text-primary)' }}>{gridData.length} adaptive patches returned</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Scan confidence:</span><br/>
                      <strong style={{ color: 'var(--primary)' }}>{localResults.confidence.toFixed(1)}% · as returned by the backend</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="field-workspace-grid">
            
            {/* Interactive scan-derived patch and severity views. */}
            <div className="card" style={{ padding: 20, background: 'var(--bg-card)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
                <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '1rem' }}>
                  <MapPin size={18} color="var(--primary)" /> Spatial Disease Map
                </h4>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button type="button" className={`btn-icon ${viewMode === 'ai_overlay' ? 'active' : ''}`} aria-pressed={viewMode === 'ai_overlay'} onClick={() => setViewMode('ai_overlay')}>
                    <Zap size={14}/> Severity Overlay
                  </button>
                  <button type="button" className={`btn-icon ${viewMode === 'heatmap' ? 'active' : ''}`} aria-pressed={viewMode === 'heatmap'} onClick={() => setViewMode('heatmap')}>
                    <Layers size={14}/> Grid Map
                  </button>
                  <button type="button" className={`btn-icon ${viewMode === 'image' ? 'active' : ''}`} aria-pressed={viewMode === 'image'} onClick={() => setViewMode('image')}>
                    <Eye size={14}/> Original
                  </button>
                  <button type="button" className={`btn-icon ${viewMode === 'gis_map' ? 'active' : ''}`} aria-pressed={viewMode === 'gis_map'} onClick={() => setViewMode('gis_map')}>
                    <MapIcon size={14}/> GIS Satellite
                  </button>
                </div>
              </div>

              {viewMode === 'gis_map' ? (
                <FieldGisMap
                  imageUrl={imageUrl}
                  heatmapDataUrl={effectiveHeatmapUrl}
                  gridData={gridData}
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

                      {/* Derived severity visualization aligned to model grid patches. */}
                      {viewMode === 'ai_overlay' && effectiveHeatmapUrl && (
                        <img
                          src={effectiveHeatmapUrl}
                          alt="Derived severity visualization from field scan output"
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
                              <button
                                key={idx}
                                type="button"
                                className={`spatial-hit-cell ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`}
                                aria-label={`Patch ${idx + 1}, ${cell.disease}, severity ${cell.severity.toFixed(1)} percent`}
                                aria-pressed={isSelected}
                                style={{
                                  left: `${cell.x_percent}%`,
                                  top: `${cell.y_percent}%`,
                                  width: `${cell.w_percent}%`,
                                  height: `${cell.h_percent}%`,
                                }}
                                onMouseEnter={() => setHoveredCell(cell)}
                                onMouseLeave={() => setHoveredCell(null)}
                                onFocus={() => setHoveredCell(cell)}
                                onBlur={() => setHoveredCell(null)}
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
                              <button
                                key={idx}
                                type="button"
                                className="spatial-grid-cell"
                                aria-label={`Patch ${idx + 1}, ${cell.disease}, severity ${cell.severity.toFixed(1)} percent`}
                                aria-pressed={isSelected}
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
                                onFocus={() => setHoveredCell(cell)}
                                onBlur={() => setHoveredCell(null)}
                                onClick={() => setSelectedCell(cell)}
                              >
                                {showCellLabels && sev > 15 && `${sev.toFixed(0)}%`}
                              </button>
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
                          <span>Patch position ({hoveredCell.x_percent.toFixed(0)}%, {hoveredCell.y_percent.toFixed(0)}%)</span>
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
                <div role="group" aria-label="Visual severity display bands, not disease severity classifications" style={{ display: 'flex', gap: 16, fontSize: '0.8rem', background: 'var(--bg-subtle)', padding: '8px 14px', borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--success)', borderRadius: 2 }} />
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Lower display band (&lt;20%)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--warning)', borderRadius: 2 }} />
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Midrange display band (20–50%)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 10, height: 10, background: 'var(--danger)', borderRadius: 2 }} />
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Higher display band (&gt;50%)</span>
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
                      <span style={{ color: 'var(--text-secondary)' }}>Non-infected area {localResults.percentage_healthy_area == null ? '(derived)' : ''}</span>
                      <strong style={{ color: 'var(--success)' }}>{healthyPct !== null ? `${healthyPct.toFixed(1)}%` : 'Unavailable'}</strong>
                    </div>
                    {healthyPct !== null && <div style={{ width: '100%', height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${healthyPct}%`, height: '100%', background: 'var(--success)', borderRadius: 3 }} />
                    </div>}
                  </div>

                  {clusters.length === 0 ? (
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>No disease clusters were returned for this scan.</div>
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
                    <CheckCircle2 size={16} color="var(--success)" /> No infected clusters were returned for this scan.
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
                    Field Scan Assistant
                  </h3>
                </div>
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Ask about values returned for this scan. Verify actionable decisions with local expertise and product labels.
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
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: aiStatus?.display_name ? 'var(--primary)' : 'var(--text-muted)', display: 'inline-block' }} />
                  {aiStatus?.display_name ? `Service-reported: ${aiStatus.display_name}` : 'Service status unavailable'}
                </div>

                <div style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px', borderRadius: 99,
                  background: 'var(--bg-subtle)',
                  border: '1px solid var(--border)',
                  fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)',
                }}>
                  <ShieldCheck size={14} color="var(--primary)" />
                  Current scan context attached
                </div>
              </div>
            </div>

            {/* Quick Action Suggestion Pills */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
                Questions about this field scan
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {[
                  {
                    icon: Layers,
                    label: 'Explain the map patches',
                    q: 'Explain how the scan patches and map overlays are displayed. Distinguish returned scan data from approximate map placement.',
                  },
                  {
                    icon: Activity,
                    label: 'Interpret scan values',
                    q: 'Explain the severity and infection values returned for this scan. State what cannot be inferred from these values alone.',
                  },
                  {
                    icon: ShieldCheck,
                    label: 'What should I verify?',
                    q: 'What additional field observations and local product-label guidance should be checked before making a treatment decision? Do not invent a dose, area, or expected outcome.',
                  },
                  {
                    icon: MapPin,
                    label: 'Review data limits',
                    q: 'Summarize which details in this scan are backend-reported and identify any missing information that limits interpretation.',
                  },
                ].map((pill, idx) => {
                  const PillIcon = pill.icon;
                  return (
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
                      <PillIcon size={14} aria-hidden="true" />
                      <span style={{ fontWeight: 600 }}>{pill.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Chat Thread Messages */}
            <div
              role="log"
              aria-label="Field scan consultation messages"
              aria-live="polite"
              aria-relevant="additions"
              style={{
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
                    Ask about the returned scan values, approximate map placement, or which missing details limit interpretation.
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
                          <strong style={{ color: 'var(--primary)' }}>{msg.provider || (msg.model ? 'Analysis service' : 'Analysis response')}{msg.model ? ` · ${msg.model}` : ''}</strong>
                        </>
                      ) : (
                        <span>You</span>
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
                    {msg.role === 'assistant' && msg.grounding_source && (
                      <span style={{ maxWidth: '92%', marginTop: 4, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        Response source: {msg.grounding_source}
                      </span>
                    )}
                  </div>
                ))
              )}

              {copilotLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', background: 'var(--bg-card)', borderRadius: 10, border: '1px solid var(--border)', width: 'fit-content' }}>
                  <div className="inference-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Preparing a response from the scan context…
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
                aria-label="Ask a question about this field scan"
                placeholder="Ask about the returned scan values or what to verify next…"
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
    </MotionConfig>
  );
}
