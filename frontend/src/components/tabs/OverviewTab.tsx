import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import NumberFlow from '@number-flow/react';
import {
  ShieldCheck, AlertTriangle, Activity, Award, CheckCircle2, ZoomIn, X, Sparkles, Bot, Cpu
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PieChart, Pie, Cell
} from 'recharts';
import EmptyState from '../ui/EmptyState';
import { Skeleton } from '../ui/Skeleton';
import WeatherWidget from '../WeatherWidget';
import { getAiStatus } from '../../services/api';
import type { FieldResult, AiProviderStatus } from '../../types';

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ color: string; name: string; value: number }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (active && payload?.length) {
    return (
      <div className="custom-tooltip glass">
        <p className="label">{label}</p>
        {payload.map((e, i) => (
          <p key={i} style={{ color: e.color }}>
            {e.name}: {e.value != null ? `${e.value}%` : 'N/A'}
          </p>
        ))}
      </div>
    );
  }
  return null;
}

interface OverviewTabProps {
  lastScan?: FieldResult | null;
}

interface ModelMetrics {
  accuracy: number;
  macro_f1?: number;
  ece_score?: number;
  has_cm: boolean;
}

type MetricsStatus = 'loading' | 'available' | 'unavailable';

interface ClassesJson {
  validation_accuracy?: number;
  macro_f1?: number;
  ece_score?: number;
  has_cm?: boolean;
}

export default function OverviewTab({ lastScan }: OverviewTabProps) {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [metricsStatus, setMetricsStatus] = useState<MetricsStatus>('loading');
  const [isZoomed, setIsZoomed] = useState(false);
  const [aiStatus, setAiStatus] = useState<AiProviderStatus | null>(null);

  useEffect(() => {
    getAiStatus().then(data => setAiStatus(data)).catch(() => {});
  }, []);

  useEffect(() => {
    const API_BASE = (import.meta.env.VITE_API_URL || '/api/v1').replace(/\/$/, '');
    fetch(`${API_BASE}/model_weights/classes.json`)
      .then((res) => {
        if (!res.ok) throw new Error('Model benchmark request failed');
        return res.json();
      })
      .then((data: ClassesJson) => {
        if (data.validation_accuracy === undefined) {
          setMetricsStatus('unavailable');
          return;
        }
        setMetrics({
          accuracy: data.validation_accuracy,
          macro_f1: data.macro_f1,
          ece_score: data.ece_score,
          has_cm: data.has_cm ?? false
        });
        setMetricsStatus('available');
      })
      .catch(() => setMetricsStatus('unavailable'));
  }, []);

  const results = lastScan;
  const measuredProgression = results?.temporal_available
    ? results.progression_data.filter((point) => point.simulated !== true)
    : [];

  let distributionData: Array<{ name: string; value: number; color: string }> = [];
  let smartRecommendation = '';
  let trendChange: string | null = null;
  let trendDirection: string | null = null;
  let coverageStr = 'N/A';

  if (results) {
    const grid = results.grid_data || [];
    let healthy = 0, moderate = 0, severe = 0;
    grid.forEach((cell) => {
      if (cell.severity < 20 || cell.disease.includes('Healthy')) healthy++;
      else if (cell.severity <= 60) moderate++;
      else severe++;
    });
    const total = Math.max(1, grid.length);
    distributionData = [
      { name: 'Healthy', value: parseFloat(((healthy / total) * 100).toFixed(1)), color: 'var(--success)' },
      { name: 'Moderate', value: parseFloat(((moderate / total) * 100).toFixed(1)), color: 'var(--warning)' },
      { name: 'Severe', value: parseFloat(((severe / total) * 100).toFixed(1)), color: 'var(--danger)' }
    ].filter(d => d.value > 0);

    const disease = results.disease_type;
    const effectiveDisease = ((disease === 'Unknown' || disease?.includes('Unknown')) && results.alternatives?.[0])
      ? results.alternatives[0].disease
      : disease;
    const severity = results.average_severity;
    const spread = results.infection_rate;
    if (effectiveDisease && effectiveDisease !== 'Unknown' && !effectiveDisease.includes('Healthy') && !effectiveDisease.includes('Unknown')) {
      smartRecommendation = `${spread > 30 ? 'High' : 'Moderate'} spread of ${effectiveDisease.split(' (')[0]} detected. Immediate ${severity > 50 ? 'aggressive ' : ''}treatment recommended within ${severity > 60 ? '48 hours' : '1 week'}. ${results.treatment}`;
    } else if (effectiveDisease?.includes('Healthy')) {
      smartRecommendation = "Field is healthy. Maintain current practices.";
    } else {
      smartRecommendation = results.treatment;
    }

    coverageStr = `${results.infection_rate.toFixed(1)}%`;
    if (results.previous_infection_rate !== undefined && results.previous_infection_rate !== null) {
      const diff = results.infection_rate - results.previous_infection_rate;
      trendChange = Math.abs(diff).toFixed(1);
      trendDirection = diff > 0 ? 'increasing' : diff < 0 ? 'decreasing' : 'stable';
      if (Math.abs(diff) < 0.1) {
        trendChange = '0';
        trendDirection = 'stable';
      }
    } else {
      trendDirection = null;
      trendChange = null;
    }
  }

  return (
    <motion.div key="overview" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>

      {/* System Status Banner — only shown when no scan */}
      {!results && (
        <div className="overview-status-banner">
          <div className="overview-status-banner__left">
            <Sparkles size={18} className="overview-status-banner__icon" />
            <div>
              <strong className="overview-status-banner__title">No live field scan yet</strong>
              <div className="overview-status-banner__sub">
                {metricsStatus === 'loading'
                  ? 'Loading model benchmark data…'
                  : metricsStatus === 'available'
                  ? 'Model validation benchmarks are shown separately from live field measurements below.'
                  : 'Model benchmark data unavailable. Run a field scan to see measured field data.'}
              </div>
            </div>
          </div>
          <span className="overview-status-banner__badge">No scan data</span>
        </div>
      )}

      {/* Stats Grid */}
      <div className="stats-grid">
        {/* Field Health Score */}
        <div className="card glass glow-on-hover">
          <div className="card-header">
            <p className="card-label">Field Health Score</p>
            <ShieldCheck size={18} color="var(--success)" />
          </div>
          <h3 className="card-value overview-card-value-flex">
            {results ? (
              <>
                <NumberFlow value={parseFloat((100 - results.average_severity).toFixed(1))} />
                <span>/100</span>
              </>
            ) : 'Data unavailable'}
          </h3>
          {results && (
            <div className="progress-bar" aria-label="Field health score">
              <motion.div
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ width: `${100 - results.average_severity}%` }}
                style={{ background: 'var(--success)' }}
              />
            </div>
          )}
          {results && trendDirection && (
            <div className="overview-trend-row">
              <span>Trend: {trendDirection}</span>
              <span className={trendDirection === 'increasing' && results.average_severity > 20 ? 'text-danger' : 'text-success'}>
                {trendDirection === 'increasing' ? '+' : trendDirection === 'decreasing' ? '-' : ''}{trendChange}%
              </span>
            </div>
          )}
          {metrics && (
            <div className="overview-val-accuracy">
              Model validation benchmark: <NumberFlow value={parseFloat(metrics.accuracy.toFixed(1))} />%
            </div>
          )}
        </div>

        {/* Dominant Disease */}
        <div className="card glass">
          <div className="card-header">
            <p className="card-label">Dominant Disease</p>
            {results?.disease_type === 'Healthy' ? (
            <ShieldCheck size={18} color={results ? 'var(--success)' : 'var(--text-muted)'} />

            ) : results?.disease_type?.includes('Unknown') ? (
              <AlertTriangle size={18} color="var(--warning)" />
            ) : (
              <Activity size={18} color="var(--primary)" />
            )}
          </div>
          <div className="overview-disease-row">
            <h3
              className="card-value overview-disease-name"
              style={{
                color: results
                  ? results.disease_type === 'Healthy'
                    ? 'var(--success)'
                    : results.disease_type?.includes('Unknown')
                    ? 'var(--warning)'
                    : 'var(--danger)'
                  : 'var(--text-muted)'
              }}
            >
              {results
                ? (results.disease_type?.includes('Unknown') && results.alternatives?.[0]
                    ? `${results.alternatives[0].disease.split(' (')[0]} (Tentative)`
                    : results.disease_type)
                : 'Data unavailable'}
            </h3>
            {results && results.confidence !== undefined && (
              <span
                className="overview-confidence-badge"
                style={{
                  background: results.confidence > 70 ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
                  color: results.confidence > 70 ? 'var(--success)' : 'var(--warning)',
                  border: `1px solid ${results.confidence > 70 ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'}`,
                }}
              >
                <NumberFlow value={parseFloat(results.confidence.toFixed(1))} />% Conf.
              </span>
            )}
          </div>
          <p className="overview-taxon-name">
            {results ? (
              results.disease_type?.includes('Rice Blast') || results.alternatives?.[0]?.disease?.includes('Rice Blast') ? 'Magnaporthe oryzae' :
              results.disease_type?.includes('Bacterial Blight') || results.alternatives?.[0]?.disease?.includes('Bacterial Blight') ? 'Xanthomonas oryzae pv. oryzae' :
              results.disease_type?.includes('Brown Spot') || results.alternatives?.[0]?.disease?.includes('Brown Spot') ? 'Bipolaris oryzae' :
              results.disease_type?.includes('Leaf Scald') || results.alternatives?.[0]?.disease?.includes('Leaf Scald') ? 'Microdochium oryzae' :
              results.disease_type?.includes('Healthy') ? 'Oryza sativa (Healthy Canopy)' :
              'Taxonomic Classification'
            ) : 'No active field scan'}
          </p>
          <p className="card-sub">{results ? `Coverage: ${coverageStr}` : 'Waiting for input — no live field scan data.'}</p>
          {results && distributionData.length > 0 && (
            <div className="overview-pie-wrapper">
              <div style={{ height: '65px' }}>
                <ResponsiveContainer width="100%" height={65} minWidth={0}>
                  <PieChart>
                    <Pie data={distributionData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={14} outerRadius={28} stroke="none">
                      {distributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip contentStyle={{ background: 'var(--bg-card)', border: 'none', borderRadius: '8px', fontSize: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="overview-pie-legend">
                {distributionData.map((d) => (
                  <span key={d.name} className="overview-pie-legend-item">
                    <span className="overview-pie-dot" style={{ background: d.color }} />
                    {d.name}: {d.value}%
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Smart Recommendation */}
        <div className="card glass">
          <div className="card-header">
            <p className="card-label">Smart Recommendation</p>
            <AlertTriangle size={18} color="var(--warning)" />
          </div>
          <p className="card-sub treatment-text overview-recommendation-text">
            {results ? smartRecommendation : 'No field scan is available. Upload a field image to view measured data and agronomic guidance.'}
          </p>
          {results && (
            <span className={`risk-badge risk-${results.spread_risk?.toLowerCase()}`}>
              Spread Risk: {results.spread_risk}
            </span>
          )}
        </div>
      </div>

      {/* Field Environment Intelligence */}
      <div className="overview-section">
        <div className="overview-section-header">
          <h3 className="overview-section-title">Field Environment Intelligence</h3>
          <span className="overview-section-badge">Live · Open-Meteo · Nominatim · GBIF</span>
        </div>
        <WeatherWidget />
      </div>

      {/* Charts Grid */}
      <div className="main-grid overview-charts-grid">
        <div className="card glass">
          <div className="overview-chart-header">
            <div>
              <h3 className="overview-chart-title">Spatio-Temporal Progression</h3>
              <p className="overview-chart-sub">
                Observed measurements and model estimates are labeled separately.
              </p>
            </div>
            <div className="chart-legend">
              <span className="legend-item"><span className="dot blue" /> Model estimate</span>
              <span className="legend-item"><span className="dot green" /> Observed</span>
            </div>
          </div>
          {measuredProgression.length > 0 ? (
            <div style={{ height: 320 }}>
              <ResponsiveContainer width="100%" height={320} minWidth={0}>
                <AreaChart data={measuredProgression}>

                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--success)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="var(--success)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.1} />
                      <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="day" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                  <RechartsTooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="predicted" stroke="var(--primary)" strokeDasharray="5 5" fill="url(#colorPredicted)" connectNulls />
                  <Area type="monotone" dataKey="actual" stroke="var(--success)" strokeWidth={3} fill="url(#colorActual)" connectNulls />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={Activity} message="No measured temporal data available. Single-scan estimates are hidden." />
          )}
        </div>

        <div className="card glass">
          <h3 className="overview-chart-title">Model Activation Features</h3>
          {results ? (
            <div style={{ height: 320 }}>
              <ResponsiveContainer width="100%" height={320} minWidth={0}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={results.spectral_radar}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <Radar name="Field" dataKey="A" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.5} />
                  <Radar name="Reference" dataKey="B" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={Activity} message="No model activation data yet" />
          )}
        </div>
      </div>

      {/* Hybrid AI Architecture Card */}
      <div className="card glass overview-ai-stack">
        <div className="overview-ai-stack__header">
          <div className="overview-ai-stack__title-row">
            <Sparkles size={18} color="var(--primary)" />
            <h3 className="overview-ai-stack__title">Hybrid AI Architecture: DINOv2 Foundation + ICAR Decision Engine</h3>
          </div>
          <div className="overview-ai-status-pill">
            <span className="overview-ai-status-dot" />
            {aiStatus?.display_name || 'AI provider status unavailable'}
          </div>
        </div>

        <div className="overview-ai-grid">
          {[
            { icon: Cpu, iconColor: 'var(--primary)', label: 'Vision Backbone', title: 'DINOv2 ViT-S/14', desc: '384-dim dense patch representation' },
            { icon: Activity, iconColor: 'var(--accent)', label: 'Spatial Topology', title: 'Graph Transformer', desc: '64-Patch adaptive grid spatial clusters' },
            { icon: Bot, iconColor: 'var(--primary)', label: 'Decision Engine', title: 'ICAR Grounded Advisory', desc: 'Evidence-based spatial containment & dosing' },
            { icon: ShieldCheck, iconColor: 'var(--success)', label: 'Knowledge Base', title: 'ICAR-IRRI 2026 PoP', desc: 'FRAC codes, PHI & tank dilution math' },
          ].map(({ icon: Icon, iconColor, label, title, desc }) => (
            <div key={label} className="overview-ai-grid-item">
              <div className="overview-ai-grid-item__label">
                <Icon size={14} color={iconColor} />
                <span>{label}</span>
              </div>
              <div className="overview-ai-grid-item__title">{title}</div>
              <div className="overview-ai-grid-item__desc">{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {metricsStatus === 'loading' && (
        <div className="card glass overview-skeleton-card" role="status" aria-label="Loading model benchmark data" aria-busy="true">
          <Skeleton width={200} height={20} style={{ marginBottom: 16 }} />
          <div className="overview-skeleton-grid">
            <Skeleton width="100%" height={70} borderRadius={10} />
            <Skeleton width="100%" height={70} borderRadius={10} />
            <Skeleton width="100%" height={70} borderRadius={10} />
          </div>
        </div>
      )}

      {metricsStatus === 'unavailable' && (
        <div className="card glass" role="status">
          <h3 className="overview-ai-stack__title">Model benchmark data unavailable</h3>
          <p className="overview-validation-desc">Live field measurements remain separate from model evaluation results.</p>
        </div>
      )}

      {/* Model Validation & Confusion Matrix */}
      {metrics?.has_cm && (
        <div className="card glass overview-validation-card">
          <div className="overview-validation-card__header">
            <div className="overview-validation-card__title-row">
              <Award size={18} color="var(--primary)" />
              <h3 className="overview-ai-stack__title">Model Validation & Performance</h3>
            </div>
            <span className="overview-validation-card__badge">Evaluated on unseen test holdouts</span>
          </div>

          <div className="overview-validation-grid">
            <div className="overview-validation-metrics">
                <p className="overview-validation-desc">
                  Model evaluation benchmark, not a live field measurement. Metrics are loaded from the backend benchmark file.
                </p>


              <div className="overview-metric-grid">
                {[
                  { label: 'Validation Accuracy', value: `${metrics.accuracy.toFixed(1)}%`, color: 'var(--primary)' },
                  { label: 'Macro F1', value: metrics.macro_f1 !== undefined ? metrics.macro_f1.toFixed(3) : 'Data unavailable', color: 'var(--accent)' },
                  { label: 'ECE Score', value: metrics.ece_score !== undefined ? metrics.ece_score.toFixed(3) : 'Data unavailable', color: 'var(--success)' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="overview-metric-box">
                    <div className="overview-metric-box__label">{label}</div>
                    <div className="overview-metric-box__value" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="overview-calibration-note">
                <CheckCircle2 size={13} color="var(--success)" />
                <span>Calibrated confidence via Temperature Scaling</span>
              </div>
            </div>

            <div className="overview-cm-wrapper">
              <div
                className="overview-cm-thumbnail"
                onClick={() => setIsZoomed(true)}
                title="Click to zoom in"
              >
                <img
                  src={`${(import.meta.env.VITE_API_URL || '/api/v1').replace(/\/$/, '')}/model_weights/confusion_matrix.png`}
                  alt="Confusion Matrix"
                  className="overview-cm-img"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
                <div className="overview-cm-zoom-hint">
                  <ZoomIn size={11} /> Click to enlarge
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen Zoom Modal */}
      {isZoomed && (
        <div className="overview-modal-backdrop" onClick={() => setIsZoomed(false)}>
          <div className="overview-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="overview-modal-header">
              <h4 className="overview-modal-title">Confusion Matrix (Full Resolution)</h4>
              <button className="btn-icon" onClick={() => setIsZoomed(false)}>
                <X size={16} />
              </button>
            </div>
            <img
              src={`${(import.meta.env.VITE_API_URL || '/api/v1').replace(/\/$/, '')}/model_weights/confusion_matrix.png`}
              alt="Confusion Matrix Large"
              className="overview-modal-img"
            />
            <p className="overview-modal-caption">
              Normalized classification distribution across 5 target classes.
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
}
