import { useState, useEffect } from 'react';
import { motion, MotionConfig } from 'framer-motion';
import NumberFlow from '@number-flow/react';
import {
  ShieldCheck, AlertTriangle, Activity, Award, ZoomIn, X, Sparkles
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

    smartRecommendation = results.treatment || 'No scan-specific guidance was returned.';

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
    <MotionConfig reducedMotion="user">
    <motion.div key="overview" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
      <header className="overview-hero">
        <div>
          <p className="overview-hero__eyebrow">AGRI PULSE / FIELD INTELLIGENCE</p>
          <h2 className="overview-hero__title">Field command center</h2>
          <p className="overview-hero__copy">A session-level view of scan output, local environment context, and model evaluation.</p>
        </div>
        <span className={`overview-hero__state${results ? ' is-active' : ''}`}>
          <span aria-hidden="true" />{results ? 'Scan loaded' : 'Awaiting scan'}
        </span>
      </header>

      {/* System Status Banner — only shown when no scan */}
      {!results && (
        <div className="overview-status-banner">
          <div className="overview-status-banner__left">
            <Sparkles size={18} className="overview-status-banner__icon" />
            <div>
              <strong className="overview-status-banner__title">No live field scan yet</strong>
              <div className="overview-status-banner__sub">
                {metricsStatus === 'loading'
                  ? 'Loading evaluation metrics separately from field measurements…'
                  : metricsStatus === 'available'
                  ? 'No session scan is attached. Evaluation metrics are shown separately below.'
                  : 'Evaluation metrics are unavailable. Run a field analysis to view scan output.'}
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
            ) : 'No scan yet'}
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
              {results ? 'Taxonomic identification is not included in this scan result.' : 'No active field scan'}
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
            {results ? smartRecommendation : 'No field scan is available in this session. Run a field analysis to view scan-specific output.'}
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
          <span className="overview-section-badge">Environment context</span>
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
                Only backend-marked observed history is shown; single-scan estimates are excluded.
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
          <h3 className="overview-chart-title">Model-derived proxy features</h3>
          {results ? (
            <div style={{ height: 320 }}>
              <ResponsiveContainer width="100%" height={320} minWidth={0}>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={results.spectral_radar}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <RechartsTooltip />
                  <Radar name="Proxy A" dataKey="A" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.5} />
                  <Radar name="Proxy B" dataKey="B" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={Activity} message="No model activation data yet" />
          )}
        </div>
      </div>

      <section className="card glass overview-ai-stack" aria-labelledby="overview-provider-title">
        <div className="overview-ai-stack__header">
          <div className="overview-ai-stack__title-row">
            <Activity size={18} color="var(--primary)" />
            <div>
              <h3 id="overview-provider-title" className="overview-ai-stack__title">Inference service</h3>
              <p className="overview-validation-desc">Provider information is operational status, not a measure of field health.</p>
            </div>
          </div>
          <div className="overview-ai-status-pill" role="status">
            <span className="overview-ai-status-dot" />
            {aiStatus?.display_name || 'Provider status unavailable'}
          </div>
        </div>
      </section>

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
      {metrics && (
        <div className="card glass overview-validation-card">
          <div className="overview-validation-card__header">
            <div className="overview-validation-card__title-row">
              <Award size={18} color="var(--primary)" />
            <h3 className="overview-ai-stack__title">Model evaluation metrics</h3>
          </div>
          <span className="overview-validation-card__badge">Values provided by benchmark file</span>
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

            </div>

            {metrics.has_cm && <div className="overview-cm-wrapper">
              <button
                type="button"
                className="overview-cm-thumbnail"
                onClick={() => setIsZoomed(true)}
                aria-label="Enlarge model confusion matrix"
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
              </button>
            </div>}
          </div>
        </div>
      )}

      {/* Fullscreen Zoom Modal */}
      {isZoomed && (
        <div className="overview-modal-backdrop" onClick={() => setIsZoomed(false)}>
          <div className="overview-modal-box" role="dialog" aria-modal="true" aria-labelledby="overview-modal-title" onClick={(e) => e.stopPropagation()}>
            <div className="overview-modal-header">
              <h4 id="overview-modal-title" className="overview-modal-title">Confusion Matrix (Full Resolution)</h4>
              <button type="button" className="btn-icon" aria-label="Close confusion matrix" onClick={() => setIsZoomed(false)}>
                <X size={16} />
              </button>
            </div>
            <img
              src={`${(import.meta.env.VITE_API_URL || '/api/v1').replace(/\/$/, '')}/model_weights/confusion_matrix.png`}
              alt="Confusion Matrix Large"
              className="overview-modal-img"
            />
            <p className="overview-modal-caption">
              Model evaluation confusion matrix supplied by the benchmark service.
            </p>
          </div>
        </div>
      )}
    </motion.div>
    </MotionConfig>
  );
}
