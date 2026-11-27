import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Radio, Sparkles, Info } from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid
} from 'recharts';
import type { FieldResult } from '../../types';

interface SpectralTabProps {
  lastScan: FieldResult | null;
}

interface NgvcData { intensity: number; value: number; }

// Illustrative reference values only; these are not measured crop or model data.
const HEALTHY_BASELINE_NGVC = Array.from({ length: 32 }, (_, i) => ({
  intensity: i * 8,
  value: Math.max(0, 0.08 - 0.002 * i + 0.001 * Math.sin(i * 0.8)),
  baseline: Math.max(0, 0.09 - 0.0022 * i + 0.0012 * Math.sin(i * 0.7)),
}));

const HEALTHY_FEATURE_REFERENCE = [
  { subject: 'Chlorophyll (proxy)', modelA: 0, modelB: 88 },
  { subject: 'Texture (illustrative)', modelA: 0, modelB: 72 },
  { subject: 'Edge (illustrative)', modelA: 0, modelB: 65 },
  { subject: 'Color (illustrative)', modelA: 0, modelB: 91 },
  { subject: 'Pattern (illustrative)', modelA: 0, modelB: 78 },
  { subject: 'Canopy (illustrative)', modelA: 0, modelB: 85 },
];

export default function SpectralTab({ lastScan }: SpectralTabProps) {
  const hasNgvc = !!(lastScan?.ngvc_data && lastScan.ngvc_data.length > 0);
  const hasFeatures = !!(lastScan?.spectral_radar && lastScan.spectral_radar.length > 0);
  const hasModelData = hasNgvc || hasFeatures;

  const mergedNgvc = useMemo(() => {
    if (!hasNgvc) return HEALTHY_BASELINE_NGVC;
    return (lastScan!.ngvc_data as NgvcData[]).slice(0, 64).map((d, i) => ({
      ...d,
      baseline: HEALTHY_BASELINE_NGVC[Math.floor(i / 2)]?.baseline ?? 0,
    }));
  }, [lastScan, hasNgvc]);

  const mergedFeatures = useMemo(() => {
    if (!hasFeatures) return HEALTHY_FEATURE_REFERENCE;
    return lastScan!.spectral_radar.map((d: { subject: string; A: number; B: number }) => ({
      subject: d.subject,
      modelA: d.A,
      modelB: d.B,
    }));
  }, [lastScan, hasFeatures]);

  return (
    <motion.div
      key="spectral"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="tab-viewport spectral-layout"
    >
      {/* Transparency Banner */}
      <div className="spectral-reference-banner">
        <Info size={16} color="var(--primary)" className="spectral-banner-icon" />
        <div>
          <div className="spectral-banner-title">
            {hasModelData
              ? `Image-derived model features for: ${lastScan?.disease_type}`
              : 'Illustrative reference example (not model output)'}
          </div>
          <div className="spectral-banner-sub">
            {hasModelData
              ? 'NGVC values are derived from the uploaded image; neural feature values are model-derived proxies, not direct physical spectral measurements. Gray curves and unscanned chart values are illustrative references only.'
              : 'The displayed curves and feature values are illustrative examples only; they are not measurements, field observations, IRRI norms, or model outputs.'}
          </div>
        </div>
        {!hasModelData && (
          <span className="spectral-reference-badge">REFERENCE MODE</span>
        )}
      </div>

      <div className="main-grid">
        {/* NGVC Spectral Curve */}
        <div className="card spectral-chart-card">
          <div className="spectral-chart-header">
            <h3 className="spectral-chart-title">
              <Radio size={17} color="var(--primary)" /> Image-derived NGVC profile
            </h3>
            <span className="spectral-badge spectral-badge--primary">Image-derived profile</span>
          </div>
          <p className="spectral-chart-desc">
            Normalized Gray Value Curve derived from the uploaded image; it is not a multispectral sensor reading or a direct chlorophyll measurement.
          </p>

          <div className="spectral-legend">
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
              {hasNgvc ? 'Uploaded image (NGVC)' : 'Illustrative example'}
            </div>
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: '#94a3b8' }} />
              Illustrative reference
            </div>
          </div>

          <div className="spectral-chart-wrapper">
            <ResponsiveContainer width="100%" height={320} minWidth={0}>
              <AreaChart data={mergedNgvc} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="intensity" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <Tooltip
                  formatter={(v, name) => [(v as number).toFixed(5), name === 'value' ? (hasNgvc ? 'Uploaded image (NGVC)' : 'Illustrative example') : 'Illustrative reference']}
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem', boxShadow: 'var(--shadow-card)' }}
                />
                <Area type="monotone" dataKey="baseline" stroke="#94a3b8" strokeWidth={1.5} fill="rgba(148,163,184,0.08)" strokeDasharray="4 2" name="baseline" />
                <Area type="monotone" dataKey="value" stroke="var(--primary)" strokeWidth={2} fill="rgba(5,150,105,0.1)" name="value" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Neural Activation Features */}
        <div className="card spectral-chart-card">
          <div className="spectral-chart-header">
            <h3 className="spectral-chart-title">
              <BarChart3 size={17} color="var(--accent)" /> Model-derived proxy features
            </h3>
            <span className="spectral-badge spectral-badge--accent">A/B proxy scores (0–100)</span>
          </div>
          <p className="spectral-chart-desc">
            {hasFeatures
              ? 'Values come from the field response’s learned-feature A/B proxy scores; they are not physical crop measurements.'
              : 'Illustrative reference values only; upload a field scan to view model-generated proxy scores.'}
          </p>

          <div className="spectral-legend">
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
              {hasFeatures ? 'Model B proxy' : 'Illustrative reference'}
            </div>
            {hasFeatures && (
              <div className="spectral-legend-item">
                <div className="spectral-legend-dot" style={{ background: 'var(--danger)' }} />
                Model A proxy
              </div>
            )}
          </div>

          <div className="spectral-chart-wrapper">
            <ResponsiveContainer width="100%" height={320} minWidth={0}>
              <BarChart data={mergedFeatures} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="subject" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={[0, 100]} />
                <Tooltip
                  formatter={(v, name) => [`${(v as number).toFixed(1)}`, name === 'modelB' ? (hasFeatures ? 'Model B proxy' : 'Illustrative reference') : 'Model A proxy']}
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem', boxShadow: 'var(--shadow-card)' }}
                />
                <Bar dataKey="modelB" fill="var(--primary)" radius={[4,4,0,0]} name="modelB" opacity={0.85} />
                {hasFeatures && (
                  <Bar dataKey="modelA" fill="var(--danger)" radius={[4,4,0,0]} name="modelA" opacity={0.8} />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Spectral Interpretation panel */}
      {hasModelData && lastScan && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card spectral-interpretation-card"
        >
          <div className="spectral-interpretation-header">
            <Sparkles size={16} color="var(--primary)" />
            <h4 className="spectral-interpretation-title">Field scan summary</h4>
            <span className="spectral-interpretation-source">Backend response · proxy features where noted</span>
          </div>
          <div className="spectral-interpretation-grid">
            {[
              { label: 'Disease Type', value: lastScan.disease_type, color: 'var(--text-primary)' },
              { label: 'Severity', value: `${lastScan.average_severity?.toFixed(1) ?? '--'}%`, color: lastScan.average_severity > 50 ? 'var(--danger)' : lastScan.average_severity > 20 ? 'var(--warning)' : 'var(--success)' },
              { label: 'Infected Area', value: `${lastScan.percentage_infected_area?.toFixed(1) ?? '--'}%`, color: 'var(--text-primary)' },
              { label: 'Confidence', value: `${lastScan.confidence?.toFixed(1) ?? '--'}%`, color: 'var(--primary)' },
            ].map(({ label, value, color }) => (
              <div key={label} className="spectral-metric-box">
                <div className="spectral-metric-box__label">{label}</div>
                <div className="spectral-metric-box__value" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
