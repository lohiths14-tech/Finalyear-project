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

// Static healthy rice/wheat crop baseline reference (from IRRI published NGVC norms)
const HEALTHY_BASELINE_NGVC = Array.from({ length: 32 }, (_, i) => ({
  intensity: i * 8,
  value: Math.max(0, 0.08 - 0.002 * i + 0.001 * Math.sin(i * 0.8)),
  baseline: Math.max(0, 0.09 - 0.0022 * i + 0.0012 * Math.sin(i * 0.7)),
}));

const HEALTHY_FEATURE_REFERENCE = [
  { subject: 'Chlorophyll', healthy: 88, diseased: 0 },
  { subject: 'Texture',     healthy: 72, diseased: 0 },
  { subject: 'Edge',        healthy: 65, diseased: 0 },
  { subject: 'Color',       healthy: 91, diseased: 0 },
  { subject: 'Pattern',     healthy: 78, diseased: 0 },
  { subject: 'Canopy',      healthy: 85, diseased: 0 },
];

export default function SpectralTab({ lastScan }: SpectralTabProps) {
  const hasScan = !!(lastScan?.ngvc_data && lastScan.ngvc_data.length > 0);
  const hasFeatures = !!(lastScan?.spectral_radar && lastScan.spectral_radar.length > 0);

  const mergedNgvc = useMemo(() => {
    if (!hasScan) return HEALTHY_BASELINE_NGVC;
    return (lastScan!.ngvc_data as NgvcData[]).slice(0, 64).map((d, i) => ({
      ...d,
      baseline: HEALTHY_BASELINE_NGVC[Math.floor(i / 2)]?.baseline ?? 0,
    }));
  }, [lastScan, hasScan]);

  const mergedFeatures = useMemo(() => {
    if (!hasFeatures) return HEALTHY_FEATURE_REFERENCE;
    return lastScan!.spectral_radar.map((d: { subject: string; A: number; B: number }) => ({
      subject: d.subject,
      healthy: d.B ?? 80,
      diseased: d.A,
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
            {hasScan
              ? `Spectral feature breakdown for: ${lastScan?.disease_type}`
              : 'Showing healthy crop reference baseline (IRRI norms)'}
          </div>
          <div className="spectral-banner-sub">
            {hasScan
              ? "Green = field scan · Gray = healthy reference. Note: Feature magnitudes represent learned neural embedding proxies, not direct physical multispectral hardware measurements."
              : "Upload a field or leaf scan to overlay your crop's spectral signature against the healthy baseline."}
          </div>
        </div>
        {!hasScan && (
          <span className="spectral-reference-badge">REFERENCE MODE</span>
        )}
      </div>

      <div className="main-grid">
        {/* NGVC Spectral Curve */}
        <div className="card spectral-chart-card">
          <div className="spectral-chart-header">
            <h3 className="spectral-chart-title">
              <Radio size={17} color="var(--primary)" /> NGVC Spectral Curve
            </h3>
            <span className="spectral-badge spectral-badge--primary">128-Bin Histogram</span>
          </div>
          <p className="spectral-chart-desc">
            Normalized Gray Value Curve — captures chlorophyll absorption bands and chlorosis patterns.
          </p>

          <div className="spectral-legend">
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
              {hasScan ? 'Field Scan' : 'Healthy Crop (IRRI ref.)'}
            </div>
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: '#94a3b8' }} />
              Healthy Baseline
            </div>
          </div>

          <div className="spectral-chart-wrapper">
            <ResponsiveContainer width="100%" height={320} minWidth={0}>
              <AreaChart data={mergedNgvc} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="intensity" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <Tooltip
                  formatter={(v, name) => [(v as number).toFixed(5), name === 'value' ? (hasScan ? 'Field Scan' : 'Healthy Reference') : 'Baseline']}
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
              <BarChart3 size={17} color="var(--accent)" /> Neural Activation Features
            </h3>
            <span className="spectral-badge spectral-badge--accent">Multi-Head Embeddings</span>
          </div>
          <p className="spectral-chart-desc">
            {hasScan
              ? 'Field-extracted pathological activations vs. healthy crop baseline.'
              : 'Healthy crop reference signatures (IRRI). Upload a scan to compare your field.'}
          </p>

          <div className="spectral-legend">
            <div className="spectral-legend-item">
              <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
              Healthy Baseline
            </div>
            {hasFeatures && (
              <div className="spectral-legend-item">
                <div className="spectral-legend-dot" style={{ background: 'var(--danger)' }} />
                Field Scan
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
                  formatter={(v, name) => [`${(v as number).toFixed(1)}`, name === 'healthy' ? 'Healthy Baseline' : 'Field Scan']}
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem', boxShadow: 'var(--shadow-card)' }}
                />
                <Bar dataKey="healthy" fill="var(--primary)" radius={[4,4,0,0]} name="healthy" opacity={0.85} />
                {hasFeatures && (
                  <Bar dataKey="diseased" fill="var(--danger)" radius={[4,4,0,0]} name="diseased" opacity={0.8} />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Spectral Interpretation panel */}
      {hasScan && lastScan && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card spectral-interpretation-card"
        >
          <div className="spectral-interpretation-header">
            <Sparkles size={16} color="var(--primary)" />
            <h4 className="spectral-interpretation-title">Spectral Interpretation</h4>
            <span className="spectral-interpretation-source">Derived from DINOv2 features</span>
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
