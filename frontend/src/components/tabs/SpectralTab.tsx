import { motion, MotionConfig } from 'framer-motion';
import { BarChart3, Radio, Sparkles, Info } from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid
} from 'recharts';
import EmptyState from '../ui/EmptyState';
import type { FieldResult } from '../../types';

interface SpectralTabProps {
  lastScan: FieldResult | null;
}

export default function SpectralTab({ lastScan }: SpectralTabProps) {
  const ngvcData = lastScan?.ngvc_data ?? [];
  const featureData = lastScan?.spectral_radar ?? [];
  const hasNgvc = ngvcData.length > 0;
  const hasFeatures = featureData.length > 0;
  const hasModelData = hasNgvc || hasFeatures;

  return (
    <MotionConfig reducedMotion="user">
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
              : 'No scan-derived spectral data available'}
          </div>
          <div className="spectral-banner-sub">
            {hasModelData
              ? 'NGVC values are derived from the uploaded image; neural feature values are model-derived proxies, not direct physical spectral measurements. Gray curves and unscanned chart values are illustrative references only.'
              : 'Upload and analyze an image to view image-derived values and model proxies. No reference curves or placeholder scores are generated.'}
          </div>
        </div>
        {!hasModelData && (
          <span className="spectral-reference-badge">NO SCAN DATA</span>
        )}
      </div>

      <div className="main-grid">
        {/* NGVC Spectral Curve */}
        <div className="card spectral-chart-card">
          <div className="spectral-chart-header">
            <h3 className="spectral-chart-title">
              <Radio size={17} color="var(--primary)" /> Image-derived NGVC profile
            </h3>
            <span className="spectral-badge spectral-badge--primary">{hasNgvc ? 'Scan data' : 'Unavailable'}</span>
          </div>
          <p className="spectral-chart-desc">
            {hasNgvc
              ? `The ${ngvcData.length} NGVC points are derived from this scan image, not from a multispectral sensor or direct chlorophyll measurement. Intensity has no verified physical unit.`
              : 'No image-derived NGVC points were returned for this scan.'}
          </p>

          {hasNgvc ? (
            <>
              <div className="spectral-legend">
                <div className="spectral-legend-item">
                  <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
                  Uploaded image (NGVC)
                </div>
              </div>
              <div className="spectral-chart-wrapper" role="img" aria-label={`Image-derived NGVC profile with ${ngvcData.length} data points`}>
                <ResponsiveContainer width="100%" height={320} minWidth={0}>
                  <AreaChart data={ngvcData} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="intensity" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                    <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(value) => [Number(value).toFixed(5), 'Image-derived NGVC']}
                      contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem', boxShadow: 'var(--shadow-card)' }}
                    />
                    <Area type="monotone" dataKey="value" stroke="var(--primary)" strokeWidth={2} fill="rgba(5,150,105,0.1)" name="Image-derived NGVC" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <EmptyState icon={Radio} message="NGVC values are unavailable because this scan returned no image-derived points." />
          )}
        </div>

        {/* Neural Activation Features */}
        <div className="card spectral-chart-card">
          <div className="spectral-chart-header">
            <h3 className="spectral-chart-title">
              <BarChart3 size={17} color="var(--accent)" /> Model-derived proxy features
            </h3>
            <span className="spectral-badge spectral-badge--accent">A/B proxy values</span>
          </div>
          <p className="spectral-chart-desc">
            {hasFeatures
              ? `${featureData.length} feature rows are returned with A/B proxy values; these are not physical crop measurements and have no physical units.`
              : 'No model-derived proxy feature rows were returned for this scan.'}
          </p>

          {hasFeatures ? (
            <>
              <div className="spectral-legend">
                <div className="spectral-legend-item">
                  <div className="spectral-legend-dot" style={{ background: 'var(--primary)' }} />
                  Model B proxy
                </div>
                <div className="spectral-legend-item">
                  <div className="spectral-legend-dot" style={{ background: 'var(--danger)' }} />
                  Model A proxy
                </div>
              </div>
              <div className="spectral-chart-wrapper" role="img" aria-label={`Model-derived proxy features with ${featureData.length} feature rows`}>
                <ResponsiveContainer width="100%" height={320} minWidth={0}>
                  <BarChart data={featureData.map(({ subject, A, B }) => ({ subject, modelA: A, modelB: B }))} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="subject" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                    <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(value, name) => [Number(value).toFixed(1), name === 'modelB' ? 'Model B proxy' : 'Model A proxy']}
                      contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem', boxShadow: 'var(--shadow-card)' }}
                    />
                    <Bar dataKey="modelB" fill="var(--primary)" radius={[4,4,0,0]} name="Model B proxy" opacity={0.85} />
                    <Bar dataKey="modelA" fill="var(--danger)" radius={[4,4,0,0]} name="Model A proxy" opacity={0.8} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <EmptyState icon={BarChart3} message="Model-derived proxy features are unavailable because this scan returned no feature rows." />
          )}
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
              { label: 'Infected Area', value: lastScan.percentage_infected_area !== undefined ? `${lastScan.percentage_infected_area.toFixed(1)}%` : 'Unavailable', color: 'var(--text-primary)' },
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
    </MotionConfig>
  );
}
