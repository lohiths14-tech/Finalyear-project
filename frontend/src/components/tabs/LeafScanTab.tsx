import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence, MotionConfig } from 'framer-motion';
import { FlaskConical, Upload, CheckCircle, AlertCircle, AlertTriangle, ChevronDown, Sparkles, Printer, Camera, Award, Cpu } from 'lucide-react';
import SkeletonCard from '../ui/SkeletonCard';
import apiClient from '../../services/api';
import { optimizeImageForUpload } from '../../utils/compress';
import LiveCameraScanner from '../ui/LiveCameraScanner';
import EconomicImpactWidget from '../ui/EconomicImpactWidget';
import ProgressionSimulator from '../ui/ProgressionSimulator';
import ClinicalCertificateModal from '../ui/ClinicalCertificateModal';
import type { LeafResult } from '../../types';

const severityColor = (s: number) => s > 70 ? 'var(--danger)' : s > 40 ? 'var(--warning)' : 'var(--success)';
const confColor = (c: number) => c >= 70 ? 'var(--success)' : c >= 40 ? 'var(--warning)' : 'var(--danger)';

function StatusIcon({ status }: { status: string }) {
  if (status === 'Healthy') return <CheckCircle size={26} color="var(--success)" />;
  if (status === 'Low confidence prediction') return <AlertTriangle size={26} color="var(--warning)" />;
  return <AlertCircle size={26} color="var(--danger)" />;
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ height: 6, borderRadius: 99, background: 'var(--bg-subtle)', overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          style={{ height: '100%', borderRadius: 99, background: confColor(value) }}
        />
      </div>
    </div>
  );
}

interface LeafScanTabProps {
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
  backendOnline: boolean | null;
  onScanComplete?: (results: LeafResult) => void;
  lastLeafScan?: LeafResult | null;
}

export default function LeafScanTab({ addToast, backendOnline, onScanComplete, lastLeafScan }: LeafScanTabProps) {
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<LeafResult | null>(lastLeafScan ?? null);
  const [preview, setPreview] = useState<string | null>(null);
  const [showAlt, setShowAlt] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isCertOpen, setIsCertOpen] = useState(false);
  const scanningRef = useRef(false);
  const previewUrlRef = useRef<string | null>(null);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  // AI Agronomist Consultation states
  const [consulting, setConsulting] = useState(false);
  const [customQuestion, setCustomQuestion] = useState('');
  const [agronomistAnswer, setAgronomistAnswer] = useState<string | null>(null);
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);

  const handleConsult = useCallback(async (questionText: string) => {
    if (!result || !questionText.trim()) return;
    setConsulting(true);
    setAgronomistAnswer(null);
    setActiveQuestion(questionText);
    try {
      const { data } = await apiClient.post<{ answer: string }>('/analyze/consult', {
        disease_type: result.disease_type,
        severity: result.severity,
        question: questionText,
      });
      setAgronomistAnswer(data.answer);
      setCustomQuestion('');
    } catch (err) {
      addToast(`Consultation failed: ${(err as Error).message}`, 'error');
    } finally {
      setConsulting(false);
    }
  }, [result, addToast]);

  const processFile = useCallback(async (file: File, alreadyLocked = false) => {
    if (!file.type.startsWith('image/')) { addToast('Please upload a valid image (JPG, PNG, WEBP).', 'error'); return; }
    if (file.size > 20 * 1024 * 1024) { addToast('File too large. Please use an image smaller than 20 MB.', 'error'); return; }
    if (scanningRef.current && !alreadyLocked) return;
    if (!alreadyLocked) scanningRef.current = true;

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const previewUrl = URL.createObjectURL(file);
    previewUrlRef.current = previewUrl;
    setPreview(previewUrl);
    setScanning(true);
    setResult(null);
    setShowAlt(false);
    setAgronomistAnswer(null);
    setActiveQuestion(null);

    try {
      addToast(`Preparing & scanning "${file.name}"…`, 'info', 3000);
      const optimizedFile = await optimizeImageForUpload(file, 1024);
      const fd = new FormData();
      fd.append('file', optimizedFile);

      const { data } = await apiClient.post<LeafResult>('/analyze/leaf', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      });
      setResult(data);
      if (onScanComplete) onScanComplete(data);

      const toastType = data.status === 'Healthy' ? 'success' : 'warning';
      addToast(`Scan complete — ${data.disease_type}`, toastType, 6000);
    } catch (err) {
      addToast(`Scan failed: ${(err as Error).message}`, 'error');
    } finally {
      scanningRef.current = false;
      setScanning(false);
    }
  }, [addToast, onScanComplete]);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFile(file);
    e.target.value = '';
  }, [processFile]);

  const handleLoadDemo = useCallback(async () => {
    if (scanningRef.current) return;
    scanningRef.current = true;
    setScanning(true);
    try {
      const res = await fetch('/samples/sample_blast.jpg');
      if (!res.ok) throw new Error('Sample image request failed');
      const blob = await res.blob();
      const file = new File([blob], 'rice_blast_leaf.jpg', { type: 'image/jpeg' });
      await processFile(file, true);
    } catch {
      scanningRef.current = false;
      setScanning(false);
      addToast('Could not load the demo leaf image. Please try uploading an image instead.', 'error');
    }
  }, [addToast, processFile]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) await processFile(file);
  }, [processFile]);

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);

  const handlePrint = () => {
    window.print();
  };

  const isHealthy = !result?.withheld && result?.status === 'Healthy';
  const isLowConf = result?.withheld || result?.status === 'Low confidence prediction';

  return (
    <MotionConfig reducedMotion="user">
    <div className="tab-viewport">
      <div className="main-grid">
        <div className="card leaf-upload-card">
          <div className="leaf-upload-card__header">
            <h3 className="leaf-upload-card__title">
              <FlaskConical size={20} color="var(--primary)" /> Leaf Micro-Diagnostic
            </h3>
            <div className="leaf-upload-card__actions">
              <button
                type="button"
                className="btn-icon btn-icon--primary"
                onClick={handleLoadDemo}
                disabled={scanning}
                title="Load verified leaf pathology sample"
              >
                <Sparkles size={14} /> Demo Leaf
              </button>
              <button
                type="button"
                className="btn-icon"
                onClick={() => setIsCameraOpen(true)}
                disabled={scanning}
                title="Open live camera scanner"
              >
                <Camera size={14} /> Live Camera
              </button>
              {result && (
                <>
                  <button className="btn-icon btn-icon--primary" onClick={() => setIsCertOpen(true)}>
                    <Award size={14} /> Certificate
                  </button>
                  <button className="btn-icon" onClick={handlePrint}>
                    <Printer size={14} /> Print
                  </button>
                </>
              )}
            </div>
          </div>
          <p className="leaf-upload-card__desc">
            Upload a clear leaf photograph to view the scan output and any diagnostics returned by the analysis service.
          </p>

          <label
            htmlFor="leaf-image-upload"
            className={`upload-zone ${dragOver ? 'drag-over' : ''} ${!backendOnline ? 'btn-disabled' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload size={36} color="var(--primary)" />
            <span style={{ marginTop: 10, fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
              {scanning ? 'Analysis in progress…' : 'Click or drop leaf photograph'}
            </span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>JPEG · PNG · WEBP · max 20 MB</span>
            <input
              id="leaf-image-upload"
              className="sr-only"
              type="file"
              accept="image/*"
              disabled={scanning || !backendOnline}
              onChange={handleUpload}
            />
          </label>

          {preview && (
            <div className="leaf-preview-wrapper">
              <img src={preview} alt="Leaf preview" className="leaf-preview-img" />
            </div>
          )}

          <div className="leaf-pro-tip">
            <strong>Capture guidance:</strong> Keep the leaf in focus with even lighting. Image quality can affect model output; results may be uncertain.
          </div>
        </div>

        <div>
          {scanning && <div role="status" aria-live="polite"><span className="sr-only">Leaf image analysis in progress</span><SkeletonCard height="440px" /></div>}

          {!scanning && result && (
            <div
              className="card"
              style={{ display: 'flex', flexDirection: 'column', gap: 0, background: 'var(--bg-card)', padding: 24 }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
                <StatusIcon status={result.withheld ? 'Low confidence prediction' : result.status} />
                <div>
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', margin: 0, fontWeight: 600 }}>
                    Scan output
                  </p>
                  <h3 style={{
                    fontSize: '1.25rem',
                    fontWeight: 700,
                    margin: 0,
                    color: isHealthy ? 'var(--success)' : isLowConf ? 'var(--warning)' : 'var(--danger)'
                  }}>
                    {result.withheld ? 'Classification withheld' : result.disease_type}
                  </h3>
                </div>
              </div>

              {result.withheld && (
                <div className="ood-warning-box" role="status">
                  <p style={{ color: 'var(--warning)', fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, margin: 0 }}>
                    <AlertTriangle size={16} /> Classification withheld by the analysis service
                  </p>
                </div>
              )}

              {result.ood?.is_ood && (
                <div className="ood-warning-box" role="status">
                  <p style={{ color: 'var(--danger)', fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, margin: 0 }}>
                    <AlertTriangle size={16} /> Out-of-distribution flag returned
                  </p>
                  {result.ood.reason && <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginTop: 4, marginBottom: 0 }}>{result.ood.reason}</p>}
                </div>
              )}

              {!result.withheld && <>
                {[
                  ['Pathology class', result.disease_type],
                  ['Image severity (reported)', `${result.severity.toFixed(1)}%`],
                ].map(([label, value]) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.86rem' }}>{label}</span>
                    <span style={{ fontWeight: 700, color: label.startsWith('Image severity') ? severityColor(result.severity) : 'var(--text-primary)' }}>{value}</span>
                  </div>
                ))}

                <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.86rem' }}>Confidence (backend-reported)</span>
                    <span style={{ fontWeight: 700, color: confColor(result.confidence) }}>{result.confidence.toFixed(1)}%</span>
                  </div>
                  <ConfidenceBar value={result.confidence} />
                </div>

              <div style={{ marginTop: 14 }}>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: 6 }}>
                  Guidance returned by the service
                </p>
                <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text-primary)' }}>
                  {result.treatment}
                </div>
                {result.disclaimer && <p className="overview-validation-desc">{result.disclaimer}</p>}
                <p className="overview-validation-desc">Guidance is shown as returned; verify treatment decisions with a qualified local agronomist.</p>
              </div>
            </>}

              {!result.withheld && <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                  <Sparkles size={16} color="var(--primary)" />
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    Scan follow-up questions
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                  {[
                    ['Interpret this result', 'Can you explain this scan result and its limitations?'],
                    ['Non-chemical options', 'What non-chemical practices may be relevant to discuss with a local agronomist?'],
                    ['Monitoring', 'What should I record in future observations for comparison?'],
                    ['Image quality', 'What image limitations should I check before relying on this result?'],
                  ].map(([label, question]) => (
                    <button
                      key={label}
                      type="button"
                      className="btn-icon"
                      onClick={() => handleConsult(question)}
                      disabled={consulting || result.withheld}
                      style={{ fontSize: '0.75rem', padding: '5px 10px' }}
                    >
                      <span>{label}</span>
                    </button>
                  ))}
                </div>

                {/* Response Block */}
                <AnimatePresence>
                  {(consulting || agronomistAnswer) && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="advisory-box"
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                        <div>
                          <p style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--primary)', margin: 0 }}>Service response · verify before action</p>
                          {activeQuestion && <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: 0 }}>Topic: "{activeQuestion}"</p>}
                        </div>
                      </div>

                      {consulting ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '6px 0' }}>
                          <div className="skeleton" style={{ height: 10, width: '90%', borderRadius: 4 }} />
                          <div className="skeleton" style={{ height: 10, width: '75%', borderRadius: 4 }} />
                        </div>
                      ) : (
                        <div style={{
                          fontSize: '0.82rem',
                          lineHeight: 1.6,
                          color: 'var(--text-primary)',
                          whiteSpace: 'pre-line',
                        }}>
                          {agronomistAnswer}
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Custom Input */}
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    type="text"
                    value={customQuestion}
                    onChange={(e) => setCustomQuestion(e.target.value)}
                    aria-label="Ask a question about the scan"
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229) handleConsult(customQuestion); }}
                    disabled={consulting || result.withheld}
                    placeholder="Ask about this result or its limitations..."
                    style={{
                      flex: 1,
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      padding: '9px 12px',
                      color: 'var(--text-primary)',
                      fontSize: '0.82rem',
                      outline: 'none',
                    }}
                  />
                  <button
                    className="btn-primary"
                    onClick={() => handleConsult(customQuestion)}
                    type="button"
                    disabled={consulting || result.withheld || !customQuestion.trim()}
                    style={{ padding: '0 16px', fontSize: '0.82rem' }}
                  >
                    Consult
                  </button>
                </div>
              </div>}

              {!result.withheld && result.alternatives && result.alternatives.length > 1 && (
                <div style={{ marginTop: 16 }}>
                    <button
                    type="button"
                    aria-expanded={showAlt}
                    onClick={() => setShowAlt((v) => !v)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--primary)', fontSize: '0.82rem', fontWeight: 600, padding: 0,
                    }}
                  >
                    <ChevronDown size={14} style={{ transform: showAlt ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                    {showAlt ? 'Hide' : 'Show'} alternative diagnoses
                  </button>

                  <AnimatePresence>
                    {showAlt && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        style={{ overflow: 'hidden', marginTop: 10 }}
                      >
                        {result.alternatives.map((alt, i) => (
                          <div
                            key={i}
                            style={{
                              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                              padding: '8px 12px',
                              background: i === 0 ? 'var(--primary-glow)' : 'var(--bg-subtle)',
                              border: i === 0 ? '1px solid rgba(16,185,129,0.25)' : '1px solid transparent',
                              borderRadius: 8, marginBottom: 6, fontSize: '0.8rem',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{
                                width: 18, height: 18, borderRadius: '50%',
                                background: i === 0 ? 'var(--primary)' : 'var(--border)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.65rem', fontWeight: 700, color: 'white', flexShrink: 0,
                              }}>{i + 1}</span>
                              <span style={{ color: 'var(--text-primary)', fontWeight: i === 0 ? 600 : 400 }}>{alt.disease}</span>
                            </div>
                            <span style={{ fontWeight: 700, color: confColor(alt.confidence) }}>
                              {alt.confidence.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
              {/* Illustrative scenario tools disclose their assumptions and are not forecasts. */}
              <div style={{ marginTop: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
                {!result.withheld && <>
                  <EconomicImpactWidget
                    diseaseType={result.disease_type}
                    severity={result.severity}
                    confidence={result.confidence}
                  />
                  <ProgressionSimulator
                    initialSeverity={result.severity}
                    diseaseType={result.disease_type}
                  />
                </>}

                {(result.quality || result.reliability || result.ood || result.recommendation || result.withheld) && (
                  <div className="scientific-drawer">
                    <button
                      type="button"
                      className="scientific-drawer-header"
                      aria-expanded={showDiagnostics}
                      onClick={() => setShowDiagnostics((open) => !open)}
                    >
                      <span className="scientific-drawer-header__label">
                        <Cpu size={14} color="var(--primary)" />
                        <strong>Optional scan diagnostics</strong>
                      </span>
                      <span className="scientific-drawer-header__action">
                        <span>{showDiagnostics ? 'Hide details' : 'Show details'}</span>
                        <ChevronDown size={14} aria-hidden="true" />
                      </span>
                    </button>

                    {showDiagnostics && (
                      <div className="scientific-drawer-content">
                        <div className="leaf-diagnostic-grid">
                          {result.quality?.quality && <div>
                            <span>Image quality</span>
                            <strong>{result.quality.quality.quality_label} · score {result.quality.quality.iqi_score}</strong>
                          </div>}
                          {result.quality?.confidence && <div>
                            <span>Quality confidence</span>
                            <strong>{result.quality.confidence.level} · {result.quality.confidence.confidence.toFixed(1)}%</strong>
                            <p>{result.quality.confidence.message}</p>
                          </div>}
                          {(result.reliability || result.quality?.reliability) && <div>
                            <span>Result reliability</span>
                            <strong>{result.reliability?.score ?? result.quality?.reliability.score} · {result.quality?.reliability.label || (result.reliability?.is_reliable ? 'Marked reliable by service' : 'Marked unreliable by service')}</strong>
                            {result.reliability?.rejection_reason && <p>{result.reliability.rejection_reason}</p>}
                          </div>}
                          {result.ood && <div>
                            <span>Out-of-distribution result</span>
                            <strong>{result.ood.is_ood ? 'Flag returned' : 'No flag returned'} · score {result.ood.ood_score}</strong>
                            {result.ood.reason && <p>{result.ood.reason}</p>}
                          </div>}
                          {result.quality?.recommendation && <div>
                            <span>Quality recommendation</span>
                            <strong>{result.quality.recommendation}</strong>
                          </div>}
                          {result.recommendation && <div>
                            <span>Additional recommendation</span>
                            <strong>{result.recommendation}</strong>
                          </div>}
                          {result.quality?.quality.issues?.length ? <div>
                            <span>Image issues returned</span>
                            <ul>{result.quality.quality.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
                          </div> : null}
                          {result.quality?.quality.suggestions?.length ? <div>
                            <span>Capture suggestions returned</span>
                            <ul>{result.quality.quality.suggestions.map((suggestion) => <li key={suggestion}>{suggestion}</li>)}</ul>
                          </div> : null}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {!scanning && !result && (
            <div className="card empty-state" style={{ background: 'var(--bg-card)', minHeight: 440, padding: 36, textAlign: 'center' }}>
              <div className="empty-state-icon-wrapper" style={{ margin: '0 auto 16px' }}>
                <FlaskConical size={36} color="var(--primary)" />
              </div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, margin: '0 0 8px 0' }}>Awaiting Leaf Scan</h3>
              <p style={{ maxWidth: 460, margin: '0 auto 20px', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.5 }}>
                Upload a close-up leaf photograph or capture with your live camera — or test immediately with our verified benchmark sample.
              </p>
              <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => setIsCameraOpen(true)}
                  style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  <Camera size={16} /> Open Camera
                </button>
                <button
                  type="button"
                  className="btn-icon"
                  onClick={handleLoadDemo}
                  style={{ padding: '8px 18px', borderColor: 'var(--primary)', color: 'var(--primary)', background: 'var(--primary-glow)', display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}
                >
                  <Sparkles size={16} /> Load Verified Leaf Demo
                </button>
              </div>
              <p className="overview-validation-desc" style={{ marginTop: 20 }}>
                Results and optional diagnostics appear only when returned by the analysis service.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Live Camera Scanner Modal */}
      <LiveCameraScanner
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={processFile}
      />

      {/* Clinical Pathology Certificate Modal */}
      <ClinicalCertificateModal
        isOpen={isCertOpen}
        onClose={() => setIsCertOpen(false)}
        scanResult={result}
      />
    </div>
    </MotionConfig>
  );
}
