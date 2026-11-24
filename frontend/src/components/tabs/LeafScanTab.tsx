import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) { addToast('Please upload a valid image (JPG, PNG, WEBP).', 'error'); return; }
    if (file.size > 20 * 1024 * 1024) { addToast('File too large. Please use an image smaller than 20 MB.', 'error'); return; }

    setPreview(URL.createObjectURL(file));
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
    try {
      setScanning(true);
      const res = await fetch('/samples/sample_blast.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'rice_blast_leaf.jpg', { type: 'image/jpeg' });
      await processFile(file);
    } catch (e) {
      console.error('Failed to load demo leaf:', e);
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

  const handlePrint = () => {
    window.print();
  };

  const isHealthy = result?.status === 'Healthy';
  const isLowConf = result?.status === 'Low confidence prediction';

  return (
    <div className="tab-viewport">
      <div className="main-grid">
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 18, background: 'var(--bg-card)', padding: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '1.15rem', fontWeight: 700, margin: 0 }}>
              <FlaskConical size={20} color="var(--primary)" /> Leaf Micro-Diagnostic
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                type="button"
                className="btn-icon"
                onClick={handleLoadDemo}
                disabled={scanning}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderColor: 'var(--primary)', color: 'var(--primary)', background: 'var(--primary-glow)' }}
                title="Load verified leaf pathology sample"
              >
                <Sparkles size={14} /> Demo Leaf
              </button>
              <button
                type="button"
                className="btn-icon"
                onClick={() => setIsCameraOpen(true)}
                disabled={scanning}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px' }}
                title="Open live camera scanner"
              >
                <Camera size={14} /> Live Camera
              </button>
              {result && (
                <>
                  <button 
                    className="btn-icon" 
                    onClick={() => setIsCertOpen(true)} 
                    style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderColor: 'var(--primary)', color: 'var(--primary)' }}
                  >
                    <Award size={14} /> Certificate
                  </button>
                  <button 
                    className="btn-icon" 
                    onClick={handlePrint} 
                    style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px' }}
                  >
                    <Printer size={14} /> Print
                  </button>
                </>
              )}
            </div>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.86rem', margin: 0 }}>
            Upload a clear, close-up leaf photograph for instant spectral pathogen profiling and ICAR agronomic guidance.
          </p>

          <label
            className={`upload-zone ${dragOver ? 'drag-over' : ''} ${!backendOnline ? 'btn-disabled' : ''}`}
            aria-label="Upload leaf image"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload size={36} color="var(--primary)" />
            <span style={{ marginTop: 10, fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
              {scanning ? 'Analyzing with Neural Engine…' : 'Click or drop leaf photograph'}
            </span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>JPEG · PNG · WEBP · max 20 MB</span>
            <input
              type="file"
              hidden
              accept="image/*"
              disabled={scanning || !backendOnline}
              onChange={handleUpload}
              aria-hidden="true"
            />
          </label>

          {preview && (
            <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)' }}>
              <img src={preview} alt="Leaf preview" style={{ width: '100%', maxHeight: 260, objectFit: 'cover' }} />
            </div>
          )}

          <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            💡 <strong>Field Pro-Tip:</strong> Photograph the boundary between healthy and infected leaf tissue under diffuse morning light for maximum diagnostic precision.
          </div>
        </div>

        <div>
          {scanning && <SkeletonCard height="440px" />}

          {!scanning && result && (
            <div
              className="card"
              style={{ display: 'flex', flexDirection: 'column', gap: 0, background: 'var(--bg-card)', padding: 24 }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
                <StatusIcon status={result.status} />
                <div>
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', margin: 0, fontWeight: 600 }}>
                    Diagnosis Result
                  </p>
                  <h3 style={{
                    fontSize: '1.25rem',
                    fontWeight: 700,
                    margin: 0,
                    color: isHealthy ? 'var(--success)' : isLowConf ? 'var(--warning)' : 'var(--danger)'
                  }}>
                    {result.disease_type}
                  </h3>
                </div>
              </div>

              {result.ood?.is_ood && (
                <div className="ood-warning-box">
                  <p style={{ color: 'var(--danger)', fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, margin: 0 }}>
                    <AlertTriangle size={16} /> Out of Distribution Detected
                  </p>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginTop: 4, margin: 0 }}>
                    {result.ood.reason || "This image does not match recognizable crop leaf patterns."}
                  </p>
                </div>
              )}

              {[
                ['Pathology Class', result.disease_type],
                ['Severity Index', `${result.severity.toFixed(1)}%`],
              ].map(([label, value]) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.86rem' }}>{label}</span>
                  <span style={{ fontWeight: 700, color: label === 'Severity Index' ? severityColor(result.severity) : 'var(--text-primary)' }}>{value}</span>
                </div>
              ))}

              <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.86rem' }}>Calibrated Confidence</span>
                  <span style={{ fontWeight: 700, color: confColor(result.confidence) }}>
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
                <ConfidenceBar value={result.confidence} />
              </div>

              <div style={{ marginTop: 14 }}>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: 6 }}>
                  Recommended Protocol
                </p>
                <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border)', borderRadius: 8, padding: 12, fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text-primary)' }}>
                  {result.treatment}
                </div>
              </div>

              {/* ── AI AGRONOMIST CONSULTATION ── */}
              <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                  <Sparkles size={16} color="var(--primary)" />
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    AI Agronomist Q&A (ICAR / IRRI Extension)
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                  {[
                    ['🧪', 'Chemical dosage', 'What fungicide should I spray and what is the exact dosage?'],
                    ['🌿', 'Organic / Bio', 'Can I use organic bio-control agents for this?'],
                    ['📊', 'Yield loss risk', 'What is the economic threshold and yield loss danger?'],
                    ['🌾', 'Cultural practices', 'How should I adjust irrigation and fertilizers?'],
                  ].map(([emoji, label, question]) => (
                    <button
                      key={label}
                      className="btn-icon"
                      onClick={() => handleConsult(question)}
                      disabled={consulting}
                      style={{ fontSize: '0.75rem', padding: '5px 10px' }}
                    >
                      <span>{emoji}</span>
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
                        <span style={{ fontSize: '1.1rem' }}>🧑‍🌾</span>
                        <div>
                          <p style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--primary)', margin: 0 }}>Agronomic Advisory Panel</p>
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
                    onKeyDown={(e) => { if (e.key === 'Enter') handleConsult(customQuestion); }}
                    disabled={consulting}
                    placeholder="Ask an agronomic question (e.g. spray timing, rain safety)..."
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
                    disabled={consulting || !customQuestion.trim()}
                    style={{ padding: '0 16px', fontSize: '0.82rem' }}
                  >
                    Consult
                  </button>
                </div>
              </div>

              {result.alternatives && result.alternatives.length > 1 && (
                <div style={{ marginTop: 16 }}>
                  <button
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
              {/* Economic ROI Impact & 14-Day Progression Simulator */}
              <div style={{ marginTop: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
                <EconomicImpactWidget
                  diseaseType={result.disease_type}
                  severity={result.severity}
                  confidence={result.confidence}
                />
                <ProgressionSimulator
                  initialSeverity={result.severity}
                  diseaseType={result.disease_type}
                />

                {/* ── COLLAPSIBLE SCIENTIFIC DIAGNOSTICS DRAWER ── */}
                <div className="scientific-drawer">
                  <div 
                    className="scientific-drawer-header"
                    onClick={() => setShowDiagnostics(!showDiagnostics)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Cpu size={14} color="var(--primary)" />
                      <strong>Scientific Diagnostics &amp; Calibration</strong>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        (OOD Energy Score · Platt Scaling · INT8 Parity)
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>{showDiagnostics ? 'Collapse' : 'Expand Details'}</span>
                      <ChevronDown size={14} style={{ transform: showDiagnostics ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                    </div>
                  </div>

                  {showDiagnostics && (
                    <div className="scientific-drawer-content">
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Calibrated ECE Score:</span><br/>
                          <strong style={{ color: 'var(--primary)' }}>&lt; 0.038 (Temperature Scaled)</strong>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>OOD Energy Score:</span><br/>
                          <strong style={{ color: result.ood?.is_ood ? 'var(--danger)' : 'var(--success)' }}>
                            {result.ood?.is_ood ? 'Rejected (Out of Domain)' : 'In-Distribution (Crop Foliage)'}
                          </strong>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Model Architecture:</span><br/>
                          <strong style={{ color: 'var(--text-primary)' }}>MobileNetV2 Vision Student (INT8)</strong>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Runtime Footprint:</span><br/>
                          <strong style={{ color: 'var(--text-primary)' }}>2.36 MB ONNX (6.65 ms p50)</strong>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
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
              <div style={{ marginTop: 24, display: 'flex', gap: 16, justifyContent: 'center', fontSize: '0.78rem', color: 'var(--text-muted)', flexWrap: 'wrap' }}>
                <span>✓ Calibrated Confidence</span>
                <span>✓ Out-Of-Distribution Rejection</span>
                <span>✓ ICAR Treatment Guidance</span>
              </div>
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
  );
}
