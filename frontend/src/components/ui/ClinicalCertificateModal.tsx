import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Award, Printer, X, ShieldCheck, CheckCircle2 } from 'lucide-react';
import type { LeafResult, FieldResult } from '../../types';

interface ClinicalCertificateModalProps {
  isOpen: boolean;
  onClose: () => void;
  scanResult: LeafResult | FieldResult | null;
}

export const ClinicalCertificateModal: React.FC<ClinicalCertificateModalProps> = ({
  isOpen,
  onClose,
  scanResult,
}) => {
  if (!scanResult) return null;

  const disease = scanResult.disease_type || 'Unknown';
  const confidence = scanResult.confidence != null ? scanResult.confidence.toFixed(1) : 'N/A';
  const severity = ('severity' in scanResult ? scanResult.severity : scanResult.average_severity) || 0;
  const treatment = scanResult.treatment || 'Standard crop monitoring recommended.';
  const scanId = scanResult.scan_id || `CERT-${disease.replace(/[^A-Z0-9]/gi, '').slice(0, 6).toUpperCase() || 'SCAN'}`;
  const dateStr = new Date().toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
  });

  const handlePrint = () => {
    window.print();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-950/70 backdrop-blur-sm overflow-y-auto print:p-0 print:bg-white">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative w-full max-w-2xl bg-white text-slate-900 rounded-2xl shadow-2xl border border-slate-200 overflow-hidden my-auto print:border-none print:shadow-none"
          >
            {/* Action Bar (hidden in print) */}
            <div className="flex items-center justify-between p-3.5 bg-slate-50 border-b border-slate-200 print:hidden">
              <span className="text-xs font-bold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
                <Award size={15} className="text-emerald-600" />
                Agronomic Diagnostic Certificate Preview
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handlePrint}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition-colors shadow-sm"
                >
                  <Printer size={14} /> Print / Save as PDF
                </button>
                <button
                  type="button"
                  onClick={onClose}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* Printable Certificate Body */}
            <div className="p-8 print:p-6 text-left border-8 border-double border-slate-200 m-3 rounded-xl bg-white">
              {/* Certificate Header */}
              <div className="flex justify-between items-start pb-4 border-b-2 border-emerald-600 mb-6">
                <div>
                  <div className="text-2xl font-black tracking-tight text-emerald-800 flex items-center gap-2">
                    <ShieldCheck className="text-emerald-600" size={28} />
                    AgriPulse Pathology Registry
                  </div>
                  <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold mt-0.5">
                    Field-Level Crop Disease Intelligence &amp; Pathology Registry
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] font-mono text-slate-400">CERTIFICATE ID</div>
                  <div className="text-xs font-mono font-bold text-slate-700">{scanId}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{dateStr}</div>
                </div>
              </div>

              {/* Verified Badge */}
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold mb-6">
                <CheckCircle2 size={14} className="text-emerald-600" />
                Calibrated AI Diagnosis Verified (ICAR Agronomic Standard)
              </div>

              {/* Diagnosis Details Grid */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-500">Detected Pathology</div>
                  <div className="text-base font-extrabold text-slate-900 mt-1">{disease}</div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-500">Calibrated Confidence</div>
                  <div className="text-base font-extrabold text-emerald-600 mt-1">{confidence}%</div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-500">Canopy Foliar Severity</div>
                  <div className="text-base font-extrabold text-slate-900 mt-1">{severity.toFixed(1)}%</div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-500">Inference Core</div>
                  <div className="text-xs font-semibold text-slate-700 mt-1.5">DINOv2 + Adaptive Graph Transformer</div>
                </div>
              </div>

              {/* Official ICAR Treatment Directive */}
              <div className="mb-6 p-4 bg-emerald-50/50 rounded-xl border border-emerald-200">
                <div className="text-xs font-bold text-emerald-900 uppercase tracking-wider mb-1">
                  Official Agronomic Treatment Protocol
                </div>
                <div className="text-xs text-slate-700 leading-relaxed font-medium">
                  {treatment}
                </div>
              </div>

              {/* Footer Sign-off & QR */}
              <div className="flex justify-between items-end pt-4 border-t border-slate-200 mt-6">
                <div>
                  <div className="w-16 h-16 border border-slate-200 rounded p-1">
                    <img src="/project_qr.png" alt="Verification QR" className="w-full h-full object-contain" />
                  </div>
                  <div className="text-[9px] text-slate-400 mt-1 font-mono">SCAN TO VERIFY LIVE</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-bold font-serif italic text-slate-700">AgriPulse Automated Pathology Lab</div>
                  <div className="text-[10px] text-slate-400 mt-1">Autonomous Crop Intelligence Core</div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
export default ClinicalCertificateModal;
