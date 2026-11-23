import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Smartphone, X, ExternalLink, Copy, Check } from 'lucide-react';

interface QrCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const QrCodeModal: React.FC<QrCodeModalProps> = ({ isOpen, onClose }) => {
  const [copied, setCopied] = React.useState(false);
  const liveUrl = 'https://finalyear-project-lohiths-projects-b447afea.vercel.app';

  const handleCopy = () => {
    navigator.clipboard.writeText(liveUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm"
          />

          {/* Modal Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 12 }}
            transition={{ type: 'spring', damping: 25, stiffness: 350 }}
            className="relative z-10 w-full max-w-sm rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-2xl text-center"
          >
            {/* Close button */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-1.5 rounded-full text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
              <X size={18} />
            </button>

            {/* Header Icon */}
            <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mx-auto mb-3">
              <Smartphone size={24} />
            </div>

            <h3 className="text-base font-bold text-slate-900 dark:text-slate-100">
              Live Examiner Mobile Handoff
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 mb-4">
              Scan with your phone camera to test Farmer Mode & voice guidance directly on mobile.
            </p>

            {/* QR Code Container */}
            <div className="relative p-3 bg-white rounded-xl border border-slate-200 shadow-inner w-56 h-56 mx-auto mb-4 flex items-center justify-center">
              <img
                src="/project_qr.png"
                alt="AgriPulse 2.0 Live QR Code"
                className="w-full h-full object-contain"
              />
            </div>

            {/* URL Display & Copy */}
            <div className="flex items-center gap-1.5 p-2 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 text-xs mb-4">
              <span className="truncate text-slate-600 dark:text-slate-300 font-mono flex-1 text-left select-all">
                {liveUrl}
              </span>
              <button
                type="button"
                onClick={handleCopy}
                className="p-1 rounded text-slate-500 hover:text-emerald-600 dark:hover:text-emerald-400"
                title="Copy link"
              >
                {copied ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
              </button>
              <a
                href={liveUrl}
                target="_blank"
                rel="noreferrer"
                className="p-1 rounded text-slate-500 hover:text-emerald-600 dark:hover:text-emerald-400"
                title="Open new tab"
              >
                <ExternalLink size={14} />
              </a>
            </div>

            <button
              onClick={onClose}
              className="w-full py-2.5 px-4 rounded-xl text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 transition-colors shadow-sm"
            >
              Done
            </button>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
export default QrCodeModal;
