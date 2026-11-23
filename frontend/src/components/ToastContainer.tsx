import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertCircle, AlertTriangle, X } from 'lucide-react';
import type { Toast } from '../types';

const ToastIcon = ({ type }: { type: Toast['type'] }) => {
  if (type === 'success') return <CheckCircle size={16} />;
  if (type === 'error') return <AlertCircle size={16} />;
  if (type === 'warning') return <AlertTriangle size={16} />;
  return <AlertCircle size={16} />;
};

interface ToastContainerProps {
  toasts: Toast[];
  removeToast: (id: number) => void;
}

export default function ToastContainer({ toasts, removeToast }: ToastContainerProps) {
  return (
    <div className="toast-container">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            className={`toast toast-${toast.type}`}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.25 }}
          >
            <ToastIcon type={toast.type} />
            <span>{toast.message}</span>
            <button className="toast-close" onClick={() => removeToast(toast.id)}>
              <X size={14} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
