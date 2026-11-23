import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import apiClient from '../../services/api';
import type { LeafResult, ScanHistoryItem, FarmerView } from '../../types';
import type { TranslationDictionary } from './i18n';

interface FarmerHistoryViewProps {
  setView: (view: FarmerView) => void;
  setResult: (result: LeafResult | null) => void;
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
  t: TranslationDictionary;
}

export const FarmerHistoryView: React.FC<FarmerHistoryViewProps> = ({
  setView,
  setResult,
  addToast,
  t,
}) => {
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data } = await apiClient.get<ScanHistoryItem[]>('/history?limit=20');
        setHistory(data);
      } catch {
        addToast('Could not load history', 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [addToast]);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <button className="btn-icon" onClick={() => setView('home')} style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 6 }}>
        <ArrowLeft size={18} /> {t.back}
      </button>
      <h3 style={{ marginBottom: 16, fontSize: '1.2rem', fontWeight: 700 }}>{t.pastScans}</h3>

      {loading ? (
        <div className="spinner" style={{ margin: '40px auto' }} />
      ) : history.length === 0 ? (
        <p style={{ color: 'var(--text-muted)' }}>No past scans found.</p>
      ) : (
        <div style={{ display: 'grid', gap: 12 }}>
          {history.map((item) => (
            <div
              key={item.id}
              className="card"
              style={{ padding: 16, textAlign: 'left', cursor: 'pointer', background: 'var(--bg-card)' }}
              onClick={() => {
                setResult(item as unknown as LeafResult);
                setView('result');
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {new Date(item.timestamp).toLocaleDateString()}
                </span>
                <span className={`certainty-badge ${item.confidence > 70 ? 'certainty-high' : 'certainty-medium'}`} style={{ fontSize: '0.65rem' }}>
                  {item.scan_type.toUpperCase()}
                </span>
              </div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, marginTop: 4, color: 'var(--text-primary)' }}>{item.disease_type}</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--primary)', fontWeight: 600 }}>Severity: {item.severity.toFixed(0)}%</div>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
};
