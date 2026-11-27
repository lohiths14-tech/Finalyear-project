import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import apiClient, { getScanDetail } from '../../services/api';
import type { FieldResult, LeafResult, ScanHistoryItem, FarmerView } from '../../types';
import type { TranslationDictionary } from './i18n';

interface FarmerHistoryViewProps {
  setView: (view: FarmerView) => void;
  setResult: (result: LeafResult | FieldResult | null) => void;
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
  t: TranslationDictionary;
}

const isFieldResult = (value: unknown): value is FieldResult => {
  if (!value || typeof value !== 'object') return false;
  const result = value as Record<string, unknown>;
  return typeof result.disease_type === 'string' &&
    typeof result.average_severity === 'number' &&
    typeof result.infection_rate === 'number' &&
    typeof result.confidence === 'number' &&
    typeof result.treatment === 'string' &&
    Array.isArray(result.grid_data);
};

const isLeafResult = (value: unknown): value is LeafResult => {
  if (!value || typeof value !== 'object') return false;
  const result = value as Record<string, unknown>;
  return typeof result.disease_type === 'string' &&
    typeof result.severity === 'number' &&
    typeof result.confidence === 'number' &&
    typeof result.status === 'string' &&
    typeof result.treatment === 'string';
};

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
              onClick={async () => {
                try {
                  const detail = await getScanDetail(item.id);
                  const scanResult = detail.full_results;
                  if (item.scan_type === 'field' && isFieldResult(scanResult)) {
                    setResult(scanResult);
                    setView('result');
                  } else if (item.scan_type === 'leaf' && isLeafResult(scanResult)) {
                    setResult(scanResult);
                    setView('result');
                  } else {
                    addToast('Detailed scan result is unavailable; the history summary is unchanged.', 'warning');
                  }
                } catch {
                  addToast('Could not load the saved scan details. The history summary is still available.', 'warning');
                }
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
