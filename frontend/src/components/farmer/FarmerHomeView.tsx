import React from 'react';
import { motion } from 'framer-motion';
import { Camera, Map, History, Upload, Info } from 'lucide-react';
import { FarmerLanguageSelector } from './FarmerLanguageSelector';
import SampleGallery from '../ui/SampleGallery';
import type { Language, OfflineQueueItem } from '../../types';
import type { TranslationDictionary } from './i18n';

interface FarmerHomeViewProps {
  t: TranslationDictionary;
  lang: Language;
  onLanguageChange: (lang: Language) => void;
  offlineQueue: OfflineQueueItem[];
  onSyncQueue: () => void;
  onScan: (e: React.ChangeEvent<HTMLInputElement>, type: 'leaf' | 'field') => void;
  onSelectSample?: (file: File) => void;
  onNavigateHistory: () => void;
}

export const FarmerHomeView: React.FC<FarmerHomeViewProps> = ({
  t,
  lang,
  onLanguageChange,
  offlineQueue,
  onSyncQueue,
  onScan,
  onSelectSample,
  onNavigateHistory,
}) => {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="farmer-home">
      <div style={{ background: '#ecfdf5', border: '1px solid rgba(16,185,129,0.3)', padding: '12px 16px', borderRadius: 12, marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12 }}>
        <Info size={20} color="var(--primary)" />
        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          {t.tip}
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, background: 'var(--bg-card)', padding: '12px 16px', borderRadius: 12, border: '1px solid var(--border)' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>{t.welcome}</h2>
        <FarmerLanguageSelector lang={lang} onLanguageChange={onLanguageChange} />
      </div>

      <div style={{ display: 'grid', gap: 16 }}>
        {offlineQueue.length > 0 && (
          <div className="action-card" onClick={onSyncQueue} style={{ background: '#eff6ff', borderColor: '#3b82f6', cursor: 'pointer' }}>
            <Upload size={32} color="#3b82f6" />
            <div>
              <div className="farmer-label" style={{ color: '#1e40af' }}>{t.syncScans} ({offlineQueue.length})</div>
              <div className="farmer-sublabel">{t.syncScansSub}</div>
            </div>
          </div>
        )}

        <label className="action-card" style={{ border: '2px solid var(--primary)', background: '#f0fdf4', cursor: 'pointer', padding: 22 }}>
          <Map size={44} color="var(--primary)" />
          <div>
            <div className="farmer-label" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)' }}>{t.scanField}</div>
            <div className="farmer-sublabel" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{t.scanFieldSub}</div>
          </div>
          <input type="file" hidden accept="image/*" onChange={(e) => onScan(e, 'field')} />
        </label>

        <label className="action-card" style={{ background: 'var(--bg-card)', cursor: 'pointer', padding: 22 }}>
          <Camera size={38} color="var(--accent)" />
          <div>
            <div className="farmer-label" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{t.scanLeaf}</div>
            <div className="farmer-sublabel" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{t.scanLeafSub}</div>
          </div>
          <input type="file" hidden accept="image/*" capture="environment" onChange={(e) => onScan(e, 'leaf')} />
        </label>

        <div className="action-card" onClick={onNavigateHistory} style={{ minHeight: 90, flexDirection: 'row', background: 'var(--bg-card)', cursor: 'pointer', padding: 18 }}>
          <History size={28} color="var(--text-muted)" />
          <div style={{ textAlign: 'left' }}>
            <div className="farmer-label" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>{t.pastScans}</div>
            <div className="farmer-sublabel" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{t.pastScansSub}</div>
          </div>
        </div>

        {onSelectSample && (
          <div style={{ marginTop: 8 }}>
            <SampleGallery onSelectSample={onSelectSample} />
          </div>
        )}
      </div>
    </motion.div>
  );
};
