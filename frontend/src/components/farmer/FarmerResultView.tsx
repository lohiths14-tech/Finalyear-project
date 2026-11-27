import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Volume2, VolumeX, Info, Download, MessageCircle } from 'lucide-react';
import { FarmerDosageCalculator, type DosageCalcResult } from './FarmerDosageCalculator';
import type { FieldResult, LeafResult } from '../../types';
import type { TranslationDictionary } from './i18n';

interface FarmerResultViewProps {
  result: LeafResult | FieldResult;
  t: TranslationDictionary;
  onBack: () => void;
  isSpeaking: boolean;
  onToggleVoice: () => void;
  landArea: number;
  landUnit: string;
  onLandAreaChange: (area: number) => void;
  onLandUnitChange: (unit: string) => void;
  dosageCalculations: DosageCalcResult | null;
  onShareWhatsApp: () => void;
  onFeedback: (positive: boolean) => void;
}

export const FarmerResultView: React.FC<FarmerResultViewProps> = ({
  result,
  t,
  onBack,
  isSpeaking,
  onToggleVoice,
  landArea,
  landUnit,
  onLandAreaChange,
  onLandUnitChange,
  dosageCalculations,
  onShareWhatsApp,
  onFeedback,
}) => {
  const diseaseKey = Object.keys(t.diseases).find(k => k.toLowerCase() === result.disease_type.toLowerCase()) || "Healthy";
  const localizedDisease = (t.diseases as Record<string, { name: string; whatIsIt: string; whatToDo: string; dosePerLiter: number; chemName: string; unit: string; frac: string; phi: number }>)[diseaseKey] || {
    name: result.disease_type,
    whatIsIt: result.farmer_friendly?.what_is_it || "Disease symptoms observed on crop.",
    whatToDo: result.treatment,
    dosePerLiter: 0,
    chemName: "Treatment",
    unit: "g",
    frac: "N/A",
    phi: 0,
  };

  const isFieldResult = 'average_severity' in result;
  const severity = isFieldResult ? result.average_severity : result.severity;
  const isHealthy = result.disease_type === 'Healthy';
  const certaintyText = result.confidence > 75 ? t.highCertainty : result.confidence > 45 ? t.medCertainty : t.lowCertainty;
  const isLowCertainty = result.confidence <= 45;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="farmer-result">
      <button className="btn-icon" onClick={onBack} style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
        <ArrowLeft size={18} /> {t.backHome}
      </button>

      <div className="card" style={{ textAlign: 'center', padding: '24px 18px', background: 'var(--bg-card)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div className={`certainty-badge ${isHealthy ? 'certainty-high' : isLowCertainty ? 'certainty-low' : 'certainty-medium'}`}>
            {certaintyText} ({result.confidence.toFixed(0)}%)
          </div>

          {/* Native Voice Audio Button */}
          <button
            onClick={onToggleVoice}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              borderRadius: 20,
              border: 'none',
              background: isSpeaking ? 'var(--danger)' : 'var(--primary)',
              color: '#ffffff',
              fontWeight: 700,
              fontSize: '0.82rem',
              cursor: 'pointer',
              boxShadow: isSpeaking ? '0 0 10px rgba(239, 68, 68, 0.5)' : '0 2px 6px rgba(16, 185, 129, 0.3)',
            }}
          >
            {isSpeaking ? <VolumeX size={16} /> : <Volume2 size={16} />}
            {isSpeaking ? t.stopVoice : t.listenVoice}
          </button>
        </div>

        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: isHealthy ? 'var(--success)' : isLowCertainty ? 'var(--warning)' : 'var(--danger)', marginBottom: 8 }}>
          {localizedDisease.name}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: 18, lineHeight: 1.5 }}>
          {localizedDisease.whatIsIt}
        </p>

        <div style={{ display: 'flex', justifyContent: 'center', gap: 18, flexWrap: 'wrap', marginBottom: 18, color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
          <span>{isFieldResult ? 'Average field severity' : 'Leaf severity'}: <strong style={{ color: 'var(--text-primary)' }}>{severity.toFixed(1)}%</strong></span>
          {isFieldResult && <span>Infected area: <strong style={{ color: 'var(--text-primary)' }}>{result.infection_rate.toFixed(1)}%</strong></span>}
          {isFieldResult && <span>Spread risk: <strong style={{ color: 'var(--text-primary)' }}>{result.spread_risk}</strong></span>}
        </div>

        {isLowCertainty && (
          <div style={{ background: 'var(--warning-glow)', border: '1px solid rgba(217, 119, 6, 0.35)', padding: 14, borderRadius: 12, marginBottom: 18, textAlign: 'left' }}>
            <p style={{ fontWeight: 800, marginBottom: 6, color: 'var(--warning)', fontSize: '0.86rem' }}>{t.lowCertaintyStepsTitle}</p>
            <ul style={{ fontSize: '0.84rem', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4, margin: 0, color: 'var(--text-secondary)' }}>
              <li>{t.step1}</li>
              <li>{t.step2}</li>
              <li>{t.step3}</li>
            </ul>
          </div>
        )}

        {/* Action Advisory Box */}
        <div style={{ 
          background: isHealthy ? 'var(--primary-glow)' : 'var(--warning-glow)', 
          border: `1px solid ${isHealthy ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`, 
          padding: 16, 
          borderRadius: 14, 
          textAlign: 'left', 
          marginBottom: 18 
        }}>
          <h4 style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, color: isHealthy ? 'var(--primary)' : 'var(--warning)', fontWeight: 800, fontSize: '0.95rem' }}>
            <Info size={18} /> {t.whatToDo}
          </h4>
          <p style={{ fontSize: '0.92rem', lineHeight: 1.6, color: 'var(--text-primary)', margin: 0 }}>
            {localizedDisease.whatToDo}
          </p>
        </div>

        {/* Knapsack Sprayer & Tank Dosage Calculator */}
        {!isHealthy && dosageCalculations && (
          <FarmerDosageCalculator
            t={t}
            landArea={landArea}
            landUnit={landUnit}
            onLandAreaChange={onLandAreaChange}
            onLandUnitChange={onLandUnitChange}
            dosageCalculations={dosageCalculations}
          />
        )}

        {/* Action Sharing Buttons */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <button className="btn-icon" style={{ padding: '10px 14px', justifyContent: 'center' }} onClick={() => window.print()}>
            <Download size={16} /> {t.report}
          </button>
          <button
            className="btn-primary"
            style={{ background: '#25D366', borderColor: '#25D366', color: '#ffffff' }}
            onClick={onShareWhatsApp}
          >
            <MessageCircle size={16} /> {t.shareWhatsApp}
          </button>
        </div>

        <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 8 }}>{t.wasHelpful}</p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 14 }}>
            <button className="btn-icon" onClick={() => onFeedback(true)} style={{ padding: '6px 16px', borderRadius: 8 }}>
              {t.yes}
            </button>
            <button className="btn-icon" onClick={() => onFeedback(false)} style={{ padding: '6px 16px', borderRadius: 8 }}>
              {t.no}
            </button>
          </div>
        </div>

        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 16, fontStyle: 'italic', lineHeight: 1.4 }}>
          {t.disclaimer}
        </p>
      </div>
    </motion.div>
  );
};
