import React from 'react';
import { Calculator, ShieldAlert } from 'lucide-react';
import type { TranslationDictionary } from './i18n';

export interface DosageCalcResult {
  acres: number;
  waterLiters: number;
  tankPumps: number;
  totalChem: number;
  perTankChem: number;
  diseaseInfo: {
    name: string;
    whatIsIt: string;
    whatToDo: string;
    dosePerLiter: number;
    chemName: string;
    unit: string;
    frac: string;
    phi: number;
  };
}

interface FarmerDosageCalculatorProps {
  t: TranslationDictionary;
  landArea: number;
  landUnit: string;
  onLandAreaChange: (area: number) => void;
  onLandUnitChange: (unit: string) => void;
  dosageCalculations: DosageCalcResult;
}

export const FarmerDosageCalculator: React.FC<FarmerDosageCalculatorProps> = ({
  t,
  landArea,
  landUnit,
  onLandAreaChange,
  onLandUnitChange,
  dosageCalculations,
}) => {
  return (
    <div style={{ background: 'var(--bg-subtle)', border: '1px solid var(--border)', borderRadius: 12, padding: 16, marginBottom: 20, textAlign: 'left' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: 'var(--primary)', fontWeight: 700, fontSize: '0.95rem' }}>
        <Calculator size={18} /> {t.dosageCalcTitle}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
        <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>{t.landSize}</label>
        <input
          type="number"
          min="0.1"
          step="0.5"
          value={landArea}
          onChange={(e) => onLandAreaChange(Math.max(0.1, parseFloat(e.target.value) || 1.0))}
          style={{ width: 76, padding: '6px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontWeight: 700 }}
        />
        <select
          value={landUnit}
          onChange={(e) => onLandUnitChange(e.target.value)}
          style={{ padding: '6px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontWeight: 600 }}
        >
          <option value="acre">Acres (ಎಕರೆ)</option>
          <option value="guntha">Gunthas (ಗುಂಟೆ)</option>
          <option value="hectare">Hectares (ಹೆಕ್ಟೇರ್)</option>
          <option value="bigha">Bighas (ಬಿಘಾ)</option>
          <option value="cent">Cents (ಸೆಂಟ್)</option>
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 8, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{t.pumpsNeeded}</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--primary)' }}>
            {dosageCalculations.tankPumps} pumps
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
            ({dosageCalculations.waterLiters} L water)
          </div>
        </div>

        <div style={{ background: 'var(--bg-card)', padding: 10, borderRadius: 8, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{t.chemNeeded}</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--accent)' }}>
            {dosageCalculations.totalChem} {dosageCalculations.diseaseInfo.unit}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
            ({dosageCalculations.perTankChem} {dosageCalculations.diseaseInfo.unit}/tank)
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 4, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
        <div><strong>{t.fracCode}</strong> {dosageCalculations.diseaseInfo.frac}</div>
        <div><strong>{t.phiDays}</strong> {dosageCalculations.diseaseInfo.phi} days before harvest</div>
      </div>

      {/* PPE Checklist */}
      <div style={{ marginTop: 12, background: 'var(--accent-glow)', padding: 12, borderRadius: 10, border: '1px solid rgba(79, 70, 229, 0.25)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-light)', marginBottom: 4 }}>
          <ShieldAlert size={15} /> {t.safetyGear}
        </div>
        <div style={{ fontSize: '0.78rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
          😷 Wear N95 Mask · 🧤 Rubber Gloves · 🥾 Boots · 🧼 Wash hands after spray
        </div>
      </div>
    </div>
  );
};
