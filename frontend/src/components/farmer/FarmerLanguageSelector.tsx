import React from 'react';
import { Globe } from 'lucide-react';
import type { Language } from '../../types';

interface FarmerLanguageSelectorProps {
  lang: Language;
  onLanguageChange: (lang: Language) => void;
}

export const FarmerLanguageSelector: React.FC<FarmerLanguageSelectorProps> = ({
  lang,
  onLanguageChange,
}) => {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <Globe size={16} color="var(--primary)" />
      <select
        value={lang}
        onChange={(e) => onLanguageChange(e.target.value as Language)}
        style={{
          background: 'var(--bg-subtle)',
          color: 'var(--text-primary)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '6px 12px',
          fontSize: '0.88rem',
          fontWeight: 700,
          outline: 'none',
          cursor: 'pointer',
        }}
      >
        <option value="kn">ಕನ್ನಡ (Kannada)</option>
        <option value="en">English</option>
        <option value="hi">हिन्दी (Hindi)</option>
        <option value="te">తెలుగు (Telugu)</option>
        <option value="ta">தமிழ் (Tamil)</option>
      </select>
    </div>
  );
};
