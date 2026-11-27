import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { FarmerResultView } from '../components/farmer/FarmerResultView';
import { I18N } from '../components/farmer/i18n';
import type { FieldResult } from '../types';

const fieldResult: FieldResult = {
  disease_type: 'Rice Blast',
  average_severity: 24,
  infection_rate: 12,
  spread_risk: 'MODERATE',
  treatment: 'Consult local extension guidance.',
  veg_coverage: 70,
  grid_data: [],
  progression_data: [],
  spectral_radar: [],
  confidence: 80,
};

describe('FarmerResultView field response', () => {
  it('renders field severity and infection rate from the field API contract', () => {
    render(
      <FarmerResultView
        result={fieldResult}
        t={I18N.en}
        onBack={vi.fn()}
        isSpeaking={false}
        onToggleVoice={vi.fn()}
        landArea={1}
        landUnit="acre"
        onLandAreaChange={vi.fn()}
        onLandUnitChange={vi.fn()}
        dosageCalculations={null}
        onShareWhatsApp={vi.fn()}
        onFeedback={vi.fn()}
      />
    );

    expect(screen.getByText(/Average field severity:/)).toHaveTextContent('24.0%');
    expect(screen.getByText(/Infected area:/)).toHaveTextContent('12.0%');
    expect(screen.queryByText(/Leaf severity:/)).not.toBeInTheDocument();
  });
});
