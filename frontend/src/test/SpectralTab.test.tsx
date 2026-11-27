import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SpectralTab from '../components/tabs/SpectralTab';

describe('SpectralTab', () => {
  it('shows unavailable states instead of generated sample curves or feature scores', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/No scan-derived spectral data available/i)).toBeInTheDocument();
    expect(screen.getByText(/NO SCAN DATA/i)).toBeInTheDocument();
    expect(screen.getByText(/no reference curves or placeholder scores are generated/i)).toBeInTheDocument();
    expect(screen.getByText(/NGVC values are unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/Model-derived proxy features are unavailable/i)).toBeInTheDocument();
    expect(screen.getAllByText('Unavailable')).toHaveLength(1);
    expect(screen.queryByText(/Illustrative example/i)).not.toBeInTheDocument();
  });

  it('renders NGVC Spectral Curve card title', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/Image-derived NGVC profile/i)).toBeInTheDocument();
  });

  it('renders model-derived proxy features card', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByRole('heading', { name: /Model-derived proxy features/i })).toBeInTheDocument();
  });

  it('labels field response measurements as image-derived proxy data', () => {
    render(
      <SpectralTab
        lastScan={{
          disease_type: 'Rice Blast',
          average_severity: 21,
          infection_rate: 12,
          spread_risk: 'MODERATE',
          treatment: 'Observe field conditions.',
          veg_coverage: 68,
          grid_data: [],
          progression_data: [],
          spectral_radar: [{ subject: 'Chlorophyll (proxy)', A: 42, B: 58 }],
          ngvc_data: [{ intensity: 0, value: 0.08 }],
          confidence: 79,
        }}
      />
    );

    expect(screen.getByText(/Image-derived model features for: Rice Blast/i)).toBeInTheDocument();
    expect(screen.getByText(/Model A proxy/i)).toBeInTheDocument();
    expect(screen.getByText(/not direct physical spectral measurements/i)).toBeInTheDocument();
  });

  it('keeps every NGVC point returned by the scan', () => {
    const ngvcData = Array.from({ length: 70 }, (_, intensity) => ({ intensity, value: intensity / 100 }));
    render(
      <SpectralTab
        lastScan={{
          disease_type: 'Rice Blast',
          average_severity: 21,
          infection_rate: 12,
          spread_risk: 'MODERATE',
          treatment: 'Observe field conditions.',
          veg_coverage: 68,
          grid_data: [],
          progression_data: [],
          spectral_radar: [],
          ngvc_data: ngvcData,
          confidence: 79,
        }}
      />
    );

    expect(screen.getByRole('img', { name: /70 data points/i })).toBeInTheDocument();
    expect(screen.getByText(/70 NGVC points are derived from this scan image/i)).toBeInTheDocument();
  });
});
