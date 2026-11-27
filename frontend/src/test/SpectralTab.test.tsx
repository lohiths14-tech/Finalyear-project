import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SpectralTab from '../components/tabs/SpectralTab';

describe('SpectralTab', () => {
  it('renders reference banner and disclaimer', () => {
    render(<SpectralTab lastScan={null} />);
    expect(
      screen.getByText(/Illustrative reference example/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/REFERENCE MODE/i)).toBeInTheDocument();
  });

  it('renders NGVC Spectral Curve card title', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/Image-derived NGVC profile/i)).toBeInTheDocument();
    expect(screen.getByText(/Image-derived profile/i)).toBeInTheDocument();
  });

  it('renders model-derived proxy features card', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/Model-derived proxy features/i)).toBeInTheDocument();
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
});
