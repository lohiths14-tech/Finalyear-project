import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import OverviewTab from '../components/tabs/OverviewTab';

vi.mock('../services/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
  },
  getAiStatus: vi.fn().mockResolvedValue({
    active_provider: 'icar_rag_local',
    model_id: 'icar-deterministic',
    display_name: 'ICAR Deterministic RAG Engine',
    has_gemini: false,
    has_openai: false,
  }),
}));

beforeEach(() => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: false,
    json: () => Promise.resolve({}),
  });
});

describe('OverviewTab', () => {
  it('shows waiting for input message when no scan is provided', async () => {
    await act(async () => {
      render(<OverviewTab lastScan={null} />);
    });
    expect(screen.getByText(/Waiting for input/i)).toBeInTheDocument();
    expect(screen.getAllByText('Data unavailable').length).toBeGreaterThan(0);
    expect(screen.queryByText(/91\.8/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Paddy Doctor Benchmark/)).not.toBeInTheDocument();
  });

  it('does not render simulated progression estimates as measured history', async () => {
    const mockScan = {
      disease_type: 'Rice Blast',
      average_severity: 60,
      infection_rate: 25,
      spread_risk: 'Medium',
      confidence: 85,
      treatment: 'Apply fungicide',
      veg_coverage: 70,
      grid_data: [],
      temporal_available: false,
      progression_data: [{ day: 'Day 1', actual: 30, predicted: 30, simulated: true, point_type: 'estimated_history' }],
      spectral_radar: [],
    };
    await act(async () => {
      render(<OverviewTab lastScan={mockScan} />);
    });
    expect(screen.getByText(/Single-scan estimates are hidden/i)).toBeInTheDocument();
    expect(screen.queryByText('Day 1')).not.toBeInTheDocument();
  });

  it('renders disease name in the Dominant Disease card when lastScan is provided', async () => {
    const mockScan = {
      disease_type: 'Rice Blast',
      average_severity: 60,
      infection_rate: 25,
      spread_risk: 'Medium',
      confidence: 85.0,
      treatment: 'Apply fungicide',
      veg_coverage: 70,
      grid_data: [],
      progression_data: [],
      spectral_radar: [],
    };
    await act(async () => {
      render(<OverviewTab lastScan={mockScan} />);
    });
    const heading = screen.getAllByText(/Rice Blast/i)[0];
    expect(heading).toBeInTheDocument();
    expect(heading.tagName).toBe('H3');
  });

  it('renders field health score when lastScan is provided', async () => {
    const mockScan = {
      disease_type: 'Brown Spot',
      average_severity: 30,
      infection_rate: 10,
      spread_risk: 'Low',
      confidence: 85.0,
      treatment: 'Apply Propiconazole 25% EC',
      veg_coverage: 85,
      grid_data: [],
      progression_data: [],
      spectral_radar: [],
    };
    await act(async () => {
      render(<OverviewTab lastScan={mockScan} />);
    });
    expect(screen.getByText('70')).toBeInTheDocument();
    expect(screen.getByText('/100')).toBeInTheDocument();
  });

  it('renders Smart Recommendation label', async () => {
    await act(async () => {
      render(<OverviewTab lastScan={null} />);
    });
    expect(screen.getByText(/Smart Recommendation/i)).toBeInTheDocument();
  });
});
