import { render, screen, act, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import FieldMapTab, { InferenceProgress } from '../components/tabs/FieldMapTab';
import FieldGisMap, { parseGoogleMapsCoordinates } from '../components/ui/FieldGisMap';
import { analyzeFieldStream, consultField } from '../services/api';

vi.mock('../services/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
  },
  analyzeField: vi.fn(),
  analyzeFieldStream: vi.fn(),
  consultField: vi.fn(),
  getAiStatus: vi.fn().mockResolvedValue({}),
}));

describe('InferenceProgress', () => {
  it('stays indeterminate until the backend reports a percentage', () => {
    render(<InferenceProgress imageUrl={null} streamProgress={null} />);
    expect(screen.getByText(/Progress percentage is unavailable/i)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
  });

  it('shows a percentage only when supplied by the backend', () => {
    render(<InferenceProgress imageUrl={null} streamProgress={{ stage: 'analysis', percent: 42, message: 'Analyzing image' }} />);
    expect(screen.getByRole('progressbar', { name: /backend-reported/i })).toHaveAttribute('aria-valuenow', '42');
  });
});

describe('FieldMapTab', () => {
  it('renders upload prompt when no scan is uploaded', async () => {
    await act(async () => {
      render(
        <FieldMapTab
          addToast={vi.fn()}
          backendOnline={true}
          onScanComplete={vi.fn()}
        />
      );
    });
    expect(screen.getByText(/Field-Level Spatial Diagnosis/i)).toBeInTheDocument();
    expect(screen.getByText(/Upload Field Image/i)).toBeInTheDocument();
  });

  it('renders empty state when no scan is loaded', async () => {
    await act(async () => {
      render(
        <FieldMapTab
          addToast={vi.fn()}
          backendOnline={true}
          onScanComplete={vi.fn()}
        />
      );
    });
    expect(screen.getByText(/Ready for Field-Level Spatial Diagnostics/i)).toBeInTheDocument();
    expect(screen.getByText(/Choose Field Image/i)).toBeInTheDocument();
  });

  it('asks safe scan questions without supplying invented coverage or field area', async () => {
    vi.mocked(analyzeFieldStream).mockResolvedValueOnce({
      disease_type: 'Field-reported disease',
      average_severity: 0,
      infection_rate: 12,
      spread_risk: '',
      treatment: '',
      veg_coverage: 0,
      grid_data: [],
      progression_data: [],
      spectral_radar: [],
      confidence: 0,
    });
    vi.mocked(consultField).mockResolvedValueOnce({
      answer: 'Check local field observations before deciding.',
      provider: 'Test service',
      model: 'Test model',
      grounding_source: 'Returned scan values',
      timestamp: '10:00',
    });

    const { container } = render(
      <FieldMapTab addToast={vi.fn()} backendOnline={true} onScanComplete={vi.fn()} />
    );
    const fileInput = container.querySelector('input[type="file"]');
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, {
      target: { files: [new File(['image'], 'field.jpg', { type: 'image/jpeg' })] },
    });

    await screen.findByText('Field Scan Assistant');
    fireEvent.click(screen.getByRole('button', { name: 'What should I verify?' }));
    await waitFor(() => expect(consultField).toHaveBeenCalledOnce());

    const payload = vi.mocked(consultField).mock.calls[0][0];
    expect(payload).toMatchObject({
      spread_risk: 'UNAVAILABLE',
      infection_rate: 12,
      average_severity: 0,
      vegetation_coverage: 0,
    });
    expect(payload).not.toHaveProperty('land_area');
    expect(payload).not.toHaveProperty('area_unit');
    expect(payload.question).toContain('Do not invent a dose, area, or expected outcome.');
    expect(await screen.findByText('Response source: Returned scan values')).toBeInTheDocument();
  });
});

describe('FieldGisMap location provenance', () => {
  it('does not invent a selected location or boundary before user input', () => {
    render(<FieldGisMap imageUrl={null} gridData={[]} />);
    expect(screen.getByText(/No field location was provided/i)).toBeInTheDocument();
    expect(screen.getByText(/Location not set · no field boundary available/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Google Maps coordinates or link/i)).toHaveValue('');
    expect(screen.getByRole('button', { name: /Copy current coordinates/i })).toBeDisabled();
  });

});

describe('parseGoogleMapsCoordinates', () => {
  it('parses standard decimal Google Maps coordinates', () => {
    const res = parseGoogleMapsCoordinates('13.9299, 75.5681');
    expect(res).toEqual({ lat: 13.9299, lon: 75.5681 });
  });

  it('parses Google Maps @lat,lon URLs and ?q= URLs', () => {
    expect(parseGoogleMapsCoordinates('https://www.google.com/maps/@13.9312,75.5701,17z')).toEqual({
      lat: 13.9312,
      lon: 75.5701,
    });
    expect(parseGoogleMapsCoordinates('https://maps.google.com/?q=12.9716,77.5946')).toEqual({
      lat: 12.9716,
      lon: 77.5946,
    });
  });

  it('parses DMS coordinates', () => {
    const res = parseGoogleMapsCoordinates(`13°55'47.6"N 75°34'05.2"E`);
    expect(res).not.toBeNull();
    expect(res!.lat).toBeCloseTo(13.92989, 4);
    expect(res!.lon).toBeCloseTo(75.56811, 4);
  });

  it('returns null for invalid coordinates', () => {
    expect(parseGoogleMapsCoordinates('not a coordinate')).toBeNull();
    expect(parseGoogleMapsCoordinates('99.5, 200.5')).toBeNull();
  });
});
