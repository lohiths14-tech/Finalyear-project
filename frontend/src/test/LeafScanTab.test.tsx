import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import LeafScanTab from '../components/tabs/LeafScanTab';
import EconomicImpactWidget from '../components/ui/EconomicImpactWidget';
import ProgressionSimulator from '../components/ui/ProgressionSimulator';

vi.mock('../services/api', () => ({
  default: {
    post: vi.fn(),
  },
}));

import apiClient from '../services/api';

const addToast = vi.fn();

const mockLeafResult = {
  disease_type: 'Rice Blast (Magnaporthe oryzae)',
  severity: 72,
  confidence: 84,
  status: 'Disease Detected',
  treatment: 'Tricyclazole 75% WP @ 0.6 g/L every 14 days.',
  scan_id: 'abc-123',
  disclaimer: 'AI suggestion only.',
  farmer_friendly: {
    simple_label: 'Leaf Rot',
    certainty: 'High Certainty' as const,
    what_is_it: 'A fungal infection.',
    what_to_do: 'Apply fungicide immediately.',
  },
};

describe('LeafScanTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockLeafResult });
  });

  it('renders the drag/click upload area', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    expect(screen.getByText(/Click or drop leaf photograph/i)).toBeInTheDocument();
  });

  it('disables the file input when backend is offline', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={false} />);
    const input = document.querySelector('input[type=file]') as HTMLInputElement;
    expect(input).toBeDisabled();
  });

  it('shows honest image capture guidance', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    expect(screen.getByText(/Capture guidance/i)).toBeInTheDocument();
  });

  it('shows toast when non-image file is selected', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    const input = document.querySelector('input[type=file]') as HTMLInputElement;
    const file = new File(['hello'], 'test.txt', { type: 'text/plain' });
    Object.defineProperty(input, 'files', { value: [file], writable: false });
    fireEvent.change(input);
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('valid image'), 'error');
  });

  it('shows toast when file exceeds 20MB', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    const input = document.querySelector('input[type=file]') as HTMLInputElement;
    const bigFile = new File([new ArrayBuffer(21 * 1024 * 1024)], 'big.jpg', { type: 'image/jpeg' });
    Object.defineProperty(input, 'files', { value: [bigFile], writable: false });
    fireEvent.change(input);
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('large'), 'error');
  });

  it('pre-populates result from lastLeafScan prop without inventing benchmark details', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} lastLeafScan={mockLeafResult} />);
    expect(screen.getAllByText(/Rice Blast/i)[0]).toBeInTheDocument();
    expect(screen.queryByText(/0\.038|MobileNetV2|6\.65 ms/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Optional scan diagnostics/)).not.toBeInTheDocument();
  });

  it('shows optional diagnostic fields only when returned by the backend', () => {
    const resultWithDiagnostics = {
      ...mockLeafResult,
      quality: {
        quality: { quality_label: 'Good', iqi_score: 92, issues: ['Slight blur'], suggestions: ['Hold camera steady'], proceed_to_inference: true },
        confidence: { confidence: 87, level: 'High', tone: 'positive', message: 'Image quality is adequate.' },
        reliability: { score: 0.82, label: 'Reliable' },
        recommendation: 'Retake if blur increases.',
        disease_label: 'Rice Blast',
      },
      reliability: { score: 0.82, is_reliable: true },
      ood: { is_ood: false, ood_score: 0.12 },
    };
    render(<LeafScanTab addToast={addToast} backendOnline={true} lastLeafScan={resultWithDiagnostics} />);
    fireEvent.click(screen.getByRole('button', { name: /Optional scan diagnostics/i }));
    expect(screen.getByText(/Good · score 92/)).toBeInTheDocument();
    expect(screen.getByText(/Slight blur/)).toBeInTheDocument();
    expect(screen.queryByText(/MobileNetV2/)).not.toBeInTheDocument();
  });

  it('does not show a disease classification when the backend withheld it', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} lastLeafScan={{ ...mockLeafResult, withheld: true }} />);
    expect(screen.getByRole('heading', { name: 'Classification withheld' })).toBeInTheDocument();
    expect(screen.queryByText(mockLeafResult.disease_type)).not.toBeInTheDocument();
    expect(screen.queryByText(/Scan follow-up questions/)).not.toBeInTheDocument();
  });

  it('does not calculate farm finances from image severity', () => {
    render(<EconomicImpactWidget diseaseType="Rice Blast" severity={72} confidence={84} />);
    expect(screen.getByText(/Farm economics not estimated/i)).toBeInTheDocument();
    expect(screen.getByText(/No verified financial inputs/i)).toBeInTheDocument();
    expect(screen.queryByText(/Potential Revenue Loss|Net Protected Profit|ROI/i)).not.toBeInTheDocument();
  });

  it('labels progression paths as illustrative and exposes their assumptions', () => {
    render(<ProgressionSimulator initialSeverity={40} diseaseType="Rice Blast" />);
    expect(screen.getByText(/Illustrative severity scenarios/i)).toBeInTheDocument();
    expect(screen.getByText(/not forecasts or treatment-efficacy estimates/i)).toBeInTheDocument();
    expect(screen.getByText(/16% of the remaining range per day/i)).toBeInTheDocument();
  });

  it('shows disease name after valid image is uploaded and scan completes', async () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    const input = document.querySelector('input[type=file]') as HTMLInputElement;
    const img = new File([new Uint8Array(100)], 'leaf.jpg', { type: 'image/jpeg' });
    Object.defineProperty(input, 'files', { value: [img], writable: false });
    await act(async () => {
      fireEvent.change(input);
      await new Promise(r => setTimeout(r, 50));
    });
    await waitFor(() => {
      expect(screen.getAllByText(/Rice Blast/i)[0]).toBeInTheDocument();
    }, { timeout: 3000 });
  });
});
