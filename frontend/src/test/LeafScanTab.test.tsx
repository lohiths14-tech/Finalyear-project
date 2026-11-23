import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import LeafScanTab from '../components/tabs/LeafScanTab';

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

  it('shows Field Pro-Tip guidance', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} />);
    expect(screen.getByText(/Field Pro-Tip/i)).toBeInTheDocument();
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

  it('pre-populates result from lastLeafScan prop', () => {
    render(<LeafScanTab addToast={addToast} backendOnline={true} lastLeafScan={mockLeafResult} />);
    expect(screen.getAllByText(/Rice Blast/i)[0]).toBeInTheDocument();
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
