import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import FarmerMode, { I18N } from '../components/FarmerMode';
import apiClient from '../services/api';

vi.mock('../services/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: [] }),
    post: vi.fn(),
  },
}));

const addToast = vi.fn();

describe('FarmerMode Component & Accessibility', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    localStorage.removeItem('agri_offline_queue');
  });

  it('renders the app title and farmer badge', () => {
    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    expect(screen.getAllByText(/AgriPulse/i)[0]).toBeInTheDocument();
    expect(screen.getByText(/Farmer Mode/i)).toBeInTheDocument();
  });

  it('renders online indicator when backend is online', () => {
    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    expect(screen.getByText('Online')).toBeInTheDocument();
  });

  it('renders offline indicator when backend is offline', () => {
    render(<FarmerMode addToast={addToast} backendOnline={false} />);
    expect(screen.getByText('Offline')).toBeInTheDocument();
  });

  it('renders all four language options in the selector', async () => {
    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    await waitFor(() => {
      expect(screen.getByText(/ಕನ್ನಡ/i)).toBeInTheDocument();
      expect(screen.getByText(/English/i)).toBeInTheDocument();
      expect(screen.getByText(/हिन्दी/i)).toBeInTheDocument();
      expect(screen.getByText(/తెలుగు/i)).toBeInTheDocument();
    });
  });

  it('switches language cleanly on dropdown change', async () => {
    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'en' } });
    expect(screen.getByText(I18N.en.welcome)).toBeInTheDocument();

    fireEvent.change(select, { target: { value: 'hi' } });
    expect(screen.getByText(I18N.hi.welcome)).toBeInTheDocument();
  });

  it('removes queued scans only after the server returns a valid result', async () => {
    const queuedScan = { id: 101, type: 'leaf', name: 'saved.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockResolvedValue({
      data: { disease_type: 'Rice Blast', severity: 35, confidence: 82, status: 'Disease Detected', treatment: 'Observe symptoms.' },
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));

    await waitFor(() => {
      expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([]);
      expect(addToast).toHaveBeenCalledWith('Successfully synced 1 offline scans.', 'success');
    });
  });

  it('does not submit the same queued scan twice during concurrent sync clicks', async () => {
    const queuedScan = { id: 103, type: 'leaf', name: 'saved.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockResolvedValue({
      data: { disease_type: 'Rice Blast', severity: 35, confidence: 82, status: 'Disease Detected', treatment: 'Observe symptoms.' },
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    const syncButton = screen.getByText(/Sync Offline Scans/i);
    fireEvent.click(syncButton);
    fireEvent.click(syncButton);

    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(1));
  });

  it('keeps queued scans when the server does not acknowledge them', async () => {
    const queuedScan = { id: 102, type: 'field', name: 'saved-field.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockRejectedValue(new Error('Connection lost'));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith('0 of 1 scans synced; 1 remain queued.', 'warning', 6000);
    });
    expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([queuedScan]);
  });

  it('contains valid ICAR dosage parameters across all disease definitions', () => {
    const diseases = ['Healthy', 'Rice Blast', 'Brown Spot', 'Bacterial Blight', 'Leaf Scald'] as const;
    for (const d of diseases) {
      const info = I18N.en.diseases[d];
      expect(info).toBeDefined();
      expect(info.name).toBeTruthy();
      expect(info.whatToDo).toBeTruthy();
      expect(info.frac).toBeDefined();
      expect(typeof info.dosePerLiter).toBe('number');
      expect(typeof info.phi).toBe('number');
    }
  });
});
