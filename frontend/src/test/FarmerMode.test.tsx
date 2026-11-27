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
    localStorage.removeItem('agri_uncertain_queue');
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

  it('keeps an incomplete leaf response queued and requires an explicit retry after unknown delivery', async () => {
    const queuedScan = { id: 102, type: 'leaf', name: 'saved.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockResolvedValue({ data: { disease_type: 'Rice Blast', severity: 35, confidence: 82, status: 'Disease Detected' } });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));

    await waitFor(() => expect(screen.getByText(/server may already have processed them/i)).toBeInTheDocument());
    expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([queuedScan]);
    expect(apiClient.post).toHaveBeenCalledTimes(1);
  });

  it('rejects an incomplete field response without removing the queued scan', async () => {
    const queuedScan = { id: 104, type: 'field', name: 'saved-field.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockResolvedValue({ data: {
      disease_type: 'Rice Blast', average_severity: 35, infection_rate: 20, spread_risk: 'low',
      treatment: 'Observe symptoms.', veg_coverage: 70, confidence: 82,
      grid_data: [], progression_data: [],
    } });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));

    await waitFor(() => expect(screen.getByText(/unknown delivery status/i)).toBeInTheDocument());
    expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([queuedScan]);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('rejects invalid leaf numeric values (%s)', async (invalidValue) => {
    const queuedScan = { id: 105, type: 'leaf', name: 'saved.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockResolvedValue({ data: {
      disease_type: 'Rice Blast', severity: invalidValue, confidence: 82,
      status: 'Disease Detected', treatment: 'Observe symptoms.',
    } });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));

    await waitFor(() => expect(screen.getByText(/unknown delivery status/i)).toBeInTheDocument());
    expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([queuedScan]);
  });

  it('does not automatically resubmit after a network failure, but permits an explicit retry', async () => {
    const queuedScan = { id: 106, type: 'leaf', name: 'saved.jpg', data: 'data:image/jpeg;base64,YQ==' };
    localStorage.setItem('agri_offline_queue', JSON.stringify([queuedScan]));
    vi.mocked(apiClient.post).mockRejectedValue(new Error('Connection lost after send'));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob(['a'], { type: 'image/jpeg' }) }));

    render(<FarmerMode addToast={addToast} backendOnline={true} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'en' } });
    fireEvent.click(screen.getByText(/Sync Offline Scans/i));
    await waitFor(() => expect(screen.getByRole('button', { name: /Explicitly retry uncertain scans/i })).toBeInTheDocument());

    fireEvent.click(screen.getByText(/Sync Offline Scans/i));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(1));
    expect(JSON.parse(localStorage.getItem('agri_offline_queue') || '[]')).toEqual([queuedScan]);

    fireEvent.click(screen.getByRole('button', { name: /Explicitly retry uncertain scans/i }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(2));
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
