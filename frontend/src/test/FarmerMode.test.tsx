import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import FarmerMode, { I18N } from '../components/FarmerMode';

vi.mock('../services/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: [] }),
    post: vi.fn(),
  },
}));

const addToast = vi.fn();

describe('FarmerMode Component & Accessibility', () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
