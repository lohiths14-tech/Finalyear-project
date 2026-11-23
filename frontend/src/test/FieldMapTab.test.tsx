import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import FieldMapTab from '../components/tabs/FieldMapTab';

vi.mock('../services/api', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
  },
  analyzeField: vi.fn(),
  consultField: vi.fn(),
  getAiStatus: vi.fn().mockResolvedValue({}),
}));

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
    expect(screen.getByText(/No Field Scan Loaded/i)).toBeInTheDocument();
    expect(screen.getByText(/Drop a field image here or click to upload/i)).toBeInTheDocument();
  });
});
