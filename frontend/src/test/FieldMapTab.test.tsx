import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import FieldMapTab from '../components/tabs/FieldMapTab';
import { parseGoogleMapsCoordinates } from '../components/ui/FieldGisMap';

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
    expect(screen.getByText(/Ready for Field-Level Spatial Diagnostics/i)).toBeInTheDocument();
    expect(screen.getByText(/Choose Field Image/i)).toBeInTheDocument();
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
