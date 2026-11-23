import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SpectralTab from '../components/tabs/SpectralTab';

describe('SpectralTab', () => {
  it('renders reference banner and disclaimer', () => {
    render(<SpectralTab lastScan={null} />);
    expect(
      screen.getByText(/Showing healthy crop reference baseline/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/REFERENCE MODE/i)).toBeInTheDocument();
  });

  it('renders NGVC Spectral Curve card title', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/NGVC Spectral Curve/i)).toBeInTheDocument();
    expect(screen.getByText(/128-Bin Histogram/i)).toBeInTheDocument();
  });

  it('renders Neural Activation Features card', () => {
    render(<SpectralTab lastScan={null} />);
    expect(screen.getByText(/Neural Activation Features/i)).toBeInTheDocument();
  });
});
