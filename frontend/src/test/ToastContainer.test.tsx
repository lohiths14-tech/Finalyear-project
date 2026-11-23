import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ToastContainer from '../components/ToastContainer';
import type { Toast } from '../types';

const makeToast = (overrides: Partial<Toast> = {}): Toast => ({
  id: 1,
  message: 'Test message',
  type: 'success',
  ...overrides,
});

describe('ToastContainer', () => {
  it('renders nothing when toast list is empty', () => {
    const { container } = render(<ToastContainer toasts={[]} removeToast={() => {}} />);
    expect(container.querySelector('.toast')).toBeNull();
  });

  it('renders toast message text', () => {
    const toasts = [makeToast({ message: 'Scan complete!' })];
    render(<ToastContainer toasts={toasts} removeToast={() => {}} />);
    expect(screen.getByText('Scan complete!')).toBeInTheDocument();
  });

  it('renders multiple toasts', () => {
    const toasts = [
      makeToast({ id: 1, message: 'First toast', type: 'success' }),
      makeToast({ id: 2, message: 'Second toast', type: 'error' }),
    ];
    render(<ToastContainer toasts={toasts} removeToast={() => {}} />);
    expect(screen.getByText('First toast')).toBeInTheDocument();
    expect(screen.getByText('Second toast')).toBeInTheDocument();
  });

  it('calls removeToast when close button is clicked', () => {
    const removeToast = vi.fn();
    const toasts = [makeToast({ id: 42 })];
    render(<ToastContainer toasts={toasts} removeToast={removeToast} />);
    const closeBtn = screen.getByRole('button');
    fireEvent.click(closeBtn);
    expect(removeToast).toHaveBeenCalledWith(42);
  });

  it('applies correct class for success type', () => {
    const toasts = [makeToast({ type: 'success' })];
    const { container } = render(<ToastContainer toasts={toasts} removeToast={() => {}} />);
    expect(container.querySelector('.toast-success')).toBeInTheDocument();
  });

  it('applies correct class for error type', () => {
    const toasts = [makeToast({ type: 'error' })];
    const { container } = render(<ToastContainer toasts={toasts} removeToast={() => {}} />);
    expect(container.querySelector('.toast-error')).toBeInTheDocument();
  });

  it('applies correct class for warning type', () => {
    const toasts = [makeToast({ type: 'warning' })];
    const { container } = render(<ToastContainer toasts={toasts} removeToast={() => {}} />);
    expect(container.querySelector('.toast-warning')).toBeInTheDocument();
  });
});
