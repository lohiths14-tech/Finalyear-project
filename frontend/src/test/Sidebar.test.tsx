import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Sidebar from '../components/Sidebar';

describe('Sidebar Component', () => {
  it('renders brand title', () => {
    render(<Sidebar activeTab="overview" setActiveTab={() => {}} backendOnline={true} />);
    expect(screen.getByText(/AgriPulse/i)).toBeInTheDocument();
  });

  it('calls setActiveTab when a navigation item is clicked', () => {
    const setActiveTab = vi.fn();
    render(<Sidebar activeTab="overview" setActiveTab={setActiveTab} backendOnline={true} />);
    
    const leafButton = screen.getByText(/Leaf Scanner/i);
    fireEvent.click(leafButton);
    
    expect(setActiveTab).toHaveBeenCalledWith('leaf');
  });

  it('shows core engine online status when backend is online', () => {
    render(<Sidebar activeTab="overview" setActiveTab={() => {}} backendOnline={true} />);
    expect(screen.getByText(/Core Engine Online/i)).toBeInTheDocument();
  });

  it('shows offline warning when backend is disconnected', () => {
    render(<Sidebar activeTab="overview" setActiveTab={() => {}} backendOnline={false} />);
    expect(screen.getByText(/Backend Offline/i)).toBeInTheDocument();
  });
});
