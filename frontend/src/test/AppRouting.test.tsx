import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App, { getInitialRoute } from '../App';

vi.mock('../context/AuthContext', () => ({
  useAuth: () => ({ apiKey: '__no_auth__', login: vi.fn() }),
}));

vi.mock('../services/api', () => ({
  default: { get: vi.fn().mockResolvedValue({ data: { status: 'healthy' } }) },
}));

vi.mock('../components/LandingPage', () => ({
  default: ({ onEnterExpert, onEnterFarmer }: { onEnterExpert: () => void; onEnterFarmer: () => void }) => (
    <main>
      <h1>Landing experience</h1>
      <button onClick={onEnterFarmer}>Enter farmer experience</button>
      <button onClick={onEnterExpert}>Enter expert dashboard</button>
    </main>
  ),
}));

vi.mock('../components/Sidebar', () => ({
  default: ({ activeTab, setActiveTab, onGoHome }: {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    onGoHome?: () => void;
  }) => (
    <nav aria-label="Dashboard navigation">
      <span>Active tab: {activeTab}</span>
      <button onClick={() => setActiveTab('history')}>Go to history</button>
      <button onClick={onGoHome}>Return home</button>
    </nav>
  ),
}));

vi.mock('../components/ToastContainer', () => ({ default: () => null }));
vi.mock('../components/ErrorBoundary', () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock('../components/tabs/OverviewTab', () => ({ default: () => <div>Overview view</div> }));
vi.mock('../components/tabs/FieldMapTab', () => ({ default: () => <div>Field view</div> }));
vi.mock('../components/tabs/LeafScanTab', () => ({ default: () => <div>Leaf view</div> }));
vi.mock('../components/tabs/SpectralTab', () => ({ default: () => <div>Spectral view</div> }));
vi.mock('../components/tabs/HistoryTab', () => ({ default: () => <div>History view</div> }));
vi.mock('../components/FarmerMode', () => ({ default: () => <div>Farmer workspace active</div> }));
vi.mock('../components/ui/QrCodeModal', () => ({ default: () => null }));
vi.mock('../components/ui/HealthSentinel', () => ({ default: () => null }));

const dashboardTitles: Record<string, string> = {
  field: 'Field Intelligence',
  overview: 'System Overview',
  spectral: 'Spectral Intelligence',
  leaf: 'Micro Leaf Scanner',
  history: 'Audit History',
};

beforeEach(() => {
  window.history.replaceState({}, '', '/');
  localStorage.clear();
  vi.clearAllMocks();
});

describe('direct dashboard routing', () => {
  it.each(Object.keys(dashboardTitles))('opens the %s dashboard tab directly', async (tab) => {
    window.history.replaceState({}, '', `/?tab=${tab}&source=bookmark`);

    render(<App />);

    expect(await screen.findByRole('heading', { name: dashboardTitles[tab] })).toBeInTheDocument();
    expect(screen.getByText(`Active tab: ${tab}`)).toBeInTheDocument();
    expect(window.location.search).toBe(`?tab=${tab}&source=bookmark`);
  });

  it('opens the landing page when no dashboard tab is provided', async () => {
    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Landing experience' })).toBeInTheDocument();
  });

  it('safely falls back to the landing page for an unsupported tab', async () => {
    window.history.replaceState({}, '', '/?tab=not-a-dashboard&source=bookmark');

    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Landing experience' })).toBeInTheDocument();
    expect(window.location.search).toBe('?tab=not-a-dashboard&source=bookmark');
  });

  it('preserves unrelated query parameters when changing dashboard tabs', async () => {
    window.history.replaceState({}, '', '/?tab=overview&campaign=spring');

    render(<App />);
    await screen.findByRole('heading', { name: dashboardTitles.overview });
    fireEvent.click(screen.getByRole('button', { name: 'Go to history' }));

    expect(window.location.search).toBe('?tab=history&campaign=spring');
    expect(screen.getByText('Active tab: history')).toBeInTheDocument();
  });

  it('retains farmer and expert entry callbacks from the landing page', async () => {
    const { unmount } = render(<App />);
    expect(await screen.findByRole('heading', { name: 'Landing experience' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Enter farmer experience' }));
    expect(await screen.findByText('Farmer workspace active')).toBeInTheDocument();

    unmount();
    window.history.replaceState({}, '', '/');
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'Enter expert dashboard' }));
    expect(await screen.findByRole('heading', { name: dashboardTitles.field })).toBeInTheDocument();
  });
});

describe('getInitialRoute', () => {
  it('maps valid tabs to direct dashboard routes and invalid tabs to landing', () => {
    expect(getInitialRoute('?tab=spectral')).toEqual({ viewMode: 'app', activeTab: 'spectral' });
    expect(getInitialRoute('?tab=history')).toEqual({ viewMode: 'app', activeTab: 'history' });
    expect(getInitialRoute('?tab=unknown')).toEqual({ viewMode: 'landing', activeTab: 'field' });
    expect(getInitialRoute('')).toEqual({ viewMode: 'landing', activeTab: 'field' });
  });
});
