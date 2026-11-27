import { useEffect, useState } from 'react';
import apiClient from '../services/api';
import LandingNavigation from './landing/LandingNavigation';
import LandingHero from './landing/LandingHero';
import LandingSections from './landing/LandingSections';
import LandingFooter from './landing/LandingFooter';
import './landing/LandingPage.css';

interface LandingPageProps {
  onEnterExpert: () => void;
  onEnterFarmer: () => void;
}

export const LandingPage = ({ onEnterExpert, onEnterFarmer }: LandingPageProps) => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    apiClient.get('/health')
      .then((response) => {
        setBackendOnline(
          response.data?.status === 'healthy' || response.data?.status === 'degraded',
        );
      })
      .catch(() => setBackendOnline(false));
  }, []);

  return (
    <div className="landing-page" id="top">
      <a className="landing-skip-link" href="#main-content">Skip to content</a>
      <LandingNavigation backendOnline={backendOnline} onEnterExpert={onEnterExpert} />
      <main id="main-content">
        <LandingHero onExplore={() => document.getElementById('platform')?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' })} onEnterExpert={onEnterExpert} />
        <LandingSections onEnterFarmer={onEnterFarmer} onEnterExpert={onEnterExpert} />
      </main>
      <LandingFooter onEnterFarmer={onEnterFarmer} onEnterExpert={onEnterExpert} />
    </div>
  );
};

export default LandingPage;
