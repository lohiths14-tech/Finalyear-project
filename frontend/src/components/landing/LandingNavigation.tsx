import { useState } from 'react';
import { ArrowUpRight, Leaf, Menu, X } from 'lucide-react';

interface LandingNavigationProps {
  backendOnline: boolean | null;
  onEnterExpert: () => void;
}

const links = [
  { href: '#platform', label: 'Platform' },
  { href: '#technology', label: 'Technology' },
  { href: '#farmer-experience', label: 'For farmers' },
  { href: '#expert-experience', label: 'For teams' },
];

export default function LandingNavigation({ backendOnline, onEnterExpert }: LandingNavigationProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const statusLabel = backendOnline === null
    ? 'Checking service'
    : backendOnline
      ? 'Service reachable'
      : 'Service unavailable';

  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="landing-header">
      <div className="landing-header__inner">
        <a className="landing-brand" href="#top" aria-label="AgriPulse home" onClick={closeMenu}>
          <span className="landing-brand__mark" aria-hidden="true"><Leaf size={19} strokeWidth={2} /></span>
          <span className="landing-brand__name">AgriPulse<span className="landing-brand__period">.</span></span>
        </a>

        <nav className="landing-nav" aria-label="Main navigation">
          {links.map((link) => <a key={link.href} href={link.href}>{link.label}</a>)}
        </nav>

        <div className="landing-header__actions">
          <span className="landing-service" role="status" aria-live="polite">
            <span className={`landing-service__dot${backendOnline === true ? ' is-online' : backendOnline === false ? ' is-offline' : ''}`} aria-hidden="true" />
            {statusLabel}
          </span>
          <button className="landing-button landing-button--header" type="button" onClick={onEnterExpert}>
            Open dashboard <ArrowUpRight size={15} aria-hidden="true" />
          </button>
          <button
            className="landing-menu-toggle"
            type="button"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-controls="landing-mobile-menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <X size={20} aria-hidden="true" /> : <Menu size={20} aria-hidden="true" />}
          </button>
        </div>
      </div>

      <div className="landing-mobile-menu" id="landing-mobile-menu" hidden={!menuOpen}>
        <nav aria-label="Mobile navigation">
          {links.map((link) => (
            <a key={link.href} href={link.href} onClick={closeMenu}>{link.label}</a>
          ))}
          <button className="landing-button landing-button--mobile" type="button" onClick={() => { closeMenu(); onEnterExpert(); }}>
            Open dashboard <ArrowUpRight size={15} aria-hidden="true" />
          </button>
        </nav>
      </div>
    </header>
  );
}
