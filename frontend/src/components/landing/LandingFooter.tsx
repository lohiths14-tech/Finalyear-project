import { ArrowUpRight, Leaf } from 'lucide-react';

interface LandingFooterProps {
  onEnterFarmer: () => void;
  onEnterExpert: () => void;
}

export default function LandingFooter({ onEnterFarmer, onEnterExpert }: LandingFooterProps) {
  return (
    <footer className="landing-footer">
      <div className="landing-footer__main">
        <div>
          <a className="landing-brand landing-brand--footer" href="#top" aria-label="AgriPulse, back to top">
            <span className="landing-brand__mark" aria-hidden="true"><Leaf size={18} /></span>
            <span className="landing-brand__name">AgriPulse<span className="landing-brand__period">.</span></span>
          </a>
          <p>Crop imagery, considered in context.</p>
        </div>
        <div className="landing-footer__links" aria-label="Get started">
          <button type="button" onClick={onEnterFarmer}>Farmer experience</button>
          <button type="button" onClick={onEnterExpert}>Expert dashboard <ArrowUpRight size={14} aria-hidden="true" /></button>
        </div>
      </div>
      <div className="landing-footer__bottom">
        <span>AgriPulse</span>
        <a href="#top">Back to top ↑</a>
      </div>
    </footer>
  );
}
