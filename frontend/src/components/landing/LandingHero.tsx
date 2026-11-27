import { ArrowDown, ArrowRight, Leaf } from 'lucide-react';

interface LandingHeroProps {
  onExplore: () => void;
  onEnterExpert: () => void;
}

export default function LandingHero({ onExplore, onEnterExpert }: LandingHeroProps) {
  return (
    <section className="landing-hero" aria-labelledby="landing-title">
      <img
        className="landing-hero__image"
        src="/images/agripulse-field-hero.png"
        alt="Morning light over rows of young rice growing in a green paddy"
        fetchPriority="high"
      />
      <div className="landing-hero__shade" aria-hidden="true" />
      <div className="landing-hero__content">
        <div className="landing-eyebrow landing-eyebrow--light"><Leaf size={14} aria-hidden="true" /> Crop imagery, thoughtfully interpreted</div>
        <h1 id="landing-title">A clearer view<br />of what&apos;s growing.</h1>
        <p className="landing-hero__description">
          AgriPulse brings crop-image analysis, field-level views, and practical workflows together—so growers and teams can make sense of what they see.
        </p>
        <div className="landing-hero__actions">
          <button className="landing-button landing-button--primary" type="button" onClick={onExplore}>
            Explore the platform <ArrowDown size={16} aria-hidden="true" />
          </button>
          <button className="landing-button landing-button--quiet" type="button" onClick={onEnterExpert}>
            Open dashboard <ArrowRight size={15} aria-hidden="true" />
          </button>
        </div>
      </div>
      <div className="landing-hero__caption">
        <span className="landing-hero__caption-line" aria-hidden="true" />
        <span>Rice field · illustrative imagery</span>
      </div>
      <a className="landing-hero__scroll" href="#platform" aria-label="Scroll to platform overview">
        <span>Scroll to explore</span><ArrowDown size={14} aria-hidden="true" />
      </a>
    </section>
  );
}
