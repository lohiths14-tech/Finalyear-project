import { ArrowRight, AudioLines, CircleHelp, Eye, FileImage, Layers3, Map, ScanLine, ShieldCheck } from 'lucide-react';

interface LandingSectionsProps {
  onEnterFarmer: () => void;
  onEnterExpert: () => void;
}

const capabilities = [
  {
    number: '01',
    icon: FileImage,
    title: 'Start with an image',
    description: 'Choose a crop leaf or field image to begin an analysis in the existing AgriPulse workspace.',
  },
  {
    number: '02',
    icon: ScanLine,
    title: 'Review image analysis',
    description: 'See the classification and image-derived signals returned by the analysis service, with their context.',
  },
  {
    number: '03',
    icon: Layers3,
    title: 'Explore the field view',
    description: 'Inspect available spatial overlays, grid views, the source image, and GIS context in the expert workspace.',
  },
];

export default function LandingSections({ onEnterFarmer, onEnterExpert }: LandingSectionsProps) {
  return (
    <>
      <section className="landing-platform landing-section" id="platform" aria-labelledby="platform-title">
        <div className="landing-section__intro">
          <div className="landing-eyebrow"><span className="landing-eyebrow__rule" aria-hidden="true" /> A practical workflow</div>
          <h2 id="platform-title">From a single leaf<br />to the wider field.</h2>
          <p className="landing-section__lede">
            One platform, two ways in: a straightforward farmer experience and a deeper workspace for field analysis.
          </p>
        </div>

        <div className="landing-capabilities">
          {capabilities.map(({ number, icon: Icon, title, description }) => (
            <article className="landing-capability" key={number}>
              <div className="landing-capability__top">
                <span className="landing-capability__number">{number}</span>
                <Icon size={19} strokeWidth={1.7} aria-hidden="true" />
              </div>
              <h3>{title}</h3>
              <p>{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-imagery landing-section" aria-labelledby="imagery-title">
        <div className="landing-imagery__copy">
          <div className="landing-eyebrow"><span className="landing-eyebrow__rule" aria-hidden="true" /> Made for crop imagery</div>
          <h2 id="imagery-title">See the image.<br />Keep the context.</h2>
          <p>Leaf and field views are designed around the image being reviewed, rather than an abstract score on its own.</p>
          <a className="landing-text-link" href="#technology">How the views are presented <ArrowRight size={15} aria-hidden="true" /></a>
        </div>
        <div className="landing-imagery__samples">
          <figure className="landing-sample landing-sample--field">
            <img src="/images/agripulse-field-hero.png" alt="Illustrative rice paddy image; not a live field analysis" loading="lazy" />
            <figcaption><span>FIELD IMAGE</span><span>Illustrative</span></figcaption>
          </figure>
          <figure className="landing-sample landing-sample--leaf">
            <img src="/samples/sample_blast.jpg" alt="Example rice leaf image from the bundled sample set; not an analysis result" loading="lazy" />
            <figcaption><span>LEAF IMAGE</span><span>Example</span></figcaption>
          </figure>
        </div>
      </section>

      <section className="landing-transparency landing-section" id="technology" aria-labelledby="technology-title">
        <div className="landing-transparency__heading">
          <div>
            <div className="landing-eyebrow"><span className="landing-eyebrow__rule" aria-hidden="true" /> Interpretation, with context</div>
            <h2 id="technology-title">Useful views.<br />Clear boundaries.</h2>
          </div>
          <p>Image-derived features and reference examples are not the same as direct sensor measurements. AgriPulse labels the distinction in the analysis experience.</p>
        </div>
        <div className="landing-transparency__grid">
          <article className="landing-transparency-card">
            <span className="landing-transparency-card__icon"><Eye size={18} aria-hidden="true" /></span>
            <h3>Image-derived signals</h3>
            <p>Profiles and feature values can be derived from an uploaded image when returned by the analysis service.</p>
          </article>
          <article className="landing-transparency-card">
            <span className="landing-transparency-card__icon"><CircleHelp size={18} aria-hidden="true" /></span>
            <h3>Reference examples</h3>
            <p>Illustrative curves and proxy values are labeled as examples, not measurements, observations, or model output.</p>
          </article>
          <article className="landing-transparency-card">
            <span className="landing-transparency-card__icon"><Map size={18} aria-hidden="true" /></span>
            <h3>Spatial inspection</h3>
            <p>Field results can be reviewed using the available original-image, grid, overlay, and GIS views.</p>
          </article>
        </div>
      </section>

      <section className="landing-pathways landing-section" aria-labelledby="pathways-title">
        <div className="landing-section__intro landing-section__intro--center">
          <div className="landing-eyebrow"><span className="landing-eyebrow__rule" aria-hidden="true" /> Choose your starting point</div>
          <h2 id="pathways-title">Built around the people<br />working with the crop.</h2>
        </div>
        <div className="landing-pathways__grid">
          <article className="landing-pathway landing-pathway--farmer" id="farmer-experience">
            <span className="landing-pathway__index">01 / FARMER EXPERIENCE</span>
            <AudioLines size={22} strokeWidth={1.6} aria-hidden="true" />
            <h3>A simpler way to begin.</h3>
            <p>Open the farmer experience to explore a more guided route into crop analysis.</p>
            <button className="landing-button landing-button--dark" type="button" onClick={onEnterFarmer}>
              Open farmer experience <ArrowRight size={15} aria-hidden="true" />
            </button>
          </article>
          <article className="landing-pathway landing-pathway--expert" id="expert-experience">
            <span className="landing-pathway__index">02 / EXPERT WORKSPACE</span>
            <ShieldCheck size={22} strokeWidth={1.6} aria-hidden="true" />
            <h3>More room to inspect.</h3>
            <p>Move into the expert dashboard for field maps, analysis views, and scan history.</p>
            <button className="landing-button landing-button--light" type="button" onClick={onEnterExpert}>
              Open expert dashboard <ArrowRight size={15} aria-hidden="true" />
            </button>
          </article>
        </div>
      </section>
    </>
  );
}
