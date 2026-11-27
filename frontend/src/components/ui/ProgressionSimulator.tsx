import type React from 'react';
import { useState } from 'react';
import { Clock, Info, TrendingDown, TrendingUp } from 'lucide-react';

interface ProgressionSimulatorProps {
  initialSeverity: number;
  diseaseType: string;
}

export const ProgressionSimulator: React.FC<ProgressionSimulatorProps> = ({
  initialSeverity,
  diseaseType,
}) => {
  const [day, setDay] = useState(0);
  const isHealthy = diseaseType.toLowerCase().includes('healthy');

  if (isHealthy) return null;

  const baselineSeverity = Math.max(0, Math.min(100, initialSeverity));
  const risingScenario = Math.min(
    95,
    Math.round(baselineSeverity + (100 - baselineSeverity) * (1 - Math.exp(-0.16 * day)))
  );
  const decliningScenario = Math.max(
    10,
    Math.round(day <= 2
      ? baselineSeverity + day * 1.5
      : Math.max(12, baselineSeverity * Math.exp(-0.08 * (day - 2))))
  );

  return (
    <section className="scenario-panel" aria-labelledby="progression-title">
      <header className="scenario-panel__header">
        <span className="scenario-panel__icon"><Clock size={17} /></span>
        <div>
          <h4 id="progression-title">Illustrative severity scenarios</h4>
          <p>No historical observations are used. These are mathematical examples, not forecasts or treatment-efficacy estimates.</p>
        </div>
        <span className="scenario-panel__day" aria-live="polite">Day {day} of 14</span>
      </header>

      <label className="scenario-panel__slider-label" htmlFor="progression-day">
        Explore day {day} <span>Scenario day (0–14)</span>
      </label>
      <input
        id="progression-day"
        type="range"
        min="0"
        max="14"
        value={day}
        onChange={(event) => setDay(Number(event.target.value))}
        aria-valuetext={`Day ${day} in an illustrative scenario`}
        className="scenario-panel__slider"
      />

      <div className="scenario-panel__assumptions">
        <div><strong>Rising assumption</strong><span>Moves toward 100%, capped at 95%, by adding 16% of the remaining range per day.</span></div>
        <div><strong>Declining assumption</strong><span>Rises 1.5 points per day for two days, then decays by 8% per day, with a 10% floor.</span></div>
      </div>

      <div className="scenario-panel__scenarios" aria-live="polite">
        <ScenarioCard
          direction="rising"
          label="Rising scenario"
          value={risingScenario}
          description="Illustrative increase based on the assumption above."
        />
        <ScenarioCard
          direction="declining"
          label="Declining scenario"
          value={decliningScenario}
          description="Illustrative decrease based on the assumption above."
        />
      </div>

      <p className="scenario-panel__note"><Info size={14} />The curves are not derived from this farm&apos;s history, weather, or a validated disease model.</p>
    </section>
  );
};

function ScenarioCard({
  direction,
  label,
  value,
  description,
}: {
  direction: 'rising' | 'declining';
  label: string;
  value: number;
  description: string;
}) {
  const Icon = direction === 'rising' ? TrendingUp : TrendingDown;

  return (
    <article className={`scenario-card scenario-card--${direction}`}>
      <div className="scenario-card__topline">
        <h5><Icon size={15} aria-hidden="true" />{label}</h5>
        <strong>{value}% severity</strong>
      </div>
      <div className="scenario-card__track" aria-hidden="true">
        <div style={{ width: `${value}%` }} />
      </div>
      <p>{description}</p>
    </article>
  );
}

export default ProgressionSimulator;
