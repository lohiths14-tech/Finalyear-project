import type React from 'react';
import { Calculator, Info } from 'lucide-react';

interface EconomicImpactWidgetProps {
  diseaseType: string;
  severity: number;
  confidence: number;
  cropType?: string;
}

export const EconomicImpactWidget: React.FC<EconomicImpactWidgetProps> = ({
  diseaseType,
  severity,
  confidence,
  cropType,
}) => (
  <section className="scenario-notice economic-impact-widget" aria-labelledby="economic-impact-title">
    <div className="scenario-notice__heading">
      <span className="scenario-notice__icon"><Calculator size={17} /></span>
      <div>
        <h4 id="economic-impact-title">Farm economics not estimated</h4>
        <span className="scenario-notice__tag">No verified financial inputs</span>
      </div>
    </div>
    <p>
      This scan does not include measured field area, expected yield, local crop price, treatment cost,
      or treatment response. A financial loss or return estimate would be speculative, so none is shown.
    </p>
    <div className="scenario-notice__context">
      <span>Scan context: {cropType || 'crop not specified'} · {diseaseType} · image severity {severity.toFixed(1)}%</span>
      <span>Model confidence ({confidence.toFixed(1)}%) is not a probability of yield loss or treatment success.</span>
    </div>
    <div className="scenario-notice__footnote">
      <Info size={15} aria-hidden="true" />
      <span>Use verified farm records and local agronomic guidance for economic decisions.</span>
    </div>
  </section>
);

export default EconomicImpactWidget;
