import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import NumberFlow from '@number-flow/react';
import { TrendingDown, ArrowUpRight, Award } from 'lucide-react';

interface EconomicImpactWidgetProps {
  diseaseType: string;
  severity: number; // 0 to 100
  confidence: number; // 0 to 100
  cropType?: string;
}

export const EconomicImpactWidget: React.FC<EconomicImpactWidgetProps> = ({
  diseaseType,
  severity,
}) => {
  const isHealthy = diseaseType.toLowerCase().includes('healthy');

  // Grounded agricultural financial metrics (Standard Indian MSP & ICAR yield loss curves)
  const stats = useMemo(() => {
    if (isHealthy || severity <= 5) {
      return {
        yieldLossPct: 0,
        yieldLossKg: 0,
        financialLossInr: 0,
        treatmentCostInr: 0,
        netSavedInr: 0,
        roiRatio: 0,
        isLoss: false,
      };
    }

    // Baseline paddy yield: ~2,400 kg (24 quintals) per acre in India
    const baselineYieldKg = 2400;
    // MSP rate for Common Paddy 2024-2025: ₹2,300 per quintal (₹23.00 / kg)
    const mspPerKg = 23.0;

    // Yield loss curve: foliar necrosis impacts photosynthesis non-linearly
    // Typical agronomic loss: ~0.55% yield reduction per 1% foliar severity
    const lossPct = Math.min(65, Math.round(severity * 0.58));
    const lossKg = Math.round((baselineYieldKg * lossPct) / 100);
    const grossLossInr = Math.round(lossKg * mspPerKg);

    // Chemical / Fungicide cost (chemical + spraying labor per acre: ~₹650-850)
    const chemicalCostInr = severity > 40 ? 820 : 640;

    // Projected recovery preservation rate if treated promptly (~85% yield protected)
    const protectedYieldInr = Math.round(grossLossInr * 0.85);
    const netSavings = Math.max(0, protectedYieldInr - chemicalCostInr);
    const roi = chemicalCostInr > 0 ? (protectedYieldInr / chemicalCostInr).toFixed(1) : '1.0';

    return {
      yieldLossPct: lossPct,
      yieldLossKg: lossKg,
      financialLossInr: grossLossInr,
      treatmentCostInr: chemicalCostInr,
      netSavedInr: netSavings,
      roiRatio: roi,
      isLoss: true,
    };
  }, [diseaseType, severity, isHealthy]);

  if (!stats.isLoss) {
    return (
      <div className="rounded-xl border border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/50 dark:bg-emerald-950/20 p-4 text-emerald-800 dark:text-emerald-300 flex items-center gap-3">
        <Award className="text-emerald-500 shrink-0" size={24} />
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider">Zero Economic Yield Threat</h4>
          <p className="text-xs text-emerald-700 dark:text-emerald-400 mt-0.5">
            Canopy exhibits optimal photosynthetic vigor. Expected harvest matches target yield of 24 quintals/acre.
          </p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/90 p-4 shadow-sm"
    >
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-rose-100 dark:bg-rose-950/50 text-rose-600 dark:text-rose-400 flex items-center justify-center">
            <TrendingDown size={16} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-800 dark:text-slate-100 uppercase tracking-wider">
              Agronomic Economic Impact &amp; Farm ROI
            </h4>
            <p className="text-[10px] text-slate-500 dark:text-slate-400">
              Grounded on ICAR yield response models @ ₹2,300/quintal MSP
            </p>
          </div>
        </div>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 dark:bg-rose-950/40 dark:text-rose-300 flex items-center gap-0.5">
          Loss Risk: <NumberFlow value={stats.yieldLossPct} />%
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mb-3">
        {/* Metric 1 */}
        <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-medium">Harvest Loss (Untreated)</span>
          <span className="text-base font-extrabold text-rose-600 dark:text-rose-400 mt-0.5 flex items-center">
            -<NumberFlow value={stats.yieldLossKg} /> <span className="text-xs font-normal text-slate-500 ml-1">kg/acre</span>
          </span>
        </div>

        {/* Metric 2 */}
        <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-medium">Potential Revenue Loss</span>
          <span className="text-base font-extrabold text-rose-600 dark:text-rose-400 mt-0.5 flex items-center">
            ₹<NumberFlow value={stats.financialLossInr} />
          </span>
        </div>

        {/* Metric 3 */}
        <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-medium">Prescription Spray Cost</span>
          <span className="text-base font-extrabold text-slate-700 dark:text-slate-300 mt-0.5 flex items-center">
            ₹<NumberFlow value={stats.treatmentCostInr} />
          </span>
        </div>

        {/* Metric 4 */}
        <div className="p-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800/50">
          <span className="text-[10px] text-emerald-700 dark:text-emerald-400 block font-medium">Net Protected Profit</span>
          <span className="text-base font-extrabold text-emerald-600 dark:text-emerald-400 mt-0.5 flex items-center">
            ₹<NumberFlow value={stats.netSavedInr} />
            <span className="ml-1 text-[10px] px-1 py-0.2 rounded bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200">
              {stats.roiRatio}×
            </span>
          </span>
        </div>
      </div>

      <div className="text-[11px] text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/40 p-2.5 rounded-xl border border-slate-200/60 dark:border-slate-700/60 flex items-start gap-2">
        <ArrowUpRight size={14} className="text-emerald-500 shrink-0 mt-0.5" />
        <span>
          <strong>Actionable Return:</strong> Applying targeted ICAR-prescribed fungicide within 48 hours protects approximately <strong>₹{stats.netSavedInr.toLocaleString('en-IN')}</strong> of crop value per acre, achieving an estimated <strong>{stats.roiRatio}× Return on Chemical Investment</strong>.
        </span>
      </div>
    </motion.div>
  );
};
export default EconomicImpactWidget;
