import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, AlertTriangle, ShieldCheck, Play, ArrowRight } from 'lucide-react';

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

  if (isHealthy) {
    return null;
  }

  // Untreated: exponential foliar fungal spore spread (sigmoid growth)
  const untreatedSeverity = Math.min(
    95,
    Math.round(initialSeverity + (100 - initialSeverity) * (1 - Math.exp(-0.16 * day)))
  );

  // Treated: arrested at day 2-3, containment & slight recovery
  const treatedSeverity = Math.max(
    10,
    Math.round(
      day <= 2
        ? initialSeverity + day * 1.5
        : Math.max(12, initialSeverity * Math.exp(-0.08 * (day - 2)))
    )
  );

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/90 p-4 shadow-sm mt-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
            <Clock size={16} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-800 dark:text-slate-100 uppercase tracking-wider">
              14-Day Dynamic "What-If" Progression Simulator
            </h4>
            <p className="text-[10px] text-slate-500 dark:text-slate-400">
              Epidemiological modeling of foliar pathogen spread: Treated vs Untreated
            </p>
          </div>
        </div>
        <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/40 px-2 py-0.5 rounded-full border border-indigo-200 dark:border-indigo-800">
          Day {day} of 14
        </span>
      </div>

      {/* Slider Controls */}
      <div className="mb-4">
        <div className="flex justify-between text-[11px] font-medium text-slate-500 mb-1">
          <span>Day 0 (Initial Scan)</span>
          <span>Day 5</span>
          <span>Day 10</span>
          <span>Day 14 (Full Cycle)</span>
        </div>
        <input
          type="range"
          min="0"
          max="14"
          value={day}
          onChange={(e) => setDay(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
        />
      </div>

      {/* Trajectory Comparison Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Untreated Path */}
        <div className="p-3 rounded-xl border border-rose-200 dark:border-rose-900/40 bg-rose-50/40 dark:bg-rose-950/20">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-bold text-rose-800 dark:text-rose-300 flex items-center gap-1.5">
              <AlertTriangle size={14} className="text-rose-600" />
              Untreated Trajectory
            </span>
            <span className="text-xs font-extrabold text-rose-600 dark:text-rose-400">
              {untreatedSeverity}% Severity
            </span>
          </div>
          <div className="w-full bg-rose-200/60 dark:bg-rose-900/50 rounded-full h-2 overflow-hidden mb-2">
            <motion.div
              className="bg-rose-600 h-full rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${untreatedSeverity}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-[10px] text-rose-700 dark:text-rose-400">
            {day === 0
              ? 'Current state without intervention.'
              : day < 7
              ? 'Pathogen expands radially across adjacent canopy patches.'
              : 'Severe leaf senescence, panicle lodging risk, and irreversible harvest drop.'}
          </p>
        </div>

        {/* Treated Path */}
        <div className="p-3 rounded-xl border border-emerald-200 dark:border-emerald-900/40 bg-emerald-50/40 dark:bg-emerald-950/20">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-bold text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
              <ShieldCheck size={14} className="text-emerald-600" />
              Prompt ICAR Treatment
            </span>
            <span className="text-xs font-extrabold text-emerald-600 dark:text-emerald-400">
              {treatedSeverity}% Severity
            </span>
          </div>
          <div className="w-full bg-emerald-200/60 dark:bg-emerald-900/50 rounded-full h-2 overflow-hidden mb-2">
            <motion.div
              className="bg-emerald-600 h-full rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${treatedSeverity}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-[10px] text-emerald-700 dark:text-emerald-400">
            {day === 0
              ? 'Fungicide applied within 48h window.'
              : day < 7
              ? 'Spore sporulation halted; necrotic perimeter quarantined.'
              : 'New tillers emerge; photosynthetic efficiency preserved at 92%.'}
          </p>
        </div>
      </div>
    </div>
  );
};
export default ProgressionSimulator;
