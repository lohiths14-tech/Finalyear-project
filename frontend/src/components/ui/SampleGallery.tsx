import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, ShieldCheck, AlertTriangle, AlertCircle, HelpCircle } from 'lucide-react';

export interface BenchmarkSample {
  id: string;
  title: string;
  pathogen: string;
  badge: string;
  badgeColor: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  imagePath: string;
  description: string;
}

export const BENCHMARK_SAMPLES: BenchmarkSample[] = [
  {
    id: 'healthy',
    title: 'Healthy Foliage',
    pathogen: 'Clean Baseline',
    badge: 'Negative Control',
    badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 border-emerald-300',
    icon: ShieldCheck,
    imagePath: '/samples/sample_healthy.jpg',
    description: 'Verifies model does not produce false positives on normal crops.',
  },
  {
    id: 'blast',
    title: 'Rice Blast',
    pathogen: 'Magnaporthe oryzae',
    badge: 'High Severity',
    badgeColor: 'bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300 border-rose-300',
    icon: AlertCircle,
    imagePath: '/samples/sample_blast.jpg',
    description: 'Spindle-shaped necrotizing lesions with chlorotic halos.',
  },
  {
    id: 'brown_spot',
    title: 'Brown Spot',
    pathogen: 'Bipolaris oryzae',
    badge: 'Fungal Lesion',
    badgeColor: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 border-amber-300',
    icon: AlertTriangle,
    imagePath: '/samples/sample_brown_spot.jpg',
    description: 'Circular brown necrotic spots testing discrete pathology.',
  },
  {
    id: 'ood_trap',
    title: 'Non-Crop Trap',
    pathogen: 'Out-Of-Distribution',
    badge: 'OOD Rejection',
    badgeColor: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300 border-indigo-300',
    icon: HelpCircle,
    imagePath: '/samples/sample_ood_trap.jpg',
    description: 'Demonstrates calibrated AI rejection on out-of-domain objects.',
  },
];

interface SampleGalleryProps {
  onSelectSample: (file: File) => void;
  disabled?: boolean;
}

export const SampleGallery: React.FC<SampleGalleryProps> = ({ onSelectSample, disabled }) => {
  const [loadingId, setLoadingId] = useState<string | null>(null);

  const handleClick = async (sample: BenchmarkSample) => {
    if (disabled || loadingId) return;
    setLoadingId(sample.id);
    try {
      const response = await fetch(sample.imagePath);
      const blob = await response.blob();
      const filename = `${sample.id}_benchmark.jpg`;
      const file = new File([blob], filename, { type: 'image/jpeg', lastModified: Date.now() });
      onSelectSample(file);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <div className="w-full mt-4">
      <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
        <Sparkles size={14} className="text-amber-500 animate-pulse" />
        <span>Live Examiner Benchmark Samples (1-Click Test)</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
        {BENCHMARK_SAMPLES.map((sample) => {
          const Icon = sample.icon;
          const isLoading = loadingId === sample.id;

          return (
            <motion.button
              key={sample.id}
              type="button"
              disabled={disabled || !!loadingId}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleClick(sample)}
              className={`group text-left relative overflow-hidden rounded-xl border p-2.5 transition-all duration-200
                bg-white/80 dark:bg-slate-900/70 backdrop-blur-sm
                border-slate-200 dark:border-slate-800
                hover:border-emerald-500 dark:hover:border-emerald-500 hover:shadow-md
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              {/* Miniature thumbnail preview with badge overlay */}
              <div className="relative w-full h-20 rounded-lg overflow-hidden mb-2 bg-slate-100 dark:bg-slate-800">
                <img
                  src={sample.imagePath}
                  alt={sample.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
                <span className={`absolute top-1 right-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full border shadow-sm backdrop-blur-md ${sample.badgeColor}`}>
                  {sample.badge}
                </span>
                {isLoading && (
                  <div className="absolute inset-0 bg-black/40 backdrop-blur-xs flex items-center justify-center text-white text-xs font-medium">
                    Loading…
                  </div>
                )}
              </div>

              {/* Title & subtitle */}
              <div className="flex items-center gap-1.5">
                <Icon size={14} className="shrink-0 text-slate-600 dark:text-slate-400 group-hover:text-emerald-500" />
                <span className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate">
                  {sample.title}
                </span>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 italic truncate mt-0.5">
                {sample.pathogen}
              </p>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};
export default SampleGallery;
