import React, { useState, useEffect } from 'react';
import apiClient from '../../services/api';
import { Cpu, RefreshCw } from 'lucide-react';

interface HealthSentinelProps {
  onStatusChange?: (isOnline: boolean) => void;
}

export const HealthSentinel: React.FC<HealthSentinelProps> = ({ onStatusChange }) => {
  const [status, setStatus] = useState<'checking' | 'online' | 'waking' | 'offline'>('checking');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const checkHealth = async () => {
    const start = performance.now();
    try {
      await apiClient.get('/health', { timeout: 10000 });
      const elapsed = Math.round(performance.now() - start);
      setLatencyMs(elapsed);
      setStatus('online');
      if (onStatusChange) onStatusChange(true);
    } catch (err: any) {
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout') || err.response?.status === 502) {
        setStatus('waking');
      } else {
        setStatus('offline');
      }
      if (onStatusChange) onStatusChange(false);
    }
  };

  useEffect(() => {
    checkHealth();
    // Poll every 30s
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, []);

  return (
    <button
      type="button"
      onClick={checkHealth}
      title="Click to re-ping AI Core"
      aria-label={`AI Core ${status}${latencyMs !== null ? `, ${latencyMs} milliseconds` : ''}. Activate to recheck.`}
      className="health-sentinel inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer transition-all duration-200 border
        bg-white/80 dark:bg-slate-900/80 backdrop-blur-md shadow-xs
        border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700"
    >
      {status === 'online' && (
        <>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="health-sentinel__text text-emerald-700 dark:text-emerald-400 font-semibold flex items-center gap-1">
            <Cpu size={13} />
            AI Core: Ready
          </span>
          {latencyMs !== null && (
            <span className="health-sentinel__latency text-[10px] text-slate-400 font-mono">
              {latencyMs}ms
            </span>
          )}
        </>
      )}

      {status === 'waking' && (
        <>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
          </span>
          <span className="health-sentinel__text text-amber-700 dark:text-amber-400 font-medium animate-pulse flex items-center gap-1">
            <RefreshCw size={12} className="animate-spin" />
            Waking Engine (~20s)...
          </span>
        </>
      )}

      {(status === 'offline' || status === 'checking') && (
        <>
          <span className="inline-flex rounded-full h-2 w-2 bg-slate-400"></span>
          <span className="health-sentinel__text text-slate-500 dark:text-slate-400 font-medium">
            {status === 'checking' ? 'Connecting…' : 'AI Core: Standby'}
          </span>
        </>
      )}
    </button>
  );
};
export default HealthSentinel;
