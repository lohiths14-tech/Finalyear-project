import React, { useState, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import apiClient from '../services/api';
import type { FieldResult, LeafResult, FarmerView, Language, OfflineQueueItem } from '../types';
import { I18N } from './farmer/i18n';
import { FarmerHomeView } from './farmer/FarmerHomeView';
import { FarmerResultView } from './farmer/FarmerResultView';
import { FarmerHistoryView } from './farmer/FarmerHistoryView';
import type { DosageCalcResult } from './farmer/FarmerDosageCalculator';

interface FarmerModeProps {
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
  backendOnline: boolean | null;
}

export { I18N };

export default function FarmerMode({ addToast, backendOnline }: FarmerModeProps) {
  const [view, setView] = useState<FarmerView>('home');
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<LeafResult | FieldResult | null>(null);
  const [scanType, setScanType] = useState<'leaf' | 'field' | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [landArea, setLandArea] = useState<number>(1.0);
  const [landUnit, setLandUnit] = useState<string>('acre');
  const [lang, setLang] = useState<Language>(() => {
    const saved = localStorage.getItem('agri_lang');
    return (saved as Language) || 'kn'; // Default to Kannada
  });
  const queueSyncLock = useRef(false);
  const [offlineQueue, setOfflineQueue] = useState<OfflineQueueItem[]>(() => {
    try {
      const saved = localStorage.getItem('agri_offline_queue');
      const parsed: unknown = saved ? JSON.parse(saved) : [];
      return Array.isArray(parsed) ? parsed as OfflineQueueItem[] : [];
    } catch {
      return [];
    }
  });

  const t = I18N[lang] || I18N.en;

  const handleLanguageChange = (newLang: Language) => {
    setLang(newLang);
    localStorage.setItem('agri_lang', newLang);
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  const stopVoice = useCallback(() => {
    try {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    } catch {
      // Ignore browser speech abort errors
    } finally {
      setIsSpeaking(false);
    }
  }, []);

  const playVoiceAdvisory = useCallback((textToSpeak: string) => {
    if (!('speechSynthesis' in window)) {
      addToast('Voice audio narration not supported on this browser.', 'warning');
      return;
    }
    try {
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      const langMap: Record<Language, string> = {
        kn: 'kn-IN',
        hi: 'hi-IN',
        te: 'te-IN',
        ta: 'ta-IN',
        en: 'en-IN',
      };
      utterance.lang = langMap[lang] || 'en-IN';
      utterance.rate = 0.9;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    } catch {
      setIsSpeaking(false);
    }
  }, [lang, addToast]);

  const saveToQueue = useCallback((file: File, type: 'leaf' | 'field') => {
    const reader = new FileReader();
    reader.onload = () => {
      let latestQueue: OfflineQueueItem[] = [];
      try {
        const saved = localStorage.getItem('agri_offline_queue');
        const parsed: unknown = saved ? JSON.parse(saved) : [];
        if (Array.isArray(parsed)) latestQueue = parsed as OfflineQueueItem[];
      } catch {
        latestQueue = offlineQueue;
      }
      const nextId = Math.max(Date.now(), ...latestQueue.map((queued) => queued.id + 1));
      const newItem: OfflineQueueItem = { id: nextId, type, name: file.name, data: reader.result as string };
      const newQueue = [...latestQueue, newItem];
      setOfflineQueue(newQueue);
      localStorage.setItem('agri_offline_queue', JSON.stringify(newQueue));
      addToast('Saved offline. We will scan when you have signal.', 'info');
    };
    reader.onerror = () => addToast('Could not save scan offline.', 'error', 4000);
    reader.readAsDataURL(file);
  }, [offlineQueue, addToast]);

  const syncQueue = useCallback(async () => {
    if (!backendOnline || offlineQueue.length === 0 || queueSyncLock.current) return;
    queueSyncLock.current = true;
    const itemsToSync = [...offlineQueue];
    let acknowledged = 0;
    addToast(`Syncing ${itemsToSync.length} offline scans...`, 'info');

    try {
      for (const item of itemsToSync) {
        try {
          const imageBlob = await fetch(item.data).then((response) => {
            if (!response.ok) throw new Error('Could not read saved scan');
            return response.blob();
          });
          const formData = new FormData();
          formData.append('file', new File([imageBlob], item.name, { type: imageBlob.type || 'image/jpeg' }));
          const response = await apiClient.post<LeafResult | FieldResult>(
            item.type === 'leaf' ? '/analyze/leaf' : '/analyze/field',
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' }, timeout: item.type === 'field' ? 300000 : 120000 }
          );
          const data = response.data;
          const validAcknowledgement = item.type === 'leaf'
            ? typeof (data as LeafResult).disease_type === 'string' && typeof (data as LeafResult).severity === 'number' && typeof (data as LeafResult).confidence === 'number'
            : typeof (data as FieldResult).disease_type === 'string' && typeof (data as FieldResult).average_severity === 'number' && typeof (data as FieldResult).infection_rate === 'number';
          if (!validAcknowledgement) throw new Error('The server response did not contain a valid scan result');

          const saved = localStorage.getItem('agri_offline_queue');
          const currentQueue: OfflineQueueItem[] = saved ? JSON.parse(saved) : [];
          const remainingQueue = currentQueue.filter((queued) => queued.id !== item.id);
          localStorage.setItem('agri_offline_queue', JSON.stringify(remainingQueue));
          setOfflineQueue(remainingQueue);
          acknowledged += 1;
        } catch {
          // Keep this item queued when delivery or the server acknowledgement is uncertain.
        }
      }

      const remaining = itemsToSync.length - acknowledged;
      if (remaining === 0) {
        addToast(`Successfully synced ${acknowledged} offline scans.`, 'success');
      } else {
        addToast(`${acknowledged} of ${itemsToSync.length} scans synced; ${remaining} remain queued.`, 'warning', 6000);
      }
    } finally {
      queueSyncLock.current = false;
    }
  }, [backendOnline, offlineQueue, addToast]);

  const resizeImage = useCallback((file: File): Promise<File> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target?.result as string;
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const MAX_WIDTH = 1024;
          let width = img.width;
          let height = img.height;

          if (width > MAX_WIDTH) {
            height *= MAX_WIDTH / width;
            width = MAX_WIDTH;
          }
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0, width, height);
          canvas.toBlob((blob) => {
            resolve(new File([blob as Blob], file.name, { type: 'image/jpeg', lastModified: Date.now() }));
          }, 'image/jpeg', 0.8);
        };
      };
    });
  }, []);

  const processFile = useCallback(async (originalFile: File, type: 'leaf' | 'field') => {
    setScanning(true);
    setScanType(type);

    if (backendOnline === false) {
      try {
        const file = await resizeImage(originalFile);
        saveToQueue(file, type);
      } catch {
        addToast('Could not save scan offline.', 'error', 4000);
      } finally {
        setScanning(false);
      }
      return;
    }

    try {
      const file = await resizeImage(originalFile);
      const fd = new FormData();
      fd.append('file', file);
      const endpoint = type === 'leaf' ? '/analyze/leaf' : '/analyze/field';
      const timeoutMs = type === 'field' ? 300000 : 120000;
      const { data } = await apiClient.post<LeafResult | FieldResult>(endpoint, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: timeoutMs,
      });
      setResult(data);
      setView('result');
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } }; message?: string; code?: string };
      const detail = axiosErr?.response?.data?.detail;
      const msg = detail || axiosErr?.message || 'Upload failed';
      addToast(`Scan failed: ${msg}`, 'error', 8000);
    } finally {
      setScanning(false);
    }
  }, [backendOnline, saveToQueue, resizeImage, addToast]);

  const handleScan = useCallback(async (e: React.ChangeEvent<HTMLInputElement>, type: 'leaf' | 'field') => {
    const originalFile = e.target.files?.[0];
    if (!originalFile) return;
    await processFile(originalFile, type);
    e.target.value = '';
  }, [processFile]);

  const dosageCalculations = useMemo<DosageCalcResult | null>(() => {
    if (!result || 'average_severity' in result) return null;
    const diseaseKey = Object.keys(t.diseases).find(k => k.toLowerCase() === result.disease_type.toLowerCase()) || "Healthy";
    const diseaseInfo = t.diseases[diseaseKey];

    const conversionToAcres: Record<string, number> = {
      acre: 1.0,
      guntha: 0.025,
      hectare: 2.471,
      bigha: 0.625,
      cent: 0.01,
    };

    const multiplier = conversionToAcres[landUnit] || 1.0;
    const acres = Math.max(0.01, landArea * multiplier);
    const waterLiters = Math.round(acres * 200);
    const tankPumps = Math.ceil(waterLiters / 16.0);
    const totalChem = Math.round(waterLiters * (diseaseInfo?.dosePerLiter || 0) * 10) / 10;
    const perTankChem = Math.round(16 * (diseaseInfo?.dosePerLiter || 0) * 10) / 10;

    return {
      acres: Math.round(acres * 100) / 100,
      waterLiters,
      tankPumps,
      totalChem,
      perTankChem,
      diseaseInfo,
    };
  }, [result, landArea, landUnit, t.diseases]);

  const handleShareWhatsApp = useCallback(() => {
    if (!result) return;
    const severity = 'average_severity' in result ? result.average_severity : result.severity;
    let text = `AgriPulse Crop Health Report\nDiagnosis: ${result.disease_type}\nSeverity: ${severity.toFixed(1)}%\nConfidence: ${result.confidence.toFixed(0)}%\nTreatment: ${result.treatment}`;

    if ('average_severity' in result) {
      text += `\nInfected area: ${result.infection_rate.toFixed(1)}%\nSpread risk: ${result.spread_risk}`;
    } else if (dosageCalculations) {
      const info = dosageCalculations.diseaseInfo;
      text += `\nLand size: ${landArea} ${landUnit} (${dosageCalculations.acres} acres)\nSpraying plan: ${dosageCalculations.tankPumps} knapsack tanks (16L each)\nChemical dose: ${dosageCalculations.totalChem} ${info.unit} ${info.chemName} (${dosageCalculations.perTankChem} ${info.unit}/tank)\nPre-harvest interval: ${info.phi} days\nFRAC Code: ${info.frac}\nAction: ${info.whatToDo}`;
    }

    window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank');
  }, [result, dosageCalculations, landArea, landUnit]);

  return (
    <div className="farmer-app-viewport" style={{ maxWidth: 560, margin: '0 auto', padding: '20px 16px 60px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, background: 'var(--bg-card)', padding: '10px 16px', borderRadius: 12, border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--text-primary)' }}>AgriPulse</span>
          <span style={{ fontSize: '0.68rem', color: 'var(--primary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', background: 'var(--primary-glow)', padding: '2px 8px', borderRadius: 99 }}>
            Farmer Mode · ರೈತ ಸೇವೆ
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.76rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
          <div className={`status-dot ${backendOnline ? 'green' : 'red'}`} />
          <span>{backendOnline ? 'Online' : 'Offline'}</span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {scanning ? (
          <motion.div key="scanning" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ textAlign: 'center', padding: '60px 20px', background: 'var(--bg-card)', borderRadius: 16, border: '1px solid var(--border)' }}>
            <div className="spinner" style={{ margin: '0 auto 20px', width: 48, height: 48 }} />
            <h3 className="farmer-label" style={{ fontSize: '1.2rem', marginBottom: 8, color: 'var(--text-primary)' }}>
              {t.analyzing}
            </h3>
            <p className="farmer-sublabel" style={{ marginBottom: 16, color: 'var(--text-secondary)' }}>
              {scanType === 'field' ? t.analyzingField : t.analyzingLeaf}
            </p>
            {scanType === 'field' && (
              <div style={{
                background: 'var(--warning-glow)',
                border: '1px solid rgba(217, 119, 6, 0.35)',
                borderRadius: 10,
                padding: '10px 14px',
                fontSize: '0.8rem',
                color: 'var(--warning)',
                maxWidth: 340,
                margin: '0 auto',
                lineHeight: 1.5,
              }}>
                {t.fieldCpuWait}
              </div>
            )}
          </motion.div>
        ) : (
          <>
            {view === 'home' && (
              <FarmerHomeView
                t={t}
                lang={lang}
                onLanguageChange={handleLanguageChange}
                offlineQueue={offlineQueue}
                onSyncQueue={syncQueue}
                onScan={handleScan}
                onSelectSample={(file) => processFile(file, 'leaf')}
                onNavigateHistory={() => setView('history')}
              />
            )}
            {view === 'result' && result && (
              <FarmerResultView
                result={result}
                t={t}
                onBack={() => { stopVoice(); setView('home'); }}
                isSpeaking={isSpeaking}
                onToggleVoice={() => {
                  if (isSpeaking) {
                    stopVoice();
                  } else {
                    const diseaseKey = Object.keys(t.diseases).find(k => k.toLowerCase() === result.disease_type.toLowerCase()) || "Healthy";
                    const loc = t.diseases[diseaseKey];
                    const spoken = `${loc?.name || result.disease_type}. ${loc?.whatIsIt || ''}. ${t.whatToDo}: ${loc?.whatToDo || result.treatment}`;
                    playVoiceAdvisory(spoken);
                  }
                }}
                landArea={landArea}
                landUnit={landUnit}
                onLandAreaChange={setLandArea}
                onLandUnitChange={setLandUnit}
                dosageCalculations={dosageCalculations}
                onShareWhatsApp={handleShareWhatsApp}
                onFeedback={(pos) => addToast(pos ? 'ಧನ್ಯವಾದಗಳು! (Thank you!)' : 'Feedback recorded.', pos ? 'success' : 'warning')}
              />
            )}
            {view === 'history' && (
              <FarmerHistoryView
                setView={setView as (v: FarmerView) => void}
                setResult={setResult}
                addToast={addToast}
                t={t}
              />
            )}
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
