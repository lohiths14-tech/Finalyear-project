import React, { useRef, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, X, RefreshCw, Zap, AlertCircle, Upload } from 'lucide-react';

interface LiveCameraScannerProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

export const LiveCameraScanner: React.FC<LiveCameraScannerProps> = ({ isOpen, onClose, onCapture }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeCameraInputRef = useRef<HTMLInputElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [capturing, setCapturing] = useState(false);

  const stopActiveStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setStream(null);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    stopActiveStream();

    if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const insecureNote =
        typeof window !== 'undefined' && !window.isSecureContext
          ? 'Live browser viewfinder requires HTTPS or localhost. Tap "Open Device Camera" below to capture directly on mobile.'
          : 'Live browser camera API is unavailable here. Use "Open Device Camera" below to capture or upload a photo.';
      setError(insecureNote);
      return;
    }

    try {
      let mediaStream: MediaStream;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
      } catch (constraintErr: any) {
        if (constraintErr?.name === 'OverconstrainedError' || constraintErr?.name === 'NotReadableError') {
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } else {
          throw constraintErr;
        }
      }

      streamRef.current = mediaStream;
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err: any) {
      console.error('Camera access error:', err);
      const name = err?.name || '';
      if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setError('No camera hardware detected on this device. Use "Open Device Camera / Upload Photo" below.');
      } else if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setError('Camera permission denied in browser. Allow camera access or use "Open Device Camera" below.');
      } else {
        setError('Could not start live viewfinder. Use "Open Device Camera / Upload Photo" below.');
      }
    }
  }, [facingMode, stopActiveStream]);

  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      stopActiveStream();
    }

    return () => {
      stopActiveStream();
    };
  }, [isOpen, startCamera, stopActiveStream]);

  const handleNativeFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file) {
      onCapture(file);
      onClose();
    }
  };

  const toggleFacing = () => {
    setFacingMode((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  const handleCapture = () => {
    if (!videoRef.current || capturing) return;
    setCapturing(true);

    try {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File([blob], `camera_scan_${Date.now()}.jpg`, { type: 'image/jpeg' });
            onCapture(file);
            onClose();
          }
          setCapturing(false);
        },
        'image/jpeg',
        0.88
      );
    } catch (err) {
      console.error('Capture error:', err);
      setCapturing(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-950/80 backdrop-blur-md">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative w-full max-w-lg rounded-2xl overflow-hidden bg-slate-900 border border-slate-700 shadow-2xl flex flex-col"
          >
            {/* Header bar */}
            <div className="flex items-center justify-between p-3.5 bg-slate-900/90 border-b border-slate-800 text-white z-10">
              <div className="flex items-center gap-2">
                <Camera size={18} className="text-emerald-400" />
                <span className="text-xs font-bold uppercase tracking-wider">Holographic Field Scanner</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={toggleFacing}
                  className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
                  title="Switch Camera"
                >
                  <RefreshCw size={15} />
                </button>
                <button
                  type="button"
                  onClick={onClose}
                  className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
                >
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* Native OS Camera / Photo Fallback Input */}
            <input
              ref={nativeCameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleNativeFileChange}
              className="hidden"
            />

            {/* Viewfinder Viewport */}
            <div className="relative w-full aspect-[4/3] bg-black flex items-center justify-center overflow-hidden">
              {error ? (
                <div className="p-5 sm:p-6 text-center text-rose-300 text-xs flex flex-col items-center gap-3 max-w-sm">
                  <AlertCircle size={26} className="text-rose-400 shrink-0" />
                  <span className="leading-relaxed">{error}</span>
                  <div className="flex flex-wrap items-center justify-center gap-2 mt-1">
                    <button
                      type="button"
                      onClick={() => nativeCameraInputRef.current?.click()}
                      className="px-3.5 py-2 bg-emerald-500 hover:bg-emerald-400 text-white font-semibold rounded-lg text-xs flex items-center gap-1.5 shadow-md shadow-emerald-500/20 transition-colors"
                    >
                      <Camera size={14} />
                      Open Device Camera
                    </button>
                    <button
                      type="button"
                      onClick={startCamera}
                      className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs transition-colors"
                    >
                      Retry Access
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {/* Holographic HUD Targeting Reticle */}
                  <div className="absolute inset-6 sm:inset-8 pointer-events-none border border-emerald-500/40 rounded-xl flex items-center justify-center">
                    {/* Corner Accents */}
                    <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-emerald-400" />
                    <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-emerald-400" />
                    <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-emerald-400" />
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-emerald-400" />

                    {/* Center Crosshair */}
                    <div className="w-8 h-8 rounded-full border border-dashed border-emerald-400/70 flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                    </div>

                    <div className="absolute bottom-2 text-[10px] font-mono tracking-wider text-emerald-300 bg-slate-900/60 px-2 py-0.5 rounded backdrop-blur-xs">
                      ALIGN CROP FOLIAGE IN RETICLE
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Shutter Controls */}
            <div className="p-3.5 sm:p-4 bg-slate-900 flex items-center justify-center gap-4 border-t border-slate-800">
              <button
                type="button"
                onClick={() => nativeCameraInputRef.current?.click()}
                className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition-colors"
                title="Capture with Device Camera or Upload Photo"
              >
                <Upload size={14} />
                <span>Device Cam / Photo</span>
              </button>
              <button
                type="button"
                disabled={!!error || !stream || capturing}
                onClick={handleCapture}
                aria-label="Capture Frame"
                className="w-14 h-14 sm:w-16 sm:h-16 rounded-full border-4 border-white/20 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 transition-all duration-150 flex items-center justify-center shadow-lg shadow-emerald-500/30 text-white"
              >
                <Zap size={22} className={capturing ? 'animate-spin' : ''} />
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
export default LiveCameraScanner;
