import React, { useRef, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, X, RefreshCw, Zap, AlertCircle } from 'lucide-react';

interface LiveCameraScannerProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

export const LiveCameraScanner: React.FC<LiveCameraScannerProps> = ({ isOpen, onClose, onCapture }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [capturing, setCapturing] = useState(false);

  const startCamera = useCallback(async () => {
    setError(null);
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Could not access camera. Please allow camera permissions in your browser.');
    }
  }, [facingMode]);

  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isOpen, facingMode]);

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

            {/* Viewfinder Viewport */}
            <div className="relative w-full aspect-4/3 bg-black flex items-center justify-center overflow-hidden">
              {error ? (
                <div className="p-6 text-center text-rose-400 text-xs flex flex-col items-center gap-2">
                  <AlertCircle size={24} />
                  <span>{error}</span>
                  <button
                    onClick={startCamera}
                    className="mt-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs"
                  >
                    Retry Access
                  </button>
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
                  <div className="absolute inset-8 pointer-events-none border border-emerald-500/40 rounded-xl flex items-center justify-center">
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
            <div className="p-4 bg-slate-900 flex items-center justify-center gap-4 border-t border-slate-800">
              <button
                type="button"
                disabled={!!error || capturing}
                onClick={handleCapture}
                className="w-16 h-16 rounded-full border-4 border-white/20 bg-emerald-500 hover:bg-emerald-400 active:scale-95 transition-all duration-150 flex items-center justify-center shadow-lg shadow-emerald-500/30 text-white"
              >
                <Zap size={24} className={capturing ? 'animate-spin' : ''} />
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
export default LiveCameraScanner;
