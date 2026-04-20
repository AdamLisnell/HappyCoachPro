import { useState, useEffect, useCallback, useRef } from 'react';
import { Camera, Circle, Square, RotateCcw, Zap, ZapOff } from 'lucide-react';
import { useCamera } from '@/hooks/useCamera';
import { SkeletonOverlay } from '@/components/analysis/SkeletonOverlay';
import * as poseDetector from '@/lib/poseDetector';
import type { PoseFrame } from '@/types';

export function RecordPage() {
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [contentArea, setContentArea] = useState({ width: 640, height: 480, offsetX: 0, offsetY: 0 });
  const [currentPose, setCurrentPose] = useState<PoseFrame | null>(null);
  const [detectorReady, setDetectorReady] = useState(false);
  const [fps, setFps] = useState(0);

  const frameNumberRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const fpsCountRef = useRef(0);
  const fpsTimerRef = useRef(0);

  const {
    videoRef,
    canvasRef,
    isStreaming,
    isRecording,
    error: cameraError,
    startCamera,
    stopCamera,
    startRecording,
    stopRecording,
    switchCamera,
  } = useCamera({ facingMode: 'environment' });

  // Initialize MediaPipe eagerly
  useEffect(() => {
    poseDetector.initialize().then(() => setDetectorReady(true)).catch(console.error);
  }, []);

  // Compute actual video content area (accounts for object-contain letterboxing)
  useEffect(() => {
    const update = () => {
      const video = videoRef.current;
      if (!video) return;
      const rect = video.getBoundingClientRect();
      const elW = rect.width;
      const elH = rect.height;
      if (elW === 0 || elH === 0) return;
      const intrinsicW = video.videoWidth || elW;
      const intrinsicH = video.videoHeight || elH;
      const videoAspect = intrinsicW / intrinsicH;
      const elementAspect = elW / elH;
      let contentW: number, contentH: number, offsetX: number, offsetY: number;
      if (videoAspect > elementAspect) {
        contentW = elW; contentH = elW / videoAspect;
        offsetX = 0; offsetY = (elH - contentH) / 2;
      } else {
        contentH = elH; contentW = elH * videoAspect;
        offsetX = (elW - contentW) / 2; offsetY = 0;
      }
      setContentArea({ width: contentW, height: contentH, offsetX, offsetY });
    };
    const video = videoRef.current;
    video?.addEventListener('loadedmetadata', update);
    video?.addEventListener('resize', update);
    window.addEventListener('resize', update);
    update();
    const id = setInterval(update, 500);
    return () => {
      video?.removeEventListener('loadedmetadata', update);
      video?.removeEventListener('resize', update);
      window.removeEventListener('resize', update);
      clearInterval(id);
    };
  }, [isStreaming, videoRef]);

  // Pose detection loop — runs directly in browser, no backend needed
  useEffect(() => {
    if (!isStreaming || !detectorReady || !showSkeleton) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const loop = () => {
      const now = performance.now();
      // ~20fps cap to leave CPU headroom on phone
      if (now - lastFrameTimeRef.current >= 50 && videoRef.current) {
        const pose = poseDetector.detectForVideo(
          videoRef.current,
          frameNumberRef.current++,
          now,
        );
        if (pose) setCurrentPose(pose);
        lastFrameTimeRef.current = now;

        // FPS counter
        fpsCountRef.current++;
        if (now - fpsTimerRef.current >= 1000) {
          setFps(fpsCountRef.current);
          fpsCountRef.current = 0;
          fpsTimerRef.current = now;
        }
      }
      animationFrameRef.current = requestAnimationFrame(loop);
    };

    loop();
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isStreaming, detectorReady, showSkeleton, videoRef]);

  const handleCameraToggle = useCallback(async () => {
    if (isStreaming) {
      stopCamera();
      setCurrentPose(null);
    } else {
      await startCamera();
    }
  }, [isStreaming, startCamera, stopCamera]);

  const handleRecordToggle = useCallback(async () => {
    if (isRecording) {
      const blob = await stopRecording();
      if (blob) {
        // Download the recorded video so user can upload to Analyze tab
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `swing-${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const error = cameraError;
  const active = isStreaming && detectorReady && showSkeleton;

  return (
    <div className="min-h-screen bg-[var(--color-surface)] flex flex-col">
      {/* Header */}
      <header className="bg-[var(--color-primary)] px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-card)] flex items-center justify-center border border-[var(--color-accent)]/30">
            <Camera className="w-5 h-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-[var(--color-text)]">HappyCoach</h1>
            <p className="text-xs text-[var(--color-accent)] uppercase tracking-wider">Swing Analyzer</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {active ? (
            <div className="flex items-center gap-1.5 text-green-400">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-mono">{fps}fps</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-gray-400">
              <ZapOff className="w-4 h-4" />
              <span className="text-xs">{detectorReady ? 'Ready' : 'Loading…'}</span>
            </div>
          )}
        </div>
      </header>

      {/* Video */}
      <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
        <video ref={videoRef} className="max-w-full max-h-full object-contain" playsInline muted />
        <canvas ref={canvasRef} className="hidden" />

        {showSkeleton && currentPose && isStreaming && (
          <div
            className="absolute pointer-events-none overflow-hidden"
            style={{
              left: contentArea.offsetX,
              top: contentArea.offsetY,
              width: contentArea.width,
              height: contentArea.height,
            }}
          >
            <SkeletonOverlay pose={currentPose} width={contentArea.width} height={contentArea.height} />
          </div>
        )}

        {isRecording && (
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/80 px-3 py-1.5 rounded-full">
            <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-sm font-medium">REC</span>
          </div>
        )}

        {error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-500/90 px-4 py-2 rounded-lg">
            <p className="text-white text-sm">{error}</p>
          </div>
        )}

        {!navigator.mediaDevices && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-primary-dark)]/95 p-6 text-center">
            <Camera className="w-12 h-12 text-yellow-400 mb-3" />
            <p className="text-[var(--color-text)] font-medium mb-2">Camera not available</p>
            <p className="text-[var(--color-text-muted)] text-sm">
              Please open this app in Safari (iOS 16.4+) or Chrome for camera access.
            </p>
          </div>
        )}

        {!isStreaming && navigator.mediaDevices && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-primary-dark)]/90">
            <Camera className="w-16 h-16 text-[var(--color-text-muted)] mb-4" />
            <p className="text-[var(--color-text-secondary)] mb-6">Camera is off</p>
            <button
              onClick={handleCameraToggle}
              className="bg-[var(--color-accent)] text-[var(--color-primary-dark)] px-6 py-3 rounded-xl font-semibold flex items-center gap-2 hover:bg-[var(--color-accent-light)] transition-colors"
            >
              <Camera className="w-5 h-5" />
              Start Camera
            </button>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="bg-[var(--color-primary)] px-4 py-6">
        <div className="flex items-center justify-center gap-6">
          <button
            onClick={switchCamera}
            disabled={!isStreaming}
            aria-label="Switch camera"
            className="w-12 h-12 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center border border-[var(--color-primary-light)] disabled:opacity-50"
          >
            <RotateCcw className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          <button
            onClick={handleRecordToggle}
            disabled={!isStreaming}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            className={`w-20 h-20 rounded-full flex items-center justify-center border-4 transition-all ${
              isRecording
                ? 'bg-red-500 border-red-400'
                : 'bg-[var(--color-surface-card)] border-[var(--color-accent)] hover:bg-[var(--color-accent)]/20'
            } disabled:opacity-50`}
          >
            {isRecording ? (
              <Square className="w-8 h-8 text-white fill-white" />
            ) : (
              <Circle className="w-10 h-10 text-[var(--color-accent)] fill-[var(--color-accent)]" />
            )}
          </button>

          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            aria-label="Toggle skeleton overlay"
            className={`w-12 h-12 rounded-full flex items-center justify-center border ${
              showSkeleton
                ? 'bg-[var(--color-accent)]/20 border-[var(--color-accent)]'
                : 'bg-[var(--color-surface-card)] border-[var(--color-primary-light)]'
            }`}
          >
            <svg
              className={`w-5 h-5 ${showSkeleton ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            >
              <circle cx="12" cy="4" r="2" />
              <line x1="12" y1="6" x2="12" y2="14" />
              <line x1="8" y1="8" x2="16" y2="8" />
              <line x1="12" y1="14" x2="8" y2="22" />
              <line x1="12" y1="14" x2="16" y2="22" />
            </svg>
          </button>
        </div>

        <p className="text-center text-[var(--color-text-muted)] text-sm mt-4">
          {!isStreaming
            ? 'Tap to start camera'
            : isRecording
            ? 'Recording… tap to stop and download'
            : 'Position yourself and tap record'}
        </p>
      </div>
    </div>
  );
}
