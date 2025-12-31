/**
 * Record Page
 * 
 * Camera view with real-time pose detection and recording.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Camera, 
  Circle, 
  Square, 
  RotateCcw, 
  Wifi, 
  WifiOff,
} from 'lucide-react';
import { useCamera } from '@/hooks/useCamera';
import { useWebSocket } from '@/hooks/useWebSocket';
import { SkeletonOverlay } from '@/components/analysis/SkeletonOverlay';

export function RecordPage() {
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [overlaySize, setOverlaySize] = useState({ width: 640, height: 480 });
  const frameNumberRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  // Camera hook
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
    captureFrame,
    switchCamera,
  } = useCamera({ facingMode: 'environment' });

  // WebSocket hook
  const {
    isConnected,
    error: wsError,
    lastPose,
    processingTime,
    frameCount,
    connect,
    disconnect,
    sendFrame,
  } = useWebSocket('ws://localhost:8000/ws/pose');

  // Update overlay size to match the DISPLAYED video size
  useEffect(() => {
    const updateOverlaySize = () => {
      if (videoRef.current) {
        const video = videoRef.current;
        const rect = video.getBoundingClientRect();
        
        if (rect.width > 0 && rect.height > 0) {
          setOverlaySize({
            width: rect.width,
            height: rect.height,
          });
        }
      }
    };

    // Update on video metadata loaded
    const video = videoRef.current;
    if (video) {
      video.addEventListener('loadedmetadata', updateOverlaySize);
      video.addEventListener('resize', updateOverlaySize);
    }

    // Also update on window resize
    window.addEventListener('resize', updateOverlaySize);

    // Initial update
    updateOverlaySize();

    // Poll for size changes (backup)
    const interval = setInterval(updateOverlaySize, 500);

    return () => {
      if (video) {
        video.removeEventListener('loadedmetadata', updateOverlaySize);
        video.removeEventListener('resize', updateOverlaySize);
      }
      window.removeEventListener('resize', updateOverlaySize);
      clearInterval(interval);
    };
  }, [isStreaming, videoRef]);

  // Frame capture loop for real-time pose detection
  useEffect(() => {
    if (!isStreaming || !isConnected || !showSkeleton) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const captureLoop = () => {
      const frame = captureFrame();
      if (frame) {
        sendFrame(frame, frameNumberRef.current++);
      }
      // Capture at ~15fps for real-time detection
      animationFrameRef.current = requestAnimationFrame(() => {
        setTimeout(captureLoop, 66); // ~15fps
      });
    };

    captureLoop();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isStreaming, isConnected, showSkeleton, captureFrame, sendFrame]);

  // Handle start/stop camera
  const handleCameraToggle = useCallback(async () => {
    if (isStreaming) {
      stopCamera();
      disconnect();
    } else {
      await startCamera();
      connect();
    }
  }, [isStreaming, startCamera, stopCamera, connect, disconnect]);

  // Handle recording toggle
  const handleRecordToggle = useCallback(async () => {
    if (isRecording) {
      const blob = await stopRecording();
      if (blob) {
        // TODO: Handle recorded video
        console.log('Recorded video blob:', blob);
      }
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const error = cameraError || wsError;

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
            <p className="text-xs text-[var(--color-accent)] uppercase tracking-wider">
              Swing Analyzer
            </p>
          </div>
        </div>
        
        {/* Connection status */}
        <div className="flex items-center gap-2">
          {isConnected ? (
            <div className="flex items-center gap-1.5 text-green-400">
              <Wifi className="w-4 h-4" />
              <span className="text-xs">{processingTime.toFixed(0)}ms</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-gray-400">
              <WifiOff className="w-4 h-4" />
              <span className="text-xs">Offline</span>
            </div>
          )}
        </div>
      </header>

      {/* Video Area */}
      <div 
        ref={videoContainerRef}
        className="flex-1 relative bg-black flex items-center justify-center overflow-hidden"
      >
        {/* Video element */}
        <video
          ref={videoRef}
          className="max-w-full max-h-full object-contain"
          playsInline
          muted
        />

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Skeleton overlay - positioned over the video */}
        {showSkeleton && lastPose && isStreaming && (
          <div 
            className="absolute pointer-events-none"
            style={{
              width: overlaySize.width,
              height: overlaySize.height,
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            <SkeletonOverlay
              pose={lastPose}
              width={overlaySize.width}
              height={overlaySize.height}
            />
          </div>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/80 px-3 py-1.5 rounded-full">
            <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-sm font-medium">REC</span>
          </div>
        )}

        {/* Frame counter */}
        {isConnected && (
          <div className="absolute top-4 right-4 bg-black/50 px-2 py-1 rounded text-xs text-white font-mono">
            Frames: {frameCount}
          </div>
        )}

        {/* Debug info */}
        {isStreaming && (
          <div className="absolute bottom-4 right-4 bg-black/50 px-2 py-1 rounded text-xs text-white font-mono">
            {overlaySize.width}x{overlaySize.height}
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-500/90 px-4 py-2 rounded-lg">
            <p className="text-white text-sm">{error}</p>
          </div>
        )}

        {/* Camera off state */}
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-primary-dark)]/90">
            <Camera className="w-16 h-16 text-[var(--color-text-muted)] mb-4" />
            <p className="text-[var(--color-text-secondary)] mb-6">
              Camera is off
            </p>
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
          {/* Switch camera */}
          <button
            onClick={switchCamera}
            disabled={!isStreaming}
            title="Switch camera"
            aria-label="Switch camera"
            className="w-12 h-12 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center border border-[var(--color-primary-light)] disabled:opacity-50"
          >
            <RotateCcw className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          {/* Record button */}
          <button
            onClick={handleRecordToggle}
            disabled={!isStreaming}
            title={isRecording ? 'Stop recording' : 'Start recording'}
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

          {/* Toggle skeleton */}
          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            title="Toggle skeleton overlay"
            aria-label="Toggle skeleton overlay"
            className={`w-12 h-12 rounded-full flex items-center justify-center border ${
              showSkeleton
                ? 'bg-[var(--color-accent)]/20 border-[var(--color-accent)]'
                : 'bg-[var(--color-surface-card)] border-[var(--color-primary-light)]'
            }`}
          >
            <svg
              className={`w-5 h-5 ${showSkeleton ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              {/* Simple skeleton icon */}
              <circle cx="12" cy="4" r="2" />
              <line x1="12" y1="6" x2="12" y2="14" />
              <line x1="8" y1="8" x2="16" y2="8" />
              <line x1="12" y1="14" x2="8" y2="22" />
              <line x1="12" y1="14" x2="16" y2="22" />
            </svg>
          </button>
        </div>

        {/* Status text */}
        <p className="text-center text-[var(--color-text-muted)] text-sm mt-4">
          {!isStreaming
            ? 'Tap to start camera'
            : isRecording
            ? 'Recording... Tap to stop'
            : 'Position yourself and tap record'}
        </p>
      </div>
    </div>
  );
}