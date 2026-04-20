/**
 * Video Player Component
 * 
 * Custom video player with playback controls and skeleton overlay support.
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Gauge
} from 'lucide-react';

interface VideoPlayerProps {
  src: string;
  onTimeUpdate?: (currentTime: number, duration: number) => void;
  onFrameChange?: (frameNumber: number) => void;
  overlay?: React.ReactNode;
  fps?: number;
}

const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2];

export function VideoPlayer({ 
  src, 
  onTimeUpdate,
  onFrameChange,
  overlay,
  fps = 30
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [contentArea, setContentArea] = useState({ width: 640, height: 480, offsetX: 0, offsetY: 0 });

  // Calculate actual video content area relative to the container div.
  // Must account for: (1) the video element's position inside the flex container,
  // and (2) the letterbox offset inside the element due to object-contain.
  const updateContentArea = useCallback(() => {
    const video = videoRef.current;
    const container = containerRef.current;
    if (!video || !container) return;

    const videoRect = video.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    const elW = videoRect.width;
    const elH = videoRect.height;
    if (elW === 0 || elH === 0) return;

    // Video element's top-left corner relative to the container
    const elLeft = videoRect.left - containerRect.left;
    const elTop = videoRect.top - containerRect.top;

    const intrinsicW = video.videoWidth || elW;
    const intrinsicH = video.videoHeight || elH;
    const videoAspect = intrinsicW / intrinsicH;
    const elementAspect = elW / elH;

    let contentW: number, contentH: number, letterboxX: number, letterboxY: number;
    if (videoAspect > elementAspect) {
      // Wider video — bars top/bottom inside element
      contentW = elW;
      contentH = elW / videoAspect;
      letterboxX = 0;
      letterboxY = (elH - contentH) / 2;
    } else {
      // Taller video — bars left/right inside element
      contentH = elH;
      contentW = elH * videoAspect;
      letterboxX = (elW - contentW) / 2;
      letterboxY = 0;
    }

    setContentArea({
      width: contentW,
      height: contentH,
      offsetX: elLeft + letterboxX,
      offsetY: elTop + letterboxY,
    });
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.addEventListener('loadedmetadata', updateContentArea);
      video.addEventListener('resize', updateContentArea);
    }
    window.addEventListener('resize', updateContentArea);
    const interval = setInterval(updateContentArea, 500);
    return () => {
      if (video) {
        video.removeEventListener('loadedmetadata', updateContentArea);
        video.removeEventListener('resize', updateContentArea);
      }
      window.removeEventListener('resize', updateContentArea);
      clearInterval(interval);
    };
  }, [src, updateContentArea]);

  // Handle time updates
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => setDuration(video.duration);
    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('ended', handleEnded);
    };
  }, []);

  // rAF loop fires at ~60fps while playing for smooth skeleton sync
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let rafId: number;
    let lastReportedTime = -1;

    const tick = () => {
      if (video && !video.paused && !video.ended) {
        const t = video.currentTime;
        setCurrentTime(t);
        if (t !== lastReportedTime) {
          onTimeUpdate?.(t, video.duration);
          onFrameChange?.(Math.floor(t * fps));
          lastReportedTime = t;
        }
      }
      rafId = requestAnimationFrame(tick);
    };

    // Also handle seek/pause-scrub via timeupdate (covers paused scrubbing)
    const handleTimeUpdate = () => {
      const t = video.currentTime;
      setCurrentTime(t);
      onTimeUpdate?.(t, video.duration);
      onFrameChange?.(Math.floor(t * fps));
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [fps, onTimeUpdate, onFrameChange]);

  // Play/Pause toggle
  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  // Restart video
  const restart = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = 0;
    video.play();
    setIsPlaying(true);
  }, []);

  // Step frame forward/backward
  const stepFrame = useCallback((direction: 'forward' | 'backward') => {
    const video = videoRef.current;
    if (!video) return;

    video.pause();
    setIsPlaying(false);

    const frameTime = 1 / fps;
    if (direction === 'forward') {
      video.currentTime = Math.min(video.currentTime + frameTime, video.duration);
    } else {
      video.currentTime = Math.max(video.currentTime - frameTime, 0);
    }
  }, [fps]);

  // Change playback speed
  const cycleSpeed = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    const currentIndex = PLAYBACK_SPEEDS.indexOf(playbackSpeed);
    const nextIndex = (currentIndex + 1) % PLAYBACK_SPEEDS.length;
    const newSpeed = PLAYBACK_SPEEDS[nextIndex];
    
    video.playbackRate = newSpeed;
    setPlaybackSpeed(newSpeed);
  }, [playbackSpeed]);

  // Seek to position
  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const newTime = parseFloat(e.target.value);
    video.currentTime = newTime;
    setCurrentTime(newTime);
  }, []);

  // Format time as MM:SS
  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col bg-black rounded-xl overflow-hidden">
      {/* Video container */}
      <div 
        ref={containerRef}
        className="relative flex items-center justify-center bg-black"
        style={{ minHeight: '300px' }}
      >
        <video
          ref={videoRef}
          src={src}
          className="max-w-full max-h-[60vh] object-contain"
          playsInline
          onClick={togglePlay}
        />

        {/* Overlay — positioned over the actual video content, not the full element box */}
        {overlay && (
          <div
            className="absolute pointer-events-none overflow-hidden"
            style={{
              left: contentArea.offsetX,
              top: contentArea.offsetY,
              width: contentArea.width,
              height: contentArea.height,
            }}
          >
            {overlay}
          </div>
        )}

        {/* Play button overlay when paused */}
        {!isPlaying && (
          <div 
            className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer"
            onClick={togglePlay}
          >
            <div className="w-20 h-20 rounded-full bg-[var(--color-accent)] flex items-center justify-center">
              <Play className="w-10 h-10 text-[var(--color-primary-dark)] ml-1" />
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="bg-[var(--color-primary)] p-4">
        {/* Timeline */}
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xs text-[var(--color-text-muted)] font-mono w-12">
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            aria-label="Video timeline"
            min={0}
            max={duration || 100}
            step={0.01}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-2 bg-[var(--color-surface-card)] rounded-full appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none 
                       [&::-webkit-slider-thumb]:w-4 
                       [&::-webkit-slider-thumb]:h-4 
                       [&::-webkit-slider-thumb]:rounded-full 
                       [&::-webkit-slider-thumb]:bg-[var(--color-accent)]
                       [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <span className="text-xs text-[var(--color-text-muted)] font-mono w-12 text-right">
            {formatTime(duration)}
          </span>
        </div>

        {/* Buttons */}
        <div className="flex items-center justify-center gap-4">
          {/* Restart */}
          <button
            onClick={restart}
            title="Restart"
            className="w-10 h-10 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center hover:bg-[var(--color-primary-light)] transition-colors"
          >
            <RotateCcw className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          {/* Step backward */}
          <button
            onClick={() => stepFrame('backward')}
            title="Previous frame"
            className="w-10 h-10 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center hover:bg-[var(--color-primary-light)] transition-colors"
          >
            <ChevronLeft className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            title={isPlaying ? 'Pause' : 'Play'}
            className="w-14 h-14 rounded-full bg-[var(--color-accent)] flex items-center justify-center hover:bg-[var(--color-accent-light)] transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-7 h-7 text-[var(--color-primary-dark)]" />
            ) : (
              <Play className="w-7 h-7 text-[var(--color-primary-dark)] ml-1" />
            )}
          </button>

          {/* Step forward */}
          <button
            onClick={() => stepFrame('forward')}
            title="Next frame"
            className="w-10 h-10 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center hover:bg-[var(--color-primary-light)] transition-colors"
          >
            <ChevronRight className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          {/* Speed */}
          <button
            onClick={cycleSpeed}
            title={`Speed: ${playbackSpeed}x`}
            className="w-10 h-10 rounded-full bg-[var(--color-surface-card)] flex items-center justify-center hover:bg-[var(--color-primary-light)] transition-colors relative"
          >
            <Gauge className="w-5 h-5 text-[var(--color-text-secondary)]" />
            <span className="absolute -bottom-1 -right-1 text-[10px] bg-[var(--color-accent)] text-[var(--color-primary-dark)] px-1 rounded font-bold">
              {playbackSpeed}x
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}