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

  // Calculate actual video content area inside the element, accounting for object-contain letterboxing
  const updateContentArea = useCallback(() => {
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
      // Video wider than box — bars on top/bottom
      contentW = elW;
      contentH = elW / videoAspect;
      offsetX = 0;
      offsetY = (elH - contentH) / 2;
    } else {
      // Video taller than box — bars on left/right
      contentH = elH;
      contentW = elH * videoAspect;
      offsetX = (elW - contentW) / 2;
      offsetY = 0;
    }
    setContentArea({ width: contentW, height: contentH, offsetX, offsetY });
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

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime, video.duration);
      
      const frameNumber = Math.floor(video.currentTime * fps);
      onFrameChange?.(frameNumber);
    };

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('ended', handleEnded);
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