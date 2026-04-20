import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { VideoUpload } from '@/components/video/VideoUpload';
import { VideoPlayer } from '@/components/video/VideoPlayer';
import { SkeletonOverlay } from '@/components/analysis/SkeletonOverlay';
import { OverallScoreBadge } from '@/components/analysis/OverallScoreBadge';
import { SwingScoreCard } from '@/components/analysis/SwingScoreCard';
import { PhaseBreakdown } from '@/components/analysis/PhaseBreakdown';
import { CoachingTipsList } from '@/components/analysis/CoachingTipsList';
import { CoachReport } from '@/components/analysis/CoachReport';
import * as poseDetector from '@/lib/poseDetector';
import { analyzeFrames } from '@/lib/swingAnalyzer';
import { saveAnalysis } from '@/lib/historyStore';
import type { PoseFrame, SwingAnalysis, GolfClub, AICoachingReport } from '@/types';

type PageState = 'upload' | 'analyzing' | 'results';
type CameraAngle = 'side' | 'behind';

const CAMERA_ANGLE_OPTIONS: { value: CameraAngle; label: string; description: string }[] = [
  { value: 'side', label: 'Side-on', description: 'Camera level with your trail shoulder, perpendicular to the target line' },
  { value: 'behind', label: 'Behind (DTL)', description: 'Camera behind you, looking down the target line' },
];

const CLUB_OPTIONS: { value: GolfClub; label: string }[] = [
  { value: 'driver', label: 'Driver' },
  { value: 'wood_3', label: '3 Wood' },
  { value: 'iron_4', label: '4 Iron' },
  { value: 'iron_5', label: '5 Iron' },
  { value: 'iron_6', label: '6 Iron' },
  { value: 'iron_7', label: '7 Iron' },
  { value: 'iron_8', label: '8 Iron' },
  { value: 'iron_9', label: '9 Iron' },
  { value: 'pitching_wedge', label: 'PW' },
  { value: 'sand_wedge', label: 'SW' },
  { value: 'lob_wedge', label: 'LW' },
  { value: 'putter', label: 'Putter' },
];

const ANALYSIS_FPS = 24; // frames per second of video to analyze

export function AnalyzePage() {
  const [pageState, setPageState] = useState<PageState>('upload');
  const [selectedClub, setSelectedClub] = useState<GolfClub>('iron_7');
  const [cameraAngle, setCameraAngle] = useState<CameraAngle>('side');
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [poseFrames, setPoseFrames] = useState<PoseFrame[]>([]);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [analyzeProgress, setAnalyzeProgress] = useState(0);
  const [videoSize, setVideoSize] = useState({ width: 640, height: 480 });
  const [swingAnalysis, setSwingAnalysis] = useState<SwingAnalysis | null>(null);
  const [coachReport, setCoachReport] = useState<AICoachingReport | null>(null);
  const [coachLoading, setCoachLoading] = useState(false);
  const [coachError, setCoachError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cancelRef = useRef(false);

  // Fetch AI coaching after analysis completes
  useEffect(() => {
    if (!swingAnalysis) return;
    setCoachLoading(true);
    setCoachError(null);

    fetch('/api/coaching', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis: swingAnalysis, cameraAngle }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error ?? 'Coaching unavailable');
        }
        return res.json() as Promise<AICoachingReport>;
      })
      .then(setCoachReport)
      .catch((e: Error) => setCoachError(e.message))
      .finally(() => setCoachLoading(false));
  }, [swingAnalysis]);

  const handleVideoSelected = useCallback(async (file: File, url: string) => {
    setVideoUrl(url);
    setPoseFrames([]);
    setSwingAnalysis(null);
    setCoachReport(null);
    setCoachError(null);
    setAnalysisError(null);
    setAnalyzeProgress(0);
    setPageState('analyzing');
    cancelRef.current = false;

    // Ensure MediaPipe is ready
    await poseDetector.initialize();

    // Create a hidden video element for frame extraction
    const video = document.createElement('video');
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = 'anonymous';

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video'));
    });

    const canvas = canvasRef.current;
    if (!canvas) return;

    const duration = video.duration;
    const step = 1 / ANALYSIS_FPS;
    const totalSteps = Math.ceil(duration / step);
    const collectedFrames: PoseFrame[] = [];
    let frameNumber = 0;

    for (let t = 0; t < duration && !cancelRef.current; t += step) {
      video.currentTime = t;
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
      });

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      setVideoSize({ width: canvas.width, height: canvas.height });

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const pose = poseDetector.detectImage(canvas, frameNumber, Math.round(t * 1000));
        if (pose) {
          collectedFrames.push(pose);
          setPoseFrames((prev) => [...prev, pose]);
        }
      }
      frameNumber++;
      setAnalyzeProgress(Math.round(((t + step) / duration) * 100));
    }

    if (cancelRef.current) return;

    if (collectedFrames.length < 3) {
      setAnalysisError('Not enough pose data detected. Try a clearer side-view video with good lighting.');
      setPageState('upload');
      return;
    }

    try {
      const fps = 1 / step;
      const result = analyzeFrames(collectedFrames, selectedClub, fps, Math.round(duration * 1000));
      setSwingAnalysis(result);
      await saveAnalysis(result).catch(() => {}); // non-blocking
      setPageState('results');
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : 'Analysis failed');
      setPageState('upload');
    }
  }, [selectedClub]);

  const handleBack = useCallback(() => {
    cancelRef.current = true;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setPoseFrames([]);
    setSwingAnalysis(null);
    setCoachReport(null);
    setPageState('upload');
    setAnalyzeProgress(0);
  }, [videoUrl]);

  // Interpolate between the two nearest analyzed frames for smooth skeleton tracking
  const currentPose = useMemo((): PoseFrame | null => {
    if (!poseFrames.length) return null;
    const targetMs = currentVideoTime * 1000;

    // Binary search for insertion point
    let lo = 0, hi = poseFrames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (poseFrames[mid].timestamp_ms < targetMs) lo = mid + 1;
      else hi = mid;
    }

    if (lo === 0) return poseFrames[0];
    if (lo >= poseFrames.length) return poseFrames[poseFrames.length - 1];

    const before = poseFrames[lo - 1];
    const after = poseFrames[lo];
    const span = after.timestamp_ms - before.timestamp_ms;
    if (span <= 0) return before;

    // Linear interpolation factor
    const t = Math.min(1, Math.max(0, (targetMs - before.timestamp_ms) / span));
    if (t < 0.01) return before;
    if (t > 0.99) return after;

    return {
      ...before,
      landmarks: before.landmarks.map((lm, i) => {
        const b = after.landmarks[i];
        if (!b) return lm;
        return {
          ...lm,
          x: lm.x + (b.x - lm.x) * t,
          y: lm.y + (b.y - lm.y) * t,
          z: lm.z + (b.z - lm.z) * t,
          visibility: lm.visibility + (b.visibility - lm.visibility) * t,
        };
      }),
    };
  }, [poseFrames, currentVideoTime]);

  const headerSubtitle = {
    upload: 'Upload a video',
    analyzing: `${analyzeProgress}% complete`,
    results: `${poseFrames.length} frames analyzed`,
  }[pageState];

  const headerTitle = {
    upload: 'Analyze Swing',
    analyzing: 'Analyzing…',
    results: 'Results',
  }[pageState];

  return (
    <div className="min-h-screen bg-[var(--color-surface)] flex flex-col">
      {/* Header */}
      <header className="bg-[var(--color-primary)] px-4 py-3 flex items-center gap-3">
        {pageState !== 'upload' && (
          <button
            onClick={handleBack}
            aria-label="Back"
            className="w-10 h-10 rounded-xl bg-[var(--color-surface-card)] flex items-center justify-center border border-[var(--color-primary-light)] flex-shrink-0"
          >
            <ArrowLeft className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>
        )}
        <div>
          <h1 className="text-lg font-bold text-[var(--color-text)]">{headerTitle}</h1>
          <p className="text-xs text-[var(--color-accent)] uppercase tracking-wider">{headerSubtitle}</p>
        </div>
      </header>

      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Upload State */}
        {pageState === 'upload' && (
          <div className="max-w-lg mx-auto p-4 mt-4 space-y-6">
            {/* Camera angle selector */}
            <div>
              <label className="block text-[var(--color-text-secondary)] text-sm font-medium mb-2">Camera angle</label>
              <div className="grid grid-cols-2 gap-2">
                {CAMERA_ANGLE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setCameraAngle(opt.value)}
                    className={`p-3 rounded-lg text-left transition-colors border ${
                      cameraAngle === opt.value
                        ? 'bg-[var(--color-accent)]/15 border-[var(--color-accent)] text-[var(--color-text)]'
                        : 'bg-[var(--color-surface-card)] border-transparent text-[var(--color-text-secondary)]'
                    }`}
                  >
                    <p className="text-sm font-medium">{opt.label}</p>
                    <p className="text-xs text-[var(--color-text-muted)] mt-0.5 leading-tight">{opt.description}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Club selector */}
            <div>
              <label className="block text-[var(--color-text-secondary)] text-sm font-medium mb-2">Club</label>
              <div className="flex flex-wrap gap-2">
                {CLUB_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setSelectedClub(opt.value)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      selectedClub === opt.value
                        ? 'bg-[var(--color-accent)] text-[var(--color-primary-dark)]'
                        : 'bg-[var(--color-surface-card)] text-[var(--color-text-secondary)]'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <VideoUpload onVideoSelected={handleVideoSelected} />

            {analysisError && (
              <div className="bg-red-500/10 border border-red-400/30 rounded-xl p-4">
                <p className="text-red-400 text-sm">{analysisError}</p>
              </div>
            )}

            <p className="text-[var(--color-text-muted)] text-sm text-center">
              Best results with a side-view recording in good lighting.
            </p>
          </div>
        )}

        {/* Analyzing State */}
        {pageState === 'analyzing' && (
          <div className="flex flex-col items-center justify-center h-full gap-6 p-8 mt-12">
            <Loader2 className="w-16 h-16 text-[var(--color-accent)] animate-spin" />
            <div className="text-center">
              <p className="text-[var(--color-text)] font-medium">Analyzing your swing…</p>
              <p className="text-[var(--color-text-muted)] text-sm mt-1">Running pose detection in your browser</p>
            </div>
            <div className="w-64 h-2 bg-[var(--color-surface-card)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--color-accent)] transition-all duration-300"
                style={{ width: `${analyzeProgress}%` }}
              />
            </div>
            <p className="text-[var(--color-text-muted)] text-sm">{poseFrames.length} frames processed</p>
          </div>
        )}

        {/* Results State */}
        {pageState === 'results' && swingAnalysis && videoUrl && (
          <div className="max-w-2xl mx-auto p-4 space-y-6">
            {/* Video with skeleton overlay */}
            <VideoPlayer
              src={videoUrl}
              onTimeUpdate={(t) => setCurrentVideoTime(t)}
              fps={ANALYSIS_FPS}
              overlay={
                currentPose ? (
                  <SkeletonOverlay pose={currentPose} width={videoSize.width} height={videoSize.height} />
                ) : null
              }
            />

            {/* Overall score */}
            <OverallScoreBadge score={swingAnalysis.overall_score} summary={swingAnalysis.summary} />

            {/* Sub-scores */}
            <div className="grid grid-cols-2 gap-3">
              {swingAnalysis.posture_score && (
                <SwingScoreCard label="Posture" swingScore={swingAnalysis.posture_score} />
              )}
              {swingAnalysis.tempo_score && (
                <SwingScoreCard label="Tempo" swingScore={swingAnalysis.tempo_score} />
              )}
              {swingAnalysis.rotation_score && (
                <SwingScoreCard label="Rotation" swingScore={swingAnalysis.rotation_score} />
              )}
              {swingAnalysis.balance_score && (
                <SwingScoreCard label="Balance" swingScore={swingAnalysis.balance_score} />
              )}
            </div>

            {/* Phase breakdown */}
            <PhaseBreakdown phases={swingAnalysis.phases} />

            {/* AI Coach report */}
            <CoachReport report={coachReport} loading={coachLoading} error={coachError} />

            {/* Rule-based tips */}
            <CoachingTipsList tips={swingAnalysis.tips} />
          </div>
        )}
      </div>
    </div>
  );
}
