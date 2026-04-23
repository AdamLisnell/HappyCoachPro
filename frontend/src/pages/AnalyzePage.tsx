import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { VideoUpload } from '@/components/video/VideoUpload';
import { VideoPlayer } from '@/components/video/VideoPlayer';
import { SkeletonOverlay } from '@/components/analysis/SkeletonOverlay';
import { SwingPathOverlay } from '@/components/analysis/SwingPathOverlay';
import { ReferenceLineOverlay } from '@/components/analysis/ReferenceLineOverlay';
import { PhaseBreakdown, PHASE_LABELS } from '@/components/analysis/PhaseBreakdown';
import { CoachingTipsList } from '@/components/analysis/CoachingTipsList';
import { CoachReport } from '@/components/analysis/CoachReport';
import * as poseDetector from '@/lib/poseDetector';
import { analyzeFrames } from '@/lib/swingAnalyzer';
import { saveAnalysis } from '@/lib/historyStore';
import type { PoseFrame, SwingAnalysis, GolfClub, AICoachingReport } from '@/types';

function ScoreGauge({ score }: { score: number }) {
  const size = 88;
  const stroke = 7;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const grade = score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';
  const ring = score >= 80 ? '#4ade80' : score >= 60 ? '#E8C547' : '#fb923c';
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,0.08)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          stroke={ring} strokeWidth={stroke} fill="none"
          strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-3xl font-black leading-none" style={{ color: ring }}>{grade}</span>
        <span className="text-[10px] font-mono text-[var(--color-text-muted)] mt-0.5">{score}/100</span>
      </div>
    </div>
  );
}

function TogglePill({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-colors border ${
        active
          ? 'bg-[var(--color-accent)] text-[var(--color-primary-dark)] border-[var(--color-accent)]'
          : 'bg-transparent text-[var(--color-text-muted)] border-[var(--color-primary-light)]'
      }`}
    >
      {label}
    </button>
  );
}

// Bidirectional EMA to smooth out per-frame MediaPipe jitter.
// alpha=1 → no smoothing; alpha≈0.45 → balanced smooth/lag trade-off.
function smoothLandmarks(frames: PoseFrame[], alpha: number): PoseFrame[] {
  if (frames.length < 2) return frames;
  const clone = frames.map(f => ({
    ...f,
    landmarks: f.landmarks.map(lm => ({ ...lm })),
  }));
  // Forward pass
  for (let i = 1; i < clone.length; i++) {
    const prev = clone[i - 1].landmarks;
    const curr = clone[i].landmarks;
    for (let j = 0; j < curr.length; j++) {
      if (!prev[j]) continue;
      curr[j].x = prev[j].x * (1 - alpha) + curr[j].x * alpha;
      curr[j].y = prev[j].y * (1 - alpha) + curr[j].y * alpha;
      curr[j].z = prev[j].z * (1 - alpha) + curr[j].z * alpha;
    }
  }
  // Backward pass (removes lag introduced by forward pass)
  for (let i = clone.length - 2; i >= 0; i--) {
    const next = clone[i + 1].landmarks;
    const curr = clone[i].landmarks;
    for (let j = 0; j < curr.length; j++) {
      if (!next[j]) continue;
      curr[j].x = next[j].x * (1 - alpha) + curr[j].x * alpha;
      curr[j].y = next[j].y * (1 - alpha) + curr[j].y * alpha;
      curr[j].z = next[j].z * (1 - alpha) + curr[j].z * alpha;
    }
  }
  return clone;
}

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

// Lower FPS on mobile — seek-based analysis is CPU-bound and iOS seek is slow
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const ANALYSIS_FPS = isMobile ? 12 : 24;
const CANVAS_MAX_WIDTH = isMobile ? 480 : 854; // smaller canvas = faster MediaPipe

interface AnalyzePageProps {
  initialBlob?: Blob | null;
  onConsumed?: () => void;
}

export function AnalyzePage({ initialBlob, onConsumed }: AnalyzePageProps = {}) {
  const [pageState, setPageState] = useState<PageState>('upload');
  const [selectedClub, setSelectedClub] = useState<GolfClub>(
    () => (localStorage.getItem('hc_default_club') as GolfClub | null) ?? 'iron_7'
  );
  const [cameraAngle, setCameraAngle] = useState<CameraAngle>(
    () => (localStorage.getItem('hc_default_angle') as CameraAngle | null) ?? 'side'
  );
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
  const [showSwingPath, setShowSwingPath] = useState(false);
  const [showReferenceLines, setShowReferenceLines] = useState(false);
  const [showAngles, setShowAngles] = useState(false);

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

    // Ensure MediaPipe is ready (GPU → CPU fallback)
    try {
      await poseDetector.initialize();
    } catch (e) {
      setAnalysisError(
        e instanceof Error ? `Pose detection unavailable: ${e.message}` : 'Failed to load pose detection model.'
      );
      setPageState('upload');
      return;
    }

    // Create a hidden video element and attach it to DOM.
    // iOS Safari requires the element to be in the DOM to draw to canvas.
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px';
    document.body.appendChild(video);
    video.src = url;

    try {
      await new Promise<void>((resolve, reject) => {
        video.onloadeddata = () => resolve();   // loadeddata = first frame decoded, safer than loadedmetadata
        video.onerror = () => reject(new Error('Failed to load video — try MP4 format'));
        setTimeout(() => reject(new Error('Video load timed out after 15s')), 15000);
        video.load();
      });
    } catch (e) {
      document.body.removeChild(video);
      setAnalysisError(e instanceof Error ? e.message : 'Video could not be loaded');
      setPageState('upload');
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) { document.body.removeChild(video); return; }

    const duration = video.duration;
    if (!duration || !isFinite(duration) || duration <= 0) {
      document.body.removeChild(video);
      setAnalysisError('Video duration unreadable. Try MP4 format.');
      setPageState('upload');
      return;
    }

    // Scale canvas down — smaller canvas = much faster MediaPipe inference
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scale = Math.min(1, CANVAS_MAX_WIDTH / vw);
    canvas.width = Math.round(vw * scale);
    canvas.height = Math.round(vh * scale);
    setVideoSize({ width: canvas.width, height: canvas.height });
    const ctx = canvas.getContext('2d')!;

    const collectedFrames: PoseFrame[] = [];
    let frameNumber = 0;
    const step = 1 / ANALYSIS_FPS;

    // Seek-based extraction — reliable on iOS, no user-gesture requirement.
    // Each seek is fast now because canvas is small and MediaPipe runs on CPU.
    const seekTo = (t: number) =>
      new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 600);  // max 600ms per seek
        video.onseeked = () => { clearTimeout(timer); resolve(); };
        video.currentTime = t;
      });

    for (let t = 0; t < duration && !cancelRef.current; t += step) {
      await seekTo(t);
      // Yield one frame to let the browser composite the decoded frame into the video element
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const pose = poseDetector.detectImage(canvas, frameNumber, Math.round(t * 1000));
      if (pose) {
        collectedFrames.push(pose);
        setPoseFrames((prev) => [...prev, pose]);
      }
      frameNumber++;
      setAnalyzeProgress(Math.min(99, Math.round(((t + step) / duration) * 100)));
    }

    document.body.removeChild(video);
    if (cancelRef.current) return;

    if (collectedFrames.length < 3) {
      setAnalysisError(
        `Pose detection failed (${collectedFrames.length} poses found). Ensure the full body is visible in good lighting, or try a different video.`
      );
      setPageState('upload');
      return;
    }

    setAnalyzeProgress(100);

    // Smooth landmark positions: forward + backward EMA to remove MediaPipe jitter
    const smoothed = smoothLandmarks(collectedFrames, 0.45);
    setPoseFrames(smoothed);

    try {
      const result = analyzeFrames(smoothed, selectedClub, ANALYSIS_FPS, Math.round(duration * 1000));
      setSwingAnalysis(result);
      await saveAnalysis(result).catch(() => {}); // non-blocking
      setPageState('results');
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : 'Analysis failed');
      setPageState('upload');
    }
  }, [selectedClub]);

  // Consume a blob handed over from RecordPage (mobile-friendly handoff, no download)
  useEffect(() => {
    if (!initialBlob) return;
    const ext = initialBlob.type.includes('mp4') ? 'mp4' : 'webm';
    const file = new File([initialBlob], `swing-${Date.now()}.${ext}`, { type: initialBlob.type || 'video/mp4' });
    const url = URL.createObjectURL(initialBlob);
    handleVideoSelected(file, url);
    onConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialBlob]);

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

  // Fixed pose at address frame — used for static reference lines
  const addressPose = useMemo((): PoseFrame | null => {
    if (!poseFrames.length || !swingAnalysis) return null;
    const idx = swingAnalysis.key_frames.address ?? 0;
    return poseFrames[Math.min(idx, poseFrames.length - 1)] ?? null;
  }, [poseFrames, swingAnalysis]);

  // Phase markers for the video timeline
  const phaseMarkers = useMemo(() => {
    if (!swingAnalysis || !poseFrames.length) return [];
    const kf = swingAnalysis.key_frames;
    return (
      [
        { label: 'A', color: '#64B5F6', key: 'address' },
        { label: 'T', color: '#FF9800', key: 'top' },
        { label: 'I', color: '#F44336', key: 'impact' },
        { label: 'F', color: '#66BB6A', key: 'finish' },
      ] as const
    )
      .filter(({ key }) => kf[key] != null)
      .map(({ label, color, key }) => ({
        label,
        color,
        time: (poseFrames[kf[key]]?.timestamp_ms ?? (kf[key] * 1000) / swingAnalysis.fps) / 1000,
      }));
  }, [swingAnalysis, poseFrames]);

  // Current phase (used for badge label, skeleton angles + highlight)
  const currentPhase = useMemo(() => {
    if (!swingAnalysis?.phases.length) return null;
    const ms = currentVideoTime * 1000;
    let best = null as typeof swingAnalysis.phases[0] | null;
    let bestDist = Infinity;
    for (const p of swingAnalysis.phases) {
      const dist = Math.abs(p.timestamp_ms - ms);
      if (dist < bestDist && dist < 600) { best = p; bestDist = dist; }
    }
    return best;
  }, [swingAnalysis, currentVideoTime]);

  const currentPhaseName = useMemo((): string | null => {
    if (!currentPhase) return null;
    return PHASE_LABELS[currentPhase.phase] ?? currentPhase.phase;
  }, [currentPhase]);

  // Key angle callouts from the impact frame
  const impactAngles = useMemo(() => {
    if (!swingAnalysis) return [];
    const impact = swingAnalysis.phases.find((p) => p.phase === 'impact');
    if (!impact) return [];
    const { spine_angle, hip_rotation, shoulder_rotation, x_factor } = impact.angles;
    return [
      spine_angle      != null ? { label: 'Spine',     value: spine_angle.toFixed(0) }      : null,
      hip_rotation     != null ? { label: 'Hips',      value: hip_rotation.toFixed(0) }     : null,
      shoulder_rotation != null ? { label: 'Shoulders', value: shoulder_rotation.toFixed(0) } : null,
      x_factor         != null ? { label: 'X-Factor',  value: x_factor.toFixed(0) }         : null,
    ].filter(Boolean) as { label: string; value: string }[];
  }, [swingAnalysis]);

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

            <div>
              <p className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest mb-2">For best results</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { t: 'Side-on', d: 'Camera perpendicular to target line' },
                  { t: 'Full frame', d: 'Head to club, full swing visible' },
                  { t: 'Good light', d: 'Bright, no backlight behind you' },
                ].map((c) => (
                  <div key={c.t} className="bg-[var(--color-surface-card)] rounded-lg p-3 border border-[var(--color-primary-light)]/40">
                    <p className="text-[var(--color-accent)] text-xs font-semibold">{c.t}</p>
                    <p className="text-[10px] text-[var(--color-text-muted)] leading-tight mt-1">{c.d}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Analyzing State */}
        {pageState === 'analyzing' && (
          <div className="max-w-2xl mx-auto p-4 space-y-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-6 h-6 text-[var(--color-accent)] animate-spin flex-shrink-0" />
              <div className="flex-1">
                <p className="text-[var(--color-text)] font-medium text-sm">Analyzing swing · {poseFrames.length} frames</p>
                <div className="w-full h-1.5 bg-[var(--color-surface-card)] rounded-full overflow-hidden mt-1.5">
                  <div className="h-full bg-[var(--color-accent-bright)] transition-all duration-300" style={{ width: `${analyzeProgress}%` }} />
                </div>
              </div>
              <span className="font-mono text-[var(--color-accent-bright)] text-sm">{analyzeProgress}%</span>
            </div>
            {/* Shimmer preview of final layout */}
            <div className="h-48 rounded-xl shimmer" />
            <div className="h-16 rounded-xl shimmer" />
            <div className="grid grid-cols-4 gap-2">
              {[0, 1, 2, 3].map((i) => <div key={i} className="h-16 rounded-xl shimmer" />)}
            </div>
            <div className="h-32 rounded-xl shimmer" />
          </div>
        )}

        {/* Results State */}
        {pageState === 'results' && swingAnalysis && videoUrl && (
          <div className="max-w-2xl mx-auto space-y-0">

            {/* Video — full bleed, no padding */}
            <VideoPlayer
              src={videoUrl}
              onTimeUpdate={(t) => setCurrentVideoTime(t)}
              fps={ANALYSIS_FPS}
              phaseMarkers={phaseMarkers}
              overlay={
                <>
                  {/* Skeleton — dimmed when swing path is active */}
                  <SkeletonOverlay
                    pose={currentPose}
                    width={videoSize.width}
                    height={videoSize.height}
                    dimmed={showSwingPath}
                    showAngles={showAngles}
                    phaseAngles={currentPhase?.angles ?? null}
                    highlightPhase={currentPhase?.phase ?? null}
                    showClubShaft={showReferenceLines}
                    showGroundLine={showReferenceLines}
                  />
                  {/* Reference lines fixed at address frame */}
                  {showReferenceLines && (
                    <ReferenceLineOverlay pose={addressPose} width={videoSize.width} height={videoSize.height} />
                  )}
                  {/* Swing path builds as video plays */}
                  {showSwingPath && (
                    <SwingPathOverlay
                      frames={poseFrames}
                      keyFrames={swingAnalysis.key_frames}
                      currentTimeMs={currentVideoTime * 1000}
                    />
                  )}
                  {/* Phase name badge */}
                  {currentPhaseName && (
                    <div className="absolute top-3 left-3 pointer-events-none">
                      <span className="bg-black/55 text-white text-[11px] font-bold uppercase tracking-widest
                                       px-2.5 py-1 rounded-full border border-white/15">
                        {currentPhaseName}
                      </span>
                    </div>
                  )}
                </>
              }
            />

            {/* Overlay toggles + score strip — sits right below video */}
            <div className="bg-[var(--color-primary)] px-4 py-3 flex items-center justify-between border-t border-[var(--color-primary-light)]">
              {/* Score gauge */}
              <div className="flex items-center gap-3">
                <ScoreGauge score={swingAnalysis.overall_score} />
                <div>
                  <div className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest">Overall</div>
                  <div className="text-xs text-[var(--color-text-secondary)]">
                    {swingAnalysis.overall_score >= 80 ? 'Great swing' : swingAnalysis.overall_score >= 60 ? 'Room to improve' : 'Needs work'}
                  </div>
                  {swingAnalysis.tempo_ratio !== undefined && (
                    <div className="text-[10px] text-[var(--color-text-muted)] font-mono mt-0.5">
                      Tempo {swingAnalysis.tempo_ratio.toFixed(1)}:1
                      {swingAnalysis.x_factor_top !== undefined && ` · X-F ${Math.round(swingAnalysis.x_factor_top)}°`}
                    </div>
                  )}
                </div>
              </div>
              {/* Toggle pills */}
              <div className="flex gap-2">
                <TogglePill active={showSwingPath} onClick={() => setShowSwingPath((v) => !v)} label="Path" />
                <TogglePill active={showReferenceLines} onClick={() => setShowReferenceLines((v) => !v)} label="Lines" />
                <TogglePill active={showAngles} onClick={() => setShowAngles((v) => !v)} label="Angles" />
              </div>
            </div>

            {/* Main data area */}
            <div className="p-4 space-y-5">

              {/* Impact key angles */}
              {impactAngles.length > 0 && (
                <div>
                  <p className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest mb-2">At Impact</p>
                  <div className="grid grid-cols-4 gap-2">
                    {impactAngles.map(({ label, value }) => (
                      <div key={label} className="bg-[var(--color-surface-card)] rounded-xl p-3 text-center">
                        <div className="text-lg font-black text-[var(--color-accent)]">{value}°</div>
                        <div className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-wide mt-0.5">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Sub-scores 2×2 */}
              <div>
                <p className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest mb-2">Scores</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    swingAnalysis.posture_score  && { label: 'Posture',   s: swingAnalysis.posture_score },
                    swingAnalysis.tempo_score    && { label: 'Tempo',     s: swingAnalysis.tempo_score },
                    swingAnalysis.rotation_score && { label: 'Rotation',  s: swingAnalysis.rotation_score },
                    swingAnalysis.balance_score  && { label: 'Balance',   s: swingAnalysis.balance_score },
                  ].filter(Boolean).map((item) => {
                    const { label, s } = item as { label: string; s: typeof swingAnalysis.posture_score };
                    if (!s) return null;
                    const col = s.score >= 80 ? 'text-green-400 border-green-400/30'
                      : s.score >= 60 ? 'text-yellow-400 border-yellow-400/30'
                      : 'text-orange-400 border-orange-400/30';
                    return (
                      <div key={label} className={`bg-[var(--color-surface-card)] rounded-xl p-3 border-l-2 ${col.split(' ')[1]}`}>
                        <div className="flex items-baseline justify-between mb-1">
                          <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide">{label}</span>
                          <span className={`text-xl font-black ${col.split(' ')[0]}`}>{s.score}</span>
                        </div>
                        <p className="text-xs text-[var(--color-text-muted)] leading-snug line-clamp-2">{s.feedback}</p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Phase breakdown */}
              <PhaseBreakdown phases={swingAnalysis.phases} />

              {/* AI Coach report */}
              <CoachReport report={coachReport} loading={coachLoading} error={coachError} />

              {/* Rule-based tips */}
              <CoachingTipsList tips={swingAnalysis.tips} />

            </div>
          </div>
        )}
      </div>
    </div>
  );
}
