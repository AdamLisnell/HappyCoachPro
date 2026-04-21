import { useEffect, useRef, useMemo } from 'react';
import type { PoseFrame } from '@/types';

interface SwingPathOverlayProps {
  frames: PoseFrame[];
  keyFrames: Record<string, number>;
}

interface Segment {
  color: string;
  points: { x: number; y: number }[];
}

// Wrist landmark indices
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;

const PHASE_COLORS = {
  backswing: 'rgba(0,200,220,0.9)',   // teal
  downswing: 'rgba(255,140,0,0.9)',   // orange
  finish:    'rgba(255,210,50,0.9)',   // yellow
};

const KEY_LABELS: Record<string, string> = {
  address: 'A', top: 'T', impact: 'I', finish: 'F',
};
const KEY_COLORS: Record<string, string> = {
  address: '#64B5F6', top: '#FF9800', impact: '#F44336', finish: '#66BB6A',
};

function getWristPoint(frame: PoseFrame): { x: number; y: number } | null {
  const lw = frame.landmarks[LEFT_WRIST];
  const rw = frame.landmarks[RIGHT_WRIST];
  // Prefer the more visible wrist
  if (lw && rw) {
    const best = lw.visibility >= rw.visibility ? lw : rw;
    if (best.visibility >= 0.3) return { x: best.x, y: best.y };
  }
  if (lw && lw.visibility >= 0.3) return { x: lw.x, y: lw.y };
  if (rw && rw.visibility >= 0.3) return { x: rw.x, y: rw.y };
  return null;
}

function drawPath(ctx: CanvasRenderingContext2D, segment: Segment, W: number, H: number, lw: number) {
  if (segment.points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = segment.color;
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.5)';
  ctx.shadowBlur = 3;
  ctx.beginPath();
  ctx.moveTo(segment.points[0].x * W, segment.points[0].y * H);
  for (const p of segment.points.slice(1)) {
    ctx.lineTo(p.x * W, p.y * H);
  }
  ctx.stroke();
  ctx.restore();
}

export function SwingPathOverlay({ frames, keyFrames }: SwingPathOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const segments = useMemo((): Segment[] => {
    if (!frames.length) return [];
    const kf = keyFrames;
    const addr = kf.address ?? 0;
    const top  = kf.top   ?? Math.floor(frames.length * 0.4);
    const imp  = kf.impact ?? Math.floor(frames.length * 0.7);
    const fin  = kf.finish ?? frames.length - 1;

    const slice = (start: number, end: number) =>
      frames
        .slice(Math.max(0, start), Math.min(frames.length, end + 1))
        .map(getWristPoint)
        .filter((p): p is { x: number; y: number } => p !== null);

    return [
      { color: PHASE_COLORS.backswing, points: slice(addr, top) },
      { color: PHASE_COLORS.downswing, points: slice(top, imp) },
      { color: PHASE_COLORS.finish,    points: slice(imp, fin) },
    ];
  }, [frames, keyFrames]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 640;
    const H = canvas.offsetHeight || 480;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // Draw each phase segment
    for (const seg of segments) {
      drawPath(ctx, seg, W, H, 3);
    }

    // Key frame dots + labels
    const kf = keyFrames;
    for (const [key, label] of Object.entries(KEY_LABELS)) {
      const idx = kf[key];
      if (idx == null || idx >= frames.length) continue;
      const pt = getWristPoint(frames[idx]);
      if (!pt) continue;
      const px = pt.x * W;
      const py = pt.y * H;
      const color = KEY_COLORS[key] ?? '#ffffff';

      // Shadow ring
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();

      // Colored dot
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Label
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, px, py);
      ctx.restore();
    }
  };

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [segments, keyFrames]);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}
