import { useEffect, useRef, useMemo } from 'react';
import type { PoseFrame } from '@/types';

interface SwingPathOverlayProps {
  frames: PoseFrame[];
  keyFrames: Record<string, number>;
  currentTimeMs: number;  // draw path only up to this timestamp
}

const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;

// Phase color by segment
const SEG_COLORS = [
  'rgba(0,200,220,0.9)',   // address → top (teal)
  'rgba(255,140,0,0.9)',   // top → impact (orange)
  'rgba(255,210,50,0.9)',  // impact → finish (yellow)
];

const KEY_META: { key: string; label: string; color: string }[] = [
  { key: 'address', label: 'A', color: '#64B5F6' },
  { key: 'top',     label: 'T', color: '#FF9800' },
  { key: 'impact',  label: 'I', color: '#F44336' },
  { key: 'finish',  label: 'F', color: '#66BB6A' },
];

function bestWrist(frame: PoseFrame): { x: number; y: number; t: number } | null {
  const lw = frame.landmarks[LEFT_WRIST];
  const rw = frame.landmarks[RIGHT_WRIST];
  const best = lw && rw ? (lw.visibility >= rw.visibility ? lw : rw)
    : (lw || rw);
  if (!best || best.visibility < 0.3) return null;
  return { x: best.x, y: best.y, t: frame.timestamp_ms };
}

export function SwingPathOverlay({ frames, keyFrames, currentTimeMs }: SwingPathOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Pre-compute all wrist points segmented by phase (static after analysis)
  const segments = useMemo(() => {
    if (!frames.length) return [];
    const kf = keyFrames;
    const addr = kf.address ?? 0;
    const top  = kf.top   ?? Math.floor(frames.length * 0.4);
    const imp  = kf.impact ?? Math.floor(frames.length * 0.7);
    const fin  = kf.finish ?? frames.length - 1;

    const slice = (s: number, e: number) =>
      frames.slice(Math.max(0, s), Math.min(frames.length, e + 1))
            .map(bestWrist)
            .filter((p): p is NonNullable<typeof p> => p !== null);

    return [
      { color: SEG_COLORS[0], points: slice(addr, top) },
      { color: SEG_COLORS[1], points: slice(top, imp) },
      { color: SEG_COLORS[2], points: slice(imp, fin) },
    ];
  }, [frames, keyFrames]);

  const drawFrame = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 640;
    const H = canvas.offsetHeight || 480;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // Draw each segment up to currentTimeMs
    for (const seg of segments) {
      const visible = seg.points.filter((p) => p.t <= currentTimeMs);
      if (visible.length < 2) continue;

      ctx.save();
      ctx.strokeStyle = seg.color;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.shadowColor = 'rgba(0,0,0,0.6)';
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.moveTo(visible[0].x * W, visible[0].y * H);
      for (const p of visible.slice(1)) ctx.lineTo(p.x * W, p.y * H);
      ctx.stroke();
      ctx.restore();
    }

    // Key frame dots (only when that frame has been reached)
    for (const { key, label, color } of KEY_META) {
      const idx = keyFrames[key];
      if (idx == null || idx >= frames.length) continue;
      const pt = bestWrist(frames[idx]);
      if (!pt || pt.t > currentTimeMs) continue;

      const px = pt.x * W;
      const py = pt.y * H;

      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.save();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 8px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, px, py);
      ctx.restore();
    }
  };

  useEffect(() => {
    drawFrame();
  }, [segments, currentTimeMs]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(drawFrame);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [segments]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}
