import { useEffect, useRef } from 'react';
import type { PoseFrame, Landmark } from '@/types';

interface ReferenceLineOverlayProps {
  pose: PoseFrame | null;
  width: number;
  height: number;
}

// Extend a line defined by two points to the canvas edges.
// Returns pixel coords of the clipped endpoints.
function extendedLine(
  x1: number, y1: number, x2: number, y2: number, W: number, H: number,
): [number, number, number, number] | null {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) return null;

  // Parametric t values at each canvas boundary
  const tCandidates: number[] = [];
  if (Math.abs(dx) > 0.001) {
    tCandidates.push(-x1 / dx);       // x = 0
    tCandidates.push((W - x1) / dx);  // x = W
  }
  if (Math.abs(dy) > 0.001) {
    tCandidates.push(-y1 / dy);       // y = 0
    tCandidates.push((H - y1) / dy);  // y = H
  }

  // Keep t values where the resulting point is inside the canvas
  const valid = tCandidates.filter((t) => {
    const px = x1 + t * dx;
    const py = y1 + t * dy;
    return px >= -1 && px <= W + 1 && py >= -1 && py <= H + 1;
  });

  if (valid.length < 2) return null;
  valid.sort((a, b) => a - b);
  const tMin = valid[0];
  const tMax = valid[valid.length - 1];

  return [
    x1 + tMin * dx, y1 + tMin * dy,
    x1 + tMax * dx, y1 + tMax * dy,
  ];
}

function midpoint(a: Landmark, b: Landmark): { x: number; y: number } {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function isVisible(lm: Landmark, min = 0.3): boolean {
  return lm.visibility >= min;
}

export function ReferenceLineOverlay({ pose, width, height }: ReferenceLineOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const W = canvas.offsetWidth || width;
    const H = canvas.offsetHeight || height;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    if (!pose?.landmarks?.length) return;
    const lm = pose.landmarks;

    const drawLine = (
      x1: number, y1: number, x2: number, y2: number,
      color: string, dash: number[], lw: number,
    ) => {
      const pts = extendedLine(x1, y1, x2, y2, W, H);
      if (!pts) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash(dash);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(pts[0], pts[1]);
      ctx.lineTo(pts[2], pts[3]);
      ctx.stroke();
      ctx.restore();
    };

    // ── Spine line ────────────────────────────────────────────────────
    const lh = lm[23]; const rh = lm[24];
    const ls = lm[11]; const rs = lm[12];
    if (lh && rh && ls && rs && isVisible(lh) && isVisible(rh) && isVisible(ls) && isVisible(rs)) {
      const hip = midpoint(lh, rh);
      const sho = midpoint(ls, rs);
      const hx = hip.x * W; const hy = hip.y * H;
      const sx = sho.x * W; const sy = sho.y * H;

      drawLine(hx, hy, sx, sy, 'rgba(100,220,255,0.75)', [10, 6], 2);

      // Angle from vertical
      const dx = sx - hx; const dy = sy - hy;
      const angle = Math.abs(Math.atan2(dx, -dy) * 180 / Math.PI);

      // Label near shoulder end
      ctx.save();
      ctx.fillStyle = 'rgba(100,220,255,0.95)';
      ctx.font = 'bold 13px system-ui, sans-serif';
      ctx.shadowColor = 'rgba(0,0,0,0.8)';
      ctx.shadowBlur = 4;
      ctx.fillText(`${angle.toFixed(0)}°`, sx + 6, sy - 6);
      ctx.restore();
    }

    // ── Shoulder line ─────────────────────────────────────────────────
    if (ls && rs && isVisible(ls) && isVisible(rs)) {
      drawLine(ls.x * W, ls.y * H, rs.x * W, rs.y * H, 'rgba(255,160,40,0.7)', [8, 5], 2);
    }

    // ── Hip line ──────────────────────────────────────────────────────
    if (lh && rh && isVisible(lh) && isVisible(rh)) {
      drawLine(lh.x * W, lh.y * H, rh.x * W, rh.y * H, 'rgba(100,255,120,0.7)', [8, 5], 2);
    }
  }, [pose, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}
