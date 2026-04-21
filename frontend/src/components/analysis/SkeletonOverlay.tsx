import { useEffect, useRef } from 'react';
import type { PoseFrame, Landmark } from '@/types';

// Golf-relevant connections only (no face)
const CONNECTIONS: [number, number][] = [
  [11, 12],               // shoulders
  [11, 23], [12, 24],     // torso
  [23, 24],               // hips
  [11, 13], [13, 15],     // left arm
  [12, 14], [14, 16],     // right arm
  [23, 25], [25, 27],     // left leg
  [24, 26], [26, 28],     // right leg
  [27, 29], [29, 31],     // left foot
  [28, 30], [30, 32],     // right foot
];

const GOLF_JOINTS = new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);

// Subtle, elegant colors
function jointColor(i: number): string {
  if (i === 11 || i === 12) return '#C9A227'; // shoulders — gold accent
  if (i === 13 || i === 14) return '#ffffff';  // elbows — white
  if (i === 15 || i === 16) return '#C9A227';  // wrists — gold (key for golf)
  if (i === 23 || i === 24) return '#64B5F6';  // hips — blue
  if (i === 25 || i === 26) return '#ffffff';  // knees — white
  return 'rgba(255,255,255,0.6)';
}

function segmentColor(s: number): string {
  if (s === 11 || s === 12) return 'rgba(201,162,39,0.7)';  // arms — gold
  if (s === 13 || s === 14) return 'rgba(201,162,39,0.7)';
  if (s === 23 && false) return ''; // unused
  return 'rgba(255,255,255,0.5)'; // body/legs — white semi-transparent
}

interface SkeletonOverlayProps {
  pose: PoseFrame | null;
  width: number;
  height: number;
  dimmed?: boolean;
}

export function SkeletonOverlay({ pose, width, height, dimmed = false }: SkeletonOverlayProps) {
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
    const alpha = dimmed ? 0.35 : 1;
    ctx.globalAlpha = alpha;

    const px = (l: Landmark) => [l.x * W, l.y * H] as [number, number];
    const vis = (l: Landmark) => l.visibility >= 0.3;

    // Draw connections — thin, elegant
    ctx.lineCap = 'round';
    for (const [a, b] of CONNECTIONS) {
      const la = lm[a]; const lb = lm[b];
      if (!la || !lb || !vis(la) || !vis(lb)) continue;
      const [x1, y1] = px(la);
      const [x2, y2] = px(lb);
      const segAlpha = Math.min(la.visibility, lb.visibility) * (dimmed ? 0.35 : 1);

      // Determine color based on which joints are in the arm chain
      const isArm = [11, 12, 13, 14, 15, 16].includes(a);
      const color = isArm ? `rgba(201,162,39,${segAlpha})` : `rgba(255,255,255,${segAlpha * 0.55})`;

      ctx.lineWidth = isArm ? 2.5 : 1.5;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw joint dots — small and clean
    for (let i = 0; i < lm.length; i++) {
      if (!GOLF_JOINTS.has(i)) continue;
      const l = lm[i];
      if (!vis(l)) continue;
      const [x, y] = px(l);
      const r = [11, 12, 15, 16, 23, 24].includes(i) ? 4 : 3;
      const color = jointColor(i);

      // Thin dark ring for contrast
      ctx.beginPath();
      ctx.arc(x, y, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${0.5 * l.visibility})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = l.visibility * (dimmed ? 0.35 : 1);
      ctx.fill();
      ctx.globalAlpha = alpha;
    }

    ctx.globalAlpha = 1;
  }, [pose, width, height, dimmed]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}
