import { useEffect, useRef } from 'react';
import type { PoseFrame, Landmark, SwingAngles, SwingPhase } from '@/types';

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

function jointColor(i: number): string {
  if (i === 11 || i === 12) return '#C9A227';
  if (i === 13 || i === 14) return '#ffffff';
  if (i === 15 || i === 16) return '#C9A227';
  if (i === 23 || i === 24) return '#64B5F6';
  if (i === 25 || i === 26) return '#ffffff';
  return 'rgba(255,255,255,0.6)';
}

interface SkeletonOverlayProps {
  pose: PoseFrame | null;
  width: number;
  height: number;
  dimmed?: boolean;
  showAngles?: boolean;
  phaseAngles?: SwingAngles | null;
  highlightPhase?: SwingPhase | null;
  showClubShaft?: boolean;
  showGroundLine?: boolean;
}

function drawBadge(ctx: CanvasRenderingContext2D, x: number, y: number, text: string) {
  ctx.save();
  ctx.font = 'bold 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const padX = 5, padY = 2.5;
  const w = ctx.measureText(text).width + padX * 2;
  const h = 14;
  const bx = x, by = y - h / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.beginPath();
  const r = 4;
  ctx.moveTo(bx + r, by);
  ctx.lineTo(bx + w - r, by);
  ctx.quadraticCurveTo(bx + w, by, bx + w, by + r);
  ctx.lineTo(bx + w, by + h - r);
  ctx.quadraticCurveTo(bx + w, by + h, bx + w - r, by + h);
  ctx.lineTo(bx + r, by + h);
  ctx.quadraticCurveTo(bx, by + h, bx, by + h - r);
  ctx.lineTo(bx, by + r);
  ctx.quadraticCurveTo(bx, by, bx + r, by);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = 'rgba(201,162,39,0.6)';
  ctx.lineWidth = 0.75;
  ctx.stroke();
  ctx.fillStyle = '#E8C547';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, bx + padX, y + 0.5);
  ctx.restore();
}

export function SkeletonOverlay({
  pose, width, height, dimmed = false,
  showAngles = false, phaseAngles = null, highlightPhase = null,
  showClubShaft = false, showGroundLine = false,
}: SkeletonOverlayProps) {
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

    // Ground line: horizontal at lower of both ankles
    if (showGroundLine) {
      const la = lm[27], ra = lm[28];
      const ys: number[] = [];
      if (la && vis(la)) ys.push(la.y);
      if (ra && vis(ra)) ys.push(ra.y);
      if (ys.length) {
        const gy = Math.max(...ys) * H;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.28)';
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(W, gy);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Club shaft: extrapolate from right shoulder → right wrist beyond wrist
    if (showClubShaft) {
      const rw = lm[16], rs = lm[12];
      if (rw && rs && rw.visibility >= 0.5 && rs.visibility >= 0.4) {
        const [sx, sy] = px(rs);
        const [wx, wy] = px(rw);
        const dx = wx - sx, dy = wy - sy;
        const len = Math.hypot(dx, dy) || 1;
        const extend = H * 0.35;
        const ex = wx + (dx / len) * extend;
        const ey = wy + (dy / len) * extend;
        ctx.save();
        ctx.strokeStyle = 'rgba(232,197,71,0.85)';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(0,0,0,0.6)';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.moveTo(wx, wy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        // Club head dot
        ctx.fillStyle = 'rgba(232,197,71,0.95)';
        ctx.beginPath();
        ctx.arc(ex, ey, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    // Connections
    ctx.lineCap = 'round';
    for (const [a, b] of CONNECTIONS) {
      const la = lm[a]; const lb = lm[b];
      if (!la || !lb || !vis(la) || !vis(lb)) continue;
      const [x1, y1] = px(la);
      const [x2, y2] = px(lb);
      const segAlpha = Math.min(la.visibility, lb.visibility) * (dimmed ? 0.35 : 1);
      const isArm = [11, 12, 13, 14, 15, 16].includes(a);
      const color = isArm ? `rgba(201,162,39,${segAlpha})` : `rgba(255,255,255,${segAlpha * 0.55})`;
      ctx.lineWidth = isArm ? 2.5 : 1.5;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Impact glow on wrists + elbows
    if (highlightPhase === 'impact') {
      const glowJoints = [13, 14, 15, 16];
      ctx.save();
      for (const j of glowJoints) {
        const l = lm[j];
        if (!l || !vis(l)) continue;
        const [x, y] = px(l);
        const grd = ctx.createRadialGradient(x, y, 2, x, y, 14);
        grd.addColorStop(0, 'rgba(255,80,80,0.75)');
        grd.addColorStop(1, 'rgba(255,80,80,0)');
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(x, y, 14, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Joint dots
    for (let i = 0; i < lm.length; i++) {
      if (!GOLF_JOINTS.has(i)) continue;
      const l = lm[i];
      if (!vis(l)) continue;
      const [x, y] = px(l);
      const r = [11, 12, 15, 16, 23, 24].includes(i) ? 4 : 3;
      const color = jointColor(i);
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

    // Angle badges
    if (showAngles && phaseAngles) {
      ctx.globalAlpha = 1;
      const place = (idx: number, value: number | null, dx = 8, dy = -4) => {
        if (value == null) return;
        const l = lm[idx];
        if (!l || !vis(l)) return;
        const [x, y] = px(l);
        drawBadge(ctx, x + dx, y + dy, `${Math.round(value)}°`);
      };
      place(13, phaseAngles.left_elbow, 8, -8);
      place(14, phaseAngles.right_elbow, -48, -8);
      // Hip rotation near hip midpoint
      if (phaseAngles.hip_rotation != null && lm[23] && lm[24]) {
        const mx = ((lm[23].x + lm[24].x) / 2) * W;
        const my = ((lm[23].y + lm[24].y) / 2) * H;
        drawBadge(ctx, mx + 10, my, `HIP ${Math.round(phaseAngles.hip_rotation)}°`);
      }
      // Shoulder rotation near shoulder midpoint
      if (phaseAngles.shoulder_rotation != null && lm[11] && lm[12]) {
        const mx = ((lm[11].x + lm[12].x) / 2) * W;
        const my = ((lm[11].y + lm[12].y) / 2) * H;
        drawBadge(ctx, mx + 10, my - 14, `SH ${Math.round(phaseAngles.shoulder_rotation)}°`);
      }
      // Spine angle mid-spine
      if (phaseAngles.spine_angle != null && lm[11] && lm[12] && lm[23] && lm[24]) {
        const sx = ((lm[11].x + lm[12].x) / 2) * W;
        const sy = ((lm[11].y + lm[12].y) / 2) * H;
        const hx = ((lm[23].x + lm[24].x) / 2) * W;
        const hy = ((lm[23].y + lm[24].y) / 2) * H;
        const mx = (sx + hx) / 2;
        const my = (sy + hy) / 2;
        drawBadge(ctx, mx + 10, my, `${Math.round(phaseAngles.spine_angle)}°`);
      }
    }

    ctx.globalAlpha = 1;
  }, [pose, width, height, dimmed, showAngles, phaseAngles, highlightPhase, showClubShaft, showGroundLine]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}
