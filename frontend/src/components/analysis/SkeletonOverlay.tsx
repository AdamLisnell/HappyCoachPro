/**
 * Skeleton Overlay Component
 * 
 * Renders pose landmarks as a skeleton over video.
 * Uses Canvas API for performant real-time rendering.
 */

import { useEffect, useRef } from 'react';
import type { PoseFrame, Landmark } from '@/types';

// Golf-relevant skeleton connections only (no face landmarks)
const SKELETON_CONNECTIONS: [number, number][] = [
  // Shoulders
  [11, 12],
  // Torso
  [11, 23], [12, 24],
  // Hips
  [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27], [27, 29], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [30, 32],
];

// Golf-relevant landmark indices to render
const GOLF_LANDMARKS = new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);

// Segment colors for skeleton lines
const getSegmentColor = (startIdx: number, endIdx: number): string => {
  // Arms
  if ([11, 12, 13, 14, 15, 16].includes(startIdx)) return '#FF9500'; // Bright orange
  // Hips / torso base
  if (startIdx === 23 && endIdx === 24) return '#00BFFF'; // Deep sky blue
  // Torso sides
  if ((startIdx === 11 && endIdx === 23) || (startIdx === 12 && endIdx === 24)) return '#39FF14'; // Neon green
  // Legs
  return '#FF3AFF'; // Bright magenta
};

// Joint colors based on body part
const getJointColor = (index: number): string => {
  if ([11, 12].includes(index)) return '#39FF14'; // Shoulders — neon green
  if ([13, 14].includes(index)) return '#FF9500'; // Elbows — bright orange
  if ([15, 16].includes(index)) return '#FFE400'; // Wrists — yellow (key for golf)
  if ([23, 24].includes(index)) return '#00BFFF'; // Hips — deep sky blue
  if ([25, 26].includes(index)) return '#FF3AFF'; // Knees — bright magenta
  if ([27, 28].includes(index)) return '#00FFCC'; // Ankles — teal
  if ([29, 30, 31, 32].includes(index)) return '#00FFCC'; // Feet — teal
  return '#FFFFFF';
};

interface SkeletonOverlayProps {
  pose: PoseFrame | null;
  width: number;
  height: number;
  showConnections?: boolean;
  showJoints?: boolean;
  jointRadius?: number;
  lineWidth?: number;
  minVisibility?: number;
}

export function SkeletonOverlay({
  pose,
  width,
  height,
  showConnections = true,
  showJoints = true,
  jointRadius = 7,
  lineWidth = 4,
  minVisibility = 0.3,
}: SkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Always match canvas drawing surface to its actual rendered size
    const dw = canvas.offsetWidth || width;
    const dh = canvas.offsetHeight || height;
    canvas.width = dw;
    canvas.height = dh;

    ctx.clearRect(0, 0, dw, dh);

    if (!pose || !pose.landmarks || pose.landmarks.length === 0) {
      return;
    }

    const landmarks = pose.landmarks;

    // Convert normalized coords to the canvas's actual pixel dimensions
    const toPixel = (lm: Landmark): [number, number] => {
      return [lm.x * dw, lm.y * dh];
    };

    // Helper to check if landmark is visible
    const isVisible = (lm: Landmark): boolean => {
      return lm.visibility >= minVisibility;
    };

    // Draw connections (skeleton lines)
    if (showConnections) {
      ctx.lineCap = 'round';

      for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];

        if (!start || !end || !isVisible(start) || !isVisible(end)) continue;

        const [x1, y1] = toPixel(start);
        const [x2, y2] = toPixel(end);
        const alpha = Math.min(start.visibility, end.visibility);
        const color = getSegmentColor(startIdx, endIdx);

        // Shadow for contrast
        ctx.lineWidth = lineWidth + 3;
        ctx.strokeStyle = `rgba(0,0,0,${alpha * 0.6})`;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Colored line
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color.replace(')', `, ${alpha})`).replace('rgb(', 'rgba(').replace('#', 'rgba(').replace(/^rgba\(([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2}), /, (_, r, g, b) => `rgba(${parseInt(r,16)},${parseInt(g,16)},${parseInt(b,16)},`);
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Draw joints (circles) — golf landmarks only
    if (showJoints) {
      landmarks.forEach((lm: Landmark, index: number) => {
        if (!GOLF_LANDMARKS.has(index) || !isVisible(lm)) return;

        const [x, y] = toPixel(lm);
        const color = getJointColor(index);
        const radius = jointRadius * (0.5 + lm.visibility * 0.5);

        ctx.globalAlpha = lm.visibility;

        // Dark outline for contrast
        ctx.beginPath();
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fill();

        // Main joint
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.globalAlpha = 1;
      });
    }

  }, [pose, width, height, showConnections, showJoints, jointRadius, lineWidth, minVisibility]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
      className="pointer-events-none"
    />
  );
}