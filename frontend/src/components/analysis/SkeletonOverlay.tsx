/**
 * Skeleton Overlay Component
 * 
 * Renders pose landmarks as a skeleton over video.
 * Uses Canvas API for performant real-time rendering.
 */

import { useEffect, useRef } from 'react';
import type { PoseFrame, Landmark } from '@/types';

// Skeleton connections (pairs of landmark indices)
const SKELETON_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  
  // Body
  [11, 12], // Shoulders
  [11, 23], [12, 24], // Torso sides
  [23, 24], // Hips
  
  // Left arm
  [11, 13], [13, 15],
  
  // Right arm
  [12, 14], [14, 16],
  
  // Left leg
  [23, 25], [25, 27], [27, 29], [29, 31],
  
  // Right leg
  [24, 26], [26, 28], [28, 30], [30, 32],
];

// Joint colors based on body part
const getJointColor = (index: number): string => {
  // Major joints (shoulders, hips)
  if ([11, 12, 23, 24].includes(index)) {
    return '#00FF41'; // Neon green
  }
  // Elbows
  if ([13, 14].includes(index)) {
    return '#FFFF00'; // Yellow
  }
  // Wrists
  if ([15, 16].includes(index)) {
    return '#FF6B00'; // Orange
  }
  // Knees
  if ([25, 26].includes(index)) {
    return '#FF00FF'; // Magenta
  }
  // Ankles and feet
  if ([27, 28, 29, 30, 31, 32].includes(index)) {
    return '#00FFFF'; // Cyan
  }
  // Face
  if (index <= 10) {
    return '#FF0080'; // Pink
  }
  return '#FFFFFF'; // White default
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
  jointRadius = 6,
  lineWidth = 3,
  minVisibility = 0.3,
}: SkeletonOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!pose || !pose.landmarks || pose.landmarks.length === 0) {
      return;
    }

    const landmarks = pose.landmarks;

    // Helper to convert normalized coords to pixel coords
    const toPixel = (lm: Landmark): [number, number] => {
      return [lm.x * width, lm.y * height];
    };

    // Helper to check if landmark is visible
    const isVisible = (lm: Landmark): boolean => {
      return lm.visibility >= minVisibility;
    };

    // Draw connections (skeleton lines)
    if (showConnections) {
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';

      for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];

        if (!start || !end || !isVisible(start) || !isVisible(end)) {
          continue;
        }

        const [x1, y1] = toPixel(start);
        const [x2, y2] = toPixel(end);

        // Gradient based on visibility
        const alpha = Math.min(start.visibility, end.visibility);
        ctx.strokeStyle = `rgba(0, 255, 65, ${alpha})`;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // Draw joints (circles)
    if (showJoints) {
      landmarks.forEach((lm: Landmark, index: number) => {
        if (!isVisible(lm)) return;

        const [x, y] = toPixel(lm);
        const color = getJointColor(index);
        const radius = jointRadius * (0.5 + lm.visibility * 0.5);

        // Outer glow
        ctx.beginPath();
        ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 0, 0, 0.5)`;
        ctx.fill();

        // Main joint
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    }

  }, [pose, width, height, showConnections, showJoints, jointRadius, lineWidth, minVisibility]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute top-0 left-0 pointer-events-none"
    />
  );
}