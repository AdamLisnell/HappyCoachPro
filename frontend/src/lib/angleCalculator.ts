import type { PoseFrame, Landmark, SwingAngles } from '@/types';

// MediaPipe landmark indices
const BP = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
} as const;

const VISIBILITY_THRESHOLD = 0.5;

function visible(lm: Landmark): boolean {
  return lm.visibility >= VISIBILITY_THRESHOLD;
}

function lm(frame: PoseFrame, index: number): Landmark | null {
  const l = frame.landmarks[index];
  return l && visible(l) ? l : null;
}

function angleAtVertex(
  p1: { x: number; y: number },
  vertex: { x: number; y: number },
  p3: { x: number; y: number },
): number | null {
  const v1x = p1.x - vertex.x;
  const v1y = p1.y - vertex.y;
  const v2x = p3.x - vertex.x;
  const v2y = p3.y - vertex.y;
  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
  if (mag1 === 0 || mag2 === 0) return null;
  return (Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * 180) / Math.PI;
}

export function calculateSpineAngle(frame: PoseFrame): number | null {
  const ls = lm(frame, BP.LEFT_SHOULDER);
  const rs = lm(frame, BP.RIGHT_SHOULDER);
  const lh = lm(frame, BP.LEFT_HIP);
  const rh = lm(frame, BP.RIGHT_HIP);
  if (!ls || !rs || !lh || !rh) return null;

  const shoulderMidX = (ls.x + rs.x) / 2;
  const shoulderMidY = (ls.y + rs.y) / 2;
  const hipMidX = (lh.x + rh.x) / 2;
  const hipMidY = (lh.y + rh.y) / 2;

  const svX = shoulderMidX - hipMidX;
  const svY = shoulderMidY - hipMidY;
  // Vertical reference points up (negative y in image coords)
  const dot = svX * 0 + svY * -1;
  const mag = Math.sqrt(svX * svX + svY * svY);
  if (mag === 0) return null;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

export function calculateShoulderRotation(frame: PoseFrame): number | null {
  const ls = lm(frame, BP.LEFT_SHOULDER);
  const rs = lm(frame, BP.RIGHT_SHOULDER);
  if (!ls || !rs) return null;
  const width = Math.abs(ls.x - rs.x);
  const normalized = Math.min(width / 0.4, 1.0);
  return (Math.acos(normalized) * 180) / Math.PI;
}

export function calculateHipRotation(frame: PoseFrame): number | null {
  const lh = lm(frame, BP.LEFT_HIP);
  const rh = lm(frame, BP.RIGHT_HIP);
  if (!lh || !rh) return null;
  const width = Math.abs(lh.x - rh.x);
  const normalized = Math.min(width / 0.3, 1.0);
  return (Math.acos(normalized) * 180) / Math.PI;
}

export function calculateXFactor(frame: PoseFrame): number | null {
  const sr = calculateShoulderRotation(frame);
  const hr = calculateHipRotation(frame);
  if (sr === null || hr === null) return null;
  return Math.abs(sr - hr);
}

export function calculateElbowAngle(frame: PoseFrame, side: 'left' | 'right'): number | null {
  const shoulder = lm(frame, side === 'left' ? BP.LEFT_SHOULDER : BP.RIGHT_SHOULDER);
  const elbow = lm(frame, side === 'left' ? BP.LEFT_ELBOW : BP.RIGHT_ELBOW);
  const wrist = lm(frame, side === 'left' ? BP.LEFT_WRIST : BP.RIGHT_WRIST);
  if (!shoulder || !elbow || !wrist) return null;
  return angleAtVertex(shoulder, elbow, wrist);
}

export function calculateKneeAngle(frame: PoseFrame, side: 'left' | 'right'): number | null {
  const hip = lm(frame, side === 'left' ? BP.LEFT_HIP : BP.RIGHT_HIP);
  const knee = lm(frame, side === 'left' ? BP.LEFT_KNEE : BP.RIGHT_KNEE);
  const ankle = lm(frame, side === 'left' ? BP.LEFT_ANKLE : BP.RIGHT_ANKLE);
  if (!hip || !knee || !ankle) return null;
  return angleAtVertex(hip, knee, ankle);
}

export function calculateAllAngles(frame: PoseFrame): SwingAngles {
  return {
    spine_angle: calculateSpineAngle(frame),
    spine_lateral: null,
    shoulder_rotation: calculateShoulderRotation(frame),
    hip_rotation: calculateHipRotation(frame),
    hip_sway: null,
    left_elbow: calculateElbowAngle(frame, 'left'),
    right_elbow: calculateElbowAngle(frame, 'right'),
    left_knee: calculateKneeAngle(frame, 'left'),
    right_knee: calculateKneeAngle(frame, 'right'),
    wrist_hinge: null,
    x_factor: calculateXFactor(frame),
  };
}
