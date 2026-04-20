import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { PoseFrame, Landmark } from '@/types';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const BODY_PART_NAMES = [
  'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
  'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee',
  'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index',
];

let imageLandmarker: PoseLandmarker | null = null;
let videoLandmarker: PoseLandmarker | null = null;
let initPromise: Promise<void> | null = null;

export async function initialize(): Promise<void> {
  if (imageLandmarker && videoLandmarker) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    const baseOptions = {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU' as const,
      },
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };

    [imageLandmarker, videoLandmarker] = await Promise.all([
      PoseLandmarker.createFromOptions(vision, { ...baseOptions, runningMode: 'IMAGE' }),
      PoseLandmarker.createFromOptions(vision, { ...baseOptions, runningMode: 'VIDEO' }),
    ]);
  })();

  return initPromise;
}

export function isReady(): boolean {
  return imageLandmarker !== null && videoLandmarker !== null;
}

function toFrame(
  landmarks: { x: number; y: number; z: number; visibility?: number }[],
  frameNumber: number,
  timestampMs: number,
): PoseFrame {
  const mapped: Landmark[] = landmarks.map((lm, i) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z,
    visibility: lm.visibility ?? 1.0,
    body_part: BODY_PART_NAMES[i] ?? null,
  }));
  const avgVis = mapped.reduce((s, l) => s + l.visibility, 0) / mapped.length;
  return { landmarks: mapped, timestamp_ms: timestampMs, frame_number: frameNumber, confidence: avgVis };
}

// For single images / video frame seeking (AnalyzePage)
export function detectImage(
  source: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement,
  frameNumber = 0,
  timestampMs = 0,
): PoseFrame | null {
  if (!imageLandmarker) return null;
  try {
    const result = imageLandmarker.detect(source);
    const lms = result.landmarks?.[0];
    if (!lms?.length) return null;
    return toFrame(lms, frameNumber, timestampMs);
  } catch {
    return null;
  }
}

// For live video stream (RecordPage) - uses temporal tracking
export function detectForVideo(
  video: HTMLVideoElement,
  frameNumber = 0,
  timestampMs?: number,
): PoseFrame | null {
  if (!videoLandmarker) return null;
  const ts = timestampMs ?? performance.now();
  try {
    const result = videoLandmarker.detectForVideo(video, ts);
    const lms = result.landmarks?.[0];
    if (!lms?.length) return null;
    return toFrame(lms, frameNumber, ts);
  } catch {
    return null;
  }
}
