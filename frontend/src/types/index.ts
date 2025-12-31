/**
 * HappyCoach2 TypeScript Types
 * 
 * Shared type definitions matching backend API schemas.
 */

// =============================================================================
// Pose Types
// =============================================================================

export interface Landmark {
  x: number;        // 0.0 - 1.0 (normalized)
  y: number;        // 0.0 - 1.0 (normalized)
  z: number;        // depth
  visibility: number; // 0.0 - 1.0 confidence
  body_part: string | null;
}

export interface PoseFrame {
  landmarks: Landmark[];
  timestamp_ms: number;
  frame_number: number;
  confidence: number;
}

// =============================================================================
// Analysis Types
// =============================================================================

export type GolfClub = 
  | 'driver' 
  | 'wood_3' 
  | 'wood_5' 
  | 'hybrid'
  | 'iron_4' 
  | 'iron_5' 
  | 'iron_6' 
  | 'iron_7' 
  | 'iron_8' 
  | 'iron_9'
  | 'pitching_wedge' 
  | 'sand_wedge' 
  | 'lob_wedge' 
  | 'putter';

export type SwingPhase = 
  | 'address' 
  | 'backswing' 
  | 'top' 
  | 'downswing' 
  | 'impact' 
  | 'follow_through' 
  | 'finish';

export interface SwingAngles {
  spine_angle: number | null;
  spine_lateral: number | null;
  shoulder_rotation: number | null;
  hip_rotation: number | null;
  hip_sway: number | null;
  left_elbow: number | null;
  right_elbow: number | null;
  left_knee: number | null;
  right_knee: number | null;
  wrist_hinge: number | null;
  x_factor: number | null;
}

export interface SwingScore {
  score: number;
  grade: string;
  feedback: string;
  details: string | null;
}

export interface CoachingTip {
  category: string;
  priority: number;
  title: string;
  description: string;
  drill: string | null;
}

export interface PhaseAnalysis {
  phase: SwingPhase;
  timestamp_ms: number;
  frame_number: number;
  angles: SwingAngles;
  score: number;
  feedback: string;
}

export interface SwingAnalysis {
  id: string;
  timestamp: string;
  video_duration_ms: number;
  total_frames: number;
  fps: number;
  club: GolfClub;
  overall_score: number;
  posture_score: SwingScore | null;
  tempo_score: SwingScore | null;
  rotation_score: SwingScore | null;
  balance_score: SwingScore | null;
  phases: PhaseAnalysis[];
  tips: CoachingTip[];
  summary: string;
  key_frames: Record<string, number>;
}

// =============================================================================
// WebSocket Types
// =============================================================================

export type WebSocketMessageType = 
  | 'frame' 
  | 'start_session' 
  | 'end_session'
  | 'pose_result' 
  | 'error' 
  | 'session_started' 
  | 'session_ended';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  data: Record<string, unknown>;
  timestamp: number;
}

export interface PoseResultData {
  frame_number: number;
  pose: PoseFrame | null;
  processing_time_ms: number;
}

// =============================================================================
// API Types
// =============================================================================

export interface HealthResponse {
  status: string;
  version: string;
  mediapipe_available: boolean;
}

export interface PoseDetectionResponse {
  success: boolean;
  pose: PoseFrame | null;
  error: string | null;
  processing_time_ms: number;
}

// =============================================================================
// App State Types
// =============================================================================

export type AppView = 'record' | 'analysis' | 'history' | 'settings';

export interface RecordingState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  videoBlob: Blob | null;
  videoUrl: string | null;
}

export interface AnalysisState {
  isAnalyzing: boolean;
  progress: number;
  result: SwingAnalysis | null;
  error: string | null;
}