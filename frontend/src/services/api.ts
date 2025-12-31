/**
 * API Service
 * 
 * HTTP client for communicating with HappyCoach2 backend.
 */

import axios from 'axios';
import type { 
  HealthResponse, 
  SwingAnalysis, 
  PoseDetectionResponse,
  GolfClub 
} from '@/types';

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes for video analysis
  headers: {
    'Content-Type': 'application/json',
  },
});

// =============================================================================
// Health
// =============================================================================

export async function checkHealth(): Promise<HealthResponse> {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
}

// =============================================================================
// Pose Detection
// =============================================================================

export async function detectPose(
  imageBase64: string,
  timestampMs: number = 0,
  frameNumber: number = 0
): Promise<PoseDetectionResponse> {
  const response = await api.post<PoseDetectionResponse>('/pose/detect', {
    image_base64: imageBase64,
    timestamp_ms: timestampMs,
    frame_number: frameNumber,
  });
  return response.data;
}

// =============================================================================
// Swing Analysis
// =============================================================================

export async function analyzeVideo(
  videoFile: File,
  club: GolfClub = 'iron_7',
  frameSkip: number = 1,
  onProgress?: (progress: number) => void
): Promise<SwingAnalysis> {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('club', club);
  formData.append('frame_skip', String(frameSkip));

  const response = await api.post<SwingAnalysis>('/analysis/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  });

  return response.data;
}

export async function getQuickAnalysis(
  club: GolfClub = 'iron_7'
): Promise<SwingAnalysis> {
  const response = await api.post<SwingAnalysis>(`/analysis/quick?club=${club}`);
  return response.data;
}

// =============================================================================
// Export
// =============================================================================

export default api;