"""
Pose Detector Service

Wrapper around MediaPipe Pose for detecting body landmarks in images/video.
Handles all MediaPipe-specific logic and converts to our domain models.

Note: MediaPipe 0.10.30+ uses a new Tasks API. This module supports both
the legacy solutions API and the new Tasks API.
"""

import cv2
import numpy as np
from typing import Optional, List, Generator, Any
from pathlib import Path

from ..domain.pose import PoseLandmark, PoseFrame, BodyPart


class PoseDetector:
    """
    Detects human body pose using MediaPipe Pose.
    
    MediaPipe Pose provides 33 body landmarks with 3D coordinates.
    This class wraps MediaPipe and converts results to our domain models.
    
    Usage:
        detector = PoseDetector()
        
        # Single image
        frame = detector.detect_pose(image)
        
        # Video file
        for frame in detector.process_video("swing.mp4"):
            print(frame.landmarks)
        
        # Cleanup
        detector.close()
    
    Or use as context manager:
        with PoseDetector() as detector:
            frame = detector.detect_pose(image)
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
    ):
        """
        Initialize the pose detector.
        
        Args:
            model_complexity: 0, 1, or 2. Higher = more accurate but slower.
                             1 is good balance for golf analysis.
            min_detection_confidence: Minimum confidence for person detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            enable_segmentation: Whether to output segmentation mask.
        """
        self._pose = None
        self._use_legacy_api = False
        
        # Try new Tasks API first (MediaPipe 0.10.30+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if needed - use pose landmarker
            base_options = python.BaseOptions(
                model_asset_path=self._get_model_path()
            )
            
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            
            self._pose = vision.PoseLandmarker.create_from_options(options)
            self._use_legacy_api = False
            
        except (ImportError, Exception) as e:
            # Fall back to legacy solutions API
            try:
                import mediapipe as mp
                self._mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
                self._pose = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=enable_segmentation,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._use_legacy_api = True
            except Exception as e2:
                raise RuntimeError(
                    f"Could not initialize MediaPipe. "
                    f"Tasks API error: {e}, Legacy API error: {e2}"
                )
    
    def _get_model_path(self) -> str:
        """Get or download the pose landmarker model."""
        import os
        import urllib.request
        
        # Model URL from MediaPipe
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        
        # Save to cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "happycoach")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "pose_landmarker.task")
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe pose model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete!")
        
        return model_path
    
    def __enter__(self) -> "PoseDetector":
        """Context manager entry."""
        return self
    
    def __exit__(
        self, 
        exc_type: Optional[type], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[Any]
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.close()
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            if self._use_legacy_api:
                try:
                    self._pose.close()
                except AttributeError:
                    pass  # Some versions don't have close()
            # Tasks API doesn't need explicit close
            self._pose = None
    # -------------------------------------------------------------------------
    # Core Detection Methods
    # -------------------------------------------------------------------------
    
    def detect_pose(
        self,
        image: np.ndarray,
        timestamp_ms: int = 0,
        frame_number: int = 0
    ) -> Optional[PoseFrame]:
        """
        Detect pose in a single image.
        
        Args:
            image: BGR image (OpenCV format) or RGB image
            timestamp_ms: Timestamp in milliseconds (for video frames)
            frame_number: Sequential frame number
            
        Returns:
            PoseFrame with 33 landmarks, or None if no person detected
        """
        # Convert BGR to RGB if needed (MediaPipe expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        if self._use_legacy_api:
            return self._detect_legacy(image_rgb, timestamp_ms, frame_number)
        else:
            return self._detect_tasks(image_rgb, timestamp_ms, frame_number)
    
    def _detect_legacy(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int,
        frame_number: int
    ) -> Optional[PoseFrame]:
        """Detect using legacy solutions API."""
        if self._pose is None:
            return None
        
        results = self._pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = self._convert_legacy_landmarks(results.pose_landmarks.landmark)
        avg_confidence = sum(lm.visibility for lm in landmarks) / len(landmarks)
        
        return PoseFrame(
            landmarks=landmarks,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            confidence=avg_confidence,
        )
    
    def _detect_tasks(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int,
        frame_number: int
    ) -> Optional[PoseFrame]:
        """Detect using new Tasks API."""
        if self._pose is None:
            return None
        
        import mediapipe as mp
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        results = self._pose.detect(mp_image)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        # Get first person's landmarks
        pose_landmarks = results.pose_landmarks[0]
        
        landmarks = self._convert_tasks_landmarks(pose_landmarks)
        avg_confidence = sum(lm.visibility for lm in landmarks) / len(landmarks)
        
        return PoseFrame(
            landmarks=landmarks,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            confidence=avg_confidence,
        )
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ) -> Generator[PoseFrame, None, None]:
        """
        Process a video file and yield pose frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            frame_skip: Process every Nth frame (1 = all, 2 = every other, etc.)
            
        Yields:
            PoseFrame for each processed frame where pose was detected
            
        Example:
            for frame in detector.process_video("swing.mp4", frame_skip=2):
                angles = AngleCalculator.calculate_all_angles(frame)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Calculate timestamp
                timestamp_ms = int((frame_count / fps) * 1000) if fps > 0 else 0
                
                # Detect pose
                pose_frame = self.detect_pose(
                    frame,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame_count
                )
                
                if pose_frame:
                    yield pose_frame
                    processed_count += 1
                
                frame_count += 1
                
                # Check max frames
                if max_frames and processed_count >= max_frames:
                    break
                    
        finally:
            cap.release()
    
    def process_video_to_list(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ) -> List[PoseFrame]:
        """
        Process video and return all frames as a list.
        
        Convenience method when you need all frames in memory.
        For large videos, use process_video() generator instead.
        """
        return list(self.process_video(video_path, max_frames, frame_skip))
    
    # -------------------------------------------------------------------------
    # Frame Extraction (for WebSocket streaming)
    # -------------------------------------------------------------------------
    
    def detect_from_base64(
        self,
        base64_image: str,
        timestamp_ms: int = 0,
        frame_number: int = 0
    ) -> Optional[PoseFrame]:
        """
        Detect pose from a base64-encoded image.
        
        Used for WebSocket communication with frontend.
        
        Args:
            base64_image: Base64 encoded JPEG/PNG image
            timestamp_ms: Timestamp in milliseconds
            frame_number: Frame number
            
        Returns:
            PoseFrame or None if detection failed
        """
        import base64
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        return self.detect_pose(image, timestamp_ms, frame_number)
    
    # -------------------------------------------------------------------------
    # Visualization (for debugging)
    # -------------------------------------------------------------------------
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        pose_frame: PoseFrame,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw pose landmarks on image.
        
        Useful for debugging and visualization.
        
        Args:
            image: Original image (will be modified)
            pose_frame: Detected pose
            draw_connections: Whether to draw lines between landmarks
            
        Returns:
            Image with landmarks drawn
        """
        h, w = image.shape[:2]
        
        # Draw each landmark as a circle
        for i, landmark in enumerate(pose_frame.landmarks):
            if landmark.is_visible(0.3):
                x, y = landmark.to_pixel(w, h)
                
                # Color based on visibility (green = high, red = low)
                color_intensity = int(landmark.visibility * 255)
                color = (0, color_intensity, 255 - color_intensity)
                
                cv2.circle(image, (x, y), 5, color, -1)
        
        # Draw connections
        if draw_connections:
            connections = self._get_pose_connections()
            
            for start_idx, end_idx in connections:
                start = pose_frame.landmarks[start_idx]
                end = pose_frame.landmarks[end_idx]
                
                if start.is_visible(0.3) and end.is_visible(0.3):
                    start_point = start.to_pixel(w, h)
                    end_point = end.to_pixel(w, h)
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        return image
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _convert_legacy_landmarks(self, mp_landmarks: Any) -> List[PoseLandmark]:
        """Convert legacy MediaPipe landmarks to our domain model."""
        landmarks = []
        
        for i, mp_lm in enumerate(mp_landmarks):
            try:
                body_part = BodyPart(i)
            except ValueError:
                body_part = None
            
            landmark = PoseLandmark(
                x=mp_lm.x,
                y=mp_lm.y,
                z=mp_lm.z,
                visibility=mp_lm.visibility,
                body_part=body_part,
            )
            landmarks.append(landmark)
        
        return landmarks
    
    def _convert_tasks_landmarks(self, pose_landmarks: Any) -> List[PoseLandmark]:
        """Convert Tasks API landmarks to our domain model."""
        landmarks = []
        
        for i, lm in enumerate(pose_landmarks):
            try:
                body_part = BodyPart(i)
            except ValueError:
                body_part = None
            
            landmark = PoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else lm.presence,
                body_part=body_part,
            )
            landmarks.append(landmark)
        
        return landmarks
    
    def _get_pose_connections(self) -> List[tuple[int, int]]:
        """Get landmark connection pairs for drawing skeleton."""
        return [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            
            # Body
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Torso sides
            (23, 24),  # Hips
            
            # Left arm
            (11, 13), (13, 15),
            
            # Right arm
            (12, 14), (14, 16),
            
            # Left leg
            (23, 25), (25, 27),
            
            # Right leg
            (24, 26), (26, 28),
            
            # Feet
            (27, 29), (29, 31),
            (28, 30), (30, 32),
        ]


# =============================================================================
# Convenience function for quick usage
# =============================================================================

def detect_pose_in_image(image_path: str) -> Optional[PoseFrame]:
    """
    Quick function to detect pose in a single image file.
    
    Usage:
        frame = detect_pose_in_image("golfer.jpg")
        if frame:
            print(f"Detected {len(frame.get_visible_landmarks())} visible landmarks")
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    with PoseDetector() as detector:
        return detector.detect_pose(image)