"""
Pose Domain Models

Data structures for representing human body pose landmarks
detected by MediaPipe.

MediaPipe Pose returns 33 landmarks:
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class BodyPart(IntEnum):
    """
    MediaPipe Pose landmark indices.
    
    These map directly to MediaPipe's 33-point pose model.
    We include the most relevant ones for golf swing analysis.
    """
    # Face
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    
    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    # Hands (for club tracking)
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    
    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Feet
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class PoseLandmark:
    """
    A single body landmark with 3D coordinates and visibility.
    
    Attributes:
        x: Horizontal position (0.0 = left edge, 1.0 = right edge)
        y: Vertical position (0.0 = top edge, 1.0 = bottom edge)
        z: Depth (smaller = closer to camera)
        visibility: Confidence score (0.0 to 1.0)
        body_part: Which body part this landmark represents
    
    Note:
        Coordinates are normalized to image dimensions.
        To get pixel coordinates: pixel_x = x * image_width
    """
    x: float
    y: float
    z: float
    visibility: float
    body_part: Optional[BodyPart] = None
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        """Check if landmark is visible above confidence threshold."""
        return self.visibility >= threshold
    
    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))
    
    def distance_to(self, other: "PoseLandmark") -> float:
        """Calculate Euclidean distance to another landmark."""
        return (
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        ) ** 0.5


@dataclass
class PoseFrame:
    """
    A complete pose detection result for a single video frame.
    
    Attributes:
        landmarks: List of 33 body landmarks
        timestamp_ms: Video timestamp in milliseconds
        frame_number: Sequential frame number
        confidence: Overall detection confidence
    """
    landmarks: list[PoseLandmark]
    timestamp_ms: int
    frame_number: int
    confidence: float
    
    def get_landmark(self, body_part: BodyPart) -> Optional[PoseLandmark]:
        """Get a specific landmark by body part."""
        index = body_part.value
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def get_visible_landmarks(self, threshold: float = 0.5) -> list[PoseLandmark]:
        """Get all landmarks above visibility threshold."""
        return [lm for lm in self.landmarks if lm.is_visible(threshold)]
    
    # -------------------------------------------------------------------------
    # Convenience methods for common landmark groups
    # -------------------------------------------------------------------------
    
    @property
    def left_arm(self) -> tuple[Optional[PoseLandmark], ...]:
        """Get left arm landmarks (shoulder, elbow, wrist)."""
        return (
            self.get_landmark(BodyPart.LEFT_SHOULDER),
            self.get_landmark(BodyPart.LEFT_ELBOW),
            self.get_landmark(BodyPart.LEFT_WRIST),
        )
    
    @property
    def right_arm(self) -> tuple[Optional[PoseLandmark], ...]:
        """Get right arm landmarks (shoulder, elbow, wrist)."""
        return (
            self.get_landmark(BodyPart.RIGHT_SHOULDER),
            self.get_landmark(BodyPart.RIGHT_ELBOW),
            self.get_landmark(BodyPart.RIGHT_WRIST),
        )
    
    @property
    def spine(self) -> tuple[Optional[PoseLandmark], ...]:
        """Get spine-related landmarks (nose, shoulders midpoint, hips midpoint)."""
        return (
            self.get_landmark(BodyPart.NOSE),
            self.get_landmark(BodyPart.LEFT_SHOULDER),
            self.get_landmark(BodyPart.RIGHT_SHOULDER),
            self.get_landmark(BodyPart.LEFT_HIP),
            self.get_landmark(BodyPart.RIGHT_HIP),
        )
    
    @property
    def left_leg(self) -> tuple[Optional[PoseLandmark], ...]:
        """Get left leg landmarks (hip, knee, ankle)."""
        return (
            self.get_landmark(BodyPart.LEFT_HIP),
            self.get_landmark(BodyPart.LEFT_KNEE),
            self.get_landmark(BodyPart.LEFT_ANKLE),
        )
    
    @property
    def right_leg(self) -> tuple[Optional[PoseLandmark], ...]:
        """Get right leg landmarks (hip, knee, ankle)."""
        return (
            self.get_landmark(BodyPart.RIGHT_HIP),
            self.get_landmark(BodyPart.RIGHT_KNEE),
            self.get_landmark(BodyPart.RIGHT_ANKLE),
        )