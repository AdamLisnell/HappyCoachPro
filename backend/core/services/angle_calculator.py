"""
Angle Calculator Service

Mathematical calculations for body angles used in golf swing analysis.
All angles are calculated in degrees (0-180).

This is pure mathematics - no external dependencies except numpy.
"""

import math
from typing import Optional, Tuple
import numpy as np

from ..domain.pose import PoseLandmark, PoseFrame, BodyPart
from ..domain.analysis import SwingAngles


class AngleCalculator:
    """
    Calculates biomechanical angles from pose landmarks.
    
    Golf-specific angles include:
    - Spine angle (forward tilt)
    - Shoulder rotation (X-factor)
    - Hip rotation
    - Elbow angles
    - Knee flex
    
    All methods are static - no state needed.
    """
    
    # -------------------------------------------------------------------------
    # Core Angle Calculations
    # -------------------------------------------------------------------------
    
    @staticmethod
    def calculate_angle(
        p1: PoseLandmark,
        p2: PoseLandmark,  # Vertex point
        p3: PoseLandmark
    ) -> Optional[float]:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        
        Uses the law of cosines to find the angle at the vertex (p2).
        
        Args:
            p1: First point
            p2: Vertex point (where angle is measured)
            p3: Third point
            
        Returns:
            Angle in degrees (0-180), or None if calculation fails
            
        Example:
            For elbow angle: shoulder -> elbow -> wrist
            angle = calculate_angle(shoulder, elbow, wrist)
        """
        # Check visibility
        if not (p1.is_visible() and p2.is_visible() and p3.is_visible()):
            return None
        
        try:
            # Vector from p2 to p1
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            
            # Vector from p2 to p3
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            # Clamp to valid range (handles floating point errors)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Convert to degrees
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
            
        except (ZeroDivisionError, ValueError):
            return None
    
    @staticmethod
    def calculate_angle_3d(
        p1: PoseLandmark,
        p2: PoseLandmark,
        p3: PoseLandmark
    ) -> Optional[float]:
        """
        Calculate 3D angle at p2 formed by p1-p2-p3.
        
        Same as calculate_angle but includes z-coordinate for depth.
        More accurate but requires good depth estimation from MediaPipe.
        """
        # Check visibility
        if not (p1.is_visible() and p2.is_visible() and p3.is_visible()):
            return None
        
        try:
            # 3D vectors
            v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return float(np.degrees(np.arccos(cos_angle)))
            
        except (ZeroDivisionError, ValueError):
            return None
    
    # -------------------------------------------------------------------------
    # Golf-Specific Angle Calculations
    # -------------------------------------------------------------------------
    
    @staticmethod
    def calculate_spine_angle(frame: PoseFrame) -> Optional[float]:
        """
        Calculate spine forward tilt angle.
        
        Measured as the angle between:
        - Vertical line (straight up)
        - Line from hip midpoint to shoulder midpoint
        
        Ideal address position: ~30-40 degrees forward tilt
        
        Returns:
            Spine angle in degrees (0 = standing straight, 90 = bent over)
        """
        left_shoulder = frame.get_landmark(BodyPart.LEFT_SHOULDER)
        right_shoulder = frame.get_landmark(BodyPart.RIGHT_SHOULDER)
        left_hip = frame.get_landmark(BodyPart.LEFT_HIP)
        right_hip = frame.get_landmark(BodyPart.RIGHT_HIP)
        
        # Check all landmarks exist and are visible
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
        
        # Type narrowing: after the check above, we know these are not None
        # But we need to explicitly check for Pylance
        if (left_shoulder is None or right_shoulder is None or 
            left_hip is None or right_hip is None):
            return None
            
        if not all([
            left_shoulder.is_visible(),
            right_shoulder.is_visible(),
            left_hip.is_visible(),
            right_hip.is_visible()
        ]):
            return None
        
        # Calculate midpoints
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        
        # Vector from hip to shoulder
        spine_vector = np.array([
            shoulder_mid_x - hip_mid_x,
            shoulder_mid_y - hip_mid_y
        ])
        
        # Vertical vector (pointing up, so negative y in image coordinates)
        vertical = np.array([0, -1])
        
        # Calculate angle
        cos_angle = np.dot(spine_vector, vertical) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cos_angle)))
    
    @staticmethod
    def calculate_shoulder_rotation(frame: PoseFrame) -> Optional[float]:
        """
        Calculate shoulder rotation relative to camera.
        
        Uses the x-distance between shoulders to estimate rotation.
        When shoulders are square to camera, distance is maximum.
        When rotated 90 degrees, shoulders appear on same x-coordinate.
        
        Returns:
            Rotation in degrees (0 = square, 90 = fully rotated)
        """
        left_shoulder = frame.get_landmark(BodyPart.LEFT_SHOULDER)
        right_shoulder = frame.get_landmark(BodyPart.RIGHT_SHOULDER)
        
        # Check landmarks exist
        if left_shoulder is None or right_shoulder is None:
            return None
            
        # Check visibility
        if not (left_shoulder.is_visible() and right_shoulder.is_visible()):
            return None
        
        # Calculate shoulder width (x-distance)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        # Normalize: assume max width is ~0.4 in normalized coordinates
        # This varies by person/camera distance, but gives reasonable estimate
        max_width = 0.4
        normalized_width = min(shoulder_width / max_width, 1.0)
        
        # Convert to rotation angle
        # width = max_width * cos(rotation)
        # rotation = arccos(width / max_width)
        rotation = np.degrees(np.arccos(normalized_width))
        
        return float(rotation)
    
    @staticmethod
    def calculate_hip_rotation(frame: PoseFrame) -> Optional[float]:
        """
        Calculate hip rotation relative to camera.
        
        Similar to shoulder rotation but for hips.
        X-factor = shoulder_rotation - hip_rotation
        
        Returns:
            Rotation in degrees (0 = square, 90 = fully rotated)
        """
        left_hip = frame.get_landmark(BodyPart.LEFT_HIP)
        right_hip = frame.get_landmark(BodyPart.RIGHT_HIP)
        
        # Check landmarks exist
        if left_hip is None or right_hip is None:
            return None
            
        # Check visibility
        if not (left_hip.is_visible() and right_hip.is_visible()):
            return None
        
        hip_width = abs(left_hip.x - right_hip.x)
        max_width = 0.3  # Hips typically narrower than shoulders in frame
        normalized_width = min(hip_width / max_width, 1.0)
        
        rotation = np.degrees(np.arccos(normalized_width))
        
        return float(rotation)
    
    @staticmethod
    def calculate_x_factor(frame: PoseFrame) -> Optional[float]:
        """
        Calculate X-Factor (shoulder-hip separation).
        
        X-Factor is the difference between shoulder rotation and hip rotation.
        Key metric at top of backswing - more separation = more power potential.
        
        Pro golfers typically achieve 45-55 degrees at top of backswing.
        
        Returns:
            X-Factor in degrees
        """
        shoulder_rot = AngleCalculator.calculate_shoulder_rotation(frame)
        hip_rot = AngleCalculator.calculate_hip_rotation(frame)
        
        if shoulder_rot is None or hip_rot is None:
            return None
            
        return abs(shoulder_rot - hip_rot)
    
    @staticmethod
    def calculate_elbow_angle(
        frame: PoseFrame,
        side: str = "left"
    ) -> Optional[float]:
        """
        Calculate elbow bend angle.
        
        Args:
            frame: Pose frame with landmarks
            side: "left" or "right"
            
        Returns:
            Elbow angle in degrees (180 = straight arm, 90 = right angle)
        """
        if side == "left":
            shoulder = frame.get_landmark(BodyPart.LEFT_SHOULDER)
            elbow = frame.get_landmark(BodyPart.LEFT_ELBOW)
            wrist = frame.get_landmark(BodyPart.LEFT_WRIST)
        else:
            shoulder = frame.get_landmark(BodyPart.RIGHT_SHOULDER)
            elbow = frame.get_landmark(BodyPart.RIGHT_ELBOW)
            wrist = frame.get_landmark(BodyPart.RIGHT_WRIST)
        
        # Check all landmarks exist before calling calculate_angle
        if shoulder is None or elbow is None or wrist is None:
            return None
        
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_knee_angle(
        frame: PoseFrame,
        side: str = "left"
    ) -> Optional[float]:
        """
        Calculate knee flex angle.
        
        Args:
            frame: Pose frame with landmarks
            side: "left" or "right"
            
        Returns:
            Knee angle in degrees (180 = straight leg, 90 = deep squat)
        """
        if side == "left":
            hip = frame.get_landmark(BodyPart.LEFT_HIP)
            knee = frame.get_landmark(BodyPart.LEFT_KNEE)
            ankle = frame.get_landmark(BodyPart.LEFT_ANKLE)
        else:
            hip = frame.get_landmark(BodyPart.RIGHT_HIP)
            knee = frame.get_landmark(BodyPart.RIGHT_KNEE)
            ankle = frame.get_landmark(BodyPart.RIGHT_ANKLE)
        
        # Check all landmarks exist before calling calculate_angle
        if hip is None or knee is None or ankle is None:
            return None
        
        return AngleCalculator.calculate_angle(hip, knee, ankle)
    
    # -------------------------------------------------------------------------
    # Complete Frame Analysis
    # -------------------------------------------------------------------------
    
    @classmethod
    def calculate_all_angles(cls, frame: PoseFrame) -> SwingAngles:
        """
        Calculate all golf-relevant angles for a frame.
        
        This is the main method to call for complete angle analysis.
        
        Args:
            frame: PoseFrame with all landmarks
            
        Returns:
            SwingAngles with all calculated values (None for any that failed)
        """
        return SwingAngles(
            spine_angle=cls.calculate_spine_angle(frame),
            spine_lateral=None,  # TODO: Implement lateral bend
            shoulder_rotation=cls.calculate_shoulder_rotation(frame),
            hip_rotation=cls.calculate_hip_rotation(frame),
            hip_sway=None,  # TODO: Implement sway detection
            left_elbow=cls.calculate_elbow_angle(frame, "left"),
            right_elbow=cls.calculate_elbow_angle(frame, "right"),
            left_knee=cls.calculate_knee_angle(frame, "left"),
            right_knee=cls.calculate_knee_angle(frame, "right"),
            wrist_hinge=None,  # TODO: Implement wrist angle
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def calculate_distance(p1: Optional[PoseLandmark], p2: Optional[PoseLandmark]) -> Optional[float]:
        """Calculate 2D distance between two landmarks."""
        if p1 is None or p2 is None:
            return None
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    @staticmethod
    def calculate_midpoint(
        p1: Optional[PoseLandmark], 
        p2: Optional[PoseLandmark]
    ) -> Optional[Tuple[float, float]]:
        """Calculate midpoint between two landmarks."""
        if p1 is None or p2 is None:
            return None
        return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)