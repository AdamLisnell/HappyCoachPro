"""
Analysis API Schemas

Pydantic models for swing analysis API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class GolfClubEnum(str, Enum):
    """Golf club types for API."""
    DRIVER = "driver"
    WOOD_3 = "wood_3"
    WOOD_5 = "wood_5"
    HYBRID = "hybrid"
    IRON_4 = "iron_4"
    IRON_5 = "iron_5"
    IRON_6 = "iron_6"
    IRON_7 = "iron_7"
    IRON_8 = "iron_8"
    IRON_9 = "iron_9"
    PITCHING_WEDGE = "pitching_wedge"
    SAND_WEDGE = "sand_wedge"
    LOB_WEDGE = "lob_wedge"
    PUTTER = "putter"


class SwingPhaseEnum(str, Enum):
    """Swing phases for API."""
    ADDRESS = "address"
    BACKSWING = "backswing"
    TOP = "top"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"


class SwingAnglesSchema(BaseModel):
    """
    Biomechanical angles measured during swing.
    
    All angles in degrees. None means couldn't be calculated.
    """
    spine_angle: Optional[float] = Field(default=None, description="Forward spine tilt (degrees)")
    spine_lateral: Optional[float] = Field(default=None, description="Side bend (degrees)")
    shoulder_rotation: Optional[float] = Field(default=None, description="Shoulder turn (degrees)")
    hip_rotation: Optional[float] = Field(default=None, description="Hip turn (degrees)")
    hip_sway: Optional[float] = Field(default=None, description="Lateral hip movement")
    left_elbow: Optional[float] = Field(default=None, description="Lead arm elbow angle")
    right_elbow: Optional[float] = Field(default=None, description="Trail arm elbow angle")
    left_knee: Optional[float] = Field(default=None, description="Lead knee flex")
    right_knee: Optional[float] = Field(default=None, description="Trail knee flex")
    wrist_hinge: Optional[float] = Field(default=None, description="Wrist cock angle")
    x_factor: Optional[float] = Field(default=None, description="Shoulder-hip separation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "spine_angle": 35.5,
                "shoulder_rotation": 85.0,
                "hip_rotation": 40.0,
                "left_elbow": 175.0,
                "x_factor": 45.0
            }
        }


class SwingScoreSchema(BaseModel):
    """
    Score for a specific aspect of the swing.
    """
    score: int = Field(..., ge=0, le=100, description="Score out of 100")
    grade: str = Field(..., description="Letter grade (A-F)")
    feedback: str = Field(..., description="Human-readable feedback")
    details: Optional[str] = Field(None, description="Additional details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 85,
                "grade": "B",
                "feedback": "Good spine angle maintained throughout swing",
                "details": "Spine angle range: 32° - 38°"
            }
        }


class CoachingTipSchema(BaseModel):
    """
    Actionable coaching advice.
    """
    category: str = Field(..., description="Aspect of swing (e.g., 'Setup', 'Impact')")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")
    title: str = Field(..., description="Short summary")
    description: str = Field(..., description="Detailed explanation")
    drill: Optional[str] = Field(None, description="Practice drill to fix issue")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Setup",
                "priority": 1,
                "title": "Improve Address Position",
                "description": "Your spine angle is too upright at address",
                "drill": "Practice your setup in front of a mirror"
            }
        }


class PhaseAnalysisSchema(BaseModel):
    """
    Analysis result for a single swing phase.
    """
    phase: SwingPhaseEnum = Field(..., description="Swing phase")
    timestamp_ms: int = Field(..., description="Video timestamp")
    frame_number: int = Field(..., description="Frame number")
    angles: SwingAnglesSchema = Field(..., description="Measured angles")
    score: int = Field(..., ge=0, le=100, description="Phase score")
    feedback: str = Field(..., description="Phase-specific feedback")


class SwingAnalysisResponse(BaseModel):
    """
    Complete swing analysis result.
    
    This is the main response from the analyze endpoint.
    """
    # Identification
    id: str = Field(..., description="Unique analysis ID")
    timestamp: datetime = Field(..., description="When analysis was performed")
    
    # Video info
    video_duration_ms: int = Field(..., description="Video duration in milliseconds")
    total_frames: int = Field(..., description="Total frames analyzed")
    fps: float = Field(..., description="Video frames per second")
    
    # Club
    club: GolfClubEnum = Field(..., description="Golf club used")
    
    # Scores
    overall_score: int = Field(..., ge=0, le=100, description="Overall swing score")
    posture_score: Optional[SwingScoreSchema] = Field(None, description="Posture analysis")
    tempo_score: Optional[SwingScoreSchema] = Field(None, description="Tempo analysis")
    rotation_score: Optional[SwingScoreSchema] = Field(None, description="Rotation analysis")
    balance_score: Optional[SwingScoreSchema] = Field(None, description="Balance analysis")
    
    # Phase details
    phases: List[PhaseAnalysisSchema] = Field(default_factory=list, description="Per-phase analysis")
    
    # Coaching
    tips: List[CoachingTipSchema] = Field(default_factory=list, description="Improvement tips")
    summary: str = Field(..., description="Text summary of analysis")
    
    # Key frames for visualization
    key_frames: dict[str, int] = Field(default_factory=dict, description="Phase -> frame number mapping")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z",
                "video_duration_ms": 5000,
                "total_frames": 150,
                "fps": 30.0,
                "club": "iron_7",
                "overall_score": 78,
                "summary": "Your iron_7 swing scored 78/100 - good."
            }
        }


class AnalyzeVideoRequest(BaseModel):
    """
    Request to analyze a video file.
    """
    video_base64: Optional[str] = Field(None, description="Base64 encoded video (for small files)")
    video_url: Optional[str] = Field(None, description="URL to video file")
    club: GolfClubEnum = Field(GolfClubEnum.IRON_7, description="Club being used")
    frame_skip: int = Field(1, ge=1, le=10, description="Process every Nth frame")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_base64": None,
                "video_url": "https://example.com/swing.mp4",
                "club": "driver",
                "frame_skip": 2
            }
        }


class AnalyzeFramesRequest(BaseModel):
    """
    Request to analyze pre-extracted pose frames.
    
    Used when frontend has already done pose detection.
    """
    frames: List[dict] = Field(..., description="List of pose frame data")
    club: GolfClubEnum = Field(GolfClubEnum.IRON_7, description="Club being used")
    fps: float = Field(30.0, description="Original video FPS")
    duration_ms: int = Field(0, description="Video duration in milliseconds")


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    mediapipe_available: bool = Field(..., description="Whether MediaPipe is working")