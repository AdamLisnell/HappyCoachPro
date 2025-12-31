"""
Pose API Schemas

Pydantic models for pose-related API requests and responses.
These define the JSON structure for communication with frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class LandmarkSchema(BaseModel):
    """
    Single body landmark in API response.
    
    Coordinates are normalized (0.0 to 1.0).
    Frontend multiplies by canvas dimensions to get pixel positions.
    """
    x: float = Field(..., ge=0.0, le=1.0, description="Horizontal position (0=left, 1=right)")
    y: float = Field(..., ge=0.0, le=1.0, description="Vertical position (0=top, 1=bottom)")
    z: float = Field(..., description="Depth (negative=closer to camera)")
    visibility: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    body_part: Optional[str] = Field(None, description="Body part name (e.g., 'LEFT_SHOULDER')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x": 0.45,
                "y": 0.32,
                "z": -0.15,
                "visibility": 0.95,
                "body_part": "LEFT_SHOULDER"
            }
        }


class PoseFrameSchema(BaseModel):
    """
    Complete pose detection result for one frame.
    
    Contains all 33 MediaPipe landmarks plus metadata.
    """
    landmarks: List[LandmarkSchema] = Field(..., description="33 body landmarks")
    timestamp_ms: int = Field(..., ge=0, description="Video timestamp in milliseconds")
    frame_number: int = Field(..., ge=0, description="Sequential frame number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall detection confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "landmarks": [
                    {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.99, "body_part": "NOSE"}
                ],
                "timestamp_ms": 1500,
                "frame_number": 45,
                "confidence": 0.92
            }
        }


class PoseDetectionRequest(BaseModel):
    """
    Request to detect pose in a base64-encoded image.
    
    Used for single-frame detection via REST API.
    """
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG image")
    timestamp_ms: int = Field(0, ge=0, description="Optional timestamp")
    frame_number: int = Field(0, ge=0, description="Optional frame number")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "/9j/4AAQSkZJRg...",
                "timestamp_ms": 0,
                "frame_number": 0
            }
        }


class PoseDetectionResponse(BaseModel):
    """
    Response from pose detection.
    """
    success: bool = Field(..., description="Whether detection succeeded")
    pose: Optional[PoseFrameSchema] = Field(None, description="Detected pose (null if no person found)")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Time taken to process in milliseconds")


# =============================================================================
# WebSocket Message Schemas
# =============================================================================

class WebSocketMessageType(str, Enum):
    """Types of WebSocket messages."""
    # Client -> Server
    FRAME = "frame"                    # Send video frame for analysis
    START_SESSION = "start_session"    # Start new analysis session
    END_SESSION = "end_session"        # End analysis session
    
    # Server -> Client
    POSE_RESULT = "pose_result"        # Pose detection result
    ERROR = "error"                    # Error message
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


class WebSocketMessage(BaseModel):
    """
    Base WebSocket message structure.
    
    All WebSocket communication uses this format.
    """
    type: WebSocketMessageType = Field(..., description="Message type")
    data: dict = Field(default_factory=dict, description="Message payload")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "frame",
                "data": {"image_base64": "..."},
                "timestamp": 1704067200000
            }
        }


class FrameMessage(BaseModel):
    """
    WebSocket message containing a video frame.
    
    Sent from frontend to backend for real-time pose detection.
    """
    image_base64: str = Field(..., description="Base64 encoded frame")
    frame_number: int = Field(0, description="Frame sequence number")


class PoseResultMessage(BaseModel):
    """
    WebSocket message containing pose detection result.
    
    Sent from backend to frontend after processing a frame.
    """
    frame_number: int = Field(..., description="Corresponding frame number")
    pose: Optional[PoseFrameSchema] = Field(None, description="Detected pose")
    processing_time_ms: float = Field(..., description="Processing time")