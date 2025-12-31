"""
API Schemas

Pydantic models for request/response validation.
"""

from .pose import (
    LandmarkSchema,
    PoseFrameSchema,
    PoseDetectionRequest,
    PoseDetectionResponse,
    WebSocketMessageType,
    WebSocketMessage,
    FrameMessage,
    PoseResultMessage,
)

from .analysis import (
    GolfClubEnum,
    SwingPhaseEnum,
    SwingAnglesSchema,
    SwingScoreSchema,
    CoachingTipSchema,
    PhaseAnalysisSchema,
    SwingAnalysisResponse,
    AnalyzeVideoRequest,
    AnalyzeFramesRequest,
    HealthResponse,
)

__all__ = [
    # Pose schemas
    "LandmarkSchema",
    "PoseFrameSchema",
    "PoseDetectionRequest",
    "PoseDetectionResponse",
    "WebSocketMessageType",
    "WebSocketMessage",
    "FrameMessage",
    "PoseResultMessage",
    # Analysis schemas
    "GolfClubEnum",
    "SwingPhaseEnum",
    "SwingAnglesSchema",
    "SwingScoreSchema",
    "CoachingTipSchema",
    "PhaseAnalysisSchema",
    "SwingAnalysisResponse",
    "AnalyzeVideoRequest",
    "AnalyzeFramesRequest",
    "HealthResponse",
]