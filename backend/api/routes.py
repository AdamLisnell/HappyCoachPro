"""
REST API Routes

FastAPI routes for golf swing analysis.
Handles HTTP requests for pose detection and swing analysis.
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .schemas import (
    PoseDetectionRequest,
    PoseDetectionResponse,
    PoseFrameSchema,
    LandmarkSchema,
    AnalyzeVideoRequest,
    SwingAnalysisResponse,
    SwingScoreSchema,
    CoachingTipSchema,
    PhaseAnalysisSchema,
    SwingAnglesSchema,
    GolfClubEnum,
    SwingPhaseEnum,
    HealthResponse,
)
from core.services import PoseDetector, SwingAnalyzer
from core.domain.analysis import GolfClub, SwingPhase

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check() -> HealthResponse:
    """
    Check if the API is running and MediaPipe is available.
    
    Returns:
        Health status and version information
    """
    # Test MediaPipe availability
    mediapipe_ok = False
    try:
        with PoseDetector() as detector:
            mediapipe_ok = True
    except Exception as e:
        logger.warning(f"MediaPipe not available: {e}")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        mediapipe_available=mediapipe_ok
    )


# =============================================================================
# Pose Detection
# =============================================================================

@router.post(
    "/pose/detect",
    response_model=PoseDetectionResponse,
    tags=["Pose Detection"],
    summary="Detect pose in a single image"
)
async def detect_pose(request: PoseDetectionRequest) -> PoseDetectionResponse:
    """
    Detect human pose in a base64-encoded image.
    
    This endpoint is useful for:
    - Testing pose detection on single images
    - Non-real-time analysis
    
    For real-time analysis, use the WebSocket endpoint instead.
    
    Args:
        request: Image data and optional metadata
        
    Returns:
        Detected pose with 33 landmarks, or error if detection failed
    """
    start_time = time.time()
    
    try:
        with PoseDetector() as detector:
            pose_frame = detector.detect_from_base64(
                request.image_base64,
                timestamp_ms=request.timestamp_ms,
                frame_number=request.frame_number
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        if pose_frame is None:
            return PoseDetectionResponse(
                success=False,
                pose=None,
                error="No person detected in image",
                processing_time_ms=processing_time
            )
        
        # Convert to API schema
        landmarks = [
            LandmarkSchema(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
                body_part=lm.body_part.name if lm.body_part else None
            )
            for lm in pose_frame.landmarks
        ]
        
        pose_schema = PoseFrameSchema(
            landmarks=landmarks,
            timestamp_ms=pose_frame.timestamp_ms,
            frame_number=pose_frame.frame_number,
            confidence=pose_frame.confidence
        )
        
        return PoseDetectionResponse(
            success=True,
            pose=pose_schema,
            error=None,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Pose detection failed: {e}")
        processing_time = (time.time() - start_time) * 1000
        return PoseDetectionResponse(
            success=False,
            pose=None,
            error=str(e),
            processing_time_ms=processing_time
        )


# =============================================================================
# Swing Analysis
# =============================================================================

@router.post(
    "/analysis/video",
    response_model=SwingAnalysisResponse,
    tags=["Swing Analysis"],
    summary="Analyze a golf swing video"
)
async def analyze_video(
    video: UploadFile = File(..., description="Video file (MP4, MOV)"),
    club: GolfClubEnum = Form(GolfClubEnum.IRON_7, description="Golf club used"),
    frame_skip: int = Form(1, ge=1, le=10, description="Process every Nth frame")
) -> SwingAnalysisResponse:
    """
    Analyze a golf swing from an uploaded video file.
    
    The video will be:
    1. Saved temporarily
    2. Processed frame-by-frame with MediaPipe
    3. Analyzed for swing phases and angles
    4. Scored and given coaching tips
    
    Args:
        video: Video file upload
        club: Type of golf club being used
        frame_skip: Skip frames for faster processing (1 = all frames)
        
    Returns:
        Complete swing analysis with scores and tips
    """
    import tempfile
    import os
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        # Create temp file with correct extension
        suffix = os.path.splitext(video.filename or ".mp4")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content = await video.read()
            temp_file.write(content)
        
        # Convert API enum to domain enum
        domain_club = GolfClub(club.value)
        
        # Analyze the video
        with SwingAnalyzer() as analyzer:
            result = analyzer.analyze_video(
                temp_path,
                club=domain_club,
                frame_skip=frame_skip
            )
        
        # Convert domain model to API response
        return _convert_analysis_to_response(result)
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post(
    "/analysis/quick",
    response_model=SwingAnalysisResponse,
    tags=["Swing Analysis"],
    summary="Quick analysis from pre-detected poses"
)
async def analyze_quick(
    club: GolfClubEnum = GolfClubEnum.IRON_7,
) -> SwingAnalysisResponse:
    """
    Quick demo analysis without video upload.
    
    Returns a sample analysis result for testing the API.
    Useful for frontend development and testing.
    """
    from datetime import datetime
    import uuid
    
    # Return demo data
    return SwingAnalysisResponse(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        video_duration_ms=5000,
        total_frames=150,
        fps=30.0,
        club=club,
        overall_score=78,
        posture_score=SwingScoreSchema(
            score=82,
            grade="B",
            feedback="Good spine angle maintained",
            details="Spine angle: 33° - 38°"
        ),
        tempo_score=SwingScoreSchema(
            score=75,
            grade="C",
            feedback="Backswing slightly rushed",
            details="Ratio: 2.5:1"
        ),
        rotation_score=SwingScoreSchema(
            score=80,
            grade="B",
            feedback="Good X-factor at top",
            details="X-factor: 42°"
        ),
        balance_score=SwingScoreSchema(
            score=76,
            grade="C",
            feedback="Minor balance shift detected",
            details="Avg knee flex: 162°"
        ),
        phases=[
            PhaseAnalysisSchema(
                phase=SwingPhaseEnum.ADDRESS,
                timestamp_ms=0,
                frame_number=0,
                angles=SwingAnglesSchema(spine_angle=35.0, left_knee=165.0),
                score=85,
                feedback="Good setup position"
            ),
            PhaseAnalysisSchema(
                phase=SwingPhaseEnum.TOP,
                timestamp_ms=1500,
                frame_number=45,
                angles=SwingAnglesSchema(shoulder_rotation=88.0, hip_rotation=45.0),
                score=80,
                feedback="Full shoulder turn achieved"
            ),
            PhaseAnalysisSchema(
                phase=SwingPhaseEnum.IMPACT,
                timestamp_ms=2500,
                frame_number=75,
                angles=SwingAnglesSchema(hip_rotation=55.0, left_elbow=172.0),
                score=75,
                feedback="Hips could be more open"
            ),
            PhaseAnalysisSchema(
                phase=SwingPhaseEnum.FINISH,
                timestamp_ms=4000,
                frame_number=120,
                angles=SwingAnglesSchema(shoulder_rotation=95.0),
                score=78,
                feedback="Balanced finish"
            ),
        ],
        tips=[
            CoachingTipSchema(
                category="Tempo",
                priority=1,
                title="Slow down backswing",
                description="Your backswing is slightly rushed. Focus on a smoother takeaway.",
                drill="Practice with a metronome at 60 BPM"
            ),
            CoachingTipSchema(
                category="Impact",
                priority=2,
                title="Open hips earlier",
                description="Start hip rotation earlier in downswing for more power.",
                drill="Pause at top, then focus on hip bump before arms"
            ),
        ],
        summary=f"Your {club.value} swing scored 78/100 - good. Focus on tempo and hip rotation.",
        key_frames={
            "address": 0,
            "top": 45,
            "impact": 75,
            "finish": 120
        }
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_analysis_to_response(result) -> SwingAnalysisResponse:
    """Convert domain SwingAnalysis to API response schema."""
    from core.domain.analysis import SwingAnalysis
    
    # Convert phase analyses
    phases = []
    for pa in result.phases:
        phases.append(PhaseAnalysisSchema(
            phase=SwingPhaseEnum(pa.phase.value),
            timestamp_ms=pa.timestamp_ms,
            frame_number=pa.frame_number,
            angles=SwingAnglesSchema(
                spine_angle=pa.angles.spine_angle,
                spine_lateral=pa.angles.spine_lateral,
                shoulder_rotation=pa.angles.shoulder_rotation,
                hip_rotation=pa.angles.hip_rotation,
                hip_sway=pa.angles.hip_sway,
                left_elbow=pa.angles.left_elbow,
                right_elbow=pa.angles.right_elbow,
                left_knee=pa.angles.left_knee,
                right_knee=pa.angles.right_knee,
                wrist_hinge=pa.angles.wrist_hinge,
            ),
            score=pa.score,
            feedback=pa.feedback
        ))
    
    # Convert tips
    tips = []
    for tip in result.tips:
        tips.append(CoachingTipSchema(
            category=tip.category,
            priority=tip.priority,
            title=tip.title,
            description=tip.description,
            drill=tip.drill
        ))
    
    # Convert scores
    def convert_score(score) -> Optional[SwingScoreSchema]:
        if score is None:
            return None
        return SwingScoreSchema(
            score=score.score,
            grade=score.grade,
            feedback=score.feedback,
            details=score.details
        )
    
    return SwingAnalysisResponse(
        id=result.id,
        timestamp=result.timestamp,
        video_duration_ms=result.video_duration_ms,
        total_frames=result.total_frames,
        fps=result.fps,
        club=GolfClubEnum(result.club.value),
        overall_score=result.overall_score,
        posture_score=convert_score(result.posture_score),
        tempo_score=convert_score(result.tempo_score),
        rotation_score=convert_score(result.rotation_score),
        balance_score=convert_score(result.balance_score),
        phases=phases,
        tips=tips,
        summary=result.summary,
        key_frames={k.value: v for k, v in result.key_frames.items()}
    )