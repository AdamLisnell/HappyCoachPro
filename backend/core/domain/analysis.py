"""
Swing Analysis Domain Models

Data structures for representing golf swing analysis results,
including phases, scores, and coaching feedback.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class SwingPhase(Enum):
    """
    The four key phases of a golf swing.
    
    Each phase has specific biomechanical characteristics:
    - ADDRESS: Setup position, weight balanced
    - BACKSWING: Club moving back, shoulder rotation
    - TOP: Top of backswing, maximum coil
    - DOWNSWING: Transition and acceleration
    - IMPACT: Club meets ball
    - FOLLOW_THROUGH: After impact, deceleration
    - FINISH: Final balanced position
    """
    ADDRESS = "address"
    BACKSWING = "backswing"
    TOP = "top"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"
    
    # Special states
    UNKNOWN = "unknown"
    TRANSITIONING = "transitioning"


class GolfClub(Enum):
    """Golf club types - affects optimal swing characteristics."""
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


@dataclass
class SwingAngles:
    """
    Key angles measured during a swing phase.
    
    All angles are in degrees (0-180).
    None means the angle couldn't be calculated (landmarks not visible).
    """
    # Spine angles
    spine_angle: Optional[float] = None          # Forward tilt
    spine_lateral: Optional[float] = None        # Side bend
    
    # Shoulder rotation (relative to hips)
    shoulder_rotation: Optional[float] = None    # X-factor at top
    
    # Hip angles
    hip_rotation: Optional[float] = None         # Hip turn
    hip_sway: Optional[float] = None             # Lateral movement
    
    # Arm angles
    left_elbow: Optional[float] = None           # Lead arm bend
    right_elbow: Optional[float] = None          # Trail arm bend
    
    # Leg angles
    left_knee: Optional[float] = None            # Lead knee flex
    right_knee: Optional[float] = None           # Trail knee flex
    
    # Wrist (if detectable)
    wrist_hinge: Optional[float] = None          # Wrist cock angle


@dataclass
class SwingScore:
    """
    Scoring for a specific aspect of the swing.
    
    Attributes:
        score: 0-100 rating
        feedback: Human-readable explanation
        details: Specific measurements or observations
    """
    score: int
    feedback: str
    details: Optional[str] = None
    
    @property
    def grade(self) -> str:
        """Convert score to letter grade."""
        if self.score >= 90:
            return "A"
        elif self.score >= 80:
            return "B"
        elif self.score >= 70:
            return "C"
        elif self.score >= 60:
            return "D"
        else:
            return "F"


@dataclass
class CoachingTip:
    """
    A specific piece of coaching advice.
    
    Attributes:
        category: What aspect of the swing this addresses
        priority: 1 (highest) to 5 (lowest) importance
        title: Short summary
        description: Detailed explanation
        drill: Optional practice drill to fix the issue
    """
    category: str
    priority: int
    title: str
    description: str
    drill: Optional[str] = None


@dataclass
class PhaseAnalysis:
    """Analysis results for a single swing phase."""
    phase: SwingPhase
    timestamp_ms: int
    frame_number: int
    angles: SwingAngles
    score: int
    feedback: str


@dataclass
class SwingAnalysis:
    """
    Complete analysis of a golf swing.
    
    This is the main result object returned after analyzing a swing video.
    """
    # Identification
    id: str
    timestamp: datetime
    
    # Video info
    video_duration_ms: int
    total_frames: int
    fps: float
    
    # Club used
    club: GolfClub
    
    # Phase-by-phase analysis
    phases: list[PhaseAnalysis] = field(default_factory=list)
    
    # Overall scores
    overall_score: int = 0
    posture_score: Optional[SwingScore] = None
    tempo_score: Optional[SwingScore] = None
    rotation_score: Optional[SwingScore] = None
    balance_score: Optional[SwingScore] = None
    
    # Coaching
    tips: list[CoachingTip] = field(default_factory=list)
    summary: str = ""
    
    # Raw data (for visualization)
    key_frames: dict[SwingPhase, int] = field(default_factory=dict)
    
    def get_phase_analysis(self, phase: SwingPhase) -> Optional[PhaseAnalysis]:
        """Get analysis for a specific phase."""
        for p in self.phases:
            if p.phase == phase:
                return p
        return None
    
    @property
    def top_tips(self) -> list[CoachingTip]:
        """Get the 3 highest priority tips."""
        sorted_tips = sorted(self.tips, key=lambda t: t.priority)
        return sorted_tips[:3]