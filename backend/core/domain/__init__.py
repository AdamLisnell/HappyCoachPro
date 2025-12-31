"""
Domain Models

Pure data structures representing golf swing analysis concepts.
No external dependencies - just Python dataclasses and enums.
"""

from .pose import PoseLandmark, PoseFrame, BodyPart
from .analysis import SwingPhase, SwingAnalysis, SwingScore, CoachingTip

__all__ = [
    "PoseLandmark",
    "PoseFrame", 
    "BodyPart",
    "SwingPhase",
    "SwingAnalysis",
    "SwingScore",
    "CoachingTip",
]