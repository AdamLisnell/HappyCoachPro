"""
Swing Analyzer Service

High-level service that orchestrates pose detection and angle calculation
to provide complete golf swing analysis.

This is the main entry point for analyzing golf swings.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from ..domain.pose import PoseFrame, BodyPart
from ..domain.analysis import (
    SwingPhase,
    SwingAnalysis,
    SwingAngles,
    SwingScore,
    CoachingTip,
    PhaseAnalysis,
    GolfClub,
)
from .pose_detector import PoseDetector
from .angle_calculator import AngleCalculator


class SwingAnalyzer:
    """
    Analyzes golf swings from video or frame sequences.
    
    This service:
    1. Processes video through PoseDetector
    2. Identifies swing phases (address, top, impact, finish)
    3. Calculates angles at each phase
    4. Scores the swing
    5. Generates coaching tips
    
    Usage:
        analyzer = SwingAnalyzer()
        
        # Analyze from video file
        result = analyzer.analyze_video("swing.mp4", club=GolfClub.DRIVER)
        print(f"Overall score: {result.overall_score}")
        
        # Or analyze from pre-detected frames
        result = analyzer.analyze_frames(frames, club=GolfClub.IRON_7)
    """
    
    # -------------------------------------------------------------------------
    # Optimal angle ranges for scoring (based on pro golfer data)
    # -------------------------------------------------------------------------
    
    OPTIMAL_ANGLES = {
        SwingPhase.ADDRESS: {
            "spine_angle": (30, 45),      # Forward tilt
            "left_knee": (155, 175),      # Slight flex
            "right_knee": (155, 175),
        },
        SwingPhase.TOP: {
            "shoulder_rotation": (80, 100),  # Full shoulder turn
            "hip_rotation": (35, 50),        # Restricted hip turn
            "left_elbow": (160, 180),        # Straight lead arm
            "right_elbow": (80, 100),        # Folded trail arm
        },
        SwingPhase.IMPACT: {
            "spine_angle": (25, 40),
            "hip_rotation": (35, 50),        # Open hips at impact
            "left_elbow": (165, 180),        # Straight lead arm
        },
        SwingPhase.FINISH: {
            "shoulder_rotation": (85, 110),  # Full rotation
            "hip_rotation": (75, 95),        # Full hip turn
        },
    }
    
    def __init__(self):
        """Initialize the swing analyzer."""
        self.pose_detector = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.angle_calculator = AngleCalculator()
    
    def close(self):
        """Release resources."""
        self.pose_detector.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # -------------------------------------------------------------------------
    # Main Analysis Methods
    # -------------------------------------------------------------------------
    
    def analyze_video(
        self,
        video_path: str,
        club: GolfClub = GolfClub.IRON_7,
        frame_skip: int = 1,
    ) -> SwingAnalysis:
        """
        Analyze a golf swing from video file.
        
        Args:
            video_path: Path to video file
            club: Type of golf club being used
            frame_skip: Process every Nth frame (higher = faster but less accurate)
            
        Returns:
            Complete SwingAnalysis with scores and tips
        """
        # Process video to get all pose frames
        frames = self.pose_detector.process_video_to_list(
            video_path,
            frame_skip=frame_skip
        )
        
        if not frames:
            raise ValueError("No poses detected in video")
        
        # Get video metadata
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0
        cap.release()
        
        return self.analyze_frames(
            frames=frames,
            club=club,
            fps=fps,
            total_frames=total_frames,
            duration_ms=duration_ms,
        )
    
    def analyze_frames(
        self,
        frames: List[PoseFrame],
        club: GolfClub = GolfClub.IRON_7,
        fps: float = 30.0,
        total_frames: int = 0,
        duration_ms: int = 0,
    ) -> SwingAnalysis:
        """
        Analyze a golf swing from pre-detected pose frames.
        
        Args:
            frames: List of PoseFrame objects
            club: Type of golf club
            fps: Video frames per second
            total_frames: Total frame count
            duration_ms: Video duration in milliseconds
            
        Returns:
            Complete SwingAnalysis
        """
        if not frames:
            raise ValueError("No frames to analyze")
        
        # Identify key phases
        key_phases = self._identify_phases(frames)
        
        # Analyze each phase
        phase_analyses = []
        for phase, frame_idx in key_phases.items():
            if frame_idx is not None and frame_idx < len(frames):
                frame = frames[frame_idx]
                angles = self.angle_calculator.calculate_all_angles(frame)
                score, feedback = self._score_phase(phase, angles)
                
                phase_analyses.append(PhaseAnalysis(
                    phase=phase,
                    timestamp_ms=frame.timestamp_ms,
                    frame_number=frame.frame_number,
                    angles=angles,
                    score=score,
                    feedback=feedback,
                ))
        
        # Calculate overall scores
        posture_score = self._calculate_posture_score(phase_analyses)
        tempo_score = self._calculate_tempo_score(frames, key_phases)
        rotation_score = self._calculate_rotation_score(phase_analyses)
        balance_score = self._calculate_balance_score(phase_analyses)
        
        # Overall score is weighted average
        overall = self._calculate_overall_score(
            posture_score, tempo_score, rotation_score, balance_score
        )
        
        # Generate coaching tips
        tips = self._generate_tips(phase_analyses, club)
        
        # Create summary
        summary = self._generate_summary(overall, phase_analyses, club)
        
        return SwingAnalysis(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            video_duration_ms=duration_ms,
            total_frames=total_frames or len(frames),
            fps=fps,
            club=club,
            phases=phase_analyses,
            overall_score=overall,
            posture_score=posture_score,
            tempo_score=tempo_score,
            rotation_score=rotation_score,
            balance_score=balance_score,
            tips=tips,
            summary=summary,
            key_frames={phase: idx for phase, idx in key_phases.items() if idx is not None},
        )
    
    # -------------------------------------------------------------------------
    # Phase Detection
    # -------------------------------------------------------------------------
    
    def _identify_phases(
        self,
        frames: List[PoseFrame]
    ) -> dict[SwingPhase, Optional[int]]:
        """
        Identify the frame indices for each swing phase.
        
        Uses wrist position and body rotation to detect:
        - ADDRESS: First stable frame with low wrist position
        - TOP: Highest wrist position (top of backswing)
        - IMPACT: Wrist at hip level during downswing
        - FINISH: Final stable position
        """
        if not frames:
            return {phase: None for phase in [
                SwingPhase.ADDRESS, SwingPhase.TOP,
                SwingPhase.IMPACT, SwingPhase.FINISH
            ]}
        
        # Track wrist heights (using left wrist as reference for right-handed golfer)
        wrist_heights = []
        for frame in frames:
            left_wrist = frame.get_landmark(BodyPart.LEFT_WRIST)
            if left_wrist and left_wrist.is_visible():
                wrist_heights.append(left_wrist.y)
            else:
                wrist_heights.append(1.0)  # Default to bottom if not visible
        
        # Find key positions
        # Note: y increases downward in image coordinates
        
        # ADDRESS: First 10% of frames, most stable position
        address_range = len(frames) // 10 or 1
        address_idx = 0  # Simplification: use first frame
        
        # TOP: Minimum wrist height (highest point in backswing)
        # Look in first 60% of swing
        search_end = int(len(frames) * 0.6)
        if search_end > 0:
            top_idx = min(range(search_end), key=lambda i: wrist_heights[i])
        else:
            top_idx = 0
        
        # IMPACT: After TOP, wrist returns to near hip level
        # Look between TOP and 80% of swing
        impact_search_start = top_idx
        impact_search_end = int(len(frames) * 0.8)
        
        hip_level = self._get_average_hip_height(frames)
        
        impact_idx = top_idx
        min_diff = float('inf')
        for i in range(impact_search_start, min(impact_search_end, len(frames))):
            diff = abs(wrist_heights[i] - hip_level)
            if diff < min_diff:
                min_diff = diff
                impact_idx = i
        
        # FINISH: Last 10% of frames
        finish_idx = len(frames) - 1
        
        return {
            SwingPhase.ADDRESS: address_idx,
            SwingPhase.TOP: top_idx,
            SwingPhase.IMPACT: impact_idx,
            SwingPhase.FINISH: finish_idx,
        }
    
    def _get_average_hip_height(self, frames: List[PoseFrame]) -> float:
        """Get average hip height across frames."""
        heights = []
        for frame in frames:
            left_hip = frame.get_landmark(BodyPart.LEFT_HIP)
            right_hip = frame.get_landmark(BodyPart.RIGHT_HIP)
            if left_hip and right_hip:
                heights.append((left_hip.y + right_hip.y) / 2)
        return sum(heights) / len(heights) if heights else 0.5
    
    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------
    
    def _score_phase(
        self,
        phase: SwingPhase,
        angles: SwingAngles
    ) -> Tuple[int, str]:
        """Score a single phase based on angles."""
        if phase not in self.OPTIMAL_ANGLES:
            return 75, "Phase analyzed"
        
        optimal = self.OPTIMAL_ANGLES[phase]
        scores = []
        feedback_parts = []
        
        for angle_name, (min_val, max_val) in optimal.items():
            actual = getattr(angles, angle_name, None)
            if actual is not None:
                if min_val <= actual <= max_val:
                    scores.append(100)
                    feedback_parts.append(f"✓ {angle_name}: {actual:.0f}°")
                else:
                    # Score based on how far from optimal
                    if actual < min_val:
                        deviation = min_val - actual
                    else:
                        deviation = actual - max_val
                    
                    # Lose 2 points per degree of deviation
                    score = max(0, 100 - int(deviation * 2))
                    scores.append(score)
                    
                    if actual < min_val:
                        feedback_parts.append(f"↑ {angle_name}: {actual:.0f}° (increase)")
                    else:
                        feedback_parts.append(f"↓ {angle_name}: {actual:.0f}° (decrease)")
        
        avg_score = int(sum(scores) / len(scores)) if scores else 75
        feedback = " | ".join(feedback_parts[:3])  # Limit feedback length
        
        return avg_score, feedback
    
    def _calculate_posture_score(
        self,
        phase_analyses: List[PhaseAnalysis]
    ) -> SwingScore:
        """Calculate overall posture score."""
        spine_angles = []
        for pa in phase_analyses:
            if pa.angles.spine_angle is not None:
                spine_angles.append(pa.angles.spine_angle)
        
        if not spine_angles:
            return SwingScore(score=75, feedback="Posture could not be fully analyzed")
        
        # Check consistency of spine angle throughout swing
        variance = max(spine_angles) - min(spine_angles)
        
        if variance < 10:
            score = 90
            feedback = "Excellent spine angle maintained throughout swing"
        elif variance < 20:
            score = 75
            feedback = "Good posture, minor spine angle changes detected"
        else:
            score = 60
            feedback = "Work on maintaining consistent spine angle"
        
        return SwingScore(
            score=score,
            feedback=feedback,
            details=f"Spine angle range: {min(spine_angles):.0f}° - {max(spine_angles):.0f}°"
        )
    
    def _calculate_tempo_score(
        self,
        frames: List[PoseFrame],
        key_phases: dict[SwingPhase, Optional[int]]
    ) -> SwingScore:
        """
        Calculate tempo score based on backswing/downswing ratio.
        
        Ideal tempo ratio is approximately 3:1 (backswing:downswing).
        """
        address_idx = key_phases.get(SwingPhase.ADDRESS, 0) or 0
        top_idx = key_phases.get(SwingPhase.TOP, 0) or 0
        impact_idx = key_phases.get(SwingPhase.IMPACT, 0) or 0
        
        backswing_frames = top_idx - address_idx
        downswing_frames = impact_idx - top_idx
        
        if downswing_frames <= 0:
            return SwingScore(score=70, feedback="Could not calculate tempo")
        
        ratio = backswing_frames / downswing_frames
        
        # Ideal ratio is 3:1
        if 2.5 <= ratio <= 3.5:
            score = 95
            feedback = "Excellent tempo! Smooth and controlled"
        elif 2.0 <= ratio <= 4.0:
            score = 80
            feedback = "Good tempo, minor adjustments possible"
        elif ratio < 2.0:
            score = 65
            feedback = "Backswing may be too quick - slow it down"
        else:
            score = 65
            feedback = "Downswing may be too slow - accelerate through impact"
        
        return SwingScore(
            score=score,
            feedback=feedback,
            details=f"Backswing/Downswing ratio: {ratio:.1f}:1"
        )
    
    def _calculate_rotation_score(
        self,
        phase_analyses: List[PhaseAnalysis]
    ) -> SwingScore:
        """Calculate rotation (X-factor) score."""
        top_phase = next(
            (pa for pa in phase_analyses if pa.phase == SwingPhase.TOP),
            None
        )
        
        if not top_phase:
            return SwingScore(score=75, feedback="Rotation could not be analyzed")
        
        shoulder_rot = top_phase.angles.shoulder_rotation
        hip_rot = top_phase.angles.hip_rotation
        
        if shoulder_rot is None or hip_rot is None:
            return SwingScore(score=75, feedback="Rotation partially analyzed")
        
        x_factor = abs(shoulder_rot - hip_rot)
        
        if x_factor >= 45:
            score = 95
            feedback = "Excellent X-factor! Great power potential"
        elif x_factor >= 35:
            score = 80
            feedback = "Good shoulder-hip separation"
        else:
            score = 65
            feedback = "Increase shoulder turn while restricting hips"
        
        return SwingScore(
            score=score,
            feedback=feedback,
            details=f"X-factor: {x_factor:.0f}° (shoulder: {shoulder_rot:.0f}°, hip: {hip_rot:.0f}°)"
        )
    
    def _calculate_balance_score(
        self,
        phase_analyses: List[PhaseAnalysis]
    ) -> SwingScore:
        """Calculate balance score based on knee flex consistency."""
        knee_angles = []
        for pa in phase_analyses:
            if pa.angles.left_knee:
                knee_angles.append(pa.angles.left_knee)
            if pa.angles.right_knee:
                knee_angles.append(pa.angles.right_knee)
        
        if not knee_angles:
            return SwingScore(score=75, feedback="Balance could not be fully analyzed")
        
        # Good balance = consistent knee flex
        avg_knee = sum(knee_angles) / len(knee_angles)
        variance = max(knee_angles) - min(knee_angles)
        
        if 150 <= avg_knee <= 175 and variance < 20:
            score = 90
            feedback = "Excellent balance maintained"
        elif 140 <= avg_knee <= 180:
            score = 75
            feedback = "Good balance, minor improvements possible"
        else:
            score = 60
            feedback = "Work on maintaining athletic knee flex"
        
        return SwingScore(
            score=score,
            feedback=feedback,
            details=f"Avg knee angle: {avg_knee:.0f}°"
        )
    
    def _calculate_overall_score(
        self,
        posture: SwingScore,
        tempo: SwingScore,
        rotation: SwingScore,
        balance: SwingScore,
    ) -> int:
        """Calculate weighted overall score."""
        weights = {
            "posture": 0.25,
            "tempo": 0.25,
            "rotation": 0.30,  # X-factor is very important
            "balance": 0.20,
        }
        
        weighted_sum = (
            posture.score * weights["posture"] +
            tempo.score * weights["tempo"] +
            rotation.score * weights["rotation"] +
            balance.score * weights["balance"]
        )
        
        return int(weighted_sum)
    
    # -------------------------------------------------------------------------
    # Coaching Tips Generation
    # -------------------------------------------------------------------------
    
    def _generate_tips(
        self,
        phase_analyses: List[PhaseAnalysis],
        club: GolfClub
    ) -> List[CoachingTip]:
        """Generate actionable coaching tips based on analysis."""
        tips = []
        
        for pa in phase_analyses:
            if pa.score < 70:
                # Generate tips for low-scoring phases
                if pa.phase == SwingPhase.ADDRESS:
                    tips.append(CoachingTip(
                        category="Setup",
                        priority=1,
                        title="Improve Address Position",
                        description=f"Your setup needs attention. {pa.feedback}",
                        drill="Practice your setup in front of a mirror, checking spine angle and knee flex.",
                    ))
                    
                elif pa.phase == SwingPhase.TOP:
                    tips.append(CoachingTip(
                        category="Backswing",
                        priority=2,
                        title="Optimize Top of Backswing",
                        description=f"Top of swing position: {pa.feedback}",
                        drill="Pause at the top of your backswing to feel the proper position.",
                    ))
                    
                elif pa.phase == SwingPhase.IMPACT:
                    tips.append(CoachingTip(
                        category="Impact",
                        priority=1,
                        title="Improve Impact Position",
                        description=f"Impact position: {pa.feedback}",
                        drill="Practice slow-motion swings focusing on hip rotation at impact.",
                    ))
        
        # Add club-specific tip
        tips.append(self._get_club_specific_tip(club))
        
        # Sort by priority
        tips.sort(key=lambda t: t.priority)
        
        return tips[:5]  # Return top 5 tips
    
    def _get_club_specific_tip(self, club: GolfClub) -> CoachingTip:
        """Get a tip specific to the club being used."""
        tips = {
            GolfClub.DRIVER: CoachingTip(
                category="Driver",
                priority=3,
                title="Driver Swing Tip",
                description="For driver, focus on sweeping through the ball with an upward angle of attack.",
                drill="Tee the ball high and practice hitting up on the ball.",
            ),
            GolfClub.IRON_7: CoachingTip(
                category="Irons",
                priority=3,
                title="Iron Swing Tip",
                description="For irons, focus on hitting down on the ball with a divot after impact.",
                drill="Place a towel 2 inches behind the ball and practice not hitting it.",
            ),
            GolfClub.PUTTER: CoachingTip(
                category="Putting",
                priority=3,
                title="Putting Tip",
                description="Keep your lower body still and rock your shoulders like a pendulum.",
                drill="Practice with a coin under each foot to feel any lower body movement.",
            ),
        }
        
        return tips.get(club, CoachingTip(
            category="General",
            priority=3,
            title="Focus on Fundamentals",
            description="Maintain good posture and tempo throughout your swing.",
            drill="Practice with alignment sticks for better consistency.",
        ))
    
    def _generate_summary(
        self,
        overall_score: int,
        phase_analyses: List[PhaseAnalysis],
        club: GolfClub
    ) -> str:
        """Generate a text summary of the analysis."""
        if overall_score >= 85:
            quality = "excellent"
        elif overall_score >= 70:
            quality = "good"
        elif overall_score >= 55:
            quality = "developing"
        else:
            quality = "needs work"
        
        low_phases = [pa for pa in phase_analyses if pa.score < 70]
        
        summary = f"Your {club.value} swing scored {overall_score}/100 - {quality}. "
        
        if low_phases:
            phase_names = [pa.phase.value for pa in low_phases]
            summary += f"Focus on improving: {', '.join(phase_names)}. "
        else:
            summary += "All phases of your swing are solid. "
        
        summary += "Keep practicing to build consistency!"
        
        return summary