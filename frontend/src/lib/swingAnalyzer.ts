import { calculateAllAngles } from './angleCalculator';
import type {
  PoseFrame,
  SwingAnalysis,
  SwingAngles,
  SwingScore,
  CoachingTip,
  PhaseAnalysis,
  SwingPhase,
  GolfClub,
} from '@/types';

type PhaseMap = Record<SwingPhase, number | null>;

const OPTIMAL_ANGLES: Partial<Record<SwingPhase, Partial<Record<keyof SwingAngles, [number, number]>>>> = {
  address: {
    spine_angle: [30, 45],
    left_knee: [155, 175],
    right_knee: [155, 175],
  },
  top: {
    shoulder_rotation: [80, 100],
    hip_rotation: [35, 50],
    left_elbow: [160, 180],
    right_elbow: [80, 100],
  },
  impact: {
    spine_angle: [25, 40],
    hip_rotation: [35, 50],
    left_elbow: [165, 180],
  },
  finish: {
    shoulder_rotation: [85, 110],
    hip_rotation: [75, 95],
  },
};

const CLUB_NAMES: Record<GolfClub, string> = {
  driver: 'Driver', wood_3: '3 Wood', wood_5: '5 Wood', hybrid: 'Hybrid',
  iron_4: '4 Iron', iron_5: '5 Iron', iron_6: '6 Iron', iron_7: '7 Iron',
  iron_8: '8 Iron', iron_9: '9 Iron',
  pitching_wedge: 'Pitching Wedge', sand_wedge: 'Sand Wedge', lob_wedge: 'Lob Wedge',
  putter: 'Putter',
};

function grade(score: number): string {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}

function makeScore(score: number, feedback: string, details?: string): SwingScore {
  return { score, grade: grade(score), feedback, details: details ?? null };
}

function scorePhase(phase: SwingPhase, angles: SwingAngles): [number, string] {
  const optimal = OPTIMAL_ANGLES[phase];
  if (!optimal) return [75, 'Phase analyzed'];

  const scores: number[] = [];
  const feedbackParts: string[] = [];

  for (const [angleName, [minVal, maxVal]] of Object.entries(optimal) as [keyof SwingAngles, [number, number]][]) {
    const actual = angles[angleName] as number | null;
    if (actual !== null) {
      if (actual >= minVal && actual <= maxVal) {
        scores.push(100);
        feedbackParts.push(`✓ ${angleName}: ${actual.toFixed(0)}°`);
      } else {
        const deviation = actual < minVal ? minVal - actual : actual - maxVal;
        const s = Math.max(0, 100 - Math.round(deviation * 2));
        scores.push(s);
        feedbackParts.push(
          `${actual < minVal ? '↑' : '↓'} ${angleName}: ${actual.toFixed(0)}° (${actual < minVal ? 'increase' : 'decrease'})`,
        );
      }
    }
  }

  const avg = scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 75;
  return [avg, feedbackParts.slice(0, 3).join(' | ')];
}

function identifyPhases(frames: PoseFrame[]): PhaseMap {
  if (!frames.length) {
    return { address: null, backswing: null, top: null, downswing: null, impact: null, follow_through: null, finish: null };
  }

  // Track left wrist height (y increases downward)
  const wristHeights = frames.map((f) => {
    const lw = f.landmarks[15];
    return lw && lw.visibility >= 0.5 ? lw.y : 1.0;
  });

  const addressIdx = 0;

  // Top = minimum wrist y (highest point) in first 60% of frames
  const searchEnd = Math.max(1, Math.round(frames.length * 0.6));
  let topIdx = 0;
  let minH = wristHeights[0];
  for (let i = 1; i < searchEnd; i++) {
    if (wristHeights[i] < minH) { minH = wristHeights[i]; topIdx = i; }
  }

  // Average hip height across frames
  let hipSum = 0;
  let hipCount = 0;
  for (const f of frames) {
    const lh = f.landmarks[23];
    const rh = f.landmarks[24];
    if (lh && rh && lh.visibility >= 0.5 && rh.visibility >= 0.5) {
      hipSum += (lh.y + rh.y) / 2;
      hipCount++;
    }
  }
  const avgHip = hipCount ? hipSum / hipCount : 0.5;

  // Impact = wrist closest to hip level, after top, in range 40-80% of frames
  const impactStart = topIdx;
  const impactEnd = Math.round(frames.length * 0.8);
  let impactIdx = topIdx;
  let minDiff = Infinity;
  for (let i = impactStart; i < Math.min(impactEnd, frames.length); i++) {
    const diff = Math.abs(wristHeights[i] - avgHip);
    if (diff < minDiff) { minDiff = diff; impactIdx = i; }
  }

  const finishIdx = frames.length - 1;

  return {
    address: addressIdx,
    backswing: null,
    top: topIdx,
    downswing: null,
    impact: impactIdx,
    follow_through: null,
    finish: finishIdx,
  };
}

function calcPostureScore(phaseAnalyses: PhaseAnalysis[]): SwingScore {
  const spineAngles = phaseAnalyses
    .map((pa) => pa.angles.spine_angle)
    .filter((a): a is number => a !== null);

  if (!spineAngles.length) return makeScore(75, 'Posture could not be fully analyzed');

  const variance = Math.max(...spineAngles) - Math.min(...spineAngles);
  if (variance < 10) return makeScore(90, 'Excellent spine angle maintained throughout swing',
    `Spine angle range: ${Math.min(...spineAngles).toFixed(0)}°–${Math.max(...spineAngles).toFixed(0)}°`);
  if (variance < 20) return makeScore(75, 'Good posture, minor spine angle changes detected',
    `Spine angle range: ${Math.min(...spineAngles).toFixed(0)}°–${Math.max(...spineAngles).toFixed(0)}°`);
  return makeScore(60, 'Work on maintaining consistent spine angle',
    `Spine angle range: ${Math.min(...spineAngles).toFixed(0)}°–${Math.max(...spineAngles).toFixed(0)}°`);
}

function calcTempoScore(frames: PoseFrame[], phases: PhaseMap): SwingScore {
  const addressIdx = phases.address ?? 0;
  const topIdx = phases.top ?? 0;
  const impactIdx = phases.impact ?? 0;

  const backswingFrames = topIdx - addressIdx;
  const downswingFrames = impactIdx - topIdx;
  if (downswingFrames <= 0) return makeScore(70, 'Could not calculate tempo');

  const ratio = backswingFrames / downswingFrames;
  const detail = `Backswing/Downswing ratio: ${ratio.toFixed(1)}:1`;

  if (ratio >= 2.5 && ratio <= 3.5) return makeScore(95, 'Excellent tempo! Smooth and controlled', detail);
  if (ratio >= 2.0 && ratio <= 4.0) return makeScore(80, 'Good tempo, minor adjustments possible', detail);
  if (ratio < 2.0) return makeScore(65, 'Backswing may be too quick — slow it down', detail);
  return makeScore(65, 'Downswing may be too slow — accelerate through impact', detail);
}

function calcRotationScore(phaseAnalyses: PhaseAnalysis[]): SwingScore {
  const topPhase = phaseAnalyses.find((pa) => pa.phase === 'top');
  if (!topPhase) return makeScore(75, 'Rotation could not be analyzed');

  const sr = topPhase.angles.shoulder_rotation;
  const hr = topPhase.angles.hip_rotation;
  if (sr === null || hr === null) return makeScore(75, 'Rotation partially analyzed');

  const xFactor = Math.abs(sr - hr);
  const detail = `X-factor: ${xFactor.toFixed(0)}° (shoulder: ${sr.toFixed(0)}°, hip: ${hr.toFixed(0)}°)`;

  if (xFactor >= 45) return makeScore(95, 'Excellent X-factor! Great power potential', detail);
  if (xFactor >= 35) return makeScore(80, 'Good shoulder-hip separation', detail);
  return makeScore(65, 'Increase shoulder turn while restricting hips', detail);
}

function calcBalanceScore(phaseAnalyses: PhaseAnalysis[]): SwingScore {
  const kneeAngles: number[] = [];
  for (const pa of phaseAnalyses) {
    if (pa.angles.left_knee !== null) kneeAngles.push(pa.angles.left_knee);
    if (pa.angles.right_knee !== null) kneeAngles.push(pa.angles.right_knee);
  }
  if (!kneeAngles.length) return makeScore(75, 'Balance could not be fully analyzed');

  const avg = kneeAngles.reduce((a, b) => a + b, 0) / kneeAngles.length;
  const variance = Math.max(...kneeAngles) - Math.min(...kneeAngles);
  const detail = `Avg knee angle: ${avg.toFixed(0)}°`;

  if (avg >= 150 && avg <= 175 && variance < 20) return makeScore(90, 'Excellent balance maintained', detail);
  if (avg >= 140 && avg <= 180) return makeScore(75, 'Good balance, minor improvements possible', detail);
  return makeScore(60, 'Work on maintaining athletic knee flex', detail);
}

function calcOverallScore(posture: SwingScore, tempo: SwingScore, rotation: SwingScore, balance: SwingScore): number {
  return Math.round(
    posture.score * 0.25 + tempo.score * 0.25 + rotation.score * 0.30 + balance.score * 0.20,
  );
}

function generateTips(phaseAnalyses: PhaseAnalysis[], club: GolfClub): CoachingTip[] {
  const tips: CoachingTip[] = [];

  for (const pa of phaseAnalyses) {
    if (pa.score < 70) {
      if (pa.phase === 'address') {
        tips.push({ category: 'Setup', priority: 1, title: 'Improve Address Position',
          description: `Your setup needs attention. ${pa.feedback}`,
          drill: 'Practice your setup in front of a mirror, checking spine angle and knee flex.' });
      } else if (pa.phase === 'top') {
        tips.push({ category: 'Backswing', priority: 2, title: 'Optimize Top of Backswing',
          description: `Top of swing position: ${pa.feedback}`,
          drill: 'Pause at the top of your backswing to feel the proper position.' });
      } else if (pa.phase === 'impact') {
        tips.push({ category: 'Impact', priority: 1, title: 'Improve Impact Position',
          description: `Impact position: ${pa.feedback}`,
          drill: 'Practice slow-motion swings focusing on hip rotation at impact.' });
      }
    }
  }

  const clubTips: Partial<Record<GolfClub, CoachingTip>> = {
    driver: { category: 'Driver', priority: 3, title: 'Driver Swing Tip',
      description: 'For driver, focus on sweeping through the ball with an upward angle of attack.',
      drill: 'Tee the ball high and practice hitting up on the ball.' },
    iron_7: { category: 'Irons', priority: 3, title: 'Iron Swing Tip',
      description: 'For irons, focus on hitting down on the ball with a divot after impact.',
      drill: 'Place a towel 2 inches behind the ball and practice not hitting it.' },
    putter: { category: 'Putting', priority: 3, title: 'Putting Tip',
      description: 'Keep your lower body still and rock your shoulders like a pendulum.',
      drill: 'Practice with a coin under each foot to feel any lower body movement.' },
  };

  tips.push(
    clubTips[club] ?? {
      category: 'General', priority: 3, title: 'Focus on Fundamentals',
      description: 'Maintain good posture and tempo throughout your swing.',
      drill: 'Practice with alignment sticks for better consistency.',
    },
  );

  tips.sort((a, b) => a.priority - b.priority);
  return tips.slice(0, 5);
}

function generateSummary(overall: number, phaseAnalyses: PhaseAnalysis[], club: GolfClub): string {
  const quality = overall >= 85 ? 'excellent' : overall >= 70 ? 'good' : overall >= 55 ? 'developing' : 'needs work';
  const clubName = CLUB_NAMES[club] ?? club;
  const lowPhases = phaseAnalyses.filter((pa) => pa.score < 70).map((pa) => pa.phase);

  let s = `Your ${clubName} swing scored ${overall}/100 — ${quality}. `;
  if (lowPhases.length) {
    s += `Focus on improving: ${lowPhases.join(', ')}. `;
  } else {
    s += 'All phases of your swing are solid. ';
  }
  s += 'Keep practicing to build consistency!';
  return s;
}

export function analyzeFrames(
  frames: PoseFrame[],
  club: GolfClub,
  fps = 30,
  durationMs = 0,
): SwingAnalysis {
  if (!frames.length) throw new Error('No frames to analyze');

  const phases = identifyPhases(frames);

  const phaseAnalyses: PhaseAnalysis[] = [];
  const analyzePhases: SwingPhase[] = ['address', 'top', 'impact', 'finish'];
  for (const phase of analyzePhases) {
    const idx = phases[phase];
    if (idx !== null && idx < frames.length) {
      const frame = frames[idx];
      const angles = calculateAllAngles(frame);
      const [s, feedback] = scorePhase(phase, angles);
      phaseAnalyses.push({
        phase,
        timestamp_ms: frame.timestamp_ms,
        frame_number: frame.frame_number,
        angles,
        score: s,
        feedback,
      });
    }
  }

  const postureScore = calcPostureScore(phaseAnalyses);
  const tempoScore = calcTempoScore(frames, phases);
  const rotationScore = calcRotationScore(phaseAnalyses);
  const balanceScore = calcBalanceScore(phaseAnalyses);
  const overall = calcOverallScore(postureScore, tempoScore, rotationScore, balanceScore);
  const tips = generateTips(phaseAnalyses, club);
  const summary = generateSummary(overall, phaseAnalyses, club);

  const keyFrames: Record<string, number> = {};
  for (const [phase, idx] of Object.entries(phases)) {
    if (idx !== null) keyFrames[phase] = idx;
  }

  const backswingFrames = (phases.top ?? 0) - (phases.address ?? 0);
  const downswingFrames = (phases.impact ?? 0) - (phases.top ?? 0);
  const tempoRatio = downswingFrames > 0 ? backswingFrames / downswingFrames : undefined;

  const topAnalysis = phaseAnalyses.find((pa) => pa.phase === 'top');
  const sr = topAnalysis?.angles.shoulder_rotation ?? null;
  const hr = topAnalysis?.angles.hip_rotation ?? null;
  const xFactorTop = sr !== null && hr !== null ? Math.abs(sr - hr) : undefined;

  return {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    video_duration_ms: durationMs,
    total_frames: frames.length,
    fps,
    club,
    overall_score: overall,
    posture_score: postureScore,
    tempo_score: tempoScore,
    rotation_score: rotationScore,
    balance_score: balanceScore,
    phases: phaseAnalyses,
    tips,
    summary,
    key_frames: keyFrames,
    tempo_ratio: tempoRatio,
    x_factor_top: xFactorTop,
  };
}
