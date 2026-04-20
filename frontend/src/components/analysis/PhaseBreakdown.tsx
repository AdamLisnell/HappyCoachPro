import type { PhaseAnalysis, SwingAngles } from '@/types';

interface Props {
  phases: PhaseAnalysis[];
}

const PHASE_LABELS: Record<string, string> = {
  address: 'Address',
  backswing: 'Backswing',
  top: 'Top',
  downswing: 'Downswing',
  impact: 'Impact',
  follow_through: 'Follow-Through',
  finish: 'Finish',
};

const ANGLE_LABELS: Partial<Record<keyof SwingAngles, string>> = {
  spine_angle: 'Spine',
  shoulder_rotation: 'Shoulders',
  hip_rotation: 'Hips',
  x_factor: 'X-Factor',
  left_elbow: 'L. Elbow',
  right_elbow: 'R. Elbow',
  left_knee: 'L. Knee',
  right_knee: 'R. Knee',
};

function scoreColor(score: number): string {
  if (score >= 80) return 'border-green-400 bg-green-400/10';
  if (score >= 60) return 'border-yellow-400 bg-yellow-400/10';
  return 'border-red-400 bg-red-400/10';
}

function scoreTextColor(score: number): string {
  if (score >= 80) return 'text-green-400';
  if (score >= 60) return 'text-yellow-400';
  return 'text-red-400';
}

export function PhaseBreakdown({ phases }: Props) {
  if (!phases.length) return null;

  return (
    <div>
      <h3 className="text-[var(--color-text)] font-semibold mb-3">Phase Breakdown</h3>
      <div className="flex gap-3 overflow-x-auto pb-2">
        {phases.map((pa) => {
          const angleEntries = (Object.entries(pa.angles) as [keyof SwingAngles, number | null][])
            .filter(([k, v]) => v !== null && k in ANGLE_LABELS)
            .slice(0, 4);

          return (
            <div
              key={pa.phase}
              className={`flex-shrink-0 w-40 rounded-xl p-3 border-l-4 bg-[var(--color-surface-card)] ${scoreColor(pa.score)}`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[var(--color-text)] text-xs font-semibold uppercase tracking-wide">
                  {PHASE_LABELS[pa.phase] ?? pa.phase}
                </span>
                <span className={`text-sm font-black ${scoreTextColor(pa.score)}`}>{pa.score}</span>
              </div>

              <div className="space-y-1">
                {angleEntries.map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs">
                    <span className="text-[var(--color-text-muted)]">{ANGLE_LABELS[k]}</span>
                    <span className="text-[var(--color-text-secondary)] font-mono">{(v as number).toFixed(0)}°</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
