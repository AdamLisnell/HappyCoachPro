import type { PhaseAnalysis, SwingAngles } from '@/types';

interface Props {
  phases: PhaseAnalysis[];
}

export const PHASE_LABELS: Record<string, string> = {
  address: 'Address',
  backswing: 'Backswing',
  top: 'Top',
  downswing: 'Downswing',
  impact: 'Impact',
  follow_through: 'Follow-Through',
  finish: 'Finish',
};

export const ANGLE_LABELS: Partial<Record<keyof SwingAngles, string>> = {
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
  if (score >= 80) return 'border-green-400/60';
  if (score >= 60) return 'border-yellow-400/60';
  return 'border-red-400/60';
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
      <div className="space-y-3">
        {phases.map((pa) => {
          const angleEntries = (Object.entries(pa.angles) as [keyof SwingAngles, number | null][])
            .filter(([k, v]) => v !== null && k in ANGLE_LABELS);

          return (
            <div
              key={pa.phase}
              className={`bg-[var(--color-surface-card)] rounded-xl p-4 border-l-4 ${scoreColor(pa.score)}`}
            >
              <div className="flex items-start gap-4">
                {/* Left: phase name + score */}
                <div className="flex-shrink-0 w-20 text-center">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--color-text-muted)] mb-1">
                    {PHASE_LABELS[pa.phase] ?? pa.phase}
                  </div>
                  <div className={`text-3xl font-black leading-none ${scoreTextColor(pa.score)}`}>
                    {pa.score}
                  </div>
                </div>

                {/* Right: angle grid */}
                <div className="flex-1">
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
                    {angleEntries.map(([k, v]) => (
                      <div key={k} className="flex justify-between text-xs">
                        <span className="text-[var(--color-text-muted)]">{ANGLE_LABELS[k]}</span>
                        <span className="text-[var(--color-text-secondary)] font-mono font-semibold">
                          {(v as number).toFixed(0)}°
                        </span>
                      </div>
                    ))}
                  </div>
                  {pa.feedback && (
                    <p className="text-[var(--color-text-muted)] text-xs mt-2 leading-relaxed">
                      {pa.feedback}
                    </p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
