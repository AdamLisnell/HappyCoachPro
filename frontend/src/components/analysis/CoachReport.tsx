import { Target, Dumbbell, BookOpen } from 'lucide-react';
import type { AICoachingReport } from '@/types';

interface Props {
  report: AICoachingReport | null;
  loading: boolean;
  error: string | null;
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[80, 60, 90, 70].map((w, i) => (
        <div key={i} className={`h-3 bg-[var(--color-surface-card)] rounded-full`} style={{ width: `${w}%` }} />
      ))}
    </div>
  );
}

export function CoachReport({ report, loading, error }: Props) {
  return (
    <div className="bg-[var(--color-surface-card)] rounded-xl p-4 border border-[var(--color-accent)]/20">
      <div className="flex items-center gap-2 mb-4">
        <BookOpen className="w-4 h-4 text-[var(--color-accent)]" />
        <h3 className="text-[var(--color-text)] font-semibold">Coach's Report</h3>
        {loading && (
          <span className="ml-auto text-xs text-[var(--color-text-muted)] animate-pulse">Generating…</span>
        )}
      </div>

      {loading && <Skeleton />}

      {error && !loading && (
        <p className="text-[var(--color-text-muted)] text-sm italic">{error}</p>
      )}

      {report && !loading && (
        <div className="space-y-4">
          {/* Narrative */}
          <div className="text-[var(--color-text-secondary)] text-sm leading-relaxed whitespace-pre-line">
            {report.narrative}
          </div>

          {/* Focus Areas */}
          {report.focus_areas.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Target className="w-3.5 h-3.5 text-[var(--color-accent)]" />
                <span className="text-[var(--color-accent)] text-xs font-semibold uppercase tracking-wide">Focus Areas</span>
              </div>
              <ol className="space-y-1.5">
                {report.focus_areas.map((area, i) => (
                  <li key={i} className="flex gap-2 text-sm text-[var(--color-text-secondary)]">
                    <span className="text-[var(--color-accent)] font-bold flex-shrink-0">{i + 1}.</span>
                    <span>{area}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {/* Practice Plan */}
          {report.practice_plan && (
            <div className="bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20 rounded-lg p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <Dumbbell className="w-3.5 h-3.5 text-[var(--color-accent)]" />
                <span className="text-[var(--color-accent)] text-xs font-semibold uppercase tracking-wide">Practice Plan</span>
              </div>
              <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed">{report.practice_plan}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
