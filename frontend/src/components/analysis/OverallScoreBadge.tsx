interface Props {
  score: number;
  summary: string | null;
  compact?: boolean;
}

const gradeColor: Record<string, string> = {
  A: 'text-green-400',
  B: 'text-lime-400',
  C: 'text-yellow-400',
  D: 'text-orange-400',
  F: 'text-red-400',
};

function scoreToGrade(score: number): string {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}

export function OverallScoreBadge({ score, summary, compact = false }: Props) {
  const g = scoreToGrade(score);
  const color = gradeColor[g] ?? 'text-white';
  const size = compact ? 88 : 144;
  const r = compact ? 38 : 44;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score / 100);

  if (compact) {
    return (
      <div className="flex flex-col items-center">
        <div className="relative" style={{ width: size, height: size }}>
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor"
              className="text-[var(--color-surface)]" strokeWidth="9" />
            <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor"
              className={color} strokeWidth="9" strokeLinecap="round"
              strokeDasharray={circumference} strokeDashoffset={offset}
              style={{ transition: 'stroke-dashoffset 1s ease' }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-2xl font-black ${color}`}>{score}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-widest">/ 100</span>
          </div>
        </div>
        <div className={`text-3xl font-black ${color} mt-1`}>{g}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4 py-6">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor"
            className="text-[var(--color-surface-card)]" strokeWidth="8" />
          <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor"
            className={color} strokeWidth="8" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-black ${color}`}>{score}</span>
          <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-widest">/ 100</span>
        </div>
      </div>
      <div className={`text-5xl font-black ${color}`}>{g}</div>
      {summary && (
        <p className="text-center text-[var(--color-text-secondary)] text-sm max-w-xs px-4">{summary}</p>
      )}
    </div>
  );
}
