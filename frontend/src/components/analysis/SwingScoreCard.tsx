import type { SwingScore } from '@/types';

interface Props {
  label: string;
  swingScore: SwingScore;
}

const gradeColor: Record<string, string> = {
  A: 'text-green-400',
  B: 'text-lime-400',
  C: 'text-yellow-400',
  D: 'text-orange-400',
  F: 'text-red-400',
};

const gradeBg: Record<string, string> = {
  A: 'border-green-400/30',
  B: 'border-lime-400/30',
  C: 'border-yellow-400/30',
  D: 'border-orange-400/30',
  F: 'border-red-400/30',
};

export function SwingScoreCard({ label, swingScore }: Props) {
  const color = gradeColor[swingScore.grade] ?? 'text-white';
  const border = gradeBg[swingScore.grade] ?? 'border-white/10';
  const circumference = 2 * Math.PI * 20;
  const offset = circumference * (1 - swingScore.score / 100);

  return (
    <div className={`bg-[var(--color-surface-card)] rounded-xl p-4 border ${border}`}>
      <div className="flex items-center gap-3 mb-2">
        <div className="relative w-12 h-12 flex-shrink-0">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 48 48">
            <circle cx="24" cy="24" r="20" fill="none" stroke="currentColor"
              className="text-[var(--color-surface)]" strokeWidth="5" />
            <circle cx="24" cy="24" r="20" fill="none" stroke="currentColor"
              className={color} strokeWidth="5" strokeLinecap="round"
              strokeDasharray={circumference} strokeDashoffset={offset}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-xs font-bold ${color}`}>{swingScore.grade}</span>
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[var(--color-text)] font-semibold text-sm">{label}</p>
          <p className={`text-2xl font-black ${color}`}>{swingScore.score}</p>
        </div>
      </div>
      <p className="text-[var(--color-text-muted)] text-xs leading-relaxed">{swingScore.feedback}</p>
      {swingScore.details && (
        <p className="text-[var(--color-text-muted)] text-xs mt-1 opacity-70">{swingScore.details}</p>
      )}
    </div>
  );
}
