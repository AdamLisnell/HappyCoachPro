import type { HistoryEntry } from '@/types';

interface Props {
  entry: HistoryEntry;
}

const gradeColor: Record<string, string> = {
  A: 'text-green-400 border-green-400/30',
  B: 'text-lime-400 border-lime-400/30',
  C: 'text-yellow-400 border-yellow-400/30',
  D: 'text-orange-400 border-orange-400/30',
  F: 'text-red-400 border-red-400/30',
};

const CLUB_NAMES: Record<string, string> = {
  driver: 'Driver', wood_3: '3 Wood', wood_5: '5 Wood', hybrid: 'Hybrid',
  iron_4: '4 Iron', iron_5: '5 Iron', iron_6: '6 Iron', iron_7: '7 Iron',
  iron_8: '8 Iron', iron_9: '9 Iron',
  pitching_wedge: 'PW', sand_wedge: 'SW', lob_wedge: 'LW', putter: 'Putter',
};

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return iso;
  }
}

export function HistoryEntryCard({ entry }: Props) {
  const colors = gradeColor[entry.grade] ?? 'text-white border-white/10';
  const [textColor, borderColor] = colors.split(' ');

  return (
    <div className={`bg-[var(--color-surface-card)] rounded-xl p-4 border ${borderColor}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[var(--color-text-muted)] text-xs">{formatDate(entry.timestamp)}</span>
            <span className="text-[var(--color-accent)] text-xs bg-[var(--color-accent)]/10 px-2 py-0.5 rounded-full">
              {CLUB_NAMES[entry.club] ?? entry.club}
            </span>
          </div>
          <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed line-clamp-2">{entry.summary}</p>
        </div>

        <div className="flex flex-col items-center flex-shrink-0">
          <span className={`text-3xl font-black ${textColor}`}>{entry.overall_score}</span>
          <span className={`text-lg font-bold ${textColor}`}>{entry.grade}</span>
        </div>
      </div>
    </div>
  );
}
