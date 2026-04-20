import { useState, useEffect } from 'react';
import { TrendingUp } from 'lucide-react';
import { getHistory } from '@/lib/historyStore';
import { HistoryEntryCard } from '@/components/history/HistoryEntryCard';
import type { HistoryEntry } from '@/types';

const gradeColors: Record<string, string> = {
  A: '#4ade80', B: '#a3e635', C: '#facc15', D: '#fb923c', F: '#f87171',
};

function MiniChart({ entries }: { entries: HistoryEntry[] }) {
  if (entries.length < 2) return null;
  const scores = [...entries].reverse().map((e) => e.overall_score);
  const W = 300;
  const H = 80;
  const pad = 10;
  const minS = Math.min(...scores) - 5;
  const maxS = Math.max(...scores) + 5;

  const points = scores.map((s, i) => {
    const x = pad + (i / (scores.length - 1)) * (W - pad * 2);
    const y = H - pad - ((s - minS) / (maxS - minS)) * (H - pad * 2);
    return `${x},${y}`;
  });

  const latest = scores[scores.length - 1];
  const grade = latest >= 90 ? 'A' : latest >= 80 ? 'B' : latest >= 70 ? 'C' : latest >= 60 ? 'D' : 'F';
  const lineColor = gradeColors[grade] ?? '#ffffff';

  return (
    <div className="bg-[var(--color-surface-card)] rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="w-4 h-4 text-[var(--color-accent)]" />
        <span className="text-[var(--color-text)] text-sm font-semibold">Score Trend</span>
        <span className="ml-auto text-[var(--color-text-muted)] text-xs">{entries.length} sessions</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-20">
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke={lineColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {points.map((pt, i) => {
          const [x, y] = pt.split(',').map(Number);
          return <circle key={i} cx={x} cy={y} r="3" fill={lineColor} />;
        })}
      </svg>
      <div className="flex justify-between text-xs text-[var(--color-text-muted)] mt-1">
        <span>Oldest</span>
        <span>Latest: <strong className="text-[var(--color-text)]">{latest}</strong></span>
      </div>
    </div>
  );
}

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory().then(setEntries).catch(console.error).finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-[var(--color-surface)] flex flex-col">
      <header className="bg-[var(--color-primary)] px-4 py-3">
        <h1 className="text-lg font-bold text-[var(--color-text)]">History</h1>
        <p className="text-xs text-[var(--color-accent)] uppercase tracking-wider">Your swing progress</p>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {loading && (
          <div className="flex justify-center pt-12">
            <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {!loading && entries.length === 0 && (
          <div className="flex flex-col items-center justify-center pt-24 text-center">
            <TrendingUp className="w-12 h-12 text-[var(--color-text-muted)] mb-4" />
            <p className="text-[var(--color-text-secondary)] font-medium">No swings yet</p>
            <p className="text-[var(--color-text-muted)] text-sm mt-1">
              Analyze a video to start tracking your progress
            </p>
          </div>
        )}

        {!loading && entries.length > 0 && (
          <>
            <MiniChart entries={entries} />
            <div className="space-y-3">
              {entries.map((entry) => (
                <HistoryEntryCard key={entry.id} entry={entry} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
