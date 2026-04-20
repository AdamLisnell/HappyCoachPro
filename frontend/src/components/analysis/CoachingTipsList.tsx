import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { CoachingTip } from '@/types';

interface Props {
  tips: CoachingTip[];
}

const PRIORITY_LABELS: Record<number, { label: string; color: string }> = {
  1: { label: 'Critical', color: 'bg-red-500/20 text-red-400 border border-red-400/30' },
  2: { label: 'Important', color: 'bg-yellow-500/20 text-yellow-400 border border-yellow-400/30' },
  3: { label: 'Suggested', color: 'bg-blue-500/20 text-blue-400 border border-blue-400/30' },
};

function PriorityBadge({ priority }: { priority: number }) {
  const p = PRIORITY_LABELS[priority] ?? PRIORITY_LABELS[3];
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${p.color}`}>
      {p.label}
    </span>
  );
}

export function CoachingTipsList({ tips }: Props) {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  if (!tips.length) return null;

  return (
    <div>
      <h3 className="text-[var(--color-text)] font-semibold mb-3">Coaching Tips</h3>
      <div className="space-y-2">
        {tips.map((tip, i) => (
          <div key={i} className="bg-[var(--color-surface-card)] rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-4 py-3 text-left"
              onClick={() => setOpenIndex(openIndex === i ? null : i)}
            >
              <div className="flex items-center gap-2 min-w-0">
                <PriorityBadge priority={tip.priority} />
                <div className="min-w-0">
                  <p className="text-[var(--color-text)] text-sm font-medium truncate">{tip.title}</p>
                  <p className="text-[var(--color-text-muted)] text-xs">{tip.category}</p>
                </div>
              </div>
              {openIndex === i
                ? <ChevronUp className="w-4 h-4 text-[var(--color-text-muted)] flex-shrink-0" />
                : <ChevronDown className="w-4 h-4 text-[var(--color-text-muted)] flex-shrink-0" />}
            </button>

            {openIndex === i && (
              <div className="px-4 pb-4 space-y-3">
                <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed">{tip.description}</p>
                {tip.drill && (
                  <div className="bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20 rounded-lg p-3">
                    <p className="text-[var(--color-accent)] text-xs font-semibold uppercase tracking-wide mb-1">Practice Drill</p>
                    <p className="text-[var(--color-text-secondary)] text-sm leading-relaxed">{tip.drill}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
