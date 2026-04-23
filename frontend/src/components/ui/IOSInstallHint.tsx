import { useEffect, useState } from 'react';
import { X, Share } from 'lucide-react';

const DISMISS_KEY = 'hc_ios_install_dismissed';

export function IOSInstallHint() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (localStorage.getItem(DISMISS_KEY) === '1') return;
    const ua = navigator.userAgent;
    const isIOS = /iPhone|iPad|iPod/.test(ua);
    const standalone = 'standalone' in navigator && (navigator as unknown as { standalone: boolean }).standalone;
    if (isIOS && !standalone) setVisible(true);
  }, []);

  if (!visible) return null;

  return (
    <div
      className="fixed left-3 right-3 z-50 bg-[var(--color-primary)] border border-[var(--color-accent)]/40 rounded-xl shadow-lg px-3 py-2 flex items-center gap-2"
      style={{ bottom: `calc(env(safe-area-inset-bottom) + 72px)` }}
      role="status"
    >
      <Share className="w-4 h-4 text-[var(--color-accent-bright)] flex-shrink-0" />
      <p className="text-xs text-[var(--color-text)] leading-tight flex-1">
        Add to Home Screen: tap <span className="font-semibold">Share</span>, then <span className="font-semibold">Add to Home Screen</span>.
      </p>
      <button
        onClick={() => { localStorage.setItem(DISMISS_KEY, '1'); setVisible(false); }}
        aria-label="Dismiss install hint"
        className="w-7 h-7 flex items-center justify-center text-[var(--color-text-muted)]"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
