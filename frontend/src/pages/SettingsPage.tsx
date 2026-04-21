import { useState, useEffect } from 'react';
import { Settings, Camera, Trash2, ChevronRight, CheckCircle } from 'lucide-react';
import type { GolfClub } from '@/types';

type Angle = 'side' | 'behind';

const CLUB_OPTIONS: { value: GolfClub; label: string }[] = [
  { value: 'driver', label: 'Driver' },
  { value: 'wood_3', label: '3 Wood' },
  { value: 'iron_4', label: '4 Iron' },
  { value: 'iron_5', label: '5 Iron' },
  { value: 'iron_6', label: '6 Iron' },
  { value: 'iron_7', label: '7 Iron' },
  { value: 'iron_8', label: '8 Iron' },
  { value: 'iron_9', label: '9 Iron' },
  { value: 'pitching_wedge', label: 'PW' },
  { value: 'sand_wedge', label: 'SW' },
  { value: 'lob_wedge', label: 'LW' },
  { value: 'putter', label: 'Putter' },
];

const ANGLE_OPTIONS: { value: Angle; label: string; description: string }[] = [
  { value: 'side', label: 'Side-on', description: 'Camera perpendicular to target line' },
  { value: 'behind', label: 'Behind (DTL)', description: 'Camera looking down the target line' },
];

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 px-1 mb-2">
      <span className="text-[var(--color-accent)]">{icon}</span>
      <span className="text-[var(--color-text-muted)] text-xs font-semibold uppercase tracking-wider">{title}</span>
    </div>
  );
}

export function SettingsPage() {
  const [defaultClub, setDefaultClub] = useState<GolfClub>('iron_7');
  const [defaultAngle, setDefaultAngle] = useState<Angle>('side');
  const [cleared, setCleared] = useState(false);

  useEffect(() => {
    const club = localStorage.getItem('hc_default_club') as GolfClub | null;
    const angle = localStorage.getItem('hc_default_angle') as Angle | null;
    if (club) setDefaultClub(club);
    if (angle) setDefaultAngle(angle);
  }, []);

  const handleClubChange = (club: GolfClub) => {
    setDefaultClub(club);
    localStorage.setItem('hc_default_club', club);
  };

  const handleAngleChange = (angle: Angle) => {
    setDefaultAngle(angle);
    localStorage.setItem('hc_default_angle', angle);
  };

  const handleClearHistory = async () => {
    try {
      await new Promise<void>((resolve, reject) => {
        const req = indexedDB.deleteDatabase('happycoach');
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
      setCleared(true);
      setTimeout(() => setCleared(false), 3000);
    } catch {
      // ignore
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-surface)] flex flex-col">
      <header className="bg-[var(--color-primary)] px-4 py-3 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-surface-card)] flex items-center justify-center border border-[var(--color-accent)]/30">
          <Settings className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-[var(--color-text)]">Settings</h1>
          <p className="text-xs text-[var(--color-accent)] uppercase tracking-wider">Preferences</p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">

        {/* Default club */}
        <div>
          <SectionHeader icon={<ChevronRight className="w-4 h-4" />} title="Default Club" />
          <div className="bg-[var(--color-surface-card)] rounded-xl p-4">
            <p className="text-[var(--color-text-muted)] text-xs mb-3">Pre-selected when you open the Analyze tab</p>
            <div className="flex flex-wrap gap-2">
              {CLUB_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => handleClubChange(opt.value)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    defaultClub === opt.value
                      ? 'bg-[var(--color-accent)] text-[var(--color-primary-dark)]'
                      : 'bg-[var(--color-primary)] text-[var(--color-text-secondary)]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Default camera angle */}
        <div>
          <SectionHeader icon={<Camera className="w-4 h-4" />} title="Default Camera Angle" />
          <div className="bg-[var(--color-surface-card)] rounded-xl p-4 space-y-2">
            <p className="text-[var(--color-text-muted)] text-xs mb-3">Pre-selected when you open the Analyze tab</p>
            {ANGLE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => handleAngleChange(opt.value)}
                className={`w-full flex items-center justify-between p-3 rounded-lg border transition-colors text-left ${
                  defaultAngle === opt.value
                    ? 'bg-[var(--color-accent)]/15 border-[var(--color-accent)]'
                    : 'bg-[var(--color-primary)] border-transparent'
                }`}
              >
                <div>
                  <p className={`text-sm font-medium ${defaultAngle === opt.value ? 'text-[var(--color-text)]' : 'text-[var(--color-text-secondary)]'}`}>
                    {opt.label}
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)] mt-0.5">{opt.description}</p>
                </div>
                {defaultAngle === opt.value && (
                  <CheckCircle className="w-5 h-5 text-[var(--color-accent)] flex-shrink-0" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Data */}
        <div>
          <SectionHeader icon={<Trash2 className="w-4 h-4" />} title="Data" />
          <div className="bg-[var(--color-surface-card)] rounded-xl overflow-hidden">
            <button
              onClick={handleClearHistory}
              disabled={cleared}
              className="w-full flex items-center justify-between px-4 py-4 transition-colors hover:bg-[var(--color-primary)] disabled:opacity-60"
            >
              <div className="text-left">
                <p className="text-sm font-medium text-red-400">Clear Swing History</p>
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">Permanently deletes all saved analyses</p>
              </div>
              {cleared ? (
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
              ) : (
                <Trash2 className="w-5 h-5 text-red-400 flex-shrink-0" />
              )}
            </button>
          </div>
          {cleared && (
            <p className="text-green-400 text-xs text-center mt-2">History cleared</p>
          )}
        </div>

        {/* About */}
        <div>
          <SectionHeader icon={<Settings className="w-4 h-4" />} title="About" />
          <div className="bg-[var(--color-surface-card)] rounded-xl px-4 py-3 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-[var(--color-text-muted)]">App</span>
              <span className="text-[var(--color-text-secondary)]">HappyCoachPro</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--color-text-muted)]">AI Model</span>
              <span className="text-[var(--color-text-secondary)]">Claude Sonnet 4.6</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--color-text-muted)]">Pose Engine</span>
              <span className="text-[var(--color-text-secondary)]">MediaPipe (on-device)</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-[var(--color-text-muted)]">Analysis FPS</span>
              <span className="text-[var(--color-text-secondary)]">24 fps</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
