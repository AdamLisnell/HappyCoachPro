import { Camera, Film, BarChart3, Settings } from 'lucide-react';

export type NavTab = 'record' | 'analyze' | 'history' | 'settings';

interface NavigationProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
}

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  const tabs: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: 'record', label: 'Record', icon: <Camera className="w-5 h-5" /> },
    { id: 'analyze', label: 'Analyze', icon: <Film className="w-5 h-5" /> },
    { id: 'history', label: 'History', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="w-5 h-5" /> },
  ];

  return (
    <nav
      className="bg-[var(--color-primary)] border-t border-[var(--color-primary-light)]"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      <div className="flex items-stretch justify-around py-1">
        {tabs.map((tab) => {
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              aria-current={active ? 'page' : undefined}
              className={`relative flex flex-col items-center justify-center gap-1 px-4 py-2 min-h-[48px] min-w-[56px] transition-colors ${
                active
                  ? 'text-[var(--color-accent-bright)]'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
              }`}
            >
              {tab.icon}
              <span className="text-[11px] font-medium tracking-wide">{tab.label}</span>
              <span
                className={`absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full transition-all ${
                  active ? 'bg-[var(--color-accent-bright)] opacity-100' : 'bg-transparent opacity-0'
                }`}
              />
            </button>
          );
        })}
      </div>
    </nav>
  );
}
