/**
 * Bottom Navigation Component
 */

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
    <nav className="bg-[var(--color-primary)] border-t border-[var(--color-primary-light)]">
      <div className="flex items-center justify-around py-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-colors ${
              activeTab === tab.id
                ? 'text-[var(--color-accent)]'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
            }`}
          >
            {tab.icon}
            <span className="text-xs font-medium">{tab.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}