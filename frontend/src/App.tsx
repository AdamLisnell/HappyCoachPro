import { useState } from 'react';
import { RecordPage } from './pages/RecordPage';
import { AnalyzePage } from './pages/AnalyzePage';
import { HistoryPage } from './pages/HistoryPage';
import { Navigation, NavTab } from './components/ui/Navigation';

function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('record');

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1">
        {activeTab === 'record' && <RecordPage />}
        {activeTab === 'analyze' && <AnalyzePage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'settings' && (
          <div className="flex items-center justify-center h-full">
            <p className="text-[var(--color-text-muted)]">Settings coming soon...</p>
          </div>
        )}
      </div>
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  );
}

export default App;