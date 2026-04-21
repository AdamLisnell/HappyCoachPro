import { useState } from 'react';
import { RecordPage } from './pages/RecordPage';
import { AnalyzePage } from './pages/AnalyzePage';
import { HistoryPage } from './pages/HistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { Navigation, NavTab } from './components/ui/Navigation';

function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('record');

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1">
        {activeTab === 'record' && <RecordPage />}
        {activeTab === 'analyze' && <AnalyzePage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'settings' && <SettingsPage />}
      </div>
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  );
}

export default App;