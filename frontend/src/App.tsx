import { useCallback, useState } from 'react';
import { RecordPage } from './pages/RecordPage';
import { AnalyzePage } from './pages/AnalyzePage';
import { HistoryPage } from './pages/HistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { Navigation, NavTab } from './components/ui/Navigation';
import { IOSInstallHint } from './components/ui/IOSInstallHint';

function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('record');
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);

  const handleRecordComplete = useCallback((blob: Blob) => {
    setRecordedBlob(blob);
    setActiveTab('analyze');
  }, []);

  const handleBlobConsumed = useCallback(() => setRecordedBlob(null), []);

  return (
    <div className="min-h-screen flex flex-col pt-[env(safe-area-inset-top)]">
      <div className="flex-1">
        {activeTab === 'record' && <RecordPage onRecordComplete={handleRecordComplete} />}
        {activeTab === 'analyze' && <AnalyzePage initialBlob={recordedBlob} onConsumed={handleBlobConsumed} />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'settings' && <SettingsPage />}
      </div>
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      <IOSInstallHint />
    </div>
  );
}

export default App;
