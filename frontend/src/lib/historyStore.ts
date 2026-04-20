import type { SwingAnalysis, HistoryEntry, GolfClub } from '@/types';

const DB_NAME = 'happycoach';
const DB_VERSION = 1;
const STORE = 'swings';

function scoreToGrade(score: number): string {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveAnalysis(analysis: SwingAnalysis): Promise<void> {
  const db = await openDB();
  const entry: HistoryEntry = {
    id: analysis.id,
    timestamp: analysis.timestamp,
    club: analysis.club as GolfClub,
    overall_score: analysis.overall_score,
    grade: scoreToGrade(analysis.overall_score),
    summary: analysis.summary,
    analysis,
  };
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).put(entry);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
  db.close();
}

export async function getHistory(limit = 50): Promise<HistoryEntry[]> {
  const db = await openDB();
  const entries = await new Promise<HistoryEntry[]>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).index('timestamp').openCursor(null, 'prev');
    const results: HistoryEntry[] = [];
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor && results.length < limit) {
        results.push(cursor.value as HistoryEntry);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    req.onerror = () => reject(req.error);
  });
  db.close();
  return entries;
}

export async function getAnalysisById(id: string): Promise<HistoryEntry | null> {
  const db = await openDB();
  const entry = await new Promise<HistoryEntry | null>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(id);
    req.onsuccess = () => resolve((req.result as HistoryEntry) ?? null);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return entry;
}
