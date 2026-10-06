import { useEffect, useState } from 'react';
import type { Client } from '../api/client';
import type { ThreadSummary } from '../types';

function when(ts: string | null) {
  if (!ts) return '';
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function ThreadsPanel({
  client,
  currentId,
  onOpen,
  onClose,
  onDeleted,
}: {
  client: Client;
  currentId: string | null;
  onOpen: (id: string) => void;
  onClose: () => void;
  onDeleted: (id: string) => void;
}) {
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    client
      .threads()
      .then(setThreads)
      .catch((e: Error) => setError(e.message));
  }, [client]);

  return (
    <aside className="threads" aria-label="Conversations">
      <div className="threads-head">
        <h2>Conversations</h2>
        <button className="btn btn-quiet" onClick={onClose}>
          Close
        </button>
      </div>
      {error && <p className="tone-bad">Couldn’t load conversations: {error}</p>}
      {threads === null && !error && <p className="muted">Loading…</p>}
      {threads?.length === 0 && <p className="muted">No conversations yet. Ask a question to start one.</p>}
      <ul>
        {threads?.map((t) => (
          <li key={t.thread_id} className={t.thread_id === currentId ? 'is-current' : undefined}>
            <button className="thread-open" onClick={() => onOpen(t.thread_id)}>
              <span className="thread-q">{t.question || '(no question)'}</span>
              <span className="thread-meta">
                {t.turns} {t.turns === 1 ? 'question' : 'questions'}, {when(t.updated_at)}
              </span>
            </button>
            {confirming === t.thread_id ? (
              <span className="thread-confirm">
                <button
                  className="btn btn-danger"
                  onClick={async () => {
                    try {
                      await client.deleteThread(t.thread_id);
                      setThreads((ts) => ts?.filter((x) => x.thread_id !== t.thread_id) ?? null);
                      onDeleted(t.thread_id);
                    } catch (e) {
                      setError((e as Error).message);
                    }
                    setConfirming(null);
                  }}
                >
                  Delete
                </button>
                <button className="btn btn-quiet" onClick={() => setConfirming(null)}>
                  Keep
                </button>
              </span>
            ) : (
              <button className="btn btn-quiet" aria-label={`Delete conversation: ${t.question}`} onClick={() => setConfirming(t.thread_id)}>
                Delete
              </button>
            )}
          </li>
        ))}
      </ul>
      <p className="muted small">Deleting removes the conversation and every search result saved with it.</p>
    </aside>
  );
}
