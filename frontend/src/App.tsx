import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { ApiError, createClient, type Client } from './api/client';
import { AnswerPanel } from './components/AnswerPanel';
import { AskForm, type AskOptions } from './components/AskForm';
import { ReviewPanel } from './components/ReviewPanel';
import { RouteMap } from './components/RouteMap';
import { ThreadsPanel } from './components/ThreadsPanel';
import { DEMO_EVENTS, DEMO_QUESTION, replay } from './demo/replay';
import { initialRun, reduceAll, runReducer } from './state/run';
import type { ChatMessage, Health, ReviewRequest, RunEvent } from './types';

const HOSTED_PREVIEW = import.meta.env.VITE_HOSTED_PREVIEW === '1';
const STORAGE_KEY = 'research-copilot.api-url';

function storedUrl(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}
function storeUrl(url: string) {
  try {
    localStorage.setItem(STORAGE_KEY, url);
  } catch {
    /* storage blocked: the URL lasts for this visit */
  }
}

type Connection =
  | { state: 'checking' }
  | { state: 'live'; health: Health }
  | { state: 'demo'; reason: string };

function pairs(messages: ChatMessage[]) {
  const out: { question: string; answer: string }[] = [];
  for (let i = 0; i + 1 < messages.length; i++) {
    if (messages[i].role === 'user' && messages[i + 1].role === 'assistant') {
      out.push({ question: messages[i].content, answer: messages[i + 1].content });
      i++;
    }
  }
  return out;
}

export default function App({ clientFactory = createClient }: { clientFactory?: (base: string) => Client }) {
  const [apiUrl, setApiUrl] = useState(() => storedUrl() ?? import.meta.env.VITE_API_URL ?? '');
  const client = useMemo(() => clientFactory(apiUrl), [apiUrl, clientFactory]);
  const [conn, setConn] = useState<Connection>(
    HOSTED_PREVIEW ? { state: 'demo', reason: 'preview' } : { state: 'checking' },
  );
  const demo = conn.state !== 'live';

  // A demo opens on the finished run, so the page shows what it does before anyone clicks.
  const [run, dispatch] = useReducer(runReducer, initialRun, () =>
    HOSTED_PREVIEW ? reduceAll(DEMO_EVENTS, DEMO_QUESTION) : initialRun,
  );
  const [history, setHistory] = useState<{ question: string; answer: string }[]>([]);
  const [showThreads, setShowThreads] = useState(false);
  const [showConn, setShowConn] = useState(false);
  const [urlDraft, setUrlDraft] = useState(apiUrl);
  const abort = useRef<AbortController | null>(null);
  const busy = run.status === 'running';

  useEffect(() => {
    if (HOSTED_PREVIEW) return;
    let cancelled = false;
    setConn({ state: 'checking' });
    client
      .health()
      .then((health) => !cancelled && setConn({ state: 'live', health }))
      .catch(() => {
        if (cancelled) return;
        setConn({ state: 'demo', reason: `Couldn’t reach the API at ${apiUrl || 'this site'}.` });
        dispatch({ kind: 'load', state: reduceAll(DEMO_EVENTS, DEMO_QUESTION) });
      });
    return () => {
      cancelled = true;
    };
  }, [client, apiUrl]);

  const consume = useCallback(async (events: AsyncGenerator<RunEvent>) => {
    try {
      for await (const event of events) dispatch({ kind: 'event', event });
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      const err = e as Error;
      dispatch({
        kind: 'fail',
        title: err instanceof ApiError ? 'The server refused the request' : 'Lost the connection to the API',
        detail: err.message,
      });
    }
  }, []);

  const ask = useCallback(
    async (question: string, options: AskOptions) => {
      abort.current?.abort();
      const ctrl = new AbortController();
      abort.current = ctrl;
      if (demo) {
        dispatch({ kind: 'start', question: DEMO_QUESTION, threadId: null });
        await consume(replay(DEMO_EVENTS, { signal: ctrl.signal }));
        return;
      }
      const followUp = run.threadId && run.status === 'done' && run.answer != null;
      if (followUp) setHistory((h) => [...h, { question: run.question, answer: run.answer! }]);
      else if (!run.threadId) setHistory([]);
      dispatch({ kind: 'start', question, threadId: run.threadId });
      await consume(client.run({ question, thread_id: run.threadId, ...options }, ctrl.signal));
    },
    [client, consume, demo, run.answer, run.question, run.status, run.threadId],
  );

  const stop = () => {
    abort.current?.abort();
    dispatch({ kind: 'fail', title: 'Stopped', detail: 'You stopped watching this run. On the server it may still finish; reopen the conversation to see the result.' });
  };

  const review = async (req: ReviewRequest) => {
    if (!run.threadId) return;
    const ctrl = new AbortController();
    abort.current = ctrl;
    dispatch({ kind: 'resume', decision: req.decision === 'approve' ? 'approve' : req.decision === 'edit' ? 'use my edit' : 'send back' });
    await consume(client.review(run.threadId, req, ctrl.signal));
  };

  const openThread = async (id: string) => {
    setShowThreads(false);
    try {
      const view = await client.thread(id);
      const msgs = view.pending_review ? view.messages.slice(0, -1) : view.messages;
      const turns = pairs(msgs);
      // Parked: every completed pair is history. Otherwise the last pair is the current turn.
      const last = view.pending_review ? undefined : turns.pop();
      setHistory(turns);
      dispatch({ kind: 'restore', threadId: id, question: view.pending_review?.question ?? last?.question ?? view.question, pending: view.pending_review });
      if (last && !view.pending_review) {
        dispatch({ kind: 'event', event: { type: 'done', answer: last.answer, dispatches: {}, revisions: 0, verdict: '', unsupported_citations: [], interventions: 0 } });
      }
      if (view.pending_review) {
        dispatch({ kind: 'event', event: { type: 'paused', payload: view.pending_review } });
      }
    } catch (e) {
      dispatch({ kind: 'fail', title: 'Couldn’t open that conversation', detail: (e as Error).message });
    }
  };

  const newConversation = () => {
    abort.current?.abort();
    setHistory([]);
    dispatch({ kind: 'reset' });
  };

  const status =
    conn.state === 'checking'
      ? { text: 'Connecting…', tone: 'muted' }
      : conn.state === 'live'
        ? { text: `${conn.health.provider}, ${conn.health.model}`, tone: conn.health.api_key_set ? 'good' : 'warn' }
        : { text: 'Demo replay', tone: 'muted' };

  return (
    <div className="app">
      <header className="top">
        <h1 className="brand">Research Copilot</h1>
        <nav className="top-actions">
          {!demo && (
            <>
              <button className="btn btn-quiet" onClick={newConversation} disabled={busy}>
                New conversation
              </button>
              <button className="btn btn-quiet" onClick={() => setShowThreads((v) => !v)} aria-expanded={showThreads}>
                Conversations
              </button>
            </>
          )}
          {HOSTED_PREVIEW ? (
            <span className={`pill pill-${status.tone}`}>{status.text}</span>
          ) : (
            <button className={`pill pill-${status.tone}`} onClick={() => setShowConn((v) => !v)} aria-expanded={showConn}>
              {status.text}
            </button>
          )}
        </nav>
      </header>

      {showConn && !HOSTED_PREVIEW && (
        <form
          className="conn"
          onSubmit={(e) => {
            e.preventDefault();
            const url = urlDraft.trim();
            storeUrl(url);
            setApiUrl(url);
            setShowConn(false);
          }}
        >
          <label htmlFor="api-url">API address</label>
          <input id="api-url" value={urlDraft} onChange={(e) => setUrlDraft(e.target.value)} placeholder="https://research-copilot-api.onrender.com" />
          <button type="submit" className="btn btn-primary">
            Connect
          </button>
          {conn.state === 'live' && !conn.health.api_key_set && (
            <p className="tone-warn small">Connected, but the server has no {conn.health.provider} API key set. Runs will stop at the first model call.</p>
          )}
          {conn.state === 'demo' && <p className="muted small">{conn.reason} Leave it empty to use this site’s own address.</p>}
        </form>
      )}

      {HOSTED_PREVIEW && (
        <p className="notice">
          This preview replays a recorded run: the real graph and event stream, with scripted model replies. Live questions need the
          backend deployed.
        </p>
      )}

      <AskForm
        key={demo ? 'demo' : 'live'}
        busy={busy}
        demo={demo}
        followUp={!demo && !!run.threadId && run.status === 'done'}
        initialQuestion={demo ? DEMO_QUESTION : ''}
        onAsk={ask}
        onStop={stop}
        demoHint={
          HOSTED_PREVIEW
            ? 'Replays a recorded run of this question.'
            : 'Replays a recorded run of this question. To ask your own, connect an API from the address pill above.'
        }
      />

      {run.problem && (
        <div className={`problem${run.status === 'error' ? ' is-error' : ''}`} role="alert">
          <strong>{run.problem.title}.</strong> {run.problem.detail}
        </div>
      )}

      <main className="board">
        <RouteMap rows={run.rows} status={run.status} />
        <div className="answer-col">
          <AnswerPanel run={run} history={history} />
          {run.status === 'paused' && run.pending && !demo && (
            <ReviewPanel key={run.pending.draft} payload={run.pending} busy={busy} onSubmit={review} />
          )}
        </div>
      </main>

      {showThreads && !demo && (
        <ThreadsPanel
          client={client}
          currentId={run.threadId}
          onOpen={openThread}
          onClose={() => setShowThreads(false)}
          onDeleted={(id) => id === run.threadId && newConversation()}
        />
      )}
    </div>
  );
}
