import { useState } from 'react';
import type { RunRequest } from '../types';

export interface AskOptions {
  mode: RunRequest['mode'];
  routing: RunRequest['routing'];
  critic: boolean;
  approve: boolean;
  plan: boolean;
  max_revisions: number;
}

export const DEFAULT_OPTIONS: AskOptions = {
  mode: 'live-search',
  routing: 'supervisor',
  critic: true,
  approve: false,
  plan: false,
  max_revisions: 2,
};

export function AskForm({
  busy,
  demo,
  followUp,
  initialQuestion,
  onAsk,
  onStop,
  demoHint = 'Replays a recorded run of this question.',
}: {
  busy: boolean;
  demo: boolean;
  followUp: boolean;
  initialQuestion: string;
  onAsk: (question: string, options: AskOptions) => void;
  onStop: () => void;
  demoHint?: string;
}) {
  const [question, setQuestion] = useState(initialQuestion);
  const [options, setOptions] = useState<AskOptions>(DEFAULT_OPTIONS);
  const set = <K extends keyof AskOptions>(k: K, v: AskOptions[K]) => setOptions((o) => ({ ...o, [k]: v }));
  const canAsk = demo || question.trim().length > 0;

  return (
    <form
      className="ask"
      onSubmit={(e) => {
        e.preventDefault();
        if (busy) return onStop();
        if (!canAsk) return;
        onAsk(question.trim(), options);
        if (!demo) setQuestion('');
      }}
    >
      <label htmlFor="question" className="ask-label">
        {followUp ? 'Ask a follow-up in this conversation' : 'Research question'}
      </label>
      <div className="ask-row">
        <textarea
          id="question"
          rows={2}
          value={question}
          readOnly={demo}
          maxLength={4000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) e.currentTarget.form?.requestSubmit();
          }}
          placeholder="e.g. What are the main approaches to evaluating retrieval-augmented generation?"
        />
        <button type="submit" className={`btn ${busy ? '' : 'btn-primary'} ask-go`} disabled={!busy && !canAsk}>
          {busy ? 'Stop' : demo ? 'Replay run' : 'Run'}
        </button>
      </div>

      {demo ? (
        <p className="ask-hint">{demoHint}</p>
      ) : (
        <fieldset className="options" disabled={busy}>
          <legend className="sr-only">Run options</legend>
          <label className="opt">
            <span>Sources</span>
            <select id="opt-mode" value={options.mode} onChange={(e) => set('mode', e.target.value as AskOptions['mode'])}>
              <option value="live-search">arXiv search</option>
              <option value="knowledge-base">My ingested documents</option>
            </select>
          </label>
          <label className="opt">
            <span>Hand-offs</span>
            <select id="opt-routing" value={options.routing} onChange={(e) => set('routing', e.target.value as AskOptions['routing'])}>
              <option value="supervisor">Supervisor decides</option>
              <option value="fixed">Fixed order (no model)</option>
            </select>
          </label>
          <label className="opt opt-check">
            <input id="opt-critic" type="checkbox" checked={options.critic} onChange={(e) => set('critic', e.target.checked)} />
            <span>Critic checks drafts</span>
          </label>
          <label className="opt">
            <span>Revisions</span>
            <select id="opt-revisions" value={options.max_revisions} onChange={(e) => set('max_revisions', Number(e.target.value))}>
              {[0, 1, 2, 3].map((n) => (
                <option key={n} value={n}>
                  up to {n}
                </option>
              ))}
            </select>
          </label>
          <label className="opt opt-check">
            <input id="opt-approve" type="checkbox" checked={options.approve} onChange={(e) => set('approve', e.target.checked)} />
            <span>I approve before it’s final</span>
          </label>
          <label className="opt opt-check">
            <input id="opt-plan" type="checkbox" checked={options.plan} onChange={(e) => set('plan', e.target.checked)} />
            <span>Split into sub-questions</span>
          </label>
        </fieldset>
      )}
    </form>
  );
}
