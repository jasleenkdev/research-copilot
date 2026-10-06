import { useEffect, useState } from 'react';
import Markdown from 'react-markdown';
import { citedIds, linkCitations } from '../citations';
import type { RunState } from '../state/run';
import type { CitationStatus } from '../types';
import { AGENT_LABEL } from './RouteMap';

const STATUS_TEXT: Record<CitationStatus, string> = {
  found: 'Found on arXiv',
  not_found: 'Not found on arXiv',
  invalid: 'Not a valid arXiv id',
  error: 'Lookup failed',
};

export function AnswerText({ text }: { text: string }) {
  return (
    <div className="prose">
      <Markdown
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer" className={href?.includes('arxiv.org/abs/') ? 'cite' : undefined}>
              {children}
            </a>
          ),
        }}
      >
        {linkCitations(text)}
      </Markdown>
    </div>
  );
}

function draftLabel(i: number, verdict?: string) {
  const v = verdict === 'approve' ? 'approved' : verdict === 'reject' ? 'rejected' : verdict ? 'review incomplete' : '';
  return `Draft ${i + 1}${v ? ` (${v})` : ''}`;
}

export function AnswerPanel({ run, history }: { run: RunState; history: { question: string; answer: string }[] }) {
  const [shown, setShown] = useState<number | null>(null); // null = follow the latest
  useEffect(() => setShown(null), [run.question, run.threadId]);

  const latest = run.drafts.length - 1;
  const index = shown ?? latest;
  const final = run.status === 'done' && shown === null && run.answer != null;
  const text = final ? run.answer! : run.drafts[index]?.text ?? '';
  const writing = run.status === 'running' && index === latest && run.rows.some((r) => r.kind === 'work' && r.agent === 'writer' && r.running);
  const checks = new Map(run.citationChecks.map((c) => [c.arxiv_id, c.status]));
  const ids = citedIds(text);

  return (
    <section className="answer" aria-label="Answer">
      {history.length > 0 && (
        <details className="history">
          <summary>
            Earlier in this conversation ({history.length} {history.length === 1 ? 'question' : 'questions'})
          </summary>
          {history.map((t, i) => (
            <div key={i} className="history-turn">
              <p className="history-q">{t.question}</p>
              <AnswerText text={t.answer} />
            </div>
          ))}
        </details>
      )}

      <h2 className="question">{run.question || 'Ask about any research topic'}</h2>

      {run.drafts.length > 1 && (
        <div className="drafts" role="tablist" aria-label="Drafts">
          {run.drafts.map((d, i) => (
            <button
              key={i}
              role="tab"
              aria-selected={i === index && !final ? true : final && i === latest}
              className={`draft-tab${d.verdict === 'reject' ? ' is-rejected' : ''}`}
              onClick={() => setShown(i === latest ? null : i)}
            >
              {draftLabel(i, d.verdict)}
            </button>
          ))}
        </div>
      )}

      {text ? (
        <>
          {shown !== null && shown !== latest && run.drafts[shown]?.critique && (
            <p className="critique-note">
              <strong>Critic:</strong> {run.drafts[shown].critique}
            </p>
          )}
          <AnswerText text={text} />
          {writing && <span className="caret" aria-hidden="true" />}
        </>
      ) : (
        <p className="answer-empty">
          {run.status === 'running'
            ? 'The Writer starts once the Researcher has notes.'
            : run.status === 'paused'
              ? 'Review the draft below.'
              : 'The answer appears here, with every arXiv citation linked and checked.'}
        </p>
      )}

      {ids.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>
          <ul>
            {ids.map((id) => {
              const st = checks.get(id);
              return (
                <li key={id}>
                  <a href={`https://arxiv.org/abs/${id}`} target="_blank" rel="noreferrer">
                    arXiv {id}
                  </a>
                  <span className={`check check-${st ?? 'none'}`}>{st ? STATUS_TEXT[st] : 'Not checked'}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {run.summary && (
        <p className="summary">
          {Object.entries(run.summary.dispatches)
            .map(([a, n]) => `${AGENT_LABEL[a] ?? a} ran ${n}×`)
            .join(', ')}
          {`. ${run.summary.revisions} ${run.summary.revisions === 1 ? 'revision' : 'revisions'}.`}
          {run.summary.unsupported.length > 0 && ` Cited but not in the notes: ${run.summary.unsupported.join(', ')}.`}
          {run.summary.interventions > 0 && ` ${run.summary.interventions} retries or trims.`}
        </p>
      )}
    </section>
  );
}
