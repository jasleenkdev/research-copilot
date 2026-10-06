import { DEMO_EVENTS, DEMO_QUESTION } from '../demo/replay';
import type { RunEvent } from '../types';
import { initialRun, reduceAll, runReducer } from './run';

const ev = (s: ReturnType<typeof runReducer>, event: RunEvent) => runReducer(s, { kind: 'event', event });

describe('the recorded demo run', () => {
  const s = reduceAll(DEMO_EVENTS, DEMO_QUESTION);

  it('ends done, with the final answer and a summary', () => {
    expect(s.status).toBe('done');
    expect(s.answer).toContain('RAGAS');
    expect(s.summary?.revisions).toBe(1);
  });

  it('draws the hand-offs in order', () => {
    const route = s.rows.filter((r) => r.kind === 'decision').map((r) => (r.kind === 'decision' ? r.to : ''));
    expect(route).toEqual(['researcher', 'writer', 'critic', 'writer', 'critic']);
  });

  it('keeps each draft with its verdict; the rejected one still holds its text', () => {
    expect(s.drafts.map((d) => d.verdict)).toEqual(['reject', 'approve']);
    expect(s.drafts[0].text).toContain('TruLens');
    expect(s.drafts[1].text).not.toContain('TruLens');
  });

  it('attaches tool calls to the agent that made them, and no row is left running', () => {
    const researcher = s.rows.find((r) => r.kind === 'work' && r.agent === 'researcher');
    expect(researcher && researcher.kind === 'work' && researcher.activity.map((a) => a.detail)).toEqual([
      'RAG evaluation framework',
      'automated evaluation retrieval augmented generation judges',
    ]);
    expect(s.rows.some((r) => r.kind === 'work' && r.running)).toBe(false);
  });

  it('marks the revision round', () => {
    expect(s.rows.some((r) => r.kind === 'marker' && r.label === 'Revision 1')).toBe(true);
  });
});

describe('endings', () => {
  const running = ev(ev(runReducer(initialRun, { kind: 'start', question: 'q', threadId: null }), { type: 'thread', thread_id: 't', is_new: true, policy: {} }), {
    type: 'node', phase: 'start', agent: 'researcher', node: 'researcher', depth: 0,
  });

  it('a stop explains itself and stops every running row', () => {
    const s = ev(running, { type: 'stopped', reason: 'quota_exhausted', detail: 'daily limit' });
    expect(s.status).toBe('stopped');
    expect(s.problem?.title).toMatch(/quota/);
    expect(s.rows.some((r) => r.kind === 'work' && r.running)).toBe(false);
  });

  it('an error is shown with its kind', () => {
    const s = ev(running, { type: 'error', kind: 'other', detail: 'KeyError: x' });
    expect(s.status).toBe('error');
    expect(s.problem?.detail).toBe('other: KeyError: x');
  });

  it('a pause holds the review payload, and resuming clears it', () => {
    const paused = ev(running, { type: 'paused', payload: { question: 'q', mode: 'live-search', draft: 'D', critique: '' } });
    expect(paused.status).toBe('paused');
    expect(paused.pending?.draft).toBe('D');
    const resumed = runReducer(paused, { kind: 'resume', decision: 'approve' });
    expect(resumed.status).toBe('running');
    expect(resumed.pending).toBeNull();
    expect(resumed.threadId).toBe('t');
  });

  it('an intervention lands on the running agent, live', () => {
    const s = ev(running, { type: 'intervention', node: 'researcher', kind: 'hint_retry', detail: 'tool_use_failed' });
    const row = s.rows.at(-1)!;
    expect(row.kind === 'work' && row.activity[0]).toEqual({ label: 'Retried after a malformed reply', detail: 'tool_use_failed', tone: 'warn' });
  });

  it('tokens only from the Writer go into the draft', () => {
    let s = ev(running, { type: 'node', phase: 'start', agent: 'writer', node: 'writer', depth: 0 });
    s = ev(s, { type: 'token', agent: 'critic', text: 'APPROVE' });
    s = ev(s, { type: 'token', agent: 'writer', text: 'Hello' });
    expect(s.drafts).toEqual([{ text: 'Hello' }]);
  });
});

describe('the Critic row', () => {
  const s = reduceAll(DEMO_EVENTS, DEMO_QUESTION);
  const critics = s.rows.filter((r) => r.kind === 'work' && r.agent === 'critic');

  it('carries its verdict even though the verdict arrives as the node ends', () => {
    expect(critics.map((r) => r.kind === 'work' && r.verdict)).toEqual(['reject', 'approve']);
  });

  it('shows each lookup once, with its result', () => {
    const first = critics[0];
    expect(first.kind === 'work' && first.activity.map((a) => a.detail)).toEqual(['2309.15217: found', '2311.09476: found', '2401.00001: found']);
  });
});

it('a resumed run does not announce the review gate twice', () => {
  let s = runReducer(initialRun, { kind: 'start', question: 'q', threadId: 't' });
  s = ev(s, { type: 'node', phase: 'start', agent: 'review_draft', node: 'review_draft', depth: 0 });
  s = runReducer(s, { kind: 'resume', decision: 'approve' });
  s = ev(s, { type: 'node', phase: 'start', agent: 'review_draft', node: 'review_draft', depth: 0 });
  expect(s.rows.map((r) => r.kind === 'marker' && r.label)).toEqual(['Waiting for your review', 'You chose: approve']);
});
