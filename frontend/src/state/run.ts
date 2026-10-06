import type { AgentName, CitationCheck, Policy, ReviewPayload, RunEvent } from '../types';

/**
 * One run, as the route map and the answer panel draw it.
 *
 * Rows are in arrival order. A `decision` row is the Supervisor handing off;
 * a `work` row is one agent's pass, filled in by the tool calls,
 * interventions and verdict that arrive while it runs; a `marker` row is a
 * turn-level event (a revision round starting, the answer being finalized).
 */
export interface Activity {
  label: string;
  detail?: string;
  tone?: 'warn';
}

export type Row =
  | { kind: 'decision'; id: number; step: number; to: string; rationale: string; override: string; proposed: string | null; brief: string }
  | {
      kind: 'work';
      id: number;
      agent: AgentName;
      running: boolean;
      activity: Activity[];
      verdict?: string;
      critique?: string;
      draftIndex?: number;
    }
  | { kind: 'marker'; id: number; label: string; tone?: 'warn' | 'good' };

export interface Draft {
  text: string;
  verdict?: string;
  critique?: string;
}

export type RunStatus = 'idle' | 'running' | 'paused' | 'done' | 'stopped' | 'error';

export interface RunState {
  status: RunStatus;
  question: string;
  threadId: string | null;
  policy: Policy | null;
  rows: Row[];
  drafts: Draft[];
  citationChecks: CitationCheck[];
  pending: ReviewPayload | null;
  answer: string | null;
  summary: { dispatches: Record<string, number>; revisions: number; unsupported: string[]; interventions: number } | null;
  problem: { title: string; detail: string } | null;
  nextId: number;
}

export const initialRun: RunState = {
  status: 'idle',
  question: '',
  threadId: null,
  policy: null,
  rows: [],
  drafts: [],
  citationChecks: [],
  pending: null,
  answer: null,
  summary: null,
  problem: null,
  nextId: 1,
};

export type RunAction =
  | { kind: 'start'; question: string; threadId: string | null }
  | { kind: 'resume'; decision: string }
  | { kind: 'event'; event: RunEvent }
  | { kind: 'fail'; title: string; detail: string }
  | { kind: 'restore'; threadId: string; question: string; pending: ReviewPayload | null }
  | { kind: 'load'; state: RunState }
  | { kind: 'reset' };

const WORKERS = new Set(['researcher', 'writer', 'critic']);

const STOP_REASONS: Record<string, string> = {
  quota_exhausted: 'The model provider’s quota is used up',
  model_not_available: 'The configured model isn’t available to this API key',
  not_configured: 'The backend isn’t configured to call a model',
};

function describeIntervention(e: Extract<RunEvent, { type: 'intervention' }>): Activity {
  switch (e.kind ?? 'trim') {
    case 'trim':
      return { label: `Trimmed ${e.part ?? 'the request'}`, detail: `${e.tokens_before} → ${e.tokens_after} tokens (limit ${e.limit})`, tone: 'warn' };
    case 'hint_retry':
      return { label: 'Retried after a malformed reply', detail: e.detail, tone: 'warn' };
    case 'too_large_retry':
      return { label: 'Retried with a smaller request', detail: e.detail, tone: 'warn' };
    case 'fallback_model':
      return { label: 'Switched to the fallback model', detail: e.detail, tone: 'warn' };
    case 'gave_up':
      return { label: 'Gave up after retrying', detail: e.detail, tone: 'warn' };
    default:
      return { label: e.kind ?? 'Intervention', detail: e.detail, tone: 'warn' };
  }
}

function lastRunning(rows: Row[], agent: string): number {
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i];
    if (r.kind === 'work' && r.agent === agent && r.running) return i;
  }
  return -1;
}

function lastOf(rows: Row[], agent: string): number {
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i];
    if (r.kind === 'work' && r.agent === agent) return i;
  }
  return -1;
}

function updateRow(rows: Row[], index: number, fn: (r: Extract<Row, { kind: 'work' }>) => Row): Row[] {
  if (index < 0) return rows;
  const row = rows[index];
  if (row.kind !== 'work') return rows;
  const next = rows.slice();
  next[index] = fn(row);
  return next;
}

function onEvent(s: RunState, e: RunEvent): RunState {
  switch (e.type) {
    case 'thread':
      return { ...s, threadId: e.thread_id, policy: e.policy, status: 'running' };

    case 'node': {
      if (e.depth !== 0) return s;
      if (e.phase === 'start' && WORKERS.has(e.node)) {
        const isWriter = e.node === 'writer';
        const row: Row = {
          kind: 'work',
          id: s.nextId,
          agent: e.node as AgentName,
          running: true,
          activity: [],
          ...(isWriter ? { draftIndex: s.drafts.length } : {}),
        };
        return {
          ...s,
          rows: [...s.rows, row],
          drafts: isWriter ? [...s.drafts, { text: '' }] : s.drafts,
          nextId: s.nextId + 1,
        };
      }
      if (e.phase === 'end' && WORKERS.has(e.node)) {
        return { ...s, rows: updateRow(s.rows, lastRunning(s.rows, e.node), (r) => ({ ...r, running: false })) };
      }
      if (e.phase === 'start' && e.node === 'start_revision') {
        const round = s.rows.filter((r) => r.kind === 'marker' && r.label.startsWith('Revision')).length + 1;
        return { ...s, rows: [...s.rows, { kind: 'marker', id: s.nextId, label: `Revision ${round}` }], nextId: s.nextId + 1 };
      }
      if (e.phase === 'start' && e.node === 'review_draft') {
        // A resume re-enters this node; the gate was already announced.
        const last = s.rows[s.rows.length - 1];
        if (last?.kind === 'marker' && last.label.startsWith('You chose')) return s;
        return { ...s, rows: [...s.rows, { kind: 'marker', id: s.nextId, label: 'Waiting for your review' }], nextId: s.nextId + 1 };
      }
      return s;
    }

    case 'decision':
      return {
        ...s,
        rows: [
          ...s.rows,
          {
            kind: 'decision',
            id: s.nextId,
            step: e.step,
            to: e.routed_to,
            rationale: e.rationale,
            override: e.override,
            proposed: e.proposed,
            brief: e.brief,
          },
        ],
        nextId: s.nextId + 1,
      };

    case 'tool': {
      if (e.phase !== 'start') return s;
      const activity: Activity =
        e.name === 'search_arxiv'
          ? { label: 'Searched arXiv', detail: e.arg }
          : e.name === 'verify_citation'
            ? { label: 'Looked up', detail: e.arg }
            : { label: e.name, detail: e.arg };
      return { ...s, rows: updateRow(s.rows, lastRunning(s.rows, e.agent), (r) => ({ ...r, activity: [...r.activity, activity] })) };
    }

    case 'intervention': {
      const agent = e.node ?? '';
      const idx = lastRunning(s.rows, agent);
      const activity = describeIntervention(e);
      if (idx >= 0) return { ...s, rows: updateRow(s.rows, idx, (r) => ({ ...r, activity: [...r.activity, activity] })) };
      return {
        ...s,
        rows: [...s.rows, { kind: 'marker', id: s.nextId, label: `${agent || 'Run'}: ${activity.label}`, tone: 'warn' }],
        nextId: s.nextId + 1,
      };
    }

    case 'token': {
      if (e.agent !== 'writer' || s.drafts.length === 0) return s;
      const drafts = s.drafts.slice();
      const last = drafts[drafts.length - 1];
      drafts[drafts.length - 1] = { ...last, text: last.text + e.text };
      return { ...s, drafts };
    }

    case 'verdict': {
      // The verdict is read from the Critic's output, so it arrives as the
      // node ends: attach it to the latest Critic row, running or not.
      const idx = lastOf(s.rows, 'critic');
      const drafts = s.drafts.slice();
      if (drafts.length) drafts[drafts.length - 1] = { ...drafts[drafts.length - 1], verdict: e.verdict, critique: e.critique };
      return {
        ...s,
        drafts,
        citationChecks: e.citation_checks,
        rows: updateRow(s.rows, idx, (r) => ({
          ...r,
          verdict: e.verdict,
          critique: e.critique,
          activity: [
            // The lookups, now with their results.
            ...r.activity.filter((a) => a.label !== 'Looked up'),
            ...e.citation_checks.map((c) => ({ label: 'Checked', detail: `${c.arxiv_id}: ${c.status.replace('_', ' ')}`, ...(c.status === 'not_found' || c.status === 'invalid' ? { tone: 'warn' as const } : {}) })),
          ],
        })),
      };
    }

    case 'paused':
      return { ...s, status: 'paused', pending: e.payload };

    case 'done':
      return {
        ...s,
        status: 'done',
        pending: null,
        answer: e.answer,
        summary: { dispatches: e.dispatches, revisions: e.revisions, unsupported: e.unsupported_citations, interventions: e.interventions },
        rows: [...s.rows, { kind: 'marker', id: s.nextId, label: 'Answer delivered', tone: 'good' }],
        nextId: s.nextId + 1,
      };

    case 'stopped':
      return {
        ...s,
        status: 'stopped',
        problem: { title: STOP_REASONS[e.reason] ?? 'The run was stopped', detail: e.detail },
        rows: s.rows.map((r) => (r.kind === 'work' && r.running ? { ...r, running: false } : r)),
      };

    case 'error':
      return {
        ...s,
        status: 'error',
        problem: { title: 'The run failed', detail: `${e.kind}: ${e.detail}` },
        rows: s.rows.map((r) => (r.kind === 'work' && r.running ? { ...r, running: false } : r)),
      };
  }
}

export function runReducer(s: RunState, a: RunAction): RunState {
  switch (a.kind) {
    case 'start':
      return { ...initialRun, status: 'running', question: a.question, threadId: a.threadId };
    case 'resume':
      return {
        ...s,
        status: 'running',
        pending: null,
        problem: null,
        rows: [...s.rows, { kind: 'marker', id: s.nextId, label: `You chose: ${a.decision}` }],
        nextId: s.nextId + 1,
      };
    case 'event':
      return onEvent(s, a.event);
    case 'fail':
      return { ...s, status: s.status === 'running' && s.rows.length === 0 ? 'idle' : 'error', problem: { title: a.title, detail: a.detail } };
    case 'restore':
      return { ...initialRun, status: a.pending ? 'paused' : 'idle', question: a.question, threadId: a.threadId, pending: a.pending };
    case 'load':
      return a.state;
    case 'reset':
      return initialRun;
  }
}

/** Replay a whole event list at once - the demo's resting state. */
export function reduceAll(events: RunEvent[], question: string): RunState {
  return events.reduce((s, event) => runReducer(s, { kind: 'event', event }), runReducer(initialRun, { kind: 'start', question, threadId: null }));
}
