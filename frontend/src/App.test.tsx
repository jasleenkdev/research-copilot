import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import type { Client } from './api/client';
import type { Health, RunEvent, ThreadView } from './types';

const HEALTH: Health = {
  status: 'ok',
  provider: 'groq',
  model: 'openai/gpt-oss-120b',
  api_key_set: true,
  checkpointer: 'AsyncSqliteSaver',
  defaults: { max_revisions: 2, critic: false, plan: false },
};

async function* from(events: RunEvent[]) {
  for (const e of events) yield e;
}

const run = (answer: string, extra: RunEvent[] = []): RunEvent[] => [
  { type: 'thread', thread_id: 't1', is_new: true, policy: {} },
  { type: 'decision', step: 1, proposed: 'researcher', routed_to: 'researcher', override: '', rationale: 'Needs evidence first.', brief: '' },
  { type: 'node', phase: 'start', agent: 'researcher', node: 'researcher', depth: 0 },
  { type: 'tool', phase: 'start', agent: 'researcher', name: 'search_arxiv', arg: 'rag evaluation' },
  { type: 'node', phase: 'end', agent: 'researcher', node: 'researcher', depth: 0 },
  { type: 'node', phase: 'start', agent: 'writer', node: 'writer', depth: 0 },
  { type: 'token', agent: 'writer', text: answer },
  { type: 'node', phase: 'end', agent: 'writer', node: 'writer', depth: 0 },
  ...extra,
];

function fakeClient(overrides: Partial<Client> = {}): Client {
  return {
    health: vi.fn(async () => HEALTH),
    threads: vi.fn(async () => []),
    thread: vi.fn(async () => ({}) as ThreadView),
    deleteThread: vi.fn(async () => {}),
    run: vi.fn(() => from(run('RAGAS scores faithfulness [2309.15217].', [{ type: 'done', answer: 'RAGAS scores faithfulness [2309.15217].', dispatches: { researcher: 1, writer: 1 }, revisions: 0, verdict: '', unsupported_citations: [], interventions: 0 }]))),
    review: vi.fn(() => from([])),
    ...overrides,
  };
}

describe('live mode', () => {
  it('shows the connected model, runs a question, and draws the hand-offs and the answer', async () => {
    const client = fakeClient();
    render(<App clientFactory={() => client} />);
    expect(await screen.findByRole('button', { name: 'groq, openai/gpt-oss-120b' })).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText('Research question'), 'How are RAG pipelines evaluated?');
    await userEvent.click(screen.getByRole('button', { name: 'Run' }));

    expect(await screen.findByText('Answer delivered')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'How are RAG pipelines evaluated?' })).toBeInTheDocument();
    expect(screen.getByText('Needs evidence first.')).toBeInTheDocument();
    expect(screen.getByText('rag evaluation')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '2309.15217' })).toHaveAttribute('href', 'https://arxiv.org/abs/2309.15217');

    const body = (client.run as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(body).toMatchObject({ question: 'How are RAG pipelines evaluated?', critic: true, approve: false, thread_id: null });
  });

  it('sends a follow-up on the same thread and keeps the earlier turn', async () => {
    const client = fakeClient();
    render(<App clientFactory={() => client} />);
    await screen.findByRole('button', { name: /groq/ });
    await userEvent.type(screen.getByLabelText('Research question'), 'First?');
    await userEvent.click(screen.getByRole('button', { name: 'Run' }));
    await screen.findByText('Answer delivered');

    await userEvent.type(screen.getByLabelText('Ask a follow-up in this conversation'), 'Second?');
    await userEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => expect(client.run).toHaveBeenCalledTimes(2));
    expect((client.run as ReturnType<typeof vi.fn>).mock.calls[1][0].thread_id).toBe('t1');
    expect(await screen.findByText(/Earlier in this conversation \(1 question\)/)).toBeInTheDocument();
  });

  it('parks at the review gate and resumes with the decision', async () => {
    const client = fakeClient({
      run: vi.fn(() => from(run('Draft text.', [{ type: 'paused', payload: { question: 'Q', mode: 'live-search', draft: 'Draft text.', critique: '' } }]))),
      review: vi.fn(() => from([{ type: 'thread', thread_id: 't1', is_new: false, policy: {} }, { type: 'done', answer: 'Draft text.', dispatches: {}, revisions: 0, verdict: '', unsupported_citations: [], interventions: 0 }])),
    });
    render(<App clientFactory={() => client} />);
    await screen.findByRole('button', { name: /groq/ });
    await userEvent.click(screen.getByLabelText('I approve before it’s final'));
    await userEvent.type(screen.getByLabelText('Research question'), 'Q');
    await userEvent.click(screen.getByRole('button', { name: 'Run' }));

    const review = await screen.findByRole('region', { name: 'Review the draft' });
    expect((client.run as ReturnType<typeof vi.fn>).mock.calls[0][0].approve).toBe(true);

    await userEvent.click(within(review).getByRole('button', { name: 'Edit it myself' }));
    const box = screen.getByLabelText('Your version of the answer');
    await userEvent.clear(box);
    await userEvent.type(box, 'My version');
    await userEvent.click(screen.getByRole('button', { name: 'Use my version' }));

    expect(client.review).toHaveBeenCalledWith('t1', { decision: 'edit', text: 'My version', note: '' }, expect.any(AbortSignal));
    expect(await screen.findByText('You chose: use my edit')).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Review the draft' })).not.toBeInTheDocument();
  });

  it('shows a refusal from the server in plain words', async () => {
    const { ApiError } = await import('./api/client');
    const client = fakeClient({
      run: vi.fn(async function* () {
        throw new ApiError(400, 'the knowledge base is empty: run `research-copilot ingest <path>` first');
      }),
    });
    render(<App clientFactory={() => client} />);
    await screen.findByRole('button', { name: /groq/ });
    await userEvent.type(screen.getByLabelText('Research question'), 'Q');
    await userEvent.click(screen.getByRole('button', { name: 'Run' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('The server refused the request. the knowledge base is empty');
  });

  it('opens a saved conversation that is waiting for review', async () => {
    const client = fakeClient({
      threads: vi.fn(async () => [{ thread_id: 'old', question: 'Older question', turns: 2, updated_at: '2026-10-06T10:00:00Z' }]),
      thread: vi.fn(async () => ({
        thread_id: 'old',
        question: 'Second',
        mode: 'live-search',
        policy: {},
        messages: [
          { role: 'user' as const, content: 'First' },
          { role: 'assistant' as const, content: 'First answer' },
          { role: 'user' as const, content: 'Second' },
        ],
        pending_review: { question: 'Second', mode: 'live-search', draft: 'Parked draft', critique: '' },
      })),
    });
    render(<App clientFactory={() => client} />);
    await screen.findByRole('button', { name: /groq/ });
    await userEvent.click(screen.getByRole('button', { name: 'Conversations' }));
    await userEvent.click(await screen.findByRole('button', { name: /^Older question/ }));

    expect(await screen.findByRole('region', { name: 'Review the draft' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Second' })).toBeInTheDocument();
    expect(screen.getByText(/Earlier in this conversation \(1 question\)/)).toBeInTheDocument();
  });

  it('asks before deleting a conversation', async () => {
    const client = fakeClient({
      threads: vi.fn(async () => [{ thread_id: 'x', question: 'Doomed', turns: 1, updated_at: null }]),
    });
    render(<App clientFactory={() => client} />);
    await screen.findByRole('button', { name: /groq/ });
    await userEvent.click(screen.getByRole('button', { name: 'Conversations' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Delete conversation: Doomed' }));
    expect(client.deleteThread).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(client.deleteThread).toHaveBeenCalledWith('x');
    await waitFor(() => expect(screen.queryByText('Doomed')).not.toBeInTheDocument());
  });
});

describe('demo mode', () => {
  it('falls back to the recorded run when the API is unreachable, showing it finished at rest', async () => {
    const client = fakeClient({ health: vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))) });
    render(<App clientFactory={() => client} />);
    expect(await screen.findByRole('button', { name: 'Demo replay' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'How are RAG pipelines evaluated?' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Draft 1 (rejected)' })).toBeInTheDocument();
    expect(screen.getByText('Answer delivered')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replay run' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Conversations' })).not.toBeInTheDocument();
  });

  it('shows the rejected draft with the Critic’s reason', async () => {
    const client = fakeClient({ health: vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))) });
    render(<App clientFactory={() => client} />);
    await userEvent.click(await screen.findByRole('tab', { name: 'Draft 1 (rejected)' }));
    expect(screen.getAllByText(/TruLens/).length).toBeGreaterThan(0);
    expect(screen.getByText(/the Writer added it/, { selector: '.critique-note' })).toBeInTheDocument();
  });
});
