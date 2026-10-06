import { ApiError, createClient } from './client';

const sse = (...events: object[]) =>
  new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });

it('POSTs a run and yields its events', async () => {
  const fetchImpl = vi.fn(async () => sse({ type: 'thread', thread_id: 't', is_new: true, policy: {} }, { type: 'done', answer: 'A' }));
  const client = createClient('https://api.example/', fetchImpl as unknown as typeof fetch);
  const out = [];
  for await (const e of client.run({ question: 'q', mode: 'live-search', routing: 'supervisor', critic: true, approve: false, plan: false, max_revisions: 2 })) out.push(e.type);
  expect(out).toEqual(['thread', 'done']);
  const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe('https://api.example/api/runs');
  expect(init.method).toBe('POST');
  expect(JSON.parse(init.body as string).question).toBe('q');
});

it('turns a refusal into an ApiError carrying the server detail', async () => {
  const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ detail: 'thread t has a draft awaiting approval' }), { status: 409 }));
  const client = createClient('', fetchImpl as unknown as typeof fetch);
  const it = client.review('t', { decision: 'approve' });
  await expect(it.next()).rejects.toEqual(new ApiError(409, 'thread t has a draft awaiting approval'));
});

it('joins FastAPI validation errors', async () => {
  const fetchImpl = vi.fn(async () => new Response(JSON.stringify({ detail: [{ msg: 'too short' }, { msg: 'bad mode' }] }), { status: 422 }));
  await expect(createClient('', fetchImpl as unknown as typeof fetch).health()).rejects.toThrow('too short; bad mode');
});
