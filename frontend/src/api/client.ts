import type { Health, ReviewRequest, RunEvent, RunRequest, ThreadSummary, ThreadView } from '../types';
import { readEventStream } from './sse';

/** A 4xx/5xx from the API, with the server's `detail` as the message. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function failure(res: Response): Promise<ApiError> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join('; ');
  } catch {
    /* not JSON: keep the status line */
  }
  return new ApiError(res.status, detail);
}

export function createClient(baseUrl: string, fetchImpl: typeof fetch = (...a) => fetch(...a)) {
  const url = (path: string) => `${baseUrl.replace(/\/+$/, '')}${path}`;

  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetchImpl(url(path), init);
    if (!res.ok) throw await failure(res);
    return (await res.json()) as T;
  }

  async function* stream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<RunEvent> {
    const res = await fetchImpl(url(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw await failure(res);
    if (!res.body) throw new ApiError(0, 'The server sent no stream.');
    yield* readEventStream(res.body);
  }

  return {
    health: () => json<Health>('/api/health'),
    threads: () => json<ThreadSummary[]>('/api/threads'),
    thread: (id: string) => json<ThreadView>(`/api/threads/${encodeURIComponent(id)}`),
    deleteThread: async (id: string) => {
      const res = await fetchImpl(url(`/api/threads/${encodeURIComponent(id)}`), { method: 'DELETE' });
      if (!res.ok) throw await failure(res);
    },
    run: (req: RunRequest, signal?: AbortSignal) => stream('/api/runs', req, signal),
    review: (threadId: string, req: ReviewRequest, signal?: AbortSignal) =>
      stream(`/api/threads/${encodeURIComponent(threadId)}/review`, req, signal),
  };
}

export type Client = ReturnType<typeof createClient>;
