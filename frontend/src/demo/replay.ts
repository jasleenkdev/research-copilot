import type { RunEvent } from '../types';
import recorded from './recorded-run.json';

/**
 * The demo: a run recorded through the real API (scripted models, the real
 * graph and event stream), replayed with pauses so it reads like a live one.
 * Re-record with `python scripts/demo_backend.py record frontend/src/demo/recorded-run.json`.
 */
export const DEMO_EVENTS = recorded as RunEvent[];
export const DEMO_QUESTION = 'How are RAG pipelines evaluated?';

export function delayFor(e: RunEvent): number {
  switch (e.type) {
    case 'token':
      return 9;
    case 'decision':
      return 650;
    case 'verdict':
      return 800;
    case 'tool':
      return e.phase === 'start' ? 550 : 120;
    case 'node':
      return e.depth === 0 && e.phase === 'start' ? 260 : 20;
    default:
      return 60;
  }
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      resolve();
    });
  });

export async function* replay(
  events: RunEvent[] = DEMO_EVENTS,
  { signal, speed = 1 }: { signal?: AbortSignal; speed?: number } = {},
): AsyncGenerator<RunEvent> {
  for (const e of events) {
    if (signal?.aborted) return;
    await sleep(delayFor(e) / speed, signal);
    if (signal?.aborted) return;
    yield e;
  }
}
