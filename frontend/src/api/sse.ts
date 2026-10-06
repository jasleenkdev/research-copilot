import type { RunEvent } from '../types';

/**
 * Incremental parser for the API's `text/event-stream` body.
 *
 * The runs are POSTs (the request carries the question and policy), so the
 * browser's EventSource, which only does GET, can't be used. This reads the
 * fetch body instead. Chunks split anywhere - mid-line, mid-event - so the
 * parser buffers until it sees the blank line that ends an event.
 */
export class SseParser {
  private buffer = '';

  push(chunk: string): RunEvent[] {
    this.buffer += chunk.replace(/\r\n/g, '\n');
    const events: RunEvent[] = [];
    let end: number;
    while ((end = this.buffer.indexOf('\n\n')) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
        .join('\n');
      if (data) events.push(JSON.parse(data) as RunEvent);
    }
    return events;
  }
}

export async function* readEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<RunEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const event of parser.push(decoder.decode(value, { stream: true }))) yield event;
    }
    for (const event of parser.push(decoder.decode() + '\n\n')) yield event;
  } finally {
    reader.releaseLock();
  }
}
