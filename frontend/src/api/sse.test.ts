import { readEventStream, SseParser } from './sse';

const enc = new TextEncoder();
function streamOf(...chunks: string[]) {
  return new ReadableStream<Uint8Array>({
    start(c) {
      chunks.forEach((s) => c.enqueue(enc.encode(s)));
      c.close();
    },
  });
}

describe('SseParser', () => {
  it('parses whole events', () => {
    const p = new SseParser();
    expect(p.push('data: {"type":"token","agent":"writer","text":"a"}\n\n')).toEqual([{ type: 'token', agent: 'writer', text: 'a' }]);
  });

  it('buffers events split across chunks, anywhere', () => {
    const p = new SseParser();
    expect(p.push('data: {"type":"tok')).toEqual([]);
    expect(p.push('en","agent":"writer","text":"hi"}\n')).toEqual([]);
    expect(p.push('\ndata: {"type":"paused","payload":{}}\n\n')).toHaveLength(2);
  });

  it('accepts CRLF line endings and ignores comments', () => {
    const p = new SseParser();
    expect(p.push(': keep-alive\r\n\r\ndata: {"type":"done"}\r\n\r\n')).toEqual([{ type: 'done' }]);
  });
});

describe('readEventStream', () => {
  it('yields every event from a byte stream, including a final one with no trailing blank line', async () => {
    const out = [];
    for await (const e of readEventStream(streamOf('data: {"type":"thread","thread_id":"t"}\n\nda', 'ta: {"type":"done"}'))) out.push(e);
    expect(out.map((e) => e.type)).toEqual(['thread', 'done']);
  });
});
