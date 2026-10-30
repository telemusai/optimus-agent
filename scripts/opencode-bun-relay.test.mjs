import { describe, expect, test } from 'bun:test';
import { createHandler, MODEL, opencodeID, prepareRequest, rewriteResponse, rewriteStream } from './opencode-bun-relay.mjs';
const native = { type: 'function', function: { name: 'ipython', description: 'Python', parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] } } };
const input = () => ({ model: MODEL, messages: [{ role: 'system', content: 'Native instructions' }, { role: 'user', content: 'Test' }], tools: [native], stream: true });
function request(body = input(), headers = {}) { return new Request('http://127.0.0.1:18997/v1/chat/completions', { method: 'POST', headers: { authorization: 'Bearer fixture-key', ...headers }, body: JSON.stringify(body) }); }
function sse(events) { return events.map(e => `data: ${typeof e === 'string' ? e : JSON.stringify(e)}\r\n\r\n`).join(''); }
async function collect(text, aliases, step = 1) {
  const bytes = new TextEncoder().encode(text); let pos = 0;
  const stream = new ReadableStream({ pull(c) { if (pos === bytes.length) c.close(); else { c.enqueue(bytes.slice(pos, pos + step)); pos = Math.min(pos + step, bytes.length); } } });
  return new Response(rewriteStream(stream, aliases)).text();
}
describe('OpenCode compatibility transport', () => {
  test('uses the timestamped 26-character OpenCode ID format', () => {
    const time = 1793400000000;
    const message = opencodeID('msg', time); const session = opencodeID('ses', time);
    expect(message).toMatch(/^msg_[0-9a-f]{26}$/); expect(session).toMatch(/^ses_[0-9a-f]{26}$/);
    expect(Number(BigInt('0x' + message.slice(4, 16)) / 0x1000n)).toBe(time % (2 ** 36));
    expect(BigInt('0x' + session.slice(4, 16))).toBe(BigInt.asUintN(48, ~(BigInt(time) * 0x1000n + 1n)));
  });
  test('retains native instructions, arguments, history and schema with executable aliases', () => {
    const raw = input(); raw.messages.push({ role: 'assistant', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'ipython', arguments: '{"code":"print(42)"}' } }] }, { role: 'tool', tool_call_id: 'call_1', content: '42' });
    const before = structuredClone(raw); const { body, aliases } = prepareRequest(raw, '/workspace');
    expect(raw).toEqual(before); expect(body.messages.slice(1)).toEqual(raw.messages);
    expect(body.messages[0].content).toContain('<env>'); expect(body.messages[0].content).toContain('/workspace');
    expect(body.tools[0]).toEqual(native); expect([...aliases.keys()]).toEqual(['bash', 'read', 'edit']);
    for (const t of body.tools) expect(t.function.parameters).toEqual(native.function.parameters);
  });
  test('does not override real native tools or silently switch model', () => {
    const raw = input(); raw.tools.push({ ...native, function: { ...native.function, name: 'read' } });
    expect(prepareRequest(raw, '/').aliases.has('read')).toBe(false);
    expect(() => prepareRequest({ ...raw, model: 'deepseek-v4.1-flash' }, '/')).toThrow();
    expect(() => prepareRequest({ ...raw, stream: false }, '/')).toThrow('stream: true');
    const disabled = prepareRequest({ ...raw, tools: [] }, '/');
    expect(disabled.toolsDisabled).toBe(true); expect(disabled.body.tool_choice).toBe('none');
    expect(() => rewriteResponse({ choices: [{ message: { tool_calls: [{ function: { name: 'read' } }] } }] }, disabled.aliases, true)).toThrow('tool-free');
  });
  test('maps nonstreaming alias names without changing arguments or IDs', () => {
    const value = { choices: [{ message: { tool_calls: [{ id: 'a', function: { name: 'read', arguments: '{"code":"print(42)"}' } }] } }] };
    expect(rewriteResponse(value, new Map([['read', 'ipython']])).choices[0].message.tool_calls[0]).toEqual({ id: 'a', function: { name: 'ipython', arguments: '{"code":"print(42)"}' } });
  });
  test('handles split UTF-8/SSE frames, split names, parallel tools, reasoning, usage and DONE', async () => {
    const events = [
      { choices: [{ index: 0, delta: { reasoning_content: '考える' } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'one', function: { name: 're', arguments: '' } }, { index: 1, id: 'two', function: { name: 'edit', arguments: '{"code":' } }] } }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { name: 'ad', arguments: '{"code":"print(42)"}' } }, { index: 1, function: { arguments: '"print(43)"}' } }] } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
      { choices: [], usage: { completion_tokens: 23 } }, '[DONE]',
    ];
    const result = await collect(sse(events), new Map([['read', 'ipython'], ['edit', 'ipython']]));
    expect(result).toContain('考える'); expect(result).toContain('[DONE]'); expect(result).toContain('"completion_tokens":23');
    const chunks = result.split('\n\n').filter(l => l.startsWith('data: {')).map(l => JSON.parse(l.slice(6)));
    const tools = chunks.flatMap(c => c.choices ?? []).flatMap(c => c.delta?.tool_calls ?? []);
    expect(tools.map(t => [t.id, t.function.name, JSON.parse(t.function.arguments).code])).toEqual([['one', 'ipython', 'print(42)'], ['two', 'ipython', 'print(43)']]);
  });
  test('rejects truncated tool streams', async () => {
    await expect(collect(sse([{ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'read' } }] } }] }, '[DONE]']), new Map())).rejects.toThrow('before tool calls finished');
  });
  test('authenticates locally, blocks browser/other hosts/routes/models, and forwards only fixed upstream', async () => {
    const seen = []; const handler = createHandler({ key: 'fixture-key', cwd: '/fixture', fetchImpl: async (url, options) => { seen.push({ url, options }); return Response.json({ choices: [] }); } });
    expect((await handler(request(input(), { authorization: 'Bearer wrong' }))).status).toBe(401);
    expect((await handler(request(input(), { origin: 'https://example.com' }))).status).toBe(403);
    expect((await handler(new Request('http://attacker.invalid/health'))).status).toBe(403);
    expect((await handler(new Request('http://127.0.0.1:18997/v1/chat/completions?target=evil', { method: 'POST', headers: { authorization: 'Bearer fixture-key' }, body: '{}' }))).status).toBe(404);
    expect((await handler(request({ ...input(), model: 'paid' }))).status).toBe(400);
    expect(seen).toHaveLength(0);
    expect((await handler(request())) .status).toBe(200);
    expect(seen[0].url).toBe('https://opencode.ai/inference/openai/v1/chat/completions');
    expect(seen[0].options.redirect).toBe('error'); expect(seen[0].options.headers.Authorization).toBe('Bearer fixture-key');
    expect(seen[0].options.headers['User-Agent']).toContain(`runtime/bun/${Bun.version}`);
  });
  test('passes upstream 429 and Retry-After through without retries or paid fallback', async () => {
    let count = 0; const handler = createHandler({ key: 'fixture-key', cwd: '/', fetchImpl: async () => { count++; return new Response('quota', { status: 429, headers: { 'retry-after': '30' } }); } });
    const response = await handler(request()); expect(response.status).toBe(429); expect(response.headers.get('retry-after')).toBe('30'); expect(await response.text()).toBe('quota'); expect(count).toBe(1);
  });
  test('propagates client cancellation and redacts connection failures', async () => {
    const abort = new AbortController(); let signal;
    const handler = createHandler({ key: 'fixture-key', cwd: '/', fetchImpl: async (_, options) => { signal = options.signal; abort.abort(); throw new Error('secret fixture-key'); } });
    const req = new Request(request(), { signal: abort.signal }); const response = await handler(req);
    expect(signal.aborted).toBe(true); expect(response.status).toBe(502); expect(await response.text()).not.toContain('fixture-key');
  });
});
