#!/usr/bin/env bun
// Optional loopback transport; Optimus remains responsible for tool execution.
import { readFileSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { join } from 'node:path';
import { randomBytes, timingSafeEqual } from 'node:crypto';

export const MODEL = 'deepseek-v4.1-flash-free';
const UPSTREAM = 'https://opencode.ai/inference/openai/v1/chat/completions';
const MAX_BYTES = 16 * 1024 * 1024;
const ALIASES = ['bash', 'read', 'edit'];

export function opencodeID(prefix, timestamp = Date.now()) {
  const time = BigInt(timestamp) * 0x1000n + 1n;
  const encoded = BigInt.asUintN(48, prefix === 'ses' ? ~time : time).toString(16).padStart(12, '0');
  return `${prefix}_${encoded}${randomBytes(7).toString('hex')}`;
}

export function prepareRequest(input, cwd) {
  if (!input || input.model !== MODEL || !Array.isArray(input.messages)) {
    throw new Error('This relay accepts only deepseek-v4.1-flash-free chat requests.');
  }
  if (input.stream !== true) throw new Error('This relay requires stream: true, including tool-free completions.');
  const body = structuredClone(input);
  const aliases = new Map();
  const tools = body.tools ?? [];
  if (!Array.isArray(tools)) throw new Error('Invalid tools.');
  const toolsDisabled = tools.length === 0;
  if (toolsDisabled) body.tool_choice = 'none';
  const python = tools.find(t => t.type === 'function' && t.function?.name === 'ipython') ?? (toolsDisabled ? { type: 'function', function: { name: 'ipython', description: 'Disabled for this completion. No tool may be called.', parameters: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] } } } : undefined);
  const names = new Set(tools.map(t => t.function?.name));
  for (const alias of ALIASES) {
    if (names.has(alias)) continue;
    if (!python) throw new Error('OpenCode compatibility requires the native ipython tool or bash/read/edit tools.');
    tools.push({ ...structuredClone(python), function: {
      ...structuredClone(python.function), name: alias,
      description: `Optimus compatibility alias for ipython. Execute Python code with the exact ipython schema. ${python.function.description ?? ''}`,
    } });
    aliases.set(alias, 'ipython');
  }
  body.tools = tools;
  // The environment block is part of the client request format. Keep real cwd/platform.
  const identity = `You are opencode, an interactive CLI tool that helps users with software engineering tasks.\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: ${JSON.stringify(cwd)}\n  Platform: ${platform()}\n  Today's date: ${new Date().toISOString().slice(0, 10)}\n</env>`;
  const instructions = aliases.size
    ? `\nThis is the Optimus transport adapter. Optimus's instructions below govern this session. ${[...aliases.keys()].join(', ')} are transport aliases for ipython: each executes Python code in the same native persistent kernel. Use their supplied JSON schema, not shell/file arguments inferred from their names. Prefer ipython. All tool execution and approvals remain in Optimus.`
    : '\nThis is the Optimus transport adapter. Optimus executes the tools using the supplied schemas and instructions below.';
  body.messages.unshift({ role: 'system', content: identity + instructions });
  return { body, aliases, toolsDisabled };
}

export function rewriteResponse(value, aliases, toolsDisabled = false) {
  for (const choice of value.choices ?? []) {
    for (const tool of choice.message?.tool_calls ?? []) {
      if (toolsDisabled) throw new Error('Upstream returned a tool call for a tool-free completion.');
      if (aliases.has(tool.function?.name)) tool.function.name = aliases.get(tool.function.name);
    }
  }
  return value;
}

// Buffer tool deltas until their finish event so split names and parallel calls map exactly.
export function rewriteStream(upstream, aliases, toolsDisabled = false) {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let pending = '';
  let bytes = 0;
  const calls = new Map();
  function frame(raw) {
    const data = raw.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
    if (!data) return raw + '\n\n';
    if (data === '[DONE]') {
      if (calls.size) throw new Error('Upstream ended before tool calls finished.');
      return raw + '\n\n';
    }
    let event;
    try { event = JSON.parse(data); } catch { return raw + '\n\n'; }
    for (const choice of event.choices ?? []) {
      const deltas = choice.delta?.tool_calls;
      if (deltas?.length) {
        if (toolsDisabled) throw new Error('Upstream returned a tool call for a tool-free completion.');
        let group = calls.get(choice.index ?? 0);
        if (!group) { group = new Map(); calls.set(choice.index ?? 0, group); }
        for (const delta of deltas) {
          const index = delta.index ?? 0;
          let call = group.get(index);
          if (!call) { call = { index, id: '', type: 'function', function: { name: '', arguments: '' } }; group.set(index, call); }
          if (delta.id) call.id += delta.id;
          if (delta.type) call.type = delta.type;
          if (delta.function?.name) call.function.name += delta.function.name;
          if (delta.function?.arguments) call.function.arguments += delta.function.arguments;
          bytes += JSON.stringify(delta).length;
          if (bytes > MAX_BYTES) throw new Error('Upstream tool response exceeded relay limit.');
        }
        delete choice.delta.tool_calls;
      }
      if (choice.finish_reason && calls.has(choice.index ?? 0)) {
        const tools = [...calls.get(choice.index ?? 0).values()].sort((a, b) => a.index - b.index);
        for (const call of tools) if (aliases.has(call.function.name)) call.function.name = aliases.get(call.function.name);
        choice.delta ??= {};
        choice.delta.tool_calls = tools;
        calls.delete(choice.index ?? 0);
      }
    }
    return `data: ${JSON.stringify(event)}\n\n`;
  }
  return new ReadableStream({
    async pull(controller) {
      try {
        while (true) {
          const match = /\r?\n\r?\n/.exec(pending);
          if (match) {
            const raw = pending.slice(0, match.index);
            pending = pending.slice(match.index + match[0].length);
            controller.enqueue(encoder.encode(frame(raw)));
            return;
          }
          const { value, done } = await reader.read();
          if (done) {
            pending += decoder.decode();
            if (pending.trim()) controller.enqueue(encoder.encode(frame(pending)));
            if (calls.size) throw new Error('Upstream closed during a tool call.');
            controller.close();
            reader.releaseLock();
            return;
          }
          pending += decoder.decode(value, { stream: true });
          if (pending.length > MAX_BYTES) throw new Error('Upstream event exceeded relay limit.');
        }
      } catch (error) { await reader.cancel().catch(() => {}); controller.error(error); }
    },
    cancel(reason) { return reader.cancel(reason); },
  });
}

function sameSecret(actual, expected) {
  const a = Buffer.from(actual ?? '');
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

export function createHandler({ key, cwd, orgID, fetchImpl = fetch, timeoutMs = 300000 }) {
  if (!key || /^(PASTE_|YOUR_)/.test(key)) throw new Error('Valid OpenCode API key required.');
  const sessions = new Map();
  return async function handle(req) {
    const url = new URL(req.url);
    if (req.headers.has('origin')) return Response.json({ error: 'Browser requests are not supported.' }, { status: 403 });
    if (url.hostname !== '127.0.0.1') return Response.json({ error: 'Loopback host required.' }, { status: 403 });
    if (req.method === 'GET' && url.pathname === '/health') return Response.json({ ok: true, transport: 'bun', model: MODEL });
    if (!sameSecret(req.headers.get('authorization'), `Bearer ${key}`)) return Response.json({ error: 'Unauthorized' }, { status: 401 });
    if (req.method !== 'POST' || url.pathname !== '/v1/chat/completions' || url.search) return Response.json({ error: 'Not found' }, { status: 404 });
    let prepared;
    try {
      const text = await req.text();
      if (Buffer.byteLength(text) > MAX_BYTES) return Response.json({ error: 'Request too large' }, { status: 413 });
      prepared = prepareRequest(JSON.parse(text), cwd);
    } catch (error) { return Response.json({ error: { message: error instanceof SyntaxError ? 'Invalid JSON' : error.message } }, { status: 400 }); }
    const incomingSession = req.headers.get('x-opencode-session') || req.headers.get('x-session-id') || randomBytes(16).toString('hex');
    let session = sessions.get(incomingSession);
    if (!session) {
      session = opencodeID('ses');
      if (sessions.size >= 4096) sessions.delete(sessions.keys().next().value);
      sessions.set(incomingSession, session);
    }
    const headers = {
      Authorization: `Bearer ${key}`, 'Content-Type': 'application/json', Accept: '*/*',
      'User-Agent': `opencode/1.19.5 ai-sdk/provider-utils/4.0.23 runtime/bun/${Bun.version}`,
      'x-opencode-client': 'cli', 'x-opencode-project': 'global',
      ...(orgID ? { 'x-opencode-org-id': orgID } : {}),
      'x-opencode-session': session, 'x-opencode-request': opencodeID('msg'),
    };
    try {
      const response = await fetchImpl(UPSTREAM, {
        method: 'POST', headers, body: JSON.stringify(prepared.body), redirect: 'error',
        signal: AbortSignal.any([req.signal, AbortSignal.timeout(timeoutMs)]),
      });
      const outHeaders = new Headers();
      for (const name of ['content-type', 'retry-after', 'x-request-id']) if (response.headers.has(name)) outHeaders.set(name, response.headers.get(name));
      outHeaders.set('cache-control', 'no-store');
      if (!response.ok) return new Response(response.body, { status: response.status, headers: outHeaders });
      if (response.headers.get('content-type')?.includes('text/event-stream')) {
        return new Response(rewriteStream(response.body, prepared.aliases, prepared.toolsDisabled), { status: response.status, headers: outHeaders });
      }
      return Response.json(rewriteResponse(await response.json(), prepared.aliases, prepared.toolsDisabled), { status: response.status, headers: outHeaders });
    } catch {
      return Response.json({ error: { message: req.signal.aborted ? 'Request cancelled' : 'OpenCode upstream connection failed' } }, { status: 502 });
    }
  };
}

if (import.meta.main) {
  const authPath = process.env.OPENCODE_AUTH_FILE || join(homedir(), '.local/share/opencode/auth.json');
  const key = process.env.OPENCODE_API_KEY || JSON.parse(readFileSync(authPath, 'utf8')).opencode?.key;
  const configResponse = await fetch('https://opencode.ai/console/api/v2/config', { headers: { Authorization: `Bearer ${key}` }, redirect: 'error', signal: AbortSignal.timeout(20000) });
  if (!configResponse.ok) throw new Error(`OpenCode configuration HTTP ${configResponse.status}`);
  const orgID = (await configResponse.json()).providers?.opencode?.headers?.['x-opencode-org-id'];
  if (typeof orgID !== 'string' || !orgID) throw new Error('OpenCode organization missing from configuration.');
  const port = Number(process.env.OPENCODE_RELAY_PORT || 18997);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid relay port.');
  const server = Bun.serve({ hostname: '127.0.0.1', port, maxRequestBodySize: MAX_BYTES, idleTimeout: 0,
    fetch: createHandler({ key, orgID, cwd: process.cwd() }),
  });
  console.log(`OpenCode Bun relay listening on http://127.0.0.1:${server.port} (${MODEL})`);
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { server.stop(true); process.exit(0); });
}
