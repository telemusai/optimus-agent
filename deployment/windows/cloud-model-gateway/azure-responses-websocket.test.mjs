import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import WebSocket, { WebSocketServer } from "ws";
import { createGatewayServer, createRoutes, estimateAzureReservation, normalizeModelRequest } from "./gateway.mjs";
import { extractLegacyDeployment, validateDeployment } from "./deployment-config.mjs";
import { TEST_DEPLOYMENT } from "./test-deployment.mjs";
const localToken = "isolated-local-token";
const auth = { authorization: `Bearer ${localToken}` };
const request = (extra = {}) => ({ type: "response.create", model: "gpt-6-astra", store: false,
  reasoning: { effort: "xhigh" }, max_output_tokens: 128_000,
  input: [{ role: "user", content: "hello" }], ...extra });
const complete = (id, output = []) => ({ type: "response.completed", response: { id, output } });
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fixture(t, options = {}) {
  const captures = [], reservations = [], logs = [], credentials = [];
  let releases = 0, inFlight = 0, opens = 0;
  const http = createServer();
  const provider = new WebSocketServer({ noServer: true });
  http.on("upgrade", (req, socket, head) => {
    opens++;
    if (options.rejectHandshake?.(opens, req)) {
      socket.end(`HTTP/1.1 ${options.rejectStatus ?? 401} Rejected\r\nRetry-After: 2\r\nContent-Length: 0\r\nConnection: close\r\n\r\n`);
    } else provider.handleUpgrade(req, socket, head, (ws) => provider.emit("connection", ws, req));
  });
  provider.on("connection", (ws, req) => {
    ws.on("message", (data) => {
      const body = JSON.parse(data);
      captures.push({ body, headers: req.headers, ws });
      if (options.respond) options.respond(ws, body, captures.length);
      else ws.send(JSON.stringify(complete(`response-${captures.length}`)));
    });
  });
  http.listen(0, "127.0.0.1"); await once(http, "listening");
  const limiter = {
    async admit(tokens, signal) {
      reservations.push(tokens);
      if (options.admit) await options.admit(signal);
      if (signal.aborted) throw signal.reason;
      inFlight++;
      let released = false;
      return { waitedMs: options.waitedMs ?? 0, waitReasons: options.waitReasons,
        release() { if (!released) { released = true; inFlight--; releases++; } } };
    },
    snapshot() { return { id: "fixture", inFlight }; },
    observeResponse(status, headers) { logs.push({ observedStatus: status, retryAfter: headers.get("retry-after") }); },
  };
  const gateway = createGatewayServer({
    deployment: options.deployment ?? TEST_DEPLOYMENT, localToken,
    log(event, value) { logs.push({ event, ...value }); },
    limiters: new Map([["gpt-6-astra", limiter], ["gpt-5.6-sol", limiter]]),
    async getAzureCredential(resource, refresh, control) {
      assert.ok(control.deadlineUnixMs <= Date.now() + 30_000);
      credentials.push({ resource, refresh, signal: control.signal });
      if (options.credential) return options.credential(resource, refresh, control);
      return { token: `isolated-azure-${refresh}`, expiresAt: Date.now() + 3_600_000 };
    },
    webSocketOptions: {
      route: { ...createRoutes(TEST_DEPLOYMENT).get("/azure-openai/v1/responses"),
        webSocketUrl: `ws://127.0.0.1:${http.address().port}/openai/v1/responses` },
      ...options.webSocketOptions,
    },
  });
  gateway.listen(0, "127.0.0.1"); await once(gateway, "listening");
  const base = `http://127.0.0.1:${gateway.address().port}`;
  const clients = [];
  t.after(async () => {
    for (const ws of clients) ws.terminate();
    for (const ws of provider.clients) ws.terminate();
    await delay(5);
    await Promise.all([new Promise((resolve) => gateway.close(resolve)), new Promise((resolve) => http.close(resolve))]);
    provider.close();
  });
  return { base, captures, reservations, credentials, logs, gateway, releases: () => releases,
    inFlight: () => inFlight, opens: () => opens,
    async connect(path = "/azure-openai/v1/responses", headers = auth) {
      const ws = new WebSocket(base.replace("http:", "ws:") + path, { headers });
      clients.push(ws); ws.on("error", () => {});
      const inbox = []; const waiters = [];
      ws.on("message", (data) => {
        const value = JSON.parse(data); const next = waiters.shift();
        if (next) next(value); else inbox.push(value);
      });
      await once(ws, "open");
      return { ws, next: () => inbox.length ? Promise.resolve(inbox.shift()) : new Promise((resolve) => waiters.push(resolve)),
        send: (body) => ws.send(JSON.stringify(body)) };
    },
  };
}

test("deployment extraction preserves HTTP URLs; configuration fails closed", () => {
  const legacy = `const TENANT_ID = "${TEST_DEPLOYMENT.tenantId}";\nconst SUBSCRIPTION_ID = "${TEST_DEPLOYMENT.subscriptionId}";\nupstreamUrl: "${TEST_DEPLOYMENT.azureOpenAiHttpUrl}",\nupstreamUrl: "${TEST_DEPLOYMENT.azureFoundryHttpUrl}",`;
  assert.deepEqual(extractLegacyDeployment(legacy), { ...TEST_DEPLOYMENT, websocketEnabled: false });
  assert.throws(() => extractLegacyDeployment(legacy + legacy), /anchor/);
  assert.throws(() => validateDeployment({ ...TEST_DEPLOYMENT, secret: "must-not-be-here" }), /configuration/);
  for (const bad of ["ws://fixture.openai.azure.com/openai/v1/responses", "wss://evil.example/openai/v1/responses",
    "wss://fixture.openai.azure.com/openai/v1/responses?key=x", "wss://user:pass@fixture.openai.azure.com/openai/v1/responses"]) {
    assert.throws(() => validateDeployment({ ...TEST_DEPLOYMENT, azureOpenAiWebSocketUrl: bad }));
  }
  const routes = createRoutes(TEST_DEPLOYMENT);
  assert.equal(routes.get("/azure-openai/v1/responses").azureResource, "https://cognitiveservices.azure.com");
  assert.equal(routes.get("/ollama/v1/chat/completions").upstreamUrl, "https://ollama.com/v1/chat/completions");
});

test("authenticated capability, isolated upgrade auth and route allowlist", { timeout: 5000 }, async (t) => {
  const f = await fixture(t);
  assert.equal((await fetch(`${f.base}/health`)).status, 401);
  const health = await (await fetch(`${f.base}/health`, { headers: auth })).json();
  assert.deepEqual(health.responsesWebSocket.models, ["gpt-5.6-sol", "gpt-6-astra"]);
  assert.equal(health.responsesWebSocket.version, 1);
  for (const [path, headers, status] of [
    ["/azure-openai/v1/responses", {}, 401], ["/ollama/v1/chat/completions", auth, 404],
    ["/azure-foundry/v1/chat/completions", auth, 404],
    ["/azure-openai/v1/responses", { ...auth, origin: "https://example.com" }, 400],
    ["/azure-openai/v1/responses", { ...auth, "x-routeworld-operation-id": "test" }, 400],
  ]) await assert.rejects(f.connect(path, headers), new RegExp(String(status)));
  assert.equal(f.opens(), 0);
  const g = await fixture(t, { deployment: { ...TEST_DEPLOYMENT, websocketEnabled: false } });
  const disabled = await (await fetch(`${g.base}/health`, { headers: auth })).json();
  assert.equal(disabled.responsesWebSocket.enabled, false);
  assert.deepEqual(disabled.responsesWebSocket.models, []);
  await assert.rejects(g.connect(), /503/);
  assert.equal(g.opens(), 0);
});

test("sequential and incremental responses preserve normalization and full-context quota", { timeout: 5000 }, async (t) => {
  const output = [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "x".repeat(1200) }] }];
  const f = await fixture(t, { waitedMs: 17, waitReasons: { fifo: 17 },
    respond(ws, body, n) { ws.send(JSON.stringify(complete(`response-${n}`, output))); } });
  const c = await f.connect();
  const first = request({ input: [{ role: "user", content: "A".repeat(2000) }] });
  c.send(first); assert.equal((await c.next()).type, "response.completed");
  const second = request({ previous_response_id: "response-1", input: [{ role: "user", content: "next" }] });
  c.send(second); await c.next();
  const full = structuredClone(second); delete full.type; delete full.previous_response_id;
  full.input = [...first.input, ...output, ...second.input];
  normalizeModelRequest(full, full.model, "responses");
  assert.equal(f.reservations[1], estimateAzureReservation(full, full.model));
  assert.ok(f.reservations[1] > f.reservations[0]);
  assert.equal(f.opens(), 1);
  assert.equal(f.captures[1].body.input.length, 1);
  assert.equal(f.captures[0].body.reasoning.effort, "xhigh");
  assert.equal(f.captures[0].body.max_output_tokens, 128_000);
  assert.equal(f.captures[0].headers.authorization, "Bearer isolated-azure-false");
  assert.equal(f.credentials[0].resource, "https://ai.azure.com");
  assert.equal(f.releases(), 2);
  c.send(request({ input: [{ role: "user", content: "compacted" }] })); await c.next();
  assert.ok(f.reservations[2] < f.reservations[1]);
  assert.equal(f.logs.filter((r) => r.event === "websocket_turn").length, 3);
  for (const record of f.logs.filter((r) => r.event === "websocket_turn")) {
    assert.deepEqual(record.rateWaitReasonsMs, { fifo: 17 });
    assert.equal(record.limiterAtResponse.id, "fixture");
  }
  assert.doesNotMatch(JSON.stringify(f.logs), /isolated-azure-false|A{100}/);
});

test("parallel chats cannot borrow previous IDs and failed references are not replayed", { timeout: 5000 }, async (t) => {
  const f = await fixture(t); const a = await f.connect(); const b = await f.connect();
  a.send(request()); await a.next();
  b.send(request({ previous_response_id: "response-1" }));
  assert.equal((await b.next()).error.code, "previous_response_not_found");
  assert.equal(f.captures.length, 1);
  assert.equal(f.reservations.length, 1);
});

test("model, effort, opaque inputs and stored responses fail before any dispatch", { timeout: 5000 }, async (t) => {
  const f = await fixture(t);
  for (const body of [request({ model: "FW-Kimi-K3" }), request({ reasoning: { effort: "invalid" } }),
    request({ store: true }), request({ n: 2 }), request({ input: [{ type: "input_file", file_id: "file-x" }] }),
    request({ conversation: "conv-opaque" }), request({ input: [{ type: "item_reference", id: "unknown" }] })]) {
    const c = await f.connect(); c.send(body); assert.equal((await c.next()).type, "error");
  }
  assert.equal(f.opens(), 0); assert.equal(f.reservations.length, 0);
});

test("concurrent create is bounded and cancels the owned in-flight provider", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { respond() {} }); const c = await f.connect();
  c.send(request());
  while (f.captures.length === 0) await delay(5);
  c.send(request()); assert.equal((await c.next()).error.code, "response_in_progress");
  await delay(20); assert.equal(f.inFlight(), 0); assert.equal(f.releases(), 1);
  assert.equal(f.captures.length, 1);
});

test("client Escape/close cancels queued admission and never dispatches", { timeout: 5000 }, async (t) => {
  let signal;
  const f = await fixture(t, { admit(s) { signal = s; return new Promise((resolve, reject) => s.addEventListener("abort", () => reject(s.reason), { once: true })); } });
  const c = await f.connect(); c.send(request()); while (!signal) await delay(5);
  c.ws.close(); await delay(20);
  assert.equal(signal.aborted, true); assert.equal(f.opens(), 0);
  const health = await (await fetch(`${f.base}/health`, { headers: auth })).json(); assert.equal(health.activeRequests, 0);
});

test("upstream handshake authorization retries only once before sending", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { rejectHandshake: (n) => n === 1 }); const c = await f.connect();
  c.send(request()); assert.equal((await c.next()).type, "response.completed");
  assert.equal(f.opens(), 2); assert.equal(f.captures.length, 1);
  assert.deepEqual(f.credentials.map((x) => x.refresh), [false, true]);
  assert.equal(f.captures[0].headers.authorization, "Bearer isolated-azure-true");
});

test("partial provider output and disconnect are never replayed", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { respond(ws) { ws.send(JSON.stringify({ type: "response.output_text.delta", delta: "partial" })); setTimeout(() => ws.terminate(), 10); } });
  const c = await f.connect(); c.send(request());
  assert.equal((await c.next()).delta, "partial"); assert.equal((await c.next()).type, "error");
  assert.equal(f.opens(), 1); assert.equal(f.captures.length, 1); assert.equal(f.releases(), 1);
});

test("upstream close diagnostics correlate the failed turn and retain safe code, reason and timing", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { waitedMs: 7, respond(ws, body, n) {
    if (n === 1) ws.send(JSON.stringify(complete("first")));
    else ws.close(1012, "Server restarting");
  } });
  const c = await f.connect(); c.send(request()); await c.next();
  c.send(request());
  const result = await c.next();
  assert.equal(result.error.code, "connection_closed");
  assert.match(result.error.message, /^Azure WebSocket connection closed \(code 1012; reason server restarting;/);
  const details = result.error.upstreamClose;
  assert.equal(details.closeCode, 1012);
  assert.equal(details.closeReason, "server restarting");
  assert.equal(details.closeReasonDisposition, "allowlisted");
  assert.match(details.requestId, /^[0-9a-f-]{36}$/);
  assert.match(details.connectionId, /^[0-9a-f-]{36}$/);
  assert.match(result.error.message, new RegExp(details.requestId));
  assert.ok(details.durationMs >= 0);
  assert.ok(details.connectionDurationMs >= 0);
  assert.equal(details.waitedMs, 7);
  assert.equal(details.firstEventMs, undefined);
  const close = f.logs.filter((r) => r.event === "websocket_upstream_close");
  const turns = f.logs.filter((r) => r.event === "websocket_turn");
  assert.equal(close.length, 1);
  assert.equal(close[0].requestId, details.requestId);
  assert.equal(close[0].connectionId, details.connectionId);
  assert.equal(turns.length, 2);
  assert.notEqual(turns[0].requestId, turns[1].requestId);
  assert.equal(turns[0].connectionId, turns[1].connectionId);
  assert.deepEqual(JSON.parse(JSON.stringify(turns[1].upstreamClose)), details);
  assert.equal(turns[1].terminal, "disconnected");
  assert.equal(f.opens(), 1); assert.equal(f.captures.length, 2); assert.equal(f.releases(), 2);
});

test("untrusted close text never leaks credentials, prompts or log controls", { timeout: 5000 }, async (t) => {
  const reason = "Bearer isolated-azure-false\r\nprivate prompt text https://example.test/?key=secret";
  const f = await fixture(t, { respond(ws) { ws.close(1008, reason); } });
  const c = await f.connect(); c.send(request({ input: "private prompt text" }));
  const result = await c.next();
  assert.equal(result.error.upstreamClose.closeCode, 1008);
  assert.equal(result.error.upstreamClose.closeReasonDisposition, "redacted");
  assert.equal(result.error.upstreamClose.closeReason, undefined);
  assert.doesNotMatch(JSON.stringify([result, f.logs]), /Bearer|isolated-azure|private prompt|example\.test|key=secret/);
  assert.equal(f.opens(), 1); assert.equal(f.captures.length, 1); assert.equal(f.releases(), 1);
});

test("abnormal close after partial output records first event timing without replay", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { respond(ws) {
    ws.send(JSON.stringify({ type: "response.output_text.delta", delta: "private output" }));
    setTimeout(() => ws.terminate(), 10);
  } });
  const c = await f.connect(); c.send(request());
  assert.equal((await c.next()).delta, "private output");
  const result = await c.next();
  const details = result.error.upstreamClose;
  assert.equal(details.closeCode, 1006);
  assert.equal(details.closeReasonDisposition, "empty");
  assert.ok(details.firstEventMs >= 0);
  assert.ok(details.firstTextMs >= details.firstEventMs);
  assert.ok(details.durationMs >= details.firstTextMs);
  assert.doesNotMatch(JSON.stringify(f.logs), /private output/);
  assert.equal(f.opens(), 1); assert.equal(f.captures.length, 1); assert.equal(f.releases(), 1);
});

test("an idle upstream close has connection correlation but no invented turn", { timeout: 5000 }, async (t) => {
  const f = await fixture(t); const c = await f.connect();
  c.send(request()); await c.next();
  f.captures[0].ws.close(1000, "normal closure");
  const result = await c.next();
  assert.equal(result.error.upstreamClose.requestId, undefined);
  assert.equal(result.error.upstreamClose.durationMs, undefined);
  assert.equal(result.error.upstreamClose.closeCode, 1000);
  const turns = f.logs.filter((r) => r.event === "websocket_turn");
  assert.equal(turns.length, 1);
  assert.equal(turns[0].terminal, "completed");
  assert.equal(turns[0].connectionId, result.error.upstreamClose.connectionId);
  assert.equal(f.releases(), 1);
});

test("connection expiry, request timeout and connection count are bounded", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { respond() {}, webSocketOptions: { connectionLifetimeMs: 90, requestTimeoutMs: 500, maxConnections: 1 } });
  const c = await f.connect(); await assert.rejects(f.connect(), /503/);
  c.send(request()); assert.equal((await c.next()).error.code, "websocket_connection_limit_reached");
  assert.equal(f.releases(), 1);
  const g = await fixture(t, { respond() {}, webSocketOptions: { requestTimeoutMs: 60 } });
  const d = await g.connect(); d.send(request()); assert.equal((await d.next()).error.code, "request_timeout");
  assert.equal(g.releases(), 1);
});

test("retained output context and slow receiver have hard memory bounds", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { webSocketOptions: { maxTotalContextBytes: 500 },
    respond(ws) { ws.send(JSON.stringify(complete("large", [{ type: "message", content: "X".repeat(1000) }]))); } });
  const c = await f.connect(); c.send(request()); assert.equal((await c.next()).error.code, "context_limit");
  assert.equal(f.releases(), 1);
  const g = await fixture(t, { webSocketOptions: { maxBufferedBytes: 100 },
    respond(ws) { ws.send(JSON.stringify({ type: "response.output_text.delta", delta: "X".repeat(200) })); } });
  const d = await g.connect(); d.send(request()); assert.equal((await d.next()).error.code, "backpressure_limit");
  assert.equal(g.releases(), 1);
});

test("repeated auth failure and 429 do not leak auth or bypass backoff", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { rejectHandshake: () => true }); const c = await f.connect(); c.send(request());
  const result = await c.next();
  assert.equal(result.status, 503); assert.equal(f.opens(), 2); assert.equal(f.captures.length, 0);
  assert.doesNotMatch(JSON.stringify(result), /isolated-azure|Bearer/);
  const g = await fixture(t, { rejectHandshake: () => true, rejectStatus: 429 });
  const d = await g.connect(); d.send(request()); assert.equal((await d.next()).status, 429);
  assert.equal(g.opens(), 1);
  assert.ok(g.logs.some((row) => row.observedStatus === 429 && row.retryAfter === "2"));
});

test("credential work is cancelled with its own connection; another chat keeps working", { timeout: 5000 }, async (t) => {
  let blockedSignal;
  let calls = 0;
  const f = await fixture(t, { credential(resource, refresh, control) {
    if (++calls !== 1) return { token: "isolated-second", expiresAt: Date.now() + 3_600_000 };
    blockedSignal = control.signal;
    return new Promise((resolve, reject) => control.signal.addEventListener("abort", () => reject(control.signal.reason), { once: true }));
  } });
  const a = await f.connect(); a.send(request()); while (!blockedSignal) await delay(5);
  const b = await f.connect(); b.send(request()); assert.equal((await b.next()).type, "response.completed");
  a.ws.close(); await delay(10); assert.equal(blockedSignal.aborted, true);
  b.send(request()); assert.equal((await b.next()).type, "response.completed");
  assert.equal(f.captures.length, 2);
});

test("HTTP server close drains upgraded sockets and releases admissions", { timeout: 5000 }, async (t) => {
  const f = await fixture(t, { respond() {} }); const c = await f.connect(); c.send(request());
  while (f.captures.length === 0) await delay(5);
  await new Promise((resolve) => f.gateway.close(resolve));
  await delay(10); assert.equal(f.inFlight(), 0); assert.equal(f.releases(), 1);
});
