import { randomUUID, timingSafeEqual } from "node:crypto";
import WebSocket, { WebSocketServer } from "ws";

export const AZURE_RESPONSES_WS_PATH = "/azure-openai/v1/responses";
export const AZURE_RESPONSES_WS_RESOURCE = "https://ai.azure.com";

function failure(message, code = "invalid_request_error", status = 400) {
  return Object.assign(new Error(message), { code, statusCode: status });
}

function rejectUpgrade(socket, status) {
  socket.end(`HTTP/1.1 ${status} Rejected\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);
}

function authenticates(header, token) {
  const provided = Buffer.from(typeof header === "string" ? header : "");
  const expected = Buffer.from(`Bearer ${token}`);
  return token.length > 0 && provided.length === expected.length && timingSafeEqual(provided, expected);
}

// A close reason is remote, untrusted text and may echo credentials or input.
// Keep only exact, known transport descriptions; never log arbitrary free text.
const SAFE_CLOSE_REASONS = new Set([
  "normal closure", "going away", "protocol error", "unsupported data",
  "invalid payload", "policy violation", "message too big", "internal server error",
  "service restart", "try again later", "server restarting", "server shutting down",
  "idle timeout", "request timeout", "connection timeout", "rate limit exceeded",
]);

function closeDetails(code, reason) {
  const text = (Buffer.isBuffer(reason) ? reason.toString("utf8") : String(reason ?? ""))
    .replace(/[\x00-\x1f\x7f-\x9f]/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
  const allowed = SAFE_CLOSE_REASONS.has(text);
  return {
    closeCode: Number.isInteger(code) && code >= 1000 && code <= 4999 ? code : undefined,
    closeReason: allowed ? text : undefined,
    closeReasonDisposition: text ? (allowed ? "allowlisted" : "redacted") : "empty",
  };
}

function inputItems(input) {
  if (typeof input === "string") return [{ role: "user", content: input }];
  if (!Array.isArray(input)) throw failure("Responses input must be an array or string");
  return input;
}

// This is a server-side transport adapter, not a generic WebSocket proxy. Only
// the configured Azure route and its gateway-owned credentials can be reached.
export function installAzureResponsesWebSockets(server, options) {
  const {
    localToken, route, getAzureCredential, limiters, normalizeModelRequest, enabled = false,
    inspectAzureMultimodalInput, estimateAzureReservation,
    onActiveChange = () => {}, log = () => {},
    maxConnections = 32, maxPayloadBytes = 64 * 1024 * 1024,
    maxTotalContextBytes = 128 * 1024 * 1024, maxBufferedBytes = 4 * 1024 * 1024,
    requestTimeoutMs = 20 * 60_000, connectionLifetimeMs = 60 * 60_000 - 5_000,
    handshakeTimeoutMs = 15_000,
  } = options;
  const wss = new WebSocketServer({ noServer: true, maxPayload: maxPayloadBytes,
    perMessageDeflate: false, closeTimeout: 1_000 });
  const connections = new Set();
  let retainedBytes = 0;

  const upgrade = (request, socket, head) => {
    if (!authenticates(request.headers.authorization, localToken)) return rejectUpgrade(socket, 401);
    if (request.method !== "GET" || request.url !== AZURE_RESPONSES_WS_PATH) return rejectUpgrade(socket, 404);
    if (!enabled) return rejectUpgrade(socket, 503);
    // Do not inherit browser origins or App V0.1's differently-scoped deadline
    // protocol into the Azure Sol/Astra transport.
    if (request.headers.origin || request.headers["sec-websocket-protocol"] ||
        Object.keys(request.headers).some((name) => name.startsWith("x-routeworld-"))) {
      return rejectUpgrade(socket, 400);
    }
    if (connections.size >= maxConnections) return rejectUpgrade(socket, 503);
    wss.handleUpgrade(request, socket, head, (client) => attach(client));
  };

  function attach(client) {
    const connectionId = randomUUID();
    let upstreamOpenedAt;
    let upstream;
    let stopped = false;
    let inFlight;
    let model;
    let previous;
    let contextBytes = 0;
    let lifetimeTimer;
    const lifetime = new AbortController();
    connections.add(client);

    function accountContext(bytes) {
      if (bytes > maxPayloadBytes || retainedBytes - contextBytes + bytes > maxTotalContextBytes) {
        throw failure("WebSocket context memory bound reached; reopen with full context", "context_limit", 413);
      }
      retainedBytes += bytes - contextBytes;
      contextBytes = bytes;
    }

    function replacePrevious(value) {
      accountContext(value ? Buffer.byteLength(JSON.stringify(value)) : 0);
      previous = value;
    }

    function finishTurn() {
      const turn = inFlight;
      if (!turn) return;
      inFlight = undefined;
      clearTimeout(turn.timer);
      turn.controller.abort();
      turn.admission?.release();
      onActiveChange(-1);
      log("websocket_turn", {
        route: route.name, model, connectionId, requestId: turn.requestId,
        reservation: turn.reservation,
        upstreamClose: turn.upstreamClose,
        waitedMs: turn.admission?.waitedMs, durationMs: Date.now() - turn.startedAt,
        rateWaitReasonsMs: turn.admission?.waitReasons,
        limiterAtResponse: turn.limiter?.snapshot?.(),
        firstEventMs: turn.firstEventMs, firstTextMs: turn.firstTextMs,
        firstThinkingMs: turn.firstThinkingMs, firstToolMs: turn.firstToolMs,
        terminal: turn.terminal ?? "disconnected",
      });
    }

    function shutdown() {
      if (stopped) return;
      stopped = true;
      clearTimeout(lifetimeTimer);
      lifetime.abort();
      finishTurn();
      replacePrevious(undefined);
      upstream?.terminate();
      connections.delete(client);
      client.close(1000);
    }

    function fail(error) {
      if (stopped) return;
      const known = error?.statusCode;
      const body = { type: "error", status: known || 502,
        error: { type: "gateway_error", code: known ? error.code ?? "gateway_error" : "gateway_error",
          message: known ? error.message : "Azure WebSocket transport failed; no request was replayed" } };
      if (error?.upstreamClose) body.error.upstreamClose = error.upstreamClose;
      if (client.readyState === WebSocket.OPEN && client.bufferedAmount < maxBufferedBytes) {
        client.send(JSON.stringify(body));
      }
      shutdown();
    }

    function forward(data) {
      if (client.readyState !== WebSocket.OPEN || client.bufferedAmount + data.length > maxBufferedBytes) {
        throw failure("WebSocket receiver is too slow", "backpressure_limit", 503);
      }
      client.send(data, { binary: false }, (error) => { if (error) shutdown(); });
    }

    function observe(data, binary) {
      if (stopped) return;
      try {
        if (binary) throw failure("Unexpected binary Azure event", "upstream_protocol_error", 502);
        const event = JSON.parse(data.toString("utf8"));
        const turn = inFlight;
        if (!turn) throw failure("Unexpected Azure event outside a response", "upstream_protocol_error", 502);
        const elapsed = Date.now() - turn.startedAt;
        turn.firstEventMs ??= elapsed;
        if (event.type === "response.output_text.delta") turn.firstTextMs ??= elapsed;
        if (String(event.type).includes("reasoning")) turn.firstThinkingMs ??= elapsed;
        if (String(event.type).includes("function_call") || event.item?.type === "function_call") turn.firstToolMs ??= elapsed;
        if (event.type === "response.completed") {
          if (typeof event.response?.id !== "string" || !Array.isArray(event.response.output)) {
            throw failure("Incomplete Azure response state", "upstream_protocol_error", 502);
          }
          replacePrevious({ id: event.response.id, model, input: turn.fullInput,
            output: event.response.output });
          turn.terminal = "completed";
          finishTurn();
        } else if (event.type === "error" || ["response.failed", "response.incomplete", "response.cancelled"].includes(event.type)) {
          turn.limiter?.observeResponse?.(Number(event.status ?? event.error?.status), new Headers());
          replacePrevious(undefined);
          turn.terminal = event.type;
          finishTurn();
        }
        forward(data);
      } catch (error) { fail(error); }
    }

    function dial(credential, signal) {
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(route.webSocketUrl, { headers: {
          authorization: `Bearer ${credential.token}`, "user-agent": "optimus-cloud-model-gateway/1",
        }, followRedirects: false, perMessageDeflate: false, maxPayload: maxPayloadBytes,
        handshakeTimeout: handshakeTimeoutMs, closeTimeout: 1_000 });
        let settled = false;
        const abort = () => { ws.terminate(); settle(failure("WebSocket request cancelled", "cancelled", 499)); };
        const settle = (error) => {
          if (settled) return;
          settled = true;
          signal.removeEventListener("abort", abort);
          if (error) reject(error); else resolve(ws);
        };
        signal.addEventListener("abort", abort, { once: true });
        ws.on("error", () => {
          if (!settled) settle(failure("Azure WebSocket handshake failed", "upstream_handshake_failed", 502));
          else if (upstream === ws && !stopped) fail(new Error("upstream socket failed"));
        });
        ws.once("unexpected-response", (request, response) => {
          const status = response.statusCode;
          const retryHeaders = new Headers();
          for (const name of ["retry-after", "retry-after-ms", "x-ms-retry-after-ms"]) {
            if (typeof response.headers[name] === "string") retryHeaders.set(name, response.headers[name]);
          }
          response.destroy();
          request.destroy();
          const error = failure("Azure WebSocket handshake rejected", "upstream_handshake_failed", 502);
          error.upstreamStatus = status;
          error.retryHeaders = retryHeaders;
          settle(error);
        });
        ws.once("open", () => {
          if (signal.aborted) return abort();
          upstreamOpenedAt = Date.now();
          ws.on("message", observe);
          ws.on("close", (code, reason) => {
            if (upstream !== ws || stopped) return;
            const turn = inFlight;
            const details = { ...closeDetails(code, reason), connectionId,
              requestId: turn?.requestId,
              connectionDurationMs: Math.max(0, Date.now() - upstreamOpenedAt),
              durationMs: turn ? Math.max(0, Date.now() - turn.startedAt) : undefined,
              waitedMs: turn?.admission?.waitedMs,
              firstEventMs: turn?.firstEventMs, firstTextMs: turn?.firstTextMs,
              firstThinkingMs: turn?.firstThinkingMs, firstToolMs: turn?.firstToolMs,
            };
            if (turn) turn.upstreamClose = details;
            log("websocket_upstream_close", { route: route.name, model, ...details });
            const reasonLabel = details.closeReason ?? details.closeReasonDisposition;
            const message = `Azure WebSocket connection closed (code ${details.closeCode ?? "unknown"}; reason ${reasonLabel}; request ${details.requestId ?? "none"}; elapsed ${details.durationMs ?? details.connectionDurationMs} ms)`;
            fail(Object.assign(failure(message, "connection_closed", 502), { upstreamClose: details }));
          });
          settle();
        });
        if (signal.aborted) abort();
      });
    }

    async function startTurn(data, binary) {
      if (stopped) return;
      if (inFlight) return fail(failure("Only one response may be in flight per connection", "response_in_progress", 409));
      const turn = { requestId: randomUUID(), startedAt: Date.now(), controller: new AbortController() };
      inFlight = turn;
      onActiveChange(1);
      turn.timer = setTimeout(() => fail(failure("Azure WebSocket request timed out", "request_timeout", 504)), requestTimeoutMs);
      const signal = AbortSignal.any([turn.controller.signal, lifetime.signal]);
      try {
        if (binary) throw failure("Binary requests are not supported");
        const body = JSON.parse(data.toString("utf8"));
        if (!body || Array.isArray(body) || body.type !== "response.create") throw failure("Expected response.create");
        if (!route.allowedModels.has(body.model)) throw failure("Model is not allowed on this gateway route");
        if (model && model !== body.model) throw failure("Reopen the connection when changing model");
        model = body.model;
        if (body.n !== undefined && Number(body.n) !== 1) throw failure("Exactly one response is supported");
        if (body.store !== false || body.background || body.stream || body.generate === false) {
          throw failure("WebSocket requests require store=false and ordinary foreground generation");
        }
        const input = inputItems(body.input);
        if (body.conversation != null || input.some((item) => item?.type === "item_reference")) {
          throw failure("Stored conversation and opaque item references are not supported; send full input");
        }
        const previousId = body.previous_response_id;
        if (previousId != null && (!previous || previous.id !== previousId || previous.model !== model)) {
          throw failure("Previous response is unavailable; resend full context on a new connection", "previous_response_not_found");
        }
        turn.fullInput = previousId == null ? input : [...previous.input, ...previous.output, ...input];
        // Reserve the entire inherited input AND generated output every turn, not
        // merely the tiny incremental frame. Vision validation sees the same state.
        const fullBody = { ...body, input: turn.fullInput };
        delete fullBody.type;
        delete fullBody.previous_response_id;
        const inspection = inspectAzureMultimodalInput(fullBody, model);
        normalizeModelRequest(body, model, route.forceReasoning);
        normalizeModelRequest(fullBody, model, route.forceReasoning);
        const fullBytes = Buffer.byteLength(JSON.stringify(fullBody));
        if (fullBytes > maxPayloadBytes) throw failure("Full context exceeds gateway byte bound", "context_limit", 413);
        accountContext(Math.max(contextBytes, fullBytes));
        turn.reservation = estimateAzureReservation(fullBody, model, inspection);
        const limiter = limiters.get(model);
        turn.limiter = limiter;
        if (!limiter) throw failure("No quota limiter configured for model", "gateway_error", 503);
        // The credential helper is operation-owned and bounded. It uses a
        // separate WS audience, never changes the working HTTP credential scope.
        const credentialForTurn = (refresh) => getAzureCredential(route.webSocketResource, refresh,
          { signal, deadlineUnixMs: Math.min(turn.startedAt + requestTimeoutMs, Date.now() + 30_000), now: Date.now });
        let credential = await credentialForTurn(false);
        if (signal.aborted) return;
        const admission = await limiter.admit(turn.reservation, signal);
        if (signal.aborted || inFlight !== turn) { admission.release(); return; }
        turn.admission = admission;
        if (credential.expiresAt - Date.now() <= 60_000) credential = await credentialForTurn(true);
        if (signal.aborted) return;
        if (!credential?.token || credential.expiresAt - Date.now() <= 60_000) throw failure("Azure authentication is temporarily unavailable", "upstream_authentication_unavailable", 503);
        if (!upstream) {
          for (let attempt = 0; ; attempt++) {
            try { upstream = await dial(credential, signal); break; }
            catch (error) {
              limiter.observeResponse?.(error.upstreamStatus, error.retryHeaders ?? new Headers());
              if (attempt > 0 || ![401, 403].includes(error.upstreamStatus)) {
                if ([401, 403].includes(error.upstreamStatus)) throw failure("Azure authentication is temporarily unavailable", "upstream_authentication_unavailable", 503);
                if (error.upstreamStatus === 429) throw failure("Azure WebSocket rate limit reached", "rate_limit_error", 429);
                throw error;
              }
              credential = await credentialForTurn(true);
              if (signal.aborted) return;
            }
          }
          if (signal.aborted) { upstream.terminate(); return; }
          clearTimeout(lifetimeTimer);
          lifetimeTimer = setTimeout(() => fail(failure("WebSocket authentication or connection lifetime expired; reopen with full context", "websocket_connection_limit_reached", 400)),
            Math.max(1, Math.min(connectionLifetimeMs, credential.expiresAt - Date.now() - 60_000)));
        }
        if (previousId == null) previous = undefined;
        // No retry, redirect or alternate transport is performed after send.
        upstream.send(JSON.stringify(body), (error) => { if (error) fail(error); });
      } catch (error) { fail(error); }
    }

    client.on("message", (data, binary) => { void startTurn(data, binary); });
    client.on("error", shutdown);
    client.on("close", shutdown);
    // Bound even authenticated sockets that never send a first request.
    lifetimeTimer = setTimeout(() => fail(failure("Idle WebSocket expired", "connection_expired", 400)), connectionLifetimeMs);
  }

  server.on("upgrade", upgrade);
  const closeConnections = () => {
    for (const client of connections) client.terminate();
    wss.close();
  };
  // Node's HTTP server.close() otherwise waits indefinitely for upgraded
  // connections, which are not included in closeAllConnections().
  const closeHttpServer = server.close;
  server.close = function close(callback) {
    closeConnections();
    return closeHttpServer.call(this, callback);
  };
  return { connections: () => connections.size };
}
