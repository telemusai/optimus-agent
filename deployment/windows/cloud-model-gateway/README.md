# Independent Windows model gateway

This is the source-managed successor to the installed Optimus gateway. It adds
Azure OpenAI Responses WebSockets; GitHub Copilot connects directly from the Rust
provider. Ollama and Foundry chat-completions HTTP behavior is unchanged.

Deployment state is deliberately excluded from Git: `deployment.json`, the local
bearer credential, encrypted secrets, trusted Azure modules, logs and PID records.
The gateway fails startup when configuration is absent or invalid. A guarded
deployment must preserve these files and the scheduled-task launch independently
of the agent. Do not launch or replace production from these tests.

`deployment-config.mjs::extractLegacyDeployment` extracts the old Azure endpoints
and identities without executing the old gateway. Verify the old source hash and
exact HTTP route equivalence before persisting its output to `deployment.json`.
It uses the documented WebSocket URL and a separately configurable Azure audience;
it never changes the working HTTP audience. Validate actual account connectivity
before setting `websocketEnabled: true`. It defaults to false, preserving HTTP.

Install only this package's locked dependency with
`npm ci --ignore-scripts --no-audit --no-fund`. Run:

```text
node --test gateway.test.mjs rate-admission.test.mjs http-proxy-acceptance.test.mjs azure-responses-websocket.test.mjs
```

These tests use injected credentials and fake upstreams, never paid providers.
The model-contract fixture contains only model metadata, not credentials.

Authenticated `/health` advertises `responsesWebSocket.version: 1`, its path and
eligible models only when the deployment explicitly enables WebSockets. Otherwise
`enabled` is false and `models` is empty, so the client retains HTTP.
Each connection permits one foreground, `store:false` response at a time. Every
turn validates and reserves the full inherited input and output, including images.
Unknown previous IDs, model changes and unbounded opaque references fail closed.
Full-context turns reset the previous window, including after compaction.

Closing the connection cancels owned credential/admission/provider work. The
adapter never retries a submitted response; only an authentication-rejected
handshake can refresh once before sending. Timeouts, connection lifetime, memory,
connection count and outbound buffering are bounded. Neither prompts nor provider
output are logged. Gateway/helper build IDs must be deployed together.
This candidate identity is `optimus-gateway-monitoring-repair-20260918.1`.

Azure quota admission preserves FIFO ordering across HTTP and WebSocket callers
within each model limiter. Slot release and cancellation wake pending requests;
token/RPM reservations, pacing, provider cooldown and restart embargo still apply.
Reservations are not refunded from provider usage/remaining-token headers, whose
cache and rate-window accounting may differ from the local conservative estimate.
Logs include measured admission `waitedMs`, `rateWaitReasonsMs` and
`limiterAtResponse`; health snapshots include `queuedRequests`. Wait-reason
durations describe observed blocking constraints and can overlap, so they must
not be summed as independent wall-clock time. No prompts or credentials are
included. Ollama does not use this limiter and is unchanged.

An unexpected Azure WebSocket close records its code and timing in
`websocket_upstream_close` and the affected `websocket_turn`. Gateway-generated
`connectionId` and per-turn `requestId` correlate these rows with the error returned
to the client. An idle close has no request ID. These IDs do not identify an Azure
request or prove the remote cause. The error keeps the `connection_closed` code.
Only exact known transport phrases are retained as `closeReason`; other remote
text is omitted with `closeReasonDisposition: "redacted"`, because it can contain
credentials or prompt text. Empty reasons are marked `empty`. No submitted turn
is retried by this diagnostic path. Existing retry and timeout policies are unchanged.

Protocol source: [Azure Responses WebSockets](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/websockets).
