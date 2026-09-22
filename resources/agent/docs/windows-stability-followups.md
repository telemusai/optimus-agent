# Windows stability follow-ups

These source changes follow `fix/astra-compaction-and-context` at
`c971060ea46d97cd454a6413f01c948bd5e486d4`. They port reusable fixes from a
native-Windows installation rather than publishing its runtime overlays,
credentials, model routes, launchers, or session data.

## Kernel and process lifetime

- A locally settled interrupted execution is not automatically evidence that
  Python has finished. Reuse sends one uniquely correlated, serialized `None`
  barrier and accepts the original completion or the barrier's completion.
  Elapsed time alone never clears execution. Late output remains correlated
  with its original request.
- Reuse waits passively instead of sending interrupts every 500 ms. A genuine
  non-responsive execution still requires the explicit Wait/Kill choice; the
  repair does not automatically restart Python or silently discard variables.
- The reuse wait is 15 seconds and the snapshot deadline is 60 seconds. These
  allowances supplement authoritative settlement; they are not its substitute.
- Windows interrupts use the main-thread signal mechanism, since cancelling a
  task on the same blocked event loop cannot interrupt synchronous Python.
  Native extension calls that do not service Python signals may still require
  an explicit kernel restart.
- Live `BashHandle` objects are excluded from snapshots, including when nested.
  Completed `BashResult` values remain serializable.
- Windows cleanup targets the owned Python process tree, including venv shims.
  A failed tree cleanup retains orphan-recovery evidence rather than treating
  a successful shim-only kill as complete cleanup.
- A narrow source-text guard rejects `subprocess.run(..., capture_output=True)`
  around `start.ps1`, `serve.ps1`, `launch.ps1`, and `run.ps1` on Windows.
  Use log files or `DEVNULL`, a hard timeout, and a separate bounded health
  check for persistent launchers. This guard is a heuristic, not a Python parser
  or a general detector of every possible inherited-pipe deadlock.

## Sessions, refinement, and subagents

Saved-session loading retains the last complete snapshot and shows progress.
It publishes one reconciled catalog after a successful load instead of doing
full-tree work per streamed record. Stale or disposed callbacks cannot replace
the current catalog. This is a UI/catalog fix, not a session migration.

Session files continue appending after an explicit pre-assistant flush. Atomic
writes preserve file fsync and rename, ignoring only Windows `EPERM` specifically
from directory `fsync`; other I/O failures still propagate. Graceful worker
shutdown allows 10 seconds on Windows, retaining 2 seconds elsewhere and 500 ms
for forced shutdown.

Refinements parse strict JSON first. Only standalone `True`, `False`, and `None`
outside quoted strings receive compatibility normalization. A malformed or
schema-invalid proposal can receive one corrective model request. Auth failures,
cancellation, and truncated output are not reinterpreted as repairable JSON.
Invalid proposals are not applied; failure evidence records categories and
fingerprints rather than raw model output. The existing transport retry policy
remains the owner of network retries.

RLM children reserve and persist bounded continuation state before threshold
compaction. Length recovery remains report-only. Runtime-owned visible results
cover both the initial task and later parent tasks, while an explicit reply
suppresses duplicate automatic delivery. Exhaustion is a structured partial
failure rather than a silent idle child. Hidden reasoning is never forwarded.

The task-ID outbox preserves undelivered results across reused-child tasks.
It does not provide distributed exactly-once delivery: a crash after the parent
accepts a report but before the child persists its acknowledgement can duplicate
that report on recovery. Receiver-side idempotency would require a separate
protocol change. Child completion now requires an exact final
`RLM_CHILD_STATUS: complete`, `blocked`, or `failed` line; markerless `stop`
responses receive at most three recovery turns. Root-agent behavior is unchanged.

Python aliases preserve `session_name`/`sessionName` and `text` while exposing
`name` and `content`. `status: "user"` remains the canonical wire value;
`activityStatus: "attached_idle"` clarifies its local meaning. Missing optional
model metadata remains unavailable, not inferred. Richer daemon-wide activity,
terminal-state, and transcript-count fields are outside this PR: adding them
requires a separate negotiated schema/compatibility change.

## Proposed global 250K policy

This is an explicit policy proposal, separate from the correctness fixes:

```text
trigger = min(250,000, model input budget - configured reserveTokens)
compact when contextTokens >= trigger
```

Smaller input budgets retain priority. Model metadata, full context windows,
maximum output, reasoning levels, and recent-message retention are not reduced.
Disabled compaction remains disabled. The proposal replaces the branch's extra
Astra-specific percentage reserve with the common rule. It is not a claim that
250K is an empirically optimal quality threshold for every model or workload;
maintainers can review this policy independently.

## Provider behavior and limits

Codex keeps cached WebSockets and incremental requests. One fresh reconnect is
allowed only before any provider response event; a partial response is never
replayed over WebSocket or SSE. Two pre-response transport failures use the
existing SSE fallback. Provider errors and cancellation do not enter this retry.

Native compaction retains the same model and full filtered context, saves and
reloads the fork's checkpoint format, and allows 20 minutes by default. Codex
subscription models are eligible; unsupported responses retain the existing
fallback. This does not add Codex routes to Azure, GitHub, or Ollama. The OpenAI
API Astra route already present in the fork is retained. Fast mode remains
optional and is not enabled by these changes.

## Validation scope

Focused TypeScript tests use fake providers, mock transports, synthetic session
catalogs, and isolated temporary session files. They cover threshold boundaries,
RLM continuation and parent-input races, partial-result delivery, cancellation,
checkpoint reload, refinement parsing, WebSocket no-replay behavior, narrow
directory-fsync handling, append durability, and worker shutdown grace.

Python tests and the opt-in real-kernel tests use an explicitly supplied isolated
Python environment. They exercise 100 sequential reads/prints, interrupt/reuse,
snapshot/restore, restart/reinitialization, and owned-process cleanup. Example:

```powershell
# From the repository root, using a dedicated test environment:
uv venv .venv-pr-tests --python 3.13
uv pip install --python .venv-pr-tests/Scripts/python.exe -e ./prime-agent-runtime dill
& ./.venv-pr-tests/Scripts/python.exe -m unittest discover -s prime-agent-runtime/test -p test_windows_stability.py -v
& ./.venv-pr-tests/Scripts/python.exe -m unittest discover -s prime-agent-runtime/test -p test_observation_aliases.py -v

$env:PRIME_STABILITY_TEST_PYTHON = (Resolve-Path .venv-pr-tests/Scripts/python.exe).Path
# Then, from packages/coding-agent:
npx tsx ../../node_modules/vitest/dist/cli.js --run test/repl-kernel-real-stability.test.ts
```

The real observation-skill test accepts `PRIME_AGENT_TEST_PYTHON` for the same
isolated environment. Set `PRIME_AGENT_CODING_AGENT_DIR` to a separate temporary
directory and add the checkout's `skills/agent-observe/src` to its `PYTHONPATH`.
Never run acceptance against a production agent directory or session.

No paid model calls, production daemons, installed launchers, gateways, or real
session transcripts are used. Source checks and focused tests do not claim a
full release build, packaged CLI acceptance, or macOS/Linux runtime validation.
