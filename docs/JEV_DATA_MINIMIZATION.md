# Jev shadow-observation data minimization

Status: implemented in this branch; applies to every Compare-mode shadow
observation the agent loop hands to `pi-jev`.

## Why

Compare mode sends a bounded summary of what the agent is doing to
`https://api.typesafe.ai/v1/systemone`. That summary leaves the machine, so the
bridge must never copy credential material, full transcripts or raw tool
arguments into a request.

## Rules

1. **Raw tool arguments are not observed.** `tool_call_observation` records tool
   identity, tool-call id and `args_omitted: true`. Tool arguments can carry
   `api_key`, `Authorization` headers, passwords or unrelated credentials, and
   the current evaluators only need tool identity to score a tool choice.
   A source guard in `crates/pi-coding-agent/src/core/jev_bridge.rs` fails the
   test suite if the production bridge names an argument-excerpt field again.
2. **Bound before redacting.** `pi_jev::redact::bounded_excerpt` reads at most
   `MAX_SCAN_CHARS` (4096) leading characters of the source, so a multi-megabyte
   tool result or final message is never fully cloned just to keep a 400
   character excerpt.
3. **Redact before the payload exists.** `pi_jev::redact::redact_text` replaces
   credential-shaped material (authorization headers, provider key prefixes,
   private-key blocks, secret-named assignment values, URL userinfo, JWT-shaped
   strings) with `[redacted]` while the excerpt is built. Agent-end excerpts are
   accumulated character by character and stop at the scan window.
4. **Residual disclosure is documented, not hand-waved.** Redaction is
   pattern-based. It does not make arbitrary private task text safe to disclose.
   Compare mode stays opt-in per session (`jev` menu, session/global settings);
   nothing enables Compare on the user's behalf, and Off costs one mode check.

## Live validation

```
TYPESAFE_API_KEY=... cargo test -p pi-jev --test live_systemone -- --ignored --nocapture
```

The live probe performs one real request, prints the endpoint, attempt counters,
response model and per-question record/skip lines, and asserts only
client-contract truth: at least one transport attempt, `applied == false`, and a
record or explicit skip for the question it asked. It never prints the
credential, the raw request or the raw response body. Use `JEV_BASE_URL` to
point it at a staging or replay endpoint.

Credential sources, in order: saved credential (Windows DPAPI envelope
`<agent-dir>/jev/typesafe.jev-credential.json`), then `TYPESAFE_API_KEY`, then
`JEV_API_KEY`. Linux builds have no DPAPI store, so the environment variables
are the working path there; a hand-written `KEY=value` file at the envelope path
is not read and is not a credential store.
