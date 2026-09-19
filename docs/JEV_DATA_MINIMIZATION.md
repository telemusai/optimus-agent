# Jev observation data minimization

Status: implemented in this branch; applies to every observation the agent loop
hands to `pi-jev`, in Compare, Active and combined mode.

## Why

Jev sends a bounded summary of what the agent is doing to
`https://api.typesafe.ai/v1/systemone`. That summary leaves the machine, so the
bridge must never copy credential material, full transcripts or raw tool
arguments into a request. Active mode sends the same bounded summary as Compare;
it differs in what it does with the answer, not in what it discloses.

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
   Every operative mode stays opt-in per session (`jev` menu, session/global
   settings); nothing enables Jev on the user's behalf, and Off costs one mode
   check.

## Modes and independent controls

| Mode | Calls the service | Effect on a run |
|---|---|---|
| Off | no feature decision calls | independent compaction keeps its setting |
| Compare | bounded shadow work | none; records recommendations |
| Active | bounded native decision boundaries | only accepted, feature-gated effects |
| Compare + Active | one request per shared boundary | records shadow and active outcomes from that same answer |

`/jev on` stays Compare. `/jev compare-active` selects the combined mode.
`/jev compact on|off|status` controls compaction independently. Disabling
compaction does not turn off Compare or the other feature gates. Compaction works
independently of decision mode, including Off and Compare, when its own toggle is on.

Tool requirement and complexity keep their existing defaults. New feature gates
are off by default. Optional-tool filtering requires explicit optional tool
names and preserves mandatory/internal and forced-choice tools. Context and
memory relevance filters only affect eligible retrieval candidates, never the
persistent store. High-impact continuation, retry and verification decisions
remain advisory. These gates do not remove the legacy Compare category set.

Compaction is a request-local projection over eligible old tool-call/result
pairs. It preserves recent/protected messages, uses bounded importance/relevance
questions and can remove or truncate only validated candidates. It commits only
when the configured minimum reduction is met. It does not rewrite transcript
history or replace normal `/compact` summarization. The six persisted algorithm
parameters are validated and remain subject to hard request/candidate limits.
See [native System One design](JEV_SYSTEM_ONE.md) for the exact boundary and
validation method.

Invalid, missing, refused, stale or unavailable decisions leave baseline behavior
unchanged. No Jev feature changes model/provider selection, permissions,
subagents, messages, depth, concurrency or budgets. Active decisions can add
bounded latency; no measured performance gain is implied.

## Records

Records are bounded metadata, not request or response transcripts. Compare rows
remain `applied: false`; Active rows state actual effects or the fallback reason.
Combined boundaries link their shadow and active rows to one request id, so a
report must count that physical request once. Applied effects do not imply a
measured speed, cost or quality improvement.

Records contain no prompt text, raw tool arguments or credential material.
Compaction records describe action/count/reduction facts, not removed text.
Absent worker telemetry is unknown, not a fabricated zero. Use the benchmark
method in the design document before making savings claims.

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

The installed Linux `optimus-agent` launcher also accepts an explicit
`<agent-dir>/jev/env` file containing `TYPESAFE_API_KEY=value` (or the
`JEV_API_KEY` alias). This is a user-managed plaintext environment file,
separate from the encrypted Windows store. It must be a regular file owned
by the current user with permissions `600`; symlinks are refused. Blank
lines, comments and matching quotes around values are accepted. The file is
parsed as data, never sourced as shell code. Existing process credentials
take precedence over the file. Relaunch the client/daemon after changing it.
Loading a key does not enable any Jev mode.

If `/jev status` reports `Credential: none`, check the active agent directory
(`PRIME_AGENT_CODING_AGENT_DIR` overrides it) and the platform's credential
source. On Windows, use `/jev key` on that computer under the account running
Optimus; a manually written JSON/key file is not a DPAPI envelope, and copying
another computer's encrypted file is not a supported way to transfer the key.
On Linux, put the assignment in `jev/env` and launch with the updated installed
launcher, or set `TYPESAFE_API_KEY` in the process environment when running from
source. The Linux `/jev key` dialog does not provide a secure saved-key store.

Settings writes use a cooperating OS lock, a unique temporary file and a
checked read generation. A conflicting or corrupt settings file is preserved;
the command reports an error so the change can be retried after reload.
This protection applies to cooperating writers, not external editors.
