# Optimus 0.1.2 stability synchronization

These changes adapt selected Prime Agent behavior to Optimus's Rust application and Python runtime. They do not restore the TypeScript application or merge upstream distribution tooling. Reviewed against upstream `8969d24c86ad3be7aff7911f71abd65f2b3e37ec`.

| Upstream change | Native adaptation |
|---|---|
| [#2423](https://github.com/PrimeIntellect-ai/prime-agent/pull/2423), [#2500](https://github.com/PrimeIntellect-ai/prime-agent/pull/2500), [#2475](https://github.com/PrimeIntellect-ai/prime-agent/pull/2475) | Chunk stdout/stderr, cap results and aggregate exception text, validate display/host payloads before dispatch, reject oversized host input frames through existing protocol repair, cap diagnostic cell source. |
| [#2463](https://github.com/PrimeIntellect-ai/prime-agent/pull/2463) | Validate memory/skill/refinement CRUD before mutation or persistence, including strict JSON metadata; retain locking and atomic replacement. Hash the bytes written without reopening a file made unreadable by an intentionally restrictive umask. |
| [#2374](https://github.com/PrimeIntellect-ai/prime-agent/pull/2374) | Allow one stale-chain retry after connection metadata; output, reasoning and tool events still prohibit replay. Clear the rejected attempt's response ID. |
| [#2424](https://github.com/PrimeIntellect-ai/prime-agent/pull/2424), [#2385](https://github.com/PrimeIntellect-ai/prime-agent/pull/2385) | Correlate tools/results using excerpt-local indices, label errors, exclude generated file inventories from iterative summary input, provide the latest retained assistant state, and bound rendered inventories to 6,000 characters with modified files prioritized. Structured file details retain their existing 200-entry-per-list limit. |
| [#2458](https://github.com/PrimeIntellect-ai/prime-agent/pull/2458) | After an announced shutdown, poll the existing socket for at most 60 seconds (configurable in the connection), rediscover the saved session by file or ID and resync its transcript. Bare session stops remain terminal; disposal cancels recovery; update recovery takes over. No daemon process is launched by shutdown recovery. |
| [#2465](https://github.com/PrimeIntellect-ai/prime-agent/pull/2465) | Goal gates include owner-session Bash liveness, kernel settlement notifies the session, and autonomous/headless continuation waits without spending continuation budget. Explicit stop/disposal cancels the wait. Ordinary user prompt admission is unchanged. |
| [#2472](https://github.com/PrimeIntellect-ai/prime-agent/pull/2472) | Treat provider `safety` failures as permanent and preserve their errors. |
| [#2425](https://github.com/PrimeIntellect-ai/prime-agent/pull/2425) | On POSIX, create/tighten the diagnostic directory to 0700 and current/rotated logs to 0600. Windows retains inherited profile-directory ACLs; Unix mode bits are not a Windows ACL guarantee. |
| [#2533](https://github.com/PrimeIntellect-ai/prime-agent/pull/2533) | Handle Opus 5.5 model ID forms as always-on adaptive thinking models; omit disabled thinking and sampling parameters. |

## Compatibility and scope

No daemon wire command, event or response shape changes. The connection adapter now consumes the existing `daemon_closing` event. This is a client-local recovery change using existing `list`, `attach`, snapshot and connection-status contracts, so the daemon protocol/schema versions do not change. An old client retains its previous shutdown behavior; a new client can recover from existing daemon announcements. A daemon without an announcement retains generic transport recovery behavior. Tests cover announced and bare closes, changed active IDs, absent optional capabilities, discovery timeout, disposal and update takeover.

Kernel protocol remains version 3 with unchanged event shapes. Larger application messages are split or explicitly truncated/rejected within existing shapes. The Rust host's frame ceiling is 32 MiB of serialized wire text, above all bounded Python text/display/host-request frames. Cancellation, snapshot formats and executable cell source are unchanged. Diagnostic source copies alone are limited to 2,048 characters.

Opaque provider compaction checkpoints, Optimus's 250K compaction policy, summary validation and bounded length-retry recovery remain in place. Retained assistant excerpts are context, never authority over user constraints. The background-work callback runs after handle locks are released; the goal continuation gate precedes goal/kernel state locks and never spans an await.

These are shared Optimus runtime fixes used by terminal, RPC and Telegram sessions. Restart recovery is implemented in the daemon connection adapter. The separate VR/3D project is not part of this repository or release.

Auxiliary summarization models and automatic quota-reset resumption are deferred: they require separate persisted settings/state, budget accounting and UI behavior. GitHub Actions remains disabled. Stable installers continue building the highest stable tag from source; this change does not install into or restart an active local session.

## Validation

Validated on Linux with locked dependencies and isolated fixtures:

- `bash scripts/check.sh`: workspace/all-target Cargo check, 24 packaging/installer tests, whitespace checks and native resource layout validation passed.
- Provider regression suite: 528 tests passed, including stale Codex socket recovery and Opus 5.5 request payloads.
- Focused coding-agent unit tests: 138 passed, 2 existing tests ignored; kernel bounds/permissions, background continuation, compaction, retry classification, daemon restart recovery, adapter routing and IPython tool plumbing.
- Compaction integration suites (`compaction_fidelity`, `compaction_length_recovery`, `compaction_summary_guard`, `compaction_observability`): 17 passed.
- `uv run --locked --project prime-agent-runtime python -m unittest discover -s prime-agent-runtime/test`: 401 tests, 8 skipped, no failures.

No paid provider calls or production profiles were used. Native Windows execution was not available on this Linux host. Existing compiler warnings remain; no dependency updates were made.
