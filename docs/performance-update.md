# Performance and Windows continuity update

Historical migration report: the TypeScript paths below identify the original changes retained in Git history. Current application code is in `crates/`; see the [Rust implementation](../README.md#implementation).

This change carries the reusable source from the Windows performance work onto the fork's current `main`. It does not install a runtime, migrate user state, modify model configuration, or supply a private gateway. Optional policies remain disabled until their own quality and deployment gates pass.

## Review map

| Area | Behavior and default | Main implementation |
| --- | --- | --- |
| Measurement | Optional bounded, asynchronous local metrics; off by default. Unavailable usage and timing fields remain unavailable. | `packages/agent/src/performance-metrics.ts`, `packages/coding-agent/src/core/performance-metrics.ts` |
| Windows persistence | Serialize asynchronous worker-descriptor writes with generation fencing; use bounded asynchronous rename backoff. File durability and narrow error handling remain. | `packages/coding-agent/src/utils/atomic-file.ts`, `packages/coding-agent/src/modes/daemon/daemon-supervisor.ts` |
| Transcript storage | Encode identical tool-result text once within new entries, then reconstruct the complete visible result on read. Existing files are not rewritten. | `packages/coding-agent/src/core/session-manager.ts` |
| Tool-result context | Optional `repeated-large-text-v1` model-facing policy, off by default; no tool execution is cached. Full output remains retrievable. | `packages/coding-agent/src/core/model-tool-output-policy.ts` |
| Retry behavior | Bounded jitter, provider retry-delay floor, cancellation, and no transparent retry after visible output. | `packages/coding-agent/src/core/provider-retry.ts` |
| Reopen responsiveness | Negotiate `history_ranges` and load a recent UI window with generation-pinned pagination; retain legacy full-history attachment. Full transcript parsing and model context are not eliminated. | daemon protocol/client, agent connection, interactive mode |
| Kernel snapshots | Optional versioned, per-variable content-addressed snapshot storage; legacy snapshots remain the default. This is not dirty-variable tracking. | `prime-agent-runtime/src/rlm/snapshot.py`, `repl.py`, TypeScript kernel manager |
| Summary updates | Optional `consolidate-repeated-v1` prompt policy, off by default; no replacement model or reduced reasoning level. | `packages/coding-agent/src/core/compaction/compaction.ts` |
| Native-compaction capability | Explicit, validated route metadata and provider handling. An unverified managed Azure route is not automatically enabled; no independent gateway is included. | `packages/ai/src/providers/openai-compaction.ts`, coding-agent model registry |

The inclusive global **250,000-token compaction trigger** remains unchanged. Smaller context-window reserve limits still take precedence. This policy does not shrink any model's advertised context window.

## Preserved user-facing fixes

- Inactive orphan subagents no longer appear as main chats. Their parent views remain available; running orphan children remain visible. No sessions are deleted. Explicit top-level forks remain top-level chats.
- The Optimus splash uses the approved compact, colored dot-outline portrait, with complete size variants and monochrome fallback instead of the large density-shaded image.
- The Telegram helper entrypoint regression test verifies the actual manager's spawn contract. Guarded Windows distributions must admit that exact helper entrypoint and its absolute profile argument without clearing `NODE_OPTIONS`; see [Windows documentation](../resources/agent/docs/windows.md). The original pairing failure was in a distribution-specific preload allowlist, not the upstream Telegram manager. This PR documents and tests that contract rather than publishing a machine-specific guard.

## Compatibility and limitations

- Daemon protocol remains version 7; schema revision is 29. History-range behavior is capability-gated and retains the legacy attach path.
- New inline tool-text encodings require the accompanying reader. Old binaries may not decode newly tagged entries. Keep pre-upgrade session backups; switching executables alone is not a safe transcript rollback.
- Snapshot v2 was slower than legacy snapshots in the measured Windows workloads. It stays disabled and is not claimed as a speed improvement. Cross-variable object identity is not guaranteed by its per-variable format.
- Model-facing output reduction and summary consolidation require quality evaluation before enablement. Native gateway compaction requires an actually verified endpoint; configuration labels alone do not establish support.
- Metrics retention is bounded per recorder, not across all historical recorder instances. Sidecars are disposable and must not become a recovery dependency.
- Reduced history transfer is not proof of faster model inference. No end-to-end provider speed or token-saving claim is made for disabled policies.
- Credentials, user sessions, private model routes, installation manifests, machine-specific launchers, and independent gateway implementation are intentionally excluded.
