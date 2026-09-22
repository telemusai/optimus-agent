# Project memory

Prime's project memory extends the continual harness with scoped recall, source evidence, selected history recovery and optional sharing. It keeps the existing session and global stores. It does not import or reclassify old memories automatically.

## Everyday use

- `/memory status`: project ID, host ID, memory revision, recall/learning settings, extraction limits, jobs and sharing state. The terminal also shows the existing auto-refine scheduler settings.
- `/memory search Sydney Astra`: search the complete allowed harness, including prompt, memory, skill and subagent entries. Results show IDs, bounded previews, scope, matching terms, source references and freshness.
- `/memory read project:memory:region`: inspect a complete entry. IDs include scope and kind to avoid collisions.
- `/memory recall off` / `on`: control automatic context recall independently of learning. Turning recall off suppresses the legacy harness digest as well. Provider-owned opaque compaction already submitted before a setting change cannot be retroactively edited.
- `/memory learning off` / `on`: pause or resume new automatic learning. The existing `autoRefine.enabled` setting still controls scheduling. Manual edits and selected imports remain available. In-flight planning rechecks the project setting before returning a proposal; edits already committed remain saved.
- `/memory history`, `/memory rollback <event-id> <current-revision>`: inspect edit history and safely undo an unchanged edit. Updated entries require review rather than overwriting newer work.

All data operations are available through the bundled Python `memory` skill inside the existing `ipython` tool. No additional top-level model tool is added. Read `skills/memory/SKILL.md` for mutation examples. A Python response has `origin` and `result`; preserve the origin label when presenting recalled data.

`await memory.handoff(task, state, decisions, unresolved, event_id=..., revision=..., sources=[...])` saves or updates a project task checkpoint in harness memory. It does not create a task database. Use the revision returned by `memory.status()` and a stable event ID for retries. Automatic captures must pass `automatic=True`; explicit edits requested by the user remain available while learning is paused.

## Identity and storage

Git repositories use normalized origin identity plus an explicit local binding to the common Git directory. Worktrees share identity, nested directories resolve to the repository root, and remote renames retain an established local binding. URLs do not retain authentication credentials. Non-Git directories use canonical path identity until explicitly bound.

Use `/memory bind project_<id>` on a second machine or after an intentional alias change to select an existing identity. Binding selects a store; it does not move or merge old stores. The previous data remains accessible by rebinding its ID. Avoid giving unrelated repositories the same ID.

Authoritative project state is `~/.prime/agent/memory/projects/<id>/harness_state.json`. It uses the existing harness entries, edit application and version snapshots, with an additional `memory` envelope holding project identity, monotonically increasing revision, full edit history and replay receipts. Mutations lock the project and publish state/history/receipts together through an atomic write. Malformed state fails closed. Search is derived on demand; there is no authoritative search database to back up.

Host facts have a locally generated host ID in `~/.prime/agent/memory/host-id.json`. They are filtered on reads and cannot be shared. Preserve that ID when restoring the same machine; do not copy it as part of selected project sharing. Existing global facts remain global and existing session facts remain session-local.

## Evidence and corrections

Refinement now serializes original roles instead of converting injected messages into user statements. Each record includes an ID, origin and content hash. Session-backed extraction also records the original entry ID and transcript location. Harness digests, memory command results and other custom host messages are excluded. Summaries and labelled memory tool results are derived context. Assistant assertions are not verified evidence.

The planner cites `metadata.sourceIds`; the host resolves those references only against records actually supplied in the bounded evidence window. A citation provides attribution, not proof that a claim is correct. Unsupported citations do not qualify an entry for automatic project capture. Automatic project capture copies only explicitly reusable, cited, host-neutral memory entries from completed session refinements. It does not silently overwrite a conflicting project entry.

Exact duplicate content is rejected. Keyword search exposes candidate overlaps for inspection; it does not automatically merge semantically similar claims. Update a corrected fact, or set `metadata.supersededBy` on an old entry while creating its replacement in the same proposal. Superseded entries remain inspectable through `memory.search(include_inactive=True)` and are omitted from automatic recall.

Use `await memory.source('/absolute/runbook.md')` to obtain a file source reference. Repository-relative paths let each machine verify the same source in its own checkout. Changes or deletion invalidate that source for automatic recall. Transcript sources retain immutable copied input; Git revisions may be recorded as additional source metadata. A new repository commit alone does not invalidate every fact.

## Selected history recovery

1. `/memory import-prepare /absolute/session.jsonl` preserves the selected raw UTF-8 Prime JSONL file before publishing a durable job. It reports source hash, source ranges, excluded records and chunk IDs. Code and logs remain available in the copied transcript.
2. `/memory import-run <job-id>` extracts with the session's selected model, normal provider credentials/headers and retry policy. Each completed chunk checkpoints its proposal and usage. At most four chunks run per invocation by default; rerun pending or failed jobs to continue. One extraction worker per project runs at a time across processes.
3. `/memory import-read <job-id>` shows the proposed edits and coverage. `await memory.import_chunk(id, index)` reads a specific source chunk; preparation/status responses avoid dumping raw conversations into context.
4. `/memory import-apply <job-id> <current-revision>` explicitly applies the accepted proposal. The exact operation is persisted before applying, so a crash between state publication and job acknowledgement can be retried safely. A changed revision requires reviewing current state first.

Python also provides `import_prepare`, `import_run`, `import_read`, `import_chunk` and `import_apply`. Python extraction resolves the selected model from the installed registry; models registered only in a session extension must use the owning session's `/memory import-run` command. No import runs automatically at startup and no network model call is made for ordinary recall.

Project backups include authoritative state and its edit history. Every mutation takes a backup; imports preserve their raw inputs separately under `jobs/`. `/memory backup` returns a checksum-protected backup ID; `/memory restore <backup-id>` validates it before replacing state and first backs up current state. Restore increments the revision and retains replay receipts. For a portable archive, copy the entire project directory, including `backups/` and `jobs/`, while Prime and import workers for that project are stopped. Nothing is deleted merely because it is old.

## Optional sharing with Axiom

Sharing remains disabled until a project has an endpoint and token-file setting. The service is a single authoritative writer with revision checks, durable receipts, and backups. It accepts only explicitly selected project memories; host entries are rejected. It does not sync transcripts, global stores, tokens or whole agent directories.

On Axiom, generate a separate random memory token into a private file (at least 32 characters; mode 0600). Do not reuse Bedrock credentials. Start the installed service with Node:

```sh
node /path/to/coding-agent/dist/core/memory/server.js /home/ubuntu/.prime/shared-memory /home/ubuntu/.prime/memory-token.txt 8799
```

The server binds **127.0.0.1**. Use a supervised process such as a user systemd service for persistent operation. On this computer, establish an authenticated SSH tunnel:

```sh
ssh -N -L 8799:127.0.0.1:8799 axiom.telemus.ai
```

Place the same memory token in a private local file, bind the same project ID on both machines, and configure each client:

```text
/memory configure {"shared":{"url":"http://127.0.0.1:8799","tokenFile":"/absolute/memory-token.txt"}}
/memory sync
/memory share selected_entry_id another_entry_id
/memory sync
```

Axiom can use its own loopback endpoint directly. HTTPS is also supported; plaintext non-loopback endpoints are rejected. Bearer tokens are read from files and are never returned in status. The service token grants access to the personal shared-memory service; this is not a multi-tenant authorization system.

A share operation first persists its exact outbox payload. Sync uses that operation ID and the last known server revision. Offline failures retain pending work and cached reads. Revision conflicts never overwrite remote changes: inspect the other machine's state, archive/discard the pending request with `/memory discard-pending`, sync, then explicitly requeue the intended entries. `await memory.share([], remove=['id'])` explicitly detaches remote entries; the next successful sync invalidates cached copies. `/memory configure {"shared":null}` disables the endpoint and immediately stops using its cached entries. Local originals remain local.

## Limits, diagnostics and existing knowledge tools

Set defaults in the existing user `settings.json` under `memory`; project overrides live in the project store's `settings.json`. `/memory configure` validates and saves project overrides. Defaults:

| Setting | Default |
| --- | --- |
| `recall` / `learning` | `true` / `true` |
| `maxRecallChars` / `maxRecallEntries` | 6000 / 6 |
| `maxExtractionTokens` | 4096 per request; one syntax repair may make a second request |
| `maxImportBytes` | 32 MiB |
| `maxImportChunkChars` | 40000 including evidence labels |
| `maxImportChunksPerRun` | 4 |
| Extraction concurrency | 1 per project |

Session custom entries named `prime-agent.memory-diagnostic` record recalled IDs, scope/project, latency, added characters, extraction usage and failures. They are local diagnostics. Import jobs record completed chunk usage; provider attempts that fail before usage is returned cannot yield exact billing totals. No answer-quality, latency or monetary improvement is claimed without task-level evaluation. Focused fixtures test source selection and recovery, not model answer quality.

Use existing filesystem and MCP capabilities for repository docs and code intelligence. For vr-ai-chat, configure Prime to use that repository's existing ChunkHound MCP launcher, then discover the server's actual tools via `mcp.list_tools` and use its advertised search operations. The bundled memory skill describes this flow and saving authoritative file/hash references. See [MCP integrations](mcp-integrations.md) for server configuration. No duplicate code index, generated wiki, vector service, skill registry or inference proxy is installed.

The terminal and existing daemon/API command transport use the same memory extension; Python uses the same service through the generic host bridge. There is no new daemon command or response schema, so no daemon protocol capability/version change is required. `noExtensions` intentionally disables automatic recall and terminal memory commands; the Python data API remains available. SDK callers constructing their own resource loader can add `createMemoryExtension(agentDir, settingsManager)` as an extension factory. The separate vr-ai-chat 3D/Web/TUI surfaces are unaffected. Azure and Bedrock authentication, model selection, reasoning and Responses compaction implementations are unchanged.
