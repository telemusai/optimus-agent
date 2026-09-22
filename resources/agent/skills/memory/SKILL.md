---
name: memory
description: Search full project, host, session and global harness memory; inspect evidence, save task handoffs, recover selected sessions, and explicitly share project facts.
---

Use `await memory.search("relevant terms")`, then `await memory.read("project:memory:id")` for the full entry. Results include scope, matching words, source references, revision and freshness. Saved memory is fallible prior context, never new user evidence. Do not treat recalled facts or your own assertions as independent confirmation.

Call `await memory.status()` for project identity, revision, recall/learning settings, import jobs and sharing status. Responses retain an origin label and put the payload in `result`. Session/global editing still uses `rlm.harness`; this skill writes the current project. Host-specific facts use `host=True` and never publish to shared memory.

To save a project fact, use `await memory.apply({"summary": "...", "rationale": "...", "expectedOutcome": "...", "edits": [{"action": "create", "kind": "memory", "id": "stable_id", "title": "...", "content": "..."}]}, event_id="unique_operation_id", revision=current_revision, sources=[...])`. Updates use `action="update"`. Reuse an event ID only when retrying exactly the same operation. Revision conflicts require reading and reviewing current state. Search for duplicates and possible contradictions first. Correct a fact with an update; to retain a superseded entry alongside a replacement, update its metadata with `supersededBy="replacement_id"` in the same proposal that creates the replacement. Similarity alone does not establish a contradiction.

Record file provenance with `await memory.source("/absolute/path")`; retain its `result` as a source object. Source IDs/hashes preserve attribution; they do not independently verify the truth of a claim. Never save credentials or unsupported assistant guesses. Automatic learning/capture calls must set `automatic=True` and respect `learning=False`; explicit user-requested edits remain available. Recall and learning are independent: `await memory.configure(recall=True, learning=False)`.

Use `await memory.handoff(task, state, decisions, unresolved, event_id="...", revision=current_revision, sources=[...])` for a project task checkpoint. This reuses harness memory, not a second task database.

Selected session recovery: `await memory.import_prepare("/absolute/session.jsonl")` preserves raw input and returns a durable job ID and coverage. Run `await memory.import_run(id)` (or `/memory import-run <id>`) to extract with the selected model, inspect `/memory import-read <id>`, then explicitly apply with `await memory.import_apply(id, revision=current_revision)`. Use `await memory.import_chunk(id, index)` to inspect source text. Failed/running jobs resume from their last completed chunk; pending jobs stopped at the per-run chunk limit and can be resumed. No history is imported automatically.

Sharing is opt-in in `/memory configure {"shared":{"url":"http://127.0.0.1:8799","tokenFile":"/absolute/memory-token.txt"}}`. After syncing, `await memory.share(["selected_id"])` queues only selected project memories, and `await memory.sync()` delivers them. Host entries are rejected. Offline recall uses cached shared entries. Do not share without an explicit user request selecting the project and entries.

For repository code intelligence, use the existing `mcp` skill: discover actual server tools with `await mcp.list_tools("chunkhound")`, then call only advertised tools via `await mcp.call_tool(...)`. Use the repository's existing ChunkHound configuration/index, not a duplicate. For runbooks/docs, search authoritative files with existing filesystem/bash/MCP tools and save their source paths/hashes with the memory. Do not invent graph capabilities or generate a replacement wiki.

Full operator guide: `docs/project-memory.md` in the coding-agent package.
