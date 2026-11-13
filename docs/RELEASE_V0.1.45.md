# Optimus Agent v0.1.45 (draft)

## Independent OCC3 modes

OCC3.1 is an optional TextMAS workflow on the selected ordinary model. It does
not require a latent-capable model, connector or server. Each eligible root user
turn runs three tool-free preparation calls (Planner, Critic and Refiner), then
the ordinary parent acts as Judger/executor with its normal tools and approvals.
Oversized role contexts first use bounded shared mapper/reducer calls and a
lossy advisory brief; every stage remains bounded and accounted, while the
parent retains its full original context. Sequential stages receive earlier
drafts. Hierarchical stages independently read the original or reduced context;
they currently execute serially. Drafts are untrusted advisory text, not hidden
reasoning, system instructions or verified evidence.

```text
/occ 3 off
/occ 3 sequential
/occ 3 hierarchical
/occ 3.2 on
/occ 3.2 off
```

`3.1` is an explicit alias for the TextMAS selector. Bare selectors do not mutate
settings. `hierarchal` and `hiearical` are accepted spelling aliases. TextMAS is
off by default and saved per chat. It does not recursively activate on children
or internal continuation messages.

OCC3.2 is the display name for the existing experimental native latent transport.
Its `occ3_handoff` API, global opt-ins, model compatibility, identity validation,
capture/consent rules and completed-use receipt requirements are unchanged.
Turning TextMAS on does not enable latent communication.

**Migration:** `/occ 3 off` now disables TextMAS only. Use `/occ 3.2 off` for
latent communication. `/occ 3 on` returns migration guidance instead of guessing
a mode. `/occ occ3 on|off` remains the legacy latent argument alias; `/occ3` is
not a slash command.

## Controls, accounting and failure handling

- The bottom status row shows delivery lights only (`TXT`, `OCC1`, `OCC2`, `OCC3.1`, `OCC3.2`); the OCC3.1 light pulses during an
  active TextMAS run and the OCC3.2 light blinks during actual native latent delivery, both distinct from
  message pulses. A selected mode, exposed tool or queued handoff is not proof
  of completed communication.
- OCC-2 and latent preferences preserve one another. TextMAS has a separate
  bounded, atomic per-session preference file. Unavailable optional status does
  not turn a confirmed write to another lane into a reported failure.
- `/stats` shows preparation overhead separately, including known token/cost
  subtotals and explicit unknown usage/cost counts. No fake replies, model
  identities or child sessions are created for accounting.
- User input and accounting reservations are persisted before provider work.
  Failed, interrupted, refused, tool-bearing or cap-exhausted preparation stops
  the turn rather than silently executing a fallback. Cancellation settles the
  waiting input and preserves unrelated queued inputs. Normal final retries
  reuse the drafts without repeating preparation.
- Generated system-prompt guidance explains the host-managed stages, their
  advisory status and the separate latent controls. It does not assert that a
  mode is enabled or that communication has completed.

See [the mode guide](../resources/agent/docs/occ3-modes.md) for bounds and costs.
This is a product adaptation, not the frozen TextMAS benchmark implementation;
previous benchmark scores are not accuracy claims for this release.

## Protocol and surfaces

Protocol 7 remains unchanged. Schema 47 adds the optional `occ31_textmas`
capability, host-owned `occ31_get_preferences`, `occ31_set_mode` and
`occ32_set_enabled` commands, and optional `occ3Modes` configuration metadata.
New clients do not send TextMAS commands to old daemons. Missing or malformed
optional metadata renders unavailable without blocking attachment. Existing
communication kind values and native receipt validation remain unchanged.

3D/Web/TUI parity: these are Optimus runtime and terminal controls. Telemus's
separate 3D and Web app surfaces are unaffected. The shared runtime also handles
eligible direct/RPC root turns. There are no renderer or OS graphics changes.
The existing live context-tree adapter is not expanded here; the visible new
accounting category is `/stats`, not a claim that the pre-existing `/usage`
context-tree path has been repaired.

## Validation and release status

The workspace, resource manifest, Python runtime and lockfiles use 0.1.45.
Completed local validation:

- Runtime integration: 9 offline SDK tests passed, covering both data-flow
  topologies, final tools/history/images, root-only activation, queued batches,
  cancellation, retry reuse, persistence failure and usage accounting.
- Focused Rust unit checks: 222 unique tests passed (261 executions with
  overlapping filters). Coverage includes controls, settings, schema/capability
  gates, indicator/queue separation, accounting and generated prompts.
- Daemon compatibility: all 7 tests passed with `TMPDIR=/tmp TMP=/tmp TEMP=/tmp`.
  The initial inherited long temporary-directory path exceeded the Unix worker
  socket limit and caused three create timeouts. No assertion was changed.
- All 11 long-session regression tests passed, including queued-work aborts,
  streaming attachment, history pagination, resizing and session commands.
- Terminal frame fixtures passed at 80, 110 and 160 columns in both themes.
- Packaging unit tests: 22 passed; native resource layout and runtime lock
  consistency passed.
- Synthetic native latent activity: all 7 tests passed. The pre-existing fixture
  now uses the strict typed source-intent digest and echoes its selected decimal
  anchor; production validation and all existing assertions are unchanged.
- The full `scripts/check.sh` gate passed: workspace/all-target Cargo checking,
  298 Python tests, whitespace checks and native resource layout validation.
  It ran with `TMPDIR=/tmp TMP=/tmp TEMP=/tmp uv run --locked --offline
  --project prime-agent-runtime --with pytest --with httpx bash scripts/check.sh`.
  Earlier attempts exposed missing test imports in the local Python environment;
  the final run supplied both test dependencies without changing repository
  dependencies or skipping checks.

### Compiled ordinary-model smoke

The compiled 0.1.45 daemon used `openai-codex/gpt-6-astra`, the selected Optimus
model route, with a private profile and a temporary copy of the existing login.
Each fresh chat received: "Reply with exactly READY. This is a simple text-only
check. Do not use any tools." Tools and latent communication were disabled for
this smoke. The normal final tool path is covered by the offline SDK tests.

| Mode | Final reply | Completed preparation calls | Normal assistant replies | Elapsed |
| --- | --- | ---: | ---: | ---: |
| Off | READY | 0 | 1 | 2.294 s |
| Sequential | READY | 3 | 1 | 7.799 s |
| Hierarchical | READY | 3 | 1 | 14.607 s |

Host preference echoes, configuration snapshots, persisted role records and
`textmasUsage.calls` agreed. Both enabled runs recorded Planner, Critic and
Refiner completion. Their preparation usage was reported, not unknown. This is
a functional smoke, not a quality or performance benchmark. The first attempted
model (`opencode/gpt-5-nano`) returned HTTP 403, "Model access is disabled," before
any successful reply; that attempt is not counted as validation. All private
credential copies were removed after their daemons shut down.

Linux is the local compilation and runtime target. Windows and macOS use shared
Rust code but have not received native-host runtime validation for this change.
OCC3.2 live latent communication is not retested by the TextMAS smoke.
GitHub Actions is out of scope. Do not publish the release or attach unvalidated
portable artifacts based on this draft.

Rollback: leave OCC3.1 off to use the ordinary execution path. Reverting this PR
restores the prior controls; the separate TextMAS preference file is ignored by
older versions. No transcript or native latent wire format is replaced.
