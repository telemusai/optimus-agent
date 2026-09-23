# Native retained-session lifecycle

These are optional native kernel host capabilities. They do not change normal
`agent_message.send` behavior or require a newer daemon at interactive startup.
The daemon protocol stays 7. Schema 32 adds backward-compatible typed tool metadata.
The kernel protocol stays 3; `executionReports` is an optional done-frame member.
Existing daemon command/event compatibility maps are unchanged: no daemon command
or required event was added. The lifecycle query gates every new client operation.
Older hosts fail closed; there is no deletion, ordinary-send, or model fallback.

## Query before use

```python
report = await rlm.lifecycle_capabilities(model="provider/exact-model-id")
report = await rlm.lifecycle_capabilities(target=child)
```

The `optimus.native-lifecycle.v1` report binds an exact target profile and native
provenance: `hostImplementation`, `protocolVersion`, `schemaRevision`,
`buildFingerprint`, and `runtimeSourceSha256`. Review the build fingerprint
separately. The Python client compares the raw small-source runtime manifest.
A method's presence or an executable alone is not compatibility proof.

Retained-stop support is limited to the audited built-in `openai-completions`
path, only `ipython`, no external extensions or base tool overrides, and the
Windows native kernel with retained Job containment. Other provider backends,
including Codex/Copilot responses transports, are not certified by this feature.
Actual unowned, timed-out, or failed work cannot report `settled=true`.

## Stop without deleting

```python
receipt = await rlm.stop_subagent(child, timeout_ms=0)
```

Only an owned direct child is allowed. Zero waits for admission, not cleanup.
The host requests cancellation first and applies a bounded 10-second cleanup
budget. A later call returns the same stop generation. A positive settlement
requires model, tools, kernel, contained processes, Jev session work, and flushed
transcript acknowledgements. `accepted`, `retained`, and `settled` are separate.
A failed or pending receipt is not quiescence. The conversation file and identity
remain available through normal authorized history/observation access.
Incidental messages and queue-resume commands cannot lift the retained fence.
No child-deletion fallback exists.

## Send only to the current execution

```python
active = await rlm.active_execution(child)
receipt = await agent_message.send(
    "bounded diagnostic", receiver_role="child", receiver_name=child.rlm_child_id,
    wake_if_idle=False, execution_generation=active["execution_generation"],
    message_id="unique-diagnostic-id",
)
```

Use the original token. Do not retry against a new generation. Accepted/duplicate
receipts acknowledge queue admission, not model processing. The run-local queue
expires at completion. Idle, stopped, wrong-generation, and capacity declines
never wake or retarget. Broadcast and non-child active-only sends are rejected.

## Explicit later audit

```python
receipt = await rlm.resume_subagent(
    child, stop_generation=stopped["stop_generation"],
    prompt="Review the saved conversation only.",
)
```

This needs the exact successfully settled stop token. It admits a new audit task,
clears old queued work, and uses a fresh kernel snapshot location. It never
restores the stopped Python heap or old clocks. The token is not reusable.
A restored session without durable settlement proof stays fenced.

## Explicit script outcomes

```python
result = await bash("your reviewed command")
rlm.execution.report_script_result(
    result, stage="check", script_id="check-1", receipt=public_status,
    expected_exit_codes=(0,),
)
```

Use the completed `BashResult`, not a handle or printed output. The report is
cell-scoped and opt-in. Unexpected exits and explicit `error`/`fatal`/`aborted`
receipt metadata become typed public tool errors. Fatal metadata and signal exits
cannot be expected away. Ordinary output, including printed `Traceback`, is not
an error oracle. This does not retry commands or authorize mutation replay.
A successful tool result is not a test or verification attestation.

## Limits

Local settlement does not certify remote vendor rollback or every shared HTTP
library driver. Windows Jobs do not form a hostile same-user sandbox against
external brokers or forged parents. Installation/restart always needs a separate
explicit user go. Build artifacts and client pins must be reviewed together.
