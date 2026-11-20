# Daemon session view disposal errors

- Return the first daemon side-question or detach/owned-session completion failure through `AgentConnection::dispose` as a fixed operation-only error, without exposing arbitrary transport/session/question text. Complete the remaining best-effort cleanup.
- Preserve existing private best-effort disposal callers and an existing native-host terminal or connection error over a later session-view cleanup error. Non-owned sessions still detach; owned-session completion and one-command/idempotent disposal semantics are unchanged.
