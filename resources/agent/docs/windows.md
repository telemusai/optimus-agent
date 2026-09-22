# Windows Setup

Prime Agent requires a bash shell on Windows. Checked locations (in order):

1. Custom path from `~/.prime/agent/settings.json`
2. Git Bash (`C:\Program Files\Git\bin\bash.exe`)
3. `bash.exe` on PATH (Cygwin, MSYS2, WSL)

For most users, [Git for Windows](https://git-scm.com/download/win) is sufficient.

## Custom Shell Path

```json
{
  "shellPath": "C:\\cygwin64\\bin\\bash.exe"
}
```

## Guarded installations and Telegram pairing

Custom `NODE_OPTIONS` preload guards are inherited by Telegram's separate Node
worker. A guard allowing only the main CLI and catalog helper will reject
`/telegram` pairing before the pairing link appears. This is an installation
entrypoint-policy issue, not a bot-token or model-provider issue.

An exact-build guard must also pin and admit
`dist/modes/telegram/worker.js` with exactly one absolute argument matching the
selected agent-profile directory. Preserve module hashes, the catalog helper's
IPC requirement and all existing entrypoint checks. Do not clear `NODE_OPTIONS`
or admit arbitrary scripts to make pairing work.

Verify the actual `startTelegramWorker` child process under that guard, using
an isolated profile and fake Telegram/model APIs: pairing link creation, invalid
and expired code rejection, message delivery, stop/restart, durable polling
offsets, attachment and zero-orphan shutdown. Testing `runTelegramWorker` only
in-process does not exercise the inherited preload guard.
