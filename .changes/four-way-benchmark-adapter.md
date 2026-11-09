---
type: added
---

Adds `research/taskbench/four_way_adapter.py` (research tooling): the four-way
benchmark adapter (txt/occ1/occ2/occ3 arms over families A-E, SEED 42) with a
live CLI harness that drives real optimus-agent daemon sessions.

- Position-aware `--daemon-socket` wiring, verified against optimus-rust 0.1.44:
  subcommands take the flag after the command name (`list --daemon-socket <sock> --json`);
  a global flag before a subcommand demotes the line to a run-mode prompt;
  run-mode takes it before `-p/--print`; `--version` prints on stderr and stays bare.
- Preflight probes the daemon via `list --json` (the `status` subcommand rejects
  `--daemon-socket`, verified rc=1 "Unknown option for status").
- Offline dry selftest: 36/36 checks green.
- Live preflight measured: cli_version 0.1.44, 14 sessions visible on the campaign
  daemon, discovery keys (activeSessionId/sessionId/sessionName/sessionFile) present.

NOT-RUN: live four-way trials (smoke dispatch follows this commit).
