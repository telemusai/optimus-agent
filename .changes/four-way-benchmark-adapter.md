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
- Held-parent spawn (2026-11-09 run4 fix): one-shot `--print` parents archived on turn
  completion and aborted their children mid-trial; the parent prompt now holds a
  non-blocking keep-alive bash handle, the adapter runs the parent CLI as a background
  Popen polled for the DONE transcript marker (new `--spawn-timeout`, default 900 s),
  and kills the CLI at cleanup, on retry, and at run end.
- Spawn completion no longer requires the literal DONE marker (run5 2026-11-10,
  run_id 20261110-001625-7ed39d: 10/10 attempts timed out because the parent ended
  its turn with a frozen transcript or a prose reply). Completion is now both children
  discovered AND (DONE marker fast path OR parent transcript unchanged for 60 s after
  both children appeared OR roster activity "idle" on two consecutive 5 s polls);
  the 900 s ceiling and the timeout kill sweep are unchanged.
- Stand-by children launch the same non-blocking keep-alive bash handle as the parent
  before replying READY (run5 probe: one-shot children archived right after READY, so
  the CLI `send` stage could not resolve them: "Unknown active session" by name and
  id). Because active children never archive themselves, cleanup now sweeps the
  trial's children by name at trial end, on every retry, and at run end, and a
  spawn-timeout stops that attempt's already-discovered children.
- Offline selftest extended from 36 to 47 checks (spawn-settled decision table,
  stand-by prompt embedding, transcript signature, no-op sweep).
- Legacy-410 guard scoped to transport-command outputs (fixwave1 probe: the
  `list --all --json` roster polls carry an idle session's firstMessage
  "legacy /occ3/v1/send|recv|release return 410", so guard_legacy_410 fired on
  100% of run records, txt arms included). The two `send` turns now run with
  `transport=True`; only their outputs (timeout notes included) feed the new
  guard_outputs list the guard scans, while stage_outputs stays the full
  diagnostic log. Selftest extended from 47 to 48 checks: a roster dump does not
  trip the scoped guard; the same string from a transport command does.
- Family-pool concurrent mode (throughput design, workers/throughput-design.md):
  `--pool-size N` (default 1 = the serial path, byte-identical) amortizes the
  ~15-minute held-parent turn over N trials. One pool prompt admits all 2N
  stand-by children in a single parent model turn (`--parents-per-pool` splits
  the pairs across M parallel parents); discovery is name-based with the
  stand-by relay verified from the child transcripts, and the parent is matched
  by its first user message equaling the exact pool prompt (unique per pool).
- Per-trial state refactor for concurrency: stage_outputs/guard_outputs and the
  held-parent handle move to per-trial/per-pool contexts (PairContext with its
  own sink lists, PoolContext/ParentUnit owning the parent Popen + session), so
  concurrent trials cannot cross-contaminate guard evidence; the serial path
  keeps the harness fields untouched.
- run_matrix pool path: resume-filtered work queue ordered round-robin across
  arms, occ3-capped batches (`--max-occ3-per-pool`, default 2, because the
  native-capture slot is serialized: RuntimeProfile max_active 1), one pool per
  batch, thread-per-pair trials, JSONL appends under a lock, and a global
  locked name counter so fwS/fwL/fwP names never collide across batches,
  pools, or retries.
- Pool teardown: parents stopped by name (archiving a parent aborts its
  children, run4 evidence), Popen kill, sid fallback when the rename never
  landed, belt-and-braces child stops, roster-shrink verification, all bounded
  by `--pool-teardown-timeout` (default 300 s). A hung parent never blocks
  discovered pairs: a missing pair is only declared at `--pool-spawn-timeout`
  (default 1200 s, no idle-based early exit - run5 showed idle rosters do not
  prove a turn is over), its work item is requeued once, and the second miss
  is terminal. A trial that fails mid-body returns to a later batch (never
  retried inside its batch; its pair is dead), bounded by --max-trials.
  `--trial-deadline` (default 3600 s) abandons an overrunning thread (daemon
  thread) and records terminal_error=trial_deadline_exceeded.
- Adapter-crash resume: pool mode preflights a sweep of roster-ACTIVE fw*
  sessions older than run start (archived sessions untouched).
- Selftest extended from 48 to 63 checks: pool flag parsing (defaults and
  values), pool prompt (2N admissions + keepalive + DONE naming every pair),
  name-counter/pair/parent naming, batch planning (occ3 cap, coverage, all-occ3
  edge, task/pair assignment), round-robin work order, concurrent JSONL append
  under a lock, per-pair guard-sink isolation, teardown naming.

NOT-RUN: live four-way trials after the held-parent and completion/standby fixes
(smoke runs 1-5 all failed at the spawn stage; both fixes are offline-verified only,
63/63 selftest checks, py_compile clean). Pool mode is additionally verified only
against an offline fake-daemon simulation (normal / missing-pair requeue /
fail-once retry / deadline / four-arm cap scenarios); no live pool run yet.
