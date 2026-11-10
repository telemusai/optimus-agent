# ITERATIONS — append-only log (campaign occ3-bench-opt-20261109)

Built 2026-11-10 by reports-init worker from `ledger/LEDGER.jsonl` (23 entries, 2026-11-09T12:58Z
through 2026-11-10T04:32Z) plus the smoke logs under `workers/`. Append-only: new iterations go at
the bottom; existing entries are never rewritten. Measured facts only.

## Commit chain on branch `perf/occ3-live-benchmark-optimisation` (verified in worktree)

| Commit | Type | What |
|---|---|---|
| `445facb53` | merge | origin/main at campaign start (= Merge PR #276) |
| `041ec9679` | test | soak snapshot fixture mirroring native OCC3 death state (cherry-pick; resolved pre-existing HEAD break) |
| `bc4a6375b` | fix(occ3) | allow explicit selector past the OCC-3 twin guard |
| `d77689a7a` | feat(taskbench) | four-way benchmark adapter + campaign daemon wiring (+ .changes fragment, 1517 insertions) |
| `da82b6ccf` | fix(taskbench) | split compound model spec into provider/model CLI flags |
| `c1790003f` | fix(taskbench) | hold live parent sessions open for four-way trials (held-parent; current HEAD) |

## 2026-11-09

### 12:58Z — twin guard refusal (guard verified, not a defect)
Spawn of `ollama-cloud/glm-5.3` child refused by occ3_twin_guard (PR #276): any subagent on
model id glm-5.3 must use the OCC-3-configured entry `dgx-glm53-occ3/glm-5.3`. The ds-executor
must NOT be the DGX model under experiment, so it runs on `ollama-cloud/glm-5.2` (different id,
no twin).

### 13:09-13:38Z — stage-1 live baseline (3 dispatch attempts)
- Run 1 (13:09Z): externally stopped at 13:15:02Z (prime-agent.explicit-stop, cause unconfirmed).
- Run 2 (13:16Z): model turn hung after 13:26:19Z tool result (main proc 0 CPU delta over 20 s);
  both executor procs killed (8892, 18680), daemon untouched. Stale 5.2-child receipts stashed
  under `probes/baseline-live-1/stale-from-5.2-child-12h59z/`.
- Run 3 (13:27Z): reduced brief (08-registry, 09-ordinary-completion, 00-SUMMARY) completed
  cleanly. **13:36:08Z: stage-1 baseline COMPLETE, 10/10 receipts.** Manifest assembled
  (BASELINE-MANIFEST.json + reports BASELINE.md). Key numbers: client ee2b9e56 (0.1.44), server
  image 9728d909, ext fingerprint cfa2f1be, runtime 79019420 gen 1 TP8 8/8, calibration 9a7394de,
  alpha artifact 0.0 / server default 0.0 / client default 0.3, max_active 1 (env unset) vs
  frontend armed-turn cap 2, registry NATIVE_READY 0 channels 146 lifetime admissions, ordinary
  completion 133 ms.

### 13:27Z — twin-guard fix committed (`bc4a6375b`)
`fix(occ3): allow explicit selector past the OCC-3 twin guard`. 3 files, +120/-19. Gate:
`cargo check --locked -p pi-coding-agent --all-targets` exit 0 (with cherry-picked fixture +
uncommitted fix at gate time). Tests: `cargo test -p pi-coding-agent --lib rlm_model_resolution`
= 11 passed / 0 failed (7 existing, 1 rewritten, 4 new regression tests). NOT committed-built:
no client cutover yet (still on 0.1.44 ee2b9e56).

### 22:47-23:06Z — DGX chat-dark incident (diagnosed, self-recovered)
- 22:47:22Z chat turn failed: OCC-3 re-enrollment after 9 h idle refused (install_refused) ->
  fail-closed. Initial client-credential diagnosis was WRONG (models.json untouched; stored
  values are runtime-resolved envelopes, not plaintext).
- Final root cause: the 12:53Z enrollment expired ~13:53Z (1 h server TTL); the first
  re-enroll-after-expiry was transiently refused; stale state cleared by 22:59Z. Fresh sessions
  enrolled fine at 22:59:56Z / 23:00:53Z; root CLI one-shot on dgx-glm53-occ3 -> "OK" in 8.8 s.
- Server-side dark-diag verdict (23:06Z): serving never restarted (Up 43 h, RestartCount 0);
  registry error_count FLAT at 146 == baseline; the 22:47:20.89Z + 22:47:21.38Z worker_control
  bursts (2x250 ms enroll retry) reached all 8 TP ranks but no registry admission op fired.
  Campaign-relevant flake: long-lived DGX sessions idling > 1 h may refuse their first re-enroll;
  fresh sessions unaffected (adapter arms are fresh CLI sessions). Receipts: probes/occ3-chat-dark-1/.
- Separate process failure acknowledged: root ended turn at 13:38Z without dispatching the next
  stage or arming a heartbeat -> 9 h campaign stall.

### 23:07Z — four-way adapter review: PASS for live use
Root review of `research/taskbench/four_way_adapter.py` verified: all four arms drive real
production paths; strict vs historical metrics fully separated; all four guards present and
fail-closed; NO DGX mutations anywhere; no secrets. Non-blocking finding: per-session occ.json
`occ3Enabled` has zero production consumers (harmless).

### 23:12Z — smoke run 1 (txt arm): dispatch-level failure (0/5)
Adapter CLI invocations never passed `--daemon-socket` -> first `--print` spawned a STRAY
standalone daemon (default profile); parent session 01a1c295 spawned on default model
`openai-codex/gpt-6-astra` instead of the requested DGX model; 4x provider_stream_failure
(model_not_found); adapter blocked 7+ min at 0.27 s CPU; stuck tree killed by root; bash exit
127; 0/5 trials; no records written.

### 23:28Z — smoke run 2 (run_id beab7b): socket fix verified, model-split defect found
With `d77689a7a` (committed before run 2): preflight on the campaign daemon OK
(daemon_keys=['sessions'], cli_version 0.1.44); list/send/rename/stop socket wiring works.
NEW DEFECT: parent spawn passed the compound spec to `--model`; the CLI IGNORES a compound
provider/model in `--model` and falls back to the default provider -> every trial errored
verbatim "The gpt-6-astra model is not supported when using Codex with a ChatGPT account".
Root live probe verified the split form `[cli --daemon-socket SOCK --provider dgx-glm53-occ3
--model glm-5.3 --print 'Reply with exactly: OK']` -> rc 0, "OK", 8.9 s.

### 23:31Z — smoke run 3 (run_id 72cfd1): split form still fell back
`da82b6ccf` split flags in use, same gpt-6-astra error -> cause: missing
`PRIME_AGENT_CODING_AGENT_DIR` env in standalone (WMI) launches. Fixed in the launcher, no code
change.

### 23:36-23:40Z — smoke run 4 (run_id eef851, WMI PID 19516): DGX path works, archive flaw found
With the env fix: no codex fallback; parent spawn turns completed; rlm admissions succeeded;
child_usage_attributed recorded. NEW DESIGN FLAW: one-shot `--print` sessions ARCHIVE when their
turn completes (verified via list --all --json) and archiving ABORTS their children (RLM child
fwL0002 failed: "Session input was aborted") and makes the parent unreachable ("Unknown active
session" on rename). All 5 trials failed within seconds. Run killed at ~23:40Z.
Root-cause chain complete: (1) missing --daemon-socket (run 1, fixed d77689a7a); (2) compound
--model spec (runs 2-3, fixed da82b6ccf); (3) missing PRIME_AGENT_CODING_AGENT_DIR (run 4,
launcher fix); (4) one-shot parent archives and aborts children (run 4, needs held-parent
redesign).

## 2026-11-10

### 00:10Z — session resume + held-parent gate
Session resumed (new root 01a1c2bb); origin/main fetched: 445facb53 unchanged (the prompt's
1614bae6 was stale); all 15 prior ledger entries and worktree state recovered. Worker
occ3-lbo-heldparent reviewed + gated the held-parent redesign: selftest 36/36, py_compile
clean, sessionFile key verified 3 ways (wire struct, live list --json 41/41, prior live
measurement); found + fixed one defect (held-parent Popen leak on mid-trial CliError,
idempotent _kill_held_parent). Committed `c1790003f`.

### 00:16Z — smoke run 5 dispatched (txt arm, 5 tasks, held-parent build)
WMI-launched (PID 24968 cmd.exe; env PRIME_AGENT_CODING_AGENT_DIR set; adapter c1790003f;
answer-timeout 240, send-timeout 180). Run id 20261110-001625-7ed39d.

### 00:51Z — mid-run root-cause: CLI send reaches only roster-ACTIVE sessions (defect #2)
The harness-level prompt delivery uses CLI `send <name> --message <text>`, but CLI send resolves
ONLY ACTIVE top-level sessions. The fw speaker/listener children go roster-inactive immediately
after their READY turn (isSessionActive=false, rosterStatus=inactive; plain `list` shows no fw*
sessions) -> every harness send fails "Unknown active session: fwL0001" (probe-verified by name
AND by session-id: both rc 1). Children DO receive via the agent-message bus (drill-proven,
parent->child and sibling->sibling). Prior taskbench used direct chat-completions, so no prior
art existed for harness->child delivery. Held-parent itself WORKS (children spawned, discovered,
stayed live; parent held until cleanup stop archived it as designed).

### 01:18Z — mid-run timeline note: brittle DONE detection (defect #1)
Task C attempt 1 hit a SECOND harness defect before the CLI-send stage: the held parent spawned
both children + keepalive but never emitted its final DONE text (transcript frozen at 25
records, last = thinking 01:01:51; CLI wrapper + worker + ping children all alive,
attachedClients=0, isStreaming=false). DONE-poll timed out at 900 s; the kill sweep fired as
designed; attempt 2 started. Two observed modes: (a) parent hangs after final thinking with NO
final assistant text; (b) parent emits final text as PROSE, not the exact
"DONE fwSxxxx=<id> fwLxxxx=<id>" literal (D2 01:47:00 "fwS0010 replied READY. Both children...").

### 02:47Z — run 5 result (exit 0, 5/5 tasks failed at spawn stage)
Exit code 0 (adapter ran cleanly and wrote results). Runtime 9033 s (~2 h 30 m vs 10-50 m
estimate). Errors: "parent spawn: DONE marker not seen in time (10/10 attempts: all 5 tasks x
2)"; trial_failed_after_2_attempts x5; 0/5 answers delivered (delivered_channel=null x5,
wire_bytes count 0). Records: 5 appended to four_way_runs.jsonl; four_way_results.json txt cell
n_trials 5, all accuracy fields null, terminal_error_rate 1.0.

### 04:31Z — run 5 verdict: SMOKE FAILED 5/5, single error class
Single error class "parent spawn: DONE marker not seen in time" (900 s DONE-poll timeout each
attempt). CORRECTION of the 00:51Z mid-run diagnosis: the claim that tasks A/B passed the DONE
stage was WRONG (inference from timing, not transcript evidence) — executor stdout shows DONE
never seen on ANY attempt. The CLI-send/inactive-children defect (root probes 00:51Z, rc 1
"Unknown active session" by name and id) remains REAL but LATENT — the run never reached the
send stage. Held-parent Popen lifecycle + kill sweeps worked exactly as designed all 10
attempts. Open: source of the legacy_410 guard firing (in all run records; needs full stdout
scan; candidate: occ3 connector legacy-route log line in CLI stderr).

### 04:32Z — worker wave dispatched (current state)
User authorized up to 10 subagents. 5 offline OCC3 design workers dispatched (throughput-design,
bench-locator, controls-design, reports-init, metrics-gaps) alongside in-flight fixwave1.
Next dependency chain: fixwave1 commit -> root verify -> ds-executor re-smoke (WMI, one at a
time) -> pooled four-way matrix.

## Defect register (root-caused; fix status)

| # | Defect | Found | Fix | Status |
|---|---|---|---|---|
| 1 | Adapter CLI calls missing `--daemon-socket` -> stray daemon, wrong model | run 1 | `d77689a7a` | FIXED (verified run 2) |
| 2 | Compound `--model provider/id` ignored by CLI -> default-provider fallback | runs 2-3 | `da82b6ccf` + launcher env | FIXED (verified run 4: DGX path works) |
| 3 | One-shot `--print` parents archive on completion, aborting children | run 4 | held-parent redesign `c1790003f` | FIXED at design level (run 5: children spawned/discovered/live, kill sweeps OK) |
| 4 | Brittle DONE detection: parent final text missing or prose-not-literal -> 900 s timeout | run 5 (10/10 attempts) | fixwave1: DONE-detection robustness | IN FLIGHT (fixwave1 owns four_way_adapter.py now) |
| 5 | CLI `send` reaches only roster-ACTIVE sessions; fw children go roster-inactive after READY | run 5 mid-run probes (latent; run never reached send stage) | fixwave1: child keepalive standby prompt so harness send reaches active children | IN FLIGHT (same fixwave1) |
| 6 | legacy_410 guard firing in all run-5 records, source unidentified | run 5 | needs full stdout scan | OPEN |

## NOT-RUN (as of this document)
- Any complete live trial on any arm (txt smoke trials all failed at parent-spawn/DONE stage;
  occ1/occ2/occ3 arms never launched).
- Pooled four-way matrix; Tracks A/B/C comparative measurements; any accuracy/wire numbers.
- Client cutover to a build carrying `bc4a6375b` (twin-guard fix) — the live client remains
  0.1.44 ee2b9e56.
