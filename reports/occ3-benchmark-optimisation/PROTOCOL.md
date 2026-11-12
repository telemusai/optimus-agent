# PROTOCOL — OCC-3 Benchmark & Optimisation Campaign (occ3-bench-opt-20261109)

Status of this document: protocol as designed and code-verified. The four-way adapter
(`research/taskbench/four_way_adapter.py`) is implemented (selftest 36/36); live bring-up is in
progress (smoke runs 1-5; see ITERATIONS.md). **No comparative four-way numbers exist yet —
NOT-RUN.** All numeric identities are from BASELINE.md / BASELINE-MANIFEST.json.

## 1. Objective

Measure end-to-end task accuracy and wire cost of the four production communication
channels between two real daemon sessions on the SAME model (`dgx-glm53-occ3/glm-5.3`),
on the 50-task dev matrix (first 10 tasks per family A-E of `research/tasks/tasks_dev.json`,
seed 42), reusing the existing taskbench task definitions and checkers. Every byte on the
measured channel travels a production seam; the adapter only orchestrates sessions and
reads durable records.

## 2. Trial topology (identical across all arms)

One small session family per trial inside the real daemon:

1. **Parent P** — a session on the campaign model whose only job is to spawn two rlm
   subagents (speaker `fwSxxxx`, listener `fwLxxxx`) with a trivial, byte-verified stand-by
   prompt, then emit a DONE marker naming both children.
2. **Harness -> agent prompts** (listener setup, speaker briefing) are delivered verbatim
   by the production CLI `send` command. No model ever relays task content.
3. **S -> L communication** (the measured channel) is emitted by the speaker session itself
   through its kernel `agent_message.send` host request or the native `occ3_handoff` tool.
4. The harness reads durable delivery records from the session transcripts on disk and
   scores the listener's final assistant text.

Why one parent + two children (fairness): plain sibling reach treats any two parentless
roots as siblings, but native handoff authorization requires an actual parent edge. One
parent with two rlm children gives all four arms the SAME topology with a real parent
edge. Rejected alternatives: speaker-spawns-listener (changes topology between arms);
session forks (ambiguous parentage, copied context).

## 3. The four arms (arm difference = exactly one production input)

All arms: speaker and listener on `dgx-glm53-occ3/glm-5.3`, same reasoning and sampling.

| Arm | Speaker authors | Production path (summary) | Receiver-side classification |
|---|---|---|---|
| **TXT** | plain English <= 120 words | body not `OCC/1`-headed -> `wrap_outgoing_body` is identity -> daemon send -> steering prompt | delivery kind `text` |
| **OCC-1** | complete OCC-1 contract (TASK/KIND/STATUS + F1..Fn) | speaker occ.json `occ2Enabled:false` -> daemon policy gate refuses OCC-2 -> body delivered as authored | delivery kind `occ1_native` |
| **OCC-2** | identical OCC-1 authoring to the OCC-1 arm | speaker occ.json `occ2Enabled:true` -> `occ2_send_policy` -> `select_with_policy` (220-char small-message floor) -> `encode_dual` (sym-v1) -> receiver strict decode + strike ledger + amortized codebook legend | delivery kind `symbolic`, or `fallback_consumed` (strict decode failed -> strike + embedded OCC-1 consumed) |
| **OCC-3** | source facts turn | speaker calls `occ3_handoff(receiver_role="sibling", receiver_name=L)`: capture armed -> source facts -> publish -> daemon admission -> control-only summary + pending marker -> listener claims injected native state -> completed-use receipt + `latent_native` restamp | `details.occ3Delivery.kind == "latent_native"` with `receiptSha256` (64-hex) + `completedUse` |

The effective channel is read from the durable receiver record (`details.occ2Delivery.kind` /
`details.occ3Delivery` on the `agent_message` custom record), never from the request.

## 4. Evaluation tracks

### Track A — protocol fidelity (one-attempt)
- One planned source->receiver attempt per (arm, task). The speaker prompt allows only
  bounded transport-error retries of the SAME send (never content retries); wrong answers
  are never retried.
- The trial records `requested_protocol` (arm) and `effective_protocol` (durable delivery
  classification) separately; disagreement is visible, never rescored.
- OCC-3 fallback: if the native path refuses/exhausts and the speaker falls back to
  ordinary messaging, the record carries `native_success:false`, the fallback label in
  `native_fallback_label`, and the effective channel of whatever actually delivered. It is
  never counted as a native success.
- Infra-level trial retries (`--max-trials`, default 2) only for spawn relay corruption,
  discovery failure or receiver-unavailable; each is recorded in the attempt chain; a
  retry never reuses a partially delivered family (fresh parent + children each attempt).

### Track B — product reliability (bounded retry/fallback)
- Measures the production client's own bounded retry/fallback budget under realistic
  conditions, with campaign defaults unmodified (BASELINE.md client configuration):
  busy retry 3 with 2/4/8 s backoff, capture retries 3, capture capacity wait 90 s,
  capture concurrency 1, enroll transport 2x250 ms, restart retry window 3700 s.
- Readout: delivered-rate and the terminal-error taxonomy per arm when production
  defaults run to their own bounds (busy / capacity / enroll-refused / dark-window /
  fallback-consumed), using the same per-trial records as Track A.
- Long-idle re-enroll flake is documented campaign context: a 12:53Z enrollment expired
  per the 1 h server TTL and its first re-enroll after 9 h idle was refused
  (install_refused) at 22:47:22Z; fresh sessions were unaffected (probes/occ3-chat-dark-1/).
  Benchmark arms are fresh CLI sessions, so exposure is bounded to trial duration.

### Track C — throughput / ops
- Wire and token accounting per arm: `wire.wire_bytes` (durable wire body),
  `wire.occ1_bytes`, `wire.dual_estimate_bytes`, `delivered_prompt_chars` (what the
  listener's model consumed), best-effort token usage from session records.
- Effective-channel distribution per arm aggregated from the durable kinds.
- Capacity/ops context from BASELINE.md: RuntimeProfile max_active 1 (registry capture
  slots; env unset), frontend BufferProfile armed-turn cap 2, max turn 900 s, live memory
  budget 256 MiB / 128 MiB workspace / 128 MiB capture / 256 records / 3600 s TTL,
  max_prefix_tokens 128, ordinary-completion HTTP baseline 133.3 ms.
- Session hygiene: one parent + two children per trial (600 sessions for the full
  4-arm x 50-task matrix); the adapter stops each parent after its trial; unique
  per-trial names; daemon maintenance reaps the rest.
- Executor discipline: one live-mutation executor at a time; WMI-launched one-shot
  executors; no DGX mutations from the adapter.

## 5. Fairness and isolation rules

1. Same model on all sessions of every arm (`dgx-glm53-occ3/glm-5.3`); reasoning and
   sampling pinned at Suite 0, matched across arms; context/output limits pinned at Suite
   0 per task budget, identical across arms.
2. Same task matrix and checkers for all arms (50 tasks, first 10 per family A-E,
   `research/tasks/tasks_dev.json`, seed 42).
3. Same topology (Section 2); the arm difference is exactly one production input
   (Section 3).
4. Strict and historical metrics are kept strictly separate in the record (Section 6);
   never conflated in aggregates.
5. Effective channel from the durable receiver record, never from the request or the arm
   label.
6. Wrong answers are never retried; infra retries get a fresh family and are recorded.
7. Fail-the-run guards (Section 7) are evaluated per trial; a tripped guard sets
   `terminal_error` and the trial is reported failed, not silently rescored.
8. No synthetic backends: provenance and gpu_free flags are checked; offline fixture
   results are never presented as live.
9. No DGX mutations from benchmark code (local CLI list/send/rename/stop + transcript
   reads + client-side occ.json only).
10. Historical scores from occ23/occ2opt campaigns are reference-only (different
    gateway/config); the contemporary four-way rerun happens on this stack.

## 6. Metric definitions and verdict taxonomy

**Historical (comparability with the 2026-10-16 baseline):**
- `historical.correct` — `research/tasks/checkers.py` family checker on the listener's
  stripped final answer (A/B/C whitespace-collapse + casefold exact; D order-insensitive
  set; E numeric 1e-6) — identical to `runner.py`.
- `historical.lenient_correct` — runner.py lenient diagnostic (last non-empty line;
  family-D newline->comma); never the headline.

**Strict (separately named, never conflated):**
- `strict.exact_match` — byte-for-byte equality of the listener's stripped final answer
  with the gold answer string. No casefold, no whitespace collapse, no set logic, no
  tolerance.
- `strict.field_match` (OCC-1/OCC-2 arms only) — strict wire fidelity: the delivered
  durable wire body must parse (production-mirrored `rlm.occ1` codec) as a complete OCC-1
  contract with contiguous finding indexes; for a dual body the embedded `#FALLBACK-OCC1`
  section is checked. Symbolic/fallback agreement itself is enforced by the production
  strict decode; a disagreement surfaces as `fallback_consumed` + a recorded strike.

**Native (OCC-3 arm):**
- `native_success` — true only with `occ3Delivery.kind == latent_native`, 64-hex
  `receiptSha256`, and `completedUse`. Fallbacks carry `native_fallback_label` and are
  never native successes.

**Per-trial verdicts:** `strict.exact_match` / `strict.field_match` /
`historical.correct` / `historical.lenient_correct` booleans; `terminal_error` string when
the trial failed; `delivered_channel` from the durable kind (`text`, `occ1_native`,
`symbolic`, `fallback_consumed`, `latent_native`; null when delivery failed);
`requested_protocol` vs `effective_protocol`; `effective_label_alpha` per handoff.

**Aggregates (`four_way_results.json`):** per-arm cells with strict + historical + native
rates, per-family cells, delivered-channel distribution, guard counts. Arms not yet run
are marked NOT-RUN with zero denominators, never zero scores.

## 7. Fail-the-run guards

| Guard | Trigger | Action |
|---|---|---|
| `legacy_410` | any stage output containing HTTP 410 / "Gone" together with a legacy route (`/occ3/v1/send`, `/occ3/v1/recv`, `/occ3/v1/release`) | terminal error, trial fails |
| `native_receipt_missing` | OCC-3 arm without `occ3Delivery.kind == latent_native` + 64-hex `receiptSha256` + `completedUse` | terminal error |
| `mislabel` | an OCC-3-labelled effective channel without proven native success (e.g. an OCC-1/text fallback labelled OCC-3) | terminal error |
| `synthetic_backend` | observed `offline-synthetic-fixture` / `synthetic-fixture` / `fixture-key` / `mock-connector` markers, `gpu_free_proven:false`, or a synthetic-looking model selector | terminal error |

Guards are evaluated per trial on that trial's stage outputs and durable records; total
failure forces guards on.

## 8. Output files and command line

- Live records: `research/taskbench/four_way_runs.jsonl` (per-trial,
  `optimus.four-way-trial.v1`); aggregates: `research/taskbench/four_way_results.json`.
  Dry runs write separate `four_way_dry_runs.jsonl` / `four_way_dry_results.json`.
- Live run (full matrix, design form):
  `python research/taskbench/four_way_adapter.py run --live --cli <optimus-rust.exe> --model "dgx-glm53-occ3/glm-5.3" --arms txt,occ1,occ2,occ3 --tasks-per-family 10 --answer-timeout 900 --resume`
- Prerequisites (standing campaign state): campaign daemon running,
  `agentCommunication.occ3Enabled` + `occ3Experimental` on, OCC-3 model endpoint live.

## 9. Current status against this protocol (honest)

- Baseline identities: COMPLETE (BASELINE.md).
- Adapter offline validation: selftest 36/36.
- Live smoke bring-up: in progress; smoke runs 1-5 exposed and fixed successive harness
  defects (ITERATIONS.md). Run 5 (txt arm only) still failed 5/5 tasks at the parent-spawn
  DONE-detection stage; fixwave1 (DONE-detection robustness + child keepalive) is in
  flight.
- Tracks A/B/C comparative measurements: NOT-RUN (no four-way numbers, no occ1/occ2/occ3
  arm has run a complete live trial yet).
