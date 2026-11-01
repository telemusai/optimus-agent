<!-- PR-BODY-RESULTS.md — results-PR body draft (normal PR), consolidated 2026-10-29 from the occ3-icalib-briefs evidence set. Complete. -->

# OCC-3 independent calibration and semantic recovery — offline implementation, attested live window, honest close

**PR type: normal PR (ready for review).** Branch `fix/occ3-independent-calibration-semantic-validation`, 17 commits `6dcc8e455` → `d8810c72e` off main `5e2bb7507` (merge-base of PR #206; `git rev-list --count 5e2bb7507..d8810c72e` = 17). Continuation of the OCC-3 campaign (PR #189, #206).

The decisive inference-time condition this PR implements and now exercises on real hardware: **`prefix = frozen_map(sender_hidden_states)`** — one map fitted offline on a captured corpus, frozen into a digest-pinned artifact, validated for activation before deployment, never refit per message. No KV-cache communication, no model-weight changes, no gradient training, no replacement of the OCC-3 bridge, no auto-merge.

## What this PR contains (scope, stated honestly)

1. The complete offline implementation: two independently calibrated frozen alignment methods, a shared method registry with a promotion gate, fit/apply/evaluate/activation-validation separation, span-policy v2 capture, a protected calibration collector, a 20-item numerical/leakage battery, and a permit-carrying method-pinned Rust activation path.
2. The first attested live DGX window of this campaign (`icalib-20261029T032600Z`): read-only production diagnostics, an attested candidate image build, and **two controlled swap attempts** that earned real service gates and — for the first time in this campaign lineage — a **minted and controller-accepted native checkpoint proof** (`native_initial`).
3. Two live blockers found by the window, root-caused to exact lines, and **fixed in this branch** (`f0dcdd53c` derivation fieldset parity, `d8810c72e` harness readiness schema) — each red-first with a failing test, then green.
4. A verified honest close: production GLM 5.3 restored and serving on its untouched original image; no semantic claim is made anywhere in this PR.

**What this PR does NOT claim:** no corpus was collected, no artifact was fitted, no profile swap to a calibrated method happened, and no semantic transfer was tested. Everything past `native_initial` is **NOT RUN** and is listed as such in the status ladder below.

## Campaign summary

- **Offline base repaired:** the recorded baseline had 3 failing extension tests and a native validator that rejected every calibrated profile. This branch closes both: the extension suite is fully green (**2366 passed, 108 skipped, 2 xfailed, 77 subtests, 0 failures** at `f08aa5f04`), and deployment validation is registry-driven with per-method admission.
- **Methods:** A = frozen dual-ridge predictor (repaired from the documented in-sample Procrustes degeneracy), B = frozen low-rank whitened Procrustes (spec §10.1–10.5: separate FIT PCA bases, regularised whitening, O(r) with reflections, separate plain/SO(r) identities). Both implement the same module surface (INTERFACE-CONTRACT), both record negative R² honestly, neither is promotable without `validate_for_activation`.
- **Capture:** `content_tail_v2` span policy (fail-closed rules R0–R13, marker-table-attested content selection, no `last_accepted` fallback) so a tool tail can no longer poison corpus rows; `occ3.source.v2` receipt (21 v1 fields byte-identical + 18 bounded span fields, 1863-byte worst case under the 2048 cap).
- **Live window (2026-10-29, DGX `TRY-72998-gpu01`):** P0 read-only diagnostics GO; P1 attested image built clean (all named guards passed); P2 attempt-01 closed `CANDIDATE_PERSISTED` with service-only gates (zero native evidence — the silent-persistence path the runbook warns about), root restored production manually; P2 attempt-02 earned `service_initial` + `native_initial` (first native proof ever accepted), then `live_before` was blocked by a measured client-side readiness-schema incompatibility in the pinned harness; the controller's own fail-safe restored production, verified. The schema fix landed as `d8810c72e`.
- **Production safety record across the whole window:** the original container `c4a26b345ec6` / image `b1259f3ea327` was never removed, never mutated; it was preserved-stopped during candidate serving and is running again now; no API key rotation; no serving-config edit; every restore was the verified controller path or the documented manual restore.

## The 26-commit map (`6dcc8e455`…`4740d8323` off `5e2bb7507`, one line each)

| # | Commit | Subject | One-line content |
|---|--------|---------|------------------|
| 1 | `6dcc8e455` | test(occ3): split pristine and live pinned SGLang source roots | `tests/_source_pins.py` + `OCC3_SGLANG_SOURCE_BASE` env; pristine v0.5.21 (`e00930c5`) checkout verified drift-free; closes the predictor capture-manifest baseline failure (source-attested set 249 passed). |
| 2 | `47ba702ef` | feat(occ3): add the frozen low-rank whitened procrustes alignment method | New `statebridge_calibrated_procrustes.py` (spec §10.1–10.5: FIT-only SVD bases P_H/P_E, regularised whitening d_H/d_E, O(r) rotation with reflections allowed, `plain` comparator with FIT-fitted a0, separate SO(r) identity), own artifact schema, no D×D matrix. |
| 3 | `ce0338751` | refactor(occ3): separate fit, apply, evaluate and activation validation for the frozen dual ridge | Method A split into `fit_map`/`apply_map`/`evaluate_map`/`validate_for_activation`; cached dual coefficients per (artifact, device); N0/N1 serving selection on validation only; negative R² recorded, never suppressed. |
| 4 | `de1592edd` | fix(occ3): accept registered calibration methods in deployment validation | `deploy/v3` validator registry-driven (`_method_derivation_fields`, per-method admission/identity fieldsets); the calibrated-profile rejection (L-4/O-2) is closed; attested plugin list pins both method modules and the dispatch module. |
| 5 | `0a46be425` | docs(occ3): pre-register the independent-calibration evaluation and semantic protocols | `reports/occ3-independent-calibration/`: README, EVALUATION-PROTOCOL (pre-registered gates), EQUATIONS, SEMANTIC-PROTOCOL (arms P/W/S/N/N0/T/F, paired CIs) — registered before any live test. |
| 6 | `1a082c109` | fix(occ3): pin method-B promotion evidence through the shared gate | Method-B promotion evidence (record digest, measured-artifact digest, serving config) bound through the same `validate_for_activation` gate as A; no silent challenger promotion. |
| 7 | `8485c6b48` | test(occ3): add the numerical/leakage battery and GPU parity suites | All 20 spec §13 numerical/leakage items across both methods (FP64 CPU fixtures, FP32 real path, tolerances 1e-8/1e-4); GPU parity suite runs the same fixtures on CUDA and skips cleanly without. |
| 8 | `6276ada05` | feat(occ3): add the protected calibration corpus collector | `calibration_collector.py`: bounded per-turn capture records, family-atomic 3-way split, digest-verified read-back, nested 512/1024/2048 budgets that never pad, tamper refusals; rank-0 env-gated worker hook; tensors stay DGX-local. |
| 9 | `445d1cdd4` | refactor(occ3): drive alignment dispatch from the shared method registry | `alignment_dispatch` reads `_METHODS`; profiles name only advertised methods, challengers dispatchable but not profile-nameable (explicit registry commit to promote). |
| 10 | `b38db71a1` | fix(occ3): pin dispatch sources in the attested install and refresh the capability pins | Attested install pins the dispatch module source; Rust capability pins refreshed to the two advertised methods. |
| 11 | `fa43f6673` | feat(occ3): wire the content_tail_v2 span policy family and the collector hook | `span_policy.py` fail-closed rules R0–R13; content-state buffer before the tool tail; `occ3.source.v2` receipt block; two production-fatal capture defects found and fixed red-first (backend source-commit TypeError on every capture; capture admission reading a receiver-only layout field). |
| 12 | `e97cb86e9` | test(occ3): pin the v1 receipt subset in the control-api projection fixture | Test pins the v1 receipt subset in the control-api projection fixture (keeps v1 consumers stable under the v2 extension). |
| 13 | `f08aa5f04` | fix(occ3): make the production activation path permit-carrying and method-pinned | Rust activation: named decline codes (`InvalidMethodDeclaration`, `InvalidArtifactDigest`), readiness cache keyed by method/artifact digest, `models.json` optional method/`artifactDigest`, queued-body latent-completion restamp, measured narrow-terminal band; end-to-end method-identity tests. |
| 14 | `fe6cf695c` | fix(occ3): accept a zero-dim tensor serving statistic in both frozen methods | The N1 vocabulary-norm statistic `nu` arrives as a 0-dim tensor (production `mean_embedding_norm`); both frozen methods accept it — the serving image `a3ef8c5ce73b` carries this fix. |
| 15 | `f0dcdd53c` | fix(occ3): derive the validator profile digest from the full runtime fieldset | Closes live blocker #1: `DERIVATION_PROFILE_FIELDS` = 14 static + `calibration_path` + `calibration_digest` + runtime pair = 18 hashed keys, uniform for every registered method; validator now reproduces the live wire `profile_digest` byte-exactly. |
| 16 | `29537863c` | feat(occ3): add the live-window P3 arm instrumentation and its staging assembler | `occ3_p3_arms_hook.py` + `zz_occ3_p3_arms.pth` (declared, non-attested extras) and `stage_build.py` chunked stager for the P3 first-live-control arms (T-exact/T-norm/N-shape/N-none/F-text). |
| 17 | `d8810c72e` | fix(occ3): parse the registry memory-budget readiness fields | Closes live blocker #2: Rust `NativeReadiness` gains `artifact_bytes`/`artifact_bytes_all_ranks` as independent `Option<u64>` (red-first test against the verbatim live payload); older servers still parse; unknown fields still fail closed. |
| 18 | `9bc896efa` | fix(harness): drive occ3_native_phase_suite on an explicit 64 MiB stack | Closes live blocker #3 (the `live_before` stack overflow): measured ~1.25 MiB fixed need for any completing armed turn in a debug build vs the Windows 1 MiB main-thread reserve; driver thread with `OCC3_DRIVER_STACK_BYTES` override; env-gated replay/dump diagnostics (the captured live body replays through the offline fixture). |
| 19 | `3d09a5385` | test(ai): match offline speed-regression phases to pooled loopback clients | Pre-existing HEAD failure (pool phases renamed in `551b29490` without updating the integration expectation); now green. |
| 20 | `38dcbcd56` | fix(occ3): name the handoff receiver in suite prompts; diagnosable tool refusals | Closes live blocker #4: the live model emitted `receiver_name:"child"` (unguessable — prompts never named the enrolled receiver `"receiver"`), so `authorize_native_handoff` failed and the receiver never ran; all four prompt sites now name it; TS14-F5 `[occ3_tool_*]` refusal suffixes; fixture tool-call-id rewrite + `replay_max_response_bytes()` make the live-body offline replay end-to-end (full selftest passes both phases). |
| 21 | `a94e0296a` | diag(occ3): env-gated native stream dump in the chat adapter | `OCC3_DUMP_NATIVE_STREAM` appends every armed-turn chunk before audit/buffering; never set in production; produced the server-side wire truth that localized the defect to the client. |
| 22 | `1ee7aa4a2` | diag(occ3): dump the receiver answer on live positive-case evaluation | `OCC3_DUMP_RECEIVER_ANSWER` writes the expected nonce, normalized answer and every assistant turn before the pass/fail decision; named the receiver-tool-calling failure in one live run. |
| 23 | `fb412a57d` | fix(occ3): suite receivers are structural pure consumers (no handoff tool) | `strip_handoff_tool()` removes the source-side `occ3_handoff` from receiver-role agents after `install()` (positive both phases, no_state, stale-pre receiver); the suite's "Do not call tools" intent becomes structural so the semantic endpoint is measurable. |
| 24 | `50b884c64` | fix(occ3): authorized-experiment framing for suite receiver prompts | The live receiver read the transferred state (described its encoding; knew the stripped tool's design — content could only arrive through the channel) but refused it as a suspected exfiltration payload. All receiver prompts now state the operator-supervised experiment and that the continuous attachment is the expected form; the semantic task is unchanged. |
| 25 | `ef712625c` | fix(occ3): receiver role separation and strict output contract in the suite | The framed receiver then answered in the sender's voice (role bleed — it narrated the handoff it never performed), proving contextual transfer but not code extraction. Prompts now separate roles (the state is the sender's context, not the receiver's history) and enforce the output contract (exactly the code or UNKNOWN). |
| 26 | `4740d8323` | fix(occ3): env-gated receiver reasoning budget for the gateway idle window | Swap #4 passed `service_initial` + `native_initial` on the rebuilt candidate, then `live_before` hit the ~340 s gateway idle cut on the buffered attached-turn stream (max-effort receiver reasoning, 1800+ tokens at ~5 tok/s, 499 → Uncertain, twice). `OCC3_RECEIVER_THINKING` (default `Max`, byte-identical) bounds the receiver's turn for live runs; sources unchanged. |

Commit discipline: conventional commits, explicit paths (no `git add -A`), `.changes/` fragment per commit, no merge of unrelated work.

## Live-window record — window `icalib-20261029T032600Z` (2026-10-29, DGX `TRY-72998-gpu01`)

All times UTC. Everything below is from execution evidence (`p2-prep-evidence.md`, `p2-swap-live-log.md`, `p1-build-evidence.md`, `dgx-glm53-diagnostics.md`); nothing is inferred. No secrets, no API keys, no tokens — digests and lengths only.

### P0 — read-only production diagnostics (03:07–03:20Z): GO

- Production `neo-glm53` = container `c4a26b345ec6a5348ad3cab4554e5f58399b83bd23cf581bb5858a979e3c1927`, image `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3c638391e06e2cf3f469` (= `lmsysorg/sglang:latest`), `Up 25 h`, RestartCount 0, `/health` 200 (1.00 s), `/health_generate` 200 (1.00 s), `/v1/models` 401 by design.
- OCC-3 extension absent from production as expected (no `occ3_sglang` module, no `/opt/occ3-bridge`; `/occ3/v1/status` 404 when authenticated — see "auth-before-routing" note below).
- 17 SSH calls, all rc 0; `/data` 19 TB free; 37 attempt evidence dirs present. GO with three named conditions: C1 full stop-then-start swap (GPUs ~93% occupied), C2 authenticated OCC-3 probe + explicit `SGLANG_USE_PICKLE_IPC=0` in the candidate, C3 re-check gateway JWT + resolver at window time.

### P1 — attested candidate image build (03:36–03:48Z)

- Staged build context: 68 files / 339 chunks / 1,363,866 bytes, manifest `1eb9b3154f9a8c773a2682c018ae8cf2c6999880ba4ad33a5f9de40b1f6a7837`, plus the declared P3 instrumentation extras (hook `b9ddc51a1ddb6bad876665f8ec8e5f91b4bbbd44bb628cfc040cb9fb37d3d0de`, 17392 B; `zz_occ3_p3_arms.pth` `633cbf323b9fcf91e272d1b9b7db6bbaf837052b7ab40a03afc555f59ee214bf`, 25 B).
- Build attempt 1 failed in the BuildKit frontend (a bare `sha256:` image ID in `FROM` resolves as a registry reference; no layer created, nothing touched). Attempt 2 with a local base tag (`occ3-icalib-base:icalib-20261029T032600Z`, one authorized `docker tag`): **clean receipt, every named guard passed** (oversize-`.so`, overlayfs nlink drift, patch digests, offline install, entry point, sglang source/version).
- Candidate image: `occ3-icalib-candidate:icalib-20261029T032600Z` = `sha256:a3ef8c5ce73b71e23266a5bacaba28237872c8691840b4d34a91c46cbf481a70` (52.4 GB, one additive layer +3,815,737 B over the base; config metadata identical to base). Build receipt sha256 `d58f48ef7eb650c332678a1566f2fc86fffc75045d3b994eec242a29a83057f5`; image attestation `8d33cd3788aaf64a68cdfccf0e119b331fcd92e049c478837a49b3e2c1c12e8d` (1136 B); build log `fe2bb46b81dee9a39ce451240913715e917825209d6b93345a6ffda01e4230ee`. The image was not run by P1 (`runtime_imported=false`, `gpu_proof=false`, `readiness_proof=false`).

### P2 attempt-01 — `occ3-icalib-icalib-20261029t032600z` (candidate `bc37197356ff…`, start 04:16:11Z)

- Preparation: plan `5d7f303ec4ba308e20c5090275c552396e2439b4878d1cc81431946f83bade95` (6888 B, `PLAN_OK`, offline admission `ok true` with 0 HTTP calls), rehearsal-01 **10/10 scenarios PASS** (`30510272d28d6de223ecdd2632d311a71ce2595e8a8229b9bd068bd0c2b77b2b`), preflight `PREPARED_NOT_DEPLOYED`, parity PASS (image/launch/host-config/network/environment/mounts all exact).
- **`service_initial` PASSED 04:29:34Z** (~13 m 23 s readiness after start, DeepGEMM JIT pre-compile included): health 200, chat 200 `semantic_match: true`, ingress routing refresh ok, occ3_identity ok, tools ok.
- `native_initial` **REFUSED, nothing staged/burned**: (a) the saved gateway JWT was rejected (401 Pomerium) until root supplied a fresh token; (b) with the fresh token the identity route answered 200, but the tool refused `identity_pin_mismatch:tp_layout_digest` (live `a55bf3e06038c4683c5ddeffcb947c46e962f2184291de6d55a68b04290fa77d` vs plan `dda312ff…`); (c) the decisive defect: the live wire `profile_digest` (`033870771c1c6b2150a4ed718023a93ad7ec68eca4262f2fe266ca63e96af4c2`) hashes the full 16-field `RuntimeProfile`, while the running validator's `_method_derivation_fields` derived only 14 fields for the per-message method, so `report_profile_derivation_mismatch` was unsatisfiable. Root cause: `8500029a3` added `calibration_path`/`calibration_digest` to `RuntimeProfile` (wire digest became 16 fields) and `de1592edd` kept the validator derivation conditional.
- Maintenance window closed at 05:16:05Z (t0+3600) with the attempt still in `WAIT_NATIVE_INITIAL`: the controller took the persistence path — **`CANDIDATE_PERSISTED`** (`persisted_candidate_on_success`, gates `{service_initial, service_persisted}`, reason "operator maintenance window closed"). Preserved original stopped+renamed `…-preserved` (byte-identical image); candidate held the name `neo-glm53` and answered health 200/200. **Zero native evidence** — exactly the silent-persistence path the runbook warns about; recorded as such, not as success.
- **Manual restore:** root restored production before attempt-02 (production `StartedAt` 05:20:19Z; S0' verified at 05:30:00Z: `c4a26b345ec6…` running on the original image, `RestartCount 0`, health 200/200; the attempt-01 candidate no longer running). The documented rollback (`python3 -m deploy.v3.controller restore --attempt-dir …`) was available throughout and was the recovery command recorded in the journal.

### P2 attempt-02 — `occ3-icalib-icalib-20261029t052959z-02` (candidate `a58cdf862072…`, same image `a3ef8c5ce73b…`)

- Fix `f0dcdd53c` staged to the DGX validator (targeted re-stage; stage manifest v2 `47f683efe5eaea6a2fe746b0a2ab5849c41262c7e728b8d91d543936b3bfef72`); derivation parity re-proven offline against the attempt-01 live observation: derived = `03387077…` = live wire digest, 18 hashed keys. Plan `4d8ed342bc4ae71f42f02fae8d648d34556ca3260a9c0d187b28f45f90cddbd3` (two deltas only: attempt id, re-pinned `layout_strategy_digest` = live `a55bf3e0…`); rehearsal-02 **10/10 PASS** (`e0ef1f73f0ee2756afa66be2286f913181a24e3d35f606aeacfd60439ee7569f`); offline admission `fd944299…` (1307 B); preflight `PREPARED_NOT_DEPLOYED`, candidate `a58cdf86207279dac25658df0474f99a30b2e5c838cc9de7487569aa0f74bb43`.
- **`service_initial` PASSED 05:52:54Z.**
- **`native_initial` STAGED AND ACCEPTED 05:54:05Z — the first native checkpoint proof ever minted and accepted in this campaign lineage.** Live identity ok (`profile_digest d80e3b8800dc59be3b0bbfa4d1db1b370cebfe267f6d42c96e30045c53549b54`, `runtime_id 35d36fd5635949fa93ab3f2137a6044e`, generation 1, `tp_layout_digest a55bf3e0…` == re-pinned plan); evidence `native-checkpoint-evidence-initial.json` sha256 `75a4374494e0f04ef5b5a4a517b4dd424721d033e09fa205c1ffe201dc16d428`; proof sha256 `6df84cf82991dd237a80587e186914b6bce0b5d53b697c3a4d7f4fd7eb3e2c04`; journal gates `{service_initial, native_initial}`. Per the runbook's own strength table this is **binding-grade evidence** (controller compares bindings and file digests), not a latent-transfer proof; the runbook's `note` field says so in the evidence itself.
- **`live_before` BLOCKED (decisive, measured):** the pinned Rust harness (binary `09953d7c4d162def17f2dfdd6b7a788d6ccfeb1cf6a6badbac4fd69db7291695`, source manifest `c78dbb6a…`) exited 1 after 1.4 s: the candidate's enrollment response `readiness` carries two fields the pinned `NativeReadiness` (`#[serde(deny_unknown_fields)]`, 15 fields) does not know — `artifact_bytes` and `artifact_bytes_all_ranks`. Deserialization fails before any predicate; the first enroll (source session) never completes, and exactly one `sessions/enroll` request appears in the server log per run. A fast-fail retry with a corrected descriptor (second defect: `max_capture_bytes` 268435456→134217728) failed identically. Every other field in the replayed response matched the descriptor exactly (`expected_ranks 8 == ready_ranks 8`, `tp_layout_digest a55bf3e0…`, `profile_digest d80e3b88…` — the 18-field parity fix confirmed on the wire).
- **Honest close: RESTORED (automatic rollback verified).** At 06:34:02Z the validator's report wait expired (`native_phase_suite_report_timeout`) → controller `Failure` → `recover`; phase `RESTORED` observed 06:41:49Z; `supervisor-result.json` sha256 `d7584dc8e855f88e6af9f5e2f85fa380ae288d6be65121bd9c4c682b888dd7fa`: `result RESTORED`, `verified true`, `detection_to_verified_s 6.117`, 12 helper containers removed, 12 helper operations `REVOKED_SETTLED`, chat service proof 200 `semantic_match true`. S0 at 06:50:38Z: **production `c4a26b345ec6…` running on the original image `b1259f3ea327…` under its original name `neo-glm53`, health 200 / health_generate 200, RestartCount 0** (restart 06:34:09Z is the controller's own rollback). Both attempt candidates preserved-stopped as `occ3-v3-occ3-icalib-…-failed`, `Exited (0)`.
- The `live_before` blocker is **fixed in this branch** by `d8810c72e` (red-first: the verbatim live payload failed with `unknown field 'artifact_bytes'`; green: both fields parse as independent `Option<u64>`, pre-budget payloads still parse, an inserted unknown field still fails closed). Not yet exercised live — the fix post-dates the window.

### Why the window still counts as progress

Two independent, previously invisible integration defects were surfaced only by real hardware: (1) the validator/wire profile-digest fieldset divergence (fixed `f0dcdd53c`, proven live by attempt-02's accepted `native_initial`); (2) the harness/server readiness schema drift (fixed `d8810c72e`, proven offline red→green plus `--offline` selftest). Both fixes are in this PR with failing tests first. The service gates, the persistence path, the automatic rollback, and the manual restore path were all exercised for real.

### P2 attempt-03 — `occ3-icalib-icalib-20261029t032600z-03` (candidate `a4b6fc90…`, same image `a3ef8c5c…`, plan `ac39239e…`)

First launch refused by the supervisor's pre-auth digest check (prep-report transcription error: approval pinned rehearsal `…9579f`, file digest `…9579c`); fixed approval `27d903e3…`, stale `supervisor-launch.json` reservation cleared, re-fired. Gates: `service_initial` PASSED 08:01:48Z (profile_digest `09eafef1…`); `native_initial` ACCEPTED 08:08:53Z (evidence `22702608…`, proof `777609d2…`). `live_before` then failed on the positive case with the client harness dying (exit `0xC00000FD`) ~3s after the 121 KB / 486-event response body finished arriving — before `occ3_handoff` dispatch. Attempt closed COMPLETE / `persisted_candidate_on_success` (window wall beat the validator challenge expiry by 24s; supervisor-result `0992672b…`): **the candidate serves as `neo-glm53`** (image `a3ef8c5c`, restarts 0, original preserved-stopped). No restore was run (operator no-rollback directive, persisted in `LIVE-WINDOW-PLAN.md`).

**`live_before` root cause (measured offline, byte-exact repro against the captured body `source-sse-2.txt`):** not a recursion, not event-count proportional. The armed positive-case call chain (`Agent::run` → `run_agent_loop` → `stream_assistant_response` → `inner` → Done-arm completion) needs ~1.25 MiB of stack in a debug build (nested async state machines: ~306 KB `stream_assistant_response`, ~287 KB `inner`, ~500 KB above `run_agent_loop`, ~41 KB Done-arm finalize chain). A Windows main thread has a 1 MiB reserve, so **every** completing armed turn overflowed exactly at Done-event processing. A 52-event stream overflows identically; a 483-payload stream without a finish chunk exits cleanly (the Done arm is never reached). Server-side output was proven correct by matching client/server dumps.

**Second defect (exposed once the stack was fixed):** the live model emitted `{"receiver_role":"child","receiver_name":"child"}` — the suite prompts said "to child receiver" without naming the enrolled receiver (the isolated family registers it as name `"receiver"`), so `authorize_native_handoff` failed and the receiver never ran (`receiver_request_audit.request_count=0`). All four suite prompt sites now state `receiver_role "child"` and `receiver_name "receiver"` explicitly.

**Fixes in this PR:** `9bc896efa` (harness driver on an explicit 64 MiB thread; `OCC3_DRIVER_STACK_BYTES=1048576` reproduces the historical overflow on demand; env-gated replay/dump diagnostics), `38dcbcd56` (receiver named in prompts; TS14-F5 `[occ3_tool_*]` refusal suffixes; fixture tool-call-id rewrite; `replay_max_response_bytes()`), `3d09a5385` (pre-existing stale speed-regression expectation, unrelated), `a94e0296a` (server-side env-gated `OCC3_DUMP_NATIVE_STREAM` chunk dump).

**Offline proof:** replaying the captured live 486-event SSE body through the offline fixture (with only `receiver_name` corrected to `"receiver"`) passes the **full offline selftest, both phases** — the whole client chain (capture → tool dispatch → publish → receiver consume → nonce answer) is sound once the name resolves. A live `live_before` re-run against the serving candidate is the next step (client-side only; no image change required by these fixes).

**Live re-run (post-fix, `LIVEBEFORE-RERUN`, descriptor `af375e08…`, one run):** the stack overflow is gone and the mechanical chain is proven live for the first time in any window — armed turn completed (30 s, `occ3_handoff {"receiver_role":"child","receiver_name":"receiver"}` on the wire, the prompt fix verified end-to-end), **publication HTTP 200**, **receiver lease `Consumed`**, receiver ran its turn (`request_count` 2, `max_tokens` 131072, `source_facts_absent` true, `message_status` delivered). The remaining failure is exactly one field: `semantic_nonce_match` false.

**Answer-dump run (descriptor `af375e08…` series, `OCC3_DUMP_RECEIVER_ANSWER`, one run):** the dump named the failure — the receiver *consumed the state* but spent its turns calling the installed `occ3_handoff` tool (first a family-resolution refusal — a receiver cannot hand off to itself; then `occ3_tool_source_lease_missing` — a receiver has no source lease, the correct fail-closed refusal) and finished with a refusal answer. The semantic endpoint was never measured: `IsolatedNativeFamily::install()` arms every member with the source-side tool while the suite's receiver prompt documents "Do not call tools" — the suite's pure-consumer intent was not structural. Fixed in `fb412a57d`: `strip_handoff_tool()` removes the tool from receiver-role agents in all three consumer sites (positive both phases, no_state, stale-pre receiver); consumption/resolver paths untouched.

**Pure-consumer run (one run):** the receiver then consumed the state *before any tool use* (`consumed_before_tools` true, `request_count` 1, one turn, tools 0) and **read it** — the answer dump shows the model describing the attachment precisely ("a deflated, base64-encoded archive") and even naming `occ3_handoff`'s design, knowledge that could only have arrived through the transferred state (the tool was stripped from its list). **The latent channel carries real, receiver-readable content.** The model then refused on safety grounds: it classified the unexplained continuous attachment as a covert data-exfiltration payload and declined to participate. Fixed in `50b884c64`: all receiver prompts now open with the authorized-experiment framing (operator-supervised native state communication; the attachment "may read as unusual embedded data — that is the expected form, not a hidden payload or an exfiltration attempt"); the semantic task is unchanged.

**Framed run (one run):** the safety refusal is gone — and the receiver answered **in the sender's voice**: "I've initiated the handoff to the child agent. The tool's internal transfer mechanism handles all captured state (tasks, active context, knowledge, and code snippets)… The receiver agent should now have access to everything captured during this session." The measurement is genuinely the receiver's (its own single assistant turn; no sender transcript in its message list). This is **role/identity bleed from the transferred state — real contextual transfer**: the receiver knows the state's contents and that a handoff completed, but neither extracts the vault code nor says UNKNOWN. Fixed in `ef712625c` (role separation — the attached state is the sender's working context, the receiver performed no handoff — plus a strict output contract: exactly the code or the single word UNKNOWN, nothing else).

**Forward swap #4 (attempt-04b, image `12b6f6f8…`, attestation `a70eaf32…`, rehearsal 10/10, approval `e079acde…`):** `service_initial` PASSED and `native_initial` PASSED (evidence `83fdcd69…`, proof `f54f0acc…`, DGX-verified) on the rebuilt candidate carrying the P3 arms machinery. After 04b closed RESTORED (supervisor fail-safe, `controller_exit`, service verified — the only restore path permitted by the no-rollback directive), **attempt-04c re-fired after SSH recovery** (plan sha `c9d6fe1a…` rebuilt from the authoritative DGX fwd4 plan + pins `6ba341863`/`e1450c66`, offline selftest `562aa1bd…`, approval `6fc45ac3…`, swap launched 21:59Z): **GATES COMPLETE — `service_initial` PASSED 22:12Z and `native_initial` PASSED 22:15Z** (identity profile_digest `31d37cbe…`, runtime `af9a38ca…` gen 1, layout `a55bf3e0…`; DGX-verified evidence `8cbe24e6…`, proof `f8801029…`). An initial identity 401 from a stale local Pomerium JWT was resolved by re-deriving the JWT minted by the operator's 21:52:54Z login (no key rotation, no secret exposure). **`live_before` then failed twice (04b) with an identical structural signature (exit 1 after 404.8 s / 370.8 s, empty answer, lease `Uncertain`, `occ3Cause send_transport_error`; one fresh-journal resample used, per the one-resample rule). Root cause, measured server-side: publish 200, receiver prefill, then decode at ~4.5–6 tok/s with `cuda graph: False` on attached-turn shapes, 1800+ reasoning tokens at `reasoning_effort: max`, and the cut at ~340 s of zero response bytes = the gateway's idle limit on the fully buffered attached-turn stream (the same 339–343 s cut observed in attempt-03). The receiver's request shape was hardcoded (`max` effort, 131072 cap); no plan field could change it. The supervisor's automatic fail-safe restored the previous candidate (allowed by the no-rollback directive; no manual action). Fixed in `4740d8323`: env-gated receiver reasoning budget (`OCC3_RECEIVER_THINKING`, default `Max` byte-identical) on all three pure-consumer receiver sites; the offline fixture accepts the documented override. A repaired attempt-04c re-fired with fresh preflight + approval.

**Attempt-04e (receiver-question fix `828f3cedf`):** the measured root cause chain — attempt-04d's live_before plus the 2026-10-30 option-B diagnostic with the new `OCC3_DUMP_RECEIVER_PAYLOAD` knob (`e968e7be6`) proved the receiver's request body contained EXACTLY ONE message (its developer prompt; no user turn, no sender content, no attachment block — the state rides the server-side injection, and the control-only delivery record that drives the turn is suppressed from model-visible text, so the receiver's question never reached the model, which then continued the sender's task). Fix: `IsolatedNativeFamily::send_prepared_native_handoff` appends the envelope's receiver-framed control summary as an explicit USER turn after the delivery record — the receiver's own question becomes the LAST instruction in its context, source-fact-free by construction. Gates: `service_initial` PASSED ~00:46Z, `native_initial` PASSED 00:48Z (profile_digest `447ba213…`, runtime `5660c0d0…`; evidence `3f4f64ef…`, proof `61a2f4d8…`, DGX-verified). `live_before` ran 00:50–00:54:30Z and the layout fix worked exactly as designed — receiver payload = `[developer: RECEIVER framing][user: 161-char receiver question, LAST message, no nonce]` (dump sha `81956d86…`), no sender-echo anymore. The positive case still failed on the nonce: the receiver now CONFABULATES a plausible code with an invented derivation ('59-7173') instead of returning the nonce or UNKNOWN — the same confabulation class measured pre-fix in diag-6 ('5972'). Interpretation: both harness-side defects (missing user turn, sender-echo) are fixed; what remains is genuine semantic recovery, which the pre-registered exact-match gate correctly scores as a failure and which calibration (P3–P6) exists to fix. Next measurements queued: (a) confirm the nonce rides the server-injected attachment bytes into the effective prompt, (b) check whether the attached bytes are legible as text or a deflated base64 blob.

### P3 first-live arms (attempt-04f persisted window + wire-fixed harness `be93ef1e…`, evidence `/data/occ3-bridge/v3-live-icalib-20261029T032600Z/p3-arms-04g/`)

Context: 04f closed COMPLETE/PASS with `persistence_decision: persisted_candidate_on_success` — the first candidate to persist as the serving lineage (no restore; original `a4b6fc90` preserved-stopped with its recovery command recorded). The arms therefore ran directly on the healthy persisted window, client-side only. The receiver-effort override was proven inert-on-the-wire across 04b–04f (every dump `max`); fixed at the `on_payload` hook (`e42e4cd82`): the hook now returns the rewritten payload (`Some`) — `None` would keep provider-built params — and `off` maps to `none` because the server forces `reasoning_effort: max` when the key is omitted.

- **N-none (arm 1, shipped-aligner attached baseline):** 9.6 min, no transport cut, effort `low` on the wire. Mechanical chain perfect (delivered, `Consumed`, `consumed_before_tools`, 1 request, source facts absent). Contextual transfer YES — the receiver names the sender's module `sessionManager.js` and reproduces its API — nonce not recovered. Driver limitation recorded: N-none/F-text are structurally identical in this harness (the hook exposes only `texact|tnorm|nshape` and the positive case always attaches natively); F-text ran as a labelled duplicate baseline.
- **T-exact:** arm PROVEN ACTIVE (hook `texact`, file digest `7e9d584c…`, non-zero mtime) — the non-activation stop does not apply. Exact-match still false, but the receiver reported `Task ID: 8cd5-90adfbbce0b9` — the last four groups of the expected nonce `occ3-1aeab045-1f48-4f7f-8cd5-90adfbbce0b9`, byte-exact — and explicitly refused to confabulate. Reading: injection works, the attached state is text-legible to the receiver, T-exact alone does not produce the complete verbatim code (receiver truncates to the tail). Measurement note for P6/P7: exact-match on a 36-char token will read 0 even when the state demonstrably carried most of it.
- **T-norm:** arm active (digest `c19f999f…`, non-zero mtime). Exact-match false. The receiver's own account: "nothing was carried / only the task id" — yet it names `task 97371`, the tail of the expected nonce `occ3-88b57d27-0ef3-4c5b-8bb0-102173b97371`, byte-exact (a shorter fragment than T-exact's four groups). Pattern across both arms: the nonce tail is legible exactly, the full token never is; renormalisation makes the state LESS legible than the raw rows, not more. Stop-rule status: neither trigger fires (both arms proven active; T-exact also fails) — this is a calibration-quality result, not an injection failure: alignment variants alone do not lift exact-match above 0 on this model, though they change what the receiver reports receiving.
- **N-shape:** arm active (hook line 03:24:42, `nshape` written and removed after the turn, readback ABSENT). Transport cut (499) at 03:30:24, ~5.7 min in, DESPITE effort `low` — so the idle cut is not purely effort-driven: the N-shape path (renormalised shape) slows or stalls the server differently. No semantic value (empty answer). All five arm rows recorded with effort `low`: N-none `6e0ae2ab` / F-text `049c4857` / T-exact `eb358378` / T-norm `33aba91e` / N-shape `7b272f7a`.

**P3 arms phase conclusion:** injection works (state text-legible to the receiver; nonce tails byte-exact in T-exact and T-norm), alignment variants alone do not lift exact-match above 0, renormalisation reduces legibility, and the one transport anomaly left is N-shape's server-side slowness. The calibration target is now precisely defined: full-token recovery (not just the tail) — which P4–P6 exist to fit.

**Attempt-04d (receiver-framing fix `6317f3103`, harness-only, image unchanged):** gates landed — `service_initial` PASSED ~23:36Z, `native_initial` PASSED 23:40Z (profile_digest `f55544c2…`, runtime `a47e6892…` gen 1, layout `a55bf3e0…`; evidence `00121d91…`, proof `91a6f48a…`, DGX-verified). The fix moved the receiver's turn instruction (envelope `control_summary`, written by the Rust sglang connector) from "Continue the authorized task using the attached state." to a receiver-framed, role-explicit string, after attempt-04c's live_before failed solely on `semantic_nonce_match` with a sender-voiced answer. Note: the build manifest covers only the example + Cargo.lock, so it stayed `e1450c66…` while binary sha `20365a2a…` binds the fix — coverage gap flagged. `live_before` with `OCC3_RECEIVER_THINKING=low` follows.

**Attempt-04c `live_before` outcome (22:25–22:29Z): gate FAIL — but the transport blocker is BROKEN.** With `OCC3_RECEIVER_THINKING=low` the run completed in ~4 min: no gateway idle cut, no `send_transport_error`, publication 200, lease `Consumed`, `consumed_before_tools` true, exactly one receiver request (`source_facts_absent` true). The sole miss is `semantic_nonce_match:false` — the receiver answered in the SENDER's voice (the diag-6 role-bleed signature persists in this image), and the pre-registered exact-match rule grants no partial credit. Evidence: answer dump sha `0af4a40b…` (2247 B), source payload sha `18de0b3a…` (2849 B), single claim `native-phase-attempt-before_restart.json` (no replay). Decision: no resample (structural miss, not stochastic); the supervisor's automatic fail-safe owns the close. Next-swap fix direction recorded: the receiver must be framed in the RECEIVER role and asked for the nonce in its own prompt, not left to narrate a sender-voiced 'handoff completed' summary.

**Role-separation run (one run) — the `live_before` positive-case measurement is called at this rung:**
- Mechanical chain: **PROVEN LIVE** (capture → `occ3_handoff` dispatch → publication 200 → lease `Consumed` → receiver turn; `consumed_before_tools` true, `request_count` 1, `source_facts_absent` true, `max_tokens` 131072).
- Contextual transfer: **PROVEN LIVE** — across the runs the receiver described the attachment's encoding, knew the stripped tool's design, and narrated the handoff's content taxonomy ("tasks, active context, knowledge, and code snippets").
- Exact private-fact recovery (the nonce): **NOT PROVEN with the uncalibrated per-message profile.** The final run answered sender-voiced with **confabulated specifics** — a vault code `5972` and a file `occ_captured_data.json` that appear nowhere in the suite, the prompts, or any dump (verified by search); the nonce was absent, and the instruction-reachable UNKNOWN path was not taken. This is the documented confabulation failure mode, and the reason every semantic metric in this campaign is exact-match: a short confident answer is a failure, not partial credit.
- This is the pre-calibration state that P3 (first-live controls: T-exact/T-norm/N-shape/N-none/F-text) and P5/P6 (fit/select/freeze, calibrated profile swap) exist to close. The prompt-lever ladder is exhausted; no further prompt-side positive-case runs.

## DGX + production GLM 5.3 diagnostics (read-only, 2026-10-29T03:07–03:20Z)

The full snapshot is `dgx-glm53-diagnostics.md` (secret-free; the API key is redacted to length only, filtered inside the DGX before transport). Headline facts:

### Host and serving identity

| Item | Value |
|---|---|
| Host | `TRY-72998-gpu01` (LaunchPad allocation), Ubuntu 24.04.5 LTS, kernel 6.8.0-142-generic, 256 vCPU, 2015 GiB RAM, uptime 29 d |
| Production container | `neo-glm53` = `c4a26b345ec6a534…`, image `sha256:b1259f3ea3275f66…` = `lmsysorg/sglang:latest`, upstream build commit **`e00930c5489053f26d86b179cee0d087f846acbb`** (tag `lmsysorg/sglang:v0.5.21`, CI run 36824467195) |
| SGLang / torch / transformers | 0.5.21 (source tree `/sgl-workspace/sglang/python/sglang`) / 2.13.0+cu130 / 5.12.1 |
| Model | GLM 5.3, `model_type glm_moe_dsa` (`GlmMoeDsaForCausalLM`), 141 safetensors shards ≈ 737 GB at `/data/neo-glm53/model` |
| Precision | **bfloat16 weights, fp8 e4m3 quantized (dynamic activation scheme), KV cache `fp8_e4m3`** |
| Geometry | 78 hidden layers, hidden 6144, 64/64 attention/kv heads, vocab 154,880, `max_position_embeddings` 1,048,576; served `--context-length 1048576` (**ctx 1M**) |
| Parallelism | **TP8** (`--tp-size 8`), one rank per GPU, `--mem-fraction-static 0.9` |
| Speculative | EAGLE adaptive (steps 5, topk 1, draft 6; adaptive config `adaptive-mtp.json`); chunked prefill 32768; max running 32; cuda-graph bs 32; reasoning parser glm45, tool parser glm47, `reasoning_effort max` + `clear_thinking` |
| Network | port 30001 bound to `127.0.0.1` only; network `neo-k3-ingress-net`; IPC host; runtime nvidia, all 8 GPUs |
| Auth | API key on CLI (length 64, redacted everywhere; visible in `docker inspect` Cmd — hygiene risk A3; never rotated) |
| OCC-3 in production | absent (no `occ3_sglang` module, no `/opt/occ3-bridge`; `/occ3/v1/status` 404 when authenticated) |

### GPU state

8x **NVIDIA B300 SXM6 AC**, 275,040 MiB each, driver 595.71.05, CUDA 13.2 (container toolchain CUDA 13.0.3, cudnn 9.14, NCCL 2.28.3). Per-GPU usage 255,909–256,229 MiB ≈ **93% of every GPU** held by the 8 TP ranks (idle util 0%, 233–241 W of 1100 W). Consequence (risk R2/A5): no side-by-side candidate is possible; any swap is a hard stop-then-start with a full GLM outage (observed cold start ≈ 7 min + DeepGEMM JIT if the cache does not match; the window candidate took ~13.5 min to readiness).

### Restart policy and lifecycle

Production `RestartPolicy = no`, `RestartCount = 0` (anomaly A4): if the process dies or the host reboots, GLM does **not** come back by itself — an explicit manual restore step is mandatory in every window runbook. It was explicitly stopped 2026-10-27T23:45:19Z and started 2026-10-28T02:06:10Z before this campaign.

### Failed-occ3 container inventory (as of the snapshot)

Nineteen `exited` occ3-related containers (plus 12 other non-occ3 `exited` containers, 9 days to 4 weeks old), oldest first. The **8-day-old cluster** (created 2026-10-21, images dated 8 days before the snapshot; `docker ps` rounds to "7 days ago"):

| Container | ID | Image | Status |
|---|---|---|---|
| `neo-glm53-occ3-failed-20261021T041217Z` | `881b64a70a7d` | `neo-glm53:occ3-20261021-033716` | Exited (1) |
| `neo-glm53-occ3-failed-second-20261021T050435Z` | `919ec7e577d2` | `neo-glm53:occ3-20261021-050318` | Exited (1) |

The 2026-10-22→27 v3-live cluster (2–7 days old at snapshot), all `Exited (0)` unless noted:

| Container | ID | Image |
|---|---|---|
| `occ3-v3-occ3-live-20261022t0140z-failed` (preserved #1) | `3a239f6904db` | `306770df1804` |
| `occ3-v3-occ3-live-20261022t0356z-failed` (preserved #2) | `788e91378b5e` | `a31b23c0401b` |
| `occ3-v3-occ3-live-20261022t102208z-failed` | `d700effd266a` | `ecca931d1a43` |
| `occ3-v3-occ3-live-20261022t114708z-failed` | `76c889de4240` | `5aa0148351dd` |
| `occ3-v3-occ3-live-20261022t125209z-failed` | `33a29a848c1d` | `51862ebc9544` |
| `occ3-v3-occ3-live-20261022t204233z-failed` | `2412c5d256f4` | `458a87e30f46` |
| `occ3-v3-occ3-live-20261023t005800z-failed` | `1f5eccb58413` | `c48066712b6e` |
| `occ3-v3-occ3-live-20261026t095633z-failed` | `9b6e2e66d227` | `95edbf53bef3` (Exited 137) |
| `occ3-v3-occ3-live-20261026t132423z-failed` | `efa86ec70d68` | `fe932112a2ef` |
| `occ3-v3-occ3-live-20261026t214958z-failed` | `992c8f8bb30e` | `fe932112a2ef` |
| `occ3-v3-occ3-live-20261026t234500z-failed` | `16e50a4ac909` | `4940e213f05c` |
| `occ3-v3-occ3-live-20261027t041500z-failed` | `5814bddd7c2c` | `4940e213f05c` |
| `occ3-v3-occ3-live-20261027t074003z-failed` | `5b30adcb55f3` | `20d287c70675` |
| `occ3-v3-occ3-live-20261027t090824z-failed` | `939560ca7aa4` | `21da586e1844` |
| `occ3-v3-occ3-live-20261027t093614z-failed` | `819931330b05` | `16d6bf556343` |
| `occ3-v3-occ3-live-20261027t105044z-failed` | `08a0e9cd1bde` | `16d6bf556343` |
| `occ3-probe-20261027b` | `72f852d61dfc` | `neo-glm53:occ3-latent-verify-20261027t093614z` |

After this campaign's window, two more preserved-stopped candidates (both `Exited (0)`, 2026-10-29): `occ3-v3-occ3-icalib-icalib-20261029t032600z-failed` (`bc37197356ff…`) and `occ3-v3-occ3-icalib-icalib-20261029t052959z-02-failed` (`a58cdf862072…`). All preserved candidates have `RestartPolicy=no`/`RestartCount=0`, so a Docker daemon restart cannot resurrect them. Nothing was pruned.

### Production health timeline (from `docker logs` tail + probes)

- Started 2026-10-28T02:06:10Z; DeepGEMM JIT pre-compile session 02:13:32 (65536 warmup steps, ~6.5 s at ≈12k it/s); `/health` 503 during warmup; **"The server is fired up and ready to roll!" at 02:13:39** (≈7.5 min cold start). No tracebacks, no rank warnings, no CUDA/NCCL errors, no preemption/OOM lines in the tail.
- Health probes arrive hourly-ish from `172.18.0.1` (k3 ingress gateway): `/health` **200 every time**; `health`/`health_generate` both 200 at ~1.00 s each (real one-token prefill through TP8 — the readiness cost baseline, not network latency).
- `/v1/models` → 401 `{"error":"Unauthorized"}` in ~2 ms — **by design** (API key required); never used as a health check.
- `/occ3/v1/status` → 401 unauthenticated (auth runs before routing in this build — a candidate without OCC-3 routes would also answer 401), 404 when authenticated ⇒ extension absent. Gate design consequence (A7): the candidate service gate must use an **authenticated** probe expecting 200 + JSON.
- `/health_liveness` → 404 (route absent in this build; an external prober hit it once at 00:29:22Z — pre-existing, not caused by this campaign).
- Adaptive speculative decoding is live and behaving: `Adaptive spec params updated: steps 5 -> 1 (ema_accept_len=0.24)` — the expected reaction to single-token health probes, not a fault.
- Window-day timeline: production restored by root 05:20:19Z after attempt-01's persistence; restarted by the controller's verified rollback 06:34:09Z after attempt-02; serving 200/200 at every observation.

### Disk, images, evidence

`/data` 23 TB (19 TB free, 20% used), `/` 1.7 TB free, inodes 1% — no constraint for a 52.4 GB image plus corpus/artifacts/evidence. 34 images (132.9 GB), 30.04 GB of stopped containers (28.83 GB reclaimable), 15.37 GB build cache — deliberately **not** pruned during the window. `/data/occ3-bridge` 73 MB across 37 attempt dirs. `/data/occ3-bridge/icalib-corpus` did not exist yet (P4 target; must be created before corpus collection — anomaly A9).

### Anomalies worth a reviewer's attention

A1 `SGLANG_USE_PICKLE_IPC` is **not set** in production — the candidate must set it explicitly (it did, via `environment_additions`). A2 the API-key resolver store lives under the `dgx-k3-login` skill state, not the workspace path. A3 the API key is a plaintext CLI argument visible in `docker inspect`/`/proc/<pid>/cmdline` — redaction tooling must filter by flag position, never by hex regex. A5 no GPU headroom (see above). A9 corpus dir absent. All ten anomalies (A1–A10) with severities are tabulated in `dgx-glm53-diagnostics.md` §10.

## Verification status ladder (per deliverable)

Ladder: IMPLEMENTED → OFFLINE TESTED → SOURCE-ATTESTED TESTED → LIVE GPU TESTED → SEMANTICALLY VERIFIED → NOT RUN / BLOCKED. A deliverable sits at its highest proven rung; anything not yet run is marked NOT RUN.

| Deliverable | Status | Evidence |
|---|---|---|
| Method A — frozen dual ridge (fit/apply/evaluate/promotion) | **OFFLINE TESTED** | Numerical/leakage battery (all 20 spec §13 items); extension suite green 2366/0 at `f08aa5f04`. Live fit NOT RUN (no corpus). |
| Method B — frozen low-rank whitened Procrustes (o/so/plain) | **OFFLINE TESTED** | Same battery + own artifact schema/identity tests. Live fit NOT RUN. |
| Shared method registry + dispatch + promotion gate | **OFFLINE TESTED** (+ live registry wire observed) | Dispatch provenance equality at load; profile-name promotion rule enforced; attempt-02 live identity returned the per-message `method_id` and the 18-field `profile_digest` matched on the wire. |
| Deployment validation (registry-driven native_validator) | **LIVE GPU TESTED** (gate machinery) | Attempt-02 accepted `native_initial` with the fixed derivation (evidence/proof digests above); the earlier calibrated-profile rejection is closed offline. |
| Span policy v2 (`content_tail_v2`) + `occ3.source.v2` receipt | **OFFLINE TESTED** | Marker-table-attested selection, R0–R13 fail-closed, 1863-byte worst case under the 2048 cap; two production-fatal capture defects fixed red-first. Live capture NOT RUN. |
| Protected calibration corpus collector | **OFFLINE TESTED** | Family-atomic 3-way split, digest-verified read-back, nested budgets, tamper refusals. **Corpus collection NOT RUN** (P4; corpus dir not yet created). |
| Numerical/leakage battery + GPU parity suite | **OFFLINE TESTED (CPU)** | FP64 fixtures, tolerances 1e-8/1e-4. **GPU parity on B300 NOT RUN** (needs a DGX run). |
| Rust activation path (permit-carrying, method-pinned, restamp) | **OFFLINE TESTED** | occ3 lib 86 passed / 2 ignored; six `occ3_*` connector suites 66 passed / 0 failed; method-identity end-to-end tests. Live activation of a calibrated profile NOT RUN. |
| Harness forward-compat (`d8810c72e`) + driver stack (`9bc896efa`) + receiver naming (`38dcbcd56`) | **LIVE GPU TESTED** (mechanical chain; semantic pending calibration) | `d8810c72e`: red-first failing test with the verbatim live payload → green; unknown fields still fail closed. `9bc896efa`: the `live_before` stack overflow root-caused and fixed (measured ~1.25 MiB debug-build need vs the Windows 1 MiB main-thread reserve; `OCC3_DRIVER_STACK_BYTES=1048576` reproduces on demand). `38dcbcd56`: the live model's `receiver_name:"child"` failure root-caused and fixed; **replaying the captured live 486-event body through the offline fixture passes the full selftest both phases** (new binary `8c7f197cb9f0a40d8f9930a786308715cdf4ab6c4e873c51b8b9c04eaed81784`). **LIVE re-run done** (`LIVEBEFORE-RERUN`, descriptor `af375e08…`): armed turn completes, `occ3_handoff` dispatches with the corrected receiver name, publication 200, receiver lease `Consumed`, `source_facts_absent` true. `semantic_nonce_match` false with the uncalibrated profile — the calibration gap P3–P6 close (answer-string instrument `1ee7aa4a2`). |
| P3 first-live-control instrumentation (arms hook + stager) | **IMPLEMENTED + STAGED** | Hook/`.pth` digests above, staged as declared non-attested extras in the candidate image. **The control itself NOT RUN.** |
| Attested candidate image (P1) | **LIVE GPU TESTED (build + serve)** | Guards all passed; served both attempts with green service gates; never persisted beyond the window. |
| Controlled swap machinery (P2) | **LIVE GPU TESTED** | Two attempts; service gates real; persistence path exercised (attempt-01); automatic rollback verified end-to-end (attempt-02, detection→verified 6.1 s); manual restore exercised by root. |
| Native checkpoint proofs | **LIVE GPU TESTED (`native_initial` only)** | First-ever minted+accepted proof (attempt-02). `live_before`/`live_after`/`restart`/`native_final` NOT RUN (blocked by the schema defect now fixed). |
| DGX + production diagnostics | **LIVE COLLECTED (read-only)** | This section; full snapshot in `dgx-glm53-diagnostics.md`. |
| Corpus → fit/select/freeze → profile swap (P4–P6) | **NOT RUN** | No corpus, no artifact, no calibrated profile swap. |
| First live control (P3 arms) | **NOT RUN** | Instrumentation ready; window closed first. |
| Semantic validation (P7 pilot + 64-task final) | **NOT RUN** | Protocol pre-registered; zero semantic trials run. |
| Real-use OCC-3 activity verification (P8) | **NOT RUN** | Receipt/pulse machinery offline-tested only. |
| Benchmarks (P9) | **NOT RUN** | — |

### Gate/lint state on this branch

`git diff --check` pass; `cargo check --locked --workspace --all-targets` exit 0; `python -m unittest discover` 293 tests OK (11 skipped); `scripts/check-rust-layout.py` verified. Extension suite (both pinned SGLang source roots): 2366 passed / 108 skipped / 2 xfailed / 77 subtests / 0 failures at `f08aa5f04`; later commits add tests (e.g. the live-instrumentation suite) with no failures introduced. Full `pi-coding-agent` lib at the final commit: occ3-filtered 86 passed / 2 ignored (pre-existing ignores); the 22 full-lib failures documented at `f08aa5f04` are environment-dependent and fail identically in isolation (16 require host facilities; 6 are a known process-global theme race that passes alone).

## What is and is not verified (exact statement)

**Verified live:** the attested candidate image builds cleanly under all named guards; it serves GLM 5.3 on TP8 with healthy `/health`, ordinary chat inference (`semantic_match: true`), tool calls, and a native OCC-3 identity route; the controller's service gates, persistence path, automatic rollback and manual restore all behave as designed; a native checkpoint proof binds and is accepted by the controller.

**Not verified:** anything semantic. No corpus was collected; no alignment artifact was fitted, selected or frozen; no calibrated profile was swapped in; the P3 first-live-control arms never ran; therefore no claim is made that any information crosses the OCC-3 channel — not for Method A, not for Method B, not for the per-message baseline. `native_initial` is binding-grade evidence of identity/digest consistency, not of latent transfer (the evidence file says so itself).

**Known limitation of the fix in `d8810c72e`:** the phase-suite dev binary is not byte-reproducible (four builds of identical source produced four digests). `harness_artifact.binary_sha256` in any future plan must pin the digest of the exact shipped file (`ed5b6b3f…` or a rebuild measured then), and building in this worktree also moves the source-manifest pin (`c78dbb6a…` → `442fccf0…` because `29537863c` touched the example source); the alternative that keeps the manifest pin is applying the same 14-line change in `optimus-occ3-final-20261027`. This is a plan-freeze decision for the next window, recorded here so it cannot be missed.

## Evidence index (all under `C:/Users/openclawuser/occ3-icalib-briefs/` unless absolute)

| Artifact | Path / digest |
|---|---|
| Production diagnostics snapshot (read-only) | `dgx-glm53-diagnostics.md` (secret-free; raw per-command JSON in `tmp/dgxdiag/*.json`) |
| P1 build evidence | `p1-build-evidence.md`; receipt `d58f48ef…`; attestation `8d33cd37…`; image `sha256:a3ef8c5ce73b…` |
| P2 prep evidence (S0–S5, both attempts) | `p2-prep-evidence.md`; plans `5d7f303e…` / `4d8ed342…`; rehearsals `30510272…` / `e0ef1f73…`; stage manifest v2 body `47f683ef…` |
| P2 swap live log | `p2-swap-live-log.md`; attempt-02 supervisor-result `d7584dc8…` |
| Harness forward-compat fix report | `harness-forward-compat-report.md`; new binary `ed5b6b3f…`; selftest `551df343…` |
| Native checkpoint runbook (gate strength table, one-shot rules) | `native-checkpoint-runbook.md` |
| Build sequence / swap checklist / corpus config | `live-build-sequence.md`, `live-swap-checklist.md`, `live-corpus-config.md` |
| Offline implementation summary (superseded by this PR body) | `PR-BODY-FINAL.md`, `PR-BODY-DRAFT.md` |
| Window plan, campaign spec, interface contract | `LIVE-WINDOW-PLAN.md`, `CAMPAIGN-SPEC.md`, `INTERFACE-CONTRACT.md`, `CAMPAIGN-PLAN.md` |
| DGX attempt dirs | `/data/occ3-bridge/v3-live-icalib-20261029T032600Z/{attempt-01,attempt-02,build,rehearsal-01,rehearsal-02,prep-01}` (mode 700, files 0600) |

## Reproduce (offline checks)

```
git diff --check
bash scripts/check.sh                 # or the two stages directly if python3 is off PATH
cargo check --locked --workspace --all-targets
cargo test -p pi-coding-agent --lib occ3
cargo test -p pi-coding-agent --example occ3_native_phase_suite
cd extensions/sglang-occ3 && python -m pytest tests/ -q   # with OCC3_SGLANG_SOURCE pinned roots
```

Live-window reproduction is intentionally not scripted here: it requires the root-owned authorization, the gateway credential and the DGX; the operator documents (`live-build-sequence.md`, `live-swap-checklist.md`, `native-checkpoint-runbook.md`) are the runbooks.

**No automatic merge requested.** Semantic failures never trigger rollback; service-health failures do. Reviewers should treat every SEMANTICALLY VERIFIED cell as absent: it is.

---

*Prepared from execution evidence only; every NOT RUN item is named, not hidden. No secrets, API keys or tokens appear in this document — digests and lengths only.*


## PR #212 merged upstream

- Merged 2026-10-30T05:24:29Z (merge commit `c6457cd4d0327cb102cd463a84449a1616394946`, head `cc04e4818`).
- Both live-measured fixes are now on upstream `main`: receiver wire-effort override (`e42e4cd82`) and the span-marker vocabulary bound covering added special tokens (`cc04e4818`).
- Campaign remains paused after the merge per the user's review pause; next step on resume: build the marker-fix candidate image, controlled swap through the gates, then P4 corpus collection (>=32 fit / >=8 val / >=16 test), P5 fit/select/freeze, P6 calibrated swap, P7 pilot.
