# Compaction R5 baseline campaign driver

`compaction_baseline_campaign.py` is campaign infrastructure. It does not change
production compaction behavior. It composes the existing R1-R4 replay,
`verify-local`, `bench-local`, R4-lane and focused test command surfaces.

## Current authorization state

The implementation is under synthetic source/control review. Real Gate A,
builds, real fixture execution and Gate B timing are **not run**. The unit tests
use injected process, telemetry and affinity adapters and non-decision values.
They never invoke a compiled example, real fixture, provider or network.

The source base is the reviewed R4 `b8ae3191a346f7387fc47389d6cd20ef38cdd58f`,
which is `b3408eadf6d354b70ceea1fafe08b15134c642a3` plus the exact approved
R4 eight-path patch. The source contract is base + patch + file hashes, not a
claim that the new R5 worktree is a clean upstream HEAD. The only planned R5
source delta is four new files: this driver, its stdlib tests, this document and
the change fragment. Admission independently invokes read-only Git probes for HEAD,
parent, branch, exact eight-path R4 commit scope, exact R4 patch bytes, tracked
changes, ignored/untracked paths and the four live R5 file hashes. Receipt claims
alone cannot admit the source.

## Roots and no-clobber contract

Real campaign artifacts use only
`C:/Users/openclawuser/Optimus-Assistant/profile/workspace/optimus-compaction-r5-baseline-artifacts`.
Fresh build targets use only
`C:/Users/openclawuser/Optimus-Assistant/profile/workspace/optimus-compaction-r5-baseline-target`
with disjoint `instrumented`, `wall` and `r3-parity` directories.

Each raw JSON, log, work directory and process receipt has a unique stage/run
path. Any existing root, output or reused path stops the stage. Stage inventory
comparison rejects missing, extra and unexpected files. Existing R4 exes may
reproduce non-timing Gate A checks only after their approved source/settings
hash checks. They are not new timing evidence. Gate B accepts only fresh target
builds bound to hashes and build logs.

Every child process records exact argv, a nonsensitive `allowlisted_env`,
sequential order, wrapper/production PIDs, running image and start/end receipt
fields. Focused99 and charcount15 use distinct isolated profile roots under
`gate-a/profiles`. Each first use creates an `.r5-profile.json` protected marker;
subsequent processes reuse only the matching isolated root. Every process carries
an exact `PRIME_AGENT_CODING_AGENT_DIR` injection and before/after full profile
file inventories. The protected marker hash must not change. Mutable test data
is admitted only inside that profile; user/global profiles are never used.
Secret-like variable names are refused. The driver rehashes source,
fixture, toolchain, script, executable and run-matrix inputs before and after
each process. Raw output identity is retained after creation. Failures stop the
affected stage; no retry replaces a raw failure record.

## Gate A: correctness only

Gate A has no timing flag and admits no distribution or performance statistic.
Its required evidence is:

1. Exact ordered R1 (34: 3 accept, 31 reject), R2 (24: 3/21), R3 (71: 4/67) and
   R4 (72: 10/62) name/outcome/intended-stage signatures.
2. Nine ordered invalid process CLI controls with intended stages and no output.
3. One all-21 replay writer per case, in frozen `index.json` order, with a fresh
   read-only reader, frozen expected summary-attempt count and R1 graph,
   raw JSONL/normalized restoration, protected state, checkpoint and tool-pair
   anchors. Each run consumes its actual `writer.json` through the adapter's
   declared output path. The all-21 aggregate is published only after every
   writer and reader passes; a missing or invalid writer leaves no aggregate.
   The small-ascii success uses context window 20000, timeout 90 and
   summary N=2. The intentional summary failure uses N=1, zero committed
   compactions, one transient notice and fresh reader parity.
4. The full-21 R3 `verify-local` route is an explicit unavailable negative.
   It uses the frozen R3 executable and the same full-21 argv as the failed
   attempt-09 route. It must return exit 1, empty stdout, no output report and
   exactly this error after line-ending normalization:
   `error: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit`.
   Its record has `status=unavailable`, `reason=compaction_input_limit`,
   `identity=large-ascii-first-in-lexical-full21` and `partial_parity=false`.
   This is not partial or all-21 parity.

   Exact parity runs only on `small-ascii`, with scope
   `small-ascii-exact-full14`. Both the existing R3 baseline and R4 candidate
   `verify-local` commands receive the single-case fixture. The report must
   contain exactly one case named `small-ascii`. All fourteen complete semantic
   sections must match in values, concrete types and array order. The comparison
   also checks full-projection hashes, exact UTF-8 serialization bytes and finite
   IEEE754 values. Extra, missing or differently ordered cases are rejected.
   Replicated chunk slicing remains non-production end-to-end evidence.
5. R13 R4 preflight on exactly seven candidate cases, independent of the seven
   R3 one-summary-call cases. The four supported timing cases must return full
   valid reports, then repeat in instrumented and wall correctness lanes with
   identical available continuation count/digest. Small-ascii retains 18
   messages and digest
   `f7a5df688012e2c31f1e36a98708a2f596a2abe6ce56fd810cabbdb5d38d317a`.
   The three named negatives must independently return native exit 101, empty
   stdout, no `out.json`/report, and the exact
   `duplicate phase mark summary_runner_entry` panic at `r4_lanes.rs:200:9`.
   Their stderr thread IDs are parsed dynamically, recorded separately from
   native process PIDs, and must not equal those PIDs; no future thread/PID is
   hardcoded. Wrong cases, argv transposition, wrong markers/locations, stdout,
   reports, wrong exits/counts, unrelated panics and invented statistics are
   rejected. These are `unavailable_noneligible` raw controls, not successes or
   Gate A evidence by themselves. Existing probe-3 sample facts are test
   fixtures only: small-emoji PID/thread/stderr SHA256
   `29160/20712/5e28b74c02892782c01c699c2d6e358e13f11f64abde7529a2e49621667f5537`,
   small-tool-huge
   `24644/2756/9a7d6957918e66bc8966781b830c68496b3b0d4f98738fab7d068d67742a360b`,
   and small-many-small
   `17680/27000/7a6f6641301a2f23f4b07c0bc0dbf6012f65661a1a37b1c15385ab849d521793`.
6. Fourteen focused release binaries with 99 passed, zero failed/ignored and
   exact ordered names, plus the charcount filter with 15 passed and exact names.
   The full workspace suite remains unrun unless separately authorized.


R13 keeps all R3 replay ordering/counts, full-21 unavailable handling, exact
small-ascii full14 parity, bytecode isolation, R9 restored identities, the R10
reopen digest, and the R12 nine-case warm / twelve-negative contracts. It does
not reuse probe-3 material as a fresh Gate A. A later real Gate A, still subject
to separate authorization, must execute the four supported cases and their full
continuations plus all three native negative controls. This R13 repair performs
only source/static/synthetic work: no Rust/build, real Gate A/B, timing, warm
producer, archive373, provider/network or delivery.

Focused99 and charcount15 do not invoke Cargo. Their exact direct test-binary
paths come from the frozen R4 provenance logs, with fourteen integration suites
and one charcount libtest binary. Each executable has a live SHA256 check, a
binary-root/name check and the exact frozen `--test-threads=1` argv (plus only
the charcount filter). Ordered expected names and the R4 log hash are also bound.
No manifest-root substitution or background rebuild is permitted. Focused
identity admission occurs before any other real Gate A child.

The R2 `self-test-r2` control uses only `small-ascii`. This is the exact
single-case fixture in the retained successful R3/R4 helper command and meets
Rust `run_r2_self_test`'s `cases.len() == 1` requirement. The 21-case root is not
admitted for this control; it remains for all-21 replay and the explicit full-21
unsupported-route negative.

The root must not treat an `unreviewed: true` Gate A summary as approval.
Independent review writes a separate verdict bound to the exact summary,
source and run-matrix hashes.

## Gate B: baseline-only fresh-process protocol

Gate B remains closed without explicit root authorization, a green Gate A
summary and independent review. The authorization is bound to both hashes, the
run matrix, source and fresh instrumented/wall build identities. Its
`protected_hashes` object must exactly match the actual plan's fixture,
toolchain, script, source, executable and run-matrix file hashes. The fresh
executable paths and SHA256 values must also match the actual executor objects.
The Gate A review and summary bind the same protected non-executable inputs;
fresh timing executables replace only the authorized Gate B executable class.
An unauthorized or stale executor is rejected before timing. It requires
`baseline_only`, `no_network` and `no_comparator`. The driver cannot create this
authorization. R4 one-shot timing rows use the explicit exception of one
observation per fresh process; no warmup or inner sample is invented.

For every Gate-A-demonstrated eligible case, wall, allocation, replay-phase and
minimal-provider each have 30 independent fresh processes. Wall is the external
parent `perf_counter_ns` spawn-to-terminal value. Allocation reports only count
and `bytes_requested`, or null values plus an unavailable reason. Replay phases
cover prepare, compact pre/post provider, append and build continuation.
Minimal-provider additionally covers conversion, serialization, dispatch
preparation and response parsing. Mock wait 25 and generation 3 are controls,
not provider performance. Provider wait/generation is not subtracted from or
included in local phase arithmetic.

Warm bench-local uses exactly nine cases, in this order:
`small-ascii`, `small-cjk`, `small-emoji`, `small-mixed-crlf`,
`small-tool-huge`, `small-many-small`, `medium-tool-huge`, `large-tool-huge`,
`xlarge-tool-huge`. Each has 30 fresh processes. R13 uses exactly four
supported one-call cases in four measured lanes with 30 processes per case/lane:
480 one-call processes plus 270 warm processes, or **750 processes**. The four
cases are `small-ascii`, `small-cjk`, `small-mixed-crlf`, and `medium-cjk`. The
three named duplicate-summary instrumentation negatives are not timing cases;
R3 one-summary-call counts do not infer R4 timing eligibility.

Twelve separate non-timed Gate A preflight negatives use the existing
instrumented `bench-local` command, one fresh process for each single-case
fixture, with warmup 10, iterations 30 and chunk context window 32768. The exact
case/history bindings are: medium-ascii 880, medium-cjk 728, medium-emoji 896,
medium-mixed-crlf 534, medium-many-small 1848, large-ascii 880, large-cjk 731,
large-emoji 938, large-mixed-crlf 554, large-many-small 1945,
near-threshold-ascii 1156 and xlarge-ascii 909. Each must return exit 1, empty
stdout, no output report and exact normalized error
`error: {case}: non-timed preflight failed: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit`.
Each record contains a status, reason, case, history, executable identity,
argv/hash, exit/error, executable/stdout/stderr/receipt hashes and no-output
confirmation. Timed observations are empty and statistics are null: no typed
skip or numeric observation is fabricated. These twelve records are Gate A
availability evidence only. They are not timing matrix rows and are never
repeated in Gate B. A changed error, executable, exit or output report fails
closed. The twelve unavailable cases are absent from the warm matrix.

Each measured warm process performs warmup 10 then 30 iterations. Only
`stats.p50_ms` from prepared numeric 30-observation producer records becomes an
outer observation. Expected skips remain nonnumeric with null statistics and
are limited to the existing `no_compaction_candidates` or
`insufficient_compaction_reduction` outcomes; the R12 repair invokes no real
warm process. The universal numeric kernels are:

- `estimate_tokens_all_messages`
- `estimate_context_tokens_full_scan_no_usable_usage`
- `estimate_context_tokens_with_usage_short_circuit`
- `find_cut_point`
- `prepare_compaction`
- `convert_to_llm`
- `serialize_conversation`
- `summary_chunk_slicing_fixture_window`
- `summary_chunk_slicing_small_window`
- `session_manager_build_session_context`
- `session_manager_get_branch`

Exactly 30 outer values are required for each of the eleven universal kernels.
Outer p95 uses nearest rank, `ordered[ceil(0.95 * n) - 1]`, for both universal
and prepared-only optional medians. This preserves the 30-value universal p95
while using the correct index for smaller prepared Jev counts.
`jev_prepare_context` is aggregated only over prepared, finite-stat processes;
one or more expected skips do not fabricate or replace observations. If all
processes expected-skip, the result records that status and no numeric Jev
comparison. Typed allocation `count`/`bytes_requested` and warm `stddev_ms`
may be finite zero where appropriate; other numeric requirements remain strict.
Inner samples do not become independent outer observations.

A future A/B function uses 30 seeded balanced randomized pairs (15 AB, 15 BA),
one fresh process per arm. It calls the retained independent R3 paired stats
implementation with outer medians, log(candidate/baseline), and deterministic
95% paired bootstrap over 10,000 resamples. Universal metrics use all 30
balanced pairs. Optional Jev comparisons use only pairs where both arms have
prepared numeric observations. If that matched subset is not AB/BA balanced,
the result states that no supported CI is available. Expected-skip observations
never become numeric zeros. This function is not executable as a baseline
comparator and creates no current speedup claim.

## Affinity, telemetry and limitations

Production-like children alone use mask `0x555`. Direct masks are observed. The
external wall path positively identifies its Python wrapper, then observes its
spawned production executable grandchild rather than comparing the wrapper to
that executable or assuming wrapper affinity proves production affinity. Every output-reported
production PID must match the observed direct production child or nested wall
grandchild, the observed `0x555` mask and the expected running image. Mismatched
or fabricated integer IDs are rejected before aggregation. Agent/system
affinity and the active power scheme are never modified.

Before/after telemetry has fixed fields for total CPU, frequency, queue,
per-logical-processor load, available memory and observed active power scheme.
A value is either real or null with an explicit unavailable reason. Null never
becomes zero. `launch_state=fresh-process`. Page cache and thermal state remain
`unobserved` unless genuinely observed. Warm means process-warm after ten
warmups, not page-cache-warm. PMU cycles, retired instructions, branch/cache
counters, live heap bytes and high-water memory are unavailable unless observed;
the driver does not estimate them.

Synthetic unit pass, Gate A correctness and Gate B raw-data completion are
separate states. None alone establishes a performance decision. Later hotspot
and optimization work needs new gates, full Gate A for each candidate, exact
same-lane A/B evidence and an honest performance threshold. No production
optimization is part of this change.

## Synthetic controls

From the new worktree, after a synthetic-only lock:

```text
python -m unittest discover -s scripts -p test_compaction_baseline_campaign.py -v
```

Set `PYTHONDONTWRITEBYTECODE=1` to preserve the exact four-path source delta.
Tests cover valid and invalid stages, source/build/lane/feature identities,
malformed/nonfinite/truncated JSON, output/PID collisions, stale hashes,
telemetry, Gate B authorization, wrong summary N, all-21 coverage, full-section
parity, provider/local separation, allocation unavailability, fresh process and
warm aggregation, prepared/skip/all-skip Jev handling and nearest-rank p95 for
2, 10 and 30 prepared values, supported matched pairs,
zero-valued allocation/stddev and PID-to-observation correlation. End-to-end
injected controls consume all 21 actual `writer.json` paths and prove no
aggregate is written after invalid or missing output. Fixed command-catalog
controls include focused99/charcount15 argv, unknown executables/options and
telemetry probes; no arbitrary plan adapter is accepted. The fixed Python
`validate --report ... --lane ... --out ...` route requires an existing report
inside the artifact root, a supported lane and an artifact-local output. A
synthetic wrong-lane input reaches the retained R4 `lane-identity` stderr and
nonzero process receipt instead of being blocked by the R5 command catalog;
unsupported lanes and escaping report paths are refused before launch. Live-Git and fresh
executor/protected-hash failures are synthetic-negative controls. Direct-binary
controls cover frozen provenance, wrong hashes/argv, forbidden Cargo/manifest
execution, missing isolation and protected profile-marker mutation. The
protected collision controls prove an unchanged existing output and continued
rejection of unregistered pre-existing outputs. Repair-r4 derives the actual
compiled collision contract from frozen Rust writer/main expressions without
running a compiled child. Durations are synthetic and non-decision. The controls do not run real campaign commands.


## Real campaign command grammar (not run in this change)

After the appropriate root authorization, the supported entry point is:

```text
python scripts/compaction_baseline_campaign.py execute-campaign --plan PLAN.json
```

For Gate B it additionally requires `--authorization gate-b/authorization.json`,
`--review gate-a/independent-review.json` and
`--gate-a-summary gate-a/summary.json`. The plan binds the source receipt,
fixture/toolchain/script/binary/run-matrix hashes, target roots, all-21 expected
summary counts and case order, ordered controls, focused commands and exact
artifact root. Gate A uses a distinct root approval with scope
`real-gate-a-after-independent-r5-source-control-review`. Executed R4 negative
controls cover seven accepted lanes/outcomes (including unavailable allocation,
provider failure/cancel and external wall), seven process rejections, four
truncated/nonfinite reader errors, a non-array schema rejection, wrong-lane
validation and an unchanged-output collision. The R14 mapping keeps truncated
and nonfinite input at `malformed-json`; the exact `b"[]"` reader now expects
`[schema] compaction-r4-lane-v1 object required` as its intended stage. The
surgically injected fake controls retain exact input/stderr bytes, argv,
exit 2, empty stdout and no reader report, and reject wrong labels/errors,
exits, reports and argv. R14 performs no real parser, compiled control, Gate
A/B, timing, build, archive, provider/network or delivery. For the collision,
the existing output is first created inside that negative control's own valid
run root and registered in the protected fixture class. Stock process layout
permits only that protected pre-existing file, never an arbitrary extra file or
outside output. The actual R4 no-clobber CLI must return exit 1 and exactly one stderr line
`error: create_new <exact protected output path>: <nonempty OS error>`. This is
derived from the unchanged R4 `write_new_json` and `main` Rust expressions; the
nonexistent `output-collision` CLI literal is not a substitute. The full
protected file identity and bytes must remain unchanged. There must be no new
parsed report, no dependent statistics, and exactly one attempt without retry
or replacement. Synthetic stderr/effect controls reject wrong prefixes, paths,
empty error text, exit 0, changed output, extra lines, retries and statistics.
The Python collision reader is only a non-compiled surrogate (FileExistsError);
it is not a claim that the later compiled R4 stage already passed. Gate B uses the
review/authorization summary hashes. The driver does not offer an arbitrary
executable or provider option.

The real Windows adapter reads telemetry through a structured read-only CIM JSON
probe plus `powercfg.exe /getactivescheme`; it does not alter power settings. It
observes descendants with read-only Windows image/affinity queries. Each real
run root must not exist before launch. All files below it are retained and
hashed in the process inventory. These native paths are not exercised by the
synthetic lock.
