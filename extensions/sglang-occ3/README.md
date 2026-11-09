# occ3-bridge: experimental SGLang connector for OCC-3

This package implements the SGLang side of the generic Optimus OCC-3 bridge
in `crates/pi-coding-agent/src/core/occ/occ3`. The generic bridge does not
require other inference engines to use SGLang's implementation.

## Status and proof boundary

The native hidden-state path is experimental and disabled by default.
Source overlays, worker/control code, native chat adapters, deployment
safety code and offline tests are under development. They are not evidence
of a successful model run, GPU-state transfer, semantic benefit, production
rollback or deployment readiness.

Salted prompt replay and radix-cache hits are a separate cache-reuse
experiment, not proof of native latent communication. A `cached_tokens`
count, HTTP 200, health result or imported plugin cannot establish that a
receiver used a sender's hidden representation. Legacy native-state
shortcuts must not be used to bypass the current admission gates.

The first isolated native integration uses the process-local experimental
`occ3_handoff` tool and real family-message authorization. The ordinary
Python `agent_message.send` path does not yet carry authenticated native
source-tool provenance. It must not be advertised as a completed native
latent handoff path.

## Intended native path

1. Authorize the source against the existing authenticated broker and an
   explicit server-owned session enrollment. Client session labels alone
   are not authorization.
2. Capture accepted target-model final-layer pre-normalization predictor
   states during the original native generation. Do not generate or
   re-prefill a second text carrier to simulate source state.
3. Retain bounded rank-local GPU state behind an owner-only opaque handle.
   Bind it to the exact model, runtime incarnation, source request and
   completed output. Do not export tensors through control RPC or logs.
4. Publish an authorized one-shot message through the native family
   controller. Preserve uncertainty after a lost reply; do not replay an
   admitted source or fall back to plaintext.
5. Align the retained state and inject a continuous prefix into one native
   receiver inference. Keep private KV pages out of shared-cache reuse and
   fence submitted GPU work before releasing owned pages.
6. Verify matching native terminal output and backend evidence before
   making a source lease or receiver completion available to tools.

Actual tensor lineage, all-rank synchronization, abort/retirement,
private-page reuse and semantic behavior still need controlled validation
on the exact target image and hardware. Offline control-flow doubles and
optional skipped Torch/CUDA tests do not close those requirements.

## Installation and activation boundary

Ordinary entry-point discovery uses `sglang.srt.plugins`, but discovery
alone does not install or enable the native hidden-state path. The native
implementation requires reviewed raw-hash-pinned source overlays for the
exact original SGLang source snapshot, fixed packaged profile/source
artifacts, explicit startup selection, and installed-source/runtime checks.

`deploy.v3.native_package` verifies or stages these artifacts in an
exclusive disposable tree. Its default startup selection is false.
Interrupted staging requires discarding that tree, not repairing a live
server. Generated `native_profile.json` and `native_sources.json` are
package data; `_native_profile_digest.py` is generated Python metadata.
Neither their presence nor a successful package build grants live approval.

Do not change server environment, weights, API keys, launch options,
reasoning effort, token budgets, adaptive MTP or ordinary cache settings to
make an experimental test pass. The optional hidden-observation dump stays
disabled unless separately approved. It is not the native transfer path.

The V3 lifecycle controller remains fail-closed while its native validator
is unavailable. Real disposable recovery and routing tests are separate
prerequisites. The historical deployment scripts and an endpoint health
check are not substitutes for those gates. No automated production swap
is authorized by this README.

## Live known-token label anchor

The attested profile pins **policy, not a mutable scalar**:

```json
{"label_anchor_alpha_allowed": [0.0, 0.3, 0.5, 0.7, 1.0],
 "label_anchor_alpha_default": 0.3, "anchor_alpha": 0.0}
```

Both calibrated candidate files carry this policy:
`deploy/native-build-inputs/native-profile-input.json.v3` and
`deploy/v3/plan-templates/occ3-icalib-persistent-candidate.json`. Allowed values
must be a nonempty, duplicate-free subset of the five coefficients above; the
default must belong to it. The default is the attested hybrid 0.3 so a client
that omits the request field is anchored at the byte-exact coefficient instead
of silently dropping to 0.0 - the exact garbling class the PR #265 client
default (env > file > client-default 0.3) fixed client-side. An explicit 0.0
request still selects zero. Old profiles without these fields keep the fixed
`label_anchor_alpha` setting (missing means zero). A conflicting legacy scalar
and new policy is rejected. If nearest-vocabulary `anchor_alpha` is nonzero,
a request selecting nonzero label alpha is rejected even if listed in the policy.

### Sweep a running server and client

The sender reads its selection **fresh for every source capture**, in this order:

1. `OCC3_LABEL_ANCHOR_ALPHA` when set; an invalid or empty value is a local error.
2. Otherwise the file selected by `OCC3_LABEL_ANCHOR_ALPHA_FILE`, or by default
   `<optimus config dir>/occ3_label_anchor_alpha` (the client's existing
   `config::get_agent_dir()` resolver, including its agent-directory override).
   The file contains one allowed
   decimal, under 32 bytes including surrounding whitespace; the value is trimmed.
3. A missing file omits the request field, so the server chooses its attested default.

A malformed, oversized or unreadable file fails locally before enrollment or
inference; it never silently falls back. Environment wins over file. Unset the
scalar environment override before starting a file-driven sweep. Environment
variables belong to the generating process: changing an export in another shell
does not change an already-running daemon. Editing its selected file **does**
affect the next capture, with no restart of either server or client:

```sh
# Use the file selected by the running client's configuration.
printf '0.3\n' > "$ALPHA_FILE"
# Run one synthetic OCC-3 handoff drill against the same running server.
printf '0.5\n' > "$ALPHA_FILE"
# Run the next drill; no rebuild/restart is required between values.
```

For concurrent editors use a temporary sibling file and an atomic rename so a
capture does not read a partial write. The existing rotating debug log (`config::get_debug_log_path()`) records
`occ3 capture label_anchor_alpha=<number|profile-default> source=<env|file|default>`,
never file contents or paths. Set server `OCC3_DIAG_TOKEN_TRACE=1` for an authorized synthetic drill;
the existing `alignment` event reports the **used** coefficient plus each final
prefix row's cosine to `E_in[accepted_token]`. See stage semantics below to compare
pre-label `aligned`, post-label `label_anchored`, and `final` evidence. Restore
`0.0` to disable the blend. A selected value outside the attested set is rejected
with a fixed server error; expanding that set is an attestation change, not a live knob.

### Binding, computation and compatibility

The optional numeric `occ3_capture.label_anchor_alpha` travels through the real
Rust `SourceCaptureIntent`, Python `CaptureIntent`, and resolved
`CaptureWorkerDescriptor`. Omitted differs from explicit zero: explicit zero can
override a nonzero default, and its explicit request field changes the intent
digest relative to omission. The resolved value is validated on all ranks and
included in existing request/control and alignment-prepare consensus digests;
no collective is added. Each capture owns an immutable coefficient and stored
prefix. Later file edits cannot change an existing handoff or its consume.

On rank zero, after the solver and before the existing TP broadcast, rows become
`(1-a) * aligned + a * E_in[accepted_token]`, in float32 then cast to prefix dtype.
The labels were checked against the retained capture ring. The existing nearest
vocabulary anchor runs afterward, unchanged. At zero the blend performs no
arithmetic; at one finite rows equal the known-token embedding. Intermediate
values retain a residual but do not guarantee exact decoding or contextual value.
Existing embeddings and bounded row scratch fit the solver workspace; no new
vocabulary-sized tensor or non-diagnostic hot-path synchronization is added.

The actual nonzero coefficient enters the capture commit and source receipt;
zero remains omitted from public receipts and omitted requests retain their old
wire bytes. Updated Rust clients default old receipts to zero and check explicitly
requested values against receipts. Install the updated client before selecting
nonzero alpha: older strict clients reject that receipt field. This optional
source-capture metadata does not change the daemon protocol. Legacy zero-profile
binding digests remain unchanged; allowed/default policy is bound when present.

### One-time installation cost; no cost to change a permitted value

Installing this plugin/client version and its allowed/default policy requires the
normal attested build once. The supported Docker lane rebuilds the derived OCI
image's native-package layer and replaces/restarts the candidate container; it
reuses the preserved original SGLang base and weights, with no calibration refit.
Thereafter, **changing the file or a per-process environment selection needs no
image/package rebuild or server/client restart**. Changing the allowed/default
policy itself still needs that attested rebuild. In-place package hot-patching
is not a supported deployment path.

In a prepared clean extension context, select the calibrated policy explicitly
(the Dockerfile reads the basename without `.v3`):

```sh
cp "$CONTEXT/deploy/native-build-inputs/native-profile-input.json.v3" \
   "$CONTEXT/deploy/native-build-inputs/native-profile-input.json"
docker build --pull=false --network=none \
  --build-arg BASE_IMAGE="$BASE_IMAGE_TAG" \
  --build-arg OCC3_EXPERIMENTAL_STARTUP=true \
  -f "$CONTEXT/deploy/Dockerfile" -t "$CANDIDATE_TAG" "$CONTEXT"
```

`BASE_IMAGE_TAG` must identify the verified preserved original base, not an
already-patched candidate. Include the reviewed source baseline/manifests and
actual calibration path/digest. `deploy.v3.native_image_build` calls
`native_package` to generate `native_profile.json`, `native_sources.json`, and
`_native_profile_digest.py` (`PROFILE_SHA256` and `SOURCE_SHA256`). These outputs
are build artifacts, not tracked Git files. Policy bytes change `PROFILE_SHA256`;
updated plugin source changes its attestation/SOURCE digest. Per-handoff selection
changes neither attestation. Update the candidate plan's image/static-input/runtime
pins for the one-time install. Historical fixed zero-profile SHA pins stay unchanged.

## Resource limits

Current bounded response handling does not prove compatibility with the
actual native default of 131072 output tokens or its 262144-token ceiling.
Do not substitute smaller fixture budgets, silently clamp requested tokens
or infer capacity from a short timeout estimate. Output-byte, context,
GPU-memory, rank and deadline limits need independent verification. A
failed or uncertain operation yields no verified lease or tool dispatch.

## Native-turn transport and diagnostics

Guarded OCC-3 client requests offer `keepalive-v1` by default, for both
source capture and receiver inference. A server that accepts sends SSE comment
keepalives while native generation runs; the client verifies the deferred
receipt before accepting completion. Servers that do not accept retain the
buffered response contract. Set client `OCC3_KEEPALIVE_STREAM=0`, `false`, or
`off`, or an empty `OCC3_KEEPALIVE_STREAM=` to disable the offer; unset, `1`, `true`, or `on` enables it. This does
not lower reasoning effort or change token budgets.

A bounded admission-busy error as the first data event of an accepted deferred
stream closes the lease as proven non-use, with the same busy-code allowlist and
retry timing as HTTP 503. Source capture then requires a fresh capture attempt;
the already-started stream is not replayed. Unknown errors, errors after data,
and receiver admission timeouts retain the uncertain outcome. Keepalive comments
do not count as data and continue during steady decode as well as stalls.

The bridge logs a one-line INFO `occ3-bridge: native turn summary` for each
native turn, including completion and interrupted turns. It reports timing,
inner-event and output-character counts, available token usage and finish
reason, transport mode, and channel/attempt identifiers. Text is off by default.
Treat `cancelled` as disconnect-class on ASGI servers below spec 2.4.

For temporary diagnosis on an **operator-owned server**, set server
`OCC3_DIAG_TURN_TEXT=<N>` to include the first and last N characters of visible
content and reasoning in that summary. The maximum is 4096 characters per
excerpt; zero or unset disables text. Newlines are escaped. The summary message
is capped at 16 KiB of ASCII text (excluding logger-added prefixes). If escaped
excerpts exceed that cap, their per-excerpt limit is repeatedly halved until
it fits, recorded as `text_truncated_log_cap`. Four full excerpts have room for
about 3900 ASCII characters each (so a requested 4096 becomes 2048); escaped
CJK and emoji need more space (4096 becomes 512 and 256 respectively for
uniform text). Metadata can reduce these effective maxima further. An oversized
metadata-only summary falls back to `summary_size_limit`. Diagnostic failures
disable collection and record a fixed `diag_error`
code, without exposing exception text or interrupting the turn. Inner-event
counts continue even after output parsing is disabled. A logger failure
is swallowed and cannot guarantee a summary line. This setting can
expose generated private content and reasoning in server logs. Enable it only
with authorization, restrict log access and retention, and disable it after
diagnosis. Request prompts, headers, keys, receipts, and authentication fields
are not collected by this diagnostic. It is not proof that transferred facts
were understood, and it does not change the attested SGLang source overlays.

### Token/position drill trace

Set **server** `OCC3_DIAG_TOKEN_TRACE=1` for a temporary synthetic-data drill.
Unset (or any other value) is off. No client change is needed: Rust source
capture enrolls the actual generation request; tokenization, accepted rows and
alignment live on the server. The plugin module is included in `_REQUIRED_PLUGIN`;
rebuild its attestation manifest when deploying. No SGLang overlay changes.

INFO `occ3-bridge: token trace` JSON lines (ASCII escaped, at most 16 KiB each):

- `capture`: first/last 64 selected token IDs, full-list SHA-256, accepted-output
  comparison and first mismatch (zero-based, null = equal), accepted span and
  absolute predictor positions (exclusive end, predictor offset -1), row count,
  and aligned-prefix byte hash. Source labels are retained locally, at most 128.
- `capture_text`: source excerpts decoded by the scheduler tokenizer when it is
  available (512 characters each); no tokenizer means IDs only.
- `consume_admit`: the same source labels/hash, placeholder holes and 64 prompt
  IDs on each side, full rewritten prompt row count, offset and private-cache flag.
  Prompt neighbors are IDs only, not decoded. Holes are **not** receiver text.
- `consume_chunk`: target prefill chunk boundaries (including zero-overlap
  chunks), mapped source labels, actual position IDs and first expected-position
  mismatch, computed DSA offset/count, actual `occ3_dsa_pin` vectors (up to
  128 entries each), rank and bounded KV slot IDs. Every TP rank records its
  slots. Draft chunks are labeled separately and show their one-row pin shift.
- `consume_settle`: saved injection slots and retraction count. The live request
  row has already been freed; this is not a terminal mapping comparison.
- `alignment`: at most 128 per-row cosines between the aligned prefix and its
  source-token embeddings, rounded to six decimals (nonfinite values are null).
  Both tensors are already present on rank zero; no vocabulary lookup is added.
- `consume_embeddings`: per-rank full retained-prefix hash, selected-row hash
  versus the freshly copied target/draft embedding slice, and rank. Capture
  comparison is null on ranks without a capture hash (currently all except zero).
  Hashes distinguish mutated stored rows from a bad copy without logging contents.

These diagnostics never inspect keys, headers, principals, state handles or
receipt objects. **Token IDs are reversible content**, and decoded source text
can contain private data. Run only authorized synthetic prompts without secrets
(including conversation history), restrict log access/retention, then disable.
Every hook is exception-isolated, including logger/decoder failures. Size overflow
emits a fixed size-limit record. Chunk, embedding and KV-value lines are per-rank;
other lines are rank zero only. No diagnostic collectives or wire fields are added. Tensor hashing copies bounded rows to CPU and can change
timing; it does not change request state, alignment, positions or inference policy.
Embedding hashes add three bounded copies per lane per rank: up to about 4.7 MB
per lane per rank when one chunk covers the full 128-row span. The draft lane
repeats these copies when its chunk overlaps the span.

Equal IDs/hashes do not establish semantic fidelity: the receiver attends to
continuous aligned embeddings, not the source IDs. KV slot IDs expose placement
and reuse across observed chunks. The value hashes below additionally detect
observed KV mutations on each traced rank. If all traces agree,
investigate alignment quality and downstream attention behavior rather than
treating source labels as text that the receiver necessarily saw.

## Offline development

Run tests through the extension's documented project Python environment:

```text
python -m pytest tests/ -q
```

Read each test's proof boundary. Some tests execute exact pinned source
methods with explicit engine, allocator or transport doubles. Optional
installed-image, Torch, native Linux, CUDA and TP8 tests need separate
approval and may be skipped locally. Use no real provider credentials or
model calls for regression tests. The repository also requires
`bash scripts/check.sh`; a passing offline gate is not live clearance.

## License

MIT, same as the repository.


### Injected KV value diagnostics

With `OCC3_DIAG_TOKEN_TRACE=1`, `consume_kv_values` hashes each TP rank's
local target MLA bytes and DSA index key **and scale** bytes. These are lifecycle
observations, not hooks on shared decode batches (which have no `occ3_context`):

- `prefill_complete`: forward completion of the chunk containing the last injected
  row. Saves the pool and original physical injection slots for later reads.
- `consume_released`: final prefill result settle, after the result copy-done wait
  and an opt-in local `runtime.fence()`, before clearing the eager flag. This is
  before the next shared decode/target-verify, not a first-decode forward hook.
- `private_release`: inside the existing private-KV release fence, before the
  allocator releases KV or frees `req.kv.req_pool_idx`. This is the last safe
  observation, not a terminal-settle read of storage that may have been reused.

A request finishing during prefill does not leave the eager cohort and therefore
has no `consume_released` point. Aborted/incomplete injection may lack a baseline;
missing saved pools skip later reads. These traces never promise three lines for
every attempt. All reads use saved physical injection slots, not neighbouring
page rows. No engine storage is written and no diagnostic collectives are added.

Each point copies at most **8 MiB** GPU-to-CPU, plus existing slot-map reads.
For N selected slots the exact copy cost is N times the sum of all MLA row
widths plus 132 bytes per allocated DSA index layer. Index tensors are built once
per point. Selected bytes from independent layer allocations are gathered on-device,
concatenated, and transferred with **one `.cpu()`**. A single gather across all
layers would require stacking whole caches or a custom pointer-array kernel;
this diagnostic instead keeps per-allocation gathers to avoid whole-pool copies
and new kernel dependencies. Logs report `bytes_copied` and `blocking_copies=1`.
CPU SHA-256 retains only per-slot digests. The last 16 injected slots (or all slots
if shorter) are mandatory; remaining budget samples the earlier region evenly.
If the mandatory tail cannot fit, the point reports `layout_exceeds_budget`.
`sampled=true` limits conclusions to selected slots. No GPU work or Torch import
occurs through these hooks when disabled. The uncompressed CUDA DSA layout is
required; invalid indices or unsupported layouts log a `skipped` reason before
indexing. Bounds use each buffer's physical capacity, not logical `pool.size`:
the paged allocator reserves page zero, so the final allocated page has valid
slots at or above `pool.size`. HIP layouts are not interpreted as CUDA layouts.

`changed=true` relative to prefill indicates mutation of observed KV bytes;
`first_changed_slot` identifies the first changed sampled physical slot in prefix
order. Unchanged hashes with wrong text point toward fidelity/alignment but do
not prove it: unsampled slots, missing rank logs and transient mutations between
observations remain invisible. For a drill, confirm in `consume_chunk` that one
target chunk covers `prefix_offset .. prefix_offset+rows`: the baseline is the
end of the chunk containing the last injected row, so earlier wrong writes inside
that chunk (or across earlier chunks) can already be baked into the baseline.
A diagnostic local-fence failure emits `skipped="fence_failed"`, not an unchanged
comparison. Synchronization can perturb or mask timing-sensitive behavior. Never treat a sampled/skipped observation as whole-pool proof.

Staging note: `deploy/v3/live-instrumentation/stage_build.py` pins the current
`PLUGIN_FILE_COUNT=38`; an inventory regression checks this against the plugin
modules. The full native package build enumerates modules and regenerates hashes.

### Opt-in alignment neighbour evidence

`OCC3_DIAG_TOKEN_TRACE=1` also emits `alignment_row` for raw predictor (`raw`),
pre-label solver (`aligned`), post-label (`label_anchored`, only when enabled),
and post-nearest (`final`) rows on rank zero. Each line contains one of at most
128 rows, cosine offsets [-1, 0, +1] (null at edges), row/embedding norms, both
anchor coefficients and target norm, rounded to four decimals. Zero-norm cosine
is undefined (null). `alignment_cast` remains before the label blend and measures
only the solver-to-buffer cast, not the anchor's effect.

At zero label alpha the existing `aligned` points remain unchanged on all ranks.
When label anchoring is enabled, `aligned` reads the original solver rows before
the blend **on rank zero only**, because no other rank owns these rows. Its
`alignment_nearest_shard` result is only rank zero's local nearest token; it
cannot establish global pre-label nearest identity. `label_anchored` and `final`
nearest scans run on all ranks after the existing broadcast. No new collective
is added. The scans use only each rank's resident embedding weight,
excluding padding via StateBridge shard ranges. It never calls embedding
forward or a collective. Vocabulary tiles are 256 rows (at most 32 MiB per
float tile at hidden width 32768); scores are at most 128 x 256. It logs local
top-1, not global rank. Merge matching capture rows offline across all ranks
by highest cosine, then smallest token ID for ties. Rounded near-ties are
ambiguous; do not infer global top-1 from one shard. Missing ranks invalidate
the global conclusion. Run one isolated capture at a time to associate rows;
no request identities are emitted. Failures remain isolated from serving.

Decision rule: a clearly higher neighbour-offset score on misread rows, or
a global nearest token equal to a neighbour, points to a capture alignment
bug. A token lost only at the solver or anchoring stage points to that
transform. Raw hidden rows far from all reasonable targets and a different
nearest token support a latent-channel fidelity limit. These are diagnostic
hypotheses, not proof: predictor hidden states and input embeddings need not
share geometry, and reading a continuous prefix is not nearest-token decoding.
The existing live evidence shows intact delivery and repeatably weak tokens,
but lacks raw/solver stage comparisons; no causal conclusion is yet justified.
All added tensor work is disabled unless the existing trace flag is exactly 1.

Precision diagnostics emit `alignment_cast` at the solver-copy boundary
(hidden/solver/buffer dtypes, float32 cosine and max absolute difference).
`cast_observed` is true only when solver and buffer dtypes differ. All current
solvers already return hidden.dtype, so this is normally a same-dtype copy
with cosine 1 and difference 0, **not a measurement of solver output rounding**.
The fp32-to-bf16 cast occurs inside apply_map/the solver and is unobserved here.
For finite normal values, a single bf16 rounding has cosine loss of order
2^-16 or smaller: substantial loss from that one rounding can be excluded
analytically, not by the copy trace. This does not exclude accumulated solver
numerical error, underflow, overflow, or earlier precision loss.

`alignment_precision` compares each final row and plain token-embedding
baseline with their own row-scaled float8_e4m3fn round trips. Each row is scaled
by 448/amax before casting and unscaled afterwards (divide first to avoid
scale overflow); zero/nonfinite rows have null cosine. Unsupported conversions
have an explicit skip reason. FP32 end-to-end replay remains skipped because
rerunning the solver/global anchor adds expensive work and collectives.

Precision decision rule: FP8 cosines around 0.999+ on misread rows, matching
the plain-embedding baseline, rule out substantial loss at this proxy boundary.
Use cast cosine as evidence only when `cast_observed` is true. Clearly worse
loss on misread rows makes precision a contributor worth testing. This proxy
is not actual KV quantization (which transforms/scales states); it cannot rule
out all numerical causes or prove real KV-storage fidelity by itself.

#### Offset conventions and raw-state logit lens

Offset `k` in neighbour cosines means `cos(values[i], E[tok[i+k]])`.
Capture `predictor_offset=-1` means raw row `i` is the hidden state at the
sequence position that **produced** `tok[i]`, not the state after consuming
that token. Normal raw-state/input-embedding cosine need not peak at offset
zero: GLM-5's input embeddings and LM head are untied. A real indexing defect
needs systematic shifted label likelihood in the model readout, not just an
isolated raw embedding neighbour win.

`alignment_logit_lens_shard` reads the verified candidate model objects
`runner.model.model.norm` (RMSNorm) and `runner.model.lm_head` (ParallelLMHead),
from GlmMoeDsaForCausalLM's DeepseekV2 base. It evaluates RMSNorm's native
formula and the unquantized head weight in bounded tiles, never calling
model/norm/head forwards or LogitsProcessor. Unsupported quantized head,
norm override, head bias, or logit transforms explicitly skip. The live
LogitsProcessor readout ignores head bias, so the lens does not add it. This is a float32
readout approximation with the normal norm-output dtype cast, not a replay
of fused inference kernels or sampling penalties/temperature. The captured
residual is the rounded fused add+norm input (the post-add residual), not an
unrounded reconstruction of the separate addends. Residual rounding therefore
also makes this an approximation of the served readout, not a bit-exact replay.

Each rank logs local top-8 IDs/logits, label logits at offsets [-1,0,+1]
(null for edges or labels owned elsewhere), actual shard ranges/vocab start,
and local logsumexp. Merge local top-8 by descending logits, then **truncate
the merged list to the global top-8**. Report exact global label rank only
for positions 1-8 in that truncated list; otherwise report `>8`. Positions
beyond eight in the untruncated union are not exact ranks: unreported tokens
from other shards may outrank them. Exact full-vocabulary rank is unavailable
without exchanging the owning rank's label logit; this trace adds no such
collective. Combine all shard logsumexp values with logaddexp to compute
untempered label probabilities. Missing ranks, rounded ties, or unsupported
readouts leave the conclusion unresolved.
The profile method ID and live anchor alpha accompany the diagnostic rows.

Run the sender greedy for this drill, or use p(label) from merged logsumexp
rather than top-1 rank. Under sampling the accepted label need not be top-1;
that alone is not evidence of an indexing defect.

| Raw logit-lens result | Interpretation |
| --- | --- |
| Label top-1/high probability at producer position -1, but aligned row misses | Map/solver loses information. This is label offset 0 in our log. |
| Label top-1 only at sequence position 0 or +1 instead of producer -1 | Indexing hypothesis. Compare adjacent raw rows; do not confuse sequence-position offsets with label offsets. |
| Label low probability, misread tokens are runner-ups | Distribution/fidelity hypothesis: hidden state encodes alternatives. |
| Label top-1 and final global nearest is label, but receiver reads incorrectly | Investigate injection/attention rather than the final map+anchor output. |
| Pre-label `aligned` cosine is weak but `label_anchored` improves | The label anchor changes the row as intended; receiver fidelity still requires a separate drill. |
| Nonzero label alpha and only rank-zero `aligned` nearest available | Do not infer global pre-label nearest identity; compare offsets and cosine, then global post-label nearest across ranks. |

All new points now pack device-side arrays and perform one small host copy
per call immediately before emission. No per-row scalar reads or per-tile
host validity checks occur. One bounded synchronous packed copy per point inside align() is an accepted
scope tradeoff for this opt-in trace; there is no asynchronous staging.
Copies include points before anchoring rather than being deferred past the
completion event, so opt-in trace overhead can still delay other ranks. Vocabulary validity stays on
device through the full loop. Logit-lens tiles share the 256-row bound.


### Diagnosing parallel handoff failures

For one controlled two-handoff repro, enable `OCC3_DIAG_TOKEN_TRACE=1`
on the server. Grep server logs for `occ3-bridge: native failure` and
`occ3-bridge: admission` and `occ3-bridge: retention`; grep client logs for `native_binding_rejected`
and `native_binding_invalid_site`. Disable the trace after the repro.

Failure lines contain an allowlisted `code`, optional source-defined `detail`,
HTTP `status`, `phase`, `dispatched`, and at most three origin frames
(module basename, function, line only). Worker preflight/control failures are
logged before their error is normalized onto the control wire. Public errors
and control replies are unchanged. No exception message, locals, request ID,
handle, principal, receipt, headers, keys or prompt text is added.

Admission lines report `operation` and `outcome` (`entry`, `error`, `exit`).
`exit` means the call exited, not that admission succeeded: look for its
preceding `error`. Registry snapshots include `active`, `max_active`, `total`,
`max_records`, `inflight`, `reservation`, `leased`, `capturing`, `revoked`,
`gpu_freed`, `charged_bytes`, `max_live_bytes`, `retained_rows`,
`admitted_capture`, and `admitted_consume`. The active slot is a registry state
predicate, not a separate semaphore. Frontend snapshots show `active`,
`max_active`, `terminal_active`, and `terminal_attempts`; backend snapshots
show `attempts` and the real control client's `pending`, `uncertain`,
`operations`, and `max_pending` counts. Worker snapshots also
show `poisoned` and `commands`. DEBUG logging adds `admission lock`
acquire/wait/release ordering on capture reservation. This lock trace reads
the same opt-in at process startup and uses the original lock directly when
opt-in or DEBUG logging is off.
`retention` lines run at quarantine, late settlement, terminal resolution,
and release-gate checks even when the gate refuses release. They contain
`pending`, `negative`, `positive`, `quarantined`, `terminal_tasks`, charge
counts, settle/failure/release/completion/cancel flags, and fixed per-kind
`abort`, `retire`, `close`, `emission`, `native`, and `proof` pending/latched
counts. Initial parsing/admission refusals and cleanup/iterator exceptions
also produce payload-free failure lines, including synchronous terminal
abort, retire, and iterator-close failures before a task exists. Snapshots are observational, not a
new reservation or a proof of GPU release.

`/occ3/v1/status.parallel_handoff_diagnostics` contains `last_error_code`,
`last_error_count`, `error_count` (saturating), `first_failure_unix_seconds`, and `admission`.
The latter contains bounded-key `counts`, `logged`, and `last` snapshots when
the flag is on, otherwise `{}`. These are **process-local**: frontend status
cannot expose rank-local worker records without changing the control protocol.
Use worker logs for those records. First failure time is not a claim that the
runtime is stuck; compare repeated failures and active-slot snapshots. Existing
metrics retain the prefix-layout counters. Lines stay below 16 KiB and logger
failures are ignored. No GPU sync or collective is added.

Client `native_binding_rejected` events identify a fixed `stage` and `reason`
for enrollment, capture, publication intent, receiver and retirement checks.
`native_binding_invalid_site` additionally reports the source line for other
control validation sites, without binding values. User-facing error text is
unchanged. The diagnostic does not repair a leaked slot or retry a handoff.

The repro build must regenerate `native_sources.json`: the attestation hashes
cover the modified plugin files. The deploy build does this; reusing an old
manifest causes `source_digest_mismatch` rather than a valid diagnostic run.
Failure counters count each exception object once across propagation boundaries;
a distinct normalized exception remains a distinct failure. Cancellation does
not count as an unknown failure in admission probes. Typed attachment refusals
use INFO; `LatentError` codes, untyped exceptions and `native_completion_failed`
use ERROR. Worker admission rows include numeric `rank`, and frontend snapshots
include replay `fences`. Client control rejection includes only numeric HTTP
`status`, never the response body. Poison-transition logs fire once per runtime
latch, and later worker unavailable gates include code and origin before wire
normalization.
