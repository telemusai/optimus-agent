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
- `consume_embeddings`: full retained-prefix hash versus capture, selected-row
  hash versus the freshly copied target/draft embedding slice. Hashes distinguish
  mutated stored rows from a bad copy, without logging tensor contents.

These diagnostics never inspect keys, headers, principals, state handles or
receipt objects. **Token IDs are reversible content**, and decoded source text
can contain private data. Run only authorized synthetic prompts without secrets
(including conversation history), restrict log access/retention, then disable.
Every hook is exception-isolated, including logger/decoder failures. Size overflow
emits a fixed size-limit record. Chunk and KV-value lines are per-rank; other
lines are rank zero only. No diagnostic collectives or wire fields are added. Tensor hashing copies bounded rows to CPU and can change
timing; it does not change request state, alignment, positions or inference policy.

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
required; invalid indices or unsupported layouts skip before indexing. HIP
layouts are not interpreted as CUDA layouts.

`changed=true` relative to prefill indicates mutation of observed KV bytes;
`first_changed_slot` identifies the first changed sampled physical slot in prefix
order. Unchanged hashes with wrong text point toward fidelity/alignment but do
not prove it: unsampled slots, missing rank logs and transient mutations between
observations remain invisible. Synchronization can perturb or mask timing-sensitive
behavior. Never treat a sampled/skipped observation as whole-pool proof.

Staging note: `deploy/v3/live-instrumentation/stage_build.py` pins
`PLUGIN_FILE_COUNT=36`; it was already stale on main (37 modules), and this PR
has 38. The live staging assembler refuses this count. The full native package
build enumerates modules instead. No deploy files are changed here; correct the
pin separately before using that assembler.
