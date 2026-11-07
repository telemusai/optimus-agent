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
  mismatch, DSA offset/count, and bounded KV slot IDs. Draft overlapping chunks
  are labeled separately and show their one-row pin shift.
- `consume_settle`: terminal request-to-KV slot mapping versus injected slots,
  with first mismatch and retraction count (if the scheduler exposes its pool).
- `consume_embeddings`: full retained-prefix hash versus capture, selected-row
  hash versus the freshly copied target/draft embedding slice. Hashes distinguish
  mutated stored rows from a bad copy, without logging tensor contents.

These diagnostics never inspect keys, headers, principals, state handles or
receipt objects. **Token IDs are reversible content**, and decoded source text
can contain private data. Run only authorized synthetic prompts without secrets
(including conversation history), restrict log access/retention, then disable.
Every hook is exception-isolated, including logger/decoder failures. Size overflow
emits a fixed size-limit record. Rank zero alone emits; no diagnostic collectives
or wire fields are added. Tensor hashing copies bounded rows to CPU and can change
timing; it does not change request state, alignment, positions or inference policy.

Equal IDs/hashes do not establish semantic fidelity: the receiver attends to
continuous aligned embeddings, not the source IDs. KV slot IDs expose placement
and reuse across observed chunks, **not** later KV content overwrites or other
ranks' memory. If all traces agree, investigate alignment quality and downstream
attention/KV behavior with separate runtime instrumentation rather than treating
source labels as text that the receiver necessarily saw.

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
