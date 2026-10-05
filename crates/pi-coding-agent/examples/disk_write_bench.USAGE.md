# disk_write_bench — disk-write benchmark harness (B1)

Standalone measurement binary for the REAL durable-write code paths behind
coding-file writes, session persistence, the edit tool, and the semantic-edge
ledger. Built to produce BEFORE numbers for the diskio optimization work:
before/after identity is proven by the parity digest gate, and before/after
numbers must come from binaries with different sha256s (record git describe +
binary sha with every matrix — the harness embeds both in every JSON report).

- Example source: `crates/pi-coding-agent/examples/disk_write_bench.rs` (new
  file; no production file modified, no new dependency).
- Measured through public entry points only:
  - `pi_coding_agent::utils::atomic_file::write_file_atomic_sync` /
    `write_file_atomic` (temp file beside destination, short-write-safe loop,
    optional fsync, Windows rename retry, optional dir fsync);
  - `pi_coding_agent::core::session_manager::SessionManager` JSONL persistence
    (first persisting append rewrites the whole file through the session
    manager's atomic stand-in; later turns append line-by-line);
  - `pi_coding_agent::core::tools::edit::execute_edit` with the default
    `LocalEditOperations` write side (`std::fs::write`), through a counting
    wrapper that only delegates to the real operations;
  - `pi_coding_agent::core::event_log::EventLog::append_sync` — the
    semantic-edge ledger pattern used per model request (one event per append,
    `durable=false`, `EventLogOptions::default()`, ledger file
    `artifacts/semantic-edges.jsonl`), plus one `durable=true` variant.

## Build

```
cd <worktree root>
export RUSTUP_HOME=C:\Users\openclawuser\optimus-rust-toolchain\rustup
export CARGO_HOME=C:\Users\openclawuser\optimus-rust-toolchain\cargo
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR=<project>\build\bench        # per-worker target dir
export CARGO_BUILD_JOBS=4
cargo build --release --example disk_write_bench
```

Binary: `%CARGO_TARGET_DIR%\release\examples\disk_write_bench.exe`.

## Subcommands

```
disk_write_bench matrix --out <json> [--runs R] [--warmup W] [--alloc-runs A] [--quick] [--keep-tmp]
disk_write_bench point  --out <json> --name <point> [--runs R] [--warmup W] [--alloc-runs A] [--keep-tmp]
disk_write_bench list
disk_write_bench help
```

Defaults: `runs=3 warmup=1 alloc-runs=1` (4 passes per point: 1 warmup +
3 measured + 1 alloc-counted). `--quick` runs a reduced 22-point smoke
matrix. `--keep-tmp` keeps the harness tmp tree for inspection.

Tmp root: `<system temp>\optimus-diskio-bench` — wiped at start and exit.
Never touches anything outside it.

## Full matrix (91 points)

- `a_sync-s{4k,32k,256k,2m}-n{1,20,200}-f{0,1}-d{0,1}` — sync atomic
  writes; sizes x file counts x (fsync, fsync_dir) variants. `(f,d)` covers
  all four combinations; real callers pass (0,0), (1,0), (1,1).
- `a_async-...` — async twin; (0,0) default and (1,1) full durability.
- `a_rw-s32k-r200-f?-d?` — 200 sequential rewrites of ONE destination
  (rename-over-existing storm) per option variant.
- `sess-t{10,100,1000}` — real SessionManager JSONL sessions, N turns of
  (user + assistant) appends. Includes an untimed `create_dir_all` control
  (100 calls on the existing sessions dir) isolating the per-append cost
  component that `SessionManager::persist` issues before every line append.
- `edit-s{4k,32k,256k}-e{1,5}` — real `execute_edit` tool calls (20 per
  pass, each from an identical reset base file) with exact read/write/access
  counts through the `EditOperations` seam.
- `evlog-s{4k,60k,500k}-d0` + `evlog-s60k-d1` — `EventLog::append_sync`
  semantic-edge ledger appends (20 single-event appends per pass, alternating
  RequestStarted/RequestFinished). The ledger is seeded through the real
  `append_sync` (batched 64 events per call, untimed) to the target size;
  every measured append includes the full `repair_tail_sync` probe (metadata
  check + read/write open + tail byte check; the torn-tail double-read path
  only triggers on a torn tail, which clean traffic never produces).

## Determinism and the parity gate

Every written byte comes from a fixed-seed generator: every run writes the
SAME bytes. After every pass the harness:

1. re-digests every file it wrote (content + sorted path list) and compares
   against the generator-derived expected digest — exit 1 on any mismatch;
2. checks no temp file was left behind;
3. atomic scenarios: asserts `before_rename` fired exactly once per write
   (the production seam);
4. session scenarios: asserts exactly one `benchsession.jsonl`, the entry
   count, the type/role sequence, the parent chain, per-entry content
   equality with the generator, and the persist-notification count; digests a
   normalized form (timestamp/id/parentId/cwd placeholders) which must be
   identical across passes;
5. edit scenarios: asserts the final file equals the expected edit result
   byte-for-byte and the counting seam saw exactly 20 reads/writes/accesses.

Before/after identity = digest equality + byte-identical files. Run-to-run
digest stability across two full matrix invocations is the determinism proof.

## Metrics per point

p50/p95/p99/mean/min/max/stddev per write (ms) over all measured passes,
first-op (cold-destination) vs warm-op distributions, total wall, alloc count
and bytes per op (counting global allocator, gated), Windows process I/O
counters per op (`GetProcessIoCounters`: read/write/other operation and byte
counts; metadata operations such as create/rename/fsync are NOT included),
persist notifications per turn, and the digests above. Every JSON report
embeds git commit/describe/branch and the exe sha256.

## Windows quirks — keep the box quiet during timing runs

- Windows Defender real-time scanning amplifies small-file writes and the
  read-after-write open in the ledger append. Keep Defender/AV idle during
  matrix runs; if you control the host, exclude the harness tmp root
  (`%TEMP%\optimus-diskio-bench`) from real-time scanning BEFORE/AFTER the
  measurement campaign only — do not change exclusions between the BEFORE and
  AFTER runs (identical environment on both sides is the contract).
- File locking: the atomic path retries renames on transient access-denied;
  close any program that might hold the tmp tree open (indexers, backup
  agents, file explorers).
- No other disk-heavy work on the host during a matrix run (other agents'
  builds, cleanup jobs, backups). The full matrix takes ~18 minutes.
- First matrix run on a fresh boot is noisier (cache warm-up everywhere);
  prefer a second run after the machine has been idle for the reported pair.

## Documented limitations

- OS-level caches are NOT controlled beyond the fresh-empty-directory rule
  (each pass writes into a new empty subdirectory). The first op per pass is
  marked cold (destination did not exist); everything else may hit warm file
  system caches. This is inherent to user-space measurement without admin
  cache-flush privileges.
- Sequential writes only; `AtomicFileWriteCoordinator` concurrency is not
  covered.
- Process I/O counters are a proxy for syscall counts: read/write ops only,
  no metadata ops (open/rename/fsync are invisible to it).
- Alloc runs measure the whole pass (including verification digests in the
  edit scenario is avoided: alloc passes verify once at the end), so per-op
  numbers are pass-level means.
- Wall-clock numbers are noisy on a shared box; treat alloc counts, byte
  counts, before_rename/persist counts, and digests as the deterministic
  signal and timing distributions as secondary evidence.

## Files

- `crates/pi-coding-agent/examples/disk_write_bench.rs` — the harness (this
  commit's only code change).
- `crates/pi-coding-agent/examples/disk_write_bench.USAGE.md` — this file.
- Reports (matrix JSONs, run tables): `<project>\reports\B1-bench\` (outside
  the repo; not committed).
