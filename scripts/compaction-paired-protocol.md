# Compaction paired outer-run protocol

## R3 evidence class

The R3 controls execute only `compaction_protocol_mock.py`. The mock values,
telemetry and source identities are synthetic non-decision inputs. No Optimus
`bench-local` timing run or speedup claim is part of R3. The driver rejects an
`authorized-r4-timing` plan unless that plan explicitly carries the later R4
authorization flag; this software switch is not authorization by itself.

## Independent components

- `compaction_paired_protocol.py` loads a manifest, verifies provenance,
  launches fresh processes, applies child-only affinity, retains raw evidence
  and admits pairs. It never implements bootstrap math.
- `compaction_paired_stats.py` reads admitted outer-run medians and recorded
  orders. It has no child-process or benchmark discovery logic. It validates
  paired-unit integrity and computes paired log ratios with a deterministic
  95% paired bootstrap CI. Inner samples are not independent observations.
- `test_compaction_paired_protocol.py` executes valid controls and each named
  rejection through the real command-line interfaces.

## Production-like manifest requirements

Use schema `r3-paired-protocol-v1` and exactly 30 outer pairs, 10 warmups and
30 inner iterations for warm lanes. Each arm must declare:

- separate source and Cargo target identities, full HEAD/base hashes, a clean
  tree and computed hashes for listed source files;
- an executable whose path, bytes and SHA-256 are checked before and after;
  the parent also queries the running process image and rejects `--stamp`;
- separate transcript/manifest fixture copies with common pre-registered
  hashes, checked before and after every process;
- the exact argument vector and allowlisted child environment;
- the expected charcount backend/diagnostic and parallel availability,
  worker cap and fallback count;
- the exact release, locked, offline build vector and verified Rust/LLVM/host
  toolchain. The driver computes compiler and Cargo binary identities.

The parallel production lane uses logical CPUs `0,2,4,6,8,10`, mask `0x555`,
for the six exposed physical cores. The parent calls affinity APIs only on the
new child handle/PID. It verifies the resulting child mask and never changes
agent or system affinity. A microkernel lane must be labeled separately and
use one CPU. Replay/process-cold plans use a separate label; fresh process
wall time includes startup and fixture work and is not page-cache-cold proof.

Every pair has recorded before/after load and active-power-scheme receipts.
The required available load fields are fixed to `cpu_total_percent`,
`processor_frequency`, `processor_queue`, `per_logical_processor` and
`available_memory_bytes`; optional relaxation is not accepted. Missing fields
or an incomplete field set fails admission. Thermal, hypervisor scheduling,
SMT/cache contention and unavailable counters remain explicit limitations. No
power or timing setting is changed.

For future authorized timing, two hashed `verify-local` receipts are required.
Their case lists must be nonempty and exactly cover the expected metric cases.
Each `full_projection_sha256` must be an object containing exactly `algorithm`,
`sections` and `note`, with the exact preserved verify-local algorithm and note.
Its `sections` object must contain exactly these 14 keys: `case`,
`fixture_manifest`, `token_outputs`, `cut_output`, `prepare_compaction`,
`conversion`, `serialization`, `replicated_chunk_slicing`,
`session_manager__raw_branch_semantic`,
`session_manager__normalized_restored_messages`, `session_manager__context`,
`session_manager__checkpoint_state`, `session_manager__protected_custom_state`
and `jev_prepare_context`. Every section value is a nonempty lowercase 64-hex
SHA-256 digest. Baseline and candidate objects must match exactly, section by
section; missing/extra content, wrong types, wrong algorithms and section
mismatches fail closed. Every warm pair separately compares nonempty, complete
non-timed preflight case coverage with valid lowercase 64-hex full-output
hashes before admission. Replay must satisfy its R1/R2 success and
fresh-process parity gates. Child output must be a JSON object. Warmup,
iteration, inner-sample, order pair and run pair/sequence contexts use strict
`int`-not-`bool` checks; floats are not accepted. Failure at any provenance,
parity, telemetry, JSON-shape, runtime, numeric or output-create-new stage
prevents the statistics output from being admitted.

## Commands

Synthetic qualification (all raw command receipts are embedded in the result):

```text
python scripts/test_compaction_paired_protocol.py R3-CONTROLS.json
```

Run one manifest:

```text
python scripts/compaction_paired_protocol.py --plan PLAN.json
```

Recompute statistics independently from retained outer medians and orders:

```text
python scripts/compaction_paired_stats.py --runs OUTER-RUNS.json --orders ORDERS.json --seed SEED --bootstrap-samples 10000 --out STATS.json
```

All JSON outputs use exclusive create-new writes. A failing protocol retains
the completed raw runs and a failure receipt with its named stage. Slow
outliers are not dropped. The full seeded order and every paired observation
are retained for review.
