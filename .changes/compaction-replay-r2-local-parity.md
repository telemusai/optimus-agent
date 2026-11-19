# Compaction replay R2 local parity

- Add non-timed `verify-local` full-output capture/comparison for production local compaction kernels and named Jev skip lanes.
- Preserve independent raw durable parity (R1), normalized restored parity, branch, checkpoint and protected-state controls.
- Make `prepare_compaction` and unexpected Jev errors fail closed before timing; scalar checksums remain auxiliary only.
- Reject invalid, duplicate, zero, negative, fractional, non-finite and excessive numeric CLI inputs as appropriate.
- Add deterministic R2 named-stage negative controls and a same-length serialization corruption control.

No production source, optimization, performance protocol or frozen fixture is changed.

Review repair:
- Present-empty numeric options now fail explicitly instead of becoming defaults.
- A real eligible Jev pair reaches `InsufficientReduction`; its bench lane has null timing and skip outcome mismatches reject at `jev-outcome-parity`.
- `capture-local` binds arm, reviewed production base, byte-identical projection schema/source, independent executable/PID runs, production and adapter identities, exact fixture bytes and effective configuration. Candidate comparison is admitted only against a frozen reviewed R1 baseline with a distinct executable.
