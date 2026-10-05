# D2: Write buffering (R5 Item 1) — flush-period probe and AFTER matrix

## Flush-period probe (t-probe-run1.json)

Protocol: release build, 40 events/s per stream, 20 s per phase, 800 single-event
appends per phase, one timed final drain; T=0 phase is the write-through control.
Continuous sibling-compile background throughout — identical across all phases, so
within-run comparisons are paired. Parity: every phase ok; cross-T digest identity holds.

| stream | T (ms) | write ops | op p50 (us) | op p95 (us) | drain (us) |
|--------|-------:|----------:|------------:|------------:|-----------:|
| evlog | 0 | 800 | 163 | 229 | 21 |
| session | 0 | 800 | 508 | 674 | 75 |
| evlog | 50 | 271 | 79 | 116 | 385 |
| session | 50 | 269 | 163 | 251 | 318 |
| evlog | 100 | 161 | 77 | 107 | 281 |
| session | 100 | 161 | 159 | 267 | 328 |
| evlog | 250 | 73 | 81 | 118 | 461 |
| session | 250 | 74 | 156 | 236 | 335 |
| evlog | 500 | 39 | 86 | 119 | 327 |
| session | 500 | 40 | 156 | 231 | 364 |
| evlog | 1000 | 20 | 82 | 173 | 322 |
| session | 1000 | 23 | 167 | 281 | 505 |

Reading: latency plateaus at T >= 50 on both streams (session 508 -> ~156 us,
evlog 163 -> ~80 us); past 50 the only tradeoff is write-op count vs crash window.
Chosen defaults: session 100 ms (5x fewer write ops, 100 ms of entries at risk),
event log 250 ms (11x fewer ops; the ledger is a crash-tolerant cache whose torn
tails are already skipped on read and truncated on append).

## AFTER matrix (reports/B1-bench/after-run1.json)

Protocol matched to the BEFORE full-run1: 89 points, runs=3 warmup=1 alloc-runs=1,
shipped defaults (no --flush-ms), fresh release exe sha256 dbdbfd9be4fb....

Hard gate: PASS — 89/89 parity ok; every point digest-identical to BEFORE
(tree digests pass-for-pass; session points match on normalized digests:
sess-t10 8772efc48490..., sess-t100 039de2c7cfab..., sess-t1000 0b1bf5fc3cd4...).

Timing (INDICATIVE ONLY — 100% CPU sibling-compile background; probe table is the
primary timing evidence):

| group | median p50 delta | notes |
|-------|----------------:|-------|
| evlog-* (non-fsync) | -95.5% | 1.3-11.7 ms -> 0.04-0.11 ms |
| evlog-s60k-d1 (fsync) | -13% | fsync dominates, expected |
| sess-t* | -83.7% | 1.16-1.21 -> ~0.19 ms; t1000 p95 1.73 -> 0.42 ms |
| a_* (76 pts, unchanged code) | +4.7% | swings -63%..+54% both ways = noise band |
| edit-* (6 pts, unchanged code) | -3.2% | swings -55%..+362% = same noise band |

The unchanged-scenario noise band brackets most a_*/edit deltas; no quiet-window
re-run is claimed necessary. Digests prove zero byte changes anywhere.

Commits (oldest first): d8c919524, 231a076ce, ca3914c37, db43944ac, ed3ac3193.
