# PR281 native retirement acceptance

**Result: passed for one source and one receiver.** This is transport/lifetime evidence, not an accuracy benchmark.

## Pinned evidence
- Approved source revision: `0209c1d1e248c3f4f103fa8cd65fff213c5fadd6`.
- Live record: `event2-smoke-result.json`, SHA-256 `a13e70139022b52323f2e653a309d50f1003c62592b04485955baf71fa4a0d3e`.
- Reviewed driver: `smoke_native_event_v2.py`, SHA-256 `39e8d80f388e9f64632ac7636800f60e6bb3405f9215047090ab05fa8f9d5d63`; saved upload evidence contains the same hash.
- Frozen proof helper: `smoke_native_header.py`, SHA-256 `7fef8e1c55c543394ef8f885eaf46c2df334701cc2a7973f51a0a9401b186532`.
- Runtime: `66a979989ecb46d497e031f9d88af406`, generation `1`; ready/expected ranks stayed **8/8**.
- Runtime profile digest: `b65cc5f39f2b969cd4bba635b1ddaf96a3d322c6d3f539bbc25b5e6cd3ca880f`.
- Container ID: `0259f807d5cbb87fec56259a990eb56ea48c839d90aa0c2267661f0e67d82542`.
- Sanitized evidence: `acceptance.json`, SHA-256 `df4e7086188b5c4c0f59542adfac0d602638fab2afd8cc7ce6118cd8c9a32ed3`. It also records model/TP/source hashes and local module hashes.

## Observed acceptance chain (UTC, 2026-11-10)
1. Source completed with a validated `occ3.source.v2` receipt. Retained rows: **32**, accepted span **200..232**. The larger union window is legitimately noncontiguous.
2. The read-only log observer was ready before receiver dispatch. A rank-0 `consume_kv_values / consume_released` event matched the source accepted-sequence digest. The event was a trigger, not proof of admission.
3. At **14:36:49.250994**, the one own-publication status response was **injected**, with the expected publication/source/binding digest.
4. The driver made **one** early retire attempt. At **14:36:49.284885**, it received **HTTP 409**, exactly `{"error":{"code":"busy"}}`.
5. At **14:36:55.426265**, the receiver delivered a validated native terminal receipt: **consumed=true, injected_state=true, fallback_used=false**. Its source handle, publication message, receiver session and runtime matched.
6. The owned observer joined. At **14:36:55.464514**, late retirement returned **HTTP 200**, **revoked=true**, **gpu_freed=true**, **state=retired** for the same source and operation.
7. The final runtime identity matched every stable identity field, with the same **8/8** ready ranks.

## Bounds and counters
- Source/receiver limits were unchanged: **512 / 2048** tokens; supported **keepalive-v1**, not terminal-sse-v1.
- Reported completion counts: source **232** (`stop`), receiver **2048** (`length`). Standard prompt/completion/total counts are preserved in JSON.
- Source HTTP lifecycle: **34.149s**. Receiver HTTP lifecycle: **6.762s**. These are client timings, not isolated GPU timings.
- Status calls: **1 of 12**, triggered only by the correlated event; header/comment status calls: **0**.
- Log reader: **480,459 bytes**, **1207 frames/lines**, **1 matched event**; error-free and joined. Limits: 180s, 4MiB, 8192 frames, 4096 lines.

## Prior attempts retained
| Start UTC | Saved record | Outcome |
|---|---|---|
| 13:13:31 | `live-smoke-result.json` | Source HTTP409 at `capture_inflight`; no receiver. Exact ordinary client unproven. No client cleanup receipt. |
| 13:32:48 | `quiet-smoke-result.json` | Nonstream source HTTP503 / `admission_timeout`; no receiver. No client cleanup receipt. |
| 13:49:20 | `stream-smoke-result.json` | Native source/receiver proof and cleanup passed; no keepalive opportunity, so early retire **not exercised**. |
| 14:04:35 | `header-smoke-result.json` | Native completion and cleanup passed; one header status was `published`, so early retire **not exercised**. |
| 14:28:38 | `event-smoke-result.json` | Driver-only broad-window contiguity overconstraint rejected after source/publication. No receiver; cleanup passed. |
| 14:36:14 | `event2-smoke-result.json` | Selected-span guard correction; complete acceptance chain above. |

## Independent local verification and limits
- **30 saved-record checks passed**: source-intent SHA-256 and publication-binding SHA-256 were recomputed; receipt, identity, rank, geometry, request and timing bindings were cross-checked.
- Driver/helper/file hashes were recomputed locally. The corrected driver also passed **228 socket-forbidden offline checks**, including all previous 216 checks.
- Native SSE bytes are not retained in saved JSON. Their SHA-256 was checked by the pinned live parser over exact native frames plus `[DONE]`, excluding comments and private receipt events. This report rechecks saved hash consistency, not raw SSE bytes.
- `gpu_freed=true` is a logical cleanup receipt, **not physical memory erasure proof**. This is not full CUDA qualification or semantic accuracy evidence.
- Busy preceded the client terminal receipt. **The server terminal-commit-before-proof window was not measured.**
- Local source hashes are not a new installed-byte attestation. Log content hashes are not globally unique attempt IDs.
- No benchmark outcome is included. Native-agent benchmark attempts and GPQA results are outside this report.
- This report used local saved artifacts only: **no runtime, remote, network or model calls**.
