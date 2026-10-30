# OCC-3 live_before diagnosis — detector overlay gap (attempt-03, window icalib-20261029T032600Z)

## Verdict
The missing tool call is a **detector error-swallowing gap in the armed buffered drain**, not a request-construction or adapter pass-through defect.

Chain (worktree `optimus-occ3-icalib-20261028`, HEAD d8810c72e):
1. `patches/native-startup.v1.json` overlays `glm47_moe_detector.parse_streaming_increment`: its `except` block checks `if _OCC3_STRICT_STREAM.get(): raise`, else logs and returns `StreamingParseResult(normal_text=current_text)` — the retained buffer is re-emitted as **prose**.
2. `extensions/sglang-occ3/occ3_sglang/chat_adapter.py` sets that ContextVar **only** in `_terminal_passthrough` (~1293, reset ~1297).
3. The armed attached-turn path is `_run`'s buffered drain (~852-865): it drains `response.body_iterator` chunk-by-chunk with the flag never set.
4. Armed turns decode eagerly at ~1 token/delta, so `occ3_handoff` arguments arrive split across many chunks; the incremental parse raises internally, the overlay swallows it, and the buffered `⟨tool_call⟩…⟨/tool_call⟩` buffer is re-emitted as normal text.
5. Result matches the live signature exactly: 576/960 generated tokens of prose, zero `tool_calls`, zero `/occ3/v1/native/publish`, `receiver_request_count 0`, `source_facts_absent`, `occ3_source_uncertain`, `[occ3_completed_binding_count]`.

The replica probe (batched deltas, same tools + `reasoning_effort=max`) never produces split arguments, so it never triggers the failure. Slowness (78-118 s) is separate and by design: capture-safe serving (eager, CUDA graphs off) at ~5-7 tok/s.

History: detector + overlay shipped with the original design (glm47 detector sha 61a62420…→e09a129d…; manifest commit 926c262ec); later chat_adapter commits (fa43f6673 content_tail_v2 + collector hook, ce13ea657 prefix marker) did not touch the overlay wiring. Not a campaign regression — an always-present latent gap.

## Ranked fixes
- **F1 (production serving, small)**: in `chat_adapter.py` `_run`, set `_OCC3_STRICT_STREAM` around the buffered `body_iterator` drain exactly as `_terminal_passthrough` does (fail-closed). A dropped attach turns into a loud 409/AttachmentError instead of silent prose. Blast radius: every attached turn on this build.
- **F2 (production serving, root cause)**: fix the incremental split-argument parse in the glm47 detector — buffer to the argument-close marker before parsing, so streaming tool calls succeed without the strict crutch. Larger, but removes the swallow entirely.
- **F3 (offline harness only)**: `occ3_native_phase_suite.rs` positive case should hard-fail when `source_tool_calls == 0` on an armed capture turn, so this class of failure reports as an error, not a semantic-absence mystery.

## RED-first test draft
`test_armed_tool_call_strict_overlay.py` (this directory). Stub detector faithful to the attested manifest contract (real glm47 source not on this machine — sha256s and before/after blocks only). Run: `python test_armed_tool_call_strict_overlay.py`.
Verified results:
- PASS control: fixed detector delivers `occ3_handoff` on both paths (schema per `occ3_state.rs` ~651, identical to live probe).
- **RED** armed buffered drain: zero tool_calls, buffer re-emitted as content (live_before reproduced).
- PASS terminal path: raises under the strict overlay (fail-closed).
- **RED** static AST check vs real `chat_adapter.py`: exactly `['_run']` drains `body_iterator` without the strict overlay.

## Calibration corpus collector
Unaffected. `latent_worker.py` `_calibration_handoff` (356-393) consumes **engine-side** rows (hidden states + token_ids) independent of the SSE parse layer. Ordinary non-tool corpus captures do not route through the detector; only tool-call serving/handoff turns break. **P4 collection can run on the serving build now.**
