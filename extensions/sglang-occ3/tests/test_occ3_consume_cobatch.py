"""Consume co-batching (A): a post-injection consume decodes in the shared batch.

Exact patched methods from the attested source; external state is stubbed.
These prove scheduling/settle branches only, not CUDA graphs, NCCL agreement
or model numerics.
"""
from __future__ import annotations

import types
import unittest
from http import HTTPStatus
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_latent_ipc_patch import PinnedCase, descriptor

SCHED = "managers/scheduler.py"
BATCH = "managers/schedule_batch.py"
POLICY = "managers/schedule_policy.py"
RESULT = "managers/scheduler_components/batch_result_processor.py"


class Abort:
    def __init__(self, message="abort", *args, **kwargs):
        self.message = message


class Row:
    """A request row; attached/eager/private flags as admission sets them."""
    def __init__(self, rid, *, attached=False, eager=False, private=False, done=False, out=()):
        self.rid, self.occ3_attached, self.occ3_force_eager = rid, attached, eager
        self.occ3_private_prefix, self.done = private, done
        self.output_ids = list(out)
        self.is_retracted = False
        self.finished_reason = self.to_finish = None
        self.inflight_middle_chunks = 0
        self.kv = NS(holds_kv=False)
        self.occ3_abort_done = False
        self.occ3_worker_descriptor = descriptor("consume") if attached else None
    def finished(self):
        # Native Req.finished(): a set finish reason is finished.
        return self.done or self.finished_reason is not None


def consume_row(rid="consume", *, eager=False, **kw):
    return Row(rid, attached=True, eager=eager, private=True, **kw)


class Batch:
    def __init__(self, reqs, eager=None):
        self.reqs = reqs
        self.occ3_force_eager = any(r.occ3_force_eager for r in reqs) if eager is None else eager
        self.occ3_context = ("ctx",) if self.occ3_force_eager else None
    def filter_batch(self):
        self.reqs = [r for r in self.reqs if not r.finished()]
        self.occ3_force_eager = any(r.occ3_force_eager for r in self.reqs)


class TestConsumeCoBatch(PinnedCase):
    def settle(self, batch, *, chunked=None, mode_extend=False, finish=(), release_ok=True):
        calls = []
        worker = types.ModuleType("occ3_sglang.latent_worker")
        def abort(s, r, reason):
            if getattr(r, "occ3_abort_done", False):
                return
            calls.append(("abort", r.rid, reason))
            r.occ3_abort_done = True
        worker.abort_request = abort
        worker.commit_result = lambda s, r, result, index, start, end, **kw: calls.append(
            ("commit", r.rid, index, start, end, kw["terminal"]))
        def release(s, r):
            calls.append(("release", r.rid))
            if not release_ok:
                return False   # record retired/expired before injection completed
            r.occ3_force_eager = False
            return True
        worker.release_consume_cohort = release
        ns = dict(FINISH_ABORT=Abort, HTTPStatus=HTTPStatus, logger=None,
                  release_kv_cache=lambda r, tree, **kw: calls.append(("free", r.rid)))
        starts = self.sources.function(RESULT, "_occ3_output_starts", "SchedulerBatchResultProcessor")
        settle = self.sources.function(RESULT, "_occ3_settle_result", "SchedulerBatchResultProcessor", **ns)
        visible = staticmethod(lambda r: len(r.output_ids))
        cls = type("P", (), {"_visible_output_len": visible, "_occ3_output_starts": classmethod(starts),
                             "_occ3_settle_result": settle})
        proc = cls()
        proc.scheduler = NS(chunked_req=chunked)
        proc.tree_cache = object()
        batch.forward_mode = NS(is_extend=lambda: mode_extend)
        batch.forward_iter = 3
        with patch.dict("sys.modules", {worker.__name__: worker}):
            before = cls._occ3_output_starts(batch)
            for r in batch.reqs:  # this step's decode appends one token per live row
                if not r.finished():
                    r.output_ids.append(5)
                    if r.rid in finish:
                        r.done = True
            proc._occ3_settle_result(batch, NS(occ3_capture_stage=None), before)
        return before, calls

    def test_ordinary_only_batch_has_no_occ3_settle(self):
        starts, calls = self.settle(Batch([Row("a"), Row("b"), Row("c")]))
        self.assertIsNone(starts)
        self.assertEqual(calls, [])

    def test_three_row_batch_settles_only_the_attached_row_at_its_index(self):
        for index in (1, 2):
            rows = [Row("a", out=[1]), Row("b", out=[1, 2]), Row("c")]
            rows[index] = consume_row(out=[7, 7])
            starts, calls = self.settle(Batch(rows))
            self.assertEqual([s is not None for s in starts], [i == index for i in range(3)])
            # Mid-decode consume: no per-step commit (no cross-rank gather).
            self.assertEqual(calls, [])
            # The step that finishes the consume journals its terminal once.
            starts, calls = self.settle(Batch(rows), finish=("consume",))
            self.assertEqual(calls, [("commit", "consume", index, 3, 4, True)])

    def test_cancelled_cobatched_consume_aborts_only_itself(self):
        rows = [Row("a"), consume_row(), Row("c")]
        rows[1].to_finish = Abort()
        rows[1].kv.holds_kv = True
        rows[0].to_finish = Abort()  # ordinary row abort is native-only
        _, calls = self.settle(Batch(rows))
        self.assertEqual(calls, [("abort", "consume", "cancelled"), ("free", "consume")])
        self.assertIsInstance(rows[1].finished_reason, Abort)
        self.assertEqual(rows[1].finished_reason.message, "occ3_cancelled")
        self.assertIsNone(rows[0].finished_reason)

    def test_chunked_consume_stays_eager_until_its_final_chunk(self):
        req = consume_row(eager=True)
        req.inflight_middle_chunks = 1
        _, calls = self.settle(Batch([req]), chunked=req, mode_extend=True)
        self.assertEqual(calls, [])
        self.assertTrue(req.occ3_force_eager)
        req.inflight_middle_chunks = 0
        _, calls = self.settle(Batch([req]), chunked=req, mode_extend=True)
        self.assertEqual(calls, [])  # still the scheduler's chunked row
        self.assertTrue(req.occ3_force_eager)
        _, calls = self.settle(Batch([req]), chunked=None, mode_extend=True)
        self.assertEqual(calls, [("release", "consume")])
        self.assertFalse(req.occ3_force_eager)

    def test_terminal_on_first_prefill_commits_and_never_releases(self):
        req = consume_row(eager=True)
        batch = Batch([req])
        batch.forward_mode = NS(is_extend=lambda: True)
        starts = self.sources.function(RESULT, "_occ3_output_starts", "SchedulerBatchResultProcessor")
        self.assertEqual(starts(NS(_visible_output_len=lambda r: len(r.output_ids)), batch), [0])
        _, calls = self.settle(batch, mode_extend=True, finish=("consume",))
        self.assertEqual(calls, [("commit", "consume", 0, 0, 1, True)])  # terminal, no cohort release

    def test_capture_keeps_per_step_commit(self):
        cap = Row("cap", attached=True, eager=True)
        _, calls = self.settle(Batch([cap]))
        self.assertEqual(calls, [("commit", "cap", 0, 0, 1, False)])

    def test_merge_with_post_injection_consume_does_not_raise(self):
        merge = self.sources.function(BATCH, "merge_batch", "ScheduleBatch")
        running = NS(reqs=[Row("a"), Row("b")], occ3_force_eager=False, occ3_context=None)
        # last_batch kept its stale cohort flag; its only row has left the cohort.
        last = NS(reqs=[consume_row()], occ3_force_eager=True, occ3_context=("stale",))
        try:
            merge(running, last)
        except RuntimeError as error:
            # Past the occ3 guard the stub cannot run native merge code.
            self.assertNotIn("mixed_cohort", str(error))
        except Exception:
            pass
        self.assertFalse(last.occ3_force_eager)
        self.assertIsNone(last.occ3_context)
        eager = NS(reqs=[consume_row(eager=True)], occ3_force_eager=True, occ3_context=("ctx",))
        with self.assertRaisesRegex(RuntimeError, "mixed_cohort"):
            merge(running, eager)

    def test_filter_recomputes_a_stale_cohort_flag_even_on_the_fast_path(self):
        fn = self.sources.function(BATCH, "filter_batch", "ScheduleBatch", strip_beam_tail=lambda s: None,
                                   Req=Row)
        state = NS(reqs=[consume_row()], occ3_force_eager=True, occ3_context=("stale",))
        fn(state)  # keep_indices == all rows: native no-op fast path
        self.assertFalse(state.occ3_force_eager)
        self.assertIsNone(state.occ3_context)
        self.assertEqual(len(state.reqs), 1)

    def test_forward_batch_rejects_a_shared_batch_with_occ3_context(self):
        init = self.sources.function("model_executor/forward_batch_info.py", "init_new", "ForwardBatch")
        with self.assertRaisesRegex(RuntimeError, "context_mismatch"):
            init(object, NS(occ3_context=("ctx",), occ3_force_eager=False), object(),
                 return_hidden_states_before_norm=False)

    def test_prefill_selection_admits_a_consume_while_ordinary_rows_decode(self):
        fn = self.sources.function(SCHED, "_occ3_prefill_selection", "Scheduler")
        pending = consume_row("consume", eager=True)
        state = NS(waiting_queue=[Row("o1"), pending], chunked_req=None)
        self.assertEqual(fn(state, Batch([Row("a"), Row("b")])), ("consume", False))
        # A capture still drains ordinary rows first.
        cap = Row("cap", attached=True, eager=True)
        state.waiting_queue = [cap]
        self.assertEqual(fn(state, Batch([Row("a")])), (None, True))
        self.assertEqual(fn(state, Batch([])), ("cap", False))

    def test_decoding_post_injection_consume_never_drains_other_prefills(self):
        fn = self.sources.function(SCHED, "_occ3_prefill_selection", "Scheduler")
        state = NS(waiting_queue=[Row("o1")], chunked_req=None)
        self.assertEqual(fn(state, Batch([Row("a"), consume_row()])), (None, False))
        # An eager row (capture turn) still drains.
        self.assertEqual(fn(state, Batch([Row("a", attached=True, eager=True)])), (None, True))
        # A chunked consume prefill keeps its exclusive extend batch.
        chunk = consume_row("chunk", eager=True)
        state.chunked_req = chunk
        self.assertEqual(fn(state, Batch([Row("a")])), ("chunk", False))

    def test_attached_abort_matching_is_exact_after_the_flag_clear(self):
        match = self.sources.function(SCHED, "_occ3_abort_matches", "Scheduler")
        attached = consume_row("consume-rid")
        self.assertFalse(match(attached, NS(abort_all=False, rid="consume")))  # no prefix match
        self.assertTrue(match(attached, NS(abort_all=False, rid="consume-rid")))
        self.assertTrue(match(Row("ordinary-1"), NS(abort_all=False, rid="ordinary")))

    def test_abort_attached_reaches_a_decoding_consume(self):
        fn = self.sources.function(SCHED, "_occ3_abort_attached", "Scheduler")
        match = self.sources.function(SCHED, "_occ3_abort_matches", "Scheduler")
        consume = consume_row("consume-rid")
        aborted = []
        state = NS(waiting_queue=[], grammar_manager=NS(grammar_queue=[]),
                   collect_inflight_reqs=lambda: [Row("a"), consume], chunked_req=None,
                   _occ3_abort_matches=match, _occ3_abort_one=lambda r, reason: aborted.append((r.rid, reason)))
        fn(state, NS(abort_all=False, rid="consume-rid", abort_message=None))
        self.assertEqual(aborted, [("consume-rid", "cancelled")])

    def test_retraction_aborts_the_attached_row_and_keeps_ordinary_rows_retractable(self):
        abort_calls = []
        worker = types.ModuleType("occ3_sglang.latent_worker")
        worker.abort_request = lambda s, r, reason: abort_calls.append((r.rid, reason))
        frees = []
        release = self.sources.function(BATCH, "release_req", FINISH_ABORT=Abort, HTTPStatus=HTTPStatus,
                                        release_kv_cache=lambda r, tree, **kw: frees.append(r.rid))
        consume = consume_row()
        consume.occ3_scheduler = object()
        with patch.dict("sys.modules", {worker.__name__: worker}):
            saved = release(req=consume, remaing_req_count=2, req_to_token_pool=None,
                            token_to_kv_pool_allocator=None, tree_cache=None, hisparse_coordinator=None)
        self.assertFalse(saved)
        self.assertEqual(abort_calls, [("consume", "retraction_unsupported")])
        self.assertEqual(frees, ["consume"])
        self.assertEqual(consume.to_finish.message, "occ3_retraction_unsupported")
        reset = self.sources.function(BATCH, "reset_for_retract", "Req")
        with self.assertRaisesRegex(RuntimeError, "retraction_unsupported"):
            reset(consume)

    def test_preemption_stays_available_beside_an_attached_row(self):
        node = self.sources.node(POLICY, "preempt_to_schedule", "PrefillAdder")
        import ast
        text = ast.unparse(node)
        self.assertNotIn("for r in self.running_batch.reqs\n        ):\n            return False", text)
        first = text.split("valid_running_reqs")[0]
        self.assertNotIn("running_batch", first)
        self.assertIn("occ3_attached", text.split("valid_running_reqs")[1])

    def test_attached_row_keeps_its_unclipped_running_reservation(self):
        fn = self.sources.function(POLICY, "_get_running_request_total_token_offset", "PrefillAdder",
                                   CLIP_MAX_NEW_TOKENS=4096)
        consume = consume_row(out=[1] * 10)
        consume.sampling_params = NS(max_new_tokens=10000)
        self.assertEqual(fn(NS(new_token_ratio=0.1), consume), 9990)
        ordinary = Row("o", out=[1] * 10)
        ordinary.sampling_params = NS(max_new_tokens=10000)
        self.assertEqual(fn(NS(new_token_ratio=0.5), ordinary), 2048)

    # ------------------------------------------------- Fable review, blocking 1
    def test_refused_cohort_release_aborts_the_consume_and_leaves_no_eager_row(self):
        req = consume_row(eager=True)
        req.kv.holds_kv = True
        _, calls = self.settle(Batch([req]), mode_extend=True, release_ok=False)
        self.assertEqual(calls, [("release", "consume"), ("abort", "consume", "commit_failed"),
                                 ("free", "consume")])
        self.assertEqual(req.finished_reason.message, "occ3_commit_failed")
        self.assertFalse(req.occ3_force_eager)
        # The settled batch merges into a running batch of ordinary rows.
        req.done = True
        last = NS(reqs=[req], occ3_force_eager=True, occ3_context=("ctx",))
        merge = self.sources.function(BATCH, "merge_batch", "ScheduleBatch")
        running = NS(reqs=[Row("a"), Row("b")], occ3_force_eager=False, occ3_context=None)
        try:
            merge(running, last)
        except RuntimeError as error:
            self.assertNotIn("mixed_cohort", str(error))
        except Exception:
            pass  # past the occ3 guard the stub cannot run native merge code
        self.assertFalse(last.occ3_force_eager)

    def test_defensive_sweep_aborts_any_eager_consume_left_after_an_extend_settle(self):
        # start None (no accepted pair this step), but no longer the chunked row.
        req = consume_row(eager=True)
        req.inflight_middle_chunks = 1
        req.kv.holds_kv = True
        _, calls = self.settle(Batch([req]), chunked=None, mode_extend=True)
        self.assertEqual(calls, [("abort", "consume", "commit_failed"), ("free", "consume")])
        self.assertFalse(req.occ3_force_eager)
        self.assertEqual(req.finished_reason.message, "occ3_commit_failed")

    def test_defensive_sweep_keeps_the_scheduler_chunked_consume(self):
        req = consume_row(eager=True)
        req.inflight_middle_chunks = 1
        _, calls = self.settle(Batch([req]), chunked=req, mode_extend=True)
        self.assertEqual(calls, [])
        self.assertTrue(req.occ3_force_eager)

    # ----------------------------------------------- Fable review, should fix 2
    def test_overshoot_tick_keeps_the_first_abort_reason(self):
        rows = [Row("a"), consume_row(), Row("c")]
        rows[1].occ3_abort_done = True
        rows[1].finished_reason = Abort("occ3_admission_timeout")
        rows[1].finished_len = 3
        rows[1].kv.holds_kv = True
        _, calls = self.settle(Batch(rows))
        self.assertEqual(calls, [])
        self.assertEqual(rows[1].finished_reason.message, "occ3_admission_timeout")
        self.assertEqual(rows[1].finished_len, 3)

    # ------------------------------------- behaviour, replacing source-text pins

    # ------------------------------------- behaviour, replacing source-text pins
    def test_deadline_sweep_aborts_a_decoding_consume_past_its_deadline(self):
        off = NS(get=lambda: 0)
        sweep = self.sources.function(SCHED, "_poll_timeout_aborts", "Scheduler",
                                      AbortReq=lambda **kw: NS(**kw), List=list,
                                      time=NS(monotonic=lambda: 50.0, perf_counter=lambda: 50.0),
                                      envs=NS(SGLANG_REQ_WAITING_TIMEOUT=off,
                                              SGLANG_REQ_RUNNING_TIMEOUT=off))
        def row(rid, deadline, *, attached=True, eager=False):
            req = Row(rid, attached=attached, eager=eager, private=attached)
            req.occ3_worker_descriptor = NS(deadline=deadline)
            req.occ3_queue_deadline = None
            return req
        late, live, ordinary = row("late", 10.0), row("live", 90.0), Row("plain")
        queued = row("queued", 90.0, eager=True)
        queued.occ3_queue_deadline = 40.0  # queue bound applies to queued rows only
        state = NS(waiting_queue=[queued], grammar_manager=NS(grammar_queue=[]),
                   collect_inflight_reqs=lambda: {late, live, ordinary}, chunked_req=None)
        aborts = sweep(state)
        self.assertEqual(sorted((a.rid, a.abort_message) for a in aborts),
                         [("late", "occ3_admission_timeout"), ("queued", "occ3_admission_timeout")])

    def test_retraction_pops_ordinary_rows_and_never_the_attached_row(self):
        retract = self.sources.function(
            BATCH, "retract_decode", "ScheduleBatch", FINISH_ABORT=Abort,
            HTTPStatus=NS(INTERNAL_SERVER_ERROR=500, SERVICE_UNAVAILABLE=503),
            beam_retraction_order=lambda order, reqs: order,
            get_disagg=lambda: NS(disaggregation_decode_retraction_backup="none"),
            logger=NS(warning=lambda *a, **k: None), List=list, Tuple=tuple, Req=Row)
        # Native order would pop the attached row first (it is last).
        rows = [Row("a"), Row("c"), Row("d"), consume_row()]
        for req in rows:
            req.beam_group = None
        released, kept = [], []
        budget = {"fits_after": 2}  # memory fits once two ordinary rows are gone
        batch = NS(reqs=rows)
        batch._get_decode_retraction_order = lambda reqs: [0, 1, 2, 3]
        batch.check_decode_mem = lambda selected_indices=None: len(released) >= budget["fits_after"]
        batch.release_req = lambda idx, remaining, offload_kv=True: released.append(rows[idx].rid) or True
        batch.filter_batch = lambda keep_indices: kept.extend(rows[i].rid for i in keep_indices)
        try:
            retract(batch)
        except Exception as error:
            if not released:
                raise  # failed before the loop: the stub no longer matches
            self.assertIsInstance(error, (AttributeError, NameError, TypeError), error)
        self.assertEqual(released, ["d", "c"])
        self.assertNotIn("consume", released)
        if kept:
            self.assertIn("consume", kept)

