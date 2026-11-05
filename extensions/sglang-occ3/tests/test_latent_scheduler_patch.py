"""Exact patched scheduler/result methods, external state is explicitly stubbed.

These tests prove bounded branches and accepted visible-token lineage. They do
not prove model inference, NCCL agreement, CUDA events or semantic transfer.
"""
from __future__ import annotations

import copy
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from http import HTTPStatus
from array import array
import re

from test_latent_ipc_patch import PinnedCase, descriptor


class Abort:
    def __init__(self, message="abort", *args, **kwargs): self.message = message
class Finish:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


class Row:
    def __init__(self, rid, opted=False, done=False):
        self.rid, self.occ3_force_eager, self.done = rid, opted, done
    def finished(self): return self.done


class Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.occ3_force_eager = any(r.occ3_force_eager for r in reqs)
    def filter_batch(self):
        self.reqs = [r for r in self.reqs if not r.finished()]
        self.occ3_force_eager = any(r.occ3_force_eager for r in self.reqs)


class HostValues:
    """Already-host normal token metadata; not a Torch/math surrogate."""
    is_cpu = True
    def __init__(self, values): self.values = values
    def tolist(self): return list(self.values)


class TestScheduler(PinnedCase):
    def test_cohort_drains_normal_and_restores_ordinary_admission(self):
        fn = self.sources.function("managers/scheduler.py", "_occ3_prefill_selection", "Scheduler")
        ordinary, latent = Row("ordinary"), Row("latent", True)
        state = NS(waiting_queue=[latent, Row("ordinary2")], chunked_req=None)
        self.assertEqual(fn(state, Batch([ordinary])), (None, True))
        self.assertEqual(fn(state, Batch([])), ("latent", False))
        ordinary.done = True
        old = Batch([ordinary])
        self.assertEqual(fn(state, old), ("latent", False))
        self.assertEqual(old.reqs, [])
        state.waiting_queue = [Row("ordinary2")]
        self.assertEqual(fn(state, Batch([latent])), (None, True))
        latent.done = True
        ended = Batch([latent])
        self.assertEqual(fn(state, ended), (None, False))
        self.assertFalse(ended.occ3_force_eager)
        self.assertEqual(ended.reqs, [])

    def test_chunked_cohort_cannot_take_other_rows(self):
        fn = self.sources.function("managers/scheduler.py", "_occ3_prefill_selection", "Scheduler")
        normal, latent = Row("ordinary"), Row("latent", True)
        state = NS(waiting_queue=[latent], chunked_req=normal)
        self.assertEqual(fn(state, Batch([])), (None, True))
        state.chunked_req = latent
        self.assertEqual(fn(state, Batch([])), ("latent", False))

    def test_full_budget_never_mutates_max_or_min_tokens(self):
        fn = self.sources.function("managers/scheduler.py", "init_req_max_new_tokens", "Scheduler", FINISH_ABORT=Abort, HTTPStatus=HTTPStatus)
        class Alloc:
            allowed = 9000
            def max_new_tokens_for_memory(self, *args, **kw): return self.allowed
        alloc = Alloc()
        state = NS(max_total_num_tokens=100000, kv_shard_widening=1, page_size=4, max_req_input_len=20000,
                   max_req_len=30000, max_new_tokens_limit=None, token_to_kv_pool_allocator=alloc,
                   sliding_window_size=None, chunked_prefill_size=32768)
        for allowed, length, budget in [(9000, 120, 9000), (4000, 120, 9000), (9000, 22000, 9000), (None, 120, 9000)]:
            alloc.allowed = allowed
            req = NS(occ3_force_eager=True, origin_input_ids=list(range(length)),
                     sampling_params=NS(max_new_tokens=budget, min_new_tokens=100), to_finish=None)
            fn(state, req)
            self.assertEqual(req.sampling_params.max_new_tokens, budget)
            self.assertEqual(req.sampling_params.min_new_tokens, 100)
            self.assertEqual(req.to_finish is None, allowed == budget and length == 120)

    def test_full_budget_admission_no_4096_clipping_for_optin(self):
        fn = self.sources.function("managers/schedule_policy.py", "_get_running_request_total_token_offset", "PrefillAdder", CLIP_MAX_NEW_TOKENS=4096)
        req = NS(occ3_force_eager=True, sampling_params=NS(max_new_tokens=10000), output_ids=[1, 2])
        state = NS(new_token_ratio=0.25)
        self.assertEqual(fn(state, req), 9998)
        req.occ3_force_eager = False
        self.assertEqual(fn(state, req), 4096 * 0.25)

    def test_deadline_uses_leader_abort_channel_exact_rid(self):
        class AbortReq:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        fn = self.sources.function("managers/scheduler.py", "_poll_timeout_aborts", "Scheduler",
            AbortReq=AbortReq, time=NS(monotonic=lambda: 100, perf_counter=lambda: 100),
            envs=NS(SGLANG_REQ_WAITING_TIMEOUT=NS(get=lambda: 0), SGLANG_REQ_RUNNING_TIMEOUT=NS(get=lambda: 0)))
        waiting, active, fresh = Row("waiting", True), Row("active", True), Row("fresh", True)
        for req, deadline in [(waiting, 50), (active, 90), (fresh, 200)]:
            req.occ3_worker_descriptor = NS(deadline=deadline)
            req.occ3_queue_deadline = 80 if req is waiting else 150
        state = NS(waiting_queue=[waiting, fresh], grammar_manager=NS(grammar_queue=[]), chunked_req=None,
                   collect_inflight_reqs=lambda: [active, active])
        result = fn(state)
        self.assertEqual([x.rid for x in result], ["waiting", "active"])
        # The sweep carries its own bounded refusal reason so the abort path
        # can distinguish admission timeout from a user cancellation.
        self.assertTrue(all(x.abort_message == "occ3_admission_timeout" for x in result))

    def test_exact_abort_identity_and_no_normal_prefix_change(self):
        fn = self.sources.function("managers/scheduler.py", "_occ3_abort_matches", "Scheduler")
        abort = NS(abort_all=False, rid="r")
        self.assertFalse(fn(Row("r-extra", True), abort))
        self.assertTrue(fn(Row("r", True), abort))
        self.assertTrue(fn(Row("r-extra", False), abort))
        abort.abort_all = True
        self.assertTrue(fn(Row("another", True), abort))

    def test_admission_gate_checks_active_cache_not_display_name(self):
        class RadixCache: pass
        class Unsupported(RadixCache): pass
        class UnifiedRadixCache: pass
        class UnsupportedUnified(UnifiedRadixCache): pass
        class ThirdPartyCache: pass
        cache_module = types.ModuleType("sglang.srt.mem_cache.radix_cache"); cache_module.RadixCache = RadixCache
        unified_module = types.ModuleType("sglang.srt.mem_cache.unified_radix_cache")
        unified_module.UnifiedRadixCache = UnifiedRadixCache
        fn = self.sources.function("managers/scheduler.py", "_occ3_validate_received", "Scheduler",
            envs=NS(SGLANG_USE_PICKLE_IPC=NS(get=lambda: False)),
            get_parallel=lambda: NS(pp_size=1, attn_cp_size=1, enable_dp_attention=False),
            BeamCoordinator=NS(request_beam_width=lambda req: 1), DisaggregationMode=NS(NULL="none"))
        state = NS(tree_cache=RadixCache(), enable_hierarchical_cache=False, enable_hicache_storage=False,
                   enable_session_radix_cache=False, enable_lmcache=False, enable_unified_cache_external_linker=False,
                   decode_offload_manager=None, enable_hisparse=False, dllm_config=None, sliding_window_size=None,
                   disaggregation_mode="none", waiting_queue=[], grammar_manager=NS(grammar_queue=[]))
        req = NS(occ3_worker_descriptor=descriptor(), rid="native-rid", session_params=None, session_id=None,
                 mm_inputs=None, input_embeds=None, positional_embed_overrides=None, lora_id=None,
                 token_type_ids=None, token_indices_to_pool=None, multi_item_delimiter_indices=None,
                 kv_hints=None, bootstrap_host=None, return_hidden_states=False, return_routed_experts=False,
                 return_indexer_topk=False, return_sampling_mask=False)
        with patch.dict("sys.modules", {cache_module.__name__: cache_module,
                                        unified_module.__name__: unified_module}):
            self.assertEqual(fn(state, req), descriptor())
            for key in ("session_id", "mm_inputs", "input_embeds", "kv_hints"):
                setattr(req, key, object())
                with self.assertRaises(ValueError): fn(state, req)
                setattr(req, key, None)
            state.tree_cache = UnifiedRadixCache()
            self.assertEqual(fn(state, req), descriptor())
            for cache in (Unsupported(), UnsupportedUnified(), ThirdPartyCache()):
                state.tree_cache = cache
                with self.assertRaisesRegex(ValueError, "occ3_unsupported_profile"): fn(state, req)
            state.tree_cache = RadixCache()
            state.waiting_queue = [Row("a", True), Row("b", True)]
            with self.assertRaisesRegex(ValueError, "busy"): fn(state, req)

    def test_incomplete_worker_admission_never_queues_or_infers(self):
        worker = types.ModuleType("occ3_sglang.latent_worker")
        calls = []
        def incomplete(*args):
            calls.append("all-rank-admission")
            raise RuntimeError("one rank missing")
        worker.admit_request = incomplete
        fn = self.sources.function("managers/scheduler.py", "handle_generate_request", "Scheduler",
            get_disagg=lambda: NS(disaggregation_bootstrap_port=1234),
            BeamCoordinator=NS(request_beam_width=lambda req: 1), Req=lambda *a, **kw: NS(), array=array)
        node = self.sources.node("managers/scheduler.py", "handle_generate_request", "Scheduler")
        # Populate all received fields accessed before the A admission seam.
        names = {n.attr for n in __import__("ast").walk(node) if isinstance(n, __import__("ast").Attribute) and isinstance(n.value, __import__("ast").Name) and n.value.id == "recv_req"}
        req = NS(**{name: None for name in names})
        req.occ3_worker_descriptor = descriptor(); req.rid = "native-rid"
        state = NS(_occ3_validate_received=lambda r: descriptor(), enable_session_radix_cache=False,
                   metrics_reporter=NS(enable_metrics=False), tokenizer=object(), model_config=NS(hf_eos_token_id=[2], vocab_size=100),
                   disaggregation_mode="none", dllm_config=None,
                   _occ3_reject_received=lambda r, code=None: calls.append("refusal"))
        with patch.dict("sys.modules", {worker.__name__: worker}): fn(state, req)
        self.assertEqual(calls, ["all-rank-admission", "refusal"])

    def test_busy_admission_decline_keeps_its_own_retryable_code(self):
        import ast as _ast
        worker = types.ModuleType("occ3_sglang.latent_worker")
        codes = []

        class QuotaDecline(RuntimeError):
            code = "quota"

        def busy(*args):
            raise QuotaDecline("rank-uniform busy decline")

        def incomplete(*args):
            raise RuntimeError("other admission failure")

        fn = self.sources.function("managers/scheduler.py", "handle_generate_request", "Scheduler",
            get_disagg=lambda: NS(disaggregation_bootstrap_port=1234),
            BeamCoordinator=NS(request_beam_width=lambda req: 1), Req=lambda *a, **kw: NS(), array=array)
        node = self.sources.node("managers/scheduler.py", "handle_generate_request", "Scheduler")
        names = {n.attr for n in _ast.walk(node) if isinstance(n, _ast.Attribute)
                 and isinstance(n.value, _ast.Name) and n.value.id == "recv_req"}

        def state_with():
            req = NS(**{name: None for name in names})
            req.occ3_worker_descriptor = descriptor(); req.rid = "native-rid"
            return NS(_occ3_validate_received=lambda r: descriptor(), enable_session_radix_cache=False,
                     metrics_reporter=NS(enable_metrics=False), tokenizer=object(),
                     model_config=NS(hf_eos_token_id=[2], vocab_size=100),
                     disaggregation_mode="none", dllm_config=None,
                     _occ3_reject_received=lambda r, code=None: codes.append(code)), req

        worker.admit_request = busy
        state, req = state_with()
        with patch.dict("sys.modules", {worker.__name__: worker}): fn(state, req)
        worker.admit_request = incomplete
        state, req = state_with()
        with patch.dict("sys.modules", {worker.__name__: worker}): fn(state, req)
        # The busy decline keeps its own bounded code so the adapter can fold
        # it into the retryable 503; every other admission failure keeps the
        # generic completion-failure code.
        self.assertEqual(codes, ["occ3_busy_batch", "occ3_admission_failed"])
        # The published armed admission window is part of the pinned patch:
        # admission declines immediately against a busy batch and an admitted
        # request may wait at most this long before the queue-bound sweep.
        self.assertIn("min(occ3_descriptor.deadline, time.monotonic() + 5.0)",
                      self.sources.text["managers/scheduler.py"])

    def test_abort_reason_threads_the_sweep_code_and_keeps_user_cancels(self):
        matches = self.sources.function("managers/scheduler.py", "_occ3_abort_matches", "Scheduler")
        reasons = []
        fn = self.sources.function("managers/scheduler.py", "_occ3_abort_attached", "Scheduler")
        armed, armed2, armed3 = Row("armed", True), Row("armed2", True), Row("armed3", True)
        sweep = NS(abort_all=False, rid="armed", abort_message="occ3_admission_timeout")
        user = NS(abort_all=False, rid="armed2", abort_message=None)
        malformed = NS(abort_all=False, rid="armed3", abort_message="occ3_BAD REASON!")
        state = NS(waiting_queue=[armed, armed2, armed3], grammar_manager=NS(grammar_queue=[]),
                   collect_inflight_reqs=lambda: [], chunked_req=None,
                   _occ3_abort_matches=lambda req, recv: matches(req, recv),
                   _occ3_abort_one=lambda req, reason: reasons.append((req.rid, reason)))
        fn(state, sweep)
        fn(state, user)
        fn(state, malformed)
        # The sweep's bounded reason threads through; user and malformed
        # aborts keep the cancelled reason.
        self.assertEqual(reasons, [("armed", "admission_timeout"),
                                   ("armed2", "cancelled"), ("armed3", "cancelled")])
        self.assertEqual(state.waiting_queue, [])

    def test_armed_cap_saturation_keeps_its_own_code(self):
        import ast as _ast
        worker = types.ModuleType("occ3_sglang.latent_worker")
        worker.admit_request = lambda *args: None
        codes = []
        fn = self.sources.function("managers/scheduler.py", "handle_generate_request", "Scheduler",
            get_disagg=lambda: NS(disaggregation_bootstrap_port=1234),
            BeamCoordinator=NS(request_beam_width=lambda req: 1), Req=lambda *a, **kw: NS(), array=array)
        node = self.sources.node("managers/scheduler.py", "handle_generate_request", "Scheduler")
        names = {n.attr for n in _ast.walk(node) if isinstance(n, _ast.Attribute)
                 and isinstance(n.value, _ast.Name) and n.value.id == "recv_req"}
        req = NS(**{name: None for name in names})
        req.occ3_worker_descriptor = descriptor(); req.rid = "native-rid"

        def saturated(recv):
            raise ValueError("occ3_busy")

        state = NS(_occ3_validate_received=saturated, enable_session_radix_cache=False,
                   metrics_reporter=NS(enable_metrics=False), tokenizer=object(),
                   model_config=NS(hf_eos_token_id=[2], vocab_size=100),
                   disaggregation_mode="none", dllm_config=None,
                   _occ3_reject_received=lambda r, code=None: codes.append(code))
        with patch.dict("sys.modules", {worker.__name__: worker}): fn(state, req)
        # Armed-cap saturation is admission unavailability with its own
        # bounded code, distinct from the generic profile refusal.
        self.assertEqual(codes, ["occ3_busy"])

    def test_context_flag_copy_and_mixed_merge_guard(self):
        path = "managers/schedule_batch.py"
        fn = self.sources.function(path, "copy", "ScheduleBatch", ScheduleBatch=lambda **kwargs: NS(**kwargs))
        node = self.sources.node(path, "copy", "ScheduleBatch")
        import ast
        attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"}
        state = NS(**{name: None for name in attrs})
        state.reqs = [Row("x", True)]; state.occ3_force_eager=True; state.occ3_context=("immutable-context",)
        state.forward_iter=17
        copied = fn(state)
        self.assertEqual(copied.occ3_context, state.occ3_context)
        self.assertTrue(copied.occ3_force_eager)
        self.assertEqual(copied.forward_iter,17)
        self.assertIsNot(copied.reqs, state.reqs)
        merge = self.sources.function(path, "merge_batch", "ScheduleBatch")
        with self.assertRaisesRegex(RuntimeError, "mixed_cohort"):
            merge(state, NS(occ3_force_eager=False, reqs=[Row("ordinary")]))
        initialize = self.sources.function("model_executor/forward_batch_info.py", "init_new", "ForwardBatch")
        with self.assertRaisesRegex(RuntimeError, "context_mismatch"):
            initialize(object, NS(occ3_context=None, occ3_force_eager=True), object(), return_hidden_states_before_norm=False)

    def test_chunked_full_budget_is_reserved_not_clipped(self):
        path = "managers/schedule_policy.py"
        swa = self.sources.function(path, "_swa_new_tokens", "PrefillAdder", CLIP_MAX_NEW_TOKENS=4096)
        add = self.sources.function(path, "add_chunked_req", "PrefillAdder", CLIP_MAX_NEW_TOKENS=4096)
        charges = []
        req = NS(occ3_force_eager=True, full_untruncated_fill_ids=list(range(20)), prefix_indices=list(range(4)),
                 output_ids=[], sampling_params=NS(max_new_tokens=10000), retracted_stain=False)
        req.set_extend_range = lambda start, end: setattr(req, "extend_range", NS(start=start,end=end,length=end-start))
        state = NS(rem_total_tokens=5000, per_req_token_overhead=4, dllm_config=None,
                   _swa_new_tokens=lambda r:swa(None,r), can_run_list=[])
        self.assertIs(add(state,req),req)
        self.assertEqual(state.can_run_list,[], "full remaining prompt/output budget must fit before any chunk")
        state.rem_total_tokens=20000
        state.memory_budget=NS(available_chunk_tokens=lambda n:n, fit_chunk=lambda **kw:kw["chunk_limit"])
        state.kv_shard_granule=0; state.prefill_delayer_single_pass=None; state.chunked_req_limit=None
        state._kv_shard_reserve_scratch=lambda **kw:True
        state._update_prefill_budget=lambda *a,**kw:charges.append(a)
        state._mamba_gap_budget_for_req=lambda r:0; state.exact_chunk_fill=False
        for chunk in (4,32):
            state.rem_chunk_tokens=chunk
            result=add(state,req)
            self.assertEqual(charges[-1][2],10000)
            self.assertEqual(result is req,chunk==4)
        req.occ3_force_eager=False; state.rem_chunk_tokens=32
        add(state,req)
        self.assertEqual(charges[-1][2],4096,"ordinary accounting retains its exact old clip")

    def test_initial_chunk_admission_retains_full_output_reservation(self):
        path="managers/schedule_policy.py"
        class AddReqResult: NO_TOKEN=1; OTHER=2
        select=self.sources.function(path,"_select_prefill_admission","PrefillAdder",
            CLIP_MAX_NEW_TOKENS=4096,AddReqResult=AddReqResult,_PrefillAdmission=lambda *a:a)
        req=NS(occ3_force_eager=True,prefix_indices=[],full_untruncated_fill_ids=list(range(20)),
               sampling_params=NS(max_new_tokens=10000))
        state=NS(ceil_paged_tokens=lambda x:x,_check_prefill_budget=lambda *a,**kw:(True,4),
                 rem_chunk_tokens=4,can_run_list=[],exact_chunk_fill=False,dllm_config=None,page_size=4,
                 _swa_new_tokens=lambda r:10000,_check_prefill_tile_budget=lambda n:None)
        self.assertEqual(select(state,req,total_tokens=10024,host_hit_length=0,swa_host_hit_length=0,
                                truncation_align_size=None),(0,4,10000,True))
        req.occ3_force_eager=False
        self.assertEqual(select(state,req,total_tokens=4120,host_hit_length=0,swa_host_hit_length=0,
                                truncation_align_size=None),(0,4,0,True))

    def test_filter_reorder_clears_stale_context_and_preserves_flags(self):
        class IndexSelection(list):
            """Index-selection-only external tensor stub; never Torch math proof."""
            def to(self,*a,**kw):return self
            def __getitem__(self,k):
                return IndexSelection([list.__getitem__(self,i) for i in k]) if isinstance(k,list) else list.__getitem__(self,k)
        nullmode=NS(need_capture=lambda:False)
        fn=self.sources.function("managers/schedule_batch.py","filter_batch","ScheduleBatch",
            strip_beam_tail=lambda s:None,torch=NS(int64="int64",tensor=lambda values,**kw:IndexSelection(values)),
            is_pin_memory_available=lambda dev:False,CaptureHiddenMode=NS(NULL=nullmode),
            get_batch_return_hidden_states_mode=lambda reqs:nullmode,Req=Row)
        rows=[Row("first",True),Row("second",True)]
        for r in rows:r.return_logprob=False;r.grammar=None
        state=NS(reqs=rows,occ3_force_eager=True,occ3_context=("stale-offsets",),device="stub",
                 model_config=NS(is_encoder_decoder=False),multimodal_inputs=None,
                 req_pool_indices=IndexSelection([10,20]),req_pool_indices_cpu=IndexSelection([10,20]),
                 seq_lens=IndexSelection([100,200]),orig_seq_lens=IndexSelection([90,190]),
                 seq_lens_cpu=IndexSelection([100,200]),input_ids=IndexSelection([1,2]),spec_info=None,
                 sampling_info=NS(filter_batch=lambda *a:None))
        fn(state,keep_indices=[1,0])
        self.assertEqual([r.rid for r in state.reqs],["second","first"])
        self.assertEqual(state.req_pool_indices,[20,10])
        self.assertTrue(state.occ3_force_eager);self.assertIsNone(state.occ3_context)
        fn(state,keep_indices=[])
        self.assertFalse(state.occ3_force_eager);self.assertIsNone(state.occ3_context);self.assertEqual(state.reqs,[])

    def test_control_dispatch_validates_bootstraps_then_handles(self):
        worker=types.ModuleType("occ3_sglang.latent_worker");calls=[]
        worker.bootstrap_worker=lambda scheduler:calls.append("bootstrap")
        def control(scheduler,payload):
            calls.append("handle")
            return dict(schema="occ3.control.result.v1",command_id="command",ok=True,result={},error="")
        worker.handle_control=control
        fn=self.sources.function("managers/scheduler.py","handle_occ3_control","Scheduler",
                                 Occ3ControlOutput=lambda **kw:NS(**kw))
        req=NS(rid="command",payload=dict(schema="occ3.control.v1",command_id="command",operation="health",payload={}))
        with patch.dict("sys.modules",{worker.__name__:worker}):
            result=fn(NS(),req)
            self.assertEqual(calls,["bootstrap","handle"]);self.assertTrue(result.payload["ok"])
            calls.clear();req.payload["payload"]={"tensor":object()}
            failed=fn(NS(),req)
            self.assertEqual(calls,[]);self.assertFalse(failed.payload["ok"])
            self.assertEqual(failed.payload["schema"],"occ3.control.result.v1")


class TestUniformCancellation(PinnedCase):
    """Exact native boundary/loops; worker clocks/events are explicit IO stubs."""
    def state(self,*,fence_fails=False):
        from occ3_sglang.latent_types import UniformRequestCancellation
        calls=[];sentinel=object()
        worker=types.ModuleType("occ3_sglang.latent_worker")
        def abort(scheduler,req,reason):
            calls.append("fence")
            if fence_fails:raise RuntimeError("uncertain submitted stream")
            calls.append("retire")
        worker.abort_request=abort
        def free(req,tree,**kw):
            calls.append("free")
            self.assertTrue(req.kv.holds_kv)
            req.kv.holds_kv=False
        reject=self.sources.function("managers/scheduler.py","_occ3_abort_one","Scheduler",
            FINISH_ABORT=Abort,HTTPStatus=HTTPStatus,release_kv_cache=free,_make_abort_req=lambda req:req)
        wrap=self.sources.function("managers/scheduler.py","_occ3_run_batch_or_cancel","Scheduler",
                                   _OCC3_CANCELLED_BATCH=sentinel)
        req=NS(rid="native-rid",occ3_worker_descriptor=descriptor(),occ3_force_eager=True,
               finished_reason=None,to_finish=None,output_ids_through_stop=[],kv=NS(holds_kv=True))
        req.finished=lambda:req.finished_reason is not None
        batch=Batch([req]);batch.batch_is_full=True
        running=Batch([]);running.batch_is_full=True
        state=NS(chunked_req=None,_pending_chunked_abort_req=None,tree_cache=object(),running_batch=running,
                 ipc_channels=NS(send_to_tokenizer=NS(send_output=lambda *a:calls.append("error"))))
        state._occ3_abort_one=types.MethodType(reject,state)
        state._occ3_run_batch_or_cancel=types.MethodType(wrap,state)
        return state,batch,req,worker,calls,sentinel,UniformRequestCancellation

    def test_exact_run_batch_clock_crossings_and_partial_submission_fenced(self):
        from contextlib import nullcontext
        for crossing in ("between-rounds","after-prepare","after-partial-target"):
            with self.subTest(crossing=crossing):
                state,batch,req,worker,calls,sentinel,cancel=self.state()
                clock=[1001.0 if crossing=="between-rounds" else 999.0]
                def prepare(scheduler,batch):
                    calls.append("prepare")
                    if clock[0]>=req.occ3_worker_descriptor.deadline:
                        raise cancel("expired",req.rid,req.occ3_worker_descriptor.attempt_id)
                    if crossing=="after-prepare":clock[0]=1001.0
                def forward(batch,**kwargs):
                    calls.append("forward-boundary")
                    if crossing=="after-partial-target":
                        calls.append("submitted-target-work")
                        clock[0]=1001.0
                    if clock[0]>=req.occ3_worker_descriptor.deadline:
                        raise cancel("expired",req.rid,req.occ3_worker_descriptor.attempt_id)
                    raise AssertionError("test must cancel before native completion")
                worker.prepare_batch=prepare
                run=self.sources.function("managers/scheduler.py","run_batch","Scheduler",
                    time=NS(monotonic=lambda:clock[0]),DisaggregationMode=NS(NULL="none",PREFILL="prefill"),
                    get_parallel=lambda:NS(pp_size=1),resolve_forward_inputs=lambda *a:None)
                batch.forward_mode=NS(is_split_prefill=lambda:False,is_prebuilt=lambda:False,is_extend=lambda:True)
                batch.extend_num_tokens=0;batch.spec_algorithm=NS(is_none=lambda:False)
                state.metrics_reporter=NS(record_scheduler_active=lambda:None)
                state.forward_ct=0;state._sched_idled=False;state.scripted_scheduler_hook=None
                state.profiler_manager=NS(_profile_batch_predicate=lambda b:None);state.forward_sleep_time=None
                state.disaggregation_mode="none";state.is_generation=True;state.enable_overlap=False;state.enable_pdmux=False
                state.pp_group=NS(is_last_rank=True);state.future_map=object()
                state._forward_isolation=lambda *a,**kw:nullcontext()
                state.model_worker=NS(forward_batch_generation=forward)
                state.run_batch=types.MethodType(run,state)
                with patch.dict("sys.modules",{worker.__name__:worker}):
                    self.assertIs(state._occ3_run_batch_or_cancel(batch),sentinel)
                self.assertEqual(calls[-4:],["fence","retire","free","error"])
                self.assertEqual(batch.reqs,[])
                self.assertFalse(batch.batch_is_full)
                self.assertFalse(state.running_batch.batch_is_full,"empty running batch must admit ordinary work again")
                self.assertTrue(req.finished());self.assertFalse(req.kv.holds_kv)
                if crossing=="between-rounds":self.assertNotIn("forward-boundary",calls)
                if crossing=="after-partial-target":
                    self.assertLess(calls.index("submitted-target-work"),calls.index("fence"))

    def test_only_uniform_bound_expired_retired_can_be_recovered(self):
        from occ3_sglang.latent_types import LatentError
        for failure in [LatentError("ranks"),LatentError("expired"),RuntimeError("GPU failure")]:
            state,batch,req,worker,calls,sentinel,cancel=self.state()
            def run(batch):raise failure
            state.run_batch=run
            with patch.dict("sys.modules",{worker.__name__:worker}):
                with self.assertRaises(type(failure)):state._occ3_run_batch_or_cancel(batch)
            self.assertEqual(calls,[]);self.assertTrue(req.kv.holds_kv)
        state,batch,req,worker,calls,sentinel,cancel=self.state()
        def mismatch(batch):raise cancel("expired","different-rid",req.occ3_worker_descriptor.attempt_id)
        state.run_batch=mismatch
        with self.assertRaisesRegex(RuntimeError,"invalid_cancellation"):
            state._occ3_run_batch_or_cancel(batch)
        self.assertEqual(calls,[])

    def test_uncertain_cancellation_fence_stays_fatal_without_free(self):
        state,batch,req,worker,calls,sentinel,cancel=self.state(fence_fails=True)
        def run(batch):raise cancel("retired",req.rid,req.occ3_worker_descriptor.attempt_id)
        state.run_batch=run
        with patch.dict("sys.modules",{worker.__name__:worker}):
            with self.assertRaisesRegex(RuntimeError,"cleanup_uncertain"):
                state._occ3_run_batch_or_cancel(batch)
        self.assertEqual(calls,["fence"])
        self.assertTrue(req.occ3_kv_quarantined);self.assertTrue(req.kv.holds_kv)

    def test_exact_normal_and_overlap_loops_serve_ordinary_after_cancel(self):
        from collections import deque
        for name in ("event_loop_normal","event_loop_overlap"):
            with self.subTest(loop=name):
                state,latent,req,worker,calls,sentinel,cancel=self.state()
                ordinary=Batch([Row("ordinary")]);ordinary.copy=lambda:ordinary
                pending=[latent,ordinary,None];ticks=[];processed=[];result=object()
                def ingest():
                    ticks.append(1)
                    if len(ticks)==3:state.gracefully_exit=True
                def plan(**kw):return NS(running_batch=Batch([]),batch_to_run=pending.pop(0))
                def run(batch):
                    if batch is latent:raise cancel("expired",req.rid,req.occ3_worker_descriptor.attempt_id)
                    return result
                state.run_batch=run;state.gracefully_exit=False;state._engine_paused=False
                state.ingest_requests=ingest;state.get_next_batch_to_run=plan
                state.running_batch=Batch([]);state.last_batch=None;state.is_generation=True;state.enable_unified_memory=False
                state.is_disable_overlap_for_batch=lambda *a,**kw:False
                state._apply_war_barrier=lambda:None
                state.process_batch_result=lambda batch,value:processed.append((batch,value))
                state.launch_batch_sample_if_needed=lambda result,batch:None
                state.on_idle=lambda:None
                loop=self.sources.function("managers/scheduler.py",name,"Scheduler",
                    _OCC3_CANCELLED_BATCH=sentinel,deque=deque,
                    envs=NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda:False)))
                with patch.dict("sys.modules",{worker.__name__:worker}):loop(state)
                self.assertEqual(processed,[(ordinary,result)])
                self.assertEqual(calls,["fence","retire","free","error"])
                self.assertIsNone(state.last_batch)
                if name=="event_loop_overlap":self.assertEqual(len(state.result_queue),0)


class TestAcceptedVisibleSpan(PinnedCase):
    def req(self, old=None, max_new=99, stop_ids=None, stop_strs=None, grammar=None):
        req_path = "managers/schedule_batch.py"
        namespace = dict(FINISH_ABORT=Abort, FINISH_MATCHED_TOKEN=Finish, FINISH_MATCHED_STR=Finish,
                         FINISHED_MATCHED_REGEX=Finish, FINISH_LENGTH=Finish, HTTPStatus=HTTPStatus, re=re)
        method_names = ["finished", "update_finish_state", "_cap_finished_len_at_max_new_tokens",
                        "_check_token_based_finish", "_check_str_based_finish", "_check_vocab_boundary_finish",
                        "_locate_str_stop_finished_len", "_stop_match_tail_len", "tail_str"]
        attrs = {name: self.sources.function(req_path, name, "Req", **namespace) for name in method_names}
        cls = type("ExactRetainedReq", (), attrs)
        req = cls()
        req.output_ids = array("q", old or [])
        req.finished_reason = req.finished_len = req.to_finish = None
        req.sampling_params = NS(ignore_eos=False, stop_token_ids=stop_ids or set(), stop_strs=stop_strs or [],
                                 stop_regex_strs=[], max_new_tokens=max_new, stop_str_max_len=50, stop_regex_max_len=0)
        req.eos_token_ids, req.vocab_size = {99}, 100
        req.tokenizer = NS(eos_token_id=99, additional_stop_token_ids=set(), decode=lambda ids: "".join(chr(96+i) for i in ids))
        req.decoded_text = ""; req.grammar=grammar; req.is_retracted=False; req.beam_group=None
        req.return_sampling_mask=req.return_logprob=req.return_hidden_states=False
        req.time_stats=NS(set_last_decode_finish_time=lambda: None)
        req.kv=NS(kv_committed_len=10, holds_kv=False)
        req.spec_verify_ct=req.spec_num_correct_drafts=req.spec_num_block_accept_tokens=req.spec_num_cap_tokens=0
        req.update_spec_correct_drafts_histogram=lambda x: None
        req.update_spec_cap_lens_histogram=lambda x: None
        req.occ3_force_eager=True; req.rid="native-rid"
        return req

    def run_decode(self, req, accepted, grammar_retained=None, lifecycle=None):
        path="managers/scheduler_components/batch_result_processor.py"
        calls=lifecycle["events"] if lifecycle is not None else []
        worker=types.ModuleType("occ3_sglang.latent_worker")
        worker.abort_request=lambda *a: calls.append(("abort",))
        worker.commit_result=lambda scheduler,r,result,index,start,end,**kw: calls.append((start,end,tuple(r.output_ids[start:end]),kw["terminal"]))
        ns=dict(FINISH_ABORT=Abort, HTTPStatus=HTTPStatus, get_observability=lambda:NS(enable_metrics=False),
                _get_speculative_output_stride=lambda r: r.speculative_output_stride)
        if lifecycle is not None:
            lifecycle["configure_worker"](worker)
            ns.update(release_kv_cache=lifecycle["release"],get_exec=lambda:NS(mamba=NS(enable_mamba_extra_buffer_lazy=False)),
                      get_disagg=lambda:NS(disaggregation_decode_enable_offload_kvcache=False),
                      get_memory=lambda:NS(enable_hisparse=False),BaseSpecWorker=type("ExternalSpecWorker",(),{}),
                      FINISH_MATCHED_TOKEN=Finish)
        methods=["_occ3_settle_result", "_resolve_spec_v2_tokens", "_normalize_decode_outputs", "process_batch_result_decode"]
        attrs={name:self.sources.function(path,name,"SchedulerBatchResultProcessor",**ns) for name in methods}
        attrs["_visible_output_len"]=staticmethod(self.sources.function(path,"_visible_output_len","SchedulerBatchResultProcessor"))
        attrs["_occ3_output_starts"]=classmethod(self.sources.function(path,"_occ3_output_starts","SchedulerBatchResultProcessor"))
        cls=type("ExactResultProcessor",(),attrs); proc=cls()
        proc.scheduler=object(); proc.enable_overlap=True; proc.enable_overlap_mlx=False
        proc.snapshot_auxiliary_output_starts=lambda *a: None
        proc.materialize_sampling_mask_output=lambda *a:None
        proc.model_worker=NS(on_verify_complete_cpu=lambda *a,**kw:calls.append("adaptive"))
        proc.advance_grammar_fsm=lambda result,batch: setattr(result,"grammar_retained_tokens",[grammar_retained or accepted])
        proc.metrics_reporter=NS(num_generated_tokens=0,update_spec_metrics=lambda *a,**kw:None,
                                 report_decode_stats=lambda *a,**kw:None,forward_ct_decode=0)
        proc.token_to_kv_pool_allocator=NS(free_group_begin=lambda:None,free_group_end=lambda:None)
        proc._maybe_update_reasoning_tokens=lambda *a:None
        proc._handle_finish_state_updated_req=lambda *a: None
        proc.output_streamer=NS(stream_output=lambda *a:calls.append("stream"))
        if lifecycle is not None:
            handle=self.sources.function(path,"_handle_finish_state_updated_req","SchedulerBatchResultProcessor",**ns)
            proc._handle_finish_state_updated_req=types.MethodType(handle,proc)
            proc.tree_cache=lifecycle["tree"];proc.draft_worker=object()
            proc._mamba_prefix_cache_update=lambda *a,**kw:None
            for name in ("_maybe_collect_routed_experts","_maybe_collect_indexer_topk","_maybe_collect_customized_info"):
                setattr(proc,name,lambda *a,**kw:None)
            proc.model_worker.prepare_for_kv_cache_release=lambda r:calls.append("native-prepare-release")
            proc.output_streamer=NS(stream_output=lifecycle["stream"])
            req.time_stats.set_completion_time=lambda:None
        result=NS(copy_done=None,auxiliary_host_output=None,routed_experts_output=None,indexer_topk_output=None,
                  logits_output=None,next_token_ids=HostValues(accepted),accept_lens=HostValues([len(accepted)]),
                  speculative_output_stride=len(accepted),num_non_draft_tokens_per_req=1,
                  block_accept_lens=None,cap_lens=None,can_run_cuda_graph=False,occ3_capture_stage=object(),
                  get_num_generated_tokens=lambda bs:len(accepted))
        batch=NS(occ3_force_eager=True,reqs=[req],forward_iter=17,forward_mode=NS(is_extend=lambda:False),
                 spec_algorithm=NS(is_none=lambda:False),return_logprob=False,batch_size=lambda:1,
                 mamba_track_mask_cpu=None)
        with patch.dict("sys.modules",{worker.__name__:worker}): proc.process_batch_result_decode(batch,result)
        return calls,result

    def test_accepted_widths_generated_start_and_terminal_clipping(self):
        for width in (1,2,4,6):
            for stop_index in range(width):
                req=self.req(old=[7,8],stop_ids={stop_index+1})
                calls,result=self.run_decode(req,list(range(1,width+1)))
                commits=[c for c in calls if isinstance(c,tuple)]
                self.assertEqual(commits,[(2,3+stop_index,tuple(range(1,stop_index+2)),True)])
                self.assertEqual(calls[-1],"stream")
                self.assertEqual(calls[0],"adaptive", "adaptive controller must remain invoked")
                self.assertIsNone(result.occ3_capture_stage)

    def test_max_length_beats_stop_after_budget(self):
        req=self.req(old=[7,8],max_new=4,stop_ids={6})
        calls,_=self.run_decode(req,[1,2,3,4,5,6])
        self.assertIn((2,4,(1,2),True),calls)

    def test_stop_string_maps_visible_end_before_commit(self):
        req=self.req(stop_strs=["bc"])
        calls,_=self.run_decode(req,[1,2,3,4,5,6])
        self.assertIn((0,3,(1,2,3),True),calls)

    def test_grammar_retained_span_not_all_verified_tokens(self):
        grammar=NS(is_terminated=lambda: True,finished=False)
        req=self.req(old=[7,8],grammar=grammar)
        calls,_=self.run_decode(req,[1,2,3,4,5,6],grammar_retained=[1,2])
        self.assertIn((2,4,(1,2),True),calls)
        self.assertEqual(req.kv.kv_committed_len,12)

    def test_abort_and_overlap_overshoot_never_commit(self):
        req=self.req(); req.to_finish=Abort()
        calls,_=self.run_decode(req,[1,2,3,4])
        self.assertIn(("abort",),calls)
        self.assertFalse(any(isinstance(c,tuple) and len(c)==4 for c in calls))
        req=self.req(old=[1]); req.finished_reason=Finish(); req.finished_len=1
        calls,_=self.run_decode(req,[2,3,4])
        self.assertFalse(any(isinstance(c,tuple) and len(c)==4 for c in calls))

    def test_middle_prefill_and_retracted_rows_have_no_output_commit(self):
        fn=self.sources.function("managers/scheduler_components/batch_result_processor.py","_occ3_output_starts","SchedulerBatchResultProcessor")
        cls=NS(_visible_output_len=lambda r:len(r.output_ids))
        req=self.req(); req.inflight_middle_chunks=1
        batch=NS(occ3_force_eager=True,reqs=[req],forward_mode=NS(is_extend=lambda:True))
        self.assertEqual(fn(cls,batch),[None])
        req.inflight_middle_chunks=0
        self.assertEqual(fn(cls,batch),[0])
        req.is_retracted=True
        self.assertEqual(fn(cls,batch),[None])

    def run_prefill(self,req,token=1,*,middle=0,lifecycle=None):
        path="managers/scheduler_components/batch_result_processor.py"
        calls=lifecycle["events"] if lifecycle is not None else []
        worker=types.ModuleType("occ3_sglang.latent_worker")
        worker.abort_request=lambda *a:calls.append("abort")
        worker.commit_result=lambda scheduler,r,result,index,start,end,**kw:calls.append((start,end,tuple(r.output_ids[start:end]),kw["terminal"]))
        def release(r,tree,**kw):
            calls.append("free")
            if not r.kv.holds_kv:raise AssertionError("double free")
            r.kv.holds_kv=False
        ns=dict(FINISH_ABORT=Abort,HTTPStatus=HTTPStatus,release_kv_cache=release,
                maybe_cache_unfinished_req=lambda *a,**kw:calls.append("cache"),
                get_memory=lambda:NS(enable_hisparse=False),logger=NS(error=lambda *a:None))
        if lifecycle is not None:
            lifecycle["configure_worker"](worker)
            ns["release_kv_cache"]=lifecycle["release"]
        methods=["_occ3_settle_result","process_batch_result_prefill","_apply_prefill_grammar"]
        attrs={name:self.sources.function(path,name,"SchedulerBatchResultProcessor",**ns) for name in methods}
        attrs["_visible_output_len"]=staticmethod(self.sources.function(path,"_visible_output_len","SchedulerBatchResultProcessor"))
        attrs["_occ3_output_starts"]=classmethod(self.sources.function(path,"_occ3_output_starts","SchedulerBatchResultProcessor"))
        proc=type("ExactPrefillResultProcessor",(),attrs)()
        proc.scheduler=object();proc.is_generation=True;proc.tree_cache=object()
        proc.token_to_kv_pool_allocator=NS(free_group_begin=lambda:None,free_group_end=lambda:None)
        for name in ["snapshot_auxiliary_output_starts","materialize_sampling_mask_output","move_logprobs_to_cpu",
                     "_validate_pp_skip_output_comm","_get_prefill_hidden_capture_mode","_maybe_update_reasoning_tokens",
                     "_maybe_collect_routed_experts","_maybe_collect_indexer_topk","_maybe_collect_customized_info"]:
            setattr(proc,name,lambda *a,**kw:None)
        proc.output_streamer=NS(stream_output=lambda *a:calls.append("stream"))
        req.inflight_middle_chunks=middle
        if lifecycle is None:
            req.kv.holds_kv=True
        else:
            proc.tree_cache=lifecycle["tree"]
            proc.output_streamer=NS(stream_output=lifecycle["stream"])
        req.time_stats=NS(set_prefill_finished_time=lambda:None,set_completion_time=lambda:None,
                          set_last_chunked_prefill_finish_time=lambda:None)
        batch=NS(occ3_force_eager=True,reqs=[req],forward_iter=17,return_hidden_states=False,return_logprob=False,
                 forward_mode=NS(is_extend=lambda:True),decoding_reqs=None,prefill_stats=None,
                 spec_algorithm=NS(is_none=lambda:False))
        result=NS(copy_done=None,auxiliary_host_output=None,routed_experts_output=None,indexer_topk_output=None,
                  logits_output=None,next_token_ids=HostValues([token]),extend_input_len_per_req=None,
                  extend_logprob_start_len_per_req=None,grammar_advanced=False,occ3_capture_stage=object(),can_run_cuda_graph=False)
        with patch.dict("sys.modules",{worker.__name__:worker}):proc.process_batch_result_prefill(batch,result)
        return calls,result

    def test_prefill_first_predictor_pair_and_terminal_eos_max(self):
        req=self.req()
        calls,result=self.run_prefill(req,token=1)
        self.assertEqual(calls,["cache",(0,1,(1,),False),"stream"])
        self.assertIsNone(result.occ3_capture_stage)
        for token,budget in [(99,99),(1,1)]:
            calls,_=self.run_prefill(self.req(max_new=budget),token=token)
            self.assertEqual(calls,["free",(0,1,(token,),True),"stream"])

    def test_prefill_middle_discards_stage_without_output(self):
        req=self.req()
        calls,result=self.run_prefill(req,middle=1)
        self.assertEqual(calls,["stream"])
        self.assertEqual(req.inflight_middle_chunks,0)
        self.assertEqual(list(req.output_ids),[])
        self.assertIsNone(result.occ3_capture_stage)

    def test_prefill_grammar_abort_fences_and_finishes_before_next_forward(self):
        def reject(token):raise ValueError("invalid grammar token")
        req=self.req(grammar=NS(accept_token=reject,is_terminated=lambda:False,finished=False))
        calls,_=self.run_prefill(req)
        self.assertEqual(calls,["cache","abort","free","stream"])
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason,Abort)
        self.assertIsNone(req.to_finish)
        self.assertFalse(req.kv.holds_kv)

    def terminal_fixture(self,req,failure=None):
        """Exact C finish/private-free code; A journal/fence and allocator IO stubs.

        No stream, NCCL, GPU residency or frontend receipt is fabricated as proof.
        The captured terminal state models what a frontend may observe at stream.
        """
        from test_latent_cache_patch import Cache,Kv,Indices
        events=[];observed=[];journal={"state":"injected"}
        tree=Cache(page=4)
        req.kv=Kv(committed=3,allocated=12);req.prefix_indices=Indices(range(3))
        req.occ3_worker_descriptor=descriptor("consume");req.occ3_scheduler=object()
        req.occ3_private_prefix=req.occ3_private_kv=req.skip_radix_cache_insert=True
        req.occ3_kv_released=req.occ3_kv_quarantined=req.occ3_kv_release_started=False
        req.multimodal_inputs=req.session=None
        actual_page_free=tree.free_kv_row;actual_row_free=tree.req_to_token_pool.free
        def page_free(kv,ranges):
            events.append("allocator-free");actual_page_free(kv,ranges)
        def row_free(r):
            events.append("row-free");actual_row_free(r)
        tree.free_kv_row=page_free;tree.req_to_token_pool.free=row_free
        private=self.sources.function("mem_cache/common.py","occ3_release_private_kv")
        release=self.sources.function("mem_cache/common.py","release_kv_cache",occ3_release_private_kv=private)
        def configure(worker):
            def prefree(scheduler,r):
                events.append("pre-free-fence")
                if failure=="pre-free-fence":raise RuntimeError("mandatory fence failed")
            def abort(scheduler,r,reason):
                events.append("cleanup-fence")
                journal["state"]="retired"
            def commit(scheduler,r,result,index,start,end,*,terminal,forward_iter):
                self.assertTrue(terminal);self.assertEqual(forward_iter,17)
                self.assertFalse(r.kv.holds_kv,"test observes original native free-before-settle order")
                self.assertTrue(r.occ3_kv_released)
                events.append("terminal-validation")
                if failure=="retired":
                    from occ3_sglang.latent_types import UniformRequestCancellation
                    raise UniformRequestCancellation("retired",r.rid,r.occ3_worker_descriptor.attempt_id)
                if failure=="terminal-validation":raise RuntimeError("lineage rejected")
                events.append("terminal-fence")
                if failure=="terminal-fence":raise RuntimeError("later terminal fence failed")
                journal["state"]="terminal";events.append("terminal-journal")
            worker.fence_private_kv_release=prefree;worker.abort_request=abort;worker.commit_result=commit
        def stream(*args):
            observed.append((journal["state"],isinstance(req.finished_reason,Abort),req.kv.holds_kv))
            events.append("stream")
        return dict(events=events,observed=observed,journal=journal,tree=tree,release=release,
                    configure_worker=configure,stream=stream)

    def test_exact_terminal_validation_failure_has_no_clean_output_or_doublefree(self):
        for mode in ("prefill","decode"):
            with self.subTest(mode=mode):
                req=self.req(max_new=1);life=self.terminal_fixture(req,"terminal-validation")
                if mode=="prefill":self.run_prefill(req,lifecycle=life)
                else:self.run_decode(req,[1,2,3],lifecycle=life)
                e=life["events"]
                self.assertLess(e.index("pre-free-fence"),e.index("allocator-free"))
                self.assertLess(e.index("row-free"),e.index("terminal-validation"))
                self.assertEqual(life["observed"],[("retired",True,False)])
                self.assertNotIn("terminal-journal",e)
                self.assertEqual(life["tree"].req_to_token_pool.frees,1)
                life["release"](req,life["tree"])
                self.assertEqual(e.count("allocator-free"),1)
                self.assertFalse(req.occ3_kv_quarantined,"already positively fenced/freed slots are not claimed quarantined")

    def test_later_terminal_fence_failure_after_positive_prefree_never_receipts(self):
        for mode in ("prefill","decode"):
            with self.subTest(mode=mode):
                req=self.req(max_new=1);life=self.terminal_fixture(req,"terminal-fence")
                if mode=="prefill":self.run_prefill(req,lifecycle=life)
                else:self.run_decode(req,[1,2,3],lifecycle=life)
                e=life["events"]
                self.assertLess(e.index("pre-free-fence"),e.index("allocator-free"))
                self.assertLess(e.index("allocator-free"),e.index("terminal-fence"))
                self.assertEqual(life["observed"],[("retired",True,False)])
                self.assertNotIn("terminal-journal",e)
                self.assertEqual(e.count("pre-free-fence"),1)
                life["release"](req,life["tree"])
                self.assertEqual(e.count("allocator-free"),1)

    def test_positive_terminal_journal_precedes_stream_after_safe_free(self):
        for mode in ("prefill","decode"):
            with self.subTest(mode=mode):
                req=self.req(max_new=1);life=self.terminal_fixture(req)
                if mode=="prefill":self.run_prefill(req,lifecycle=life)
                else:self.run_decode(req,[1,2,3],lifecycle=life)
                e=life["events"]
                self.assertLess(e.index("pre-free-fence"),e.index("allocator-free"))
                self.assertLess(e.index("allocator-free"),e.index("terminal-validation"))
                self.assertLess(e.index("terminal-fence"),e.index("terminal-journal"))
                self.assertLess(e.index("terminal-journal"),e.index("stream"))
                self.assertEqual(life["observed"],[("terminal",False,False)])

    def test_failed_mandatory_prefree_fence_never_reuses_allocator(self):
        for mode in ("prefill","decode"):
            with self.subTest(mode=mode):
                req=self.req(max_new=1);life=self.terminal_fixture(req,"pre-free-fence")
                with self.assertRaisesRegex(RuntimeError,"quarantined"):
                    if mode=="prefill":self.run_prefill(req,lifecycle=life)
                    else:self.run_decode(req,[1,2,3],lifecycle=life)
                self.assertTrue(req.kv.holds_kv);self.assertTrue(req.occ3_kv_quarantined)
                self.assertNotIn("allocator-free",life["events"])
                self.assertEqual(life["observed"],[])

    def test_retirement_after_worker_before_result_settle_has_no_clean_output(self):
        for mode in ("prefill","decode"):
            with self.subTest(mode=mode):
                req=self.req(max_new=1);life=self.terminal_fixture(req,"retired")
                if mode=="prefill":self.run_prefill(req,lifecycle=life)
                else:self.run_decode(req,[1,2,3],lifecycle=life)
                self.assertEqual(life["observed"],[("retired",True,False)])
                self.assertNotIn("terminal-journal",life["events"])
                self.assertEqual(life["events"].count("allocator-free"),1)
                life["release"](req,life["tree"])
                self.assertEqual(life["events"].count("allocator-free"),1)

    def test_actual_A_stale_stage_iteration_is_refused_before_any_tensor_use(self):
        from occ3_sglang import latent_worker as worker
        calls=[]
        class NoTensorUse:
            def __getattr__(self,name):raise AssertionError("stale stage reached tensor operation: "+name)
        runtime=NS(torch=NoTensorUse(),poisoned=False)
        runtime.resolve=lambda value:NS(accepted_count=0)
        def gather(value):
            calls.append((value["phase"],value["error"]))
            return [dict(value) for _ in range(8)] # Explicit bounded all-rank IO stub, not NCCL proof.
        runtime.gather=gather
        runtime.preflight=types.MethodType(worker.WorkerRuntime.preflight,runtime)
        req=self.req();req.output_ids.append(1);req.occ3_worker_descriptor=descriptor()
        req.kv.holds_kv=False
        stage=NS(iteration=16,committed=False)
        result=NS(occ3_capture_stage=stage)
        batch=NS(reqs=[req],forward_iter=17)
        proc=NS(scheduler=object(),tree_cache=object(),_visible_output_len=lambda r:len(r.output_ids))
        settle=self.sources.function("managers/scheduler_components/batch_result_processor.py",
            "_occ3_settle_result","SchedulerBatchResultProcessor",FINISH_ABORT=Abort,HTTPStatus=HTTPStatus)
        with patch.object(worker,"_runtime",return_value=runtime),patch.object(worker,"abort_request",side_effect=lambda *a:calls.append("abort")):
            settle(proc,batch,result,[0])
        self.assertEqual(calls,[("commit_lineage","lineage"),"abort"])
        self.assertFalse(stage.committed)
        self.assertIsNone(result.occ3_capture_stage)
        self.assertIsInstance(req.finished_reason,Abort)


if __name__=="__main__":
    unittest.main()
