"""Capture reservation (hold): worker policy and exact patched scheduler seams.

The scheduler loop in this file is a small model. Forward passes, KV and
tokens are deterministic stand-ins. The exact pinned sources are only the
named patched methods: handle_generate_request, _occ3_prefill_selection,
_poll_timeout_aborts, abort_request, _occ3_abort_attached,
_occ3_reject_received, _occ3_capture_hold_retract, pause_generation,
continue_generation, collect_inflight_reqs, _add_request_to_queue,
_abort_on_queued_limit, _set_or_validate_priority, and schedule_batch
retract_all, release_req and Req.reset_for_retract. Admission and the hold
policy are the real latent_worker code. Ranks are threads that meet in a
barrier for every gather, so an unmatched collective fails the test.

This is not a SGLang, GPU, NCCL or TP8 runtime proof.
"""
from __future__ import annotations

import ast
import copy
import logging
import sys
import threading
import time as real_time
import types
from array import array
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_types, latent_worker as w
from occ3_sglang.latent_types import LatentError
from test_latent_ipc_patch import AppliedSource
from test_latent_types import capture

SCHED = "managers/scheduler.py"
BATCH = "managers/schedule_batch.py"
INFLIGHT = "latent_worker.admit_request.capture_inflight"
DT = 0.05  # model seconds per scheduler iteration
EXACT = ("handle_generate_request", "_occ3_prefill_selection", "_poll_timeout_aborts",
         "abort_request", "_occ3_abort_attached", "_occ3_reject_received",
         "_occ3_capture_hold_retract", "pause_generation", "continue_generation",
         "collect_inflight_reqs", "_add_request_to_queue", "_abort_on_queued_limit",
         "_set_or_validate_priority")


@pytest.fixture(scope="module")
def sources():
    applied = AppliedSource()
    yield applied
    applied.close()


class RankTime:
    """time module stand-in for latent_worker: each rank thread has its own clock."""
    def __init__(self):
        self.local = threading.local()

    def monotonic(self):
        return self.local.clock.now

    def __getattr__(self, name):
        return getattr(real_time, name)


class Clock:
    def __init__(self, skew):
        self.now = 100.0 + skew

    def monotonic(self):
        return self.now

    perf_counter = monotonic


class Bus:
    """all_gather stand-in. Every rank must enter the same collective."""
    def __init__(self, ranks):
        self.slots = [None] * ranks
        self.result = None
        self.calls = 0
        self.barrier = threading.Barrier(ranks, action=self._snapshot, timeout=20)

    def _snapshot(self):
        self.result = list(self.slots)
        self.calls += 1

    def gather(self, rank, value):
        self.slots[rank] = value
        self.barrier.wait()
        return list(self.result)


class Runtime:
    poisoned = False

    def __init__(self, bus, rank):
        self.bus, self.rank = bus, rank
        self.record = NS(state="reserved", admitted_capture=False, state_generation=123)
        self.lie = None

    def resolve(self, value, **kw):
        return self.record

    def gather(self, value):
        return self.bus.gather(self.rank, value if self.lie is None else self.lie(value))

    def preflight(self, phase, check, *, descriptor=None):
        value, error, declined = None, "", None
        try:
            value = check()
        except LatentError as exc:
            declined, error = exc, exc.code
        outcomes = self.gather({"phase": phase, "binding": value, "error": error})
        if any(item != outcomes[0] for item in outcomes):
            raise LatentError("ranks")
        if error:
            raise declined
        return value


class Abort:
    def __init__(self, message="abort", status_code=None):
        self.message = message


class AbortReq:
    def __init__(self, rid="", abort_all=False, finished_reason=None, abort_message=None):
        self.rid, self.abort_all = rid, abort_all
        self.finished_reason, self.abort_message = finished_reason, abort_message


class Batch:
    def __init__(self, reqs, extend=False):
        self.reqs = list(reqs)
        self.batch_is_full = False
        self.forward_mode = NS(is_extend=lambda: extend)

    @property
    def occ3_force_eager(self):
        return any(r.occ3_force_eager for r in self.reqs)

    def filter_batch(self):
        self.reqs = [r for r in self.reqs if not r.finished()]


def token(rid, position):
    return (sum(map(ord, rid)) * 31 + position * 7) % 9973


def reference(rid, count):
    return [token(rid, position) for position in range(count)]


def generate(rid, new_tokens, *, descriptor=None, chunks=1, input_embeds=None):
    """Tokenized request stand-in. Every field the seam reads is filled in make_rank."""
    return NS(rid=rid, new_tokens=new_tokens, occ3_worker_descriptor=descriptor,
              chunks=chunks, input_embeds=input_embeds)


def make_row_class(sources):
    reset = sources.function(BATCH, "reset_for_retract", "Req",
                             torch=NS(empty=lambda *a, **kw: [], int64="int64"),
                             DecLockRefParams=lambda: None, array=array,
                             truncate_weight_version_events=lambda events, **kw: events)

    class Row:
        """Request row. reset_for_retract is the exact pinned Req method."""
        reset_for_retract = reset

        def __init__(self, rid, input_text, input_ids, sampling_params, **kw):
            self.rid = rid
            self.origin_input_ids = array("q", input_ids)
            self.origin_input_ids_unpadded = self.origin_input_ids
            self.full_untruncated_fill_ids = self.origin_input_ids
            self.sampling_params = sampling_params
            self.output_ids = array("q")
            self.finished_reason = self.to_finish = None
            self.kv = NS(holds_kv=False, holds_mamba=False, cache_protected_len=0, kv_committed_len=0)
            self.session = self.multimodal_inputs = self.positional_embed_overrides = None
            self.occ3_worker_descriptor = None
            self.occ3_attached = self.occ3_force_eager = self.occ3_private_prefix = False
            self.input_embeds = kw.get("input_embeds")
            self.beam_group = None
            self.retraction_count = 0
            self.is_retracted = self.is_prefill_only = False
            self.return_logprob = self.return_sampling_mask = False
            self.priority = kw.get("priority")
            self.extend_range = NS(end=len(self.origin_input_ids))
            self.time_stats = NS(set_wait_queue_entry_time=lambda: None, wait_queue_entry_time=0.0)
            self.cache_request_handle = None
            self.weight_version_events = None
            self.send_token_offset = 0
            self.chunks = self.chunks_left = 1

        def finished(self):
            return self.finished_reason is not None

    return Row


class Rank:
    """One TP rank: exact patched decision methods around a model loop."""
    def __init__(self, sources, bus, index, rank_time, *, max_queued=None):
        self.index, self.rank_time = index, rank_time
        self.clock = Clock(skew=index * 3.7)  # rank clocks differ on purpose
        self.occ3_runtime = Runtime(bus, index)
        self.sent, self.freed, self.trace = [], [], []
        self.waiting_queue, self.chunked_req = [], None
        self.running_batch, self.last_batch = Batch([]), None
        self.result_queue, self.cur_batch_for_debug = [], None
        self._engine_paused = False
        self._pending_chunked_abort_req = None
        self.max_queued_requests = max_queued
        self.processed_tokens_counter = 0
        self.enable_overlap = self.enable_session_radix_cache = False
        self.enable_priority_scheduling = self.abort_on_priority_when_disabled = False
        self.disaggregation_mode = "null"
        self.dllm_config = self.mm_receiver = self.decode_offload_manager = None
        self.hisparse_coordinator = None
        self.req_to_token_pool = self.token_to_kv_pool_allocator = object()
        self.tokenizer, self.max_req_input_len = object(), 1 << 20
        self.model_config = NS(hf_eos_token_id=[2], vocab_size=100)
        self.metrics_collector = None
        self.metrics_reporter = NS(enable_metrics=False, last_gen_throughput=0.0,
                                   current_scheduler_metrics_enabled=False)
        self.kv_events_publisher = NS(publish_kv_events=lambda: None)
        self.tree_cache = NS(cache_controller=None)
        self.spec_algorithm = NS(is_dflash_family=lambda: False, is_uno=lambda: False,
                                 is_none=lambda: True)
        self.grammar_manager = NS(grammar_queue=[], process_req_with_grammar=lambda req: False,
                                  abort_requests=lambda recv: None)
        self.beam_coordinator = NS(retire_group=lambda req: None)
        self.ipc_channels = NS(send_to_tokenizer=NS(send_output=self._send))
        self.Row = make_row_class(sources)
        self._bind(sources)

    def _bind(self, sources):
        mode = NS(NULL="null", PREFILL="prefill", DECODE="decode")
        release_req = sources.function(
            BATCH, "release_req", get_disagg=lambda: NS(disaggregation_mode="null"),
            release_kv_cache=self._release_kv, evict_from_tree_cache=lambda tree, count: None,
            envs=NS(SGLANG_RETRACT_DECODE_STEPS=NS(get=lambda: 20)))
        retract_all = sources.function(BATCH, "retract_all", release_req=release_req)
        namespace = dict(
            AbortReq=AbortReq, HTTPStatus=HTTPStatus, DisaggregationMode=mode, time=self.clock,
            array=array, sys=sys, logger=logging.getLogger("occ3.sim"), FINISH_ABORT=Abort,
            Req=self._row, BeamCoordinator=NS(request_beam_width=lambda recv: 1),
            envs=NS(SGLANG_REQ_WAITING_TIMEOUT=NS(get=lambda: 0),
                    SGLANG_REQ_RUNNING_TIMEOUT=NS(get=lambda: 0),
                    SGLANG_RUST_SERVER=NS(get=lambda: False)),
            get_parallel=lambda: NS(pp_size=1, pp_rank=0),
            get_disagg=lambda: NS(disaggregation_bootstrap_port=1),
            get_serving=lambda: NS(allow_auto_truncate=False),
            get_device=lambda: NS(mlx_enable_sampling=False),
            validate_input_length=lambda *a: None, release_kv_cache=self._release_kv,
            retract_all=retract_all,
            _make_abort_req=lambda req, finished_reason=None: AbortReq(
                rid=req.rid, finished_reason=finished_reason),
            PauseGenerationReqInput=lambda **kw: NS(**kw),
            ContinueGenerationReqInput=lambda **kw: NS(**kw))
        for name in EXACT:
            setattr(self, name, types.MethodType(
                sources.function(SCHED, name, "Scheduler", **namespace), self))
        self._occ3_abort_matches = sources.function(SCHED, "_occ3_abort_matches", "Scheduler")
        node = sources.node(SCHED, "handle_generate_request", "Scheduler")
        self.recv_fields = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == "recv_req"}

    # Stubs for scheduler parts outside this change.
    def _occ3_validate_received(self, recv_req):
        return latent_types.decode_descriptor(recv_req.occ3_worker_descriptor)

    def _occ3_abort_one(self, req, reason):
        raise AssertionError("an OCC-3 row must never be aborted by the reservation: " + reason)

    def _release_aborted_request(self, req):
        raise AssertionError("an ordinary row must never be dropped by the reservation")

    def _maybe_namespace_elastic_radix_cache(self, req):
        return None

    def validate_dllm_request(self, req):
        return None

    def init_req_max_new_tokens(self, req):
        return None

    def _prefetch_kvcache(self, req):
        return None

    def process_batch_result(self, batch, result):
        raise AssertionError("no overlap result is pending in this model")

    def _row(self, rid, input_text, input_ids, sampling_params, **kw):
        return self.Row(rid, input_text, input_ids, sampling_params, **kw)

    def _send(self, output, context):
        self.sent.append(("abort", output.rid, (output.finished_reason or {}).get("message")))

    def _release_kv(self, req, tree, **kw):
        assert req.kv.holds_kv, "double free"
        req.kv.holds_kv = False
        self.freed.append(req.rid)

    def recv(self, request):
        """This rank's copy of the broadcast request, as the scheduler receives it."""
        fields = {name: None for name in self.recv_fields}
        fields.update(rid=request.rid, input_ids=[1, 2, 3], input_embeds=request.input_embeds,
                      sampling_params=NS(max_new_tokens=request.new_tokens),
                      occ3_worker_descriptor=request.occ3_worker_descriptor,
                      logprob_start_len=-1, return_logprob=False, stream=True,
                      chunks=request.chunks)
        return NS(**fields)

    def poll(self):
        """Leader only: the one place a clock decides for a reservation."""
        return self._poll_timeout_aborts()

    def step(self, items):
        self.rank_time.local.clock = self.clock
        for item in items:
            if isinstance(item, AbortReq):
                self.abort_request(copy.copy(item))
            else:
                recv = self.recv(item) if not hasattr(item, "sampling_params") else item
                before = len(self.waiting_queue)
                self.handle_generate_request(recv)
                for row in self.waiting_queue[before:]:
                    if row.rid == recv.rid and getattr(recv, "chunks", 1) > 1:
                        row.chunks = row.chunks_left = recv.chunks
        self._run()

    # Model of get_next_batch_to_run, run_batch and process_batch_result.
    def _run(self):
        last = self.last_batch
        if last is not None and last.forward_mode.is_extend():
            for row in last.reqs:
                if (not row.finished() and row is not self.chunked_req
                        and row not in self.running_batch.reqs):
                    self.running_batch.reqs.append(row)
        self.running_batch.filter_batch()
        rid, drain = self._occ3_prefill_selection(self.running_batch)
        new = []
        if not (drain and self.chunked_req is None):
            if self.chunked_req is not None:
                new.append(self.chunked_req)
            for row in list(self.waiting_queue):
                if drain or (rid is not None and row.rid != rid):
                    continue
                self.waiting_queue.remove(row)
                new.append(row)
        if new:
            batch = Batch(new, extend=True)
            for row in new:
                self._prefill(row)
        elif self.running_batch.reqs:
            batch = self.running_batch
            for row in batch.reqs:
                self._emit(row)
        else:
            batch = None
        if batch is not None:
            eager = [row.occ3_force_eager for row in batch.reqs]
            assert all(eager) or not any(eager), "capture cohort must stay isolated"
            assert len(batch.reqs) == 1 or not any(eager), "capture runs alone"
        self.trace.append(tuple(row.rid for row in batch.reqs) if batch else ())
        self.last_batch = batch

    def _prefill(self, row):
        if row.is_retracted:
            row.is_retracted, row.chunks_left = False, row.chunks
        row.kv.holds_kv = True
        if row.chunks_left > 1:
            row.chunks_left -= 1
            self.chunked_req = row
            return
        if self.chunked_req is row:
            self.chunked_req = None
        self._emit(row)

    def _emit(self, row):
        value = token(row.rid, len(row.output_ids))
        row.output_ids.append(value)
        self.sent.append(("token", row.rid, value))
        if len(row.output_ids) >= row.sampling_params.max_new_tokens:
            row.finished_reason = "length"
            self._release_kv(row, self.tree_cache)
            self.sent.append(("done", row.rid, None))

    def rows(self):
        seen = [*self.waiting_queue, *self.running_batch.reqs,
                *(self.last_batch.reqs if self.last_batch else [])]
        if self.chunked_req is not None:
            seen.append(self.chunked_req)
        return {id(row): row for row in seen}.values()

    def view(self):
        hold = getattr(self, "occ3_capture_hold", None)
        return (tuple(row.rid for row in self.waiting_queue),
                tuple(sorted((row.rid, tuple(row.output_ids)) for row in self.running_batch.reqs)),
                None if self.chunked_req is None else self.chunked_req.rid,
                None if hold is None else (hold.rid, hold.seq),
                tuple(self.sent), tuple(self.freed), tuple(self.trace),
                (self.occ3_runtime.record.state, self.occ3_runtime.record.admitted_capture))


class World:
    def __init__(self, sources, monkeypatch, ranks=2, **kw):
        self.bus = Bus(ranks)
        self.rank_time = RankTime()
        monkeypatch.setattr(w, "time", self.rank_time)
        monkeypatch.setattr(w, "_runtime", lambda scheduler=None: scheduler.occ3_runtime)
        self.ranks = [Rank(sources, self.bus, index, self.rank_time, **kw) for index in range(ranks)]
        self.pool = ThreadPoolExecutor(ranks)
        self.elapsed = 0.0

    @property
    def leader(self):
        return self.ranks[0]

    def tick(self, *incoming):
        local = self.leader.poll()
        items = [*incoming, *local]
        for future in [self.pool.submit(rank.step, items) for rank in self.ranks]:
            future.result()
        for rank in self.ranks:
            rank.clock.now += DT
        self.elapsed += DT
        views = [rank.view() for rank in self.ranks]
        assert all(view == views[0] for view in views), "ranks left lockstep"
        return local

    def run(self, seconds, *, each=None):
        for _ in range(round(seconds / DT)):
            self.tick(*(each(self) if each else ()))

    def close(self):
        self.pool.shutdown(wait=False)

    def stream(self, rid):
        return [value for kind, r, value in self.leader.sent if kind == "token" and r == rid]

    def refusals(self, rid):
        return [value for kind, r, value in self.leader.sent if kind == "abort" and r == rid]

    def done(self, rid):
        return ("done", rid, None) in self.leader.sent

    def running(self, rid):
        return any(rid in step for step in self.leader.trace)


@pytest.fixture
def world(sources, monkeypatch):
    for name in ("OCC3_CAPTURE_BUSY_WAIT_SECONDS", "OCC3_CAPTURE_DRAIN_SECONDS", "OCC3_CAPTURE_PREEMPT"):
        monkeypatch.delenv(name, raising=False)
    made = []

    def make(**kw):
        made.append(World(sources, monkeypatch, **kw))
        return made[-1]

    yield make
    for item in made:
        item.close()


def test_quiet_scheduler_admits_at_once_without_a_reservation(world):
    sim = world()
    sim.tick(generate("rid", 4, descriptor=capture()))
    assert sim.bus.calls == 1  # the admission preflight only
    assert getattr(sim.leader, "occ3_capture_hold", None) is None
    assert sim.leader.occ3_runtime.record.state == "capturing"
    sim.run(0.5)
    assert sim.stream("rid") == reference("rid", 4) and sim.done("rid")
    assert sim.refusals("rid") == []
