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
from test_latent_registry import consume
from test_latent_types import capture

SCHED = "managers/scheduler.py"
BATCH = "managers/schedule_batch.py"
INFLIGHT = "latent_worker.admit_request.capture_inflight"
DT = 0.0625  # model seconds per scheduler iteration, exact in binary
KNOBS = ("OCC3_CAPTURE_BUSY_WAIT_SECONDS", "OCC3_CAPTURE_DRAIN_SECONDS", "OCC3_CAPTURE_PREEMPT",
         "SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT")
EXACT = ("handle_generate_request", "_occ3_prefill_selection", "_poll_timeout_aborts",
         "abort_request", "_occ3_abort_attached", "_occ3_reject_received",
         "_occ3_capture_hold_retract", "pause_generation", "continue_generation",
         "collect_inflight_reqs", "_add_request_to_queue", "_abort_on_queued_limit",
         "_set_or_validate_priority")


def ticks(seconds):
    return round(seconds / DT)


class Exact:
    """Exact methods of the applied sources, extracted like AppliedSource.function.
    Each file is parsed once and each method is compiled once."""
    def __init__(self, applied):
        self.applied, self.text, self.trees, self.codes = applied, applied.text, {}, {}

    def node(self, path, name, scope=None):
        if path not in self.trees:
            self.trees[path] = ast.parse(self.text[path])
        body = self.trees[path].body
        if scope:
            body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == scope).body
        return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)

    def function(self, path, name, scope=None, **namespace):
        key = (path, name, scope)
        if key not in self.codes:
            node = copy.deepcopy(self.node(path, name, scope))
            node.decorator_list = []
            module = ast.Module(body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                type_ignores=[])
            self.codes[key] = compile(ast.fix_missing_locations(module), f"{path}:{node.lineno}", "exec")
        scope_ = dict(namespace)
        exec(self.codes[key], scope_)
        return scope_[name]


@pytest.fixture(scope="module")
def sources():
    applied = AppliedSource()
    yield Exact(applied)
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
        self.log = []
        self.barrier = threading.Barrier(ranks, action=self._snapshot, timeout=20)

    def _snapshot(self):
        self.result = list(self.slots)
        self.log.append(self.result)

    def gather(self, rank, value):
        self.slots[rank] = value
        self.barrier.wait()
        return list(self.result)

    def holds(self):
        return [values for values in self.log if "hold" in values[0]]


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
    """Tokenized request stand-in. Every field the seam reads is filled in Rank.recv."""
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
    def __init__(self, sources, bus, index, rank_time, *, skew=3.75, max_queued=None,
                 allow_drop=False):
        self.index, self.rank_time, self.allow_drop = index, rank_time, allow_drop
        self.clock = Clock(skew=index * skew)  # rank clocks differ on purpose
        self.occ3_runtime = Runtime(bus, index)
        self.sent, self.freed, self.trace, self.made, self.dropped = [], [], [], [], []
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
        if not self.allow_drop:
            raise AssertionError("an ordinary row must never be dropped by the reservation")
        self.dropped.append(req.rid)

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
        row = self.Row(rid, input_text, input_ids, sampling_params, **kw)
        self.made.append(row)
        return row

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
                continue
            recv = self.recv(item)
            before = len(self.made)
            self.handle_generate_request(recv)
            for row in self.made[before:]:
                row.chunks = row.chunks_left = item.chunks
        if not self._engine_paused:
            self._run()

    # Model of get_next_batch_to_run, run_batch and process_batch_result.
    def _run(self):
        last = self.last_batch
        if last is not None and last.forward_mode.is_extend():
            for row in last.reqs:
                if (not row.finished() and row is not self.chunked_req
                        and row not in self.running_batch.reqs):
                    self.running_batch.reqs.append(row)
        for row in self.running_batch.reqs:
            if row.to_finish is not None and not row.finished():
                row.finished_reason = row.to_finish
                self._release_kv(row, self.tree_cache)
                self.sent.append(("aborted", row.rid, None))
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

    def row(self, rid):
        return next(row for row in reversed(self.made) if row.rid == rid)

    def busy(self):
        rows = [*self.collect_inflight_reqs()]
        if self.chunked_req is not None:
            rows.append(self.chunked_req)
        return any(not row.finished() for row in rows)

    def view(self):
        hold = getattr(self, "occ3_capture_hold", None)
        return (tuple(row.rid for row in self.waiting_queue),
                tuple(sorted((row.rid, tuple(row.output_ids)) for row in self.running_batch.reqs)),
                None if self.chunked_req is None else self.chunked_req.rid,
                None if hold is None else (hold.rid, hold.seq),
                tuple(self.sent), tuple(self.freed), tuple(self.trace),
                tuple((row.rid, row.retraction_count) for row in self.made),
                (self.occ3_runtime.record.state, self.occ3_runtime.record.admitted_capture))


class World:
    def __init__(self, sources, monkeypatch, ranks=2, **kw):
        self.bus = Bus(ranks)
        self.rank_time = RankTime()
        monkeypatch.setattr(w, "time", self.rank_time)
        monkeypatch.setattr(w, "_runtime", lambda scheduler=None: scheduler.occ3_runtime)
        self.ranks = [Rank(sources, self.bus, index, self.rank_time, **kw) for index in range(ranks)]
        self.pool = ThreadPoolExecutor(ranks)
        self.ticks, self.busy, self.signals = 0, [], []

    @property
    def leader(self):
        return self.ranks[0]

    def tick(self, *incoming):
        local = self.leader.poll()
        self.signals.extend((self.ticks, item.abort_message) for item in local)
        items = [*incoming, *local]
        for future in [self.pool.submit(rank.step, items) for rank in self.ranks]:
            future.result()
        for rank in self.ranks:
            rank.clock.now += DT
        self.ticks += 1
        self.busy.append(self.leader.busy())
        views = [rank.view() for rank in self.ranks]
        assert all(view == views[0] for view in views), "ranks left lockstep"

    def run(self, seconds, *sources_):
        for _ in range(ticks(seconds)):
            self.tick(*[item for source in sources_ for item in source(self)])

    def close(self):
        self.pool.shutdown(wait=False)

    def each(self, action):
        for rank in self.ranks:
            action(rank)

    def stream(self, rid):
        return [value for kind, r, value in self.leader.sent if kind == "token" and r == rid]

    def refusals(self, rid):
        return [value for kind, r, value in self.leader.sent if kind == "abort" and r == rid]

    def done(self, rid):
        return ("done", rid, None) in self.leader.sent

    def first_run(self, rid):
        """Iteration of the first forward that holds this request."""
        return next((at for at, step in enumerate(self.leader.trace) if rid in step), None)

    def last_run(self, rid):
        return max((at for at, step in enumerate(self.leader.trace) if rid in step), default=None)

    def record(self):
        return self.leader.occ3_runtime.record

    def assert_clean(self):
        """No leaked hold, pause, queue slot or KV after the scenario."""
        for rank in self.ranks:
            assert getattr(rank, "occ3_capture_hold", None) is None
            assert not getattr(rank, "occ3_capture_hold_replay", False)
            assert not rank._engine_paused and rank.chunked_req is None
            assert rank.waiting_queue == [] and not rank.busy()
            assert not any(row.kv.holds_kv for row in rank.made)
            for rid in {row.rid for row in rank.made}:
                rows = [row for row in rank.made if row.rid == rid]
                # One release per retraction and one at the end of a row that ran.
                assert rank.freed.count(rid) == sum(
                    row.finished() + row.retraction_count for row in rows)


def load(every, tokens, *, start=0, count=None, prefix="load"):
    """Ordinary traffic: one request every `every` seconds, `tokens` tokens each."""
    period = ticks(every)

    def source(sim):
        index, late = divmod(sim.ticks - ticks(start), period)
        if index < 0 or late or (count is not None and index >= count):
            return []
        return [generate(f"{prefix}-{index}", tokens)]

    return source


def once(at, *requests):
    return lambda sim: list(requests) if sim.ticks == ticks(at) else []


class Sender:
    """Frontend stand-in: sends one capture and sends the same request again
    one cadence after each busy refusal, while the start window is open."""
    def __init__(self, request, *, at=0.0, interval=1.0, window=8.0):
        self.request, self.at, self.interval, self.window = request, ticks(at), ticks(interval), ticks(window)
        self.seen, self.due, self.sent_at = 0, ticks(at), []

    def __call__(self, sim):
        refusals = sim.refusals(self.request.rid)
        if len(refusals) > self.seen:
            self.seen = len(refusals)
            planned = sim.ticks + self.interval
            if refusals[-1] == "occ3_busy_batch" and planned <= self.at + self.window:
                self.due = planned
        if self.due is not None and sim.ticks >= self.due:
            self.due = None
            self.sent_at.append(sim.ticks)
            return [self.request]
        return []


@pytest.fixture
def world(sources, monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    made = []

    def make(**kw):
        made.append(World(sources, monkeypatch, **kw))
        return made[-1]

    yield make
    for item in made:
        item.close()


def cap(new_tokens=8, **changes):
    return generate(changes.get("native_rid", "rid"), new_tokens, descriptor=capture(**changes))


def retractions(sim):
    return {row.rid: row.retraction_count for rank in sim.ranks for row in rank.made
            if row.retraction_count}


def hold_gathers(sim):
    """(kind, rid, verdict) of every reservation collective, in order."""
    return [("park", values[0]["rid"], values[0]["eligible"]) if values[0]["hold"] == "park"
            else ("resolve", values[0]["rid"], values[0]["action"]) for values in sim.bus.holds()]


def test_quiet_scheduler_admits_at_once_without_a_reservation(world):
    sim = world()
    sim.run(1.0, once(0, cap(4)))
    assert len(sim.bus.log) == 1 and sim.bus.holds() == [] and sim.signals == []
    assert sim.first_run("rid") == 0
    assert sim.stream("rid") == reference("rid", 4) and sim.done("rid")
    assert sim.refusals("rid") == [] and sim.record().state == "capturing"
    sim.assert_clean()


def test_capture_is_admitted_while_other_requests_leave_no_idle_gap(world):
    """Default knobs. Ordinary requests overlap all the time, so the scheduler
    has no idle instant. The reservation holds new prefills, the running rows
    finish inside the drain window and the capture is admitted. Nothing is
    refused, retracted or re-sent."""
    sim = world()
    count, tokens, sent = 24, ticks(1.0), ticks(2.0)
    sender = Sender(cap(8), at=2.0)
    sim.run(16.0, load(0.5, tokens, count=count), sender)
    assert sender.sent_at == [sent] and sim.refusals("rid") == []
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "admit")]
    assert all(sim.busy[:sent])  # no idle instant until the reservation made one
    waited = sim.first_run("rid") - sent
    assert 0 < waited <= ticks(2.0)
    assert sim.stream("rid") == reference("rid", 8) and sim.done("rid")
    turn = sim.last_run("rid") - sim.first_run("rid") + 1
    for index in range(count):
        rid = f"load-{index}"
        assert sim.stream(rid) == reference(rid, tokens) and sim.done(rid)
        # A new prefill waits for the drain window and the capture turn at most.
        assert sim.first_run(rid) - index * ticks(0.5) <= ticks(2.0) + turn
    assert retractions(sim) == {}
    sim.assert_clean()


def test_polling_alone_is_refused_at_every_attempt_under_the_same_traffic(world, monkeypatch):
    """Control for the test above. With the reservation disabled, which is the
    earlier behaviour, the same traffic refuses every attempt of the window."""
    monkeypatch.setenv("OCC3_CAPTURE_BUSY_WAIT_SECONDS", "0")
    sim = world()
    sender = Sender(cap(8), at=2.0, window=10.0)
    sim.run(13.0, load(0.5, ticks(1.0)), sender)
    assert all(sim.busy) and sim.bus.holds() == [] and sim.signals == []
    assert len(sender.sent_at) >= 8
    assert sim.refusals("rid") == ["occ3_busy_batch"] * len(sender.sent_at)
    assert sim.first_run("rid") is None and sim.record().state == "reserved"


def test_f4_shape_drain_and_resend_admit_right_after_the_decode_without_retraction(world):
    """Field case F4 shape: one unrelated request decodes for 6.5 s after the
    first decline and the server is idle after it. Default knobs (window 10 s,
    drain 2 s, preempt off) and the frontend 1 s resend cadence admit the
    capture in the first iteration after that decode."""
    sim = world()
    sent, decode = ticks(1.0), ticks(6.5)
    sender = Sender(cap(8), at=1.0)
    sim.run(10.0, once(0, generate("unrelated", sent + decode + 1)), sender)
    assert sim.last_run("unrelated") == sent + decode
    assert sender.sent_at == [sent, sent + ticks(2.0) + 1 + ticks(1.0),
                              sent + 2 * (ticks(2.0) + 1 + ticks(1.0))]
    assert sim.refusals("rid") == ["occ3_busy_batch"] * 2
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline"),
                                 ("park", "rid", True), ("resolve", "rid", "decline"),
                                 ("park", "rid", True), ("resolve", "rid", "admit")]
    assert sim.first_run("rid") == sim.last_run("unrelated") + 1
    assert (sim.first_run("rid") - sent) * DT <= 10.0
    assert retractions(sim) == {}
    assert sim.stream("unrelated") == reference("unrelated", sent + decode + 1)
    assert sim.stream("rid") == reference("rid", 8) and sim.done("rid")
    sim.assert_clean()


def test_retract_mode_admits_at_the_drain_bound_while_decodes_outlast_the_window(world, monkeypatch):
    """Opt-in retract. Two requests decode for 12 s, longer than the whole busy
    window. The capture is admitted exactly at the drain bound. The retracted
    rows resume first, from the same token, and every stream is exact."""
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world()
    long, sent = ticks(12.0), ticks(1.0)
    sim.run(16.0, once(0, generate("long-a", long), generate("long-b", long)),
            Sender(cap(8), at=1.0), load(0.5, ticks(1.0), start=1.5, count=4))
    assert sim.refusals("rid") == [] and sim.first_run("rid") - sent == ticks(2.0)
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "retract")]
    assert retractions(sim) == {"long-a": 1, "long-b": 1}
    resumed = sim.last_run("rid") + 1
    assert sim.leader.trace[resumed][:2] == ("long-a", "long-b")
    assert set(sim.leader.trace[resumed][2:]) == {f"load-{index}" for index in range(4)}
    for rid, count in (("long-a", long), ("long-b", long), ("rid", 8),
                       *((f"load-{index}", ticks(1.0)) for index in range(4))):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()


def test_retract_mode_restarts_a_chunked_prefill_and_keeps_every_stream_exact(world, monkeypatch):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    monkeypatch.setenv("OCC3_CAPTURE_DRAIN_SECONDS", "0.5")
    sim = world()
    sim.run(12.0, once(0, generate("decode", ticks(3.0))),
            once(0.25, generate("chunked", ticks(1.0), chunks=40)), once(0.5, cap(4)))
    assert sim.first_run("rid") - ticks(0.5) == ticks(0.5) and sim.refusals("rid") == []
    assert retractions(sim) == {"decode": 1, "chunked": 1}
    for rid, count in (("decode", ticks(3.0)), ("chunked", ticks(1.0)), ("rid", 4)):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()


def test_retract_mode_leaves_rows_alone_when_they_finish_inside_the_drain(world, monkeypatch):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world()
    sim.run(4.0, once(0, generate("short", ticks(1.5))), once(1.0, cap(4)),
            once(1.25, generate("later", 4)))
    assert sim.first_run("rid") == sim.last_run("short") + 1 < ticks(1.0) + ticks(2.0)
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "admit")]
    assert retractions(sim) == {} and sim.refusals("rid") == []
    assert sim.first_run("later") == sim.last_run("rid") + 1
    for rid, count in (("short", ticks(1.5)), ("later", 4), ("rid", 4)):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()


@pytest.mark.parametrize("drain", ["0", "0.5", "1", "2", "4"])
def test_retract_mode_bound_follows_the_drain_knob(world, monkeypatch, drain):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    monkeypatch.setenv("OCC3_CAPTURE_DRAIN_SECONDS", drain)
    sim = world()
    sent = ticks(1.0)
    sim.run(float(drain) + 3.0, once(0, generate("long", ticks(20.0))), once(1.0, cap(4)),
            once(1.0, generate("later", 4)))
    waited = max(ticks(float(drain)), 1)  # the signal needs one iteration
    assert sim.first_run("rid") - sent == waited and sim.refusals("rid") == []
    assert sim.signals == [(sent + waited, "occ3_capture_hold_1")]
    # The new prefill and the retracted row wait for the capture turn only.
    assert sim.first_run("later") == sim.last_run("rid") + 1
    assert sim.leader.trace[sim.last_run("rid") + 1] == ("long", "later")
    assert retractions(sim) == {"long": 1}


@pytest.mark.parametrize("drain", ["0.5", "2", "3"])
def test_default_mode_declines_at_the_drain_bound_and_frees_new_prefills(world, monkeypatch, drain):
    monkeypatch.setenv("OCC3_CAPTURE_DRAIN_SECONDS", drain)
    sim = world()
    sent, held = ticks(1.0), ticks(float(drain))
    sim.run(float(drain) + 2.0, once(0, generate("long", ticks(20.0))), once(1.0, cap(4)),
            once(1.25, generate("later", 4)))
    assert sim.signals == [(sent + held, "occ3_capture_hold_1")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
    assert sim.first_run("later") == sent + held  # the same iteration as the decline
    assert retractions(sim) == {} and sim.record().state == "reserved"


def test_decline_after_the_bound_is_mutation_free_and_the_same_request_is_admitted_later(world):
    sim = world()
    request = cap(4)
    sim.run(3.5, once(0, generate("long", ticks(4.0))), once(1.0, request))
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
    assert len(sim.bus.log) == 3  # the declined preflight, the park and the resolve
    for rank in sim.ranks:
        record = rank.occ3_runtime.record
        assert (record.state, record.admitted_capture, record.state_generation) == ("reserved", False, 123)
        assert not rank.occ3_runtime.poisoned
        attempt, = [row for row in rank.made if row.rid == "rid"]
        assert not (attempt.occ3_force_eager or attempt.occ3_attached or attempt.kv.holds_kv)
        assert attempt.occ3_worker_descriptor is None and not attempt.output_ids
        assert attempt not in rank.waiting_queue and "rid" not in rank.freed
        assert rank.row("long").retraction_count == 0
    sim.run(2.0)  # the ordinary decode ends
    sim.tick(request)
    sim.run(1.0)
    assert sim.refusals("rid") == ["occ3_busy_batch"]
    assert sim.stream("rid") == reference("rid", 4) and sim.done("rid")
    assert sim.stream("long") == reference("long", ticks(4.0)) and sim.done("long")
    sim.assert_clean()


def test_four_ranks_with_skewed_clocks_reach_the_same_decisions(world, monkeypatch):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world(ranks=4)
    assert len({rank.clock.now for rank in sim.ranks}) == 4
    sim.run(8.0, once(0, generate("long", ticks(5.0))), once(1.0, cap(8)),
            load(0.5, 8, start=1.5, count=3))
    assert all(len({repr(value) for value in values}) == 1 for values in sim.bus.holds())
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "retract")]
    assert sim.first_run("rid") - ticks(1.0) == ticks(2.0)
    for rank in sim.ranks:
        assert rank.row("long").retraction_count == 1 and not rank.occ3_runtime.poisoned
        assert rank.occ3_runtime.record.admitted_capture
    assert sim.stream("long") == reference("long", ticks(5.0)) and sim.done("rid")
    sim.assert_clean()


@pytest.mark.parametrize("phase", ["park", "resolve"])
def test_one_rank_that_disagrees_makes_every_rank_decline(world, monkeypatch, phase):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world(ranks=3)

    def lie(value):
        if phase == "park" and value.get("hold") == "park":
            return {**value, "eligible": False}
        if phase == "resolve" and "action" in value:
            return {**value, "action": "admit"}
        return value

    sim.ranks[2].occ3_runtime.lie = lie
    sim.run(4.0, once(0, generate("long", ticks(6.0))), once(1.0, cap(4)))
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
    assert sim.signals == ([] if phase == "park" else [(ticks(3.0), "occ3_capture_hold_1")])
    assert retractions(sim) == {}
    for rank in sim.ranks:
        assert rank.occ3_runtime.record.state == "reserved" and not rank.occ3_runtime.poisoned
        assert getattr(rank, "occ3_capture_hold", None) is None
    sim.run(4.0)
    assert sim.stream("long") == reference("long", ticks(6.0))
    sim.assert_clean()


def test_ranks_whose_clocks_split_on_the_deadline_decline_together(world, monkeypatch):
    """The leader is before the descriptor deadline and would retract. The
    other rank's clock is past it. The gathered verdict declines on both."""
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world()
    sim.run(4.0, once(0, generate("long", ticks(6.0))), once(1.0, cap(4, deadline=103.5)))
    assert [sorted({value["action"] for value in values}) for values in sim.bus.holds()
            if "action" in values[0]] == [["decline", "retract"]]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and retractions(sim) == {}
    assert sim.signals == [(ticks(3.0), "occ3_capture_hold_1")]


def test_the_descriptor_deadline_ends_a_reservation_before_the_drain_bound(world):
    sim = world(skew=0.0)
    sim.run(4.0, once(0, generate("long", ticks(6.0))), once(1.0, cap(4, deadline=102.0)))
    assert sim.signals == [(ticks(2.0), "occ3_capture_hold_1")]
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.record().state == "reserved"


def test_only_the_leader_reads_a_clock_for_the_reservation(sources):
    ingest = ast.unparse(sources.node(SCHED, "ingest_requests", "Scheduler"))
    assert "attn_tp_rank == 0" in ingest and "local_reqs = self._poll_timeout_aborts()" in ingest
    poll = ast.unparse(sources.node(SCHED, "_poll_timeout_aborts", "Scheduler"))
    assert "now >= occ3_hold.due_at" in poll and "abort_message=occ3_hold.signal" in poll
    # No other patched method compares a clock with the reservation.
    text = sources.text[SCHED]
    assert text.count("occ3_hold.due_at") == 1 and text.count(".due_at") == 1
    selection = ast.unparse(sources.node(SCHED, "_occ3_prefill_selection", "Scheduler"))
    assert "time." not in selection


def parked(world, **kw):
    sim = world(**kw)
    sim.run(1.5, once(0, generate("load-a", ticks(5.0))), once(1.0, cap(4)),
            once(1.25, generate("load-b", 4)))
    assert all(rank.occ3_capture_hold.rid == "rid" for rank in sim.ranks)
    assert sim.first_run("load-b") is None and sim.refusals("rid") == []
    return sim


def test_client_abort_releases_a_parked_capture_and_later_signals_abort_nothing(world):
    sim = parked(world)
    sim.tick(AbortReq(rid="rid"))  # disconnect, or the frontend cleanup after its own bound
    assert sim.refusals("rid") == ["occ3_cancelled"] and sim.record().state == "reserved"
    assert all(rank.occ3_capture_hold is None for rank in sim.ranks)
    assert sim.first_run("load-b") == sim.ticks - 1  # held prefill starts in that iteration
    sim.tick(AbortReq(rid="rid", abort_message="occ3_capture_hold_1"))
    sim.tick(AbortReq(rid="load", abort_message="occ3_capture_hold_9"))
    sim.run(6.0)
    assert sim.signals == [] and sim.first_run("rid") is None
    assert sim.refusals("rid") == ["occ3_cancelled"]
    for rid, count in (("load-a", ticks(5.0)), ("load-b", 4)):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()


def test_abort_all_releases_the_hold_and_keeps_the_stock_abort(world):
    sim = parked(world, allow_drop=True)
    sim.tick(AbortReq(abort_all=True))
    assert sim.refusals("rid") == ["occ3_cancelled"]
    assert all(rank.occ3_capture_hold is None and rank.dropped == ["load-b"] for rank in sim.ranks)
    sim.run(0.5)
    assert ("aborted", "load-a", None) in sim.leader.sent and sim.signals == []
    sim.assert_clean()


def test_a_stale_signal_never_aborts_the_admitted_capture(world):
    sim = world()
    turn = ticks(2.0)
    sim.run(2.0, once(0, generate("short", ticks(1.5))), once(1.0, cap(turn)))
    assert sim.signals == [(sim.first_run("rid"), "occ3_capture_hold_1")]
    assert not sim.done("rid")
    for seq in (1, 2):
        sim.tick(AbortReq(rid="rid", abort_message=f"occ3_capture_hold_{seq}"))
    sim.run(3.0)
    assert sim.stream("rid") == reference("rid", turn) and sim.done("rid")
    assert sim.refusals("rid") == []
    sim.assert_clean()


def test_a_second_capture_is_refused_while_one_is_parked(world):
    sim = world()
    other = cap(4, native_rid="rid-2", attempt_id="attempt-2", capture_id="capture-2")
    sim.run(4.0, once(0, generate("short", ticks(2.0))), once(1.0, cap(4)), once(1.5, other))
    assert hold_gathers(sim) == [("park", "rid", True), ("park", "rid-2", False),
                                 ("resolve", "rid", "admit")]
    assert sim.refusals("rid-2") == ["occ3_busy_batch"] and sim.refusals("rid") == []
    assert sim.first_run("rid") == sim.last_run("short") + 1 and sim.done("rid")
    sim.assert_clean()


def test_a_paused_engine_never_parks_and_never_admits_a_parked_capture(world):
    sim = world()
    pause = lambda flag: sim.each(lambda rank: setattr(rank, "_engine_paused", flag))
    sim.run(1.0, once(0, generate("long", ticks(4.0))))
    pause(True)
    sim.tick(cap(4))
    assert sim.refusals("rid") == ["occ3_busy_batch"]
    assert hold_gathers(sim) == [("park", "rid", False)]
    pause(False)
    sim.tick(cap(4))
    pause(True)  # an operator pause while the capture is parked
    sim.run(2.5)
    assert hold_gathers(sim)[1:] == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] * 2 and sim.record().state == "reserved"
    pause(False)
    sim.run(4.0)
    assert sim.stream("long") == reference("long", ticks(4.0)) and sim.first_run("rid") is None
    sim.assert_clean()


@pytest.mark.parametrize("mode", ["off", "retract"])
def test_a_full_queue_declines_instead_of_dropping_a_request(world, monkeypatch, mode):
    """max_queued_requests=2. The capture and each retracted row need a slot."""
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", mode)
    sim = world(max_queued=2)
    running = ticks(1.5) if mode == "off" else ticks(6.0)
    sim.run(8.0, once(0, generate("run", running)), once(1.0, cap(4)),
            once(1.25, generate("held-1", 4)), once(1.5, generate("held-2", 4)))
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
    assert retractions(sim) == {} and sim.record().state == "reserved"
    for rid, count in (("run", running), ("held-1", 4), ("held-2", 4)):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
        assert sim.refusals(rid) == []
    sim.assert_clean()


@pytest.mark.parametrize("name, value", [("SGLANG_REQ_WAITING_TIMEOUT", "30"),
                                         ("SGLANG_REQ_RUNNING_TIMEOUT", "0.5"),
                                         ("SGLANG_REQ_WAITING_TIMEOUT", "soon")])
def test_retract_is_not_used_while_a_stock_timeout_sweep_is_enabled(world, monkeypatch, name, value):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    monkeypatch.setenv(name, value)
    sim = world()
    sim.run(4.0, once(0, generate("long", ticks(6.0))), once(1.0, cap(4)))
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and retractions(sim) == {}


@pytest.mark.parametrize("name, value", [("SGLANG_REQ_WAITING_TIMEOUT", "-1"),
                                         ("SGLANG_REQ_RUNNING_TIMEOUT", "0")])
def test_a_disabled_timeout_sweep_does_not_block_retract(world, monkeypatch, name, value):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    monkeypatch.setenv(name, value)
    sim = world()
    sim.run(4.0, once(0, generate("long", ticks(6.0))), once(1.0, cap(4)))
    assert hold_gathers(sim)[-1] == ("resolve", "rid", "retract") and sim.done("rid")


@pytest.mark.parametrize("kind", ["occ3_attached", "input_embeds", "beam_group"])
def test_retract_mode_declines_for_a_row_the_pause_path_cannot_resume(world, monkeypatch, kind):
    monkeypatch.setenv("OCC3_CAPTURE_PREEMPT", "retract")
    sim = world()
    embeds = [[0.0]] * 3 if kind == "input_embeds" else None
    sim.run(0.5, once(0, generate("special", ticks(4.0), input_embeds=embeds)))
    if kind == "occ3_attached":
        sim.each(lambda rank: vars(rank.row("special")).update(
            occ3_attached=True, occ3_worker_descriptor=NS(deadline=1e9)))
    if kind == "beam_group":
        sim.each(lambda rank: setattr(rank.row("special"), "beam_group", object()))
    sim.tick(cap(4))
    sim.run(6.0)
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and retractions(sim) == {}
    assert sim.stream("special") == reference("special", ticks(4.0)) and sim.done("special")
    sim.assert_clean()


def gate(**state):
    def gather(value):
        raise AssertionError("no collective is allowed before the rank-uniform gate")

    return NS(occ3_runtime=NS(gather=gather, poisoned=False), waiting_queue=[], **state)


def quota(detail=INFLIGHT):
    return LatentError("quota", detail=detail)


def test_rank_uniform_gates_return_before_any_collective(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(w, "_runtime", lambda scheduler=None: scheduler.occ3_runtime)
    recv = NS(rid="rid", occ3_worker_descriptor=capture())
    scheduler = gate()
    assert not w.capture_hold_park(scheduler, recv, LatentError("replay"))
    assert not w.capture_hold_park(scheduler, recv, RuntimeError("other admission failure"))
    receiver = NS(rid="receiver-rid", occ3_worker_descriptor=consume())
    assert not w.capture_hold_park(scheduler, receiver, quota())
    assert not w.capture_hold_park(scheduler, NS(rid="rid", occ3_worker_descriptor={}), quota())
    assert not w.capture_hold_park(gate(occ3_capture_hold_replay=True), recv, quota())
    for name, value in (("OCC3_CAPTURE_BUSY_WAIT_SECONDS", "0"),
                        ("OCC3_CAPTURE_BUSY_WAIT_SECONDS", "11"),
                        ("OCC3_CAPTURE_DRAIN_SECONDS", "soon"),
                        ("OCC3_CAPTURE_PREEMPT", "maybe")):
        monkeypatch.setenv(name, value)
        assert not w.capture_hold_park(scheduler, recv, quota())
        monkeypatch.delenv(name)
    assert not hasattr(scheduler, "occ3_capture_hold")


def test_a_quota_decline_for_another_reason_is_not_parked(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(w, "_runtime", lambda scheduler=None: scheduler.occ3_runtime)
    monkeypatch.setattr(w, "time", NS(monotonic=lambda: 100.0))
    seen = []
    scheduler = NS(occ3_runtime=NS(gather=lambda value: seen.append(value) or [value] * 2),
                   waiting_queue=[])
    recv = NS(rid="rid", occ3_worker_descriptor=capture())
    assert not w.capture_hold_park(scheduler, recv, LatentError("quota"))
    assert seen == [{"hold": "park", "rid": "rid", "eligible": False}]
    assert w.capture_hold_park(scheduler, recv, quota())
    hold = scheduler.occ3_capture_hold
    assert (hold.rid, hold.seq, hold.signal) == ("rid", 1, "occ3_capture_hold_1")
    assert (hold.parked_at, hold.due_at, hold.deadline) == (100.0, 102.0, 1000.0)
    assert hold.recv_req is recv


def test_reservation_knobs_default_to_drain_only():
    settings = latent_types.capture_hold_settings({})
    assert (settings.window, settings.drain, settings.preempt) == (10.0, 2.0, "off")
    assert latent_types.CAPTURE_PREEMPT_DEFAULT == "off"
    assert latent_types.CAPTURE_PREEMPT_MODES == ("off", "retract")
    short = latent_types.capture_hold_settings({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "1.5"})
    assert (short.window, short.drain) == (1.5, 1.5)
    off = latent_types.capture_hold_settings({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "0"})
    assert (off.window, off.drain) == (0.0, 0.0)
    full = latent_types.capture_hold_settings({"OCC3_CAPTURE_DRAIN_SECONDS": "10",
                                               "OCC3_CAPTURE_PREEMPT": "retract"})
    assert (full.window, full.drain, full.preempt) == (10.0, 10.0, "retract")


@pytest.mark.parametrize("environ, code", [
    ({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "10.5"}, "capture_busy_wait_seconds_invalid"),
    ({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "-1"}, "capture_busy_wait_seconds_invalid"),
    ({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "nan"}, "capture_busy_wait_seconds_invalid"),
    ({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": ""}, "capture_busy_wait_seconds_invalid"),
    ({"OCC3_CAPTURE_DRAIN_SECONDS": "10.5"}, "capture_drain_seconds_invalid"),
    ({"OCC3_CAPTURE_DRAIN_SECONDS": "-0.5"}, "capture_drain_seconds_invalid"),
    ({"OCC3_CAPTURE_DRAIN_SECONDS": "inf"}, "capture_drain_seconds_invalid"),
    ({"OCC3_CAPTURE_BUSY_WAIT_SECONDS": "2", "OCC3_CAPTURE_DRAIN_SECONDS": "3"},
     "capture_drain_seconds_invalid"),
    ({"OCC3_CAPTURE_PREEMPT": "Retract"}, "capture_preempt_invalid"),
    ({"OCC3_CAPTURE_PREEMPT": ""}, "capture_preempt_invalid"),
])
def test_invalid_reservation_knobs_carry_fixed_codes(environ, code):
    with pytest.raises(ValueError) as caught:
        latent_types.capture_hold_settings(environ)
    assert str(caught.value) == code


def plain(**changes):
    return NS(**{"occ3_attached": False, "occ3_force_eager": False, "input_embeds": None,
                 "beam_group": None, **changes})


@pytest.mark.parametrize("preempt, state, rows, expected", [
    ("off", {}, [], "admit"),
    ("off", {}, [plain()], "decline"),
    ("retract", {}, [plain(), plain()], "retract"),
    ("retract", {}, [], "admit"),
    ("retract", {"_engine_paused": True}, [], "decline"),
    ("retract", {"max_queued_requests": 3, "waiting_queue": [plain(), plain()]}, [], "admit"),
    ("retract", {"max_queued_requests": 3, "waiting_queue": [plain(), plain(), plain()]}, [], "decline"),
    ("retract", {"max_queued_requests": 3, "waiting_queue": [plain()]}, [plain()], "retract"),
    ("retract", {"max_queued_requests": 3, "waiting_queue": [plain()]}, [plain(), plain()], "decline"),
    ("retract", {"waiting_queue": [plain(occ3_force_eager=True)]}, [], "decline"),
    ("retract", {"waiting_queue": [plain(occ3_attached=True)]}, [plain()], "decline"),
    ("off", {"grammar_manager": NS(grammar_queue=[plain(occ3_force_eager=True)])}, [], "decline"),
    ("retract", {}, [plain(), plain(occ3_force_eager=True)], "decline"),
    ("retract", {}, [plain(occ3_attached=True)], "decline"),
    ("retract", {}, [plain(input_embeds=[[0.0]])], "decline"),
    ("retract", {}, [plain(beam_group=object())], "decline"),
])
def test_resolution_verdict_table(monkeypatch, preempt, state, rows, expected):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(w, "time", NS(monotonic=lambda: 100.0))
    scheduler = NS(**{"waiting_queue": [], **state})
    settings = latent_types.CaptureHoldSettings(10.0, 2.0, preempt)
    hold = NS(deadline=101.0)
    assert w._capture_hold_action(scheduler, hold, settings, rows) == expected
    assert w._capture_hold_action(scheduler, NS(deadline=100.0), settings, rows) == "decline"
    if expected == "retract":
        assert w._capture_hold_action(scheduler, hold, None, rows) == "decline"
        monkeypatch.setenv("SGLANG_REQ_WAITING_TIMEOUT", "5")
        assert w._capture_hold_action(scheduler, hold, settings, rows) == "decline"


def test_a_drained_reservation_declines_while_an_occ3_row_is_queued(world):
    """The scheduler drains inside the window, but a queued OCC-3 row runs
    first. A capture admitted behind it would spend its one-shot record and
    then wait for that row's whole turn."""
    sim = world()
    sim.run(1.5, once(0, generate("short", ticks(2.0))), once(1.0, cap(4)),
            once(1.25, generate("occ3-row", ticks(1.0))))
    sim.each(lambda rank: vars(rank.row("occ3-row")).update(
        occ3_attached=True, occ3_queue_deadline=None, occ3_worker_descriptor=NS(deadline=1e9)))
    sim.run(4.0)
    drained = sim.last_run("short") + 1
    assert sim.signals == [(drained, "occ3_capture_hold_1")]
    assert hold_gathers(sim) == [("park", "rid", True), ("resolve", "rid", "decline")]
    assert sim.refusals("rid") == ["occ3_busy_batch"] and sim.first_run("rid") is None
    assert sim.record().state == "reserved" and not sim.record().admitted_capture
    assert sim.first_run("occ3-row") == drained
    for rid, count in (("short", ticks(2.0)), ("occ3-row", ticks(1.0))):
        assert sim.stream(rid) == reference(rid, count) and sim.done(rid)
    sim.assert_clean()
