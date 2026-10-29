"""Offline injection-lineage regression using pinned native method bodies.

No SGLang model or CUDA runtime is loaded. The unpatched, pinned checkout is
resolved from OCC3_SGLANG_SOURCE_BASE (see _source_pins.py for the runbook).
"""
from __future__ import annotations
import ast
import contextlib
import dataclasses
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from _source_pins import pristine_root, srt_dir
from occ3_sglang import latent_worker as worker
from occ3_sglang.attachment_contract import ConsumeWorkerDescriptor
from occ3_sglang.latent_types import RuntimeProfile, RequestSlice

ROOT = Path(__file__).resolve().parents[3]
SG = srt_dir(pristine_root())
NS = types.SimpleNamespace

def source_method(text, owner, name, scope):
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SG / owner), "exec"), scope)
    return scope[name]


EXTRA_SOURCE_PINS = {'model_executor/runner/eager_runner.py': 'a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5', 'model_executor/cuda_graph_buffer_registry.py': '839a1a077a6c7ec50148d4a6481e37880baa8947432db7ac09574b7e601f0d78'}


def patched_source(path):
    text = (SG / path).read_text(encoding="utf-8")
    if path in EXTRA_SOURCE_PINS:
        assert hashlib.sha256(text.encode()).hexdigest() == EXTRA_SOURCE_PINS[path]
    for manifest_path in (ROOT / "extensions/sglang-occ3/patches").glob("*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            if entry["path"] == path:
                assert hashlib.sha256(text.encode()).hexdigest() == entry["before_sha256"]
                for replacement in entry["replacements"]:
                    assert text.count(replacement["before"]) == 1
                    text = text.replace(replacement["before"], replacement["after"], 1)
                assert hashlib.sha256(text.encode()).hexdigest() == entry["after_sha256"]
    return text


def profile(**changes):
    return RuntimeProfile("epoch", 9, "model-hash", "glm", **changes)


def consume():
    return ConsumeWorkerDescriptor(
        attempt_id="consume", reservation_id="reserve-consume", native_rid="receiver-rid",
        principal_digest="a" * 64, attachment_digest="e" * 64, receiver_session_id="receiver",
        message_id="message", consume_nonce="nonce", opaque_handle="synthetic-state",
        state_generation=123, runtime_generation=9, source_capture_commit_id="commit",
        runtime_epoch="epoch", model_fingerprint="model-hash", method_id=profile().method,
        tp_layout_digest="b" * 64, expected_rank_count=8, expected_prefix_tokens=8,
        hidden_dim=6144, dtype="bfloat16", deadline=900.0)


class ShapeOnlyBuffer:
    """No tensor data; records that the actual hook reached its copy call."""
    is_cuda = True
    dtype = "bfloat16"
    def __init__(self, rows=8, copies=None):
        self.shape = (rows, 6144)
        self.copies = copies if copies is not None else []
    def __getitem__(self, value):
        return self
    def copy_(self, value):
        self.copies.append(True)
    def record_stream(self, stream):
        pass


@dataclasses.dataclass
class ShapeOnlyBatch:
    occ3_context: object
    occ3_force_eager: bool = True
    occ3_effective_embeds: object = None
    occ3_capture_rows: object = None
    batch_size: int = 1
    input_ids: object = None
    positions: object = None
    input_embeds: object = None
    forward_mode: object = None
    attn_cp_metadata: object = None
    global_num_tokens_cpu: object = None
    token_indices_to_pool: object = None
    def needs_forward_metadata_init(self):
        return False


def observe_injection_span(no_copy, outcome):
    if not (SG / "model_executor/runner/eager_runner.py").is_file():
        pytest.skip("Pinned SGLang source absent; set OCC3_SGLANG_SOURCE_BASE")
    descriptor = consume()
    item = RequestSlice(descriptor, 0, 8, 8 if outcome == "no_prefix" else 0, 0, True)
    mode = NS(is_extend=lambda **kw: True, is_target_verify=lambda: False,
              is_split_prefill=lambda: False, is_decode=lambda: False,
              is_cuda_graph=lambda: False)
    batch = ShapeOnlyBatch((item,), input_ids=NS(shape=(8,)), positions=object(), forward_mode=mode)
    record = NS(prefix=ShapeOnlyBuffer(), events=[], inflight=0, injected_positions=set(),
                injection_commit_id=None, capture_commit_id="commit", state="leased")
    stream = NS(wait_event=lambda event: None)
    runtime = NS(poisoned=False, resolve=lambda descriptor: record,
                 preflight=lambda phase, fn, **kw: fn(), profile=profile(), device="synthetic",
                 event=lambda: object(), torch=NS(cuda=NS(current_stream=lambda device: stream)))
    layer_module = types.ModuleType("sglang.srt.layers.layer_boundary")
    layer_module.get_attn_tp_context = lambda: NS(input_scattered=False)
    seen = []
    copies = []
    def model_body(ids, positions, actual_batch, *args):
        seen.append(actual_batch)
        if outcome == "before_copy":
            raise RuntimeError("synthetic forward failure")
        worker.inject_target_embeddings(actual_batch, ShapeOnlyBuffer(copies=copies))
        if outcome == "after_copy":
            raise RuntimeError("synthetic forward failure")
        return object()
    native_forward = source_method(patched_source("models/deepseek_v2.py"),
        "DeepseekV2ForCausalLM", "forward", {
            "get_attn_tp_context": lambda: NS(maybe_input_scattered=lambda fb: contextlib.nullcontext())})
    model = NS(model=model_body, capture_aux_hidden_states=False,
               pp_group=NS(is_last_rank=True), lm_head=None,
               logits_processor=lambda *args: NS())
    model.forward = types.MethodType(native_forward, model)
    scope = dict(contextlib=contextlib, replace=dataclasses.replace,
                 envs=NS(SGLANG_EAGER_INPUT_NO_COPY=NS(get=lambda: no_copy)),
                 is_cp_active=lambda fb: False,
                 device_timer_ctx=lambda *args: contextlib.nullcontext(), _is_hip=False)
    eager_source = patched_source("model_executor/runner/eager_runner.py")
    extract_source = patched_source("model_executor/cuda_graph_buffer_registry.py")
    extract = source_method(extract_source, "CudaGraphBufferRegistry", "extract_buffer", {})
    registry = NS(_slots={}, fill_from=lambda *args, **kwargs: None)
    registry.extract_buffer = types.MethodType(extract, registry)
    load = source_method(eager_source, "EagerRunner", "load_batch", scope)
    execute = source_method(eager_source, "EagerRunner", "_execute_extend", scope)
    eager = NS(enable_pdmux=False, _eager_registry=registry)
    eager.load_batch = types.MethodType(load, eager)
    runner = NS(is_draft_worker=False, model=model,
                _extend_forward_kwargs=lambda *args: {}, prefill_cuda_graph_runner=None,
                device_timer=None, attn_backend=object(), device="cuda", hisparse_coordinator=None,
                decode_cuda_graph_runner=None, pp_group=NS(is_last_rank=True),
                _prepare_eager_forward_batch=lambda fb: None,
                _maybe_execute_deferred_mamba_cow_and_clear=lambda fb: None)
    eager.model_runner = runner
    runner.eager_runner = NS(execute=lambda fb, **kwargs: execute(eager, fb, **kwargs))
    scope.update(has_forward_context=lambda: True, get_global_dwdp_manager=lambda: None,
                 EagerRunner=type("UnusedEagerType", (), {}), ModelRunnerOutput=lambda **kw: NS(**kw))
    forward = source_method(patched_source("model_executor/model_runner.py"), "ModelRunner", "_forward_raw", scope)
    with patch.object(worker, "_RUNTIME", runtime), patch.dict(sys.modules, {layer_module.__name__: layer_module}), \
            patch.object(worker, "record_forward_start", lambda *args: None), \
            patch.object(worker, "record_forward_completion", lambda *args: None):
        if outcome in ("before_copy", "after_copy"):
            with pytest.raises(RuntimeError, match="synthetic forward failure"):
                forward(runner, batch, None)
        else:
            forward(runner, batch, None)
        assert copies == ([True] if outcome in ("success", "after_copy") else [])
        assert seen[0] is not batch
        assert batch.occ3_effective_embeds is None
        observed = {"injected_count": len(record.injected_positions),
                    "commit_present": record.injection_commit_id is not None}
        if outcome == "success":
            assert seen[0].occ3_effective_embeds is None
            event_count, commit = len(record.events), record.injection_commit_id
            worker.mark_injection(seen[0])  # A repeated marker must be a no-op.
            assert len(record.events) == event_count
            batch.occ3_context = (RequestSlice(descriptor, 0, 8, 8, 0, True),)
            forward(runner, batch, None)  # A later non-prefix chunk cannot reuse the span.
            assert seen[1] is not seen[0] and seen[1].occ3_effective_embeds is None
            assert len(record.injected_positions) == 8 and record.injection_commit_id == commit
            assert copies == [True]
    return observed


@pytest.mark.parametrize("no_copy", [False, True])
@pytest.mark.parametrize("outcome", ["success", "before_copy", "after_copy", "no_prefix"])
def test_native_eager_forward_preserves_injection_journal(no_copy, outcome):
    observed = observe_injection_span(no_copy, outcome)
    assert observed == {"injected_count": 8 if outcome == "success" else 0,
                        "commit_present": outcome == "success"}


@pytest.mark.parametrize("mode_name", ["TARGET_VERIFY", "DECODE"])
def test_decode_does_not_reinject_prefill_embeddings(mode_name):
    from enum import IntEnum, auto

    if not (SG / "model_executor/forward_batch_info.py").is_file():
        pytest.skip("Pinned SGLang source absent; set OCC3_SGLANG_SOURCE_BASE")
    mode_source = patched_source("model_executor/forward_batch_info.py")
    mode_node = next(n for n in ast.parse(mode_source).body
                     if isinstance(n, ast.ClassDef) and n.name == "ForwardMode")
    scope = {"IntEnum": IntEnum, "auto": auto}
    exec(compile(ast.Module(body=[mode_node], type_ignores=[]), str(SG / "forward_batch_info.py"), "exec"), scope)
    mode = scope["ForwardMode"]
    descriptor = consume()
    record = NS(prefix=ShapeOnlyBuffer(), events=[], inflight=0,
                injected_positions=set(range(8)), injection_commit_id="completed-prefill")
    stream = NS(wait_event=lambda event: None)
    runtime = NS(poisoned=False, resolve=lambda descriptor, **kw: record,
                 profile=profile(), device="synthetic", event=lambda: object(),
                 gather=lambda value: [value] * 8,
                 torch=NS(cuda=NS(current_stream=lambda device: stream)))
    runtime.preflight = types.MethodType(worker.WorkerRuntime.preflight, runtime)
    scheduled = NS(reqs=[NS(occ3_worker_descriptor=descriptor, output_ids=[1])],
                   forward_mode=mode.DECODE, extend_lens=[8], prefix_lens=[0])
    layer_module = types.ModuleType("sglang.srt.layers.layer_boundary")
    layer_module.get_attn_tp_context = lambda: NS(input_scattered=False)
    copies = []
    with patch.object(worker, "_RUNTIME", runtime), patch.dict(sys.modules, {layer_module.__name__: layer_module}):
        worker.prepare_batch(NS(occ3_runtime=runtime), scheduled)
        assert scheduled.occ3_context[0].extend_len == 0
        # Native EAGLE changes DECODE into TARGET_VERIFY; is_extend() includes it.
        if mode_name == "TARGET_VERIFY":
            assert mode.TARGET_VERIFY.is_extend()
        batch = ShapeOnlyBatch(scheduled.occ3_context, forward_mode=mode[mode_name])
        worker.inject_target_embeddings(batch, ShapeOnlyBuffer(rows=6, copies=copies))
        worker.mark_injection(batch)
    assert copies == [] and batch.occ3_effective_embeds is None
    assert record.events == [] and record.inflight == 0
    assert record.injected_positions == set(range(8))
    assert record.injection_commit_id == "completed-prefill"


class CpuTorch:
    """Real CPU Torch operators; only the CUDA stream surface is stubbed."""

    def __init__(self, torch, stream):
        self._torch = torch
        self.cuda = NS(current_stream=lambda device=None: stream)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def test_only_accepted_verify_rows_enter_the_retained_ring(monkeypatch):
    """Ring content, not counters: accepted rows survive, rejected rows cannot.

    Real CPU Torch tensors drive the actual capture stage and commit path. The
    accepted path is a permutation of the verify rows, so a "first N rows" read
    cannot pass. CUDA events/streams are the file's existing two-callable stub.
    """
    torch = pytest.importorskip("torch", reason="Actual CPU Torch is required; no tensor stub")
    from occ3_sglang.attachment_contract import CaptureWorkerDescriptor

    width, ring_size, accepted = 6, 8, [3, 1, 5]
    prompt = [11, 12, 13]
    descriptor = CaptureWorkerDescriptor(
        attempt_id="capture", reservation_id="reserve-capture", native_rid="rid",
        principal_digest="a" * 64, runtime_epoch="epoch", model_fingerprint="model-hash",
        method_id=profile().method, deadline=900.0, capture_id="c" * 64,
        source_session_id="source", source_turn_nonce="nonce", runtime_generation=9,
        span_policy="whole_completed_response_tail_v1",
        max_prefix_tokens=ring_size, max_capture_bytes=1 << 20)
    hidden_dim = profile().hidden_dim
    # One unique sentinel per verify row; rejected rows are interleaved with the
    # accepted set instead of trailing it.
    hidden = (torch.arange(1, width + 1, dtype=torch.float32)[:, None] * 1e6
              + torch.arange(hidden_dim, dtype=torch.float32)[None, :])
    # Predictor offset -1: the accepted column j predicts the token at prompt+j.
    positions = torch.tensor([50, 3, 51, 2, 52, 4], dtype=torch.int64)
    predict = torch.tensor([70, 73, 71, 90, 72, 95], dtype=torch.int64)
    accept_index = torch.tensor([[3, 1, 5, -1, -1, -1]], dtype=torch.int64)
    accept_lens = torch.tensor([len(accepted)], dtype=torch.int32)
    context = (RequestSlice(descriptor, 0, width, ring_size, 0, True),)
    record = NS(events=[], live_refs=[], inflight=0, accepted_count=0, retained=[])
    stream = NS(wait_event=lambda event: None)
    runtime = NS(torch=CpuTorch(torch, stream), device="cpu", profile=profile(), poisoned=False,
                 resolve=lambda value, **kwargs: record, gather=lambda value: [value] * 8,
                 event=lambda: NS(query=lambda: True))
    runtime.preflight = types.MethodType(worker.WorkerRuntime.preflight, runtime)
    runtime.reap_live = types.MethodType(worker.WorkerRuntime.reap_live, runtime)
    monkeypatch.setattr(worker, "_RUNTIME", runtime)
    events = runtime.event()
    logits = NS(occ3_capture_rows=worker.CaptureRows(context, hidden, positions, "verify", events, 7))
    forward_batch = NS(occ3_context=context, occ3_force_eager=True, batch_size=1)
    stage = worker.stage_verify(forward_batch, logits, predict, accept_lens, accept_index, positions)
    assert stage is not None and stage.hidden.shape == (1, width, hidden_dim)
    assert stage.counts.tolist() == [len(accepted)]
    request = NS(rid="rid", occ3_worker_descriptor=descriptor, retraction_count=0,
                 origin_input_ids=prompt, output_ids=[int(predict[i]) for i in accepted])
    worker.commit_result(NS(occ3_runtime=runtime), request, NS(occ3_capture_stage=stage),
                         0, 0, len(accepted), terminal=False, forward_iter=7)
    written_hidden, written_tokens = record.retained
    span = slice(0, len(accepted))
    assert record.accepted_count == len(accepted) and stage.committed
    assert torch.equal(written_hidden[span], hidden[accepted])
    assert torch.equal(written_tokens[span], predict[accepted])
    for rejected in range(width):
        if rejected in accepted:
            continue
        assert not any(torch.equal(written_hidden[i], hidden[rejected]) for i in range(len(accepted)))
