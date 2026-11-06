"""DSA pin producer and CUDA-graph wiring inside the attested manifests.

Applies the scheduler-cache-ipc and capture-eager-mtp manifests through the
real applier to pinned e00930c5 sources and checks the patched producer and
the graph runners (prefill, decode/target-verify, EAGLE draft, draft-extend):
the original graphs never capture the pin, the decode-family runners capture
a pinned variant per bs from a static view, replay picks the set by the batch
pin, a missing pinned bs runs eager, and the switch off restores a8e98b759.
Source evidence only; not GPU, kernel or CUDA-graph replay evidence.
"""
from __future__ import annotations

import ast
import textwrap
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[1]
ROOTS = [Path(p) for p in (os.environ.get("OCC3_ATTESTED_SRT", ""),
                           os.environ.get("OCC3_PRISTINE_RUNNER_SRT", "")) if p]
FBI = "model_executor/forward_batch_info.py"
RUNNERS = {
    "model_executor/runner/decode_cuda_graph_runner.py": ("capture_prepare", "execute"),
    "speculative/eagle_draft_cuda_graph_runner.py": ("capture_one_shape", "execute"),
    "speculative/eagle_draft_extend_cuda_graph_runner.py": ("capture_one_shape", "execute"),
}


def _manifest(name):
    return json.loads((EXT / "patches" / name).read_text(encoding="utf-8"))


def test_pin_paths_are_owned_by_one_manifest_each():
    owners = {}
    for name in ("capture-eager-mtp.v1.json", "scheduler-cache-ipc.v1.json",
                 "native-startup.v2.json", "dsa-pin.v1.json", "dsa-mtp-seed.v1.json"):
        for f in _manifest(name)["files"]:
            assert f["path"] not in owners, (f["path"], owners.get(f["path"]), name)
            owners[f["path"]] = name
    assert owners[FBI] == "scheduler-cache-ipc.v1.json"
    for path in RUNNERS:
        assert owners[path] == "capture-eager-mtp.v1.json"
    assert owners["layers/attention/dsa/dsa_indexer.py"] == "dsa-pin.v1.json"


def _applier():
    spec = importlib.util.spec_from_file_location("_occ3_pin_wiring_source_patch",
                                                  EXT / "deploy/v3/source_patch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find(path):
    for root in ROOTS:
        if (root / path).is_file():
            return root / path
    return None


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    applier = _applier()
    out = {}
    for name in ("scheduler-cache-ipc.v1.json", "capture-eager-mtp.v1.json"):
        manifest = _manifest(name)
        sources = {f["path"]: _find(f["path"]) for f in manifest["files"]}
        if not all(sources.values()):
            pytest.skip("pinned sources absent; set OCC3_ATTESTED_SRT and OCC3_PRISTINE_RUNNER_SRT")
        root = tmp_path_factory.mktemp("occ3-pin-" + name.split(".")[0]) / "srt"
        for f in manifest["files"]:
            data = sources[f["path"]].read_bytes()
            assert hashlib.sha256(data).hexdigest() == f["before_sha256"], f["path"]
            target = root / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        proof = applier.apply(applier.prepare(root, manifest))
        assert proof["applied"] is True
        for f in manifest["files"]:
            out[f["path"]] = (root / f["path"]).read_text(encoding="utf-8")
    return out


def _method(text, name, cls=None):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (cls is None or node.name == cls):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.get_source_segment(text, item)
    raise AssertionError(name)


def test_producer_rebuilds_pin_from_batch_reqs_in_init_new(patched):
    text = patched[FBI]
    assert "    occ3_dsa_pin: Optional[tuple] = None\n" in text
    init_new = _method(text, "init_new", "ForwardBatch")
    build = init_new.index("ret.occ3_dsa_pin = _occ3_dsa_pin.build(")
    assert "record=not model_runner.is_draft_worker" in init_new
    # Built before every early return (idle path included), after the
    # dataclass is constructed: no stale value from another batch survives.
    assert init_new.index("ret = cls(") < build < init_new.index("if ret.forward_mode.is_idle():")
    tbo = _method(text, "can_run_tbo", "ForwardBatch")
    assert "if self.occ3_dsa_pin is not None:" in tbo


@pytest.mark.parametrize("path", sorted(RUNNERS))
def test_can_run_graph_pinned_batches_need_a_pinned_graph(patched, path, monkeypatch):
    """Ordinary batches keep the unpinned graph; a pinned batch replays the
    pinned variant only when every decode-family runner captured one for its
    size (dsa_pin.graph_ok); otherwise it runs eager (a8e98b759 behaviour)."""
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin

    method = _method(patched[path], "can_run_graph")
    guard = method.index('occ3_pin = getattr(forward_batch, "occ3_dsa_pin", None)')
    assert method.index('"occ3_context", None') < guard < method.index("spec_info")
    cut = method.index("\n", method.index("return False", guard)) + 1
    head = method[:cut] + "        return 'graph'\n"
    ns = {"ForwardBatch": object}
    exec(compile(textwrap.dedent(head), path, "exec"), ns)
    can_run = ns["can_run_graph"]
    pin = (None, NS(shape=(3,)), None)
    pinned = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=pin)
    plain = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=None)
    ok = {"value": True}
    monkeypatch.setattr(dsa_pin, "graph_ok", lambda raw_bs: ok["value"] and raw_bs == 3)
    runner = NS(_occ3_pin_bs=frozenset({4}))
    assert can_run(runner, plain) == "graph"
    assert can_run(runner, NS()) == "graph"
    assert can_run(runner, pinned) == "graph"
    ok["value"] = False  # some runner lacks the pinned graph: eager
    assert can_run(runner, pinned) is False
    ok["value"] = True
    assert can_run(NS(_occ3_pin_bs=frozenset()), pinned) is False  # flag off
    assert can_run(NS(), pinned) is False
    assert can_run(runner, NS(occ3_force_eager=True)) is False


def _exec_method(text, name, cls=None, **globs):
    ns = dict(globs)
    exec(compile(textwrap.dedent(_method(text, name, cls)), name, "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("path", sorted(RUNNERS))
def test_unpinned_capture_has_no_pin_and_pinned_capture_uses_the_static_view(patched, path):
    """N6: the unpinned capture batch never carries a pin; the pinned one
    carries the runner's GraphPin view, and its key is the pinned variant."""
    capture_name, _ = RUNNERS[path]
    capture = _method(patched[path], capture_name)
    assert "init_new(" not in capture
    # The only pin assignment in capture is gated on the pinned-capture flag
    # and uses the static view.
    assert capture.count("occ3_dsa_pin =") == 1
    gate = capture.index('if getattr(self, "_occ3_pin_capture", False):')
    assign = capture.index("forward_batch.occ3_dsa_pin = self._occ3_dsa_pin.view(")
    assert gate < assign and capture[gate:assign].count("\n") <= 2

    # Exercise the gate on its own (dedented two-line block).
    block = capture[gate: capture.index("\n", assign) + 1]
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin
    torch = pytest.importorskip("torch")
    graph_pin = dsa_pin.GraphPin(8, "cpu")
    for flag in (False, True):
        fb = NS(occ3_dsa_pin=None)
        env = {"self": NS(_occ3_pin_capture=flag, _occ3_dsa_pin=graph_pin),
               "forward_batch": fb, "bs": 4, "num_seqs": 4}
        exec(compile(textwrap.dedent(block), path, "exec"), env)
        if flag:
            assert fb.occ3_dsa_pin[0].data_ptr() == graph_pin.start.data_ptr()
            assert fb.occ3_dsa_pin[0].shape[0] == 4
        else:
            assert fb.occ3_dsa_pin is None

    # Graph key: unpinned key unchanged, pinned capture gets the pin label.
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class ShapeKey:
        size: int
        stream_idx: object = None
        variant_label: object = None
        attention_variant: object = None

    make = _exec_method(patched[path], "_make_graph_key", ShapeKey=ShapeKey)
    assert make(NS(_occ3_pin_capture=False), 4) == ShapeKey(size=4)
    assert make(NS(), 4) == ShapeKey(size=4)
    assert make(NS(_occ3_pin_capture=True), 4) == ShapeKey(size=4, variant_label=dsa_pin.PIN_LABEL)


def test_pinned_set_is_captured_right_after_each_unpinned_shape(patched):
    """Same capture session and pool, same bs list (or its env cap)."""
    path = "model_executor/runner/decode_cuda_graph_runner.py"
    calls = []
    from types import SimpleNamespace as NS
    import contextlib

    class Range(list):
        def set_description(self, _):
            pass

    def run(pin_bs):
        calls.clear()
        runner = NS(model_runner=NS(device="cpu", gpu_id=0, model=None, tp_group=None),
                    capture_bs=[1, 2, 4], record_nolora_graph=False,
                    attention_graph_variants=None, compile_bs=[], captured_req_width=1,
                    _occ3_pin_bs=pin_bs, _occ3_pin_capture=False)
        runner.capture_one_shape = lambda bs, *a: calls.append((bs, runner._occ3_pin_capture))
        fn = _exec_method(
            patched[path], "_capture_one_stream",
            get_available_gpu_memory=lambda *a, **k: 1.0,
            get_parallel=lambda: NS(tp_rank=1),
            tqdm=NS(tqdm=Range),
            _set_capture_lora_variant=lambda v: None,
            _set_capture_attention_variant=lambda v: None,
            torch_compile_decoration=NS(patch_model=lambda *a, **k: contextlib.nullcontext(None)),
            Optional=object,
        )
        fn(runner)
        return list(calls)

    assert run(frozenset({1, 2, 4})) == [(4, False), (4, True), (2, False), (2, True), (1, False), (1, True)]
    assert run(frozenset({1, 2})) == [(4, False), (2, False), (2, True), (1, False), (1, True)]
    # Flag off: exactly the a8e98b759 capture sequence.
    assert run(frozenset()) == [(4, False), (2, False), (1, False)]


@pytest.mark.parametrize("path", sorted(RUNNERS))
def test_replay_picks_the_pinned_set_only_for_pinned_batches(patched, path):
    """Zero + load the pin, swap to the pinned key, replay, publish forced."""
    _, replay_name = RUNNERS[path]
    replay = _method(patched[path], replay_name)
    gate = replay.index("occ3_pinned = (")
    cond = replay[gate: replay.index("\n", replay.index('"_occ3_dsa_pin", None)', gate))]
    assert 'getattr(forward_batch, "occ3_dsa_pin", None) is not None' in cond
    assert 'getattr(self, "_occ3_dsa_pin", None) is not None' in cond
    load = replay.index("self._occ3_dsa_pin.load(forward_batch,")
    key = replay.index("_occ3_pinned_key(", load)
    call = replay.index("replay(", key) if "decode_cuda" in path else replay.index("self._replay_graph(", key)
    publish = replay.index("self._occ3_dsa_pin.publish(forward_batch,")
    assert gate < load < key < call < publish
    # Every pin step sits under `if occ3_pinned:`.
    for at in (load, publish):
        before = replay[:at].rstrip().rsplit("\n", 2)
        assert any("if occ3_pinned:" in line for line in before), path


def test_graph_ok_requires_every_runner_and_falls_back_eager_for_a_missing_bs(monkeypatch):
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin

    monkeypatch.setattr(dsa_pin, "_graph_runners", {})
    monkeypatch.delenv(dsa_pin.GRAPHS_ENV, raising=False)
    monkeypatch.delenv(dsa_pin.GRAPH_MAX_BS_ENV, raising=False)
    mr = NS(is_draft_worker=False, model_config=NS(hf_config=NS(index_topk=2048)))
    draft = NS(is_draft_worker=True, model_config=NS(hf_config=NS(index_topk=2048)))

    class Target: pass
    class Draft: pass
    class DraftExtend: pass

    assert dsa_pin.graph_ok(3) is False  # nothing registered yet
    assert dsa_pin.graph_bs(Target(), mr, [1, 2, 4, 8]) == {1, 2, 4, 8}
    monkeypatch.setenv(dsa_pin.GRAPH_MAX_BS_ENV, "4")
    assert dsa_pin.graph_bs(Draft(), draft, [1, 2, 4, 8]) == {1, 2, 4}
    monkeypatch.delenv(dsa_pin.GRAPH_MAX_BS_ENV)
    assert dsa_pin.graph_bs(DraftExtend(), draft, [1, 2, 4, 8]) == {1, 2, 4, 8}
    assert dsa_pin.graph_ok(3) is True     # pads to 4 everywhere
    assert dsa_pin.graph_ok(5) is False    # bs 8 has no pinned draft graph: eager
    assert dsa_pin.graph_ok(9) is False    # beyond max bs: eager
    # Unsupported runner shapes register an empty pinned set.
    assert dsa_pin.graph_bs(Target(), mr, [1, 2]) == {1, 2}
    assert dsa_pin.graph_bs(NS(disable_padding=True), mr, [1, 2]) == frozenset()
    assert dsa_pin.graph_ok(1) is False
    # Non-DSA models never get pinned graphs.
    monkeypatch.setattr(dsa_pin, "_graph_runners", {})
    assert dsa_pin.graph_bs(Target(), NS(model_config=NS(hf_config=NS())), [1]) == frozenset()


def test_flag_off_restores_a8e98b759(monkeypatch):
    """OCC3_DSA_PIN_GRAPHS=0: no pinned set, no GraphPin, pinned batches eager."""
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin

    monkeypatch.setattr(dsa_pin, "_graph_runners", {})
    mr = NS(is_draft_worker=False, model_config=NS(hf_config=NS(index_topk=2048)))
    for value in ("0", "false", "off", "NO"):
        monkeypatch.setenv(dsa_pin.GRAPHS_ENV, value)
        assert dsa_pin.graphs_enabled() is False
        assert dsa_pin.graph_bs(NS(), mr, [1, 2, 4]) == frozenset()
        assert dsa_pin.graph_ok(1) is False
    monkeypatch.setenv(dsa_pin.GRAPHS_ENV, "1")
    assert dsa_pin.graphs_enabled() is True


@pytest.mark.parametrize("path", sorted(RUNNERS))
def test_init_builds_graph_pin_only_when_a_pinned_set_exists(patched, path):
    text = patched[path]
    init = _method(text, "__init__")
    at = init.index("self._occ3_pin_bs = _occ3_dsa_pin.graph_bs(self, model_runner, self.capture_bs)")
    assert init.index("self.max_bs = max(self.capture_bs)") < at
    assert "_occ3_dsa_pin.GraphPin(self.max_bs, model_runner.device)\n            if self._occ3_pin_bs\n            else None" in init
    assert "self._occ3_pin_capture = False" in init


def test_graph_pin_load_never_raises_and_zeroes_on_mismatch():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin

    gp = dsa_pin.GraphPin(4, "cpu")
    gp.start.fill_(9); gp.count.fill_(9); gp.forced.fill_(9)
    pin = (torch.tensor([3, 5], dtype=torch.int32), torch.tensor([8, 0], dtype=torch.int32),
           torch.zeros(2, dtype=torch.int32))
    fb = NS(occ3_dsa_pin=pin)
    assert gp.load(fb, 2) is True
    assert gp.start.tolist() == [3, 5, 0, 0] and gp.count.tolist() == [8, 0, 0, 0]
    assert gp.forced.tolist() == [0, 0, 0, 0]
    gp.forced[0] = 7
    gp.publish(fb, 2)
    assert pin[2].tolist() == [7, 0]
    faults = dsa_pin.faults
    bad = NS(occ3_dsa_pin=pin)
    assert gp.load(bad, 3) is False  # shape mismatch: fault, zero, count 0
    assert dsa_pin.faults == faults + 1
    assert gp.count.tolist() == [0, 0, 0, 0] and gp.start.tolist() == [0, 0, 0, 0]
    assert bad.occ3_dsa_pin is None  # note_fault dropped the batch pin
    assert gp.load(NS(occ3_dsa_pin=None), 2) is False


def test_no_pinned_prefill_graph(patched):
    """Prefill stays as at a8e98b759: no pin code beyond the eager guard."""
    text = patched["model_executor/runner/prefill_cuda_graph_runner.py"]
    assert "GraphPin" not in text and "_occ3_dsa_pin" not in text
    assert "publish(forward_batch" not in text


PREFILL = "model_executor/runner/prefill_cuda_graph_runner.py"


def test_prefill_graph_runner_rejects_pinned_batches(patched):
    """Fable #2: pinned consume chunks never replay a pin-less prefill graph."""
    text = patched[PREFILL]
    method = _method(text, "can_run_graph")
    guard = method.index('if getattr(forward_batch, "occ3_dsa_pin", None) is not None:')
    # Ahead of every native eligibility check (which reads live batch fields).
    assert guard < method.index("global_num_tokens_cpu")
    # Execute the patched early-return prefix on stub batches.
    head = method[: method.index("        # DP check")] + "        return 'native'\n"
    ns = {"ForwardBatch": object}
    exec(compile("import textwrap\n" + textwrap.dedent(head), PREFILL, "exec"), ns)
    can_run = ns["can_run_graph"]
    from types import SimpleNamespace as NS

    pinned = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=("s", "c", "f"))
    plain = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=None)
    legacy = NS()
    assert can_run(None, pinned) is False
    assert can_run(None, plain) == "native"
    assert can_run(None, legacy) == "native"
    assert can_run(None, NS(occ3_force_eager=True)) is False


def test_pin_covers_every_chunk_and_decode_of_an_attached_request():
    """The producer keys on occ3_attached, which cohort release never clears.

    release_consume_cohort clears only occ3_force_eager after the injected
    chunk, so later prefill chunks and every decode step still carry the pin
    (and therefore take the eager path in every graph runner).
    """
    import inspect
    from types import SimpleNamespace as NS

    from occ3_sglang import dsa_pin, latent_worker
    from test_latent_registry import consume

    src = inspect.getsource(latent_worker.release_consume_cohort)
    assert "req.occ3_force_eager = False" in src and "occ3_attached" not in src
    req = NS(occ3_attached=True, occ3_force_eager=True,
             occ3_worker_descriptor=consume(prefix_offset=0, expected_prefix_tokens=64))
    assert dsa_pin.pin_rows([req]) == ([0], [64])  # injected chunk
    req.occ3_force_eager = False  # cohort released after the injected chunk
    assert dsa_pin.pin_rows([req]) == ([0], [64])  # later chunks
    assert dsa_pin.pin_rows([NS(occ3_attached=False), req]) == ([0, 0], [0, 64])  # decode


def test_two_batch_overlap_gets_no_pinned_graph_set(monkeypatch):
    """Fable F6: TBO splits captured batches; no pinned set, pinned batches eager."""
    import sys, types
    from types import SimpleNamespace as NS
    from occ3_sglang import dsa_pin

    monkeypatch.setattr(dsa_pin, "_graph_runners", {})
    monkeypatch.delenv(dsa_pin.GRAPHS_ENV, raising=False)
    monkeypatch.delenv(dsa_pin.GRAPH_MAX_BS_ENV, raising=False)
    mr = NS(is_draft_worker=False, model_config=NS(hf_config=NS(index_topk=2048)))
    draft = NS(is_draft_worker=True, model_config=NS(hf_config=NS(index_topk=2048)))
    exec_flag = {"value": False}
    ctx = types.ModuleType("sglang.srt.runtime_context")
    ctx.get_exec = lambda: NS(overlap=NS(enable_two_batch_overlap=exec_flag["value"]))
    monkeypatch.setitem(sys.modules, ctx.__name__, ctx)

    class Target: pass
    class Draft: pass

    assert dsa_pin.graph_bs(Target(), mr, [1, 2, 4]) == {1, 2, 4}
    assert dsa_pin.graph_bs(Draft(), draft, [1, 2, 4]) == {1, 2, 4}
    assert dsa_pin.graph_ok(2) is True
    # Decode runner attribute (copied from get_exec at the pinned commit).
    tbo = Target(); tbo.enable_two_batch_overlap = True
    assert dsa_pin.graph_bs(tbo, mr, [1, 2, 4]) == frozenset()
    assert dsa_pin.graph_ok(2) is False  # one runner without a pinned set: all eager
    # The EAGLE runners read the global exec config (no runner attribute).
    exec_flag["value"] = True
    assert dsa_pin.graph_bs(Target(), mr, [1, 2, 4]) == frozenset()
    assert dsa_pin.graph_bs(Draft(), draft, [1, 2, 4]) == frozenset()
    assert dsa_pin.graph_ok(1) is False and dsa_pin.graph_ok(4) is False
    exec_flag["value"] = False
    # server_args flag also excludes.
    tbo_mr = NS(is_draft_worker=False, model_config=NS(hf_config=NS(index_topk=2048)),
                server_args=NS(enable_two_batch_overlap=True))
    assert dsa_pin.graph_bs(Target(), tbo_mr, [1, 2]) == frozenset()
    # An unreadable exec config fails closed (no pinned set).
    def broken():
        raise RuntimeError("no exec")
    ctx.get_exec = broken
    assert dsa_pin.graph_bs(Target(), mr, [1, 2]) == frozenset()
