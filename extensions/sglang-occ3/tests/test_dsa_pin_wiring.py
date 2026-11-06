"""DSA pin producer and CUDA-graph wiring inside the attested manifests.

Applies the scheduler-cache-ipc and capture-eager-mtp manifests through the
real applier to pinned e00930c5 sources and checks the patched producer and
the three graph runners (decode/target-verify, EAGLE draft, draft-extend).
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
def test_graph_runner_static_pin_buffers(patched, path):
    text = patched[path]
    capture_name, replay_name = RUNNERS[path]
    init = _method(text, "__init__")
    assert "self._occ3_dsa_pin = _Occ3GraphPin(self.max_bs, model_runner.device)" in init
    assert init.index("self.max_bs = max(self.capture_bs)") < init.index("_Occ3GraphPin(self.max_bs")
    capture = _method(text, capture_name)
    view = capture.index("forward_batch.occ3_dsa_pin = self._occ3_dsa_pin.view(")
    # Present on the capture batch before the captured forward is defined/run.
    if "def run_once" in capture:
        assert view < capture.index("def run_once")
    else:
        assert view < capture.index("return forward_batch, attn_backend, pp_proxy_tensors")
    replay = _method(text, replay_name)
    load = replay.index("self._occ3_dsa_pin.load(forward_batch,")
    publish = replay.index("self._occ3_dsa_pin.publish(forward_batch,")
    call = replay.index("_replay_graph(") if "_replay_graph(" in replay else replay.index("self.backend.replay(")
    assert load < call < publish
    assert replay.count("self._occ3_dsa_pin.load(") == 1


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
    (and therefore take the eager prefill path above, or the decode/verify
    graphs that load the static pin buffers).
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
