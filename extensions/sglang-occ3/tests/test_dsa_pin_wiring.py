"""DSA pin producer and CUDA-graph wiring inside the attested manifests.

Applies the scheduler-cache-ipc and capture-eager-mtp manifests through the
real applier to pinned e00930c5 sources and checks the patched producer and
the graph runners (prefill, decode/target-verify, EAGLE draft, draft-extend):
no graph captures the pin and every runner rejects pinned batches.
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
def test_graph_runners_capture_without_a_pin_and_reject_pinned_batches(patched, path):
    """Opus #1: no captured graph carries the pin; pinned batches run eager.

    Ordinary batches keep the original unpinned graphs at their original
    cost: capture, replay and __init__ contain no pin code, and the pin guard
    is the first eligibility check after the request-local occ3 checks.
    """
    text = patched[path]
    assert "GraphPin" not in text and "_occ3_dsa_pin" not in text
    capture_name, replay_name = RUNNERS[path]
    for name in ("__init__", capture_name, replay_name):
        assert "occ3_dsa_pin" not in _method(text, name)
    method = _method(text, "can_run_graph")
    guard = method.index('if getattr(forward_batch, "occ3_dsa_pin", None) is not None:')
    assert method.index('"occ3_context", None') < guard < method.index("spec_info")
    # Execute the patched early-return prefix on stub batches.
    cut = method.index("\n", method.index("return False", guard)) + 1
    head = method[:cut] + "        return 'graph'\n"
    ns = {"ForwardBatch": object}
    exec(compile(textwrap.dedent(head), path, "exec"), ns)
    can_run = ns["can_run_graph"]
    from types import SimpleNamespace as NS

    pinned = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=("s", "c", "f"))
    plain = NS(occ3_force_eager=False, occ3_context=None, occ3_dsa_pin=None)
    assert can_run(None, pinned) is False
    assert can_run(None, plain) == "graph"
    assert can_run(None, NS()) == "graph"
    assert can_run(None, NS(occ3_force_eager=True)) is False


def test_capture_batches_carry_no_pin(patched):
    """Capture builds ForwardBatch(...) directly; the field default is None.

    The producer runs only in ForwardBatch.init_new, which graph capture
    never calls, so every captured graph sees occ3_dsa_pin None.
    """
    assert "    occ3_dsa_pin: Optional[tuple] = None\n" in patched[FBI]
    for path, (capture_name, _) in RUNNERS.items():
        capture = _method(patched[path], capture_name)
        assert "init_new(" not in capture and "occ3_dsa_pin" not in capture


def test_no_graph_runner_has_pin_replay_hooks(patched):
    for path in list(RUNNERS) + ["model_executor/runner/prefill_cuda_graph_runner.py"]:
        text = patched[path]
        assert ".load(forward_batch" not in text or "_occ3_dsa_pin" not in text
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
