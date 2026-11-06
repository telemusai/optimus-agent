"""Backport of sgl-project/sglang#42565 (ragged MTP seed -> physical slots).

Applies patches/dsa-mtp-seed.v1.json to the pinned SGLang v0.5.21 sources
through the real applier, then runs the exact patched methods
(prepare_mtp_seed_indices, write_mtp_seed) on CPU tensors. Ports the upstream
unit tests from test/registered/unit/layers/attention/test_index_topk_share.py.
Not GPU, kernel or CUDA-graph evidence.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

EXT = Path(__file__).resolve().parents[1]
MANIFEST = EXT / "patches/dsa-mtp-seed.v1.json"
EXPECTED_PATHS = {
    "layers/attention/dsa_backend.py",
    "layers/attention/index_topk_share.py",
    "models/deepseek_common/attention_forward_methods/forward_mha.py",
}
ATTESTED = Path(os.environ.get("OCC3_ATTESTED_SRT", "/nonexistent-occ3-attested"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _applier():
    spec = importlib.util.spec_from_file_location("_occ3_dsa_seed_source_patch", EXT / "deploy/v3/source_patch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_shape_without_source():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "optimus.sglang-occ3.source-patch.v1"
    assert manifest["source_revision"] == "e00930c5489053f26d86b179cee0d087f846acbb"
    assert manifest["patch_set"] == "dsa-mtp-seed-v1"
    assert {f["path"] for f in manifest["files"]} == EXPECTED_PATHS
    others = set()
    for name in ("capture-eager-mtp.v1.json", "scheduler-cache-ipc.v1.json", "native-startup.v2.json"):
        others |= {f["path"] for f in json.loads((EXT / "patches" / name).read_text())["files"]}
    assert not (others & EXPECTED_PATHS)


@pytest.fixture(scope="module")
def pinned(tmp_path_factory):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not all((ATTESTED / f["path"]).is_file() for f in manifest["files"]):
        pytest.skip("exact attested SGLang DSA sources absent; set OCC3_ATTESTED_SRT")
    applier = _applier()
    root = tmp_path_factory.mktemp("occ3-dsa-seed") / "srt"
    raw = {}
    for f in manifest["files"]:
        data = (ATTESTED / f["path"]).read_bytes()
        assert sha(data) == f["before_sha256"]
        raw[f["path"]] = data
        target = root / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    plan = applier.prepare(root, manifest)
    proof = applier.apply(plan)
    patched = {f["path"]: (root / f["path"]).read_text(encoding="utf-8") for f in manifest["files"]}
    for f in manifest["files"]:
        assert sha(patched[f["path"]].encode()) == f["after_sha256"]
    return NS(root=root, manifest=manifest, raw=raw, patched=patched, applier=applier, proof=proof)


def test_apply_is_exact_and_second_apply_refused(pinned):
    assert pinned.proof["applied"] is True and pinned.proof["patch_set"] == "dsa-mtp-seed-v1"
    with pytest.raises(pinned.applier.PatchError, match="original_digest_mismatch"):
        pinned.applier.prepare(pinned.root, pinned.manifest)
    assert "def prepare_mtp_seed_indices" in pinned.patched["layers/attention/dsa_backend.py"]
    assert "def write_mtp_seed" in pinned.patched["layers/attention/index_topk_share.py"]
    mha = pinned.patched["models/deepseek_common/attention_forward_methods/forward_mha.py"]
    assert "write_mtp_seed(forward_batch, seed_buf, topk_indices)" in mha
    assert "seed_buf[: src.shape[0]].copy_(src)" not in mha


def _function(text, owner, name, scope):
    tree = ast.parse(text)
    body = tree.body if owner is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    ns = dict(scope)
    exec(compile(ast.fix_missing_locations(module), f"<pinned:{name}>", "exec"), ns)
    return ns[name]


class _Method:
    PAGED, RAGGED = 1, 2


def _capture(pinned, torch, indices, *, fused=True, ragged=True, flattened=True, select=None):
    # Upstream fixture: two requests share their first two physical slots.
    # Logical index 5 is outside the four-slot pool but maps to slot 2.
    table = torch.tensor([[3, 1, 0], [3, 1, 2]], dtype=torch.int32)
    metadata = NS(page_table_1=table, page_table_1_flattened=table.flatten() if flattened else None,
                  indexer_seq_lens_cpu=torch.tensor([3, 3]))
    prepare = _function(pinned.patched["layers/attention/dsa_backend.py"], "DeepseekSparseAttnBackend",
                        "prepare_mtp_seed_indices", {"torch": torch, "TopkTransformMethod": _Method})
    backend = NS(use_fused_topk=fused, forward_metadata=metadata,
                 get_topk_transform_method=lambda mode: _Method.RAGGED if ragged else _Method.PAGED)
    backend.prepare_mtp_seed_indices = lambda t, mode: prepare(backend, t, mode)
    write = _function(pinned.patched["layers/attention/index_topk_share.py"], None, "write_mtp_seed", {})
    rows = len(indices) if select is None else len(select)
    capture = torch.full((rows + 1, indices.shape[1]), -99, dtype=torch.int32)
    batch = NS(forward_mode="extend", spec_info=NS(dsa_seed_topk_capture=capture, dsa_seed_topk_select=select))
    mha = sys.modules.setdefault("sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha",
                                 type(sys)("forward_mha_fixture"))
    saved = getattr(mha, "resolve_attn_backend", None)
    mha.resolve_attn_backend = lambda fb: backend
    try:
        write(batch, capture, indices)
    finally:
        if saved is None:
            del mha.resolve_attn_backend
        else:
            mha.resolve_attn_backend = saved
    assert torch.all(capture[-1] == -99)
    return capture[:-1]


@pytest.mark.parametrize("flattened", [True, False])
@pytest.mark.parametrize("use_select", [False, True])
def test_ragged_seed_maps_shared_prefix_and_preserves_padding(pinned, flattened, use_select):
    torch = pytest.importorskip("torch")
    select = torch.tensor([2, 0]) if use_select else None
    indices = torch.tensor([[0, 2, -1], [3, 4, -1], [5, 1, -1]], dtype=torch.int32)
    original = indices.clone()
    expected = torch.tensor([[3, 0, -1], [3, 1, -1], [2, 1, -1]], dtype=torch.int32)
    if select is not None:
        expected = expected[select]
    actual = _capture(pinned, torch, indices, flattened=flattened, select=select)
    assert torch.equal(actual, expected)
    assert torch.equal(indices, original)


@pytest.mark.parametrize("fused,ragged", [(True, False), (False, True), (False, False)])
def test_paged_and_unfused_seed_contracts_are_preserved(pinned, fused, ragged):
    torch = pytest.importorskip("torch")
    indices = torch.tensor([[2, 0, -1], [1, 2, -1]], dtype=torch.int32)
    assert torch.equal(_capture(pinned, torch, indices, fused=fused, ragged=ragged), indices)


@pytest.mark.parametrize("invalid", [-2, 6])
def test_invalid_ragged_indices_are_not_clamped(pinned, invalid):
    torch = pytest.importorskip("torch")
    with pytest.raises((IndexError, RuntimeError)):
        _capture(pinned, torch, torch.tensor([[invalid, -1]], dtype=torch.int32))


def test_pinned_seed_range_survives_export(pinned):
    """B + C: a pinned logical range in a ragged seed maps to its physical slots."""
    torch = pytest.importorskip("torch")
    indices = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    seed = _capture(pinned, torch, indices)
    assert torch.equal(seed, torch.tensor([[3, 1, 0], [3, 1, 2]], dtype=torch.int32))
