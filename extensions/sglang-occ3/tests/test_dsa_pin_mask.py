"""DSA indexer per-row OCC-3 pin (patches/dsa-pin.v1.json), pure-torch CPU.

Runs the exact patched Indexer._occ3_pin_mask / _mask_init_and_local_tokens
extracted from the pinned v0.5.21 source after the real applier, then selects
with torch.topk as the unfused backend does. Not kernel or CUDA-graph evidence.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

EXT = Path(__file__).resolve().parents[1]
MANIFEST = EXT / "patches/dsa-pin.v1.json"
EXPECTED_PATHS = {"layers/attention/dsa/dsa_indexer.py"}
ATTESTED = Path(os.environ.get("OCC3_ATTESTED_SRT", "/nonexistent-occ3-attested"))


def test_manifest_shape_and_no_overlap():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["patch_set"] == "dsa-pin-v1"
    assert {f["path"] for f in manifest["files"]} == EXPECTED_PATHS
    others = set()
    for path in (EXT / "patches").glob("*.json"):
        if path != MANIFEST:
            others |= {f["path"] for f in json.loads(path.read_text())["files"]}
    assert not (others & EXPECTED_PATHS)


@pytest.fixture(scope="module")
def indexer(tmp_path_factory):
    torch = pytest.importorskip("torch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rel = manifest["files"][0]["path"]
    if not (ATTESTED / rel).is_file():
        pytest.skip("exact attested dsa_indexer.py absent; set OCC3_ATTESTED_SRT")
    spec = importlib.util.spec_from_file_location("_occ3_dsa_pin_patch", EXT / "deploy/v3/source_patch.py")
    applier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = applier
    spec.loader.exec_module(applier)
    root = tmp_path_factory.mktemp("occ3-dsa-pin") / "srt"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes((ATTESTED / rel).read_bytes())
    applier.apply(applier.prepare(root, manifest))
    tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    ns = {"torch": torch, "Optional": object}
    for name in ("_occ3_pin_mask", "_occ3_paged_row_batch", "_mask_init_and_local_tokens"):
        node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), name, "exec"), ns)
    obj = NS(num_init_tokens=0, num_local_tokens=0, _occ3_pin_mask=ns["_occ3_pin_mask"])
    mask = lambda *a, **k: ns["_mask_init_and_local_tokens"](obj, *a, **k)
    return NS(torch=torch, mask=mask, pin_mask=ns["_occ3_pin_mask"],
              paged_rows=ns["_occ3_paged_row_batch"], text=(root / rel).read_text(encoding="utf-8"))


def _pin(torch, starts, counts):
    n = len(starts)
    return (torch.tensor(starts, dtype=torch.int32), torch.tensor(counts, dtype=torch.int32),
            torch.zeros(n, dtype=torch.int32))


def _select(torch, logits, lengths, row_starts, topk):
    # Unfused reference selection (dsa_topk_backend._topk_unfused semantics).
    cols = torch.arange(logits.shape[1])[None, :]
    rs = row_starts[:, None]
    valid = (cols >= rs) & (cols < rs + lengths[:, None])
    masked = logits.masked_fill(~valid, float("-inf"))
    scores, idx = torch.topk(masked, topk, dim=-1)
    return idx.masked_fill(scores == float("-inf"), -1) - torch.where(scores == float("-inf"), 0, rs)


def _ident(torch, n):
    return torch.arange(n)


def test_paged_pin_selects_range_and_leaves_ordinary_rows_bitwise(indexer):
    torch = indexer.torch
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 600, generator=g) - 5.0
    original = logits.clone()
    lengths = torch.tensor([600, 500, 600, 300], dtype=torch.int32)
    pin = _pin(torch, [0, 0, 10, 0], [0, 128, 128, 0])
    indexer.mask(logits, lengths, pin=pin, row_batch=indexer.paged_rows(4, 1, "cpu"))
    assert torch.equal(logits[0], original[0]) and torch.equal(logits[3], original[3])
    topk = 256
    sel = _select(torch, logits, lengths, torch.zeros(4, dtype=torch.int32), topk)
    for row, base in ((1, 0), (2, 10)):
        chosen = set(sel[row].tolist())
        assert set(range(base, base + 128)) <= chosen
        assert len(chosen) == topk and -1 not in chosen
    assert pin[2].tolist() == [0, 128, 128, 0]


def test_count_zero_is_bitwise_noop_and_causal_guard(indexer):
    torch = indexer.torch
    logits = torch.randn(3, 64)
    original = logits.clone()
    lengths = torch.tensor([64, 64, 20], dtype=torch.int32)
    pin = _pin(torch, [0, 0, 5], [0, 0, 128])
    indexer.mask(logits, lengths, pin=pin, row_batch=_ident(torch, 3))
    assert torch.equal(logits[:2], original[:2])
    # Causal: only positions < length are forced (5..19).
    assert torch.isinf(logits[2, 5:20]).all()
    assert torch.equal(logits[2, 20:], original[2, 20:]) and torch.equal(logits[2, :5], original[2, :5])
    assert pin[2].tolist() == [0, 0, 15]
    # No pin argument, or a pin without a row map: the native path is untouched.
    other = original.clone()
    indexer.mask(other, lengths)
    indexer.mask(other, lengths, pin=_pin(torch, [0] * 3, [128] * 3), row_batch=None)
    assert torch.equal(other, original)


def test_ragged_flattened_index_space_and_chunk_slices(indexer):
    torch = indexer.torch
    # Two requests concatenated: KV [0,300) req0, [300,700) req1; rows of req1
    # carry ks=300. Pin req1 positions [0,128) -> flattened columns [300,428).
    rows_req0, rows_req1 = 3, 4
    ks = torch.tensor([0] * rows_req0 + [300] * rows_req1, dtype=torch.int32)
    tb = torch.tensor([0] * rows_req0 + [1] * rows_req1, dtype=torch.int32)
    lengths = torch.tensor([298, 299, 300, 397, 398, 399, 400], dtype=torch.int32)
    logits = torch.randn(7, 700) - 5.0
    original = logits.clone()
    pin = _pin(torch, [0, 0], [0, 128])  # one entry per batch row
    indexer.mask(logits, lengths, ks, pin=pin, row_batch=tb)
    assert torch.equal(logits[:rows_req0], original[:rows_req0])
    assert torch.isinf(logits[rows_req0:, 300:428]).all()
    assert torch.equal(logits[rows_req0:, :300], original[rows_req0:, :300])
    assert pin[2].tolist() == [0, 128]
    sel = _select(torch, logits, lengths, ks, 200)
    for row in range(rows_req0, 7):
        assert set(range(128)) <= set(sel[row].tolist())
    # Chunk path: the whole per-batch pin with the chunk's row map slice
    # matches the unchunked call, and forced accumulates by max.
    chunked = original.clone()
    pin2 = _pin(torch, [0, 0], [0, 128])
    for s, e in ((0, 2), (2, 5), (5, 7)):
        indexer.mask(chunked[s:e], lengths[s:e], ks[s:e], pin=pin2, row_batch=tb[s:e])
    assert torch.equal(chunked, logits)
    assert pin2[2].tolist() == [0, 128]


def test_marker_offset_pin_follows_prefix_offset(indexer):
    torch = indexer.torch
    # Marker layout: span starts at prefix_offset 900 inside a 3000-row row.
    logits = torch.randn(1, 3000) - 5.0
    lengths = torch.tensor([3000], dtype=torch.int32)
    pin = _pin(torch, [900], [96])
    indexer.mask(logits, lengths, pin=pin, row_batch=_ident(torch, 1))
    sel = set(_select(torch, logits, lengths, torch.zeros(1, dtype=torch.int32), 2048)[0].tolist())
    assert set(range(900, 996)) <= sel and len(sel) == 2048
    assert pin[2].tolist() == [96]


def test_expanded_rows_follow_static_width_like_seqlens_expanded(indexer):
    torch = indexer.torch
    draft = 3
    lengths = torch.tensor([1000, 1000], dtype=torch.int32).repeat_interleave(draft) + torch.arange(draft).repeat(2).int()
    logits = torch.randn(2 * draft, 1100)
    original = logits.clone()
    rows = indexer.paged_rows(2 * draft, draft, "cpu")
    assert rows.tolist() == [0, 0, 0, 1, 1, 1]
    pin = _pin(torch, [0, 7], [0, 64])
    indexer.mask(logits, lengths, pin=pin, row_batch=rows)
    assert torch.equal(logits[:draft], original[:draft])
    assert torch.isinf(logits[draft:, 7:71]).all()
    assert pin[2].tolist() == [0, 64]


def test_graph_padded_rows_beyond_live_batch_are_noops(indexer):
    torch = indexer.torch
    # A graph captured at bs=4 replays a raw_bs=2 batch: slots 2,3 are zeroed.
    logits = torch.randn(4, 300)
    original = logits.clone()
    lengths = torch.tensor([300, 300, 1, 1], dtype=torch.int32)
    pin = _pin(torch, [0, 10, 0, 0], [0, 50, 0, 0])
    indexer.mask(logits, lengths, pin=pin, row_batch=indexer.paged_rows(4, 1, "cpu"))
    assert torch.equal(logits[[0, 2, 3]], original[[0, 2, 3]])
    assert pin[2].tolist() == [0, 50, 0, 0]


def test_patched_call_sites_pass_row_maps(indexer):
    text = indexer.text
    assert text.count('getattr(forward_batch, "occ3_dsa_pin", None)') == 3
    assert "row_batch=token_to_batch_idx," in text
    assert "token_to_batch_idx[start:end]" in text
    assert "self._occ3_paged_row_batch(logits.shape[0], next_n, logits.device)" in text
    assert ".item()" not in text.split("def _occ3_pin_mask")[1].split("def _mask_init_and_local_tokens")[0]
