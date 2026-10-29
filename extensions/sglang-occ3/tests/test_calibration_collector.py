"""Calibration corpus collector: records, corpus writing, budgets, tamper refusal.

Spec sections 6, 7, 14, 15: whole-turn capture records under the predictor
convention, a family-grouped three-way split frozen on disk, nested
deterministic M budgets over the FIT partition, and a load path that verifies
every digest in the chain and refuses a tampered corpus.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

VOCAB = 512
DIM = 32


def _table(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, DIM, generator=generator)


def _lookup(seed=7):
    table = _table(seed)

    def lookup(token_id):
        if isinstance(token_id, int) and 0 <= token_id < VOCAB:
            return table[token_id]
        return None

    return lookup


def _record(index, rows=40, family=None, seed=0, span="content_span_v2", hidden=None,
            tokens=None, positions=None):
    from occ3_sglang.calibration_collector import CaptureRecord
    generator = torch.Generator().manual_seed(seed + 17 * index)
    if hidden is None:
        hidden = torch.randn(rows, DIM, generator=generator)
    if tokens is None:
        tokens = torch.randint(0, VOCAB, (rows,), generator=generator)
    if positions is None:
        positions = torch.arange(rows, dtype=torch.int64) * 3 + 17
    return CaptureRecord(turn_id=f"turn-{index:04d}",
                         family_id=(family if family is not None
                                    else f"fam-{index:04d}"),
                         hidden=hidden, token_ids=tokens,
                         absolute_positions=positions, span_policy=span,
                         span_metadata={"selected_rows": int(rows)})


def _records(count=12, rows=40, family_size=3, seed=0):
    return [_record(index, rows=rows, seed=seed,
                    family=(f"fam-{index // family_size:04d}" if family_size else None))
            for index in range(count)]


def _collect(records, *, provenance=None, **kwargs):
    from occ3_sglang.calibration_collector import Collector
    collector = Collector(provenance=provenance, **kwargs)
    lookup = _lookup()
    for record in records:
        collector.add(record, embedding_lookup=lookup)
    return collector


def _write(collector, path, **kwargs):
    kwargs.setdefault("fit_turns", 6)
    kwargs.setdefault("validation_turns", 3)
    kwargs.setdefault("test_turns", 3)
    return collector.write(path, **kwargs)


def _expected(records, fit=6, validation=3, test=3):
    from occ3_sglang.calibration_corpus import TurnBlock, split_manifest
    table = _table()
    blocks = [TurnBlock(turn_id=record.turn_id, hidden=record.hidden.clone(),
                        embeddings=torch.stack(
                            [table[int(value)] for value in record.token_ids], 0),
                        family=record.family_id)
              for record in records]
    return split_manifest(blocks, fit_turns=fit, validation_turns=validation,
                          test_turns=test)


def _reseal(manifest):
    body = {key: value for key, value in manifest.items()
            if key != "corpus_manifest_digest"}
    manifest["corpus_manifest_digest"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_floats(value):
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(_no_floats(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_floats(child) for child in value)
    return True


def _rewrite(path, manifest):
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8")


def test_capture_record_refuses_invalid_turns():
    from occ3_sglang.calibration_collector import CaptureRecord
    from occ3_sglang.latent_types import LatentError
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(8, DIM, generator=generator)
    tokens = torch.randint(0, VOCAB, (8,), generator=generator)
    positions = torch.arange(8, dtype=torch.int64) * 2

    def build(**overrides):
        values = dict(turn_id="turn-x", family_id="fam-x", hidden=hidden,
                      token_ids=tokens, absolute_positions=positions,
                      span_policy="content_span_v2", span_metadata={"rows": 8})
        values.update(overrides)
        return CaptureRecord(**values)

    build()
    for overrides, code in (
            ({"turn_id": "turn x"}, "profile"),
            ({"family_id": "fam x"}, "profile"),
            ({"span_policy": "content span"}, "profile"),
            ({"hidden": hidden.double()}, "profile"),
            ({"hidden": torch.randn(1, DIM, generator=generator)}, "profile"),
            ({"hidden": torch.randn(8, 2, generator=generator)}, "profile"),
            ({"token_ids": tokens.float()}, "profile"),
            ({"token_ids": tokens[:4]}, "profile"),
            ({"token_ids": tokens.clone().neg()}, "profile"),
            ({"absolute_positions": positions.flip(0)}, "profile"),
            ({"absolute_positions": positions.clone().neg()}, "profile"),
            ({"span_metadata": {"tensor": hidden}}, "invalid"),
            ({"span_metadata": {"items": ["y" * 140] * 128}}, "quota"),
    ):
        with pytest.raises(LatentError) as caught:
            build(**overrides)
        assert caught.value.code == code
    broken = hidden.clone()
    broken[0, 0] = float("nan")
    with pytest.raises(LatentError) as caught:
        build(hidden=broken)
    assert caught.value.code == "numerical"
    with pytest.raises(LatentError) as caught:
        build(hidden=torch.zeros(0, DIM),
              token_ids=torch.zeros(0, dtype=torch.long),
              absolute_positions=torch.zeros(0, dtype=torch.long))
    assert caught.value.code == "profile"


def test_collector_refuses_duplicates_caps_and_bounds():
    from occ3_sglang.calibration_collector import Collector
    from occ3_sglang.calibration_corpus import turn_order_key
    from occ3_sglang.latent_types import LatentError
    records = _records(count=3)
    collector = Collector(per_turn_row_cap=64)
    assert collector.add(records[0]) == 40
    with pytest.raises(LatentError) as caught:
        collector.add(records[0])
    assert caught.value.code == "identity"
    assert [collector.turn_ids()[0]] == [records[0].turn_id]
    with pytest.raises(LatentError) as caught:
        Collector(per_turn_row_cap=16).add(_record(50, rows=40))
    assert caught.value.code == "profile"
    bounded = Collector(max_turns=2)
    bounded.add(records[0])
    bounded.add(records[1])
    with pytest.raises(LatentError) as caught:
        bounded.add(records[2])
    assert caught.value.code == "quota"
    with pytest.raises(LatentError) as caught:
        Collector(max_total_bytes=4096).add(_record(51, rows=40))
    assert caught.value.code == "quota"
    mixed = Collector()
    mixed.add(_record(60, rows=6, hidden=torch.randn(6, DIM)))
    with pytest.raises(LatentError) as caught:
        mixed.add(_record(61, rows=6, hidden=torch.randn(6, 16)))
    assert caught.value.code == "profile"
    ordered = Collector()
    for record in records:
        ordered.add(record)
    assert ordered.turn_ids() == tuple(sorted((record.turn_id for record in records),
                                               key=turn_order_key))


def test_resolve_embeddings_refuses_unknown_tokens_and_records_refusals():
    from occ3_sglang.calibration_collector import Collector
    from occ3_sglang.latent_types import LatentError
    record = _record(0, rows=12)
    collector = Collector()
    with pytest.raises(LatentError) as caught:
        collector.resolve_embeddings(record, _lookup())
    assert caught.value.code == "identity"

    def unknown(token_id):
        return None

    with pytest.raises(LatentError) as caught:
        collector.add(record, embedding_lookup=unknown)
    assert caught.value.code == "identity"
    assert collector.turn_count == 0
    refusals = collector.refusals()
    assert refusals[0]["turn_id"] == record.turn_id
    assert refusals[0]["token_id"] == int(record.token_ids[0])
    assert refusals[0]["reason"] == "unknown"

    def raising(token_id):
        raise KeyError(token_id)

    collector.add(record)
    with pytest.raises(LatentError) as caught:
        collector.resolve_embeddings(record, raising)
    assert caught.value.code == "identity"
    assert collector.refusals()[-1]["reason"] == "lookup"

    def wrong_shape(token_id):
        return torch.zeros(DIM + 1)

    with pytest.raises(LatentError) as caught:
        collector.resolve_embeddings(record, wrong_shape)
    assert caught.value.code == "profile"
    assert collector.refusals()[-1]["reason"] == "shape"

    def nonfinite(token_id):
        row = torch.zeros(DIM)
        row[0] = float("nan")
        return row

    with pytest.raises(LatentError) as caught:
        collector.resolve_embeddings(record, nonfinite)
    assert caught.value.code == "numerical"
    assert collector.refusals()[-1]["reason"] == "nonfinite"

    embeddings = collector.resolve_embeddings(record, _lookup())
    table = _table()
    assert torch.equal(embeddings, torch.stack(
        [table[int(value)] for value in record.token_ids], 0))
    with pytest.raises(LatentError) as caught:
        collector.resolve_embeddings(record, _lookup())
    assert caught.value.code == "identity"


def test_write_load_round_trip_matches_split_manifest(tmp_path):
    from occ3_sglang.calibration_collector import load, read_manifest
    from occ3_sglang.calibration_corpus import PARTITION_FIELDS, SPLIT_SCHEMA, evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    records = _records()
    summary = _write(_collect(records), tmp_path)
    expected = _expected(records)
    loaded = load(tmp_path)
    assert set(loaded) == {"schema", "fit", "validation", "test", "turns",
                           "targets", "manifest_digest"}
    assert loaded["schema"] == SPLIT_SCHEMA
    assert loaded["manifest_digest"] == expected["manifest_digest"]
    assert loaded["turns"] == expected["turns"]
    for name in ("fit", "validation", "test"):
        partition = loaded[name]
        assert set(partition) == set(PARTITION_FIELDS)
        assert partition["turn_ids"] == expected[name]["turn_ids"]
        assert partition["row_turn_ids"] == expected[name]["row_turn_ids"]
        assert partition["manifest_digest"] == expected[name]["manifest_digest"]
        assert torch.equal(partition["hidden"], expected[name]["hidden"])
        assert torch.equal(partition["embeddings"], expected[name]["embeddings"])
    artifact = fit_map(loaded["fit"], {"rho": 0.02})
    record = evaluate_map(apply_map, artifact, loaded["validation"])
    assert record["rows"] == int(loaded["validation"]["hidden"].shape[0])
    manifest = read_manifest(tmp_path)
    assert manifest["corpus_manifest_digest"] == summary["corpus_manifest_digest"]


def test_family_grouping_keeps_families_whole(tmp_path):
    from occ3_sglang.calibration_collector import load
    records = _records(count=18, rows=8, family_size=2, seed=3)
    _write(_collect(records), tmp_path, fit_turns=8, validation_turns=4, test_turns=6)
    loaded = load(tmp_path)
    assignment = {}
    for name in ("fit", "validation", "test"):
        assert loaded[name]["turn_ids"]
        for turn_id in loaded[name]["turn_ids"]:
            assignment[turn_id] = name
    families = {}
    for record in records:
        families.setdefault(record.family_id, []).append(record.turn_id)
    for members in families.values():
        assert len({assignment[turn_id] for turn_id in members}) == 1
    assert loaded["turns"] == {"fit": 8, "validation": 4, "test": 6}
    ids = [set(loaded[name]["turn_ids"]) for name in ("fit", "validation", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == {record.turn_id for record in records}


def test_budget_selections_are_nested_capped_and_fit_only():
    from occ3_sglang.calibration_collector import M_ROW_BUDGETS, budget_selections
    from occ3_sglang.calibration_corpus import turn_order_key
    counts = [40] * 22 + [200, 90]
    ids = sorted((f"turn-{index:04d}" for index in range(len(counts))),
                 key=turn_order_key)
    row_turn_ids = tuple(turn_id for turn_id, count in zip(ids, counts)
                         for _ in range(count))
    rows = sum(counts)
    partition = {"hidden": torch.zeros(rows, 8),
                 "embeddings": torch.zeros(rows, 8),
                 "row_turn_ids": row_turn_ids, "turn_ids": tuple(ids),
                 "manifest_digest": ""}
    assert M_ROW_BUDGETS == (512, 1024, 2048)
    selections = budget_selections(partition, per_turn_cap=32)
    assert selections["512"]["rows"] == 512
    assert selections["512"]["turns"] == len(counts)
    assert selections["1024"]["rows"] == 768
    assert selections["2048"]["rows"] == 768
    smaller = selections["512"]["rows_per_turn"]
    larger = selections["1024"]["rows_per_turn"]
    assert set(smaller) <= set(larger)
    for turn_id, indices in smaller.items():
        assert indices == larger[turn_id][:len(indices)]
    for selection in selections.values():
        for turn_id, indices in selection["rows_per_turn"].items():
            position = ids.index(turn_id)
            start = sum(counts[:position])
            assert len(indices) <= 32
            assert indices == sorted(indices)
            assert all(start <= index < start + counts[position]
                       for index in indices)
    outside = {"turn-9999"}
    for selection in selections.values():
        assert set(selection["rows_per_turn"]).isdisjoint(outside)


def test_manifest_budget_selections_match_the_partition(tmp_path):
    from occ3_sglang.calibration_collector import budget_selections, load, read_manifest
    records = _records(count=48, rows=40, family_size=3, seed=9)
    _write(_collect(records), tmp_path, fit_turns=24, validation_turns=12,
           test_turns=12, budget_per_turn_cap=32)
    loaded = load(tmp_path)
    expected = budget_selections(loaded["fit"], per_turn_cap=32)
    manifest = read_manifest(tmp_path)
    assert manifest["partitions"]["fit"]["budget_selections"] == expected
    fit_ids = set(loaded["fit"]["turn_ids"])
    others = set(loaded["validation"]["turn_ids"]) | set(loaded["test"]["turn_ids"])
    for selection in expected.values():
        assert set(selection["rows_per_turn"]) <= fit_ids
        assert set(selection["rows_per_turn"]).isdisjoint(others)
    assert expected["512"]["rows"] == 512
    assert expected["1024"]["rows"] == 768
    assert expected["2048"]["rows"] == 768


def test_budget_selections_never_pad_and_report_actual_rows(tmp_path):
    from occ3_sglang.calibration_collector import read_manifest
    records = _records(count=9, rows=20, family_size=3, seed=5)
    summary = _write(_collect(records), tmp_path, fit_turns=3, validation_turns=3,
                     test_turns=3, budget_per_turn_cap=16)
    assert summary["budgets"]["512"] == {"rows": 48, "turns": 3}
    selection = read_manifest(tmp_path)["partitions"]["fit"]["budget_selections"]["512"]
    assert selection["rows"] == 48 < 512
    assert selection["turns"] == 3
    assert all(len(indices) == 16 for indices in selection["rows_per_turn"].values())


def test_validation_mutation_cannot_change_the_fit_partition(tmp_path):
    from occ3_sglang.calibration_collector import load, read_manifest
    records = _records()
    _write(_collect(records), tmp_path / "first")
    target = read_manifest(tmp_path / "first")["partitions"]["validation"]["turn_ids"][0]
    generator = torch.Generator().manual_seed(777)
    changed = [dataclasses.replace(
        record, hidden=torch.randn(record.hidden.shape, generator=generator))
        if record.turn_id == target else record for record in records]
    _write(_collect(changed), tmp_path / "second")
    left = load(tmp_path / "first")
    right = load(tmp_path / "second")
    assert left["fit"]["turn_ids"] == right["fit"]["turn_ids"]
    assert left["fit"]["manifest_digest"] == right["fit"]["manifest_digest"]
    assert torch.equal(left["fit"]["hidden"], right["fit"]["hidden"])
    assert left["validation"]["turn_ids"] == right["validation"]["turn_ids"]
    assert (left["validation"]["manifest_digest"]
            != right["validation"]["manifest_digest"])
    assert not torch.equal(left["validation"]["hidden"], right["validation"]["hidden"])
    first = read_manifest(tmp_path / "first")["partitions"]["fit"]["budget_selections"]
    second = read_manifest(tmp_path / "second")["partitions"]["fit"]["budget_selections"]
    assert first == second


def test_tampered_corpora_are_refused(tmp_path):
    from occ3_sglang.calibration_collector import load
    from occ3_sglang.latent_types import LatentError
    records = _records(count=6, rows=12, family_size=2, seed=11)
    _write(_collect(records), tmp_path / "corpus", fit_turns=2, validation_turns=2,
           test_turns=2)

    def copy(name):
        import shutil
        target = tmp_path / name
        shutil.copytree(tmp_path / "corpus", target)
        return target

    with pytest.raises(LatentError) as caught:
        load(tmp_path / "absent")
    assert caught.value.code == "profile"

    flipped = copy("flipped")
    path = flipped / "fit_hidden.pt"
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x40
    path.write_bytes(bytes(raw))
    with pytest.raises(LatentError) as caught:
        load(flipped)
    assert caught.value.code == "identity"

    edited = copy("edited")
    manifest = json.loads((edited / "manifest.json").read_text(encoding="utf-8"))
    manifest["partitions"]["fit"]["manifest_digest"] = "0" * 64
    _rewrite(edited / "manifest.json", manifest)
    with pytest.raises(LatentError) as caught:
        load(edited)
    assert caught.value.code == "identity"

    sealed = copy("sealed")
    path = sealed / "fit_embeddings.pt"
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    path.write_bytes(bytes(raw))
    manifest = json.loads((sealed / "manifest.json").read_text(encoding="utf-8"))
    manifest["partitions"]["fit"]["embeddings"]["sha256"] = hashlib.sha256(
        bytes(raw)).hexdigest()
    _reseal(manifest)
    _rewrite(sealed / "manifest.json", manifest)
    with pytest.raises(LatentError) as caught:
        load(sealed)
    assert caught.value.code == "identity"

    tokens = copy("tokens")
    path = tokens / "fit_token_ids.pt"
    changed = torch.load(path, weights_only=True, map_location="cpu").clone()
    changed[0] += 1
    torch.save(changed, path)
    manifest = json.loads((tokens / "manifest.json").read_text(encoding="utf-8"))
    manifest["partitions"]["fit"]["token_ids"]["sha256"] = _file_digest(path)
    _reseal(manifest)
    _rewrite(tokens / "manifest.json", manifest)
    with pytest.raises(LatentError) as caught:
        load(tokens)
    assert caught.value.code == "identity"

    budgeted = copy("budgeted")
    manifest = json.loads((budgeted / "manifest.json").read_text(encoding="utf-8"))
    selections = manifest["partitions"]["fit"]["budget_selections"]
    key = next(iter(selections))
    turn_id = next(iter(selections[key]["rows_per_turn"]))
    selections[key]["rows_per_turn"][turn_id].pop()
    _reseal(manifest)
    _rewrite(budgeted / "manifest.json", manifest)
    with pytest.raises(LatentError) as caught:
        load(budgeted)
    assert caught.value.code == "identity"


def test_manifest_carries_provenance_offset_and_corpus_digest(tmp_path):
    from occ3_sglang.calibration_collector import load, read_manifest
    from occ3_sglang.statebridge_calibrated import (PREDICTOR_POSITION_OFFSET,
                                                   corpus_digest)
    provenance = {"model_fingerprint": "sha256:" + "a" * 64,
                   "checkpoint_revision": "rev-1",
                   "tokenizer_fingerprint": "sha256:" + "b" * 64,
                   "embedding_id": "glm53-tp8-vocab",
                   "capture_method_id": "glm53.final-prenorm.capture.v1",
                   "capture_span_policy": "content_span_v2"}
    summary = _write(_collect(_records(), provenance=provenance), tmp_path)
    manifest = read_manifest(tmp_path)
    assert manifest["provenance"] == provenance
    assert manifest["predictor_position_offset"] == PREDICTOR_POSITION_OFFSET == -1
    loaded = load(tmp_path)
    fit = manifest["partitions"]["fit"]
    assert fit["corpus_digest"] == corpus_digest(
        loaded["fit"]["hidden"], loaded["fit"]["embeddings"],
        turn_ids=loaded["fit"]["turn_ids"])
    assert len((tmp_path / "manifest.json").read_bytes()) < 1024 * 1024
    assert _no_floats(manifest)
    assert _no_floats(summary)
    assert _no_floats(read_manifest(tmp_path)["partitions"])


def test_write_refuses_unresolved_turns(tmp_path):
    from occ3_sglang.calibration_collector import Collector
    from occ3_sglang.latent_types import LatentError
    collector = Collector()
    for record in _records(count=6, rows=8, family_size=2):
        collector.add(record)
    with pytest.raises(LatentError) as caught:
        _write(collector, tmp_path, fit_turns=2, validation_turns=2, test_turns=2)
    assert caught.value.code == "profile"


def test_embeddings_file_is_optional_and_rebuilt_via_lookup(tmp_path):
    from occ3_sglang.calibration_collector import load
    from occ3_sglang.latent_types import LatentError
    records = _records()
    _write(_collect(records), tmp_path / "lean", include_embeddings=False)
    assert not (tmp_path / "lean" / "fit_embeddings.pt").exists()
    with pytest.raises(LatentError) as caught:
        load(tmp_path / "lean")
    assert caught.value.code == "profile"
    lean = load(tmp_path / "lean", embedding_lookup=_lookup())
    _write(_collect(records), tmp_path / "full")
    full = load(tmp_path / "full")
    for name in ("fit", "validation", "test"):
        assert lean[name]["manifest_digest"] == full[name]["manifest_digest"]
        assert torch.equal(lean[name]["hidden"], full[name]["hidden"])
        assert torch.equal(lean[name]["embeddings"], full[name]["embeddings"])
