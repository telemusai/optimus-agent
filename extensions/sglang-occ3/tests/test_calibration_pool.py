"""Retained-capture row alignment, explicit partition writes, and pooling.

Three red-first contracts of the supplemental-collection design:

1. The collector ACCEPTS a retained content-only span (the v3
   ``content_tail_no_tool_v2`` policy) whose rows equal the receipt's
   ``span_content_count`` even though the captured window carried a thinking
   tail: the selected span is exactly the content run.
2. The collector REFUSES, fail-closed with the existing ``profile`` decline
   class, a v3 record whose rows disagree with ``span_content_count`` (the
   tokenize-aligned off-by-one shape). Row alignment is by SPAN content
   rows, never by a tokenize count.
3. The offline pool tool merges a legacy corpus with supplemental corpora:
   legacy partitions are preserved, supplemental turns land in the
   partitions their own corpus assigned them, bitwise-identical duplicates
   collapse, and any turn or embedding-row conflict is refused.
"""
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

VOCAB = 512
DIM = 32
LEGACY_POLICY = "content_tail_v2"
RETAINED_POLICY = "content_tail_no_tool_v2"


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


def _span_metadata(policy, rows, *, thinking=61, accepted_total=902):
    """A bounded receipt block shaped like SpanSelectionV2.span_block."""
    return {"span_schema": "occ3.span.v1", "span_policy": policy,
            "span_marker_profile": "glm45_glm47_xml_kv.v1",
            "span_marker_schema": "occ3.span.markers.v1",
            "span_reasoning_terminator": "thinking_end",
            "span_turn_end_attested": True,
            "span_thinking_count": int(thinking),
            "span_content_count": int(rows),
            "span_tool_call_count": 0,
            "span_tool_arguments_count": 0,
            "span_control_count": 0,
            "span_unknown_count": 0,
            "span_window_start_position": 463,
            "span_window_end_position": 463 + int(thinking) + int(rows),
            "span_contiguous": True,
            "span_contains_final_content": True,
            "span_contains_tool_call": policy != RETAINED_POLICY,
            "span_message_truncated": True,
            "capture_commit_id": hashlib.sha256(
                f"{policy}:{rows}".encode("ascii")).hexdigest(),
            "accepted_total": int(accepted_total)}


def _record(index, family, rows=31, policy=RETAINED_POLICY, seed=0,
            span_content_count=None, turn_id=None):
    from occ3_sglang.calibration_collector import CaptureRecord
    generator = torch.Generator().manual_seed(seed + 31 * index)
    hidden = torch.randn(rows, DIM, generator=generator)
    tokens = torch.randint(0, VOCAB, (rows,), generator=generator)
    positions = torch.arange(rows, dtype=torch.int64) * 3 + 17
    metadata = _span_metadata(policy, rows)
    if span_content_count is not None:
        metadata["span_content_count"] = int(span_content_count)
    return CaptureRecord(turn_id=turn_id or f"turn-{index:04d}",
                         family_id=family, hidden=hidden, token_ids=tokens,
                         absolute_positions=positions, span_policy=policy,
                         span_metadata=metadata)


def _corpus(records, path, *, fit_turns=1, validation_turns=1, test_turns=1,
            seed=7, provenance=None):
    from occ3_sglang.calibration_collector import Collector
    collector = Collector(provenance=provenance or {
        "method_id": "statebridge.glm53.final-prenorm-predictor."
                     "continuous-prefix.v1",
        "artifact_digest": "none", "hidden_dim": str(DIM),
        "layer_index": "77", "marker_profile_max_len": "64"})
    lookup = _lookup(seed)
    for record in records:
        collector.add(record, embedding_lookup=lookup)
    return collector.write(path, fit_turns=fit_turns,
                           validation_turns=validation_turns,
                           test_turns=test_turns)


# --------------------------------------------------------------------------- #
# 1 + 2: the v3 row-alignment rule
# --------------------------------------------------------------------------- #

def test_retained_policy_constant_matches_span_policy_module():
    from occ3_sglang import calibration_collector, span_policy
    assert (calibration_collector.RETAINED_CONTENT_ONLY_POLICY
            == span_policy.POLICY_NO_TOOL == RETAINED_POLICY)


def test_retained_content_only_span_is_accepted_with_thinking_tail(tmp_path):
    """(i) A retained content-only span with a thinking tail is accepted.

    The window metadata still carries the thinking rows (span_thinking_count
    61); the selected span is the 31-row content run, so rows must equal
    span_content_count and the frozen manifest records exactly that.
    """
    from occ3_sglang.calibration_collector import Collector, read_manifest
    record = _record(0, "fam-a", rows=31)
    assert record.span_metadata["span_thinking_count"] == 61
    collector = Collector()
    assert collector.add(record, embedding_lookup=_lookup()) == 31
    for index, family in ((1, "fam-b"), (2, "fam-c")):
        collector.add(_record(index, family, rows=6), embedding_lookup=_lookup())
    path = tmp_path / "corpus"
    summary = collector.write(path, fit_turns=1, validation_turns=1,
                             test_turns=1)
    manifest = read_manifest(path)
    assert summary["turns"] == {"fit": 1, "validation": 1, "test": 1}
    turn = next(entry for entry in manifest["turns"]
                if entry["turn_id"] == record.turn_id)
    assert turn["span_policy"] == RETAINED_POLICY
    assert turn["rows"] == 31
    assert turn["span_metadata"]["span_content_count"] == 31
    assert turn["span_metadata"]["span_thinking_count"] == 61


@pytest.mark.parametrize("offset", [1, -1])
def test_retained_off_by_one_rows_are_refused_fail_closed(tmp_path, offset):
    """(ii) A tokenize-aligned off-by-one is declined with code 'profile'.

    Pre-fix this record was silently accepted; the misaligned rows would
    poison the pooled fit. NEVER align by tokenize_count: the span content
    count is the only admissible row count for a retained capture.
    """
    from occ3_sglang.latent_types import LatentError
    with pytest.raises(LatentError) as failure:
        _record(0, "fam-a", rows=31, span_content_count=31 + offset)
    assert str(failure.value.code) == "profile"


def test_retained_missing_or_malformed_content_count_is_refused():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_collector import CaptureRecord
    base = _record(0, "fam-a", rows=31)
    for value in (None, "31", 31.0, True, -1):
        metadata = dict(base.span_metadata)
        metadata["span_content_count"] = value
        with pytest.raises(LatentError) as failure:
            CaptureRecord(turn_id=base.turn_id, family_id=base.family_id,
                          hidden=base.hidden, token_ids=base.token_ids,
                          absolute_positions=base.absolute_positions,
                          span_policy=RETAINED_POLICY, span_metadata=metadata)
        assert str(failure.value.code) == "profile"
    empty = dict(base.span_metadata)
    del empty["span_content_count"]
    with pytest.raises(LatentError) as failure:
        CaptureRecord(turn_id=base.turn_id, family_id=base.family_id,
                      hidden=base.hidden, token_ids=base.token_ids,
                      absolute_positions=base.absolute_positions,
                      span_policy=RETAINED_POLICY, span_metadata=empty)
    assert str(failure.value.code) == "profile"


def test_legacy_span_policies_stay_admissible_without_alignment():
    """The p4 legacy corpus (content_tail_v2) is pooled unchanged.

    Only the retained v3 policy carries the content-count invariant, so the
    legacy turns of the archived corpus stay admissible exactly as written.
    """
    from occ3_sglang.calibration_collector import CaptureRecord, Collector
    base = _record(0, "fam-a", rows=31, policy=LEGACY_POLICY)
    record = CaptureRecord(turn_id=base.turn_id, family_id=base.family_id,
                           hidden=base.hidden, token_ids=base.token_ids,
                           absolute_positions=base.absolute_positions,
                           span_policy=LEGACY_POLICY,
                           span_metadata=_span_metadata(LEGACY_POLICY, 44))
    assert Collector().add(record, embedding_lookup=_lookup()) == 31


# --------------------------------------------------------------------------- #
# write_partitions + load_turns (the pooling primitives)
# --------------------------------------------------------------------------- #

def _families(count, per_family, policy=RETAINED_POLICY, start=0):
    records = []
    index = start
    for family in range(count):
        for _ in range(per_family):
            records.append(_record(index, f"fam-{family:02d}", rows=6,
                                  policy=policy, seed=index))
            index += 1
    return records


def test_write_partitions_matches_split_manifest_digest(tmp_path):
    """An explicit assignment equal to the greedy split is byte-identical."""
    from occ3_sglang.calibration_collector import Collector, read_manifest
    records = _families(3, 2)
    plain = tmp_path / "plain"
    explicit = tmp_path / "explicit"
    summary_a = _corpus(records, plain)
    manifest = read_manifest(plain)
    assignment = {name: list(manifest["partitions"][name]["turn_ids"])
                 for name in ("fit", "validation", "test")}
    provenance = {"method_id": "statebridge.glm53.final-prenorm-predictor."
                                "continuous-prefix.v1",
                  "artifact_digest": "none", "hidden_dim": str(DIM),
                  "layer_index": "77", "marker_profile_max_len": "64"}
    collector = Collector(provenance=provenance)
    for record in records:
        collector.add(record, embedding_lookup=_lookup())
    summary_b = collector.write_partitions(
        explicit, assignment, targets=manifest["split"]["targets"])
    assert (summary_b["corpus_manifest_digest"]
            == summary_a["corpus_manifest_digest"])
    assert summary_b["turns"] == summary_a["turns"]
    assert (read_manifest(explicit)["corpus_manifest_digest"]
            == manifest["corpus_manifest_digest"])


def test_write_partitions_records_actual_targets_when_unset(tmp_path):
    from occ3_sglang.calibration_collector import Collector, load
    records = _families(4, 2)
    collector = Collector()
    for record in records:
        collector.add(record, embedding_lookup=_lookup())
    path = tmp_path / "corpus"
    collector.write_partitions(path, {"fit": [r.turn_id for r in records[:4]],
                                      "validation": [r.turn_id for r in records[4:6]],
                                      "test": [r.turn_id for r in records[6:8]]})
    partitions = load(path)
    assert partitions["turns"] == {"fit": 4, "validation": 2, "test": 2}
    assert partitions["targets"] == {"fit": 4, "validation": 2, "test": 2}


def test_write_partitions_refuses_broken_assignments(tmp_path):
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_collector import Collector
    records = _families(3, 2)
    collector = Collector()
    for record in records:
        collector.add(record, embedding_lookup=_lookup())
    ids = [r.turn_id for r in records]
    broken = [
        {"fit": ids[:2], "validation": ids[2:4]},                     # missing test
        {"fit": ids[:2], "validation": ids[2:4], "test": ids[3:6]},  # duplicate
        {"fit": ids[:2], "validation": ids[2:4], "test": ids[6:]},    # unassigned
        {"fit": [], "validation": ids[:4], "test": ids[4:8]},         # empty fit
        {"fit": ids[:1] + ids[2:3], "validation": ids[3:5],
         "test": ids[5:8]},                                           # family split
    ]
    for partitions in broken:
        with pytest.raises(LatentError):
            collector.write_partitions(tmp_path / "corpus", partitions)


def test_load_turns_round_trips_per_turn_records(tmp_path):
    from occ3_sglang.calibration_collector import load_turns
    records = _families(3, 2)
    path = tmp_path / "corpus"
    _corpus(records, path)
    from occ3_sglang.calibration_collector import turn_order_key
    turns = load_turns(path)
    assert [turn["turn_id"] for turn in turns] == sorted(
        (record.turn_id for record in records), key=turn_order_key)
    by_id = {record.turn_id: record for record in records}
    for turn in turns:
        record = by_id[turn["turn_id"]]
        assert turn["family_id"] == record.family_id
        assert turn["span_policy"] == record.span_policy
        assert turn["rows"] == int(record.hidden.shape[0])
        assert torch.equal(turn["hidden"], record.hidden)
        assert torch.equal(turn["token_ids"], record.token_ids)
        assert torch.equal(turn["absolute_positions"], record.absolute_positions)
        assert torch.equal(turn["embeddings"], torch.stack(
            [_table()[int(value)] for value in record.token_ids], 0))


# --------------------------------------------------------------------------- #
# 3: the offline pool tool
# --------------------------------------------------------------------------- #

def _pool_tool():
    from deploy.v3 import calibration_pool
    return calibration_pool


def _write_corpus(records, path, *, seed=7, provenance=None):
    return _corpus(records, path, seed=seed, provenance=provenance)


def test_pool_merges_legacy_partitions_with_supplemental(tmp_path):
    """(iii-a) Merge correctness: legacy partitions preserved, supplemental
    turns land in the partitions their own corpus assigned them."""
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir)
    supplemental = _families(3, 2, policy=RETAINED_POLICY, start=100)
    supplemental_dir = tmp_path / "supplemental"
    _write_corpus(supplemental, supplemental_dir)
    out = tmp_path / "pooled"
    summary = tool.pool(str(legacy_dir), [str(supplemental_dir)], str(out))

    from occ3_sglang.calibration_collector import load, read_manifest
    manifest = read_manifest(out)
    partitions = load(out)
    legacy_by_id = {record.turn_id: record for record in legacy}
    supplemental_by_id = {record.turn_id: record for record in supplemental}
    expected = {}
    for record in legacy + supplemental:
        expected[record.turn_id] = record
    from occ3_sglang.calibration_collector import turn_order_key
    assert len(manifest["turns"]) == 12
    assert manifest["turns"] == sorted(
        manifest["turns"], key=lambda entry: turn_order_key(entry["turn_id"]))
    seen = {}
    for entry in manifest["turns"]:
        seen[entry["turn_id"]] = entry["partition"]
    for record in legacy:
        assert seen[record.turn_id]  # legacy turns all present
    # every family stayed whole, and every partition holds 2+2 turns
    for name in ("fit", "validation", "test"):
        assert partitions["turns"][name] == 4
        rows = 0
        for turn_id, count in zip(manifest["partitions"][name]["turn_ids"],
                                  manifest["partitions"][name]["row_counts"]):
            record = expected[turn_id]
            rows += count
            assert count == int(record.hidden.shape[0])
        assert manifest["partitions"][name]["rows"] == rows
    # the supplemental turns are the retained v3 policy, the legacy are not
    policies = {entry["turn_id"]: entry["span_policy"]
                for entry in manifest["turns"]}
    assert all(policies[record.turn_id] == RETAINED_POLICY
               for record in supplemental)
    assert all(policies[record.turn_id] == LEGACY_POLICY
               for record in legacy)
    assert summary["deduped_turns"] == []
    assert len(summary["pooled_from"]) == 2
    # the pooled manifest chains verify and the budgets rebuilt
    assert set(manifest["partitions"]["fit"]["budget_selections"]) == {
        "512", "1024", "2048"}
    assert partitions["manifest_digest"] == manifest["split"]["manifest_digest"]


def test_pool_keeps_legacy_turn_in_its_own_partition(tmp_path):
    """Legacy partitions preserved: a supplemental corpus cannot move a
    legacy family's partition, and identical turns dedupe silently."""
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir)
    out = tmp_path / "pooled"
    summary = tool.pool(str(legacy_dir), [str(legacy_dir)], str(out))
    from occ3_sglang.calibration_collector import read_manifest
    manifest = read_manifest(out)
    assert len(manifest["turns"]) == 6
    assert sorted(summary["deduped_turns"]) == sorted(
        record.turn_id for record in legacy)
    assert summary["pooled_from"][0] == summary["pooled_from"][1]


def test_pool_refuses_turn_conflict(tmp_path):
    """(iii-b) The same turn id with divergent content is refused."""
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir)
    # supplemental corpus reuses one legacy turn id with different rows
    conflicting = _record(0, "fam-99", rows=6, policy=RETAINED_POLICY,
                          seed=999, turn_id=legacy[0].turn_id)
    supplemental = [conflicting] + _families(2, 2, policy=RETAINED_POLICY,
                                             start=100)
    supplemental_dir = tmp_path / "supplemental"
    _write_corpus(supplemental, supplemental_dir, seed=9)
    with pytest.raises(tool.PoolError) as failure:
        tool.pool(str(legacy_dir), [str(supplemental_dir)],
                  str(tmp_path / "out"))
    assert failure.value.code == "identity"


def test_pool_refuses_embedding_row_conflict(tmp_path):
    """(iii-c) One token id resolving to different rows is refused.

    The supplemental corpus reuses the legacy token ids (so the merged
    lookup must merge them) but was written against a different embedding
    table: the same id now resolves to two different rows.
    """
    from occ3_sglang.calibration_collector import CaptureRecord
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir, seed=7)
    supplemental = []
    for index, record in enumerate(legacy):
        rows = int(record.token_ids.shape[0])
        supplemental.append(CaptureRecord(
            turn_id=f"turn-s{index:04d}", family_id=f"sup-{index % 3}",
            hidden=torch.randn(rows, DIM, generator=torch.Generator().manual_seed(500 + index)),
            token_ids=record.token_ids.clone(),
            absolute_positions=torch.arange(rows, dtype=torch.int64) * 2 + 5,
            span_policy=RETAINED_POLICY,
            span_metadata=_span_metadata(RETAINED_POLICY, rows)))
    supplemental_dir = tmp_path / "supplemental"
    # seed 9: a DIFFERENT embedding table for the same token ids
    _write_corpus(supplemental, supplemental_dir, seed=9)
    with pytest.raises(tool.PoolError) as failure:
        tool.pool(str(legacy_dir), [str(supplemental_dir)],
                  str(tmp_path / "out"))
    assert failure.value.code == "identity"


def test_pool_refuses_v3_row_misalignment_in_a_written_corpus(tmp_path):
    """A corpus that passes its own digest chain but carries a misaligned
    retained turn is refused when the turns re-enter CaptureRecord."""
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir)
    supplemental = _families(3, 2, policy=RETAINED_POLICY, start=100)
    supplemental_dir = tmp_path / "supplemental"
    _write_corpus(supplemental, supplemental_dir)
    # tamper one span_content_count and reseal the manifest digest chain
    manifest = json.loads((supplemental_dir / "manifest.json").read_text())
    target = manifest["turns"][0]
    assert target["span_policy"] == RETAINED_POLICY
    target["span_metadata"]["span_content_count"] = target["rows"] + 1
    body = {key: value for key, value in manifest.items()
            if key != "corpus_manifest_digest"}
    manifest["corpus_manifest_digest"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    (supplemental_dir / "manifest.json").write_text(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(tool.PoolError) as failure:
        tool.pool(str(legacy_dir), [str(supplemental_dir)],
                  str(tmp_path / "out"))
    assert failure.value.code == "record_refused"


def test_pool_cli_writes_summary(tmp_path, capsys):
    tool = _pool_tool()
    legacy = _families(3, 2, policy=LEGACY_POLICY, start=0)
    legacy_dir = tmp_path / "legacy"
    _write_corpus(legacy, legacy_dir)
    supplemental = _families(3, 2, policy=RETAINED_POLICY, start=100)
    supplemental_dir = tmp_path / "supplemental"
    _write_corpus(supplemental, supplemental_dir)
    out = tmp_path / "pooled"
    json_out = tmp_path / "summary.json"
    code = tool.main(["--legacy", str(legacy_dir), "--supplemental",
                      str(supplemental_dir), "--out", str(out),
                      "--json-out", str(json_out)])
    assert code == 0
    payload = json.loads(json_out.read_text())
    assert payload["schema"] == tool.POOL_SCHEMA
    assert payload["turns"] == {"fit": 4, "validation": 4, "test": 4}
    assert (out / "manifest.json").is_file()
