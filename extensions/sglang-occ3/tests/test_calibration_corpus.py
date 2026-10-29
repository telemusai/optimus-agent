"""Three-way manifests, shared evaluation, the promotion gate and grid selection."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _model(rank, dim, seed=1000):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rank, dim, generator=generator) / rank ** 0.5,
            torch.randn(rank, dim, generator=generator) / rank ** 0.5)


def _turns(count=16, rows=40, dim=32, rank=8, kappa=0.9, seed=0, independent=False,
           constant_embeddings=False, family_size=0, start=0):
    from occ3_sglang.calibration_corpus import TurnBlock
    left, right = _model(rank, dim)
    other_right = _model(rank + 1, dim, seed=2000)[1]
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for index in range(start, start + count):
        factor = torch.randn(rows, rank, generator=generator)
        noise_h = torch.randn(rows, dim, generator=generator)
        noise_e = torch.randn(rows, dim, generator=generator)
        if independent:
            hidden = torch.randn(rows, dim, generator=generator)
            embeddings = torch.randn(rows, dim, generator=generator)
        elif constant_embeddings:
            hidden = torch.randn(rows, dim, generator=generator)
            embeddings = torch.full((rows, dim), 0.5)
        else:
            hidden = kappa ** 0.5 * (factor @ left) + (1.0 - kappa) ** 0.5 * noise_h
            embeddings = kappa ** 0.5 * (factor @ right) + (1.0 - kappa) ** 0.5 * noise_e
        tokens = (kappa ** 0.5 * (torch.randn(rows, rank + 1, generator=generator) @ other_right)
                  + (1.0 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator))
        family = None if family_size == 0 else f"fam-{index // family_size:03d}"
        blocks.append(TurnBlock(turn_id=f"turn-{index:03d}", hidden=hidden,
                                embeddings=embeddings, reference_embeddings=tokens,
                                wrong_sender_hidden=torch.randn(rows, dim,
                                                                generator=generator),
                                family=family))
    return blocks


def _manifest(blocks, fit_turns=8, validation_turns=4, test_turns=4):
    from occ3_sglang.calibration_corpus import split_manifest
    return split_manifest(blocks, fit_turns=fit_turns, validation_turns=validation_turns,
                          test_turns=test_turns)


def _fitted(blocks=None, provenance=True):
    from occ3_sglang.statebridge_calibrated import CalibrationProvenance, fit_map
    manifest = _manifest(blocks if blocks is not None else _turns())
    identity = None
    if provenance:
        identity = CalibrationProvenance(
            layer_index=77, model_fingerprint="sha256:" + "a" * 64,
            checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
            embedding_id="glm53-tp8-vocab",
            capture_method_id="glm53.final-prenorm.capture.v1")
    artifact = fit_map(manifest["fit"], {"rho": 0.02}, provenance=identity)
    return manifest, artifact


def _validation_block(manifest, blocks, index=0):
    turn_ids = list(manifest["validation"]["turn_ids"])
    wanted = turn_ids[index]
    return next(block for block in blocks if block.turn_id == wanted)


def test_turn_block_refuses_a_shape_mismatch_and_bad_family():
    from occ3_sglang.calibration_corpus import TurnBlock
    from occ3_sglang.latent_types import LatentError
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(8, 16, generator=generator)
    with pytest.raises(LatentError):
        TurnBlock(turn_id="turn-x", hidden=hidden, embeddings=torch.randn(8, 32,
                                                                         generator=generator))
    with pytest.raises(LatentError):
        TurnBlock(turn_id="turn x", hidden=hidden, embeddings=hidden.clone())
    with pytest.raises(LatentError):
        TurnBlock(turn_id="turn-x", hidden=hidden, embeddings=hidden.clone(),
                   reference_embeddings=torch.randn(4, 16,
                                                     generator=generator))
    with pytest.raises(LatentError):
        TurnBlock(turn_id="turn-x", hidden=hidden, embeddings=hidden.clone(),
                   family="family 9")
    TurnBlock(turn_id="turn-x", hidden=hidden, embeddings=hidden.clone(),
              family="fam-9")


def test_duplicate_turn_ids_are_refused():
    from occ3_sglang.calibration_corpus import TurnBlock, ordered_blocks
    from occ3_sglang.latent_types import LatentError
    blocks = _turns()
    duplicate = TurnBlock(turn_id=blocks[0].turn_id, hidden=blocks[1].hidden,
                          embeddings=blocks[1].embeddings)
    with pytest.raises(LatentError) as caught:
        ordered_blocks((blocks[0], duplicate))
    assert caught.value.code == "identity"


def test_split_manifest_partitions_are_whole_turn_disjoint_and_covering():
    from occ3_sglang.calibration_corpus import PARTITION_FIELDS
    blocks = _turns()
    manifest = _manifest(blocks)
    for name in ("fit", "validation", "test"):
        partition = manifest[name]
        assert set(partition) == set(PARTITION_FIELDS)
        assert len(partition["manifest_digest"]) == 64
        assert len(partition["row_turn_ids"]) == partition["hidden"].shape[0]
        assert set(partition["row_turn_ids"]) == set(partition["turn_ids"])
    ids = [set(manifest[name]["turn_ids"]) for name in ("fit", "validation", "test")]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == {block.turn_id for block in blocks}
    assert manifest["turns"] == {"fit": 8, "validation": 4, "test": 4}


def test_split_manifest_is_deterministic_regardless_of_input_order():
    blocks = _turns()
    first = _manifest(blocks)
    second = _manifest(list(reversed(blocks)))
    assert first["manifest_digest"] == second["manifest_digest"]
    for name in ("fit", "validation", "test"):
        assert first[name]["manifest_digest"] == second[name]["manifest_digest"]
        assert first[name]["turn_ids"] == second[name]["turn_ids"]


def test_split_manifest_keeps_families_whole():
    blocks = _turns(family_size=4)
    manifest = _manifest(blocks)
    family_turns = {}
    for index, block in enumerate(blocks):
        family_turns.setdefault(block.family, set()).add(block.turn_id)
    for name in ("fit", "validation", "test"):
        assigned = set(manifest[name]["turn_ids"])
        for family, turns in family_turns.items():
            assert not turns & assigned or turns <= assigned


def test_split_manifest_refuses_impossible_or_bad_targets():
    from occ3_sglang.latent_types import LatentError
    blocks = _turns(count=15, family_size=5)
    with pytest.raises(LatentError) as caught:
        _manifest(blocks, fit_turns=12, validation_turns=2, test_turns=2)
    assert caught.value.code == "profile"
    blocks = _turns()
    with pytest.raises(LatentError) as caught:
        _manifest(blocks, fit_turns=0, validation_turns=4, test_turns=4)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        _manifest(blocks, fit_turns=8.0, validation_turns=4, test_turns=4)
    assert caught.value.code == "profile"


def test_row_budget_is_a_supported_value_and_reported():
    from occ3_sglang.calibration_corpus import M_ROW_BUDGETS, select_calibration_rows
    from occ3_sglang.latent_types import LatentError
    assert M_ROW_BUDGETS == (512, 1024, 2048)
    blocks = _turns(count=12, rows=200)
    with pytest.raises(LatentError) as caught:
        select_calibration_rows(blocks, 600)
    assert caught.value.code == "profile"
    selected, used = select_calibration_rows(blocks, 1024)
    assert used == 1000 <= 1024
    assert used == sum(block.rows for block in selected)


def test_m_truncation_is_deterministic_and_takes_whole_turns():
    from occ3_sglang.calibration_corpus import ordered_blocks, select_calibration_rows
    blocks = _turns(count=12, rows=200)
    ordered = ordered_blocks(blocks)
    first = select_calibration_rows(ordered, 512)
    assert first[1] == 400
    assert [block.turn_id for block in first[0]] == [
        block.turn_id for block in ordered[:2]]
    assert select_calibration_rows(list(reversed(ordered)), 512) == first
    assert select_calibration_rows(ordered, 1024)[1] == 1000
    assert select_calibration_rows(ordered, 2048)[1] == 2000
    for budget, rows in ((512, 400), (1024, 1000), (2048, 2000)):
        selected, used = select_calibration_rows(ordered, budget)
        assert used == sum(block.rows for block in selected) == rows


def test_evaluation_records_every_contract_key():
    from occ3_sglang.calibration_corpus import RECORD_KEYS, VIEW_KEYS, evaluate_map
    from occ3_sglang.calibration_corpus import validate_metrics_record
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert set(record) == set(RECORD_KEYS)
    assert validate_metrics_record(record)
    assert record["rows"] == manifest["validation"]["hidden"].shape[0]
    assert record["turns"] == 4
    assert set(record["pooled"]) == set(VIEW_KEYS)
    assert set(record["turn_macro"]) == set(VIEW_KEYS) - {"rows"}
    assert set(record["per_turn"]) == set(manifest["validation"]["turn_ids"])
    for view in record["per_turn"].values():
        assert set(view) == set(VIEW_KEYS)
    assert record["r2_standard"] > 0.5
    assert record["raw_row_norm_mean"] > 0.0
    assert record["undefined_denominators"] == []


def test_raw_and_served_metrics_are_kept_distinct():
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    raw = evaluate_map(apply_map, artifact, manifest["validation"])
    assert raw["served_r2_standard"] == raw["r2_standard"]
    assert raw["served_row_norm_mean"] == raw["raw_row_norm_mean"]
    nu = 5.0 * raw["raw_row_norm_mean"]
    served = evaluate_map(apply_map, artifact, manifest["validation"],
                          serving={"kind": "n1"}, nu=nu)
    assert served["served_row_norm_mean"] == pytest.approx(nu, rel=1e-4)
    assert served["served_r2_standard"] != served["r2_standard"]
    assert served["served_r2_standard"] < served["r2_standard"]
    assert served["nu"] == nu and served["serving"] == {"kind": "n1"}


def test_negative_r2_is_recorded_never_suppressed():
    from occ3_sglang.calibration_corpus import evaluate_map, validate_metrics_record
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted(_turns(independent=True, seed=3))
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert validate_metrics_record(record)
    assert record["r2_standard"] is None or record["r2_standard"] < 0.3


def test_zero_denominators_are_recorded_as_undefined():
    from occ3_sglang.calibration_corpus import evaluate_map, validate_metrics_record
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted(_turns(constant_embeddings=True, seed=5))
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert validate_metrics_record(record)
    assert record["r2_standard"] is None and record["r2_vs_fit_mean"] is None
    assert record["sst_test"] == 0.0 and record["sse_vs_fit_mean"] == 0.0
    assert set(record["undefined_denominators"]) >= {"r2_standard", "r2_vs_fit_mean"}
    first_turn = manifest["validation"]["turn_ids"][0]
    assert f"r2_standard@{first_turn}" in record["undefined_denominators"]


def test_turn_macro_is_the_unweighted_mean_over_turns():
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    values = [record["per_turn"][turn_id]["r2_standard"]
              for turn_id in manifest["validation"]["turn_ids"]]
    assert record["turn_macro"]["r2_standard"] == pytest.approx(sum(values) / len(values))
    assert "rows" not in record["turn_macro"]


def test_evaluation_refuses_a_malformed_partition():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    with pytest.raises(LatentError) as caught:
        evaluate_map(apply_map, artifact, {**manifest["validation"], "extra": 1})
    assert caught.value.code == "profile"
    broken = {**manifest["validation"],
              "row_turn_ids": manifest["validation"]["row_turn_ids"][:-1]}
    with pytest.raises(LatentError) as caught:
        evaluate_map(apply_map, artifact, broken)
    assert caught.value.code == "profile"


def test_serving_n1_requires_a_vocabulary_norm():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    with pytest.raises(LatentError) as caught:
        evaluate_map(apply_map, artifact, manifest["validation"], serving={"kind": "n1"})
    assert caught.value.code == "profile"


def test_record_digest_is_canonical():
    from occ3_sglang.calibration_corpus import evaluate_map, record_digest
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    first = evaluate_map(apply_map, artifact, manifest["validation"])
    second = evaluate_map(apply_map, artifact, manifest["validation"])
    assert record_digest(first) == record_digest(second)
    assert len(record_digest(first)) == 64
    altered = dict(first, rows=first["rows"] + 1)
    assert record_digest(altered) != record_digest(first)


def test_validate_metrics_record_rejects_tampering():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_corpus import evaluate_map, validate_metrics_record
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert validate_metrics_record(record)
    with pytest.raises(LatentError):
        validate_metrics_record({k: v for k, v in record.items() if k != "rows"})
    with pytest.raises(LatentError):
        validate_metrics_record(dict(record, r2_standard="high"))
    with pytest.raises(LatentError):
        validate_metrics_record(dict(record, rows=3.5))
    with pytest.raises(LatentError):
        validate_metrics_record(dict(record, schema="other"))


def test_the_gate_allows_refuses_and_binds():
    from occ3_sglang.calibration_corpus import evaluate_map, validate_for_activation
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert validate_for_activation(artifact, record, {"min_r2_standard": 0.1,
                                                      "min_turns": 1}) == ("allow", None)
    assert validate_for_activation(artifact, record, {"min_r2_standard": 0.99}) == (
        "refuse", "min_r2_standard")
    assert validate_for_activation(artifact, record, {"r2_standard": 0.1}) == (
        "refuse", "policy")
    assert validate_for_activation(artifact, record, {"min_nonsense": 1}) == (
        "refuse", "policy")
    assert validate_for_activation(artifact, record, {}) == ("refuse", "policy")
    assert validate_for_activation(artifact, {"rows": 1}, {}) == ("refuse", "record")
    assert validate_for_activation(artifact, record, {"min_mean_row_cosine": 2.0}) == (
        "refuse", "min_mean_row_cosine")
    other_manifest, other = _fitted(_turns(seed=91))
    other_record = evaluate_map(apply_map, other, other_manifest["validation"])
    assert validate_for_activation(artifact, other_record, {"min_r2_standard": 0.1}) == (
        "refuse", "artifact_digest")


def test_the_gate_refuses_unbound_artifacts_and_undefined_metrics():
    from occ3_sglang.calibration_corpus import evaluate_map, validate_for_activation
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted(provenance=False)
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert validate_for_activation(artifact, record, {"min_r2_standard": 0.1}) == (
        "refuse", "unbound")
    flat_manifest, flat = _fitted(_turns(constant_embeddings=True, seed=7))
    flat_record = evaluate_map(apply_map, flat, flat_manifest["validation"])
    assert validate_for_activation(flat, flat_record, {"min_r2_standard": 0.1}) == (
        "refuse", "undefined:r2_standard")


def test_bind_validation_evidence_pins_the_record():
    from dataclasses import replace
    from occ3_sglang.calibration_corpus import (bind_validation_evidence, evaluate_map,
                                                record_digest, validate_for_activation)
    from occ3_sglang.statebridge_calibrated import (TENSOR_NAMES, apply_map,
                                                    artifact_digest)
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    policy = {"min_r2_standard": 0.1}
    pinned = bind_validation_evidence(artifact, record, {"kind": "n1"})
    assert pinned.binding.validation_record_digest == record_digest(record)
    assert pinned.binding.validation_artifact_digest == record["artifact_digest"]
    assert pinned.binding.serving == {"kind": "n1"}
    assert pinned.content_digest is None
    for name in TENSOR_NAMES:
        assert torch.equal(getattr(pinned, name), getattr(artifact, name))
    assert validate_for_activation(pinned, record, policy) == ("allow", None)
    tampered = dict(record, r2_standard=record["r2_standard"] + 1.0)
    assert validate_for_activation(pinned, tampered, policy) == (
        "refuse", "validation_evidence")
    assert artifact_digest(pinned) != record["artifact_digest"]


def test_bind_validation_evidence_refuses_a_foreign_record():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_corpus import bind_validation_evidence, evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    other_manifest, other = _fitted(_turns(seed=93))
    other_record = evaluate_map(apply_map, other, other_manifest["validation"])
    with pytest.raises(LatentError) as caught:
        bind_validation_evidence(artifact, other_record, None)
    assert caught.value.code == "identity"


def test_grid_selection_runs_the_rho_grid_on_validation_only():
    from occ3_sglang.calibration_corpus import (RHO_GRID, evaluate_map,
                                                select_calibrated_map)
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(_turns(seed=13))
    selection = select_calibrated_map(fit_map, apply_map, manifest["fit"],
                                      manifest["validation"], nu=2.0)
    rhos = sorted(candidate["hyperparams"]["rho"]
                  for candidate in selection["candidates"])
    assert rhos == sorted(RHO_GRID)
    manual = []
    for rho in RHO_GRID:
        artifact = fit_map(manifest["fit"], {"rho": rho})
        manual.append((rho, evaluate_map(apply_map, artifact,
                                         manifest["validation"])["r2_standard"]))
    best = max(manual, key=lambda pair: pair[1])[0]
    assert selection["hyperparams"] == {"rho": best}
    assert (selection["validation_record"]["partition_manifest_digest"]
            == manifest["validation"]["manifest_digest"])
    served_best = max(selection["serving_candidates"],
                      key=lambda entry: entry["served_r2_standard"])["variant"]
    assert selection["serving_variant"] == served_best


def test_grid_selection_never_touches_the_test_partition():
    from occ3_sglang.calibration_corpus import select_calibrated_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    blocks = _turns(seed=17)
    manifest = _manifest(blocks)
    seen = {"fit": [], "apply": []}

    def fit_spy(partition, hyperparams, **kwargs):
        seen["fit"].append(partition["manifest_digest"])
        return fit_map(partition, hyperparams, **kwargs)

    def apply_spy(artifact, hidden, **kwargs):
        seen["apply"].append(hidden)
        return apply_map(artifact, hidden, **kwargs)

    select_calibrated_map(fit_spy, apply_spy, manifest["fit"], manifest["validation"],
                          nu=1.0)
    assert set(seen["fit"]) == {manifest["fit"]["manifest_digest"]}
    for hidden in seen["apply"]:
        assert torch.equal(hidden, manifest["validation"]["hidden"])
        assert not torch.equal(hidden, manifest["test"]["hidden"])


def test_grid_n1_selection_requires_a_vocabulary_norm():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.calibration_corpus import select_calibrated_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(_turns(seed=19))
    with pytest.raises(LatentError) as caught:
        select_calibrated_map(fit_map, apply_map, manifest["fit"], manifest["validation"],
                              serving_grid=("n1",))
    assert caught.value.code == "profile"
    selection = select_calibrated_map(fit_map, apply_map, manifest["fit"],
                                      manifest["validation"], serving_grid=("n1",), nu=2.0)
    assert selection["serving_variant"] == "n1" and selection["serving"] == {"kind": "n1"}


def test_controls_separate_correct_from_shuffled_and_null():
    from occ3_sglang.calibration_corpus import CONTROL_KINDS, score_controls
    from occ3_sglang.statebridge_calibrated import apply_map
    blocks = _turns()
    manifest, artifact = _fitted(blocks)
    scalar = float(manifest["fit"]["embeddings"].norm(dim=-1).mean())
    scores = score_controls(apply_map, artifact, _validation_block(manifest, blocks),
                            target_norm=scalar)
    assert set(scores) == set(CONTROL_KINDS)
    correct = scores["correct"]["cosine"]
    assert correct > 0.8
    for control in ("shuffled", "null", "wrong-sender", "reference-token-only"):
        assert scores[control]["attached"] == 1
        assert scores[control]["cosine"] < correct - 0.3, (control, scores[control])


def test_no_attachment_has_no_score_and_reference_tokens_are_not_the_state():
    from occ3_sglang.calibration_corpus import SOURCE_NONE, SOURCE_TOKEN, score_controls
    from occ3_sglang.statebridge_calibrated import apply_map
    blocks = _turns()
    manifest, artifact = _fitted(blocks)
    scalar = float(manifest["fit"]["embeddings"].norm(dim=-1).mean())
    block = _validation_block(manifest, blocks, index=1)
    scores = score_controls(apply_map, artifact, block, target_norm=scalar)
    assert scores["no-attachment"] == {"kind": "no-attachment", "source": SOURCE_NONE,
                                       "rows": 0, "attached": 0, "cosine": None,
                                       "relative_error": None}
    assert scores["reference-token-only"]["source"] == SOURCE_TOKEN
    assert scores["reference-token-only"]["rows"] == block.rows
    assert scores["correct"]["source"] == "hidden"


def test_shuffled_and_null_rows_are_mathematically_distinct():
    from occ3_sglang.calibration_corpus import control_rows
    block = _turns(count=1)[0]
    _, correct = control_rows(block, "correct")
    _, shuffled = control_rows(block, "shuffled")
    _, null = control_rows(block, "null")
    _, other = control_rows(block, "wrong-sender")
    assert torch.equal(shuffled, torch.flip(block.hidden, [0]))
    assert torch.equal(torch.flip(shuffled, [0]), block.hidden)
    assert not torch.equal(shuffled, correct)
    assert float(null.abs().sum()) == 0.0
    assert torch.equal(other, block.wrong_sender_hidden)
    assert not torch.equal(other, correct)


def test_separation_math():
    from occ3_sglang.calibration_corpus import separation
    result = separation([1.0, 1.0, -1.0], [0.0, -1.0, -2.0])
    assert result["trials"] == 3
    assert result["delta_accuracy"] == pytest.approx(2.0 / 3.0)
    assert result["delta_margin"] == pytest.approx(1.0)


def test_the_digest_free_fallback_allows_unequal_validation_rows():
    """A hand-built artifact without a content digest still validates a record
    measured on a differently sized validation partition."""
    import dataclasses as _dc

    from occ3_sglang.calibration_corpus import evaluate_map, validate_for_activation
    from occ3_sglang.statebridge_calibrated import apply_map
    manifest, artifact = _fitted()
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert record["rows"] != artifact.rows
    bare = _dc.replace(artifact, content_digest=None)
    policy = {"min_r2_standard": 0.1, "min_turns": 1}
    assert validate_for_activation(bare, record, policy) == ("allow", None)
    assert validate_for_activation(bare, dict(record, method_id="other"), policy) == (
        "refuse", "artifact_digest")
