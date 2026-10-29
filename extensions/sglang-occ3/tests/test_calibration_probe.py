"""Tests for the scalar relation probe used in the live window."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _fixture(rows=128, dim=512, rank=16, kappa=0.75, seed=0):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    right = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    factor = torch.randn(rows, rank, generator=generator)
    hidden = kappa ** 0.5 * (factor @ left) + (1.0 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
    embeddings = kappa ** 0.5 * (factor @ right) + (1.0 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
    return hidden, embeddings


def test_probe_returns_the_fixed_key_set_and_no_content():
    from occ3_sglang.calibration_probe import PROBE_KEYS, probe_relation
    hidden, embeddings = _fixture()
    result = probe_relation(hidden, embeddings)
    assert set(result) == set(PROBE_KEYS)
    assert isinstance(result["canonical_decile"], list)
    assert all(type(v) in (int, float) for k, v in result.items()
               if k not in ("canonical_decile",) and v is not None)


def test_canonical_correlations_are_high_when_the_relation_exists():
    from occ3_sglang.calibration_probe import probe_relation
    hidden, embeddings = _fixture(kappa=0.9, seed=3)
    result = probe_relation(hidden, embeddings)
    assert result["canonical_top8"] > 0.5, result["canonical_top8"]


def test_canonical_correlations_do_not_separate_related_from_independent():
    """Whitening makes two full-rank clouds congruent, so the spectrum saturates.

    This is the reason the probe also reports held-out within-turn R^2 through
    the frozen dual-ridge fit/apply contract.
    """
    from occ3_sglang.calibration_probe import probe_relation
    related, related_e = _fixture(kappa=0.9, seed=11)
    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(128, 512, generator=generator)
    embeddings = torch.randn(128, 512, generator=generator)
    independent = probe_relation(hidden, embeddings)
    paired = probe_relation(related, related_e)
    assert independent["canonical_top8"] > 0.90
    assert abs(independent["canonical_top8"] - paired["canonical_top8"]) < 0.1


def test_heldout_within_turn_r2_separates_related_from_independent():
    from occ3_sglang.calibration_probe import probe_relation
    related, related_e = _fixture(kappa=0.9, seed=11)
    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(128, 512, generator=generator)
    embeddings = torch.randn(128, 512, generator=generator)
    paired = probe_relation(related, related_e)
    independent = probe_relation(hidden, embeddings)
    assert paired["heldout_r2_within_turn"] > 0.05, paired["heldout_r2_within_turn"]
    assert independent["heldout_r2_within_turn"] < paired["heldout_r2_within_turn"]


def test_a_turn_too_short_for_a_contract_fit_records_undefined_not_zero():
    from occ3_sglang.calibration_probe import probe_relation, validate_probe
    hidden, embeddings = _fixture(rows=32, dim=512, seed=9)
    assert hidden.shape[0] < 66
    result = probe_relation(hidden, embeddings)
    assert result["heldout_r2_within_turn"] is None
    assert validate_probe(result)
    # A turn that can hold a contract-sized fit records a real measurement.
    measurable = probe_relation(*_fixture(rows=128, dim=512, seed=10))
    assert measurable["heldout_r2_within_turn"] is not None


def test_probe_detects_the_shipped_prefix_as_a_copy_of_the_embeddings():
    from occ3_sglang.align_diagnostics import alignment_diagnostics
    from occ3_sglang.calibration_probe import probe_relation
    from occ3_sglang.statebridge import align_hidden_sequence
    hidden, embeddings = _fixture(rows=128, dim=512, seed=7)
    target = float(embeddings.norm(dim=-1).mean())
    prefix, _ = align_hidden_sequence(hidden, embeddings, target)
    result = probe_relation(hidden, embeddings, prefix)
    assert result["cos_prefix_to_embedding"] > 0.999
    assert result["hidden_roll_sensitivity"] < 0.5


def test_probe_refuses_a_short_message():
    from occ3_sglang.calibration_probe import probe_relation
    from occ3_sglang.latent_types import LatentError
    hidden, embeddings = _fixture(rows=1)
    with pytest.raises(LatentError):
        probe_relation(hidden, embeddings)


def test_the_permuted_sensitivity_uses_an_explicit_generator():
    from occ3_sglang.calibration_probe import probe_relation
    hidden, embeddings = _fixture(rows=128, dim=512, seed=12)
    default = probe_relation(hidden, embeddings)
    seeded_zero = probe_relation(hidden, embeddings,
                                 generator=torch.Generator().manual_seed(0))
    assert default["hidden_permute_sensitivity"] == pytest.approx(
        seeded_zero["hidden_permute_sensitivity"])
    explicit = probe_relation(hidden, embeddings,
                              generator=torch.Generator().manual_seed(13))
    seeded = probe_relation(hidden, embeddings,
                            generator=torch.Generator().manual_seed(13))
    assert explicit["hidden_permute_sensitivity"] == seeded["hidden_permute_sensitivity"]


# Pooled probe: the turn-disjoint number the next live window must report.

def _turns(count=24, rows=16, dim=64, rank=8, kappa=0.9, seed=0):
    from occ3_sglang.calibration_corpus import TurnBlock
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    right = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    blocks = []
    for index in range(count):
        factor = torch.randn(rows, rank, generator=generator)
        hidden = kappa ** 0.5 * (factor @ left) + (1.0 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
        embeddings = kappa ** 0.5 * (factor @ right) + (1.0 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
        blocks.append(TurnBlock(turn_id=f"turn-{index:02d}", hidden=hidden, embeddings=embeddings))
    return tuple(blocks)


def _fit_apply():
    from occ3_sglang import statebridge_calibrated
    return statebridge_calibrated.fit_map, statebridge_calibrated.apply_map


def test_per_turn_probe_schema_is_not_extended_in_place():
    from occ3_sglang.calibration_corpus import RECORD_KEYS
    from occ3_sglang.calibration_probe import POOLED_PROBE_KEYS, PROBE_KEYS
    assert "heldout_r2_within_turn" in PROBE_KEYS and len(PROBE_KEYS) >= 20
    assert POOLED_PROBE_KEYS == RECORD_KEYS
    assert set(POOLED_PROBE_KEYS) != set(PROBE_KEYS)
    assert "canonical_decile" in set(PROBE_KEYS) - set(POOLED_PROBE_KEYS)
    assert "mean_row_cosine" in set(POOLED_PROBE_KEYS) - set(PROBE_KEYS)


def test_pooled_probe_reports_the_turn_disjoint_schema():
    from occ3_sglang.calibration_corpus import validate_metrics_record
    from occ3_sglang.calibration_probe import POOLED_PROBE_KEYS, probe_pooled_turn_disjoint
    fit, apply = _fit_apply()
    report = probe_pooled_turn_disjoint(_turns(), fit, apply, row_budget=512,
                                        fit_turns=12, validation_turns=4, test_turns=8)
    assert set(report) == set(POOLED_PROBE_KEYS)
    assert validate_metrics_record(report)
    assert report["turns"] >= 1 and report["rows"] > 0
    assert report["method_id"].endswith("calibrated-dual-ridge.v1")
    # The turn-disjoint number is recorded, never gated: it can be negative.
    assert report["r2_standard"] is None or report["r2_standard"] > -1.0
    assert report["mean_row_cosine"] > 0.5, report["mean_row_cosine"]
    assert report["raw_row_norm_mean"] > 0


def test_pooled_probe_scores_raw_and_served_views_when_a_nu_is_given():
    from occ3_sglang.calibration_probe import probe_pooled_turn_disjoint
    fit, apply = _fit_apply()
    report = probe_pooled_turn_disjoint(_turns(), fit, apply, row_budget=512,
                                        fit_turns=12, validation_turns=4, test_turns=8,
                                        serving={"kind": "n1"}, nu=1.25)
    assert report["serving"] == {"kind": "n1"} and report["nu"] == 1.25
    assert report["served_row_norm_mean"] == pytest.approx(1.25, rel=1e-3)
    assert report["raw_row_norm_mean"] != pytest.approx(1.25, rel=1e-3)


def test_pooled_probe_refuses_turns_outside_the_capture_domain():
    from occ3_sglang.calibration_corpus import TurnBlock
    from occ3_sglang.calibration_probe import probe_pooled_turn_disjoint
    from occ3_sglang.latent_types import LatentError
    fit, apply = _fit_apply()
    wide = torch.randn(200, 64)
    blocks = list(_turns())
    blocks[0] = TurnBlock(turn_id=blocks[0].turn_id, hidden=wide, embeddings=wide.clone())
    with pytest.raises(LatentError) as caught:
        probe_pooled_turn_disjoint(tuple(blocks), fit, apply, row_budget=512,
                                   fit_turns=12, validation_turns=4, test_turns=8)
    assert caught.value.code == "profile"
    narrow = torch.randn(16, 16)
    blocks = list(_turns())
    blocks[0] = TurnBlock(turn_id=blocks[0].turn_id, hidden=narrow, embeddings=narrow.clone())
    with pytest.raises(LatentError) as caught:
        probe_pooled_turn_disjoint(tuple(blocks), fit, apply, row_budget=512,
                                   fit_turns=12, validation_turns=4, test_turns=8)
    assert caught.value.code == "profile"
