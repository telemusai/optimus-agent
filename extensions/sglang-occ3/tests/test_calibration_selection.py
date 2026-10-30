"""Non-torch coverage of the P5 selection and record machinery.

The selection driver and the held-out scoring helper need torch for their
fits and tensors, so their end-to-end tests live in the torch-dependent
files. Everything decided in plain Python is pinned here and runs without a
torch install: the (r, eta) selection grid, the fit/val/test gap math, and
the extended evaluation-record schema (norm distribution, gap and
identifier-recovery fields).
"""
from __future__ import annotations

import math

import pytest


def _view(rows=2, **overrides):
    from occ3_sglang.calibration_corpus import VIEW_KEYS
    values = {"r2_standard": 0.5, "r2_vs_fit_mean": 0.4, "sse": 10.0, "sst_test": 20.0,
              "sse_vs_fit_mean": 25.0, "relative_frobenius": 0.5, "mean_row_cosine": 0.9,
              "raw_row_norm_mean": 1.0, "raw_row_norm_std": 0.1, "raw_row_norm_p05": 0.9,
              "raw_row_norm_p95": 1.1, "served_r2_standard": 0.45,
              "served_r2_vs_fit_mean": 0.35, "served_relative_frobenius": 0.55,
              "served_mean_row_cosine": 0.88, "served_row_norm_mean": 2.0,
              "served_row_norm_std": 0.2, "served_row_norm_p05": 1.8,
              "served_row_norm_p95": 2.2, "rows": rows}
    values.update(overrides)
    assert set(values) == set(VIEW_KEYS)
    return values


def _record(**overrides):
    from occ3_sglang.calibration_corpus import (GAP_FIELDS, IDENTIFIER_FIELDS,
                                                 RECORD_KEYS, RECORD_SCHEMA, VIEW_KEYS)
    view = _view()
    record = {"schema": RECORD_SCHEMA, "method_id": "method-1",
              "artifact_digest": "a" * 64, "corpus_digest": "b" * 64,
              "fit_manifest_digest": "c" * 64, "partition_manifest_digest": "d" * 64,
              "serving": None, "nu": None, "rows": 2, "turns": 1,
              "output_variance": 1.0, "output_effective_rank": 4.0,
              "served_output_variance": 1.0, "served_output_effective_rank": 4.0,
              "per_turn": {"turn-1": dict(view)},
              "turn_macro": {name: view[name] for name in VIEW_KEYS if name != "rows"},
              "pooled": dict(view), "undefined_denominators": []}
    record.update({name: view[name] for name in VIEW_KEYS[:-1]})
    record.update({name: None for name in GAP_FIELDS})
    record.update({name: None for name in IDENTIFIER_FIELDS})
    record.update(overrides)
    assert set(record) == set(RECORD_KEYS)
    return record


# -- the (r, eta) selection grid ----------------------------------------------

def test_the_default_selection_grid_is_the_frozen_method_grid():
    from occ3_sglang import statebridge_calibrated_procrustes as procrustes
    from occ3_sglang.calibration_corpus import (PROCRUSTES_ETA_GRID,
                                                 PROCRUSTES_RANK_GRID,
                                                 _procrustes_grid)
    # The default selection rank axis is exactly {32, 64, 128, 256}. r=16
    # stays a legal fit hyperparameter (the module grid keeps it, and the
    # small fixtures pin it), so only the DEFAULT SELECTION grid drops it.
    assert PROCRUSTES_RANK_GRID == (32, 64, 128, 256)
    assert 16 in procrustes.RANK_GRID and 16 not in PROCRUSTES_RANK_GRID
    assert set(PROCRUSTES_RANK_GRID) <= set(procrustes.RANK_GRID)
    assert PROCRUSTES_ETA_GRID == procrustes.ETA_GRID == (0.001, 0.01, 0.1)
    grid = _procrustes_grid(None, None, "o")
    assert len(grid) == len(PROCRUSTES_RANK_GRID) * len(PROCRUSTES_ETA_GRID) == 12
    # Rank-major then eta-major, both ascending: ties in the selection keep
    # the earlier grid entry.
    assert grid[0] == {"r": 32, "eta": 0.001, "variant": "o"}
    assert grid[1] == {"r": 32, "eta": 0.01, "variant": "o"}
    assert grid[-1] == {"r": 256, "eta": 0.1, "variant": "o"}
    assert [entry["r"] for entry in grid] == sorted(entry["r"] for entry in grid)


def test_the_selection_grid_accepts_narrowed_axes_and_refuses_broken_ones():
    from occ3_sglang import statebridge_calibrated_procrustes as procrustes
    from occ3_sglang.calibration_corpus import _procrustes_grid
    from occ3_sglang.latent_types import LatentError
    narrowed = _procrustes_grid((16, 32), (0.01,), "plain")
    assert narrowed == ({"r": 16, "eta": 0.01, "variant": "plain"},
                        {"r": 32, "eta": 0.01, "variant": "plain"})
    # r=16 may be selected explicitly; the fit module remains the authority on
    # whether a rank is a legal hyperparameter at all.
    assert all(entry["r"] in procrustes.RANK_GRID for entry in narrowed)
    # None selects the frozen defaults (proved above); an empty or a
    # duplicated axis is a broken grid and refuses.
    for broken in ((), (32, 32)):
        with pytest.raises(LatentError):
            _procrustes_grid(broken, None, "o")
        with pytest.raises(LatentError):
            _procrustes_grid(None, broken, "o")


# -- fit/val/test gaps ---------------------------------------------------------

def test_record_gaps_are_signed_left_minus_right_per_available_pair():
    from occ3_sglang.calibration_corpus import GAP_FIELDS, GAP_PAIRS, _record_gaps
    records = {"fit": _record(r2_standard=0.90, relative_frobenius=0.20),
               "validation": _record(r2_standard=0.70, relative_frobenius=0.30),
               "test": _record(r2_standard=0.60, relative_frobenius=0.40)}
    gaps = _record_gaps(records)
    assert set(gaps) == set(GAP_FIELDS)
    assert gaps["r2_standard_gap_fit_validation"] == pytest.approx(0.20)
    assert gaps["r2_standard_gap_fit_test"] == pytest.approx(0.30)
    assert gaps["r2_standard_gap_validation_test"] == pytest.approx(0.10)
    # A positive r2 gap is the overfit direction; the same direction on the
    # error metric is a NEGATIVE relative_frobenius gap (fit error lower).
    assert gaps["relative_frobenius_gap_fit_test"] == pytest.approx(-0.20)
    # A pair is measured only when both partitions were provided.
    partial = _record_gaps({"fit": records["fit"], "test": records["test"]})
    assert set(partial) == {name for name in GAP_FIELDS if name.endswith("_fit_test")}
    # An undefined metric propagates as None, never as a fabricated zero; the
    # defined metrics of the same pair are still measured.
    undefined = _record_gaps({"fit": _record(r2_standard=None,
                                             relative_frobenius=0.20),
                               "test": records["test"]})
    assert undefined["r2_standard_gap_fit_test"] is None
    assert undefined["relative_frobenius_gap_fit_test"] == pytest.approx(-0.20)
    assert set(GAP_PAIRS) == {("fit", "validation"), ("fit", "test"),
                              ("validation", "test")}


# -- the extended evaluation-record schema ------------------------------------

def test_the_record_schema_carries_the_p5_metric_fields():
    from occ3_sglang.calibration_corpus import (GAP_FIELDS, IDENTIFIER_FIELDS,
                                                RECORD_KEYS, VIEW_KEYS)
    for name in ("raw_row_norm_std", "raw_row_norm_p05", "raw_row_norm_p95",
                 "served_row_norm_std", "served_row_norm_p05", "served_row_norm_p95"):
        assert name in VIEW_KEYS
        assert name in RECORD_KEYS
    assert set(GAP_FIELDS) <= set(RECORD_KEYS)
    assert set(IDENTIFIER_FIELDS) <= set(RECORD_KEYS)
    assert set(IDENTIFIER_FIELDS) == {
        "identifier_margin_mean", "identifier_margin_p05", "identifier_recovery_rate",
        "served_identifier_margin_mean", "served_identifier_margin_p05",
        "served_identifier_recovery_rate"}
    # The gap and margin fields are top-level record fields, not view fields:
    # a gap is a cross-partition fact and a margin is a pooled held-out one.
    assert set(GAP_FIELDS).isdisjoint(VIEW_KEYS)
    assert set(IDENTIFIER_FIELDS).isdisjoint(VIEW_KEYS)


def test_validate_metrics_record_accepts_measured_and_unmeasured_p5_fields():
    from occ3_sglang.calibration_corpus import validate_metrics_record
    assert validate_metrics_record(_record()) is True
    measured = _record(identifier_margin_mean=0.3, identifier_margin_p05=0.1,
                       identifier_recovery_rate=0.75,
                       served_identifier_margin_mean=0.2,
                       served_identifier_margin_p05=0.0,
                       served_identifier_recovery_rate=0.5,
                       r2_standard_gap_fit_test=0.2,
                       relative_frobenius_gap_fit_validation=-0.1)
    assert validate_metrics_record(measured) is True
    # Not-measured stays None; a fabricated zero or an int is a tampered record.
    assert validate_metrics_record(_record(identifier_margin_mean=None)) is True
    for name, value in (("identifier_margin_mean", 0), ("identifier_recovery_rate", 1),
                        ("r2_standard_gap_fit_test", "0.2"),
                        ("served_identifier_margin_p05", float("nan")),
                        ("served_identifier_recovery_rate", float("inf"))):
        with pytest.raises(Exception):
            validate_metrics_record(_record(**{name: value}))
    # Removing a new key is tampering, exactly like removing an old one.
    for name in ("identifier_margin_mean", "r2_standard_gap_fit_test",
                 "raw_row_norm_p95"):
        broken = _record()
        del broken[name]
        with pytest.raises(Exception):
            validate_metrics_record(broken)


def test_gap_and_margin_fields_are_undefined_when_not_measured():
    """The synthetic record mirrors what evaluate_map emits: the P5 fields the
    single-partition evaluation cannot know are present but None."""
    from occ3_sglang.calibration_corpus import GAP_FIELDS, IDENTIFIER_FIELDS
    record = _record()
    for name in GAP_FIELDS + IDENTIFIER_FIELDS:
        assert record[name] is None


def test_percentile_is_linear_interpolation_on_sorted_values():
    """The p05/p95 convention: torch.quantile's default linear interpolation."""
    from occ3_sglang.calibration_corpus import _percentile

    class _Row:
        def __init__(self, values):
            self._values = list(values)

        def numel(self):
            return len(self._values)

        def __getitem__(self, index):
            return self._values[index]

    ordered = _Row([1.0, 2.0, 3.0, 4.0])
    assert _percentile(ordered, 0.0) == 1.0
    assert _percentile(ordered, 1.0) == 4.0
    assert _percentile(ordered, 0.25) == pytest.approx(1.75)
    assert _percentile(ordered, 0.05) == pytest.approx(1.15)
    assert _percentile(ordered, 0.95) == pytest.approx(3.85)
    single = _Row([7.5])
    assert _percentile(single, 0.05) == _percentile(single, 0.95) == 7.5
    assert math.isfinite(_percentile(ordered, 0.5))
