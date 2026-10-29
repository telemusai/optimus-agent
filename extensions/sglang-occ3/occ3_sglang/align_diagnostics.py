"""Bounded scalar diagnostics for one StateBridge alignment.

The receiver reads a continuous prefix that is produced inside the worker on
device. To decide whether that prefix carries sender information we need
measurements from real GLM-5.3 states, but raw tensors must never leave the
authorised GPU state lifecycle.

This module therefore returns a fixed, small dictionary of int and float
scalars. It allocates only n x n and rank x rank workspaces; the largest
temporary is the 2n x d stacked matrix that align_hidden_sequence already
builds. No vocabulary gather, no d x d covariance, no tensor leaves the call.

Everything here is pure: torch is passed in, nothing is imported at module
load, and the call never mutates its inputs.
"""
from __future__ import annotations

import math

from .latent_types import LatentError, require

# Fixed key set. The control plane and the receipts may rely on the names and
# on the types; adding a key is a schema change and must be versioned with the
# method identifier.
DIAGNOSTIC_KEYS = (
    "rows", "hidden_dim", "union_rank",
    "hidden_norm_mean", "embedding_norm_mean", "prefix_norm_mean",
    "hidden_eff_rank", "embedding_eff_rank",
    "whitened_hidden_sv_max", "whitened_hidden_sv_min",
    "whitened_embedding_sv_max", "whitened_embedding_sv_min",
    "procrustes_residual", "cos_prefix_to_embedding", "relative_prefix_error",
    "hidden_sensitivity", "shuffled_hidden_sensitivity",
    "nonfinite_count", "anchor_alpha",
)
_FLOATS = ("hidden_norm_mean", "embedding_norm_mean", "prefix_norm_mean",
           "hidden_eff_rank", "embedding_eff_rank",
           "whitened_hidden_sv_max", "whitened_hidden_sv_min",
           "whitened_embedding_sv_max", "whitened_embedding_sv_min",
           "procrustes_residual", "cos_prefix_to_embedding", "relative_prefix_error",
           "hidden_sensitivity", "shuffled_hidden_sensitivity", "anchor_alpha")


def effective_rank(matrix):
    """Participation ratio of the squared singular values, in [1, n]."""
    torch = _torch_of(matrix)
    centred = matrix - matrix.mean(0, keepdim=True)
    gram = centred @ centred.mT
    values = torch.linalg.eigvalsh((gram + gram.mT) * 0.5).clamp_min(0.0)
    total = values.sum()
    require(bool(torch.isfinite(total)) and float(total) > 0, "numerical")
    return float((total * total) / (values * values).sum().clamp_min(torch.finfo(values.dtype).tiny))


def _torch_of(reference):
    import torch
    return torch


def _solve(hidden, embeddings, target_norm, *, ridge, eigen_floor, rank_rtol, torch):
    """Mirror of statebridge.align_hidden_sequence that keeps its intermediates."""
    h, e = hidden.float(), embeddings.float()
    mh, me = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
    hc, ec = h - mh, e - me
    _, singular, vh = torch.linalg.svd(torch.cat((hc, ec), dim=0), full_matrices=False)
    threshold = max(float(singular[0]) * rank_rtol, eigen_floor)
    rank = min(int((singular > threshold).sum()), 2 * (h.shape[0] - 1))
    require(rank > 0, "numerical")
    q = vh[:rank].mT
    bh, be = hc @ q, ec @ q
    eye = torch.eye(rank, dtype=h.dtype, device=h.device)
    ch = bh.mT @ bh / h.shape[0] + ridge * eye
    ce = be.mT @ be / h.shape[0] + ridge * eye
    half = _sym(ce, 0.5, eigen_floor, torch)
    whitened_h = bh @ _sym(ch, -0.5, eigen_floor, torch)
    whitened_e = be @ _sym(ce, -0.5, eigen_floor, torch)
    u, _, v = torch.linalg.svd(whitened_h.mT @ whitened_e, full_matrices=False)
    rotation = u @ v
    z = (whitened_h @ rotation @ half) @ q.mT + me
    norms = z.norm(dim=-1, keepdim=True)
    target = torch.as_tensor(target_norm, device=z.device, dtype=z.dtype)
    require(bool(torch.isfinite(target)) and float(target) > 0, "numerical")
    require(bool((norms > eigen_floor).all()), "numerical")
    prefix = z * (target / norms)
    residual = float(((whitened_h @ rotation - whitened_e).norm() /
                      whitened_e.norm().clamp_min(torch.finfo(z.dtype).tiny)))
    return {"prefix": prefix, "rank": rank, "residual": residual,
            "whitened_h": whitened_h, "whitened_e": whitened_e,
            "embedding": e, "prefix_h": prefix}


def _sym(matrix, power, floor, torch):
    values, vectors = torch.linalg.eigh((matrix + matrix.mT) * 0.5)
    return (vectors * values.clamp_min(floor).pow(power)) @ vectors.mT


def alignment_diagnostics(hidden, embeddings, prefix=None, *, target_norm,
                          ridge=1e-3, eigen_floor=1e-6, rank_rtol=1e-5,
                          anchor_alpha=0.0, sensitivity=True, generator=None):
    """Return DIAGNOSTIC_KEYS scalars for one alignment. Never raises on content.

    sensitivity=True costs two extra thin solves (row-rolled hidden states and
    row-shuffled hidden states). Both are deterministic permutations; the
    shuffle draws from an explicit torch.Generator (a fresh local one when
    none is passed), never from the serving model RNG.
    """
    torch = _torch_of(hidden)
    require(hidden.ndim == 2 and hidden.shape == embeddings.shape, "profile")
    require(hidden.device == embeddings.device, "profile")
    with torch.no_grad():
        require(bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(embeddings).all()), "numerical")
        main = _solve(hidden, embeddings, target_norm, ridge=ridge, eigen_floor=eigen_floor,
                      rank_rtol=rank_rtol, torch=torch)
        z = main["prefix"] if prefix is None else prefix.float()
        e = main["embedding"]
        target = torch.as_tensor(target_norm, device=e.device, dtype=e.dtype)
        calibrated = e * (target / e.norm(dim=-1, keepdim=True).clamp_min(eigen_floor))
        cos = (z * calibrated).sum(-1) / (z.norm(dim=-1).clamp_min(eigen_floor) *
                                          calibrated.norm(dim=-1).clamp_min(eigen_floor))
        hidden_sensitivity = shuffled_sensitivity = 0.0
        if sensitivity:
            rolled = torch.roll(hidden, 1, 0)
            other = _solve(rolled, embeddings, target_norm, ridge=ridge, eigen_floor=eigen_floor,
                           rank_rtol=rank_rtol, torch=torch)["prefix"]
            hidden_sensitivity = float((other - z).norm() / z.norm().clamp_min(eigen_floor))
            if generator is None:
                generator = torch.Generator(device=hidden.device).manual_seed(0)
            permuted = hidden[torch.randperm(hidden.shape[0], generator=generator,
                                            device=hidden.device)]
            other = _solve(permuted, embeddings, target_norm, ridge=ridge, eigen_floor=eigen_floor,
                           rank_rtol=rank_rtol, torch=torch)["prefix"]
            shuffled_sensitivity = float((other - z).norm() / z.norm().clamp_min(eigen_floor))
        wh = main["whitened_h"]
        we = main["whitened_e"]
        result = {
            "rows": int(hidden.shape[0]), "hidden_dim": int(hidden.shape[1]),
            "union_rank": int(main["rank"]),
            "hidden_norm_mean": float(hidden.float().norm(dim=-1).mean()),
            "embedding_norm_mean": float(e.norm(dim=-1).mean()),
            "prefix_norm_mean": float(z.norm(dim=-1).mean()),
            "hidden_eff_rank": effective_rank(hidden),
            "embedding_eff_rank": effective_rank(embeddings),
            "whitened_hidden_sv_max": float(wh.max()),
            "whitened_hidden_sv_min": float(wh.abs().min()),
            "whitened_embedding_sv_max": float(we.max()),
            "whitened_embedding_sv_min": float(we.abs().min()),
            "procrustes_residual": float(main["residual"]),
            "cos_prefix_to_embedding": float(cos.mean()),
            "relative_prefix_error": float((z - calibrated).norm() / calibrated.norm().clamp_min(eigen_floor)),
            "hidden_sensitivity": hidden_sensitivity,
            "shuffled_hidden_sensitivity": shuffled_sensitivity,
            "nonfinite_count": int((~torch.isfinite(z)).sum()),
            "anchor_alpha": float(anchor_alpha),
        }
    validate(result)
    return result


def validate(result):
    """Fixed key set, exact int/float types, finite values, bounded magnitudes."""
    require(isinstance(result, dict) and set(result) == set(DIAGNOSTIC_KEYS), "profile")
    for key in DIAGNOSTIC_KEYS:
        value = result[key]
        require(type(value) in (int, float) and not isinstance(value, bool), "profile")
        if isinstance(value, float):
            require(value == value and abs(value) != float("inf"), "numerical")
    require(result["rows"] >= 2 and result["hidden_dim"] > 2 * result["rows"], "profile")
    require(0 < result["union_rank"] <= 2 * (result["rows"] - 1), "profile")
    require(result["nonfinite_count"] == 0, "numerical")
    return True


def workspace_elements(rows, hidden_dim):
    """Upper bound on any single temporary this module allocates, in elements."""
    require(2 <= rows and hidden_dim > 2 * rows, "profile")
    rank = 2 * (rows - 1)
    return max(2 * rows * hidden_dim, rows * rank, rank * rank, rows * rows)


CALIBRATED_DIAGNOSTIC_KEYS = (
    "rows", "hidden_dim", "serving_n1",
    "hidden_norm_mean", "embedding_norm_mean",
    "raw_prefix_norm_mean", "served_prefix_norm_mean",
    "raw_cos_prefix_to_embedding", "served_cos_prefix_to_embedding",
    "raw_relative_prefix_error", "served_relative_prefix_error",
    "raw_predicted_r2", "served_predicted_r2",
    "hidden_sensitivity", "shuffled_hidden_sensitivity",
    "raw_copier_flag", "served_copier_flag",
    "nonfinite_count", "anchor_alpha",
)
CALIBRATED_FLOATS = ("hidden_norm_mean", "embedding_norm_mean",
                     "raw_prefix_norm_mean", "served_prefix_norm_mean",
                     "raw_cos_prefix_to_embedding", "served_cos_prefix_to_embedding",
                     "raw_relative_prefix_error", "served_relative_prefix_error",
                     "raw_predicted_r2", "served_predicted_r2",
                     "hidden_sensitivity", "shuffled_hidden_sensitivity", "anchor_alpha")
# LIVE-WINDOW-1 Section 5: any candidate whose cosine to the embedding copier
# stays above this is not a candidate, whatever its other numbers say.
COPIER_COSINE_LIMIT = 0.9
_DIAGNOSTIC_FLOOR = 1e-6


def _method_surface(artifact):
    """The artifact method's apply/serving callables, through the registry."""
    import importlib

    from .native_profile import method_descriptor
    try:
        descriptor = method_descriptor(artifact.method_id)
        module = importlib.import_module(descriptor["artifact_module"])
    except Exception:
        raise LatentError("profile") from None
    return module.apply_map, module.serving_transform


def _cosine_mean(left, right, torch):
    norms = left.norm(dim=-1).clamp_min(_DIAGNOSTIC_FLOOR) * \
        right.norm(dim=-1).clamp_min(_DIAGNOSTIC_FLOOR)
    return float(((left * right).sum(-1) / norms).mean())


def frozen_map_diagnostics(artifact, hidden, embeddings, *, target_norm=None,
                           anchor_alpha=0.0):
    """CALIBRATED_DIAGNOSTIC_KEYS scalars for one frozen map, any registered method.

    The prefix is the method's apply_map (raw rows, no renormalisation inside
    the apply) followed by its serving transform under the artifact binding's
    serving config. The raw and served metric families are computed and kept
    separate: raw rows against the captured embeddings, served rows against
    the embeddings renormalised with the same statistic. The sensitivity terms
    and the copier flag use the SAME artifact and serving normalisation in
    both terms (evaluation protocol v2 R1): a shuffled output is never
    compared against a differently-normalised correct output.
    """
    apply, serve = _method_surface(artifact)
    torch = _torch_of(hidden)
    require(hidden.ndim == 2 and hidden.shape == embeddings.shape, "profile")
    require(hidden.shape[0] >= 2 and hidden.device == embeddings.device, "profile")
    require(hidden.shape[1] == artifact.hidden_dim, "profile")
    binding = artifact.binding
    require(binding is not None, "profile")
    serving = getattr(binding, "serving", None)
    if serving is not None:
        require(isinstance(target_norm, (int, float)) and not isinstance(target_norm, bool)
                and math.isfinite(target_norm) and target_norm > 0.0, "profile")
    with torch.no_grad():
        h, e = hidden.float(), embeddings.float()
        require(bool(torch.isfinite(h).all()) and bool(torch.isfinite(e).all()), "numerical")
        z_raw = apply(artifact, h, torch_module=torch)
        z_served = serve(serving, z_raw, nu=target_norm)
        require(z_raw.shape == e.shape and z_served.shape == e.shape, "profile")
        # The served reference carries the same normalisation as the served
        # rows; the raw family compares raw rows against the captured scale.
        if serving is None:
            served_target = e
        else:
            norms = e.norm(dim=-1, keepdim=True).clamp_min(_DIAGNOSTIC_FLOOR)
            served_target = e * (float(target_norm) / norms)
        raw_cos = _cosine_mean(z_raw, e, torch)
        served_cos = _cosine_mean(z_served, served_target, torch)

        def view(predicted, target):
            residual = float((predicted - target).pow(2).sum())
            baseline = float((target - target.mean(0, keepdim=True)).pow(2).sum())
            require(math.isfinite(residual) and math.isfinite(baseline) and baseline > 0.0,
                    "numerical")
            return {"relative": float((predicted - target).norm()
                                      / target.norm().clamp_min(_DIAGNOSTIC_FLOOR)),
                    "r2": 1.0 - residual / baseline}

        raw = view(z_raw, e)
        served = view(z_served, served_target)
        # R1: both terms of every sensitivity use the same artifact and the
        # same serving transform; the denominator is the correct served output.
        rolled = serve(serving, apply(artifact, torch.roll(h, 1, 0), torch_module=torch),
                       nu=target_norm)
        reversed_rows = serve(serving, apply(artifact, torch.flip(h, [0]).contiguous(),
                                             torch_module=torch),
                              nu=target_norm)
        floor = max(float(z_served.norm()), _DIAGNOSTIC_FLOOR)
        result = {
            "rows": int(h.shape[0]), "hidden_dim": int(h.shape[1]),
            "serving_n1": int(serving is not None),
            "hidden_norm_mean": float(h.norm(dim=-1).mean()),
            "embedding_norm_mean": float(e.norm(dim=-1).mean()),
            "raw_prefix_norm_mean": float(z_raw.norm(dim=-1).mean()),
            "served_prefix_norm_mean": float(z_served.norm(dim=-1).mean()),
            "raw_cos_prefix_to_embedding": raw_cos,
            "served_cos_prefix_to_embedding": served_cos,
            "raw_relative_prefix_error": raw["relative"],
            "served_relative_prefix_error": served["relative"],
            "raw_predicted_r2": raw["r2"],
            "served_predicted_r2": served["r2"],
            "hidden_sensitivity": float((rolled - z_served).norm() / floor),
            "shuffled_hidden_sensitivity": float((reversed_rows - z_served).norm() / floor),
            "raw_copier_flag": int(raw_cos > COPIER_COSINE_LIMIT),
            "served_copier_flag": int(served_cos > COPIER_COSINE_LIMIT),
            "nonfinite_count": int((~torch.isfinite(z_served)).sum()),
            "anchor_alpha": float(anchor_alpha),
        }
    validate_calibrated(result)
    return result


def validate_calibrated(result):
    """Fixed key set, exact int/float types, finite values, bounded magnitudes."""
    require(isinstance(result, dict) and set(result) == set(CALIBRATED_DIAGNOSTIC_KEYS), "profile")
    for key in CALIBRATED_DIAGNOSTIC_KEYS:
        value = result[key]
        require(type(value) in (int, float) and not isinstance(value, bool), "profile")
        if isinstance(value, float):
            require(value == value and abs(value) != float("inf"), "numerical")
    require(result["rows"] >= 2 and result["hidden_dim"] > 2, "profile")
    require(result["serving_n1"] in (0, 1), "profile")
    require(result["raw_copier_flag"] in (0, 1) and result["served_copier_flag"] in (0, 1),
            "profile")
    require(result["nonfinite_count"] == 0, "numerical")
    for key in ("raw_cos_prefix_to_embedding", "served_cos_prefix_to_embedding"):
        require(-1.0 <= result[key] <= 1.0, "numerical")
    return True
