"""Message-specific StateBridge alignment. Torch is imported only when invoked.

No dense hidden_dim squared covariance or vocabulary gather is constructed.
These functions support real CPU Torch references; worker entry points require CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass

from .latent_types import ALIGNMENT, LatentError, require


@dataclass(frozen=True)
class AlignmentInfo:
    method: str
    union_rank: int
    retained_count: int
    hidden_dim: int


def _torch():
    import torch
    return torch


def _symmetric_power(matrix, power, floor):
    torch = _torch()
    vals, vectors = torch.linalg.eigh((matrix + matrix.mT) * 0.5)
    return (vectors * vals.clamp_min(floor).pow(power)) @ vectors.mT


def align_hidden_sequence(hidden, embeddings, target_norm, *, ridge=1e-3,
                          eigen_floor=1e-6, rank_rtol=1e-5):
    """Thin union-subspace Procrustes with identity completion outside union.

    The full proper rotation uses an unused complement sign when necessary.
    That sign cannot change the observed centered rows. Degenerate zero-info
    messages refuse instead of yielding a fabricated carrier.
    """
    torch = _torch()
    require(hidden.ndim == 2 and hidden.shape == embeddings.shape, "profile")
    n, d = hidden.shape
    require(2 <= n <= 128 and d > 2 * n and ridge > 0 and eigen_floor > 0, "profile")
    require(hidden.device == embeddings.device, "profile")
    with torch.no_grad():
        h, e = hidden.float(), embeddings.float()
        require(bool(torch.isfinite(h).all()) and bool(torch.isfinite(e).all()), "numerical")
        mh, me = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
        hc, ec = h - mh, e - me
        _, singular, vh = torch.linalg.svd(torch.cat((hc, ec), dim=0), full_matrices=False)
        require(bool(torch.isfinite(singular).all()) and float(singular[0]) > 0, "numerical")
        threshold = max(float(singular[0]) * rank_rtol, eigen_floor)
        rank = min(int((singular > threshold).sum()), 2 * (n - 1))
        require(rank > 0, "numerical")
        q = vh[:rank].mT
        bh, be = hc @ q, ec @ q
        eye = torch.eye(rank, dtype=h.dtype, device=h.device)
        ch = bh.mT @ bh / n + ridge * eye
        ce = be.mT @ be / n + ridge * eye
        x = bh @ _symmetric_power(ch, -0.5, eigen_floor)
        y = be @ _symmetric_power(ce, -0.5, eigen_floor)
        u, _, v = torch.linalg.svd(x.mT @ y, full_matrices=False)
        z = (x @ (u @ v) @ _symmetric_power(ce, 0.5, eigen_floor)) @ q.mT + me
        norms = z.norm(dim=-1, keepdim=True)
        target = torch.as_tensor(target_norm, device=z.device, dtype=z.dtype)
        require(target.numel() == 1 and bool(torch.isfinite(target)) and float(target) > 0, "numerical")
        require(bool(torch.isfinite(z).all()) and bool((norms > eigen_floor).all()), "numerical")
        z = z * (target / norms)
        require(bool(torch.isfinite(z).all()), "numerical")
        return z.to(dtype=hidden.dtype).contiguous(), AlignmentInfo(ALIGNMENT, rank, n, d)


def shard_ranges(embedding):
    """(local_start, local_end, global_start); exclude every padded row."""
    indices = embedding.shard_indices
    org = indices.org_vocab_end_index - indices.org_vocab_start_index
    added = indices.added_vocab_end_index - indices.added_vocab_start_index
    ranges = [(0, org, indices.org_vocab_start_index)]
    if added:
        offset = indices.num_org_elements_padded
        ranges.append((offset, offset + added, indices.added_vocab_start_index))
    require(sum(b - a for a, b, _ in ranges) > 0, "profile")
    require(embedding.weight.ndim == 2 and embedding.weight.shape[1] == embedding.embedding_dim, "profile")
    return ranges


def mean_embedding_norm(embedding, *, tp_group=None, tile_rows=1024):
    torch = _torch()
    weight = embedding.weight
    total, error = None, ""
    try:
        total = torch.zeros(2, dtype=torch.float64, device=weight.device)
        with torch.no_grad():
            for start, end, _ in shard_ranges(embedding):
                for offset in range(start, end, tile_rows):
                    tile = weight[offset:min(offset + tile_rows, end)].float()
                    norms = tile.norm(dim=-1)
                    require(bool(torch.isfinite(norms).all()), "numerical")
                    total[0] += norms.double().sum()
                    total[1] += norms.numel()
    except Exception:
        error = "numerical"
    if embedding.tp_size > 1:
        require(tp_group is not None, "ranks")
        errors = [None] * torch.distributed.get_world_size(tp_group.cpu_group)
        torch.distributed.all_gather_object(errors, error, group=tp_group.cpu_group)
        require(not any(errors), "numerical")
        torch.distributed.all_reduce(total, group=tp_group.device_group)
    else:
        require(not error, "numerical")
    require(float(total[1]) == embedding.num_embeddings, "ranks")
    return (total[0] / total[1]).float()


def anchor_to_labels(prefix, embeddings, *, alpha=0.0):
    """Blend known accepted-token embeddings before TP broadcast and vocab anchor.

    Zero returns the original object without arithmetic. Scratch is bounded by
    the existing 128-row solver workspace; no vocabulary-sized tensors or syncs.
    """
    require(type(alpha) in (int, float) and alpha in (0.0, 0.3, 0.5, 0.7, 1.0), "profile")
    if alpha == 0:
        return prefix
    require(prefix.shape == embeddings.shape and prefix.device == embeddings.device, "profile")
    with _torch().no_grad():
        return ((1 - alpha) * prefix.float() + alpha * embeddings.float()).to(prefix.dtype).contiguous()


def anchor_tiled(prefix, embedding, *, alpha=0.3, tp_group=None, tile_rows=1024):
    """Exact global max cosine + smallest-token tie; never gather vocab weights."""
    torch = _torch()
    require(alpha in (0.0, 0.3) and 1 <= tile_rows <= 4096, "profile")
    if alpha == 0:
        return prefix
    local_error = ""
    try:
        with torch.no_grad():
            query = prefix.float()
            query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scores = torch.full((prefix.shape[0],), -float("inf"), device=prefix.device)
            ids = torch.full((prefix.shape[0],), 2**62, dtype=torch.int64, device=prefix.device)
            for start, end, global_start in shard_ranges(embedding):
                for offset in range(start, end, tile_rows):
                    tile = embedding.weight[offset:min(offset + tile_rows, end)].float()
                    tile = tile / tile.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    local_score, local_idx = (query @ tile.mT).max(dim=-1)
                    token = local_idx + offset - start + global_start
                    better = (local_score > scores) | ((local_score == scores) & (token < ids))
                    scores = torch.where(better, local_score, scores)
                    ids = torch.where(better, token, ids)
    except Exception:
        local_error = "numerical"
    if embedding.tp_size > 1:
        require(tp_group is not None, "ranks")
        errors = [None] * torch.distributed.get_world_size(tp_group.cpu_group)
        torch.distributed.all_gather_object(errors, local_error, group=tp_group.cpu_group)
        require(not any(errors), "numerical")
    else:
        require(not local_error, "numerical")
    with torch.no_grad():
        if embedding.tp_size > 1:
            require(tp_group is not None, "ranks")
            maximum = scores.clone()
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=tp_group.device_group)
            ids = torch.where(scores == maximum, ids, torch.full_like(ids, 2**62))
            torch.distributed.all_reduce(ids, op=torch.distributed.ReduceOp.MIN, group=tp_group.device_group)
        require(bool((ids < embedding.num_embeddings).all()), "numerical")
        nearest = embedding(ids)
        return ((1 - alpha) * prefix.float() + alpha * nearest.float()).to(prefix.dtype).contiguous()


def workspace_bound(count, hidden_dim, ranks, tile_rows, alpha):
    """Conservative aggregate plugin tensor plan incl. 64MiB solver allowance.

    Native-library internal allocator peaks must additionally be measured in the
    root-only exact-image test; this number is not hardware peak attestation.
    """
    require(2 <= count <= 128 and hidden_dim > 2 * count and ranks > 0)
    solver = 4 * (12 * count * hidden_dim + 64 * count * count) + 64 * 1024 * 1024
    norms = ranks * 4 * (2 * tile_rows * hidden_dim + 4 * tile_rows)
    anchors = ranks * 4 * (3 * tile_rows * hidden_dim + count * tile_rows) if alpha else 0
    return max(solver, norms, anchors)
