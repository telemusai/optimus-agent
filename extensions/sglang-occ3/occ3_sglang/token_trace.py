"""Opt-in token/geometry evidence; no transport identities or tensor contents."""
import hashlib
import json
import logging
import os
from functools import wraps

LOG = logging.getLogger("occ3_sglang.token_trace")
PREFIX = "occ3-bridge: token trace "
MAX_LOG_CHARS = 16 * 1024
WINDOW = 64


def isolated(function):
    """Keep even tensor-copy, decoder and logging failures off the request path."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            if os.environ.get("OCC3_DIAG_TOKEN_TRACE") == "1":
                return function(*args, **kwargs)
        except Exception:
            pass  # Never log exception strings: they can contain request data.
        return None
    return call


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def first_mismatch(left, right):
    """Zero-based mismatch, including a length-only difference; None means equal."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def summary(ids, tokenizer=None):
    """Only bounded excerpts are decoded; the digest covers the full ID sequence."""
    result = {"count": len(ids), "sha256": ids_hash(ids),
              "first_ids": ids[:WINDOW], "last_ids": ids[-WINDOW:]}
    if tokenizer is not None:
        for name, values in (("first_text", ids[:WINDOW]), ("last_text", ids[-WINDOW:])):
            result[name] = tokenizer.decode(values, skip_special_tokens=False)[:512]
    return result


def emit(event, **fields):
    encoded = json.dumps({"event": event, **fields}, ensure_ascii=True, separators=(",", ":"))
    if len(PREFIX) + len(encoded) > MAX_LOG_CHARS:
        encoded = json.dumps({"event": event, "diag_error": "size_limit"})
    LOG.info("%s%s", PREFIX, encoded)


@isolated
def capture(runtime, record, req, tokens, start, end):
    """Record at most 128 source labels beside their continuous rows, not on wire."""
    if runtime.rank != 0:
        return
    if not 0 <= end - start <= 128 or tokens.numel() > 128:
        return
    labels = [int(value) for value in tokens.tolist()]
    accepted = list(req.output_ids[start:end])
    record.occ3_trace_labels = tuple(labels)
    record.occ3_trace_prefix_hash = tensor_hash(record.prefix)
    record.occ3_trace_hash = ids_hash(labels)
    record.occ3_trace_start = len(req.origin_input_ids) - 1 + start
    emit("capture", labels=summary(labels), accepted=summary(accepted),
         first_mismatch=first_mismatch(accepted, labels), accepted_start=start, accepted_end=end,
         predictor_start=record.occ3_trace_start,
         predictor_end=record.occ3_trace_start + len(labels), predictor_offset=-1,
         rows=int(record.prefix.shape[0]), prefix_sha256=record.occ3_trace_prefix_hash)


@isolated
def capture_text(runtime, record, scheduler):
    if runtime.rank == 0 and hasattr(record, "occ3_trace_labels"):
        emit("capture_text", source_labels=summary(list(record.occ3_trace_labels),
             getattr(scheduler, "tokenizer", None)))


@isolated
def admit(runtime, record, req, scheduler, descriptor):
    """Read the rewritten prompt; IDs in the attachment are placeholders, not text."""
    if runtime.rank != 0:
        return
    at, count = descriptor.prefix_offset, descriptor.expected_prefix_tokens
    ids = req.origin_input_ids
    labels = list(getattr(record, "occ3_trace_labels", ()))
    emit("consume_admit", source_labels=summary(labels, getattr(scheduler, "tokenizer", None)),
         source_hash_matches=ids_hash(labels) == getattr(record, "occ3_trace_hash", None),
         prompt_rows=len(ids), prefix_offset=at, prefix_rows=count,
         before=summary(list(ids[max(0, at-WINDOW):at])),
         holes=summary(list(ids[at:at+count])),
         after=summary(list(ids[at+count:at+count+WINDOW])),
         private_cache=bool(req.occ3_private_kv), labels_are_not_receiver_tokens=True)


@isolated
def chunk(runtime, record, item, forward_batch, local, count, source, lane):
    """Observe actual RoPE positions and mapped source rows, including empty chunks."""
    labels = list(getattr(record, "occ3_trace_labels", ()))
    positions = forward_batch.positions
    actual = [int(x) for x in positions[local:local+count].tolist()]
    shift = 1 if lane == "draft" else 0
    expected = list(range(item.prefix_len + local, item.prefix_len + local + count))
    cache = getattr(forward_batch, "out_cache_loc", None)
    slots = [] if cache is None else [int(x) for x in cache[local:local+count].tolist()]
    if lane == "target" and slots:
        saved = getattr(record, "occ3_trace_slots", {})
        saved.update(zip(expected, slots))
        record.occ3_trace_slots = saved
    emit("consume_chunk", lane=lane, rank=runtime.rank,
         dsa_pin=pin_summary(getattr(forward_batch, "occ3_dsa_pin", None)), source_sha256=getattr(record, "occ3_trace_hash", None),
         chunk_start=item.prefix_len, chunk_end=item.prefix_len + item.extend_len,
         local_start=local, source_start=source, rows=count,
         source_labels=summary(labels[source:source+count]),
         positions=summary(actual), first_position_mismatch=first_mismatch(expected, actual),
         chunk_position_first=int(positions[0]), chunk_position_last=int(positions[-1]),
         kv_slots=summary(slots), draft_shift=shift,
         dsa_pin_offset=max(0, item.descriptor.prefix_offset-shift),
         dsa_pin_expected=item.descriptor.expected_prefix_tokens -
             (1 if shift and item.descriptor.prefix_offset == 0 else 0))


@isolated
def target_chunk(runtime, item, forward_batch, hidden_states):
    from .latent_worker import _span
    local, count, source = _span(item, item.extend_len)
    record = runtime.resolve(item.descriptor)
    chunk(runtime, record, item, forward_batch, local, count, source, "target")
    copied(runtime, record, hidden_states, local, count, source, "target")


def tensor_hash(tensor):
    """Hash at most 128 aligned rows; no floating point or raw bytes enter logs."""
    if tensor.ndim != 2 or tensor.shape[0] > 128 or tensor.numel() > 128 * 32768:
        return None
    import torch
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@isolated
def copied(runtime, record, values, local, count, source, lane):
    """Each rank verifies its local injection; capture comparison needs a baseline."""
    if not count:
        return
    original = tensor_hash(record.prefix)
    selected = tensor_hash(record.prefix[source:source+count])
    actual = tensor_hash(values[local:local+count])
    capture_hash = getattr(record, "occ3_trace_prefix_hash", None)
    emit("consume_embeddings", lane=lane, rank=runtime.rank,
         source_sha256=getattr(record, "occ3_trace_hash", None),
         prefix_sha256=original, prefix_matches_capture=(original == capture_hash
             if original is not None and capture_hash is not None else None),
         source_start=source, rows=count, selected_sha256=selected, copied_sha256=actual,
         copy_matches=selected is not None and selected == actual)


@isolated
def draft_chunk(runtime, record, item, forward_batch, values, local, count, source):
    chunk(runtime, record, item, forward_batch, local, count, source, "draft")
    copied(runtime, record, values, local, count, source, "draft")


@isolated
def settled(runtime, record, scheduler, req, descriptor):
    """The request row is already freed; report saved injection mapping only."""
    if runtime.rank != 0:
        return
    at, count = descriptor.prefix_offset, descriptor.expected_prefix_tokens
    saved = getattr(record, "occ3_trace_slots", {})
    initial = [saved.get(position) for position in range(at, at+count)]
    emit("consume_settle", source_sha256=getattr(record, "occ3_trace_hash", None),
         kv_slots=summary(initial), initial_mapping_complete=all(x is not None for x in initial),
         mapping_source="injection", prefix_offset=at, rows=count,
         num_retractions=req.retraction_count)


def pin_summary(pin):
    """Read the actual forward pin's three bounded integer vectors, not its repr."""
    if pin is None:
        return None
    if len(pin) != 3 or any(value.ndim != 1 or value.numel() > 128 for value in pin):
        return {"skipped": "pin_shape"}
    return [[int(x) for x in value.tolist()] for value in pin]


@isolated
def alignment(runtime, prefix, embeddings):
    """Compare aligned and token-embedding rows already resident on this rank."""
    if runtime.rank != 0:
        return
    if (prefix.ndim != 2 or prefix.shape != embeddings.shape or
            not 0 < prefix.shape[0] <= 128 or prefix.numel() > 128 * 32768):
        return
    import math
    import torch
    values = torch.nn.functional.cosine_similarity(
        prefix.detach().float(), embeddings.detach().float(), dim=1).cpu().tolist()
    emit("alignment", rank=runtime.rank, rows=len(values),
         row_cosine=[round(v, 6) if math.isfinite(v) else None for v in values])


@isolated
def kv_saved(runtime, record, descriptor, point, fence=False):
    """Use owned physical slots before release; never read a freed request row."""
    pool = getattr(record, "occ3_trace_kv_pool", None)
    if pool is None:
        return
    at, count = descriptor.prefix_offset, descriptor.expected_prefix_tokens
    saved = getattr(record, "occ3_trace_slots", {})
    slots = [saved.get(i) for i in range(at, at+count)]
    if fence:
        try:
            runtime.fence()
        except Exception:
            emit("consume_kv_values", point=point, rank=runtime.rank,
                 skipped="fence_failed")
            return
    kv_snapshot(runtime, record, pool, slots, point)


KV_COPY_LIMIT = 8 * 1024 * 1024


def kv_buffers(pool):
    """Physical CUDA DSA layout: MLA rows; paged index keys then FP32 scales."""
    result = []
    for layer, buf in enumerate(pool.kv_buffer):
        if len(buf.shape) != 3 or buf.shape[1] != 1 or buf.element_size() != 1:
            raise ValueError("unsupported_mla_layout")
        result.append(("mla", layer, buf, buf.shape[2]))
    for layer, buf in enumerate(getattr(pool, "index_k_with_scale_buffer", ())):
        if not buf.shape[0]:
            continue
        if (pool.page_size != 64 or pool.index_head_dim != 128 or
                pool.index_kpool != 1 or pool.index_kpool_compress or
                len(buf.shape) != 2 or buf.shape[1] != 64 * 132 or
                buf.element_size() != 1):
            raise ValueError("unsupported_index_layout")
        result.append(("index", layer, buf, 132))
    return result


def kv_indices(slots, device):
    """Build bounded gather indices once per point on the pool's device."""
    import torch
    if torch.version.hip:
        raise ValueError("unsupported_hip_layout")
    ids = torch.tensor(slots, dtype=torch.long, device=device)
    page, row = ids // 64, ids % 64
    return (ids, page[:, None],
            row[:, None] * 128 + torch.arange(128, device=device)[None, :],
            64 * 128 + row[:, None] * 4 + torch.arange(4, device=device)[None, :])


def kv_selected(kind, buf, slots, indices):
    """Gather independently allocated buffers without copying whole caches."""
    import torch
    if not slots or any(type(i) is not int or i < 0 for i in slots):
        raise ValueError("invalid_slots")
    capacity = buf.shape[0] * (64 if kind == "index" else 1)
    if max(slots) >= capacity:
        raise ValueError("invalid_buffer_slots")
    ids, page, keys, scales = indices
    if kind == "mla":
        selected = buf.index_select(0, ids).reshape(len(slots), -1)
    elif kind == "index":
        selected = torch.cat((buf[page, keys], buf[page, scales]), dim=1)
    else:
        raise ValueError("unsupported_index_layout")
    return selected.contiguous().view(torch.uint8).reshape(-1)


def kv_bytes(kind, buf, slots):
    """Standalone accessor for layout tests; point reads use kv_point_bytes."""
    indices = kv_indices(slots, buf.device)
    return kv_selected(kind, buf, slots, indices).cpu().numpy().tobytes()


def kv_point_bytes(buffers, slots):
    """One bounded D2H transfer; a gather per allocation avoids whole-pool copies."""
    import torch
    device = buffers[0][2].device
    if any(buf.device != device for _, _, buf, _ in buffers):
        raise ValueError("unsupported_mla_layout")
    indices = kv_indices(slots, device)
    pieces = [kv_selected(kind, buf, slots, indices) for kind, _, buf, _ in buffers]
    data = torch.cat(pieces).cpu().numpy().tobytes()
    result, offset = [], 0
    for _, _, _, width in buffers:
        length = len(slots) * width
        result.append(data[offset:offset+length])
        offset += length
    if offset != len(data):
        raise ValueError("copy_length")
    return result


@isolated
def kv_snapshot(runtime, record, pool, slots, point):
    """Per-rank bounded D2H copy; compare original physical slots, no collectives."""
    try:
        buffers = kv_buffers(pool)
        if not slots or any(type(i) is not int or i < 0 for i in slots):
            raise ValueError("invalid_slots")
        # Paged allocators reserve page zero: the last allocated page can hold
        # slots >= pool.size. Physical buffer capacity, not logical size, bounds reads.
        for kind, _, buf, _ in buffers:
            capacity = buf.shape[0] * (64 if kind == "index" else 1)
            if max(slots) >= capacity:
                raise ValueError("invalid_buffer_slots")
        width = sum(entry[3] for entry in buffers)
        if not width or width > KV_COPY_LIMIT:
            raise ValueError("layout_exceeds_budget")
        limit = KV_COPY_LIMIT // width
        tail = min(16, len(slots))
        if limit < tail:
            raise ValueError("layout_exceeds_budget")
        chosen = slots
        if len(slots) > limit:
            rest = limit - tail
            head = len(slots) - tail
            spaced = ([slots[i * (head-1) // (rest-1)] for i in range(rest)]
                      if rest > 1 else slots[:rest])
            chosen = spaced + slots[-tail:]
        digests = [hashlib.sha256() for _ in chosen]
        whole = hashlib.sha256()
        copied = 0
        point_bytes = kv_point_bytes(buffers, chosen)
        for (_, _, _, row_bytes), data in zip(buffers, point_bytes):
            if len(data) != len(chosen) * row_bytes:
                raise ValueError("copy_length")
            whole.update(data)
            copied += len(data)
            for i, digest in enumerate(digests):
                digest.update(data[i*row_bytes:(i+1)*row_bytes])
        current = [d.hexdigest() for d in digests]
        baseline = getattr(record, "occ3_trace_kv_baseline", None)
        comparable = baseline is not None and baseline[0] == chosen
        mismatch = first_mismatch(baseline[1], current) if comparable else None
        if point == "prefill_complete":
            record.occ3_trace_kv_baseline = (chosen, current)
        emit("consume_kv_values", point=point, rank=runtime.rank,
             source_sha256=getattr(record, "occ3_trace_hash", None),
             sha256=whole.hexdigest(), changed=(mismatch is not None) if comparable else None,
             first_changed_slot=chosen[mismatch] if mismatch is not None else None,
             sampled=len(chosen) != len(slots), slots=len(slots), hashed_slots=len(chosen),
             bytes_copied=copied, blocking_copies=1,
             baseline_comparable=comparable)
    except Exception as error:
        reason = str(error) if type(error) is ValueError else "snapshot_failed"
        emit("consume_kv_values", point=point, rank=runtime.rank,
             skipped=reason if reason in {"invalid_slots", "invalid_buffer_slots",
                 "unsupported_mla_layout", "unsupported_index_layout", "layout_exceeds_budget",
                 "copy_length", "unsupported_hip_layout"} else "snapshot_failed")


@isolated
def kv_forward(model_runner, forward_batch, point):
    """Only eager prefill owns context; shared decode deliberately has none."""
    from .latent_worker import _runtime, ConsumeWorkerDescriptor
    context = getattr(forward_batch, "occ3_context", None)
    if (point != "prefill_complete" or not context or
            getattr(model_runner, "is_draft_worker", False)):
        return
    item = context[0]
    if type(item.descriptor) is not ConsumeWorkerDescriptor:
        return
    runtime = _runtime()
    record = runtime.resolve(item.descriptor)
    at, count = item.descriptor.prefix_offset, item.descriptor.expected_prefix_tokens
    saved = getattr(record, "occ3_trace_slots", {})
    slots = [saved.get(i) for i in range(at, at+count)]
    if any(i is None for i in slots) or hasattr(record, "occ3_trace_kv_pool"):
        return
    record.occ3_trace_kv_pool = model_runner.token_to_kv_pool
    kv_snapshot(runtime, record, model_runner.token_to_kv_pool, slots, point)


def rounded(value):
    """JSON-safe scalar; undefined/nonfinite geometry is null."""
    import math
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


@isolated
def alignment_stage(runtime, values, embeddings, tokens, stage):
    """Observe resident rows before/after each transform, without retaining tensors."""
    if runtime.rank != 0 or stage not in ("raw", "aligned", "final"):
        return
    import torch
    if (values.ndim != 2 or values.shape != embeddings.shape or
            not 0 < values.shape[0] <= 128 or not 0 < values.shape[1] <= 32768 or
            tokens.ndim != 1 or tokens.numel() != values.shape[0]):
        return
    labels = tokens.tolist()
    if any(type(t) is not int or not 0 <= t < runtime.embedding.num_embeddings for t in labels):
        return
    with torch.no_grad():
        v, e = values.detach().float(), embeddings.detach().float()
        vn, en = v.norm(dim=1), e.norm(dim=1)
        cos = torch.nn.functional.normalize(v, dim=1) @ torch.nn.functional.normalize(e, dim=1).mT
        for i, token in enumerate(labels):
            offsets = [rounded(cos[i, j]) if 0 <= j < len(labels) and vn[i] > 0 and en[j] > 0
                       else None for j in (i-1, i, i+1)]
            emit("alignment_row", rank=0, stage=stage, row=i, token=token,
                 dtype=str(values.dtype), cosine_offsets=offsets,
                 row_norm=rounded(vn[i]), embedding_norm=rounded(en[i]),
                 anchor_alpha=rounded(runtime.profile.anchor_alpha),
                 target_norm=None if runtime.target_norm is None else rounded(runtime.target_norm))


@isolated
def alignment_nearest(runtime, prefix, tokens):
    """Shard-local top-1 only: no embedding forward, gather, or collective."""
    import torch
    from .statebridge import shard_ranges
    embedding = runtime.embedding
    if (prefix.ndim != 2 or not 0 < prefix.shape[0] <= 128 or
            not 0 < prefix.shape[1] <= 32768 or tokens.ndim != 1 or
            tokens.numel() != prefix.shape[0]):
        return
    labels = tokens.tolist()
    if any(type(t) is not int or not 0 <= t < embedding.num_embeddings for t in labels):
        return
    ranges = shard_ranges(embedding)
    weight = embedding.weight
    if weight.shape[1] != prefix.shape[1] or weight.device != prefix.device:
        return
    if any(not (0 <= a <= b <= weight.shape[0] and 0 <= g <= g+b-a <= embedding.num_embeddings)
           for a, b, g in ranges):
        return
    with torch.no_grad():
        query = torch.nn.functional.normalize(prefix.detach().float(), dim=1)
        scores = torch.full((len(labels),), -float("inf"), device=prefix.device)
        ids = torch.full((len(labels),), 2**62, dtype=torch.int64, device=prefix.device)
        for start, end, global_start in ranges:
            for offset in range(start, end, 256):
                tile = weight[offset:min(offset+256, end)].detach().float()
                if not bool(torch.isfinite(tile).all()):
                    return
                tile = torch.nn.functional.normalize(tile, dim=1)
                score, index = (query @ tile.mT).max(dim=1)
                token = index + offset - start + global_start
                better = (score > scores) | ((score == scores) & (token < ids))
                scores = torch.where(better, score, scores)
                ids = torch.where(better, token, ids)
        for i, (token, score) in enumerate(zip(ids.tolist(), scores.tolist())):
            emit("alignment_nearest_shard", rank=runtime.rank, row=i, token=labels[i],
                 top1_id=token, top1_cosine=rounded(score), top1_hit=token == labels[i],
                 scope="local_shard", ranges=ranges)


def precision_cosine(left, right):
    """Per-row cosine, with null for zero or nonfinite rows; inputs stay untouched."""
    import torch
    a, b = left.detach().float(), right.detach().float()
    values = torch.nn.functional.cosine_similarity(a, b, dim=1)
    valid = (torch.isfinite(a).all(dim=1) & torch.isfinite(b).all(dim=1) &
             (a.norm(dim=1) > 0) & (b.norm(dim=1) > 0))
    return [rounded(v) if ok else None for v, ok in zip(values.tolist(), valid.tolist())]


@isolated
def alignment_cast(runtime, hidden, solved, copied):
    """Measure the actual solver-to-serving-buffer cast, before anchoring."""
    if runtime.rank != 0:
        return
    import torch
    if (solved.ndim != 2 or solved.shape != copied.shape or solved.shape != hidden.shape or
            not 0 < solved.shape[0] <= 128 or not 0 < solved.shape[1] <= 32768):
        return
    with torch.no_grad():
        cosine = precision_cosine(solved, copied)
        difference = (solved.detach().float() - copied.detach().float()).abs().amax(dim=1).tolist()
        for i, (cos, delta) in enumerate(zip(cosine, difference)):
            emit("alignment_cast", rank=0, row=i, hidden_dtype=str(hidden.dtype),
                 solver_dtype=str(solved.dtype), buffer_dtype=str(copied.dtype),
                 cosine=cos, max_abs_diff=rounded(delta))


@isolated
def alignment_precision(runtime, prefix, embeddings):
    """Unscaled FP8 round-trip proxy only, not a simulation of the KV encoder."""
    if runtime.rank != 0:
        return
    import torch
    if (prefix.ndim != 2 or prefix.shape != embeddings.shape or
            not 0 < prefix.shape[0] <= 128 or not 0 < prefix.shape[1] <= 32768):
        return
    dtype = getattr(torch, "float8_e4m3fn", None)
    reason = None
    served, baseline = [None] * prefix.shape[0], [None] * prefix.shape[0]
    if dtype is None:
        reason = "float8_unavailable"
    else:
        try:
            with torch.no_grad():
                served = precision_cosine(prefix, prefix.detach().to(dtype).float())
                baseline = precision_cosine(embeddings, embeddings.detach().to(dtype).float())
        except Exception:
            reason = "float8_conversion_unsupported"
    for i, (cos, ref) in enumerate(zip(served, baseline)):
        emit("alignment_precision", rank=0, row=i, anchored_dtype=str(prefix.dtype),
             embedding_dtype=str(embeddings.dtype), fp8_cosine=cos, embedding_fp8_cosine=ref,
             fp8_skipped=reason, fp32_end_to_end_skipped="requires_solver_and_global_anchor_replay")
