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
    if runtime.rank != 0:
        return
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
    emit("consume_chunk", lane=lane, source_sha256=getattr(record, "occ3_trace_hash", None),
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
    if runtime.rank != 0 or not count:
        return
    original = tensor_hash(record.prefix)
    selected = tensor_hash(record.prefix[source:source+count])
    actual = tensor_hash(values[local:local+count])
    emit("consume_embeddings", lane=lane, source_sha256=getattr(record, "occ3_trace_hash", None),
         prefix_sha256=original, prefix_matches_capture=(original is not None and
             original == getattr(record, "occ3_trace_prefix_hash", None)),
         source_start=source, rows=count, selected_sha256=selected, copied_sha256=actual,
         copy_matches=selected is not None and selected == actual)


@isolated
def draft_chunk(runtime, record, item, forward_batch, values, local, count, source):
    chunk(runtime, record, item, forward_batch, local, count, source, "draft")
    copied(runtime, record, values, local, count, source, "draft")


@isolated
def settled(runtime, record, scheduler, req, descriptor):
    """Compare terminal slot mapping, not KV values; no engine-owned state changes."""
    if runtime.rank != 0:
        return
    at, count = descriptor.prefix_offset, descriptor.expected_prefix_tokens
    pool = scheduler.req_to_token_pool.req_to_token
    slots = [int(x) for x in pool[req.req_pool_idx, at:at+count].tolist()]
    saved = getattr(record, "occ3_trace_slots", {})
    initial = [saved.get(position) for position in range(at, at+count)]
    if hasattr(record, "occ3_trace_kv_pool"):
        kv_snapshot(runtime, record, record.occ3_trace_kv_pool, initial, "terminal_settle")
    emit("consume_settle", source_sha256=getattr(record, "occ3_trace_hash", None),
         kv_slots=summary(slots), initial_mapping_complete=all(x is not None for x in initial),
         first_slot_mismatch=first_mismatch(initial, slots),
         prefix_offset=at, rows=count, num_retractions=req.retraction_count)


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


def kv_bytes(kind, buf, slots):
    """Gather only selected token bytes, never their neighbouring page rows."""
    import torch
    if torch.version.hip:
        raise ValueError("unsupported_hip_layout")
    ids = torch.tensor(slots, dtype=torch.long, device=buf.device)
    if kind == "mla":
        selected = buf.index_select(0, ids).reshape(len(slots), -1)
    else:
        page, row = ids // 64, ids % 64
        keys = buf[page[:, None], row[:, None] * 128 +
                   torch.arange(128, device=buf.device)[None, :]]
        scales = buf[page[:, None], 64 * 128 + row[:, None] * 4 +
                     torch.arange(4, device=buf.device)[None, :]]
        selected = torch.cat((keys, scales), dim=1)
    return selected.contiguous().view(torch.uint8).cpu().numpy().tobytes()


@isolated
def kv_snapshot(runtime, record, pool, slots, point):
    """Rank zero only, bounded D2H copies; compare original physical slots."""
    if runtime.rank != 0:
        return
    try:
        buffers = kv_buffers(pool)
        if (not slots or any(type(i) is not int or i < 0 or i >= pool.size
                             for i in slots)):
            raise ValueError("invalid_slots")
        for kind, _, buf, _ in buffers:
            capacity = buf.shape[0] * (64 if kind == "index" else 1)
            if max(slots) >= capacity:
                raise ValueError("invalid_buffer_slots")
        width = sum(entry[3] for entry in buffers)
        if not width or width > KV_COPY_LIMIT:
            raise ValueError("layout_exceeds_budget")
        limit = KV_COPY_LIMIT // width
        chosen = slots if len(slots) <= limit else [
            slots[i * (len(slots)-1) // (limit-1)] for i in range(limit)] if limit > 1 else slots[:1]
        digests = [hashlib.sha256() for _ in chosen]
        whole = hashlib.sha256()
        copied = 0
        for kind, layer, buf, row_bytes in buffers:
            data = kv_bytes(kind, buf, chosen)
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
             bytes_copied=copied, blocking_copies=len(buffers),
             baseline_comparable=comparable)
    except Exception as error:
        reason = str(error) if type(error) is ValueError else "snapshot_failed"
        emit("consume_kv_values", point=point, rank=runtime.rank,
             skipped=reason if reason in {"invalid_slots", "invalid_buffer_slots",
                 "unsupported_mla_layout", "unsupported_index_layout", "layout_exceeds_budget",
                 "copy_length", "unsupported_hip_layout"} else "snapshot_failed")


@isolated
def kv_forward(model_runner, forward_batch, point):
    from .latent_worker import _runtime, ConsumeWorkerDescriptor
    runtime = _runtime()
    if runtime.rank != 0 or getattr(model_runner, "is_draft_worker", False):
        return
    item = forward_batch.occ3_context[0]
    record = runtime.resolve(item.descriptor)
    if type(item.descriptor) is not ConsumeWorkerDescriptor:
        return
    at, count = item.descriptor.prefix_offset, item.descriptor.expected_prefix_tokens
    saved = getattr(record, "occ3_trace_slots", {})
    slots = [saved.get(i) for i in range(at, at+count)]
    if any(i is None for i in slots):
        return
    if point == "prefill_complete":
        if hasattr(record, "occ3_trace_kv_pool"):
            return
        record.occ3_trace_kv_pool = model_runner.token_to_kv_pool
    else:
        mode = forward_batch.forward_mode
        if mode.is_extend() and not mode.is_target_verify():
            return
        if (not hasattr(record, "occ3_trace_kv_pool") or
                getattr(record, "occ3_trace_kv_decode", False)):
            return
        record.occ3_trace_kv_decode = True
    kv_snapshot(runtime, record, model_runner.token_to_kv_pool, slots, point)
