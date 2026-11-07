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
    emit("consume_settle", source_sha256=getattr(record, "occ3_trace_hash", None),
         kv_slots=summary(slots), initial_mapping_complete=all(x is not None for x in initial),
         first_slot_mismatch=first_mismatch(initial, slots),
         prefix_offset=at, rows=count, num_retractions=req.retraction_count)
