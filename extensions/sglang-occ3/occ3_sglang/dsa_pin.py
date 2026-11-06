"""OCC-3 DSA indexer pin: per-batch-row forced range for attached consume rows.

The DSA indexer keeps the top ``index_topk`` keys per query. The injected
continuous prefix is a contiguous span ``[prefix_offset, prefix_offset +
expected_prefix_tokens)`` of an attached consume row; the attested
``dsa-pin`` manifest forces exactly that span into each indexer selection.

``ForwardBatch.occ3_dsa_pin`` is ``None`` (every batch without an attached
consume row) or ``(start, count, forced)``: three int32 device tensors with
one entry per *batch row*, rebuilt from ``batch.reqs`` by index in every
``ForwardBatch.init_new``. It is never keyed by ``req_pool_idx``, so a
filtered, merged or reordered batch cannot carry a stale pin. The indexer
expands rows to logits rows itself (``row // width`` for decode,
target-verify and draft-extend; ``token_to_batch_idx`` for ragged prefill),
exactly the expansion ``get_seqlens_expanded()`` uses. ``forced`` receives the
per-row number of pin lanes forced in the forward. The indexer builds a
logits-independent plan (row map, lane indices, validity, ``forced``) once per
forward on its first layer and caches it on the forward batch as
``occ3_dsa_pin_plan``; each layer then only gathers, fills and scatters.

No CUDA graph carries the pin. Every graph runner (prefill, decode and
target-verify, EAGLE draft, draft-extend) captures with ``occ3_dsa_pin``
None and its ``can_run_graph`` returns False for a batch that carries a pin,
so ordinary batches replay the original unpinned graphs at their original
cost and a pinned batch runs eager with this per-batch tensor.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_bounds_logged = False
_fault_logged = False
# Process-wide count of pin faults (a pin that could not be built or
# applied). Admission snapshots it on the request; terminal settle fails an
# over-top-k consume whose window saw a fault, so a fault never yields a
# completed-use receipt and never raises on the shared scheduler path.
faults = 0

# Fixed lane width of the indexer pin; must stay strictly below index_topk.
PIN_LANES = 128


def pin_rows(reqs, shift=0):
    """Host-side (start, count) per batch row; zeros for every non-pinned row.

    Only an attached consume row with a bounded descriptor is pinned.
    Returns None when no row is pinned, so ordinary batches carry no tensor.
    This runs inside ``ForwardBatch.init_new`` for every batch, so it never
    raises: an out-of-bounds descriptor (unreachable after admission, which
    bounds both values) is logged once and treated as count 0 (no pin).

    ``shift`` is 1 for the EAGLE/NextN draft runner. Its input is shifted by
    one row (latent_worker._effective uses prefix_span(..., shift=1)), so
    prefix row 0 sits at draft row ``offset - 1`` and the draft pin starts
    there with the same count. At offset 0 prefix row 0 never reaches the
    draft and the start stays 0.
    """
    global _bounds_logged
    from .attachment_contract import ConsumeWorkerDescriptor

    starts, counts, any_pin = [], [], False
    for req in reqs:
        descriptor = getattr(req, "occ3_worker_descriptor", None)
        start = count = 0
        if getattr(req, "occ3_attached", False) is True and type(descriptor) is ConsumeWorkerDescriptor:
            count = descriptor.expected_prefix_tokens
            start = descriptor.prefix_offset
            if (type(count) is int and type(start) is int
                    and 0 < count <= PIN_LANES and 0 <= start <= 2**20):
                any_pin = True
                if shift and start > 0:
                    start -= 1
            else:
                if not _bounds_logged:
                    _bounds_logged = True
                    logger.error("occ3 dsa pin descriptor out of bounds; row left unpinned")
                start = count = 0
        starts.append(start)
        counts.append(count)
    return (starts, counts) if any_pin else None


def note_fault(forward_batch=None):
    """Record a pin fault without raising: log once, drop the batch pin.

    Called from the indexer call sites and from :func:`build` when pin work
    raises. The forward then runs unpinned (ordinary rows are unaffected
    either way); only the OCC request is failed later, at terminal settle.
    """
    global faults, _fault_logged
    faults += 1
    if forward_batch is not None:
        try:
            forward_batch.occ3_dsa_pin = None
            forward_batch.occ3_dsa_pin_plan = None
        except Exception:
            pass
    if not _fault_logged:
        _fault_logged = True
        logger.error("occ3 dsa pin fault; forward runs unpinned and the OCC request fails at settle")


def build(batch, device, *, record, draft=None):
    """Build ``ForwardBatch.occ3_dsa_pin`` for one forward from ``batch.reqs``.

    ``record`` (target runner only) stores ``(forced, row)`` on each pinned
    request so terminal settle can read the last forward's pin-hit count; the
    draft runner applies the same pin, in its shifted row space, and never
    overwrites the record. ``draft`` defaults to ``not record``: init_new
    passes ``record=not model_runner.is_draft_worker``.
    """
    try:
        return _build(batch, device, record=record, draft=draft)
    except Exception:
        # init_new runs for every batch: never raise there.
        note_fault(None)
        return None


def _build(batch, device, *, record, draft):
    if draft is None:
        draft = not record
    reqs = list(getattr(batch, "reqs", None) or ())
    rows = pin_rows(reqs, shift=1 if draft else 0)
    if rows is None:
        return None
    import torch

    starts, counts = rows
    start = torch.tensor(starts, dtype=torch.int32).to(device, non_blocking=True)
    count = torch.tensor(counts, dtype=torch.int32).to(device, non_blocking=True)
    forced = torch.zeros(len(reqs), dtype=torch.int32, device=device)
    if record:
        for index, req in enumerate(reqs):
            if counts[index]:
                req.occ3_dsa_pin_forced = (forced, index)
    return start, count, forced


def read_forced(req):
    """Bounded pin-hit count from the request's last target forward, or None."""
    held = getattr(req, "occ3_dsa_pin_forced", None)
    if held is None:
        return None
    forced, index = held
    return max(0, min(PIN_LANES, int(forced[index].item())))
