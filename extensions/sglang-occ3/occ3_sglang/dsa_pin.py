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

CUDA graph runners own a static ``[max_bs]`` copy (:class:`GraphPin`): the
captured graph reads it, every replay zeroes all slots and copies the live
rows, so padded and free slots always carry count 0 (a bitwise no-op).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_bounds_logged = False

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


def build(batch, device, *, record, draft=None):
    """Build ``ForwardBatch.occ3_dsa_pin`` for one forward from ``batch.reqs``.

    ``record`` (target runner only) stores ``(forced, row)`` on each pinned
    request so terminal settle can read the last forward's pin-hit count; the
    draft runner applies the same pin, in its shifted row space, and never
    overwrites the record. ``draft`` defaults to ``not record``: init_new
    passes ``record=not model_runner.is_draft_worker``.
    """
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


class GraphPin:
    """Static [max_bs] pin buffers owned by one CUDA graph runner.

    ``view(bs)`` is attached to the capture batch so the graph reads these
    buffers. ``load`` runs before every replay: it zeroes all slots (padded
    and free slots get count 0) and copies the live rows; it uses only
    fixed-shape device copies and fills. ``publish`` copies the graph's forced
    counts back to the live batch's tensor after replay.
    """

    def __init__(self, max_bs, device):
        import torch

        self.max_bs = int(max_bs)
        self.start = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self.count = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self.forced = torch.zeros(self.max_bs, dtype=torch.int32, device=device)

    def view(self, bs):
        return self.start[:bs], self.count[:bs], self.forced[:bs]

    def load(self, forward_batch, raw_bs):
        self.start.zero_()
        self.count.zero_()
        self.forced.zero_()
        pin = getattr(forward_batch, "occ3_dsa_pin", None)
        if pin is None:
            return
        if not 0 <= raw_bs <= self.max_bs or pin[0].shape[0] < raw_bs:
            raise RuntimeError("occ3_dsa_pin_shape")
        self.start[:raw_bs].copy_(pin[0][:raw_bs])
        self.count[:raw_bs].copy_(pin[1][:raw_bs])

    def publish(self, forward_batch, raw_bs):
        pin = getattr(forward_batch, "occ3_dsa_pin", None)
        if pin is not None:
            pin[2][:raw_bs].copy_(self.forced[:raw_bs])
