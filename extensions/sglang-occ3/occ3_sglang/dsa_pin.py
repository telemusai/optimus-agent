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
per-row number of pin lanes forced in the forward. A row is pinned only while
its sequence can exceed ``index_topk`` (:func:`needs_pin`); shorter attached
consumes carry no pin and their batches keep the ordinary graphs. The indexer builds a
logits-independent plan (row map, lane indices, validity, ``forced``) once per
forward on its first layer and caches it on the forward batch as
``occ3_dsa_pin_plan``; each indexer layer then only gathers, fills and scatters
(GLM-5.3: 21 full layers plus NextN run their own indexer; the shared
layers reuse the pinned indices).

The original CUDA graphs never carry the pin: every graph runner captures
them with ``occ3_dsa_pin`` None, so ordinary batches replay them at their
original cost with no pin work. The decode-family runners (decode and
target-verify, EAGLE draft, draft-extend) also capture a pinned variant per
batch size that reads a static :class:`GraphPin` view; a pinned batch
replays it when every such runner has one for its size (:func:`graph_ok`),
otherwise it runs eager. Prefill graphs have no pinned variant: pinned
prefill chunks run eager.
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

# Install-time threshold, set once per process by the latent runtime
# (configure) after the rank gather: (index_topk, spec_rows). None means the
# pin is not active on this rank and build() pins nothing. Both values are
# static server config, identical on every rank, so the pin decision (and
# with it the graph/eager choice) is rank-uniform.
_threshold = None


def configure(index_topk, spec_rows=1):
    """Arm (or with ``index_topk=None`` disarm) the over-top-k pin threshold.

    ``spec_rows`` is the margin a forward can add beyond the request's
    committed tokens as seen by ``init_new``. latent_worker passes twice the
    EAGLE draft-token width (2 without speculation): one step for this
    forward's query rows and one for the output that overlap scheduling has
    not appended yet. Pinning early only costs the eager path; it never
    drops a pin that settle later requires.
    """
    global _threshold
    if type(index_topk) is int and index_topk > 0:
        rows = spec_rows if type(spec_rows) is int and spec_rows > 0 else 1
        _threshold = (index_topk, rows)
    else:
        _threshold = None


def sequence_rows(req):
    """Committed sequence rows of a request: prompt plus visible output.

    Host ints only (list lengths replicated on every rank by the scheduler).
    Terminal settle (latent_worker._pin_evidence_required) and the producer
    use this same count, so a receipt needs pin evidence only for a request
    that the producer pinned on its last forward.
    """
    return (len(getattr(req, "origin_input_ids", ()) or ())
            + len(getattr(req, "output_ids", ()) or ()))


def needs_pin(req, index_topk, spec_rows=1):
    """True when this forward's causal length can exceed ``index_topk``.

    At or below the top-k the indexer already keeps every valid key, so the
    injected rows are always selected and the pin is a no-op; such a row is
    not pinned and its batch keeps the ordinary graphs. A receiver that grows
    past the top-k during decode is pinned from the first forward whose rows
    (committed rows plus this forward's ``spec_rows``) cross it.
    """
    return sequence_rows(req) + spec_rows > index_topk


def pin_rows(reqs, shift=0, threshold=None):
    """Host-side (start, count) per batch row; zeros for every non-pinned row.

    Only an attached consume row with a bounded descriptor is pinned, and,
    when ``threshold`` = (index_topk, spec_rows) is given, only while
    :func:`needs_pin` holds for it.
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
        if (getattr(req, "occ3_attached", False) is True
                and type(descriptor) is ConsumeWorkerDescriptor
                and (threshold is None or needs_pin(req, *threshold))):
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
    threshold = _threshold
    if threshold is None:
        # Pin not active on this rank: no over-top-k consume is admitted.
        return None
    reqs = list(getattr(batch, "reqs", None) or ())
    rows = pin_rows(reqs, shift=1 if draft else 0, threshold=threshold)
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


# ---------------------------------------------------------------------------
# Pinned CUDA graph variants
#
# Each decode-family graph runner (target decode/verify, EAGLE draft decode,
# EAGLE draft extend) captures a second graph per batch size whose forward
# batch carries a static pin view (GraphPin). The original unpinned graphs are
# captured exactly as before and keep serving every batch without a pinned
# row, with no pin work. A batch with a pinned row replays the pinned graph
# for its padded batch size when every registered runner has one; otherwise
# every runner runs it eager (the a8e98b759 behaviour). The choice is a pure
# function of the raw batch size and static capture lists, so target, draft
# and draft-extend always agree for a batch.
# ---------------------------------------------------------------------------

GRAPHS_ENV = "OCC3_DSA_PIN_GRAPHS"
GRAPH_MAX_BS_ENV = "OCC3_DSA_PIN_GRAPH_MAX_BS"
PIN_LABEL = "occ3_pin"

# runner key -> (sorted capture bs tuple, frozenset of pinned bs)
_graph_runners = {}


def graphs_enabled():
    """Pinned graph set switch; default ON. "0"/"false"/"off"/"no" disable it."""
    import os

    value = os.environ.get(GRAPHS_ENV, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _graph_max_bs():
    import os

    raw = os.environ.get(GRAPH_MAX_BS_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("occ3 dsa pin: ignoring non-integer %s", GRAPH_MAX_BS_ENV)
        return None
    return value if value >= 0 else None


def _dsa_model(model_runner):
    try:
        topk = getattr(model_runner.model_config.hf_config, "index_topk", None)
    except Exception:
        return False
    return type(topk) is int and topk > PIN_LANES


def _two_batch_overlap(runner, model_runner):
    """True when two-batch overlap (TBO) is or may be enabled.

    TBO splits a captured batch into two sub-batches, which the per-row pin
    view does not mirror, so no pinned graph set is captured and pinned
    batches run eager in every runner (graph_ok then fails for all sizes).
    At the pinned commit the flag is ``get_exec().overlap.
    enable_two_batch_overlap`` (the decode runner copies it to
    ``runner.enable_two_batch_overlap``; the EAGLE runners do not), with
    ``server_args.enable_two_batch_overlap`` as the source. An unreadable
    flag counts as enabled (fail closed to eager).
    """
    if getattr(runner, "enable_two_batch_overlap", False) is True:
        return True
    args = getattr(model_runner, "server_args", None)
    if getattr(args, "enable_two_batch_overlap", False) is True:
        return True
    try:
        from sglang.srt.runtime_context import get_exec
    except Exception:
        return False  # no runtime context (offline/test): runner and args decide
    try:
        return bool(get_exec().overlap.enable_two_batch_overlap)
    except Exception:
        return True


def graph_bs(runner, model_runner, capture_bs):
    """Batch sizes that get a pinned graph for ``runner``; () disables it.

    Empty when the switch is off, the model has no DSA indexer top-k, or the
    runner keys or pads graphs in a way the pinned set does not mirror
    (disabled padding, DP MLP gather, compact ragged verify, PD multiplexing,
    two-batch overlap).
    ``OCC3_DSA_PIN_GRAPH_MAX_BS`` caps the list; the default is the full
    capture list. The result is registered for :func:`graph_ok`.
    """
    key = (type(runner).__name__, bool(getattr(model_runner, "is_draft_worker", False)))
    pinned = ()
    try:
        if (graphs_enabled() and _dsa_model(model_runner)
                and not getattr(runner, "disable_padding", False)
                and not getattr(runner, "require_mlp_tp_gather", False)
                and not getattr(runner, "ragged_verify_mode", False)
                and not getattr(runner, "enable_pdmux", False)
                and not _two_batch_overlap(runner, model_runner)):
            cap = _graph_max_bs()
            pinned = tuple(sorted(int(b) for b in capture_bs if cap is None or int(b) <= cap))
        _graph_runners[key] = (tuple(sorted(int(b) for b in capture_bs)), frozenset(pinned))
    except Exception:
        _graph_runners[key] = ((), frozenset())
        pinned = ()
    return frozenset(pinned)


def graph_ok(raw_bs):
    """True when every registered runner has a pinned graph for ``raw_bs``.

    Pure function of the raw batch size and the static capture lists, so the
    target, draft and draft-extend runners make the same pinned-graph vs
    eager choice for a batch (no mix of graph and eager state for a pin).
    """
    if not _graph_runners or not graphs_enabled():
        return False
    try:
        raw_bs = int(raw_bs)
    except Exception:
        return False
    for capture, pinned in _graph_runners.values():
        bucket = next((b for b in capture if b >= raw_bs), None)
        if bucket is None or bucket not in pinned:
            return False
    return True


def pinned_key(shape_key):
    """Graph key of the pinned variant: the same key with a pin label."""
    import dataclasses

    label = getattr(shape_key, "variant_label", None)
    label = PIN_LABEL if label is None else f"{label}+{PIN_LABEL}"
    if dataclasses.is_dataclass(shape_key):
        return dataclasses.replace(shape_key, variant_label=label)
    if hasattr(shape_key, "_replace"):
        return shape_key._replace(variant_label=label)
    fields = ("size", "stream_idx", "variant_label", "attention_variant")
    values = {f: getattr(shape_key, f, None) for f in fields}
    values["variant_label"] = label
    return type(shape_key)(**values)


class GraphPin:
    """Static [max_bs] pin buffers owned by one CUDA graph runner.

    ``view(bs)`` is attached to the pinned capture batch so the pinned graph
    reads these buffers. ``load`` runs before every pinned replay: it zeroes
    all slots (padded and free slots get count 0) and copies the live rows,
    using only fixed-shape device copies and fills. It never raises: on a
    shape mismatch it records a fault, leaves the buffers zero and returns
    False, so the pinned graph replays with count 0 (equal to unpinned) and
    the OCC request fails at settle (_require_pin_evidence). ``publish``
    copies the graph's forced counts back to the live batch's tensor.
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
            return False
        try:
            if not 0 <= int(raw_bs) <= self.max_bs or pin[0].shape[0] < raw_bs \
                    or pin[1].shape[0] < raw_bs:
                raise ValueError("occ3_dsa_pin_shape")
            self.start[:raw_bs].copy_(pin[0][:raw_bs])
            self.count[:raw_bs].copy_(pin[1][:raw_bs])
        except Exception:
            self.start.zero_()
            self.count.zero_()
            note_fault(forward_batch)
            return False
        return True

    def publish(self, forward_batch, raw_bs):
        pin = getattr(forward_batch, "occ3_dsa_pin", None)
        if pin is None:
            return
        try:
            pin[2][:raw_bs].copy_(self.forced[:raw_bs])
        except Exception:
            note_fault(None)
