"""Bounded, tokenizer-free classification of accepted OCC-3 capture spans.

Why this module exists
----------------------
The rank-local worker that commits accepted states (``latent_worker.align``,
lines 303-363) sees only accepted token ids and their absolute positions.  The
tokenizer, the chat template and the reasoning/tool parsers all live in the
tokenizer-manager process.  A worker therefore cannot decode text, cannot
re-tokenize a rendered marker string, and must not try.  It can compare integer
ids.  This module turns that comparison into an explicit, auditable answer:
which accepted rows are reasoning, which are final assistant content, which are
tool-call syntax, which are tool arguments, which are structural control
markers, and which stay unknown.

Matching contract
------------------
Every marker is an exact token-id sequence.  A marker matches at index ``i``
only when ``token_ids[i + k] == marker[k]`` for every ``k``.  There is no text,
no substring search, no decoding and no prefix/suffix tolerance.

Classification contract
------------------------
Classifying an index needs positive evidence.  The tokens of the first
occurrence of each supplied marker become ``CONTROL_MARKERS``.  Then:

* rows before a ``thinking_end`` marker -> ``THINKING``;
* rows after that marker -> ``CONTENT``, but only when the marker was actually
  supplied and observed (without it, reasoning cannot be told from content, so
  those rows stay ``UNKNOWN``);
* rows after ``tool_call_begin`` -> ``TOOL_CALL`` (syntax);
* rows after ``tool_call_end`` -> ``TOOL_ARGUMENTS``;
* rows after ``message_end`` -> ``UNKNOWN`` (a later message is not audited);
* everything else -> ``UNKNOWN``.

Fail-closed contract
--------------------
Contradictory or overlapping markers raise :class:`SpanContractError` with
``code == "lineage"`` instead of returning a classification.  A caller that
supplies no markers at all gets every row classified ``UNKNOWN``; nothing is
guessed from token values, positions or counts.

Privacy contract
----------------
No function returns, formats, logs or hashes token ids, decoded text or hidden
states.  Records carry counts, index/position ranges and booleans only.
``SpanAudit.summary()`` is a fixed allowlist of ints and bools, and ``repr`` of
every record is bounded and id-free.  ``SpanMarkers`` prints marker lengths,
never marker ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .latent_types import LatentError

__all__ = [
    "THINKING", "CONTENT", "TOOL_CALL", "TOOL_ARGUMENTS", "CONTROL_MARKERS",
    "UNKNOWN", "CLASSES", "POLICIES", "SUMMARY_KEYS", "MAX_ACCEPTED",
    "MAX_MARKER_TOKENS", "MAX_RUNS", "SpanContractError", "SpanMarkers",
    "ClassSpan", "SpanRun", "SpanAudit", "SpanSelection", "classify_accepted",
    "span_from_policy",
]

THINKING = "THINKING"
CONTENT = "CONTENT"
TOOL_CALL = "TOOL_CALL"
TOOL_ARGUMENTS = "TOOL_ARGUMENTS"
CONTROL_MARKERS = "CONTROL_MARKERS"
UNKNOWN = "UNKNOWN"

#: Fixed class order for every tuple in this module.
CLASSES = (THINKING, CONTENT, TOOL_CALL, TOOL_ARGUMENTS, CONTROL_MARKERS, UNKNOWN)

POLICY_LAST_ACCEPTED = "last_accepted"
POLICY_FINAL_CONTENT = "final_content"
POLICY_FINAL_CONTENT_NO_TOOL = "final_content_no_tool"
POLICY_FINAL_CONTENT_LAST64 = "final_content_last64"
POLICY_FINAL_CONTENT_LAST32 = "final_content_last32"

#: Supported span policies, most permissive first.
POLICIES = (POLICY_LAST_ACCEPTED, POLICY_FINAL_CONTENT, POLICY_FINAL_CONTENT_NO_TOOL,
            POLICY_FINAL_CONTENT_LAST64, POLICY_FINAL_CONTENT_LAST32)

_TAIL_LIMITS = {POLICY_FINAL_CONTENT_LAST64: 64, POLICY_FINAL_CONTENT_LAST32: 32}

#: Hard bounds keep every audit O(n) in time and O(n) bytes of scratch.
MAX_ACCEPTED = 1 << 20
MAX_MARKER_TOKENS = 32
MAX_MARKER_ID = 2 ** 63 - 1
MAX_RUNS = 16

#: Fixed `SpanAudit.summary()` allowlist.  Ints and bools only.
SUMMARY_KEYS = (
    "total_accepted", "retained_count",
    "thinking", "final_content", "tool_call", "tool_arguments",
    "control_markers", "unknown",
    "window_start_position", "window_end_position",
    "contains_final_content", "contains_tool_call", "message_truncated",
    "contiguous", "tool_call_complete", "markers_supplied", "marker_repeats",
    "runs_truncated",
)

_MARKER_FIELDS = ("thinking_end_ids", "tool_call_begin_ids", "tool_call_end_ids",
                  "message_end_ids")


class SpanContractError(LatentError, ValueError):
    """Fail-closed span decline.

    A ``LatentError`` so worker seams handle it like every other decline, and a
    ``ValueError`` so ordinary callers can catch it.  ``code`` is always a code
    from the shared ``LatentError`` set; span contradictions use ``"lineage"``.
    """

    def __init__(self, code: str = "lineage"):
        super().__init__(code)


def _marker_ids(value) -> Tuple[int, ...] | None:
    """Normalise one optional marker to a bounded tuple of plain ints."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise SpanContractError("lineage")
    try:
        items = tuple(value)
    except TypeError:
        raise SpanContractError("lineage") from None
    if not items:
        # An empty marker would match everywhere.  Treat it as "not supplied".
        return None
    if len(items) > MAX_MARKER_TOKENS:
        raise SpanContractError("lineage")
    for item in items:
        if type(item) is not int or item < 0 or item > MAX_MARKER_ID:
            raise SpanContractError("lineage")
    return items


@dataclass(frozen=True, repr=False)
class SpanMarkers:
    """Immutable token-id marker set produced by the tokenizer-manager process.

    Each field is None when that marker is unknown to the caller.  Sequences are
    normalised to tuples of plain ints; an empty sequence means "not supplied".
    """

    thinking_end_ids: Tuple[int, ...] | None = None
    tool_call_begin_ids: Tuple[int, ...] | None = None
    tool_call_end_ids: Tuple[int, ...] | None = None
    message_end_ids: Tuple[int, ...] | None = None

    def __post_init__(self):
        for name in _MARKER_FIELDS:
            object.__setattr__(self, name, _marker_ids(getattr(self, name)))

    @property
    def supplied(self) -> Tuple[str, ...]:
        """Field names that carry a usable marker, in fixed order."""
        return tuple(name for name in _MARKER_FIELDS if getattr(self, name) is not None)

    @property
    def complete(self) -> bool:
        """True when a tool-call delimiter pair is either whole or absent."""
        beginning = self.tool_call_begin_ids
        end = self.tool_call_end_ids
        return (beginning is None) == (end is None)

    def wire(self) -> dict:
        """Bounded descriptor: marker lengths only, never marker ids."""
        return {name: (None if getattr(self, name) is None else len(getattr(self, name)))
                for name in _MARKER_FIELDS}

    def __repr__(self) -> str:
        lengths = ", ".join("%s=%s" % (name, "None" if value is None else value)
                            for name, value in self.wire().items())
        return "SpanMarkers(%s)" % lengths


@dataclass(frozen=True, repr=False)
class ClassSpan:
    """Index and position range of one class inside the retained window.

    ``first_index``/``last_index`` are accepted-sequence indices, inclusive.
    ``first_position``/``last_position`` are absolute positions, inclusive.
    An empty class reports -1 in every field.
    """

    kind: str
    count: int
    first_index: int
    last_index: int
    first_position: int
    last_position: int

    @property
    def empty(self) -> bool:
        return self.count == 0

    def __repr__(self) -> str:
        return "ClassSpan(%s, count=%d, index=%d..%d, position=%d..%d)" % (
            self.kind, self.count, self.first_index, self.last_index,
            self.first_position, self.last_position)


@dataclass(frozen=True, repr=False)
class SpanRun:
    """One maximal run of a single class inside the retained window.

    ``end_index`` and ``end_position`` are exclusive.  ``contiguous`` is True
    only when every adjacent pair inside the run is one absolute position apart.
    """

    kind: str
    start_index: int
    end_index: int
    start_position: int
    end_position: int
    contiguous: bool

    @property
    def length(self) -> int:
        return self.end_index - self.start_index

    def __repr__(self) -> str:
        return "SpanRun(%s, index=%d..%d, position=%d..%d, contiguous=%s)" % (
            self.kind, self.start_index, self.end_index, self.start_position,
            self.end_position, self.contiguous)


@dataclass(frozen=True, repr=False)
class SpanAudit:
    """Frozen, token-free description of one retained accepted window.

    The record holds counts, ranges and flags only.  It cannot reproduce a token
    id, a decoded string or a hidden state, so it is safe to log, attach to a
    receipt and forward across a process boundary.
    """

    total_accepted: int
    retained_count: int
    window_start_index: int
    window_end_index: int
    selected_positions: Tuple[int, int]
    spans: Tuple[ClassSpan, ...]
    runs: Tuple[SpanRun, ...]
    runs_truncated: bool
    markers_supplied: Tuple[str, ...]
    marker_occurrences: Tuple[Tuple[str, int], ...]
    contains_final_content: bool
    contains_tool_call: bool
    message_truncated: bool
    contiguous: bool
    tool_call_complete: bool

    @property
    def counts(self) -> dict:
        """Per-class counts in fixed :data:`CLASSES` order."""
        return {span.kind: span.count for span in self.spans}

    def count(self, kind: str) -> int:
        return self.span(kind).count

    def span(self, kind: str) -> ClassSpan:
        if kind not in CLASSES:
            raise SpanContractError("lineage")
        for span in self.spans:
            if span.kind == kind:
                return span
        raise SpanContractError("lineage")

    def index_range(self, kind: str) -> Tuple[int, int]:
        """Inclusive (first_index, last_index), or (-1, -1) when empty."""
        span = self.span(kind)
        return (span.first_index, span.last_index)

    def position_range(self, kind: str) -> Tuple[int, int]:
        """Inclusive (first_position, last_position), or (-1, -1) when empty."""
        span = self.span(kind)
        return (span.first_position, span.last_position)

    def last_run(self, kind: str) -> SpanRun | None:
        """Last run of ``kind``, or None when empty or the run list overflowed."""
        if kind not in CLASSES:
            raise SpanContractError("lineage")
        if self.runs_truncated:
            return None
        found = None
        for run in self.runs:
            if run.kind == kind:
                found = run
        return found

    def summary(self) -> dict:
        """Fixed allowlist of ints and bools.  No token ids, no text."""
        counts = self.counts
        return {
            "total_accepted": self.total_accepted,
            "retained_count": self.retained_count,
            "thinking": counts[THINKING],
            "final_content": counts[CONTENT],
            "tool_call": counts[TOOL_CALL],
            "tool_arguments": counts[TOOL_ARGUMENTS],
            "control_markers": counts[CONTROL_MARKERS],
            "unknown": counts[UNKNOWN],
            "window_start_position": self.selected_positions[0],
            "window_end_position": self.selected_positions[1],
            "contains_final_content": self.contains_final_content,
            "contains_tool_call": self.contains_tool_call,
            "message_truncated": self.message_truncated,
            "contiguous": self.contiguous,
            "tool_call_complete": self.tool_call_complete,
            "markers_supplied": len(self.markers_supplied),
            "marker_repeats": sum(1 for _, count in self.marker_occurrences if count > 1),
            "runs_truncated": self.runs_truncated,
        }

    def __repr__(self) -> str:
        counts = self.counts
        return ("SpanAudit(total=%d, retained=%d, thinking=%d, content=%d, "
                "tool_call=%d, tool_arguments=%d, control=%d, unknown=%d, "
                "contiguous=%s, truncated=%s)" % (
                    self.total_accepted, self.retained_count, counts[THINKING],
                    counts[CONTENT], counts[TOOL_CALL], counts[TOOL_ARGUMENTS],
                    counts[CONTROL_MARKERS], counts[UNKNOWN], self.contiguous,
                    self.message_truncated))


@dataclass(frozen=True, repr=False)
class SpanSelection:
    """One policy result over the accepted sequence.

    Fields are indices into the accepted sequence; ``end_index`` is exclusive.
    Iterating the record yields ``(start_index, end_index)`` so callers may write
    ``start, end = span_from_policy(audit, policy)``; ``contiguous`` and
    ``clamped`` must be read as attributes because a bare two-tuple cannot carry
    them.
    """

    policy: str
    start_index: int
    end_index: int
    contiguous: bool
    clamped: bool
    requested: int | None

    @property
    def bounds(self) -> Tuple[int, int]:
        return (self.start_index, self.end_index)

    @property
    def length(self) -> int:
        return self.end_index - self.start_index

    def __iter__(self):
        yield self.start_index
        yield self.end_index

    def __repr__(self) -> str:
        return "SpanSelection(%s, %d..%d, contiguous=%s, clamped=%s)" % (
            self.policy, self.start_index, self.end_index, self.contiguous,
            self.clamped)


@dataclass(frozen=True)
class _MarkerHit:
    """First non-overlapping occurrence of one marker kind."""

    start: int
    end: int
    occurrences: int


def _matches(ids: Sequence[int], start: int, marker: Tuple[int, ...]) -> bool:
    for offset, value in enumerate(marker):
        if ids[start + offset] != value:
            return False
    return True


def _find(ids: Sequence[int], count: int, marker: Tuple[int, ...] | None) -> _MarkerHit | None:
    """First exact token-id run plus the number of non-overlapping repeats.

    O(count * len(marker)) time, no slice copies: the first id is compared before
    the remaining ids so ordinary positions cost one comparison.
    """
    if marker is None:
        return None
    width = len(marker)
    if width > count:
        return None
    first = marker[0]
    limit = count - width
    start = end = -1
    occurrences = 0
    index = 0
    while index <= limit:
        if ids[index] == first and _matches(ids, index, marker):
            occurrences += 1
            if start < 0:
                start, end = index, index + width
            index += width
        else:
            index += 1
    if start < 0:
        return None
    return _MarkerHit(start, end, occurrences)


def _size(value) -> int:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise SpanContractError("lineage")
    try:
        return len(value)
    except TypeError:
        raise SpanContractError("lineage") from None


def _check_ids(token_ids, count: int) -> None:
    for index in range(count):
        value = token_ids[index]
        if type(value) is not int or value < 0 or value > MAX_MARKER_ID:
            raise SpanContractError("lineage")


def _check_positions(positions, count: int) -> bool:
    """Validate positions and report whether the window is unit-stride."""
    contiguous = True
    previous = -1
    for index in range(count):
        value = positions[index]
        if type(value) is not int or value < 0 or value > MAX_MARKER_ID:
            raise SpanContractError("lineage")
        if previous >= 0:
            if value <= previous:
                raise SpanContractError("lineage")
            if value != previous + 1:
                contiguous = False
        previous = value
    return contiguous


def _boundary_checks(thinking: _MarkerHit | None, begin: _MarkerHit | None,
                     finish: _MarkerHit | None, message: _MarkerHit | None,
                     alternate: _MarkerHit | None = None) -> None:
    """Reject marker layouts that cannot describe one assistant message.

    ``alternate`` is the optional second reasoning terminator (doc
    SOURCE-SPAN-SELECTION.md 4.3): reasoning may end at ``thinking_end`` or at
    ``tool_call_start``. It is normally the same marker as ``begin``, so the
    begin-ordering checks below deliberately allow that equality.
    """
    if finish is not None and begin is None:
        raise SpanContractError("lineage")
    if begin is not None:
        if thinking is not None and begin.start < thinking.end:
            _decline()
        if finish is not None and finish.start < begin.end:
            _decline()
    if thinking is not None and message is not None and message.start < thinking.end:
        _decline()
    if begin is not None and message is not None and message.start < begin.end:
        _decline()
    if finish is not None and message is not None and message.start < finish.end:
        _decline()
    if alternate is not None:
        if thinking is not None and alternate.start < thinking.end:
            _decline()  # a second terminator inside the reasoning head
        if begin is not None and alternate.start > begin.start:
            _decline()  # a second terminator inside the tool region
        if message is not None and message.start < alternate.end:
            _decline()


def _decline():
    raise SpanContractError("lineage")


def _run_of(kind: str, start: int, end: int, positions) -> SpanRun:
    first = positions[start]
    last = positions[end - 1]
    return SpanRun(kind=kind, start_index=start, end_index=end, start_position=first,
                   end_position=last + 1, contiguous=(last - first) == (end - start - 1))


def classify_accepted(token_ids: Sequence[int], positions: Sequence[int],
                      markers: SpanMarkers, *, accepted_total: int | None = None,
                      reasoning_end_ids: Sequence[int] | None = None) -> SpanAudit:
    """Classify a retained accepted window without a tokenizer.

    ``token_ids`` and ``positions`` are parallel sequences for the retained rows,
    in accepted order.  ``positions`` must be strictly increasing; a gap only
    clears ``contiguous`` and never invents rows.

    ``accepted_total`` is the whole accepted sequence length when the caller knows
    it (``WorkerRuntime.align`` does: ``record.accepted_count``).  When it is
    larger than the retained window, ``message_truncated`` is True because a
    marker before the window cannot have been observed.

    ``reasoning_end_ids`` is the optional second reasoning terminator: reasoning
    ends at the FIRST of ``thinking_end`` or this marker (the GLM template lets
    ``tool_call_start`` close a thinking block with no thinking-end at all).
    Rows before that first terminator are ``THINKING`` instead of ``UNKNOWN``.
    The marker does not join ``markers_supplied``/``marker_occurrences``, which
    keep describing the four ``SpanMarkers`` fields only.

    Raises :class:`SpanContractError` (``code == "lineage"``) on inconsistent
    input, contradictory markers or a bounded-resource breach.  Never guesses.
    """
    if type(markers) is not SpanMarkers:
        raise SpanContractError("lineage")
    count = _size(token_ids)
    if _size(positions) != count:
        raise SpanContractError("lineage")
    if count == 0:
        raise SpanContractError("lineage")
    if count > MAX_ACCEPTED:
        raise SpanContractError("quota")
    if accepted_total is None:
        total = count
    elif type(accepted_total) is int and count <= accepted_total <= MAX_ACCEPTED:
        total = accepted_total
    else:
        raise SpanContractError("lineage")
    _check_ids(token_ids, count)
    contiguous = _check_positions(positions, count)

    thinking = _find(token_ids, count, markers.thinking_end_ids)
    alternate = (_find(token_ids, count, _marker_ids(reasoning_end_ids))
                 if reasoning_end_ids is not None else None)
    begin = _find(token_ids, count, markers.tool_call_begin_ids)
    finish = _find(token_ids, count, markers.tool_call_end_ids)
    message = _find(token_ids, count, markers.message_end_ids)
    _boundary_checks(thinking, begin, finish, message, alternate)

    # The reasoning terminator is the FIRST of the two; rows before it are
    # THINKING even when no thinking-end marker occurs in the window.
    if alternate is not None and (thinking is None or alternate.start < thinking.start):
        terminator = alternate
    else:
        terminator = thinking

    flags = bytearray(count)
    for hit in (thinking, alternate, begin, finish, message):
        if hit is not None:
            for index in range(hit.start, hit.end):
                flags[index] = 1

    thinking_start = -1 if terminator is None else terminator.start
    thinking_end = -1 if terminator is None else terminator.end
    begin_end = -1 if begin is None else begin.end
    finish_end = -1 if finish is None else finish.end
    message_end = -1 if message is None else message.end

    counts = dict.fromkeys(CLASSES, 0)
    first_index = dict.fromkeys(CLASSES, -1)
    last_index = dict.fromkeys(CLASSES, -1)
    first_position = dict.fromkeys(CLASSES, -1)
    last_position = dict.fromkeys(CLASSES, -1)
    runs = []
    truncated_runs = False

    def push(kind, start, end):
        nonlocal truncated_runs
        if len(runs) < MAX_RUNS:
            runs.append(_run_of(kind, start, end, positions))
        else:
            truncated_runs = True

    current = None
    run_start = 0
    for index in range(count):
        if flags[index]:
            kind = CONTROL_MARKERS
        elif message_end >= 0 and index >= message_end:
            kind = UNKNOWN
        elif begin_end >= 0 and index >= begin_end:
            kind = TOOL_ARGUMENTS if (finish_end >= 0 and index >= finish_end) else TOOL_CALL
        elif thinking_start >= 0 and index < thinking_start:
            kind = THINKING
        elif thinking_end >= 0 and index >= thinking_end:
            kind = CONTENT
        else:
            kind = UNKNOWN
        counts[kind] += 1
        if first_index[kind] < 0:
            first_index[kind] = index
            first_position[kind] = positions[index]
        last_index[kind] = index
        last_position[kind] = positions[index]
        if kind != current:
            if current is not None:
                push(current, run_start, index)
            current, run_start = kind, index
    push(current, run_start, count)

    spans = tuple(ClassSpan(kind=kind, count=counts[kind], first_index=first_index[kind],
                            last_index=last_index[kind], first_position=first_position[kind],
                            last_position=last_position[kind]) for kind in CLASSES)
    occurrences = []
    for name, hit in zip(_MARKER_FIELDS, (thinking, begin, finish, message)):
        if getattr(markers, name) is not None:
            occurrences.append((name, 0 if hit is None else hit.occurrences))

    return SpanAudit(
        total_accepted=total,
        retained_count=count,
        window_start_index=0,
        window_end_index=count,
        selected_positions=(positions[0], positions[count - 1] + 1),
        spans=spans,
        runs=tuple(runs),
        runs_truncated=truncated_runs,
        markers_supplied=markers.supplied,
        marker_occurrences=tuple(occurrences),
        contains_final_content=counts[CONTENT] > 0,
        contains_tool_call=(counts[TOOL_CALL] + counts[TOOL_ARGUMENTS]) > 0,
        message_truncated=total > count,
        contiguous=contiguous,
        tool_call_complete=(begin is None or finish is not None),
    )


def span_from_policy(audit: SpanAudit, policy: str) -> SpanSelection | None:
    """Map a span policy onto accepted-sequence indices.

    Returns None when the policy cannot be satisfied: no content run exists, the
    turn carries tool-call syntax under ``final_content_no_tool``, or the run list
    overflowed so the last run is unknown.  ``SpanSelection.contiguous`` is False
    whenever the selected rows are not one absolute position apart; a selection
    never claims contiguity it cannot prove.
    """
    if type(audit) is not SpanAudit:
        raise SpanContractError("lineage")
    if policy not in POLICIES:
        raise SpanContractError("lineage")
    if policy == POLICY_LAST_ACCEPTED:
        if audit.retained_count < 1:
            return None
        return SpanSelection(policy=policy, start_index=0, end_index=audit.retained_count,
                             contiguous=audit.contiguous, clamped=False, requested=None)
    run = audit.last_run(CONTENT)
    if run is None:
        return None
    if policy == POLICY_FINAL_CONTENT_NO_TOOL and audit.contains_tool_call:
        return None
    requested = _TAIL_LIMITS.get(policy)
    start = run.start_index
    if requested is not None:
        start = max(run.start_index, run.end_index - requested)
    clamped = requested is not None and run.length < requested
    return SpanSelection(policy=policy, start_index=start, end_index=run.end_index,
                         contiguous=run.contiguous, clamped=clamped, requested=requested)
