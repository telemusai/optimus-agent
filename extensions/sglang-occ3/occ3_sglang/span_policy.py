"""The content_tail_v2 span-policy family: R0-R13 selection, fail-closed.

Why this module exists
----------------------
``span_audit`` classifies a retained window and ``span_markers`` attests marker
ids, but nothing selected a span: the worker captures the last N accepted rows
under the single constant policy ``whole_completed_response_tail_v1``.  This
module implements the versioned ``content_tail_v2`` family of
``reports/occ3-alignment-repair/SOURCE-SPAN-SELECTION.md`` section 4: rules
R0-R13, in order, first failure wins, over the retained window the worker can
actually prove.  It never falls back to ``last_accepted`` (R13) and never
invents a marker (R0): every decline is a ``LatentError`` with the existing
``"profile"``/``"lineage"`` codes.

Coordinate convention
---------------------
The caller supplies the retained window rows (``token_ids``/``positions``, in
position order, possibly with gaps) plus ``position_base``: the absolute
sequence position of accepted index 0, i.e. ``len(prompt) - 1`` under the
predictor convention (a row at absolute position p predicts the accepted token
at p+1; the constant offset is -1).  Accepted-index bounds are derived as
``positions[i] - position_base``; the receipt's ``accepted_span_start/_end``
stay accepted-index bounds of the ALIGNED span, exactly like the v1 receipt.

Rules (first failure wins; codes are the shared LatentError vocabulary)
-----------------------------------------------------------------------
R0  attestation   no marker table for the profile, or a required role
                  unresolved, or a bad stop-id set                    profile
R1  bounds        fewer than 2 or more than 128 retained rows, ragged
                  rows, non-integer ids, non-increasing positions    lineage
R2  window        the window does not end at the finish boundary
                  identity       (``positions[-1] + 1 - base == accepted_total``),
                  or the aligned span is not exactly as wide as its
                  selected rows (a gap inside the selection)          lineage
R3  turn end      the last retained row's id is not in the stop set  lineage
R4  reasoning     neither thinking-end nor tool-call-start occurs in
                  terminator     the window                          lineage
R5  reasoning     rows before the first terminator are REASONING, the
                  region         terminator being thinking-end OR
                  tool-call-start (reasoning_parser.py:936,165-177)
R6  tool-call     tool-call-start occurs without tool-call-end       lineage
                  completeness
R7  content       the maximal CONTENT run after the terminator, before
                  region         the first tool-call-start, stopping at the
                  stop row (a CONTROL row must bound the run)
R8  non-empty     that run is empty                                 lineage
R9  no-tool       content_tail_no_tool_v2 while the window holds any
                  policy          tool syntax                        lineage
R10 marker inside any marker sequence overlaps the selected run at any
    selection     offset (a partial marker at a run edge included)    lineage
R11 cross-check   req.reasoning_tokens is available and disagrees with
                  the R5 boundary (thinking_end terminator only;
                  a tool-start-terminated block never reaches here:
                  R8 declines it first)                            lineage
R12 bounded       the run list overflowed, or the fixed tail limit is
                  resources      unattainable within the window     lineage
R13 default       any rule that cannot be evaluated                lineage
                  (never a fallback to last_accepted)

The marker wire
---------------
Marker ids never travel in the worker descriptor (its leaves are
str/int/float only).  They ride the private per-request channel
``sampling_params.custom_params`` exactly like ``__sglang_reasoning_end_token_ids``
(bounded to 32 ids per role).  :func:`wire_marker_entries` builds the channel
entries from an attested table on the tokenizer-manager side;
:func:`marker_table_from_wire` rebuilds the table fail-closed on the worker
side.  The wire carries profile, schema and roundtrip attestation, never
marker text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Sequence, Tuple

from .attachment_contract import (MARKER_SCHEMA_V1, SPAN_POLICIES_V2, SPAN_SCHEMA_V1,
                                  SOURCE_SCHEMA_V2)
from .latent_types import LatentError
from .span_audit import (CLASSES, CONTENT, MAX_ACCEPTED, MAX_MARKER_ID,
                          MAX_MARKER_TOKENS, SpanContractError, _check_positions,
                          _find, _matches, classify_accepted)
from .span_markers import (PROFILES, ROLES, MarkerTable, resolve_marker_table,
                           resolve_stop_ids)

__all__ = [
    "POLICY_NO_TOOL", "POLICY_CONTENT", "POLICY_LAST64", "POLICY_LAST32",
    "TAIL_LIMITS", "TERMINATOR_THINKING_END", "TERMINATOR_TOOL_CALL_START",
    "TERMINATOR_NONE", "MARKER_PARAM_PREFIX", "SPAN_SOURCE_SCHEMA",
    "SpanSelectionV2", "select_span_v2", "span_block",
    "wire_marker_entries", "marker_table_from_wire", "resolve_runtime_markers",
]

POLICY_NO_TOOL = "content_tail_no_tool_v2"
POLICY_CONTENT = "content_tail_v2"
POLICY_LAST64 = "content_tail_last64_v2"
POLICY_LAST32 = "content_tail_last32_v2"

#: Fixed tail limits of the last-N members (doc section 4.4).
TAIL_LIMITS = {POLICY_LAST64: 64, POLICY_LAST32: 32}

TERMINATOR_THINKING_END = "thinking_end"
TERMINATOR_TOOL_CALL_START = "tool_call_start"
TERMINATOR_NONE = "none"

SPAN_SOURCE_SCHEMA = SOURCE_SCHEMA_V2

#: The private custom_params channel keys (doc sections 3.2, 6.1).
MARKER_PARAM_PREFIX = "__optimus_occ3_span_marker_"
PROFILE_PARAM = MARKER_PARAM_PREFIX + "profile"
SCHEMA_PARAM = MARKER_PARAM_PREFIX + "schema"
ROUNDTRIP_PARAM = MARKER_PARAM_PREFIX + "roundtrip"
STOP_PARAM = "__optimus_occ3_span_stop_ids"
#: The only role a profile may declare absent; the wire then omits its key.
_ABSENTABLE_ROLE = "thinking_start"
_NON_TURN_END_ROLES = tuple(role for role in ROLES if role != "turn_end")


class SpanPolicyError(SpanContractError):
    """A decline of the v2 family; ``code`` is "profile" or "lineage"."""


def _decline(code: str = "lineage"):
    raise SpanPolicyError(code)


def _require(condition: bool, code: str = "lineage") -> None:
    if not condition:
        _decline(code)


def _marker_tuple(value, role: str) -> Tuple[int, ...] | None:
    """Normalise one wire marker to a bounded tuple of plain ints, or None."""
    if value is None:
        return None
    _require(isinstance(value, (list, tuple)), "profile")
    items = tuple(value)
    _require(0 < len(items) <= MAX_MARKER_TOKENS, "profile")
    for item in items:
        _require(type(item) is int and 0 <= item <= MAX_MARKER_ID, "profile")
    return items


def _check_table(marker_table) -> None:
    """R0: a table for a reviewed profile with every required role resolved."""
    _require(type(marker_table) is MarkerTable, "profile")
    _require(marker_table.profile in PROFILES and marker_table.schema == MARKER_SCHEMA_V1,
             "profile")
    for role in _NON_TURN_END_ROLES:
        ids = marker_table.by_role()[role]
        _require(ids is not None or role == _ABSENTABLE_ROLE, "profile")
    _require(marker_table.turn_end_ids is not None, "profile")


def _check_stop_ids(stop_ids) -> FrozenSet[int]:
    """R0: the stop set is a non-empty bounded set of plain ids."""
    _require(type(stop_ids) is frozenset and 0 < len(stop_ids) <= MAX_MARKER_TOKENS,
             "profile")
    for item in stop_ids:
        _require(type(item) is int and 0 <= item <= MAX_MARKER_ID, "profile")
    return stop_ids


def _scan_all(ids: Sequence[int], count: int, marker: Tuple[int, ...] | None) -> Tuple[Tuple[int, int], ...]:
    """Every non-overlapping occurrence of an exact marker sequence."""
    if marker is None:
        return ()
    width = len(marker)
    if width > count:
        return ()
    found = []
    index = 0
    limit = count - width
    while index <= limit:
        if _matches(ids, index, marker):
            found.append((index, index + width))
            index += width
        else:
            index += 1
    return tuple(found)


@dataclass(frozen=True, repr=False)
class SpanSelectionV2:
    """One v2 policy result over the retained window; receipt-safe scalars only.

    ``start_index``/``end_index`` are window indices of the ALIGNED span (end
    exclusive).  ``accepted_span_start``/``accepted_span_end`` are its
    accepted-index bounds - the v2 receipt carries them unchanged in the v1
    fields.  ``counts`` follows :data:`span_audit.CLASSES` order.
    """

    policy: str
    terminator: str
    start_index: int
    end_index: int
    accepted_span_start: int
    accepted_span_end: int
    window_start_index: int
    window_end_index: int
    window_start_position: int
    window_end_position: int
    counts: Tuple[int, ...]
    contiguous: bool
    message_truncated: bool
    contains_final_content: bool
    contains_tool_call: bool
    clamped: bool
    requested: int | None

    @property
    def retained_rows(self) -> int:
        return self.end_index - self.start_index

    def span_block(self, marker_table: MarkerTable) -> Dict[str, object]:
        """The bounded occ3.span.v1 receipt block (doc section 6.2)."""
        _check_table(marker_table)
        return {
            "span_schema": SPAN_SCHEMA_V1,
            "span_policy": self.policy,
            "span_marker_profile": marker_table.profile,
            "span_marker_schema": marker_table.schema,
            "span_reasoning_terminator": self.terminator,
            "span_turn_end_attested": True,  # R3 fired before this record exists
            "span_thinking_count": self.counts[CLASSES.index("THINKING")],
            "span_content_count": self.counts[CLASSES.index("CONTENT")],
            "span_tool_call_count": self.counts[CLASSES.index("TOOL_CALL")],
            "span_tool_arguments_count": self.counts[CLASSES.index("TOOL_ARGUMENTS")],
            "span_control_count": self.counts[CLASSES.index("CONTROL_MARKERS")],
            "span_unknown_count": self.counts[CLASSES.index("UNKNOWN")],
            "span_window_start_position": self.window_start_position,
            "span_window_end_position": self.window_end_position,
            "span_contiguous": self.contiguous,
            "span_contains_final_content": self.contains_final_content,
            "span_contains_tool_call": self.contains_tool_call,
            "span_message_truncated": self.message_truncated,
        }


def select_span_v2(token_ids: Sequence[int], positions: Sequence[int], *, position_base: int,
                   accepted_total: int, policy: str, marker_table: MarkerTable,
                   stop_ids: FrozenSet[int], reasoning_tokens: int | None = None) -> SpanSelectionV2:
    """Select the v2 span over one retained window, or decline (R0-R13).

    ``token_ids``/``positions`` are the retained window rows in position
    order.  ``position_base`` is the absolute position of accepted index 0
    (``len(prompt) - 1``).  ``accepted_total`` is the whole accepted count of
    the turn.  ``reasoning_tokens`` is the scheduler's own boundary count when
    available (R11 cross-check).
    """
    # R0: attestation.
    if policy not in SPAN_POLICIES_V2:
        _decline("profile")
    _check_table(marker_table)
    _check_stop_ids(stop_ids)
    _require(type(position_base) is int and 0 <= position_base <= MAX_MARKER_ID, "profile")
    _require(type(accepted_total) is int and 0 < accepted_total <= MAX_ACCEPTED, "lineage")

    # R1: bounds (span_audit's own validators raise SpanContractError "lineage").
    # The retained window is the union of the v1 ring and the content buffer, so
    # it can legitimately hold more rows than the negotiated ring cap; the
    # ALIGNED span is what must fit the cap, checked after selection below.
    count = len(token_ids)
    _require(len(positions) == count and 2 <= count <= MAX_ACCEPTED, "lineage")
    for index in range(count):
        value = token_ids[index]
        _require(type(value) is int and 0 <= value <= MAX_MARKER_ID, "lineage")
    contiguous = _check_positions(positions, count)

    # R2 (window identity, first half): the window ends at the finish boundary
    # and starts inside the accepted sequence.
    window_start_position = positions[0]
    window_end_position = positions[count - 1] + 1
    window_start_index = window_start_position - position_base
    window_end_index = window_end_position - position_base
    _require(0 <= window_start_index, "lineage")
    _require(window_end_index == accepted_total, "lineage")

    # R3: turn end.
    _require(token_ids[count - 1] in stop_ids, "lineage")

    # R4/R6: the terminator and tool-call completeness need first occurrences.
    markers = marker_table.span_markers()
    by_role = marker_table.by_role()
    thinking = _find(token_ids, count, by_role["thinking_end"])
    tool_start = _find(token_ids, count, by_role["tool_call_start"])
    tool_end = _find(token_ids, count, by_role["tool_call_end"])
    _require(thinking is not None or tool_start is not None, "lineage")
    _require(tool_start is None or tool_end is not None, "lineage")
    if thinking is not None and (tool_start is None or thinking.start < tool_start.start):
        terminator, terminator_kind = thinking, TERMINATOR_THINKING_END
    else:
        terminator, terminator_kind = tool_start, TERMINATOR_TOOL_CALL_START

    # R5/R7: classification with the second reasoning terminator.
    audit = classify_accepted(token_ids, positions, markers, accepted_total=accepted_total,
                             reasoning_end_ids=by_role["tool_call_start"])

    # R12 (first half): a truncated run list cannot answer "the last content run".
    _require(not audit.runs_truncated, "lineage")
    run = audit.last_run(CONTENT)
    # R8: no content run exists; v2 declines, it never falls back.
    _require(run is not None, "lineage")
    # R7: the run is the region after the R5 boundary, before the first
    # tool-call-start, and stops at a marker row (tool-call-start or the
    # stop row) rather than at the window edge.
    _require(run.start_index == terminator.end, "lineage")
    _require(tool_start is None or run.end_index <= tool_start.start, "lineage")
    _require(run.end_index < count, "lineage")

    # R9: the no-tool policy refuses any tool syntax anywhere in the window.
    if policy == POLICY_NO_TOOL:
        tool_syntax = audit.contains_tool_call
        for role in ("tool_call_start", "tool_call_end", "argument_key_start",
                     "argument_key_end", "argument_value_start", "argument_value_end"):
            if _scan_all(token_ids, count, by_role[role]):
                tool_syntax = True
        _require(not tool_syntax, "lineage")

    # R10: no marker sequence may overlap the selected run at any offset.
    for role in ROLES:
        for start, end in _scan_all(token_ids, count, by_role[role]):
            _require(start >= run.end_index or end <= run.start_index, "lineage")

    # R11: two independent measurements of the reasoning boundary.
    if reasoning_tokens is not None and terminator_kind == TERMINATOR_THINKING_END:
        _require(type(reasoning_tokens) is int and reasoning_tokens >= 0, "lineage")
        _require(reasoning_tokens == window_start_index + terminator.end, "lineage")

    # R12 (second half): an unattainable tail limit is refused, not clamped-to-nothing.
    requested = TAIL_LIMITS.get(policy)
    _require(requested is None or requested <= count, "lineage")

    start = run.start_index
    clamped = False
    if requested is not None:
        start = max(run.start_index, run.end_index - requested)
        # Clamped means the fixed tail limit actually cut rows off the run,
        # never that the run was shorter than the limit.
        clamped = run.length > requested

    # R2 (window identity, second half): the aligned span is exactly as wide
    # as its rows - no gap inside the selection.
    _require(positions[start] - position_base >= 0, "lineage")
    _require((run.end_index - start)
             == (positions[run.end_index - 1] + 1) - positions[start], "lineage")
    # Receipt bounds: the aligned span fits the negotiated ring cap, and every
    # class count fits the v2 receipt block's u32 bound (128 per class).
    _require(2 <= run.end_index - start <= 128, "lineage")
    for value in audit.counts.values():
        _require(value <= 128, "lineage")

    return SpanSelectionV2(
        policy=policy, terminator=terminator_kind,
        start_index=start, end_index=run.end_index,
        accepted_span_start=positions[start] - position_base,
        accepted_span_end=positions[run.end_index - 1] + 1 - position_base,
        window_start_index=window_start_index, window_end_index=window_end_index,
        window_start_position=window_start_position,
        window_end_position=window_end_position,
        counts=tuple(audit.counts[kind] for kind in CLASSES),
        contiguous=contiguous, message_truncated=audit.message_truncated,
        contains_final_content=audit.contains_final_content,
        contains_tool_call=audit.contains_tool_call,
        clamped=clamped, requested=requested)


def span_block(selection: SpanSelectionV2, marker_table: MarkerTable) -> Dict[str, object]:
    """Module-level alias of :meth:`SpanSelectionV2.span_block`."""
    return selection.span_block(marker_table)


def wire_marker_entries(marker_table: MarkerTable, stop_ids: FrozenSet[int]) -> Dict[str, object]:
    """The private custom_params channel entries for one marker table.

    One bounded int list per role, plus profile, schema and the round-trip
    attestation flag.  Marker text never enters the channel.  The turn-end
    role is a marker, not the stop set: the stop set travels separately because
    a GLM turn can end on more than one id.
    """
    _check_table(marker_table)
    _check_stop_ids(stop_ids)
    entries = {PROFILE_PARAM: marker_table.profile,
               SCHEMA_PARAM: marker_table.schema,
               ROUNDTRIP_PARAM: int(bool(marker_table.roundtrip_attested)),
               STOP_PARAM: sorted(stop_ids)}
    for role in ROLES:
        ids = marker_table.by_role()[role]
        if ids is not None:
            entries[MARKER_PARAM_PREFIX + role] = list(ids)
    return entries


def marker_table_from_wire(custom_params) -> Tuple[MarkerTable, FrozenSet[int]]:
    """Rebuild the marker table from the channel, fail-closed (worker side).

    A missing role key is tolerated only for the profile's absentable role.
    The rebuilt table is cross-checked with the marker module's own equality
    and prefix rules, so a wire that could reorder matching is refused.
    """
    from .span_markers import _check_table as _check_resolved
    _require(isinstance(custom_params, dict), "profile")
    profile = custom_params.get(PROFILE_PARAM)
    schema = custom_params.get(SCHEMA_PARAM)
    roundtrip = custom_params.get(ROUNDTRIP_PARAM)
    _require(type(profile) is str and profile in PROFILES, "profile")
    _require(schema == MARKER_SCHEMA_V1, "profile")
    _require(type(roundtrip) is int and roundtrip in (0, 1), "profile")
    stop_wire = custom_params.get(STOP_PARAM)
    _require(isinstance(stop_wire, (list, tuple)) and 0 < len(stop_wire) <= MAX_MARKER_TOKENS,
             "profile")
    stop_ids = frozenset(stop_wire)
    _check_stop_ids(stop_ids)
    values = {}
    for role in _NON_TURN_END_ROLES:
        ids = _marker_tuple(custom_params.get(MARKER_PARAM_PREFIX + role), role)
        _require(ids is not None or role == _ABSENTABLE_ROLE, "profile")
        values[role + "_ids"] = ids
    values["turn_end_ids"] = _marker_tuple(custom_params.get(MARKER_PARAM_PREFIX + "turn_end"),
                                           "turn_end")
    _require(values["turn_end_ids"] is not None, "profile")
    table = MarkerTable(profile=profile, schema=MARKER_SCHEMA_V1,
                        roundtrip_attested=bool(roundtrip), provenance=(),
                        **values)
    _check_resolved(table)
    return table, stop_ids


def resolve_runtime_markers(tokenizer):
    """Resolve and cache the marker table plus stop ids on a tokenizer.

    The cache lives on the tokenizer object itself, so the adapter, the
    backend and any other tokenizer-manager seam share one attested table per
    process.  ``thinking_start`` is allowlisted absent: the GLM template may
    prefill it and a window can legitimately contain no thinking-start row.
    Raises :class:`span_markers.MarkerResolutionError` (code ``profile``) when
    the tokenizer cannot attest the table.
    """
    cached = getattr(tokenizer, "_occ3_span_markers", None)
    if type(cached) is tuple and len(cached) == 2:
        return cached
    table = resolve_marker_table(tokenizer, allow_absent=(_ABSENTABLE_ROLE,))
    stop_ids = resolve_stop_ids(tokenizer)
    cached = (table, stop_ids)
    try:
        setattr(tokenizer, "_occ3_span_markers", cached)
    except Exception:
        pass
    return cached
