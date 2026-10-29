"""Offline tests for the tokenizer-free accepted-span auditor.

No SGLang, no Torch, no GPU, no tokenizer: the auditor compares integer ids.
"""
from __future__ import annotations

import dataclasses
import time

import pytest

from occ3_sglang.latent_types import LatentError
from occ3_sglang.span_audit import (
    CLASSES, CONTENT, CONTROL_MARKERS, MAX_MARKER_TOKENS, POLICIES, SUMMARY_KEYS,
    THINKING, TOOL_ARGUMENTS, TOOL_CALL, UNKNOWN, SpanAudit, SpanContractError,
    SpanMarkers, classify_accepted, span_from_policy,
)

THINK = 900        # thinking end
TOOL_BEGIN = 910   # tool-call syntax open
TOOL_END = 920     # tool-call syntax close
MSG_END = 990      # assistant message end


def full_markers(**changes):
    values = dict(thinking_end_ids=(THINK,), tool_call_begin_ids=(TOOL_BEGIN,),
                  tool_call_end_ids=(TOOL_END,), message_end_ids=(MSG_END,))
    values.update(changes)
    return SpanMarkers(**values)


def classify(ids, markers=None, positions=None, **kwargs):
    markers = full_markers() if markers is None else markers
    positions = list(range(len(ids))) if positions is None else positions
    return classify_accepted(ids, positions, markers, **kwargs)


def test_frozen_records_carry_no_ids():
    audit = classify([1, 2, THINK, 5, 6, MSG_END])
    assert isinstance(audit, SpanAudit)
    with pytest.raises(dataclasses.FrozenInstanceError):
        audit.total_accepted = 0
    markers = full_markers()
    with pytest.raises(dataclasses.FrozenInstanceError):
        markers.thinking_end_ids = (1,)
    assert "900" not in repr(markers) and "900" not in repr(audit)


def test_marker_at_index_zero():
    audit = classify([THINK, 1, 2, 3])
    assert audit.counts[THINKING] == 0
    assert audit.counts[CONTROL_MARKERS] == 1
    assert audit.counts[CONTENT] == 3
    assert audit.contains_final_content is True
    assert audit.index_range(CONTROL_MARKERS) == (0, 0)
    assert audit.position_range(CONTENT) == (1, 3)
    selection = span_from_policy(audit, "final_content")
    assert selection.bounds == (1, 4)
    assert selection.contiguous is True and selection.clamped is False


def test_marker_at_end():
    audit = classify([1, 2, 3, THINK])
    assert audit.counts[THINKING] == 3
    assert audit.counts[CONTENT] == 0
    assert audit.contains_final_content is False
    assert audit.index_range(CONTENT) == (-1, -1)
    assert span_from_policy(audit, "final_content") is None
    assert span_from_policy(audit, "final_content_last64") is None
    last = span_from_policy(audit, "last_accepted")
    assert last.bounds == (0, 4) and last.contiguous is True


def test_marker_absent_stays_unknown():
    audit = classify([1, 2, 3, 4])
    assert audit.counts[UNKNOWN] == 4
    assert set(audit.counts) == set(CLASSES)
    assert audit.contains_final_content is False
    assert audit.contains_tool_call is False
    assert audit.markers_supplied == ("thinking_end_ids", "tool_call_begin_ids",
                                      "tool_call_end_ids", "message_end_ids")
    assert ("thinking_end_ids", 0) in audit.marker_occurrences
    assert span_from_policy(audit, "final_content") is None
    assert span_from_policy(audit, "final_content_no_tool") is None


def test_marker_repeated():
    audit = classify([1, THINK, 2, 3, THINK, 4], full_markers(
        tool_call_begin_ids=None, tool_call_end_ids=None, message_end_ids=None))
    assert audit.counts[THINKING] == 1
    assert audit.counts[CONTROL_MARKERS] == 1
    assert audit.counts[CONTENT] == 4          # the repeat is classified by region
    assert ("thinking_end_ids", 2) in audit.marker_occurrences
    assert audit.summary()["marker_repeats"] == 1


def test_thinking_then_tool_call():
    ids = [11, 12, THINK, 21, 22, TOOL_BEGIN, 31, 32, TOOL_END, 41, 42, MSG_END]
    audit = classify(ids)
    assert audit.counts[THINKING] == 2
    assert audit.counts[CONTROL_MARKERS] == 4
    assert audit.counts[CONTENT] == 2
    assert audit.counts[TOOL_CALL] == 2
    assert audit.counts[TOOL_ARGUMENTS] == 2
    assert audit.counts[UNKNOWN] == 0
    assert audit.contains_tool_call is True and audit.tool_call_complete is True
    assert audit.index_range(TOOL_CALL) == (6, 7)
    assert audit.index_range(TOOL_ARGUMENTS) == (9, 10)
    assert span_from_policy(audit, "final_content_no_tool") is None
    mixed = span_from_policy(audit, "final_content")
    assert mixed.bounds == (3, 5) and mixed.contiguous is True


def test_tool_call_turn_has_no_final_content():
    audit = classify([11, THINK, TOOL_BEGIN, 31, TOOL_END, 41, MSG_END])
    assert audit.counts[CONTENT] == 0
    assert audit.contains_final_content is False
    assert audit.contains_tool_call is True
    assert span_from_policy(audit, "final_content") is None
    assert span_from_policy(audit, "final_content_no_tool") is None


def test_truncated_tool_call_is_reported_not_raised():
    audit = classify([11, THINK, TOOL_BEGIN, 31, 32])
    assert audit.tool_call_complete is False
    assert audit.counts[TOOL_CALL] == 2
    assert audit.counts[TOOL_ARGUMENTS] == 0
    assert audit.contains_tool_call is True


def test_content_only_selection():
    audit = classify([1, 2, THINK, 5, 6, 7, MSG_END])
    assert audit.counts[CONTENT] == 3
    assert audit.counts[UNKNOWN] == 0
    no_tool = span_from_policy(audit, "final_content_no_tool")
    assert no_tool.bounds == (3, 6)
    assert no_tool.contiguous is True and no_tool.clamped is False
    assert span_from_policy(audit, "last_accepted").bounds == (0, 7)


def test_tail_policies_clamp_to_short_content():
    ids = [THINK] + [5] * 10
    audit = classify(ids)
    for policy, wanted in (("final_content_last64", 64), ("final_content_last32", 32)):
        selection = span_from_policy(audit, policy)
        assert selection.bounds == (1, 11)
        assert selection.clamped is True
        assert selection.requested == wanted
        assert selection.length == 10


def test_tail_policies_clamp_to_long_content():
    ids = [THINK] + [5] * 100
    audit = classify(ids)
    last64 = span_from_policy(audit, "final_content_last64")
    assert last64.bounds == (37, 101)
    assert last64.clamped is False and last64.requested == 64
    last32 = span_from_policy(audit, "final_content_last32")
    assert last32.bounds == (69, 101)
    assert last32.clamped is False and last32.requested == 32
    exact = span_from_policy(classify([THINK] + [5] * 64), "final_content_last64")
    assert exact.bounds == (1, 65) and exact.clamped is False


def test_discontiguous_positions_never_claim_contiguity():
    audit = classify([1, 2, 3, THINK, 7, 8, 9], positions=[0, 1, 2, 3, 10, 11, 12])
    assert audit.contiguous is False
    assert audit.selected_positions == (0, 13)
    assert span_from_policy(audit, "last_accepted").contiguous is False
    inner = span_from_policy(audit, "final_content")
    assert inner.bounds == (4, 7)
    assert inner.contiguous is True          # this run is unit-stride
    gap = classify([THINK, 7, 8], positions=[0, 1, 3])
    assert span_from_policy(gap, "final_content").contiguous is False


def test_discontiguous_inner_selection_is_flagged():
    audit = classify([THINK, 7, 8], positions=[0, 1, 3])
    assert audit.contiguous is False
    assert audit.runs[-1].contiguous is False
    selection = span_from_policy(audit, "final_content_last64")
    assert selection.contiguous is False


def test_unknown_markers_classify_nothing():
    audit = classify([1, 2, 3, 4, 5], markers=SpanMarkers())
    assert audit.counts[UNKNOWN] == 5
    assert audit.markers_supplied == ()
    assert audit.contains_final_content is False
    assert audit.contains_tool_call is False
    assert audit.marker_occurrences == ()
    summary = audit.summary()
    assert summary["markers_supplied"] == 0 and summary["unknown"] == 5
    assert span_from_policy(audit, "final_content") is None
    # Empty sequences are "not supplied", never a match at every index.
    assert SpanMarkers(thinking_end_ids=[]) == SpanMarkers()
    assert classify([1, 2], markers=SpanMarkers(thinking_end_ids=[])).counts[UNKNOWN] == 2


def test_summary_is_a_fixed_int_bool_allowlist():
    ids = [123456789, 987654321, THINK, 55555555, 66666666, MSG_END]
    audit = classify(ids)
    summary = audit.summary()
    assert tuple(summary) == SUMMARY_KEYS
    assert set(summary) == set(SUMMARY_KEYS)
    for key, value in summary.items():
        assert type(value) in (int, bool), key
    values = set(summary.values())
    for token_id in ids:
        assert token_id not in values
    assert "123456789" not in repr(audit)
    assert "123456789" not in str(audit.summary())


@pytest.mark.parametrize("markers", [
    dict(tool_call_end_ids=(TOOL_END,)),                                   # end without begin
    dict(thinking_end_ids=(TOOL_BEGIN,), tool_call_begin_ids=(TOOL_BEGIN,)),  # tool before think end
    dict(message_end_ids=(THINK,)),                                        # message ends in thinking
    dict(tool_call_begin_ids=(910, 911), tool_call_end_ids=(911, 912)),   # overlapping pair
    dict(thinking_end_ids=(911,), tool_call_begin_ids=(910, 911)),        # overlap across kinds
])
def test_contradictory_markers_fail_closed(markers):
    ids = [11, 910, 911, THINK, 12, TOOL_END, MSG_END]
    with pytest.raises(SpanContractError) as caught:
        classify(ids, full_markers(**markers))
    assert caught.value.code == "lineage"
    assert isinstance(caught.value, LatentError) and isinstance(caught.value, ValueError)


@pytest.mark.parametrize("ids,positions,markers,kwargs", [
    ([1, 2], [0], full_markers(), {}),                       # length mismatch
    ([], [], full_markers(), {}),                           # empty window
    ([1, True], [0, 1], full_markers(), {}),                # bool is not a token id
    ([1, -2], [0, 1], full_markers(), {}),                  # negative id
    ([1, 2], [5, 5], full_markers(), {}),                   # positions not increasing
    ([1, 2], [3, 2], full_markers(), {}),                   # positions decreasing
    ([1, 2], [0, "1"], full_markers(), {}),                 # non-int position
    ([1, 2], [0, 1], full_markers(), {"accepted_total": 1}),  # total below window
    ([1, 2], [0, 1], full_markers(), {"accepted_total": True}),  # bool total
    ([1, 2], [0, 1], {"thinking_end_ids": (1,)}, {}),       # not a SpanMarkers
])
def test_invalid_input_fails_closed(ids, positions, markers, kwargs):
    with pytest.raises(SpanContractError) as caught:
        classify_accepted(ids, positions, markers, **kwargs)
    assert caught.value.code == "lineage"


@pytest.mark.parametrize("marker", [
    tuple(range(MAX_MARKER_TOKENS + 1)),   # marker longer than the bound
    "think",                               # decoded text is not a marker
    b"\x00\x01",                          # bytes are not ids
    (1, True),                             # bool is not a token id
    (-1,),                                 # negative id
    (1.5,),                                # non-int id
])
def test_invalid_markers_fail_closed(marker):
    with pytest.raises(SpanContractError) as caught:
        SpanMarkers(thinking_end_ids=marker)
    assert caught.value.code == "lineage"


def test_unknown_policy_fails_closed():
    audit = classify([1, THINK, 2])
    for policy in ("final_content_last128", "", None, "LAST_ACCEPTED"):
        with pytest.raises(SpanContractError):
            span_from_policy(audit, policy)
    with pytest.raises(SpanContractError):
        span_from_policy({"total_accepted": 1}, "last_accepted")


def test_accepted_total_marks_message_truncated():
    audit = classify([1, THINK, 2, 3], accepted_total=9)
    assert audit.total_accepted == 9 and audit.retained_count == 4
    assert audit.message_truncated is True
    assert audit.summary()["message_truncated"] is True
    whole = classify([1, THINK, 2, 3])
    assert whole.message_truncated is False


def test_marker_wire_and_complete_flags():
    markers = full_markers()
    assert markers.complete is True
    assert markers.wire() == {"thinking_end_ids": 1, "tool_call_begin_ids": 1,
                              "tool_call_end_ids": 1, "message_end_ids": 1}
    assert SpanMarkers(tool_call_begin_ids=(910,)).complete is False
    assert SpanMarkers(tool_call_end_ids=(920,)).complete is False


def test_policies_are_the_documented_set():
    assert POLICIES == ("last_accepted", "final_content", "final_content_no_tool",
                        "final_content_last64", "final_content_last32")
    audit = classify([1, THINK, 2])
    for policy in POLICIES:
        selection = span_from_policy(audit, policy)
        assert selection is None or selection.policy == policy


def test_module_is_pure_python():
    path = __import__("occ3_sglang.span_audit", fromlist=["__file__"]).__file__
    source = open(path, "r", encoding="utf-8").read()
    for forbidden in ("import torch", "import sglang", "from sglang", "open(",
                      "requests.", "socket"):
        assert forbidden not in source


def test_performance_guard_on_4096_ids():
    ids = [1000 + (index % 97) for index in range(4096)]
    ids[100] = THINK
    ids[2000] = TOOL_BEGIN
    ids[2500] = TOOL_END
    ids[4095] = MSG_END
    positions = list(range(4096))
    markers = full_markers()
    best = None
    for _ in range(3):
        start = time.perf_counter()
        audit = classify_accepted(ids, positions, markers)
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    assert audit.retained_count == 4096
    assert audit.counts[CONTROL_MARKERS] == 4
    assert best < 0.05, best


def test_marker_sequences_accept_array_and_tuple_inputs():
    from array import array
    ids = array("l", [1, THINK, 2, 3])
    audit = classify_accepted(ids, array("l", [0, 1, 2, 3]), full_markers(
        tool_call_begin_ids=None, tool_call_end_ids=None, message_end_ids=None))
    assert audit.counts[CONTENT] == 2
    assert audit.counts[THINKING] == 1
    tuple_audit = classify((1, THINK, 2, 3), full_markers(
        tool_call_begin_ids=None, tool_call_end_ids=None, message_end_ids=None),
        positions=(0, 1, 2, 3))
    assert tuple_audit.summary() == audit.summary()


def test_overflowed_run_list_fails_closed_for_policies():
    audit = classify([1, THINK, 2, 3])
    assert span_from_policy(audit, "final_content").bounds == (2, 4)
    forced = dataclasses.replace(audit, runs_truncated=True)
    assert forced.last_run(CONTENT) is None
    assert span_from_policy(forced, "final_content") is None
    assert span_from_policy(forced, "last_accepted").bounds == (0, 4)


def test_counts_follow_the_fixed_class_order():
    audit = classify([1, THINK, 2])
    assert tuple(audit.counts) == CLASSES
    assert tuple(span.kind for span in audit.spans) == CLASSES
    assert [span.count for span in audit.spans] == [1, 1, 0, 0, 1, 0]


def test_alternate_reasoning_terminator_closes_thinking_at_tool_start():
    # The GLM template lets tool_call_start close a thinking block with no
    # thinking-end at all: the first of the two terminators wins.
    audit = classify([1, 2, TOOL_BEGIN, 4, TOOL_END, MSG_END], reasoning_end_ids=(TOOL_BEGIN,))
    assert audit.counts[THINKING] == 2
    assert audit.counts[CONTENT] == 0
    assert audit.contains_tool_call is True
    # The window still holds no thinking-end occurrence.
    assert audit.marker_occurrences[0] == ("thinking_end_ids", 0)


def test_alternate_terminator_after_a_real_thinking_end_loses():
    audit = classify([1, THINK, 3, TOOL_BEGIN, 4, TOOL_END, MSG_END], reasoning_end_ids=(TOOL_BEGIN,))
    assert audit.counts[THINKING] == 1
    assert audit.counts[CONTENT] == 1
    assert audit.marker_occurrences[0] == ("thinking_end_ids", 1)


def test_alternate_terminator_never_joins_marker_occurrences():
    audit = classify([1, 2, TOOL_BEGIN, 4, TOOL_END, MSG_END], reasoning_end_ids=(TOOL_BEGIN,))
    assert [name for name, _ in audit.marker_occurrences] == [
        "thinking_end_ids", "tool_call_begin_ids", "tool_call_end_ids", "message_end_ids"]
    assert audit.markers_supplied == ("thinking_end_ids", "tool_call_begin_ids",
                                      "tool_call_end_ids", "message_end_ids")


def test_alternate_terminator_inside_the_reasoning_head_declines():
    with pytest.raises(SpanContractError):
        classify([TOOL_BEGIN, 1, 2, THINK, 5, MSG_END], reasoning_end_ids=(TOOL_BEGIN,))


def test_alternate_terminator_inside_the_tool_region_declines():
    with pytest.raises(SpanContractError):
        classify([1, THINK, 3, TOOL_BEGIN, 4, TOOL_END, MSG_END], reasoning_end_ids=(TOOL_END,))


def test_alternate_terminator_overlapping_the_message_end_declines():
    with pytest.raises(SpanContractError):
        classify([1, 2, 3, MSG_END], reasoning_end_ids=(MSG_END,))


def test_selection_unpacks_as_two_bounds():
    audit = classify([1, THINK, 2, 3])
    start, end = span_from_policy(audit, "final_content")
    assert (start, end) == (2, 4)
    selection = span_from_policy(audit, "last_accepted")
    assert selection.bounds == (0, 4) and selection.length == 4
