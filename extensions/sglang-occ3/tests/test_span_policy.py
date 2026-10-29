"""Offline tests for the content_tail_v2 selection rules R0-R13.

No SGLang, no Torch, no GPU, no tokenizer ids beyond the fake marker set. The
selector compares integers; declines are LatentError "profile"/"lineage" and
never fall back to last_accepted (R13).
"""
from __future__ import annotations

import pytest

from occ3_sglang.attachment_contract import MARKER_SCHEMA_V1, SOURCE_SCHEMA_V2, SPAN_SCHEMA_V1
from occ3_sglang.latent_types import LatentError
from occ3_sglang.span_markers import DEFAULT_PROFILE, MarkerTable
from occ3_sglang.span_policy import (
    POLICY_CONTENT, POLICY_LAST32, POLICY_LAST64, POLICY_NO_TOOL,
    PROFILE_PARAM, ROUNDTRIP_PARAM, SCHEMA_PARAM, STOP_PARAM,
    TERMINATOR_THINKING_END, MARKER_PARAM_PREFIX, SpanPolicyError,
    marker_table_from_wire, select_span_v2, wire_marker_entries,
)

T_START, THINK = 100, 101
TC_START, TC_END = 102, 103
AK_START, AK_END = 104, 105
AV_START, AV_END = 106, 107
EOS = 108
STOP = frozenset((EOS,))

#: The GLM-attested single-token spellings of test_span_markers.ADDED.
BASE = {T_START: "thinking_start", THINK: "thinking_end", TC_START: "tool_call_start",
        TC_END: "tool_call_end", AK_START: "argument_key_start", AK_END: "argument_key_end",
        AV_START: "argument_value_start", AV_END: "argument_value_end", EOS: "turn_end"}


def table(**changes):
    ids = {"thinking_start": (T_START,), "thinking_end": (THINK,),
           "tool_call_start": (TC_START,), "tool_call_end": (TC_END,),
           "argument_key_start": (AK_START,), "argument_key_end": (AK_END,),
           "argument_value_start": (AV_START,), "argument_value_end": (AV_END,),
           "turn_end": (EOS,)}
    ids.update(changes)
    values = {role + "_ids": seq for role, seq in ids.items()}
    return MarkerTable(profile=DEFAULT_PROFILE, schema=MARKER_SCHEMA_V1,
                       roundtrip_attested=True, provenance=(), **values)


def window(tokens, base=9, total=None):
    return list(tokens), [base + i for i in range(len(tokens))], base, total


def declines(case, code="lineage", **kwargs):
    kwargs.setdefault("policy", POLICY_CONTENT)
    kwargs.setdefault("marker_table", table())
    kwargs.setdefault("stop_ids", STOP)
    tokens, positions, base, total = case
    # span_audit's own validators raise the parent SpanContractError; worker
    # seams see either as a LatentError with the shared code.
    with pytest.raises(LatentError) as caught:
        select_span_v2(tokens, positions, position_base=base,
                       accepted_total=total if total is not None else len(tokens), **kwargs)
    assert caught.value.code == code
    assert isinstance(caught.value, (SpanPolicyError,)) or type(caught.value).__name__ == "SpanContractError"


def selects(case, **kwargs):
    kwargs.setdefault("policy", POLICY_CONTENT)
    kwargs.setdefault("marker_table", table())
    kwargs.setdefault("stop_ids", STOP)
    tokens, positions, base, total = case
    return select_span_v2(tokens, positions, position_base=base,
                          accepted_total=total if total is not None else len(tokens), **kwargs)


def test_r0_unadmitted_policy_and_bad_attestation_decline_profile():
    good = window([1, THINK, 2, EOS])
    declines(good, code="profile", policy="whole_completed_response_tail_v1")
    declines(good, code="profile", policy="guessed_content_tail_v2")
    # R0: an empty stop set cannot attest a turn end.
    declines(good, code="profile", marker_table=table(), stop_ids=frozenset())


def test_r1_bounds_ragged_and_order_decline_lineage():
    declines(window([THINK, EOS]), code="lineage")                      # 1 row
    declines(window([]), code="lineage")                                # 0 rows
    declines(([THINK, 2, EOS], [10, 10, 11], 9, 3), code="lineage")     # not increasing
    declines(([THINK, 2, EOS], [10, 11], 9, 3), code="lineage")          # ragged
    declines(([THINK, 2**63], [10, 11, 12], 9, 3), code="lineage")      # id out of bounds
    declines((["x", 2, EOS], [10, 11, 12], 9, 3), code="lineage")       # non-int id


def test_r2_window_must_end_at_the_finish_boundary():
    # The window is short of the finish boundary: the last accepted row is missing.
    declines(([1, THINK, 2, EOS], [9, 10, 11, 12], 9, 5), code="lineage")
    # A gap inside the selection is not silently accepted.
    declines(([1, THINK, 2, 3, EOS], [9, 10, 11, 13, 14], 9, 6), code="lineage")


def test_r3_turn_end_must_be_a_stop_row():
    declines(([1, THINK, 2, 999], [9, 10, 11, 12], 9, 4), code="lineage")


def test_r4_window_without_a_terminator_declines():
    declines(([1, 2, EOS], [9, 10, 11], 9, 3), code="lineage")


def test_r6_incomplete_tool_call_declines():
    declines(([THINK, 1, TC_START, 2, EOS], [9, 10, 11, 12, 13], 9, 5), code="lineage")


def test_r7_content_run_must_start_at_the_terminator_and_stop_at_a_marker():
    # An argument delimiter between the terminator and the run breaks adjacency.
    declines(([THINK, AK_START, 1, EOS], [9, 10, 11, 12], 9, 4), code="lineage")


def test_r8_no_content_run_declines_and_never_falls_back():
    # Tool-call-start-terminated block: reasoning closes at the tool marker and
    # the content run is empty - R8 declines, it never selects the tool tail.
    declines(([1, THINK, TC_START, 4, TC_END, EOS], [9, 10, 11, 12, 13, 14], 9, 6),
             code="lineage")
    declines(([THINK, EOS], [9, 10], 9, 2), code="lineage")


def test_r9_no_tool_policy_refuses_any_tool_syntax_in_the_window():
    for tokens in ([THINK, 1, TC_START, TC_END, EOS], [THINK, 1, AK_START, 2, EOS],
                  [THINK, 1, AV_START, 2, EOS]):
        declines((tokens, [9 + i for i in range(len(tokens))], 9, len(tokens)),
                 code="lineage", policy=POLICY_NO_TOOL)
    # The same window is admissible under the plain content policy: the run
    # before the first tool-call-start is selected.
    selection = selects(([THINK, 1, 2, TC_START, TC_END, EOS],
                         [9 + i for i in range(6)], 9, 6), policy=POLICY_CONTENT)
    assert (selection.accepted_span_start, selection.accepted_span_end) == (1, 3)


def test_r10_partial_marker_inside_the_run_declines():
    # thinking_end attested as the 2-token sequence (101, 109): the sequence
    # recurs inside the content run and overlaps the selection at an offset
    # that is not a flagged occurrence.
    two = table(thinking_end=(THINK, 109))
    declines(([7, THINK, 109, 3, THINK, 109, EOS], [9 + i for i in range(7)], 9, 7),
             code="lineage", marker_table=two)


def test_r11_reasoning_boundary_cross_check():
    case = window([1, 2, THINK, 3, 4, EOS])
    assert selects(case, reasoning_tokens=3).accepted_span_start == 3
    declines(case, code="lineage", reasoning_tokens=2)
    declines(case, code="lineage", reasoning_tokens=True)


def test_r12_unattainable_tail_limit_declines():
    declines(([THINK, 1, 2, EOS], [9, 10, 11, 12], 9, 4),
             code="lineage", policy=POLICY_LAST32)


def test_receipt_bounds_refuse_oversized_selection_and_counts():
    # A 130-row content run cannot fit the receipt's 128-row bound.
    long = [THINK] + [50 + (i % 40) for i in range(130)] + [EOS]
    declines((long, [9 + i for i in range(len(long))], 9, len(long)), code="lineage")


def test_content_policy_selects_the_full_run():
    selection = selects(window([1, 2, THINK, 3, 4, 5, EOS]))
    assert selection.retained_rows == 3
    assert (selection.accepted_span_start, selection.accepted_span_end) == (3, 6)
    assert selection.terminator == TERMINATOR_THINKING_END
    assert selection.contiguous is True and selection.message_truncated is False
    assert selection.contains_final_content is True and selection.contains_tool_call is False
    counts = dict(zip(("THINKING", "CONTENT", "TOOL_CALL", "TOOL_ARGUMENTS",
                       "CONTROL_MARKERS", "UNKNOWN"), selection.counts))
    assert counts == {"THINKING": 2, "CONTENT": 3, "TOOL_CALL": 0,
                      "TOOL_ARGUMENTS": 0, "CONTROL_MARKERS": 2, "UNKNOWN": 0}


def test_last64_clamps_a_long_run_and_last32_keeps_a_short_one():
    long_run = [THINK] + [20 + (i % 30) for i in range(70)] + [EOS]
    clamped = selects((long_run, [9 + i for i in range(len(long_run))], 9, len(long_run)),
                      policy=POLICY_LAST64)
    assert clamped.retained_rows == 64 and clamped.clamped is True
    assert clamped.accepted_span_start == 1 + (70 - 64) and clamped.requested == 64
    # A run shorter than the limit inside a window that can still hold it:
    # nothing is cut, so clamped is False.
    short_tokens = [7] * 20 + [THINK] + [8] * 10 + [EOS]
    short = selects((short_tokens, [9 + i for i in range(32)], 9, 32), policy=POLICY_LAST32)
    assert short.retained_rows == 10 and short.clamped is False and short.requested == 32
    assert (short.accepted_span_start, short.accepted_span_end) == (21, 31)


def test_noncontiguous_union_window_selects_across_the_gap():
    # The gap is BEFORE the run (dropped reasoning rows), not inside it.
    selection = selects(([1, THINK, 3, 4, 5, EOS], [9, 10, 11, 12, 13, 15], 9, 7),
                        policy=POLICY_CONTENT)
    assert selection.contiguous is False
    assert (selection.accepted_span_start, selection.accepted_span_end) == (2, 5)
    assert selection.retained_rows == 3


def test_span_block_is_the_bounded_receipt_record():
    selection = selects(window([1, THINK, 2, 3, EOS]))
    block = selection.span_block(table())
    assert block["span_schema"] == SPAN_SCHEMA_V1
    assert block["span_policy"] == POLICY_CONTENT
    assert block["span_marker_profile"] == DEFAULT_PROFILE
    assert block["span_marker_schema"] == MARKER_SCHEMA_V1
    assert block["span_reasoning_terminator"] == TERMINATOR_THINKING_END
    assert block["span_turn_end_attested"] is True
    assert block["span_content_count"] == 2 and block["span_control_count"] == 2
    assert block["span_contiguous"] is True
    assert block["span_message_truncated"] is False


def test_wire_round_trip_is_exact_and_fail_closed():
    markers, stop_ids = table(), frozenset((EOS, 109))
    entries = wire_marker_entries(markers, stop_ids)
    assert entries[PROFILE_PARAM] == DEFAULT_PROFILE
    assert entries[SCHEMA_PARAM] == MARKER_SCHEMA_V1 and entries[ROUNDTRIP_PARAM] == 1
    assert entries[STOP_PARAM] == [EOS, 109]
    assert entries[MARKER_PARAM_PREFIX + "thinking_end"] == [THINK]
    rebuilt, rebuilt_stops = marker_table_from_wire(entries)
    assert rebuilt.by_role() == markers.by_role()
    assert rebuilt.profile == markers.profile and rebuilt.schema == markers.schema
    assert rebuilt_stops == stop_ids
    # An absent thinking_start survives the round trip; marker text never does.
    absent = table(thinking_start=None)
    entries_absent = wire_marker_entries(absent, stop_ids)
    assert MARKER_PARAM_PREFIX + "thinking_start" not in entries_absent
    rebuilt_absent, _ = marker_table_from_wire(entries_absent)
    assert rebuilt_absent.thinking_start_ids is None


def test_wire_refusals_are_fixed_profile_codes():
    entries = wire_marker_entries(table(), frozenset((EOS,)))
    bad = dict(entries); bad[ROUNDTRIP_PARAM] = 2
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire(bad)
    assert caught.value.code == "profile"
    bad = dict(entries); del bad[MARKER_PARAM_PREFIX + "turn_end"]
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire(bad)
    assert caught.value.code == "profile"
    bad = dict(entries); del bad[MARKER_PARAM_PREFIX + "tool_call_end"]
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire(bad)
    assert caught.value.code == "profile"
    bad = dict(entries); bad[STOP_PARAM] = []
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire(bad)
    assert caught.value.code == "profile"
    bad = dict(entries); bad[PROFILE_PARAM] = "invented_profile.v9"
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire(bad)
    assert caught.value.code == "profile"
    with pytest.raises(SpanPolicyError) as caught:
        marker_table_from_wire("not-a-dict")
    assert caught.value.code == "profile"


def test_declines_are_never_a_fallback_record():
    # R13: every unanswerable rule raises; nothing like last_accepted exists.
    tokens = [THINK, 1, EOS]
    positions = [9, 10, 11]
    with pytest.raises(SpanPolicyError):
        select_span_v2(tokens, positions, position_base=9, accepted_total=99,
                      policy=POLICY_CONTENT, marker_table=table(), stop_ids=STOP)
