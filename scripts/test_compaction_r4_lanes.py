#!/usr/bin/env python3
"""Non-timing R4 synthetic and schema controls. No performance admission."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import sys

from compaction_r4_lanes import Reject, load_report, validate_external_wall, validate_lane_report
import compaction_r4_lanes_mock as mock

checks = []


def accepted(name, value, external=False):
    try:
        (validate_external_wall if external else validate_lane_report)(value)
    except Exception as error:
        raise AssertionError(f"{name}: valid control rejected: {error}") from error
    checks.append({"name": name, "accepted": True})


def rejected(name, value, stage, external=False):
    try:
        (validate_external_wall if external else validate_lane_report)(value)
    except Reject as error:
        text = str(error)
        if not text.startswith("[") or stage not in text:
            raise AssertionError(f"{name}: wrong rejection {text}, wanted {stage}") from error
        checks.append({"name": name, "rejected": True, "intended_stage": stage, "error": text})
        return
    raise AssertionError(f"{name}: invalid control accepted")


def changed(value, path, replacement):
    value = copy.deepcopy(value)
    target = value
    for key in path[:-1]:
        target = target[key]
    if replacement is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = replacement
    return value


class Delete:
    pass


DELETE = Delete()


def main():
    for lane in ["wall", "allocation", "replay-phase", "minimal-provider"]:
        accepted(f"{lane}_accepted", mock.lane_report(lane))
    for outcome in ["failure", "cancelled"]:
        accepted(f"minimal_provider_{outcome}_receipt_accepted", mock.lane_report("minimal-provider", outcome))
    accepted("external_wall_accepted", mock.external_wall(), external=True)

    wall = mock.lane_report("wall")
    rejected("wrong_schema", changed(wall, ["schema"], "r3"), "schema")
    rejected("wall_internal_duration_leak", changed(wall, ["internal_ms"], 1), "wall-lane-exclusivity")
    wall_leak = mock.lane_report("wall")
    wall_leak["fixture"]["phase_ms"] = 1
    rejected("wall_nested_duration_leak", wall_leak, "wall-lane-exclusivity")
    rejected("wall_provider_field_leak", dict(wall, provider=mock.provider()), "wall-lane-exclusivity")
    rejected("wall_allocation_field_leak", dict(wall, allocation={"status": "unavailable"}), "report-fields")
    rejected("wall_instrumentation_leak", changed(wall, ["build", "instrumentation"], ["phase-marks"]), "instrumentation-identity")
    rejected("wall_wrong_build", changed(wall, ["build", "configuration"], "instrumented"), "build-lane-identity")
    rejected("wall_wrong_feature", changed(wall, ["build", "feature"], None), "build-feature")
    rejected("warm_page_cache_inference", changed(mock.external_wall(), ["observation", "page_cache_state"], "warm"), "external-observation", external=True)
    rejected("external_wall_identity", changed(mock.external_wall(), ["lane"], "replay-phase"), "external-wall-identity", external=True)
    rejected("external_wall_nonfinite", changed(mock.external_wall(), ["external_wall", "duration_ms"], math.inf), "external-wall-duration-ms", external=True)
    rejected("external_wall_bool", changed(mock.external_wall(), ["external_wall", "duration_ms"], True), "external-wall-duration-ms", external=True)
    rejected("external_wall_pid_mismatch", changed(mock.external_wall(), ["child", "pid"], 999), "child-pid-identity", external=True)
    rejected("external_wall_child_lane_mismatch", changed(mock.external_wall(), ["child_report"], mock.lane_report("allocation")), "lane-identity", external=True)

    allocation = mock.lane_report("allocation")
    rejected("allocation_zero_reported_as_available", changed(allocation, ["allocation", "count"], 0), "allocation-count")
    rejected("allocation_bool_rejected", changed(allocation, ["allocation", "count"], True), "allocation-count")
    rejected("allocation_float_rejected", changed(allocation, ["allocation", "bytes_requested"], 1.5), "allocation-bytes")
    rejected("allocation_string_rejected", changed(allocation, ["allocation", "count"], "19"), "allocation-count")
    rejected("allocation_missing_counter", changed(allocation, ["allocation", "count"], DELETE), "allocation")
    rejected("allocation_extra_phase_lane", dict(allocation, local_replay={}), "report-fields")
    unavailable = changed(allocation, ["allocation"], {"status": "unavailable", "source": None, "count": None, "bytes_requested": None, "reason": "supported counters absent"})
    accepted("allocation_unavailable_null_reason_accepted", unavailable)
    rejected("allocation_unavailable_zero", changed(unavailable, ["allocation", "count"], 0), "allocation-unavailable")
    rejected("allocation_unavailable_estimate", changed(unavailable, ["allocation", "bytes_requested"], 10), "allocation-unavailable")
    rejected("allocation_unavailable_missing_reason", changed(unavailable, ["allocation", "reason"], DELETE), "allocation")

    provider = mock.lane_report("minimal-provider")
    rejected("provider_bool_wait", changed(provider, ["provider", "wait", "requested_ms"], False), "provider-wait-value")
    rejected("provider_float_wait", changed(provider, ["provider", "wait", "requested_ms"], 1.5), "provider-wait-value")
    rejected("provider_negative_wait", changed(provider, ["provider", "wait", "requested_ms"], -1), "provider-wait-value")
    rejected("provider_estimated_wait", changed(provider, ["provider", "wait", "observed_duration_ms"], 25), "provider-wait-estimate")
    rejected("provider_wait_reason_missing", changed(provider, ["provider", "wait", "unavailable_reason"], ""), "provider-wait-reason")
    rejected("provider_generation_bool", changed(provider, ["provider", "generation", "summary_render_iterations"], True), "provider-generation-units")
    rejected("provider_generation_float", changed(provider, ["provider", "generation", "summary_render_iterations"], 2.5), "provider-generation-units")
    rejected("provider_generation_zero", changed(provider, ["provider", "generation", "summary_render_iterations"], 0), "provider-generation-units")
    rejected("provider_generation_duration_claim", changed(provider, ["provider", "generation", "generated_units_are_not_duration"], False), "provider-generation-source")
    rejected("provider_call_count_bool", changed(provider, ["provider", "call_count"], True), "provider-call-count")
    rejected("provider_success_without_call", changed(provider, ["provider", "call_count"], 0), "provider-call-accounting")
    rejected("provider_multiple_uncontrolled_calls", changed(provider, ["provider", "call_count"], 2), "provider-call-accounting")
    rejected("provider_extra_local_duration", changed(provider, ["provider", "local_ms"], 1), "provider")

    rejected("phase_bool_duration", changed(provider, ["local_replay", "phases", 0, "duration_ms"], True), "phase-duration")
    rejected("phase_string_duration", changed(provider, ["local_replay", "phases", 0, "duration_ms"], "1"), "phase-duration")
    rejected("phase_nan_duration", changed(provider, ["local_replay", "phases", 0, "duration_ms"], math.nan), "phase-duration")
    rejected("phase_negative_duration", changed(provider, ["local_replay", "phases", 0, "duration_ms"], -1), "phase-duration")
    rejected("phase_provider_cpu_leak", changed(provider, ["local_replay", "phases", 0, "cpu_scope"], "provider-plus-local"), "provider-local-separation")
    rejected("phase_overhead_unlabeled", changed(provider, ["local_replay", "phases", 0, "overhead"], ""), "phase-overhead")
    rejected("phase_boundary_unlabeled", changed(provider, ["local_replay", "phases", 0, "boundary"], ""), "phase-boundary")
    rejected("phase_duplicate", changed(provider, ["local_replay", "phases", 1], mock.phase("prepare_compaction")), "phase-coverage")
    rejected("phase_missing_continuation", changed(provider, ["local_replay", "phases"], provider["local_replay"]["phases"][:-1]), "phase-coverage")
    rejected("provider_policy_removed", changed(provider, ["local_replay", "provider_time_policy"], "local includes provider"), "provider-local-separation")
    rejected("provider_extra_phase_field", changed(provider, ["local_replay", "phases", 0, "provider_wait_ms"], 0), "phase-row")

    rejected("continuation_count_zero", changed(provider, ["continuation", "message_count"], 0), "continuation-message-count")
    rejected("continuation_count_bool", changed(provider, ["continuation", "message_count"], True), "continuation-message-count")
    rejected("continuation_compaction_count_zero", changed(provider, ["continuation", "compaction_entries"], 0), "continuation-accounting")
    rejected("continuation_compaction_count_two", changed(provider, ["continuation", "compaction_entries"], 2), "continuation-accounting")
    rejected("continuation_digest_short", changed(provider, ["continuation", "context_sha256"], "abc"), "continuation-digest")
    rejected("continuation_digest_uppercase", changed(provider, ["continuation", "context_sha256"], "B" * 64), "continuation-digest")
    rejected("continuation_success_unavailable", changed(provider, ["continuation", "status"], "unavailable"), "continuation-accounting")
    failure = mock.lane_report("minimal-provider", "failure")
    rejected("failure_committed_continuation", changed(failure, ["continuation"], mock.continuation()), "continuation-accounting")
    rejected("failure_zero_count_instead_null", changed(failure, ["continuation", "message_count"], 0), "continuation-unavailable")
    rejected("failure_empty_error", changed(failure, ["outcome", "error"], ""), "outcome-error")
    cancelled = mock.lane_report("minimal-provider", "cancelled")
    rejected("cancelled_provider_called", changed(cancelled, ["provider", "call_count"], 1), "provider-cancellation")
    rejected("cancelled_success_injection", changed(cancelled, ["provider", "injection"], "none"), "provider-injection")
    rejected("cancelled_phase_leak", changed(cancelled, ["local_replay", "phases"], [mock.phase("prepare_compaction")]), "cancellation-phase-receipt")
    accepted("failure_receipt_correct", failure)
    accepted("cancelled_receipt_correct", cancelled)

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r4-synthetic-controls.json")
    malformed = out.parent / "malformed-r4.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text('{"schema": ', encoding="utf-8")
    try:
        load_report(malformed)
    except Reject as error:
        if "malformed-json" not in str(error):
            raise
        checks.append({"name": "malformed_json_rejected", "rejected": True, "intended_stage": "malformed-json", "error": str(error)})
    else:
        raise AssertionError("malformed JSON accepted")
    nonfinite = out.parent / "nonfinite-r4.json"
    nonfinite.write_text('{"x": NaN}', encoding="utf-8")
    try:
        load_report(nonfinite)
    except Reject as error:
        checks.append({"name": "nonfinite_json_rejected", "rejected": True, "intended_stage": "malformed-json", "error": str(error)})
    else:
        raise AssertionError("nonfinite JSON accepted")

    names = [check["name"] for check in checks]
    if len(names) != len(set(names)):
        raise AssertionError("control names not unique")
    result = {
        "schema": "r4-synthetic-controls-v1",
        "status": "ok",
        "synthetic_durations": "non-decision",
        "check_count": len(checks),
        "accepted": sum(check.get("accepted") is True for check in checks),
        "rejected": sum(check.get("rejected") is True for check in checks),
        "checks": checks,
        "unrun": ["Optimus timing distributions", "performance decisions", "real provider/network"],
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(checks)} controls)")


if __name__ == "__main__":
    main()
