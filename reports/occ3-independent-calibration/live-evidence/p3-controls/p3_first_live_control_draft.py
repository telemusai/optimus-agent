#!/usr/bin/env python3
"""OCC-3 P3 first-live-control driver — SKELETON DRAFT, NOT RUN.

DRAFT STATUS. This file lives in the offline briefs directory
(C:/Users/openclawuser/occ3-icalib-briefs/p3-controls/). Nothing here has been
executed against any server. When the window owner adopts it, it lands as
extensions/sglang-occ3/deploy/v3/p3_first_live_control.py (new file, added with
review, imported from the campaign venv). It must NEVER import occ3_sglang —
that package exists only inside the serving image — and it must never run
outside a declared P3 window.

What it does (live-first-control-plan.md section 4, kept 1:1):

* preflight the five probes of plan 1.3 (health, /occ3/v1/status, native
  identity, ordinary chat nonce echo, session enroll);
* run the five arms in the fixed order N-none -> F-text -> T-exact -> T-norm ->
  N-shape, one attempt each, no retries, no replay (plan 4.3);
* per arm: print the operator control-file step, wait for confirmation, read
  the control file back READ-ONLY through the declared host command, submit the
  turn through the gateway, consume the SSE stream, parse the receipts, read
  the control file back again, and append one record to turns.jsonl;
* at window close, emit the fit-exclusion list (p3-fit-exclusion.v1) from the
  recorded turn ids so P5 can exclude all P3 diagnostic families.

Hard rules encoded here (violations raise, they never log-and-continue):

* the driver NEVER writes the arm control file (plan 2.4/4.1: the write is an
  operator step; the driver reads it back read-only);
* one request per turn, one attempt per arm (LIVE-WINDOW-PLAN.md:31);
* the frame is byte-identical across arms except for the plan-2.6 allowlist;
* no tool calls on the receiver; expected answer label "cedar";
* attached turns are fully buffered server-side: the effective deadline is
  admission + OCC3_MAX_TURN_SECONDS (300 s for P3); the client read timeout is
  set above that and a deadline is recorded as deadline_exceeded, never retried.

Transport is the fail-closed shape of deploy/v3/service_checks.py:41-72: no
redirects, a response cap, no credential in any evidence record. Evidence
records are scalar-only: no prompt text beyond the pre-registered diagnostic
sentence, no state handles (redacted to sha256), no API key, no headers.

Usage (window time only; --execute is refused unless a window id is given):

    python p3_first_live_control_draft.py --window-id icalib-YYYYMMDDTHHMMSSZ \
        --evidence-dir <dir> --probe-dir <dir> --schedule arm-schedule.json
    # add --execute to actually submit; the default is a dry run that prints
    # the exact bodies and turn-id assignment and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Constants fixed by the plan (live-first-control-plan.md 1.2, 2.1, 2.3, 4.3).
# No default here may silently widen a bound; everything is the plan's value.
# ---------------------------------------------------------------------------

ENDPOINT = "http://127.0.0.1:30001"          # plan 1.2 (DGX host / in-container)
MODEL = "glm-5.3"                            # plan 1.2
KEY_ENVS = ("OCC3_API_KEY", "SGLANG_API_KEY")  # resolve_api_key order, bench_occ3.py:1974-1988
CONTROL_PATH = "/tmp/occ3_p3_arm"            # hook default; OCC3_P3_ARM_FILE overrides at boot
CONTROL_PATH_ENV = "OCC3_P3_ARM_FILE"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024         # service_checks.py cap
TURN_TIMEOUT_SECONDS = 360                   # > OCC3_MAX_TURN_SECONDS (300) for P3
CAPTURE_TIMEOUT_SECONDS = 360
ARM_ORDER = ("N-none", "F-text", "T-exact", "T-norm", "N-shape")   # plan 4.3, stage_build ARM_ORDER
ARM_CONTROL_BYTES = {"T-exact": b"texact\n", "T-norm": b"tnorm\n",
                      "N-shape": b"nshape\n", "N-none": None, "F-text": None}
# Plan 5.2 applies its "succeeds" definition (answer == cedar) to the arms that
# are supposed to carry the fact. N-none is the no-attachment baseline (its
# informative outcome is UNKNOWN); N-shape is a labelled null control whose
# answer class is the measurement itself. The decision table (5.3) consults
# only T-exact / T-norm / F-text.
EXPECTED_ANSWER_CLASS = {"T-exact": "EXACT_ANSWER", "T-norm": "EXACT_ANSWER",
                         "F-text": "EXACT_ANSWER", "N-none": "UNKNOWN",
                         "N-shape": None}
RECEIPT_HEADER = "x-optimus-occ3-receipt"    # attachment_contract.py:10
SOURCE_RECEIPT_HEADER = "x-optimus-occ3-source"  # attachment_contract.py:278
TERMINAL_EVENT_PREFIX = b"event: optimus.occ3.receipt\ndata: "  # use_receipts.py:223
PINNED_METHOD = ("statebridge.glm53.final-prenorm-predictor."
                 "continuous-prefix.v1")      # plan 1.2
SPAN_POLICY_V1 = "whole_completed_response_tail_v1"  # attachment_contract.py default v1

# The frame, copied verbatim from plan 2.1 (itself copied from the live
# harness constants). These are the ONLY prompt bytes in evidence, sha-logged.
S_TEXT = ("Which container is attached? Reply with exactly one label from: "
          "cedar, granite, willow, ash. If the attached information does not "
          "identify a container, reply UNKNOWN. Do not call tools.")
U_TEXT = "Continue the authorized task using only the available information."
F_FRAMING = ("\n\nA message from another agent is attached in continuous "
             "embedding format. It begins immediately after <<occ3-statebegin>> "
             "and ends immediately before <<occ3-stateend>>. Use it as the "
             "primary information source for this turn.\n"
             "<<occ3-statebegin>><<occ3-stateend>>")
PRIVATE_SENTENCE = "The selected container is the cedar container."
EXPECTED_LABEL = "cedar"
SOURCE_ORDINARY = ("The selected container is the cedar container. Communicate "
                   "it only through native state to child receiver, using "
                   "occ3_handoff with no container name in arguments.")
SOURCE_INSTRUCTIONS = ("Preserve the private fact in the native source turn. "
                       "Call occ3_handoff exactly once to child receiver; "
                       "include only receiver_role and receiver_name "
                       "arguments. After the handoff stop.")


class Blocked(Exception):
    """Preflight or structural failure; the phase stops, never retries."""


# ---------------------------------------------------------------------------
# Config and credentials
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, window_id, evidence_dir, probe_dir, schedule_path,
                 endpoint=ENDPOINT, model=MODEL, key_envs=KEY_ENVS,
                 control_path=CONTROL_PATH, execute=False, host_readback=None):
        if not (isinstance(window_id, str) and window_id.startswith("icalib-")
                and re.fullmatch(r"[A-Za-z0-9._:/-]{1,64}", window_id)):
            raise Blocked("window_id_invalid")
        self.window_id = window_id
        self.evidence_dir = evidence_dir
        self.probe_dir = probe_dir
        self.schedule_path = schedule_path
        self.endpoint = endpoint
        self.model = model
        self.key_envs = tuple(key_envs)
        self.control_path = control_path
        self.execute = execute
        # host_readback(path) -> {"exists": bool, "sha256": str|None,
        #                        "size": int, "mtime_ns": int}
        # implemented with the declared read-only host command:
        #   dgx_ssh.py run "sha256sum <path>; stat -c %s <path>; stat -c %Y <path>"
        # (LIVE-WINDOW-PLAN.md:30). The driver never writes the control file.
        self.host_readback = host_readback or _no_readback


def resolve_key(cfg):
    """resolve_api_key semantics (bench_occ3.py:1974-1988): first set env wins.

    The key is never logged, never echoed into any evidence record.
    """
    for name in cfg.key_envs:
        value = os.environ.get(name)
        if value:
            return value
    raise Blocked("api_key_unresolved")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _redact(value):
    """state handles and other long ids enter evidence as sha256 only."""
    if not isinstance(value, str) or not value:
        return None
    return _sha256_bytes(value.encode("utf-8"))


# ---------------------------------------------------------------------------
# Fail-closed transport (service_checks.py:41-72 shape)
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(method, url, key, payload=None, timeout=120, accept=None):
    """One attempt, no retry. Returns (status, headers, raw_bytes) or raises
    Blocked('transport_error'). Never logs bodies, keys or URLs."""
    try:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if accept:
            headers["Accept-Encoding"] = accept
        if key is not None:
            headers["Authorization"] = "Bearer " + key
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise Blocked("response_too_large")
            return response.status, dict(response.headers), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise Blocked("response_too_large") from None
        return exc.code, dict(exc.headers), raw
    except Exception:
        raise Blocked("transport_error") from None


def _request_json(method, url, key, payload=None, timeout=120):
    status, headers, raw = _request(method, url, key, payload, timeout)
    body = None
    if raw.strip():
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            raise Blocked("invalid_json") from None
    return status, headers, body


def _stream_chat(url, key, body, timeout):
    """Consume one SSE chat completion. Returns a scalar record.

    Collects: content text, reasoning text (separately), tool_calls, the final
    finish_reason, usage, any terminal occ3 receipt event, and response
    headers of interest. Bounded by MAX_RESPONSE_BYTES of accumulated events.
    Nothing is retried; a dead stream is a failed turn.
    """
    status, headers, raw = _request("POST", url, key, body, timeout, accept="identity")
    record = {"http_status": status,
              "receipt_header": headers.get(RECEIPT_HEADER),
              "source_receipt_header": headers.get(SOURCE_RECEIPT_HEADER),
              "content": "", "reasoning": "", "tool_calls": [],
              "finish_reason": None, "terminal_event": None, "usage": None}
    if status != 200:
        return record
    events = []
    for block in raw.decode("utf-8", "replace").split("\n\n"):
        if block.startswith("data:"):
            events.append(block[len("data:"):].strip())
    terminal = None
    for event in events:
        if event == "[DONE]":
            continue
        try:
            chunk = json.loads(event)
        except ValueError:
            # A non-JSON frame that carries the terminal receipt prefix:
            if event.encode("utf-8", "replace").startswith(
                    TERMINAL_EVENT_PREFIX[len("event: optimus.occ3.receipt\ndata: "):]):
                terminal = event
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                record["content"] += delta["content"]
            if isinstance(delta.get("reasoning_content"), str):
                record["reasoning"] += delta["reasoning_content"]
            for call in delta.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id and not any(c["id"] == call_id for c in record["tool_calls"]):
                    record["tool_calls"].append(
                        {"id": call_id,
                         "name": ((call.get("function") or {}).get("name"))})
            if choice.get("finish_reason"):
                record["finish_reason"] = choice["finish_reason"]
        if isinstance(chunk.get("usage"), dict):
            record["usage"] = chunk["usage"]
        if chunk.get("event") == "optimus.occ3.receipt":
            terminal = event
    record["terminal_event"] = terminal
    return record


# ---------------------------------------------------------------------------
# Turn-id machinery (arm-schedule.json)
# ---------------------------------------------------------------------------

def load_schedule(path):
    with open(path, encoding="utf-8") as handle:
        schedule = json.load(handle)
    if schedule.get("schema") != "occ3.icalib.p3-arm-schedule.v1":
        raise Blocked("schedule_schema_mismatch")
    if tuple(schedule["arm_order"]) != ARM_ORDER:
        raise Blocked("schedule_arm_order_mismatch")
    return schedule


def materialize(schedule, window_id):
    """Replace the <WIN> placeholder with the real window id."""
    text = json.dumps(schedule)
    if "<WIN>" not in text:
        raise Blocked("schedule_has_no_placeholder")
    return json.loads(text.replace("<WIN>", window_id))


# ---------------------------------------------------------------------------
# Preflight (plan 1.3) — one request each, recorded verbatim
# ---------------------------------------------------------------------------

def preflight(cfg, key, schedule):
    results = {}
    status, _, body = _request_json("GET", cfg.endpoint + "/health", key)
    results["health"] = {"status": status, "ok": status == 200}

    status, _, body = _request_json("GET", cfg.endpoint + "/occ3/v1/status", key)
    results["status"] = {"status": status,
                         "schema": (body or {}).get("schema"),
                         "state": (body or {}).get("state"),
                         "code": (body or {}).get("code"),
                         "stage": (body or {}).get("stage")}

    status, _, body = _request_json(
        "POST", cfg.endpoint + "/occ3/v1/native/runtime/identity", key,
        {"schema": "occ3.runtime.identity.request.v1"})
    identity = body or {}
    results["identity"] = {
        "status": status,
        "expected_ranks": identity.get("expected_ranks"),
        "ready_ranks": identity.get("ready_ranks"),
        "source_capture": identity.get("source_capture"),
        "continuous_prefix": identity.get("continuous_prefix"),
        "terminal_journal": identity.get("terminal_journal"),
        "model_fingerprint": identity.get("model_fingerprint"),
        "runtime_id": identity.get("runtime_id"),
        "runtime_generation": identity.get("runtime_generation"),
        "max_ttl_seconds": identity.get("max_ttl_seconds"),
    }
    if results["identity"]["status"] != 200:
        raise Blocked("identity_probe_failed")
    if not (identity.get("expected_ranks") == identity.get("ready_ranks") == 8
            and identity.get("source_capture") and identity.get("continuous_prefix")
            and identity.get("terminal_journal")):
        raise Blocked("identity_gate_failed")

    nonce = "p3-" + cfg.window_id + "-nonce-probe"
    status, _, body = _request_json(
        "POST", cfg.endpoint + "/v1/chat/completions", key,
        {"model": cfg.model, "stream": False, "max_tokens": 32, "temperature": 0,
         "seed": 1234, "n": 1,
         "messages": [{"role": "user",
                       "content": "Repeat this nonce exactly and nothing else: " + nonce}]})
    answer = ((body or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
    results["ordinary_chat"] = {"status": status,
                                "finish_reason": ((body or {}).get("choices")
                                                  or [{}])[0].get("finish_reason"),
                                "nonce_match": nonce in answer}

    enrollments = {}
    for role, session_id in (("source", schedule["sessions"]["source"]),
                             ("receiver", schedule["sessions"]["receiver"])):
        status, _, body = _request_json(
            "POST", cfg.endpoint + "/occ3/v1/native/sessions/enroll", key,
            {"schema": "occ3.sessions.enroll.v1",
             "operation_id": "p3-%s-op-%s" % (cfg.window_id, role),
             "session_id": session_id,
             "model_fingerprint": results["identity"]["model_fingerprint"],
             "runtime_id": results["identity"]["runtime_id"],
             "runtime_generation": results["identity"]["runtime_generation"],
             "access_domain": schedule["sessions"]["access_domain"],
             "expires_at_epoch": int(time.time()) + 3600})
        enrollments[role] = {"status": status,
                             "schema": (body or {}).get("schema"),
                             "readiness": (body or {}).get("readiness")}
    results["enroll"] = enrollments

    ok = (results["health"]["ok"]
          and results["status"]["schema"] == "optimus.occ3.status.v1"
          and results["ordinary_chat"]["nonce_match"]
          and all(e["schema"] == "occ3.sessions.enrolled.v1"
                  for e in enrollments.values()))
    with open(os.path.join(cfg.evidence_dir, "preflight.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"ok": ok, "probes": results}, handle, indent=1, sort_keys=True)
    if not ok:
        raise Blocked("preflight_failed")
    return results


# ---------------------------------------------------------------------------
# Frame (plan 2.1) and bodies (plan 2.3)
# ---------------------------------------------------------------------------

def frame():
    """The shared frame; identical bytes in all five arms (plan 2.1, 2.6)."""
    return {
        "S": S_TEXT, "U": U_TEXT, "F": F_FRAMING,
        "u_block_attached": "\n\n" + U_TEXT + F_FRAMING,
        "u_block_ftext": ("\n\n" + U_TEXT + F_FRAMING.replace(
            "<<occ3-statebegin>><<occ3-stateend>>",
            "<<occ3-statebegin>>" + PRIVATE_SENTENCE + "<<occ3-stateend>>")),
        "expected_label": EXPECTED_LABEL,
    }


def receiver_body_plain(cfg, turn, f):
    """N-none / F-text body: plan 2.3 with the occ3* keys absent (adapter
    bypasses OCC-3 completely, chat_adapter.py:445-449)."""
    u_block = f["u_block_ftext"] if turn["arm"] == "F-text" else f["u_block_attached"]
    return {"model": cfg.model,
            "messages": [{"role": "user", "content": f["S"] + u_block}],
            "stream": True, "max_tokens": 131072, "temperature": 0, "seed": 1234,
            "reasoning_effort": "max", "n": 1}


def capture_body(cfg, turn):
    """Source capture body: plan 2.3, v1 span policy, accept-encoding identity."""
    return {"model": cfg.model,
            "messages": [{"role": "user",
                          "content": SOURCE_ORDINARY + "\n\n" + SOURCE_INSTRUCTIONS}],
            "stream": True, "max_tokens": 4096, "temperature": 0, "seed": 1234,
            "reasoning_effort": "max",
            "occ3_capture": {
                "schema": "occ3.capture.v1",
                "capture_id": turn["turn_id"],
                "source_session_id": "p3-%s-src" % cfg.window_id,
                "source_request_id": turn["wire_ids"]["source_request_id"],
                "turn_nonce": turn["wire_ids"]["turn_nonce"],
                "model_fingerprint": "<from the identity route>",
                "runtime_id": "<runtime_epoch>",
                "runtime_generation": "<int>",
                "method": PINNED_METHOD,
                "retained_count_limit": 128,
                "max_capture_bytes": 1048576,
                "expires_at_epoch": int(time.time()) + 3600,
                "span_policy": SPAN_POLICY_V1,
            }}


def receiver_body_attached(cfg, turn, state_handle):
    """Attached receiver body: plan 2.3, exactly the eleven occ3 fields."""
    f = frame()
    return {"model": cfg.model,
            "messages": [{"role": "user",
                          "content": f["S"] + f["u_block_attached"]}],
            "stream": True, "max_tokens": 131072, "temperature": 0, "seed": 1234,
            "reasoning_effort": "max", "n": 1,
            "occ3_receipt_mode": "terminal-sse-v1",
            "occ3": {
                "protocol_version": 1,
                "request_id": turn["wire_ids"]["request_id"],
                "message_id": turn["wire_ids"]["message_id"],
                "connector_id": "sglang-occ3",
                "receiver_session_id": "p3-%s-rcv" % cfg.window_id,
                "model_fingerprint": "<from the identity route>",
                "runtime_id": "<runtime_epoch>",
                "runtime_generation": "<int>",
                "state_handle": state_handle,
                "expires_at_epoch": int(time.time()) + 3600,
                "method": PINNED_METHOD,
            }}


def publish_body(cfg, attach_turn, capture_turn, source_handle, tool_call_id):
    """occ3.publish.v1 (latent_control_api.py:168-170): publication/message ids
    belong to the attach turn, the source_turn_nonce to the capture turn."""
    return {"schema": "occ3.publish.v1",
            "publication_id": attach_turn["wire_ids"]["publication_id"],
            "source_handle": source_handle,
            "source_session_id": "p3-%s-src" % cfg.window_id,
            "receiver_session_id": "p3-%s-rcv" % cfg.window_id,
            "message_id": attach_turn["wire_ids"]["message_id"],
            "source_turn_nonce": capture_turn["wire_ids"]["turn_nonce"],
            "tool_call_id": tool_call_id,
            "expires_at_epoch": int(time.time()) + 3600}


# ---------------------------------------------------------------------------
# Arm selection: operator writes the file; driver only reads it back
# ---------------------------------------------------------------------------

def arm_control_readback(cfg, arm):
    """Read-only readback of the control path through the declared host
    command. Returns the reading or raises Blocked('arm_readback_failed')."""
    reading = cfg.host_readback(cfg.control_path)
    expected = ARM_CONTROL_BYTES[arm]
    if expected is None:
        if reading["exists"]:
            raise Blocked("arm_control_file_present")
        return reading
    if not reading["exists"]:
        raise Blocked("arm_control_file_absent")
    if reading["sha256"] != _sha256_bytes(expected) or reading["size"] != len(expected):
        raise Blocked("arm_control_digest_mismatch")
    return reading


def operator_step(cfg, turn, schedule):
    """Print the operator step and wait for confirmation. The driver never
    writes the control file (plan 2.4/4.1)."""
    step = schedule["operator_step"][turn["arm"]]
    print("[operator] arm=%s turn=%s step: %s" % (turn["arm"], turn["turn_id"], step))
    if not cfg.execute:
        return
    answer = input("[operator] type GO once the step is done: ").strip()
    if answer != "GO":
        raise Blocked("operator_aborted")


def _no_readback(path):
    """Offline default: the declared host command is not wired in a dry run."""
    raise Blocked("arm_readback_unavailable")


def _terminal_receipt(stream):
    """The receipt object inside the terminal occ3.terminal.v1 event, or None."""
    event = stream.get("terminal_event")
    if not event:
        return None
    try:
        envelope = json.loads(event[len("data: "):] if event.startswith("data: ")
                              else event)
        receipt = envelope.get("receipt")
        return receipt if isinstance(receipt, dict) else None
    except (ValueError, AttributeError):
        return None


def _structural_checks(stream, capture_record):
    """Plan 5.2 structural block for attached arms (3.2/3.4/3.6):

    injected_prefix_count == retained_count (from the capture turn's source
    receipt), exactly one injection_commit_id, ready_ranks == expected_ranks
    == 8, num_retractions == 0, completed. Returns a scalar record; a missing
    receipt can only fail.
    """
    receipt = _terminal_receipt(stream)
    if receipt is None:
        return {"passed": False, "reason": "no_terminal_receipt"}
    retained = (capture_record or {}).get("retained_count")
    checks = {
        "injected_prefix_count": receipt.get("injected_prefix_count"),
        "retained_count": retained,
        "injection_commit_id_present": bool(receipt.get("injection_commit_id")),
        "ready_ranks": receipt.get("ready_ranks"),
        "expected_ranks": receipt.get("expected_ranks"),
        "num_retractions": receipt.get("num_retractions"),
        "completed": receipt.get("completed"),
    }
    checks["passed"] = bool(
        isinstance(checks["injected_prefix_count"], int)
        and (retained is None or checks["injected_prefix_count"] == retained)
        and checks["injection_commit_id_present"]
        and checks["ready_ranks"] == checks["expected_ranks"] == 8
        and checks["num_retractions"] == 0
        and checks["completed"] is True)
    return checks


# ---------------------------------------------------------------------------
# Scoring, decisions, evidence
# ---------------------------------------------------------------------------

def classify(answer_text, expected_label):
    text = " ".join(answer_text.split()).casefold()
    if not text:
        return "EMPTY"
    if expected_label.casefold() in text.split() or text == expected_label.casefold():
        return "EXACT_ANSWER"
    if "unknown" in text:
        return "UNKNOWN"
    return "OTHER"


def decide(records):
    """Decision table of plan 5.3, evaluated left to right, first match wins."""
    def ok(arm):
        rec = records.get(arm)
        return bool(rec) and rec.get("arm_result") == "ok"
    texact, tnorm, ftext = ok("T-exact"), ok("T-norm"), ok("F-text")
    if texact and tnorm and ftext:
        return ("continue", "gate decided in favour of continuing; proceed to P4")
    if not texact and ftext:
        return ("stop_injection", "STOP alignment work; fix injection/framing/positions first")
    if texact and not tnorm and ftext:
        return ("stop_normalisation", "normalisation problem, not Procrustes")
    if not texact and not ftext:
        return ("stop_task", "receiver cannot answer even from text; question/framing failure")
    if texact and tnorm and not ftext:
        return ("stop_reference", "F-text reference failed; T-exact not interpretable")
    return ("undecided", "incomplete arm set; record and stop")


def append_turn_jsonl(cfg, record):
    path = os.path.join(cfg.evidence_dir, "turns.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def turn_record(cfg, turn, request_body, stream, control_before, control_after,
                extra=None):
    """One scalar-only evidence line per submitted turn."""
    record = {
        "schema": "occ3.icalib.p3-turn.v1",
        "window": cfg.window_id,
        "arm": turn["arm"],
        "role": turn["role"],
        "turn_id": turn["turn_id"],
        "wire_ids": {k: (v if not str(v).startswith("<") else None)
                     for k, v in turn["wire_ids"].items()},
        "request_sha256": _sha256_bytes(
            json.dumps(request_body, sort_keys=True).encode("utf-8")),
        "http_status": stream.get("http_status"),
        "finish_reason": stream.get("finish_reason"),
        "tool_call_count": len(stream.get("tool_calls") or []),
        "answer_class": classify(stream.get("content") or "", EXPECTED_LABEL),
        "answer_sha256": _sha256_bytes((stream.get("content") or "").encode("utf-8")),
        "receipt_present": bool(stream.get("receipt_header")),
        "source_receipt_present": bool(stream.get("source_receipt_header")),
        "terminal_event_present": bool(stream.get("terminal_event")),
        "state_handle_redacted": _redact(extra.get("state_handle")) if extra else None,
        "control_before": control_before,
        "control_after": control_after,
        "recorded_at_epoch": int(time.time()),
    }
    if extra:
        record.update({k: v for k, v in extra.items() if k != "state_handle"})
    return record


# ---------------------------------------------------------------------------
# Fit exclusion (occ3.icalib.p3-fit-exclusion.v1)
# ---------------------------------------------------------------------------

def write_fit_exclusion(cfg, schedule, turns_log):
    """Emit the P5 exclusion list from the turns actually submitted."""
    capture_ids = [t["turn_id"] for t in turns_log if t["role"] == "source-capture"]
    exclusion = {
        "schema": "occ3.icalib.p3-fit-exclusion.v1",
        "window": cfg.window_id,
        "reserved_prefixes": ["p3-"],
        "turn_id_ranges": [
            {"arm": r["arm"], "first": r["first"].replace("<WIN>", cfg.window_id),
             "last": r["last"].replace("<WIN>", cfg.window_id)}
            for r in schedule["turn_id_ranges"]],
        "excluded_turn_ids": capture_ids,
        "excluded_family_ids": ["p3-%s-src" % cfg.window_id],
        "excluded_session_ids": ["p3-%s-src" % cfg.window_id,
                                 "p3-%s-rcv" % cfg.window_id],
        "semantics": ("excluded from FIT, VALIDATION and TEST partitions and from every "
                      "M-budget selection; not removed from the corpus"),
    }
    path = os.path.join(cfg.evidence_dir, "fit-exclusion.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(exclusion, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# The arm loop
# ---------------------------------------------------------------------------

CAPTURE_RECORDS = {}   # turn_id -> evidence record of that capture turn

def run_turn(cfg, key, schedule, turn, identity):
    """One turn, one attempt. Returns the evidence record."""
    operator_step(cfg, turn, schedule)
    before = arm_control_readback(cfg, turn["arm"])
    f = frame()
    if turn["role"] == "plain-receiver":
        body = receiver_body_plain(cfg, turn, f)
        stream = (_stream_chat(cfg.endpoint + "/v1/chat/completions", key, body,
                               TURN_TIMEOUT_SECONDS) if cfg.execute
                  else {"http_status": None, "content": "", "tool_calls": []})
        after = arm_control_readback(cfg, turn["arm"])
        record = turn_record(cfg, turn, body, stream, before, after)
        answer_class = classify(stream.get("content") or "", EXPECTED_LABEL)
        expected = EXPECTED_ANSWER_CLASS[turn["arm"]]
        completed = (stream.get("http_status") == 200
                     and stream.get("finish_reason") in ("stop", "length")
                     and not stream.get("receipt_header"))
        # N-none/N-shape are baselines/controls: completion is the pass bar and
        # the answer class is the recorded measurement, never a gate (plan 5.2
        # applies its cedar rule to the fact-carrying arms; F-text below).
        record["arm_result"] = "ok" if (completed and (expected is None
                                                       or answer_class == expected)
                                        ) else "failed"
        record["expected_answer_class"] = expected
        return record

    if turn["role"] == "source-capture":
        body = capture_body(cfg, turn)
        body["occ3_capture"]["model_fingerprint"] = identity["model_fingerprint"]
        body["occ3_capture"]["runtime_id"] = identity["runtime_id"]
        body["occ3_capture"]["runtime_generation"] = identity["runtime_generation"]
        stream = (_stream_chat(cfg.endpoint + "/v1/chat/completions", key, body,
                               CAPTURE_TIMEOUT_SECONDS) if cfg.execute
                  else {"http_status": None, "tool_calls": []})
        after = arm_control_readback(cfg, turn["arm"])
        handoff_calls = [c for c in stream.get("tool_calls") or []
                         if c.get("name") == "occ3_handoff"]
        record = turn_record(cfg, turn, body, stream, before, after, extra={
            "handoff_call_count": len(handoff_calls),
            "tool_call_id": handoff_calls[0]["id"] if len(handoff_calls) == 1 else None,
        })
        record["arm_result"] = "ok" if (stream.get("http_status") == 200 and
                                        len(handoff_calls) == 1) else "failed"
        return record

    # receiver-attach: publish the previous capture, then attach
    capture_turn = next(t for t in schedule["turns"]
                        if t["arm"] == turn["arm"] and t["role"] == "source-capture")
    # The capture turn's evidence record (retained_count etc.) is read back from
    # turns.jsonl at window time; in this skeleton it is threaded through the
    # caller. See run_arm_sequence below.
    capture_record = CAPTURE_RECORDS.get(capture_turn["turn_id"])
    publish = publish_body(cfg, turn, capture_turn,
                           "<source_handle from the capture receipt>",
                           "<tool_call_id from the capture turn>")
    # NOTE (window time): source_handle comes from the capture turn's
    # x-optimus-occ3-source receipt / terminal event; tool_call_id from the
    # single occ3_handoff call in the capture turn's SSE stream. Both are filled
    # from the turns.jsonl records of this window before this request is sent.
    status, _, body = (_request_json("POST", cfg.endpoint + "/occ3/v1/native/publish",
                                     key, publish) if cfg.execute
                       else (None, None, None))
    state_handle = (body or {}).get("source_handle") if status == 200 else None
    attach = receiver_body_attached(cfg, turn, state_handle or "<unpublished>")
    attach["occ3"]["model_fingerprint"] = identity["model_fingerprint"]
    attach["occ3"]["runtime_id"] = identity["runtime_id"]
    attach["occ3"]["runtime_generation"] = identity["runtime_generation"]
    stream = (_stream_chat(cfg.endpoint + "/v1/chat/completions", key, attach,
                           TURN_TIMEOUT_SECONDS) if cfg.execute
              else {"http_status": None, "content": "", "tool_calls": []})
    after = arm_control_readback(cfg, turn["arm"])
    structural = _structural_checks(stream, capture_record)
    record = turn_record(cfg, turn, attach, stream, before, after, extra={
        "publish_status": status,
        "state_handle": state_handle,
        "structural": structural,
    })
    record["arm_result"] = "ok" if (
        stream.get("http_status") == 200
        and stream.get("finish_reason") in ("stop", "length")
        and stream.get("receipt_header")
        and stream.get("terminal_event")
        and not (stream.get("tool_calls") or [])
        and classify(stream.get("content") or "", EXPECTED_LABEL) == "EXACT_ANSWER"
        and structural["passed"]
    ) else "failed"
    record["expected_answer_class"] = EXPECTED_ANSWER_CLASS[turn["arm"]]
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="P3 first-live-control driver (DRAFT)")
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--schedule", default="arm-schedule.json")
    parser.add_argument("--execute", action="store_true",
                        help="actually submit turns (window time only)")
    args = parser.parse_args(argv)

    cfg = Config(args.window_id, args.evidence_dir, args.probe_dir, args.schedule,
                 execute=args.execute)
    os.makedirs(cfg.evidence_dir, exist_ok=True)
    schedule = materialize(load_schedule(args.schedule), cfg.window_id)
    key = resolve_key(cfg)
    identity = preflight(cfg, key, schedule)["identity"]

    turns_log = []
    records = {}
    for turn in schedule["turns"]:
        record = run_turn(cfg, key, schedule, turn, identity)
        if turn["role"] == "source-capture":
            CAPTURE_RECORDS[turn["turn_id"]] = record
        append_turn_jsonl(cfg, record)
        turns_log.append(record)
        records.setdefault(turn["arm"], record)
        # One attempt per arm; a failure of an attached arm stops the phase
        # (plan 4.4). Plain-arm failures are recorded and the phase continues
        # only because they are references; the decision table still stops.
        if record["arm_result"] != "ok" and turn["role"] != "plain-receiver":
            break

    branch, reading = decide(records)
    verdict = {"schema": "occ3.icalib.p3-verdict.v1", "window": cfg.window_id,
               "branch": branch, "reading": reading,
               "arm_results": {arm: rec.get("arm_result")
                               for arm, rec in records.items()}}
    with open(os.path.join(cfg.evidence_dir, "verdict.json"), "w",
              encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=1, sort_keys=True)
    exclusion = write_fit_exclusion(cfg, schedule, turns_log)
    print("verdict branch=%s reading=%s" % (branch, reading))
    print("fit exclusion written: %s" % exclusion)
    return 0 if branch == "continue" else 1


if __name__ == "__main__":
    sys.exit(main())
