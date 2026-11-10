"""Four-way benchmark adapter: TXT / OCC-1 / OCC-2 / OCC-3-native.

Campaign occ3-bench-opt-20261109. Measures end-to-end task accuracy of the
four production communication channels between two REAL daemon sessions on
the SAME model (dgx-glm53-occ3/glm-5.3), on the 50-task taskbench dev
matrix (first 10 tasks per family A-E), reusing the existing task
definitions (research/tasks/tasks_dev.json) and checkers
(research/tasks/checkers.py).

Why CLI-driven real sessions (not the old gateway):
  The historical runner (runner.py) drove a local model gateway
  (127.0.0.1:43120) that no longer exists, and model-behavioral prompt
  guidance alone cannot measure the production transport. This adapter
  drives the production stack end to end:
    - speaker/listener are real daemon sessions (children of one short-lived
      parent session, so they are true siblings with a real parent edge),
    - TXT/OCC-1/OCC-2 travel through the production agent_message kernel
      seam (wrap_outgoing_body at the sender, strict-decode + strike ledger
      + codebook render at the receiver),
    - OCC-3 travels through the native occ3_handoff tool (capture ->
      publication -> admission -> injection -> completed-use receipt).
  The adapter only orchestrates: it never reimplements a codec, never
  fabricates receipts, and never builds a second communication framework.

Topology (all four arms identical):
  parent P (--print one-shot, campaign model) spawns speaker S and
  listener L via its rlm skill with a trivial, verifiable stand-by prompt;
  the harness then delivers the real prompts to S and L verbatim through
  the production CLI `send` command (exact harness bytes, no model relay),
  S sends/hands off to sibling L, L's follow-up turn produces the answer,
  and the harness scores L's final assistant text against the gold answer.

Modes:
  selftest            offline fixture self-test (no model, no network, no CLI)
  run --dry           plan + validate the full selected matrix offline
  run --live          the real thing (requires the explicit --live flag)

Outputs:
  research/taskbench/four_way_runs.jsonl     one record per (arm, task)
  research/taskbench/four_way_results.json   aggregates incl. strict metrics
Dry modes write four_way_dry_runs.jsonl / four_way_dry_results.json instead,
so live results are never polluted by fixture data.

Usage (live, exactly what the executor runs):
  python research/taskbench/four_way_adapter.py run --live ^
    --cli "C:/Users/openclawuser/optimus-assistant-main-preparation-20261012-030555/candidate/optimus-0.1.44-main-20261109-122335/bin/optimus-rust.exe" ^
    --model "dgx-glm53-occ3/glm-5.3" ^
    --arms txt,occ1,occ2,occ3 --tasks-per-family 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # worktree root
sys.path.insert(0, str(ROOT / "prime-agent-runtime" / "src"))

from rlm.occ1 import Occ1Error, Occ1Message  # noqa: E402  (same codec the runner uses)

# ---------------------------------------------------------------------------
# Constants (mirror the production sources named in the design doc)
# ---------------------------------------------------------------------------

CAMPAIGN_ID = "occ3-bench-opt-20261109"
TRIAL_SCHEMA = "optimus.four-way-trial.v1"
ARMS = ("txt", "occ1", "occ2", "occ3")
FAMILIES = ("A", "B", "C", "D", "E")
SEED = 42
OCC0_WORD_LIMIT = 120

# crates/pi-coding-agent/src/core/occ/selection.rs
SMALL_MESSAGE_CHAR_FLOOR = 220
OCC1_TOKENS_PER_CHAR = 0.4336
SILO_REFERENCE_TOKEN_RATIO = 0.6032
CODEBOOK_TOKENS = 161.0
EXPECTED_REMAINING_MESSAGES = 8

# crates/pi-coding-agent/src/core/agent_messages.rs
DEFAULT_AGENT_MESSAGE_MAX_CHARS = 16384
OCC1_HEADER = "OCC/1"
OCC2_HEADER = "OCC/2"
OCC2_FALLBACK_SENTINEL = "#FALLBACK-OCC1"
AGENT_MESSAGE_CUSTOM_TYPE = "agent_message"
OCC2_DELIVERY_KINDS = (
    "symbolic",
    "fallback_consumed",
    "occ1_native",
    "occ2_disabled_by_strikes",
    "text",
)
OCC3_DELIVERY_LATENT_NATIVE = "latent_native"
OCC3_DELIVERY_PENDING = "pending"
# extensions/sglang-occ3/benchmarks/bench_occ3.py:944-978: dead legacy routes.
OCC3_LEGACY_ROUTES = ("/occ3/v1/send", "/occ3/v1/recv", "/occ3/v1/release")

DEFAULT_CLI = (
    r"C:\Users\openclawuser\optimus-assistant-main-preparation-20261012-030555"
    r"\candidate\optimus-0.1.44-main-20261109-122335\bin\optimus-rust.exe"
)
DEFAULT_MODEL = "dgx-glm53-occ3/glm-5.3"
# CLI subcommands that select the daemon via --daemon-socket AFTER the subcommand
# name (a global flag BEFORE a subcommand demotes the rest to a run-mode prompt).
DEFAULT_DAEMON_SOCKET = r"\\.\pipe\optimus-assistant"
DAEMON_SUBCOMMANDS = {"list", "send", "rename", "stop", "attach", "agents"}

TASKS_FILE = ROOT / "research" / "tasks" / "tasks_dev.json"
CHECKERS_FILE = ROOT / "research" / "tasks" / "checkers.py"
RUNS_JSONL = HERE / "four_way_runs.jsonl"
RESULTS_JSON = HERE / "four_way_results.json"
DRY_RUNS_JSONL = HERE / "four_way_dry_runs.jsonl"
DRY_RESULTS_JSON = HERE / "four_way_dry_results.json"

OCC2_OCC1_PREFERENCE = {"schema": "optimus.occ-session.v1", "occ2Enabled": True, "occ3Enabled": False}
OCC1_OCC2_OFF_PREFERENCE = {"schema": "optimus.occ-session.v1", "occ2Enabled": False, "occ3Enabled": False}

# Run5 evidence (2026-11-10, run_id 20261110-001625-7ed39d): one-shot children
# archive themselves right after their READY turn, so the later CLI `send`
# cannot resolve them ("Unknown active session: fwL0001", by name AND id).
# Probe-verified: a session holding a non-blocking keep-alive bash handle
# stays live and roster-active (the held-parent pattern). The stand-by task
# therefore makes each child launch that same handle before replying READY.
STANDBY_PROMPT = (
    "Stand by for the benchmark. First keep your session alive: from your "
    "Python kernel launch a NON-BLOCKING long-lived shell handle with exactly "
    "h = bash('ping -n 7200 127.0.0.1') (it returns immediately with a handle; "
    "do NOT wait for it, do NOT inspect it, do NOT close it). Then reply "
    "READY and nothing else."
)

# Run5 evidence: parents finish their turn without writing the literal DONE
# marker (transcript frozen after the thinking block, or a prose reply), so
# spawn completion also accepts a quiet or idle parent once both children
# have been discovered.
PARENT_QUIET_S = 60.0

# ---------------------------------------------------------------------------
# Task loading + checkers (reused verbatim from the existing framework)
# ---------------------------------------------------------------------------


def load_tasks() -> list[dict]:
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return data["tasks"]


def load_checkers():
    ns: dict = {}
    exec(CHECKERS_FILE.read_text(encoding="utf-8"), ns)
    return ns["check_answer"]


def pick_tasks(tasks: list[dict], per_family: int) -> list[dict]:
    selected: list[dict] = []
    for fam in FAMILIES:
        fam_tasks = [t for t in tasks if t["family"] == fam]
        selected.extend(fam_tasks[:per_family])
    return selected


# ---------------------------------------------------------------------------
# Scoring: historical family metrics + separately-named strict metrics
# ---------------------------------------------------------------------------


def lenient_answer_variants(answer: str, family: str) -> list[str]:
    """The runner.py lenient diagnostics; never the headline."""
    variants = [answer]
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if lines:
        variants.append(lines[-1])
    if family == "D":
        normalized = answer.replace("\n", ",").replace(";", ",")
        variants.append(normalized)
        if lines:
            variants.append(",".join(lines))
    return variants


def score_answer(task: dict, answer: str | None) -> dict:
    """Historical (checker) + strict byte-equality metrics, kept separate."""
    check_answer = load_checkers()
    out = {
        "historical_correct": None,
        "historical_lenient_correct": None,
        "strict_exact_match": None,
        "strict_answer_bytes": None,
    }
    if answer is None:
        return out
    stripped = answer.strip()
    out["strict_answer_bytes"] = len(stripped.encode("utf-8"))
    out["historical_correct"] = bool(check_answer(task, stripped))
    out["historical_lenient_correct"] = any(
        check_answer(task, v) for v in lenient_answer_variants(stripped, task["family"])
    )
    # Strict: byte-for-byte equality with the gold answer after leading/trailing
    # whitespace strip only. No whitespace collapse, no casefold, no set logic,
    # no numeric tolerance. Deliberately harder than the historical checker.
    out["strict_exact_match"] = stripped == str(task["gold_answer"]).strip()
    return out


def strict_field_match(speaker_occ1_body: str | None, recorded_wire_body: str | None) -> bool | None:
    """Wire-fidelity strictness for OCC-1/OCC-2 arms: the speaker-authored
    OCC-1 field set must survive the round trip byte-identically.

    For a plain OCC-1 delivery the durable wire body IS the authored body.
    For a dual OCC-2 delivery the wire body is `symbolic + sentinel + occ1`;
    the embedded OCC-1 section must decode to the same field list."""
    if speaker_occ1_body is None or recorded_wire_body is None:
        return None
    try:
        expected = Occ1Message.decode(speaker_occ1_body)
    except Exception:
        return None
    candidate = recorded_wire_body
    if candidate.startswith(OCC2_HEADER) and f"\n{OCC2_FALLBACK_SENTINEL}\n" in candidate:
        candidate = candidate.split(f"\n{OCC2_FALLBACK_SENTINEL}\n", 1)[1]
    try:
        got = Occ1Message.decode(candidate)
    except Exception:
        return False
    return expected.fields == got.fields


def strict_wire_fidelity(wire_body: str) -> bool | None:
    """Wire-fidelity strictness for OCC-1/OCC-2 arms on the DELIVERED body.

    Plain OCC-1: the wire body must parse as a valid complete contract.
    Dual OCC-2: the embedded OCC-1 fallback section must parse as a valid
    complete contract (the symbolic/fallback agreement itself is enforced
    by the production strict decode; a disagreement shows up as the
    fallback_consumed delivery kind plus a recorded strike)."""
    if not wire_body:
        return None
    if wire_body.startswith(OCC2_HEADER):
        if f"\n{OCC2_FALLBACK_SENTINEL}\n" not in wire_body:
            return False
        section = wire_body.split(f"\n{OCC2_FALLBACK_SENTINEL}\n", 1)[1]
    elif wire_body.startswith(OCC1_HEADER):
        section = wire_body
    else:
        return None
    ok, _error, _count = validate_occ1_contract(section)
    return ok


# ---------------------------------------------------------------------------
# OCC-1 fixture synthesis + codec validation (offline, no model)
# ---------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^\s*\[([A-Z])(\d+)\]\s*(.*?)\s*$", re.M)


def _fixture_facts(sender_input: str) -> list[str]:
    """Family-aware fact extraction for the dry fixture.

    Families A/D/E carry [F<n>] ids, family C carries [S<n>] step ids, and
    family B is a free-form state/event log: its non-empty lines become the
    findings. Live trials never use this; the speaker model authors its own
    complete contract from the same briefing."""
    facts: list[str] = []
    for match in _FACT_LINE.finditer(sender_input):
        facts.append(match.group(3))
    if facts:
        return facts
    for line in sender_input.splitlines():
        line = line.strip()
        if line:
            facts.append(line[:200])
    return facts


def synthesize_occ1_body(task: dict) -> str:
    """Deterministic OCC-1 contract fixture carrying every briefing fact.

    Used by dry mode to validate the codec path and to predict the OCC-2
    selection floor; never used as a live message body."""
    lines = [OCC1_HEADER, f"TASK={task['task_id']}", "KIND=TASK", "STATUS=pending"]
    for index, fact in enumerate(_fixture_facts(task["sender_input"]), start=1):
        lines.append(f"F{index}={fact}")
    return "\n".join(lines) + "\n"


def validate_occ1_contract(body: str) -> tuple[bool, str | None, int]:
    """Parse with the production-mirrored codec and check contract rules."""
    try:
        message = Occ1Message.decode(body)
    except Occ1Error as exc:
        return False, f"occ1 decode: {exc}", 0
    for required in ("TASK", "KIND", "STATUS"):
        if not message.has(required):
            return False, f"missing required field {required}", 0
    if message.get("KIND") not in (
        "TASK", "PROGRESS", "RESULT", "EVIDENCE", "REQUEST_DETAIL",
        "ACK", "CORRECTION", "BLOCKED", "CANCELLED",
    ):
        return False, "unknown KIND", 0
    finding_indexes = sorted(
        int(m.group(1)) for m in re.finditer(r"^F(\d+)=", body, re.M)
    )
    if finding_indexes and finding_indexes != list(range(1, len(finding_indexes) + 1)):
        return False, "finding indexes not contiguous from 1", 0
    return True, None, len(message.fields)


def predict_occ2_selection(occ1_body: str) -> dict:
    """Mirror selection.rs policy v1 (development path) offline.

    Live behaviour is decided by the production selector; this prediction
    only tells the harness whether the OCC-2 arm is expected to engage, so
    a surprise is visible as an effective-label mismatch."""
    chars = len(occ1_body.rstrip("\n"))
    if chars < SMALL_MESSAGE_CHAR_FLOOR:
        return {"predicted": "occ1", "reason": "message below size floor", "chars": chars}
    savings = (1.0 - SILO_REFERENCE_TOKEN_RATIO) * OCC1_TOKENS_PER_CHAR * chars
    break_even = CODEBOOK_TOKENS / savings if savings > 0 else float("inf")
    if savings <= 0:
        return {"predicted": "occ1", "reason": "no expected savings", "chars": chars}
    if EXPECTED_REMAINING_MESSAGES < break_even:
        return {"predicted": "occ1", "reason": "codebook not amortized", "chars": chars}
    return {
        "predicted": "occ2",
        "reason": "selected on expected savings",
        "chars": chars,
        "expected_savings_tokens": round(savings, 1),
    }


def wire_accounting(body: str) -> dict:
    """Byte accounting for the wire forms (occ2.rs sizes)."""
    occ1_bytes = len(body.encode("utf-8"))
    dual = None
    if body.startswith(OCC1_HEADER):
        # Symbolic section estimate: sym-v1 shortens keys; the true dual size
        # is measured live from the durable delivery record. The dry figure
        # uses the conservative occ1*2.09 corpus factor for planning only.
        dual = int(occ1_bytes * 2.09)
    return {
        "occ1_bytes": occ1_bytes,
        "occ1_chars": len(body.rstrip("\n")),
        "dual_estimate_bytes": dual,
    }


# ---------------------------------------------------------------------------
# Guards (fail-the-run conditions)
# ---------------------------------------------------------------------------


def guard_legacy_410(stage_outputs: list[str]) -> bool:
    """Any HTTP 410 mentioning a legacy /occ3/v1 route fails the run."""
    for text in stage_outputs:
        if not text:
            continue
        if "410" in text and any(route in text for route in OCC3_LEGACY_ROUTES):
            return True
        if "Gone" in text and any(route in text for route in OCC3_LEGACY_ROUTES):
            return True
    return False


def guard_native_receipt(arm: str, occ3_delivery: dict | None) -> tuple[bool, str | None]:
    """A native arm without a completed-use receipt fails the run."""
    if arm != "occ3":
        return False, None
    if not occ3_delivery:
        return True, "occ3 arm has no occ3Delivery marker at all"
    kind = occ3_delivery.get("kind")
    receipt = occ3_delivery.get("receiptSha256") or occ3_delivery.get("receipt_sha256")
    completed = occ3_delivery.get("completedUse", occ3_delivery.get("completed_use"))
    if kind != OCC3_DELIVERY_LATENT_NATIVE or not receipt or not completed:
        return True, (
            f"occ3 arm delivery marker kind={kind!r} receipt={bool(receipt)} "
            f"completed_use={completed!r}; completed-use receipt missing"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt)):
        return True, "occ3 receipt digest is not 64-hex sha256"
    return False, None


def guard_mislabel(arm: str, effective_label: str, native_success: bool | None) -> str | None:
    """An OCC-1/text fallback must never be labelled OCC-3."""
    if arm != "occ3":
        return None
    if effective_label == "OCC3" and native_success is not True:
        return f"arm occ3 labelled {effective_label} without proven native success"
    return None


def guard_synthetic_backend(collected_texts: list[str], model: str) -> str | None:
    """Stub/synthetic backend detection on the observed identities."""
    joined = "\n".join(t for t in collected_texts if t)
    for marker in ("offline-synthetic-fixture", "synthetic-fixture", "fixture-key", "mock-connector"):
        if marker.lower() in joined.lower():
            return f"synthetic backend marker {marker!r} observed"
    if re.search(r"\bgpu_free_proven\"\s*:\s*false\b", joined):
        return "gpu_free_proven false observed"
    if "synthetic" in model.lower() or "fixture" in model.lower():
        return f"model selector {model!r} looks synthetic"
    return None


# ---------------------------------------------------------------------------
# Prompts (the only model-visible arm differences)
# ---------------------------------------------------------------------------


def parent_spawn_prompt(speaker_name: str, listener_name: str) -> str:
    return f"""You are the setup agent for a two-agent benchmark trial. Your only job is to spawn two subagents with the rlm skill, keep your session alive, and then finish your turn.

In your Python kernel run exactly these two spawns (both children inherit your model; do not request a different model):

handle_l = await rlm({STANDBY_PROMPT!r}, name={listener_name!r})
handle_s = await rlm({STANDBY_PROMPT!r}, name={speaker_name!r})

Do not modify the sub-task text. Do not send the children any messages. Do not inspect them. After both admissions return, keep your session alive by launching a NON-BLOCKING long-lived shell handle in your Python kernel with exactly:

h = bash('ping -n 7200 127.0.0.1')

It returns immediately with a handle. Do NOT wait for it, do NOT inspect it, do NOT close it. Then reply with exactly:
DONE {speaker_name}=<its rlm_child_id> {listener_name}=<its rlm_child_id>"""


def listener_setup_prompt(task: dict, speaker_name: str) -> str:
    return f"""This message is from the benchmark operator, not from Agent A. It sets up your task.

You are Agent B, the receiver in a two-agent protocol. Your sibling Agent A (session name: {speaker_name}) will next send you ONE communication: it may be plain English text, OCC-1 structured KEY=VALUE text, symbolic OCC-2 text with a codebook legend, or a native state handoff attached to a control message. Use ONLY the content Agent A's communication carries.

Your question, to be answered ONLY from Agent A's content:
{task['receiver_task']}

Rules:
- Until Agent A's content has arrived, reply only READY.
- When it has arrived, answer the question. Your final response must be ONLY the answer: a value, a name, a number, or a comma-separated list, exactly as the question requests. No explanation, no prefix, no quotes.
- Never guess from your own assumptions; if A's content does not contain the answer, reply exactly: UNKNOWN"""


def speaker_prompt_txt(task: dict, listener_name: str) -> str:
    return f"""You are Agent A, the sender in a two-agent protocol. Agent B (your sibling, session name: {listener_name}) will receive only your message and must answer a question about this briefing. You do not know which detail B will need, so convey the complete content of the briefing.

Briefing (your sender input):
{task['sender_input']}

Write your message in plain English. Hard limit: at most {OCC0_WORD_LIMIT} words. No greeting, no commentary, no markdown.

Then send it from your Python kernel with exactly:
await agent_message.send(message, receiver_role="sibling", receiver_name={listener_name!r})

where message is your plain-English text as one string. If the send fails with a transport error or sibling-not-found error, retry the same send up to 5 times. Do not send anything else. After a successful send, reply with exactly: SENT"""


def _speaker_occ1_authoring(task: dict) -> str:
    return f"""Encode your message as a complete OCC-1 contract carrying EVERY fact of the briefing:
- Line 1 is exactly: OCC/1
- TASK={task['task_id']}
- KIND=TASK
- STATUS=pending
- One F<n> line per briefing fact, indexes contiguous from 1, value = the fact text (no newlines inside a value).
- Optional final line: NEXT=answer the receiver question when it arrives

The message must parse as a valid OCC-1 contract (OCC/1 header, KEY=VALUE lines, uppercase keys)."""


def speaker_prompt_occ1(task: dict, listener_name: str) -> str:
    return f"""You are Agent A, the sender in a two-agent protocol. Agent B (your sibling, session name: {listener_name}) will receive only your message and must answer a question about this briefing. You do not know which detail B will need, so convey the complete content of the briefing.

Briefing (your sender input):
{task['sender_input']}

{_speaker_occ1_authoring(task)}

Then send it from your Python kernel with exactly:
await agent_message.send(message, receiver_role="sibling", receiver_name={listener_name!r})

where message is your OCC-1 contract text as one string. If the send fails with a transport error or sibling-not-found error, retry the same send up to 5 times. Do not send anything else. After a successful send, reply with exactly: SENT"""


def speaker_prompt_occ3(task: dict, listener_name: str) -> str:
    return f"""You are Agent A, the sender in a two-agent protocol. Agent B (your sibling, session name: {listener_name}) must answer a question about this briefing using only what you transfer. You do not know which detail B will need, so transfer the complete content of the briefing.

Briefing (your sender input):
{task['sender_input']}

Transfer the briefing to your sibling with the NATIVE occ3_handoff tool (do NOT use agent_message.send for the briefing):
1. First call occ3_handoff with receiver_role="sibling" and receiver_name={listener_name!r}. It returns capture_armed and sends nothing.
2. In your NEXT response: emit the complete briefing facts as your assistant text (all facts, plainly and completely, no tools in that response), then call occ3_handoff again with the same receiver_role and receiver_name, with no other tool calls in that response.
3. The second call publishes the captured turn and admits the handoff. After admission, do not repeat the handoff. Reply with exactly: HANDED_OFF

If the tool reports the server or receiver is busy, call occ3_handoff again after a short wait to re-arm, then emit the facts again, as the tool text instructs. If the handoff falls back to ordinary messaging, follow the tool's fallback instruction and report exactly: NATIVE_FALLBACK"""


def speaker_prompt(arm: str, task: dict, listener_name: str) -> str:
    if arm == "txt":
        return speaker_prompt_txt(task, listener_name)
    if arm == "occ1":
        return speaker_prompt_occ1(task, listener_name)
    if arm == "occ2":
        # Identical authoring to occ1: the arm difference is purely the
        # transport preference (speaker occ.json occ2Enabled), so the wire
        # comparison is controlled.
        return speaker_prompt_occ1(task, listener_name)
    if arm == "occ3":
        return speaker_prompt_occ3(task, listener_name)
    raise ValueError(f"unknown arm {arm}")


def arm_occ2_preference(arm: str) -> dict:
    if arm == "occ1":
        return dict(OCC1_OCC2_OFF_PREFERENCE)  # the `/occ 2 off` state
    return dict(OCC2_OCC1_PREFERENCE)  # default-on state


# ---------------------------------------------------------------------------
# Session transcript parsing (read-only, on-disk JSONL artifacts)
# ---------------------------------------------------------------------------


def parse_session_records(session_file: str | Path) -> list[dict]:
    """Tolerant reader for a daemon session JSONL file.

    Each line is a file entry whose `message` member carries the agent
    message (role/content/customType/details). Unknown lines are skipped."""
    records: list[dict] = []
    try:
        text = Path(session_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("message"), dict):
            records.append(entry["message"])
        elif isinstance(entry, dict):
            records.append(entry)
    return records


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _role(message: dict) -> str:
    return str(message.get("role", ""))


def extract_deliveries(records: list[dict]) -> list[dict]:
    """All agent_message custom records with their durable details."""
    out = []
    for message in records:
        if _role(message) != "custom":
            continue
        if message.get("customType") != AGENT_MESSAGE_CUSTOM_TYPE:
            continue
        details = message.get("details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                details = None
        out.append(
            {
                "text": _message_text(message),
                "details": details if isinstance(details, dict) else {},
            }
        )
    return out


def extract_answer_after(records: list[dict], message_id: str | None) -> str | None:
    """The last assistant text after the last delivered agent message.

    `message_id` pins the delivered record when present; without it the
    last agent-message record is used (single-delivery trials)."""
    deliveries = extract_deliveries(records)
    if not deliveries:
        return None
    if message_id:
        pinned = [d for d in deliveries if d["details"].get("id") == message_id]
        anchor_index = None
        for index, message in enumerate(records):
            if (
                _role(message) == "custom"
                and message.get("customType") == AGENT_MESSAGE_CUSTOM_TYPE
            ):
                details = message.get("details")
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except json.JSONDecodeError:
                        details = {}
                if isinstance(details, dict) and details.get("id") == message_id:
                    anchor_index = index
        if anchor_index is None and pinned:
            return None
    else:
        anchor_index = None
        for index in range(len(records) - 1, -1, -1):
            if (
                _role(records[index]) == "custom"
                and records[index].get("customType") == AGENT_MESSAGE_CUSTOM_TYPE
            ):
                anchor_index = index
                break
    if anchor_index is None:
        anchor_index = -1
    answer = None
    for message in records[anchor_index + 1:]:
        if _role(message) == "assistant":
            text = _message_text(message).strip()
            if text:
                answer = text
    return answer


def find_speaker_send_receipt(records: list[dict]) -> dict | None:
    """Best-effort speaker-side view: the last emitted agent-message receipt."""
    found = None
    for message in records:
        text = _message_text(message)
        if "deliveryStatus" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = json.loads(text[start:end])
                if isinstance(parsed, dict) and "deliveryStatus" in parsed:
                    found = parsed
            except (ValueError, json.JSONDecodeError):
                continue
    return found


# ---------------------------------------------------------------------------
# CLI orchestration (live mode)
# ---------------------------------------------------------------------------


class CliError(RuntimeError):
    pass


class LiveHarness:
    def __init__(self, cli: str, model: str, run_dir: Path, answer_timeout_s: float,
                 send_timeout_s: float, verbose: bool = False,
                 sessions_dir: str | None = None,
                 daemon_socket: str = r"\\.\pipe\optimus-assistant",
                 spawn_timeout_s: float = 900.0,
                 parent_quiet_s: float = PARENT_QUIET_S):
        self.cli = cli
        self.model = model
        self.run_dir = run_dir
        self.answer_timeout_s = answer_timeout_s
        self.send_timeout_s = send_timeout_s
        self.verbose = verbose
        self.sessions_dir = sessions_dir
        self.daemon_socket = daemon_socket
        self.spawn_timeout_s = spawn_timeout_s
        self.parent_quiet_s = parent_quiet_s
        self.stage_outputs: list[str] = []
        # Transport-scoped subset fed to guard_legacy_410 (see _run): the full
        # stage_outputs log also collects roster dumps, whose unrelated
        # sessions' firstMessages can mention legacy routes + 410.
        self.guard_outputs: list[str] = []
        self._parent_proc: subprocess.Popen | None = None
        # Named sessions owned by the trial currently in flight (children then
        # parent); swept best-effort at cleanup and on any trial failure.
        self._trial_sessions: list[str] = []

    def _provider_model_flags(self) -> list[str]:
        # The CLI takes --provider and --model as SEPARATE flags; a compound
        # "provider/model" passed to --model is silently ignored and the run
        # falls back to the default provider (verified live 2026-11-09 run2:
        # compound -> codex/gpt-6-astra refused; split form -> rc 0 in 8.9s).
        if "/" in self.model:
            prov, _, mid = self.model.partition("/")
            return ["--provider", prov, "--model", mid]
        return ["--model", self.model]

    def _build_cmd(self, args: list[str]) -> list[str]:
        # Position-aware daemon-socket wiring (root-probed 2026-11-09): a global
        # flag BEFORE a subcommand demotes the rest to a run-mode PROMPT, so for
        # subcommands the flag must come after the subcommand name; before
        # -p/--print it is a plain run-mode global flag; everything else (e.g.
        # --version, which prints EMPTY when combined with the flag) stays bare.
        cmd = [self.cli, *args]
        if self.daemon_socket and args:
            if args[0] in DAEMON_SUBCOMMANDS:
                cmd = [self.cli, args[0], "--daemon-socket", self.daemon_socket, *args[1:]]
            elif args[0] in ("-p", "--print"):
                cmd = [self.cli, "--daemon-socket", self.daemon_socket, *args]
        return cmd

    def _run(self, args: list[str], timeout: float, check: bool = True,
             transport: bool = False) -> subprocess.CompletedProcess:
        # transport=True tags a command that exercises the message-transport
        # path (the CLI `send` turns). Only transport outputs reach
        # self.guard_outputs, the list guard_legacy_410 scans, so roster dumps
        # (list --all --json carries every session's firstMessage, and an
        # unrelated idle session mentions legacy /occ3/v1 routes + 410)
        # cannot trip the guard; stage_outputs keeps the full diagnostic log.
        cmd = self._build_cmd(args)
        if self.verbose:
            print(f"  [cli] {' '.join(args[:6])}{' ...' if len(args) > 6 else ''}", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            note = f"timeout: {' '.join(args[:4])}: {exc}"
            self.stage_outputs.append(note)
            if transport:  # a transport timeout string may legitimately mention routes
                self.guard_outputs.append(note)
            raise CliError(f"CLI timeout: {' '.join(args[:4])}") from exc
        outputs = [proc.stdout or "", proc.stderr or ""]
        self.stage_outputs.extend(outputs)
        if transport:
            self.guard_outputs.extend(outputs)
        if check and proc.returncode != 0:
            raise CliError(
                f"CLI failed ({proc.returncode}): {' '.join(args[:4])}: "
                f"{(proc.stderr or proc.stdout or '')[:400]}"
            )
        return proc

    # -- discovery ---------------------------------------------------------

    def list_sessions(self, include_saved: bool = True) -> list[dict]:
        args = ["list"]
        if include_saved:
            args.append("--all")
        args.append("--json")
        proc = self._run(args, timeout=120, check=False)
        try:
            data = json.loads(proc.stdout.strip() or "{}")
        except json.JSONDecodeError:
            return []
        sessions = data.get("sessions") or data.get("agents") or []
        if isinstance(sessions, dict):
            sessions = list(sessions.values())
        return [s for s in sessions if isinstance(s, dict)]

    def preflight(self) -> dict:
        if not os.path.isfile(self.cli):
            raise CliError(f"CLI binary not found: {self.cli}")
        proc = self._run(["--version"], timeout=60, check=False)
        status = self._run(["list", "--json"], timeout=120, check=False)
        try:
            status_data = json.loads(status.stdout.strip() or "{}")
        except json.JSONDecodeError:
            status_data = {}
        return {
            "cli": self.cli,
            "cli_version": (proc.stderr or proc.stdout or "").strip()[:200],
            "daemon_status": status_data,
        }

    def session_artifact_dir(self, session_file: str) -> Path:
        # session_manager.rs get_session_artifact_path_for_file:
        # <dirname(dirname(session_file))>/session-artifacts/<id>/
        path = Path(session_file)
        session_id = path.stem
        return path.parent.parent / "session-artifacts" / session_id

    def write_occ_preference(self, session_file: str, preference: dict) -> Path:
        artifact_dir = self.session_artifact_dir(session_file)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target = artifact_dir / "occ.json"
        target.write_text(json.dumps(preference) + "\n", encoding="utf-8")
        return target

    def _kill_held_parent(self) -> None:
        """Kill the held parent CLI if it is still running (idempotent)."""
        if self._parent_proc is not None and self._parent_proc.poll() is None:
            self._parent_proc.kill()

    def _done_marker_seen(self, session_file: str | Path | None, marker: str) -> bool:
        """Pure transcript check (no subprocess): True when an assistant
        message in the session file contains the DONE marker."""
        if not session_file:
            return False
        records = parse_session_records(session_file)
        return any(
            _role(m) == "assistant" and marker in _message_text(m)
            for m in records
        )

    @staticmethod
    def _transcript_signature(session_file: str | Path | None) -> tuple[int, float] | None:
        """(size, mtime) of a session transcript; None when absent/unreadable."""
        if not session_file:
            return None
        try:
            stat = Path(session_file).stat()
        except OSError:
            return None
        return (stat.st_size, stat.st_mtime)

    @staticmethod
    def _spawn_settled(both_seen_at: float | None, sig: tuple[int, float] | None,
                       sig_at: float | None, idle_polls: int, now: float,
                       quiet_s: float) -> bool:
        """Spawn-completion decision once both children are discovered.

        The DONE marker is checked separately as the fast path. Here the
        parent counts as settled when it stopped producing output: the roster
        reported activity "idle" on two consecutive polls (5 s apart; run5
        mode (a): the parent never writes any final assistant text), or its
        transcript signature was unchanged for quiet_s AND at least quiet_s
        passed since both children appeared (run5 mode (b): prose reply)."""
        if idle_polls >= 2:
            return True
        if both_seen_at is None or sig is None or sig_at is None:
            return False
        return (now - both_seen_at >= quiet_s) and (now - sig_at >= quiet_s)

    def sweep_trial_sessions(self) -> None:
        """Best-effort stop of the current trial's named sessions (idempotent).

        Stand-by children hold non-blocking keep-alive bash handles, so they
        stay roster-ACTIVE after their READY turn and never archive
        themselves: they must be stopped by name. Order: children first, then
        the parent session, then the held parent CLI process."""
        for name in list(getattr(self, "_trial_sessions", []) or []):
            self._run(["stop", name], timeout=120, check=False)
        self._kill_held_parent()

    # -- one trial ---------------------------------------------------------

    def run_trial(self, arm: str, task: dict, trial_index: int) -> dict:
        t_start = time.perf_counter()
        self.stage_outputs = []  # full log, kept for diagnostics
        self.guard_outputs = []  # transport-scoped log; guards are evaluated per trial
        trial_id = f"t{trial_index:04d}-{arm}-{task['task_id']}"
        speaker_name = f"fwS{trial_index:04d}"
        listener_name = f"fwL{trial_index:04d}"
        parent_name = f"fwP{trial_index:04d}"
        # Children hold keep-alive handles once spawned, so every named
        # session of this trial must be swept at cleanup and on failure.
        self._trial_sessions = [speaker_name, listener_name, parent_name]
        chain: list[dict] = []
        record = {
            "schema": TRIAL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "run_id": None,
            "trial_id": trial_id,
            "task_id": task["task_id"],
            "family": task["family"],
            "difficulty": task["difficulty"],
            "seed": SEED,
            "mode": arm,
            "arm": arm,
            "requested_protocol": arm,
            "speaker": self.model,
            "listener": self.model,
            "model": self.model,
            "native_method": None,
            "native_success": None,
            "native_fallback_label": None,
            "effective_protocol": None,
            "effective_label_alpha": None,
            "identities": {},
            "delivery": {},
            "chain": chain,
            "a": {},
            "b": {},
            "strict": {},
            "historical": {},
            "wire": {},
            "tokens": {},
            "terminal_error": None,
            "guards": {
                "legacy_410": False,
                "native_receipt_missing": False,
                "mislabel": None,
                "synthetic_backend": None,
            },
        }

        def stage(name: str, fn):
            t0 = time.perf_counter()
            try:
                value = fn()
                chain.append({"stage": name, "status": "ok", "duration_s": round(time.perf_counter() - t0, 2)})
                return value
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                chain.append({
                    "stage": name,
                    "status": "error",
                    "duration_s": round(time.perf_counter() - t0, 2),
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                })
                raise

        # 1. parent spawn turn (held parent) ----------------------------------
        def _spawn_parent():
            # A mid-trial failure skips the cleanup stage and would orphan the
            # previous attempt's held parent CLI; sweep any leftover first.
            self._kill_held_parent()
            _spawn_started_at = time.time()
            before = {s.get("activeSessionId") or s.get("sessionId") for s in self.list_sessions()}
            prompt = parent_spawn_prompt(speaker_name, listener_name)
            # A one-shot --print session ARCHIVES when its turn completes,
            # which aborts its children and makes the parent unreachable for
            # rename (run4 evidence 2026-11-09). The parent now holds a
            # non-blocking keep-alive bash handle, so it stays live: launch it
            # in the background, poll for completion, and kill the process
            # handle at cleanup. Completion = both children discovered AND the
            # parent no longer producing output: the DONE marker (fast path)
            # or, since run5 (2026-11-10) showed parents that never write it,
            # a quiet transcript or an idle roster entry.
            self._parent_proc = subprocess.Popen(
                self._build_cmd(["--print", *self._provider_model_flags(), "--cwd", str(self.run_dir), prompt]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            done_marker = f"DONE {speaker_name}"
            sessions = []
            children: dict[str, dict] = {}
            parent = None
            both_seen_at: float | None = None
            parent_sig: tuple[int, float] | None = None
            parent_sig_at: float | None = None
            parent_idle_polls = 0
            deadline = time.perf_counter() + self.spawn_timeout_s
            while time.perf_counter() < deadline:
                sessions = self.list_sessions()
                children = {}
                parent = None
                for entry in sessions:
                    name = entry.get("sessionName") or entry.get("name") or ""
                    sid = entry.get("activeSessionId") or entry.get("sessionId")
                    if name in (speaker_name, listener_name) and sid not in before:
                        children[name] = entry
                    elif sid not in before and entry.get("cwd") == str(self.run_dir):
                        parent = entry
                if parent is None:
                    # fall back: the remaining new session
                    new = [s for s in sessions if (s.get("activeSessionId") or s.get("sessionId")) not in before]
                    parent = next((s for s in new if (s.get("sessionName") or s.get("name")) not in (speaker_name, listener_name)), None)
                if len(children) == 2 and both_seen_at is None:
                    both_seen_at = time.perf_counter()
                if parent is not None and len(children) == 2 and self._done_marker_seen(parent.get("sessionFile"), done_marker):
                    break  # fast path: explicit DONE marker
                if parent is not None and both_seen_at is not None:
                    now = time.perf_counter()
                    sig = self._transcript_signature(parent.get("sessionFile"))
                    if sig is None or sig != parent_sig:
                        parent_sig = sig
                        parent_sig_at = now
                    if str(parent.get("activity", "")).lower() == "idle":
                        parent_idle_polls += 1
                    else:
                        parent_idle_polls = 0
                    if self._spawn_settled(
                        both_seen_at, sig, parent_sig_at, parent_idle_polls,
                        now, self.parent_quiet_s,
                    ):
                        break  # settled path: parent stopped producing output
                time.sleep(5)
            else:
                self._kill_held_parent()
                # Never leak this attempt's roster-active children (they now
                # hold keep-alive handles and would survive the parent kill).
                for child_name in (speaker_name, listener_name):
                    if child_name in children:
                        self._run(["stop", child_name], timeout=120, check=False)
                raise CliError(
                    "parent spawn: no completion signal (DONE marker, quiet transcript, "
                    "or idle parent) in time"
                )
            missing = {speaker_name, listener_name} - set(children)
            if missing and self.sessions_dir and os.path.isdir(self.sessions_dir):
                # Filesystem fallback: children spawned since the snapshot
                # whose first user message is the stand-by prompt.
                for path in Path(self.sessions_dir).rglob("*.jsonl"):
                    try:
                        stat = path.stat()
                        if stat.st_mtime < _spawn_started_at - 5:
                            continue
                        if path.stat().st_size > 4_000_000:
                            continue
                        records = parse_session_records(path)
                        users = [_message_text(m) for m in records if _role(m) == "user"]
                        if users and users[0].strip() == STANDBY_PROMPT:
                            name = speaker_name if speaker_name in missing else listener_name
                            children[name] = {
                                "sessionFile": str(path),
                                "sessionName": name,
                            }
                            missing.discard(name)
                            if not missing:
                                break
                    except OSError:
                        continue
            if missing:
                raise CliError(f"child sessions not discovered after spawn: {sorted(missing)}")
            if parent is None:
                # fall back: the remaining new session
                new = [s for s in sessions if (s.get("activeSessionId") or s.get("sessionId")) not in before]
                parent = next((s for s in new if (s.get("sessionName") or s.get("name")) not in (speaker_name, listener_name)), None)
            if parent is None:
                raise CliError("parent session not discovered after spawn")
            self._run(["rename", str(parent.get("activeSessionId") or parent.get("sessionId")), parent_name], timeout=60)
            return {
                "parent": parent,
                "speaker": children[speaker_name],
                "listener": children[listener_name],
            }

        family = stage("spawn_family", _spawn_parent)
        speaker_entry = family["speaker"]
        listener_entry = family["listener"]
        speaker_file = speaker_entry.get("sessionFile") or speaker_entry.get("session_file")
        listener_file = listener_entry.get("sessionFile") or listener_entry.get("session_file")
        if not speaker_file or not listener_file:
            raise CliError("child session files not reported by list --json")
        record["identities"] = {
            "parent_session": family["parent"].get("activeSessionId") or family["parent"].get("sessionId"),
            "speaker_session": speaker_entry.get("activeSessionId") or speaker_entry.get("sessionId"),
            "listener_session": listener_entry.get("activeSessionId") or listener_entry.get("sessionId"),
            "speaker_name": speaker_name,
            "listener_name": listener_name,
            "speaker_session_file": str(speaker_file),
            "listener_session_file": str(listener_file),
            "client": "four_way_adapter",
            "server": "optimus daemon (production)",
        }

        # 2. verify the spawn relay stayed verbatim ---------------------------
        def _verify_standby():
            for label, path in (("speaker", speaker_file), ("listener", listener_file)):
                records = parse_session_records(path)
                users = [_message_text(m) for m in records if _role(m) == "user"]
                first = users[0].strip() if users else ""
                if first != STANDBY_PROMPT:
                    raise CliError(
                        f"{label} spawn prompt relay corrupted: {first[:120]!r}"
                    )
            return True

        stage("verify_spawn_relay", _verify_standby)

        # 3. speaker transport preference (arm control) -----------------------
        def _write_preference():
            target = self.write_occ_preference(str(speaker_file), arm_occ2_preference(arm))
            return str(target)

        record["delivery"]["speaker_occ_preference"] = stage(
            "speaker_occ_preference", _write_preference
        )

        # 4. listener setup turn ---------------------------------------------
        def _listener_setup():
            self._run(
                ["send", listener_name, "--message", listener_setup_prompt(task, speaker_name)],
                timeout=600,
                transport=True,
            )
            deadline = time.perf_counter() + self.send_timeout_s
            while time.perf_counter() < deadline:
                records = parse_session_records(listener_file)
                deliveries = extract_deliveries(records)
                assistants = [
                    _message_text(m).strip() for m in records if _role(m) == "assistant" and _message_text(m).strip()
                ]
                if deliveries and assistants and assistants[-1].upper().startswith("READY"):
                    return True
                time.sleep(5)
            raise CliError("listener did not acknowledge setup in time")

        stage("listener_setup", _listener_setup)

        # 5. speaker send turn -------------------------------------------------
        def _speaker_send():
            prompt = speaker_prompt(arm, task, listener_name)
            self._run(
                ["send", speaker_name, "--message", prompt],
                timeout=900,
                transport=True,
            )
            return True

        stage("speaker_send", _speaker_send)

        # 6. wait for the delivery + answer ----------------------------------
        def _await_answer():
            deadline = time.perf_counter() + self.answer_timeout_s
            delivery = None
            answer = None
            while time.perf_counter() < deadline:
                records = parse_session_records(listener_file)
                deliveries = extract_deliveries(records)
                if len(deliveries) >= 2:  # setup + A's message
                    delivery = deliveries[-1]
                    answer = extract_answer_after(records, delivery["details"].get("id"))
                    if answer is not None:
                        return {"delivery": delivery, "answer": answer}
                time.sleep(5)
            if delivery is not None:
                raise CliError("delivery present but listener never answered")
            raise CliError("no delivered agent message observed in listener transcript")

        outcome = stage("await_answer", _await_answer)
        delivery = outcome["delivery"]
        answer = outcome["answer"]
        details = delivery["details"]
        occ2_delivery = details.get("occ2Delivery") or {}
        occ3_delivery = details.get("occ3Delivery")
        wire_body = details.get("message") or ""
        record["delivery"].update(
            {
                "message_id": details.get("id"),
                "wire_body": wire_body,
                "occ2_delivery_kind": occ2_delivery.get("kind"),
                "occ2_pair": occ2_delivery.get("pair"),
                "occ2_strike_recorded": occ2_delivery.get("strikeRecorded", occ2_delivery.get("strike_recorded")),
                "occ3_delivery": occ3_delivery,
                "delivered_prompt_chars": len(delivery["text"]),
            }
        )
        record["native_method"] = (occ3_delivery or {}).get("methodId") or (occ3_delivery or {}).get("method_id")

        # effective channel: from the durable receiver record, never the ask
        if arm == "occ3":
            receipt_ok, receipt_error = guard_native_receipt(arm, occ3_delivery)
            record["guards"]["native_receipt_missing"] = receipt_ok
            if occ3_delivery and (
                occ3_delivery.get("kind") == OCC3_DELIVERY_LATENT_NATIVE
            ):
                record["native_success"] = True
                record["effective_protocol"] = "occ3-native"
                record["effective_label_alpha"] = "OCC3"
            else:
                record["native_success"] = False
                # fallback: the control body is an ordinary text delivery
                record["native_fallback_label"] = "text"
                record["effective_protocol"] = "text"
                record["effective_label_alpha"] = "TXT"
            if receipt_error:
                record["terminal_error"] = f"native_receipt_missing: {receipt_error}"
        else:
            kind = occ2_delivery.get("kind") or "unknown"
            record["effective_protocol"] = kind
            record["effective_label_alpha"] = {
                "symbolic": "OCC2",
                "fallback_consumed": "OCC2fallback",
                "occ1_native": "OCC1",
                "occ2_disabled_by_strikes": "OCC1",
                "text": "TXT",
            }.get(kind, kind)

        # 7. speaker transcript (send-side view, best effort) ------------------
        speaker_records = parse_session_records(speaker_file)
        receipt = find_speaker_send_receipt(speaker_records)
        record["a"] = {
            "send_receipt": receipt,
            "speaker_final_text": next(
                (
                    _message_text(m).strip()
                    for m in reversed(speaker_records)
                    if _role(m) == "assistant" and _message_text(m).strip()
                ),
                None,
            ),
        }

        # 8. wire accounting + scoring -----------------------------------------
        record["wire"] = wire_accounting(wire_body or "")
        record["wire"]["wire_bytes"] = len((wire_body or "").encode("utf-8"))
        record["wire"]["predicted_occ2_selection"] = (
            predict_occ2_selection(synthesize_occ1_body(task)) if arm == "occ2" else None
        )
        record["wire"]["strict_wire_fidelity"] = strict_wire_fidelity(wire_body or "")
        record["b"] = {"answer": answer, "answer_source": "listener_transcript"}
        scores = score_answer(task, answer)
        record["historical"] = {
            "correct": scores["historical_correct"],
            "lenient_correct": scores["historical_lenient_correct"],
        }
        record["strict"] = {
            "exact_match": scores["strict_exact_match"],
            "answer_bytes": scores["strict_answer_bytes"],
            "field_match": strict_wire_fidelity(wire_body or "") if arm in ("occ1", "occ2") else None,
        }

        # 9. guards -------------------------------------------------------------
        record["guards"]["legacy_410"] = guard_legacy_410(self.guard_outputs)
        record["guards"]["mislabel"] = guard_mislabel(arm, record["effective_label_alpha"] or "", record["native_success"])
        record["guards"]["synthetic_backend"] = guard_synthetic_backend(
            [json.dumps(record["delivery"]), json.dumps(record["identities"]), self.model],
            self.model,
        )
        tripped = [name for name, value in record["guards"].items() if value is True]
        if record["guards"]["mislabel"]:
            tripped.append("mislabel")
        if record["guards"]["synthetic_backend"]:
            tripped.append("synthetic_backend")
        if tripped:
            record["terminal_error"] = record["terminal_error"] or f"guard tripped: {sorted(set(tripped))}"

        record["duration_s"] = round(time.perf_counter() - t_start, 2)

        # 10. cleanup (best effort) ---------------------------------------------
        def _cleanup():
            # Children first (they hold keep-alive handles and stay
            # roster-active), then the parent session, then the held CLI.
            self.sweep_trial_sessions()
            return True

        stage("cleanup", _cleanup)
        return record


# ---------------------------------------------------------------------------
# Dry fixture self-test (no model, no network, no CLI execution)
# ---------------------------------------------------------------------------


def dry_selftest(tasks_per_family: int, cli: str, model: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1. task matrix ---------------------------------------------------------
    tasks = pick_tasks(load_tasks(), tasks_per_family)
    expected_ids = [f"{fam}-dev-{i:03d}" for fam in FAMILIES for i in range(1, tasks_per_family + 1)]
    check("task_matrix", [t["task_id"] for t in tasks] == expected_ids,
          f"{len(tasks)} tasks, ids {[t['task_id'] for t in tasks][:3]}...")

    # 2. checkers dispatch ----------------------------------------------------
    check_answer = load_checkers()
    probe = {"family": "A", "gold_answer": "sa-east"}
    check("checker_text", check_answer(probe, "sa-east") and not check_answer(probe, "SA-EAST!"))
    probe_d = {"family": "D", "gold_answer": "alpha, beta"}
    check("checker_set", check_answer(probe_d, "beta, alpha") and not check_answer(probe_d, "alpha"))
    probe_e = {"family": "E", "gold_answer": "42.0"}
    check("checker_number", check_answer(probe_e, "42") and not check_answer(probe_e, "43"))

    # 3. strict vs historical separation --------------------------------------
    mixed = {"family": "A", "gold_answer": "sa-east"}
    scores = score_answer(mixed, "  SA-EAST  ")
    check("strict_vs_historical",
          scores["historical_correct"] is True and scores["strict_exact_match"] is False,
          f"historical={scores['historical_correct']} strict={scores['strict_exact_match']}")
    scores2 = score_answer(mixed, "sa-east")
    check("strict_positive", scores2["strict_exact_match"] is True)

    # 4. prompts build for every (arm, task) ----------------------------------
    built = 0
    for arm in ARMS:
        for task in tasks:
            prompt = speaker_prompt(arm, task, "fwL0001")
            if not prompt or task["sender_input"] not in prompt:
                check(f"prompt_build_{arm}", False, task["task_id"])
                break
            if "receiver_name" not in prompt.replace("receiver_name=", ""):
                pass
            built += 1
        else:
            check(f"prompt_build_{arm}", True, f"{len(tasks)} prompts")
    check("listener_setup", all(
        listener_setup_prompt(t, "fwS0001").strip() for t in tasks
    ))
    check("parent_spawn", bool(parent_spawn_prompt("fwS0001", "fwL0001")))

    # 5. OCC-1 fixture per task: codec + contract + floor ----------------------
    occ1_ok = 0
    floor_ok = 0
    for task in tasks:
        body = synthesize_occ1_body(task)
        ok, error, field_count = validate_occ1_contract(body)
        if not ok:
            check("occ1_fixture", False, f"{task['task_id']}: {error}")
            break
        if wire_accounting(body)["occ1_chars"] >= SMALL_MESSAGE_CHAR_FLOOR:
            floor_ok += 1
        occ1_ok += 1
    else:
        check("occ1_fixture", occ1_ok == len(tasks), f"{occ1_ok}/{len(tasks)} valid")
    prediction = predict_occ2_selection(synthesize_occ1_body(tasks[0]))
    check("occ2_prediction_shape", "predicted" in prediction and "reason" in prediction)

    # 6. strict field match (wire fidelity) ------------------------------------
    body = synthesize_occ1_body(tasks[0])
    dual = f"OCC/2 D=sym-v1@1 C=0123456789ab\nT={tasks[0]['task_id']}\n#FALLBACK-OCC1\n{body}"
    check("field_match_plain", strict_field_match(body, body) is True)
    check("field_match_dual", strict_field_match(body, dual) is True)
    corrupted = body.replace("STATUS=pending", "STATUS=done")
    check("field_match_corrupt", strict_field_match(body, corrupted) is False)

    # 7. guards ------------------------------------------------------------------
    check("guard_410", guard_legacy_410(["POST /occ3/v1/send -> HTTP 410 Gone"]) is True)
    check("guard_410_clean", guard_legacy_410(["HTTP 200 ok"]) is False)
    ok, err = guard_native_receipt("occ3", None)
    check("guard_receipt_absent", ok and err is not None, str(err))
    ok, err = guard_native_receipt("occ3", {"kind": "latent_native", "receiptSha256": "a" * 64, "completedUse": True})
    check("guard_receipt_ok", not ok)
    ok, err = guard_native_receipt("occ3", {"kind": "pending"})
    check("guard_receipt_pending", ok)
    check("guard_mislabel", guard_mislabel("occ3", "OCC3", False) is not None)
    check("guard_mislabel_ok", guard_mislabel("occ3", "OCC3", True) is None)
    check("guard_mislabel_txt", guard_mislabel("txt", "TXT", None) is None)
    check("guard_synthetic", guard_synthetic_backend(["declared offline-synthetic-fixture"], model) is not None)
    check("guard_synthetic_gpu", guard_synthetic_backend(['"gpu_free_proven": false'], model) is not None)
    check("guard_synthetic_clean", guard_synthetic_backend(["normal"], model) is None)

    # 8. transcript parsing on a fixture ----------------------------------------
    fixture_session = [
        {"role": "user", "content": STANDBY_PROMPT},
        {"role": "assistant", "content": "READY"},
        {
            "role": "custom",
            "customType": AGENT_MESSAGE_CUSTOM_TYPE,
            "content": "Agent-to-agent message received.",
            "details": {
                "id": "agentmsg_test1",
                "message": "OCC/1\nTASK=t\nKIND=TASK\nSTATUS=pending\nF1=region is sa-east\n",
                "occ2Delivery": {"kind": "occ1_native", "pair": "s->l", "strikeRecorded": False},
            },
        },
        {"role": "assistant", "content": "sa-east"},
    ]
    deliveries = extract_deliveries(fixture_session)
    check("parse_deliveries", len(deliveries) == 1 and deliveries[0]["details"]["id"] == "agentmsg_test1")
    answer = extract_answer_after(fixture_session, "agentmsg_test1")
    check("parse_answer", answer == "sa-east")
    answer2 = extract_answer_after(fixture_session, None)
    check("parse_answer_unpinned", answer2 == "sa-east")

    # 9. trial record schema + JSONL round trip ----------------------------------
    trial = {
        "schema": TRIAL_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "run_id": "dry",
        "trial_id": "dry-occ1-A-dev-001",
        "task_id": "A-dev-001",
        "family": "A",
        "seed": SEED,
        "mode": "occ1",
        "arm": "occ1",
        "requested_protocol": "occ1",
        "effective_protocol": "occ1_native",
        "effective_label_alpha": "OCC1",
        "strict": {"exact_match": True, "field_match": True},
        "historical": {"correct": True},
        "guards": {"legacy_410": False, "native_receipt_missing": False, "mislabel": None, "synthetic_backend": None},
        "terminal_error": None,
    }
    line = json.dumps(trial, ensure_ascii=True)
    check("jsonl_roundtrip", json.loads(line)["trial_id"] == trial["trial_id"])

    # 10. aggregate over fixture records -------------------------------------------
    fixture_trials = []
    for arm in ARMS:
        for task in tasks[:2]:
            fixture_trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "run_id": "dry",
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "mode": arm,
                    "arm": arm,
                    "requested_protocol": arm,
                    "effective_protocol": "occ3-native" if arm == "occ3" else ("symbolic" if arm == "occ2" else ("occ1_native" if arm == "occ1" else "text")),
                    "effective_label_alpha": {"txt": "TXT", "occ1": "OCC1", "occ2": "OCC2", "occ3": "OCC3"}[arm],
                    "native_success": True if arm == "occ3" else None,
                    "strict": {"exact_match": task["family"] in ("A", "B"), "field_match": True if arm in ("occ1", "occ2") else None},
                    "historical": {"correct": True, "lenient_correct": True},
                    "wire": {"wire_bytes": 512, "occ1_bytes": 300},
                    "tokens": {"listener_prompt_tokens": None},
                    "duration_s": 42.0,
                    "terminal_error": None,
                    "guards": {"legacy_410": False, "native_receipt_missing": False, "mislabel": None, "synthetic_backend": None},
                }
            )
    agg = aggregate(fixture_trials)
    check("aggregate_cells", set(agg["cells"].keys()) == set(ARMS), str(sorted(agg["cells"].keys())))
    check("aggregate_strict", agg["cells"]["occ1"]["strict_exact_accuracy"] is not None)
    check("aggregate_native", agg["cells"]["occ3"]["native_success_rate"] == 1.0)

    # 11. CLI path configured (existence only; dry never executes it) -------------
    check("cli_path_configured", os.path.isfile(cli), cli)

    # 12. held-spawn completion + active-child standby (run5 fixes) ---------------
    check("standby_keepalive",
          "bash('ping -n 7200 127.0.0.1')" in STANDBY_PROMPT
          and "do NOT wait for it" in STANDBY_PROMPT
          and "reply ready" in STANDBY_PROMPT.lower(),
          STANDBY_PROMPT[:100])
    check("standby_embedded", repr(STANDBY_PROMPT) in parent_spawn_prompt("fwS0001", "fwL0001"),
          "parent prompt must relay the stand-by prompt verbatim")
    sig = LiveHarness._transcript_signature(Path(__file__))
    check("transcript_signature", isinstance(sig, tuple) and len(sig) == 2
          and all(isinstance(v, (int, float)) for v in sig), str(sig))
    check("transcript_signature_missing",
          LiveHarness._transcript_signature(Path("no/such/file.jsonl")) is None)
    check("spawn_settled_idle",
          LiveHarness._spawn_settled(0.0, None, None, 2, 10.0, 60.0) is True)
    check("spawn_settled_quiet",
          LiveHarness._spawn_settled(0.0, (1, 1.0), 0.0, 0, 61.0, 60.0) is True)
    check("spawn_settled_early",
          LiveHarness._spawn_settled(0.0, (1, 1.0), 0.0, 0, 59.0, 60.0) is False)
    check("spawn_settled_growing",
          LiveHarness._spawn_settled(0.0, (1, 1.0), 55.0, 0, 61.0, 60.0) is False)
    check("spawn_settled_nosig",
          LiveHarness._spawn_settled(0.0, None, None, 0, 100.0, 60.0) is False)
    check("spawn_settled_unset",
          LiveHarness._spawn_settled(None, (1, 1.0), 0.0, 1, 100.0, 60.0) is False)
    probe_harness = LiveHarness(cli="offline-probe", model="probe", run_dir=HERE,
                                answer_timeout_s=1.0, send_timeout_s=1.0)
    try:
        probe_harness.sweep_trial_sessions()  # no named sessions: no CLI call
        check("sweep_noop_offline", probe_harness._trial_sessions == [])
    except Exception as exc:  # noqa: BLE001 - selftest must report, not crash
        check("sweep_noop_offline", False, f"{type(exc).__name__}: {exc}")

    # 13. legacy-410 guard scoped to transport-command outputs ------------------
    roster_dump = json.dumps({
        "sessions": [
            {
                "sessionName": "C Clean up 2",
                "activeSessionId": "01a1c082-0000-4000-8000-000000000000",
                "firstMessage": "legacy /occ3/v1/send|recv|release return 410 (do not call them)",
            }
        ]
    })

    def _fake_subprocess_run(cmd, **_kwargs):  # offline: the CLI is never executed
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=roster_dump, stderr="")

    _real_subprocess_run = subprocess.run
    subprocess.run = _fake_subprocess_run
    try:
        probe_harness.stage_outputs = []
        probe_harness.guard_outputs = []
        # roster poll (list --all --json): same 410 + route text, not transport
        probe_harness._run(["list", "--all", "--json"], timeout=5, check=False)
        roster_only = guard_legacy_410(probe_harness.guard_outputs)
        # the SAME string as the output of a transport command must still trip
        probe_harness._run(["send", "fwL0001", "--message", "ping"], timeout=5, check=False, transport=True)
        after_transport = guard_legacy_410(probe_harness.guard_outputs)
        full_log_would_trip = guard_legacy_410(probe_harness.stage_outputs)
    finally:
        subprocess.run = _real_subprocess_run
    check(
        "guard_410_transport_scope",
        roster_only is False and after_transport is True and full_log_would_trip is True,
        f"roster_only={roster_only} after_transport={after_transport} full_log={full_log_would_trip}",
    )

    print(f"selftest: {len(checks) - len(failures)}/{len(checks)} checks passed")
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" ({detail})" if detail and not ok else ""))
    return not failures, failures


# ---------------------------------------------------------------------------
# Aggregation (historical family metrics + strict metrics, separately named)
# ---------------------------------------------------------------------------


def aggregate(trials: list[dict]) -> dict:
    def pct(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
        }

    agg = {
        "meta": {
            "campaign_id": CAMPAIGN_ID,
            "protocol": "four-way adapter: TXT / OCC-1 / OCC-2 / OCC-3-native over real daemon sessions",
            "schema": TRIAL_SCHEMA,
            "model": trials[0].get("model") if trials else None,
            "seed": SEED,
            "tasks_file": str(TASKS_FILE.relative_to(ROOT)),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_trials": len(trials),
            "notes": [
                "historical_* metrics use research/tasks/checkers.py exactly as the 2026-10-16 baseline runner did",
                "strict_* metrics are separately named byte/field equality and are never conflated with historical",
                "effective channel labels come from the receiver's durable delivery record, not the requested arm",
            ],
        },
        "cells": {},
        "per_family": {},
        "delivered_channel_distribution": {},
        "guards": {},
    }

    for arm in ARMS:
        recs = [r for r in trials if r.get("arm") == arm]
        cell = {
            "n_trials": len(recs),
            "historical_accuracy": pct([r.get("historical", {}).get("correct") for r in recs]),
            "historical_lenient_accuracy": pct([r.get("historical", {}).get("lenient_correct") for r in recs]),
            "strict_exact_accuracy": pct([r.get("strict", {}).get("exact_match") for r in recs]),
            "strict_field_match_accuracy": pct([r.get("strict", {}).get("field_match") for r in recs]),
            "terminal_error_rate": pct([r.get("terminal_error") is not None for r in recs]),
            "wire_bytes": stats([r.get("wire", {}).get("wire_bytes") for r in recs]),
            "duration_s": stats([r.get("duration_s") for r in recs]),
        }
        if arm == "occ3":
            cell["native_success_rate"] = pct([r.get("native_success") for r in recs])
            cell["native_fallback_rate"] = pct([r.get("native_success") is False for r in recs])
        kinds = {}
        for r in recs:
            kind = r.get("effective_protocol")
            kinds[kind] = kinds.get(kind, 0) + 1
        agg["delivered_channel_distribution"][arm] = kinds
        agg["cells"][arm] = cell

    for arm in ARMS:
        for fam in FAMILIES:
            recs = [r for r in trials if r.get("arm") == arm and r.get("family") == fam]
            if not recs:
                continue
            agg["per_family"][f"{arm}|{fam}"] = {
                "n": len(recs),
                "historical_accuracy": pct([r.get("historical", {}).get("correct") for r in recs]),
                "strict_exact_accuracy": pct([r.get("strict", {}).get("exact_match") for r in recs]),
            }

    guard_counts = {}
    for r in trials:
        for name, value in (r.get("guards") or {}).items():
            if value is True or (isinstance(value, str) and value):
                guard_counts[name] = guard_counts.get(name, 0) + 1
    agg["guards"] = guard_counts
    return agg


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def run_matrix(args) -> int:
    tasks = pick_tasks(load_tasks(), args.tasks_per_family)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            print(f"unknown arm {arm!r}; valid: {', '.join(ARMS)}", file=sys.stderr)
            return 2
    live = bool(args.live)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(args.run_dir) if args.run_dir else HERE / "four_way_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = RUNS_JSONL if live else DRY_RUNS_JSONL
    out_json = RESULTS_JSON if live else DRY_RESULTS_JSON

    if not live:
        print(f"DRY mode (no --live): validating {len(arms)} arms x {len(tasks)} tasks offline")
        ok, failures = dry_selftest(args.tasks_per_family, args.cli, args.model)
        if not ok:
            print("dry validation failed:", *failures, sep="\n  ", file=sys.stderr)
            return 1
        print(f"dry validation passed; live command would be:")
        print(
            f'  python research/taskbench/four_way_adapter.py run --live '
            f'--cli "{args.cli}" --model "{args.model}" '
            f'--arms {",".join(arms)} --tasks-per-family {args.tasks_per_family}'
        )
        return 0

    print(f"LIVE mode: {len(arms)} arms x {len(tasks)} tasks; run_id={run_id}")
    harness = LiveHarness(
        cli=args.cli,
        model=args.model,
        run_dir=run_dir,
        answer_timeout_s=args.answer_timeout,
        send_timeout_s=args.send_timeout,
        verbose=args.verbose,
        sessions_dir=args.sessions_dir,
        daemon_socket=args.daemon_socket,
        spawn_timeout_s=args.spawn_timeout,
    )
    preflight = harness.preflight()
    print(f"preflight: cli={Path(args.cli).name} daemon_keys={sorted(preflight['daemon_status'])[:6]}")

    completed: set[tuple[str, str]] = set()
    if out_jsonl.exists() and args.resume:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    completed.add((rec.get("arm"), rec.get("task_id")))
                except json.JSONDecodeError:
                    continue

    trials: list[dict] = []
    index = 0
    for arm in arms:
        for task in tasks:
            if (arm, task["task_id"]) in completed:
                print(f"[skip] {arm} {task['task_id']} (already recorded)")
                continue
            index += 1
            print(f"[{arm}] {task['task_id']} ...", flush=True)
            record = None
            for attempt in range(1, args.max_trials + 1):
                try:
                    record = harness.run_trial(arm, task, index)
                    record["run_id"] = run_id
                    record["attempt_count"] = attempt
                    break
                except (CliError,) as exc:
                    print(f"  trial error (attempt {attempt}/{args.max_trials}): {exc}", flush=True)
                    # The failed attempt's children hold keep-alive handles and
                    # stay roster-active; sweep them before the next attempt.
                    harness.sweep_trial_sessions()
                    index += 1
            if record is None:
                record = {
                    "schema": TRIAL_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "run_id": run_id,
                    "trial_id": f"t{index:04d}-{arm}-{task['task_id']}",
                    "task_id": task["task_id"],
                    "family": task["family"],
                    "difficulty": task.get("difficulty"),
                    "seed": SEED,
                    "mode": arm,
                    "arm": arm,
                    "requested_protocol": arm,
                    "model": args.model,
                    "terminal_error": f"trial_failed_after_{args.max_trials}_attempts",
                    "guards": {
                        "legacy_410": guard_legacy_410(harness.guard_outputs),
                        "native_receipt_missing": arm == "occ3",
                        "mislabel": None,
                        "synthetic_backend": None,
                    },
                }
            trials.append(record)
            with out_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            err = record.get("terminal_error")
            answer = (record.get("b") or {}).get("answer")
            print(
                f"  -> effective={record.get('effective_protocol')} "
                f"strict={record.get('strict', {}).get('exact_match')} "
                f"historical={record.get('historical', {}).get('correct')} "
                f"answer={str(answer)[:40]!r} error={err}",
                flush=True,
            )

    # Final belt-and-braces: never leave the last trial's sessions or its
    # held parent CLI running.
    harness.sweep_trial_sessions()

    agg = aggregate(trials)
    out_json.write_text(json.dumps(agg, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {out_json} and appended {len(trials)} trials to {out_jsonl}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd")

    st = sub.add_parser("selftest", help="offline fixture self-test (no model, no network)")
    st.add_argument("--tasks-per-family", type=int, default=10)
    st.add_argument("--cli", default=os.environ.get("FOUR_WAY_CLI", DEFAULT_CLI))
    st.add_argument("--model", default=DEFAULT_MODEL)

    run = sub.add_parser("run", help="run the four-way matrix")
    run.add_argument("--live", action="store_true",
                    help="LIVE mode: real daemon sessions, real model calls. Without this flag only the dry validation runs.")
    run.add_argument("--dry", action="store_true", help="alias for omitting --live")
    run.add_argument("--arms", default="txt,occ1,occ2,occ3")
    run.add_argument("--tasks-per-family", type=int, default=10)
    run.add_argument("--cli", default=os.environ.get("FOUR_WAY_CLI", DEFAULT_CLI))
    run.add_argument("--model", default=os.environ.get("FOUR_WAY_MODEL", DEFAULT_MODEL))
    run.add_argument("--run-dir", default=None)
    run.add_argument("--daemon-socket",
                    default=os.environ.get("FOUR_WAY_DAEMON_SOCKET", DEFAULT_DAEMON_SOCKET),
                    help="daemon pipe the adapter attaches to (default: the campaign daemon)")
    run.add_argument("--sessions-dir", default=os.environ.get("OPTIMUS_SESSIONS_DIR"),
                     help="daemon sessions dir; enables the filesystem fallback for child discovery")
    run.add_argument("--answer-timeout", type=float, default=900.0)
    run.add_argument("--spawn-timeout", type=float, default=900.0,
                    help="seconds to wait for the held parent's completion signal (DONE marker, quiet transcript, or idle parent)")
    run.add_argument("--send-timeout", type=float, default=600.0)
    run.add_argument("--max-trials", type=int, default=2,
                    help="infra-level trial attempts (never retries wrong answers)")
    run.add_argument("--resume", action="store_true", help="skip (arm, task) pairs already in the JSONL")
    run.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.cmd == "selftest":
        ok, failures = dry_selftest(args.tasks_per_family, args.cli, args.model)
        return 0 if ok else 1
    if args.cmd == "run":
        return run_matrix(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
