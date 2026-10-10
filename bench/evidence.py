"""Local artifact binding and selection rules for scoring and attribution."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from bench.protocol import (PROVENANCE_SCHEMA, canonical_hash, evaluator_identity,
                            question_set_hash, validate_question_protocol)

SCORE_FLAGS = {None, "missing_record", "driver_error", "empty_answer", "eval_error", "unknown_eval_type"}


QUESTION_BINDING_SOURCE = "caller_declared_scoring_question"
QUESTION_BINDING_SCHEMA = "memory-bench-caller-question-declaration/1"
MAX_PROTOCOL_CHARS = 256
IDENTITY_FIELDS = ("run_id", "benchmark", "variant", "env_id")


def prepare_question_declaration(question: dict) -> dict:
    """Prepare a NEW driver's gold-free input; never annotate old predictions.

    The full scoring-question hash is opaque to the driver. Echoing it is a
    caller declaration, not independent proof of gold or actual presentation.
    """
    family = question.get("family")
    if family not in {"locomo", "locomo_plus", "longmemeval", "lme_v2"}:
        raise ValueError("unsupported scoring-question family")
    validate_question_protocol(question, family)
    visible = question.get("presented_question", question.get("question"))
    if not isinstance(visible, str) or not visible.strip():
        raise ValueError("question requires explicit presented text")
    for key in ("qid", "env_id"):
        if not isinstance(question.get(key), str) or not question[key]:
            raise ValueError("question requires nonempty qid/env_id")
    return {
        "qid": question["qid"], "env_id": question["env_id"], "question": visible,
        "question_protocol": question["question_protocol"],
        "scoring_protocol": question["eval"]["protocol"],
        "question_sha256": canonical_hash(question),
    }


def run_identity(record) -> tuple[str, str, str, str]:
    if not isinstance(record, dict) or any(
        not isinstance(record.get(key), str) or not record[key].strip() for key in IDENTITY_FIELDS
    ):
        raise ValueError("run/snapshot identity requires nonempty run_id/benchmark/variant/env_id")
    return tuple(record[key] for key in IDENTITY_FIELDS)


def effective_variant(records, requested=None):
    values = {record.get("variant") for record in records.values()}
    if len(values) > 1:
        raise ValueError("selected records have ambiguous variants")
    value = next(iter(values), requested)
    if requested is not None and value != requested:
        raise ValueError("selected record variant conflicts with explicit selection")
    return value


def validate_driver_question(record, question, family) -> None:
    if record is None:
        return  # no prediction is being bound or rescored
    identity = run_identity(record)
    expected = prepare_question_declaration(question)
    if identity[1] != family or identity[3] != expected["env_id"]:
        raise ValueError("run/question benchmark/environment identity mismatch")
    for key in ("question_protocol", "scoring_protocol"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or not 1 <= len(value) <= MAX_PROTOCOL_CHARS:
            raise ValueError("strict scoring requires bounded caller question protocol declarations")
        if value != expected[key]:
            raise ValueError("caller question protocol declaration mismatch")
    digest = record.get("question_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("strict scoring requires a 64-hex caller scoring-question hash")
    if digest.lower() != expected["question_sha256"]:
        raise ValueError("caller scoring-question hash mismatch")
    if record.get("question_binding_source") != QUESTION_BINDING_SOURCE:
        raise ValueError("strict scoring requires the caller-declared binding source marker")
    if record.get("qid") != expected["qid"] or record.get("question") != expected["question"]:
        raise ValueError("declared driver question text/id mismatch")


def jsonl_records(data: bytes) -> list:
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def read_questions(directory) -> tuple[list, list]:
    questions, files, seen = [], [], set()
    for path in sorted(Path(directory).glob("qs_*.jsonl")):
        data = path.read_bytes()
        files.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest()})
        for row in jsonl_records(data):
            qid = row.get("qid")
            if not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError("question qids must be nonempty unique strings")
            seen.add(qid)
            questions.append(row)
    if not questions:
        raise ValueError("no common-format questions found")
    return questions, files


def select_run_records(data: bytes, variant=None, dedup="last") -> tuple[dict, int]:
    if dedup not in {"error", "first", "last"}:
        raise ValueError("invalid dedup policy")
    selected, duplicates = {}, 0
    variants = set()
    for row in jsonl_records(data):
        if variant is not None and row.get("variant") != variant:
            continue
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid:
            raise ValueError("run qids must be nonempty strings")
        value = row.get("variant")
        if value is not None and not isinstance(value, str):
            raise ValueError("run variants must be strings or null")
        variants.add(value)
        if row.get("answer_text") is not None and not isinstance(row["answer_text"], str):
            raise ValueError("answer_text must be a string or null")
        if qid in selected:
            duplicates += 1
            if dedup == "error":
                raise ValueError("duplicate run qid; choose an explicit first/last policy")
            if dedup == "first":
                continue
        selected[qid] = row
    if variant is None and len(variants) > 1:
        raise ValueError("mixed run variants require an explicit --variant")
    return selected, duplicates


def native_failure(record) -> dict:
    record = record or {}
    stop = str(record.get("stop_reason") or "").lower()
    return {"driver_error": bool(record.get("error")),
            "stop_error": stop in {"error", "aborted", "cancelled", "canceled"},
            "timed_out": bool(record.get("timed_out")) or stop in {"timeout", "timed_out"}}


def record_binding(record):
    return None if record is None else {"variant": record.get("variant"), "sha256": canonical_hash(record)}


def make_provenance(all_questions, question_files, questions, records, variant, dedup, identity) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA,
        "question_set_sha256": question_set_hash(all_questions),
        "question_files": question_files,
        "evaluator": identity,
        "driver_question_binding": {"schema": QUESTION_BINDING_SCHEMA, "source": QUESTION_BINDING_SOURCE,
            "claim": "caller-declared local artifact consistency only; not actual presentation or gold proof"},
        "selection": {"variant": variant, "effective_variant": effective_variant(records, variant),
                      "dedup": dedup, "qids": [q["qid"] for q in questions]},
        "selected_records": {q["qid"]: record_binding(records.get(q["qid"])) for q in questions},
    }


def validate_scores(scores, raw_run, all_questions, question_files, *, family, variant, dedup,
                    allow_legacy_unverified=False):
    if not isinstance(scores, dict) or scores.get("schema") != "memory-bench-scores/1":
        raise ValueError("unsupported score schema")
    if scores.get("bench") != family:
        raise ValueError("score benchmark mismatch")
    meta = scores.get("run") or {}
    if meta.get("benchmark") not in {None, family}:
        raise ValueError("score benchmark metadata mismatch")
    if variant is not None and meta.get("variant") not in {None, variant}:
        raise ValueError("score variant metadata mismatch")
    records, _ = select_run_records(raw_run, variant, dedup)
    whole_hash = hashlib.sha256(raw_run).hexdigest()
    if scores.get("run_sha256") is not None and scores["run_sha256"] != whole_hash:
        raise ValueError("score run hash mismatch")
    rows = scores.get("results")
    if not isinstance(rows, list):
        raise ValueError("score results must be a list")
    by_qid = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str) or row["qid"] in by_qid:
            raise ValueError("score qids must be unique strings")
        score = row.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("scores must be finite numbers in [0,1]")
        if row.get("flag") not in SCORE_FLAGS:
            raise ValueError("unsupported score flag")
        by_qid[row["qid"]] = row
    provenance = scores.get("scoring_provenance")
    if provenance is None:
        if not allow_legacy_unverified:
            raise ValueError("legacy scores need --allow-legacy-unverified; no stage inference is allowed")
        return all_questions, records, by_qid, "unverified-legacy"
    if not isinstance(provenance, dict) or provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported scoring provenance schema")
    if scores.get("run_sha256") != whole_hash:
        raise ValueError("bound scores require the exact run hash")
    sample = next(iter(records.values()), None)
    expected_meta = ({} if sample is None else {"run_id": sample.get("run_id"),
                     "variant": sample.get("variant"), "benchmark": sample.get("benchmark", family)})
    if meta != expected_meta:
        raise ValueError("score sampled-run metadata mismatch")
    if provenance.get("question_set_sha256") != question_set_hash(all_questions):
        raise ValueError("score question-set hash mismatch")
    if provenance.get("question_files") != question_files:
        raise ValueError("score question-file hash mismatch")
    identity = provenance.get("evaluator") or {}
    try:
        budget = identity["judge_settings"]["locomo_plus"]["max_tokens"]
    except (KeyError, TypeError):
        raise ValueError("missing evaluator request identity") from None
    if identity != evaluator_identity(budget):
        raise ValueError("score evaluator/protocol fingerprint mismatch")
    declared_binding = provenance.get("driver_question_binding") or {}
    if (declared_binding.get("schema") != QUESTION_BINDING_SCHEMA
            or declared_binding.get("source") != QUESTION_BINDING_SOURCE):
        raise ValueError("strict scores require a caller question declaration policy")
    selection = provenance.get("selection") or {}
    if selection.get("effective_variant") != effective_variant(records, variant):
        raise ValueError("score effective variant mismatch")
    if selection.get("variant") != variant or selection.get("dedup") != dedup:
        raise ValueError("score variant/dedup selection mismatch")
    qids = selection.get("qids")
    if not isinstance(qids, list) or any(not isinstance(qid, str) for qid in qids) or len(qids) != len(set(qids)):
        raise ValueError("invalid selected question identity")
    question_map = {q["qid"]: q for q in all_questions}
    if set(qids) != set(by_qid) or not set(qids).issubset(question_map):
        raise ValueError("score selected-question coverage mismatch")
    questions = [question_map[qid] for qid in qids]
    expected_bindings = {q["qid"]: record_binding(records.get(q["qid"])) for q in questions}
    if provenance.get("selected_records") != expected_bindings:
        raise ValueError("selected run-record hash mismatch")
    effective_variants = {record.get("variant") for record in records.values()}
    if effective_variants and meta.get("variant") not in effective_variants:
        raise ValueError("score selected-variant metadata mismatch")
    for question in questions:
        validate_question_protocol(question, family)
        qid = question["qid"]
        row, record = by_qid[qid], records.get(qid)
        validate_driver_question(record, question, family)
        binding = expected_bindings[qid]
        if (row.get("question_sha256") != canonical_hash(question)
                or row.get("run_record_sha256") != (binding["sha256"] if binding else None)
                or row.get("scoring_protocol") != question["eval"]["protocol"]):
            raise ValueError("score row content/protocol binding mismatch")
        for field, expected in (("env_id", question.get("env_id")), ("category", question.get("category")),
                                ("eval_type", question["eval"].get("type"))):
            if row.get(field) != expected:
                raise ValueError("score row metadata mismatch")
        expected_flag = ("missing_record" if record is None else "driver_error" if record.get("error")
                         else "empty_answer" if not (record.get("answer_text") or "").strip() else None)
        if expected_flag is not None and row.get("flag") != expected_flag:
            raise ValueError("score failure flag conflicts with selected run")
        if expected_flag is None and row.get("flag") in {"missing_record", "driver_error", "empty_answer"}:
            raise ValueError("score failure flag conflicts with selected run")
        if row.get("flag") and row["score"] != 0:
            raise ValueError("flagged score must retain the zero convention")
        if record is not None and row.get("native_failure") != native_failure(record):
            raise ValueError("native failure diagnostics mismatch")
    return questions, records, by_qid, "verified-local-artifact-consistency"
