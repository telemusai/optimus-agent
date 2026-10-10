"""Content identities for the publishable local evaluation protocol.

Hashes bind declared local artifacts, not provider identity or model truth.
No files or environment values are read at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

EVALUATOR_VERSION = "optimus-memory-evaluation/2.0.0"
LOCAL_SCORING_VERSION = "optimus-local-scoring/1.0.0"
PROVENANCE_SCHEMA = "memory-bench-scoring-provenance/2"
CAPTURE_PROTOCOL_VERSION = "optimus-memory-capture/temporal-authority/1.0.0"
QUESTION_PROTOCOLS = {
    "locomo": "optimus-locomo-conversion/2.0.0",
    "locomo_plus": "optimus-normalized-cues/1.0.0",
    "longmemeval": "optimus-longmemeval-adapter/2.0.0",
    "lme_v2": "optimus-lme-v2-adapter/2.0.0",
}
SCORING_PROTOCOLS = {
    "locomo": LOCAL_SCORING_VERSION,
    "locomo_plus": LOCAL_SCORING_VERSION,
    "longmemeval": "optimus-longmemeval-dgx-port/2.0.0",
    "lme_v2": "optimus-lme-v2-dgx-port/2.0.0",
}
SOURCE_FILES = (
    "protocol.py", "evidence.py", "io.py", "eval/evaluate.py", "eval/deterministic.py", "eval/porter.py",
    "eval/judges.py", "eval/local_scoring.py", "eval/local_adapters.py",
    "converters/common.py", "converters/locomo.py", "converters/locomo_plus.py",
    "converters/longmemeval.py", "converters/lme_v2.py",
)


def canonical_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def question_set_hash(questions) -> str:
    return canonical_hash(sorted(questions, key=lambda question: question["qid"]))


def annotate_question(question: dict, family: str) -> dict:
    question["family"] = family
    question["question_protocol"] = QUESTION_PROTOCOLS[family]
    question["eval"]["protocol"] = SCORING_PROTOCOLS[family]
    return question


def validate_question_protocol(question: dict, family: str) -> None:
    if (question.get("family") != family
            or question.get("question_protocol") != QUESTION_PROTOCOLS[family]
            or (question.get("eval") or {}).get("protocol") != SCORING_PROTOCOLS[family]):
        raise ValueError("question protocol mismatch; regenerate common inputs with this adapter")


def evaluator_identity(lp_max_tokens=None) -> dict:
    budget = int(os.environ.get("LP_JUDGE_MAX_TOKENS", "512")) if lp_max_tokens is None else lp_max_tokens
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("LP judge token budget must be a positive integer")
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    descriptor = {
        "version": EVALUATOR_VERSION,
        "question_protocols": QUESTION_PROTOCOLS,
        "scoring_protocols": SCORING_PROTOCOLS,
        "source_sha256": sources,
        "judge_settings": {
            "provider": "dgx-glm53", "model": "glm-5.3", "jev": False,
            "longmemeval": {"temperature": 0, "max_tokens": 2048},
            "locomo_plus": {"temperature": 0, "max_tokens": budget},
            "lme_v2": {"temperature": None, "max_tokens": 4096},
        },
    }
    return dict(descriptor, sha256=canonical_hash(descriptor))
