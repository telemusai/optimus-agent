"""Read caller-normalized cue environments; no upstream stitching/duration code.

Input: locomo_plus_normalized.json containing [{"env": <common environment>,
"qid": str, "question": str, "evidence_cue": str, "category": str}].
The caller supplies licensed, already ordered sessions and cue locations.
Raw LoCoMo-Plus time-gap strings and conversation stitching are not supported.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from bench.protocol import annotate_question, QUESTION_PROTOCOLS
from .common import SEED, get_data_root, family_dir, clean_outputs, stable_sample, env_stats, write_json, write_jsonl

FAMILY = "locomo_plus"
DEV_N = 120


def load_rows():
    path = Path(get_data_root()) / "locomo_plus_normalized.json"
    if not path.is_file():
        raise ValueError("LoCoMo-Plus raw duration/stitching source is not distributed; "
                         "supply explicit locomo_plus_normalized.json with preordered common environments")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("normalized cues must be a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str) or not row["qid"]:
            raise ValueError("normalized cue requires a string qid")
        if row["qid"] in seen:
            raise ValueError("normalized cue qids must be unique")
        seen.add(row["qid"])
        env = row.get("env")
        if not isinstance(env, dict) or not isinstance(env.get("env_id"), str):
            raise ValueError("normalized cue requires an explicit common environment")
        if not isinstance(env.get("sessions"), list):
            raise ValueError("normalized cue requires ordered sessions")
        for session in env["sessions"]:
            if not isinstance(session.get("events"), list):
                raise ValueError("normalized sessions require ordered events")
            for event in session["events"]:
                if event.get("role") not in {"user", "assistant"} or not isinstance(event.get("text"), str):
                    raise ValueError("normalized events require a text and role")
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("question", "evidence_cue")):
            raise ValueError("normalized cue requires question and evidence_cue strings")
    if len({row["env"]["env_id"] for row in rows}) != len(rows):
        raise ValueError("normalized cue environments must be isolated per question")
    return rows


def build_question(row):
    return annotate_question({
        "qid": row["qid"], "env_id": row["env"]["env_id"], "question": row["question"],
        "answer": None, "evidence": list(row["env"].get("ground_truth_refs", [])),
        "category": row.get("category", "normalized-cue"),
        "eval": {"type": "llm_judge", "params": {"evidence_cue": row["evidence_cue"]}},
    }, FAMILY)


def _emit(rows, directory):
    clean_outputs(directory)
    environments = []
    for row in rows:
        env = copy.deepcopy(row["env"])
        env["family"] = FAMILY
        env["question_protocol"] = QUESTION_PROTOCOLS[FAMILY]
        env_id = env["env_id"]
        # Do not let untrusted identifiers create subdirectories or escape roots.
        if any(char in env_id for char in ("/", "\\", ":")) or env_id in {".", "..", ""}:
            raise ValueError("normalized environment id must be a simple filename component")
        file, qs_file = f"env_{env_id}.json", f"qs_{env_id}.jsonl"
        write_json(str(Path(directory) / file), env)
        write_jsonl(str(Path(directory) / qs_file), [build_question(row)])
        environments.append(dict(env_stats(env), file=file, qs_file=qs_file, n_questions=1))
    chars = sum(item["chars"] for item in environments)
    manifest = {
        "family": FAMILY, "protocol": QUESTION_PROTOCOLS[FAMILY], "seed": SEED,
        "sampling": {"method": "stable qid sample of caller-normalized inputs; not upstream stratification"},
        "n_envs": len(rows), "n_questions": len(rows), "total_env_chars": chars,
        "estimated_ingest_tokens": chars // 4, "envs": environments,
    }
    write_json(str(Path(directory) / "manifest.json"), manifest)
    return manifest


def build(dev_only=True):
    rows = load_rows()
    by_qid = {row["qid"]: row for row in rows}
    chosen = stable_sample(SEED, sorted(by_qid), DEV_N)
    manifest = _emit([by_qid[qid] for qid in chosen], family_dir(FAMILY))
    full_dir = family_dir(FAMILY, full=True)
    if dev_only:
        write_json(str(Path(full_dir) / "manifest.json"), {
            "family": FAMILY, "protocol": QUESTION_PROTOCOLS[FAMILY],
            "n_questions": len(rows), "questions": [build_question(row) for row in rows],
        })
    else:
        _emit(rows, full_dir)
    return manifest
