"""LoCoMo-shaped data adapter using Optimus local protocols, not upstream code.

Numeric categories retain their input identity. Scoring and presentation use
optimus-local-scoring/1.0.0 and optimus-locomo-conversion/2.0.0. Category-5
alternatives are shuffled BEFORE letter labels are assigned. Regenerate both
common questions and runs; historical uniform-letter records are not compatible.
Data acquisition/licensing is the caller's responsibility.
"""

from __future__ import annotations

import json
import os
import random
import re

from bench.protocol import annotate_question

from .common import (clean_outputs, SEED, get_data_root, family_dir, write_json, write_jsonl,
                     parse_locomo_datetime, iso, stable_sample, largest_remainder,
                     tolerant_dia_refs, env_stats)

FAMILY = "locomo"
DEV_N = 300

CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop",
             5: "adversarial"}


def load_rows():
    with open(os.path.join(get_data_root(), "locomo10.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _turn_text(m: dict, speaker: str) -> str:
    text = f"{speaker}: {m['text']}"
    if m.get("blip_caption"):
        text += f" (shared image: {m['blip_caption']})"
    return text


def build_env(conv: dict, all_qa_parsed: list) -> dict:
    cv = conv["conversation"]
    sa, sb = cv["speaker_a"], cv["speaker_b"]
    sess_keys = sorted((k for k in cv if re.fullmatch(r"session_\d+", k)),
                       key=lambda k: int(k.split("_")[1]))
    sessions = []
    ext_ids = set()
    for sk in sess_keys:
        ts = iso(parse_locomo_datetime(cv[sk + "_date_time"]))
        events = []
        for m in cv[sk]:
            speaker = m["speaker"]
            role = "user" if speaker == sa else "assistant"
            events.append({"role": role, "text": _turn_text(m, speaker),
                           "ext_id": m["dia_id"], "ts": ts})
            ext_ids.add(m["dia_id"])
        sessions.append({"session_id": sk, "ts": ts, "events": events})
    gt_refs = sorted({r for qa in all_qa_parsed for r in qa["evidence"]
                      if r in ext_ids})
    return {
        "env_id": conv["sample_id"],
        "sessions": sessions,
        "ground_truth_refs": gt_refs,
        "family": FAMILY,
        "notes": {
            "speaker_a": sa, "speaker_b": sb,
            "speaker_a_role": "user",
            "n_sessions": len(sessions),
            "multimodal_turns": sum(
                1 for sk in sess_keys for m in cv[sk] if m.get("blip_caption")),
        },
    }


def build_question(conv: dict, idx: int, qa: dict, env_ext_ids=None) -> dict:
    category = int(qa["category"])
    if category not in CAT_NAMES:
        raise ValueError("unsupported numeric question category")
    qid = f"{conv['sample_id']}_q{idx}"
    references = tolerant_dia_refs(qa.get("evidence", []))
    missing = [] if env_ext_ids is None else [ref for ref in references if ref not in env_ext_ids]
    question = {
        "qid": qid, "env_id": conv["sample_id"], "question": qa["question"],
        "answer": None if qa.get("answer") is None else str(qa["answer"]),
        "evidence": [ref for ref in references if ref not in missing],
        "category": str(category), "category_name": CAT_NAMES[category],
    }
    if missing:
        question["unresolved_evidence_refs"] = missing
    if category == 5:
        alternatives = [
            {"text": "Insufficient information in this record.", "abstention": True},
            {"text": str(qa.get("adversarial_answer", "")), "abstention": False},
        ]
        random.Random(f"{SEED}:{qid}").shuffle(alternatives)
        mapping = {label: option["text"] for label, option in zip(("a", "b"), alternatives)}
        expected = next(label for label, option in zip(("a", "b"), alternatives) if option["abstention"])
        options = [f"({label}) {text}" for label, text in mapping.items()]
        question.update({
            "options": options, "option_map": mapping, "abstention_label": expected,
            "presented_question": qa["question"] + "\nSelect one alternative:\n" + "\n".join(options),
            "eval": {"type": "abstain_f1", "params": {"option_map": mapping, "abstention_label": expected}},
        })
    else:
        params = {"variant": "comma_multi_answer" if category == 1 else "plain"}
        if category == 3:
            params["gold_truncate_at"] = ";"
        question["eval"] = {"type": "f1", "params": params}
    return annotate_question(question, FAMILY)


def parse_all_qas(rows):
    """All QAs with tolerant evidence parsing + per-question metadata."""
    out = []
    for conv in rows:
        for idx, qa in enumerate(conv["qa"]):
            out.append({
                "conv": conv["sample_id"],
                "idx": idx,
                "qid": f"{conv['sample_id']}_q{idx}",
                "category": qa["category"],
                "evidence": tolerant_dia_refs(qa.get("evidence", [])),
                "raw_evidence": qa.get("evidence", []),
                "has_answer": qa.get("answer") is not None,
            })
    return out


def dev_selection(all_qas):
    """300 QAs: category totals via largest remainder, then per-conversation
    proportional allocation inside each category (guarantees coverage of all
    10 conversations)."""
    by_cat = {}
    for qa in all_qas:
        by_cat.setdefault(qa["category"], []).append(qa)
    cat_alloc = largest_remainder(DEV_N, {c: len(v) for c, v in by_cat.items()})
    selected = []
    for cat, qas in sorted(by_cat.items()):
        conv_w = {}
        for qa in qas:
            conv_w[qa["conv"]] = conv_w.get(qa["conv"], 0) + 1
        conv_alloc = largest_remainder(cat_alloc[cat], conv_w)
        for conv, k in sorted(conv_alloc.items()):
            cell = sorted(qa["qid"] for qa in qas if qa["conv"] == conv)
            selected += [(conv, q) for q in stable_sample(SEED, cell, k)]
    return selected, cat_alloc, {}


def build(dev_only=True):
    rows = load_rows()
    out_dir = family_dir(FAMILY)
    clean_outputs(out_dir)
    all_qas = parse_all_qas(rows)
    selected, cat_alloc, _ = dev_selection(all_qas)
    sel_by_conv = {}
    for conv, qid in selected:
        sel_by_conv.setdefault(conv, []).append(qid)

    qa_lookup = {qa["qid"]: qa for qa in all_qas}
    conv_lookup = {c["sample_id"]: c for c in rows}

    envs_meta = []
    total_chars = 0
    for sample_id in sorted(conv_lookup):
        conv = conv_lookup[sample_id]
        conv_qas = [qa for qa in all_qas if qa["conv"] == sample_id]
        env = build_env(conv, conv_qas)
        st = env_stats(env)
        total_chars += st["chars"]
        write_json(os.path.join(out_dir, f"env_{sample_id}.json"), env)

        env_ext_ids = {e["ext_id"] for s in env["sessions"] for e in s["events"]}
        qrows = []
        sel = sorted(sel_by_conv.get(sample_id, []))
        for qid in sel:
            qa_meta = qa_lookup[qid]
            raw_qa = conv["qa"][qa_meta["idx"]]
            qrows.append(build_question(conv, qa_meta["idx"], raw_qa, env_ext_ids))
        write_jsonl(os.path.join(out_dir, f"qs_{sample_id}.jsonl"), qrows)
        envs_meta.append({
            "env_id": sample_id, "file": f"env_{sample_id}.json",
            "qs_file": f"qs_{sample_id}.jsonl",
            "n_questions": len(qrows),
            **{k: st[k] for k in ("n_sessions", "n_events", "chars", "est_tokens")},
        })

    n_q = sum(e["n_questions"] for e in envs_meta)
    manifest = {
        "family": FAMILY,
        "seed": SEED,
        "sampling": {
            "method": "stratified 300 by category (largest remainder) with "
                      "per-conversation proportional allocation; within-cell "
                      "md5(seed:qid) ordering",
            "category_alloc": {str(c): n for c, n in sorted(cat_alloc.items())},
            "category_labels": {str(k): v for k, v in CAT_NAMES.items()},
        },
        "n_envs": len(envs_meta),
        "n_questions": n_q,
        "total_env_chars": total_chars,
        "estimated_ingest_tokens": total_chars // 4,
        "envs": envs_meta,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    # ---- full planning manifest (all 1986) ----
    full_dir = family_dir(FAMILY, full=True)
    full_rows = []
    for qa in all_qas:
        conv = conv_lookup[qa["conv"]]
        raw_qa = conv["qa"][qa["idx"]]
        full_rows.append({
            "qid": qa["qid"], "env_id": qa["conv"], "category": str(qa["category"]),
            "evidence": qa["evidence"],
            "answer": raw_qa.get("answer"),
            "adversarial_answer": raw_qa.get("adversarial_answer"),
        })
    write_json(os.path.join(full_dir, "manifest.json"), {
        "family": FAMILY,
        "n_questions": len(full_rows),
        "note": "planning manifest for full 1986-QA runs; env files in the parent "
                "directory already cover the full conversations",
        "questions": full_rows,
    })

    if not dev_only:
        clean_outputs(family_dir(FAMILY, full=True))
        # emit full qs files under full/ (env files are question-independent)
        for sample_id in sorted(conv_lookup):
            conv = conv_lookup[sample_id]
            with open(os.path.join(out_dir, f"env_{sample_id}.json"), encoding="utf-8") as handle:
                env = json.load(handle)
            env_ext_ids = {e["ext_id"] for s in env["sessions"] for e in s["events"]}
            qrows = []
            for idx, qa in enumerate(conv["qa"]):
                qrows.append(build_question(conv, idx, qa, env_ext_ids))
            write_jsonl(os.path.join(full_dir, f"qs_{sample_id}.jsonl"), qrows)
    return manifest
