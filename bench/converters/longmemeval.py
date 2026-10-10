"""Local LongMemEval-S data adapter with separately licensed MIT judge templates.

Question-specific sessions are date-sorted; empty sessions are skipped. This
adapter retains campaign abstention detection (suffix, type and answer-session
conditions), which is not generally identical to the upstream qid substring
rule. It is versioned as a local adapter, not a full upstream-equivalent runner.
"""

from __future__ import annotations

import json
import os

from bench.protocol import annotate_question

from .common import (clean_outputs, SEED, get_data_root, family_dir, write_json, write_jsonl,
                     parse_lme_date, iso, stable_sample, largest_remainder,
                     env_stats)

FAMILY = "longmemeval"
DEV_N = 100


def load_rows():
    with open(os.path.join(get_data_root(), "longmemeval_s.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def is_abstention(q) -> bool:
    """True for the 6 injected-decoy abstention questions."""
    return (q["question_id"].endswith("_abs")
            and q["question_type"] == "single-session-user"
            and any(a.endswith("_abs") for a in q["answer_session_ids"]))


def build_env(q: dict) -> dict:
    qid = q["question_id"]
    order = sorted(range(len(q["haystack_dates"])),
                   key=lambda i: (parse_lme_date(q["haystack_dates"][i]), i))
    sessions = []
    sid_seen = {}
    for i in order:
        sid = q["haystack_session_ids"][i]
        date = q["haystack_dates"][i]
        msgs = q["haystack_sessions"][i]
        if not msgs:  # this adapter skips empty sessions
            continue
        n = sid_seen.get(sid, 0)
        sid_seen[sid] = n + 1
        sess_sid = sid if n == 0 else f"{sid}#{n + 1}"
        ts = iso(parse_lme_date(date))
        events = [{"role": m["role"], "text": m["content"], "ext_id": sid, "ts": ts}
                  for m in msgs]
        sessions.append({"session_id": sess_sid, "ts": ts, "events": events})
    return {
        "env_id": qid,
        "sessions": sessions,
        "ground_truth_refs": list(q["answer_session_ids"]),
        "family": FAMILY,
        "notes": {
            "question_date": q["question_date"],
            "haystack_sessions_total": len(q["haystack_sessions"]),
            "empty_sessions_skipped": len(q["haystack_sessions"]) - len(sessions),
            "duplicate_session_ids": sorted(s for s, c in sid_seen.items() if c > 1),
            "abstention_decoy": is_abstention(q),
        },
    }


def build_question(q: dict) -> dict:
    ans = q["answer"]
    if isinstance(ans, (int, float)):
        ans = str(ans)
    return annotate_question({
        "qid": q["question_id"],
        "env_id": q["question_id"],
        "question": q["question"],
        "answer": ans,
        "evidence": list(q["answer_session_ids"]),
        "category": q["question_type"],
        "eval": {
            "type": "llm_judge",
            "params": {
                "prompt_variant": q["question_type"],
                "abstention": is_abstention(q),
                "judge_model": "glm-5.3",
                "source": "MIT LongMemEval templates; local DGX adapter",
            },
        },
        # extras (documented in H3):
        "question_date": q["question_date"],
        "abstention": is_abstention(q),
    }, FAMILY)


def dev_selection(rows):
    """Stratified 100 by question_type (proportional, largest remainder),
    force-including all 6 abstention items (inside the single-session-user
    stratum)."""
    by_id = {q["question_id"]: q for q in rows}
    by_type = {}
    for q in rows:
        by_type.setdefault(q["question_type"], []).append(q["question_id"])
    alloc = largest_remainder(DEV_N, {k: len(v) for k, v in by_type.items()})
    selected = {}
    for t, ids in sorted(by_type.items()):
        k = alloc[t]
        if t == "single-session-user":
            # keep only true abstention items (oracle -> injected decoy sessions)
            forced = [i for i in sorted(ids) if i.endswith("_abs")
                      and any(a.endswith("_abs")
                              for a in by_id[i]["answer_session_ids"])]
            chosen = list(dict.fromkeys(forced))
            rest = [i for i in sorted(ids) if i not in chosen]
            chosen += stable_sample(SEED, rest, k - len(chosen))
            selected[t] = chosen
        else:
            selected[t] = stable_sample(SEED, sorted(ids), k)
    return selected, alloc


def build(dev_only=True):
    rows = load_rows()
    out_dir = family_dir(FAMILY)
    clean_outputs(out_dir)
    selected, alloc = dev_selection(rows)

    sel_ids = set()
    for v in selected.values():
        sel_ids.update(v)
    by_id = {q["question_id"]: q for q in rows}

    envs_meta = []
    total_chars = 0
    for qid in sorted(sel_ids):
        q = by_id[qid]
        env = build_env(q)
        st = env_stats(env)
        total_chars += st["chars"]
        write_json(os.path.join(out_dir, f"env_{qid}.json"), env)
        write_jsonl(os.path.join(out_dir, f"qs_{qid}.jsonl"), [build_question(q)])
        envs_meta.append({
            "env_id": qid, "file": f"env_{qid}.json", "qs_file": f"qs_{qid}.jsonl",
            "category": q["question_type"], "abstention": is_abstention(q),
            **{k: st[k] for k in ("n_sessions", "n_events", "chars", "est_tokens")},
        })

    manifest = {
        "family": FAMILY,
        "seed": SEED,
        "sampling": {
            "method": "stratified by question_type; largest-remainder allocation; "
                      "within-stratum md5(seed:qid) ordering; all abstention items forced in",
            "strata_alloc": {t: len(v) for t, v in sorted(selected.items())},
        },
        "n_envs": len(envs_meta),
        "n_questions": len(envs_meta),
        "total_env_chars": total_chars,
        "estimated_ingest_tokens": total_chars // 4,
        "envs": envs_meta,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    # ---- full planning manifest (all 500) ----
    full_dir = family_dir(FAMILY, full=True)
    full_rows = []
    for q in rows:
        n_empty = sum(1 for s in q["haystack_sessions"] if not s)
        chars = sum(len(m["content"]) for s in q["haystack_sessions"] for m in s)
        full_rows.append({
            "qid": q["question_id"],
            "env_id": q["question_id"],
            "category": q["question_type"],
            "abstention": is_abstention(q),
            "question_date": q["question_date"],
            "answer": str(q["answer"]) if isinstance(q["answer"], (int, float)) else q["answer"],
            "evidence": list(q["answer_session_ids"]),
            "n_sessions": len(q["haystack_sessions"]),
            "n_nonempty_sessions": len(q["haystack_sessions"]) - n_empty,
            "n_msgs": sum(len(s) for s in q["haystack_sessions"]),
            "chars": chars,
            "est_tokens": chars // 4,
        })
    write_json(os.path.join(full_dir, "manifest.json"), {
        "family": FAMILY,
        "n_questions": len(full_rows),
        "note": "planning manifest for full 500-question runs",
        "questions": full_rows,
    })

    if not dev_only:
        clean_outputs(family_dir(FAMILY, full=True))
        # emit the full env/qs set under full/ (dev files in the parent stay intact)
        for q in rows:
            write_json(os.path.join(full_dir, f"env_{q['question_id']}.json"),
                       build_env(q))
            write_jsonl(os.path.join(full_dir, f"qs_{q['question_id']}.jsonl"),
                        [build_question(q)])
    return manifest
