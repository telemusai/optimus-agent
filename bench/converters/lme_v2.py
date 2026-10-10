"""LongMemEval-V2 -> common format (authority-preserving adapter, revision 1).

Two envs (web, enterprise). Each env = the domain's shared 100-trajectory
haystack (lme_v2_small_haystack.json); selected trajectories remain in source
file order, with states in source list order. One session per
trajectory.

Authority model (adapter revision 1; see reports/continuation/
lme-authority-proposal/README.md):

  * header event (assistant role): "Goal: <goal>\nOutcome: <outcome>\nStart:
    <start_url>". The goal/outcome are dataset task metadata with no documented
    speaker; they stay assistant-origin (non-citable assertions) and are NEVER
    relabeled user. Text, position, and ext_id are byte-identical to revision 0.
  * per state, TWO events instead of one mixed user-labeled blob:
      - observation event (toolResult role): "URL: <url>\nObservation:\n
        <accessibility_tree capped>" plus a tool_result descriptor. Browser
        state is a tool observation, not a user confirmation.
      - decision event (assistant role): "Action: <action>" / "Thought:
        <thought>" when present. Agent decisions are assistant assertions, not
        verified facts.
    Event order per state is observation then decision (the recorded decision
    responds to that state's observation); all field text is preserved
    verbatim, only the grouping changes.

ext_id: "<traj_id>:header" / "<traj_id>:s<state_index>:obs" /
"<traj_id>:s<state_index>:act". Screenshots are NOT ingested (text-only run;
accessibility_tree is the text equivalent). Each accessibility tree is capped
at the first AXTREE_CAP chars (truncation marker appended; stats recorded in
env notes). Every event keeps ts: None; the dataset carries no event times and
none are invented.

No retrieval oracle exists in the dataset (questions carry no trajectory
ids), so ground_truth_refs = [] (noted in env).
"""

from __future__ import annotations

import json
import os

from bench.protocol import annotate_question, CAPTURE_PROTOCOL_VERSION, QUESTION_PROTOCOLS

from .common import (clean_outputs, SEED, get_data_root, family_dir, write_json, write_jsonl,
                     stable_sample, largest_remainder, decode_lme_v2_eval,
                     env_stats)

FAMILY = "lme_v2"
DEV_N = 120
AXTREE_CAP = int(os.environ.get("LME_V2_AXTREE_CAP", "8000"))

# Adapter revision marker for fresh-capture identification. Revision 0 is the
# historical single-user-event-per-state format; this revision splits authority.
AUTHORITY_REVISION = 1
TOOL_NAME = "lme_v2_observation"

AUTHORITY_POLICY = {
    "revision": AUTHORITY_REVISION,
    "header": "assistant: dataset task metadata (goal/outcome/start); no documented "
              "speaker, never relabeled user",
    "observation": "toolResult: browser URL + accessibility tree are environment "
                   "observations (citable tool evidence, not user statements)",
    "decision": "assistant: recorded agent action/thought are agent assertions, "
                "not verified facts",
    "user": "none: the dataset records no user turns; no user origin is invented",
    "missing_ts": "absent: dataset has no event times; no synthetic fallback",
}


def load_questions():
    qs = []
    with open(os.path.join(get_data_root(), "lme_v2_questions.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def load_haystack():
    with open(os.path.join(get_data_root(), "lme_v2_small_haystack.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def cap_accessibility_tree(at: str) -> tuple:
    """Cap the accessibility tree at AXTREE_CAP chars; return (text, marker)."""
    capped, marker = at, ""
    if len(at) > AXTREE_CAP:
        capped = at[:AXTREE_CAP]
        marker = f"\n[accessibility tree truncated: first {AXTREE_CAP} of {len(at)} chars]"
    return capped, marker


def observation_text(s: dict) -> tuple:
    """Browser state observation (tool authority). URL plus capped tree."""
    parts = ["URL: " + (s.get("url") or "")]
    at = s.get("accessibility_tree") or ""
    capped, marker = cap_accessibility_tree(at)
    parts.append("Observation:\n" + capped + marker)
    return "\n".join(parts), marker != ""


def decision_text(s: dict) -> str:
    """Agent decision (assistant authority). Action/Thought when present."""
    parts = []
    if s.get("action"):
        parts.append("Action: " + s["action"])
    if s.get("thought"):
        parts.append("Thought: " + s["thought"])
    return "\n".join(parts)


def build_domain_env(domain: str, traj_ids: list) -> tuple:
    """Stream trajectories.jsonl once, keep the domain's haystack trajs."""
    wanted = set(traj_ids)
    sessions = []
    stats = {"states_total": 0, "states_truncated": 0,
             "axtree_chars_raw": 0, "axtree_chars_capped": 0,
             "events_header": 0, "events_observation": 0, "events_decision": 0,
             "states_without_decision": 0}
    with open(os.path.join(get_data_root(), "trajectories.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t["id"] not in wanted:
                continue
            header_text = (f"Goal: {t['goal']}\nOutcome: {t['outcome']}"
                           f"\nStart: {t['start_url']}")
            events = [{
                "role": "assistant",
                "text": header_text,
                "ext_id": f"{t['id']}:header",
                "ts": None,
                "authority": "dataset_task_metadata",
            }]
            stats["events_header"] += 1
            for s in t["states"]:
                obs_text, truncated = observation_text(s)
                stats["states_total"] += 1
                stats["states_truncated"] += int(truncated)
                stats["axtree_chars_raw"] += len(s.get("accessibility_tree") or "")
                stats["axtree_chars_capped"] += min(
                    len(s.get("accessibility_tree") or ""), AXTREE_CAP)
                obs_id = f"{t['id']}:s{s['state_index']}:obs"
                events.append({
                    "role": "toolResult",
                    "text": obs_text,
                    "ext_id": obs_id,
                    "ts": None,
                    "authority": "browser_observation",
                    "tool_result": {
                        "toolCallId": obs_id,
                        "toolName": TOOL_NAME,
                        "isError": False,
                    },
                })
                stats["events_observation"] += 1
                decision = decision_text(s)
                if decision:
                    events.append({
                        "role": "assistant",
                        "text": decision,
                        "ext_id": f"{t['id']}:s{s['state_index']}:act",
                        "ts": None,
                        "authority": "agent_decision",
                    })
                    stats["events_decision"] += 1
                else:
                    stats["states_without_decision"] += 1
            sessions.append({
                "session_id": t["id"],
                "ts": None,
                "events": events,
                "traj_meta": {"outcome": t.get("outcome"),
                              "environment": t.get("environment")},
            })
    found = {s["session_id"] for s in sessions}
    if found != wanted:
        raise RuntimeError(
            f"haystack mismatch for {domain}: missing {len(wanted - found)}, "
            f"extra {len(found - wanted)}")
    env = {
        "env_id": domain,
        "sessions": sessions,  # selected trajectory file order preserved
        "ground_truth_refs": [],
        "family": FAMILY,
        "authority_revision": AUTHORITY_REVISION,
        "capture_protocol": CAPTURE_PROTOCOL_VERSION,
        "question_protocol": QUESTION_PROTOCOLS[FAMILY],
        "authority_policy": AUTHORITY_POLICY,
        "notes": {
            "oracle": "none: LongMemEval-V2 provides no per-question trajectory "
                      "ground truth; retrieval Recall@K/P@K not computable (see S3)",
            "haystack": "lme_v2_small_haystack.json (shared 100-traj haystack per domain)",
            "axtree_cap_chars": AXTREE_CAP,
            "screenshots": "not ingested (text-only); 29 errors-gotchas questions "
                           "reference images that are NOT provided to the model",
            "authority": "adapter revision 1: per-state events split into toolResult "
                         "observation (url + capped axtree) and assistant decision "
                         "(action/thought); header stays assistant dataset metadata; "
                         "no user origin is invented",
            **stats,
        },
    }
    return env, stats


def build_question(q: dict) -> dict:
    return annotate_question({
        "qid": q["id"],
        "env_id": q["domain"],
        "question": q["question"],
        "answer": q["answer"],
        "evidence": [],
        "category": q["question_type"],
        "eval": decode_lme_v2_eval(q["eval_function"]),
        # extras:
        "domain": q["domain"],
        "environment": q["environment"],
        "image": q.get("image"),
        "raw_eval_function": q["eval_function"],
    }, FAMILY)


def dev_selection(questions):
    """120 questions stratified by question_type x domain; all 29 gotchas in."""
    gotchas = sorted(q["id"] for q in questions if q["question_type"] == "errors-gotchas")
    rest = [q for q in questions if q["question_type"] != "errors-gotchas"]
    cells = {}
    for q in rest:
        cells.setdefault((q["question_type"], q["domain"]), []).append(q["id"])
    alloc = largest_remainder(DEV_N - len(gotchas),
                              {k: len(v) for k, v in cells.items()})
    selected = list(gotchas)
    strata = {"errors-gotchas(all)": len(gotchas)}
    for (t, d), ids in sorted(cells.items()):
        k = alloc[(t, d)]
        strata[f"{t}|{d}"] = k
        selected += stable_sample(SEED, sorted(ids), k)
    return selected, strata


def build(dev_only=True):
    out_dir = family_dir(FAMILY)
    clean_outputs(out_dir)
    questions = load_questions()
    haystack = load_haystack()

    # one shared haystack per domain (verified: distinct set per domain)
    dom_traj = {}
    for dom in ("web", "enterprise"):
        ids = None
        for q in questions:
            if q["domain"] == dom:
                cur = tuple(haystack[q["id"]])
                if ids is None:
                    ids = cur
                elif cur != ids:
                    raise RuntimeError(f"haystack not shared within {dom}")
        dom_traj[dom] = list(ids)

    selected, strata = dev_selection(questions)
    sel_by_dom = {"web": [], "enterprise": []}
    for qid in selected:
        q = next(x for x in questions if x["id"] == qid)
        sel_by_dom[q["domain"]].append(qid)

    envs_meta = []
    total_chars = 0
    for dom in ("web", "enterprise"):
        env, stats = build_domain_env(dom, dom_traj[dom])
        st = env_stats(env)
        total_chars += st["chars"]
        write_json(os.path.join(out_dir, f"env_{dom}.json"), env)

        qrows = [build_question(next(x for x in questions if x["id"] == qid))
                 for qid in sorted(sel_by_dom[dom])]
        write_jsonl(os.path.join(out_dir, f"qs_{dom}.jsonl"), qrows)
        envs_meta.append({
            "env_id": dom, "file": f"env_{dom}.json", "qs_file": f"qs_{dom}.jsonl",
            "n_questions": len(qrows),
            **{k: st[k] for k in ("n_sessions", "n_events", "chars", "est_tokens")},
            "states_truncated": stats["states_truncated"],
            "axtree_chars_raw": stats["axtree_chars_raw"],
            "axtree_chars_capped": stats["axtree_chars_capped"],
            "events_header": stats["events_header"],
            "events_observation": stats["events_observation"],
            "events_decision": stats["events_decision"],
        })

    manifest = {
        "family": FAMILY,
        "authority_revision": AUTHORITY_REVISION,
        "capture_protocol": CAPTURE_PROTOCOL_VERSION,
        "question_protocol": QUESTION_PROTOCOLS[FAMILY],
        "seed": SEED,
        "sampling": {
            "method": "stratified by question_type x domain (largest remainder over "
                      "non-gotcha questions); all 29 errors-gotchas forced in; "
                      "within-cell md5(seed:qid) ordering",
            "strata_alloc": strata,
        },
        "n_envs": len(envs_meta),
        "n_questions": sum(e["n_questions"] for e in envs_meta),
        "total_env_chars": total_chars,
        "estimated_ingest_tokens": total_chars // 4,
        "envs": envs_meta,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    # ---- full planning manifest (all 451) ----
    full_dir = family_dir(FAMILY, full=True)
    write_json(os.path.join(full_dir, "manifest.json"), {
        "family": FAMILY,
        "authority_revision": AUTHORITY_REVISION,
        "capture_protocol": CAPTURE_PROTOCOL_VERSION,
        "question_protocol": QUESTION_PROTOCOLS[FAMILY],
        "n_questions": len(questions),
        "note": "planning manifest for full 451-question runs; the two env files "
                "in the parent directory already cover the full haystacks",
        "questions": [build_question(q) for q in questions],
    })

    if not dev_only:
        clean_outputs(family_dir(FAMILY, full=True))
        # emit full qs files under full/ (envs are question-independent)
        for dom in ("web", "enterprise"):
            qrows = [build_question(q) for q in questions if q["domain"] == dom]
            write_jsonl(os.path.join(full_dir, f"qs_{dom}.jsonl"), qrows)
    return manifest
