"""Shared utilities for the memory-bench dataset converters.

Common intermediate format (H1 contract) output:
  data/common/<family>/env_<env_id>.json   -- one ingest environment
  data/common/<family>/qs_<env_id>.jsonl   -- one line per question for that env
  data/common/<family>/manifest.json      -- dev-subset manifest
  data/common/<family>/full/manifest.json -- full-dataset planning manifest

Determinism: all sampling is keyed on md5(f"{seed}:{key}") ordering (stable
across Python versions/platforms); strata use largest-remainder allocation with
ties broken by key sort order.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path

SEED = 20261007

WORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("MEMORY_BENCH_DATA_ROOT") or os.path.join(WORK_ROOT, "data")
COMMON_ROOT = os.environ.get("MEMORY_BENCH_COMMON_ROOT") or os.path.join(DATA_ROOT, "common")


def configure_paths(data_root=None, common_root=None) -> None:
    """Set explicit CLI roots before converting. No files are read here."""
    global DATA_ROOT, COMMON_ROOT
    if data_root is not None:
        DATA_ROOT = os.fspath(data_root)
    if common_root is not None:
        COMMON_ROOT = os.fspath(common_root)
    elif data_root is not None and not os.environ.get("MEMORY_BENCH_COMMON_ROOT"):
        COMMON_ROOT = os.path.join(DATA_ROOT, "common")


def get_data_root() -> str:
    return DATA_ROOT


def _checked_output(path) -> str:
    root = Path(COMMON_ROOT).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("converter output escapes the configured common root")
    return os.fspath(resolved)


# ------------------------------------------------------------------ io helpers
def family_dir(family: str, full: bool = False) -> str:
    d = os.path.join(COMMON_ROOT, family)
    if full:
        d = os.path.join(d, "full")
    d = _checked_output(d)
    os.makedirs(d, exist_ok=True)
    return d


def clean_outputs(dirpath: str) -> int:
    """Remove stale env_*.json / qs_*.jsonl from a converter output dir so
    regeneration after a selection change never leaves orphans."""
    import glob as _glob
    dirpath = _checked_output(dirpath)
    n = 0
    for pat in ("env_*.json", "qs_*.jsonl"):
        for p in _glob.glob(os.path.join(dirpath, pat)):
            os.remove(_checked_output(p))
            n += 1
    return n


def write_json(path: str, obj) -> None:
    path = _checked_output(path)
    tmp = _checked_output(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def write_jsonl(path: str, rows) -> None:
    path = _checked_output(path)
    tmp = _checked_output(path + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------ sampling
def sample_key(seed: int, key: str) -> str:
    """Stable pseudo-random ordering key (hex digest)."""
    return hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()


def stable_sample(seed: int, items, k: int):
    """Deterministic sample of k items from a list.

    Items are ordered by md5(f"{seed}:{repr(item)}"); the first k are returned
    in that (stable, platform-independent) order. Falls back to all items when
    k >= len(items).
    """
    if k >= len(items):
        return list(items)
    ordered = sorted(items, key=lambda it: (sample_key(seed, repr(it)), repr(it)))
    return ordered[:k]


def largest_remainder(total: int, weights: dict) -> dict:
    """Allocate `total` across keys proportionally to weights (largest remainder).

    Ties on remainder are broken by key sort order. Keys with zero weight get
    zero unless total cannot be placed otherwise.
    """
    wsum = sum(weights.values())
    if wsum <= 0 or total <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / wsum for k, w in weights.items()}
    alloc = {k: int(v // 1) for k, v in exact.items()}
    rem = total - sum(alloc.values())
    by_rem = sorted(exact.keys(), key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in by_rem[:rem]:
        alloc[k] += 1
    return alloc


# ------------------------------------------------------------------ dates
def parse_lme_date(s: str) -> _dt.datetime:
    """LongMemEval haystack/question date: '2023/05/20 (Sat) 02:21'."""
    return _dt.datetime.strptime(s, "%Y/%m/%d (%a) %H:%M")


def parse_locomo_datetime(s: str) -> _dt.datetime:
    """LoCoMo session datetime: '1:56 pm on 8 May, 2023'."""
    return _dt.datetime.strptime(s, "%I:%M %p on %d %B, %Y")


def iso(dt_: _dt.datetime) -> str:
    return dt_.replace(microsecond=0).isoformat()


# ------------------------------------------------------------------ locomo helpers
def ab_lines(text: str):
    """'A: ...'/'B: ...' lines -> [('A'|'B', text), ...]; blank lines skipped."""
    out = []
    for line in text.split("\n"):
        m = re.match(r"^([AB]):\s*(.*)$", line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def tolerant_dia_refs(evidence) -> list:
    """Tolerant LoCoMo evidence parser.

    Handles the malformed raw refs ('D8:6; D9:17', 'D:11:26', 'D30:05',
    space-separated runs, bare 'D'). Returns normalized 'D<n>:<k>' strings in
    first-seen order (deduplicated).
    """
    out, seen = [], set()
    items = evidence if isinstance(evidence, list) else [evidence]
    for e in items:
        for part in re.split(r"[;,\s]+", str(e)):
            part = part.strip()
            m = re.match(r"^D:?(\d+):0*(\d+)$", part)
            if m:
                ref = "D" + str(int(m.group(1))) + ":" + str(int(m.group(2)))
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
    return out


# Modified Apache-2.0 eval-spec adaptation from LongMemEval-V2
# 2cc8c540bdb87fe6761629b585e727e1c4704520. See ../THIRD_PARTY_NOTICES.md.
# ------------------------------------------------------------------ eval decoding (LME-V2)
_LME_V2_EVAL_TYPES = {
    "norm_phrase_set_match": "phrase_set",
    "norm_phrase_set_match_ordered": "phrase_set_ordered",
    "mc_choice_match": "mc_choice",
    "mc_choice_set_match": "mc_choice_set",
    "llm_abstention_checker": "llm_abstention",
    "llm_gotchas_checker": "llm_gotchas",
}


def decode_lme_v2_eval(spec: str) -> dict:
    """'func|k=v|k=v' -> {"type": <mapped>, "params": {...}} (S3 decoding)."""
    parts = [p for p in spec.split("|") if p]
    fn = parts[0]
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            params[k] = v
    if fn not in _LME_V2_EVAL_TYPES:
        raise ValueError(f"unknown eval_function: {spec}")
    return {"type": _LME_V2_EVAL_TYPES[fn], "params": params, "function": fn}


# ------------------------------------------------------------------ budgets
def env_stats(env: dict) -> dict:
    n_events = sum(len(s["events"]) for s in env["sessions"])
    chars = sum(len(e["text"]) for s in env["sessions"] for e in s["events"])
    return {
        "env_id": env["env_id"],
        "n_sessions": len(env["sessions"]),
        "n_events": n_events,
        "chars": chars,
        "est_tokens": chars // 4,
        "n_ground_truth_refs": len(set(env.get("ground_truth_refs", []))),
    }
