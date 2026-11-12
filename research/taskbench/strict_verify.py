"""Strict extraction metrics for the taskbench campaign (occ3-bench-opt-20261109).

New standalone module. It does NOT modify runner.py, four_way_adapter.py, or
any other existing file; the adapter imports it later.

Why this module exists
----------------------
The historical taskbench "exact_match" verifier means SUBSTRING matching:
an answer counts when it CONTAINS the expected value somewhere. That is a
recovery-of-fact signal, not a complete-message-recovery signal: an answer
like "The region is sa-east, as Agent A said." hits the substring metric
while its bytes differ from the gold answer. The campaign must

  (a) preserve that historical metric unchanged, under its historical name,
      so historical numbers stay comparable, and
  (b) add SEPARATELY-NAMED strict byte/field equality metrics, so a substring
      success is never silently reported as complete-message recovery.

All functions here are pure, stdlib-only, and make zero model/network calls.

Public API (exact signatures)
-----------------------------
  historical_exact(answer, expected) -> bool
  strict_byte_equal(a, b) -> bool
  strict_field_match(answer, expected_fields) -> dict
  partial_recovery(answer, expected) -> float

Normalization rules (documented, minimal)
-----------------------------------------
- historical_exact applies the historical taskbench normalization on BOTH
  sides before the substring test: collapse internal whitespace runs to
  single spaces, strip, casefold (a verbatim mirror of
  research/tasks/checkers.py::normalize_text). Then it tests containment:
  normalized(expected) in normalized(answer). An empty/None expected never
  hits (vacuous containment is not recovery).
- strict_byte_equal trims SURROUNDING Unicode whitespace only
  (Python str.strip()) and then compares the UTF-8 byte sequences. No
  whitespace collapse, no casefold, no punctuation stripping, no numeric
  tolerance. None is equal only to None.
- strict_field_match parses KEY=VALUE lines tolerantly (optional spaces
  around "=", optional OCC/1 header, code fences skipped, and for a dual
  OCC-2 body only the section after the #FALLBACK-OCC1 sentinel is parsed)
  but compares VALUES strictly with strict_byte_equal semantics.
- partial_recovery splits the expected value into facts (collection items,
  KEY=VALUE values for a structured body, comma-separated parts otherwise)
  and reports the fraction recoverable from the answer by the same
  normalized-containment test historical_exact uses. It is partial-credit
  reporting only and is NEVER a substitute for the strict metrics.

Relationship to four_way_adapter.py: that file keeps its own
strict_field_match(speaker_occ1_body, recorded_wire_body) for WIRE fidelity
(author-authored field set surviving the round trip). This module's
strict_field_match(answer, expected_fields) scores the listener's ANSWER
against expected fields. Different inputs, different semantics; both strict.
"""

from __future__ import annotations

import re
import sys

__all__ = [
    "historical_exact",
    "strict_byte_equal",
    "strict_field_match",
    "partial_recovery",
]

# ---------------------------------------------------------------------------
# Shared helpers (private)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")

# A KEY=VALUE line: bare identifier key, optional spaces around "=".
# Lines that do not match (prose, "OCC/1", "OCC/2 D=... C=..." headers,
# "#FALLBACK-OCC1" sentinels, code fences) are simply not fields.
_KV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

_OCC2_FALLBACK_SENTINEL = "#FALLBACK-OCC1"


def _normalize_historical(value) -> str:
    """Historical taskbench normalization (checkers.py normalize_text mirror).

    None -> ""; otherwise collapse whitespace runs to single spaces, strip,
    casefold. Byte-level identity is NOT preserved by design: this is the
    historical lenient-extraction normalizer, kept verbatim so the preserved
    metric matches its historical behaviour."""
    if value is None:
        return ""
    return _WS.sub(" ", str(value)).strip().casefold()


def _as_text(value) -> str | None:
    """Coerce a value to str for strict comparison. None stays None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value
    return str(value)


def _field_section(text: str) -> str:
    """For a dual OCC-2 body, only the embedded OCC-1 section is fields."""
    if _OCC2_FALLBACK_SENTINEL in text:
        return text.split(_OCC2_FALLBACK_SENTINEL, 1)[1]
    return text


def _parse_field_occurrences(text: str) -> dict[str, list[str]]:
    """Parse KEY=VALUE occurrences from text.

    Returns key -> ordered list of stripped values (one per occurrence).
    Parsing is tolerant (spaces around "=", header/fence/prose lines
    skipped); value COMPARISON elsewhere in this module is strict."""
    occurrences: dict[str, list[str]] = {}
    for line in _field_section(text).splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        match = _KV_LINE.match(line)
        if match:
            occurrences.setdefault(match.group(1), []).append(match.group(2))
    return occurrences


def _expected_facts(expected) -> list[str]:
    """Split an expected value into recoverable facts.

    - None -> no facts.
    - list/tuple/set -> one fact per item (str()).
    - dict -> one fact per value.
    - str -> structured (ALL content lines are KEY=VALUE, e.g. an OCC-1
      body): the VALUES are the facts; otherwise comma-separated parts;
      otherwise the single stripped string.
    Empty facts are dropped."""
    if expected is None:
        return []
    if isinstance(expected, (list, tuple, set)):
        items = [str(item) for item in expected]
    elif isinstance(expected, dict):
        items = [str(value) for value in expected.values()]
    else:
        text = str(expected)
        lines = [ln.strip() for ln in _field_section(text).splitlines()]
        content = [
            ln for ln in lines if ln and not ln.startswith("```") and ln != "OCC/1"
        ]
        if content and all(_KV_LINE.match(ln) for ln in content):
            items = [match.group(2) for match in map(_KV_LINE.match, content) if match]
        elif "," in text:
            items = text.split(",")
        else:
            items = [text]
    facts = [str(item).strip() for item in items]
    return [fact for fact in facts if fact]


# ---------------------------------------------------------------------------
# (1) Historical metric, name and semantics preserved
# ---------------------------------------------------------------------------


def historical_exact(answer, expected) -> bool:
    """Historical taskbench "exact_match": SUBSTRING matching, preserved.

    True iff the normalized expected value occurs as a contiguous substring
    of the normalized answer. Both sides get the historical normalization
    (whitespace collapse + strip + casefold, mirroring
    research/tasks/checkers.py::normalize_text).

    A True here means the expected value is RECOVERABLE from the answer
    text. It does NOT mean the answer equals the expected value; pair it
    with strict_byte_equal / strict_field_match before claiming complete
    recovery. Empty or None expected -> False (vacuous containment is not
    recovery). None answer -> False."""
    needle = _normalize_historical(expected)
    if not needle:
        return False
    return needle in _normalize_historical(answer)


# ---------------------------------------------------------------------------
# (2) Strict byte equality
# ---------------------------------------------------------------------------


def strict_byte_equal(a, b) -> bool:
    """Exact byte equality after a documented minimal normalization.

    Normalization: trim SURROUNDING Unicode whitespace only (str.strip()).
    Everything else — internal whitespace, case, punctuation, numeric form —
    is compared exactly, as UTF-8 bytes. No casefold, no whitespace collapse,
    no set logic, no tolerance.

    Examples:
      strict_byte_equal("  sa-east\n", "sa-east")    -> True   (trim only)
      strict_byte_equal("SA-EAST", "sa-east")         -> False  (case differs)
      strict_byte_equal("a  b", "a b")               -> False  (inner spaces)
      strict_byte_equal("sa-east.", "sa-east")       -> False  (punctuation)

    None is equal only to None; anything else involving None is False."""
    if a is None or b is None:
        return a is b
    text_a = _as_text(a)
    text_b = _as_text(b)
    if text_a is None or text_b is None:
        return False
    stripped_a = text_a.strip()
    stripped_b = text_b.strip()
    try:
        return stripped_a.encode("utf-8") == stripped_b.encode("utf-8")
    except UnicodeEncodeError:
        # Lone surrogates or similar: fall back to code-point equality.
        return stripped_a == stripped_b


# ---------------------------------------------------------------------------
# (3) Strict per-field extraction for KEY=VALUE / structured tasks
# ---------------------------------------------------------------------------


def strict_field_match(answer, expected_fields) -> dict:
    """Per-field exact extraction of KEY=VALUE fields from an answer.

    expected_fields: either a mapping {field name -> expected value}, or a
    string (an OCC-1 body, or a dual OCC-2 body; its KEY=VALUE lines become
    the expected fields). Parsing of the ANSWER is tolerant: optional spaces
    around "=", optional OCC/1 header, code fences skipped, and for a dual
    body only the section after #FALLBACK-OCC1 counts. Comparison is strict:
    field names must match exactly (case-sensitive) and values must satisfy
    strict_byte_equal (trim surrounding whitespace only).

    Semantics of "match": every required (expected) field is present AND has
    the exactly-correct value AND no required field is altered (a present
    field with a wrong or conflicting duplicate value fails). Extra fields
    (present in the answer but not required) are REPORTED but do not by
    themselves fail the match — apply an exact-field-set policy at the
    caller using the returned "extra" list if needed.

    Returns a report dict:
      {
        "match": bool,
        "fields": {name: {"present": bool, "correct": bool,
                          "expected": str | None, "got": str | None,
                          "conflict": bool}},
        "missing": [names of required fields absent from the answer],
        "wrong":   [names of required fields present but incorrect],
        "extra":   [names of answer fields that were not required],
        "error":   None | str,
      }
    The lists are sorted for deterministic reporting. expected_fields=None
    or an unparseable expected body -> match=False with "error" set (never a
    silent vacuous True)."""
    report = {
        "match": False,
        "fields": {},
        "missing": [],
        "wrong": [],
        "extra": [],
        "error": None,
    }
    if expected_fields is None:
        report["error"] = "expected_fields is None"
        return report
    if isinstance(expected_fields, dict):
        expected = {
            str(name): value for name, value in expected_fields.items()
        }
    elif isinstance(expected_fields, str):
        occurrences = _parse_field_occurrences(expected_fields)
        if not occurrences:
            report["error"] = "expected_fields string has no KEY=VALUE lines"
            return report
        expected = {name: values[0] for name, values in occurrences.items()}
    else:
        report["error"] = (
            "expected_fields must be a mapping or a KEY=VALUE string, got "
            + type(expected_fields).__name__
        )
        return report

    answer_text = _as_text(answer)
    if answer_text is None:
        answer_text = ""
    got_occurrences = _parse_field_occurrences(answer_text)

    for name, expected_value in expected.items():
        expected_text = _as_text(expected_value)
        occurrences = got_occurrences.get(name, [])
        present = bool(occurrences)
        conflict = len({value for value in occurrences}) > 1
        got_value = occurrences[0] if occurrences else None
        correct = present and not conflict and strict_byte_equal(
            got_value, expected_text
        )
        report["fields"][name] = {
            "present": present,
            "correct": bool(correct),
            "expected": expected_text,
            "got": got_value,
            "conflict": conflict,
        }
        if not present:
            report["missing"].append(name)
        elif not correct:
            report["wrong"].append(name)

    report["extra"] = sorted(set(got_occurrences) - set(expected))
    report["missing"].sort()
    report["wrong"].sort()
    report["match"] = not report["missing"] and not report["wrong"]
    return report


# ---------------------------------------------------------------------------
# (4) Partial-credit fact recovery (never a substitute for strict)
# ---------------------------------------------------------------------------


def partial_recovery(answer, expected) -> float:
    """Fraction of expected facts recoverable from the answer.

    The expected value is split into facts (see _expected_facts: collection
    items; KEY=VALUE values for a structured body; comma-separated parts
    otherwise). A fact counts as recovered when it is recoverable from the
    answer under the historical normalized-containment test used by
    historical_exact. Returns recovered/total in [0.0, 1.0]; no expected
    facts at all -> 0.0 (vacuous completeness is never reported).

    This is PARTIAL-CREDIT REPORTING ONLY. A 1.0 here means every expected
    fact is somewhere in the answer — it says nothing about byte-exact or
    field-exact recovery, and it must never be reported as a strict or
    complete-message-recovery success."""
    facts = _expected_facts(expected)
    if not facts:
        return 0.0
    haystack = _normalize_historical(answer)
    recovered = 0
    for fact in facts:
        needle = _normalize_historical(fact)
        if needle and needle in haystack:
            recovered += 1
    return recovered / len(facts)


# ---------------------------------------------------------------------------
# Selftest (no model calls, no network, no filesystem writes)
# ---------------------------------------------------------------------------


def _selftest() -> tuple[int, int, list[str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # -- historical_exact: substring semantics, name preserved --------------
    check("hx_exact_hit", historical_exact("sa-east", "sa-east"))
    check(
        "hx_substring_hit",
        historical_exact("The region is sa-east, as stated.", "sa-east"),
        "substring inside a sentence must hit the historical metric",
    )
    check(
        "hx_substring_hit_but_bytes_differ",
        historical_exact("The region is sa-east.", "sa-east")
        and not strict_byte_equal("The region is sa-east.", "sa-east"),
        "core case: historical hit, strict byte miss",
    )
    check(
        "hx_normalization",
        historical_exact("The  REGION is\n\tSA-EAST.", "sa-east"),
        "whitespace collapse + casefold on both sides",
    )
    check("hx_miss", not historical_exact("eu-west-1", "sa-east"))
    check("hx_empty_expected", not historical_exact("anything", ""))
    check("hx_none_answer", not historical_exact(None, "sa-east"))
    check(
        "hx_punctuation_not_rescued",
        not historical_exact("sa-eas", "sa-east"),
        "partial token overlap is not containment",
    )

    # -- strict_byte_equal: trim-only normalization, exact bytes -------------
    check("sbe_exact", strict_byte_equal("sa-east", "sa-east"))
    check("sbe_trim_only", strict_byte_equal("  sa-east\n", "sa-east"))
    check("sbe_case_differs", not strict_byte_equal("SA-EAST", "sa-east"))
    check("sbe_inner_ws_differs", not strict_byte_equal("a  b", "a b"))
    check("sbe_trailing_period", not strict_byte_equal("Stage the new key.", "Stage the new key"))
    check("sbe_none_none", strict_byte_equal(None, None))
    check("sbe_none_vs_empty", not strict_byte_equal(None, ""))
    check("sbe_prefix", not strict_byte_equal("sa-east-1", "sa-east"))

    # -- strict_field_match: per-field extraction ----------------------------
    ok_answer = "OCC/1\nTASK=A-dev-001\nKIND=TASK\nSTATUS=pending\nF1=sa-east"
    rep = strict_field_match(ok_answer, {"TASK": "A-dev-001", "F1": "sa-east"})
    check(
        "sfm_full_match",
        rep["match"] and rep["missing"] == [] and rep["wrong"] == [],
        f"got {rep}",
    )
    check(
        "sfm_extra_reported",
        sorted(rep["extra"]) == ["KIND", "STATUS"],
        f"extra={rep['extra']}",
    )
    rep = strict_field_match("F1=eu-west", {"F1": "sa-east"})
    check(
        "sfm_present_but_wrong",
        not rep["match"] and rep["wrong"] == ["F1"] and rep["missing"] == [],
        f"got {rep}",
    )
    rep = strict_field_match("F1=sa-east", {"F1": "sa-east", "F2": "missing"})
    check(
        "sfm_missing_field",
        not rep["match"] and rep["missing"] == ["F2"] and rep["wrong"] == [],
        f"got {rep}",
    )
    rep = strict_field_match("F1=a\nF1=b", {"F1": "a"})
    check(
        "sfm_conflicting_duplicate",
        not rep["match"] and rep["wrong"] == ["F1"] and rep["fields"]["F1"]["conflict"],
        f"got {rep}",
    )
    rep = strict_field_match("F1=  sa-east  ", {"F1": "sa-east"})
    check("sfm_value_trim_ok", rep["match"], "value trim-only normalization")
    rep = strict_field_match("F1=SA-EAST", {"F1": "sa-east"})
    check("sfm_value_case_strict", not rep["match"], "value case is strict")
    rep = strict_field_match(ok_answer, "OCC/1\nTASK=A-dev-001\nF1=sa-east")
    check(
        "sfm_expected_body_string",
        rep["match"] and rep["extra"] == ["KIND", "STATUS"],
        f"expected_fields as OCC-1 body string; got extra={rep['extra']}",
    )
    dual = "OCC/2 D=sym C=01\nT=x\n#FALLBACK-OCC1\nOCC/1\nT=hello\nF1=sa-east"
    rep = strict_field_match(dual, {"T": "hello", "F1": "sa-east"})
    check(
        "sfm_dual_body_after_sentinel",
        rep["match"] and rep["extra"] == [],
        f"symbolic section before sentinel must not count; got {rep}",
    )
    rep = strict_field_match("no fields here", {"F1": "x"})
    check("sfm_prose_answer", not rep["match"] and rep["missing"] == ["F1"])
    rep = strict_field_match("F1=x", None)
    check(
        "sfm_expected_none",
        not rep["match"] and rep["error"] is not None,
        "None expected_fields never yields vacuous True",
    )
    rep = strict_field_match("F1=x", "not a body")
    check(
        "sfm_expected_unparseable",
        not rep["match"] and rep["error"] is not None,
        "unparseable expected body never yields vacuous True",
    )

    # -- partial_recovery: fraction of expected facts -------------------------
    got = partial_recovery("alpha and beta", "alpha, beta, gamma")
    check("pr_two_of_three", abs(got - 2 / 3) < 1e-12, f"got {got}")
    check("pr_full", partial_recovery("alpha and beta", "alpha, beta") == 1.0)
    check(
        "pr_full_but_not_strict",
        partial_recovery("alpha and beta", "alpha, beta") == 1.0
        and not strict_byte_equal("alpha and beta", "alpha, beta"),
        "partial 1.0 is never a strict success",
    )
    got = partial_recovery("F2=b", "OCC/1\nF1=a\nF2=b\nF3=c")
    check("pr_structured_one_of_three", abs(got - 1 / 3) < 1e-12, f"got {got}")
    check("pr_list_expected", abs(partial_recovery("a c", ["a", "b", "c"]) - 2 / 3) < 1e-12)
    check("pr_none_answer", partial_recovery(None, "sa-east") == 0.0)
    check("pr_empty_expected", partial_recovery("anything", "") == 0.0)
    check("pr_miss", partial_recovery("eu-west", "sa-east") == 0.0)

    failures = [f"{name}: {detail}" for name, ok, detail in checks if not ok]
    return len(checks) - len(failures), len(checks), failures


if __name__ == "__main__":
    passed, total, failures = _selftest()
    print(f"strict_verify selftest: {passed}/{total} checks passed")
    for failure in failures:
        print(f"  [FAIL] {failure}")
    sys.exit(0 if passed == total else 1)
