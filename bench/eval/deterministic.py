"""Licensed LME-V2 metric adaptations and independent Optimus local dispatch.

LME-V2 functions below are modified from xiaowu0162/LongMemEval-V2 at
2cc8c540bdb87fe6761629b585e727e1c4704520, evaluation/qa_eval_metrics.py.
Licensed under Apache-2.0; see ../licenses/LongMemEval-V2-Apache-2.0.txt
and ../THIRD_PARTY_NOTICES.md. Modified: stdlib-only types/dispatch and
campaign eval parameter adaptation. This is not a full upstream harness.

LoCoMo source is not distributed. Its new local metrics are implemented in
local_scoring.py from a separate specification, not an upstream port.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .local_adapters import score_local

# ---------------------------------------------------------------------------
# Apache-2.0 LME-V2 function adaptations; source pin and modifications in notices
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS: Sequence[str] = (",", ";")


def normalize_phrase(
    text,
    *,
    lower: bool = True,
    normalize_hyphen: bool = True,
    strip_punct: bool = True,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if normalize_hyphen:
        text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[,;]", " ", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_phrases(
    text,
    *,
    separators=DEFAULT_SEPARATORS,
    **normalize_kwargs,
):
    if text is None:
        return []
    separator_list = list(separators)
    if not separator_list:
        normalized = normalize_phrase(text, **normalize_kwargs)
        return [normalized] if normalized else []
    pattern = "|".join(re.escape(sep) for sep in separator_list)
    parts = re.split(pattern, text)
    normalized_parts = [
        normalize_phrase(part, **normalize_kwargs) for part in parts
    ]
    return [part for part in normalized_parts if part]


def norm_phrase_set_match(
    prediction,
    answer,
    *,
    separators=DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    for phrase in set(answer_phrases):
        pattern = r"\b%s\b" % re.escape(phrase)
        if re.search(pattern, normalized_pred) is None:
            return False
    return True


def norm_phrase_set_match_ordered(
    prediction,
    answer,
    *,
    separators=DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    start = 0
    for phrase in answer_phrases:
        pattern = r"\b%s\b" % re.escape(phrase)
        match = re.search(pattern, normalized_pred[start:])
        if match is None:
            return False
        start += match.end()
    return True


def mc_choice_match(
    prediction,
    answer,
    *,
    strip_chars: str = ".",
    require_non_empty: bool = True,
    **_,
) -> bool:
    if prediction is None or answer is None:
        return False
    if not isinstance(prediction, str):
        prediction = str(prediction)
    if not isinstance(answer, str):
        answer = str(answer)
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", prediction.lower())
    candidate = boxed_match.group(1) if boxed_match else prediction
    cleaned = re.sub(r"\b(choice|option)\b", "", candidate, flags=re.IGNORECASE)
    for ch in strip_chars:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().upper()
    expected = answer.strip().upper()
    if require_non_empty and (not cleaned or not expected):
        return False
    return cleaned == expected


_MULTI_SELECT_FILLER_WORDS = {
    "AND",
    "ANSWER",
    "ANSWERS",
    "CHOICE",
    "CHOICES",
    "FINAL",
    "LETTER",
    "LETTERS",
    "OPTION",
    "OPTIONS",
}


def _extract_multi_select_letters(text) -> list:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    chunks = re.findall(r"[A-Z]+", text.upper())
    letters: list = []
    for chunk in chunks:
        if chunk in _MULTI_SELECT_FILLER_WORDS:
            continue
        letters.extend(list(chunk))
    return letters


def mc_choice_set_match(
    prediction,
    answer,
    *,
    require_non_empty: bool = True,
    **_,
) -> bool:
    pred_letters = _extract_multi_select_letters(prediction)
    answer_letters = _extract_multi_select_letters(answer)
    if require_non_empty and (not pred_letters or not answer_letters):
        return False
    return set(pred_letters) == set(answer_letters)


def extract_boxed_answer(text: str) -> str:
    """Balanced-brace-aware \boxed{...} parser adapted from licensed LongMemEval-V2.

    Uses the LAST \boxed{ occurrence. Falls back to the whole stripped text
    when no marker is present or the parsed content is empty.
    """
    text = text or ""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def is_unknown(parsed_answer: str) -> bool:
    return parsed_answer.strip().lower() == "unknown"


# ---------------------------------------------------------------------------
# Dispatcher used by evaluate.py
# ---------------------------------------------------------------------------

def _phrase_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map eval.params (common-format) to norm_phrase_set_match kwargs."""
    return {
        "lower": bool(params.get("lower", True)),
        "normalize_hyphen": bool(params.get("normalize_hyphen", True)),
        "strip_punct": bool(params.get("strip_punct", True)),
    }


def score_deterministic(
    eval_type: str,
    params: Dict[str, Any],
    prediction,
    answer,
    question: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch one deterministic evaluation.

    Returns {"score": float, "details": {...}}; score is always in [0, 1].
    ``prediction`` is the model's (possibly family-preprocessed) answer text;
    ``answer`` is the gold. Local choice scoring requires explicit option
    mapping in params from the new converter; legacy questions are not accepted.
    """
    params = params or {}
    prediction = "" if prediction is None else str(prediction)
    answer = "" if answer is None else str(answer)

    if eval_type in {"f1", "abstain_f1", "em"}:
        return score_local(eval_type, params, prediction, answer, question)

    if eval_type == "phrase_set":
        kwargs = _phrase_kwargs(params)
        ok = norm_phrase_set_match(
            prediction, answer,
            separators=list(params.get("separators") or DEFAULT_SEPARATORS),
            require_non_empty=bool(params.get("require_non_empty", True)),
            **kwargs,
        )
        return {"score": 1.0 if ok else 0.0, "details": {
            "metric": "lme_v2_norm_phrase_set_match",
            "separators": params.get("separators"),
        }}

    if eval_type == "phrase_set_ordered":
        kwargs = _phrase_kwargs(params)
        ok = norm_phrase_set_match_ordered(
            prediction, answer,
            separators=list(params.get("separators") or DEFAULT_SEPARATORS),
            require_non_empty=bool(params.get("require_non_empty", True)),
            **kwargs,
        )
        return {"score": 1.0 if ok else 0.0, "details": {
            "metric": "lme_v2_norm_phrase_set_match_ordered",
            "separators": params.get("separators"),
        }}

    if eval_type == "mc_choice":
        ok = mc_choice_match(
            prediction, answer,
            require_non_empty=bool(params.get("require_non_empty", True)),
        )
        return {"score": 1.0 if ok else 0.0, "details": {"metric": "lme_v2_mc_choice_match"}}

    if eval_type == "mc_choice_set":
        ok = mc_choice_set_match(
            prediction, answer,
            require_non_empty=bool(params.get("require_non_empty", True)),
        )
        return {"score": 1.0 if ok else 0.0, "details": {"metric": "lme_v2_mc_choice_set_match"}}

    raise ValueError(f"Unknown deterministic eval type: {eval_type!r}")


DETERMINISTIC_TYPES = {"f1", "abstain_f1", "em", "phrase_set", "phrase_set_ordered", "mc_choice", "mc_choice_set"}
