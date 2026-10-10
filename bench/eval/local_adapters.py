"""Original category routing for Optimus-local-scoring/1.0.0.

This is not a LoCoMo upstream implementation. Generic metric mathematics lives
in the separately authored local_scoring module. Choice scoring uses only the
explicit label association recorded by the versioned local converter.
"""
from __future__ import annotations

from .local_scoring import SCORING_VERSION, lexical_f1, mean_best_match, token_set_equal
from .porter import porter_stem


def score_local(kind, parameters, prediction, reference, question=None):
    if kind == "f1":
        if parameters.get("variant") == "comma_multi_answer":
            value = mean_best_match(prediction.split(","), reference.split(","), porter_stem)
        else:
            delimiter = parameters.get("gold_truncate_at")
            expected = reference.partition(delimiter)[0].strip() if delimiter else reference
            value = lexical_f1(prediction, expected, porter_stem)
    elif kind == "em":
        value = float(token_set_equal(prediction, reference))
    elif kind == "abstain_f1":
        options = parameters.get("option_map")
        expected_label = parameters.get("abstention_label")
        if (not isinstance(options, dict) or set(options) != {"a", "b"}
                or expected_label not in options or not all(isinstance(text, str) for text in options.values())):
            raise ValueError("local choice scoring needs explicit option_map and abstention_label; regenerate questions")
        choice = prediction.strip().casefold()
        accepted = {expected_label, f"({expected_label})", options[expected_label].strip().casefold()}
        value = float(choice in accepted)
    else:
        raise ValueError("unsupported local scoring kind")
    return {"score": value, "details": {"metric": SCORING_VERSION, "kind": kind}}
