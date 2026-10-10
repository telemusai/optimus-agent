# Third-party source notices and modifications

This directory is not an MIT-only bundle. Original Optimus scaffolding remains
under the repository license. The specific copied/adapted source below retains
its upstream license. Complete pinned license texts are in `licenses/`.
The current distribution excludes CC-BY-NC LoCoMo evaluator/presentation source
and LoCoMo-Plus upstream prompts, parser, duration helper, and stitching code.
No MemoryLake code or artifacts are included.

## LongMemEval — MIT

Copyright (c) 2024 Di Wu. Complete permission and warranty notice:
[`licenses/LongMemEval-MIT.txt`](licenses/LongMemEval-MIT.txt).

Source: `xiaowu0162/LongMemEval`, commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc`,
`src/evaluation/evaluate_qa.py` (`get_anscheck_prompt`).

Boundary: `eval/judges.py` constants `LME_TEMPLATE_SINGLE`,
`LME_TEMPLATE_TEMPORAL`, `LME_TEMPLATE_KNOWLEDGE_UPDATE`,
`LME_TEMPLATE_PREFERENCE`, `LME_TEMPLATE_ABSTENTION`, `LME_TEMPLATES`, and
`build_longmemeval_prompt`. The adapter `judge_longmemeval` uses those templates.
Modified: separated templates/functions, explicit DGX transport, cache protocol,
stdlib client, 2048-token campaign budget instead of upstream 10, and offline
opt-in policy. Whole-harness equivalence is not claimed.

## LongMemEval-V2 — Apache-2.0

Source project: `xiaowu0162/LongMemEval-V2`, commit
`2cc8c540bdb87fe6761629b585e727e1c4704520`,
`evaluation/qa_eval_metrics.py`. Complete license:
[`licenses/LongMemEval-V2-Apache-2.0.txt`](licenses/LongMemEval-V2-Apache-2.0.txt).
The inspected pinned source supplies no extra copyright header or NOTICE file;
no substitute copyright owner/year is invented here.

Boundaries:
- `eval/deterministic.py`: `DEFAULT_SEPARATORS`, `normalize_phrase`,
  `split_phrases`, `norm_phrase_set_match`, `norm_phrase_set_match_ordered`,
  `mc_choice_match`, `_MULTI_SELECT_FILLER_WORDS`, `_extract_multi_select_letters`,
  `mc_choice_set_match`, `extract_boxed_answer`, and `is_unknown`.
- `eval/judges.py`: both `V2_*_JUDGE_SYSTEM_PROMPT` constants,
  `v2_build_abstention_judge_messages`, `v2_build_gotchas_judge_messages`,
  `_v2_stringify_text`, `_v2_strip_markdown_code_fence`, and
  `v2_parse_llm_binary_judgement`; two `judge_lme_v2_*` adapters.
- `converters/common.py`: `decode_lme_v2_eval` and its function-name mapping are
  a limited campaign adaptation of the eval-function specification.

Modified: stdlib typing/dispatch, limited bool/string eval-option decoder,
common-format adapter, boxed-answer integration, separate DGX transport,
`max_tokens` rather than `max_completion_tokens`, shorter timeout/retry policy,
cache namespace and source binding. The decoder does not reproduce every
upstream option form. No newly expanded general-input parity is claimed.

## NLTK — Apache-2.0

Copyright (C) 2001-2026 NLTK Project. Complete license:
[`licenses/NLTK-Apache-2.0.txt`](licenses/NLTK-Apache-2.0.txt).
Source: `nltk/nltk`, commit `b417a98a9497b4a27067043e35e64c388e060686`,
`nltk/stem/porter.py`. The inspected pinned tree has no NOTICE file.

Boundary: `eval/porter.py`, including class `PorterStemmer`, its rule lists and
method bodies. This is a **modified source port**, not a no-copy implementation.
Modified: removed NLTK dependencies/base class, CLI and extended documentation;
added standalone helpers and type hints; iterative consonant flags. Algorithm
citation: M. Porter, “An algorithm for suffix stripping,” Program 14.3 (1980),
130–137, with NLTK extensions. Original local scoring can accept this stemmer as
a caller-supplied token transform; that does not relicense the stemmer as MIT.

## Independent local scoring

`eval/local_scoring.py` is new implementation from a math/rubric-only task,
written without reading the held/reference evaluators. Authorship process,
source digest, tests and limitations are recorded in `ORIGINAL_SCORING.md` and
`PROVENANCE.json`. Its label is `optimus-local-scoring/1.0.0`, not “official”
LoCoMo or LoCoMo-Plus scoring. Integration/category routing and normalized-data
adapters are new Optimus glue. This statement describes the process; it is not
an external legal opinion or a general claim of exclusive authorship.

License/source URLs and raw license hashes are pinned in `PROVENANCE.json`.
Public license reads used unauthenticated read-only HTTPS. Data rights are
separate: no real datasets, source archives, historical scores or caches ship.
