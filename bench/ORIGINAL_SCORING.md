# Independent local scoring

Identity: `optimus-local-scoring/1.0.0`. This is a new local metric, not official
LoCoMo or LoCoMo-Plus scoring and not an equivalence claim.

`eval/local_scoring.py` was written by the separate `astra-original-metrics`
worker from the parent's mathematical/rubric-only task. The worker reports no
inspection of held evaluators, old prompts, benchmark answers, reference repos,
or cached third-party evaluation source. It read only AGENTS.md and runtime
message/edit API instructions. Generic token/set/best-match utilities were
requested by their mathematical definitions, without supplied source code.

The verified proposal had 74 synthetic stdlib tests. Its raw source SHA256 is
`0e5d64df9e01064fa22755675b82314a2623fb63b20151ac68074053d160d4ba`; its normalized UTF-8/LF SHA256 is
`36f82134ea0e383c64f3addb61709b029f02f3fc1b106dc4584c1cc8ab9cbd8b`. The source is included unchanged. Tests are included in
`tests/test_local_scoring.py` with only their import changed to the package path.
The repository portability suite also loads those tests. No model was called.

Local lexical policy: lowercase, delete ASCII punctuation, split whitespace,
remove a/an/the (not “and”), then apply an optional per-token callable. F1 is
multiset overlap; the empty/empty F1 is zero. Token-set equality and the directed
mean of reference-best candidate F1 are separate generic operations.

Local semantic policy separates trusted instructions from JSON-encoded evidence.
The parser accepts exactly a bare JSON object with `correct: bool` and a
nonblank `reason` of at most 400 characters. It rejects numeric coercion,
extra/missing/duplicate fields, fences, prose, nonfinite constants and oversize
responses. This only validates response syntax; it does not prove judge quality
or immunity to prompt injection. Request hashes include the actual rubric/data.

`local_adapters.py` contains separate new integration glue: category routing,
optional licensed NLTK stemming, comma-part aggregation, and exact local choice
association. Those additions are not claimed to be an independently reviewed
upstream evaluator. Authorship provenance describes this process; it is not a
legal opinion, rights grant, provider identity attestation, or benchmark result.
