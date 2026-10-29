# OCC-3 causal semantic experiment protocol

Protocol version 2 - pre-registered before final-test data collection.

This protocol is fixed before any final-test data collection. Every gate, arm, task family,
scoring rule and threshold below is pre-registered. Nothing in this file may be changed after
final-test answers are read; a failed final run can only be followed by a new declared
protocol version and a fresh test set (section 12).

## 1. What this experiment can decide

- CLAIM 1: sender-specific information crosses through an H-derived continuous representation
  without ordinary text delivery.
- CLAIM 2: the representation transfers additional information beyond what the sender's
  selected discrete tokens convey.

The pre-registered gate in section 10 addresses CLAIM 1 only. CLAIM 2 needs additional
controlled tasks and is not claimed by this campaign's gate.

## 2. Participants and runtime

- Two independent Optimus sessions act as sender and receiver on the exact same compatible
  GLM-5.3 runtime (FP8, TP8, pinned SGLang 0.5.21 on DGX).
- The Ollama coding model is never one of the participants.
- Single-use state leases are maintained. Consumed handles are never replayed. Fresh
  authorised captures are used, or a separately authorised experimental snapshot mechanism
  with new one-shot leases. The production one-shot contract is not weakened.

## 3. First live control (runs before any alignment tuning)

The existing framed injection is tested with a deliberate diagnostic reference before either
alignment method is tuned or scored. Sender-private, multi-token diagnostic message (example):
"The selected container is the cedar container." The receiver gets a neutral question and the
same public candidate set in every arm.

| Arm | Content | Role |
|---|---|---|
| T-exact | Actual input-token embeddings of that message, constructed inside the authorised diagnostic runtime | readability oracle |
| T-norm | The same embeddings with the proposed serving norm calibration | normalisation diagnostic |
| N-shape | Same number and positions of injected rows, labelled null or mean-embedding control | shape control |
| N-none | No attached prefix | sequence-geometry baseline, kept separate |
| F-text | The message in ordinary text | answerability reference |

These are diagnostic controls, not hidden-state success claims. Checks in every arm: exact
tokenisation and message boundaries; marker insertion inside the intended chat turn; correct
sequential positions and causal attention mask; chunked-prefill overlap; no reinjection during
TARGET_VERIFY; identical ordinary framing within each matched contrast; correct execution on
all required TP ranks.

Pre-registered stop rules:

1. If T-exact fails while F-text succeeds, stop alignment tuning and fix
   injection/framing/position handling first. No alignment measurement is acceptance
   evidence before this gate is decided.
2. If T-exact succeeds but T-norm fails, investigate normalisation. Do not attribute that
   result to Procrustes.
3. A real private answer is never put into the free-generation receiver prompt; the
   embedding-only reference is an explicit oracle arm.

## 4. Semantic experiment arms

| Arm | Reserved-row content | Role |
|---|---|---|
| P | Correct sender H through the frozen candidate map | positive |
| W | Wrong sender H carrying a different private fact/answer | primary negative control |
| S | Row-permuted H through the SAME frozen candidate map | order/correspondence ablation |
| N | Matched-shape null (labelled zeros) or calibrated mean-prefix control | primary negative control |
| N0 | No attachment at all, plain request | no-attachment baseline, separately labelled |
| T | Selected reference-token embeddings only | interpretation reference (token channel) |
| F | Full-text reference of the same message | interpretation reference (text ceiling) |

- Prefix length, insertion position, ordinary prompt, decoding configuration and task
  difficulty are matched across arms wherever possible. N0 and N are never merged: they have
  different sequence geometry.
- W is scored both ways: recovery of the intended sender's answer AND recovery of the wrong
  sender's answer. A receiver switching to the wrong sender's answer is evidence that the
  attached state controls its output.
- S is an order/correspondence ablation, not a universal information-free control: a row
  permutation does not always destroy message meaning. W and N are the primary negative
  controls. (The offline control library scores `shuffled` as a row reversal for
  reproducibility without a seed, `occ3_sglang/calibration_corpus.py:348-350`; a live S arm
  uses a genuine row permutation and records it.)
- T and F are interpretation references. Neither is a pass condition for CLAIM 1
  (`EVALUATION-PROTOCOL.md` section 2, item D).

## 5. Task families

Balanced sender-private tasks, four families:

1. Simple category recovery (four-way candidate lists).
2. Random assignments involving invented entities.
3. Small private facts requiring a receiver decision.
4. Short exact strings, as a separate fidelity test.

- A single `hello` success is a smoke test, not a scientific gate, and is never reported as
  evidence.
- Public candidate lists are allowed in every arm of a four-way task, including the correct
  label's presence in the list. That is not answer leakage. What must remain private is which
  candidate is correct and the fact-to-answer binding.
- The leakage audit must therefore not contain a broken assertion that forbids every
  occurrence of the correct label even in a balanced public candidate list.

## 6. Leakage audit (per trial)

Mechanically assert that the private answer does not appear in: ordinary messages, tool
arguments, metadata, routing labels, filenames, error strings, telemetry, the state handle and
the artifact name. Reuse `tests/test_semantic_leakage.py`. An arm that fails the audit is
excluded from counted successes and reported as a leakage failure.

## 7. Scoring

Primary endpoint: actual generated answers, scored as exact semantic accuracy on the
four-way tasks (and exact-string accuracy for the fidelity family, reported separately).

Candidate-answer likelihood scores (secondary):

```
score(a | c) = sum_j log P(a_j | c, a_1, ..., a_(j-1))
margin = score(correct | c) - max score(wrong_candidate | c)
```

- The actual answer-token boundary and tokenisation are used. GLM may produce reasoning before
  the final answer; the first generated token is not automatically an answer token.
- A score may be called a "first-token forced-choice margin" only when all of these hold:
  (a) it is evaluated at the intended answer position; (b) every candidate is validated as
  exactly one token at that position; (c) the requested candidate probabilities are actually
  returned. A missing candidate from top-k output is not probability zero.
- If teacher-forced scoring is needed, it is kept separate from free-generation trials, all
  candidate continuations are scored symmetrically, and a ground-truth continuation is never
  fed back into a free-generation run and counted as recovery.
- Normal reasoning effort is never reduced globally to simplify scoring. If reliable scoring
  is unavailable, it is marked unavailable and generation accuracy remains the primary
  endpoint.

## 8. Pilot

- 16 balanced tasks, used for debugging only. No pilot number is acceptance evidence.
- After the pilot, freeze: the selected map, its artifact, the framing and the analysis
  script. The pilot may not tune thresholds.

## 9. Final semantic test

- 64 fresh paired tasks, disjoint from the pilot and from all calibration turns.
- P, W and N run on every final task. T and F run as interpretation references.
- Every trial uses a fresh authorised capture; consumed handles are never replayed.

## 10. Pre-registered promotion gate (fixed before final testing)

Engineering release criteria, not mathematical constants. All must hold:

1. P exact semantic accuracy >= 0.75 on the four-way task.
2. P minus W accuracy >= 0.20.
3. P minus N accuracy >= 0.20.
4. Paired 95% confidence intervals for both improvements exclude 0.
5. Every counted P success has valid native completed-use evidence (a completed-use receipt)
   and passes the no-text-leakage audit.

These are not adjustable after reading final-test answers. The gate decides CLAIM 1 only
(section 1).

## 11. Statistics

- Bootstrap over paired independent tasks; never token rows, never GPU ranks.
- Facts from the same source family are kept in the same resampling unit
  (family-grouped resampling).
- Uncertainty is reported and multiple primary comparisons are accounted for.
- Reporting rules: accuracy over all attempts; accuracy conditional on completed native use;
  rejections, timeouts, uncertain outcomes and semantic misses reported separately and never
  merged. Interrupted trials are never silently dropped.
- An Uncertain attempt is never resent as ordinary text.

## 12. After a failed final run

If the final run fails the gate, the result is reported as a failure. If additional testing is
needed, a new protocol version and a fresh test set are declared first. Testing is never
continued until a favourable confidence interval appears.
