# OCC-3 evaluation protocol, version 2

Protocol version 2 - pre-registered before final-test data collection.

This document is the versioned replacement for the pre-registered acceptance rules of the
PR #204 experiment plan (`reports/occ3-alignment-repair/EXPERIMENT-PLAN.md`, section 4) and
the PR #205 live-window numeric rule (`reports/occ3-alignment-repair/LIVE-WINDOW-1.md`,
"Verdict", the `analyze_probe.py` rule). It is fixed before any final-test data collection in
this campaign. No acceptance rule in it may be changed after final-test answers are read; a
change after a failed run requires a new protocol version and a fresh test set
(section 8).

## 1. What stands unchanged

- The PR #205 semantic result stands: 0 of 3 diagnostic trials recovered. Per the individual
  receipts this is 2 completed consumptions with semantic misses plus 1 uncertain attempt
  (handoff not admitted, no answer text), not 3 completed consumptions.
- The PR #205 verdict stands: INCONCLUSIVE (median within-turn held-out R2 0.1011 in the
  0.10-0.30 dead band; the cosine bound failed by a wide margin).
- No old failed trial is reclassified as a success or as a completed consumption. Old
  reports are history; where their prose over-claims, `README.md` section "Historical
  corrections" is the correction of record.
- The live transport facts stand: capture, admission, one-shot delivery, receipts and TP8
  worked on the real runtime in the PR #205 window.

## 2. Why the old rules are replaced (spec section 3, items A-G)

The old rules were a mixture of too strong and too weak. Each correction below is fixed in
this protocol before data collection.

| Item | Old rule | Verdict on the old rule |
|---|---|---|
| A | FEASIBLE required `median cos <= 0.30`; copier cosine 0.9 read as disqualifying | too strong |
| B | a low cosine was read as supportive | too weak |
| C | `cos <= 0.30` as a universal semantic acceptance gate | withdrawn |
| D | a result only counted if `P` beat `T` by `Delta_accuracy >= 0.10` | too strong |
| E | the two possible claims were conflated | both, in different places |
| F | within-turn held-out R2 was the deciding number | too weak |
| G | numerical sensitivity read as semantic evidence | too weak |

### A - High cosine is not intrinsically failure

Old rule: FEASIBLE required `median cos <= 0.30`
(`reports/occ3-alignment-repair/LIVE-WINDOW-1.md`, "Verdict"); "any candidate whose
cos(prefix, E) stays above 0.9 is not a candidate" (same file, section 5, item 3).
Too strong: high cosine similarity to token embeddings is not intrinsically a failure; an
independently fitted H-only map could legitimately predict useful token-like embeddings
accurately. Version 2: cosine to reference embeddings is a recorded diagnostic (copier
flag), never an acceptance gate.

### B - Low cosine is not proof of a useful channel

Old rule: the other direction of the same threshold, where a low cosine was read as
supportive. Too weak: random vectors have low cosine and convey nothing useful.
Version 2: no cosine threshold confers or removes acceptance. Acceptance comes from state
sensitivity measured under R1 plus receiver-task results.

### C - The universal cosine gate is withdrawn

`cos(prefix, reference_embedding) <= 0.30` is withdrawn as a semantic acceptance gate
entirely, for the reasons in A and B. It appears nowhere in this protocol.

### D - No requirement to beat exact references

Old rule: "a result is only counted if `P` also beats `T` by `Delta_accuracy >= 0.10`"
(`reports/occ3-alignment-repair/EXPERIMENT-PLAN.md`, section 4). Too strong: latent
communication does not need to beat an exact token-embedding or full-text reference merely to
establish that communication occurred. Version 2: `T` and `F` are interpretation references,
not pass conditions.

### E - Two claims, explicitly separated

CLAIM 1: sender-specific information crosses through an H-derived continuous representation
without ordinary text delivery. CLAIM 2: the representation transfers additional information
beyond what the sender's selected discrete tokens convey. The old plan conflated them.
CLAIM 1 is decidable with the P/W/N arms and does not require beating an answer-containing
text reference on a task where that reference is already perfect. CLAIM 2 is stronger and
needs additional controlled tasks beyond this campaign's final test; no CLAIM 2 success is
claimed by this protocol's gate.

### F - A within-turn split is not a lower bound

Old rule: the deciding number was a within-turn held-out R2, described as a "pessimistic
bound" (`LIVE-WINDOW-1.md`, section 3.2), and the internal fit gate used a row-order 80/20
split inside a single turn. Too weak as generalisation evidence: a within-turn split is not
a lower bound on turn-disjoint generalisation, and more calibration rows do not guarantee
improvement. Version 2: whole-turn partitions only (R3); within-turn measurements may be
collected as diagnostics and are never acceptance numbers.

### G - Numerical sensitivity is not semantic sensitivity

Old rule: hidden-state roll sensitivity 0.479 was reported beside the 0/3 semantic result.
Too weak as semantic evidence: numerical H-sensitivity does not prove semantic
H-sensitivity, and high sensitivity can be noise amplification. Version 2: the frozen-map
state sensitivity metric (R1) is measured with the same frozen artifact and the same
normalisation in both terms, and every sensitivity result is paired with receiver-task
results. Sensitivity alone never confers acceptance.

## 3. New acceptance rules

### R1 - Frozen-map state sensitivity

For a frozen artifact f and its selected serving normalisation (and anchoring, if any):

```
S_wrong = || f(H_correct) - f(H_wrong) ||_F / max( || f(H_correct) ||_F, numerical_floor )
```

- The SAME frozen artifact, normalisation and anchoring appear in both terms. A raw shuffled
  output is never compared against a normalised correct output; that measures scaling
  mismatch, not state dependence.
- H_wrong is a matched-shape control: a different sender's states (wrong-sender arm W) or a
  row permutation of the correct states (arm S, an order/correspondence ablation).
- For the historical per-message baseline, any reported sensitivity must state explicitly
  whether the intervention refits the map or holds the fit fixed; the two are different
  experiments.
- Sensitivity is reported together with receiver-task results, never alone.

### R2 - First live control gate

Before either alignment method is tuned or scored, the existing framed injection must pass the
diagnostic reference control (spec section 5): T-exact (actual input-token embeddings of the
sender-private diagnostic message) versus F-text (the same message in ordinary text), with
T-norm, N-shape and N-none as the accompanying controls. The full arm definitions and stop
rules are pre-registered in `SEMANTIC-PROTOCOL.md` section 3. The governing rule: if T-exact
fails while F-text succeeds, alignment tuning stops and injection/framing/position handling
is fixed first. No alignment measurement is acceptance evidence before this gate is decided.

### R3 - Partition discipline

- Three partitions exist: FIT, VALIDATION, FINAL TEST, split on whole source turns, never
  rows. Related turns are grouped by source fact/context family before splitting so
  near-duplicate prompts never cross partitions (`occ3_sglang/calibration_corpus.py:196`,
  `split_manifest`).
- The split manifest is frozen (with digests per partition) before measurement. FIT fits;
  VALIDATION selects method, rank, regularisation and serving normalisation; FINAL TEST is
  never used for fitting or selection, and is never refitted after its results are read.
- All fitted statistics come from FIT only. A message is centred with the frozen mu_H, never
  its own mean; PCA bases are never recomputed from a message; E_message never fits or
  modifies its own transformation.
- Only accepted rows with attested reference embeddings are retained. No zero padding and no
  duplicated rows to reach an M target.

### R4 - Promotion gate and measurement/promotion separation

- Promotion happens only in `validate_for_activation(artifact, validation_record, policy)`
  (`occ3_sglang/calibration_corpus.py:634`), which consumes a policy of fixed numeric
  thresholds and a validation record bound to the artifact digest. `fit_map` and
  `evaluate_map` never gate.
- Research measurement and deployment promotion are different operations. A poor or negative
  measurement is recorded, not suppressed: negative R2 values are reported as they are, and
  an undefined denominator is recorded as undefined, never fabricated as zero.
- An artifact that fails promotion may still be measured. An explicitly authorised research
  profile may not masquerade as a promoted method.

## 4. What is measured

Every metric in `EQUATIONS.md` section 7 is reported on independent held-out turns, before
serving normalisation and again after it, with raw and served values kept distinct. Per-turn
metrics, the turn-macro average and the pooled row-weighted view are all reported; row-count
imbalance across turns is documented rather than hidden by pooling. The fit/validation/test
error gap is reported. `r2_standard` and `r2_vs_fit_mean` are both named and reported
separately; neither is called "held-out R2" without qualification.

## 5. Dataset plan

Starting engineering sample sizes, not statistical laws:

| Partition | Minimum | Use |
|---|---|---|
| FIT source turns | >= 32 independent turns | fit means, PCA bases, covariance scales and maps |
| VALIDATION turns | >= 8 | select method, rank, regularisation, serving normalisation |
| Alignment-test turns | >= 16 | independent alignment metrics (FINAL TEST of the alignment stage) |
| Final semantic test | 64 trials, separate task set | causal semantic decision (`SEMANTIC-PROTOCOL.md` section 9) |

- Row budgets: M in {512, 1024, 2048} for the FIT rows, selected as nested deterministic row
  selections across budgets so the smaller budget is a prefix of the larger
  (`occ3_sglang/calibration_corpus.py:245`, `select_calibration_rows`; turn order is the
  ascending sha256 of the turn id, never insertion order).
- Per-turn contribution is capped so one long reasoning trace cannot dominate the fit. Actual
  rows and unique turns are reported for every budget.
- Resource limits are recorded before collection. A smaller run is a pilot, not a completed
  final validation, and is reported as such.
- Shared vocabulary across partitions is allowed; shared private evaluation facts are not.

## 6. Prohibitions

- Do not select rank or regularisation from final-test answers.
- Do not refit on final-test turns after inspecting their results.
- Do not add a metric after seeing the data.
- Do not reclassify an old failed trial.
- Do not put a real private answer into a free-generation receiver prompt (the embedding-only
  reference is an explicit oracle arm; `SEMANTIC-PROTOCOL.md` section 3).

## 7. Claim coverage of this protocol

This protocol's gates can at most establish CLAIM 1 (section 2, item E) on the pre-registered
task families. They cannot establish CLAIM 2. Any statement about information beyond the
sender's selected discrete tokens requires additional controlled tasks and is out of scope
for the promotion gate in `SEMANTIC-PROTOCOL.md` section 10.

## 8. Version control of this protocol

- Version 2 replaces the PR #204/#205 rules as of this document.
- If the final semantic run fails, the result is reported as a failure. Additional testing
  requires a new protocol version and a fresh test set. Testing is never continued until a
  favourable confidence interval appears.
- Every change to this file before final-test data collection is recorded in the campaign
  reports with its reason; no change after data collection starts.
