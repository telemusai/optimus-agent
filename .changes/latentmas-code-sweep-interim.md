- Added HumanEval+ (n=164) benchmark results for baseline, TextMAS sequential
  and TextMAS hierarchical on GLM-5.3: 86.6%, 87.2% and 90.9% accuracy.
- Added MBPP+ (n=378) baseline and TextMAS sequential results: 80.2% both.
- Added per-item sanitized result rows (correctness, tokens, latency) for the
  completed code-task and ARC conditions with source sha256 provenance.
- Added records of two earlier GSM8K item-19 native OCC-3 authorization
  attempts (one rejected before work, one failed_uncertain after 137.7 s);
  neither attempt has a score.
- Added completed GPQA-Diamond baseline results: 198 rows (160 historical +
  38 continuation rows joined by index), 73/198 correct, with segment provenance.
- Added MBPP+ (n=378) TextMAS hierarchical results: 323/378 correct, 85.45%
  accuracy, McNemar vs baseline b=6, c=26, p=0.000535.
- Added ARC-Easy (n=300) baseline results: 285/300 correct, 95.0% accuracy;
  TextMAS sequential: 290/300, 96.67%, McNemar b=6, c=11, p=0.332;
  TextMAS hierarchical: 200/300, 66.67%.
- Added the full ARC-Challenge sweep: baseline 286/300 (95.3%), TextMAS
  sequential 297/300 (99.0%, McNemar p=0.0074), hierarchical 215/300 (71.7%).
- Added a fresh full-300 GSM8K sweep: baseline 280/300 (93.3%), TextMAS
  sequential 277/300 (92.3%), hierarchical 286/300 (95.3%).
- Added a data-quality note for the 2026-11-11 sweep: ARC-Challenge sequential
  and hierarchical rows deduplicated after a restart race (first row per item
  kept; scores unchanged; latencies not comparable); mixed 16/32 client
  in-flight concurrency across conditions.
- Added MedQA TextMAS hierarchical, completing the 27/27 matrix: 109/300
  (36.33%), McNemar b=103, c=38, p=4.1e-08; low hierarchical letter-choice
  scores now span GPQA, ARC-Easy, ARC-Challenge and MedQA.
- Added the completed, pinned-scorer GSM8K native OCC-3 v7 cohort: 17/19
  eligible items correct (89.5%), exact McNemar p=1.0 vs paired baseline;
  item 19 completed after the 0.1.45 daemon upgrade. Published safe rows,
  summary and public scores. The interim 16/17 transcript-join figure is
  superseded by the pinned-scorer 15/17 batch result (item 2 formatting loss).
- Added native OCC-3 v7 ARC-Easy and ARC-Challenge a-cohorts (items 0-9):
  10/10 correct on each, with transport/cleanup/format gates valid. Paired
  baselines score 10/10 each; sequential scores 10/10 and 9/10 respectively
  (ARC-Challenge b=0, c=1, p=1.0); hierarchical scores 6/10 and 7/10.
  Published safe rows, summaries and public scores.
- Added separately scored native ARC-Easy items 10-19 and ARC-Challenge
  items 10-19, all correct, with their public scorer reports. Native totals
  are 20/20 on the first 20 items of each task, by arithmetic across disjoint
  score reports, not a combined scorer run. Paired single-agent baselines also
  score 20/20, so these items show no native gain.
- Added recovered historical native sandbox aggregates: HumanEval+ and MBPP+
  8/9 each. Full scorer/sandbox source validation and per-item coverage remain
  pending; no transport, latency, cost or channel-only gain is inferred.
- Added safe exclusion metadata for HumanEval+ item 9, MBPP+ item 9 attempts
  and ARC-Easy item 12. Failed attempts do not count as scored answers. The
  follow-up audit of the failed ARC-Easy item-12 attempt found no transfer
  and clean server state; item 12 is scored from a separate fresh attempt.
- Added a results-at-a-glance table for all nine benchmarks and a native
  OCC-3 v7 progress table: 77 of 100 planned first-20 items scored.
- Fixed stale NOT-RUN and sandbox prose to match completed HTTP conditions
  and the limited native evidence. Four-call TextMAS remains separate from
  the one-call baseline; execution budgets are not matched.
