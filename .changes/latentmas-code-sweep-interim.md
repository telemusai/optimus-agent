Added:
  - HumanEval+ (n=164) benchmark results for baseline, TextMAS sequential and
    TextMAS hierarchical on GLM-5.3: 86.6%, 87.2% and 90.9% accuracy.
  - MBPP+ (n=378) baseline and TextMAS sequential results: 80.2% both.
  - Per-item sanitized result rows (correctness, tokens, latency) for the eight
    completed code-task and ARC-Easy conditions with source sha256 provenance.
  - Record of the two later GSM8K item-19 native OCC-3 authorization attempts
    (one rejected before work, one failed_uncertain after 137.7 s); no score.
  - Completed the GPQA-Diamond baseline results file: 198 rows (160 historical
    + 38 continuation rows joined by index), 73/198 correct, with segment
    provenance recorded in the summary JSON.
  - MBPP+ (n=378) TextMAS hierarchical results: 323/378 correct, 85.45%
    accuracy, McNemar vs baseline b=6, c=26, p=0.000535.
  - ARC-Easy (n=300) baseline results: 285/300 correct, 95.0% accuracy.
  - ARC-Easy (n=300) TextMAS sequential results: 290/300 correct, 96.67%
    accuracy, McNemar vs baseline b=6, c=11, p=0.332.
  - ARC-Easy TextMAS hierarchical (200/300, 66.67%) and the full ARC-Challenge
    sweep: baseline 286/300 (95.3%), TextMAS sequential 297/300 (99.0%,
    McNemar p=0.0074), TextMAS hierarchical 215/300 (71.7%).
  - Fresh full-300 GSM8K sweep: baseline 280/300 (93.3%), TextMAS sequential
    277/300 (92.3%), TextMAS hierarchical 286/300 (95.3%).
  - Data-quality note for the 2026-11-11 sweep: ARC-Challenge sequential and
    hierarchical rows deduplicated after a restart race (one row per item
    kept; accuracy unaffected; latencies not comparable); mixed 16/32 client
    in-flight concurrency across conditions.
