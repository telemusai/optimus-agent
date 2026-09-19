# Mandatory Codex wrap-up: repeated failed compaction

Status: OPEN — required by the user after the Flash teams finish; not resolved by the worker's later recovery.

Evidence: C:/Users/openclawuser/optimus-jev-compare-20260919/COMPACTION_INCIDENT_20260919.json.
Related existing repairs: CF-01 (retry amplification) and CF-02 (incomplete summarization/terminal diagnostics). Treat this as linked live evidence, not automatically a separate duplicate bug.

## Required at the big wrap-up
1. Review the final integrated CF-01/CF-02 changes against this incident, and implement any missing correction in isolated source.
2. Investigate why the summarizer returned incomplete responses. Identical 13,107 output-token counts are a possible output-limit clue, not a proven stop reason. Inspect captured terminal metadata or reproduce with a fake provider; do not guess or enlarge live model limits.
3. Prove repeated threshold failures do not cause a fresh expensive attempt after every small continuation/tool step. Recovery/backoff must be bounded, preserve responsiveness, and not permanently prevent successful compaction.
4. Prove rejected/partial summaries never replace the existing conversation/checkpoint, never lose pending human or agent messages, and never replay partially completed tool work.
5. Test explicit output-limit termination separately from unknown incomplete, cancellation, refusal/filtering and transport interruption. Only allow specifically justified bounded recovery; never accept a truncated summary just to suppress the error.
6. Verify manual/overflow recovery and subsequent successful automatic compaction still work, and that diagnostics distinguish an actual compaction failure from eventual success.
7. Record exact candidate source, test commands/results, remaining uncertainty, and a separate verdict for this incident in the wrap-up report. If not proved fixed, keep it OPEN and state the blocker; do not count eventual production recovery as a repair.

Codex owns independent final verification after the teams finish. Existing authors/reviewers may include supporting tests now, but this instruction does not ask them to stop, restart or begin another broad audit. The running Optimus installation remains unchanged. No cutover, restart or PR without renewed authority.
