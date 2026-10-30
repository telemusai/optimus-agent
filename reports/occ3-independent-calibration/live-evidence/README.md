# Live-evidence landing — occ3-icalib briefs (2026-10-29 window)

Provenance: copied from the root-owned operator briefs directory
`C:/Users/openclawuser/occ3-icalib-briefs/` (never committed before) so the repo
carries the live P2/P3 record of PR #213's campaign window. Curated by the root
agent on 2026-10-30 during the full-recovery campaign (branch
`fix/occ3-glm53-full-message-recovery`). No content was edited; selection only.

## Contents

- `PR-BODY-RESULTS.md` — consolidated results body of the independent-calibration
  PR (#213): offline implementation, first attested live DGX window
  (`icalib-20261029T032600Z`), two controlled swap attempts, minted
  `native_initial` checkpoint proof, two live blockers root-caused and fixed.
  Explicitly records what was NOT run (corpus, fit, calibrated swap, semantics).
- `overnight-supervision.log` — supervisor log of the overnight period between
  the first attested window and the follow-up swap (SSH binding expiry gaps,
  staged-attempt discipline, observer idle-correct behavior).
- `p3-controls/` — the five-arm first-live-control kit as designed pre-#213:
  arm schedule (json + md), fit-exclusion format + template, the draft control
  driver `p3_first_live_control_draft.py`, and the arm control files
  (`arms/control-*.txt`). NOTE (post-audit): these control files implement
  server-side arm selection only; N-none and F-text both mapped to an absent
  control file, i.e. the original NATIVE aligner. The five live arms recorded on
  2026-10-30 (window `04f`) therefore measured attached-baseline duplicates for
  those two arms. The corrected driver/arms surface is implemented in the
  full-recovery branch; see `reports/occ3-fullrecovery/` in the campaign
  workspace for the driver-arms audit.
- `livebefore-diagnosis/report.md` — pre-window diagnosis of the
  live-before-candidate partial-recovery behavior.
- `livebefore-diagnosis/corrected-probe/` — five timestamped corrected-probe
  turns (2026-10-29): source payload JSON, raw source SSE stream, and receiver
  answer for each. Primary evidence for: contextual transfer working through the
  attached baseline, exact value/nonce recovery NOT demonstrated, and the
  receiver answering from context rather than the latent state. The 115514Z turn
  has no receiver answer (attempt aborted; retried at 120104Z).

## Excluded (still operator-local)

Raw truncation/window experiment dumps (`lf*`, `trunc*`, `var*`,
`offline-crash-dump.bin`), p2-swap-live-log.md and dgx-glm53-diagnostics.md
remain in the operator briefs directory; PR-BODY-RESULTS.md summarizes them.

## Integrity

Files copied verbatim (timestamps preserved by copy2). No secrets: scanned for
hex-64 keys, JWTs, bearer tokens, private keys, `sk-` patterns; all 64-hex
matches are sha256 digests or container/file ids.
