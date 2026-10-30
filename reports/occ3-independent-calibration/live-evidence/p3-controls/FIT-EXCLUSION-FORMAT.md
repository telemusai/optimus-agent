# P3 fit-exclusion turn-id list format (DRAFT — OFFLINE, NOT RUN)

Purpose: give P5 a machine-checkable format that excludes ALL P3 diagnostic families from the
fit, validation and test partitions and from every M-budget selection, without deleting anything
from the corpus. This is mandated by the staged arm-order document
(`deploy/v3/live-instrumentation/stage_build.py`, key `p5_exclusion`):
"P3 families are excluded from the P5 fit by an explicit turn-id list recorded in the window
evidence; they are not removed from the corpus."

Companion files: `fit-exclusion.template.json` (the schema, with `<WIN>` placeholders),
`arm-schedule.json` (the turn-id ranges the template references), `turns.jsonl` (the driver's
per-turn log the filled list is generated from).

## 1. Why turn ids are the right key

The corpus row keys are fixed by the collector
(`extensions/sglang-occ3/occ3_sglang/calibration_collector.py`):

* `turn_id` = the capture's `capture_id` (`latent_worker.py:389`,
  `capture.submit(record.descriptor.capture_id, ...)`);
* `family_id` = the source session id (same call, second argument);
* both must match `_IDENTIFIER = [A-Za-z0-9._:/-]{1,256}`
  (`calibration_corpus.py`; same class as `attachment_contract.py:31`).

Only source-capture turns can ever reach the corpus (receiver attaches are consume-only;
plain arms have no OCC-3 keys at all). P3's three source captures are therefore the complete
set of possible P3 corpus rows, and they all share ONE source session, so they are ONE corpus
family. The exclusion list still enumerates every turn id and session id of the window, so a
misconfigured future capture (for example a v2-policy retry of the cedar message) is also caught
by the reserved-prefix rule.

## 2. The format: `occ3.icalib.p3-fit-exclusion.v1`

One JSON object per window, written by the driver at window close into the window evidence
directory as `fit-exclusion.json` (template with `<WIN>` placeholders:
`fit-exclusion.template.json`). Fields:

| field | type | content |
|---|---|---|
| `schema` | str | `occ3.icalib.p3-fit-exclusion.v1` |
| `window` | str | the window id (`icalib-YYYYMMDDTHHMMSSZ`) |
| `reserved_prefixes` | [str] | `["p3-"]`; any corpus turn_id/family_id starting with one is excluded, every window |
| `turn_id_grammar` | str | `p3-<WIN>-t<NNNN>` (documented so P5 can parse/generate ids) |
| `turn_id_ranges` | [obj] | per arm: `arm`, `roles`, `first`, `last` — contiguous, in execution order |
| `excluded_turn_ids` | [str] | the capture ids actually used this window (corpus `turn_id` key) |
| `excluded_family_ids` | [str] | the source session id(s) (corpus `family_id` key) |
| `excluded_session_ids` | [str] | every enrolled P3 session (source + receiver) |
| `other_diagnostic_namespaces` | obj | reserved prefixes of other diagnostic phases (`p7-` pre-declared) |
| `application` | obj | where and how P5 applies the exclusion (below) |
| `generated_from` | obj | `turns.jsonl` path + sha256, schedule path |

Fill rules at window close (driver, `write_fit_exclusion`):

1. `excluded_turn_ids` = every `turn_id` in `turns.jsonl` with `role == "source-capture"`
   (submitted or not — exclusion is unconditional, even for failed arms).
2. `excluded_family_ids` = `["p3-<WIN>-src"]`; `excluded_session_ids` = source + receiver
   sessions.
3. `turn_id_ranges` copied from `arm-schedule.json` with `<WIN>` substituted.
4. `generated_from.turns_jsonl_sha256` = sha256 of the closed `turns.jsonl`.

## 3. How P5 applies it (binding)

1. Load the corpus (partitions are rebuilt at fit time; the frozen files are never edited).
2. Drop every turn whose `turn_id` is in `excluded_turn_ids`, OR whose `turn_id` starts with any
   `reserved_prefixes` entry, OR whose `family_id` is in `excluded_family_ids` or starts with any
   `reserved_prefixes` entry. Exclusion is family-atomic, matching `split_manifest`'s own
   family-atomic rule, so the remaining split stays deterministic in (turn ids, families,
   targets) alone.
3. Run `calibration_corpus.split_manifest` on the remainder; fit on FIT, select on VALIDATION.
4. Record in the P5 report: the excluded-set sha256, the counts excluded per rule, and the audit
   line for defense in depth (below).
5. Must hold: no excluded turn in any partition; no excluded row in any M-budget selection
   (budgets are prefixes of the round-robin FIT sequence, `calibration_collector.py` write());
   nothing deleted from the corpus files.

## 4. Defense in depth (why the list should be empty anyway)

* P3 captures declare `span_policy = whole_completed_response_tail_v1` (v1). The calibration
  handoff runs only when `prepared["span_selection"]` exists, and only a v2 span policy builds it
  (`latent_worker.py:367`; plan 2.3 "Consequence of v1"). So a P3 source capture cannot enter the
  corpus even with the collector enabled.
* The P2 swap enables `OCC3_CALIBRATION_CAPTURE_DIR`, so the collector is LIVE during P3 — the
  exclusion list is the mandatory second line of defence, not decoration.
* P5 audit rule: report any corpus turn carrying a v1 span policy (`whole_completed_response_tail_v1`);
  P4 collection uses v2 content spans, so a v1 row is a diagnostic leak by definition.

## 5. Worked example (placeholder window)

With `<WIN>` = `icalib-20261107T041500Z`:

* ranges: N-none `t0001..t0001`, F-text `t0002..t0002`, T-exact `t0003..t0004`,
  T-norm `t0005..t0006`, N-shape `t0007..t0008`;
* `excluded_turn_ids`: `p3-icalib-20261107T041500Z-t0003`, `...-t0005`, `...-t0007`;
* `excluded_family_ids`: `p3-icalib-20261107T041500Z-src`;
* prefix rule catches any future typo'd id in the same namespace.

All ids satisfy `[A-Za-z0-9._:/-]{1,256}` (letters, digits, hyphens only).
