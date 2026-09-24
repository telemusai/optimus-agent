# Selective Prime Agent v0.9.6 fixes

Reviewed Prime Agent [v0.9.6 / `e260085dd8f7`](https://github.com/PrimeIntellect-ai/prime-agent/releases/tag/v0.9.6) against Optimus main `c24898bcc9e1` on 24 September 2026. Optimus 0.1.4 adapts these correctness fixes to its Rust application and Python runtime:

| Upstream change | Optimus integration |
| --- | --- |
| [#2471](https://github.com/PrimeIntellect-ai/prime-agent/pull/2471) | Rebind restored workspace functions to live globals in both legacy and CAS-v2 snapshots; include defaults, closures, partials, containers and missing globals. Preserve staged commit and interrupt protection. |
| [#2645](https://github.com/PrimeIntellect-ai/prime-agent/pull/2645), [#2533](https://github.com/PrimeIntellect-ai/prime-agent/pull/2533) | Refresh Anthropic subscription identity and explain that identity in login guidance. Add ten Opus 5.5 provider routes; retain existing adaptive thinking handling. |
| [#2497](https://github.com/PrimeIntellect-ai/prime-agent/pull/2497) | Cache successful credential commands only, preserving Windows process and PowerShell handling. |
| [#2491](https://github.com/PrimeIntellect-ai/prime-agent/pull/2491) | Forward proxy service-tier settings and settle streams on terminal events or truncated EOF. |
| [#2388](https://github.com/PrimeIntellect-ai/prime-agent/pull/2388) | Preserve a cancelled collection receipt after child deletion and allow bound deleted names to be reused. Keep deletion ownership, startup reservations, ambiguity checks and retained-stop protections. |
| [#2519](https://github.com/PrimeIntellect-ai/prime-agent/pull/2519) | Use declared Prime Inference reasoning parameters/efforts and preserve selected teams under runtime or environment API-key overrides. Explicit team environment settings take precedence. |
| [#2432](https://github.com/PrimeIntellect-ai/prime-agent/pull/2432) | Wait for a complete, target-specific replacement snapshot during session switching, with timeout, supersession, cancellation and reconnect handling. |

The model overlay is copied from the reviewed [catalog revision `d3f43ad14d87`](https://github.com/PrimeIntellect-ai/prime-agent-catalog/blob/d3f43ad14d87caa99662d7a62f0d658ab12afa86/models/catalog.v1.json). Existing bundled models and subscription overlays retain precedence. Catalog presence does not establish access for any particular account.

There are no daemon wire changes or external dependency upgrades. Tests use local fixtures and synthetic credentials, without paid provider calls. Windows-specific paths are preserved and covered by conditional fixtures, but this release was validated on Linux.

Optional auxiliary compaction models, quota parking, operator commands, catalog migration and performance experiments remain out of scope. Optimus retains its existing JEV integration and harness helpers; upstream #2670 reverted the intermediate helper removal. GitHub Actions remain disabled.

The broader Python regression run also reproduces an existing failure on untouched main: `test_interrupt_during_bash_notifier_construction_rolls_back_activity` reports `CORO_CREATED` instead of `CORO_CLOSED`. This selective sync does not alter bash notifier construction.
