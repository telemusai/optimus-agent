# OCC3.1 hierarchical advisory overlap

- Changed: Preserve the original serial hierarchical role path by default and in sequential mode, including independent hierarchical contexts and ordered Planner/Critic/Refiner results.
- Added: Add explicit `PRIME_AGENT_OCC3_TEXTMAS_HIERARCHICAL_CONCURRENCY=2|3` opt-in for bounded, work-conserving hierarchical overlap. An unset or invalid setting uses the safe serial default; full parent gating and existing cancellation/accounting remain unchanged.
- Fixed: Prevent unconditional concurrent role admission from breaking capacity-one providers. There is no generic retry or serial fallback after an admitted request fails or is refused.
