---
"When a chat has a live native OCC-3 enrollment, occ3_handoff is the preferred communication channel."
---

## What

- The RLM system prompt states the OCC-3 preference only when the occ3_handoff
  tool is active for the session (live enrollment): native handoffs first for
  compatible-family fact exchange, ordinary OCC-1/OCC-2 messaging for
  non-compatible receivers, conversational replies, and busy/refused/exhausted
  handoffs.
- /occ status reports "OCC-3: PREFERRED (live native enrollment...)" when the
  tool is active, keeping ON/OFF wording unchanged otherwise.
