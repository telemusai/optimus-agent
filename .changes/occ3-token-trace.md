Add opt-in bounded OCC-3 token, alignment cosine, embedding-hash, actual DSA pin, position and per-rank KV-slot diagnostics for synthetic handoff drills.

Hash each TP rank's MLA/DSA bytes at injected `prefill_complete`, nonterminal `consume_released` (after copy-done and a local fence, before leaving eager), and `private_release` (before freeing KV). Terminal settle reports only saved injection slots. Reserve the last 16 slots before spacing samples, with one combined CPU transfer per point and exception-isolated, default-off hooks. The live staging module-count pin was already stale on main (37); deploy files remain unchanged.

Validate physical MLA/DSA capacity rather than logical pool size, retaining valid final-page slots and logging out-of-storage skips explicitly.
