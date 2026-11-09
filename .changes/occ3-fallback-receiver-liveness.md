---
"Mark OCC-1 fallback re-emissions delivered into hydrated (non-resident) receivers honestly."
---

## What

- AgentSessionMessageReceipt carries an optional deliveredToHydrated marker, set only
  by the daemon send path that had to hydrate (restore from saved session artifacts) a
  non-resident receiver to deliver.
- The OCC-3 fallback acknowledgment closes on the new delivered_to_hydrated status but
  records it in the notice reason (ordinary_delivery:delivered_to_hydrated) instead of
  a clean `delivered`.
- In-flight spawn hydration (wait_for_hydrating_child) is NOT marked: a live child
  will consume the message.
