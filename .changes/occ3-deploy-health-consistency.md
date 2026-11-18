---
"Fix OCC-3 deploy/rollback health-timeout consistency and rollback image guard."
---

## What

- occ3_controller.py and v3 lifecycle.py now apply the same deployment-health
  default (1500 s) as deploy.sh / rollback.sh, and the controller honors the
  shared OCC3_HEALTH_TIMEOUT override (absent/empty/garbage falls back safely).
- rollback.sh refuses to remove the deployed container when its running image
  does not match the OCC3_STATE_NEW_IMAGE recorded by deploy.sh, instead of
  destroying a newer manual deploy.
- The deploy runtime state file occ3-deploy-state.env is gitignored.
- occ3-queue.md documents the actual 90 s capture-capacity wait default
  (OCC3_CAPTURE_CAPACITY_WAIT_SECONDS, clamp 5-240 s).
