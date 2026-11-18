#!/usr/bin/env bash
# OCC-3 deployment watchdog (DGX-side, model-independent).
#
# Runs the controller under a hard timeout. If the controller exits
# non-zero (including SIGTERM/SIGKILL of the controller itself, or a
# hang killed by timeout), the watchdog runs the rollback subcommand,
# which restores and VERIFIES the previous service. The watchdog never
# retries the deployment (no automatic repeated redeployment).
#
# Usage: setsid nohup bash occ3_watchdog.sh <key-file> <new-key-file> <recovery-dir> &
set -u

KEY_FILE="$1"        # current api key (also the rollback key)
NEW_KEY_FILE="$2"    # rotated key the candidate will serve (required)
RECOVERY_DIR="$3"
CONTROLLER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/occ3_controller.py"
DEADLINE_S="${OCC3_DEADLINE_S:-3000}"

LOG="$RECOVERY_DIR/watchdog.log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

{
  echo "$(stamp) watchdog start (pid $$, deadline ${DEADLINE_S}s)"
  timeout --signal=TERM --kill-after=60 "$DEADLINE_S" \
      python3 "$CONTROLLER" deploy --key-file "$KEY_FILE" \
      --new-key-file "$NEW_KEY_FILE" --recovery-dir "$RECOVERY_DIR"
  rc=$?
  echo "$(stamp) controller exited rc=$rc"
  if [ "$rc" -ne 0 ]; then
      echo "$(stamp) watchdog: automatic rollback"
      # The controller's rollback verification uses the same health budget
      # as the deploy gate (OCC3_HEALTH_TIMEOUT, default 1500); the watchdog
      # hard-kill must cover that plus the stop/rename/start overhead, or a
      # slow-restoring original is killed mid-verification and reported as
      # a failed rollback.
      OCC3_HEALTH_TIMEOUT="${OCC3_HEALTH_TIMEOUT:-1500}"
      ROLLBACK_KILL_S=$((OCC3_HEALTH_TIMEOUT + 600))
      timeout --signal=TERM --kill-after=60 "$ROLLBACK_KILL_S" \
          python3 "$CONTROLLER" rollback --key-file "$KEY_FILE" \
          --recovery-dir "$RECOVERY_DIR"
      # rollback serves the OLD key again; flip the local env back if the
      # watcher had already rotated it (it only rotates on SUCCESS/gate8,
      # so normally this is a no-op safeguard)
      rb_rc=$?
      echo "$(stamp) watchdog: rollback rc=$rb_rc"
      if [ "$rb_rc" -ne 0 ]; then
          echo "$(stamp) WATCHDOG ALERT: rollback did not verify; manual action required:"
          echo "  docker rename $([ -n "$(docker ps -aq --filter name=^/neo-glm53-pre-occ3$ 2>/dev/null)" ] && echo neo-glm53-pre-occ3 || echo '<none>') neo-glm53 && docker start neo-glm53"
      fi
  fi
  echo "$(stamp) watchdog done"
} >> "$LOG" 2>&1
