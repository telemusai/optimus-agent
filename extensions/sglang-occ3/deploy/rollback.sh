#!/usr/bin/env bash
# rollback.sh -- restore the pre-occ3 neo-glm53 SGLang container on the
# DGX host. Reverses deploy.sh:
#
#   1. stop and remove the occ3-bridge container currently named neo-glm53;
#   2. rename neo-glm53-pre-occ3 back to neo-glm53;
#   3. restore its original restart policy (deploy.sh had set it to 'no');
#   4. start it;
#   5. poll GET http://127.0.0.1:30001/health until 200 (timeout 1500s);
#   6. confirm the plugin routes are gone (informational) and report.
#
# The original container and base image are never deleted by deploy.sh, so
# this script only removes OCC-3-added state. The API key is read from
# docker inspect only (in memory); it is never printed, logged, or written
# to a file.
#
# Safety: before removing the deployed container, this script verifies that
# its image matches the OCC3_STATE_NEW_IMAGE recorded by deploy.sh. A
# mismatch means a newer manual deploy already replaced it while the
# preserved container lingered; rollback refuses rather than destroying the
# running container.
#
# Usage (on the DGX host, from extensions/sglang-occ3/deploy):
#   sudo bash rollback.sh
#
# Environment overrides (all optional):
#   BASE_CONTAINER        default neo-glm53
#   PRE_CONTAINER         default neo-glm53-pre-occ3
#   HOST_IP / HOST_PORT   health endpoint (default 127.0.0.1:30001)
#   OCC3_HEALTH_TIMEOUT   seconds to wait for /health (default 1500)
#   OCC3_POLL_INTERVAL    seconds between polls (default 5)
#   OCC3_STOP_TIMEOUT    docker stop timeout in seconds (default 300)
#   OCC3_LOG_FILE         log file (default rollback-occ3.log next to this
#                         script)

set -euo pipefail
export LC_ALL=C

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BASE_CONTAINER="${BASE_CONTAINER:-neo-glm53}"
PRE_CONTAINER="${PRE_CONTAINER:-neo-glm53-pre-occ3}"
HOST_IP="${HOST_IP:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-30001}"
OCC3_HEALTH_TIMEOUT="${OCC3_HEALTH_TIMEOUT:-1500}"
OCC3_POLL_INTERVAL="${OCC3_POLL_INTERVAL:-5}"
OCC3_STOP_TIMEOUT="${OCC3_STOP_TIMEOUT:-300}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${OCC3_LOG_FILE:-$SCRIPT_DIR/rollback-occ3.log}"
STATE_FILE="$SCRIPT_DIR/occ3-deploy-state.env"

HEALTH_URL="http://$HOST_IP:$HOST_PORT/health"
STATUS_URL="http://$HOST_IP:$HOST_PORT/occ3/v1/status"

# In-memory API key (extracted from docker inspect; never printed or saved).
API_KEY=""

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE" >&2
}

die() {
    log "FATAL: $*"
    exit 1
}

container_exists() {
    docker inspect "$1" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]
}

# Extract the API key from a container's launch command (--api-key VALUE or
# --api-key=VALUE). Output is captured by the caller; it is never logged.
extract_api_key() {
    docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' "$1" 2>/dev/null | awk '
        prev == "--api-key" { print; exit }
        /^--api-key=/      { sub(/^--api-key=/, ""); print; exit }
        { prev = $0 }
    '
}

# HTTP status code for GET $1, with the Authorization header built from the
# in-memory key. The key never appears on any process command line: it is
# piped to curl through a stdin config. (/health is exempt from auth in
# SGLang 0.5.21; the header is sent anyway for uniformity.)
auth_get_code() {
    local url="$1" code=""
    if [[ -n "$API_KEY" ]]; then
        code="$(printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" \
            | curl -sS --config - -o /dev/null -w '%{http_code}' --max-time 10 \
                "$url" 2>/dev/null || true)"
    else
        code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
            "$url" 2>/dev/null || true)"
    fi
    if [[ ! "$code" =~ ^[0-9]{3}$ ]]; then
        code="000"
    fi
    printf '%s\n' "$code"
}

wait_for_http_200() {
    local url="$1" watch_container="$2" timeout_s="$3"
    local t0=$SECONDS deadline=$(( SECONDS + timeout_s )) code="" last=""
    log "waiting for $url to return 200 (timeout ${timeout_s}s, poll every ${OCC3_POLL_INTERVAL}s)"
    while (( SECONDS < deadline )); do
        code="$(auth_get_code "$url")"
        if [[ "$code" == "200" ]]; then
            log "$url -> 200 (took $(( SECONDS - t0 ))s)"
            return 0
        fi
        if [[ "$code" != "$last" ]]; then
            log "  $url -> HTTP $code, waiting..."
            last="$code"
        fi
        if ! container_running "$watch_container"; then
            log "  container $watch_container is no longer running; giving up"
            return 1
        fi
        sleep "$OCC3_POLL_INTERVAL"
    done
    log "  timed out after ${timeout_s}s (last HTTP code: ${code:-none})"
    return 1
}

# ---------------------------------------------------------------------------
# Prerequisites and state checks
# ---------------------------------------------------------------------------
log "=== occ3-bridge rollback start (user=$(id -un), host=$(hostname)) ==="

command -v docker >/dev/null 2>&1 || die "docker CLI not found"
command -v curl >/dev/null 2>&1 || die "curl not found"
touch "$LOG_FILE" 2>/dev/null || die "cannot write log file $LOG_FILE"
[[ -w "$LOG_FILE" ]] || die "log file $LOG_FILE is not writable"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (run with sudo or from the docker group)"

if ! container_exists "$PRE_CONTAINER"; then
    if container_exists "$BASE_CONTAINER"; then
        die "$PRE_CONTAINER not found while $BASE_CONTAINER exists; nothing to roll back (deploy.sh may not have run)"
    fi
    die "neither $BASE_CONTAINER nor $PRE_CONTAINER exists; nothing to roll back"
fi

if [[ -f "$STATE_FILE" ]]; then
    # State file written by deploy.sh; contains no secrets.
    # shellcheck disable=SC1090
    . "$STATE_FILE"
    log "deploy state found: image ${OCC3_STATE_NEW_IMAGE:-unknown} (deployed ${OCC3_STATE_DEPLOYED_AT:-unknown})"
fi

# ---------------------------------------------------------------------------
# 1. Remove the occ3-bridge container (only after proving it is ours)
# ---------------------------------------------------------------------------
if container_exists "$BASE_CONTAINER"; then
    # Image guard: deploy.sh recorded the image it deployed in $STATE_FILE.
    # If the running container's image no longer matches, an intervening
    # (manual) deploy replaced it; removing it here would destroy a newer
    # container while restoring the stale preserved one. Refuse instead.
    if [[ -n "${OCC3_STATE_NEW_IMAGE:-}" ]]; then
        running_image="$(docker inspect -f '{{.Config.Image}}' "$BASE_CONTAINER" 2>/dev/null || true)"
        if [[ "$running_image" != "$OCC3_STATE_NEW_IMAGE" ]]; then
            die "$BASE_CONTAINER runs image '${running_image:-<unknown>}' but the deploy state records '$OCC3_STATE_NEW_IMAGE' -- refusing to remove a container this deploy did not create. Inspect both containers; if you really want to restore $PRE_CONTAINER, remove $BASE_CONTAINER manually and rerun this script."
        fi
        log "image guard passed: $BASE_CONTAINER runs the deployed image $OCC3_STATE_NEW_IMAGE"
    else
        log "note: deploy state records no image; cannot verify $BASE_CONTAINER identity before removal"
    fi
    if container_running "$BASE_CONTAINER"; then
        log "stopping occ3 container $BASE_CONTAINER (timeout ${OCC3_STOP_TIMEOUT}s)"
        docker stop -t "$OCC3_STOP_TIMEOUT" "$BASE_CONTAINER" >/dev/null
    else
        log "$BASE_CONTAINER is not running"
    fi
    log "removing occ3 container $BASE_CONTAINER"
    docker rm "$BASE_CONTAINER" >/dev/null
else
    log "$BASE_CONTAINER does not exist (earlier deploy never started it); nothing to remove"
fi

# ---------------------------------------------------------------------------
# 2. Rename the preserved container back and restore its restart policy
# ---------------------------------------------------------------------------
docker rename "$PRE_CONTAINER" "$BASE_CONTAINER"
log "renamed $PRE_CONTAINER -> $BASE_CONTAINER"

if [[ -f "$STATE_FILE" ]] && [[ -n "${OCC3_STATE_RESTART_POLICY:-}" ]]; then
    docker update --restart "${OCC3_STATE_RESTART_POLICY}" "$BASE_CONTAINER" >/dev/null
    log "restored restart policy '${OCC3_STATE_RESTART_POLICY}'"
else
    log "note: no deploy state file; leaving the restart policy as-is"
fi

# ---------------------------------------------------------------------------
# 3. Start the restored container
# ---------------------------------------------------------------------------
docker start "$BASE_CONTAINER" >/dev/null
log "started $BASE_CONTAINER"

# ---------------------------------------------------------------------------
# 4. Wait for the server to come back (B300 weight load takes minutes)
# ---------------------------------------------------------------------------
API_KEY="$(extract_api_key "$BASE_CONTAINER")"
if [[ -z "$API_KEY" ]]; then
    log "no --api-key flag found; polling without an Authorization header"
fi

if ! wait_for_http_200 "$HEALTH_URL" "$BASE_CONTAINER" "$OCC3_HEALTH_TIMEOUT"; then
    log "FAILURE: the restored server did not become healthy"
    log "  inspect with: docker logs --tail 200 $BASE_CONTAINER"
    exit 1
fi
log "health check passed: $HEALTH_URL -> 200"

# ---------------------------------------------------------------------------
# 5. Confirm the plugin is gone (informational)
# ---------------------------------------------------------------------------
code="$(auth_get_code "$STATUS_URL")"
log "plugin endpoint after rollback: $STATUS_URL -> HTTP $code (non-200 expected; 404 confirms the routes are gone)"

if [[ -f "$STATE_FILE" ]]; then
    log "occ3 image from the deploy: ${OCC3_STATE_NEW_IMAGE:-unknown}"
    log "remove it manually if no longer needed: docker rmi ${OCC3_STATE_NEW_IMAGE:-<tag>}"
    rm -f -- "$STATE_FILE"
    log "removed deploy state file $STATE_FILE"
fi

log "=== SUCCESS: $BASE_CONTAINER restored to its pre-occ3 state ==="
log "  log file: $LOG_FILE"
