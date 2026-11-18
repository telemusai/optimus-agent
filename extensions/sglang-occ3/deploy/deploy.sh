#!/usr/bin/env bash
# deploy.sh -- additive deployment of the occ3-bridge SGLang plugin onto the
# running neo-glm53 SGLang server (NVIDIA DGX B300 container host).
#
# The plugin package (occ3-bridge) registers an SGLang plugin entry point
# (group sglang.srt.plugins, name occ3_bridge). SGLang discovers plugins
# only at process start; there is no hot-load path, so the server process
# is restarted exactly once. Steps:
#
#   1. verify prerequisites (docker, curl, base container present/running);
#   2. capture the running container's launch configuration via
#      docker inspect (image digest, entrypoint, cmd, env, ports, volumes,
#      network, restart policy, gpus, shm-size, ulimits, plus ipc, pid,
#      runtime, user, workdir, init, tmpfs, extra hosts, memory/cpu limits);
#   3. write the server API key to the root-only file /tmp/.occ3_deploy_key
#      and use it from there; the key is never printed or logged, and the
#      file is scrubbed on exit;
#   4. build the derived image (base digest + plugin only), tagged
#      neo-glm53:occ3-<UTC date>;
#   5. preflight the plugin entry point inside the derived image
#      (no GPU, no ports) and abort BEFORE any mutation if it cannot load;
#   6. mutate: rename neo-glm53 -> neo-glm53-pre-occ3 (preserved, stopped),
#      then docker run -d a NEW neo-glm53 with the captured configuration
#      on the derived image;
#   7. poll GET http://127.0.0.1:30001/health until 200 (timeout 1500s;
#      B300 weight loading takes minutes);
#   8. verify the plugin: GET /occ3/v1/status, expecting 200 JSON with the
#      occ3 version;
#   9. report success, or failure with rollback instructions.
#
# Usage (on the DGX host, from extensions/sglang-occ3/deploy):
#   sudo bash deploy.sh
#   sudo env OCC3_DRY_RUN=1 bash deploy.sh        # capture+build+preflight only
#   sudo env OCC3_SET_PLUGIN_WHITELIST=1 bash deploy.sh
#
# Environment overrides (all optional):
#   BASE_CONTAINER               default neo-glm53
#   PRE_CONTAINER                default neo-glm53-pre-occ3
#   HOST_IP / HOST_PORT          health endpoint (default 127.0.0.1:30001,
#                                otherwise derived from the captured port map)
#   OCC3_HEALTH_TIMEOUT          seconds to wait for /health (default 1500)
#   OCC3_POLL_INTERVAL           seconds between polls (default 5)
#   OCC3_STOP_TIMEOUT            docker stop timeout in seconds (default 300)
#   OCC3_GPU_SETTLE              seconds to wait after stopping the old
#                                container before starting the new one (default 15)
#   OCC3_SET_PLUGIN_WHITELIST    1 = set SGLANG_PLUGINS=occ3_bridge on the new
#                                container (see README.md)
#   OCC3_PREFLIGHT_IMPORT        0 = skip the plugin import in the preflight
#                                (default 1)
#   OCC3_DRY_RUN                 1 = no mutation of the running service
#   OCC3_ALLOW_STOPPED_BASE       1 = allow deploying when the base container
#                                exists but is stopped
#   OCC3_LOG_FILE                log file (default deploy-occ3.log next to this
#                                script)
#
# Security notes:
#   - The API key is never printed, logged, or written anywhere except the
#     root-only /tmp/.occ3_deploy_key (0600), which is removed on exit.
#   - Captured env values are never logged; variable NAMES only. The
#     captured command line is logged in redacted form.
#   - If docker inspect fails, the script aborts before any mutation; the
#     rename and docker run happen only after the image build and the
#     plugin preflight have succeeded.

set -euo pipefail
export LC_ALL=C

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BASE_CONTAINER="${BASE_CONTAINER:-neo-glm53}"
PRE_CONTAINER="${PRE_CONTAINER:-neo-glm53-pre-occ3}"
KEY_FILE="/tmp/.occ3_deploy_key"
OCC3_HEALTH_TIMEOUT="${OCC3_HEALTH_TIMEOUT:-1500}"
OCC3_POLL_INTERVAL="${OCC3_POLL_INTERVAL:-5}"
OCC3_STOP_TIMEOUT="${OCC3_STOP_TIMEOUT:-300}"
OCC3_GPU_SETTLE="${OCC3_GPU_SETTLE:-15}"
OCC3_SET_PLUGIN_WHITELIST="${OCC3_SET_PLUGIN_WHITELIST:-0}"
OCC3_PREFLIGHT_IMPORT="${OCC3_PREFLIGHT_IMPORT:-1}"
OCC3_DRY_RUN="${OCC3_DRY_RUN:-0}"
OCC3_ALLOW_STOPPED_BASE="${OCC3_ALLOW_STOPPED_BASE:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
LOG_FILE="${OCC3_LOG_FILE:-$SCRIPT_DIR/deploy-occ3.log}"
STATE_FILE="$SCRIPT_DIR/occ3-deploy-state.env"

HOST_IP="${HOST_IP:-}"
HOST_PORT="${HOST_PORT:-}"

STAGE=""
NEW_IMAGE=""

# ---------------------------------------------------------------------------
# Logging: every step goes to the console and to $LOG_FILE with a UTC
# timestamp.
# ---------------------------------------------------------------------------
log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE" >&2
}

die() {
    log "FATAL: $*"
    exit 1
}

# ---------------------------------------------------------------------------
# Cleanup: scrub the API key file and the staging area (the captured env and
# cmd files contain the key). Runs on every exit path.
# ---------------------------------------------------------------------------
cleanup() {
    local rc=$?
    if [[ -n "${STAGE:-}" && -d "$STAGE" ]]; then
        rm -rf -- "$STAGE" || true
        log "scrubbed staging directory $STAGE (held captured env/cmd)"
    fi
    if [[ -e "$KEY_FILE" ]]; then
        rm -f -- "$KEY_FILE" || true
        log "scrubbed API key file $KEY_FILE"
    fi
    exit "$rc"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
container_exists() {
    docker inspect "$1" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]
}

inspect_f() {
    # docker inspect with a Go template against the capture container.
    docker inspect -f "$1" "$CAPTURE_CONTAINER"
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
# root-only key file. The key never appears on any process command line: it
# is piped to curl through a stdin config.
auth_get_code() {
    local url="$1" code=""
    if [[ -s "$KEY_FILE" ]]; then
        code="$(printf 'header = "Authorization: Bearer %s"\n' "$(cat "$KEY_FILE")" \
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

# Body of GET $1 with the same authentication; used for the status check.
auth_get_body() {
    local url="$1"
    if [[ -s "$KEY_FILE" ]]; then
        printf 'header = "Authorization: Bearer %s"\n' "$(cat "$KEY_FILE")" \
            | curl -sS --config - --max-time 15 "$url" 2>/dev/null || true
    else
        curl -sS --max-time 15 "$url" 2>/dev/null || true
    fi
}

# Print the given arguments on one line with anything that looks like an API
# key redacted. Used for LOGGING ONLY (never for the actual docker run).
redact_args() {
    local arg redact_next=0 out=""
    for arg in "$@"; do
        if (( redact_next )); then
            out+="***REDACTED*** "
            redact_next=0
        else
            case "$arg" in
                --api-key|--admin-api-key)
                    out+="$arg "
                    redact_next=1
                    ;;
                --api-key=*|--admin-api-key=*)
                    out+="${arg%%=*}=***REDACTED*** "
                    ;;
                *)
                    out+="$arg "
                    ;;
            esac
        fi
    done
    printf '%s' "${out% }"
}

# Poll GET $1 until HTTP 200 or timeout; give up early if $2 stops running.
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
# Step 1: prerequisites
# ---------------------------------------------------------------------------
log "=== occ3-bridge deployment start (user=$(id -un), host=$(hostname)) ==="

command -v docker >/dev/null 2>&1 || die "docker CLI not found"
command -v curl >/dev/null 2>&1 || die "curl not found"
[[ -r "$DOCKERFILE" ]] || die "Dockerfile not found at $DOCKERFILE"
[[ -f "$EXT_ROOT/pyproject.toml" ]] || die "pyproject.toml not found at $EXT_ROOT"
[[ -d "$EXT_ROOT/occ3_sglang" ]] || die "package source not found at $EXT_ROOT/occ3_sglang"
touch "$LOG_FILE" 2>/dev/null || die "cannot write log file $LOG_FILE"
[[ -w "$LOG_FILE" ]] || die "log file $LOG_FILE is not writable"

docker info >/dev/null 2>&1 || die "docker daemon not reachable (run with sudo or from the docker group)"

if container_exists "$BASE_CONTAINER"; then
    if ! container_running "$BASE_CONTAINER"; then
        if [[ "$OCC3_ALLOW_STOPPED_BASE" != "1" ]]; then
            die "$BASE_CONTAINER is not running; expected a live server (override with OCC3_ALLOW_STOPPED_BASE=1)"
        fi
        log "WARNING: $BASE_CONTAINER is stopped; continuing (OCC3_ALLOW_STOPPED_BASE=1)"
    fi
    CAPTURE_CONTAINER="$BASE_CONTAINER"
elif container_exists "$PRE_CONTAINER"; then
    log "note: $BASE_CONTAINER missing, $PRE_CONTAINER present -- resuming an earlier deploy attempt"
    CAPTURE_CONTAINER="$PRE_CONTAINER"
else
    die "neither $BASE_CONTAINER nor $PRE_CONTAINER exists; nothing to deploy onto"
fi

# Remember the original restart policy from a previous attempt (this deploy
# sets the preserved container to --restart=no; see the mutation phase).
if [[ -f "$STATE_FILE" ]]; then
    # State file written by this script; contains no secrets.
    # shellcheck disable=SC1090
    . "$STATE_FILE"
    log "found previous deploy state: image ${OCC3_STATE_NEW_IMAGE:-unknown}, restart policy ${OCC3_STATE_RESTART_POLICY:-unknown}"
fi

# Write the API key to the root-only file, used by every authenticated call.
API_KEY="$(extract_api_key "$CAPTURE_CONTAINER")"
if [[ -n "$API_KEY" ]]; then
    ( umask 077; printf '%s\n' "$API_KEY" > "$KEY_FILE" )
    chmod 600 "$KEY_FILE"
    log "API key found in the launch flags; stored in $KEY_FILE (0600); never printed or logged"
else
    log "no --api-key flag found; proceeding without an Authorization header"
fi
unset API_KEY

# ---------------------------------------------------------------------------
# Step 2: capture the current launch configuration (docker inspect). Aborts
# before any mutation if inspection fails.
# ---------------------------------------------------------------------------
STAGE="$(mktemp -d /tmp/occ3-deploy.XXXXXX)"
chmod 700 "$STAGE"
log "staging captured configuration in $STAGE (0700, removed on exit)"

docker inspect "$CAPTURE_CONTAINER" > "$STAGE/inspect.json" \
    || die "docker inspect failed for $CAPTURE_CONTAINER -- aborting before any mutation"

inspect_f '{{range .Config.Cmd}}{{println .}}{{end}}'        > "$STAGE/cmd.txt"
inspect_f '{{range .Config.Entrypoint}}{{println .}}{{end}}' > "$STAGE/entrypoint.txt"
inspect_f '{{range .Config.Env}}{{println .}}{{end}}'        > "$STAGE/env.txt"

mapfile -t CMD_ARGS        < "$STAGE/cmd.txt"
mapfile -t ENTRYPOINT_ARGS < "$STAGE/entrypoint.txt"

if (( ${#CMD_ARGS[@]} == 0 && ${#ENTRYPOINT_ARGS[@]} == 0 )); then
    die "captured container has neither entrypoint nor cmd; refusing to replicate"
fi

BASE_IMAGE_REF="$(inspect_f '{{.Config.Image}}')"
BASE_IMAGE_ID="$(inspect_f '{{.Image}}')"
NETWORK_MODE="$(inspect_f '{{.HostConfig.NetworkMode}}')"
RESTART_NAME="$(inspect_f '{{.HostConfig.RestartPolicy.Name}}')"
RESTART_MAXRETRY="$(inspect_f '{{.HostConfig.RestartPolicy.MaximumRetryCount}}')"
SHM_SIZE="$(inspect_f '{{.HostConfig.ShmSize}}')"
IPC_MODE="$(inspect_f '{{.HostConfig.IpcMode}}')"
PID_MODE="$(inspect_f '{{.HostConfig.PidMode}}')"
RUNTIME="$(inspect_f '{{.HostConfig.Runtime}}')"
CONT_USER="$(inspect_f '{{.Config.User}}')"
CONT_WORKDIR="$(inspect_f '{{.Config.WorkingDir}}')"
CONT_INIT="$(inspect_f '{{.HostConfig.Init}}')"
MEMORY="$(inspect_f '{{.HostConfig.Memory}}')"
NANOCPUS="$(inspect_f '{{.HostConfig.NanoCpus}}')"

PORT_LINES="$(inspect_f '{{range $p, $b := .HostConfig.PortBindings}}{{range $b}}{{printf "%s|%s|%s\n" $p .HostIp .HostPort}}{{end}}{{end}}')"
mapfile -t BINDS       < <(inspect_f '{{range .HostConfig.Binds}}{{println .}}{{end}}')
mapfile -t ULIMITS     < <(inspect_f '{{range .HostConfig.Ulimits}}{{printf "%s|%d|%d\n" .Name .Soft .Hard}}{{end}}')
mapfile -t TMPFS       < <(inspect_f '{{range $k, $v := .HostConfig.Tmpfs}}{{printf "%s|%s\n" $k $v}}{{end}}')
mapfile -t EXTRA_HOSTS < <(inspect_f '{{range .HostConfig.ExtraHosts}}{{println .}}{{end}}')
mapfile -t NETWORKS    < <(inspect_f '{{range $k, $v := .NetworkSettings.Networks}}{{printf "%s\n" $k}}{{end}}')

# Sanitize: docker 29.x emits a trailing newline even for empty template
# ranges, so every mapfile above can hold a single EMPTY element. That
# would defeat the ${#ARR[@]} == 0 checks below (mount fallback) and
# append empty-valued flags (-v "", --tmpfs "", ...) to the launch line.
drop_blank() {
    # usage: drop_blank VAR; filters empty/whitespace-only elements.
    local -n __arr="$1"
    local __kept=()
    local __e
    for __e in ${__arr[@]+"${__arr[@]}"}; do
        [[ -n "${__e//[[:space:]]/}" ]] && __kept+=( "$__e" )
    done
    __arr=( "${__kept[@]}" )
}
drop_blank BINDS
drop_blank ULIMITS
drop_blank TMPFS
drop_blank EXTRA_HOSTS
drop_blank NETWORKS

# Restart policy: rebuild the docker run argument; a previous attempt's
# state file keeps the ORIGINAL policy (this deploy sets the preserved
# container to --restart=no, see the mutation phase).
RESTART_ARG=""
if [[ -n "${OCC3_STATE_RESTART_POLICY:-}" ]]; then
    RESTART_ARG="$OCC3_STATE_RESTART_POLICY"
elif [[ -n "$RESTART_NAME" && "$RESTART_NAME" != "no" ]]; then
    RESTART_ARG="$RESTART_NAME"
    if [[ "$RESTART_NAME" == "on-failure" && "$RESTART_MAXRETRY" -gt 0 ]]; then
        RESTART_ARG="on-failure:$RESTART_MAXRETRY"
    fi
fi

# GPU access: reconstruct --gpus from the device requests.
GPUS_ARGS=()
GPUS_SPEC=""
GPUS_JSON="$(inspect_f '{{json .HostConfig.DeviceRequests}}')"
if [[ "$GPUS_JSON" != "null" && "$GPUS_JSON" != "[]" && -n "$GPUS_JSON" ]]; then
    log "captured device requests: $GPUS_JSON"
    while IFS='|' read -r driver devids count; do
        [[ -n "${driver}${devids}${count}" ]] || continue
        spec=""
        if [[ -n "$devids" ]]; then
            spec="device=$devids"
        elif [[ "$count" -gt 0 ]]; then
            spec="$count"
        else
            spec="all"
        fi
        if [[ -n "$driver" && "$driver" != "nvidia" ]]; then
            spec="driver=$driver,$spec"
        fi
        GPUS_ARGS+=( --gpus "$spec" )
        GPUS_SPEC+="${GPUS_SPEC:+ }$spec"
    done < <(inspect_f '{{range .HostConfig.DeviceRequests}}{{printf "%s|%s|%d\n" .Driver (join .DeviceIDs ",") .Count}}{{end}}')
fi

# Ports: rebuild -p arguments from the port bindings.
PORT_ARGS=()
if [[ -n "$PORT_LINES" ]]; then
    while IFS='|' read -r cport hostip hostport; do
        [[ -n "$cport" ]] || continue
        proto="${cport##*/}"
        cnum="${cport%%/*}"
        suffix=""
        if [[ "$proto" != "tcp" && "$proto" != "$cport" ]]; then
            suffix="/$proto"
        fi
        if [[ -n "$hostip" ]]; then
            PORT_ARGS+=( -p "${hostip}:${hostport}:${cnum}${suffix}" )
        else
            PORT_ARGS+=( -p "${hostport}:${cnum}${suffix}" )
        fi
    done <<< "$PORT_LINES"
fi

# Volumes: prefer the raw bind strings (they keep propagation modes); fall
# back to --mount reconstruction from .Mounts when no binds were captured.
MOUNT_ARGS=()
if (( ${#BINDS[@]} == 0 )); then
    while IFS='|' read -r mtype mname msrc mdest mrw; do
        [[ -n "$mtype" ]] || continue
        ro=""
        if [[ "$mrw" == "false" ]]; then
            ro=",readonly"
        fi
        case "$mtype" in
            bind)   MOUNT_ARGS+=( --mount "type=bind,source=$msrc,target=$mdest$ro" ) ;;
            volume) MOUNT_ARGS+=( --mount "type=volume,source=$mname,target=$mdest$ro" ) ;;
            *)      : ;;  # tmpfs and others are covered by HostConfig.Tmpfs
        esac
    done < <(inspect_f '{{range .Mounts}}{{printf "%s|%s|%s|%s|%t\n" .Type .Name .Source .Destination .RW}}{{end}}')
fi

# Network mode.
if [[ "$NETWORK_MODE" == "container:"* ]]; then
    die "unsupported network mode '$NETWORK_MODE' (container:...); refusing to replicate"
fi
NETWORK_ARG=()
if [[ -n "$NETWORK_MODE" && "$NETWORK_MODE" != "default" && "$NETWORK_MODE" != "bridge" ]]; then
    NETWORK_ARG=( --network "$NETWORK_MODE" )
fi

# Host endpoint for the health/status checks: explicit override, else the
# captured port map (preferring the SGLang port 30001), else the default.
if [[ -z "$HOST_PORT" && -n "$PORT_LINES" ]]; then
    bind_first=""
    bind_want=""
    while IFS='|' read -r cport hostip hostport; do
        [[ -n "$cport" ]] || continue
        if [[ -z "$bind_first" ]]; then
            bind_first="${hostip:-127.0.0.1}|$hostport"
        fi
        if [[ "$cport" == "30001"* ]]; then
            bind_want="${hostip:-127.0.0.1}|$hostport"
            break
        fi
    done <<< "$PORT_LINES"
    chosen="${bind_want:-$bind_first}"
    HOST_IP="${chosen%%|*}"
    HOST_PORT="${chosen##*|}"
fi
HOST_IP="${HOST_IP:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-30001}"
HEALTH_URL="http://$HOST_IP:$HOST_PORT/health"
STATUS_URL="http://$HOST_IP:$HOST_PORT/occ3/v1/status"
log "health endpoint: $HEALTH_URL"
log "plugin status endpoint: $STATUS_URL"

# Idempotency: if the current server already serves the plugin, do nothing.
if container_exists "$BASE_CONTAINER"; then
    early_code="$(auth_get_code "$STATUS_URL")"
    if [[ "$early_code" == "200" ]]; then
        log "$BASE_CONTAINER already serves $STATUS_URL (HTTP 200): occ3-bridge is live, nothing to do"
        exit 0
    fi
    if container_exists "$PRE_CONTAINER"; then
        die "both $BASE_CONTAINER and $PRE_CONTAINER exist and the plugin is not live; run ./rollback.sh first (it restores $BASE_CONTAINER from $PRE_CONTAINER)"
    fi
fi

# Optional plugin whitelist (see README.md). Applied after the idempotency
# check so a no-op rerun does not modify anything.
if [[ "$OCC3_SET_PLUGIN_WHITELIST" == "1" ]]; then
    sed -i '/^SGLANG_PLUGINS=/d' "$STAGE/env.txt"
    printf 'SGLANG_PLUGINS=occ3_bridge\n' >> "$STAGE/env.txt"
    log "SGLANG_PLUGINS=occ3_bridge whitelist enabled on the new container"
fi

# Captured configuration summary (secrets redacted; env names only).
ENV_COUNT="$(wc -l < "$STAGE/env.txt" | tr -d '[:space:]')"
ENV_NAMES="$(awk -F= '{print $1}' "$STAGE/env.txt" | sort | paste -sd, -)"
log "captured configuration of $CAPTURE_CONTAINER:"
log "  image:      $BASE_IMAGE_REF ($BASE_IMAGE_ID)"
log "  entrypoint: $(redact_args ${ENTRYPOINT_ARGS[@]+"${ENTRYPOINT_ARGS[@]}"})"
log "  cmd:        $(redact_args ${CMD_ARGS[@]+"${CMD_ARGS[@]}"})"
log "  env:        $ENV_COUNT variables (names only): $ENV_NAMES"
log "  ports:      $(printf '%s' "$PORT_LINES" | tr '\n' ' ')"
log "  volumes:    ${BINDS[*]:-none}"
if (( ${#MOUNT_ARGS[@]} > 0 )); then
    log "  mounts:     ${MOUNT_ARGS[*]}"
fi
log "  network:    ${NETWORK_MODE:-default} (attached: ${NETWORKS[*]:-unknown})"
log "  restart:    ${RESTART_ARG:-none}"
log "  gpus:       ${GPUS_SPEC:-none}"
log "  shm-size:   $SHM_SIZE bytes"
log "  ulimits:    ${ULIMITS[*]:-none}"
log "  ipc: ${IPC_MODE:-(default)}  pid: ${PID_MODE:-(default)}  runtime: ${RUNTIME:-(default)}"
log "  user: ${CONT_USER:-(default)}  workdir: ${CONT_WORKDIR:-(default)}  init: $CONT_INIT  memory: ${MEMORY:-0}  nano-cpus: ${NANOCPUS:-0}"

# Baseline: how is the current server doing before anything changes?
baseline="$(auth_get_code "$HEALTH_URL")"
log "baseline health check: $HEALTH_URL -> HTTP $baseline"
if [[ "$baseline" != "200" ]]; then
    log "WARNING: the current server did not answer 200 before the deploy; continuing"
fi

# ---------------------------------------------------------------------------
# Step 3: build the derived image (additive: base digest + plugin only)
# ---------------------------------------------------------------------------
BASE_REF="$(inspect_f '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null | head -1 || true)"
if [[ -z "$BASE_REF" ]]; then
    if [[ "$BASE_IMAGE_REF" == *"@sha256:"* ]]; then
        BASE_REF="$BASE_IMAGE_REF"
    elif docker image inspect "$BASE_IMAGE_ID" >/dev/null 2>&1; then
        # Local-only (untagged, never-pulled) image: pin the exact image
        # ID with a local tag. Same content, no registry involved.
        if ! docker tag "$BASE_IMAGE_ID" occ3-base:occ3 2>>"$LOG_FILE"; then
            die "cannot tag the base image id $BASE_IMAGE_ID"
        fi
        BASE_REF="occ3-base:occ3"
    else
        BASE_REF="$BASE_IMAGE_REF"
    fi
    log "no repo digest found; using fallback base reference $BASE_REF"
fi
log "base image reference (digest-pinned, read via docker inspect): $BASE_REF"

NEW_TAG="${BASE_CONTAINER}:occ3-$(date -u +%Y%m%d-%H%M%S)"
log "building derived image $NEW_TAG (context $EXT_ROOT, Dockerfile $DOCKERFILE)"
if ! docker build --build-arg "BASE_IMAGE=$BASE_REF" -t "$NEW_TAG" \
        -f "$DOCKERFILE" "$EXT_ROOT" </dev/null 2>&1 | tee -a "$LOG_FILE"; then
    die "docker build failed -- no mutation has been performed"
fi
NEW_IMAGE="$NEW_TAG"
log "derived image built: $NEW_IMAGE"

# ---------------------------------------------------------------------------
# Step 4: preflight the plugin inside the derived image (no GPU, no ports).
# Aborts BEFORE any mutation if the plugin cannot be discovered or loaded.
# ---------------------------------------------------------------------------
PREFLIGHT_SCRIPT='
import importlib.metadata as md
import importlib.util
import pathlib
import sys

eps = [ep for ep in md.entry_points(group="sglang.srt.plugins") if ep.name == "occ3_bridge"]
if not eps:
    sys.exit("FAIL: no entry point occ3_bridge in group sglang.srt.plugins (was occ3-bridge installed?)")
ep = eps[0]
module, _, attr = ep.value.partition(":")
spec = importlib.util.find_spec(module)
if spec is None or not spec.origin:
    sys.exit("FAIL: entry point module " + module + " is not present in the image")
compile(pathlib.Path(spec.origin).read_text(), spec.origin, "exec")
if not attr:
    sys.exit("FAIL: entry point has no attribute part")
print("OK: occ3_bridge -> " + ep.value + " (syntax ok)")
'
if [[ "$OCC3_PREFLIGHT_IMPORT" == "1" ]]; then
    PREFLIGHT_SCRIPT="$PREFLIGHT_SCRIPT"'
try:
    func = ep.load()
except Exception as exc:
    sys.exit("FAIL: importing the entry point raised: %r" % (exc,))
if not callable(func):
    sys.exit("FAIL: entry point target is not callable")
print("OK: entry point imports cleanly")
'
fi
log "preflight: verifying the occ3_bridge entry point inside $NEW_IMAGE"
if ! timeout 300 docker run --rm --network none --entrypoint python3 \
        "$NEW_IMAGE" -c "$PREFLIGHT_SCRIPT" </dev/null 2>&1 | tee -a "$LOG_FILE"; then
    die "plugin preflight failed in $NEW_IMAGE -- aborting before any mutation (is occ3_sglang/plugin.py in the build context?)"
fi
log "plugin preflight passed"

# ---------------------------------------------------------------------------
# Assemble the docker run arguments: the new container replicates the
# captured configuration; only the image differs.
# ---------------------------------------------------------------------------
RUN_ARGS=( -d --name "$BASE_CONTAINER" )
RUN_ARGS+=( --env-file "$STAGE/env.txt" )

if (( ${#ENTRYPOINT_ARGS[@]} > 0 )); then
    RUN_ARGS+=( --entrypoint "${ENTRYPOINT_ARGS[0]}" )
fi
if (( ${#NETWORK_ARG[@]} > 0 )); then
    RUN_ARGS+=( "${NETWORK_ARG[@]}" )
fi
if [[ -n "$RESTART_ARG" ]]; then
    RUN_ARGS+=( --restart "$RESTART_ARG" )
fi
if (( ${#GPUS_ARGS[@]} > 0 )); then
    RUN_ARGS+=( "${GPUS_ARGS[@]}" )
fi
if [[ -n "$RUNTIME" && "$RUNTIME" != "runc" ]]; then
    RUN_ARGS+=( --runtime "$RUNTIME" )
fi
if [[ -n "$IPC_MODE" ]]; then
    RUN_ARGS+=( --ipc "$IPC_MODE" )
fi
if [[ -n "$PID_MODE" ]]; then
    RUN_ARGS+=( --pid "$PID_MODE" )
fi
if [[ -n "$CONT_USER" ]]; then
    RUN_ARGS+=( --user "$CONT_USER" )
fi
if [[ -n "$CONT_WORKDIR" ]]; then
    RUN_ARGS+=( --workdir "$CONT_WORKDIR" )
fi
if [[ "$CONT_INIT" == "true" ]]; then
    RUN_ARGS+=( --init )
fi
if [[ "$SHM_SIZE" -gt 0 ]]; then
    RUN_ARGS+=( --shm-size "$SHM_SIZE" )
fi
if (( ${#ULIMITS[@]} > 0 )); then
    for uline in ${ULIMITS[@]+"${ULIMITS[@]}"}; do
        IFS='|' read -r uname usoft uhard <<< "$uline"
        [[ -n "$uname" ]] || continue
        RUN_ARGS+=( --ulimit "${uname}=${usoft}:${uhard}" )
    done
fi
if (( ${#BINDS[@]} > 0 )); then
    for bind in ${BINDS[@]+"${BINDS[@]}"}; do
        RUN_ARGS+=( -v "$bind" )
    done
fi
if (( ${#MOUNT_ARGS[@]} > 0 )); then
    RUN_ARGS+=( "${MOUNT_ARGS[@]}" )
fi
if (( ${#TMPFS[@]} > 0 )); then
    for tline in ${TMPFS[@]+"${TMPFS[@]}"}; do
        tdest="${tline%%|*}"
        topts="${tline#*|}"
        if [[ -n "$topts" && "$topts" != "$tline" ]]; then
            RUN_ARGS+=( --tmpfs "${tdest}:${topts}" )
        else
            RUN_ARGS+=( --tmpfs "$tdest" )
        fi
    done
fi
if (( ${#EXTRA_HOSTS[@]} > 0 )); then
    for hline in ${EXTRA_HOSTS[@]+"${EXTRA_HOSTS[@]}"}; do
        RUN_ARGS+=( --add-host "$hline" )
    done
fi
if (( ${#PORT_ARGS[@]} > 0 )); then
    RUN_ARGS+=( "${PORT_ARGS[@]}" )
fi
if [[ "$MEMORY" -gt 0 ]]; then
    RUN_ARGS+=( --memory "$MEMORY" )
fi
if [[ "$NANOCPUS" -gt 0 ]]; then
    RUN_ARGS+=( --cpus "$(awk -v n="$NANOCPUS" 'BEGIN { printf "%.3f", n / 1000000000 }')" )
fi

# Process arguments: docker run takes a single --entrypoint value, so a
# multi-element entrypoint is replicated as entrypoint[0] plus the remaining
# elements prepended to the command.
PROC_ARGS=()
if (( ${#ENTRYPOINT_ARGS[@]} > 1 )); then
    PROC_ARGS+=( "${ENTRYPOINT_ARGS[@]:1}" )
fi
PROC_ARGS+=( ${CMD_ARGS[@]+"${CMD_ARGS[@]}"} )

# ---------------------------------------------------------------------------
# Dry-run gate: everything above leaves the running service untouched.
# ---------------------------------------------------------------------------
if [[ "$OCC3_DRY_RUN" == "1" ]]; then
    log "dry run (OCC3_DRY_RUN=1): stopping before any mutation"
    log "planned launch (redacted):"
    log "  docker run ${RUN_ARGS[*]} \\"
    log "      $NEW_IMAGE $(redact_args ${PROC_ARGS[@]+"${PROC_ARGS[@]}"})"
    log "dry run complete; no changes were made to the running service"
    exit 0
fi

# Persist rollback state before mutating anything (no secrets in this file).
{
    printf '# Written by deploy.sh; consumed and removed by rollback.sh. No secrets.\n'
    printf 'OCC3_STATE_NEW_IMAGE=%s\n' "$NEW_IMAGE"
    printf 'OCC3_STATE_BASE_REF=%s\n' "$BASE_REF"
    printf 'OCC3_STATE_RESTART_POLICY=%s\n' "${RESTART_ARG:-no}"
    printf 'OCC3_STATE_DEPLOYED_AT=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
} > "$STATE_FILE"
chmod 600 "$STATE_FILE"
log "wrote deploy state to $STATE_FILE"

# ---------------------------------------------------------------------------
# Step 5: mutation. Happens only after build + preflight succeeded.
# ---------------------------------------------------------------------------
log "=== mutation phase begins (service downtime starts here) ==="

if container_exists "$PRE_CONTAINER"; then
    log "$PRE_CONTAINER already exists (earlier attempt); skipping rename"
else
    docker rename "$BASE_CONTAINER" "$PRE_CONTAINER" \
        || die "docker rename failed; nothing else has been changed"
    log "renamed $BASE_CONTAINER -> $PRE_CONTAINER (original container preserved for rollback)"
fi

if container_running "$PRE_CONTAINER"; then
    log "stopping $PRE_CONTAINER (timeout ${OCC3_STOP_TIMEOUT}s)"
    docker stop -t "$OCC3_STOP_TIMEOUT" "$PRE_CONTAINER" >/dev/null
fi

# Keep the preserved container from auto-starting on a docker daemon restart
# (restart policy 'always' would fight the new container for the port).
# rollback.sh restores the original policy from $STATE_FILE.
docker update --restart=no "$PRE_CONTAINER" >/dev/null
log "set $PRE_CONTAINER restart policy to 'no' (original: ${RESTART_ARG:-none}; rollback.sh restores it)"

log "waiting ${OCC3_GPU_SETTLE}s for GPU memory release before starting the new container"
sleep "$OCC3_GPU_SETTLE"

log "starting new container $BASE_CONTAINER from $NEW_IMAGE"
if ! NEW_CID="$(docker run "${RUN_ARGS[@]}" "$NEW_IMAGE" ${PROC_ARGS[@]+"${PROC_ARGS[@]}"})"; then
    log "docker run failed; run ./rollback.sh to restore the previous server"
    die "docker run failed for the new $BASE_CONTAINER"
fi
log "new container started: $NEW_CID"

# Re-attach any extra networks beyond the primary one.
if (( ${#NETWORKS[@]} > 1 )); then
    for net in ${NETWORKS[@]+"${NETWORKS[@]}"}; do
        if [[ "$net" == "$NETWORK_MODE" || "$net" == "bridge" ]]; then
            continue
        fi
        if docker network connect "$net" "$BASE_CONTAINER" 2>/dev/null; then
            log "attached extra network $net to $BASE_CONTAINER"
        else
            log "WARNING: failed to attach extra network $net to $BASE_CONTAINER"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Step 6: wait for the server to come back (B300 weight load takes minutes)
# ---------------------------------------------------------------------------
if ! wait_for_http_200 "$HEALTH_URL" "$BASE_CONTAINER" "$OCC3_HEALTH_TIMEOUT"; then
    log "FAILURE: the new server did not become healthy"
    log "  inspect with: docker logs --tail 200 $BASE_CONTAINER"
    log "  restore the previous server with: ./rollback.sh"
    exit 1
fi
log "health check passed: $HEALTH_URL -> 200"

# ---------------------------------------------------------------------------
# Step 7: verify the plugin is actually serving
# ---------------------------------------------------------------------------
code="$(auth_get_code "$STATUS_URL")"
if [[ "$code" != "200" ]]; then
    log "FAILURE: plugin endpoint $STATUS_URL returned HTTP $code (expected 200)"
    log "  the server is healthy but occ3-bridge is not serving; likely causes:"
    log "  - the plugin failed to load: docker logs $BASE_CONTAINER 2>&1 | grep -i plugin"
    log "  - SGLANG_PLUGINS is set without occ3_bridge in it"
    log "  restore the previous server with: ./rollback.sh"
    exit 1
fi
body="$(auth_get_body "$STATUS_URL")"
if [[ -s "$KEY_FILE" ]]; then
    # Defensive: never let the key reach the log, even inside a status body.
    body="${body//$(cat "$KEY_FILE")/***}"
fi
log "plugin status: $STATUS_URL -> 200 body: ${body:0:400}"
if ! printf '%s' "$body" | grep -q '"version"'; then
    log "WARNING: status body has no \"version\" field; check the route shape"
fi

log "=== SUCCESS: occ3-bridge is live on $BASE_CONTAINER ==="
log "  new image:     $NEW_IMAGE"
log "  new container: $NEW_CID"
log "  old container: $PRE_CONTAINER (stopped, preserved)"
log "  health:        $HEALTH_URL -> 200"
log "  plugin status: $STATUS_URL -> 200"
log "  to roll back:  ./rollback.sh"
log "  log file:      $LOG_FILE"
