# occ3-bridge deployment (SGLang plugin on the DGX `neo-glm53` server)

Deployment tooling for the `occ3-bridge` SGLang plugin on the NVIDIA DGX
B300 container host that serves GLM-5.3 FP8 (TP8) through SGLang 0.5.21.

The plugin is a pip package (`occ3-bridge`) that registers the entry point
`occ3_bridge` in the group `sglang.srt.plugins`. SGLang discovers plugins in
that group at process start only -- there is no hot-load path -- so the
deployment restarts the server process exactly once and keeps everything
additive: the original container and image are preserved, and a documented
rollback restores the previous state.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds the derived image: running base image (digest-pinned) + the `occ3-bridge` package, nothing else. |
| `deploy.sh` | Captures the live container configuration, builds, preflights, and rolls the server onto the derived image. |
| `rollback.sh` | Restores the pre-occ3 container and waits for it to come back healthy. |
| `deploy-occ3.log`, `rollback-occ3.log` | Timestamped run logs (created by the scripts). |
| `occ3-deploy-state.env` | Small non-secret state file written by `deploy.sh` and removed by `rollback.sh`. |

## Verified target environment (2026-10-21 inspection)

- Container: `neo-glm53` (do not touch `neo-k3-ingress` or `launchpad-traefik`).
- Image: `lmsysorg/sglang`, digest
  `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
  (52.4 GB), SGLang 0.5.21.
- Endpoint: `127.0.0.1:30001` (host port map; external traffic goes through
  `launchpad-traefik` / Pomerium).
- Model: GLM-5.3 FP8, `--tp-size 8`, `--context-length 1048576`, EAGLE
  speculative decoding, `--kv-cache-dtype fp8_e4m3`, launch command protected
  by `--api-key`.
- GPUs are ~99% committed: a second model instance is impossible; the
  deployment must attach to the existing server process.

## Preconditions

On the DGX host (Ubuntu, docker CLI, curl, bash):

1. The extension tree must be present on the host, complete and buildable:
   `occ3_sglang/` (including the plugin entry module `occ3_sglang/plugin.py`)
   and `pyproject.toml`. deploy.sh preflights the entry point inside the
   built image and aborts before touching the running server if anything is
   missing.
2. `docker info` must work (run as root or from the `docker` group).
3. The `neo-glm53` container must exist and run (override with
   `OCC3_ALLOW_STOPPED_BASE=1` to deploy onto a stopped base).

Copy the tree to the host first, for example:

```
rsync -av --exclude .git \
    extensions/sglang-occ3/ <dgx-host>:/data/occ3/sglang-occ3/
```

## What deploy.sh does, step by step

Run it from the `deploy/` directory on the DGX host:

```
sudo bash deploy.sh
```

Every step is logged to `deploy-occ3.log` with UTC timestamps.

1. **Prerequisites.** Checks `docker`, `curl`, the Dockerfile, the package
   source, a reachable docker daemon, and that the base container exists
   and is running. Aborts before any mutation if `docker inspect` fails.
2. **Capture.** `docker inspect` of the running container into a 0700
   staging directory: image reference and digest, entrypoint, command line,
   environment (to a root-only env file), port bindings, volumes, networks,
   restart policy, GPU device requests, shm-size, ulimits, ipc/pid modes,
   runtime, user, workdir, init flag, tmpfs mounts, extra hosts, and
   memory/CPU limits.
3. **API key handling.** The `--api-key` value is extracted from the launch
   command into the root-only file `/tmp/.occ3_deploy_key` (0600), used for
   all authenticated HTTP checks, and scrubbed on exit. It is never printed
   or logged (environment values are never logged at all; the command line
   is logged in redacted form).
4. **Idempotency check.** If the current `neo-glm53` already serves
   `/occ3/v1/status` with HTTP 200, the script reports "already deployed"
   and exits without changing anything.
5. **Build.** Builds the derived image from the base image pinned by the
   digest read at runtime (`docker inspect --format '{{index .RepoDigests
   0}}'`), tagged `neo-glm53:occ3-<UTC date-time>`. The only change over
   the base image is the installed `occ3-bridge` package (`pip install
   --no-deps`, no apt packages).
6. **Preflight.** Runs a short, CPU-only container from the new image and
   verifies that the `occ3_bridge` entry point is registered, present, and
   importable. Any failure aborts the deploy before the running service is
   touched.
7. **Mutation (only after build + preflight succeeded).** Renames
   `neo-glm53` to `neo-glm53-pre-occ3` (the original container is
   preserved, not deleted), stops it, sets its restart policy to `no` (so a
   docker daemon restart cannot start both containers into a port
   conflict; rollback restores the original policy), waits a few seconds
   for GPU memory release, then `docker run -d` starts a NEW `neo-glm53`
   with the captured configuration on the derived image.
8. **Health wait.** Polls `GET http://127.0.0.1:30001/health` with the
   Authorization header from the key file until HTTP 200 (timeout 1500 s;
   loading the B300 weights takes minutes). Fails fast if the container
   dies.
9. **Plugin verification.** `GET /occ3/v1/status` with the same
   authentication; expects HTTP 200 and a JSON body containing the occ3
   version.
10. **Report.** On success, prints the new image tag, container id, the
    preserved old container, and the rollback command. On failure at or
    after the mutation phase, prints the rollback instructions.

A no-op rehearsal is available:

```
sudo env OCC3_DRY_RUN=1 bash deploy.sh     # capture + build + preflight only
```

## Restart caveat (read before deploying)

SGLang loads plugins only at process start. Deploying therefore restarts
the server process once:

- Every in-flight request and every model session powered by this server
  is killed at `docker stop`. Schedule the deploy in a maintenance window.
- The KV/radix cache starts cold; expect the usual minutes-long weight
  load before `/health` turns 200 (the script waits up to 1500 s).
- External access through `launchpad-traefik` / Pomerium is interrupted
  for the same interval.
- Keep `rollback.sh` at hand; the whole restore path is exercised only by
  the scripts in this directory.

## Verifying the plugin is live

`/health`, `/ready` and `/metrics` are exempt from authentication in
SGLang 0.5.21, but every other route (including `/occ3/v1/*`) requires the
server API key. Extract it from the running container (never print it) and
call the status route:

```
KEY=$(docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' neo-glm53 \
      | awk 'prev=="--api-key"{print; exit} {prev=$0}')
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:30001/occ3/v1/status
```

Expected: HTTP 200 and a JSON body with the occ3 version, for example:

```
{"plugin":"occ3_bridge","version":"0.1.0", ...}
```

A 404 means the plugin did not load (check `docker logs neo-glm53 2>&1 |
grep -i plugin`). A 401 means the Authorization header is missing or wrong.

## Rollback

```
sudo bash rollback.sh
```

What it does:

1. Stops and removes the occ3-bridge container currently named
   `neo-glm53` (only the OCC-3-added container is ever removed).
2. Renames `neo-glm53-pre-occ3` back to `neo-glm53`.
3. Restores the original restart policy from
   `occ3-deploy-state.env` (deploy.sh had set it to `no`).
4. Starts the restored container.
5. Polls `GET http://127.0.0.1:30001/health` until 200 (timeout 1500 s).
6. Reports the plugin endpoint status (a 404 there confirms the plugin
   routes are gone) and removes the state file.

The API key is only read from `docker inspect` into memory for the health
poll; it is never printed, logged, or written to a file. The derived image
(`neo-glm53:occ3-...`) is not deleted automatically; remove it with
`docker rmi <tag>` once the rollback is confirmed. The base image and the
original container are never deleted by deploy or rollback.

Manual equivalent, for reference:

```
docker stop -t 300 neo-glm53 && docker rm neo-glm53
docker rename neo-glm53-pre-occ3 neo-glm53
docker update --restart <original-policy> neo-glm53   # from occ3-deploy-state.env
docker start neo-glm53
# then poll: curl -s http://127.0.0.1:30001/health
```

## SGLANG_PLUGINS whitelist

SGLang 0.5.21 auto-discovers every installed entry point in
`sglang.srt.plugins`. Setting the environment variable `SGLANG_PLUGINS`
(comma-separated entry point names) restricts loading to the named
plugins only; when it is unset or empty, all discovered plugins load.
To whitelist only this plugin, set:

```
SGLANG_PLUGINS=occ3_bridge
```

deploy.sh can do this for the new container:

```
sudo env OCC3_SET_PLUGIN_WHITELIST=1 bash deploy.sh
```

(the value is appended to the replicated environment before the new
container is started; re-running with the flag after a deploy is a no-op
unless you roll back first). To set it permanently at the image level
instead, add `ENV SGLANG_PLUGINS=occ3_bridge` to the Dockerfile -- that
option is off by default to keep the derived image byte-equivalent to the
base apart from the plugin package.

## Security notes

- No secret is stored in any file in this directory. The API key exists,
  transiently, only in `/tmp/.occ3_deploy_key` (0600) during `deploy.sh`,
  and in memory during `rollback.sh`.
- The key never appears in the run logs, in captured summaries (env values
  are never logged; the command line is redacted), or on any process
  command line (it is piped to curl through a stdin config).
- The staging directory with the captured env/command lives in a 0700
  directory under `/tmp` and is deleted on every exit path.
- Note that `docker inspect neo-glm53` reveals the launch command
  (including the API key) to every user with docker access, exactly as it
  did before the deploy; this tooling does not change that exposure.
- `docker run` receives the captured command line as arguments, so the
  key is briefly visible in `/proc` to other processes on the host while
  that command executes. Run the deploy from a trusted shell as root.

## Replicated container settings

deploy.sh replicates from `docker inspect`: image (replaced by the derived
one), entrypoint, command, environment (via `--env-file`), port bindings,
volumes (`-v` binds, with `--mount` reconstruction as fallback), network
mode (plus re-attaching extra networks), restart policy, GPU device
requests (`--gpus ...`), shm-size, ulimits, ipc/pid modes, runtime, user,
workdir, init flag, tmpfs mounts, extra hosts, and memory/CPU limits.

Not replicated (harmless for this server, reviewed during the verified
inspection): container hostname, labels, log driver configuration,
`--device` entries other than GPUs, capability/security options, and
tty/interactive mode.

## Assumptions and limitations

- Bash 4.4+ (Ubuntu), docker CLI with BuildKit, curl. No `jq` dependency:
  all parsing uses `docker inspect` Go templates.
- Container command line and environment entries must not contain literal
  newlines (they never do for this server).
- Containers using `--network container:<id>` are refused (not the case
  here).
- If the box has multiple networks attached to `neo-glm53`, the primary
  one is passed to `docker run` and extras are re-attached afterwards.
- The deploy does not push images anywhere and does not touch other
  containers (`neo-k3-ingress`, `launchpad-traefik`) or the model weights
  under `/data/neo-glm53`.

## Controller-based deployment (canonical)

`deploy.sh`/`rollback.sh` are superseded by `occ3_controller.py`, which
fixes the root cause of the 2026-10-21 outage (docker 29.x range
templates append a trailing blank line even for non-empty ranges, so
line-based capture injected an empty-string argument into the
replicated launch command):

- `docker inspect` is read as structured JSON; Entrypoint/Cmd/env are
  carried as exact arrays, never newline-split.
- The candidate container is created WITHOUT starting it, and its
  docker-recorded configuration is diffed element-wise against the
  working original. Only image, name, and the rotated api-key element
  may differ.
- Any failure after the mutation begins triggers an automatic rollback
  that restores the preserved original and re-verifies it (health,
  authenticated chat, ingress path).
- `occ3_watchdog.sh` supervises a deploy with a hard timeout and runs
  the rollback even if the controller itself is killed; it never
  retries the deployment.
- `disposable-test` rehearses the three failure scenarios (startup
  crash, health timeout, validation failure) against throwaway
  containers, proving the rollback restores and verifies service.
- `occ3_key_watcher.py` runs on the client VM and flips the local
  User-scope `GLM53_API_KEY` only after the controller reports
  SUCCESS, keeping the local key in sync with the rotated server key.

See `tests/test_controller.py` for the offline regression tests.
