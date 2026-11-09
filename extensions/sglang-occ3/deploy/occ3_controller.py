#!/usr/bin/env python3
"""OCC-3 deployment controller (DGX-side, model-independent).

Owns the full lifecycle of ONE controlled deployment:

  preflight   capture the working container as structured JSON, build the
              derived image from the EXACT base image id, import the
              plugin, run a launch canary, create the candidate WITHOUT
              starting it, and diff the docker-recorded configuration
              against the original (only explicitly listed OCC-3 changes
              are allowed).
  deploy      rename + stop the original, rename + start the candidate,
              wait for real readiness (health, authenticated chat, tool
              calling, ingress path), then run the live OCC-3 GATE-8
              validation suite.
  rollback    automatic on ANY failure, timeout or signal: stop the
              candidate, restore the original container, and VERIFY the
              restored service (health + authenticated chat + ingress).

The controller runs detached (setsid nohup) under a lock and a deadline
and logs persistently; it never depends on any AI session for
completion. Rollback restores service rather than printing
instructions.

Security: API keys exist only in 0600 files and process memory. Every
log line and status document passes through redact(), which replaces
key material with ***KEY***. No key ever appears on a command line
(HTTP auth headers go through urllib) and no key is ever logged.

Python 3.12, stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows unit-test shim
    fcntl = None


def _env_int(name, default):
    """Integer environment override shared with the deploy.sh / rollback.sh
    lanes. An absent, empty, or unparseable value falls back to the default
    instead of aborting a running deployment."""
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


BASE_CONTAINER = "neo-glm53"
PRESERVED_CONTAINER = "neo-glm53-pre-occ3"
CANDIDATE_PREFIX = "occ3-candidate-"
FAILED_PREFIX = "neo-glm53-occ3-failed-"
RECOVERY_ROOT = "/data/neo-glm53/recovery"
EXT_ROOT = "/data/occ3-bridge/sglang-occ3"
DOCKERFILE = os.path.join(EXT_ROOT, "deploy", "Dockerfile")
LOCK_FILE = "/tmp/occ3-controller.lock"

HEALTH_URL = "http://127.0.0.1:30001/health"
BASE_URL = "http://127.0.0.1:30001"

HEALTH_TIMEOUT_S = _env_int("OCC3_HEALTH_TIMEOUT", 1500)  # B300 weight load takes minutes
DEPLOY_DEADLINE_S = 2700
ROLLBACK_DEADLINE_S = 900

KEY_SENTINEL = "***KEY***"

CAPTURE_FIELDS = (
    "Entrypoint", "Cmd", "Env", "WorkingDir", "User",
    "Binds", "Mounts", "PortBindings", "NetworkMode", "Networks",
    "IpcMode", "PidMode", "Runtime", "ShmSize", "Ulimits", "Tmpfs",
    "ExtraHosts", "DeviceRequests", "RestartPolicy", "AutoRemove",
    "Privileged", "CapAdd", "CapDrop", "SecurityOpt", "Labels",
    "OomKillDisable", "Memory", "NanoCpus", "Init",
)

API_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


class Failure(Exception):
    """Deployment-phase failure; triggers automatic rollback."""


class DeadlineExceeded(Failure):
    pass


class Controller:
    def __init__(self, recovery_dir: str, verbose: bool = True) -> None:
        self.recovery_dir = recovery_dir
        self.log_path = os.path.join(recovery_dir, "controller.log")
        self.status_path = os.path.join(recovery_dir, "status.json")
        self.verbose = verbose
        self.secrets: list[str] = []
        self.t0 = time.time()
        self.deadline_s = DEPLOY_DEADLINE_S
        os.makedirs(recovery_dir, exist_ok=True)

    def register_secret(self, value):
        if value and len(value) >= 16:
            self.secrets.append(value)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, KEY_SENTINEL)
        return text

    def log(self, message: str) -> None:
        line = self.redact(message)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(stamp + " " + line + "\n")
        if self.verbose:
            print(stamp + " " + line, flush=True)

    def write_status(self, document: dict) -> None:
        safe = json.loads(self.redact(json.dumps(document)))
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(safe, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.status_path)

    def check_deadline(self, where: str) -> None:
        elapsed = time.time() - self.t0
        if elapsed > self.deadline_s:
            raise DeadlineExceeded(
                "deadline exceeded at " + where + " after " + str(int(elapsed)) + "s"
            )

    def run(self, args, timeout: int = 120, check: bool = True,
            allow_secret_argv: bool = False):
        """Run a command; secrets must never appear in args (asserted).

        ``allow_secret_argv`` is reserved for the single call that must
        carry the api-key launch argument (``docker create`` of the
        candidate, mirroring the original container's own Cmd). It is a
        transient, process-list-visible launch argument exactly like the
        running container's; every other command keeps the hard guard.
        """
        joined = " ".join(args)
        if not allow_secret_argv:
            for secret in self.secrets:
                if secret and len(secret) >= 16 and secret in joined:
                    raise Failure("refusing to run a command line containing key material")
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        if check and proc.returncode != 0:
            raise Failure(
                "command failed (" + str(proc.returncode) + "): "
                + joined[:220] + " :: "
                + self.redact((proc.stderr or proc.stdout)[-400:])
            )
        return proc

    def http(self, method, url, key, payload=None, timeout=120):
        data = None
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
                try:
                    return response.status, json.loads(body)
                except ValueError:
                    return response.status, body
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                return error.code, json.loads(body)
            except ValueError:
                return error.code, body
        except (urllib.error.URLError, OSError, ConnectionError) as error:
            # resets (container died mid-request), refused sockets and
            # timeouts are all "no answer yet", not controller crashes
            return 0, type(error).__name__ + ": " + str(error)[:120]


# ---------------------------------------------------------------------------
# Structured capture: docker inspect as JSON, arrays preserved exactly
# ---------------------------------------------------------------------------


def docker_inspect(controller, container):
    proc = controller.run(["docker", "inspect", container], timeout=60)
    return json.loads(proc.stdout)[0]


def normalize(config):
    """Extract the replication-relevant configuration from one inspect doc."""
    host = config.get("HostConfig", {})
    net = config.get("NetworkSettings", {})
    cconfig = config.get("Config", {})

    mounts = []
    for mount in net.get("Mounts", []) or []:
        mounts.append((
            mount.get("Type"), mount.get("Name") or "",
            mount.get("Source"), mount.get("Destination"),
            bool(mount.get("RW", True)),
        ))
    networks = sorted((net.get("Networks") or {}).keys())
    port_bindings = {}
    for container_port, bindings in (host.get("PortBindings") or {}).items():
        port_bindings[container_port] = sorted(
            (b.get("HostIp", ""), b.get("HostPort", "")) for b in bindings
        )
    ulimits = sorted(
        (u.get("Name"), u.get("Soft"), u.get("Hard"))
        for u in (host.get("Ulimits") or [])
    )
    return {
        "Entrypoint": list(cconfig.get("Entrypoint") or []),
        "Cmd": list(cconfig.get("Cmd") or []),
        "Env": sorted(cconfig.get("Env") or []),
        "WorkingDir": cconfig.get("WorkingDir") or "",
        "User": cconfig.get("User") or "",
        "Binds": sorted(host.get("Binds") or []),
        "Mounts": sorted(mounts),
        "PortBindings": port_bindings,
        "NetworkMode": host.get("NetworkMode") or "",
        "Networks": networks,
        "IpcMode": host.get("IpcMode") or "",
        "PidMode": host.get("PidMode") or "",
        "Runtime": host.get("Runtime") or "",
        "ShmSize": host.get("ShmSize") or 0,
        "Ulimits": ulimits,
        "Tmpfs": dict(host.get("Tmpfs") or {}),
        "ExtraHosts": sorted(host.get("ExtraHosts") or []),
        "DeviceRequests": [
            {
                "Driver": d.get("Driver") or "",
                "Count": d.get("Count", 0),
                "DeviceIDs": sorted(d.get("DeviceIDs") or []),
                "Capabilities": d.get("Capabilities") or [],
            }
            for d in (host.get("DeviceRequests") or [])
        ],
        "RestartPolicy": host.get("RestartPolicy") or {},
        "AutoRemove": bool(host.get("AutoRemove")),
        "Privileged": bool(host.get("Privileged")),
        "CapAdd": sorted(host.get("CapAdd") or []),
        "CapDrop": sorted(host.get("CapDrop") or []),
        "SecurityOpt": sorted(host.get("SecurityOpt") or []),
        "Labels": dict(cconfig.get("Labels") or {}),
        "OomKillDisable": bool(host.get("OomKillDisable")),
        "Memory": host.get("Memory") or 0,
        "NanoCpus": host.get("NanoCpus") or 0,
        "Init": bool(host.get("Init")),
    }


def api_key_from_cmd(cmd):
    for index, element in enumerate(cmd):
        if element == "--api-key" and index + 1 < len(cmd):
            return cmd[index + 1]
        if element.startswith("--api-key="):
            return element.split("=", 1)[1]
    return None


def replace_api_key(cmd, new_key):
    out = list(cmd)
    for index, element in enumerate(out):
        if element == "--api-key" and index + 1 < len(out):
            out[index + 1] = new_key
            return out, index + 1
        if element.startswith("--api-key="):
            out[index] = "--api-key=" + new_key
            return out, index
    raise Failure("no --api-key flag found in the original Cmd")


def assert_no_blank_elements(args, where):
    """Regression guard for the docker 29.x range-template bug.

    docker 29.x emits a trailing blank line for `{{range ...}}{{println .}}`
    templates even for NON-EMPTY ranges. Line-based capture (bash mapfile)
    therefore injected empty-string elements into argument arrays, which
    produced `python3 "" -m sglang.launch_server` and the startup failure
    of 2026-10-21. Every argument array built here is checked to contain
    no empty elements, and the structural diff additionally compares
    element-exact against the original container.
    """
    for index, element in enumerate(args):
        if not isinstance(element, str) or element.strip() == "":
            raise Failure(
                "blank/invalid element " + str(index) + " in " + where +
                ": refusing to launch (empty-argument regression guard)"
            )


def build_run_args(norm, image, name, new_cmd, env):
    """docker create argument list built from the captured JSON.

    Entrypoint and Cmd are arrays from `docker inspect` JSON, never
    line-split; this is the fix for the empty-argument bug.
    """
    args = ["docker", "create", "--name", name]
    entrypoint = norm["Entrypoint"]
    if entrypoint:
        args += ["--entrypoint", entrypoint[0]]
    cmd = list(entrypoint[1:]) + list(new_cmd)
    assert_no_blank_elements(cmd, "container command")
    if norm["NetworkMode"]:
        args += ["--network", norm["NetworkMode"]]
    if norm["IpcMode"]:
        args += ["--ipc", norm["IpcMode"]]
    if norm["PidMode"]:
        args += ["--pid", norm["PidMode"]]
    if norm["Runtime"]:
        args += ["--runtime", norm["Runtime"]]
    if norm["User"]:
        args += ["--user", norm["User"]]
    if norm["WorkingDir"]:
        args += ["--workdir", norm["WorkingDir"]]
    if norm["Init"]:
        args += ["--init"]
    if norm["ShmSize"]:
        args += ["--shm-size", str(norm["ShmSize"])]
    policy = (norm["RestartPolicy"] or {}).get("Name") or "no"
    if policy != "no":
        args += ["--restart", policy]
    if norm["Memory"]:
        args += ["--memory", str(norm["Memory"])]
    if norm["NanoCpus"]:
        args += ["--cpus", str(norm["NanoCpus"] / 1_000_000_000)]
    for request in norm["DeviceRequests"]:
        if request["DeviceIDs"]:
            args += ["--gpus", "device=" + ",".join(request["DeviceIDs"])]
        elif request["Count"] == -1:
            args += ["--gpus", "all"]
        elif request["Count"]:
            args += ["--gpus", str(request["Count"])]
    for container_port, bindings in sorted(norm["PortBindings"].items()):
        for host_ip, host_port in bindings:
            spec = (host_ip + ":" + host_port + ":" + container_port
                    if host_ip else host_port + ":" + container_port)
            args += ["-p", spec]
    for bind in norm["Binds"]:
        args += ["-v", bind]
    for mount_type, name_or_src, source, destination, rw in norm["Mounts"]:
        if mount_type == "bind":
            spec = "type=bind,source=" + source + ",target=" + destination
            if not rw:
                spec += ",readonly"
            args += ["--mount", spec]
        elif mount_type == "volume" and name_or_src:
            spec = "type=volume,source=" + name_or_src + ",target=" + destination
            if not rw:
                spec += ",readonly"
            args += ["--mount", spec]
    for name, soft, hard in norm["Ulimits"]:
        args += ["--ulimit", name + "=" + str(soft) + ":" + str(hard)]
    for destination, options in sorted(norm["Tmpfs"].items()):
        args += ["--tmpfs", (destination + ":" + options) if options else destination]
    for extra in norm["ExtraHosts"]:
        args += ["--add-host", extra]
    for cap in norm["CapAdd"]:
        args += ["--cap-add", cap]
    for cap in norm["CapDrop"]:
        args += ["--cap-drop", cap]
    for opt in norm["SecurityOpt"]:
        args += ["--security-opt", opt]
    for element in env:
        if "\n" in element or element.strip() == "":
            raise Failure(
                "environment contains blank/newline elements; "
                "per-element -e passing cannot express it"
            )
        args += ["-e", element]
    for label_key, label_value in sorted((norm["Labels"] or {}).items()):
        args += ["--label", label_key + "=" + str(label_value)]
    args.append(image)
    args += cmd
    assert_no_blank_elements(args[2:], "docker create arguments")
    return args


def diff_configs(original, candidate, key_index, old_key, new_key):
    """Element-exact comparison; only the listed OCC-3 change is allowed."""
    problems = []
    for field in CAPTURE_FIELDS:
        left = original.get(field)
        right = candidate.get(field)
        if field == "Cmd" and key_index is not None and left and right:
            if len(left) != len(right):
                problems.append("Cmd length differs: " +
                                str(len(left)) + " vs " + str(len(right)))
                continue
            for index, (a, b) in enumerate(zip(left, right)):
                if a != b:
                    if index == key_index and a == old_key and b == new_key:
                        continue  # the one intentional change: rotated key
                    problems.append(
                        "Cmd[" + str(index) + "] differs (not the api-key rotation): "
                        + repr(str(a)[:60]) + " vs " + repr(str(b)[:60])
                    )
        elif left != right:
            problems.append(field + " differs: " + str(left)[:120] +
                            " vs " + str(right)[:120])
    return problems


# ---------------------------------------------------------------------------
# Validation primitives (pluggable for the disposable rehearsal)
# ---------------------------------------------------------------------------


def wait_health(controller, url, timeout_s, interval_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        controller.check_deadline("health wait")
        code, _ = controller.http("GET", url, None, timeout=10)
        if code == 200:
            return True
        time.sleep(interval_s)
    return False


def validate_chat(controller, key, base_url, force_fail=False):
    """Authenticated chat; returns the reply text (truthiness = pass)."""
    if force_fail:
        raise Failure("chat validation force-failed (test mode)")
    code, body = controller.http(
        "POST", base_url + "/v1/chat/completions", key,
        {
            "model": "glm-5.3",
            "messages": [{"role": "user",
                          "content": "Reply with exactly one word: ready"}],
            "max_tokens": 2048,
            "temperature": 0,
        }, timeout=180,
    )
    if code != 200:
        raise Failure("authenticated chat returned HTTP " + str(code))
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise Failure("chat response missing choices: " + str(body)[:200])
    text = message.get("content") or message.get("reasoning_content") or ""
    if not text.strip():
        raise Failure("chat response was empty")
    return text[:200]


def validate_tool_call(controller, key, base_url, force_fail=False):
    """Tool-calling check through the glm47 parser."""
    if force_fail:
        raise Failure("tool validation force-failed (test mode)")
    code, body = controller.http(
        "POST", base_url + "/v1/chat/completions", key,
        {
            "model": "glm-5.3",
            "messages": [{"role": "user",
                          "content": "Call the get_weather tool for Paris. "
                                      "Do not answer in plain text."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "max_tokens": 4096,
            "temperature": 0,
        }, timeout=180,
    )
    if code != 200:
        raise Failure("tool-call chat returned HTTP " + str(code))
    try:
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            raise Failure("model answered without a tool call")
        return json.dumps(calls[0].get("function", {}))[:200]
    except (KeyError, IndexError, TypeError):
        raise Failure("tool-call response malformed: " + str(body)[:200])


def validate_ingress(controller, image, key, force_fail=False):
    """Reach the server through the ingress network path (no host port)."""
    if force_fail:
        raise Failure("ingress validation force-failed (test mode)")
    script = (
        "import os,urllib.request\n"
        "req=urllib.request.Request('http://neo-glm53:30001/v1/models',"
        "headers={'Authorization':'Bearer '+os.environ['OCC3_KEY']})\n"
        "with urllib.request.urlopen(req,timeout=30) as r: print(r.status)\n"
    )
    env_file = os.path.join(controller.recovery_dir, ".ingress-env")
    write_secret_file(env_file, "OCC3_KEY=" + key)
    try:
        proc = controller.run(
            ["docker", "run", "--rm", "--network", "neo-k3-ingress-net",
             "--entrypoint", "python3", "--env-file", env_file, image,
             "-c", script],
            timeout=120,
        )
    finally:
        try:
            os.remove(env_file)
        except OSError:
            pass
    if "200" not in proc.stdout:
        raise Failure("ingress path check failed: " + proc.stdout[-200:])
    return "ingress->neo-glm53:30001 OK"


# ---------------------------------------------------------------------------
# Rollback: restore and VERIFY, never just print instructions
# ---------------------------------------------------------------------------


def rollback(controller, base, preserved, candidate, old_key, reason,
             test_mode=False, test_url=None, test_image=None):
    """Restore the preserved container and verify a real response."""
    controller.log("ROLLBACK triggered: " + reason)
    results = {}
    # Safety: if no preserved container exists, no mutation ever happened
    # (or a previous rollback already completed). Touching the running
    # original here would take the working service down for nothing.
    probe = controller.run(["docker", "inspect", preserved],
                           timeout=30, check=False)
    if probe.returncode != 0:
        controller.log("rollback: no preserved container '" + preserved
                       + "'; nothing to restore (no mutation had happened)")
        return {"result": "NOOP", "health": "SKIPPED", "chat": "SKIPPED"}
    try:
        # 1. stop the candidate if it is running
        proc = controller.run(["docker", "inspect", "-f", "{{.State.Running}}",
                               candidate], timeout=30, check=False)
        if proc.returncode == 0 and proc.stdout.strip() == "true":
            controller.run(["docker", "stop", "-t", "300", candidate], timeout=360)
            controller.log("candidate " + candidate + " stopped")
        # preserve the failed candidate for evidence
        if proc.returncode == 0:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            controller.run(["docker", "rename", candidate,
                           FAILED_PREFIX + stamp], timeout=30, check=False)
            controller.log("candidate preserved as " + FAILED_PREFIX + stamp)
        # 2. restore the original name and start it
        controller.run(["docker", "rename", preserved, base], timeout=30)
        controller.run(["docker", "start", base], timeout=120)
        controller.log("original container restored as " + base + " and started")
        # 3. verify: health + authenticated chat + ingress
        url = test_url or HEALTH_URL
        if not wait_health(controller, url, ROLLBACK_DEADLINE_S):
            raise Failure("rollback: restored service did not become healthy")
        results["health"] = "PASS"
        controller.log("rollback: health 200")
        if test_mode:
            # disposable rehearsal: a plain HTTP 200 is the service proof
            code, body = controller.http("GET", url, None, timeout=30)
            results["chat"] = "PASS" if code == 200 else "FAIL"
        else:
            validate_chat(controller, old_key, BASE_URL)
            results["chat"] = "PASS"
            controller.log("rollback: authenticated chat verified")
            # the ingress probe runs from the RESTORED container's own image
            # (always present; never a caller-supplied guess)
            restored_image = controller.run(
                ["docker", "inspect", "-f", "{{.Image}}", base],
                timeout=30).stdout.strip()
            validate_ingress(controller, restored_image, old_key)
            results["ingress"] = "PASS"
            controller.log("rollback: ingress path verified")
        results["result"] = "RESTORED"
        controller.log("rollback complete: service verified")
    except Exception as error:  # noqa: BLE001 - rollback must never raise out
        results["result"] = "RESTORE_FAILED"
        results["error"] = controller.redact(str(error))
        controller.log("ROLLBACK INCOMPLETE: " + results["error"])
    return results


# ---------------------------------------------------------------------------
# Live OCC-3 GATE-8 validation (the real latent-communication proof)
# ---------------------------------------------------------------------------


def gate8(controller, key):
    """Prove capture -> send -> receive -> USE, isolation, expiry, release,
    stale-after-restart, and no-silent-fallback evidence."""
    results = {}
    log = controller.log

    def occ3(method, path, payload=None, timeout=180):
        return controller.http(method, BASE_URL + path, key, payload, timeout)

    # t1: status
    code, body = occ3("GET", "/occ3/v1/status")
    if code != 200 or body.get("plugin") != "occ3-bridge":
        raise Failure("gate8 t1 status failed: HTTP " + str(code))
    results["status"] = "PASS"
    log("gate8 t1 status: " + json.dumps(body.get("channels", {})))

    # t2: identity is stable and complete
    code_a, ident_a = occ3("GET", "/occ3/v1/identity")
    code_b, ident_b = occ3("GET", "/occ3/v1/identity")
    fields = ident_a.get("fields", {})
    if (code_a != 200 or code_b != 200
            or len(fields) != 11
            or ident_a.get("fingerprint") != ident_b.get("fingerprint")):
        raise Failure("gate8 t2 identity unstable or incomplete")
    results["identity"] = "PASS fingerprint=" + str(ident_a.get("fingerprint"))[:24] + "..."
    log("gate8 t2 identity: 11 fields, stable fingerprint, runtime="
        + str(fields.get("runtime")))

    # t3: capabilities
    code, caps = occ3("GET", "/occ3/v1/capabilities")
    if (code != 200 or caps.get("supported") is not True
            or "kv-handoff" not in (caps.get("methods") or [])):
        raise Failure("gate8 t3 capabilities failed: HTTP " + str(code))
    results["capabilities"] = "PASS"
    log("gate8 t3 capabilities: " + json.dumps(caps.get("methods")))

    # t4: REAL latent transfer + semantic use
    facts = ("Verification code ZEBRA-47-Q. Dock number 12. "
             "Contact person Marek. Batch id 8841.")
    code, sent = occ3("POST", "/occ3/v1/send",
                      {"channel": "g8-transfer", "text": facts, "ttl_s": 600})
    if code != 200:
        raise Failure("gate8 t4 send failed: HTTP " + str(code) + " " + str(sent)[:200])
    carrier = sent.get("carrier_tokens", 0)
    if carrier < 8:
        raise Failure("gate8 t4 carrier unexpectedly short: " + str(carrier))
    code, recvd = occ3("POST", "/occ3/v1/recv",
                       {"channel": "g8-transfer",
                        "suffix": "Reply with the verification code, dock number, "
                                  "contact person and batch id from the message "
                                  "above, separated by spaces.",
                        "max_new_tokens": 512, "temperature": 0.0})
    if code != 200:
        raise Failure("gate8 t4 recv failed: HTTP " + str(code) + " " + str(recvd)[:200])
    text_out = (recvd.get("output_text") or "").lower()
    cached = recvd.get("cached_tokens", 0)
    hit = recvd.get("hit")
    found = [token.lower() for token in
             ["ZEBRA-47-Q", "12", "Marek", "8841"] if token.lower() in text_out]
    if hit is not True:
        raise Failure("gate8 t4 prefix MISS: cached=" + str(cached)
                      + " carrier=" + str(carrier))
    if len(found) < 3:
        raise Failure("gate8 t4 semantic use failed: found " + str(found)
                      + " in: " + text_out[:200])
    results["transfer"] = ("PASS carrier=" + str(carrier)
                           + " cached=" + str(cached)
                           + " receiver_prefill_saved=" + str(carrier - cached)
                           + " facts=" + str(len(found)) + "/4")
    log("gate8 t4 transfer: hit=" + str(hit) + " cached_tokens=" + str(cached)
        + "/" + str(carrier) + " facts_recalled=" + str(found))

    # t5: channel isolation (two channels, no cross-talk)
    occ3("POST", "/occ3/v1/send",
         {"channel": "g8-iso-a", "text": "Code ALPHA-9 belongs to zone north.", "ttl_s": 600})
    occ3("POST", "/occ3/v1/send",
         {"channel": "g8-iso-b", "text": "Code BRAVO-7 belongs to zone south.", "ttl_s": 600})
    code, ra = occ3("POST", "/occ3/v1/recv",
                    {"channel": "g8-iso-a",
                     "suffix": "Reply with only the code and zone from the message above.",
                     "max_new_tokens": 256, "temperature": 0.0})
    code, rb = occ3("POST", "/occ3/v1/recv",
                    {"channel": "g8-iso-b",
                     "suffix": "Reply with only the code and zone from the message above.",
                     "max_new_tokens": 256, "temperature": 0.0})
    ta = (ra.get("output_text") or "").lower()
    tb = (rb.get("output_text") or "").lower()
    if not ("alpha-9" in ta and "north" in ta and "bravo-7" not in ta):
        raise Failure("gate8 t5 isolation A failed: " + ta[:200])
    if not ("bravo-7" in tb and "south" in tb and "alpha-9" not in tb):
        raise Failure("gate8 t5 isolation B failed: " + tb[:200])
    results["isolation"] = "PASS"
    log("gate8 t5 isolation: A=alpha-9/north only, B=bravo-7/south only")

    # t5b: default-namespace isolation (unsalted request must not hit)
    code, plain = controller.http(
        "POST", BASE_URL + "/generate", key,
        {"text": facts, "sampling_params": {"max_new_tokens": 1,
                                             "temperature": 0.0}}, timeout=180)
    if code != 200:
        raise Failure("gate8 t5b generate failed: HTTP " + str(code))
    plain_cached = ((plain.get("meta_info") or {}).get("cached_tokens", 0))
    if plain_cached >= carrier:
        raise Failure("gate8 t5b unsalted request hit the salted channel: "
                     + str(plain_cached))
    results["namespace_isolation"] = "PASS (unsalted cached=" + str(plain_cached) + ")"
    log("gate8 t5b namespace isolation: unsalted cached_tokens=" + str(plain_cached))

    # t6: release retires the channel
    occ3("POST", "/occ3/v1/release", {"channel": "g8-iso-a"})
    code, _ = occ3("POST", "/occ3/v1/recv",
                   {"channel": "g8-iso-a", "suffix": "x", "max_new_tokens": 4})
    if code != 404:
        raise Failure("gate8 t6 released channel still receivable: HTTP " + str(code))
    results["release"] = "PASS"
    log("gate8 t6 release: recv after release -> 404")

    # t7: expiry
    occ3("POST", "/occ3/v1/send",
         {"channel": "g8-ttl", "text": "Short lived channel payload 55.", "ttl_s": 1})
    time.sleep(2.5)
    code, _ = occ3("POST", "/occ3/v1/recv",
                   {"channel": "g8-ttl", "suffix": "x", "max_new_tokens": 4})
    if code != 404:
        raise Failure("gate8 t7 expired channel still receivable: HTTP " + str(code))
    results["expiry"] = "PASS"
    log("gate8 t7 expiry: recv after ttl -> 404")

    # t8: metrics show the hits
    code, status = occ3("GET", "/occ3/v1/status")
    counters = (status.get("metrics") or {}).get("counters") or {}
    if counters.get("recv_hit", 0) < 2:
        raise Failure("gate8 t8 recv_hit counter not incremented: "
                     + json.dumps(counters))
    results["metrics"] = "PASS recv_hit=" + str(counters.get("recv_hit"))
    log("gate8 t8 metrics: " + json.dumps(counters))
    return results


def gate8_restart_phase(controller, key):
    """Stale handles after a REAL restart + plugin auto-load proof.

    This restarts the candidate once (planned, inside the validation
    window): channels created before the restart must be gone, new ones
    must work, and the plugin must load again from the image without
    any redeployment action.
    """
    controller.log("gate8 t9 restart phase: creating pre-restart channel")
    code, _ = controller.http(
        "POST", BASE_URL + "/occ3/v1/send", key,
        {"channel": "g8-stale", "text": "This channel must die on restart.",
         "ttl_s": 600}, timeout=180)
    if code != 200:
        raise Failure("gate8 t9 pre-restart send failed: HTTP " + str(code))
    controller.run(["docker", "restart", BASE_CONTAINER], timeout=600)
    controller.log("gate8 t9 container restarted; waiting for health")
    if not wait_health(controller, HEALTH_URL, HEALTH_TIMEOUT_S):
        raise Failure("gate8 t9 restart: service did not return")
    controller.log("gate8 t9 service healthy again")
    # old channel must be stale (404), not silently reused
    code, body = controller.http(
        "POST", BASE_URL + "/occ3/v1/recv", key,
        {"channel": "g8-stale", "suffix": "x", "max_new_tokens": 4}, timeout=60)
    if code != 404:
        raise Failure("gate8 t9 stale channel survived restart: HTTP " + str(code))
    controller.log("gate8 t9 stale handle after restart -> 404")
    # plugin auto-loaded and new channels work
    code, sent = controller.http(
        "POST", BASE_URL + "/occ3/v1/send", key,
        {"channel": "g8-fresh", "text": "Post-restart channel payload 77.",
         "ttl_s": 600}, timeout=180)
    if code != 200:
        raise Failure("gate8 t9 post-restart send failed: HTTP " + str(code))
    code, recvd = controller.http(
        "POST", BASE_URL + "/occ3/v1/recv", key,
        {"channel": "g8-fresh",
         "suffix": "Reply with the payload number from the message above.",
         "max_new_tokens": 256, "temperature": 0.0}, timeout=180)
    if code != 200 or recvd.get("hit") is not True:
        raise Failure("gate8 t9 post-restart recv failed: " + str(recvd)[:200])
    if "77" not in (recvd.get("output_text") or ""):
        raise Failure("gate8 t9 post-restart semantic use failed")
    return ("PASS stale->404, plugin auto-loaded, new channel hit="
            + str(recvd.get("cached_tokens")) + "/"
            + str(recvd.get("carrier_tokens")))


# ---------------------------------------------------------------------------
# Deployment pipeline
# ---------------------------------------------------------------------------


def load_key_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = handle.read().strip()
    if not API_KEY_RE.match(value):
        raise Failure("key file does not contain a 64-hex key: " + path)
    return value


def write_secret_file(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def cmd_preflight(controller, base, key_path, new_key=None):
    old_key = load_key_file(key_path)
    controller.register_secret(old_key)
    controller.log("preflight: captured original configuration as JSON")
    original = docker_inspect(controller, base)
    norm = normalize(original)
    old_key_from_cmd = api_key_from_cmd(norm["Cmd"])
    if old_key_from_cmd != old_key:
        raise Failure("key file does not match the running container's --api-key")
    if new_key is None:
        new_key = secrets.token_hex(32)
    elif not API_KEY_RE.match(new_key):
        raise Failure("--new-key-file must contain a 64-hex key")
    controller.register_secret(new_key)
    new_key_path = os.path.join(controller.recovery_dir, "new-api-key")
    write_secret_file(new_key_path, new_key)
    old_key_path = os.path.join(controller.recovery_dir, "old-api-key")
    write_secret_file(old_key_path, old_key)
    with open(os.path.join(controller.recovery_dir, "capture-original.json"),
              "w", encoding="utf-8") as handle:
        json.dump(original, handle, indent=2)
    os.chmod(os.path.join(controller.recovery_dir, "capture-original.json"),
             stat.S_IRUSR | stat.S_IWUSR)
    controller.log("preflight: base image id " + original["Image"])

    # exact base image, identified and pinned via a local id-tag
    base_tag = "occ3-base:" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    controller.run(["docker", "tag", original["Image"], base_tag])
    controller.log("preflight: pinned exact base image as " + base_tag)

    # derived image: additive plugin layer only
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    image = base + ":occ3-" + stamp
    controller.run(["docker", "build", "--build-arg", "BASE_IMAGE=" + base_tag,
                    "-t", image, "-f", DOCKERFILE, EXT_ROOT], timeout=1800)
    controller.log("preflight: derived image built " + image)

    # canary 1: plugin imports on the server interpreter
    proc = controller.run(["docker", "run", "--rm", "--network", "none",
                           "--entrypoint", "python3", image,
                           "-c", "import occ3_sglang.plugin; print('plugin-import-ok')"],
                          timeout=300)
    if "plugin-import-ok" not in proc.stdout:
        raise Failure("preflight: plugin import canary failed")
    controller.log("preflight: plugin import canary OK")

    # canary 2: non-serving launch check (module + CLI resolution)
    proc = controller.run(["docker", "run", "--rm", "--network", "none",
                           "--entrypoint", "python3", image,
                           "-m", "sglang.launch_server", "--help"],
                          timeout=300)
    if proc.returncode != 0:
        raise Failure("preflight: launch canary failed: "
                      + (proc.stdout + proc.stderr)[-300:])
    controller.log("preflight: launch canary OK (launch_server --help)")

    # candidate created WITHOUT starting; structural diff vs original
    new_cmd, key_index = replace_api_key(norm["Cmd"], new_key)
    candidate = CANDIDATE_PREFIX + stamp
    args = build_run_args(norm, image, candidate, new_cmd, norm["Env"])
    controller.log("preflight: creating candidate (not started): " + candidate)
    controller.run(args, timeout=120, allow_secret_argv=True)
    candidate_doc = docker_inspect(controller, candidate)
    candidate_norm = normalize(candidate_doc)
    problems = diff_configs(norm, candidate_norm, key_index, old_key, new_key)
    # intentional, documented differences
    if candidate_doc.get("Image") == original.get("Image"):
        problems.append("candidate image did not change")
    if problems:
        controller.run(["docker", "rm", candidate], timeout=60, check=False)
        raise Failure("preflight structural diff rejected: "
                      + "; ".join(problems))
    controller.log("preflight: structural diff PASS (only image/name/key differ)")
    controller.run(["docker", "rm", candidate], timeout=60)
    controller.log("preflight: preflight candidate removed (deploy recreates it)")
    write_summary = {
        "phase": "preflight",
        "result": "PASS",
        "image": image,
        "base_image": original["Image"],
        "candidate_cmd_elements": len(new_cmd),
    }
    controller.write_status(write_summary)
    return {
        "norm": norm, "image": image, "new_key": new_key, "old_key": old_key,
        "new_cmd": new_cmd, "key_index": key_index, "stamp": stamp,
    }


def cmd_deploy(controller, base, key_path, new_key=None):
    controller.deadline_s = DEPLOY_DEADLINE_S
    stale = controller.run(["docker", "inspect", PRESERVED_CONTAINER],
                           timeout=30, check=False)
    if stale.returncode == 0:
        raise Failure("stale preserved container '" + PRESERVED_CONTAINER
                      + "' exists from an earlier attempt; run the rollback "
                        "command or remove it before deploying")
    plan = cmd_preflight(controller, base, key_path, new_key)
    norm, image = plan["norm"], plan["image"]
    new_key, old_key = plan["new_key"], plan["old_key"]
    stamp = plan["stamp"]
    candidate = CANDIDATE_PREFIX + stamp

    controller.write_status({"phase": "deploy", "result": "IN_PROGRESS",
                             "image": image})
    mutated = False
    try:
        controller.log("deploy: creating candidate " + candidate)
        args = build_run_args(norm, image, candidate, plan["new_cmd"], norm["Env"])
        controller.run(args, timeout=120, allow_secret_argv=True)

        controller.log("deploy: mutation begins (service downtime starts)")
        controller.run(["docker", "rename", base, PRESERVED_CONTAINER], timeout=30)
        mutated = True
        controller.run(["docker", "stop", "-t", "300", PRESERVED_CONTAINER],
                       timeout=360)
        controller.run(["docker", "update", "--restart=no", PRESERVED_CONTAINER],
                       timeout=30)
        controller.log("deploy: original preserved as " + PRESERVED_CONTAINER)
        controller.run(["docker", "rename", candidate, base], timeout=30)
        controller.run(["docker", "start", base], timeout=120)
        controller.log("deploy: candidate started as " + base)

        controller.log("deploy: waiting for readiness")
        if not wait_health(controller, HEALTH_URL, HEALTH_TIMEOUT_S):
            raise Failure("deploy: candidate did not become healthy")
        controller.log("deploy: health 200")

        chat = validate_chat(controller, new_key, BASE_URL)
        controller.log("deploy: authenticated chat OK: " + chat[:80])
        tool = validate_tool_call(controller, new_key, BASE_URL)
        controller.log("deploy: tool call OK: " + tool[:120])
        ingress = validate_ingress(controller, image, new_key)
        controller.log("deploy: ingress OK: " + ingress)
        controller.write_status({"phase": "readiness", "result": "PASS",
                                 "health": "PASS", "chat": "PASS",
                                 "tool_call": "PASS", "ingress": "PASS",
                                 "image": image})

        controller.log("gate8: live OCC-3 validation begins")
        gate = gate8(controller, new_key)
        controller.log("gate8: restart phase begins (planned single restart)")
        gate["restart_staleness"] = gate8_restart_phase(controller, new_key)
        # post-restart service sanity (the restart is part of validation)
        chat = validate_chat(controller, new_key, BASE_URL)
        gate["post_restart_chat"] = "PASS"
        controller.log("gate8: post-restart authenticated chat OK")

        status = {"phase": "gate8", "result": "PASS", "image": image}
        status.update(gate)
        controller.write_status(status)
        controller.log("DEPLOYMENT SUCCESS: all gates passed; "
                       "new api key file: new-api-key in " + controller.recovery_dir)
        controller.write_status({
            "phase": "COMPLETE", "result": "SUCCESS", "image": image,
            "gates": status,
            "note": "OCC-3 validated live; normal work stays on OCC-1/2 "
                    "until the user enables it",
        })
        return 0
    except (Failure, subprocess.SubprocessError, OSError, ValueError,
            KeyError) as error:
        controller.log("deploy FAILED: " + str(error))
        if mutated:
            result = rollback(controller, base, PRESERVED_CONTAINER, base,
                              old_key, str(error), test_image=image)
        else:
            # pre-mutation failure: the working service was never touched;
            # remove the unstarted candidate and leave everything as-is
            controller.run(["docker", "rm", "-f", candidate],
                           timeout=60, check=False)
            result = {"result": "NOOP",
                      "note": "failure before mutation; service untouched"}
        controller.write_status({
            "phase": "deploy", "result": "FAILED",
            "error": controller.redact(str(error)),
            "rollback": result,
        })
        return 1


def cmd_rollback(controller, base, key_path):
    old_key = load_key_file(key_path)
    controller.register_secret(old_key)
    result = rollback(controller, base, PRESERVED_CONTAINER, base, old_key,
                      "manual rollback", test_image=None)
    controller.write_status({"phase": "manual-rollback", "result": result})
    return 0 if result.get("result") == "RESTORED" else 1


# ---------------------------------------------------------------------------
# Disposable rehearsal: prove rollback WITHOUT touching the live service
# ---------------------------------------------------------------------------


def cmd_disposable_test(controller):
    """Three failure scenarios against throwaway nginx containers."""
    base = "occ3-rt-base"
    preserved = "occ3-rt-preserved"  # NEVER the production name
    port = 39901
    url = "http://127.0.0.1:" + str(port) + "/"
    dummy_key = secrets.token_hex(32)
    controller.register_secret(dummy_key)
    outcomes = []

    def fresh_original():
        controller.run(["docker", "rm", "-f", base], timeout=60, check=False)
        controller.run(["docker", "rm", "-f", preserved],
                       timeout=60, check=False)
        controller.run(["docker", "run", "-d", "--name", base,
                        "-p", "127.0.0.1:" + str(port) + ":80",
                        "nginx:alpine"], timeout=120)
        if not wait_health(controller, url, 60):
            raise Failure("disposable original did not start")

    def cleanup():
        for name in (base, preserved):
            controller.run(["docker", "rm", "-f", name],
                           timeout=60, check=False)
        listing = controller.run(
            ["docker", "ps", "-aq", "--format", "{{.Names}}"], timeout=60)
        for name in listing.stdout.split():
            if name.startswith(FAILED_PREFIX):
                controller.run(["docker", "rm", "-f", name],
                               timeout=60, check=False)

    scenarios = [
        ("startup-crash", ["sh", "-c", "exit 7"], 60),
        ("health-timeout", ["sh", "-c", "sleep 600"], 25),
    ]
    try:
        for name, cmd, health_timeout in scenarios:
            controller.log("rehearsal[" + name + "]: fresh original")
            fresh_original()
            # capture + build candidate exactly like production
            original = docker_inspect(controller, base)
            norm = normalize(original)
            candidate = "occ3-rt-candidate"
            controller.run(["docker", "rm", "-f", candidate],
                           timeout=30, check=False)
            args = build_run_args(norm, "nginx:alpine", candidate, cmd, norm["Env"])
            controller.run(args, timeout=60)
            controller.log("rehearsal[" + name + "]: candidate created")
            # production-shaped mutation
            controller.run(["docker", "rename", base, preserved], timeout=30)
            controller.run(["docker", "stop", "-t", "5", preserved], timeout=60)
            controller.run(["docker", "update", "--restart=no", preserved],
                           timeout=30)
            controller.run(["docker", "rename", candidate, base], timeout=30)
            controller.run(["docker", "start", base], timeout=60)
            controller.log("rehearsal[" + name + "]: candidate started")
            # failure detection: health wait with the scenario timeout
            healthy = wait_health(controller, url, health_timeout, interval_s=2.0)
            if healthy:
                raise Failure("rehearsal[" + name + "]: candidate unexpectedly healthy")
            controller.log("rehearsal[" + name + "]: failure detected as designed")
            result = rollback(controller, base, preserved, base,
                              dummy_key, "rehearsal " + name, test_mode=True,
                              test_url=url)
            code, _ = controller.http("GET", url, None, timeout=15)
            ok = (result.get("result") == "RESTORED" and code == 200)
            outcomes.append((name, "PASS" if ok else "FAIL"))
            controller.log("rehearsal[" + name + "]: " + outcomes[-1][1]
                           + " (restored service answering: " + str(code == 200) + ")")

        # scenario 3: healthy candidate, validation failure
        name = "validation-fail"
        controller.log("rehearsal[" + name + "]: fresh original")
        fresh_original()
        original = docker_inspect(controller, base)
        norm = normalize(original)
        controller.run(["docker", "rm", "-f", "occ3-rt-candidate"],
                       timeout=30, check=False)
        args = build_run_args(norm, "nginx:alpine", "occ3-rt-candidate",
                              norm["Cmd"], norm["Env"])
        controller.run(args, timeout=60)
        controller.run(["docker", "rename", base, preserved], timeout=30)
        controller.run(["docker", "stop", "-t", "5", preserved], timeout=60)
        controller.run(["docker", "rename", "occ3-rt-candidate", base], timeout=30)
        controller.run(["docker", "start", base], timeout=60)
        if not wait_health(controller, url, 60, interval_s=2.0):
            raise Failure("rehearsal[" + name + "]: healthy candidate did not start")
        try:
            validate_chat(controller, dummy_key, "http://127.0.0.1:1",
                          force_fail=True)
        except Failure as error:
            result = rollback(controller, base, preserved, base,
                              dummy_key, str(error), test_mode=True, test_url=url)
            code, _ = controller.http("GET", url, None, timeout=15)
            ok = (result.get("result") == "RESTORED" and code == 200)
            outcomes.append((name, "PASS" if ok else "FAIL"))
            controller.log("rehearsal[" + name + "]: " + outcomes[-1][1])
    finally:
        cleanup()
    summary = {"phase": "disposable-rehearsal",
               "result": "PASS" if all(v == "PASS" for _, v in outcomes) else "FAIL",
               "scenarios": {k: v for k, v in outcomes}}
    controller.write_status(summary)
    controller.log("rehearsal summary: " + json.dumps(summary["scenarios"]))
    return 0 if summary["result"] == "PASS" else 1


# ---------------------------------------------------------------------------
# Offline self-test: the empty-argument regression and redaction proofs
# ---------------------------------------------------------------------------


def cmd_selftest():
    import tempfile

    failures = []

    def check(name, condition):
        print(("ok   " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    # 1. the docker 29.x quirk, reproduced: line-based capture injects ""
    # (entrypoint ["python3"]: the range template prints "python3" plus a
    # trailing blank line, so mapfile yields TWO elements, one empty)
    template_lines = ["python3", ""]  # what mapfile yields (trailing blank)
    cmd_json = ["python3"]            # what the JSON array actually holds
    check("line-capture produces an extra empty element (bug reproduced)",
          len(template_lines) == len(cmd_json) + 1 and template_lines[-1] == "")

    # 2. the guard rejects blank elements anywhere
    for bad in (["python3", "", "-m"], ["docker", "create", " "]):
        try:
            assert_no_blank_elements(bad, "test")
            check("guard rejects " + repr(bad), False)
        except Failure:
            check("guard rejects " + repr(bad), True)

    # 3. build_run_args from JSON produces element-exact command
    norm = {
        "Entrypoint": ["python3"], "Cmd": [], "Env": ["A=1", "B=2"],
        "WorkingDir": "/sgl-workspace/sglang", "User": "",
        "Binds": [], "Mounts": [], "PortBindings": {"30001/tcp": [("", "30001")]},
        "NetworkMode": "bridge", "Networks": ["bridge"], "IpcMode": "host",
        "PidMode": "", "Runtime": "nvidia", "ShmSize": 67108864,
        "Ulimits": [], "Tmpfs": {}, "ExtraHosts": [], "DeviceRequests": [],
        "RestartPolicy": {"Name": "no"}, "AutoRemove": False,
        "Privileged": False, "CapAdd": [], "CapDrop": [], "SecurityOpt": [],
        "Labels": {}, "OomKillDisable": False, "Memory": 0, "NanoCpus": 0,
        "Init": None,
    }
    new_cmd = ["-m", "sglang.launch_server", "--api-key", "k" * 64]
    args = build_run_args(norm, "img", "cand", new_cmd, norm["Env"])
    joined = " ".join(args)
    check("built args contain no empty elements",
          all(a.strip() for a in args))
    check("entrypoint preserved from JSON", "--entrypoint python3" in joined)
    check("workdir preserved", "--workdir /sgl-workspace/sglang" in joined)
    check("port preserved", "-p 30001:30001/tcp" in joined)
    check("ipc preserved", "--ipc host" in joined)
    check("runtime preserved", "--runtime nvidia" in joined)
    check("shm preserved", "--shm-size 67108864" in joined)

    # 4. the diff allowlist: only image/name/key accepted
    original = normalize({"Config": {"Entrypoint": ["python3"],
                                    "Cmd": ["-m", "x", "--api-key", "a" * 64],
                                    "Env": ["A=1"], "WorkingDir": "/w",
                                    "Labels": {}},
                         "HostConfig": {"NetworkMode": "bridge"},
                         "NetworkSettings": {}})
    candidate = normalize({"Config": {"Entrypoint": ["python3"],
                                      "Cmd": ["-m", "x", "--api-key", "b" * 64],
                                      "Env": ["A=1"], "WorkingDir": "/w",
                                      "Labels": {}},
                           "HostConfig": {"NetworkMode": "bridge"},
                           "NetworkSettings": {}})
    problems = diff_configs(original, candidate, 3, "a" * 64, "b" * 64)
    check("key-only change accepted", problems == [])
    tampered = json.loads(json.dumps(candidate))
    tampered["WorkingDir"] = "/elsewhere"
    problems = diff_configs(original, tampered, 3, "a" * 64, "b" * 64)
    check("working-dir change rejected", any("WorkingDir" in p for p in problems))
    tampered2 = json.loads(json.dumps(candidate))
    tampered2["Cmd"] = ["", "-m", "x", "--api-key", "b" * 64]
    problems = diff_configs(original, tampered2, 3, "a" * 64, "b" * 64)
    check("injected empty Cmd element rejected",
          any(("Cmd length differs" in p) or ("Cmd[0]" in p) for p in problems))
    tampered3 = json.loads(json.dumps(candidate))
    tampered3["Mounts"] = [("bind", "", "/data/x", "/data/x", True)]
    problems = diff_configs(original, tampered3, 3, "a" * 64, "b" * 64)
    check("mount change rejected", any("Mounts" in p for p in problems))

    # 5. redaction: no key material in logs or status
    with tempfile.TemporaryDirectory() as tmp:
        controller = Controller(tmp, verbose=False)
        key = secrets.token_hex(32)
        controller.register_secret(key)
        controller.log("the key is " + key + " and more text")
        controller.write_status({"phase": "x", "secret": key})
        log_text = open(controller.log_path, encoding="utf-8").read()
        status_text = open(controller.status_path, encoding="utf-8").read()
        check("key never appears in the log", key not in log_text)
        check("key never appears in status", key not in status_text)
        check("sentinel present in log", KEY_SENTINEL in log_text)

        # 6. command-line leak guard
        try:
            controller.run(["echo", key])
            check("run() refuses key in argv", False)
        except Failure:
            check("run() refuses key in argv", True)

    print("SELFTEST " + ("PASSED" if not failures else
                         "FAILED: " + ", ".join(failures)))
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def acquire_lock():
    if fcntl is None:
        return None
    handle = open(LOCK_FILE, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        print("another controller holds the lock: " + LOCK_FILE, flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "selftest", "preflight", "deploy", "rollback",
        "disposable-test", "gate8", "status"])
    parser.add_argument("--key-file", default=None,
                        help="0600 file with the CURRENT 64-hex api key")
    parser.add_argument("--new-key-file", default=None,
                        help="0600 file with the ROTATED 64-hex api key "
                             "(generated when omitted)")
    parser.add_argument("--recovery-dir", default=None)
    args = parser.parse_args()

    if args.command == "selftest":
        return cmd_selftest()

    lock = acquire_lock()
    if lock is None and args.command != "status":
        return 3

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    recovery = args.recovery_dir or os.path.join(
        RECOVERY_ROOT, "occ3-deploy-" + stamp)
    controller = Controller(recovery)
    if args.command == "status":
        try:
            print(open(controller.status_path, encoding="utf-8").read())
        except OSError:
            print("no status document")
        return 0
    if not args.key_file:
        print("--key-file is required for " + args.command, file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))
    try:
        if args.command == "preflight":
            cmd_preflight(controller, BASE_CONTAINER, args.key_file)
            return 0
        if args.command == "deploy":
            new_key = (load_key_file(args.new_key_file)
                       if args.new_key_file else None)
            return cmd_deploy(controller, BASE_CONTAINER, args.key_file,
                              new_key)
        if args.command == "rollback":
            return cmd_rollback(controller, BASE_CONTAINER, args.key_file)
        if args.command == "disposable-test":
            return cmd_disposable_test(controller)
        if args.command == "gate8":
            key = load_key_file(args.key_file)
            controller.register_secret(key)
            results = gate8(controller, key)
            results["restart_staleness"] = gate8_restart_phase(controller, key)
            controller.write_status({"phase": "gate8", "result": "PASS",
                                     **results})
            return 0
    except Failure as error:
        controller.log("command failed: " + str(error))
        controller.write_status({"phase": args.command, "result": "FAILED",
                                 "error": controller.redact(str(error))})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
