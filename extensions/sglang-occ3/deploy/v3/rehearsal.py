"""Real disposable Docker fault rehearsals using the production controller/guardian.

No production name, GPU, model mount, key or network is used. Fixture replies
prove recovery mechanics only, never GLM inference or actual ingress/Gate8.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import time

from .docker_api import DockerAPI, container_id, image_id
from .lifecycle import (Failure, Journal, REHEARSAL_LABEL, Services, checked_inspect,
                        is_running, prepare, private_json, process_identity, read_json,
                        require_running, wait_ready)
from .supervisor import launch

FIXTURE_KEY = "occ3-disposable-fixture-not-a-serving-credential"
DECLINED_FIXTURE_CODE = "unsupported_profile"
FIXTURE = r"""
import argparse, json, re, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
parser = argparse.ArgumentParser()
parser.add_argument('--api-key', required=True)
key = parser.parse_args().api_key
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def answer(self, status, value=None):
        data = json.dumps(value).encode() if value is not None else b''
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def authorized(self):
        return self.headers.get('Authorization') == 'Bearer ' + key
    def do_GET(self):
        if self.path == '/health':
            return self.answer(HEALTH_STATUS)
        if not self.authorized():
            return self.answer(401, {'error': 'unauthorized'})
        if self.path == '/occ3/v1/status':
            if OCC3_DECLINED:
                return self.answer(200, {'schema': 'optimus.occ3.status.v1', 'state': 'DECLINED',
                                         'code': OCC3_DECLINE_CODE, 'stage': 'install', 'attempt': 'deadbeef',
                                         'experimental_selected': True, 'preflight_reached': True,
                                         'backend_constructed': True, 'worker_readiness_reached': False,
                                         'hooks_prepared': True, 'route_mount_attempted': False,
                                         'rollback_completed': True, 'plugin': 'occ3-bridge',
                                         'version': '0.1.0', 'enabled': False, 'model_fingerprint': None,
                                         'channels': {}, 'metrics': {}})
            return self.answer(200, {'schema': 'optimus.occ3.status.v1', 'state': 'NATIVE_READY',
                                     'code': 'installed_experimental_not_live_validated'})
        return self.answer(200, {'data': [{'id': 'fixture-model'}]})
    def do_POST(self):
        if not self.authorized():
            return self.answer(401, {'error': 'unauthorized'})
        if self.path == '/occ3/v1/native/runtime/identity':
            if OCC3_IDENTITY_ABSENT:
                return self.answer(404, {'detail': 'Not Found'})
            if OCC3_DECLINED:
                return self.answer(503, {'error': {'code': 'not_ready'}})
            return self.answer(200, {'schema': 'occ3.runtime.identity.v1', 'runtime_id': 'fixture-runtime',
                                     'runtime_generation': 1, 'model_fingerprint': 'sha256:fixture',
                                     'expected_ranks': 8, 'ready_ranks': 8, 'source_capture': True,
                                     'continuous_prefix': True, 'terminal_journal': True})
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        text = body['messages'][-1]['content']
        nonce = re.search(r'OCC3_(?:READY|TOOL)_[a-f0-9]+', text).group(0)
        message = {'role': 'assistant', 'content': nonce}
        finish = 'stop'
        if body.get('tools'):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call-fixture', 'type': 'function', 'function': {
                    'name': 'occ3_echo', 'arguments': json.dumps({'nonce': nonce})}}]}
            finish = 'tool_calls'
        self.answer(200, {'id': 'fixture-' + uuid.uuid4().hex, 'model': body['model'],
                         'choices': [{'message': message, 'finish_reason': finish}],
                         'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
HTTPServer(('0.0.0.0', 8081), Handler).serve_forever()
"""

SCENARIOS = (
    {"name": "startup-exit", "image": "exit"},
    {"name": "running-unready", "image": "unready"},
    {"name": "controller-kill-before", "image": "healthy", "pause": "before_mutation", "signal": "kill"},
    {"name": "controller-kill-after-rename", "image": "healthy", "pause": "after_rename", "signal": "kill"},
    {"name": "controller-kill-after-start", "image": "healthy", "pause": "after_start", "signal": "kill"},
    {"name": "controller-hang-after-start", "image": "healthy", "pause": "after_start", "signal": "stop"},
    {"name": "supervisor-kill-before", "image": "healthy", "pause": "before_mutation", "signal": "supervisor-kill"},
    {"name": "supervisor-kill-after-start", "image": "healthy", "pause": "after_start", "signal": "supervisor-kill"},
    {"name": "occ3-identity-absent", "image": "identity-absent"},
    {"name": "occ3-declined", "image": "identity-declined"},
)
# One optional capability scenario, appended only when a validated calibrated
# input was supplied on the command line. Exit fixture only: the run must never
# reach an identity/admission validation phase.
CALIBRATED_SCENARIO = {"name": "calibrated-bind-startup-exit", "image": "exit", "calibrated": True}

# The pre-mutation admission receipt every rehearsal plan journals before any
# mutation: lifecycle.Services.pre_mutation_admission returns exactly this for
# rehearsal mode (a different shape here means a production-mode receipt, which
# must never appear in a rehearsal journal).
CALIBRATED_ADMISSION_RECEIPT = {"ok": True, "scope": "disposable_faults_only_not_gate8"}

# Post-start validation/acceptance phases the exit-fixture scenario can never
# legitimately reach; a recovery phase such as RESTORED is legitimate and is
# never rejected on its own.
CALIBRATED_FORBIDDEN_PHASES = frozenset({"WAIT_NATIVE_INITIAL", "WAIT_NATIVE_FINAL",
                                          "RESTART_REQUESTED", "POST_RESTART",
                                          "GATES_COMPLETE", "COMPLETE"})

# Durable journal files that survive any later phase overwrite: the validation
# checkpoints and external native proof receipts (controller checkpoint()),
# and the validation restart permit (supervisor permit_restart).
CALIBRATED_FORBIDDEN_FILES = ("checkpoint-initial.json", "checkpoint-final.json",
                              "native-proof-initial.json", "native-proof-final.json",
                              "restart-permit.json")


def rehearsal_scenarios(calibrated_input):
    """The scenario list: today's ten, plus the capability scenario only when
    a validated calibrated input is present. Names stay unique and the
    capability scenario is restricted to the exit fixture."""
    scenarios = list(SCENARIOS)
    if calibrated_input is None:
        return scenarios
    if CALIBRATED_SCENARIO["image"] != "exit":
        raise Failure("calibrated rehearsal requires the exit fixture")
    names = [scenario["name"] for scenario in scenarios] + [CALIBRATED_SCENARIO["name"]]
    if len(set(names)) != len(names):
        raise Failure("rehearsal scenario names must be unique")
    return scenarios + [CALIBRATED_SCENARIO]


def load_calibrated_input(path):
    """Bounded exact-field parse of the optional calibrated capability input.

    The document carries exactly calibrated_artifact and
    calibrated_profile_evidence; the two plan pin fields are derived from the
    provided profile sha256 (never campaign constants) and the whole capability
    is validated through the committed calibrated_capability schema before any
    scenario runs. The input never overrides any other plan field.
    """
    from .docker_config import calibrated_capability
    try:
        document = read_json(path)
    except OSError:
        # Missing or unreadable input: a sanitized refusal, never file content.
        raise Failure("calibrated input unreadable") from None
    except json.JSONDecodeError:
        raise Failure("invalid calibrated input document") from None
    if not isinstance(document, dict) or set(document) != {"calibrated_artifact",
                                                            "calibrated_profile_evidence"}:
        raise Failure("invalid calibrated input document")
    evidence = document["calibrated_profile_evidence"]
    if not isinstance(evidence, dict):
        raise Failure("invalid calibrated input document")
    sha256 = evidence.get("sha256")
    fragment = dict(document)
    fragment["static_input"] = {"sha256": sha256}
    fragment["native_validator"] = {"static_input_sha256": sha256}
    try:
        return calibrated_capability(fragment)
    except ValueError:
        raise Failure("invalid calibrated input capability") from None


def calibrated_plan_keys(calibrated_input):
    """The four capability plan fields from a validated input, or none."""
    if calibrated_input is None:
        return {}
    return {"calibrated_artifact": {"sha256": calibrated_input["sha256"],
                                     "size": calibrated_input["size"]},
            "calibrated_profile_evidence": {"path": calibrated_input["profile_path"],
                                            "sha256": calibrated_input["profile_sha256"],
                                            "bytes": calibrated_input["profile_bytes"]},
            "static_input": {"sha256": calibrated_input["profile_sha256"]},
            "native_validator": {"static_input_sha256": calibrated_input["profile_sha256"]}}


def fixture_environment(values):
    """Parse a bounded plain-Python image environment; no executable hooks."""
    allowed = {"PATH", "LANG", "LC_ALL", "PYTHON_VERSION", "PYTHON_SHA256", "GPG_KEY",
               "PYTHON_PIP_VERSION", "PYTHON_SETUPTOOLS_VERSION", "PYTHON_GET_PIP_URL", "PYTHON_GET_PIP_SHA256"}
    if not isinstance(values, list) or len(values) > 32:
        raise Failure("invalid inherited fixture environment")
    result = {}
    for item in values:
        if not isinstance(item, str) or "=" not in item or "\x00" in item or len(item) > 8192:
            raise Failure("unsafe inherited fixture environment")
        key, value = item.split("=", 1)
        if key not in allowed or key in result:
            raise Failure("unsafe inherited fixture environment")
        result[key] = value
    return result


def assert_fixture_image(document, expected_image):
    config = document.get("Config", {})
    if (document.get("Id") != expected_image or document.get("Os") != "linux" or
            config.get("OnBuild") or config.get("Volumes") or config.get("Healthcheck") or
            config.get("Entrypoint") not in (None, []) or config.get("Cmd") not in (["python3"], ["python"]) or
            config.get("User") not in (None, "")):
        raise Failure("fixture requires a vetted plain Linux Python image profile")
    fixture_environment(config.get("Env", []))
    if any("nvidia" in str(key).lower() or "cuda" in str(key).lower() for key in (config.get("Labels") or {})):
        raise Failure("GPU image is not a disposable CPU fixture base")


def assert_fixture_policy(document, owner, expected_image, *, port, image_document,
                          resource_nonce, unstarted=True):
    """Independent literal role policy, never a rebuild of the create request.

    Engine null/absent/empty collections are equivalent only for empty fields.
    The two spellings of no-new-privileges are the only security normalization.
    Dynamic endpoint IDs may be empty before start. No static address is allowed.
    """
    from .lifecycle import assert_rehearsal_original
    from .docker_api import container_id
    assert_fixture_image(image_document, expected_image)
    container_id(document.get("Id"))
    if (not re.fullmatch(r"[a-f0-9]{16}", owner) or not 1024 <= port <= 65535 or
            not re.fullmatch(r"[a-f0-9]{32}", resource_nonce)):
        raise Failure("invalid disposable identity or port")
    host, config, state = document.get("HostConfig", {}), document.get("Config", {}), document.get("State", {})
    config_fields = set("""
        Hostname Domainname User AttachStdin AttachStdout AttachStderr ExposedPorts Tty
        OpenStdin StdinOnce Env Cmd Healthcheck ArgsEscaped Image Volumes WorkingDir
        Entrypoint NetworkDisabled MacAddress OnBuild Labels StopSignal StopTimeout Shell
    """.split())
    host_fields = set("""
        Binds ContainerIDFile LogConfig NetworkMode PortBindings RestartPolicy AutoRemove VolumeDriver
        VolumesFrom ConsoleSize CapAdd CapDrop CgroupnsMode Dns DnsOptions DnsSearch ExtraHosts GroupAdd
        IpcMode Cgroup Links OomScoreAdj PidMode Privileged PublishAllPorts ReadonlyRootfs SecurityOpt
        UTSMode UsernsMode ShmSize Runtime Isolation CpuShares Memory NanoCpus CgroupParent BlkioWeight
        BlkioWeightDevice BlkioDeviceReadBps BlkioDeviceWriteBps BlkioDeviceReadIOps BlkioDeviceWriteIOps
        CpuPeriod CpuQuota CpuRealtimePeriod CpuRealtimeRuntime CpusetCpus CpusetMems Devices
        DeviceCgroupRules DeviceRequests MemoryReservation MemorySwap MemorySwappiness OomKillDisable
        PidsLimit Ulimits CpuCount CpuPercent IOMaximumIOps IOMaximumBandwidth Mounts MaskedPaths
        ReadonlyPaths Tmpfs Init StorageOpt Sysctls KernelMemory KernelMemoryTCP Annotations
    """.split())
    if set(config) - config_fields or set(host) - host_fields:
        raise Failure("unsupported actual fixture inspect field")
    if (config.get("MacAddress") or config.get("ArgsEscaped") or
            config.get("StopSignal") not in (None, "", "SIGTERM") or
            config.get("StopTimeout") not in (None, 1)):
        raise Failure("unexpected actual fixture process settings")
    for field in ("CpuShares", "Memory", "NanoCpus", "BlkioWeight", "BlkioWeightDevice", "BlkioDeviceReadBps",
                  "BlkioDeviceWriteBps", "BlkioDeviceReadIOps", "BlkioDeviceWriteIOps", "CpuPeriod", "CpuQuota",
                  "CpuRealtimePeriod", "CpuRealtimeRuntime", "CpusetCpus", "CpusetMems", "MemoryReservation",
                  "MemorySwap", "OomKillDisable", "Ulimits", "CpuCount", "CpuPercent", "IOMaximumIOps",
                  "IOMaximumBandwidth", "KernelMemory", "KernelMemoryTCP", "OomScoreAdj"):
        if host.get(field):
            raise Failure("unexpected actual fixture resource settings")
    if (host.get("PidsLimit") not in (None, 0, -1) or host.get("MemorySwappiness") not in (None, -1) or
            host.get("ShmSize") not in (None, 67108864) or host.get("ConsoleSize") not in (None, [], [0, 0])):
        raise Failure("unexpected actual fixture resource defaults")
    # Require the Engine's standard read-only/masked proc protections when exposed.
    # Additional masks/restrictions do not increase fixture access.
    for field, required in (("MaskedPaths", {"/proc/kcore", "/proc/keys", "/sys/firmware"}),
                            ("ReadonlyPaths", {"/proc/sys", "/proc/sysrq-trigger"})):
        paths = host.get(field)
        if paths is not None and (not isinstance(paths, list) or
                any(not isinstance(path, str) or not path.startswith("/") for path in paths) or
                not required <= set(paths)):
            raise Failure("actual fixture proc protections weakened")
    if (document.get("Image") != expected_image or document.get("Name") != "/occ3-v3-rehearsal-" + owner + "-original" or
            (unstarted and (state.get("Status") != "created" or state.get("Running") is not False or state.get("Restarting"))) or
            config.get("Image") != expected_image or config.get("Entrypoint") != ["python3", "-I", "-S", "/fixture.py"] or
            config.get("Cmd") != ["--api-key", FIXTURE_KEY] or config.get("User") != "65534:65534" or
            config.get("WorkingDir") != "/" or config.get("Hostname") != "occ3-v3-rehearsal-" + owner or
            config.get("Domainname") not in (None, "") or config.get("ExposedPorts") != {"8081/tcp": {}} or
            config.get("Volumes") or config.get("OnBuild") or config.get("Shell") or
            any(config.get(field) for field in ("Tty", "OpenStdin", "StdinOnce", "NetworkDisabled", "AttachStdin")) or
            config.get("Healthcheck", {}).get("Test") != ["NONE"]):
        raise Failure("actual disposable fixture config is outside CPU role policy")
    if any(value for key, value in config.get("Healthcheck", {}).items() if key != "Test"):
        raise Failure("unexpected actual fixture healthcheck settings")
    assert_rehearsal_original(document, {"rehearsal_owner": owner})
    expected_env = fixture_environment(image_document.get("Config", {}).get("Env", []))
    expected_env.update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="-1")
    actual_env = config.get("Env")
    if not isinstance(actual_env, list) or len(actual_env) != len(expected_env):
        raise Failure("actual fixture environment differs")
    parsed = {}
    for item in actual_env:
        if not isinstance(item, str) or "=" not in item or "\x00" in item:
            raise Failure("actual fixture environment differs")
        key, value = item.split("=", 1)
        if key in parsed:
            raise Failure("duplicate actual fixture environment")
        parsed[key] = value
    if parsed != expected_env:
        raise Failure("actual fixture environment differs")
    labels = dict(image_document.get("Config", {}).get("Labels") or {})
    labels.update({REHEARSAL_LABEL: owner, "occ3.v3.resource-nonce": resource_nonce})
    if config.get("Labels") != labels:
        raise Failure("actual fixture labels differ")
    if (document.get("Mounts") != [] or host.get("Runtime") != "runc" or host.get("NetworkMode") != "bridge" or
            host.get("PortBindings") != {"8081/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]} or
            host.get("ReadonlyRootfs") is not True or host.get("CapDrop") != ["ALL"] or
            host.get("SecurityOpt") not in (["no-new-privileges"], ["no-new-privileges:true"]) or
            host.get("IpcMode") != "private" or host.get("CgroupnsMode") != "private" or
            host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0} or
            host.get("LogConfig") != {"Type": "json-file", "Config": {}} or
            any(host.get(field) for field in ("Privileged", "AutoRemove", "PublishAllPorts", "Init"))):
        raise Failure("actual fixture host isolation differs")
    # No host namespace, supplementary group, custom DNS, device or mount access.
    for field in ("Binds", "Mounts", "VolumesFrom", "Tmpfs", "Devices", "DeviceRequests", "DeviceCgroupRules",
                  "CapAdd", "GroupAdd", "Dns", "DnsOptions", "DnsSearch", "ExtraHosts", "Links", "Sysctls",
                  "StorageOpt", "Annotations", "PidMode", "UTSMode", "UsernsMode", "Cgroup", "CgroupParent",
                  "ContainerIDFile", "VolumeDriver", "Isolation"):
        if host.get(field):
            raise Failure("unexpected actual fixture host access")
    settings = document.get("NetworkSettings", {})
    ports = settings.get("Ports")
    inactive_ports = (None, {}, {"8081/tcp": None}, {"8081/tcp": []})
    active_ports = {"8081/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}
    if (ports not in inactive_ports and (unstarted or ports != active_ports)) or (unstarted and settings.get("IPAddress")):
        raise Failure("unexpected actual fixture published endpoint")
    networks = settings.get("Networks")
    if not isinstance(networks, dict) or set(networks) != {"bridge"}:
        raise Failure("unexpected actual fixture network membership")
    endpoint = networks["bridge"]
    if not isinstance(endpoint, dict):
        raise Failure("invalid actual fixture network endpoint")
    for field in ("IPAMConfig", "Aliases", "Links", "DriverOpts", "GwPriority"):
        if endpoint.get(field):
            raise Failure("unexpected actual fixture endpoint settings")
    if unstarted and (endpoint.get("IPAddress") or endpoint.get("GlobalIPv6Address")):
        raise Failure("unexpected unstarted fixture endpoint address")
    if endpoint.get("NetworkID") not in (None, ""):
        container_id(endpoint["NetworkID"])


def assert_unstarted_fixture(document, owner, expected_image, *, port, image_document, resource_nonce):
    return assert_fixture_policy(document, owner, expected_image, port=port,
        image_document=image_document, resource_nonce=resource_nonce, unstarted=True)


def fixture_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture_configuration(document):
    # Full stable inspect config, not state or dynamically assigned IPs.
    return {key: document.get(key) for key in ("Image", "Config", "HostConfig", "Mounts")}


def fixture_request_projection(document, request):
    config = {key: document["Config"].get(key) for key in request if key != "HostConfig"}
    # Docker may append inherited image Env after explicit overrides. The literal
    # fixture policy already rejects duplicates and checks every key/value.
    config["Env"] = dict(item.split("=", 1) for item in config["Env"])
    host = {key: document["HostConfig"].get(key) for key in request["HostConfig"]}
    # The independent role policy already requires disabled healthcheck and
    # empty device lists; these are Docker's documented empty encodings.
    config["Healthcheck"] = {"Test": config["Healthcheck"]["Test"]}
    for field in ("Devices", "DeviceRequests"):
        if host.get(field) is None:
            host[field] = []
    if host.get("SecurityOpt") == ["no-new-privileges:true"]:
        host["SecurityOpt"] = ["no-new-privileges"]
    return {"Config": config, "HostConfig": host, "Mounts": document.get("Mounts")}


def reconcile_original_fixture(api, ledger_path):
    """One read-only exact lookup for cleanup identity, never resume/start/recreate."""
    ledger = read_json(ledger_path)
    try:
        if (ledger.get("schema") != "occ3-basic-fixture/v1" or
                not re.fullmatch(r"[a-f0-9]{16}", ledger.get("owner", "")) or
                not re.fullmatch(r"[a-f0-9]{32}", ledger.get("resource_nonce", "")) or
                ledger.get("name") != "occ3-v3-rehearsal-" + ledger["owner"] + "-original" or
                fixture_digest(ledger.get("request")) != ledger.get("request_sha256")):
            raise Failure("invalid exact fixture ledger")
        image = api.inspect_image(image_id(ledger["image_id"]))
        assert_fixture_image(image, ledger["image_id"])
        if fixture_digest(image.get("Config", {})) != ledger["image_configuration_sha256"]:
            raise Failure("fixture image configuration changed")
        identity = ledger.get("container_id")
        if identity:
            container_id(identity)
        document = api.inspect(identity or ledger["name"])
        actual_id = container_id(document.get("Id"))
        if identity and actual_id != identity:
            raise Failure("immutable fixture response ID mismatch")
        assert_fixture_policy(document, ledger["owner"], ledger["image_id"], port=ledger["port_contract"]["host_port"],
            image_document=image, resource_nonce=ledger["resource_nonce"], unstarted=not ledger.get("start_intent"))
        if fixture_digest(fixture_request_projection(document, ledger["request"])) != ledger["expected_configuration_sha256"]:
            raise Failure("fixture differs from complete planned request")
        digest = fixture_digest(fixture_configuration(document))
        if ledger.get("actual_configuration_sha256") not in (None, digest):
            raise Failure("actual fixture configuration changed")
        ledger.update(container_id=actual_id, actual_configuration_sha256=digest,
            reconciliation="verified_identity_for_cleanup_only", cleanup_admissible=True,
            observed_running=document.get("State", {}).get("Running") is True)
    except Exception as error:
        ledger.update(reconciliation="unresolved", cleanup_admissible=False, reconciliation_error=type(error).__name__)
    private_json(ledger_path, ledger)
    return ledger


def create_original_fixture(api, image, owner, port, name, *, ledger_path):
    # Durable owner/name/config precedes create; verified full ID precedes start.
    if not re.fullmatch(r"[a-f0-9]{16}", owner) or name != "occ3-v3-rehearsal-" + owner + "-original":
        raise Failure("invalid exact basic fixture name")
    image_document = api.inspect_image(image)
    assert_fixture_image(image_document, image)
    nonce = secrets.token_hex(16)
    request = original_request(image, owner, port, resource_nonce=nonce)
    expected = deepcopy(request)
    expected_env = fixture_environment(image_document.get("Config", {}).get("Env", []))
    expected_env.update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="-1")
    expected["Env"] = expected_env
    expected["Labels"] = {**(image_document.get("Config", {}).get("Labels") or {}), **request["Labels"]}
    projection = {"Config": {key: value for key, value in expected.items() if key != "HostConfig"},
                  "HostConfig": expected["HostConfig"], "Mounts": []}
    ledger = {"schema": "occ3-basic-fixture/v1", "owner": owner, "resource_nonce": nonce,
        "name": name, "image_id": image_id(image), "container_id": None, "request": request,
        "request_sha256": fixture_digest(request), "expected_configuration_sha256": fixture_digest(projection),
        "image_configuration_sha256": fixture_digest(image_document.get("Config", {})),
        "network_contract": {"mode": "bridge", "extra_endpoints": False},
        "port_contract": {"host_ip": "127.0.0.1", "host_port": port, "container_port": "8081/tcp"},
        "phase": "CREATE_INTENT", "start_intent": False, "cleanup_admissible": False,
        "cleanup_complete": False, "result": "PENDING"}
    private_json(ledger_path, ledger, exclusive=True)
    try:
        identity = container_id(api.create(request, name))
        # Returned ID is evidence, not yet permission to start or clean.
        ledger.update(returned_id=identity, phase="VERIFY_CREATED")
        private_json(ledger_path, ledger)
        document = checked_inspect(api, identity, image)
        assert_unstarted_fixture(document, owner, image, port=port, image_document=image_document, resource_nonce=nonce)
        if fixture_digest(fixture_request_projection(document, request)) != ledger["expected_configuration_sha256"]:
            raise Failure("actual fixture differs from planned complete request")
        ledger.update(container_id=identity, actual_configuration_sha256=fixture_digest(fixture_configuration(document)),
                      phase="START_INTENT", start_intent=True, cleanup_admissible=True)
        private_json(ledger_path, ledger)
        api.start(identity)
        ledger.update(phase="START_ACKNOWLEDGED", result="STARTED_NOT_SCENARIO_PASS")
        private_json(ledger_path, ledger)
        return identity
    except BaseException as error:
        ledger.update(result="FAIL", failure=type(error).__name__, uncertain=True, cleanup_admissible=False)
        with contextlib.suppress(Exception):
            private_json(ledger_path, ledger)
            reconcile_original_fixture(api, ledger_path)
        raise


def cleanup_original_fixture(api, ledger_path):
    """Explicit cleanup only, with a fresh complete read-only identity validation."""
    ledger = reconcile_original_fixture(api, ledger_path)
    if not ledger.get("cleanup_admissible") or ledger.get("reconciliation") != "verified_identity_for_cleanup_only":
        raise Failure("unresolved fixture ownership blocks cleanup")
    identity = container_id(ledger["container_id"])
    ledger["phase"] = "CLEANUP_INTENT"
    private_json(ledger_path, ledger)
    if ledger["observed_running"]:
        api.stop(identity, seconds=1)
    api.remove(identity)
    ledger.update(phase="REMOVED", cleanup_complete=True, cleanup_admissible=False)
    private_json(ledger_path, ledger)
    return {"container_id": identity, "cleanup_complete": True}


def signal_owned_process(pid, expected_identity, signum):
    # A pidfd pins the process across PID reuse; a check followed by os.kill does not.
    if (not isinstance(expected_identity, dict) or expected_identity.get("pid") != pid or
            not expected_identity.get("start_ticks") or not hasattr(os, "pidfd_open") or
            not hasattr(signal, "pidfd_send_signal")):
        raise Failure("safe pidfd fault injection is unavailable")
    descriptor = os.pidfd_open(pid, 0)
    try:
        if process_identity(pid) != expected_identity:
            raise Failure("fault victim process identity mismatch")
        signal.pidfd_send_signal(descriptor, signum, None, 0)
    finally:
        os.close(descriptor)


def assert_scenario_evidence(scenario, state, recovery, controller, candidate, injected):
    if recovery.get("verified") is not True:
        raise Failure("disposable recovery was not verified")
    before = scenario.get("pause") == "before_mutation"
    expected = "NOOP" if before else "RESTORED"
    if recovery.get("result") != expected:
        raise Failure("recovery did not exercise the requested failure stage")
    if scenario.get("signal") and not injected:
        raise Failure("fault was not actually injected")
    operations = [item["operation"] for item in state.get("history", [])]
    if before:
        if state.get("mutation_intent") or "rename_original" in operations:
            raise Failure("before-mutation fault occurred after mutation")
    elif not state.get("mutation_intent"):
        raise Failure("requested post-mutation fault never mutated the fixture")
    if scenario["image"] in ("exit", "unready"):
        if "start_candidate" not in operations or not state.get("original_stopped"):
            raise Failure("startup fault never started the intended candidate")
    if scenario["image"] == "exit" and candidate.get("State", {}).get("ExitCode") != 7:
        raise Failure("startup-exit fixture did not exit with 7")
    if scenario["image"] == "unready":
        observations = state.get("readiness_observations", {})
        if (not state.get("candidate_started") or not state.get("candidate_epoch") or
                controller.get("error") != "RunningUnready" or observations.get("checks", 0) <= 0 or
                observations.get("last_http_status") != 503 or observations.get("last_ok") is not False or
                controller.get("heartbeat", 0) - observations.get("started_monotonic", 0) < state["plan"]["startup_timeout_s"]):
            raise Failure("running-unready HTTP503 and readiness deadline were not exercised")
    if scenario["image"] == "identity-declined":
        decline = state.get("occ3_decline") or {}
        evidence = decline.get("evidence") or {}
        if (decline.get("gate") != "service_initial" or decline.get("field") != "occ3_identity" or
                evidence.get("error_code") != "occ3_declined" or
                evidence.get("declined_code") != DECLINED_FIXTURE_CODE or
                evidence.get("stage") != "install" or
                evidence.get("rollback_completed") is not True or
                state.get("restart_count") != 0 or "service_initial" in state.get("gates", {})):
            raise Failure("declined candidate did not fail fast with its exact decline code")
        return {"expected_recovery": expected, "failure_stage_proven": True,
                "declined_code": evidence.get("declined_code")}
    return {"expected_recovery": expected, "failure_stage_proven": True}


def build_fixtures(api, base_image, owner, *, ledger_dir):
    if not re.fullmatch(r"[a-f0-9]{16}", owner):
        raise Failure("invalid fixture build owner")
    image_id(base_image)
    assert_fixture_image(api.inspect_image(base_image), base_image)
    dockerfile = ("FROM " + base_image + "\nCOPY fixture.py /fixture.py\n").encode()
    scripts = {
        "original": "OCC3_IDENTITY_ABSENT = False\nOCC3_DECLINED = False\nHEALTH_STATUS = 200\n" + FIXTURE,
        "healthy": "OCC3_IDENTITY_ABSENT = False\nOCC3_DECLINED = False\n# distinct candidate fixture image\nHEALTH_STATUS = 200\n" + FIXTURE,
        "exit": "OCC3_IDENTITY_ABSENT = False\nraise SystemExit(7)\n",
        "unready": "OCC3_IDENTITY_ABSENT = False\nOCC3_DECLINED = False\nHEALTH_STATUS = 503\n" + FIXTURE,
        "identity-absent": "OCC3_IDENTITY_ABSENT = True\nOCC3_DECLINED = False\nHEALTH_STATUS = 200\n" + FIXTURE,
        "identity-declined": ("OCC3_IDENTITY_ABSENT = False\nOCC3_DECLINED = True\nOCC3_DECLINE_CODE = "
                              + repr(DECLINED_FIXTURE_CODE) + "\nHEALTH_STATUS = 200\n" + FIXTURE),
    }
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(mode=0o700, exist_ok=False)
    images = {}
    for role, source in scripts.items():
        files = {"Dockerfile": dockerfile, "fixture.py": source.encode()}
        tag = "occ3-v3-rehearsal-" + owner + ":" + role
        ledger = {"schema": "occ3-basic-build/v1", "owner": owner, "resource_nonce": secrets.token_hex(16),
                  "role": role, "tag": tag, "base_image_id": base_image, "image_id": None,
                  "files_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in files.items()},
                  "build_policy": {"pull": False, "network": "none", "dockerfile_run_steps": False},
                  "phase": "BUILD_INTENT", "cleanup_complete": False, "artifacts_retained": True}
        path = ledger_dir / (role + ".json")
        private_json(path, ledger, exclusive=True)
        try:
            identity = image_id(api.build(files, tag))
            ledger.update(phase="BUILD_ACKNOWLEDGED", image_id=identity)
            private_json(path, ledger)
            images[role] = identity
        except BaseException as error:
            ledger.update(phase="BUILD_UNCERTAIN", failure=type(error).__name__, result="FAIL",
                          unresolved_backend_artifact=True)
            with contextlib.suppress(Exception):
                private_json(path, ledger)
            raise
    return images


def original_request(image, owner, port, *, resource_nonce=None):
    if not re.fullmatch(r"[a-f0-9]{16}", owner) or not 1024 <= port <= 65535:
        raise Failure("invalid disposable identity or port")
    resource_nonce = resource_nonce or owner + "0" * 16
    if not re.fullmatch(r"[a-f0-9]{32}", resource_nonce):
        raise Failure("invalid fixture resource nonce")
    return {"Image": image, "Entrypoint": ["python3", "-I", "-S", "/fixture.py"],
            "Cmd": ["--api-key", FIXTURE_KEY],
            "Env": ["NVIDIA_VISIBLE_DEVICES=void", "CUDA_VISIBLE_DEVICES=-1"],
            "Labels": {REHEARSAL_LABEL: owner, "occ3.v3.resource-nonce": resource_nonce}, "ExposedPorts": {"8081/tcp": {}},
            "User": "65534:65534", "WorkingDir": "/", "Hostname": "occ3-v3-rehearsal-" + owner,
            "Healthcheck": {"Test": ["NONE"]}, "Tty": False,
            "HostConfig": {"Runtime": "runc", "DeviceRequests": [], "Devices": [],
                           "NetworkMode": "bridge", "AutoRemove": False,
                           "ReadonlyRootfs": True, "Privileged": False, "CapDrop": ["ALL"],
                           "SecurityOpt": ["no-new-privileges"], "IpcMode": "private", "CgroupnsMode": "private",
                           "PidMode": "", "UTSMode": "", "UsernsMode": "", "PublishAllPorts": False,
                           "LogConfig": {"Type": "json-file", "Config": {}},
                           "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                           "PortBindings": {"8081/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}}}


def cleanup_owned(api, identities, owner):
    """Only exact created IDs with matching ownership labels, never a name/prefix scan."""
    for identity, expected_image in identities:
        document = checked_inspect(api, identity, expected_image)
        if document.get("Config", {}).get("Labels", {}).get(REHEARSAL_LABEL) != owner:
            raise Failure("refuse cleanup of an unowned container")
        if document.get("State", {}).get("Running"):
            api.stop(identity, seconds=1)
        api.remove(identity)


def persist_calibrated_echo(output, candidate):
    """Persist the sanitized capability echo before any assertion or cleanup.

    Explicit allowlist only: container identity, the create-time mount
    declaration (HostConfig.Mounts) and the resolved top-level Mounts records,
    each labeled by its source document, plus the bounded state fields.
    Never Config.Cmd, never Config.Env, never the whole Config or HostConfig
    objects. A post-write grep must find no credential material; a failure
    here raises into the except branch, which skips cleanup and keeps the
    container, so the evidence then survives twice.
    """
    state = candidate.get("State") or {}
    echo = {"schema": "occ3-calibrated-rehearsal-echo/v1",
            "source_document": "candidate-inspect",
            "container_id": candidate.get("Id"),
            "image_id": candidate.get("Image"),
            "name": candidate.get("Name"),
            "declared_hostconfig_mounts": deepcopy(candidate.get("HostConfig", {}).get("Mounts")),
            "resolved_actual_mounts": deepcopy(candidate.get("Mounts")),
            "state": {key: state.get(key) for key in ("Status", "ExitCode", "Running")}}
    path = Path(output) / "calibrated-candidate-echo.json"
    private_json(path, echo, exclusive=True)
    text = path.read_text(encoding="utf-8")
    if "--api-key" in text or FIXTURE_KEY in text:
        raise Failure("calibrated candidate echo is not sanitized")
    return path


def assert_calibrated_capability_evidence(state, original, current, candidate, plan,
                                           journal_directory):
    """Create/echo/restore evidence for the one calibrated bind; fail closed.

    The immutable journal binding must equal the pair re-derived from the
    verified staged profile; the candidate must carry exactly one read-only
    bind, in both the create-time declaration and the resolved record, with
    only the committed bounded daemon spellings; and the restored disposable
    original must carry no mounts.

    Admission versus validation, on durable journal evidence: the unconditional
    pre-mutation admission receipt must still be present with the rehearsal
    tooling scope (the supervisor journals it before the controller starts, and
    the controller journals it again before the first mutation), while any
    post-start validation or acceptance progression is refused - forbidden
    phases, journaled gates (every production gate is post-start), a restart
    request or consumed restart, a persistence record, and the checkpoint,
    native-proof and restart-permit evidence files. All of those outlive a
    later recovery-phase overwrite, so a final RESTORED phase cannot conceal
    progression; legitimate recovery states are not rejected, and no timing
    assumption is made about the STARTING phase.
    """
    from .docker_config import (_CALIBRATION_MOUNT_MODES, _CALIBRATION_MOUNT_PROPAGATION,
                                calibrated_capability, derive_calibrated_binding,
                                verify_calibrated_profile)
    capability = calibrated_capability(plan)
    if capability is None:
        raise Failure("calibrated scenario plan lost its capability")
    derived = derive_calibrated_binding(verify_calibrated_profile(capability), original)
    expected = {"calibration_source": derived["source"],
                "calibration_destination": derived["destination"],
                "calibration_basename": derived["basename"]}
    if any(state.get(key) != value for key, value in expected.items()):
        raise Failure("calibrated journal binding differs from the verified derived pair")
    if state.get("validation_admission") != CALIBRATED_ADMISSION_RECEIPT:
        raise Failure("calibrated scenario lost its pre-mutation admission receipt")
    declared = (candidate.get("HostConfig") or {}).get("Mounts")
    if (not isinstance(declared, list) or len(declared) != 1
            or declared[0] != {"Type": "bind", "Source": derived["source"],
                               "Target": derived["destination"], "ReadOnly": True}):
        raise Failure("calibrated candidate declaration is not the exact one read-only bind")
    records = candidate.get("Mounts")
    if not isinstance(records, list) or len(records) != 1:
        raise Failure("calibrated candidate resolved mounts are not exactly one")
    record = records[0]
    if (record.get("Type") != "bind" or record.get("Source") != derived["source"]
            or record.get("Destination") != derived["destination"]
            or record.get("RW") is not False
            or record.get("Mode", "") not in _CALIBRATION_MOUNT_MODES
            or record.get("Propagation", "") not in _CALIBRATION_MOUNT_PROPAGATION):
        raise Failure("calibrated candidate resolved record is not the bounded read-only bind")
    if current.get("Mounts") != []:
        raise Failure("restored disposable original carries mounts")
    if state.get("phase") in CALIBRATED_FORBIDDEN_PHASES:
        raise Failure("calibrated scenario progressed into a forbidden validation phase")
    if state.get("gates"):
        raise Failure("calibrated scenario journaled a validation gate")
    if state.get("restart_count") != 0 or "restart_request" in state:
        raise Failure("calibrated scenario requested or consumed a validation restart")
    if state.get("persistence") is not None:
        raise Failure("calibrated scenario journaled a persistence record")
    for name in CALIBRATED_FORBIDDEN_FILES:
        if (Path(journal_directory) / name).exists():
            raise Failure("calibrated scenario left validation evidence: " + name)
    return {"calibration_source": derived["source"],
            "calibration_destination": derived["destination"],
            "calibration_basename": derived["basename"],
            "declared_exactly_one_readonly_bind": True,
            "resolved_record_bounded": True,
            "restored_original_mounts_empty": True,
            "pre_mutation_admission_retained": True,
            "post_start_validation_absent": True}


def run_scenario(api, images, output, scenario, calibrated_input=None):
    if os.name != "posix":
        raise Failure("real Docker rehearsals require Linux")
    owner = secrets.token_hex(8)
    attempt_id = "rehearsal-" + owner
    name = "occ3-v3-rehearsal-" + owner + "-original"
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    identities, journal, fault_at = [], None, None
    result = {"scenario": scenario["name"], "result": "FAIL", "scope": "disposable_fixture_only"}
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        ledger_path = output / "original-fixture.json"
        result["original_fixture_ledger"] = str(ledger_path)
        original_id = create_original_fixture(api, images["original"], owner, port, name, ledger_path=ledger_path)
        identities.append((original_id, images["original"]))
        original = checked_inspect(api, original_id, images["original"])
        plan = {
            "attempt_id": attempt_id, "mode": "rehearsal", "rehearsal_owner": owner,
            "original_id": original_id, "original_image_id": images["original"],
            "candidate_image_id": images[scenario["image"]], "model": "fixture-model",
            "base_url": "http://127.0.0.1:" + str(port), "ingress_base_url": "http://127.0.0.1:" + str(port),
            "docker_socket": api.socket_path, "fault_pause": scenario.get("pause"),
            "startup_timeout_s": 5, "deployment_timeout_s": 45, "rollback_timeout_s": 20,
            "proof_timeout_s": 5, "controller_timeout_s": 3, "supervisor_lease_s": 2,
            "poll_interval_s": 0.1, "request_timeout_s": 2, "stop_timeout_s": 1,
            "probe_timeout_s": 10, "validation_timeout_s": 10,
        }
        plan.update(calibrated_plan_keys(calibrated_input if scenario.get("calibrated") else None))
        service = Services(original, plan, api=api)
        wait_ready(api, original_id, images["original"], service.health, time.monotonic() + 10, interval=0.1)
        result["attempt_dir"] = str(output / "attempt")
        journal = prepare(api, plan, output / "attempt")
        state = journal.read()
        identities.append((state["candidate_id"], state["candidate_image_id"]))
        approval = {"action": "one-controlled-swap", "approved": True,
                    "attempt_id": attempt_id, "original_id": original_id,
                    "original_image_id": images["original"], "candidate_id": state["candidate_id"],
                    "candidate_image_id": state["candidate_image_id"], "expires_at": time.time() + 90}
        approval_path = output / "approval.json"
        private_json(approval_path, approval, exclusive=True)
        launched = launch(journal, approval_path)
        started = time.monotonic()
        end = started + 75
        injected = False
        while time.monotonic() < end:
            fault_path = journal.directory / "fault-checkpoint.json"
            if scenario.get("signal") and fault_path.exists() and not injected:
                checkpoint = read_json(fault_path)
                if checkpoint.get("attempt_id") != attempt_id or checkpoint.get("point") != scenario["pause"]:
                    raise Failure("fault checkpoint identity mismatch")
                state = journal.read()
                lease = read_json(journal.directory / "supervisor.json")
                victim = launched["supervisor_pid"] if scenario["signal"] == "supervisor-kill" else checkpoint["controller_pid"]
                identity = lease["identity"] if scenario["signal"] == "supervisor-kill" else state.get("controller_identity")
                fault_at = time.monotonic()
                signal_owned_process(victim, identity, signal.SIGSTOP if scenario["signal"] == "stop" else signal.SIGKILL)
                injected = True
            final_path = journal.directory / ("recovery-result.json" if scenario.get("signal") == "supervisor-kill" else "supervisor-result.json")
            if final_path.exists():
                recovery = read_json(final_path)
                if recovery.get("verified") is not True or recovery.get("result") not in ("RESTORED", "NOOP"):
                    raise Failure("disposable original was not verified restored")
                state = journal.read()
                current = require_running(api, original_id, images["original"])
                from .docker_config import assert_candidate_parity
                assert_candidate_parity(original, current, images["original"])
                if current["Name"] != "/" + name:
                    raise Failure("disposable original name not restored")
                if not service.chat().get("ok"):
                    raise Failure("independent fixture semantic verification failed")
                candidate = checked_inspect(api, state["candidate_id"], state["candidate_image_id"])
                if scenario.get("calibrated"):
                    # Sanitized echo first, so the daemon's mount spelling
                    # survives cleanup; then the fail-closed assertions.
                    persist_calibrated_echo(output, candidate)
                    result["calibrated_capability_evidence"] = assert_calibrated_capability_evidence(
                        state, original, current, candidate, plan, journal.directory)
                controller_state = read_json(journal.directory / "controller.json")
                result["scenario_evidence"] = assert_scenario_evidence(
                    scenario, state, recovery, controller_state, candidate, injected)
                detections = [recovery.get("started_monotonic", started)]
                controller_evidence = journal.directory / "controller.json"
                if controller_evidence.exists():
                    pulse = read_json(controller_evidence)
                    if pulse.get("status") == "recovering":
                        detections.append(pulse["heartbeat"])
                requested = journal.directory / "recover-request.json"
                if requested.exists():
                    detections.append(read_json(requested)["detected_monotonic"])
                detected = min(detections)
                start_events = [x["at"] for x in state["history"] if x["operation"] == "start_candidate"]
                candidate_start = start_events[0] if start_events else None
                result.update(result="PASS", actual_signal_injected=injected, recovery=recovery,
                              fault_to_detection_s=max(0, detected - fault_at) if fault_at else None,
                              fault_to_verified_s=time.monotonic() - fault_at if fault_at else None,
                              candidate_start_monotonic=candidate_start,
                              failure_detected_monotonic=detected,
                              start_to_failure_detection_s=detected - candidate_start if candidate_start else None,
                              detection_to_verified_s=time.monotonic() - detected,
                              duration_s=time.monotonic() - started, original_id=original_id,
                              candidate_id=state["candidate_id"], production_inference_tested=False)
                break
            time.sleep(0.1)
        else:
            raise Failure("disposable supervision deadline expired")
    except BaseException as error:
        result["error"] = type(error).__name__
    finally:
        # On failed recovery keep owned evidence; do not race an active supervisor.
        if result["result"] == "PASS":
            try:
                cleanup_owned(api, list(reversed(identities)), owner)
                result["exact_id_cleanup"] = True
                ledger = read_json(output / "original-fixture.json")
                ledger.update(phase="REMOVED", cleanup_complete=True, cleanup_admissible=False)
                private_json(output / "original-fixture.json", ledger)
                if journal:
                    candidate_ledger = journal.directory / "candidate-create-intent.json"
                    record = read_json(candidate_ledger)
                    record.update(phase="REMOVED", cleanup_complete=True, cleanup_admissible=False)
                    private_json(candidate_ledger, record)
            except Exception as error:
                result.update(result="FAIL", cleanup_error=type(error).__name__)
        private_json(output / "result.json", result)
    return result


def run_rehearsals(api, base_image, output_dir, calibrated_input=None):
    if os.name != "posix":
        raise Failure("Docker rehearsals require Linux")
    output = Path(output_dir)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        images = build_fixtures(api, image_id(base_image), secrets.token_hex(8), ledger_dir=output / "build-intents")
    except BaseException as error:
        result = {"schema": "occ3-disposable-rehearsal/v1", "scope": "disposable_fixture_only", "result": "FAIL",
                  "build_error": type(error).__name__, "build_intent_directory": str(output / "build-intents"),
                  "build_cleanup_complete": False, "production_touched": False, "live_glm_or_occ3_proven": False}
        private_json(output / "result.json", result)
        return result
    private_json(output / "images.json", images, exclusive=True)
    scenarios = rehearsal_scenarios(calibrated_input)
    outcomes = []
    for scenario in scenarios:
        outcome = run_scenario(api, images, output / scenario["name"], scenario,
                               calibrated_input=calibrated_input)
        outcomes.append(outcome)
        if outcome["result"] != "PASS":
            break
    result = {"schema": "occ3-disposable-rehearsal/v1", "scope": "disposable_fixture_only",
              "result": "PASS" if len(outcomes) == len(scenarios) and all(x["result"] == "PASS" for x in outcomes) else "FAIL",
              "scenarios": outcomes, "production_touched": False,
              "calibrated_capability_scenario": calibrated_input is not None,
              "live_glm_or_occ3_proven": False, "images_retained": images}
    private_json(output / "result.json", result)
    return result
