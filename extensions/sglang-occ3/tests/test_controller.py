"""Offline unit tests for the OCC-3 deployment controller.

Covers the regression that caused the 2026-10-21 outage: docker 29.x
range templates append a trailing blank line even for non-empty ranges,
so line-based (mapfile) capture injected empty-string arguments into
the replicated launch command. The controller must (a) build every
argument array from `docker inspect` JSON, element-exact, (b) reject
any blank element, and (c) structurally reject any candidate whose
recorded configuration differs from the original beyond the explicitly
allowed OCC-3 changes.

The 2026-10-21 DGX disposable rehearsal additionally proved (defect CD-1)
that Docker Engine 29.6.2 coerces an explicitly present null
HostConfig.OomKillDisable to false at create while an omitted key inspects
as null, so both original spellings are covered and the v3 create/parity
path is exercised through an offline model of exactly that coercion. The
same rehearsal proved (defect CD-2) that the supervisor fence directory must
default to the attempt directory so a non-root deployment needs no system
path. The rehearsal re-run proved (defect CD-3) that the same engine
populates endpoint DNSNames [current-name, 12-char-id, Config.Hostname]
when a running container is renamed, so the original's hostname is covered
as generated identity material on both sides of the DNSNames parity.
Rehearsal round 3 proved (defect CD-4) that the populated list is never
rewritten afterwards: it survives stop, rename-back and restart with the
stale previous-name entry intact, so the lifecycle's own staging names
(prefix + "-candidate"/"-preserved"/"-failed") are generated identity
material on both sides of the DNSNames parity as well.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy"
spec = importlib.util.spec_from_file_location(
    "occ3_controller", DEPLOY_DIR / "occ3_controller.py"
)
controller_mod = importlib.util.module_from_spec(spec)
sys.modules.setdefault("occ3_controller", controller_mod)
spec.loader.exec_module(controller_mod)


OLD_KEY = "a" * 64
NEW_KEY = "b" * 64


def make_norm(**overrides):
    norm = {
        "Entrypoint": ["python3"],
        "Cmd": [],
        "Env": ["A=1", "B=2"],
        "WorkingDir": "/sgl-workspace/sglang",
        "User": "",
        "Binds": [],
        "Mounts": [],
        "PortBindings": {"30001/tcp": [("", "30001")]},
        "NetworkMode": "neo-k3-ingress-net",
        "Networks": ["neo-k3-ingress-net"],
        "IpcMode": "host",
        "PidMode": "",
        "Runtime": "nvidia",
        "ShmSize": 67108864,
        "Ulimits": [],
        "Tmpfs": {},
        "ExtraHosts": [],
        "DeviceRequests": [{"Driver": "", "Count": -1,
                            "DeviceIDs": [], "Capabilities": [["gpu"]]}],
        "RestartPolicy": {"Name": "no"},
        "AutoRemove": False,
        "Privileged": False,
        "CapAdd": [],
        "CapDrop": [],
        "SecurityOpt": [],
        "Labels": {},
        "OomKillDisable": False,
        "Memory": 0,
        "NanoCpus": 0,
        "Init": None,
    }
    norm.update(overrides)
    return norm


# Real DGX originals inspect OomKillDisable as null (daemon default); an
# explicit false original must behave identically. The v1/v2 normalizer
# coerces both spellings to the same value, and the tests below prove it.
OOM_SPELLINGS = [None, False]


# --- the outage regression -------------------------------------------------


def test_docker29_trailing_blank_reproduces_the_outage_bug():
    """Line-based capture of a one-element entrypoint yields an extra ""."""
    template_lines = ["python3", ""]  # mapfile over the range output
    json_array = ["python3"]
    assert len(template_lines) == len(json_array) + 1
    assert template_lines[-1] == ""


def test_blank_elements_are_rejected_everywhere():
    for bad in (["python3", "", "-m"], ["x", " "], [""]):
        with pytest.raises(controller_mod.Failure):
            controller_mod.assert_no_blank_elements(bad, "unit")


@pytest.mark.parametrize("oom_kill_disable", OOM_SPELLINGS)
def test_built_args_carry_no_blank_elements(oom_kill_disable):
    new_cmd = ["-m", "sglang.launch_server", "--api-key", NEW_KEY]
    args = controller_mod.build_run_args(
        make_norm(OomKillDisable=oom_kill_disable), "img", "cand", new_cmd, ["A=1"]
    )
    assert all(isinstance(a, str) and a.strip() for a in args)


@pytest.mark.parametrize("oom_kill_disable", OOM_SPELLINGS)
def test_built_args_replicate_the_captured_configuration(oom_kill_disable):
    new_cmd = ["-m", "sglang.launch_server", "--api-key", NEW_KEY]
    args = controller_mod.build_run_args(
        make_norm(OomKillDisable=oom_kill_disable), "img", "cand", new_cmd, ["A=1", "B=2"]
    )
    joined = " ".join(args)
    assert "--entrypoint python3" in joined
    assert "--workdir /sgl-workspace/sglang" in joined
    assert "--ipc host" in joined
    assert "--runtime nvidia" in joined
    assert "--shm-size 67108864" in joined
    assert "--gpus all" in joined
    assert "--network neo-k3-ingress-net" in joined
    assert "-p 30001:30001/tcp" in joined
    assert "-e A=1" in joined and "-e B=2" in joined
    # the command tail is exactly entrypoint+cmd with no blanks
    tail = args[args.index("img") + 1:]
    assert tail == ["-m", "sglang.launch_server", "--api-key", NEW_KEY]


@pytest.mark.parametrize("oom_kill_disable", OOM_SPELLINGS)
def test_multi_element_entrypoint_moves_tail_into_command(oom_kill_disable):
    norm = make_norm(Entrypoint=["python3", "-u"], Cmd=[], OomKillDisable=oom_kill_disable)
    args = controller_mod.build_run_args(
        norm, "img", "cand", ["-m", "x"], []
    )
    tail = args[args.index("img") + 1:]
    assert tail == ["-u", "-m", "x"]


# --- the structural diff allowlist ------------------------------------------


def original_and_candidate(mutate=None, oom_kill_disable=False):
    original = make_norm(Cmd=["-m", "x", "--api-key", OLD_KEY],
                         OomKillDisable=oom_kill_disable)
    candidate = make_norm(Cmd=["-m", "x", "--api-key", NEW_KEY],
                          OomKillDisable=oom_kill_disable)
    if mutate:
        mutate(candidate)
    return original, candidate


@pytest.mark.parametrize("oom_kill_disable", OOM_SPELLINGS)
def test_key_rotation_is_the_only_accepted_cmd_change(oom_kill_disable):
    original, candidate = original_and_candidate(oom_kill_disable=oom_kill_disable)
    problems = controller_mod.diff_configs(
        original, candidate, 3, OLD_KEY, NEW_KEY
    )
    assert problems == []


def test_injected_empty_argument_is_rejected():
    original, candidate = original_and_candidate(
        mutate=lambda c: c.update(Cmd=["", "-m", "x", "--api-key", NEW_KEY])
    )
    problems = controller_mod.diff_configs(
        original, candidate, 3, OLD_KEY, NEW_KEY
    )
    assert problems


@pytest.mark.parametrize("oom_kill_disable", OOM_SPELLINGS)
def test_any_other_field_change_is_rejected(oom_kill_disable):
    for mutate in (
        lambda c: c.update(WorkingDir="/elsewhere"),
        lambda c: c.update(Binds=["/host:/ctr"]),
        lambda c: c["PortBindings"].update({"9000/tcp": [("", "9000")]}),
        lambda c: c.update(Networks=["other-net"]),
        lambda c: c.update(Env=["A=1", "B=2", "C=3"]),
        lambda c: c.update(Runtime="runc"),
        lambda c: c.update(IpcMode=""),
        lambda c: c["DeviceRequests"].clear(),
        lambda c: c.update(ShmSize=64),
        lambda c: c.update(OomKillDisable=not oom_kill_disable
                           if isinstance(oom_kill_disable, bool) else True),
    ):
        original, candidate = original_and_candidate(mutate=mutate,
                                                     oom_kill_disable=oom_kill_disable)
        problems = controller_mod.diff_configs(
            original, candidate, 3, OLD_KEY, NEW_KEY
        )
        assert problems, mutate


def test_wrong_key_position_is_rejected():
    original, candidate = original_and_candidate(
        mutate=lambda c: c.update(Cmd=["-m", "x", NEW_KEY, "--api-key"])
    )
    problems = controller_mod.diff_configs(
        original, candidate, 3, OLD_KEY, NEW_KEY
    )
    assert problems


# --- key hygiene -------------------------------------------------------------


def test_replace_api_key_keeps_position():
    cmd = ["-m", "x", "--api-key", OLD_KEY, "--tp-size", "8"]
    out, index = controller_mod.replace_api_key(cmd, NEW_KEY)
    assert index == 3
    assert out[3] == NEW_KEY
    assert out[:3] == cmd[:3] and out[4:] == cmd[4:]


def test_api_key_from_cmd_forms():
    assert controller_mod.api_key_from_cmd(["--api-key", "k"]) == "k"
    assert controller_mod.api_key_from_cmd(["--api-key=k2"]) == "k2"
    assert controller_mod.api_key_from_cmd(["-m", "x"]) is None


def test_controller_redacts_keys_from_logs_and_status(tmp_path):
    controller = controller_mod.Controller(str(tmp_path), verbose=False)
    controller.register_secret(OLD_KEY)
    controller.register_secret(NEW_KEY)
    controller.log("running with " + OLD_KEY + " then " + NEW_KEY)
    controller.write_status({"phase": "t", "key": NEW_KEY})
    log_text = (tmp_path / "controller.log").read_text(encoding="utf-8")
    status_text = json.loads((tmp_path / "status.json").read_text())
    assert OLD_KEY not in log_text and NEW_KEY not in log_text
    assert controller_mod.KEY_SENTINEL in log_text
    assert status_text["key"] == controller_mod.KEY_SENTINEL


def test_controller_refuses_secrets_in_argv(tmp_path):
    controller = controller_mod.Controller(str(tmp_path), verbose=False)
    controller.register_secret(OLD_KEY)
    with pytest.raises(controller_mod.Failure):
        controller.run(["echo", OLD_KEY])


# --- v3 controller: Docker 29.6.2 OomKillDisable create coercion (CD-1) -----
#
# DGX rehearsal evidence (2026-10-21, TRY-72998-gpu01, Docker Engine 29.6.2,
# API 1.55, four independent container pairs): every original inspects
# HostConfig.OomKillDisable as null; a create body carrying that copied
# explicit null produces a candidate that inspects false; an omitted key
# inspects null. The offline model below implements exactly that evidenced
# coercion and nothing else: all other fields round-trip verbatim.

_EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from deploy.v3 import docker_config as v3_dc
from deploy.v3 import supervisor as v3_sup

V3_ORIGINAL_IMAGE = "sha256:" + "a" * 64
V3_CANDIDATE_IMAGE = "sha256:" + "b" * 64
V3_KEY = "synthetic-v3-private-credential"


def v3_original(oom_kill_disable=None):
    """Raw inspect shaped like the real DGX originals (OomKillDisable null)."""
    return {
        "Id": "a" * 64, "Name": "/original", "Image": V3_ORIGINAL_IMAGE,
        "Config": {
            "Image": V3_ORIGINAL_IMAGE, "Hostname": "a" * 12,
            "Entrypoint": ["python3"],
            "Cmd": ["-m", "sglang.launch_server", "--api-key", V3_KEY],
            "Env": ["A=1"], "WorkingDir": "/work", "User": "", "Labels": {},
        },
        "HostConfig": {
            "NetworkMode": "bridge", "Runtime": "runc", "Binds": None, "Mounts": [],
            "PortBindings": {}, "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False, "Privileged": False, "Memory": 0,
            "OomKillDisable": oom_kill_disable,
        },
        "Mounts": [],
        "NetworkSettings": {"Networks": {"bridge": {"NetworkID": "net-1", "Aliases": None}}},
    }


def docker29_create(original, body, name="candidate"):
    """Offline Docker 29.6.2 create+inspect model for the candidate body.

    Only the evidenced coercion is modeled: an explicitly present null
    OomKillDisable becomes false; an omitted key is materialized as null.
    Every other create-body value round-trips verbatim, and the endpoint
    runtime fields reset like a created-not-started container.
    """
    candidate = deepcopy(original)
    candidate.update(Id="c" * 64, Name="/" + name, Image=body["Image"])
    candidate["Config"] = {key: deepcopy(value) for key, value in body.items()
                           if key not in ("HostConfig", "NetworkingConfig")}
    host = deepcopy(body["HostConfig"])
    if "OomKillDisable" in host:
        if host["OomKillDisable"] is None:
            host["OomKillDisable"] = False
    else:
        host["OomKillDisable"] = None
    candidate["HostConfig"] = host
    candidate["State"] = {"Status": "created", "Running": False}
    network = candidate["NetworkSettings"]["Networks"]["bridge"]
    network.update(NetworkID="", EndpointID="", IPAddress="", MacAddress="", DNSNames=None)
    return candidate


@pytest.mark.parametrize("oom_kill_disable", [None, False])
def test_v3_docker29_oomkilldisable_original_roundtrip_parity(oom_kill_disable):
    """CD-1 regression: a null original must survive create + exact parity.

    Pre-fix this failed for the null original: build_create_request copied
    the raw null into the create body, the daemon model coerced it to false,
    and the exact HostConfig parity raised ConfigError("HostConfig_differs")
    before any mutation, exactly like the DGX rehearsal.
    """
    original = v3_original(oom_kill_disable)
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    candidate = docker29_create(original, body)
    proof = v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE)
    assert proof["result"] == "PASS" and proof["host_config_exact"] is True


def test_v3_builder_omits_oomkilldisable_only_for_null_originals():
    original = v3_original(None)
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    assert "OomKillDisable" not in body["HostConfig"]
    original = v3_original(False)
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    assert body["HostConfig"]["OomKillDisable"] is False
    original = v3_original(True)
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    assert body["HostConfig"]["OomKillDisable"] is True


def test_v3_parity_tolerates_only_the_oomkilldisable_null_false_spelling():
    original = v3_original(None)
    coerced = deepcopy(original)
    coerced.update(Id="c" * 64, Name="/candidate", Image=V3_CANDIDATE_IMAGE)
    coerced["HostConfig"]["OomKillDisable"] = False
    v3_dc.assert_candidate_parity(original, coerced, V3_CANDIDATE_IMAGE)
    omitted = deepcopy(coerced)
    omitted["HostConfig"].pop("OomKillDisable")
    v3_dc.assert_candidate_parity(original, omitted, V3_CANDIDATE_IMAGE)
    # A true original is a real setting change and is never tolerated.
    enabled = v3_original(True)
    flipped = deepcopy(enabled)
    flipped.update(Id="c" * 64, Name="/candidate", Image=V3_CANDIDATE_IMAGE)
    for value in (None, False):
        flipped["HostConfig"]["OomKillDisable"] = value
        with pytest.raises(v3_dc.ConfigError, match="HostConfig_differs"):
            v3_dc.assert_candidate_parity(enabled, flipped, V3_CANDIDATE_IMAGE)
    # The tolerance is scoped to this one field: every other HostConfig
    # difference, including absent-versus-null, still fails closed.
    for mutation in (
        lambda host: host.update(PidsLimit=5),
        lambda host: host.update(PidsLimit=None),
        lambda host: host.update(Memory=1),
    ):
        altered = deepcopy(original)
        altered.update(Id="c" * 64, Name="/candidate", Image=V3_CANDIDATE_IMAGE)
        mutation(altered["HostConfig"])
        with pytest.raises(v3_dc.ConfigError, match="HostConfig_differs"):
            v3_dc.assert_candidate_parity(original, altered, V3_CANDIDATE_IMAGE)


# --- v3 controller: rename-populated DNSNames identity (CD-3) ----------------
#
# The 2026-10-21 DGX rehearsal round 2 proved (defect CD-3) that Docker
# Engine 29.6.2 (API 1.55) populates endpoint DNSNames with
# [current-name, 12-char-id, Config.Hostname] when a RUNNING container is
# renamed, and the populated list survives stop/start, while a never-renamed
# container inspects DNSNames=null. The mutation renames the running
# original to the preserved name before stopping it, so recovery's first
# parity check compared the populated current inspect against the null
# pre-mutation snapshot and the surviving Config.Hostname entry raised
# ConfigError("network_dns_names_differ"). The original's hostname is now
# generated identity material on both sides of the DNSNames comparison,
# exactly like name/ID aliases; every other DNSNames entry stays exact.

DNS_RUN_ID = "2d672cf9fa9d12d3a1ba511532e609639d4b7261f3d107089a3665fb51563d0a"
DNS_RUN_PREFIX = "occ3-v3-rehearsal-ee915404e705653d"
DNS_RUN_IMAGE = "sha256:019387f49fe2f86c4b698918689217f8407c2597b60177de104d4bb7e28241a0"
DNS_RUN_NETWORK = "bf868b8103a9428238cedba8c5073e853da9adcad627552221ea260510706625"


def v3_dns_snapshot():
    """The retained rehearsal-2 run-1 original snapshot, identity-exact.

    Id, Name, Config.Hostname, image, network identity and the null DNSNames
    reproduce the retained run-1 journal document; dense launch fields are
    abbreviated exactly like v3_original.
    """
    return {
        "Id": DNS_RUN_ID, "Name": "/" + DNS_RUN_PREFIX + "-original",
        "Image": DNS_RUN_IMAGE,
        "Config": {
            "Image": DNS_RUN_IMAGE, "Hostname": DNS_RUN_PREFIX,
            "Entrypoint": ["python3"],
            "Cmd": ["-m", "sglang.launch_server", "--api-key", V3_KEY],
            "Env": ["A=1"], "WorkingDir": "/work", "User": "", "Labels": {},
        },
        "HostConfig": {
            "NetworkMode": "bridge", "Runtime": "runc", "Binds": None, "Mounts": [],
            "PortBindings": {}, "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False, "Privileged": False, "Memory": 0,
        },
        "Mounts": [],
        "NetworkSettings": {"Networks": {"bridge": {
            "NetworkID": DNS_RUN_NETWORK,
            "EndpointID": "3b8b4d440aa51337df61a03969a7256f20a7b107069f75367eea7178746f13d6",
            "Aliases": None, "DNSNames": None, "IPAMConfig": None, "Links": None,
            "DriverOpts": None, "GwPriority": 0,
            "IPAddress": "172.17.0.2", "MacAddress": "1a:22:9e:87:c6:36",
        }}},
        "State": {"Status": "running", "Running": True},
    }


def docker29_rename(document, new_name):
    """Offline Docker 29.6.2 daemon model for container rename (round 3).

    Renaming a RUNNING container replaces the endpoint's generated identity
    list with [new-name, 12-char-id, Config.Hostname] (order-preserving
    dedup, so a hostname equal to the 12-char id matches the production
    shape). Renaming a STOPPED or never-started container changes only the
    name: an already populated DNSNames list keeps its stale entries (the
    rename history survives), and a null list stays null. The previous
    always-overwriting model could not represent the stale-entry state at
    all, which is exactly why defect CD-4 was invisible to the offline tests.
    """
    renamed = deepcopy(document)
    renamed["Name"] = "/" + new_name
    if (document.get("State") or {}).get("Running") is True:
        for endpoint in renamed["NetworkSettings"]["Networks"].values():
            endpoint["DNSNames"] = list(dict.fromkeys(
                [new_name, document["Id"][:12], document["Config"]["Hostname"]]))
            endpoint["EndpointID"] = ""
    return renamed


def docker29_stop(document):
    """Stop tears the running endpoint down but never touches DNSNames."""
    stopped = deepcopy(document)
    stopped["State"] = {"Status": "exited", "Running": False}
    for endpoint in stopped["NetworkSettings"]["Networks"].values():
        endpoint.update(EndpointID="", Gateway="", IPAddress="", MacAddress="")
    return stopped


def docker29_start(document, user_defined_network=False):
    """Offline Docker 29.6.2 start model.

    A start never rewrites an already populated DNSNames list (round-3
    probe: restart keeps the stale rename history). The default bridge
    leaves a null list null (rounds 1-3 fixtures); a user-defined network
    (the production shape) populates a null list with [name, 12-char-id].
    Per-start endpoint material is regenerated like the real engine, which
    also changes a container's MacAddress across restarts.
    """
    started = deepcopy(document)
    started["State"] = {"Status": "running", "Running": True}
    for endpoint in started["NetworkSettings"]["Networks"].values():
        if not endpoint.get("DNSNames") and user_defined_network:
            endpoint["DNSNames"] = [started["Name"].lstrip("/"), started["Id"][:12]]
        endpoint["EndpointID"] = "e" * 64
        endpoint["MacAddress"] = "02:42:ac:11:00:02"
    return started


def test_v3_recovery_parity_after_running_rename_populates_dnsnames():
    """CD-3 regression: the exact rehearsal-2 run-1 recovery comparison.

    Pre-fix this raised ConfigError("network_dns_names_differ") exactly like
    both independent DGX runs: the pre-mutation snapshot inspected
    DNSNames=null while the renamed, stopped original carried
    [preserved-name, 12-char-id, hostname] and only the hostname entry
    survived the identity subtraction.
    """
    snapshot = v3_dns_snapshot()
    current = docker29_stop(docker29_rename(snapshot, DNS_RUN_PREFIX + "-preserved"))
    proof = v3_dc.assert_candidate_parity(snapshot, current, DNS_RUN_IMAGE)
    assert proof["result"] == "PASS" and proof["network_options_exact"] is True


def test_v3_recovery_parity_with_production_shaped_snapshot_dnsnames():
    """Production recovery shape: Config.Hostname equals the 12-char id and
    the running original's snapshot already carries populated DNSNames
    [name, 12-char-id], as observed on the production containers."""
    snapshot = v3_dns_snapshot()
    snapshot["Config"]["Hostname"] = DNS_RUN_ID[:12]
    endpoint = snapshot["NetworkSettings"]["Networks"]["bridge"]
    endpoint["DNSNames"] = [DNS_RUN_PREFIX + "-original", DNS_RUN_ID[:12]]
    current = docker29_stop(docker29_rename(snapshot, DNS_RUN_PREFIX + "-preserved"))
    v3_dc.assert_candidate_parity(snapshot, current, DNS_RUN_IMAGE)


def test_v3_candidate_dnsnames_may_carry_the_preserved_hostname():
    """Completion shape: a candidate renamed while running carries the
    preserved original Hostname inside DNSNames. That hostname is original
    identity, not candidate identity, and must be tolerated exactly like
    the name/ID aliases a renamed candidate takes over."""
    snapshot = v3_dns_snapshot()
    candidate = deepcopy(snapshot)
    candidate.update(Id="7" * 64, Name="/" + DNS_RUN_PREFIX + "-original",
                     Image=V3_CANDIDATE_IMAGE)
    candidate["Config"]["Image"] = V3_CANDIDATE_IMAGE
    endpoint = candidate["NetworkSettings"]["Networks"]["bridge"]
    endpoint.update(EndpointID="", DNSNames=[DNS_RUN_PREFIX + "-original",
                                             "7" * 12, DNS_RUN_PREFIX])
    v3_dc.assert_candidate_parity(snapshot, candidate, V3_CANDIDATE_IMAGE)


def test_v3_dnsnames_difference_beyond_hostname_identity_still_rejected():
    """The hostname subtraction must not over-relax: every other DNSNames
    entry is still compared exactly, added or dropped on either side."""
    snapshot = v3_dns_snapshot()
    current = docker29_stop(docker29_rename(snapshot, DNS_RUN_PREFIX + "-preserved"))
    endpoint = current["NetworkSettings"]["Networks"]["bridge"]
    endpoint["DNSNames"] = endpoint["DNSNames"] + ["rogue-dns-entry"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(snapshot, current, DNS_RUN_IMAGE)
    populated = deepcopy(snapshot)
    kept = populated["NetworkSettings"]["Networks"]["bridge"]
    kept.update(Aliases=["genuine-service-alias"],
                DNSNames=[DNS_RUN_PREFIX + "-original", DNS_RUN_ID[:12],
                          "genuine-service-alias"])
    renamed = docker29_stop(docker29_rename(populated, DNS_RUN_PREFIX + "-preserved"))
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(populated, renamed, DNS_RUN_IMAGE)


# --- v3 controller: stale rename history in DNSNames (CD-4) ------------------
#
# The 2026-10-21 DGX rehearsal round 3 proved (defect CD-4) that Docker
# Engine 29.6.2 (API 1.55) never rewrites an endpoint DNSNames list after
# the rename that populated it: renaming a RUNNING container sets
# [new-name, 12-char-id, Config.Hostname], and that list then survives
# stop, rename-back and restart with the stale previous-name entry intact
# (round-3 probe; both rehearsal runs). The recovery rename dance
# (original -> preserved -> back) therefore leaves the stale preserved-name
# inside the restored original, and the FINAL recovery parity check raised
# ConfigError("network_dns_names_differ") after entry parity, full restore
# and chat/ingress proofs had already passed. A container that has ever
# been renamed carries its rename history in DNSNames forever, so the
# lifecycle's own staging names are generated identity material on both
# sides of the comparison, exactly like name/ID aliases and Hostname
# (the CD-1/CD-3 generated-identity class).

DNS3_RUN_ID = "b82b58164748bba9fb5fc7d0280c3c004ecde708519838efb15e730aeb8ffd60"
DNS3_RUN_PREFIX = "occ3-v3-rehearsal-15b4a59d6b31e03c"
DNS3_RUN_IMAGE = "sha256:2f777af66dc04aacb19917de77236a7b6556d5430a86979846a53dae6aacac29"
DNS3_CANDIDATE_IMAGE = "sha256:7066c059e734fa959d70c85e7961665622e339222fa24a488ef0d1e99419bf00"
DNS3_RUN_NETWORK = "bf868b8103a9428238cedba8c5073e853da9adcad627552221ea260510706625"


def v3_dns3_snapshot():
    """The retained rehearsal-3 run-1 original snapshot, identity-exact.

    Id, Name, Config.Hostname, image, network identity, endpoint material
    and the null DNSNames reproduce the retained run-1 journal document;
    dense launch fields are abbreviated exactly like v3_original.
    """
    return {
        "Id": DNS3_RUN_ID, "Name": "/" + DNS3_RUN_PREFIX + "-original",
        "Image": DNS3_RUN_IMAGE,
        "Config": {
            "Image": DNS3_RUN_IMAGE, "Hostname": DNS3_RUN_PREFIX,
            "Entrypoint": ["python3"],
            "Cmd": ["-m", "sglang.launch_server", "--api-key", V3_KEY],
            "Env": ["A=1"], "WorkingDir": "/work", "User": "", "Labels": {},
        },
        "HostConfig": {
            "NetworkMode": "bridge", "Runtime": "runc", "Binds": None, "Mounts": [],
            "PortBindings": {}, "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False, "Privileged": False, "Memory": 0,
        },
        "Mounts": [],
        "NetworkSettings": {"Networks": {"bridge": {
            "NetworkID": DNS3_RUN_NETWORK,
            "EndpointID": "565197a7bad782ed4346121a3608358928c5235dbaccaa90b71e5635e6bfc534",
            "Aliases": None, "DNSNames": None, "IPAMConfig": None, "Links": None,
            "DriverOpts": None, "GwPriority": 0,
            "IPAddress": "172.17.0.2", "MacAddress": "4e:1e:f4:19:3b:2a",
        }}},
        "State": {"Status": "running", "Running": True},
    }


def docker29_recovery_rename_dance(snapshot, prefix):
    """The recovery's exact rename dance on the Docker 29.6.2 model:
    running rename to the preserved name, stop, stopped rename back,
    restart. The restored container keeps the stale preserved-name in
    DNSNames forever, exactly like both retained rehearsal-3 runs."""
    return docker29_start(docker29_rename(
        docker29_stop(docker29_rename(snapshot, prefix + "-preserved")),
        prefix + "-original"))


def test_v3_recovery_final_parity_after_rename_back_keeps_stale_preserved_name():
    """CD-4 regression: the exact rehearsal-3 recovery FINAL comparison.

    Pre-fix this raised ConfigError("network_dns_names_differ") exactly
    like both independent DGX runs: the pre-mutation snapshot inspected
    DNSNames=null while the restored original still carried
    [preserved-name, 12-char-id, hostname], and the stale preserved-name
    was the only entry surviving the identity subtraction. The previous
    daemon model always overwrote DNSNames on rename, so no offline test
    could even represent this defect.
    """
    snapshot = v3_dns3_snapshot()
    current = docker29_recovery_rename_dance(snapshot, DNS3_RUN_PREFIX)
    assert current["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] == [
        DNS3_RUN_PREFIX + "-preserved", DNS3_RUN_ID[:12], DNS3_RUN_PREFIX]
    assert current["Name"] == "/" + DNS3_RUN_PREFIX + "-original"
    proof = v3_dc.assert_candidate_parity(snapshot, current, DNS3_RUN_IMAGE)
    assert proof["result"] == "PASS" and proof["network_options_exact"] is True


def test_v3_completion_candidate_parity_with_stale_preserved_name():
    """CD-4 regression: the completion candidate check after a recovery.

    Production shape: the running original's DNSNames is populated at
    start [name, 12-char-id] (observed on the production containers) and
    the restored original additionally carries the stale preserved-name
    from its rename history. A later attempt against that restored
    original completes with a fresh candidate renamed to the original
    name; the supervisor's completion candidate check then compares the
    stale left side against the fresh running candidate, which pre-fix
    raised ConfigError("network_dns_names_differ").
    """
    snapshot = v3_dns3_snapshot()
    snapshot["Config"]["Hostname"] = DNS3_RUN_ID[:12]
    snapshot["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] = [
        DNS3_RUN_PREFIX + "-original", DNS3_RUN_ID[:12]]
    restored = docker29_recovery_rename_dance(snapshot, DNS3_RUN_PREFIX)
    assert restored["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] == [
        DNS3_RUN_PREFIX + "-preserved", DNS3_RUN_ID[:12]]
    body = v3_dc.build_create_request(restored, DNS3_CANDIDATE_IMAGE,
                                      DNS3_RUN_PREFIX + "-candidate")
    candidate = docker29_create(restored, body, DNS3_RUN_PREFIX + "-candidate")
    renamed = docker29_rename(candidate, DNS3_RUN_PREFIX + "-original")
    assert renamed["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] is None
    running = docker29_start(renamed, user_defined_network=True)
    assert running["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] == [
        DNS3_RUN_PREFIX + "-original", "c" * 12]
    v3_dc.assert_candidate_parity(restored, running, DNS3_CANDIDATE_IMAGE)


def test_v3_stale_rename_history_subtraction_does_not_over_relax():
    """Guards: only generated identity and the lifecycle's own staging-name
    grammar are subtracted, on both sides. A rogue non-generated entry added
    on either side, a genuine user DNS entry missing from either side, and
    a near-miss staging spelling all still fail closed."""
    snapshot = v3_dns3_snapshot()
    current = docker29_recovery_rename_dance(snapshot, DNS3_RUN_PREFIX)
    names = current["NetworkSettings"]["Networks"]["bridge"]["DNSNames"]
    rogue_right = deepcopy(current)
    rogue_right["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] = names + ["rogue-dns-entry"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(snapshot, rogue_right, DNS3_RUN_IMAGE)
    rogue_left = deepcopy(snapshot)
    rogue_left["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] = ["rogue-dns-entry"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(rogue_left, current, DNS3_RUN_IMAGE)
    # A genuine user alias (equal on both sides) whose DNSNames entry is
    # lost by the restored side still fails.
    alias_snapshot = deepcopy(snapshot)
    alias_snapshot["NetworkSettings"]["Networks"]["bridge"].update(
        Aliases=["genuine-service-alias"],
        DNSNames=[DNS3_RUN_PREFIX + "-original", DNS3_RUN_ID[:12], "genuine-service-alias"])
    alias_current = deepcopy(current)
    alias_current["NetworkSettings"]["Networks"]["bridge"]["Aliases"] = ["genuine-service-alias"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(alias_snapshot, alias_current, DNS3_RUN_IMAGE)
    # A non-generated DNS entry the snapshot never had still fails.
    extra_right = deepcopy(current)
    extra_right["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] = names + ["genuine-service-alias"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(snapshot, extra_right, DNS3_RUN_IMAGE)
    # Grammar precision: a near-miss staging spelling (31-hex nonce tail)
    # is not the lifecycle's name and is compared exactly.
    near_miss = deepcopy(current)
    near_miss["NetworkSettings"]["Networks"]["bridge"]["DNSNames"] = names + [
        DNS3_RUN_PREFIX + "-preserved-n34654a6d9b4bd27098dfdb633863dfe"]
    with pytest.raises(v3_dc.ConfigError, match="network_dns_names_differ"):
        v3_dc.assert_candidate_parity(snapshot, near_miss, DNS3_RUN_IMAGE)


# --- v3 supervisor: attempt-local fence default (CD-2) ----------------------
#
# The 2026-10-21 DGX rehearsal needed a one-time root mkdir/chown for the
# /var/lib/occ3-v3/attempt-fences default. The default is now inside the
# attempt directory; deployments wanting a shared host-wide fence namespace
# still pass fence_root explicitly.


def test_v3_supervisor_fence_default_is_attempt_dir_local(tmp_path, monkeypatch):
    from test_v3_lifecycle import prepared
    api, journal, services = prepared(tmp_path)
    monkeypatch.setattr(v3_sup, "process_identity",
                        lambda pid: {"pid": pid, "start_ticks": "offline"})
    guardian = v3_sup.Supervisor(api, journal, services=services, lock_root=tmp_path / "locks")
    assert guardian.fence_root == journal.directory / "fences"
    overridden = v3_sup.Supervisor(api, journal, services=services, lock_root=tmp_path / "locks",
                                   fence_root=tmp_path / "system-fences")
    assert overridden.fence_root == Path(tmp_path) / "system-fences"


def test_v3_supervisor_attempt_local_fence_needs_no_system_path(tmp_path, monkeypatch):
    from test_v3_lifecycle import approval, prepared
    api, journal, services = prepared(tmp_path)
    monkeypatch.setattr(v3_sup, "process_identity",
                        lambda pid: {"pid": pid, "start_ticks": "offline"})
    def broken_spawn():
        raise OSError("synthetic failure")
    guardian = v3_sup.Supervisor(api, journal, services=services, process_factory=broken_spawn,
                                 lock_root=tmp_path / "locks")
    assert guardian.run(approval(journal)) == 1
    fence = journal.directory / "fences" / (journal.read()["original_id"] + ".json")
    assert fence.exists()
    assert journal.read()["phase"] == "NOOP"


# --- health timeout consistency with the shell deploy lanes (PR #258) -------
#
# deploy.sh and rollback.sh default OCC3_HEALTH_TIMEOUT to 1500 s; the
# controller must apply the same default and honor the same override so no
# lane can roll back a working 13.5-minute B300 weight load.


def load_controller_fresh():
    """Exec the controller source into a brand-new module object so the
    environment-dependent constants can be exercised without disturbing the
    shared module instance the other tests import."""
    fresh_spec = importlib.util.spec_from_file_location(
        "occ3_controller_env_probe", DEPLOY_DIR / "occ3_controller.py"
    )
    module = importlib.util.module_from_spec(fresh_spec)
    fresh_spec.loader.exec_module(module)
    return module


def test_controller_health_timeout_default_matches_shell_lanes(monkeypatch):
    monkeypatch.delenv("OCC3_HEALTH_TIMEOUT", raising=False)
    assert load_controller_fresh().HEALTH_TIMEOUT_S == 1500


def test_controller_health_timeout_reads_the_shared_env_override(monkeypatch):
    monkeypatch.setenv("OCC3_HEALTH_TIMEOUT", "2400")
    assert load_controller_fresh().HEALTH_TIMEOUT_S == 2400


def test_controller_health_timeout_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OCC3_HEALTH_TIMEOUT", "not-a-number")
    assert load_controller_fresh().HEALTH_TIMEOUT_S == 1500


def test_controller_rollback_verification_budget_tracks_the_health_budget(monkeypatch):
    # The rollback lane waits for the RESTORED original to pass the same
    # health gate, so its budget must never be shorter than the health
    # budget: a slow-restoring original would false-fail under the
    # controller lane while rollback.sh (OCC3_HEALTH_TIMEOUT) passes.
    monkeypatch.delenv("OCC3_HEALTH_TIMEOUT", raising=False)
    monkeypatch.delenv("OCC3_ROLLBACK_DEADLINE_S", raising=False)
    assert load_controller_fresh().ROLLBACK_DEADLINE_S == 1500

    monkeypatch.setenv("OCC3_HEALTH_TIMEOUT", "2400")
    assert load_controller_fresh().ROLLBACK_DEADLINE_S == 2400

    # An explicit rollback deadline still wins (an operator may want a
    # tighter or looser verification lane than the health gate).
    monkeypatch.setenv("OCC3_ROLLBACK_DEADLINE_S", "1800")
    assert load_controller_fresh().ROLLBACK_DEADLINE_S == 1800

    # ...but a shorter-than-health explicit value is still honored as the
    # operator's explicit choice; the floor only applies to the default.
    monkeypatch.setenv("OCC3_ROLLBACK_DEADLINE_S", "600")
    assert load_controller_fresh().ROLLBACK_DEADLINE_S == 600


def test_watchdog_rollback_kill_covers_the_controller_verification_budget():
    # The watchdog hard-kills the controller rollback lane; it must read the
    # same health budget (plus overhead) rather than a fixed 900s that no
    # longer covers the verification wait.
    script = (DEPLOY_DIR / "occ3_watchdog.sh").read_text()
    assert 'OCC3_HEALTH_TIMEOUT="${OCC3_HEALTH_TIMEOUT:-1500}"' in script
    assert "ROLLBACK_KILL_S=$((OCC3_HEALTH_TIMEOUT + 600))" in script
    assert "timeout --signal=TERM --kill-after=60 \"$ROLLBACK_KILL_S\"" in script
