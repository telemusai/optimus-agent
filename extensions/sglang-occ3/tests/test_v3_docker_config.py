"""Offline regression tests for private Engine bodies and independent parity."""
from copy import deepcopy
import json

import pytest

from deploy.v3 import docker_config as dc

IMAGE = "sha256:" + "b" * 64
KEY = "synthetic-private-credential"


@pytest.fixture
def original():
    return {
        "Id": "a" * 64, "Name": "/original", "Image": "sha256:" + "a" * 64,
        "Config": {
            "Image": "sha256:" + "a" * 64, "Hostname": "a" * 12,
            "Entrypoint": ["python3"], "Cmd": ["-m", "sglang.launch_server",
                "--model-path", "/data/model", "--api-key", KEY,
                "--speculative-adaptive-config", "/data/adaptive.json"],
            "Env": ["A=1", "A=2", "EMPTY=", "NVIDIA_VISIBLE_DEVICES=all", "SECRET=" + KEY],
            "User": "1000:1000", "WorkingDir": "/work", "Labels": {"owner": "original"},
            "Healthcheck": {"Test": ["CMD", "health"], "Interval": 1000000},
            "ExposedPorts": {"30001/tcp": {}}, "Volumes": None,
        },
        "HostConfig": {
            "Binds": None, "Mounts": [{"Type": "bind", "Source": "/data", "Target": "/data"}],
            "NetworkMode": "serving", "Runtime": "nvidia", "IpcMode": "host",
            "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "DeviceIDs": None,
                                "Capabilities": [["gpu", "compute"]], "Options": {"key": "value"}}],
            "PortBindings": {"30001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "30001"}]},
            "RestartPolicy": {"Name": "on-failure", "MaximumRetryCount": 3},
            "SecurityOpt": ["label=disable"], "GroupAdd": ["1001"],
            "CapAdd": None, "CapDrop": None, "Privileged": False,
            "AutoRemove": False, "Memory": 987654, "NanoCpus": 123,
            "ReadonlyRootfs": True, "LogConfig": {"Type": "json-file", "Config": {"max-size": "10m"}},
            "PidsLimit": 1000, "Ulimits": [{"Name": "nofile", "Soft": 100, "Hard": 200}],
            "MaskedPaths": ["/proc/acpi"], "ReadonlyPaths": ["/proc/sys"],
        },
        "Mounts": [{"Type": "bind", "Source": "/data", "Destination": "/data",
                    "RW": True, "Mode": "", "Propagation": "rprivate"}],
        "NetworkSettings": {"Networks": {"serving": {
            "IPAMConfig": None, "Links": None, "Aliases": None, "DriverOpts": None,
            "GwPriority": 0, "NetworkID": "network-id", "EndpointID": "old-endpoint",
            "IPAddress": "172.18.0.2", "MacAddress": "02:42:ac:12:00:02",
            "DNSNames": ["original", "a" * 12],
        }}},
    }


def candidate_from(original):
    candidate = deepcopy(original)
    candidate.update(Id="c" * 64, Name="/candidate", Image=IMAGE)
    candidate["Config"]["Image"] = "candidate:tag"
    net = candidate["NetworkSettings"]["Networks"]["serving"]
    net.update(NetworkID="", EndpointID="", IPAddress="", MacAddress="", DNSNames=None)
    return candidate


def test_builder_preserves_every_config_host_field_and_credential(original):
    original["Config"]["Entrypoint"] = ["python3", "-u", "", " "]
    original["Config"]["Cmd"].extend(["--legitimate-empty", "", " "])
    before = deepcopy(original)
    body = dc.build_create_request(original, "candidate:tag", "candidate")
    assert {k: v for k, v in body.items() if k not in {"Image", "HostConfig", "NetworkingConfig"}} == {
        k: v for k, v in original["Config"].items() if k != "Image"}
    assert body["HostConfig"] == original["HostConfig"]
    assert dc.extract_api_key(original) == KEY
    assert "name" not in body and "Name" not in body
    body["Cmd"].append("change")
    body["HostConfig"]["Mounts"][0]["Source"] = "/other"
    assert original == before


def test_parity_accepts_only_documented_metadata_changes(original):
    proof = dc.assert_candidate_parity(original, candidate_from(original), IMAGE)
    assert proof["actual_mount_count"] == 1 and proof["result"] == "PASS"
    assert KEY not in json.dumps(proof) and "Cmd" not in proof


@pytest.mark.parametrize("field,value", [
    ("Entrypoint", ["python3", "-u"]), ("Cmd", ["x"]), ("Env", ["A=2", "A=1"]),
    ("WorkingDir", "/other"), ("Hostname", "candidate"), ("User", "0"),
    ("Labels", {}), ("ExposedPorts", {}), ("Healthcheck", {"Test": ["NONE"]}),
])
def test_exact_config_parity(original, field, value):
    candidate = candidate_from(original)
    candidate["Config"][field] = value
    with pytest.raises(dc.ConfigError, match="Config") as error:
        dc.assert_candidate_parity(original, candidate, IMAGE)
    assert KEY not in str(error.value)


@pytest.mark.parametrize("field,value", [
    ("Runtime", "runc"), ("IpcMode", "private"), ("Memory", 1), ("NanoCpus", 0),
    ("SecurityOpt", []), ("GroupAdd", []), ("ReadonlyRootfs", False), ("CapDrop", ["ALL"]),
    ("RestartPolicy", {"Name": "on-failure", "MaximumRetryCount": 0}),
    ("PortBindings", {}), ("Ulimits", []), ("PidsLimit", 1), ("LogConfig", {}),
    ("MaskedPaths", []), ("ReadonlyPaths", []),
])
def test_exact_host_config_parity(original, field, value):
    candidate = candidate_from(original)
    candidate["HostConfig"][field] = value
    with pytest.raises(dc.ConfigError, match="HostConfig"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


@pytest.mark.parametrize("field,value", [("Driver", ""), ("Count", 1), ("DeviceIDs", ["0"]),
                                       ("Capabilities", [["gpu"]]), ("Options", {})])
def test_device_request_full_fields(original, field, value):
    candidate = candidate_from(original)
    candidate["HostConfig"]["DeviceRequests"][0][field] = value
    with pytest.raises(dc.ConfigError, match="HostConfig"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


def test_missing_actual_mount_cannot_hide_in_false_normalizer(original, monkeypatch):
    candidate = candidate_from(original)
    candidate["Mounts"] = []
    false_normalizer = lambda doc: doc["NetworkSettings"].get("Mounts", [])
    assert false_normalizer(original) == false_normalizer(candidate) == []
    monkeypatch.setattr(dc, "build_create_request", lambda *args: {})
    monkeypatch.setattr(dc, "_declared_mounts", lambda *args: None)
    with pytest.raises(dc.ConfigError, match="actual_mount_destinations"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


@pytest.mark.parametrize("field,value", [("Source", "/wrong"), ("Destination", "/missing"),
                                       ("RW", False), ("Mode", "ro"), ("Propagation", "private")])
def test_raw_mount_fields_independent(original, field, value):
    candidate = candidate_from(original)
    candidate["Mounts"][0][field] = value
    with pytest.raises(dc.ConfigError, match="actual_mount"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


def test_declared_mount_options_must_match_even_when_raw_mounts_match(original):
    candidate = candidate_from(original)
    candidate["HostConfig"]["Mounts"][0]["BindOptions"] = {"NonRecursive": True}
    with pytest.raises(dc.ConfigError, match="HostConfig"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


@pytest.mark.parametrize("mutation", [
    lambda d: d["HostConfig"]["Mounts"][0].update(Type="tmpfs"),
    lambda d: d["HostConfig"]["Mounts"][0].update(UnknownOption=True),
    lambda d: d["HostConfig"]["Mounts"][0].update(BindOptions={"Propagation": "rshared"}),
    lambda d: d["HostConfig"].update(VolumesFrom=["other"]),
    lambda d: d["HostConfig"].update(Binds=["/x:/x:Z"]),
    lambda d: d["HostConfig"].update(UnknownField=True),
    lambda d: d["Config"].update(UnknownField=True),
])
def test_unsupported_configuration_is_refused(original, mutation):
    mutation(original)
    with pytest.raises(dc.ConfigError):
        dc.build_create_request(original, IMAGE, "candidate")


def test_legacy_binds_preserved_and_mapped(original):
    original["HostConfig"].pop("Mounts")
    original["HostConfig"]["Binds"] = ["/data:/data:ro,rprivate"]
    original["Mounts"][0].update(RW=False, Mode="ro,rprivate")
    assert dc.build_create_request(original, IMAGE, "candidate")["HostConfig"]["Binds"] == ["/data:/data:ro,rprivate"]


def test_network_static_configuration_and_aliases_preserved(original):
    net = original["NetworkSettings"]["Networks"]["serving"]
    net.update(Aliases=["original", "stable", "a" * 12],
               IPAMConfig={"IPv4Address": "172.18.0.2", "IPv6Address": "", "LinkLocalIPs": []},
               DriverOpts={"endpoint-option": "value"}, GwPriority=4,
               DNSNames=["original", "stable", "a" * 12])
    body = dc.build_create_request(original, IMAGE, "candidate")
    endpoint = body["NetworkingConfig"]["EndpointsConfig"]["serving"]
    assert endpoint["Aliases"] == ["original", "stable"]
    assert endpoint["IPAMConfig"] == net["IPAMConfig"]
    candidate = candidate_from(original)
    candidate["NetworkSettings"]["Networks"]["serving"].update(
        Aliases=["original", "stable", "candidate", "c" * 12],
        DNSNames=["original", "stable", "candidate", "c" * 12])
    dc.assert_candidate_parity(original, candidate, IMAGE)
    candidate["NetworkSettings"]["Networks"]["serving"]["IPAMConfig"]["IPv4Address"] = "172.18.0.3"
    with pytest.raises(dc.ConfigError, match="network_endpoint"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


def test_network_alias_or_network_id_change_rejected(original):
    candidate = candidate_from(original)
    net = candidate["NetworkSettings"]["Networks"]["serving"]
    net["Aliases"] = ["unexpected"]
    with pytest.raises(dc.ConfigError, match="aliases"):
        dc.assert_candidate_parity(original, candidate, IMAGE)
    net["Aliases"] = None
    net["NetworkID"] = "different-network"
    with pytest.raises(dc.ConfigError, match="network_id"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


def test_image_tag_is_not_image_identity(original):
    with pytest.raises(dc.ConfigError, match="immutable"):
        dc.assert_candidate_parity(original, candidate_from(original), "candidate:tag")
    with pytest.raises(dc.ConfigError, match="image_id"):
        dc.assert_candidate_parity(original, candidate_from(original), "sha256:" + "e" * 64)


def test_credential_extraction_equals_form_and_ambiguity(original):
    original["Config"]["Cmd"] = ["--api-key=" + KEY]
    assert dc.extract_api_key(original) == KEY
    original["Config"]["Cmd"].extend(["--api-key", KEY])
    with pytest.raises(dc.ConfigError):
        dc.extract_api_key(original)


def test_probe_is_nonserving_cpu_only_and_permission_faithful(original):
    before = deepcopy(original)
    probe = dc.build_artifact_probe_request(original, IMAGE, "probe")
    assert KEY not in json.dumps(probe)
    assert probe["Entrypoint"] == ["python3", "-I", "-S"]
    assert probe["Healthcheck"] == {"Test": ["NONE"]}
    assert probe["User"] == original["Config"]["User"]
    assert probe["WorkingDir"] == original["Config"]["WorkingDir"]
    host = probe["HostConfig"]
    for field in ("Mounts", "Binds", "GroupAdd", "SecurityOpt", "CapDrop", "ReadonlyRootfs"):
        assert host[field] == original["HostConfig"][field]
    assert host["Runtime"] == "runc" and host["DeviceRequests"] == host["Devices"] == []
    assert not host["Privileged"] and host["NetworkMode"] == "none"
    assert host["PortBindings"] == {} and host["IpcMode"] == "private"
    assert "NVIDIA_VISIBLE_DEVICES=void" in probe["Env"]
    spec = json.loads(probe["Cmd"][-1])
    assert spec["model_path"] == spec["tokenizer_path"] == "/data/model"
    assert spec["required_json"] == ["/data/adaptive.json"]
    assert original == before


@pytest.mark.parametrize("args", [
    ["--model-path", "/outside"], ["--model-path", "hf://model"],
    ["--model-path", "/data/model", "--model-path", "/data/model"],
    ["--model-path", "/data/model", "--load-format", "pt"],
    ["--model-path", "/data/model", "--config", "/data/server.yaml"],
])
def test_probe_rejects_unsupported_or_ambiguous_paths(original, args):
    original["Config"]["Cmd"] = ["-m", "sglang.launch_server", *args]
    with pytest.raises(dc.ConfigError):
        dc.build_artifact_probe_request(original, IMAGE, "probe")


def test_probe_rejects_shell_and_privilege(original):
    original["Config"]["Entrypoint"] = ["sh", "-c"]
    with pytest.raises(dc.ConfigError, match="launch_form"):
        dc.build_artifact_probe_request(original, IMAGE, "probe")
    original["Config"]["Entrypoint"] = ["python3"]
    original["HostConfig"]["Privileged"] = True
    with pytest.raises(dc.ConfigError, match="privilege"):
        dc.build_artifact_probe_request(original, IMAGE, "probe")


@pytest.mark.parametrize("port", ["", "0", "99999", "not-a-port"])
def test_dynamic_or_invalid_published_ports_rejected(original, port):
    original["HostConfig"]["PortBindings"]["30001/tcp"][0]["HostPort"] = port
    with pytest.raises(dc.ConfigError):
        dc.build_create_request(original, IMAGE, "candidate")


def test_actual_published_ports_must_match_declared_ports(original):
    original["NetworkSettings"]["Ports"] = {"30001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "30001"}]}
    with pytest.raises(dc.ConfigError, match="actual_published"):
        dc.build_create_request(original, IMAGE, "candidate")


def test_actual_mount_mode_cannot_contradict_rw(original):
    original["Mounts"][0]["Mode"] = "ro"
    with pytest.raises(dc.ConfigError, match="mode_inconsistent"):
        dc.build_create_request(original, IMAGE, "candidate")


def actual_probe_from(original):
    body = dc.build_artifact_probe_request(original, IMAGE, "probe")
    config = {key: deepcopy(value) for key, value in body.items() if key != "HostConfig"}
    return {"Image": IMAGE, "Config": config, "HostConfig": deepcopy(body["HostConfig"]),
            "State": {"Status": "created", "Running": False},
            "Mounts": deepcopy(original["Mounts"]), "NetworkSettings": {"Networks": {"none": {}}}}


def test_independent_probe_parity_accepts_actual_safe_inspect(original):
    probe = actual_probe_from(original)
    proof = dc.assert_artifact_probe_parity(original, probe, IMAGE)
    assert proof["result"] == "PASS" and proof["cpu_only_configuration"] is True
    assert KEY not in json.dumps(proof)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(Mounts=[]),
    lambda p: p["HostConfig"].update(Runtime="nvidia"),
    lambda p: p["HostConfig"].update(DeviceRequests=[{"Count": -1}]),
    lambda p: p["HostConfig"].update(Devices=[{"PathOnHost": "/dev/nvidia0"}]),
    lambda p: p["HostConfig"].update(Privileged=True),
    lambda p: p["HostConfig"].update(NetworkMode="host"),
    lambda p: p["HostConfig"].update(GroupAdd=[]),
    lambda p: p["HostConfig"].update(CapAdd=["SYS_ADMIN"]),
    lambda p: p["Config"].update(User="0"),
    lambda p: p["Config"].update(Entrypoint=["python3"]),
    lambda p: p["Config"].update(Healthcheck={"Test": ["CMD", "unsafe"]}),
    lambda p: p["Config"]["Env"].append("NVIDIA_VISIBLE_DEVICES=all"),
    lambda p: p["Config"]["Env"].append("SECRET=" + KEY),
    lambda p: p["State"].update(Status="exited"),
])
def test_probe_actual_isolation_and_mounts_not_builder_claims(original, mutation, monkeypatch):
    probe = actual_probe_from(original)
    mutation(probe)
    monkeypatch.setattr(dc, "build_artifact_probe_request", lambda *args: probe)
    with pytest.raises(dc.ConfigError) as error:
        dc.assert_artifact_probe_parity(original, probe, IMAGE)
    assert KEY not in str(error.value)


def test_probe_spec_must_match_original_not_builder_result(original, monkeypatch):
    probe = actual_probe_from(original)
    spec = json.loads(probe["Config"]["Cmd"][-1])
    spec["model_path"] = "/image-bundled-fake-model"
    probe["Config"]["Cmd"][-1] = json.dumps(spec)
    monkeypatch.setattr(dc, "build_artifact_probe_request", lambda *args: probe)
    with pytest.raises(dc.ConfigError, match="probe_spec_not_original"):
        dc.assert_artifact_probe_parity(original, probe, IMAGE)


@pytest.mark.parametrize("extra", [
    ["--quantization", "bitsandbytes"], ["--model-impl", "transformers"],
    ["--model-loader-extra-config", "{}"], ["--speculative-draft-model-path", "/data/other"],
    ["--speculative-algorithm", "NGRAM"], ["--tokenizer-mode", "slow"],
])
def test_probe_refuses_unaudited_loader_flags(original, extra):
    original["Config"]["Cmd"].extend(extra)
    with pytest.raises(dc.ConfigError):
        dc.build_artifact_probe_request(original, IMAGE, "probe")


def test_probe_accepts_valueless_enable_http2_without_widening_the_spec(original):
    plain = deepcopy(original)
    original["Config"]["Cmd"].append("--enable-http2")
    probe = dc.build_artifact_probe_request(original, IMAGE, "probe")
    plain_probe = dc.build_artifact_probe_request(plain, IMAGE, "probe")
    # Transport-only flag: the probe command and spec are identical to the
    # launch without it, and the executed command stays the isolated stdlib
    # preflight, never a serving launch. Structural checks on execution
    # positions only: the spec payload is data and may legitimately carry
    # loader names (loader_profile names an sglang profile), so it is compared
    # parsed, never substring-scanned for serving markers.
    assert probe["Entrypoint"] == ["python3", "-I", "-S"]
    assert probe["Cmd"] == plain_probe["Cmd"]
    assert len(probe["Cmd"]) == 3 and probe["Cmd"][:2] == [dc.PROBE_SCRIPT_PATH, "--spec"]
    assert "-m" not in probe["Cmd"] and "sglang.launch_server" not in probe["Cmd"]
    assert "sglang.launch_server" not in probe["Entrypoint"]
    assert json.loads(probe["Cmd"][-1]) == json.loads(plain_probe["Cmd"][-1])
    # The independent parity check accepts the real-shaped probe inspect too.
    proof = dc.assert_artifact_probe_parity(original, actual_probe_from(original), IMAGE)
    assert proof["result"] == "PASS" and proof["isolated_stdlib_command_exact"] is True


def test_candidate_clone_preserves_the_valueless_enable_http2_flag(original):
    original["Config"]["Cmd"].append("--enable-http2")
    body = dc.build_create_request(original, IMAGE, "candidate")
    # The candidate clone carries the exact original launch args, flag included.
    assert body["Cmd"] == original["Config"]["Cmd"]
    assert body["Cmd"][-1] == "--enable-http2"
    original["Path"] = "python3"
    original["Args"] = deepcopy(original["Config"]["Cmd"])
    candidate = candidate_from(original)
    dc.assert_candidate_parity(original, candidate, IMAGE)


@pytest.mark.parametrize("flag", ["--enable-otel", "--enable-http2-v2", "--http2", "--enable"])
def test_probe_still_refuses_unaudited_valueless_flags(original, flag):
    original["Config"]["Cmd"].append(flag)
    with pytest.raises(dc.ConfigError):
        dc.build_artifact_probe_request(original, IMAGE, "probe")


def test_framed_enable_http2_is_gate_accepted_and_spec_inert(original):
    # The allowlist gate is name-based (x.split("=", 1)[0]), so the framed
    # spelling passes exactly like --model-path=/x always has; no _flag query
    # reads it, so the spec stays identical. Documents the boundary honestly.
    plain = deepcopy(original)
    original["Config"]["Cmd"].append("--enable-http2=off")
    probe = dc.build_artifact_probe_request(original, IMAGE, "probe")
    assert probe["Cmd"] == dc.build_artifact_probe_request(plain, IMAGE, "probe")["Cmd"]


def test_independent_parity_still_refuses_unaudited_valueless_flags(original, monkeypatch):
    probe = actual_probe_from(original)
    original["Config"]["Cmd"].append("--enable-otel")
    monkeypatch.setattr(dc, "build_artifact_probe_request", lambda *args: probe)
    with pytest.raises(dc.ConfigError, match="unsupported_external_launch_configuration"):
        dc.assert_artifact_probe_parity(original, probe, IMAGE)


def test_effective_path_and_args_cannot_differ_from_original(original):
    original["Path"] = "python3"
    original["Args"] = deepcopy(original["Config"]["Cmd"])
    candidate = candidate_from(original)
    candidate["Args"].insert(0, "")
    with pytest.raises(dc.ConfigError, match="effective_launch"):
        dc.assert_candidate_parity(original, candidate, IMAGE)


@pytest.mark.parametrize("variable", ["LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "GCONV_PATH", "BASH_ENV"])
def test_actual_probe_inherited_executable_hooks_rejected(original, variable):
    probe = actual_probe_from(original)
    probe["Config"]["Env"] = [item for item in probe["Config"]["Env"] if not item.startswith(variable + "=")]
    probe["Config"]["Env"].append(variable + "=/private/hook")
    with pytest.raises(dc.ConfigError, match="environment"):
        dc.assert_artifact_probe_parity(original, probe, IMAGE)


def test_probe_extra_environment_must_be_safe_and_match_original(original):
    original["Config"]["Env"].append("LANG=C.UTF-8")
    probe = actual_probe_from(original)
    probe["Config"]["Env"].append("LANG=C.UTF-8")
    dc.assert_artifact_probe_parity(original, probe, IMAGE)
    probe["Config"]["Env"].append("LD_DEBUG=libs")
    with pytest.raises(dc.ConfigError, match="inherited"):
        dc.assert_artifact_probe_parity(original, probe, IMAGE)


@pytest.mark.parametrize("change", [
    {"Env": ["LD_PRELOAD=/private.so"]}, {"Env": ["LD_AUDIT=/private.so"]},
    {"Env": ["PYTHONPATH=/private"]}, {"Env": ["PYTHONSTARTUP=/private.py"]},
    {"Env": ["LD_DEBUG=libs"]}, {"Volumes": {"/model": {}}}, {"OnBuild": ["RUN unsafe"]},
])
def test_probe_image_metadata_refuses_hooks_volumes_onbuild(original, change):
    image = {"Id": IMAGE, "Config": {"Env": [], **change}}
    with pytest.raises(dc.ConfigError) as error:
        dc.assert_probe_image(image, IMAGE, original)
    assert "/private" not in str(error.value)


def test_probe_image_baseline_metadata_vetted_without_changing_serving_env(original):
    original["Config"]["Env"].extend(["LANG=C.UTF-8", "LD_LIBRARY_PATH=/approved/base/lib"])
    before = deepcopy(original)
    image = {"Id": IMAGE, "Config": {"Env": ["LANG=C.UTF-8", "LD_LIBRARY_PATH=/approved/base/lib"], "Volumes": None, "OnBuild": None}}
    assert dc.assert_probe_image(image, IMAGE, original)["environment_vetted"] is True
    probe = dc.build_artifact_probe_request(original, IMAGE, "probe")
    for key in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
        assert key + "=" in probe["Env"]
    assert dc.build_create_request(original, IMAGE, "candidate")["Env"] == before["Config"]["Env"]
    assert original == before


def test_public_probe_environment_returns_independent_fixed_overrides():
    first = dc.probe_environment()
    first.append("EXTRA=mutation")
    assert "EXTRA=mutation" not in dc.probe_environment()
    assert {"LD_PRELOAD=", "LD_AUDIT=", "LD_LIBRARY_PATH=", "PYTHONPATH=", "NVIDIA_VISIBLE_DEVICES=void"} <= set(dc.probe_environment())
