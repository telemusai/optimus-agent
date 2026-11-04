"""Offline build-driver and Dockerfile tests; no Docker, network or GPU.

The offline pip install is intercepted at the driver's subprocess seam and
never executed here: the real pip run happens only inside a future
root-approved container build stage. The isolated-verify tests execute the
driver's real child script with throwaway interpreter environments, which is
offline evidence about the ``-I`` staging-shadow boundary, not an image,
wheel or runtime proof.
"""
import ast
import base64
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from deploy.v3 import native_image_build as nib
from deploy.v3 import native_package as package


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))


PROFILE = {"model_fingerprint": "fixture-fingerprint-not-production", "model_id": "fixture-model",
           "expected_ranks": 8, "hidden_dim": 6144, "layer_index": 77,
           "max_prefix_tokens": 128, "max_live_bytes": 256 * 1024 ** 2,
           "workspace_bytes": 128 * 1024 ** 2, "max_records": 256, "max_active": 1,
           "max_ttl_seconds": 3600, "anchor_alpha": 0.0, "tile_rows": 256,
           "method": package.METHOD}
MANIFEST_TARGETS = {"capture-eager-mtp.v1.json": "a.py",
                    "native-startup.v2.json": "http_server.py",
                    "scheduler-cache-ipc.v1.json": "sub/b.py"}


class Fixture:
    """Synthetic disposable trees mirroring the real build inputs."""

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.sg, self.plugin = root / "srt", root / "occ3_sglang"
        self.sg.mkdir()
        self.plugin.mkdir()
        (self.sg / "sub").mkdir()
        self.originals = {"a.py": b"VALUE = 1\n", "sub/b.py": b"VALUE = 2\n",
                          "http_server.py": b"VALUE = 3\n", "untouched.py": b"VALUE = 4\n"}
        for path, raw in self.originals.items():
            (self.sg / path).write_bytes(raw)
        # CRLF bytes in one plugin source must survive staging, install and proof.
        self.plugin_sources = {"__init__.py": b"raise RuntimeError('plugin must never be imported')\n",
                               "native_profile.py": b"import torch\r\nVALUE = 1\r\n",
                               "native_install.py": b"import torch\nVALUE = 3\n",
                               "plugin.py": b"import sglang\nVALUE = 2\n",
                               "startup_state.py": b"VALUE = 4\n"}
        for path, raw in self.plugin_sources.items():
            (self.plugin / path).write_bytes(raw)
        self.staged = root / "native-build-inputs"
        self.staged.mkdir()
        write_json(self.staged / "source-baseline-expanded.json",
                   {"schema": package.BASELINE_SCHEMA, "source_revision": package.SOURCE_REVISION,
                    "sglang_version": package.SOURCE_VERSION,
                    "files": [{"path": path, "sha256": digest(raw)}
                              for path, raw in sorted(self.originals.items())]})
        write_json(self.staged / "native-profile-input.json", PROFILE)
        for name, path in MANIFEST_TARGETS.items():
            original = self.originals[path]
            patched = original.replace(b"VALUE", b"PATCHED")
            write_json(self.staged / name,
                       {"schema": package.source_patch.SCHEMA,
                        "source_revision": package.SOURCE_REVISION,
                        "source_version": package.SOURCE_VERSION,
                        "patch_set": "fixture-" + name.split(".")[0],
                        "files": [{"path": path, "before_sha256": digest(original),
                                   "after_sha256": digest(patched),
                                   "replacements": [{"before": "VALUE", "after": "PATCHED"}]}]})

    def arguments(self, **overrides):
        values = {"--base-image": "occ3-original-working:local",
                  "--sglang-srt": str(self.sg),
                  "--occ3-sglang": str(self.plugin),
                  "--staged-inputs": str(self.staged),
                  "--experimental-startup": None,
                  "--disposable-tree": True}
        values.update(overrides)
        argv = []
        for flag, value in values.items():
            if value is None:
                continue
            argv.append(flag)
            if value is not True:
                argv.append(value)
        return argv

    def artifacts_present(self):
        return all((self.plugin / name).is_file() for name in package.OUTPUT_NAMES)


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


class FakeExecute:
    """Subprocess seam double; no pip, network or child process runs here."""

    def __init__(self, tools=None, pip_code=0, verify_receipt=None, tamper=None):
        self.calls = []
        self.pip_artifacts_present = None
        self.tools = {"pip": "24.0", "setuptools": "75.1.0"} if tools is None else tools
        self.pip_code = pip_code
        self.receipt = {"ok": True, "mode": "isolated-interpreter", "isolated": True,
                        "installed_package_files": 7, "installed_sglang_files": 4,
                        "sglang_version": package.SOURCE_VERSION,
                        "distribution": "occ3-bridge==0.1.0",
                        "entry_point": "occ3_bridge=occ3_sglang.plugin:load",
                        "runtime_imported": False, "gpu_proof": False,
                        "readiness_proof": False} if verify_receipt is None else verify_receipt
        self.tamper = tamper

    def __call__(self, argv, *, timeout, env=None, cwd=None):
        self.calls.append(list(argv))
        if self.tamper is not None:
            self.tamper(argv)
        if argv[1:3] == ["-I", "-c"] and argv[3] == nib.TOOLS_PROBE_SOURCE:
            return 0, json.dumps(self.tools) + "\n", ""
        if argv[1:3] == ["-m", "pip"]:
            # Intercepted boundary: the container build stage alone runs pip.
            return self.pip_code, "", "synthetic pip boundary: never executed offline"
        if len(argv) > 3 and argv[3] == nib.ISOLATED_VERIFY_SOURCE:
            return 0, json.dumps(self.receipt) + "\n", ""
        raise AssertionError("unexpected subprocess argv " + repr(argv[:4]))

    def kinds(self):
        kinds = []
        for argv in self.calls:
            if argv[1:3] == ["-I", "-c"] and argv[3] == nib.TOOLS_PROBE_SOURCE:
                kinds.append("tools")
            elif len(argv) > 3 and argv[3] == nib.ISOLATED_VERIFY_SOURCE:
                kinds.append("verify")
            else:
                kinds.append("other")
        return kinds


def pip_argv(project):
    return [sys.executable, "-m", "pip", "install", *nib.PIP_INSTALL_FLAGS, str(project)]


def run_build(fixture, monkeypatch, fake):
    monkeypatch.setattr(nib, "_execute", fake)
    receipt = nib.build_layer(fixture.arguments())
    fake.pip_index = next((i for i, argv in enumerate(fake.calls)
                           if argv[1:3] == ["-m", "pip"]), None)
    if fake.pip_index is not None:
        fake.pip_artifacts_present = fixture.artifacts_present()
    return receipt


def test_offline_build_receipt_reports_staged_inputs_tools_and_install(fixture, monkeypatch):
    fake = FakeExecute(tamper=lambda argv: None)
    monkeypatch.setattr(nib, "_execute", fake)
    receipt = nib.build_layer(fixture.arguments())
    assert receipt["schema"] == nib.SCHEMA and receipt["ok"] is True
    assert receipt["layer_only"] is True and receipt["docker_operation"] is False
    assert receipt["base_image"] == "occ3-original-working:local"
    assert receipt["experimental_startup_selected"] is False
    assert receipt["runtime_imported"] is receipt["gpu_proof"] is receipt["readiness_proof"] is False
    names = set(nib.STAGED_NAMES)
    assert set(receipt["input_receipt"]) == names
    for name in names:
        raw = (fixture.staged / name).read_bytes()
        assert receipt["input_receipt"][name] == {"sha256": digest(raw), "bytes": len(raw)}
    assert receipt["staged"] == {"sglang_srt_root": str(fixture.sg),
                                 "occ3_sglang_root": str(fixture.plugin),
                                 "plugin_source_files": 5, "sg_source_files": 4,
                                 "patched_source_files": 3}
    assert receipt["install"] == {"flags": list(nib.PIP_INSTALL_FLAGS), "offline": True,
                                 "index": False, "build_isolation": False,
                                 "network_fallback": False,
                                 "bytecode_compilation": False,
                                 "project_root": str(fixture.root)}
    assert receipt["build_tools"]["pip"] == "24.0"
    assert receipt["build_tools"]["setuptools"] == "75.1.0"
    assert receipt["installed"]["mode"] == "isolated-interpreter"
    assert receipt["package_receipt"]["applied"] is True
    assert receipt["package_receipt"]["experimental_startup_selected"] is False
    # The overlay and generated artifacts exist in the staged tree.
    assert fixture.artifacts_present()
    assert (fixture.sg / "a.py").read_bytes() == b"PATCHED = 1\n"
    assert (fixture.sg / "untouched.py").read_bytes() == fixture.originals["untouched.py"]
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse((fixture.plugin / "_native_profile_digest.py").read_bytes()).body}
    assert assignments["EXPERIMENTAL_STARTUP"] is False


def test_experimental_startup_requires_exact_true_string(fixture, monkeypatch):
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    receipt = nib.build_layer(fixture.arguments(**{"--experimental-startup": "true"}))
    assert receipt["experimental_startup_selected"] is True
    assert receipt["package_receipt"]["experimental_startup_selected"] is True
    module = ast.parse((fixture.plugin / "_native_profile_digest.py").read_bytes())
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body}
    assert assignments["EXPERIMENTAL_STARTUP"] is True
    for invalid in ("1", "yes", "TRUE", "True", ""):
        with pytest.raises(nib.BuildError, match="invalid_experimental_startup_flag"):
            nib.parse_options(fixture.arguments(**{"--experimental-startup": invalid}))


def test_parse_rejects_missing_disposable_tree_and_bad_values(fixture):
    with pytest.raises(nib.BuildError, match="explicit_disposable_tree_required"):
        nib.parse_options(fixture.arguments(**{"--disposable-tree": None}))
    with pytest.raises(nib.BuildError, match="invalid_arguments"):
        nib.parse_options(fixture.arguments(**{"--base-image": None}))
    for invalid in ("has space", "tag$bad", ""):
        with pytest.raises(nib.BuildError, match="invalid_base_image"):
            nib.parse_options(fixture.arguments(**{"--base-image": invalid}))
    with pytest.raises(nib.BuildError, match="invalid_path_argument"):
        nib.parse_options(fixture.arguments(**{"--sglang-srt": "relative/path"}))


def test_staged_inputs_directory_is_exact_and_closed(fixture, monkeypatch):
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    (fixture.staged / "extra.json").write_bytes(b"{}")
    with pytest.raises(nib.BuildError, match="staged_input_unexpected"):
        nib.build_layer(fixture.arguments())
    (fixture.staged / "extra.json").unlink()
    (fixture.staged / "sub").mkdir()
    with pytest.raises(nib.BuildError, match="staged_input_unexpected"):
        nib.build_layer(fixture.arguments())
    (fixture.staged / "sub").rmdir()
    (fixture.staged / "native-startup.v2.json").unlink()
    with pytest.raises(nib.BuildError, match="staged_input_missing"):
        nib.build_layer(fixture.arguments())
    (fixture.staged / "native-startup.v2.json").mkdir()
    with pytest.raises(nib.BuildError, match="staged_input_unsupported"):
        nib.build_layer(fixture.arguments())
    (fixture.staged / "native-startup.v2.json").rmdir()
    (fixture.staged / "native-startup.v2.json").write_bytes(b"[]")
    with pytest.raises(package.PackageError):
        nib.build_layer(fixture.arguments())


def test_staged_input_symlink_and_size_are_refused(fixture, monkeypatch):
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    target = fixture.staged / "native-profile-input.json"
    raw = target.read_bytes()
    target.unlink()
    (fixture.root / "elsewhere.json").write_bytes(raw)
    target.symlink_to(fixture.root / "elsewhere.json")
    with pytest.raises(nib.BuildError, match="staged_input_unsupported"):
        nib.build_layer(fixture.arguments())
    target.unlink()
    target.write_bytes(raw)
    (fixture.staged / "source-baseline-expanded.json").write_bytes(b" " * (package.MAX_BASELINE_BYTES + 1))
    with pytest.raises(nib.BuildError, match="staged_input_oversize"):
        nib.build_layer(fixture.arguments())
    (fixture.staged / "source-baseline-expanded.json").write_bytes(b"{}")
    (fixture.staged / "native-profile-input.json").write_bytes(b" " * (package.MAX_PROFILE_BYTES + 1))
    with pytest.raises(nib.BuildError, match="staged_input_oversize"):
        nib.build_layer(fixture.arguments())


def test_offline_pip_command_is_exact_and_intercepted_at_the_boundary(fixture, monkeypatch):
    """The pip argv is asserted here; the container build stage alone runs it."""
    seen = []

    def record(argv):
        if argv[1:3] == ["-m", "pip"]:
            seen.append(list(argv))

    fake = FakeExecute(tamper=record)
    receipt = run_build(fixture, monkeypatch, fake)
    assert receipt["ok"] is True
    assert seen == [pip_argv(fixture.root)]
    command = " ".join(seen[0])
    assert "--no-compile" in command
    for forbidden in ("--index-url", "--find-links", "--requirement", "http://",
                      "https://", "--build-isolation", "--user", "--target"):
        assert forbidden not in command


def test_package_install_happens_after_overlay_and_artifacts(fixture, monkeypatch):
    fake = FakeExecute()
    receipt = run_build(fixture, monkeypatch, fake)
    assert receipt["ok"] is True and fake.pip_index is not None
    # The generated artifacts must already exist when pip runs, so the wheel
    # carries the packaged JSON data and digest module.
    assert fake.pip_artifacts_present is True
    assert fake.kinds() == ["tools", "other", "verify"]


def test_missing_old_or_broken_build_tools_fail_before_pip(fixture, monkeypatch):
    for tools, code in (({"pip": None, "setuptools": "75.0"}, "build_tools_missing"),
                       ({"pip": "24.0", "setuptools": None}, "build_tools_missing"),
                       ({"pip": "20.3", "setuptools": "75.0"}, "build_tool_version_unsupported"),
                       ({"pip": "24.0", "setuptools": "63.9"}, "build_tool_version_unsupported"),
                       ({"pip": "x.y", "setuptools": "75.0"}, "build_tool_version_unsupported")):
        fake = FakeExecute(tools=tools)
        monkeypatch.setattr(nib, "_execute", fake)
        with pytest.raises(nib.BuildError, match=code):
            nib.build_layer(fixture.arguments())
        assert fake.kinds() == ["tools"]


def test_failed_pip_is_a_fixed_code_and_leaves_no_activation(tmp_path, monkeypatch, capsys):
    unselected = Fixture(tmp_path / "unselected")
    monkeypatch.setattr(nib, "_execute", FakeExecute(pip_code=1))
    assert nib.main(unselected.arguments()) == 1
    result = json.loads(capsys.readouterr().out.strip())
    assert result == {"ok": False, "schema": nib.SCHEMA, "code": "pip_install_failed",
                      "runtime_imported": False, "gpu_proof": False, "readiness_proof": False}
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node
                   in ast.parse((unselected.plugin / "_native_profile_digest.py").read_bytes()).body}
    assert assignments["EXPERIMENTAL_STARTUP"] is False
    # An explicitly selected experimental build also fails closed: no
    # verification runs and the disposable layer is discarded.
    selected = Fixture(tmp_path / "selected")
    fake = FakeExecute(pip_code=1)
    monkeypatch.setattr(nib, "_execute", fake)
    assert nib.main(selected.arguments(**{"--experimental-startup": "true"})) == 1
    assert "verify" not in fake.kinds()
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node
                   in ast.parse((selected.plugin / "_native_profile_digest.py").read_bytes()).body}
    assert assignments["EXPERIMENTAL_STARTUP"] is True


@pytest.mark.parametrize("receipt", ({"ok": False, "code": "installed_bytes_mismatch"},
                                     {"ok": True, "unexpected": "shape"}, "not-json"))
def test_verify_failure_is_a_fixed_code(fixture, monkeypatch, receipt):
    monkeypatch.setattr(nib, "_execute", FakeExecute(verify_receipt=receipt))
    with pytest.raises(nib.BuildError, match="installed_verification_failed"):
        nib.build_layer(fixture.arguments())


def test_no_runtime_imports_or_installs_during_offline_build(fixture, monkeypatch):
    original_import = builtins.__import__

    def refuse_runtime(name, *args, **kwargs):
        assert not name.startswith(("sglang", "torch", "occ3_sglang")), "runtime import attempted"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_runtime)
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    receipt = nib.build_layer(fixture.arguments())
    assert receipt["ok"] is True


def test_package_errors_surface_as_fixed_prefixed_codes(fixture, monkeypatch, capsys):
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    baseline = json.loads((fixture.staged / "source-baseline-expanded.json").read_bytes())
    baseline["files"][0]["sha256"] = "0" * 64
    write_json(fixture.staged / "source-baseline-expanded.json", baseline)
    assert nib.main(fixture.arguments()) == 1
    line = capsys.readouterr().out.strip().splitlines()
    assert len(line) == 1
    result = json.loads(line[0])
    assert result == {"ok": False, "schema": nib.SCHEMA,
                      "code": "package_baseline_source_digest_mismatch",
                      "runtime_imported": False, "gpu_proof": False, "readiness_proof": False}


def test_spec_composition_binds_installed_and_source_bytes(fixture, monkeypatch):
    monkeypatch.setattr(nib, "_execute", FakeExecute())
    plan = package.prepare(fixture.sg, fixture.plugin,
                           fixture.staged / "source-baseline-expanded.json",
                           fixture.staged / "native-profile-input.json",
                           [fixture.staged / name for name in nib.MANIFEST_NAMES])
    package.apply(plan, disposable=True)
    spec = nib.compose_verification_spec(plan, fixture.sg)
    assert set(spec) == {"dist", "package_files", "project_root", "sglang_root",
                         "sglang_files", "sglang_version"}
    assert spec["dist"] == {"name": "occ3-bridge", "version": "0.1.0",
                            "entry_group": "sglang.srt.plugins",
                            "entry_name": "occ3_bridge",
                            "entry_value": "occ3_sglang.plugin:load"}
    assert spec["sglang_version"] == package.SOURCE_VERSION
    assert spec["project_root"] == str(fixture.root)
    assert set(spec["package_files"]) == (set(fixture.plugin_sources) | set(package.OUTPUT_NAMES))
    for name, value in spec["package_files"].items():
        raw = (fixture.plugin / name).read_bytes()
        assert value == digest(raw)
    final = {entry["path"]: entry["sha256"] for entry in spec["sglang_files"]}
    assert set(final) == set(fixture.originals)
    for path, value in final.items():
        raw = (fixture.sg / path).read_bytes()
        assert value == digest(raw)
    assert final["a.py"] != digest(fixture.originals["a.py"])
    assert final["untouched.py"] == digest(fixture.originals["untouched.py"])


def test_oversized_spec_is_refused_before_any_child_process():
    spec = {"dist": {}, "package_files": {"a%d.py" % index: "0" * 64 for index in range(40000)},
            "sglang_root": "/x", "sglang_files": [], "sglang_version": "0.5.21"}
    with pytest.raises(nib.BuildError, match="spec_oversize"):
        nib.run_isolated_verify(sys.executable, spec)


def test_driver_constants_pin_the_finite_interface():
    assert nib.MANIFEST_NAMES == ("capture-eager-mtp.v1.json", "native-startup.v2.json",
                                  "scheduler-cache-ipc.v1.json")
    assert nib.STAGED_NAMES == ("capture-eager-mtp.v1.json", "native-profile-input.json",
                                "native-startup.v2.json", "scheduler-cache-ipc.v1.json",
                                "source-baseline-expanded.json")
    assert nib.PIP_INSTALL_FLAGS == ("--no-index", "--no-deps", "--no-cache-dir",
                                     "--no-build-isolation", "--no-compile",
                                     "--disable-pip-version-check")
    assert nib.ALLOWED_DIST_INFO == ("RECORD", "METADATA", "WHEEL", "entry_points.txt",
                                     "top_level.txt", "INSTALLER", "REQUESTED", "direct_url.json")
    assert nib.REQUIRED_DIST_INFO == ("RECORD", "METADATA", "WHEEL", "INSTALLER", "direct_url.json")
    assert nib.INSTALLER_CONTENT == b"pip\n"
    assert nib.ENTRY_VALUE == "occ3_sglang.plugin:load"
    assert nib.ENTRY_GROUP == "sglang.srt.plugins" and nib.ENTRY_NAME == "occ3_bridge"


def test_driver_source_imports_stdlib_and_packaging_only():
    tree = ast.parse(Path(nib.__file__).read_bytes())
    allowed = {"argparse", "hashlib", "json", "os", "re", "stat", "subprocess", "sys",
               "tempfile", "pathlib", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root in allowed | {"deploy", ""}
            if node.module in ("deploy.v3.native_package", "deploy.v3.source_patch"):
                assert node.level == 0
    source = Path(nib.__file__).read_text(encoding="utf-8")
    for forbidden in ("--index-url", "--find-links", "--extra-index-url", "https://"):
        assert forbidden not in source


def test_child_script_is_static_and_read_only():
    compile(nib.ISOLATED_VERIFY_SOURCE, "<isolated-verify>", "exec")
    tree = ast.parse(nib.ISOLATED_VERIFY_SOURCE)
    allowed = {"base64", "hashlib", "json", "os", "stat", "sys", "sysconfig",
               "importlib", "pathlib"}
    forbidden_names = {"open", "exec", "eval", "compile", "__import__", "input",
                       "globals", "vars", "setattr", "delattr"}
    forbidden_attributes = {"write_bytes", "write_text", "unlink", "mkdir", "rmdir", "rename",
                             "replace", "symlink_to", "chmod", "touch", "system", "popen",
                             "run", "call", "check_call", "check_output"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes


# --- Real isolated-interpreter verification in throwaway environments ---


def create_venv(root):
    target = root / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)],
                   check=True, capture_output=True, timeout=300)
    return target


def venv_interpreter(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def venv_purelib(venv):
    result = subprocess.run([str(venv_interpreter(venv)), "-I", "-c",
                             "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                            check=True, capture_output=True, text=True, timeout=120)
    return Path(result.stdout.strip())


def record_row(path, raw):
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode("ascii")
    return "%s,sha256=%s,%d" % (path, encoded, len(raw))


def install_synthetic(venv, plan, project_root, *, sglang_version=package.SOURCE_VERSION):
    """Model the exact layout a real offline ``pip install <local-dir>``
    lays down: wheel data files, the three installer-generated files, real
    RECORD hashes for every generated file and the hashless RECORD self-row.
    No bytecode, matching --no-compile."""
    site = venv_purelib(venv)
    package_dir = site / "occ3_sglang"
    package_dir.mkdir()
    for relative, record in plan.plugin_files:
        (package_dir / relative).write_bytes(record.raw)
    for name, raw in plan.artifacts:
        (package_dir / name).write_bytes(raw)
    dist_info = site / nib.DIST_INFO
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: occ3-bridge\nVersion: 0.1.0\n", encoding="utf-8")
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: synthetic-offline\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nocc3_bridge = occ3_sglang.plugin:load\n", encoding="utf-8")
    (dist_info / "top_level.txt").write_text("occ3_sglang\n", encoding="utf-8")
    # Installer-generated files exactly as pip writes them for a local
    # directory install: INSTALLER unconditionally, REQUESTED for a
    # user-supplied requirement, direct_url.json with an empty dir_info and
    # the local project file:// URL, no trailing newline.
    (dist_info / "INSTALLER").write_bytes(nib.INSTALLER_CONTENT)
    (dist_info / "REQUESTED").write_bytes(b"")
    (dist_info / "direct_url.json").write_bytes(
        json.dumps({"dir_info": {}, "url": "file://" + str(project_root)},
                   sort_keys=True).encode("utf-8"))
    rows = [record_row("occ3_sglang/" + relative, record.raw)
            for relative, record in plan.plugin_files]
    rows.extend(record_row("occ3_sglang/" + name, raw) for name, raw in plan.artifacts)
    for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt",
                 "INSTALLER", "REQUESTED", "direct_url.json"):
        rows.append(record_row("%s/%s" % (nib.DIST_INFO, name),
                              (dist_info / name).read_bytes()))
    rows.append("%s/RECORD,," % nib.DIST_INFO)
    (dist_info / "RECORD").write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")
    sglang_info = site / ("sglang-%s.dist-info" % sglang_version)
    sglang_info.mkdir()
    (sglang_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sglang\nVersion: %s\n" % sglang_version, encoding="utf-8")
    return site


def rewrite_dist_info_file(dist_info, name, raw):
    """Tamper helper: rewrite one dist-info file AND keep its RECORD row
    consistent, so the content check (not the hash check) is exercised."""
    (dist_info / name).write_bytes(raw)
    rows = [row for row in (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
            if row and not row.startswith("%s/%s," % (nib.DIST_INFO, name))]
    rows.append(record_row("%s/%s" % (nib.DIST_INFO, name), raw))
    (dist_info / "RECORD").write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")


def prepared_plan(fixture):
    return package.prepare(fixture.sg, fixture.plugin,
                           fixture.staged / "source-baseline-expanded.json",
                           fixture.staged / "native-profile-input.json",
                           [fixture.staged / name for name in nib.MANIFEST_NAMES])


def child_result(interpreter, spec, *, cwd=None, env=None):
    raw = nib._canonical(spec)
    handle, path = tempfile.mkstemp(prefix="occ3-child-", suffix=".json")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
        result = subprocess.run([str(interpreter), "-I", "-c", nib.ISOLATED_VERIFY_SOURCE, path],
                                capture_output=True, text=True, cwd=cwd, env=env, timeout=300)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines, result.stderr
        return json.loads(lines[-1]), result.returncode
    finally:
        os.unlink(path)


def test_real_isolated_verify_proves_the_installed_tree(tmp_path):
    fixture = Fixture(tmp_path)
    plan = prepared_plan(fixture)
    package.apply(plan, disposable=True)
    venv = create_venv(tmp_path)
    site = install_synthetic(venv, plan, fixture.root)
    spec = nib.compose_verification_spec(plan, fixture.sg)
    receipt = nib.run_isolated_verify(venv_interpreter(venv), spec)
    assert receipt["ok"] is True and receipt["mode"] == "isolated-interpreter"
    assert receipt["installed_package_files"] == len(spec["package_files"]) == 8
    assert receipt["installed_sglang_files"] == 4
    assert receipt["sglang_version"] == package.SOURCE_VERSION
    assert receipt["entry_point"] == "occ3_bridge=occ3_sglang.plugin:load"
    assert receipt["runtime_imported"] is receipt["gpu_proof"] is False
    # Generated JSON package data is installed byte-exact, including the
    # plugin source that carries CRLF bytes.
    artifacts = dict(plan.artifacts)
    assert (site / "occ3_sglang" / "native_profile.json").read_bytes() == artifacts["native_profile.json"]
    assert (site / "occ3_sglang" / "native_profile.py").read_bytes() == fixture.plugin_sources["native_profile.py"]


@pytest.mark.parametrize("tamper,code", [
    ("installed-byte", "installed_bytes_mismatch"),
    ("installed-missing", "installed_membership_mismatch"),
    ("installed-extra", "installed_membership_mismatch"),
    ("installed-symlink", "installed_file_unsupported"),
    ("oversize-installed-file", "file_oversize"),
    ("installer-wrong-content", "installer_invalid"),
    ("installer-missing", "installer_metadata_missing"),
    ("requested-nonempty", "requested_invalid"),
    ("direct-url-scheme", "direct_url_invalid"),
    ("direct-url-editable", "direct_url_invalid"),
    ("direct-url-wrong-root", "direct_url_invalid"),
    ("direct-url-stale-record", "record_hash_mismatch"),
    ("dist-info-stray", "record_unexpected_dist_info_entry"),
    ("record-hash", "record_hash_mismatch"),
    ("record-hash-missing", "record_hash_missing"),
    ("record-duplicate-row", "record_duplicate_entry"),
    ("record-traversal-row", "record_invalid"),
    ("record-extra", "record_unexpected_entry"),
    ("record-missing", "record_missing_package_entry"),
    ("entry-point", "entry_point_mismatch"),
    ("dist-version", "distribution_identity_mismatch"),
    ("duplicate-dist", "duplicate_distribution"),
    ("sglang-missing", "sglang_distribution_missing"),
    ("sglang-version", "sglang_version_mismatch"),
    ("sglang-source", "sglang_source_mismatch"),
])
def test_real_isolated_verify_detects_installed_and_source_tampering(tmp_path, tamper, code):
    fixture = Fixture(tmp_path)
    plan = prepared_plan(fixture)
    package.apply(plan, disposable=True)
    venv = create_venv(tmp_path)
    site = install_synthetic(venv, plan, fixture.root)
    package_dir = site / "occ3_sglang"
    dist_info = site / nib.DIST_INFO
    if tamper == "installed-byte":
        (package_dir / "plugin.py").write_bytes(b"tampered\n")
    elif tamper == "installed-missing":
        (package_dir / "plugin.py").unlink()
    elif tamper == "installed-extra":
        (package_dir / "stray.pyc").write_bytes(b"junk")
    elif tamper == "installed-symlink":
        target = package_dir / "plugin.py"
        raw = target.read_bytes()
        target.unlink()
        (tmp_path / "outside.bin").write_bytes(raw)
        target.symlink_to(tmp_path / "outside.bin")
    elif tamper == "oversize-installed-file":
        # Reduced-bound oversize proof: the child refuses before reading.
        (package_dir / "plugin.py").write_bytes(b"x" * (nib.MAX_FILE_BYTES + 1))
    elif tamper == "installer-wrong-content":
        rewrite_dist_info_file(dist_info, "INSTALLER", b"pip")
    elif tamper == "installer-missing":
        (dist_info / "INSTALLER").unlink()
    elif tamper == "requested-nonempty":
        rewrite_dist_info_file(dist_info, "REQUESTED", b"junk")
    elif tamper == "direct-url-scheme":
        rewrite_dist_info_file(dist_info, "direct_url.json",
                               json.dumps({"dir_info": {}, "url": "http://evil/x"},
                                          sort_keys=True).encode("utf-8"))
    elif tamper == "direct-url-editable":
        rewrite_dist_info_file(dist_info, "direct_url.json",
                               json.dumps({"dir_info": {"editable": True},
                                           "url": "file://" + str(fixture.root)},
                                          sort_keys=True).encode("utf-8"))
    elif tamper == "direct-url-wrong-root":
        rewrite_dist_info_file(dist_info, "direct_url.json",
                               json.dumps({"dir_info": {},
                                           "url": "file://" + str(tmp_path / "other")},
                                          sort_keys=True).encode("utf-8"))
    elif tamper == "direct-url-stale-record":
        # Content tampered without updating RECORD: the hash binding fires.
        (dist_info / "direct_url.json").write_bytes(
            json.dumps({"dir_info": {}, "url": "file://" + str(tmp_path / "other")},
                       sort_keys=True).encode("utf-8"))
    elif tamper == "dist-info-stray":
        (dist_info / "rogue.txt").write_bytes(b"stray")
    elif tamper == "record-hash-missing":
        rows = (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
        for index, row in enumerate(rows):
            if row.startswith("occ3_sglang/plugin.py,"):
                rows[index] = row.split(",")[0] + ","
                break
        (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif tamper == "record-duplicate-row":
        rows = (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
        rows.append(rows[0])
        (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif tamper == "record-traversal-row":
        with (dist_info / "RECORD").open("a", encoding="utf-8") as stream:
            stream.write("../evil.py,,\n")
    elif tamper == "record-hash":
        rows = (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
        parts = rows[0].split(",")
        parts[1] = "sha256=" + base64.urlsafe_b64encode(b"\x01" * 32).rstrip(b"=").decode("ascii")
        rows[0] = ",".join(parts)
        (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif tamper == "record-extra":
        with (dist_info / "RECORD").open("a", encoding="utf-8") as stream:
            stream.write("outside/rogue.py,sha256=%s,1\n" % ("A" * 43))
    elif tamper == "record-missing":
        rows = [row for row in (dist_info / "RECORD").read_text(encoding="utf-8").splitlines()
                if not row.startswith("occ3_sglang/plugin.py")]
        (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif tamper == "entry-point":
        (dist_info / "entry_points.txt").write_text(
            "[sglang.srt.plugins]\nocc3_bridge = occ3_sglang.other:load\n", encoding="utf-8")
    elif tamper == "dist-version":
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: occ3-bridge\nVersion: 9.9.9\n", encoding="utf-8")
    elif tamper == "duplicate-dist":
        extra = site / "occ3_bridge.egg-info"
        extra.mkdir()
        (extra / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: occ3-bridge\nVersion: 0.1.0\n", encoding="utf-8")
    elif tamper == "sglang-missing":
        import shutil
        shutil.rmtree(site / ("sglang-%s.dist-info" % package.SOURCE_VERSION))
    elif tamper == "sglang-version":
        old = site / ("sglang-%s.dist-info" % package.SOURCE_VERSION)
        new = site / "sglang-0.5.20.dist-info"
        old.rename(new)
        (new / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.20\n", encoding="utf-8")
    elif tamper == "sglang-source":
        (fixture.sg / "untouched.py").write_bytes(b"TAMPERED = 1\n")
    spec = nib.compose_verification_spec(plan, fixture.sg)
    result, returncode = child_result(venv_interpreter(venv), spec)
    assert returncode == 1 and result["ok"] is False and result["code"] == code, result
    with pytest.raises(nib.BuildError, match="installed_verification_failed"):
        nib.run_isolated_verify(venv_interpreter(venv), spec)


def test_isolated_verify_ignores_cwd_and_pythonpath_shadow(tmp_path):
    """A staged shadow in the build directory cannot satisfy verification."""
    fixture = Fixture(tmp_path)
    plan = prepared_plan(fixture)
    package.apply(plan, disposable=True)
    venv = create_venv(tmp_path)
    install_synthetic(venv, plan, fixture.root)
    shadow = tmp_path / "shadow"
    egg = shadow / "occ3_bridge.egg-info"
    egg.mkdir(parents=True)
    (egg / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: occ3-bridge\nVersion: 0.1.0\n", encoding="utf-8")
    shadow_package = shadow / "occ3_sglang"
    shadow_package.mkdir()
    (shadow_package / "__init__.py").write_bytes(b"SHADOW = 'stage only'\n")
    (shadow_package / "plugin.py").write_bytes(b"SHADOW = 'wrong bytes'\n")
    spec = nib.compose_verification_spec(plan, fixture.sg)
    env = {"PYTHONPATH": str(shadow), "PATH": os.environ.get("PATH", "")}
    result, returncode = child_result(venv_interpreter(venv), spec, cwd=shadow, env=env)
    assert returncode == 0 and result["ok"] is True
    assert result["installed_package_files"] == 8


def test_shadow_only_environment_cannot_pass_verification(tmp_path):
    """No installed wheel plus a cwd/PYTHONPATH shadow must fail, not pass."""
    fixture = Fixture(tmp_path)
    plan = prepared_plan(fixture)
    package.apply(plan, disposable=True)
    venv = create_venv(tmp_path)  # empty: no distribution installed
    shadow = tmp_path / "shadow"
    egg = shadow / "occ3_bridge.egg-info"
    egg.mkdir(parents=True)
    (egg / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: occ3-bridge\nVersion: 0.1.0\n", encoding="utf-8")
    shadow_package = shadow / "occ3_sglang"
    shadow_package.mkdir()
    for name, raw in fixture.plugin_sources.items():
        (shadow_package / name).write_bytes(raw)
    for name, raw in plan.artifacts:
        (shadow_package / name).write_bytes(raw)
    spec = nib.compose_verification_spec(plan, fixture.sg)
    env = {"PYTHONPATH": str(shadow), "PATH": os.environ.get("PATH", "")}
    result, returncode = child_result(venv_interpreter(venv), spec, cwd=shadow, env=env)
    assert returncode == 1 and result["code"] == "distribution_missing"


def test_child_refuses_spec_file_beyond_its_own_bound(tmp_path):
    """Reduced-bound proof that the child's spec read is size-checked before
    any parse, with no whole-file read of an oversize input."""
    fixture = Fixture(tmp_path)
    plan = prepared_plan(fixture)
    package.apply(plan, disposable=True)
    venv = create_venv(tmp_path)
    install_synthetic(venv, plan, fixture.root)
    junk = tmp_path / "oversize-spec.json"
    junk.write_bytes(b'{"pad": "' + b"a" * (nib.MAX_SPEC_BYTES + 16) + b'"}')
    assert junk.stat().st_size > nib.MAX_SPEC_BYTES
    result = subprocess.run([str(venv_interpreter(venv)), "-I", "-c",
                             nib.ISOLATED_VERIFY_SOURCE, str(junk)],
                            capture_output=True, text=True, timeout=300)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    receipt = json.loads(lines[-1])
    assert result.returncode == 1
    assert receipt == {"ok": False, "code": "spec_size_invalid",
                       "mode": "isolated-interpreter"}


# --- Owned Dockerfile static contract ---


DOCKERFILE = Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile"


def dockerfile_instructions():
    text = DOCKERFILE.read_text(encoding="utf-8")
    joined = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1].rstrip() + " " + stripped
        else:
            joined.append(stripped)
    return [line.split(None, 1) for line in joined]


def test_base_image_is_required_and_used_verbatim():
    instructions = dockerfile_instructions()
    assert instructions[0] == ["ARG", "BASE_IMAGE"]
    assert ["FROM", "${BASE_IMAGE}"] in instructions
    for verb, value in instructions:
        if verb == "ARG" and value.startswith("BASE_IMAGE"):
            assert value == "BASE_IMAGE"
    assert sum(1 for verb, _ in instructions if verb == "FROM") == 1


def test_dockerfile_has_no_serving_or_runtime_mutations():
    forbidden = {"ENV", "ENTRYPOINT", "CMD", "USER", "HEALTHCHECK", "EXPOSE", "VOLUME",
                 "LABEL", "MAINTAINER", "ONBUILD", "SHELL", "STOPSIGNAL", "ADD"}
    instructions = dockerfile_instructions()
    assert not forbidden & {verb for verb, _ in instructions}
    workdir = [value for verb, value in instructions if verb == "WORKDIR"]
    assert workdir == ["/opt/occ3-bridge"]


def test_dockerfile_copy_set_is_bounded_to_staged_inputs():
    allowed = {"occ3_sglang/", "pyproject.toml", "README.md*",
               "deploy/v3/artifact_preflight.py", "deploy/v3/__init__.py",
               "deploy/v3/source_patch.py", "deploy/v3/native_package.py",
               "deploy/v3/native_image_build.py",
               "patches/capture-eager-mtp.v1.json", "patches/native-startup.v2.json",
               "patches/scheduler-cache-ipc.v1.json",
               "deploy/native-build-inputs/source-baseline-expanded.json",
               "deploy/native-build-inputs/native-profile-input.json"}
    copies = [value for verb, value in dockerfile_instructions() if verb == "COPY"]
    assert copies, "no COPY instructions found"
    for value in copies:
        parts = value.split()
        sources, destination = parts[:-1], parts[-1]
        assert set(sources) <= allowed, sources
        assert destination in {"./", "./occ3_sglang/", "./deploy/v3/", "./native-build-inputs/"}


def test_single_run_invokes_driver_with_pinned_paths_and_flags():
    runs = [value for verb, value in dockerfile_instructions() if verb == "RUN"]
    assert len(runs) == 1
    text = " ".join(runs[0].split())
    assert "python3 -m deploy.v3.native_image_build" in text
    assert '--base-image "${BASE_IMAGE}"' in text
    assert "--sglang-srt /sgl-workspace/sglang/python/sglang/srt" in text
    assert "--occ3-sglang /opt/occ3-bridge/occ3_sglang" in text
    assert "--staged-inputs /opt/occ3-bridge/native-build-inputs" in text
    assert '--experimental-startup "${OCC3_EXPERIMENTAL_STARTUP}"' in text
    assert text.endswith("--disposable-tree")


def test_experimental_startup_arg_defaults_to_false():
    assert ["ARG", "OCC3_EXPERIMENTAL_STARTUP=false"] in dockerfile_instructions()


def test_dockerfile_has_no_network_or_pip_or_legacy_paths():
    text = DOCKERFILE.read_text(encoding="utf-8")
    instruction_text = " ".join(value for _, value in dockerfile_instructions()).lower()
    for forbidden in ("pip install", "curl", "wget", "apt-get", "apk add", "yum", "dnf",
                      "ssh ", "deploy.sh", "occ3_controller", "--network="):
        assert forbidden not in instruction_text, forbidden
    assert "salted KV-radix" not in text
    assert "Experimental SGLang connector for the generic Optimus OCC-3 bridge" in text
