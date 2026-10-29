"""Offline disposable-tree packaging tests; no model/runtime imports or Docker."""
import ast
import builtins
import dataclasses
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import socket

import pytest

from deploy.v3 import native_package as package

PRIVATE = "PRIVATE_INPUT_MUST_NOT_ESCAPE"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))


# Fabricated postapply tree pairs for _verify_tree_result (deploy/v3/native_package.py:437-458).
# Directory st_nlink cannot be driven through the real filesystem on NTFS (it already
# reports 1), so overlay2-style collapses are fabricated from one real _tree()
# snapshot; the pairs are therefore identical on NTFS and POSIX, and only the
# postapply seam ever observes the fabrication.
PRE_COLLAPSE_NLINK = 3  # ext4 directory with one subdirectory: 2 + child directories
COLLAPSED_NLINK = 1  # overlay2 merged view after copy-up
_PRODUCTION_TREE = package._tree  # captured at import, before any test can patch it


def tree_with_directory_nlink(tree, nlink):
    """Copy of tree with every directory signature's st_nlink (sig[3]) set to nlink."""
    entries = []
    for name, kind, signature in tree.entries:
        if kind == "directory":
            signature = list(signature)
            signature[3] = nlink
            signature = tuple(signature)
        entries.append((name, kind, signature))
    return package.Tree(tree.root, tuple(sorted(entries)), tree.parents)


def tree_with_entry_field(tree, name, index, value):
    """Copy of tree with a single entry's signature[index] set to value."""
    entries = []
    for entry_name, kind, signature in tree.entries:
        if entry_name == name:
            signature = list(signature)
            signature[index] = value
            signature = tuple(signature)
        entries.append((entry_name, kind, signature))
    return package.Tree(tree.root, tuple(sorted(entries)), tree.parents)


def fabricated_postapply(monkeypatch, plan, before, after):
    """Serve a fabricated before/after pair to the postapply check: _verify_tree_result
    reads its before side from plan.trees and its after side from the module-level
    _tree lookup it performs per root (native_package.py:439). Only the sglang_srt
    tree is fabricated; the plugin tree stays real, and because the caller never
    invokes package.apply, the preapply _recheck_inputs full-tuple comparison
    (native_package.py:302) never observes the fabrication."""
    monkeypatch.setattr(package, "_tree",
                        lambda root: after if root == plan.sglang_srt else _PRODUCTION_TREE(root))
    return dataclasses.replace(plan, trees=(before, plan.trees[1]))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.sg, self.plugin = root / "srt", root / "occ3_sglang"
        self.sg.mkdir()
        self.plugin.mkdir()
        (self.sg / "sub").mkdir()
        self.originals = {"a.py": b"VALUE = 1\n", "sub/b.py": b"VALUE = 2\n",
                          "http_server.py": b"VALUE = 3\n", "untouched.py": b"VALUE = 4\n"}
        for path, raw in self.originals.items():
            (self.sg / path).write_bytes(raw)
        self.plugin_sources = {"__init__.py": b"raise RuntimeError('plugin must never be imported')\n",
                               "native_profile.py": b"import torch\r\nVALUE = 1\r\n",
                               "native_install.py": b"import torch\nVALUE = 3\n",
                               "plugin.py": b"import sglang\nVALUE = 2\n",
                               "startup_state.py": b"VALUE = 4\n"}
        for path, raw in self.plugin_sources.items():
            (self.plugin / path).write_bytes(raw)
        self.baseline_path, self.profile_path = root / "baseline.json", root / "profile.json"
        self.baseline = {"schema": package.BASELINE_SCHEMA, "source_revision": package.SOURCE_REVISION,
                         "sglang_version": package.SOURCE_VERSION,
                         "files": [{"path": path, "sha256": digest(raw)} for path, raw in self.originals.items()]}
        self.profile = {"model_fingerprint": "fixture-fingerprint-not-production", "model_id": "fixture-model",
                        "expected_ranks": 8, "hidden_dim": 6144, "layer_index": 77,
                        "max_prefix_tokens": 128, "max_live_bytes": 256 * 1024**2,
                        "workspace_bytes": 128 * 1024**2, "max_records": 256, "max_active": 1,
                        "max_ttl_seconds": 3600, "anchor_alpha": 0.0, "tile_rows": 256,
                        "method": package.METHOD}
        write_json(self.baseline_path, self.baseline)
        write_json(self.profile_path, self.profile)
        self.manifests, self.manifest_paths = [], []
        for index, path in enumerate(("a.py", "sub/b.py", "http_server.py")):
            original = self.originals[path]
            patched = original.replace(b"VALUE", b"PATCHED")
            manifest = {"schema": package.source_patch.SCHEMA, "source_revision": package.SOURCE_REVISION,
                        "source_version": package.SOURCE_VERSION, "patch_set": "fixture-" + str(index),
                        "files": [{"path": path, "before_sha256": digest(original), "after_sha256": digest(patched),
                                   "replacements": [{"before": "VALUE", "after": "PATCHED"}]}]}
            target = root / ("fixture-" + str(index) + ".json")
            write_json(target, manifest)
            self.manifests.append(manifest)
            self.manifest_paths.append(target)

    def prepare(self, **kwargs):
        return package.prepare(self.sg, self.plugin, self.baseline_path, self.profile_path, self.manifest_paths, **kwargs)

    def no_writes(self):
        assert all((self.sg / path).read_bytes() == raw for path, raw in self.originals.items())
        assert all(not (self.plugin / name).exists() for name in package.OUTPUT_NAMES)

    def cli(self, action):
        args = [action, "--sglang-srt", str(self.sg), "--occ3-sglang", str(self.plugin),
                "--baseline-json", str(self.baseline_path), "--profile-json", str(self.profile_path)]
        for path in self.manifest_paths:
            args.extend(("--manifest", str(path)))
        return args


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_complete_canonical_artifacts_match_loader_schema_and_exact_source_bytes(fixture):
    plan = fixture.prepare()
    fixture.no_writes()
    result = package.apply(plan, disposable=True)
    assert result["ok"] and result["applied"]
    assert result["runtime_imported"] is result["gpu_proof"] is result["readiness_proof"] is False
    assert result["experimental_startup_selected"] is False
    profile_raw = (fixture.plugin / "native_profile.json").read_bytes()
    source_raw = (fixture.plugin / "native_sources.json").read_bytes()
    sources = json.loads(source_raw)
    assert profile_raw == package._canonical(fixture.profile)
    assert source_raw == package._canonical(sources)
    assert set(sources) == {"schema", "source_revision", "sglang_version", "manifests", "files"}
    assert sources["schema"] == "optimus.occ3.native-sources.v1"
    assert len(sources["files"]) == 9
    assert any(entry["root"] == "occ3_sglang" and entry["path"] == "native_install.py" for entry in sources["files"])
    assert any(entry["root"] == "occ3_sglang" and entry["path"] == "startup_state.py" for entry in sources["files"])
    for entry in sources["files"]:
        assert set(entry) == {"root", "path", "sha256"}
        root = fixture.sg if entry["root"] == "sglang_srt" else fixture.plugin
        assert digest((root / entry["path"]).read_bytes()) == entry["sha256"]
        assert entry["path"] != "_native_profile_digest.py"
    for entry in sources["manifests"]:
        assert set(entry) == {"name", "sha256"}
        assert digest((fixture.root / entry["name"]).read_bytes()) == entry["sha256"]
    assert [entry["name"] for entry in sources["manifests"]] == sorted(path.name for path in fixture.manifest_paths)
    module_raw = (fixture.plugin / "_native_profile_digest.py").read_bytes()
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(module_raw).body}
    assert assignments == {"PROFILE_SHA256": digest(profile_raw), "SOURCE_SHA256": digest(source_raw),
                           "EXPERIMENTAL_STARTUP": False}
    assert {entry["name"]: entry["sha256"] for entry in result["artifacts"]} == {
        name: digest((fixture.plugin / name).read_bytes()) for name in package.OUTPUT_NAMES}
    assert (fixture.plugin / "native_profile.py").read_bytes() == fixture.plugin_sources["native_profile.py"]
    with pytest.raises(package.PackageError, match="already_exists"):
        fixture.prepare()


def test_no_runtime_imports_network_or_source_execution(fixture, monkeypatch):
    original_import = builtins.__import__
    def refuse_runtime(name, *args, **kwargs):
        assert not name.startswith(("sglang", "torch", "occ3_sglang")), "runtime import attempted"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", refuse_runtime)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempted"))
    plan = fixture.prepare()
    assert package.apply(plan, disposable=True)["gpu_proof"] is False


def test_module_import_is_inert(monkeypatch):
    monkeypatch.setattr(package.source_patch, "prepare", lambda *a: pytest.fail("prepare during import"))
    monkeypatch.setattr(package.source_patch, "apply", lambda *a: pytest.fail("apply during import"))
    importlib.reload(package)


@pytest.mark.parametrize("flag", [False, True])
def test_experimental_startup_is_explicit_literal_build_choice(fixture, flag):
    plan = fixture.prepare(experimental_startup=flag)
    assert package.receipt(plan)["experimental_startup_selected"] is flag
    module = dict(plan.artifacts)["_native_profile_digest.py"]
    assert b"EXPERIMENTAL_STARTUP = " + repr(flag).encode() + b"\n" in module
    assert b"getenv" not in module and b"import" not in module


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_nonbool_startup_flag_rejected_without_writes(fixture, flag):
    with pytest.raises(package.PackageError, match="startup_flag"):
        fixture.prepare(experimental_startup=flag)
    fixture.no_writes()


def test_cli_verify_is_no_write_and_apply_requires_disposable_opt_in(fixture, capsys):
    assert package.main(fixture.cli("verify") + ["--experimental-startup"]) == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["applied"] is False and proof["experimental_startup_selected"] is True
    fixture.no_writes()
    assert package.main(fixture.cli("apply")) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "explicit_disposable_tree_required"
    fixture.no_writes()
    assert package.main(fixture.cli("apply") + ["--disposable-tree"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is True


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=True),
    lambda value: value.pop("files"),
    lambda value: value.update(schema="unknown"),
    lambda value: value.update(source_revision="0" * 40),
    lambda value: value.update(sglang_version="0.5.22"),
    lambda value: value["files"].append(deepcopy(value["files"][0])),
    lambda value: value["files"][0].update(path="../outside.py"),
    lambda value: value["files"][0].update(path="/absolute.py"),
    lambda value: value["files"][0].update(path="sub//b.py"),
    lambda value: value["files"][0].update(path="a.json"),
    lambda value: value["files"][0].update(sha256="A" * 64),
    lambda value: value["files"][0].update(extra="value"),
    lambda value: value.update(files=[]),
    lambda value: value.update(files=value["files"] * 33),
])
def test_baseline_strict_fields_revision_paths_counts_and_duplicates(fixture, mutation):
    mutation(fixture.baseline)
    write_json(fixture.baseline_path, fixture.baseline)
    with pytest.raises(package.PackageError):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(runtime_epoch="forbidden"),
    lambda value: value.update(source_attestation_sha256="0" * 64),
    lambda value: value.pop("model_fingerprint"),
    lambda value: value.update(model_fingerprint="has whitespace"),
    lambda value: value.update(expected_ranks=True),
    lambda value: value.update(tile_rows=0),
    lambda value: value.update(max_records=4097),
    lambda value: value.update(max_ttl_seconds=3601),
    lambda value: value.update(max_live_bytes=2**30 + 1),
    lambda value: value.update(anchor_alpha=float("inf")),
    lambda value: value.update(anchor_alpha=0.5),
    lambda value: value.update(hidden_dim=12),
    lambda value: value.update(method="unknown"),
])
def test_static_profile_rejects_runtime_fields_unsafe_primitives_and_known_bounds(fixture, mutation):
    mutation(fixture.profile)
    write_json(fixture.profile_path, fixture.profile)
    with pytest.raises(package.PackageError):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("which", ["baseline", "profile", "manifest"])
def test_duplicate_json_keys_are_not_silently_discarded(fixture, which):
    path = {"baseline": fixture.baseline_path, "profile": fixture.profile_path, "manifest": fixture.manifest_paths[0]}[which]
    raw = path.read_bytes()
    path.write_bytes(b'{"duplicated":1,"duplicated":2,' + raw[1:])
    with pytest.raises(package.PackageError, match="duplicate_json_field"):
        fixture.prepare()
    fixture.no_writes()


def test_manifest_cannot_self_attest_a_changed_original(fixture):
    path = fixture.sg / "a.py"
    changed = b"VALUE = 999\n"
    path.write_bytes(changed)
    entry = fixture.manifests[0]["files"][0]
    entry.update(before_sha256=digest(changed), after_sha256=digest(changed.replace(b"VALUE", b"PATCHED")))
    write_json(fixture.manifest_paths[0], fixture.manifests[0])
    with pytest.raises(package.PackageError, match="baseline_source_digest"):
        fixture.prepare()
    assert path.read_bytes() == changed
    assert not (fixture.plugin / "native_profile.json").exists()


def test_every_patch_requires_independent_baseline_coverage(fixture):
    fixture.baseline["files"] = [entry for entry in fixture.baseline["files"] if entry["path"] != "http_server.py"]
    write_json(fixture.baseline_path, fixture.baseline)
    with pytest.raises(package.PackageError, match="not_covered"):
        fixture.prepare()
    fixture.no_writes()


def test_disjoint_manifest_paths_are_mandatory_before_any_write(fixture):
    fixture.manifests[1]["files"] = deepcopy(fixture.manifests[0]["files"])
    write_json(fixture.manifest_paths[1], fixture.manifests[1])
    with pytest.raises(package.PackageError, match="overlapping_manifest_paths"):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_exactly_three_manifests_required(fixture, count):
    fixture.manifest_paths = (fixture.manifest_paths * 2)[:count]
    with pytest.raises(package.PackageError, match="exactly_three"):
        fixture.prepare()
    fixture.no_writes()


def test_corrupt_third_manifest_and_syntax_do_not_apply_earlier_plans(fixture):
    entry = fixture.manifests[2]["files"][0]
    entry["replacements"] = [{"before": "VALUE = 3", "after": PRIVATE + " = !"}]
    entry["after_sha256"] = digest((PRIVATE + " = !\n").encode())
    write_json(fixture.manifest_paths[2], fixture.manifests[2])
    with pytest.raises(package.PackageError, match="source_patch_validation_failed"):
        fixture.prepare()
    fixture.no_writes()


def test_unpatched_attested_source_and_plugin_syntax_checked(fixture):
    (fixture.plugin / "plugin.py").write_bytes(b"INVALID = !\n")
    with pytest.raises(package.PackageError, match="syntax"):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("raw", [b"\xef\xbb\xbfVALUE = 1\n", b"VALUE = 1\0\n", b"VALUE = '\xff'\n"])
def test_plugin_non_utf8_bom_null_refused(fixture, raw):
    (fixture.plugin / "plugin.py").write_bytes(raw)
    with pytest.raises(package.PackageError):
        fixture.prepare()
    fixture.no_writes()


def test_attested_sg_raw_lf_is_not_normalized_even_with_matching_new_baseline(fixture):
    path = fixture.sg / "untouched.py"
    raw = fixture.originals["untouched.py"].replace(b"\n", b"\r\n")
    path.write_bytes(raw)
    for entry in fixture.baseline["files"]:
        if entry["path"] == "untouched.py":
            entry["sha256"] = digest(raw)
    write_json(fixture.baseline_path, fixture.baseline)
    with pytest.raises(package.PackageError, match="raw_lf"):
        fixture.prepare()
    assert path.read_bytes() == raw


@pytest.mark.parametrize("name", ["sub", "__pycache__"])
def test_nested_or_generated_plugin_directories_are_explicitly_refused(fixture, name):
    (fixture.plugin / name).mkdir()
    with pytest.raises(package.PackageError, match="plugin_flat_source_only"):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("name", package.OUTPUT_NAMES)
def test_preexisting_artifacts_are_never_overwritten_or_repaired(fixture, name):
    path = fixture.plugin / name
    path.write_bytes(b"existing")
    with pytest.raises(package.PackageError, match="already_exists"):
        fixture.prepare()
    assert path.read_bytes() == b"existing"
    assert (fixture.sg / "a.py").read_bytes() == fixture.originals["a.py"]


@pytest.mark.parametrize("target", ["plugin", "sg", "baseline"])
def test_hardlinks_are_refused_for_all_input_kinds(fixture, target):
    path = {"plugin": fixture.plugin / "plugin.py", "sg": fixture.sg / "a.py", "baseline": fixture.baseline_path}[target]
    try:
        os.link(path, fixture.root / "hardlink")
    except OSError:
        pytest.skip("hardlink unavailable")
    with pytest.raises(package.PackageError):
        fixture.prepare()
    fixture.no_writes()


@pytest.mark.parametrize("target", ["plugin", "sg", "baseline"])
def test_symlinks_are_refused_for_all_input_kinds(fixture, target):
    path = {"plugin": fixture.plugin / "plugin.py", "sg": fixture.sg / "a.py", "baseline": fixture.baseline_path}[target]
    real = fixture.root / "real-source"
    path.rename(real)
    try:
        path.symlink_to(real)
    except OSError:
        real.rename(path)
        pytest.skip("symlink unavailable")
    with pytest.raises(package.PackageError):
        fixture.prepare()
    assert not (fixture.plugin / "native_profile.json").exists()


@pytest.mark.parametrize("target", ["plugin", "sg", "manifest", "baseline", "profile", "extra"])
def test_prepared_input_or_tree_identity_drift_refused_before_any_apply(fixture, monkeypatch, target):
    plan = fixture.prepare()
    if target == "extra":
        (fixture.sg / "new.py").write_bytes(b"NEW = True\n")
    else:
        path = {"plugin": fixture.plugin / "plugin.py", "sg": fixture.sg / "a.py",
                "manifest": fixture.manifest_paths[2], "baseline": fixture.baseline_path,
                "profile": fixture.profile_path}[target]
        raw = path.read_bytes()
        path.unlink()
        path.write_bytes(raw)
    monkeypatch.setattr(package.source_patch, "apply", lambda plan: pytest.fail("write phase reached"))
    # 2026-10-22: PackageError reasons now propagate with their own code (TS2 diagnosis
    # lesson); the drift codes below are the real pre-apply refusals.
    with pytest.raises(package.PackageError, match="source_tree_(membership_)?identity_drift|input_identity_drift|input_parent_drift"):
        package.apply(plan, disposable=True)
    fixture.no_writes()


def test_output_appearing_after_prepare_is_never_overwritten(fixture):
    plan = fixture.prepare()
    (fixture.plugin / "native_profile.json").write_bytes(b"foreign")
    with pytest.raises(package.PackageError, match="generated_output_already_exists"):
        package.apply(plan, disposable=True)
    assert (fixture.plugin / "native_profile.json").read_bytes() == b"foreign"
    assert (fixture.sg / "a.py").read_bytes() == fixture.originals["a.py"]


def test_partial_artifact_write_fails_requires_discard_and_cannot_resume(fixture, monkeypatch):
    plan = fixture.prepare()
    original_write = package._write_fresh
    def fail_second(root, name, raw):
        if name == "native_sources.json":
            raise OSError(PRIVATE)
        original_write(root, name, raw)
    monkeypatch.setattr(package, "_write_fresh", fail_second)
    with pytest.raises(package.PackageError, match="discard_tree"):
        package.apply(plan, disposable=True)
    assert (fixture.plugin / "native_profile.json").exists()
    assert not (fixture.plugin / "native_sources.json").exists()
    assert (fixture.sg / "a.py").read_bytes() == b"PATCHED = 1\n"
    with pytest.raises(package.PackageError):
        fixture.prepare()


def test_failed_overlay_or_false_apply_receipt_cannot_create_metadata(fixture, monkeypatch):
    plan = fixture.prepare()
    monkeypatch.setattr(package.source_patch, "apply", lambda plan: {"applied": True})
    # the false receipt is caught by source verification, whose code now propagates
    with pytest.raises(package.PackageError, match="applied_source_mismatch"):
        package.apply(plan, disposable=True)
    fixture.no_writes()


def test_overlayfs_directory_nlink_collapse_is_tolerated_after_apply(fixture, monkeypatch):
    """Replaces the deleted variant that monkeypatched package._tree across the whole
    apply() flow: on NTFS it was vacuous (directory st_nlink already reports 1, so
    forcing sig[3]=1 changed nothing) and on POSIX it would have failed the
    apply-entry recheck instead of the postapply check, because _recheck_inputs
    compares FULL 9-field trees (native_package.py:302) and the patched _tree drifted
    the before side too. The faithful seam is the postapply comparison alone: apply()
    rechecks the real tree, source_patch.apply's writes trigger the overlay2 copy-up
    collapse, and only afterwards does _verify_tree_result compare
    (native_package.py:490-493). The collapse is not reproducible through the real
    NTFS filesystem, so the before/after pair is fabricated from one _tree()
    snapshot: before carries an ext4-style directory nlink, after the collapsed
    merged-view nlink, every other field identical on both platforms."""
    plan = fixture.prepare()
    snapshot = package._tree(plan.sglang_srt)
    before = tree_with_directory_nlink(snapshot, PRE_COLLAPSE_NLINK)
    after = tree_with_directory_nlink(snapshot, COLLAPSED_NLINK)
    fabricated = fabricated_postapply(monkeypatch, plan, before, after)
    package._verify_tree_result(fabricated, outputs=False)  # admitted: no PackageError
    # Anti-vacuity guard: the pair differs in st_nlink and in nothing else.
    before_directories = {name: sig for name, kind, sig in before.entries if kind == "directory"}
    after_directories = {name: sig for name, kind, sig in after.entries if kind == "directory"}
    assert before_directories
    for name, signature in before_directories.items():
        assert after_directories[name][3] != signature[3]
        assert after_directories[name][:3] == signature[:3] and after_directories[name][4:] == signature[4:]
    fixture.no_writes()
    # The same fabricated pair with one replaced directory inode is still refused.
    replaced = tree_with_entry_field(after, "sub", 1, after_directories["sub"][1] + 1)
    fabricated_postapply(monkeypatch, plan, before, replaced)
    with pytest.raises(package.PackageError, match="source_directory_identity_drift"):
        package._verify_tree_result(fabricated, outputs=False)
    fixture.no_writes()


def test_directory_replacement_is_still_refused_without_nlink(fixture, monkeypatch):
    """The nlink exclusion must not weaken the wholesale-directory-replacement guard:
    a changed dev/ino (delete + recreate) still fails directory identity."""
    plan = fixture.prepare()
    original_tree = package._tree

    def replaced_tree(root):
        tree = original_tree(root)
        if root == plan.sglang_srt:
            entries = []
            for name, kind, sig in tree.entries:
                sig = list(sig)
                if kind == "directory" and name == "sub":
                    sig[1] = sig[1] + 1  # different inode: directory was replaced
                entries.append((name, kind, tuple(sig)))
            return package.Tree(tree.root, tuple(sorted(entries)), tree.parents)
        return tree

    monkeypatch.setattr(package, "_tree", replaced_tree)
    # refused by the pre-apply tree identity recheck (dev/ino pinned) before any write
    with pytest.raises(package.PackageError, match="source_tree_identity_drift"):
        package.apply(plan, disposable=True)
    fixture.no_writes()


@pytest.mark.parametrize("index,refused",
                         [(0, True), (1, True), (2, True), (3, False), (7, True), (8, True)],
                         ids=["dev-drift", "ino-drift", "mode-drift", "nlink-collapse", "uid-drift",
                              "gid-drift"])
def test_postapply_directory_identity_pins_dev_ino_mode_and_metadata_tail(fixture, monkeypatch, index, refused):
    """Pin the exact postapply directory identity contract: _verify_tree_result's
    directory branch (native_package.py:456) compares (signature[0:3], signature[7:])
    — dev, ino, mode and the uid/gid tail — and deliberately excludes st_nlink. If that
    slice ever silently dropped mode or ino, the drift cases here fail; if it grew to
    pin st_nlink, the collapse case fails. Every case starts from the tolerated nlink
    collapse, so a refusal can only come from the pinned field."""
    plan = fixture.prepare()
    snapshot = package._tree(plan.sglang_srt)
    before = tree_with_directory_nlink(snapshot, PRE_COLLAPSE_NLINK)
    after = tree_with_directory_nlink(snapshot, COLLAPSED_NLINK)
    before_signature = {name: sig for name, kind, sig in before.entries}["sub"]
    if refused:
        after = tree_with_entry_field(after, "sub", index, before_signature[index] + 1)
    after_signature = {name: sig for name, kind, sig in after.entries}["sub"]
    assert after_signature[index] != before_signature[index]  # anti-vacuity: drift is real
    fabricated = fabricated_postapply(monkeypatch, plan, before, after)
    if refused:
        with pytest.raises(package.PackageError, match="source_directory_identity_drift"):
            package._verify_tree_result(fabricated, outputs=False)
    else:
        package._verify_tree_result(fabricated, outputs=False)
    fixture.no_writes()


def test_postapply_unmodified_file_identity_still_pins_nlink(fixture, monkeypatch):
    """The directory nlink tolerance must not leak into file entries: unmodified files
    keep FULL 9-field signature equality (native_package.py:457-458), so a drifted
    file st_nlink is refused even though every directory is untouched. Fabricated
    from one snapshot, so the drift is equally real on NTFS and POSIX."""
    plan = fixture.prepare()
    snapshot = package._tree(plan.sglang_srt)
    signature = {name: sig for name, kind, sig in snapshot.entries}["untouched.py"]
    after = tree_with_entry_field(snapshot, "untouched.py", 3, signature[3] + 1)
    fabricated = fabricated_postapply(monkeypatch, plan, snapshot, after)
    with pytest.raises(package.PackageError, match="unmodified_tree_file_drift"):
        package._verify_tree_result(fabricated, outputs=False)
    fixture.no_writes()


def test_post_apply_source_or_artifact_drift_never_returns_success(fixture, monkeypatch):
    plan = fixture.prepare()
    original_write = package._write_fresh
    def corrupt(root, name, raw):
        original_write(root, name, raw)
        if name == "_native_profile_digest.py":
            (root / "native_sources.json").write_bytes(b"{}")
    monkeypatch.setattr(package, "_write_fresh", corrupt)
    # the corrupted artifact is caught by the readback, whose code now propagates
    with pytest.raises(package.PackageError, match="generated_artifact_mismatch"):
        package.apply(plan, disposable=True)


def test_bounded_oversize_refused_without_writes(fixture, monkeypatch):
    monkeypatch.setattr(package, "MAX_FILE_BYTES", 16)
    with pytest.raises(package.PackageError, match="oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_base_image_binary_object_exceeding_source_bound_is_accepted(fixture, monkeypatch):
    """Live-window abort 2026-10-22: compiled .so objects from the immutable base
    image exceed the 2 MiB source bound; the tree walk must admit them under the
    separate binary traversal bound while never reading their contents."""
    monkeypatch.setattr(package, "MAX_FILE_BYTES", 64)
    (fixture.sg / "rust_extensions").mkdir()
    (fixture.sg / "rust_extensions" / "_server.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF" + b"0" * 96)
    plan = fixture.prepare()
    entry = [item for item in plan.trees[0].entries if item[0].endswith(".so")]
    assert entry and entry[0][1] == "file"
    fixture.no_writes()


def test_binary_object_beyond_traversal_bound_is_refused(fixture, monkeypatch):
    monkeypatch.setattr(package, "MAX_BINARY_FILE_BYTES", 8)
    (fixture.sg / "_grpc.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF" + b"0" * 40)
    with pytest.raises(package.PackageError, match="tree_file_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_oversize_python_source_still_refused_under_binary_bound(fixture, monkeypatch):
    monkeypatch.setattr(package, "MAX_FILE_BYTES", 16)
    (fixture.sg / "oversize.py").write_bytes(b"VALUE = 1\n" + b"#" * 40)
    with pytest.raises(package.PackageError, match="tree_file_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_cli_errors_never_echo_private_input_or_paths(fixture, capsys):
    fixture.profile["model_id"] = PRIVATE + " has spaces"
    write_json(fixture.profile_path, fixture.profile)
    assert package.main(fixture.cli("verify")) == 1
    raw = capsys.readouterr().out
    assert PRIVATE not in raw and str(fixture.root) not in raw
    assert json.loads(raw)["ok"] is False
    fixture.no_writes()


@pytest.mark.parametrize("name", ["__init__.py", "native_profile.py", "native_install.py", "plugin.py"])
def test_required_packaged_loader_installer_and_plugin_modules_cannot_be_omitted(fixture, name):
    (fixture.plugin / name).unlink()
    with pytest.raises(package.PackageError, match="plugin_package_missing"):
        fixture.prepare()
    fixture.no_writes()


def test_duplicate_manifest_basenames_are_refused_even_at_distinct_paths(fixture):
    alternate = fixture.root / "other"
    alternate.mkdir()
    replacement = alternate / fixture.manifest_paths[0].name
    replacement.write_bytes(fixture.manifest_paths[1].read_bytes())
    fixture.manifest_paths[1] = replacement
    with pytest.raises(package.PackageError, match="manifest_name"):
        fixture.prepare()
    fixture.no_writes()


def test_all_combined_source_entries_fit_loader_128_file_limit(fixture):
    for index in range(121):
        (fixture.plugin / ("extra_" + str(index) + ".py")).write_bytes(b"VALUE = 0\n")
    with pytest.raises(package.PackageError, match="source_attestation_file_count"):
        fixture.prepare()
    fixture.no_writes()


def test_source_json_size_and_input_memory_limits_are_checked_before_apply(fixture, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(package, "MAX_SOURCES_BYTES", 64)
        with pytest.raises(package.PackageError, match="source_attestation_oversize"):
            fixture.prepare()
    monkeypatch.setattr(package, "MAX_INPUT_BYTES", 32)
    with pytest.raises(package.PackageError, match="input_set_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_unmodified_baseline_file_syntax_is_validated_not_only_patched_files(fixture):
    broken = b"VALUE = !\n"
    (fixture.sg / "untouched.py").write_bytes(broken)
    for entry in fixture.baseline["files"]:
        if entry["path"] == "untouched.py":
            entry["sha256"] = digest(broken)
    write_json(fixture.baseline_path, fixture.baseline)
    with pytest.raises(package.PackageError, match="source_syntax_invalid"):
        fixture.prepare()
    assert not (fixture.plugin / "native_profile.json").exists()


def test_missing_attested_file_refuses_all_writes(fixture):
    (fixture.sg / "untouched.py").unlink()
    with pytest.raises(package.PackageError):
        fixture.prepare()
    assert not (fixture.plugin / "native_profile.json").exists()
    assert (fixture.sg / "a.py").read_bytes() == fixture.originals["a.py"]


def test_partial_overlay_failure_is_not_success_or_resumable(fixture, monkeypatch):
    plan = fixture.prepare()
    def partial(overlay):
        overlay.files[0].path.write_bytes(overlay.files[0].patched)
        raise package.source_patch.PatchError("synthetic_partial_write")
    monkeypatch.setattr(package.source_patch, "apply", partial)
    with pytest.raises(package.PackageError, match="discard_tree"):
        package.apply(plan, disposable=True)
    assert not (fixture.plugin / "native_profile.json").exists()
    with pytest.raises(package.PackageError):
        fixture.prepare()


def test_final_merged_plan_digest_is_a_real_source_patch_manifest_digest(fixture):
    plan = fixture.prepare()
    merged = {"schema": package.source_patch.SCHEMA, "source_revision": package.SOURCE_REVISION,
              "source_version": package.SOURCE_VERSION, "patch_set": "occ3-native-package",
              "files": sorted([entry for manifest in fixture.manifests for entry in manifest["files"]],
                              key=lambda entry: entry["path"])}
    independently_prepared = package.source_patch.prepare(fixture.sg, merged)
    assert plan.overlay.manifest_sha256 == independently_prepared.manifest_sha256
    assert plan.overlay.files == independently_prepared.files



def test_environment_does_not_enable_experimental_startup(fixture, monkeypatch):
    monkeypatch.setenv("EXPERIMENTAL_STARTUP", "True")
    monkeypatch.setenv("OCC3_EXPERIMENTAL_STARTUP", "1")
    plan = fixture.prepare()
    assert plan.experimental_startup is False
    assert b"EXPERIMENTAL_STARTUP = False\n" in dict(plan.artifacts)["_native_profile_digest.py"]



@pytest.mark.parametrize("identifier", ["organization/model", "x" * 256, "_leading", "/:.-"])
@pytest.mark.parametrize("field", ["model_id", "model_fingerprint"])
def test_profile_identifier_matches_current_shared_consumer_contract(fixture, identifier, field):
    fixture.profile[field] = identifier
    write_json(fixture.profile_path, fixture.profile)
    plan = fixture.prepare()
    assert json.loads(dict(plan.artifacts)["native_profile.json"])[field] == identifier
    fixture.no_writes()


@pytest.mark.parametrize("identifier", ["x" * 257, "has space", "line\nfeed", "nonascii-\u00e9", "", "null\0byte",
                                          True, 42, None, [], {}])
def test_profile_identifier_outside_shared_contract_refused_without_write(fixture, identifier):
    fixture.profile["model_id"] = identifier
    write_json(fixture.profile_path, fixture.profile)
    with pytest.raises(package.PackageError, match="invalid_profile_identifier"):
        fixture.prepare()
    fixture.no_writes()


def test_merged_manifest_respects_same_byte_bound_as_normal_source_patch_prepare(fixture, monkeypatch):
    for path, manifest in zip(fixture.manifest_paths, fixture.manifests):
        path.write_bytes(package._canonical(manifest))
    maximum = max(len(path.read_bytes()) for path in fixture.manifest_paths)
    monkeypatch.setattr(package.source_patch, "MAX_MANIFEST_BYTES", maximum)
    # Positive controls establish that failure is the merged bound, not an
    # individual input read/parse or an individual source patch preparation.
    for manifest in fixture.manifests:
        assert package.source_patch.prepare(fixture.sg, manifest).files
    merged = {"schema": package.source_patch.SCHEMA, "source_revision": package.SOURCE_REVISION,
              "source_version": package.SOURCE_VERSION, "patch_set": "occ3-native-package",
              "files": sorted([entry for manifest in fixture.manifests for entry in manifest["files"]],
                              key=lambda entry: entry["path"])}
    assert len(package._canonical(merged)) > maximum
    with pytest.raises(package.source_patch.PatchError, match="manifest_too_large"):
        package.source_patch.prepare(fixture.sg, merged)
    with pytest.raises(package.PackageError, match="merged_manifest_oversize"):
        fixture.prepare()
    fixture.no_writes()



def fixture_final_source_bytes(fixture):
    # Independent fixture arithmetic: manifests replace VALUE by PATCHED in
    # exactly three SG files; the fourth and every plugin module are unchanged.
    patched = {"a.py", "sub/b.py", "http_server.py"}
    return (sum(len(raw.replace(b"VALUE", b"PATCHED")) if path in patched else len(raw)
                for path, raw in fixture.originals.items())
            + sum(len((fixture.plugin / name).read_bytes()) for name in fixture.plugin_sources))


def test_final_source_byte_budget_accepts_exact_consumer_boundary(fixture, monkeypatch):
    expected = fixture_final_source_bytes(fixture)
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", expected)
    plan = fixture.prepare()
    assert plan.source_bytes == expected
    assert package.receipt(plan)["produced_source_bytes"] == expected
    proof = package.apply(plan, disposable=True)
    sources = json.loads((fixture.plugin / "native_sources.json").read_bytes())
    actual = sum(len(((fixture.sg if entry["root"] == "sglang_srt" else fixture.plugin) / entry["path"]).read_bytes())
                 for entry in sources["files"])
    assert actual == proof["produced_source_bytes"] == expected
    assert len((fixture.plugin / "_native_profile_digest.py").read_bytes()) > 0
    # JSON, input manifests and the generated digest are deliberately excluded.
    assert sum(len(path.read_bytes()) for path in fixture.manifest_paths) > expected


def test_final_source_byte_budget_refuses_one_byte_over_before_any_write(fixture, monkeypatch):
    expected = fixture_final_source_bytes(fixture)
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", expected - 1)
    for manifest in fixture.manifests:
        assert package.source_patch.prepare(fixture.sg, manifest).files
    with pytest.raises(package.PackageError, match="produced_source_total_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_final_source_limit_includes_patch_growth_not_just_original_bytes(fixture, monkeypatch):
    before = sum(len(raw) for raw in fixture.originals.values()) + sum(len(raw) for raw in fixture.plugin_sources.values())
    assert fixture_final_source_bytes(fixture) > before
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", before)
    with pytest.raises(package.PackageError, match="produced_source_total_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_final_source_limit_includes_every_flat_plugin_module(fixture, monkeypatch):
    limit = fixture_final_source_bytes(fixture)
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", limit)
    (fixture.plugin / "additional_module.py").write_bytes(b"# extra\n")
    with pytest.raises(package.PackageError, match="produced_source_total_oversize"):
        fixture.prepare()
    fixture.no_writes()


def test_source_and_all_input_budgets_are_distinct_contracts(fixture, monkeypatch):
    expected = fixture_final_source_bytes(fixture)
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", expected)
    assert package.MAX_INPUT_BYTES == 64 * 1024 * 1024
    assert fixture.prepare().source_bytes == expected
    # Reducing the separate input budget still fails even when final sources fit.
    monkeypatch.setattr(package, "MAX_INPUT_BYTES", expected)
    with pytest.raises(package.PackageError, match="input_set_oversize"):
        fixture.prepare()
    fixture.no_writes()



def test_final_source_budget_is_rechecked_at_prewrite_barrier(fixture, monkeypatch):
    plan = fixture.prepare()
    monkeypatch.setattr(package, "MAX_SOURCE_TOTAL", plan.source_bytes - 1)
    monkeypatch.setattr(package.source_patch, "apply", lambda _: pytest.fail("write phase reached"))
    # the budget recheck's own code now propagates (was masked as discard_tree)
    with pytest.raises(package.PackageError, match="produced_source_total_oversize"):
        package.apply(plan, disposable=True)
    fixture.no_writes()



def test_identifier_alphabet_matches_both_current_consumers_without_importing_them():
    root = Path(__file__).resolve().parents[1] / "occ3_sglang"
    for name in ("native_profile.py", "attachment_contract.py"):
        tree = ast.parse((root / name).read_bytes())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "_IDENTIFIER" for target in node.targets)]
        assert len(assignments) == 1
        declaration = assignments[0].value
        assert isinstance(declaration, ast.Call)
        assert package._IDENTIFIER.pattern == ast.literal_eval(declaration.args[0])


def test_offline_prepare_and_apply_survive_a_blocked_registry_import(fixture, monkeypatch):
    """The build path never imports occ3_sglang, so even a poisoned registry
    module cannot be pulled in: the static registry read is the only source."""
    import sys as _sys
    import types as _types

    class _Poison:
        def __getattr__(self, name):
            raise AssertionError("runtime import attempted: occ3_sglang." + name)

    monkeypatch.setitem(_sys.modules, "occ3_sglang", _Poison())
    monkeypatch.setitem(_sys.modules, "occ3_sglang.native_profile", _Poison())
    plan = fixture.prepare()
    assert package.apply(plan, disposable=True)["ok"] is True
