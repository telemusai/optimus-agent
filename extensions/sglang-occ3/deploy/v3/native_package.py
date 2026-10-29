"""Offline packaging for an exclusive disposable derived-image source tree.

No SGLang, Torch, or OCC3 runtime imports at module load and none anywhere in
the validate/apply path: the shipped method ids are read by parsing the
registry source (occ3_sglang/native_profile.py) instead of importing the
package, so the build stage stays pure-file validation and the registry stays
the only place a method id is written.
Hashes and syntax are packaging proof, not installed-source, GPU, model-identity,
or readiness proof. On any interrupted or failed apply, discard the disposable
tree; this is never an in-place repair.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from . import source_patch

BASELINE_SCHEMA = "optimus.occ3.source-baseline.v1"
SOURCES_SCHEMA = "optimus.occ3.native-sources.v1"
RECEIPT_SCHEMA = "optimus.occ3.native-package.v1"
SOURCE_REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"
SOURCE_VERSION = "0.5.21"
PROFILE_FIELDS = frozenset({
    "model_fingerprint", "model_id", "expected_ranks", "hidden_dim", "layer_index",
    "max_prefix_tokens", "max_live_bytes", "workspace_bytes", "max_records",
    "max_active", "max_ttl_seconds", "anchor_alpha", "tile_rows", "method",
})
# Frozen-artifact reference: optional, because the per-message method names none.
OPTIONAL_PROFILE_FIELDS = frozenset({"calibration_path", "calibration_digest"})
OUTPUT_NAMES = ("native_profile.json", "native_sources.json", "_native_profile_digest.py")
MAX_PROFILE_BYTES = 16384
MAX_SOURCES_BYTES = 65536
MAX_BASELINE_BYTES = 65536
MAX_FILE_BYTES = 2 * 1024 * 1024
# Traversal bound for non-source files in the packaged trees (compiled shared
# objects and data shipped by the immutable base image). The base image's
# rust/grpc/multimodal extensions exceed the 2 MiB source bound; this is a
# metadata-walk bound only — file contents at this size are never read,
# hashed into source manifests, or attested as reviewed source.
MAX_BINARY_FILE_BYTES = 64 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
# Matches native_install._MAX_SOURCE_TOTAL; only native_sources.files bytes.
MAX_SOURCE_TOTAL = 32 * 1024 * 1024
MAX_TREE_ENTRIES = 8192
MAX_TREE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_FILES = 128
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")


class PackageError(Exception):
    """Fixed non-sensitive error codes only."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def shipped_methods():
    """Method ids a packaged profile may pin, read from the registry that owns them.

    Import-based view for callers that may import the runtime package (tests,
    parity checks). The validate/apply path never calls this: it reads the
    same registry through _static_registry() by parsing the source, because
    the offline build stage must not import occ3_sglang.
    """
    try:
        from occ3_sglang import native_profile
    except ImportError:
        raise PackageError("method_registry_unavailable") from None
    return tuple(native_profile.advertised_methods())


_REGISTRY_MODULE = "native_profile.py"
# Separate from MAX_FILE_BYTES: source bounds are build-input policy, while the
# registry read is an internal metadata lookup with its own fixed bound.
_REGISTRY_MAX_BYTES = 2 * 1024 * 1024


def _registry_source(name):
    """Bounded read of one registry-side module from the local package tree."""
    path = Path(__file__).resolve().parents[2] / "occ3_sglang" / name
    try:
        raw = path.read_bytes()
    except OSError:
        raise PackageError("method_registry_unavailable") from None
    _need(len(raw) <= _REGISTRY_MAX_BYTES, "method_registry_unavailable")
    return raw


def _registry_constants(tree, name):
    """Module-level constant assignments: name -> value (imports excluded)."""
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (str, int, float, bool))):
            constants[node.targets[0].id] = node.value.value
    return constants


def _registry_imports(tree):
    """Relative constant imports: local alias -> (module file, original name)."""
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level >= 1:
            module = node.module.lstrip(".")
            if "/" in module or chr(92) in module or not _NAME.fullmatch(module):
                continue
            for alias in node.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = (module + ".py", alias.name)
    return imports


def _registry_value(node, environment):
    """Evaluate one registry expression: constants, names, nothing else."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)
            and isinstance(node.op, (ast.USub, ast.UAdd))):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    raise PackageError("method_registry_unavailable")


def _registry_call_entries(tree, name):
    """The _method_entry calls building _METHODS, in registry order."""
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_METHODS"):
            value = node.value
            elements = None
            if isinstance(value, ast.DictComp) and len(value.generators) == 1:
                iterator = value.generators[0].iter
                if isinstance(iterator, (ast.Tuple, ast.List)):
                    elements = iterator.elts
            elif isinstance(value, (ast.Tuple, ast.List)):
                elements = value.elts
            if elements is None:
                raise PackageError("method_registry_unavailable")
            entries = [item for item in elements if isinstance(item, ast.Call)]
            _need(len(entries) == len(elements) and entries, "method_registry_unavailable")
            return entries
    raise PackageError("method_registry_unavailable")


def _registry_entry_function(tree):
    """The _method_entry definition: parameter names plus its return dict."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_method_entry":
            arguments = node.args
            _need(not arguments.posonlyargs and not arguments.vararg
                  and not arguments.kwonlyargs and not arguments.kwarg
                  and not arguments.defaults, "method_registry_unavailable")
            parameters = [item.arg for item in arguments.args]
            body = node.body
            _need(len(body) == 1 and isinstance(body[0], ast.Return)
                  and isinstance(body[0].value, ast.Dict), "method_registry_unavailable")
            return parameters, body[0].value
    raise PackageError("method_registry_unavailable")


def _static_registry():
    """Method descriptors parsed from the registry source; never an import.

    The registry module (occ3_sglang/native_profile.py) is the single place a
    native method id is written, but the offline build path must not import
    occ3_sglang, so its _METHODS table is read by parsing the module and its
    relative constant imports. The parse understands exactly the registry's
    own constructs and fails closed on anything else; the imported view
    (shipped_methods) must stay equal and is pinned by tests.
    """
    raw = _registry_source(_REGISTRY_MODULE)
    try:
        tree = ast.parse(raw.decode("utf-8"), _REGISTRY_MODULE)
    except (SyntaxError, ValueError, UnicodeError, RecursionError):
        raise PackageError("method_registry_unavailable") from None
    environment = dict(_registry_constants(tree, _REGISTRY_MODULE))
    for local, (module, original) in _registry_imports(tree).items():
        imported = ast.parse(_registry_source(module).decode("utf-8"), module)
        constants = _registry_constants(imported, module)
        _need(original in constants, "method_registry_unavailable")
        environment[local] = constants[original]
    parameters, returned = _registry_entry_function(tree)
    registry = {}
    for call in _registry_call_entries(tree, _REGISTRY_MODULE):
        _need(isinstance(call.func, ast.Name) and call.func.id == "_method_entry",
              "method_registry_unavailable")
        _need(len(call.args) <= len(parameters), "method_registry_unavailable")
        bound = dict(zip(parameters, call.args))
        for keyword in call.keywords:
            _need(keyword.arg in parameters, "method_registry_unavailable")
            bound[keyword.arg] = keyword.value
        _need(len(bound) == len(parameters), "method_registry_unavailable")
        scope = {name: _registry_value(value, environment)
                 for name, value in bound.items()}
        descriptor = {}
        for key, value in zip(returned.keys, returned.values):
            _need(isinstance(key, ast.Constant) and isinstance(key.value, str),
                  "method_registry_unavailable")
            descriptor[key.value] = _registry_value(value, scope)
        method_id = descriptor.get("method_id")
        _need(type(method_id) is str and _IDENTIFIER.fullmatch(method_id) is not None,
              "method_registry_unavailable")
        _need(type(descriptor.get("requires_calibration_artifact")) is bool,
              "method_registry_unavailable")
        _need(method_id not in registry, "method_registry_unavailable")
        registry[method_id] = descriptor
    _need(registry, "method_registry_unavailable")
    return registry


def default_method():
    """The method a profile pins when it does not name one: the per-message method."""
    try:
        from occ3_sglang import native_profile
    except ImportError:
        raise PackageError("method_registry_unavailable") from None
    return native_profile.PER_MESSAGE_METHOD


def __getattr__(name):
    # Compatibility shim: METHOD was a local literal. It now resolves to the
    # registry's default method instead of a copy that can silently drift.
    if name == "METHOD":
        return default_method()
    raise AttributeError(name)


def _need(condition, code):
    if not condition:
        raise PackageError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate_json_field")
        result[key] = value
    return result


def _integer(value):
    _need(len(value) <= 20, "invalid_integer")
    number = int(value)
    _need(-(1 << 63) <= number <= (1 << 64) - 1, "invalid_integer")
    return number


def _constant(_value):
    raise PackageError("invalid_json_constant")


def _json(raw):
    _need(not raw.startswith(b"\xef\xbb\xbf") and b"\0" not in raw, "invalid_json_encoding")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                          parse_int=_integer, parse_constant=_constant)
    except PackageError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise PackageError("invalid_json") from None


def _keys(value, fields, code, optional=frozenset()):
    _need(type(value) is dict and fields <= set(value) <= fields | optional, code)
    return value


def _digest(value):
    _need(type(value) is str and _HEX.fullmatch(value) is not None, "invalid_digest")
    return value


def _relative(value):
    _need(type(value) is str and len(value) <= 180
          and re.fullmatch(r"[A-Za-z0-9_/-]+[.]py", value) is not None, "invalid_source_path")
    path = PurePosixPath(value)
    _need(not path.is_absolute() and str(path) == value
          and all(part not in {".", ".."} and _NAME.fullmatch(part) is not None for part in path.parts),
          "invalid_source_path")
    return value


def _metadata(info):
    return (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_uid, info.st_gid)


def _safe_info(path, *, directory=False):
    info = path.lstat()
    _need(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
          "filesystem_link_unsupported")
    _need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
          "filesystem_type_unsupported")
    if not directory:
        _need(info.st_nlink == 1, "filesystem_hardlink_unsupported")
        _need(stat.S_IMODE(info.st_mode) & 0o7000 == 0, "filesystem_special_mode")
    return info


def _absolute(path):
    return Path(os.path.abspath(path))


def _parents(path):
    """Record ancestor identities, not unrelated parent directory contents."""
    records = []
    for parent in reversed((path, *path.parents)):
        info = _safe_info(parent, directory=True)
        records.append((parent, (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid)))
    return tuple(records)


@dataclass(frozen=True)
class InputFile:
    path: Path
    raw: bytes
    mode: int
    signature: tuple
    parents: tuple


@dataclass(frozen=True)
class Tree:
    root: Path
    entries: tuple
    parents: tuple


@dataclass(frozen=True)
class PackagePlan:
    sglang_srt: Path
    occ3_sglang: Path
    inputs: tuple[InputFile, ...]
    sg_files: tuple[tuple[str, InputFile], ...]
    plugin_files: tuple[tuple[str, InputFile], ...]
    trees: tuple[Tree, Tree]
    overlay: source_patch.PatchPlan
    manifests: tuple[tuple[str, str], ...]
    baseline_sha256: str
    experimental_startup: bool
    source_bytes: int
    artifacts: tuple[tuple[str, bytes], ...]


def _read(path, maximum):
    path = _absolute(path)
    parents = _parents(path.parent)
    before = _safe_info(path)
    _need(before.st_size <= maximum, "input_oversize")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        actual = os.fstat(fd)
        _need(_metadata(before) == _metadata(actual), "input_identity_drift")
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        _need(len(raw) <= maximum and len(raw) == before.st_size, "input_oversize")
        _need(_metadata(after) == _metadata(before) == _metadata(_safe_info(path)), "input_identity_drift")
        _need(_parents(path.parent) == parents, "input_parent_drift")
        return InputFile(path, raw, stat.S_IMODE(before.st_mode), _metadata(before), parents)
    finally:
        if fd is not None:
            os.close(fd)


def _tree(root):
    root = _absolute(root)
    parents = _parents(root)
    entries, folded, total = [], set(), 0
    pending = [(root, "", 0)]
    while pending:
        path, relative, depth = pending.pop()
        _need(depth <= 24, "tree_depth_exceeded")
        info = path.lstat()
        directory = stat.S_ISDIR(info.st_mode)
        info = _safe_info(path, directory=directory)
        _need(relative.casefold() not in folded, "tree_path_collision")
        folded.add(relative.casefold())
        entries.append((relative, "directory" if directory else "file", _metadata(info)))
        _need(len(entries) <= MAX_TREE_ENTRIES, "tree_entry_limit")
        if directory:
            with os.scandir(path) as children:
                for child in children:
                    _need(len(pending) + len(entries) < MAX_TREE_ENTRIES, "tree_entry_limit")
                    name = child.name if not relative else relative + "/" + child.name
                    pending.append((Path(child.path), name, depth + 1))
        else:
            limit = MAX_FILE_BYTES if relative.endswith(".py") else MAX_BINARY_FILE_BYTES
            _need(info.st_size <= limit, "tree_file_oversize")
            total += info.st_size
            _need(total <= MAX_TREE_BYTES, "tree_byte_limit")
    return Tree(root, tuple(sorted(entries)), parents)


def _source(raw, relative, *, attested_sg):
    _need(not raw.startswith(b"\xef\xbb\xbf") and b"\0" not in raw, "source_encoding_invalid")
    if attested_sg:
        _need(b"\r" not in raw, "attested_source_requires_raw_lf")
    try:
        raw.decode("utf-8", errors="strict")
        compile(raw, relative, "exec", dont_inherit=True)
    except (SyntaxError, ValueError, UnicodeError, RecursionError):
        raise PackageError("source_syntax_invalid") from None


def _profile(value):
    _keys(value, PROFILE_FIELDS, "invalid_profile_fields", OPTIONAL_PROFILE_FIELDS)
    path_value = value.get("calibration_path", "")
    digest_value = value.get("calibration_digest", "")
    _need(type(path_value) is str and type(digest_value) is str
          and bool(path_value) == bool(digest_value), "invalid_profile_calibration_reference")
    if path_value:
        _need(0 < len(path_value) <= 512 and path_value.isprintable()
              and os.path.isabs(path_value), "invalid_profile_calibration_path")
        _digest(digest_value)
    for name in ("model_fingerprint", "model_id", "method"):
        _need(type(value[name]) is str and _IDENTIFIER.fullmatch(value[name]) is not None,
              "invalid_profile_identifier")
    for name in PROFILE_FIELDS - {"model_fingerprint", "model_id", "method", "anchor_alpha"}:
        _need(type(value[name]) is int and (0 if name == "layer_index" else 1) <= value[name] < 2**64,
              "invalid_profile_integer")
    _need(type(value["anchor_alpha"]) in (int, float) and math.isfinite(value["anchor_alpha"]),
          "invalid_profile_alpha")
    # Current audited RuntimeProfile bounds, not evidence of actual model/TP/GPU state.
    # A packaged profile may only pin a method the installed registry advertises.
    # The registry is parsed, not imported: the build path stays pure-file.
    _need(value["method"] in _static_registry(), "profile_method_not_advertised")
    _need((value["expected_ranks"], value["hidden_dim"], value["layer_index"]) == (8, 6144, 77)
          and value["max_prefix_tokens"] in (8, 16, 32, 64, 128)
          and value["max_active"] == 1 and value["max_records"] <= 4096
          and value["max_live_bytes"] <= 2**30 and value["workspace_bytes"] <= 2**30
          and value["max_ttl_seconds"] <= 3600 and value["tile_rows"] <= 4096
          and value["anchor_alpha"] in (0.0, 0.3), "profile_outside_audited_bounds")
    raw = _canonical(value)
    _need(len(raw) <= MAX_PROFILE_BYTES, "profile_oversize")
    return raw


def _baseline(value):
    _keys(value, {"schema", "source_revision", "sglang_version", "files"}, "invalid_baseline_fields")
    _need(value["schema"] == BASELINE_SCHEMA and value["source_revision"] == SOURCE_REVISION
          and value["sglang_version"] == SOURCE_VERSION, "baseline_revision_mismatch")
    files = value["files"]
    _need(type(files) is list and 1 <= len(files) <= MAX_SOURCE_FILES, "baseline_file_count")
    result, seen = {}, set()
    for entry in files:
        _keys(entry, {"path", "sha256"}, "invalid_baseline_file")
        path = _relative(entry["path"])
        _need(path.casefold() not in seen, "duplicate_baseline_path")
        seen.add(path.casefold())
        result[path] = _digest(entry["sha256"])
    return result


def _fresh(root):
    _parents(root)
    for name in OUTPUT_NAMES:
        _need(not os.path.lexists(root / name), "generated_output_already_exists")


def _recheck_inputs(plan):
    changed = {item.relative: item.patched for item in plan.overlay.files}
    actual_total = (sum(len(changed.get(path, record.raw)) for path, record in plan.sg_files)
                    + sum(len(record.raw) for _, record in plan.plugin_files))
    _need(actual_total == plan.source_bytes, "produced_source_size_drift")
    _need(actual_total <= MAX_SOURCE_TOTAL, "produced_source_total_oversize")
    _fresh(plan.occ3_sglang)
    _need(tuple(_tree(tree.root) for tree in plan.trees) == plan.trees, "source_tree_identity_drift")
    for record in (*plan.inputs, *(item for _, item in plan.sg_files), *(item for _, item in plan.plugin_files)):
        current = _read(record.path, max(len(record.raw), 1))
        _need(current == record, "input_identity_drift")


def prepare(sglang_srt, occ3_sglang, baseline_path, profile_path, manifest_paths, *, experimental_startup=False):
    """Validate all inputs, three plans, final bytes and tree identities; no writes."""
    try:
        _need(type(experimental_startup) is bool, "invalid_experimental_startup_flag")
        sg, plugin = _absolute(sglang_srt), _absolute(occ3_sglang)
        _need(sg != plugin and not sg.is_relative_to(plugin) and not plugin.is_relative_to(sg),
              "overlapping_package_roots")
        _fresh(plugin)
        trees = (_tree(sg), _tree(plugin))
        _need(type(manifest_paths) in (list, tuple) and len(manifest_paths) == 3, "exactly_three_manifests_required")
        paths = tuple(_absolute(path) for path in (baseline_path, profile_path, *manifest_paths))
        _need(len({str(path).casefold() for path in paths}) == 5, "duplicate_input_path")
        limits = (MAX_BASELINE_BYTES, MAX_PROFILE_BYTES, *([source_patch.MAX_MANIFEST_BYTES] * 3))
        inputs = tuple(_read(path, limit) for path, limit in zip(paths, limits))
        baseline = _baseline(_json(inputs[0].raw))
        profile_raw = _profile(_json(inputs[1].raw))
        original = []
        total_bytes = sum(len(record.raw) for record in inputs)
        _need(total_bytes <= MAX_INPUT_BYTES, "input_set_oversize")
        for relative, expected in sorted(baseline.items()):
            record = _read(sg.joinpath(*PurePosixPath(relative).parts), MAX_FILE_BYTES)
            total_bytes += len(record.raw)
            _need(total_bytes <= MAX_INPUT_BYTES, "input_set_oversize")
            _need(_sha(record.raw) == expected, "baseline_source_digest_mismatch")
            _source(record.raw, relative, attested_sg=True)
            original.append((relative, record))
        plugin_records = []
        for relative, kind, _signature in trees[1].entries:
            if not relative:
                continue
            _need(kind == "file" and "/" not in relative and relative.endswith(".py"), "plugin_flat_source_only")
            _relative(relative)
            record = _read(plugin / relative, MAX_FILE_BYTES)
            total_bytes += len(record.raw)
            _need(total_bytes <= MAX_INPUT_BYTES, "input_set_oversize")
            _source(record.raw, relative, attested_sg=False)
            plugin_records.append((relative, record))
        required_modules = {"__init__.py", "native_profile.py", "native_install.py",
                            "plugin.py", "startup_state.py"}
        _need(required_modules <= {path for path, _ in plugin_records}, "plugin_package_missing")
        _need(len(original) + len(plugin_records) <= MAX_SOURCE_FILES, "source_attestation_file_count")
        manifests, plans, manifest_files, patched_paths, patch_sets, names = [], [], [], set(), set(), set()
        for record in inputs[2:]:
            name = record.path.name
            _need(_NAME.fullmatch(name) is not None and name not in (".", "..")
                  and name.casefold() not in names, "invalid_manifest_name")
            names.add(name.casefold())
            manifest = _json(record.raw)
            # Prepare each whole manifest before any mutation; baseline is an
            # independent required input, never copied from manifest claims.
            plan = source_patch.prepare(sg, manifest)
            _need(plan.patch_set not in patch_sets, "duplicate_patch_set")
            patch_sets.add(plan.patch_set)
            for item in plan.files:
                _need(item.relative.casefold() not in patched_paths, "overlapping_manifest_paths")
                patched_paths.add(item.relative.casefold())
                _need(item.relative in baseline and _sha(item.original) == baseline[item.relative],
                      "manifest_not_covered_by_baseline")
                _source(item.patched, item.relative, attested_sg=True)
            plans.append(plan)
            manifest_files.extend(manifest["files"])
            manifests.append((name, _sha(record.raw)))
        files = tuple(sorted((item for plan in plans for item in plan.files), key=lambda item: item.relative))
        _need(len(files) <= source_patch.MAX_FILES
              and sum(len(item.original) + len(item.patched) for item in files) <= source_patch.MAX_TOTAL_BYTES,
              "merged_patch_set_oversize")
        manifest_entries = [{"name": name, "sha256": digest} for name, digest in sorted(manifests)]
        merged_manifest = {"schema": source_patch.SCHEMA, "source_revision": SOURCE_REVISION,
                           "source_version": SOURCE_VERSION, "patch_set": "occ3-native-package",
                           "files": sorted(manifest_files, key=lambda item: item["path"])}
        merged_raw = _canonical(merged_manifest)
        _need(len(merged_raw) <= source_patch.MAX_MANIFEST_BYTES, "merged_manifest_oversize")
        overlay = source_patch.PatchPlan(sg, "occ3-native-package", _sha(merged_raw), files)
        final_bytes = {item.relative: item.patched for item in files}
        source_bytes = (sum(len(final_bytes.get(path, record.raw)) for path, record in original)
                        + sum(len(record.raw) for _, record in plugin_records))
        _need(source_bytes <= MAX_SOURCE_TOTAL, "produced_source_total_oversize")
        source_entries = [{"root": "sglang_srt", "path": path, "sha256": _sha(final_bytes.get(path, record.raw))}
                          for path, record in original]
        source_entries.extend({"root": "occ3_sglang", "path": path, "sha256": _sha(record.raw)}
                              for path, record in plugin_records)
        sources = {"schema": SOURCES_SCHEMA, "source_revision": SOURCE_REVISION,
                   "sglang_version": SOURCE_VERSION, "manifests": manifest_entries, "files": source_entries}
        source_raw = _canonical(sources)
        _need(len(source_raw) <= MAX_SOURCES_BYTES, "source_attestation_oversize")
        digest_raw = ("# Baked disposable-image packaging digests; not runtime readiness.\n"
                      "PROFILE_SHA256 = '" + _sha(profile_raw) + "'\n"
                      "SOURCE_SHA256 = '" + _sha(source_raw) + "'\n"
                      "EXPERIMENTAL_STARTUP = " + repr(experimental_startup) + "\n").encode("ascii")
        _source(digest_raw, OUTPUT_NAMES[2], attested_sg=False)
        result = PackagePlan(sg, plugin, inputs, tuple(original), tuple(plugin_records), trees, overlay,
                             tuple(sorted(manifests)), _sha(inputs[0].raw), experimental_startup, source_bytes,
                             tuple(zip(OUTPUT_NAMES, (profile_raw, source_raw, digest_raw))))
        _recheck_inputs(result)
        return result
    except PackageError:
        raise
    except source_patch.PatchError:
        raise PackageError("source_patch_validation_failed") from None
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise PackageError("package_validation_failed") from None


def receipt(plan, *, applied=False):
    return {"ok": True, "schema": RECEIPT_SCHEMA, "source_revision": SOURCE_REVISION,
            "sglang_version": SOURCE_VERSION, "applied": applied, "runtime_imported": False,
            "gpu_proof": False, "readiness_proof": False, "baseline_sha256": plan.baseline_sha256,
            "experimental_startup_selected": plan.experimental_startup,
            "produced_source_bytes": plan.source_bytes,
            "manifests": [{"name": name, "sha256": digest} for name, digest in plan.manifests],
            "sg_source_files": len(plan.sg_files), "plugin_source_files": len(plan.plugin_files),
            "patched_source_files": len(plan.overlay.files),
            "merged_manifest_sha256": plan.overlay.manifest_sha256,
            "artifacts": [{"name": name, "sha256": _sha(raw)} for name, raw in plan.artifacts]}


def _verify_source_result(plan):
    changed = {item.relative: item for item in plan.overlay.files}
    for relative, record in plan.sg_files:
        actual = _read(record.path, MAX_FILE_BYTES)
        if relative in changed:
            _need(actual.raw == changed[relative].patched and actual.mode == record.mode, "applied_source_mismatch")
        else:
            _need(actual == record, "unmodified_source_drift")
    for _, record in plan.plugin_files:
        _need(_read(record.path, MAX_FILE_BYTES) == record, "plugin_source_drift")
    for record in plan.inputs:
        _need(_read(record.path, max(len(record.raw), 1)) == record, "input_identity_drift")


def _verify_tree_result(plan, *, outputs):
    for before in plan.trees:
        after = _tree(before.root)
        _need(after.parents == before.parents, "source_parent_drift")
        previous = {name: (kind, signature) for name, kind, signature in before.entries}
        current = {name: (kind, signature) for name, kind, signature in after.entries}
        additions = set(OUTPUT_NAMES) if outputs and before.root == plan.occ3_sglang else set()
        _need(set(current) == set(previous) | additions, "source_tree_membership_drift")
        modified = {item.relative for item in plan.overlay.files} if before.root == plan.sglang_srt else set()
        for name, (kind, signature) in previous.items():
            now_kind, now = current[name]
            _need(now_kind == kind, "source_tree_type_drift")
            if kind == "directory":
                # Directory identity pins dev, inode, mode and the metadata tail.
                # st_nlink is deliberately excluded: it counts child directories and
                # is a filesystem-implementation detail — on overlay2 a copied-up
                # directory's merged view collapses nlink even though the directory
                # itself is untouched (dev/ino/mode/uid/gid stay stable). A wholesale
                # directory replacement still fails on dev/ino change.
                _need((signature[0:3], signature[7:]) == (now[0:3], now[7:]), "source_directory_identity_drift")
            elif name not in modified:
                _need(now == signature, "unmodified_tree_file_drift")


def _write_fresh(root, name, raw):
    _parents(root)
    path = root / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(raw)
            handle.flush()
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        if os.name == "posix":
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if fd is not None:
            os.close(fd)


def apply(plan, *, disposable=False):
    """All validation precedes writes. Any failure mandates discarding the tree."""
    _need(disposable is True, "explicit_disposable_tree_required")
    try:
        _recheck_inputs(plan)
        source_patch.apply(plan.overlay)
        _verify_source_result(plan)
        _verify_tree_result(plan, outputs=False)
        _fresh(plan.occ3_sglang)
        for name, raw in plan.artifacts:
            _write_fresh(plan.occ3_sglang, name, raw)
        _verify_source_result(plan)
        _verify_tree_result(plan, outputs=True)
        for name, raw in plan.artifacts:
            actual = _read(plan.occ3_sglang / name, MAX_SOURCES_BYTES)
            _need(actual.raw == raw and (os.name != "posix" or actual.mode == 0o644), "generated_artifact_mismatch")
        return receipt(plan, applied=True)
    except PackageError:
        raise
    except BaseException:
        raise PackageError("package_apply_failed_discard_tree") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "apply"))
    parser.add_argument("--sglang-srt", type=Path, required=True)
    parser.add_argument("--occ3-sglang", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--disposable-tree", action="store_true")
    parser.add_argument("--experimental-startup", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "apply":
            _need(args.disposable_tree, "explicit_disposable_tree_required")
        plan = prepare(args.sglang_srt, args.occ3_sglang, args.baseline_json, args.profile_json, args.manifest,
                       experimental_startup=args.experimental_startup)
        result = apply(plan, disposable=True) if args.action == "apply" else receipt(plan)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except PackageError as error:
        print(json.dumps({"ok": False, "code": error.code, "runtime_imported": False, "gpu_proof": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
