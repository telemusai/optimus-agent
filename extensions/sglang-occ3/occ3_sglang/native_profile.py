"""Pinned CPU-only metadata and a shared Linux serving-container incarnation.

Nothing runs on import. These loaders are only for explicit trusted installation.
The image and its baked digest module, root, and the effective UID are trusted.
This is not isolation from a malicious same-UID administrator. A worker replacement
under the same PID-namespace init must separately invalidate all-rank membership.
"""
from __future__ import annotations

import errno
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from types import MappingProxyType

from .latent_types import ALIGNMENT as PER_MESSAGE_ALIGNMENT
from .latent_types import METHOD as PER_MESSAGE_METHOD
from .statebridge_calibrated import ARTIFACT_SCHEMA as CALIBRATED_SCHEMA
from .statebridge_calibrated import METHOD_ID as CALIBRATED_METHOD
from .statebridge_calibrated_procrustes import ARTIFACT_SCHEMA as PROCRUSTES_SCHEMA
from .statebridge_calibrated_procrustes import METHOD_ID as PROCRUSTES_METHOD
from .statebridge_calibrated_procrustes import METHOD_ID_PLAIN as PROCRUSTES_METHOD_PLAIN
from .statebridge_calibrated_procrustes import METHOD_ID_SO as PROCRUSTES_METHOD_SO

__all__ = ["ProfileError", "load_pinned_profile", "load_source_attestation",
           "get_runtime_incarnation", "PER_MESSAGE_METHOD", "CALIBRATED_METHOD",
           "advertised_methods", "registered_methods", "method_descriptor",
           "require_method", "receipt_descriptor", "validate_method_binding"]

_PROFILE_LIMIT = 16384
_SOURCE_LIMIT = 65536
_RECORD_LIMIT = 2048
_U64_MAX = (1 << 64) - 1
_LOCK_TIMEOUT = 2.0
_LOCK_INTERVAL = 0.01
_RUNTIME_DIRECTORY = "optimus-occ3-native"
_RECORD_NAME = "incarnation.json"
_LOCK_NAME = "incarnation.lock"
_RECORD_SCHEMA = "optimus.occ3.runtime-incarnation.v1"
_SOURCE_SCHEMA = "optimus.occ3.native-sources.v1"
_SOURCE_REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_PROFILE_FIELDS = frozenset({
    "model_fingerprint", "model_id", "expected_ranks", "hidden_dim", "layer_index",
    "max_prefix_tokens", "max_live_bytes", "workspace_bytes", "max_records",
    "max_active", "max_ttl_seconds", "anchor_alpha", "tile_rows", "method",
})
# The frozen-artifact reference (absolute path plus expected artifact_digest) is
# optional in a pinned profile: the per-message method names none, and an older
# pinned file stays valid. load_pinned_profile always returns the pair, empty
# when absent, so the runtime profile has one shape.
_OPTIONAL_PROFILE_FIELDS = frozenset({"calibration_path", "calibration_digest"})
_RUNTIME_FIELDS = frozenset({"runtime_epoch", "runtime_generation",
                            "incarnation_digest", "profile_sha256",
                            "source_attestation_sha256"})
_ERRORS = frozenset({"profile_invalid", "profile_io", "profile_digest",
                     "profile_unsafe", "profile_oversize", "runtime_unsupported",
                     "runtime_io", "runtime_unsafe", "runtime_record",
                     "runtime_parent", "runtime_busy", "runtime_profile_changed",
                     "runtime_generation_exhausted"})


class ProfileError(RuntimeError):
    """Only a fixed code is safe to expose or log."""

    def __init__(self, code="profile_invalid"):
        self.code = code if code in _ERRORS else "profile_invalid"
        super().__init__(self.code)


def _require(condition, code="profile_invalid"):
    if not condition:
        raise ProfileError(code)

# Advertised native methods. One representation per method id: the calibrated
# map is a different map from the per-message Procrustes fit, so it is a
# different method and can never be served under the per-message id. The
# representation string describes the captured state (which both methods
# consume), not the map; the source receipt carries the state representation
# and the negotiated method separately.
_METHOD_FIELDS = frozenset({"method_id", "representation", "alignment", "fit_source",
                            "solver", "predictor_position_offset", "precision",
                            "requires_calibration_artifact", "artifact_module",
                            "artifact_schema"})
_RECEIPT_FIELDS = ("method_id", "representation", "alignment", "predictor_position_offset",
                   "precision")


def _method_entry(method_id, alignment, fit_source, solver, requires_calibration_artifact,
                  artifact_module, artifact_schema):
    return {"method_id": method_id, "representation": "final_layer_prenorm_predictor",
            "alignment": alignment, "fit_source": fit_source, "solver": solver,
            "predictor_position_offset": -1, "precision": "float32",
            "requires_calibration_artifact": requires_calibration_artifact,
            "artifact_module": artifact_module, "artifact_schema": artifact_schema}


# The advertised table is also the offline deployment surface: the packaging
# static parse reads exactly this assignment (deploy/v3/native_package
# _static_registry), and a pinned profile may name only an advertised id.
_METHODS = {entry["method_id"]: MappingProxyType(entry) for entry in (
    _method_entry(PER_MESSAGE_METHOD, PER_MESSAGE_ALIGNMENT, "message-rows",
                  "occ3_sglang.statebridge.align_hidden_sequence", False, None, None),
    _method_entry(CALIBRATED_METHOD, "frozen-dual-ridge.v1", "independent-calibration-turns",
                  "occ3_sglang.statebridge_calibrated.apply_map", True,
                  "occ3_sglang.statebridge_calibrated", CALIBRATED_SCHEMA),
)}
_METHOD_IDS = tuple(_METHODS)
# Registered challengers (spec section 10): dispatchable frozen methods whose
# own artifact modules live in this same registry. They are not yet pinnable by
# a packaged profile: the packaging static parse reads only the table above, so
# merging them into the advertised surface belongs to the deployment wave.
_CHALLENGER_METHODS = {entry["method_id"]: MappingProxyType(entry) for entry in (
    _method_entry(PROCRUSTES_METHOD, "frozen-lowrank-whitened-procrustes.v1",
                  "independent-calibration-turns",
                  "occ3_sglang.statebridge_calibrated_procrustes.apply_map", True,
                  "occ3_sglang.statebridge_calibrated_procrustes", PROCRUSTES_SCHEMA),
    _method_entry(PROCRUSTES_METHOD_SO, "frozen-lowrank-whitened-procrustes-so.v1",
                  "independent-calibration-turns",
                  "occ3_sglang.statebridge_calibrated_procrustes.apply_map", True,
                  "occ3_sglang.statebridge_calibrated_procrustes", PROCRUSTES_SCHEMA),
    _method_entry(PROCRUSTES_METHOD_PLAIN, "frozen-lowrank-whitened-procrustes-plain.v1",
                  "independent-calibration-turns",
                  "occ3_sglang.statebridge_calibrated_procrustes.apply_map", True,
                  "occ3_sglang.statebridge_calibrated_procrustes", PROCRUSTES_SCHEMA),
)}
_REGISTERED = MappingProxyType({**_METHODS, **_CHALLENGER_METHODS})
_REGISTERED_IDS = tuple(_REGISTERED)


def _validate_method(descriptor):
    _require(set(descriptor) == _METHOD_FIELDS)
    for name in ("method_id", "representation", "alignment", "fit_source", "solver"):
        item = descriptor[name]
        _require(type(item) is str and _IDENTIFIER.fullmatch(item) is not None)
    _require(descriptor["solver"].count(".") == 2)
    _require(type(descriptor["predictor_position_offset"]) is int
             and descriptor["predictor_position_offset"] == -1)
    _require(descriptor["precision"] == "float32")
    _require(type(descriptor["requires_calibration_artifact"]) is bool)
    if descriptor["requires_calibration_artifact"]:
        for name in ("artifact_module", "artifact_schema"):
            item = descriptor[name]
            _require(type(item) is str and _IDENTIFIER.fullmatch(item) is not None)
    else:
        _require(descriptor["artifact_module"] is None and descriptor["artifact_schema"] is None)
    return descriptor


def advertised_methods():
    """Every method a pinned profile may name, in registry order.

    The packaging path parses exactly this table offline, so it is the
    shippable surface; the registered challengers are dispatchable for
    research through registered_methods() but are not yet pinnable here.
    """
    for descriptor in _METHODS.values():
        _validate_method(descriptor)
    return _METHOD_IDS


def registered_methods():
    """Every method id this registry holds, advertised or challenger, in registry order."""
    for descriptor in _REGISTERED.values():
        _validate_method(descriptor)
    return _REGISTERED_IDS


def method_descriptor(method_id):
    """Immutable identity of one registered method and its solver."""
    _require(type(method_id) is str and method_id in _REGISTERED)
    return _validate_method(_REGISTERED[method_id])


def require_method(method_id):
    """Return method_id when it is advertised; a fixed code otherwise."""
    method_descriptor(method_id)
    return method_id


def receipt_descriptor(method_id):
    """The scalar method identity a source receipt may carry. No calibration data."""
    descriptor = method_descriptor(method_id)
    return {name: descriptor[name] for name in _RECEIPT_FIELDS}


def validate_method_binding(binding, profile, *, negotiated_method_id=None):
    """Refuse a calibration binding that does not match the negotiated method.

    The per-message method takes no artifact at all, so a calibrated artifact
    presented under it is refused instead of served. Every artifact-requiring
    method dispatches its binding type through the registry: the binding class
    must live in the method's own artifact module, and its method id,
    dimensions, layer, model fingerprint, precision and predictor position all
    equal the pinned profile and the registry entry. No single binding class
    is assumed for all methods.
    """
    _validate_profile(profile)
    negotiated = profile["method"] if negotiated_method_id is None else negotiated_method_id
    _require(negotiated == profile["method"])
    descriptor = method_descriptor(negotiated)
    if not descriptor["requires_calibration_artifact"]:
        _require(binding is None)
        return descriptor
    _require(binding is not None
             and type(binding).__module__ == descriptor["artifact_module"], "profile")
    _require(binding.method_id == negotiated)
    _require(binding.hidden_dim == profile["hidden_dim"])
    _require(binding.layer_index == profile["layer_index"])
    _require(binding.model_fingerprint == profile["model_fingerprint"])
    _require(binding.precision == descriptor["precision"])
    _require(binding.predictor_position_offset == descriptor["predictor_position_offset"])
    return descriptor


def _sha(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _uint(value, minimum=0):
    return type(value) is int and minimum <= value <= _U64_MAX


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _json_integer(value):
    _require(len(value) <= 20)
    number = int(value)
    _require(-(1 << 63) <= number <= _U64_MAX)
    return number


def _invalid_constant(_value):
    raise ProfileError()


def _decode_json(raw, limit):
    _require(0 < len(raw) <= limit, "profile_oversize")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                       parse_int=_json_integer, parse_constant=_invalid_constant)

    def check(item, depth=0):
        _require(depth <= 8)
        if type(item) is dict:
            _require(len(item) <= 64)
            for key, child in item.items():
                _require(len(key) <= 128)
                check(child, depth + 1)
        elif type(item) is list:
            _require(len(item) <= 128)
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            _require(len(item) <= 1024)
        else:
            _require(type(item) in (int, float, bool) or item is None)
            if type(item) is float:
                _require(math.isfinite(item))
    check(value)
    _require(type(value) is dict)
    return value


def _read_bounded(fd, limit, code):
    chunks = []
    remaining = limit + 1
    while remaining:
        chunk = os.read(fd, min(remaining, 8192))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    _require(len(raw) <= limit, code)
    return raw


def _same_file(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _not_reparse(info):
    return not (getattr(info, "st_file_attributes", 0) & 0x400)


def _check_regular(info, *, private, code):
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
             and _not_reparse(info), code)
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if private:
            _require(info.st_uid == os.geteuid() and mode == 0o600, code)
        else:
            _require(info.st_uid in (0, os.geteuid()) and not mode & 0o7022, code)


def _directory_flags():
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _check_parent(info, *, sticky=False):
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid in (0, os.geteuid()),
             "runtime_unsafe")
    mode = stat.S_IMODE(info.st_mode)
    _require(not mode & 0o022 or (sticky and info.st_uid == 0 and mode & stat.S_ISVTX),
             "runtime_unsafe")


def _open_absolute_directory(path):
    """Walk trusted absolute parents without following any filesystem symlink."""
    _require(os.path.isabs(path), "runtime_unsafe")
    fd = os.open("/", _directory_flags())
    try:
        _check_parent(os.fstat(fd))
        for component in path.split("/"):
            if not component:
                continue
            _require(component not in (".", ".."), "runtime_unsafe")
            child = os.open(component, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = child
            _check_parent(os.fstat(fd), sticky=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _baked_digest(attribute):
    baked = importlib.import_module("._native_profile_digest", __package__)
    value = getattr(baked, attribute, None)
    _require(_sha(value), "profile_digest")
    return value


def _read_packaged(name, limit):
    directory = os.path.dirname(__file__)
    fd = None
    parent = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if os.name == "posix":
            parent = _open_absolute_directory(directory)
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _check_regular(before, private=False, code="profile_unsafe")
            fd = os.open(name, flags, dir_fd=parent)
        else:
            # Windows can test artifacts, but cannot establish a runtime incarnation.
            parent_info = os.lstat(directory)
            _require(stat.S_ISDIR(parent_info.st_mode) and _not_reparse(parent_info),
                     "profile_unsafe")
            path = os.path.join(directory, name)
            before = os.lstat(path)
            _check_regular(before, private=False, code="profile_unsafe")
            fd = os.open(path, flags)
        actual = os.fstat(fd)
        _check_regular(actual, private=False, code="profile_unsafe")
        _require(_same_file(before, actual), "profile_unsafe")
        _require(0 < actual.st_size <= limit, "profile_oversize")
        raw = _read_bounded(fd, limit, "profile_oversize")
        after = os.fstat(fd)
        _require(_same_file(actual, after) and len(raw) == after.st_size,
                 "profile_unsafe")
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        if parent is not None:
            os.close(parent)


def _artifact(name, attribute, limit):
    expected = _baked_digest(attribute)
    raw = _read_packaged(name, limit)
    _require(hashlib.sha256(raw).hexdigest() == expected, "profile_digest")
    return _decode_json(raw, limit), expected


def _validate_profile(value):
    _require(_PROFILE_FIELDS <= set(value) <= _PROFILE_FIELDS | _OPTIONAL_PROFILE_FIELDS)
    path_value = value.get("calibration_path", "")
    digest_value = value.get("calibration_digest", "")
    _require(type(path_value) is str and type(digest_value) is str
             and bool(path_value) == bool(digest_value))
    if path_value:
        # An absolute path, no control bytes, and a digest that pins its content.
        _require(0 < len(path_value) <= 512 and path_value.isprintable()
                 and os.path.isabs(path_value))
        _require(_HEX.fullmatch(digest_value) is not None)
        _require(value["method"] != PER_MESSAGE_METHOD)
    for name in ("model_fingerprint", "model_id", "method"):
        item = value[name]
        _require(type(item) is str and _IDENTIFIER.fullmatch(item) is not None)
    # A pinned profile may only name an advertised method; an unknown or a
    # silently substituted representation is refused before any runtime use.
    _require(value["method"] in _METHODS)
    for name in _PROFILE_FIELDS - {"model_fingerprint", "model_id", "method", "anchor_alpha"}:
        _require(_uint(value[name], minimum=0 if name == "layer_index" else 1))
    _require(type(value["anchor_alpha"]) in (int, float)
             and math.isfinite(value["anchor_alpha"]))
    # RuntimeProfile owns semantic/model/TP/cache validation. No second profile type.
    return value


def _validate_sources(value):
    _require(set(value) == {"schema", "source_revision", "sglang_version", "manifests", "files"})
    _require(value["schema"] == _SOURCE_SCHEMA and value["source_revision"] == _SOURCE_REVISION
             and value["sglang_version"] == "0.5.21")
    manifests, files = value["manifests"], value["files"]
    _require(type(manifests) is list and 1 <= len(manifests) <= 128)
    _require(type(files) is list and 1 <= len(files) <= 128)
    seen = set()
    for entry in manifests:
        _require(type(entry) is dict and set(entry) == {"name", "sha256"})
        name = entry["name"]
        _require(type(name) is str and _NAME.fullmatch(name) is not None
                 and name not in (".", "..") and name not in seen and _sha(entry["sha256"]))
        seen.add(name)
    seen.clear()
    for entry in files:
        _require(type(entry) is dict and set(entry) == {"root", "path", "sha256"})
        _require(entry["root"] in ("sglang_srt", "occ3_sglang") and _sha(entry["sha256"]))
        path = entry["path"]
        _require(type(path) is str and len(path) <= 512 and path.endswith(".py"))
        parts = path.split("/")
        _require(all(_NAME.fullmatch(part) is not None and part not in (".", "..")
                     for part in parts))
        _require(path != "_native_profile_digest.py" or entry["root"] != "occ3_sglang")
        key = (entry["root"], path)
        _require(key not in seen)
        seen.add(key)
    return value


def load_pinned_profile() -> dict:
    """Return the exact flat static profile; never an epoch or readiness claim."""
    try:
        value, _ = _artifact("native_profile.json", "PROFILE_SHA256", _PROFILE_LIMIT)
        return _validate_profile(value)
    except ProfileError as exc:
        raise ProfileError(exc.code) from None
    except Exception:
        raise ProfileError("profile_io") from None


def load_source_attestation() -> dict:
    """Return pinned expectations, not proof of the installed source contents."""
    try:
        value, _ = _artifact("native_sources.json", "SOURCE_SHA256", _SOURCE_LIMIT)
        return _validate_sources(value)
    except ProfileError as exc:
        raise ProfileError(exc.code) from None
    except Exception:
        raise ProfileError("profile_io") from None


def _parse_pid1_start(raw):
    _require(0 < len(raw) <= 4096 and raw.startswith(b"1 (") and b"\0" not in raw,
             "runtime_parent")
    closing = raw.rfind(b")")
    _require(closing >= 3 and raw[closing + 1:closing + 2] == b" ", "runtime_parent")
    fields = raw[closing + 2:].split()
    # fields[0] is field 3 (state), so starttime (field 22) is fields[19].
    _require(len(fields) >= 20 and len(fields[0]) == 1
             and fields[0] in (b"R", b"S", b"D", b"Z", b"T", b"t", b"X", b"x", b"K", b"W", b"P", b"I"),
             "runtime_parent")
    token = fields[19]
    _require(1 <= len(token) <= 20 and token.isdigit(), "runtime_parent")
    start = int(token)
    _require(_uint(start), "runtime_parent")
    return start


def _parent_digest(boot_raw, namespace_inode, pid1_raw):
    _require(0 < len(boot_raw) <= 128 and _uint(namespace_inode, 1), "runtime_parent")
    boot = boot_raw.decode("ascii").strip()
    _require(str(uuid.UUID(boot)) == boot, "runtime_parent")
    value = {"boot_id": boot, "pid_namespace_inode": namespace_inode,
             "pid1_start_ticks": _parse_pid1_start(pid1_raw)}
    return hashlib.sha256(_canonical(value)).hexdigest()


def _kernel_bytes(directory, name, maximum):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                 dir_fd=directory)
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), "runtime_parent")
        return _read_bounded(fd, maximum, "runtime_parent")
    finally:
        os.close(fd)


def _namespace_inode(proc_fd, pid):
    _require(pid in ("self", 1), "runtime_parent")
    # /proc/self resolves the caller even if procfs was mounted in an ancestor
    # PID namespace. A numeric getpid() lookup would inspect the wrong process.
    flags = _directory_flags() if pid == 1 else os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    process = os.open(str(pid), flags, dir_fd=proc_fd)
    namespace = None
    try:
        namespace = os.open("ns", _directory_flags(), dir_fd=process)
        # This one kernel-owned procfs magic link must be followed to stat nsfs.
        return os.stat("pid", dir_fd=namespace).st_ino
    finally:
        if namespace is not None:
            os.close(namespace)
        os.close(process)


def _read_parent_digest():
    proc = _open_absolute_directory("/proc")
    boot_dir = init = None
    try:
        boot_dir = _open_absolute_directory("/proc/sys/kernel/random")
        boot = _kernel_bytes(boot_dir, "boot_id", 128)
        namespace_inode = _namespace_inode(proc, "self")
        _require(namespace_inode == _namespace_inode(proc, 1), "runtime_parent")
        init = os.open("1", _directory_flags(), dir_fd=proc)
        return _parent_digest(boot, namespace_inode, _kernel_bytes(init, "stat", 4096))
    finally:
        if init is not None:
            os.close(init)
        if boot_dir is not None:
            os.close(boot_dir)
        os.close(proc)


def _open_runtime_directory():
    parent = _open_absolute_directory("/tmp")
    directory = None
    try:
        try:
            os.mkdir(_RUNTIME_DIRECTORY, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
        directory = os.open(_RUNTIME_DIRECTORY, _directory_flags(), dir_fd=parent)
        info = os.fstat(directory)
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                 and stat.S_IMODE(info.st_mode) == 0o700, "runtime_unsafe")
        named = os.stat(_RUNTIME_DIRECTORY, dir_fd=parent, follow_symlinks=False)
        _require(_same_file(info, named), "runtime_unsafe")
        result, directory = directory, None
        return result
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


@contextmanager
def _locked_directory():
    import fcntl

    directory = _open_runtime_directory()
    lock = None
    try:
        lock = os.open(_LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                       | os.O_CLOEXEC | os.O_NONBLOCK, 0o600, dir_fd=directory)
        info = os.fstat(lock)
        _check_regular(info, private=True, code="runtime_unsafe")
        _require(info.st_size == 0, "runtime_unsafe")
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EINTR):
                    raise
                _require(time.monotonic() < deadline, "runtime_busy")
                time.sleep(_LOCK_INTERVAL)
        named = os.stat(_LOCK_NAME, dir_fd=directory, follow_symlinks=False)
        _check_regular(named, private=True, code="runtime_unsafe")
        _require(_same_file(info, named), "runtime_unsafe")
        yield directory
    finally:
        # close releases flock; never unlink/recreate the shared lock inode.
        if lock is not None:
            os.close(lock)
        os.close(directory)


def _read_record(directory):
    try:
        fd = os.open(_RECORD_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                     | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        _check_regular(info, private=True, code="runtime_unsafe")
        _require(0 < info.st_size <= _RECORD_LIMIT, "runtime_record")
        try:
            value = _decode_json(_read_bounded(fd, _RECORD_LIMIT, "runtime_record"),
                                 _RECORD_LIMIT)
            _require(set(value) == _RUNTIME_FIELDS | {"schema"}, "runtime_record")
            _require(value["schema"] == _RECORD_SCHEMA, "runtime_record")
            epoch = value["runtime_epoch"]
            _require(type(epoch) is str and len(epoch) == 32, "runtime_record")
            parsed = uuid.UUID(hex=epoch)
            _require(parsed.hex == epoch and parsed.version == 4
                     and parsed.variant == uuid.RFC_4122, "runtime_record")
            _require(_uint(value["runtime_generation"], 1), "runtime_record")
            for name in ("incarnation_digest", "profile_sha256", "source_attestation_sha256"):
                _require(_sha(value[name]), "runtime_record")
            return value
        except Exception:
            raise ProfileError("runtime_record") from None
    finally:
        os.close(fd)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _write_record(directory, value):
    temporary = ".incarnation-" + uuid.uuid4().hex + ".tmp"
    fd = None
    created = None
    replaced = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                     | os.O_CLOEXEC | os.O_NONBLOCK, 0o600, dir_fd=directory)
        created = os.fstat(fd)
        _check_regular(created, private=True, code="runtime_unsafe")
        raw = _canonical(value)
        _require(len(raw) <= _RECORD_LIMIT, "runtime_record")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            _require(written > 0, "runtime_io")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, _RECORD_NAME, src_dir_fd=directory, dst_dir_fd=directory)
        replaced = True
        os.fsync(directory)
    finally:
        if fd is not None:
            os.close(fd)
        if created is not None and not replaced:
            # Remove only this call's own unchanged temporary inode, never sweep.
            try:
                named = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
                if _same_file(created, named):
                    os.unlink(temporary, dir_fd=directory)
            except OSError:
                pass


def get_runtime_incarnation() -> dict:
    """Converge frontend and TP workers on one parent-bound opaque identity.

    A new boot/PID namespace/PID1 birth advances a valid retained generation. If
    /tmp was lost, generation starts at 1 but the random epoch is still fresh.
    No process-local cache, environment override, model import, or GPU work.
    """
    _require(sys.platform.startswith("linux") and os.name == "posix", "runtime_unsupported")
    try:
        profile, profile_sha = _artifact("native_profile.json", "PROFILE_SHA256", _PROFILE_LIMIT)
        sources, source_sha = _artifact("native_sources.json", "SOURCE_SHA256", _SOURCE_LIMIT)
        _validate_profile(profile)
        _validate_sources(sources)
        with _locked_directory() as directory:
            incarnation = _read_parent_digest()
            previous = _read_record(directory)
            if previous is not None and previous["incarnation_digest"] == incarnation:
                _require(previous["profile_sha256"] == profile_sha
                         and previous["source_attestation_sha256"] == source_sha,
                         "runtime_profile_changed")
                value = previous
            else:
                generation = 1 if previous is None else previous["runtime_generation"] + 1
                _require(generation <= _U64_MAX, "runtime_generation_exhausted")
                value = {"schema": _RECORD_SCHEMA, "runtime_epoch": uuid.uuid4().hex,
                         "runtime_generation": generation, "incarnation_digest": incarnation,
                         "profile_sha256": profile_sha, "source_attestation_sha256": source_sha}
                _write_record(directory, value)
            return {name: value[name] for name in sorted(_RUNTIME_FIELDS)}
    except ProfileError as exc:
        raise ProfileError(exc.code) from None
    except Exception:
        raise ProfileError("runtime_io") from None
