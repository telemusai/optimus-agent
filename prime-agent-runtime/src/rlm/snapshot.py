"""Versioned durable storage for Python kernel namespace snapshots.

The legacy reader/writer stays in :mod:`rlm.repl`. This module implements the
explicitly opted-in CAS v2 representation. It never guesses whether a Python
value changed: callers serialize every eligible name on the kernel thread.
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import hashlib
import json
import os
import re
import signal
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from .snapshot_serializer import SnapshotSerializationMetrics, dump_snapshot_value
from .snapshot_restore import prepare_restored_values

CAS_FORMAT = "prime-agent-kernel-snapshot-cas"
CAS_VERSION = 2
CAS_FORMAT_MARKER = b"prime-agent-kernel-snapshot-cas-v2\n"
FORMAT_FILENAME = "FORMAT"
CURRENT_FILENAME = "CURRENT.json"
BLOBS_DIRNAME = "blobs"
GENERATIONS_DIRNAME = "generations"
MAX_POINTER_BYTES = 16 * 1024
MAX_GENERATION_BYTES = 512 * 1024 * 1024
DEFAULT_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES = 16 * 1024 * 1024
MAX_GC_ENTRIES = 4096
MAX_GC_DELETIONS = 128

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")
_ACTIVE_RESTORE_LOCK = threading.RLock()
_ACTIVE_RESTORE_BLOBS: dict[str, dict[str, int]] = {}


class SnapshotStoreError(Exception):
    """A visible snapshot format, integrity, or persistence failure."""


class SnapshotSizeLimitExceeded(Exception):
    pass


class CappedWriter:
    def __init__(self, sink: Any, limit: int) -> None:
        self._sink = sink
        self._limit = limit
        self.written = 0

    def write(self, chunk: Any) -> int:
        size = len(chunk)
        if self.written + size > self._limit:
            raise SnapshotSizeLimitExceeded()
        self._sink.write(chunk)
        self.written += size
        return size


class _NullWriter:
    def write(self, chunk: Any) -> int:
        return len(chunk)


def cas_state_present(root: str) -> bool:
    """True for any filesystem entry at the v2 root, including broken links."""
    return os.path.lexists(root)


def cas_root_for_legacy_path(path: str) -> str:
    """Derive the v2 sibling used when an older host supplies only `path`."""
    return f"{path[:-len('.dill')]}.v2" if path.endswith(".dill") else f"{path}.v2"


def _thread_cpu_ns() -> int | None:
    clock = getattr(time, "thread_time_ns", None)
    return clock() if clock is not None else None


def _elapsed_ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) / 1_000_000


def _safe_str(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return "<exception str() failed>"


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _require_absolute_root(root: str) -> str:
    if not isinstance(root, str) or not os.path.isabs(root):
        raise SnapshotStoreError("CAS root must be an absolute path")
    return os.path.abspath(root)


def _require_directory(path: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SnapshotStoreError(f"snapshot directory unavailable: {_safe_str(error)}") from error
    if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise SnapshotStoreError("snapshot directory is not a plain directory")
    return info


def _owned_path(root: str, *parts: str) -> str:
    candidate = os.path.abspath(os.path.join(root, *parts))
    try:
        if os.path.commonpath((root, candidate)) != root:
            raise SnapshotStoreError("snapshot path escapes CAS root")
    except ValueError as error:
        raise SnapshotStoreError("snapshot path is on a different volume") from error
    return candidate


def _require_owned_parent(root: str, path: str) -> None:
    parent = os.path.dirname(path)
    _require_directory(root)
    _require_directory(parent)
    root_real = os.path.realpath(root)
    parent_real = os.path.realpath(parent)
    try:
        if os.path.commonpath((root_real, parent_real)) != root_real:
            raise SnapshotStoreError("snapshot parent escapes CAS root")
    except ValueError as error:
        raise SnapshotStoreError("snapshot parent is on a different volume") from error


def _read_regular_file(path: str, *, max_bytes: int, expected_size: int | None = None) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise SnapshotStoreError(f"snapshot file missing: {os.path.basename(path)}") from error
    if _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise SnapshotStoreError(f"snapshot file is not a plain file: {os.path.basename(path)}")
    if before.st_size > max_bytes:
        raise SnapshotStoreError(f"snapshot file exceeds metadata limit: {os.path.basename(path)}")
    if expected_size is not None and before.st_size != expected_size:
        raise SnapshotStoreError(f"snapshot file size mismatch: {os.path.basename(path)}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise SnapshotStoreError(f"snapshot file open failed: {os.path.basename(path)}") from error
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise SnapshotStoreError(f"snapshot file changed while opening: {os.path.basename(path)}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise SnapshotStoreError(f"snapshot file ended early: {os.path.basename(path)}")
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)

    try:
        after = os.lstat(path)
    except OSError as error:
        raise SnapshotStoreError(f"snapshot file disappeared: {os.path.basename(path)}") from error
    if _is_reparse(after) or not _same_file(opened, after) or after.st_size != len(data):
        raise SnapshotStoreError(f"snapshot file changed while reading: {os.path.basename(path)}")
    return data


def _read_json_file(path: str, *, max_bytes: int, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _read_regular_file(
        path,
        max_bytes=max_bytes,
        expected_size=expected.get("size") if expected is not None else None,
    )
    if expected is not None and hashlib.sha256(data).hexdigest() != expected.get("sha256"):
        raise SnapshotStoreError(f"snapshot metadata hash mismatch: {os.path.basename(path)}")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotStoreError(f"invalid snapshot JSON: {os.path.basename(path)}") from error
    if not isinstance(value, dict):
        raise SnapshotStoreError(f"snapshot JSON is not an object: {os.path.basename(path)}")
    return value


def _encode_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _directory_fsync_unsupported(error: OSError) -> bool:
    if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
        return True
    return os.name == "nt" and error.errno in {errno.EACCES, errno.EPERM}


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        if _directory_fsync_unsupported(error):
            return
        raise
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            if not _directory_fsync_unsupported(error):
                raise
    finally:
        os.close(fd)


def _atomic_write_bytes(path: str, data: bytes, written: list[int]) -> None:
    parent = os.path.dirname(path)
    fd, temp_path = tempfile.mkstemp(dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short snapshot write")
            written[0] += count
            view = view[count:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp_path, path)
        _fsync_directory(parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _validate_format(root: str) -> None:
    marker_path = _owned_path(root, FORMAT_FILENAME)
    _require_owned_parent(root, marker_path)
    marker = _read_regular_file(marker_path, max_bytes=128)
    if marker != CAS_FORMAT_MARKER:
        raise SnapshotStoreError("unsupported or corrupt CAS FORMAT marker")


def _initialize_for_write(root: str, written: list[int]) -> None:
    root = _require_absolute_root(root)
    parent = os.path.dirname(root)
    _require_directory(parent)
    if not os.path.lexists(root):
        os.mkdir(root)
        _fsync_directory(parent)
    _require_directory(root)

    marker_path = _owned_path(root, FORMAT_FILENAME)
    if os.path.lexists(marker_path):
        _validate_format(root)
    else:
        with os.scandir(root) as entries:
            if next(entries, None) is not None:
                raise SnapshotStoreError("CAS root has state but its FORMAT marker is missing")
        _atomic_write_bytes(marker_path, CAS_FORMAT_MARKER, written)
        _validate_format(root)

    for dirname in (BLOBS_DIRNAME, GENERATIONS_DIRNAME):
        path = _owned_path(root, dirname)
        if not os.path.lexists(path):
            os.mkdir(path)
            _fsync_directory(root)
        _require_directory(path)


def _validate_reference(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SnapshotStoreError(f"CURRENT {field} reference is invalid")
    generation = value.get("generation")
    digest = value.get("sha256")
    size = value.get("size")
    if not isinstance(generation, str) or not _GENERATION_RE.fullmatch(generation):
        raise SnapshotStoreError(f"CURRENT {field} generation is invalid")
    if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
        raise SnapshotStoreError(f"CURRENT {field} hash is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > MAX_GENERATION_BYTES:
        raise SnapshotStoreError(f"CURRENT {field} size is invalid")
    return {"generation": generation, "sha256": digest, "size": size}


def _read_pointer(root: str, *, required: bool) -> dict[str, Any] | None:
    current_path = _owned_path(root, CURRENT_FILENAME)
    _require_owned_parent(root, current_path)
    if not os.path.lexists(current_path):
        if required:
            raise SnapshotStoreError("CAS FORMAT exists but CURRENT.json is missing")
        return None
    pointer = _read_json_file(current_path, max_bytes=MAX_POINTER_BYTES)
    if pointer.get("format") != CAS_FORMAT or pointer.get("version") != CAS_VERSION:
        raise SnapshotStoreError("CURRENT.json format/version is invalid")
    current = _validate_reference(pointer.get("current"), "current")
    previous_value = pointer.get("previous")
    previous = None if previous_value is None else _validate_reference(previous_value, "previous")
    return {"format": CAS_FORMAT, "version": CAS_VERSION, "current": current, "previous": previous}


def _validate_reason_list(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise SnapshotStoreError(f"generation {field} is invalid")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("reason"), str):
            raise SnapshotStoreError(f"generation {field} entry is invalid")
        result.append({"name": item["name"], "reason": item["reason"]})
    return result


def _validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SnapshotStoreError(f"generation {field} is invalid")
    if len(set(value)) != len(value):
        raise SnapshotStoreError(f"generation {field} contains duplicates")
    return list(value)


def _read_generation(
    root: str,
    reference: dict[str, Any],
    *,
    max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    max_variable_bytes: int = DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES,
) -> dict[str, Any]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise SnapshotStoreError("restore aggregate size cap is invalid")
    if (
        isinstance(max_variable_bytes, bool)
        or not isinstance(max_variable_bytes, int)
        or max_variable_bytes < 0
    ):
        raise SnapshotStoreError("restore per-variable size cap is invalid")
    generation = reference["generation"]
    path = _owned_path(root, GENERATIONS_DIRNAME, f"{generation}.json")
    _require_owned_parent(root, path)
    value = _read_json_file(path, max_bytes=MAX_GENERATION_BYTES, expected=reference)
    if value.get("format") != CAS_FORMAT or value.get("version") != CAS_VERSION:
        raise SnapshotStoreError("generation format/version is invalid")
    if value.get("generation") != generation:
        raise SnapshotStoreError("generation identity does not match CURRENT")

    declared_max_bytes = value.get("maxBytes")
    declared_max_variable_bytes = value.get("maxVariableBytes")
    if (
        isinstance(declared_max_bytes, bool)
        or not isinstance(declared_max_bytes, int)
        or declared_max_bytes < 0
    ):
        raise SnapshotStoreError("generation aggregate size cap is invalid")
    if (
        isinstance(declared_max_variable_bytes, bool)
        or not isinstance(declared_max_variable_bytes, int)
        or declared_max_variable_bytes < 0
    ):
        raise SnapshotStoreError("generation per-variable size cap is invalid")

    entries_value = value.get("entries")
    if not isinstance(entries_value, list):
        raise SnapshotStoreError("generation entries are invalid")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    logical_bytes = 0
    for item in entries_value:
        if not isinstance(item, dict):
            raise SnapshotStoreError("generation entry is invalid")
        name = item.get("name")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(name, str) or name in names:
            raise SnapshotStoreError("generation entry name is invalid or duplicated")
        if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
            raise SnapshotStoreError("generation blob hash is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SnapshotStoreError("generation blob size is invalid")
        if size > declared_max_variable_bytes or size > max_variable_bytes:
            raise SnapshotStoreError(f"generation entry exceeds per-variable restore cap: {name}")
        names.add(name)
        logical_bytes += size
        if logical_bytes > declared_max_bytes or logical_bytes > max_bytes:
            raise SnapshotStoreError("generation entries exceed aggregate restore cap")
        entries.append({"name": name, "sha256": digest, "size": size})

    saved_names = _validate_string_list(value.get("savedNames"), "savedNames")
    if saved_names != sorted(names):
        raise SnapshotStoreError("generation savedNames do not match entries")
    skipped = _validate_reason_list(value.get("skipped"), "skipped")
    pruned = _validate_string_list(value.get("pruned"), "pruned")
    if pruned != sorted(pruned):
        raise SnapshotStoreError("generation pruned names are not sorted")
    if value.get("logicalSerializedBytes") != logical_bytes:
        raise SnapshotStoreError("generation logical byte count is invalid")
    envelope_bytes = value.get("legacyEnvelopeBytes")
    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise SnapshotStoreError("generation legacy envelope byte count is invalid")
    if envelope_bytes < logical_bytes:
        raise SnapshotStoreError("generation legacy envelope is smaller than its logical blobs")
    if envelope_bytes > declared_max_bytes or envelope_bytes > max_bytes:
        raise SnapshotStoreError("generation exceeds aggregate restore cap")

    return {
        **value,
        "entries": entries,
        "savedNames": saved_names,
        "skipped": skipped,
        "pruned": pruned,
        "logicalSerializedBytes": logical_bytes,
        "legacyEnvelopeBytes": envelope_bytes,
        "maxBytes": declared_max_bytes,
        "maxVariableBytes": declared_max_variable_bytes,
    }


def _active_root_key(root: str) -> str:
    return os.path.normcase(os.path.realpath(root))


def _register_active(root: str, digests: set[str]) -> None:
    key = _active_root_key(root)
    with _ACTIVE_RESTORE_LOCK:
        active = _ACTIVE_RESTORE_BLOBS.setdefault(key, {})
        for digest in digests:
            active[digest] = active.get(digest, 0) + 1


def _unregister_active(root: str, digests: set[str]) -> None:
    key = _active_root_key(root)
    with _ACTIVE_RESTORE_LOCK:
        active = _ACTIVE_RESTORE_BLOBS.get(key)
        if active is None:
            return
        for digest in digests:
            remaining = active.get(digest, 0) - 1
            if remaining > 0:
                active[digest] = remaining
            else:
                active.pop(digest, None)
        if not active:
            _ACTIVE_RESTORE_BLOBS.pop(key, None)


def _active_digests(root: str) -> set[str]:
    key = _active_root_key(root)
    with _ACTIVE_RESTORE_LOCK:
        return set(_ACTIVE_RESTORE_BLOBS.get(key, {}))


@contextlib.contextmanager
def open_cas_generation(
    root: str,
    source: str = "current",
    *,
    max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    max_variable_bytes: int = DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES,
) -> Iterator[tuple[dict[str, Any], dict[str, bytes]]]:
    """Validate and pin one committed generation through the caller's use."""
    root = _require_absolute_root(root)
    digests: set[str] = set()
    # Registration and GC share this lock. A supported restore therefore owns
    # the full pointer/manifest validation-to-use window, not only blob reads.
    with _ACTIVE_RESTORE_LOCK:
        _require_directory(root)
        _validate_format(root)
        pointer = _read_pointer(root, required=True)
        assert pointer is not None
        if source == "current":
            reference = pointer["current"]
        elif source == "previous":
            reference = pointer["previous"]
            if reference is None:
                raise SnapshotStoreError("no previous CAS generation is retained")
        else:
            raise SnapshotStoreError("CAS restore source must be current or previous")
        generation = _read_generation(
            root,
            reference,
            max_bytes=max_bytes,
            max_variable_bytes=max_variable_bytes,
        )
        digests = {entry["sha256"] for entry in generation["entries"]}
        _register_active(root, digests)

    try:
        blobs: dict[str, bytes] = {}
        for entry in generation["entries"]:
            digest = entry["sha256"]
            path = _owned_path(root, BLOBS_DIRNAME, f"{digest}.blob")
            _require_owned_parent(root, path)
            data = _read_regular_file(path, max_bytes=entry["size"], expected_size=entry["size"])
            if hashlib.sha256(data).hexdigest() != digest:
                raise SnapshotStoreError(f"blob hash mismatch for {digest}")
            blobs[entry["name"]] = data
        yield generation, blobs
    finally:
        _unregister_active(root, digests)


def _count_envelope(dill: Any, payload: dict[str, bytes], limit: int) -> int | None:
    writer = CappedWriter(_NullWriter(), limit)
    try:
        dill.dump(payload, writer)
    except SnapshotSizeLimitExceeded:
        return None
    return writer.written


def _serialize_namespace(
    ns: dict[str, Any],
    *,
    max_bytes: int,
    max_variable_bytes: int,
    prune_oversized: bool,
    always_skip: set[str],
    handle_type: type[Any],
) -> tuple[dict[str, bytes], list[dict[str, str]], list[str], int, int, dict[str, float | int | None]]:
    try:
        import dill
    except Exception as error:
        raise SnapshotStoreError(f"dill unavailable: {_safe_str(error)}") from error
    dill.settings["recurse"] = True

    wall_start = time.monotonic_ns()
    cpu_start = _thread_cpu_ns()
    payload: dict[str, bytes] = {}
    skipped: list[dict[str, str]] = []
    oversized: list[str] = []
    total = 0
    variable_metrics = SnapshotSerializationMetrics()
    missing = object()
    for name in list(ns.keys()):
        if not isinstance(name, str) or name.startswith("_") or name in always_skip:
            continue
        value = ns.get(name, missing)
        if value is missing:
            skipped.append({"name": name, "reason": "deleted during snapshot"})
            continue
        if isinstance(value, handle_type):
            skipped.append(
                {
                    "name": name,
                    "reason": "BashHandle is a runtime-owned process handle and cannot be snapshotted",
                }
            )
            continue
        remaining = max_bytes - total
        limit = max_variable_bytes if prune_oversized else min(max_variable_bytes, remaining)
        import io

        buffer = io.BytesIO()
        serialization_started = time.monotonic_ns()
        try:
            dump_snapshot_value(dill, value, buffer, CappedWriter(buffer, limit))
            blob = buffer.getvalue()
        except SnapshotSizeLimitExceeded:
            if not prune_oversized and remaining < max_variable_bytes:
                skipped.append({"name": name, "reason": "exceeds aggregate snapshot size cap"})
            else:
                skipped.append({"name": name, "reason": "exceeds per-variable snapshot size cap"})
                oversized.append(name)
            continue
        except Exception as error:
            skipped.append({"name": name, "reason": f"{type(error).__name__}: {_safe_str(error)[:200]}"})
            continue
        finally:
            variable_metrics.record(name, time.monotonic_ns() - serialization_started)
        if total + len(blob) > max_bytes:
            skipped.append({"name": name, "reason": "exceeds aggregate snapshot size cap"})
            continue
        payload[name] = blob
        total += len(blob)

    envelope_bytes = _count_envelope(dill, payload, max_bytes)
    if envelope_bytes is None:
        items = list(payload.items())
        empty_bytes = _count_envelope(dill, {}, max_bytes)
        if empty_bytes is None:
            raise SnapshotStoreError("snapshot exceeds aggregate snapshot size cap")
        low, high = 0, len(items) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if _count_envelope(dill, dict(items[:mid]), max_bytes) is None:
                high = mid - 1
            else:
                low = mid
        for name, _ in items[low:]:
            skipped.append({"name": name, "reason": "exceeds aggregate snapshot size cap"})
        payload = dict(items[:low])
        envelope_bytes = _count_envelope(dill, payload, max_bytes)
        if envelope_bytes is None:
            raise SnapshotStoreError("snapshot exceeds aggregate snapshot size cap")

    logical_bytes = sum(len(blob) for blob in payload.values())
    cpu_end = _thread_cpu_ns()
    wall_end = time.monotonic_ns()
    metrics: dict[str, float | int | None] = {
        "serialization_wall_ms": _elapsed_ms(wall_start, wall_end),
        "serialization_cpu_ms": _elapsed_ms(cpu_start, cpu_end),
        "serialized_bytes": logical_bytes,
        **variable_metrics.summarize(payload),
    }
    return payload, skipped, oversized, envelope_bytes, logical_bytes, metrics


def _write_blob(root: str, digest: str, blob: bytes, written: list[int]) -> None:
    path = _owned_path(root, BLOBS_DIRNAME, f"{digest}.blob")
    _require_owned_parent(root, path)
    if os.path.lexists(path):
        existing = _read_regular_file(path, max_bytes=len(blob), expected_size=len(blob))
        if hashlib.sha256(existing).hexdigest() != digest or existing != blob:
            raise SnapshotStoreError(f"existing CAS blob does not match its hash: {digest}")
        return
    _atomic_write_bytes(path, blob, written)
    persisted = _read_regular_file(path, max_bytes=len(blob), expected_size=len(blob))
    if hashlib.sha256(persisted).hexdigest() != digest or persisted != blob:
        raise SnapshotStoreError(f"persisted CAS blob validation failed: {digest}")


def _new_generation_id(root: str) -> str:
    for _ in range(16):
        generation = uuid.uuid4().hex
        path = _owned_path(root, GENERATIONS_DIRNAME, f"{generation}.json")
        if not os.path.lexists(path):
            return generation
    raise SnapshotStoreError("could not allocate a unique snapshot generation")


def _reference_for(generation: str, data: bytes) -> dict[str, Any]:
    return {"generation": generation, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _retained_state(
    root: str,
    pointer: dict[str, Any],
    *,
    max_bytes: int,
    max_variable_bytes: int,
) -> tuple[set[str], set[str]]:
    retained_generations: set[str] = set()
    retained_blobs = _active_digests(root)
    for key in ("current", "previous"):
        reference = pointer.get(key)
        if reference is None:
            continue
        generation = _read_generation(
            root,
            reference,
            max_bytes=max_bytes,
            max_variable_bytes=max_variable_bytes,
        )
        retained_generations.add(reference["generation"])
        retained_blobs.update(entry["sha256"] for entry in generation["entries"])
    return retained_generations, retained_blobs


def _bounded_gc(
    root: str,
    pointer: dict[str, Any],
    *,
    max_bytes: int,
    max_variable_bytes: int,
) -> None:
    """Delete only strict, unretained v2 files; ambiguity makes this a no-op."""
    try:
        # Holding the registration lock through unlink closes the race where a
        # restore validates a reference just as GC decides the blob is inactive.
        with _ACTIVE_RESTORE_LOCK:
            retained_generations, retained_blobs = _retained_state(
                root,
                pointer,
                max_bytes=max_bytes,
                max_variable_bytes=max_variable_bytes,
            )
            planned: list[str] = []
            specs = (
                (GENERATIONS_DIRNAME, re.compile(r"^([0-9a-f]{32})\.json$"), retained_generations),
                (BLOBS_DIRNAME, re.compile(r"^([0-9a-f]{64})\.blob$"), retained_blobs),
            )
            for dirname, pattern, retained in specs:
                directory = _owned_path(root, dirname)
                _require_directory(directory)
                with os.scandir(directory) as iterator:
                    entries = []
                    for index, entry in enumerate(iterator):
                        if index >= MAX_GC_ENTRIES:
                            return
                        entries.append(entry)
                for entry in entries:
                    match = pattern.fullmatch(entry.name)
                    if match is None:
                        return
                    info = entry.stat(follow_symlinks=False)
                    if _is_reparse(info) or not stat.S_ISREG(info.st_mode):
                        return
                    if match.group(1) not in retained:
                        planned.append(entry.path)
                        if len(planned) >= MAX_GC_DELETIONS:
                            break
                if len(planned) >= MAX_GC_DELETIONS:
                    break

            for path in planned:
                info = os.lstat(path)
                if _is_reparse(info) or not stat.S_ISREG(info.st_mode):
                    return
            changed_directories: set[str] = set()
            for path in planned:
                os.remove(path)
                changed_directories.add(os.path.dirname(path))
            for directory in changed_directories:
                _fsync_directory(directory)
    except Exception:
        # GC is postcommit and best-effort. No cleanup failure may turn a
        # durably published generation into a falsely reported snapshot error.
        return


def snapshot_cas_v2(
    ns: dict[str, Any],
    root: str,
    max_bytes: int,
    max_variable_bytes: int,
    prune_oversized: bool,
    always_skip: set[str],
    handle_type: type[Any],
    committed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize all eligible names and atomically publish one CAS generation."""
    total_start = time.monotonic_ns()
    metrics: dict[str, float | int | None] = {
        "serialization_wall_ms": None,
        "serialization_cpu_ms": None,
        "serialized_bytes": None,
        "write_ms": None,
        "written_bytes": 0,
        "total_wall_ms": None,
    }
    written = [0]
    try:
        payload, skipped, oversized, envelope_bytes, logical_bytes, serialization = _serialize_namespace(
            ns,
            max_bytes=max_bytes,
            max_variable_bytes=max_variable_bytes,
            prune_oversized=prune_oversized,
            always_skip=always_skip,
            handle_type=handle_type,
        )
        metrics.update(serialization)
        write_start = time.monotonic_ns()
        _initialize_for_write(root, written)
        root = _require_absolute_root(root)
        pointer = _read_pointer(root, required=False)
        if pointer is not None:
            # A generation promoted to `previous` must still be complete now.
            with open_cas_generation(
                root,
                "current",
                max_bytes=max_bytes,
                max_variable_bytes=max_variable_bytes,
            ):
                pass

        entries: list[dict[str, Any]] = []
        for name, blob in payload.items():
            digest = hashlib.sha256(blob).hexdigest()
            _write_blob(root, digest, blob, written)
            entries.append({"name": name, "sha256": digest, "size": len(blob)})

        saved = sorted(payload)
        pruned = sorted(name for name in oversized if name in ns) if prune_oversized else []
        generation_id = _new_generation_id(root)
        generation = {
            "format": CAS_FORMAT,
            "version": CAS_VERSION,
            "generation": generation_id,
            "entries": entries,
            "savedNames": saved,
            "skipped": skipped,
            "pruned": pruned,
            "logicalSerializedBytes": logical_bytes,
            "legacyEnvelopeBytes": envelope_bytes,
            "maxBytes": max_bytes,
            "maxVariableBytes": max_variable_bytes,
            "pythonVersion": os.sys.version.split()[0],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        generation_bytes = _encode_json(generation)
        generation_path = _owned_path(root, GENERATIONS_DIRNAME, f"{generation_id}.json")
        _require_owned_parent(root, generation_path)
        _atomic_write_bytes(generation_path, generation_bytes, written)
        generation_ref = _reference_for(generation_id, generation_bytes)
        # Read through the reference before publishing it.
        _read_generation(
            root,
            generation_ref,
            max_bytes=max_bytes,
            max_variable_bytes=max_variable_bytes,
        )

        next_pointer = {
            "format": CAS_FORMAT,
            "version": CAS_VERSION,
            "current": generation_ref,
            "previous": pointer["current"] if pointer is not None else None,
        }
        pointer_bytes = _encode_json(next_pointer)
        pointer_path = _owned_path(root, CURRENT_FILENAME)
        _require_owned_parent(root, pointer_path)

        parked: list[int] = []
        previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: parked.append(signum))
        try:
            _atomic_write_bytes(pointer_path, pointer_bytes, written)
            metrics["write_ms"] = _elapsed_ms(write_start, time.monotonic_ns())
            metrics["written_bytes"] = written[0]
            for name in pruned:
                ns.pop(name, None)
            result: dict[str, Any] = {
                "saved": saved,
                "skipped": skipped,
                "pruned": pruned,
                "bytes": envelope_bytes,
                "format": "cas-v2",
                "generation": generation_id,
                "logical_bytes": logical_bytes,
                "written_bytes": written[0],
                "backward_readable": False,
                "metrics": metrics,
            }
            if committed is not None:
                committed.append(result)
            _bounded_gc(
                root,
                next_pointer,
                max_bytes=max_bytes,
                max_variable_bytes=max_variable_bytes,
            )
            metrics["total_wall_ms"] = _elapsed_ms(total_start, time.monotonic_ns())
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        return result
    except Exception as error:
        metrics["written_bytes"] = written[0]
        metrics["total_wall_ms"] = _elapsed_ms(total_start, time.monotonic_ns())
        return {"error": f"CAS v2 snapshot failed: {_safe_str(error)}", "metrics": metrics}


def restore_cas_v2(
    ns: dict[str, Any],
    root: str,
    restore_skip: set[str],
    source: str = "current",
    committed: list[dict[str, Any]] | None = None,
    *,
    max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    max_variable_bytes: int = DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES,
) -> dict[str, Any]:
    try:
        import dill
    except Exception as error:
        return {"error": f"dill unavailable: {_safe_str(error)}"}

    try:
        staged: dict[str, Any] = {}
        failed: list[dict[str, str]] = []
        with open_cas_generation(
            root,
            source,
            max_bytes=max_bytes,
            max_variable_bytes=max_variable_bytes,
        ) as (generation, blobs):
            for entry in generation["entries"]:
                name = entry["name"]
                if name in restore_skip:
                    continue
                try:
                    staged[name] = dill.loads(blobs[name])
                except Exception as error:
                    failed.append({"name": name, "reason": f"{type(error).__name__}: {_safe_str(error)[:200]}"})

            prepared, backfill, revive_failed = prepare_restored_values(staged, ns, restore_skip)
            result: dict[str, Any] = {
                "restored": sorted(prepared),
                "failed": failed + revive_failed,
                "format": "cas-v2",
                "generation": generation["generation"],
            }
            if source == "previous":
                result["rolled_back"] = True
                result["unsaved_work_possible"] = True

            previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: None)
            try:
                for name, value in prepared.items():
                    ns[name] = value
                for name, value in backfill:
                    if name not in ns:
                        ns[name] = value
                if committed is not None:
                    committed.append(result)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        return result
    except SnapshotStoreError as error:
        return {"error": f"CAS v2 load failed: {_safe_str(error)}"}


def read_cas_payload(
    root: str,
    source: str = "current",
    *,
    max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    max_variable_bytes: int = DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Return validated per-name bytes for an explicit legacy export."""
    with open_cas_generation(
        root,
        source,
        max_bytes=max_bytes,
        max_variable_bytes=max_variable_bytes,
    ) as (generation, blobs):
        return dict(blobs), generation
