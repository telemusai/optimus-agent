"""Native builtin subgraph snapshots, retaining dill's custom-object semantics."""

from __future__ import annotations

import copyreg
import datetime
import decimal
import io
import pickle
import time
from typing import Any

from .snapshot_safety import reject_file_handle


class SnapshotPathMetrics:
    """Content-free counters for one freshly serialized variable."""

    def __init__(self) -> None:
        self.values: dict[str, float | int] = {
            "serialization_native_values": 0,
            "serialization_dill_values": 0,
            "serialization_native_ms": 0.0,
            "serialization_dill_ms": 0.0,
            "serialization_native_probe_ms": 0.0,
            "serialization_native_probe_bytes": 0,
            "serialization_native_probe_attempts": 0,
            "serialization_native_probe_rejected": 0,
            "serialization_fragment_prepare_ms": 0.0,
            "serialization_fragment_write_ms": 0.0,
            "serialization_fragment_bytes": 0,
            "serialization_fragment_segments": 0,
            "serialization_buffer_reset_ms": 0.0,
            "serialization_blob_extract_ms": 0.0,
        }

    def add(self, key: str, value: float | int) -> None:
        self.values[key] += max(0, value)

    def elapsed(self, key: str, started: int) -> None:
        self.add(key, (time.monotonic_ns() - started) / 1_000_000)


class SnapshotSerializationMetrics:
    """Bounded per-save accounting; variable names never leave this object."""

    def __init__(self) -> None:
        self._durations_ns: dict[str, int] = {}
        self._path_totals: dict[str, float | int] | None = None
        self._probe_ms: dict[str, float] = {}

    def record(self, name: str, elapsed_ns: int, paths: SnapshotPathMetrics | None = None) -> None:
        self._durations_ns[name] = max(0, elapsed_ns)
        if paths is not None:
            if self._path_totals is None:
                self._path_totals = {key: 0 for key in paths.values}
            for key, value in paths.values.items():
                self._path_totals[key] += value
            probe_ms = paths.values["serialization_native_probe_ms"]
            if probe_ms:
                self._probe_ms[name] = self._probe_ms.get(name, 0.0) + probe_ms

    def summarize(self, saved: Any) -> dict[str, float | int]:
        saved_names = set(saved)
        saved_ns = sum(elapsed for name, elapsed in self._durations_ns.items() if name in saved_names)
        total_ns = sum(self._durations_ns.values())
        result = {
            "serialization_max_variable_ms": max(self._durations_ns.values(), default=0) / 1_000_000,
            "serialization_slow_variables": sum(elapsed >= 100_000_000 for elapsed in self._durations_ns.values()),
            "serialization_saved_ms": saved_ns / 1_000_000,
            "serialization_skipped_ms": (total_ns - saved_ns) / 1_000_000,
        }
        if self._path_totals is not None:
            result.update(self._path_totals)
            result["serialization_native_probe_saved_ms"] = sum(
                elapsed for name, elapsed in self._probe_ms.items() if name in saved_names
            )
            result["serialization_native_probe_skipped_ms"] = sum(
                elapsed for name, elapsed in self._probe_ms.items() if name not in saved_names
            )
        return result


class _RequiresDill(Exception):
    pass


_NATIVE_VALUE_TYPES = (
    datetime.date,
    datetime.datetime,
    datetime.time,
    datetime.timedelta,
    datetime.timezone,
    decimal.Decimal,
)


def _native_dispatch_compatible(dill: Any) -> bool:
    if dill.Pickler.persistent_id is not pickle._Pickler.persistent_id:
        return False
    if getattr(dill.Pickler, "reducer_override", None) is not None:
        return False
    expected = {
        type(None): pickle._Pickler.save_none,
        bool: pickle._Pickler.save_bool,
        int: pickle._Pickler.save_long,
        float: pickle._Pickler.save_float,
        bytes: pickle._Pickler.save_bytes,
        str: pickle._Pickler.save_str,
        list: pickle._Pickler.save_list,
        dict: dill._dill.save_module_dict,
        tuple: pickle._Pickler.save_tuple,
        set: pickle._Pickler.save_set,
        frozenset: pickle._Pickler.save_frozenset,
    }
    return all(dill.Pickler.dispatch.get(kind) is handler for kind, handler in expected.items())


class _PrimitivePickler(pickle.Pickler):
    def __init__(self, writer: Any, dill: Any, *, protocol: int | None = None) -> None:
        super().__init__(writer, protocol=dill.settings["protocol"] if protocol is None else protocol)
        self._dill = dill

    def reducer_override(self, value: Any) -> Any:
        reject_file_handle(value)
        # The C pickler bypasses this hook only for exact builtin primitives and
        # containers. Exact immutable standard-library values have the same
        # importable reducers under dill. Subclasses, custom tzinfo, closures and
        # overridden reducer registrations still fall back before invoking them.
        for trusted in _NATIVE_VALUE_TYPES:
            if type(value) is trusted or value is trusted:
                if self._dill.Pickler.dispatch.get(trusted) is not None or trusted in copyreg.dispatch_table:
                    raise _RequiresDill()
                return NotImplemented
        raise _RequiresDill()


class _ProbeLimitExceeded(Exception):
    pass


class _SubgraphPickler(_PrimitivePickler):
    def persistent_id(self, value: Any) -> None:
        # Unlike ordinary dictionaries, dill can serialize a module namespace
        # by reference. C pickle bypasses reducer_override for exact dicts.
        if type(value) is dict and "__name__" in value:
            raise _RequiresDill()
        return None


class _ProbeWriter:
    def __init__(self, buffer: io.BytesIO, limit: int) -> None:
        self.buffer = buffer
        self.limit = limit
        self.last_write_start = 0

    def write(self, data: bytes) -> int:
        if self.buffer.tell() + len(data) > self.limit:
            raise _ProbeLimitExceeded()
        self.last_write_start = self.buffer.tell()
        return self.buffer.write(data)


def _value_fragment(
    buffer: io.BytesIO, writer: _ProbeWriter, protocol: int
) -> tuple[memoryview | bytes, ...] | None:
    """Keep immutable backing bytes; only replace the final frame's size."""
    data = buffer.getvalue()
    if data[:2] != bytes((pickle.PROTO[0], protocol)) or not data.endswith(pickle.STOP):
        return None
    view = memoryview(data)
    if protocol >= 4:
        # Validate the final write boundary, never search user payload opcodes.
        start = max(2, writer.last_write_start)
        if data[start:start + 1] == pickle.FRAME:
            size = int.from_bytes(view[start + 1:start + 9], "little")
            if size != len(data) - start - 9 or size < 1:
                return None
            return (view[2:start + 1], (size - 1).to_bytes(8, "little"), view[start + 9:-1])
        if len(data) - start >= 4:
            return None
    return (view[2:-1],)


def _write_fragment(pickler: Any, writer: Any, fragment: tuple[memoryview | bytes, ...]) -> None:
    write_segments = getattr(writer, "write_segments", None)
    file_write = pickler._file_write
    if (
        write_segments is None
        or getattr(file_write, "__self__", None) is not writer
        or getattr(file_write, "__func__", None) is not getattr(type(writer), "write", None)
    ):
        # Unknown sinks or wrapped dill writes keep the original behavior.
        file_write(b"".join(fragment))
    else:
        write_segments(fragment)


def _dump_with_dill(
    dill: Any, value: Any, writer: Any, metrics: SnapshotPathMetrics | None = None
) -> None:
    """Keep dill's graph/reducer semantics, accelerating only builtin subgraphs."""
    pickler = dill.Pickler(writer, protocol=dill.settings["protocol"])
    original_persistent_id = pickler.persistent_id

    def persistent_id(obj: Any) -> Any:
        reject_file_handle(obj)
        return original_persistent_id(obj)

    pickler.persistent_id = persistent_id
    original_save = pickler.save
    # A custom dispatch table may change even primitive values inside a graph.
    can_accelerate = _native_dispatch_compatible(dill) and pickler.proto >= 3
    if not can_accelerate:
        try:
            pickler.dump(value)
        finally:
            del pickler.persistent_id
        return

    def save(obj: Any, save_persistent_id: bool = True) -> None:
        nonlocal can_accelerate
        if not can_accelerate:
            original_save(obj, save_persistent_id=save_persistent_id)
            return
        kind = type(obj)
        if (
            can_accelerate
            and kind in (list, dict, tuple)
            and len(obj) >= 128
            and id(obj) not in pickler.memo
            and (kind is not dict or "__name__" not in obj)
            and _native_dispatch_compatible(dill)
        ):
            buffer = io.BytesIO()
            probe_writer = _ProbeWriter(buffer, max(0, getattr(writer, "_limit", 16 << 20)))
            native = _SubgraphPickler(
                probe_writer, dill, protocol=pickler.proto
            )
            native.memo = pickler.memo
            probe_started = time.monotonic_ns()
            probe_ended = None
            if metrics is not None:
                metrics.add("serialization_native_probe_attempts", 1)
            try:
                native.dump(obj)
            except (_RequiresDill, pickle.PicklingError, _ProbeLimitExceeded):
                if metrics is not None:
                    metrics.add("serialization_native_probe_rejected", 1)
                # Do not repeatedly probe a mixed graph at every nested level.
                # Revert this value's remaining traversal to unwrapped dill.
                can_accelerate = False
                pickler.save = original_save
            else:
                probe_ended = time.monotonic_ns()
                prepare_started = probe_ended
                fragment = _value_fragment(buffer, probe_writer, pickler.proto)
                if metrics is not None:
                    metrics.elapsed("serialization_fragment_prepare_ms", prepare_started)
                if fragment is not None:
                    # Never nest FRAME opcodes. Publish the native memo only
                    # after all segments succeed, including cap preflight.
                    if pickler.proto >= 4:
                        pickler.framer.end_framing()
                    write_started = time.monotonic_ns()
                    try:
                        _write_fragment(pickler, writer, fragment)
                    finally:
                        if metrics is not None:
                            metrics.elapsed("serialization_fragment_write_ms", write_started)
                    if pickler.proto >= 4:
                        pickler.framer.start_framing()
                    pickler.memo = native.memo.copy()
                    if metrics is not None:
                        metrics.add("serialization_fragment_bytes", sum(len(part) for part in fragment))
                        metrics.add("serialization_fragment_segments", len(fragment))
                    return
                if metrics is not None:
                    metrics.add("serialization_native_probe_rejected", 1)
                can_accelerate = False
                pickler.save = original_save
            finally:
                if metrics is not None:
                    metrics.add("serialization_native_probe_ms", (
                        (probe_ended if probe_ended is not None else time.monotonic_ns()) - probe_started
                    ) / 1_000_000)
                    metrics.add("serialization_native_probe_bytes", buffer.tell())
        original_save(obj, save_persistent_id=save_persistent_id)

    pickler.save = save
    try:
        pickler.dump(value)
    finally:
        # The wrapper closes over this pickler and its memo. Do not retain the
        # entire snapshot graph until cyclic GC happens to run.
        del pickler.save
        del pickler.persistent_id


def dump_snapshot_value(
    dill: Any, value: Any, buffer: Any, writer: Any, metrics: SnapshotPathMetrics | None = None
) -> None:
    """Serialize afresh on every save, including in-place mutations and cycles."""
    started = time.monotonic_ns()
    try:
        if not _native_dispatch_compatible(dill):
            raise _RequiresDill()
        _PrimitivePickler(writer, dill).dump(value)
    except (_RequiresDill, pickle.PicklingError):
        if metrics is not None:
            metrics.elapsed("serialization_native_ms", started)
            metrics.add("serialization_dill_values", 1)
        reset_started = time.monotonic_ns()
        buffer.seek(0)
        buffer.truncate()
        writer.written = 0
        if metrics is not None:
            metrics.elapsed("serialization_buffer_reset_ms", reset_started)
        dill_started = time.monotonic_ns()
        try:
            _dump_with_dill(dill, value, writer, metrics)
        finally:
            if metrics is not None:
                metrics.elapsed("serialization_dill_ms", dill_started)
    except BaseException:
        if metrics is not None:
            metrics.elapsed("serialization_native_ms", started)
            metrics.add("serialization_native_values", 1)
        raise
    else:
        if metrics is not None:
            metrics.elapsed("serialization_native_ms", started)
            metrics.add("serialization_native_values", 1)
