"""Keep snapshot file handles from reopening paths during kernel recovery."""

from __future__ import annotations

import functools
import io
import pickle
from typing import Any


class SnapshotFileHandleError(pickle.PicklingError):
    pass


def reject_file_handle(value: Any) -> None:
    # In-memory buffers carry data, whereas file/stream handles carry live I/O.
    if isinstance(value, io.IOBase) and type(value) not in (io.BytesIO, io.StringIO):
        raise SnapshotFileHandleError("file/stream handles cannot be snapshotted")


@functools.lru_cache(maxsize=4)
def _guarded_unpickler(base: type[Any]) -> type[Any]:
    class SnapshotUnpickler(base):
        def find_class(self, module: str, name: str) -> Any:
            # Reject before REDUCE calls dill's constructor, including old blobs.
            if module in ("dill._dill", "dill.dill") and name == "_create_filehandle":
                raise SnapshotFileHandleError("snapshot contains a file handle; refusing to reopen its path")
            return super().find_class(module, name)

    return SnapshotUnpickler


def load_snapshot_value(dill: Any, source: Any) -> Any:
    stream = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source
    return _guarded_unpickler(dill.Unpickler)(stream).load()
