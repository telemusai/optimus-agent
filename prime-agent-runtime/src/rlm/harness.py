"""Persistent harness-state helpers for Prime Agent's RLM kernel.

The state model is intentionally small: it records prompt notes, memory,
skills, subagent specs, and refinement events in the session-local harness
store by default; pass ``global_=True`` for the cross-session global store.
Execution still belongs to Prime Agent's TypeScript host and the existing
``rlm.run`` recursion bridge.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Iterator, Literal

if os.name == "nt":
    import msvcrt
else:
    import fcntl

HarnessKind = Literal["prompt", "memory", "skill", "subagent"]
HarnessScope = Literal["local", "global"]

_DEFAULT_FILE_NAME = "harness_state.json"
_DEFAULT_HARNESS_DIR_NAME = "harness"
# Recovery copies of unparsable state live beside the state file, named by content
# hash so a repeat quarantine of identical bytes is idempotent.
_CORRUPT_STATE_PREFIX = "harness_state.corrupt-"
# Transient Windows rename retries, mirroring WIN32_RENAME_ATTEMPTS in
# crates/pi-coding-agent/src/utils/atomic_file.rs.
_WIN32_RENAME_ATTEMPTS = 5
_STATE_LOCK_TIMEOUT_SECONDS = 1.0
_KINDS: tuple[HarnessKind, ...] = ("prompt", "memory", "skill", "subagent")
_state_cache: dict[tuple[Path, HarnessScope], "HarnessState"] = {}


@contextmanager
def _state_write_lock(state_path: Path) -> Iterator[None]:
    """Coordinate Python/Rust writers using a crash-released OS file lock.

    Keep the stable lock file: unlinking it would let a new writer lock a
    different inode while another writer still holds the original one. The
    Rust writer locks this same file; on Windows our first-byte lock conflicts
    with its whole-file LockFileEx lock.
    """
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b") as lock_file:
        deadline = time.monotonic() + _STATE_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError("Harness state is being saved by another writer; retry the operation.") from error
                time.sleep(0.01)
        try:
            yield
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_refinement_id() -> str:
    """Mint a refinement id in the canonical `refine_<17-digit timestamp>` form.

    The Rust and TypeScript refiners mint ids this way (see
    `generate_refinement_id` in
    crates/pi-coding-agent/src/core/refinement/refinement.rs and
    packages/coding-agent/src/core/refinement/refinement.ts), so a kernel-side
    refinement is addressable by the same id format as a host-side one. Rows
    written by older kernels keep their `refine_NNNN` ids; they are read as-is.
    """
    digits = "".join(character for character in _now() if character.isdigit())
    return f"refine_{digits[:17]}"


def _is_transient_replace_error(error: OSError) -> bool:
    """Windows reports the destination being held open as a sharing violation.

    The accepted codes mirror `is_transient_windows_rename_error` in
    crates/pi-coding-agent/src/utils/atomic_file.rs (EPERM/EACCES/EBUSY and
    ERROR_ACCESS_DENIED/ERROR_SHARING_VIOLATION/ERROR_LOCK_VIOLATION).
    """
    transient_errno = {errno.EPERM, errno.EACCES, errno.EBUSY}
    if error.errno in transient_errno:
        return True
    return os.name == "nt" and getattr(error, "winerror", None) in {5, 32, 33}


def _replace_with_retry(source: Path, destination: Path) -> None:
    """`os.replace` with the bounded Windows rename retry the Rust writer uses.

    A transient sharing violation must not abort a completed save; a persistent
    failure still raises, so a lost refinement is never silent.
    """
    attempt = 1
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if not _is_transient_replace_error(error) or attempt >= _WIN32_RENAME_ATTEMPTS:
                raise
            time.sleep(0.01 * attempt)
            attempt += 1


def _slug(raw: str, fallback: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return (normalized or fallback)[:80]


def _agent_dir() -> Path:
    raw = (
        os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
        or os.environ.get("PI_CODING_AGENT_DIR")
        or str(Path.home() / ".prime" / "agent")
    )
    return Path(raw).expanduser().resolve()


def _resolve_global_flag(global_: bool = False, extra: dict[str, Any] | None = None) -> bool:
    extra = dict(extra or {})
    if "global" in extra:
        value = extra.pop("global")
        if not isinstance(value, bool):
            raise TypeError(f"global must be a bool, got {type(value).__name__}")
        global_ = value
    if extra:
        unexpected = next(iter(extra))
        raise TypeError(f"unexpected keyword argument {unexpected!r}")
    return bool(global_)


def _strip_scope_prefix(id: str | None, global_: bool) -> tuple[str | None, bool]:
    # overview() displays entries as [local:id]/[global:id]; accept those ids
    # verbatim. A global: prefix routes to the global store unless the caller
    # already forced a scope via global_.
    if isinstance(id, str):
        scope, sep, rest = id.partition(":")
        if sep and rest and scope in ("local", "global"):
            return rest, global_ or scope == "global"
    return id, global_


def _env_dir(name: str) -> str | None:
    # Set-but-empty env values must behave as unset; a bare "" would skip the
    # session-dir fallback and land local writes in the global agent-dir default.
    value = (os.environ.get(name) or "").strip()
    return value or None


def _state_file(state_dir: str | Path | None = None, *, global_: bool = False) -> Path:
    root: str | Path | None = state_dir
    if root is None:
        root = _env_dir("RLM_GLOBAL_HARNESS_STATE_DIR") if global_ else _env_dir("RLM_HARNESS_STATE_DIR")
    if root is None and not global_ and (session_dir := _env_dir("RLM_SESSION_DIR")):
        root = Path(session_dir) / _DEFAULT_HARNESS_DIR_NAME
    if root is None and not global_:
        raise RuntimeError(
            "Local harness state requires RLM_HARNESS_STATE_DIR or RLM_SESSION_DIR. "
            "Use get_harness_state(global_=True) for global state."
        )
    if root:
        return Path(root).expanduser().resolve() / _DEFAULT_FILE_NAME
    return _agent_dir() / _DEFAULT_HARNESS_DIR_NAME / _DEFAULT_FILE_NAME


@dataclass
class HarnessEntry:
    """A reusable prompt, memory, skill, or subagent record."""

    id: str
    kind: HarnessKind
    title: str
    content: str
    path: str = "general"
    scope: HarnessScope = "local"
    reference: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = 1


@dataclass
class RefinementEvent:
    """A recorded online harness-refinement pass."""

    id: str
    trigger: str
    changes: list[str]
    evidence: str = ""
    outcome: str = ""
    created_at: str = field(default_factory=_now)


_ENTRY_FIELDS = {field.name for field in fields(HarnessEntry)}
_REFINEMENT_FIELDS = {field.name for field in fields(RefinementEvent)}


def _validate_python_skill_reference(reference: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise ValueError("skill entries require a Python reference")
    normalized = dict(reference)
    if normalized.get("type") != "python":
        raise ValueError("skill reference.type must be 'python'")
    if not any(isinstance(normalized.get(key), str) and normalized[key] for key in ("import", "python_import")):
        raise ValueError("skill reference requires a Python import")
    if not any(isinstance(normalized.get(key), str) and normalized[key] for key in ("callable", "call_pattern")):
        raise ValueError("skill reference requires a callable or call_pattern")
    return normalized


def _require_text(entry_name: str, field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Harness entry {entry_name!r}: {field_name} must be a non-empty string")


def _entry_name(id: Any, title: Any) -> str:
    return id if isinstance(id, str) and id else title if isinstance(title, str) else "<unnamed>"


def _validate_entry_fields(kind: str, id: str, title: Any, content: Any, *, path: Any,
                           reference: Any, arguments: Any, metadata: Any, source: Any,
                           existing: HarnessEntry | None) -> None:
    for field_name, value in (("id", id), ("title", title), ("content", content), ("source", source)):
        _require_text(id, field_name, value)
    if path is not None:
        _require_text(id, "path", path)
    for field_name, value in (("reference", reference), ("arguments", arguments), ("metadata", metadata)):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"Harness entry {id!r}: {field_name} must be a dict")
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Harness entry {id!r}: {field_name} must contain JSON values") from error
    if kind == "skill" and (reference is not None or existing is None):
        _validate_python_skill_reference(reference)


class HarnessState:
    """CRUD store for reset-free harness refinement state."""

    def __init__(
        self,
        file_path: str | Path | None = None,
        *,
        in_memory: bool = False,
        scope: HarnessScope = "local",
        local_write_error: str | None = None,
    ):
        # in_memory mode never resolves or touches a path. It is the safe fallback when
        # path resolution itself fails, so constructing it cannot re-raise that error.
        if in_memory:
            self.file_path: Path | None = None
        else:
            self.file_path = (
                Path(file_path).expanduser().resolve()
                if file_path
                else _state_file(global_=(scope == "global"))
            )
        self.scope: HarnessScope = scope
        # When set, local mutations raise instead of vanishing into a volatile
        # store; reads and global_=True delegation keep working.
        self._local_write_error = local_write_error
        self.entries: dict[HarnessKind, dict[str, HarnessEntry]] = {kind: {} for kind in _KINDS}
        self.refinements: list[RefinementEvent] = []
        # "missing" (no file yet), "loaded", "corrupt" (readable, unparsable), or
        # "unreadable" (access error). Only the last two make a write unsafe.
        self._load_status: str = "missing"
        self._load_reason: str | None = None
        self._global_target_state_dir: Path | None = None
        # mtime of the file as of the last load/save, used to detect out-of-process
        # writes (e.g. the host `/refine` command) and avoid clobbering them.
        self._loaded_mtime: int | None = None
        # The bytes we actually read, not the mtime of a later filesystem probe.
        self._loaded_generation: str | None = None
        self._needs_reload = False
        self.load()

    def _ensure_local_writable(self) -> None:
        if self._local_write_error is not None:
            raise RuntimeError(self._local_write_error)

    def _disk_mtime(self) -> int | None:
        if self.file_path is None:
            return None
        try:
            return self.file_path.stat().st_mtime_ns
        except OSError:
            return None

    def _sync_from_disk(self) -> None:
        """Reload if another process rewrote the state file since we last touched it.

        The kernel keeps a long-lived ``HarnessState`` in memory while the host
        ``/refine`` command rewrites the same file from a separate process. Without
        this guard the next in-kernel ``save()`` would overwrite host edits with a
        stale snapshot. We re-read whenever the on-disk mtime no longer matches the
        value recorded at our last load/save.
        """
        try:
            generation = self._disk_generation()
        except OSError:
            self.load()
            return
        if (
            self._needs_reload
            or self._load_status == "unreadable"
            or generation != self._loaded_generation
            or self._disk_mtime() != self._loaded_mtime
        ):
            self.load()

    def _disk_generation(self) -> str | None:
        if self.file_path is None:
            return None
        try:
            return hashlib.sha256(self.file_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None

    def _read_state_payload(self) -> tuple[dict[str, Any], str, str | None]:
        """Read and parse the state payload, classifying it for CF-04 safety.

        Returns ``(data, status, reason)``. An access error is "unreadable"
        (bytes unknown, save must fail closed); readable but unparsable bytes
        (including invalid UTF-8) are "corrupt" (recovery copy before rewrite);
        a valid JSON non-object is also "corrupt".
        """
        try:
            raw = self.file_path.read_bytes()
        except OSError as error:
            self._loaded_generation = None
            return {}, "unreadable", str(error)
        self._loaded_generation = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            # Invalid UTF-8 is readable corruption, not an access error: the
            # bytes survive quarantine and the next save may proceed (review
            # defect D1, review-glm; parity with the Rust classifier).
            return {}, "corrupt", str(error)
        try:
            data = json.loads(text)
        except ValueError as error:
            return {}, "corrupt", str(error)
        if not isinstance(data, dict):
            # json.loads returns non-dict types for valid JSON like `null`, `[]`,
            # or a bare string; those are corrupt state, not an empty store.
            return (
                {},
                "corrupt",
                f"state file is a JSON {type(data).__name__}, not an object",
            )
        return data, "loaded", None

    def load(self) -> "HarnessState":
        self._needs_reload = False
        if self.file_path is None:
            self._loaded_mtime = None
            self._loaded_generation = None
            self._load_status = "missing"
            self._load_reason = None
            return self
        # Classify through an explicit metadata probe: Path.exists() reports
        # false for any stat error, including an ACL denial, which would
        # masquerade an inaccessible store as a new one (review defect D4,
        # review-glm).
        try:
            os.lstat(self.file_path)
        except FileNotFoundError:
            self._loaded_mtime = None
            self._loaded_generation = None
            self._load_status = "missing"
            self._load_reason = None
            self.entries = {kind: {} for kind in _KINDS}
            self.refinements = []
            return self
        except OSError as error:
            # An access error is not "no state": the bytes are unknown, so the
            # next save must not replace them. load() still returns an empty view.
            self._loaded_mtime = None
            self._loaded_generation = None
            self._load_status = "unreadable"
            self._load_reason = str(error)
            data = {}
        else:
            data, status, reason = self._read_state_payload()
            self._load_status = status
            self._load_reason = reason
        mtime = self._disk_mtime()

        entries: dict[HarnessKind, dict[str, HarnessEntry]] = {kind: {} for kind in _KINDS}
        raw_entries = data.get("entries", {})
        if isinstance(raw_entries, dict):
            for kind in _KINDS:
                raw_kind_entries = raw_entries.get(kind, {})
                if not isinstance(raw_kind_entries, dict):
                    continue
                for entry_id, raw_entry in raw_kind_entries.items():
                    if isinstance(raw_entry, dict):
                        entry_data = {key: value for key, value in raw_entry.items() if key in _ENTRY_FIELDS}
                        entry_data["id"] = str(entry_id)
                        entry_data["kind"] = kind
                        if not isinstance(entry_data.get("title"), str) or not isinstance(
                            entry_data.get("content"), str
                        ):
                            continue
                        if not isinstance(entry_data.get("path"), str):
                            entry_data["path"] = "general"
                        if entry_data.get("scope") not in ("local", "global"):
                            entry_data["scope"] = self.scope
                        if not isinstance(entry_data.get("source"), str):
                            entry_data["source"] = "agent"
                        version = entry_data.get("version", 1)
                        if isinstance(version, str):
                            try:
                                version = int(version)
                            except ValueError:
                                version = 1
                        if not isinstance(version, int):
                            version = 1
                        entry_data["version"] = version
                        if not isinstance(entry_data.get("reference"), dict):
                            entry_data["reference"] = {}
                        if not isinstance(entry_data.get("arguments"), dict):
                            entry_data["arguments"] = {}
                        if not isinstance(entry_data.get("metadata"), dict):
                            entry_data["metadata"] = {}
                        entries[kind][str(entry_id)] = HarnessEntry(**entry_data)
        self.entries = entries

        self.refinements = []
        raw_refinements = data.get("refinements", [])
        if isinstance(raw_refinements, list):
            for raw_event in raw_refinements:
                if isinstance(raw_event, dict):
                    event_data = {key: value for key, value in raw_event.items() if key in _REFINEMENT_FIELDS}
                    if not isinstance(event_data.get("id"), str) or not isinstance(
                        event_data.get("trigger"), str
                    ):
                        continue
                    changes = event_data.get("changes")
                    if isinstance(changes, str):
                        event_data["changes"] = [changes]
                    elif isinstance(changes, list):
                        event_data["changes"] = [str(change) for change in changes]
                    elif not isinstance(changes, list):
                        continue
                    self.refinements.append(RefinementEvent(**event_data))
        self._loaded_mtime = mtime
        return self

    def _global_target(self, global_: bool, extra: dict[str, Any] | None = None) -> "HarnessState | None":
        if not _resolve_global_flag(global_, extra):
            return None
        target = get_harness_state(state_dir=self._global_target_state_dir, global_=True)
        if self.file_path is not None and target.file_path == self.file_path and target.scope == self.scope:
            return None
        return target

    def _classify_on_disk(self) -> tuple[str, str | None, str | None]:
        """Classify the current on-disk bytes at save time.

        Mirrors the Rust save-time re-classification: a writer that corrupts the
        file while preserving its mtime (backup/restore tools, racing writers
        landing inside the sync window) must not escape the quarantine decision
        by hiding behind the cached load-time status (review defect D3,
        review-glm).
        """
        if self.file_path is None:
            return ("missing", None, None)
        try:
            os.lstat(self.file_path)
        except FileNotFoundError:
            return ("missing", None, None)
        except OSError as error:
            # A stat denial is an access failure, not an empty store (review
            # defect D4, review-glm).
            return ("unreadable", str(error), None)
        try:
            raw = self.file_path.read_bytes()
        except OSError as error:
            return ("unreadable", str(error), None)
        generation = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            return ("corrupt", str(error), generation)
        if not isinstance(data, dict):
            return ("corrupt", f"state file is a JSON {type(data).__name__}, not an object", generation)
        return ("loaded", None, generation)

    def save(self) -> "HarnessState":
        if self.file_path is None:
            # in_memory fallback: nothing to persist.
            return self
        try:
            with _state_write_lock(self.file_path):
                return self._save_locked()
        except Exception:
            # The mutation may already be in our cached view, but is not durable.
            # Subsequent reads/mutations must reload instead of presenting it as saved.
            self._needs_reload = True
            raise

    def _save_locked(self) -> "HarnessState":
        # Fail closed: when the state file could not be read, replacing it would
        # destroy bytes nobody has seen. A corrupt (readable) file is copied aside
        # first; an unreadable file blocks the write entirely. The classification
        # comes from the bytes on disk right now, not from the cached load-time
        # status.
        status, reason, generation = self._classify_on_disk()
        if self._needs_reload or self._load_status == "unreadable" or status == "unreadable":
            raise RuntimeError(
                f"Harness state at {self.file_path} could not be read "
                f"({reason or self._load_reason or 'reload required'}); refusing to overwrite unreadable state."
            )
        if generation != self._loaded_generation:
            raise RuntimeError(
                "Harness state changed since it was loaded; reload and retry the operation. "
                "The newer state was not overwritten."
            )
        if status == "corrupt":
            self._quarantine_corrupt_state()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": 1,
            "entries": {
                kind: {entry_id: asdict(entry) for entry_id, entry in records.items()}
                for kind, records in self.entries.items()
            },
            "refinements": [asdict(event) for event in self.refinements],
        }
        # Atomic replace on the real file: aliases survive, readers never see a torn file.
        target_path = Path(os.path.realpath(self.file_path))
        temp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            existing_mode = stat.S_IMODE(os.stat(target_path).st_mode)
        except FileNotFoundError:
            existing_mode = None
        mode = existing_mode if existing_mode is not None else 0o600
        serialized = json.dumps(data, indent=2, ensure_ascii=False)
        try:
            # Create no looser than the destination; retain the umask for new files.
            descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                # Durability parity with the Rust harness writer and the CAS store:
                # a completed save must survive power loss.
                f.flush()
                os.fsync(f.fileno())
            if existing_mode is not None:
                os.chmod(temp_path, existing_mode)
            _replace_with_retry(temp_path, target_path)
        finally:
            temp_path.unlink(missing_ok=True)
        # The rewrite reconciled the file, so the status is healthy again.
        self._load_status = "loaded"
        self._load_reason = None
        self._loaded_mtime = self._disk_mtime()
        # Hash the exact bytes written; a restrictive umask may make reopening
        # this newly created file impossible even though its save succeeded.
        written = serialized.replace("\n", os.linesep).encode("utf-8")
        self._loaded_generation = hashlib.sha256(written).hexdigest()
        self._needs_reload = False
        return self

    def _quarantine_corrupt_state(self) -> Path | None:
        """Copy unparsable state bytes aside before the rewrite.

        The copy is content-addressed, so quarantining identical bytes twice is
        idempotent. A file that cannot be copied aside raises, so the caller never
        silently destroys content it could not preserve.
        """
        if self.file_path is None:
            return None
        try:
            raw = self.file_path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Harness state at {self.file_path} could not be read ({error}); "
                "refusing to overwrite unreadable state."
            ) from error
        digest = hashlib.sha256(raw).hexdigest()[:16]
        backup_path = self.file_path.with_name(f"{_CORRUPT_STATE_PREFIX}{digest}.json")
        # Atomic, unconditional rewrite through temp + fsync + the bounded
        # rename retry: a torn partial copy from a killed process must never be
        # pinned as the recovery copy by an exists() skip (review defect D2,
        # review-glm). The content-addressed name keeps repeat quarantine
        # idempotent in the success case.
        temp_path = backup_path.with_name(f"{backup_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            _replace_with_retry(temp_path, backup_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return backup_path

    def upsert(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        id, global_ = _strip_scope_prefix(id, global_)
        if target := self._global_target(global_, kwargs):
            return target.upsert(
                kind,
                title,
                content,
                id=id,
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )
        self._ensure_local_writable()
        self._sync_from_disk()
        return self._upsert(
            kind,
            title,
            content,
            id=id,
            path=path,
            reference=reference,
            arguments=arguments,
            metadata=metadata,
            source=source,
        )

    def _upsert(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
    ) -> HarnessEntry:
        # Caller is responsible for syncing from disk first. create()/update() sync
        # once and then call this directly so their existence check and the write are
        # not separated by a second reload (which could turn create-or-fail into a
        # silent update).
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {_KINDS}")

        _require_text(_entry_name(id, title), "title", title)
        if id is not None:
            _require_text(_entry_name(id, title), "id", id)
        entry_id = id or _slug(title, kind)
        existing = self.entries[kind].get(entry_id)
        _validate_entry_fields(kind, entry_id, title, content, path=path, reference=reference,
                               arguments=arguments, metadata=metadata, source=source, existing=existing)
        if existing:
            existing.title = title
            existing.content = content
            # Preserve path/reference/arguments/metadata when the caller omits them
            # (None) so updating only an entry's title or content does not reset its
            # grouping path or wipe a skill's reference/argument contract. An explicit
            # value (including {}) still overwrites.
            if path is not None:
                existing.path = path
            if reference is not None:
                existing.reference = dict(reference)
            if arguments is not None:
                existing.arguments = dict(arguments)
            if metadata is not None:
                existing.metadata = dict(metadata)
            existing.source = source
            existing.updated_at = _now()
            existing.version += 1
            entry = existing
        else:
            entry = HarnessEntry(
                id=entry_id,
                kind=kind,
                title=title,
                content=content,
                path=path if path is not None else "general",
                scope=self.scope,
                reference=dict(reference or {}),
                arguments=dict(arguments or {}),
                metadata=dict(metadata or {}),
                source=source,
            )
            self.entries[kind][entry_id] = entry
        self.save()
        return entry

    def get(self, kind: HarnessKind, id: str, *, global_: bool = False, **kwargs: Any) -> HarnessEntry | None:
        id, global_ = _strip_scope_prefix(id, global_)
        if target := self._global_target(global_, kwargs):
            return target.get(kind, id)
        self._sync_from_disk()
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {_KINDS}")
        return self.entries[kind].get(id)

    def delete(self, kind: HarnessKind, id: str, *, global_: bool = False, **kwargs: Any) -> bool:
        id, global_ = _strip_scope_prefix(id, global_)
        if target := self._global_target(global_, kwargs):
            return target.delete(kind, id)
        self._ensure_local_writable()
        self._sync_from_disk()
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {_KINDS}")
        _require_text(_entry_name(id, None), "id", id)
        if id not in self.entries[kind]:
            return False
        del self.entries[kind][id]
        self.save()
        return True

    def list(self, kind: HarnessKind | None = None, *, global_: bool = False, **kwargs: Any) -> list[HarnessEntry]:
        if target := self._global_target(global_, kwargs):
            return target.list(kind)
        self._sync_from_disk()
        kinds = [kind] if kind else list(_KINDS)
        records: list[HarnessEntry] = []
        for current_kind in kinds:
            if current_kind not in self.entries:
                raise ValueError(f"unknown harness kind {current_kind!r}; expected one of {_KINDS}")
            records.extend(self.entries[current_kind].values())
        return sorted(records, key=lambda entry: (entry.kind, entry.path, entry.title, entry.id))

    def create(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        id, global_ = _strip_scope_prefix(id, global_)
        if target := self._global_target(global_, kwargs):
            return target.create(
                kind,
                title,
                content,
                id=id,
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )
        self._ensure_local_writable()
        self._sync_from_disk()
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {_KINDS}")
        _require_text(_entry_name(id, title), "title", title)
        if id is not None:
            _require_text(_entry_name(id, title), "id", id)
        entry_id = id or _slug(title, kind)
        if entry_id in self.entries[kind]:
            raise ValueError(f"{kind} entry {entry_id!r} already exists")
        return self._upsert(
            kind,
            title,
            content,
            id=entry_id,
            path=path,
            reference=reference,
            arguments=arguments,
            metadata=metadata,
            source=source,
        )

    def update(
        self,
        kind: HarnessKind,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        id, global_ = _strip_scope_prefix(id, global_)
        if target := self._global_target(global_, kwargs):
            return target.update(
                kind,
                id,
                title,
                content,
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )
        self._ensure_local_writable()
        self._sync_from_disk()
        if kind not in self.entries:
            raise ValueError(f"unknown harness kind {kind!r}; expected one of {_KINDS}")
        _require_text(_entry_name(id, title), "id", id)
        if id not in self.entries[kind]:
            raise ValueError(f"{kind} entry {id!r} does not exist")
        return self._upsert(
            kind,
            title,
            content,
            id=id,
            path=path,
            reference=reference,
            arguments=arguments,
            metadata=metadata,
            source=source,
        )

    def create_memory(
        self,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.create("memory", title, content, id=id, path=path, metadata=metadata, global_=global_, **kwargs)

    def update_memory(
        self,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.update("memory", id, title, content, path=path, metadata=metadata, global_=global_, **kwargs)

    def delete_memory(self, id: str, *, global_: bool = False, **kwargs: Any) -> bool:
        return self.delete("memory", id, global_=global_, **kwargs)

    def create_prompt_note(
        self,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "policy",
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.create("prompt", title, content, id=id, path=path, metadata=metadata, global_=global_, **kwargs)

    def update_prompt_note(
        self,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.update("prompt", id, title, content, path=path, metadata=metadata, global_=global_, **kwargs)

    def delete_prompt_note(self, id: str, *, global_: bool = False, **kwargs: Any) -> bool:
        return self.delete("prompt", id, global_=global_, **kwargs)

    def create_skill(
        self,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.create(
            "skill",
            title,
            content,
            id=id,
            path=path,
            reference=_validate_python_skill_reference(reference),
            arguments=arguments,
            metadata=metadata,
            global_=global_,
            **kwargs,
        )

    def update_skill(
        self,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        # Only validate a reference when one is supplied; omitting it preserves the
        # existing reference (see _upsert) rather than forcing every title/content-only
        # update to re-send the full Python reference.
        validated_reference = _validate_python_skill_reference(reference) if reference is not None else None
        return self.update(
            "skill",
            id,
            title,
            content,
            path=path,
            reference=validated_reference,
            arguments=arguments,
            metadata=metadata,
            global_=global_,
            **kwargs,
        )

    def delete_skill(self, id: str, *, global_: bool = False, **kwargs: Any) -> bool:
        return self.delete("skill", id, global_=global_, **kwargs)

    def create_subagent(
        self,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.create("subagent", title, content, id=id, path=path, metadata=metadata, global_=global_, **kwargs)

    def update_subagent(
        self,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> HarnessEntry:
        return self.update("subagent", id, title, content, path=path, metadata=metadata, global_=global_, **kwargs)

    def delete_subagent(self, id: str, *, global_: bool = False, **kwargs: Any) -> bool:
        return self.delete("subagent", id, global_=global_, **kwargs)

    def record_refinement(
        self,
        trigger: str,
        changes: list[str] | str,
        *,
        evidence: str = "",
        outcome: str = "",
        id: str | None = None,
        global_: bool = False,
        **kwargs: Any,
    ) -> RefinementEvent:
        if target := self._global_target(global_, kwargs):
            return target.record_refinement(trigger, changes, evidence=evidence, outcome=outcome, id=id)
        self._ensure_local_writable()
        self._sync_from_disk()
        _require_text("refinement", "trigger", trigger)
        if id is not None:
            _require_text("refinement", "id", id)
        if not isinstance(changes, (str, list)):
            raise ValueError("Refinement changes must be a string or a list of strings")
        normalized_changes = [changes] if isinstance(changes, str) else list(changes)
        for change in normalized_changes:
            _require_text("refinement", "change", change)
        if not isinstance(evidence, str) or not isinstance(outcome, str):
            raise ValueError("Refinement evidence and outcome must be strings")
        event_id = id or generate_refinement_id()
        event = RefinementEvent(
            id=event_id,
            trigger=trigger,
            changes=normalized_changes,
            evidence=evidence,
            outcome=outcome,
        )
        self.refinements.append(event)
        self.save()
        return event

    def plan_refinement(
        self,
        observation: str,
        *,
        failing_component: str = "",
        next_step: str = "",
    ) -> list[str]:
        target = f" for {failing_component}" if failing_component else ""
        plan = [
            f"Diagnose the repeated failure or opportunity{target}: {observation}",
            "Update the smallest useful prompt note, memory item, skill, or subagent spec.",
            "Run the next action with the changed harness state, then record the outcome.",
        ]
        if next_step:
            plan.append(f"Immediate validation step: {next_step}")
        return plan

    def overview(self, *, max_entries_per_kind: int = 20, global_: bool = False, **kwargs: Any) -> str:
        if target := self._global_target(global_, kwargs):
            return target.overview(max_entries_per_kind=max_entries_per_kind)
        self._sync_from_disk()
        lines = [
            f"Harness state ({self.scope}): {self.file_path}",
            "Call contract: installed Python skills use await <skill_import>(...) or a matching shell CLI; "
            "harness skill entries are Python REPL skills and must include a Python reference plus arguments. "
            "Spawn a subagent spec by composing a concise task prompt and calling "
            "handle = await rlm('sub-task'); admission returns immediately with rlm_child_id, name, session_dir, "
            "and model, never the child's answer. Results arrive only through explicit agent_message replies or "
            "files; children reply with await agent_message.send(message, receiver_role='parent'). Use "
            "await rlm.list_subagents() to recover direct child handles and await agent_message.send(..., "
            "receiver_role='child', receiver_name=handle.name) for follow-ups.",
        ]
        for kind in _KINDS:
            records = self.list(kind)[:max_entries_per_kind]
            lines.append(f"{kind}: {len(self.entries[kind])}")
            for entry in records:
                summary = entry.content.strip().replace("\n", " ")
                if len(summary) > 120:
                    summary = f"{summary[:117]}..."
                argument_summary = ""
                if entry.kind == "skill" and entry.arguments:
                    argument_text = json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True)
                    if len(argument_text) > 120:
                        argument_text = f"{argument_text[:117]}..."
                    argument_summary = f" args={argument_text}"
                reference_summary = ""
                if entry.kind == "skill" and entry.reference:
                    reference_text = json.dumps(entry.reference, ensure_ascii=False, sort_keys=True)
                    if len(reference_text) > 120:
                        reference_text = f"{reference_text[:117]}..."
                    reference_summary = f" ref={reference_text}"
                lines.append(
                    f"  - [{entry.scope}:{entry.id}] {entry.title} ({entry.path}, v{entry.version})"
                    f"{reference_summary}{argument_summary}: {summary}"
                )
            overflow = len(self.entries[kind]) - len(records)
            if overflow > 0:
                lines.append(f"  - +{overflow} more")
        if self.refinements:
            lines.append(f"refinements: {len(self.refinements)}")
            for event in self.refinements[-5:]:
                lines.append(f"  - [{event.id}] {event.trigger}: {', '.join(event.changes)}")
        else:
            lines.append("refinements: 0")
        return "\n".join(lines)

    def snapshot(self, *, global_: bool = False, **kwargs: Any) -> dict[str, Any]:
        if target := self._global_target(global_, kwargs):
            return target.snapshot()
        self._sync_from_disk()
        return {
            "file_path": str(self.file_path),
            "scope": self.scope,
            "entries": {
                kind: {entry_id: asdict(entry) for entry_id, entry in records.items()}
                for kind, records in self.entries.items()
            },
            "refinements": [asdict(event) for event in self.refinements],
        }


def get_harness_state(
    state_dir: str | Path | None = None, *, global_: bool = False, **kwargs: Any
) -> HarnessState:
    """Return the cached local harness state, or global when requested."""
    global_ = _resolve_global_flag(global_, kwargs)
    file_path = _state_file(state_dir, global_=global_)
    scope: HarnessScope = "global" if global_ else "local"
    cache_key = (file_path, scope)
    state = _state_cache.get(cache_key)
    if state is None:
        state = HarnessState(file_path, scope=scope)
        # Recorded at construction only: an instance created from env defaults must
        # keep targeting RLM_GLOBAL_HARNESS_STATE_DIR even when a later explicit
        # state_dir call aliases the same local file. An explicit dir that merely
        # aliases the env resolution must not sandbox later global_=True writes
        # either, so pin only when the explicit dir actually diverges.
        if state_dir is not None:
            try:
                env_file: Path | None = _state_file(global_=global_)
            except RuntimeError:
                env_file = None
            if file_path != env_file:
                state._global_target_state_dir = Path(state_dir).expanduser().resolve()
        _state_cache[cache_key] = state
    return state


__all__ = [
    "generate_refinement_id",
    "HarnessEntry",
    "HarnessKind",
    "HarnessScope",
    "HarnessState",
    "RefinementEvent",
    "get_harness_state",
]
