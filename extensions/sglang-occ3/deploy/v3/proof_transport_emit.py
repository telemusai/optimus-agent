r"""Emit-only validator challenge transport (POSIX-only, read-only).

Prints the phase challenge frame for a validated journal context to
stdout: one ASCII header line ``N=<canonical unsigned decimal> H=<64 lowercase hex>\n`` followed by exactly ``4*((N+2)//3)`` unwrapped base64
characters of the captured challenge bytes, then EOF. Nothing else is
ever written to stdout. The helper performs ZERO intentional filesystem
mutation: it opens the journal root, ``journal.json`` and the fixed
challenge basename read-only and closes each descriptor exactly once on
every path.

Scope (emit-only slice, root GO agentmsg_c9aab840): byte retrieval of the
phase challenge only. No place verb exists in this module; no publication,
no staging, no launcher, no network, no DGX access, no journal updates.
The frame header is integrity framing, never authorization: the future
Windows caller holds its own independently authorized expected length and
digest (three-way trust, phase-1 component); this helper's self-header is
not that authority. Caller authentication (SSH/OS) happens outside this
helper; the helper validates context only.

Provenance:
- The strict JSON loader discipline (``_Reject`` control flow,
  ``_no_duplicates``, ``_reject_constant``, ``_json_depth_ok``,
  ``_strict_loads``) is inherited from ``deploy/v3/native_validator.py``:
  byte bound first, strict UTF-8, BOM refused, depth 32, duplicate-key and
  non-finite refusal.
- The POSIX reader below (per-component lstat walk, O_DIRECTORY|
  O_NOFOLLOW root capture, dir_fd-relative leaf opens with fd-fstat
  hardening) is NEW helper behavior, not an inherited guarantee: the
  existing ``_safe_read`` is an lstat-then-open TOCTOU reader and is
  deliberately NOT used here.

Field policy is the P-consumer contract resolved by root (agentmsg_
c9aab840 F3): exact CHALLENGE_SCHEMA, requested phase, journal-equal
attempt and candidate, non-empty strings, EXACT METHOD_ID, nonce (the
``challenge`` field) a non-empty string of at most 256 characters with no
charset restriction, ``expected_ranks`` an int (never bool) within the
u32 consumer range, and structural ``expires_at > created_at`` only - no
wall clock, no derived equality.

Disclosed, not eliminated: privileged swaps of ancestors before the root
directory is captured, and concurrent same-inode mutation of an already
open leaf. The held root fd pins the opened directory and leaves are
resolved relative to it, but a digest binds content, never placement.

Exit codes: 0 ok; 2 usage; 4 verified refusal (fixed value-free stderr
codes only); 8 internal (unexpected condition or stdout failure). A
nonzero exit means the consumer must discard the entire frame even if a
partial frame reached stdout.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import stat
import sys

# --- fixed vocabulary (provenance: suite + deploy/v3) ---
CHALLENGE_SCHEMA = "optimus.occ3.validator-phase-challenge.v1"
METHOD_ID = "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1"
CHALLENGE_CLASSES = ("challenge-before_restart", "challenge-after_restart")
JOURNAL_NAME = "journal.json"
BEFORE_STATE = "WAIT_NATIVE_INITIAL"
AFTER_STATE = "POST_RESTART"
# Watcher TERMINAL_SPEC (cutover_watch.py) plus the controller TERMINAL set
# (lifecycle.py): a deny-list superset under the per-class allow-lists.
TERMINAL_STATES = frozenset((
    "COMPLETE", "CANDIDATE_PERSISTED", "RESTORED", "FAILED", "ABORTED",
    "NOOP", "RESTORE_FAILED", "PREFLIGHT_FAILED",
))
MAX_JSON_DEPTH = 32
MAX_JOURNAL_BYTES = 4 * 1024 * 1024      # lifecycle.read_json default bound
MAX_CHALLENGE_BYTES = 4096               # native_validator MAX_CHALLENGE_BYTES
MAX_NONCE_CHARS = 256                    # suite read_phase_challenge bound
MAX_EXPECTED_RANKS = 4294967295          # u32 consumer range (suite struct)
MAX_JOURNAL_PATH_CHARS = 1024
HEADER_CAP_BYTES = 77

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 4
EXIT_INTERNAL = 8


# ---------------------------------------------------------------------------
# Inherited strict loader (native_validator.py discipline, verbatim).
# ---------------------------------------------------------------------------

class _Reject(Exception):
    """Internal control flow: one specific refusal code, never partial state."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise _Reject(code)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Reject("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise _Reject("non_finite_number")


def _json_depth_ok(value):
    """Bounded pre-parse depth scan (inherited, verbatim logic)."""
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
    depth = 0
    in_string = False
    escaped = False
    for char in value:
        code = ord(char)
        if in_string:
            if escaped:
                escaped = False
            elif code == 0x5C:  # backslash
                escaped = True
            elif code == 0x22:  # closing quote
                in_string = False
            continue
        if code == 0x22:  # opening quote
            in_string = True
        elif code in (0x7B, 0x5B):  # { or [
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return False
        elif code in (0x7D, 0x5D):  # } or ]
            depth -= 1
            if depth < 0:
                return False
    return not in_string and depth == 0


def _strict_loads(data, limit):
    """Strict UTF-8-only document loader: byte bound, decode, depth, parse."""
    if not isinstance(data, (bytes, bytearray)) or len(data) > limit:
        raise _Reject("file_too_large")
    try:
        text = bytes(data).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _Reject("malformed_json") from None
    if text.startswith("\ufeff"):
        # Explicit fail-closed BOM policy: no BOM is accepted.
        raise _Reject("malformed_json")
    if not _json_depth_ok(text):
        raise _Reject("malformed_json")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates,
                          parse_constant=_reject_constant)
    except _Reject:
        raise
    except (ValueError, RecursionError):
        raise _Reject("malformed_json") from None


# ---------------------------------------------------------------------------
# Usage grammar (pure: no lstat, no open, platform-independent).
# ---------------------------------------------------------------------------

def _journal_path_ok(path):
    """Absolute ASCII POSIX grammar: leading '/', printable ASCII only, no
    backslash, no empty/'.'/'..' components (so no trailing '/' and no
    '//' anywhere), not '/' itself, at most 1024 characters."""
    if not isinstance(path, str):
        return False
    if len(path) < 2 or len(path) > MAX_JOURNAL_PATH_CHARS:
        return False
    if not path.startswith("/"):
        return False
    for char in path:
        code = ord(char)
        if code < 0x20 or code > 0x7e:
            return False
    if "\\" in path:
        return False
    for component in path[1:].split("/"):
        if component in ("", ".", ".."):
            return False
    return True


def _parse_usage(argv):
    """Fixed two-flag vocabulary (phase-1 parity): --journal/--class pairs in
    any order, each at most once, both required, nothing else. Pure."""
    if len(argv) % 2 != 0:
        return None
    journal = None
    klass = None
    index = 0
    while index < len(argv):
        flag = argv[index]
        value = argv[index + 1]
        if flag == "--journal" and journal is None:
            journal = value
        elif flag == "--class" and klass is None:
            klass = value
        else:
            return None
        index += 2
    if journal is None or klass is None:
        return None
    if klass not in CHALLENGE_CLASSES:
        return None
    if not _journal_path_ok(journal):
        return None
    return journal, klass


# ---------------------------------------------------------------------------
# POSIX-only hardened reader (NEW helper behavior; see module docstring).
# ---------------------------------------------------------------------------

def _platform_ok():
    return os.name == "posix"


def _posix_caps_ok():
    """Required POSIX capabilities: O_DIRECTORY, O_NOFOLLOW, geteuid and
    dir_fd-aware os.open. Missing capabilities fail closed."""
    return (hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "geteuid")
            and os.open in os.supports_dir_fd)


def _close_fd(fd):
    """Release one leaf descriptor; best effort, exactly once per open."""
    try:
        os.close(fd)
    except OSError:
        pass


def _close_root(fd):
    """Release the held root directory descriptor; best effort, once."""
    try:
        os.close(fd)
    except OSError:
        pass


def _open_journal_root(path):
    """Hardened capture of the journal root directory (NEW reader behavior).

    Per-component lstat walk rejects existing symlink or non-directory
    ancestors; the full path is then opened O_RDONLY|O_DIRECTORY|O_NOFOLLOW
    and the held descriptor is fstat-checked: directory, owned by the
    effective uid, no group/other permission bits. The fd pins the opened
    directory; leaves are never re-resolved through the absolute path."""
    components = path.split("/")[1:]
    prefix = ""
    for component in components:
        prefix = prefix + "/" + component
        try:
            info = os.lstat(prefix)
        except OSError:
            raise _Reject("journal_root_missing")
        if stat.S_ISLNK(info.st_mode):
            raise _Reject("journal_root_symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise _Reject("journal_root_not_directory")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _Reject("journal_root_symlink")
        raise _Reject("journal_root_open_failed")
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise _Reject("journal_root_not_directory")
        if info.st_uid != os.geteuid():
            raise _Reject("journal_root_owner_refused")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise _Reject("journal_root_mode_refused")
    except _Reject:
        _close_root(fd)
        raise
    except OSError:
        _close_root(fd)
        raise _Reject("journal_root_open_failed")
    return fd


def _open_leaf(root_fd, basename, missing_code, open_code):
    """dir_fd-relative O_NOFOLLOW open of one fixed leaf name, with fd-fstat
    hardening: regular file, owned by the effective uid, no group/other
    permission bits, exactly one hard link (NEW reader behavior)."""
    try:
        fd = os.open(basename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _Reject("symlink_refused")
        if exc.errno == errno.ENOENT:
            raise _Reject(missing_code)
        raise _Reject(open_code)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _Reject("leaf_not_regular")
        if info.st_uid != os.geteuid():
            raise _Reject("leaf_owner_refused")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise _Reject("leaf_mode_refused")
        if info.st_nlink != 1:
            raise _Reject("leaf_hardlink_refused")
    except _Reject:
        _close_fd(fd)
        raise
    except OSError:
        _close_fd(fd)
        raise _Reject(open_code)
    return fd


def _read_bounded(fd, limit):
    """Bounded capture: read at most limit+1 bytes total across legitimate
    short reads; None means oversize or a failed read (the caller maps it
    to its fixed refusal code). No single-read assumption is made."""
    chunks = []
    total = 0
    while total <= limit:
        try:
            chunk = os.read(fd, limit + 1 - total)
        except OSError:
            return None
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
    return None


def _read_leaf(root_fd, basename, limit, missing_code, open_code,
               too_large_code):
    """Single-open bounded capture of one leaf relative to the HELD root fd.
    The captured buffer is the only bytes source for validation, header and
    base64 (same-open-fd discipline: no re-open, no second read)."""
    fd = _open_leaf(root_fd, basename, missing_code, open_code)
    if fd is None:
        raise _Reject(open_code)
    try:
        data = _read_bounded(fd, limit)
    finally:
        _close_fd(fd)
    if data is None:
        raise _Reject(too_large_code)
    return data


# ---------------------------------------------------------------------------
# Context gates (journal) and challenge field policy (P-consumer).
# ---------------------------------------------------------------------------

def _load_journal(root_fd):
    data = _read_leaf(root_fd, JOURNAL_NAME, MAX_JOURNAL_BYTES,
                      "journal_missing", "journal_open_failed",
                      "journal_too_large")
    document = _strict_loads(data, MAX_JOURNAL_BYTES)
    if not isinstance(document, dict):
        raise _Reject("journal_invalid")
    return document


def _check_state(klass, journal):
    """Deny-by-default context gates; returns the requested phase name."""
    phase = klass[len("challenge-"):]
    attempt_id = journal.get("attempt_id")
    candidate_id = journal.get("candidate_id")
    state_phase = journal.get("phase")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise _Reject("journal_invalid")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise _Reject("journal_invalid")
    if not isinstance(state_phase, str):
        raise _Reject("journal_invalid")
    if "persistence_decision" in journal:
        raise _Reject("persistence_decision_present")
    if state_phase in TERMINAL_STATES:
        raise _Reject("state_refused")
    if phase == "before_restart":
        if state_phase != BEFORE_STATE:
            raise _Reject("state_refused")
    else:
        if state_phase != AFTER_STATE:
            raise _Reject("state_refused")
        if journal.get("restart_done") is not True:
            raise _Reject("restart_not_completed")
        restart_count = journal.get("restart_count")
        if type(restart_count) is not int or isinstance(restart_count, bool) \
                or restart_count != 1:
            raise _Reject("restart_count_invalid")
    return phase


def _check_challenge(document, phase, journal):
    """P-consumer binding and field policy for the captured challenge."""
    if not isinstance(document, dict):
        raise _Reject("challenge_invalid")
    if document.get("schema") != CHALLENGE_SCHEMA:
        raise _Reject("challenge_invalid")
    if document.get("phase") != phase:
        raise _Reject("challenge_invalid")
    attempt_id = document.get("attempt_id")
    candidate_id = document.get("candidate_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise _Reject("challenge_invalid")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise _Reject("challenge_invalid")
    started = document.get("expected_candidate_started_at")
    if not isinstance(started, str) or not started:
        raise _Reject("challenge_invalid")
    fingerprint = document.get("model_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise _Reject("challenge_invalid")
    if document.get("method_id") != METHOD_ID:
        raise _Reject("challenge_invalid")
    nonce = document.get("challenge")
    if not isinstance(nonce, str) or not nonce or len(nonce) > MAX_NONCE_CHARS:
        raise _Reject("challenge_invalid")
    ranks = document.get("expected_ranks")
    if type(ranks) is not int or isinstance(ranks, bool) \
            or ranks < 0 or ranks > MAX_EXPECTED_RANKS:
        raise _Reject("challenge_invalid")
    created = document.get("created_at")
    expires = document.get("expires_at")
    if type(created) is not int or isinstance(created, bool):
        raise _Reject("challenge_invalid")
    if type(expires) is not int or isinstance(expires, bool):
        raise _Reject("challenge_invalid")
    if expires <= created:
        raise _Reject("challenge_invalid")
    if attempt_id != journal["attempt_id"] or candidate_id != journal["candidate_id"]:
        raise _Reject("challenge_binding_failed")


# ---------------------------------------------------------------------------
# Frame construction and output.
# ---------------------------------------------------------------------------

def _frame_bytes(data):
    """Canonical frame: header line plus unwrapped base64 of the captured
    bytes. The 77-byte header cap holds by construction for every bounded
    challenge (len(data) <= 4096 gives at most 74 header bytes)."""
    header = "N=" + str(len(data)) + " H=" + hashlib.sha256(data).hexdigest() + "\n"
    return header.encode("ascii") + base64.b64encode(data)


def _emit(frame):
    """Single fail-closed write: the reported count must be exactly
    len(frame) (type-is-int, never bool or None, never short or
    overlong) before the flush runs; an ordinary exception from the
    write/flush acquisition or operation (Exception, never
    BaseException) refuses identically. Partial stdout may remain and
    the consumer discards it on the nonzero exit - no
    all-or-nothing-output claim."""
    try:
        stream = sys.stdout.buffer
        written = stream.write(frame)
        if type(written) is not int or written != len(frame):
            return False
        stream.flush()
    except Exception:
        return False
    return True


def _refuse(exit_code, code):
    """Fixed value-free stderr refusal; never payload, digests or raw
    exception text. The code is a short fixed literal, so the 4096 bound
    holds by construction."""
    try:
        sys.stderr.write(code + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    return exit_code


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def _emit_frame(root_fd, klass):
    journal = _load_journal(root_fd)
    phase = _check_state(klass, journal)
    data = _read_leaf(root_fd, "validator-challenge-" + phase + ".json",
                      MAX_CHALLENGE_BYTES, "challenge_missing",
                      "challenge_open_failed", "challenge_too_large")
    document = _strict_loads(data, MAX_CHALLENGE_BYTES)
    _check_challenge(document, phase, journal)
    if not _emit(_frame_bytes(data)):
        return _refuse(EXIT_INTERNAL, "output_refused")
    return EXIT_OK


def _run(argv):
    if argv is None:
        argv = sys.argv[1:]
    parsed = _parse_usage(list(argv))
    if parsed is None:
        return _refuse(EXIT_USAGE, "usage_refused")
    journal_arg, klass = parsed
    if not _platform_ok():
        return _refuse(EXIT_REFUSED, "platform_refused")
    if not _posix_caps_ok():
        return _refuse(EXIT_REFUSED, "capability_refused")
    try:
        root_fd = _open_journal_root(journal_arg)
    except _Reject as exc:
        return _refuse(EXIT_REFUSED, exc.code)
    if root_fd is None:
        return _refuse(EXIT_REFUSED, "journal_root_open_failed")
    try:
        return _emit_frame(root_fd, klass)
    except _Reject as exc:
        return _refuse(EXIT_REFUSED, exc.code)
    finally:
        _close_root(root_fd)


def main(argv=None):
    """Emit-only entry point. Returns the process exit code; never raises
    and never prints raw exception text (fixed codes only)."""
    try:
        return _run(argv)
    except Exception:
        return _refuse(EXIT_INTERNAL, "internal_refused")


if __name__ == "__main__":
    sys.exit(main())
