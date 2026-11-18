"""Atomic no-replace publication for Linux (renameat2 RENAME_NOREPLACE).

Kernel-atomic exclusion: the existence check and the move are ONE operation,
so a final that appears concurrently can never be overwritten, and the final
name only ever transitions from nonexistent to complete content.

Certainty classes (tightening addendum; authoritative):
- rc == 0                     rename happened. Success "written" only when the
                              parent-directory fsync also succeeds; any fsync
                              failure is the fixed NONZERO refusal
                              publish_uncertain (final stays, never deleted).
- EEXIST / ENOTEMPTY          final existed at the rename; caller runs the
                              strict same-binding idempotency/conflict check.
- ENOSYS / EINVAL / EOPNOTSUPP publish_unsupported (syscall/kernel/filesystem
                              does not provide the operation).
- EPERM                       publish_refused_policy: an honest policy-denial
                              class (seccomp/apparmor/LSM); NOT evidence the
                              syscall is unsupported.
- any other errno             publish_unknown: the outcome is uncertain. The
                              caller may report bounded safe identity facts
                              (staging_present / final_present) and nothing
                              else; no retry, no deletion, no recovery.

Reviewer RC3: a successful rename CONSUMES the staging name. The publisher
never unlinks the staging path after a successful rename - an unlink there
could delete a name recreated by another actor. Cleanup unlink happens ONLY
before the rename (own inode identity match) and in the non-success paths.
If the staging name unexpectedly persists after rc == 0, that is classified
publish_uncertain (an unverified name), never deleted.

Non-Linux platforms fail closed before any ctypes work. No os.rename,
os.replace, os.link or shell fallback exists anywhere in this module.
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys

__all__ = ["RenameNotSupported", "RenameWouldOverwrite", "RenamePolicyRefused",
           "RenameUnknown", "RENAME_NOREPLACE", "AT_FDCWD", "rename_noreplace",
           "safe_identity"]

# Kernel uapi constants (linux/fs.h RENAME_NOREPLACE; linux/fcntl.h AT_FDCWD):
# stable ABI values, never architecture-guessed syscall numbers.
RENAME_NOREPLACE = 1
AT_FDCWD = -100

_IS_LINUX = sys.platform.startswith("linux")


class RenameNotSupported(Exception):
    """ENOSYS/EINVAL/EOPNOTSUPP or non-Linux: the operation is unavailable."""


class RenameWouldOverwrite(Exception):
    """EEXIST/ENOTEMPTY: the final name already exists."""


class RenamePolicyRefused(Exception):
    """EPERM: policy denial; unsupported-vs-refused is NOT decidable here."""


class RenameUnknown(Exception):
    """Any other errno: the outcome is uncertain; no deletion, no retry."""


_RESOLVED = None


def _resolve():
    """Typed libc renameat2 symbol, or the fixed unsupported refusal.

    Resolution is by real symbol only (glibc >= 2.28). A missing symbol,
    a non-Linux platform, or a resolution error is unsupported - never a
    fallback to another primitive.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if not _IS_LINUX:
        raise RenameNotSupported("linux_only")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = getattr(libc, "renameat2", None)
    except (OSError, ValueError):
        raise RenameNotSupported("libc_unavailable") from None
    if fn is None:
        raise RenameNotSupported("renameat2_absent")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p,
                   ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _RESOLVED = fn
    return fn


def rename_noreplace(staging, final, *, rename_fn=None):
    """Atomically move a fully-written staging file onto `final`.

    `rename_fn` is a test-only injection seam for the resolved libc call: a
    fake sets the errno with ctypes.set_errno(...) and returns nonzero (or
    returns 0 for success), exactly like the real symbol. Injection bypasses
    the platform gate BY DESIGN so offline tests never execute a syscall; a
    REAL call (no injection) always resolves through _resolve(), which
    enforces the Linux gate.
    """
    fn = rename_fn if rename_fn is not None else _resolve()
    rc = fn(AT_FDCWD, os.fsencode(staging), AT_FDCWD, os.fsencode(final),
            RENAME_NOREPLACE)
    if rc == 0:
        return
    code = ctypes.get_errno()
    if code in (errno.EEXIST, errno.ENOTEMPTY):
        raise RenameWouldOverwrite()
    if code == errno.EPERM:
        raise RenamePolicyRefused()
    if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise RenameNotSupported("errno")
    raise RenameUnknown(code)


def safe_identity(staging, final):
    """Bounded safe identity facts for an UNKNOWN outcome. Reads no content.

    Returns exactly (staging_present, final_present) as booleans; lstat only.
    This is the only inspection permitted after an uncertain rename, and its
    result implies no deletion, retry or recovery action.
    """
    def _present(path):
        try:
            os.lstat(path)
            return True
        except OSError:
            return False
    return _present(staging), _present(final)
