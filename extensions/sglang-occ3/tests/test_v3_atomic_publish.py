"""atomic_publish wrapper tests: injected syscall outcomes, platform honesty.

Every syscall-outcome test runs through the injection seam (ctypes.set_errno
fakes) and executes NO real syscall. Real-Linux behavior tests are marked
linux-only and skipped elsewhere: non-Linux platforms never pretend to prove
renameat2 semantics (a separate Linux-native smoke, owned by the sole DGX
executor, is required before deployment use).
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import atomic_publish as pub


def fake(errno_value=None, rc=None, consume=True):
    """A renameat2 fake: sets errno, returns rc; optionally performs the move."""
    def rename_fn(old_fd, old, new_fd, new, flags):
        old_path, new_path = os.fsdecode(old), os.fsdecode(new)
        if rc == 0:
            if consume:
                os.rename(old_path, new_path)  # faithful: consumes the name
            return 0
        ctypes.set_errno(errno_value)
        return rc
    return rename_fn


def test_non_linux_refuses_before_any_ctypes(monkeypatch):
    monkeypatch.setattr(pub, "_IS_LINUX", False)
    monkeypatch.setattr(pub, "_RESOLVED", None)
    with pytest.raises(pub.RenameNotSupported):
        pub.rename_noreplace("/old", "/new")
    # No CDLL work happened: _RESOLVED is still None.


# Certain classes: the kernel (or the platform gate) guarantees the final
# was untouched, so those assertions are kernel-backed. Uncertain classes
# (RenameUnknown) carry NO kernel guarantee - the fake performed no move, so
# the assertions bound themselves to the OBSERVED fixture (the fake's own
# behavior), never to a kernel claim about real renameat2.
CERTAIN = [
    (errno.EEXIST, pub.RenameWouldOverwrite),
    (errno.ENOTEMPTY, pub.RenameWouldOverwrite),
    (errno.ENOSYS, pub.RenameNotSupported),
    (errno.EINVAL, pub.RenameNotSupported),
    (errno.EOPNOTSUPP, pub.RenameNotSupported),
    (errno.EPERM, pub.RenamePolicyRefused),
]
UNCERTAIN = [
    (errno.EIO, pub.RenameUnknown),
    (errno.EACCES, pub.RenameUnknown),
    (errno.EXDEV, pub.RenameUnknown),
    (errno.ENOENT, pub.RenameUnknown),
]


@pytest.mark.parametrize("errno_value,exception", CERTAIN + UNCERTAIN)
def test_errno_classification_observation_bounded(tmp_path, errno_value, exception):
    staging = tmp_path / "staged"
    staging.write_bytes(b"x")
    final = tmp_path / "final"
    with pytest.raises(exception):
        pub.rename_noreplace(str(staging), str(final),
                             rename_fn=fake(errno_value=errno_value, rc=-1))
    # OBSERVED fixture behavior (the fake made no move): final absent, staging
    # present. For CERTAIN classes this is kernel-backed; for UNCERTAIN classes
    # it is an observation of this fake only - no kernel guarantee is implied.
    assert not final.exists()
    assert staging.exists()


@pytest.mark.parametrize("errno_value,exception", CERTAIN)
def test_errno_classification_certain_final_untouched(tmp_path, errno_value, exception):
    """Certain classes only: the kernel guarantees the final was untouched."""
    staging = tmp_path / "staged"
    staging.write_bytes(b"x")
    final = tmp_path / "final"
    final.write_bytes(b"existing")  # final ALREADY present: certain classes prove preservation
    with pytest.raises(exception):
        pub.rename_noreplace(str(staging), str(final),
                             rename_fn=fake(errno_value=errno_value, rc=-1))
    assert final.read_bytes() == b"existing"  # kernel-backed for certain classes


def test_success_consumes_staging(tmp_path):
    staging = tmp_path / "staged"
    staging.write_bytes(b"payload")
    final = tmp_path / "final"
    pub.rename_noreplace(str(staging), str(final), rename_fn=fake(rc=0))
    assert final.read_bytes() == b"payload"
    assert not staging.exists()


def test_success_without_consumption_is_caller_visible(tmp_path):
    # rc == 0 but the name persists (injection lying): the wrapper returns
    # success; detection of the unverified name is the CALLER's contract
    # (admission_pin classifies publish_uncertain). The wrapper itself never
    # deletes anything.
    staging = tmp_path / "staged"
    staging.write_bytes(b"payload")
    final = tmp_path / "final"
    pub.rename_noreplace(str(staging), str(final), rename_fn=fake(rc=0, consume=False))
    assert staging.exists()
    assert not final.exists()


def test_safe_identity_reads_only_presence(tmp_path):
    staging = tmp_path / "staged"
    staging.write_bytes(b"secret-free")
    final = tmp_path / "final"
    staging_present, final_present = pub.safe_identity(str(staging), str(final))
    assert staging_present is True and final_present is False


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="real renameat2 requires Linux; non-Linux platforms "
                           "must not pretend to prove kernel behavior")
class TestRealLinux:
    def test_real_renameat2_noreplace_semantics(self, tmp_path):
        staging = tmp_path / "staged"
        staging.write_bytes(b"payload")
        final = tmp_path / "final"
        pub.rename_noreplace(str(staging), str(final))
        assert final.read_bytes() == b"payload"
        assert not staging.exists()
        # nlink == 1 at rest: the validator's extra_hardlink_refused can
        # never fire from this publication.
        assert os.stat(final).st_nlink == 1
        # Would-overwrite on a second publication.
        other = tmp_path / "staged2"
        other.write_bytes(b"other")
        with pytest.raises(pub.RenameWouldOverwrite):
            pub.rename_noreplace(str(other), str(final))
        assert final.read_bytes() == b"payload"
        assert other.exists()
