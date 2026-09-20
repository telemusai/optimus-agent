"""Test-owned subprocess for the Windows cross-runtime checked-save proof."""
import os
import sys
import time
from pathlib import Path

repository, root, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert os.name == "nt", "Windows lock interoperability only"
assert sys.version_info[:3] == (3, 11, 15), "fresh candidate Python must match live 3.11.15"
sys.path.insert(0, str(repository / "prime-agent-runtime" / "src"))
from rlm.harness import HarnessEntry, HarnessState, _state_write_lock, __file__ as harness_file

assert Path(harness_file).resolve() == (
    repository / "prime-agent-runtime" / "src" / "rlm" / "harness.py"
).resolve(), "test must call the candidate source checked-save implementation"
state_path = root / "harness_state.json"
lock_path = root / "harness_state.json.lock"
original = state_path.read_bytes()
original_mtime = state_path.stat().st_mtime_ns
lock_bytes = b"synthetic stable harness lock\r\n"
lock_identity = lock_path.stat()


def wait_marker(name):
    deadline = time.monotonic() + 15
    while not (root / name).exists():
        assert time.monotonic() < deadline, f"handshake timed out: {name}"
        time.sleep(0.01)


def add_marker(state, name):
    state.entries["memory"][name] = HarnessEntry(
        id=name, kind="memory", title=name, content="synthetic interoperability marker",
        path="test-only", scope="local",
    )


if mode == "python-contender":
    state = HarnessState(state_path)
    stale = HarnessState(state_path)
    add_marker(state, "python-unsaved")
    started = time.monotonic()
    try:
        state.save()
    except RuntimeError as error:
        assert "another writer" in str(error), str(error)
    else:
        raise AssertionError("Python checked save ignored the Rust lock")
    assert 0.8 <= time.monotonic() - started < 5, "contention must be bounded"
    assert state_path.read_bytes() == original
    assert os.path.samestat(lock_identity, lock_path.stat())
    (root / "python-refused").write_bytes(b"refused")
    wait_marker("rust-exited-and-saved")
    newer = state_path.read_bytes()
    assert newer != original
    assert state_path.stat().st_mtime_ns == original_mtime
    add_marker(stale, "python-stale")
    try:
        stale.save()
    except RuntimeError as error:
        assert "changed since it was loaded" in str(error), str(error)
    else:
        raise AssertionError("Python accepted stale bytes with unchanged mtime")
    assert state_path.read_bytes() == newer
    state.load()
    assert "rust-peer" in state.entries["memory"]
    assert "python-unsaved" not in state.entries["memory"]
    add_marker(state, "python-success")
    state.save()
    loaded = HarnessState(state_path)
    assert {"seed", "rust-peer", "python-success"} <= loaded.entries["memory"].keys()
elif mode == "python-holder":
    assert lock_path.read_bytes() == lock_bytes
    with _state_write_lock(state_path):
        (root / "python-ready").write_bytes(b"ready")
        wait_marker("release-python")
        assert state_path.read_bytes() == original
    state = HarnessState(state_path)
    add_marker(state, "python-peer")
    state.save()
    stat = state_path.stat()
    os.utime(state_path, ns=(stat.st_atime_ns, original_mtime))
    assert state_path.stat().st_mtime_ns == original_mtime
else:
    raise AssertionError(f"unknown test mode: {mode}")
assert os.path.samestat(lock_identity, lock_path.stat())
assert lock_path.read_bytes() == lock_bytes
print(f"{mode}: passed; Python {sys.version.split()[0]}; candidate source harness")
