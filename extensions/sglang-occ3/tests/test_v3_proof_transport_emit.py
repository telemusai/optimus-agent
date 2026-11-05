"""Offline tests for the emit-only proof-transport helper.

Seam discipline (the sanctioned seam): PRIVATE helper monkeypatching in
TEST CODE ONLY - the module under test has no public parameter and no CLI
platform bypass. The POSIX-only helpers (_platform_ok, _posix_caps_ok,
_open_journal_root, _close_root, _open_leaf) are faked; journal.json and
the challenge bytes are still read through REAL descriptors (real
_read_bounded, real _close_fd) against real temporary files, so bounded
reads, close-once accounting, strict loads, context gates, frame
construction, output and exit codes all run real code.

NOT-RUN (honest): Linux-only mechanics - the per-component lstat walk,
O_NOFOLLOW/dir_fd opens, fd-fstat owner/mode/nlink hardening - are only
SIMULATED here (Windows-logic/seam evidence); no POSIX host executed them.
In-place same-inode mutation during the read window is a disclosed
residual and no test asserts immunity to it. No network, no DGX, no
producer invocation, no journal mutation anywhere in this file.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import proof_transport_emit as pte

ATTEMPT = "attempt-" + "a" * 32
CANDIDATE = "candidate-" + "b" * 32
NONCE = "n" * 64
GOOD_PATH = "/seam/journal-root"

# Every stderr code the module can emit (kept in lockstep with the module
# source; test_module_codes_match_vocabulary cross-checks the union).
FIXED_CODES = {
    "usage_refused", "platform_refused", "capability_refused",
    "internal_refused", "output_refused",
    "journal_root_missing", "journal_root_symlink",
    "journal_root_not_directory", "journal_root_open_failed",
    "journal_root_owner_refused", "journal_root_mode_refused",
    "journal_missing", "journal_open_failed", "journal_too_large",
    "journal_invalid", "state_refused", "persistence_decision_present",
    "restart_not_completed", "restart_count_invalid",
    "challenge_missing", "challenge_open_failed", "challenge_too_large",
    "challenge_invalid", "challenge_binding_failed",
    "file_too_large", "malformed_json", "duplicate_json_key",
    "non_finite_number", "symlink_refused", "leaf_not_regular",
    "leaf_owner_refused", "leaf_mode_refused", "leaf_hardlink_refused",
}


def journal_doc(**extra):
    doc = {"attempt_id": ATTEMPT, "candidate_id": CANDIDATE,
           "phase": "WAIT_NATIVE_INITIAL"}
    doc.update(extra)
    return doc


def after_journal_doc(**extra):
    doc = {"attempt_id": ATTEMPT, "candidate_id": CANDIDATE,
           "phase": "POST_RESTART", "restart_done": True,
           "restart_count": 1}
    doc.update(extra)
    return doc


def challenge_doc(phase="before_restart", **extra):
    doc = {
        "schema": pte.CHALLENGE_SCHEMA,
        "phase": phase,
        "attempt_id": ATTEMPT,
        "candidate_id": CANDIDATE,
        "expected_candidate_started_at": "2026-11-05T00:00:00Z",
        "model_fingerprint": "fp-" + "c" * 40,
        "method_id": pte.METHOD_ID,
        "expected_ranks": 8,
        "challenge": NONCE,
        "created_at": 1000,
        "expires_at": 2000,
    }
    doc.update(extra)
    return doc


def jdump(doc):
    return json.dumps(doc).encode("utf-8")


def challenge_name(klass):
    return "validator-challenge-" + klass[len("challenge-"):] + ".json"


class FailingBuffer(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.fail = False

    def write(self, data):
        if self.fail:
            raise OSError("simulated output failure")
        return super().write(data)


def _arm_short(data):
    """Stores ONLY the reported prefix (partial stdout is real)."""
    return data[:len(data) - 1], len(data) - 1


def _arm_none(data):
    return data, None


def _arm_bool(data):
    return data, True


def _arm_negative(data):
    return data, -1


def _arm_overlong(data):
    return data, len(data) + 1


class CountingBuffer(io.BytesIO):
    """Single-write double: records every write/flush call, stores what
    the arm chooses, reports the arm's chosen count."""

    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        self.write_calls = []
        self.flush_calls = 0

    def write(self, data):
        self.write_calls.append(len(data))
        stored, reported = self.transform(data)
        if stored is not None:
            super().write(stored)
        return reported

    def flush(self):
        self.flush_calls += 1
        super().flush()


class PrefixThenRaiseBuffer(io.BytesIO):
    """Single write stores a prefix, then raises (partial write)."""

    def __init__(self):
        super().__init__()
        self.write_calls = []
        self.flush_calls = 0

    def write(self, data):
        self.write_calls.append(len(data))
        super().write(data[:10])
        raise OSError("simulated write failure after prefix")

    def flush(self):
        self.flush_calls += 1
        super().flush()


class FlushRaisesBuffer(io.BytesIO):
    """Full-count single write, then the flush raises."""

    def __init__(self):
        super().__init__()
        self.write_calls = []
        self.flush_calls = 0

    def write(self, data):
        self.write_calls.append(len(data))
        return super().write(data)

    def flush(self):
        self.flush_calls += 1
        raise OSError("simulated flush failure")


class RuntimeErrorBuffer(io.BytesIO):
    """Ordinary non-OSError write exception: proves the except Exception
    routing (RuntimeError) inside the write operation refuses to
    output_refused, not internal_refused."""

    def __init__(self):
        super().__init__()
        self.write_calls = []
        self.flush_calls = 0

    def write(self, data):
        self.write_calls.append(len(data))
        super().write(data[:10])
        raise RuntimeError("ordinary simulated failure")

    def flush(self):
        self.flush_calls += 1
        super().flush()


class FakeStdout(object):
    def __init__(self):
        self.buffer = FailingBuffer()


class EmitSeam(object):
    """Fakes ONLY the POSIX-only private helpers; leaves run real code.

    The fake _open_leaf mirrors the real contract: a missing file raises
    _Reject(missing_code) exactly like the real ENOENT path.
    """

    def __init__(self, monkeypatch, tmp_path):
        self.tmp = Path(tmp_path)
        self.root_opens = []
        self.root_closes = []
        self.leaf_opens = []
        self.leaf_fd_closes = []
        self.fake_root = "fake-root-fd"
        monkeypatch.setattr(pte, "_platform_ok", lambda: True)
        monkeypatch.setattr(pte, "_posix_caps_ok", lambda: True)
        monkeypatch.setattr(pte, "_open_journal_root", self._open_root)
        monkeypatch.setattr(pte, "_close_root", self._close_root)
        monkeypatch.setattr(pte, "_open_leaf", self._open_leaf)
        real_close = pte._close_fd

        def counting_close(fd):
            self.leaf_fd_closes.append(fd)
            real_close(fd)

        monkeypatch.setattr(pte, "_close_fd", counting_close)

    def _open_root(self, path):
        self.root_opens.append(path)
        return self.fake_root

    def _close_root(self, fd):
        self.root_closes.append(fd)

    def _open_leaf(self, root_fd, basename, missing_code, open_code):
        self.leaf_opens.append(basename)
        target = self.tmp / basename
        if not target.exists():
            raise pte._Reject(missing_code)
        return os.open(str(target), os.O_RDONLY | getattr(os, "O_BINARY", 0))

    def write(self, basename, data):
        (self.tmp / basename).write_bytes(data)


def run_emit(seam, monkeypatch, journal_bytes, challenge_bytes, klass,
             stdout_fail=False, argv=None, write=True):
    """Write real fixture files (unless write=False), run main, return
    (code, stdout, stderr)."""
    if write:
        if journal_bytes is not None:
            seam.write("journal.json", journal_bytes)
        if challenge_bytes is not None:
            seam.write(challenge_name(klass), challenge_bytes)
    out = FakeStdout()
    out.buffer.fail = stdout_fail
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    if argv is None:
        argv = ["--journal", GOOD_PATH, "--class", klass]
    code = pte.main(list(argv))
    return code, out.buffer.getvalue(), err.getvalue()


def decode_frame(frame):
    """Independent full-grammar reference decoder (test-side only)."""
    assert isinstance(frame, bytes) and frame
    newline = frame.find(b"\n")
    assert newline != -1
    header = frame[:newline + 1]
    payload = frame[newline + 1:]
    match = re.match(rb"^N=([0-9]+) H=([0-9a-f]{64})\n$", header)
    assert match is not None, header
    digits, digest = match.group(1), match.group(2)
    length = int(digits)
    assert digits == str(length).encode("ascii")
    assert len(header) <= pte.HEADER_CAP_BYTES
    assert len(payload) == 4 * ((length + 2) // 3)
    data = base64.b64decode(payload, validate=True)
    assert len(data) == length
    assert hashlib.sha256(data).hexdigest() == digest.decode("ascii")
    return data


def snapshot_tree(root):
    """Names/contents/permissions snapshot; never timestamps (reads may
    legitimately update atime - no such promise is made or tested)."""
    root = Path(root)
    state = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            state[rel] = ("symlink", os.readlink(str(path)))
        elif path.is_dir():
            state[rel] = ("dir", oct(info.st_mode))
        else:
            state[rel] = ("file", oct(info.st_mode), path.read_bytes())
    return state


def _cd(**over):
    doc = challenge_doc()
    doc.update(over)
    return jdump(doc)


# ---------------------------------------------------------------------------
# 16 offline tests (synchronized contract inventory).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("klass,journal", [
    ("challenge-before_restart", "before"),
    ("challenge-after_restart", "after"),
])
def test_happy_path_both_phases_full_frame_grammar(
        tmp_path, monkeypatch, klass, journal):
    if journal == "before":
        doc = journal_doc()
    else:
        doc = after_journal_doc()
    phase = klass[len("challenge-"):]
    chal = jdump(challenge_doc(phase=phase))
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(doc), chal, klass)
    assert code == 0
    assert err == ""
    assert decode_frame(out) == chal
    assert seam.root_opens == [GOOD_PATH]
    assert seam.root_closes == [seam.fake_root]
    assert seam.leaf_opens == ["journal.json", challenge_name(klass)]
    assert len(seam.leaf_fd_closes) == 2


def test_stdout_is_exactly_the_frame(tmp_path, monkeypatch):
    seam = EmitSeam(monkeypatch, tmp_path)
    chal = jdump(challenge_doc())
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()), chal,
                              "challenge-before_restart")
    assert code == 0 and err == ""
    expected = ("N=%d H=%s\n" % (len(chal),
                                  hashlib.sha256(chal).hexdigest())
                ).encode("ascii") + base64.b64encode(chal)
    assert out == expected
    # flag order is free (phase-1 parity): swapped argv still emits the
    # exact same frame
    code2, out2, err2 = run_emit(
        seam, monkeypatch, jdump(journal_doc()), chal,
        "challenge-before_restart",
        argv=["--class", "challenge-before_restart", "--journal", GOOD_PATH])
    assert code2 == 0 and out2 == expected and err2 == ""


@pytest.mark.parametrize("field,value", [
    ("attempt_id", "attempt-" + "z" * 32),
    ("candidate_id", "candidate-" + "z" * 32),
])
def test_attempt_candidate_mismatch_refuses(
        tmp_path, monkeypatch, field, value):
    seam = EmitSeam(monkeypatch, tmp_path)
    chal = jdump(challenge_doc(**{field: value}))
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()), chal,
                              "challenge-before_restart")
    assert code == 4
    assert out == b""
    assert err == "challenge_binding_failed\n"


@pytest.mark.parametrize("state", sorted(pte.TERMINAL_STATES))
def test_all_terminal_states_refuse(tmp_path, monkeypatch, state):
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc(phase=state)),
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == "state_refused\n"


@pytest.mark.parametrize("state", [
    "WAIT_NATIVE_FINAL", "RESTART_REQUESTED", "", "wait_native_initial",
])
def test_unknown_phase_string_refuses(tmp_path, monkeypatch, state):
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc(phase=state)),
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == "state_refused\n"


def test_persistence_decision_present_refuses(tmp_path, monkeypatch):
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(
        seam, monkeypatch,
        jdump(journal_doc(persistence_decision="persist")),
        jdump(challenge_doc()), "challenge-before_restart")
    assert code == 4 and out == b""
    assert err == "persistence_decision_present\n"


@pytest.mark.parametrize("extra,remove,expected", [
    ({}, ("restart_done", "restart_count"), "restart_not_completed"),
    ({"restart_done": False}, (), "restart_not_completed"),
    ({"restart_done": True}, ("restart_count",), "restart_count_invalid"),
    ({"restart_done": True, "restart_count": 0}, (), "restart_count_invalid"),
    ({"restart_done": True, "restart_count": 2}, (), "restart_count_invalid"),
    ({"restart_done": True, "restart_count": True}, (), "restart_count_invalid"),
    ({"restart_done": True, "restart_count": "1"}, (), "restart_count_invalid"),
    ({"restart_done": "yes", "restart_count": 1}, (), "restart_not_completed"),
    ({"phase": "WAIT_NATIVE_INITIAL", "restart_done": True,
      "restart_count": 1}, (), "state_refused"),
])
def test_after_class_restart_gates(tmp_path, monkeypatch, extra, remove,
                                    expected):
    seam = EmitSeam(monkeypatch, tmp_path)
    journal = after_journal_doc()
    journal.update(extra)
    # dict.update cannot remove keys, so the two intended missing-key
    # arms omit them explicitly (round21 postmortem: the module was
    # correct; the fixtures were not)
    for key in remove:
        journal.pop(key)
    # post-construction preconditions: every requested removal is
    # truly absent; the restart_count-only arm keeps restart_done True
    for key in remove:
        assert key not in journal
    if remove == ("restart_count",):
        assert journal.get("restart_done") is True
        assert "restart_count" not in journal
    code, out, err = run_emit(seam, monkeypatch, jdump(journal),
                              jdump(challenge_doc(phase="after_restart")),
                              "challenge-after_restart")
    assert code == 4 and out == b"" and err == expected + "\n"


def test_missing_challenge_and_journal_refuse(tmp_path, monkeypatch):
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()), None,
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == "challenge_missing\n"
    other = tmp_path / "no-journal"
    other.mkdir()
    seam2 = EmitSeam(monkeypatch, other)
    code, out, err = run_emit(seam2, monkeypatch, None,
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == "journal_missing\n"


@pytest.mark.parametrize("challenge_bytes,expected", [
    (b"{not json", "malformed_json"),
    (b'{"schema":"a","schema":"b"}', "duplicate_json_key"),
    (b'{"x": NaN}', "non_finite_number"),
    (b"\xef\xbb\xbf{}", "malformed_json"),
    (b'"just a string"', "challenge_invalid"),
    (b"[1, 2, 3]", "challenge_invalid"),
    (b"[" * 33 + b"]" * 33, "malformed_json"),
    (b'{"p":"' + b"a" * 4090 + b'"}', "challenge_too_large"),
    (_cd(schema="other.v1"), "challenge_invalid"),
    (_cd(phase="after_restart"), "challenge_invalid"),
    (_cd(method_id="other.method.id"), "challenge_invalid"),
    (_cd(challenge="n" * 257), "challenge_invalid"),
    (_cd(challenge=""), "challenge_invalid"),
    (_cd(challenge=123), "challenge_invalid"),
    (_cd(expected_ranks=True), "challenge_invalid"),
    (_cd(expected_ranks=-1), "challenge_invalid"),
    (_cd(expected_ranks=4294967296), "challenge_invalid"),
    (_cd(expected_ranks=4294967295), None),
    (_cd(expected_ranks=0), None),
    (_cd(created_at=True), "challenge_invalid"),
    (_cd(expires_at="2000"), "challenge_invalid"),
    (_cd(expires_at=1000), "challenge_invalid"),
    (_cd(expires_at=999), "challenge_invalid"),
    (_cd(model_fingerprint=""), "challenge_invalid"),
    (_cd(expected_candidate_started_at=""), "challenge_invalid"),
    (_cd(attempt_id=""), "challenge_invalid"),
])
def test_challenge_document_policy_refuses(
        tmp_path, monkeypatch, challenge_bytes, expected):
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()),
                              challenge_bytes, "challenge-before_restart")
    if expected is None:
        assert code == 0 and err == ""
        assert decode_frame(out) == challenge_bytes
    else:
        assert code == 4 and out == b"" and err == expected + "\n"


@pytest.mark.parametrize("code_rej", [
    "symlink_refused", "leaf_not_regular", "leaf_owner_refused",
    "leaf_mode_refused", "leaf_hardlink_refused",
])
def test_leaf_hardening_refusals_map_to_exit_4(
        tmp_path, monkeypatch, code_rej):
    seam = EmitSeam(monkeypatch, tmp_path)

    def fake_open_leaf(root_fd, basename, missing_code, open_code):
        raise pte._Reject(code_rej)

    monkeypatch.setattr(pte, "_open_leaf", fake_open_leaf)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()),
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == code_rej + "\n"


@pytest.mark.parametrize("code_rej", [
    "journal_root_symlink", "journal_root_not_directory",
    "journal_root_owner_refused", "journal_root_mode_refused",
    "journal_root_missing",
])
def test_root_hardening_refusals_map_to_exit_4(
        tmp_path, monkeypatch, code_rej):
    seam = EmitSeam(monkeypatch, tmp_path)

    def fake_open_root(path):
        raise pte._Reject(code_rej)

    monkeypatch.setattr(pte, "_open_journal_root", fake_open_root)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()),
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 4 and out == b"" and err == code_rej + "\n"
    assert seam.root_closes == []


@pytest.mark.parametrize("bad", [
    "relative/path",
    "\\server\\share",
    "/trailing/",
    "/double//slash",
    "/dot/./x",
    "/dotdot/../x",
    "/",
    "/too-long" + "/x" * 600,
    "/non-ascii-\u00e9",
    "/nul\x00byte",
    "C:/not/posix",
    "",
])
def test_journal_path_grammar_refuses_exit_2(bad):
    # usage grammar is pure: these exit 2 on EVERY platform, before the
    # platform guard and before any filesystem read
    code = pte.main(["--journal", bad, "--class", "challenge-before_restart"])
    assert code == 2


@pytest.mark.parametrize("argv", [
    [],
    ["--journal", "/ok/path"],
    ["--class", "challenge-before_restart"],
    ["--journal", "/ok/path", "--class", "challenge-before_restart", "extra"],
    ["--journal", "/ok/path", "--journal", "/ok/path",
     "--class", "challenge-before_restart"],
    ["--journal", "/ok/path", "--class", "challenge-before_restart",
     "--class", "challenge-after_restart"],
    ["--place", "/ok/path", "--class", "challenge-before_restart"],
    ["--journal"],
    ["--class"],
    ["--journal", "/ok/path", "--class", "place"],
    ["--journal", "/ok/path", "--class", ""],
    ["--journal", "", "--class", "challenge-before_restart"],
])
def test_usage_matrix_exits_2(monkeypatch, argv):
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    code = pte.main(list(argv))
    assert code == 2
    assert err.getvalue() == "usage_refused\n"


def test_same_open_fd_single_capture_and_short_reads(
        tmp_path, monkeypatch):
    # One open and ONE bounded capture per leaf; the frame derives from the
    # captured buffer only (no re-open, no second read). Pathname
    # replacement before the open simply yields the new file through the
    # same gates; in-place same-inode mutation during the read window is a
    # DISCLOSED residual - no immunity is asserted here.
    seam = EmitSeam(monkeypatch, tmp_path)
    reads = []
    real_read = pte._read_bounded

    def counting_read(fd, limit):
        reads.append((fd, limit))
        return real_read(fd, limit)

    monkeypatch.setattr(pte, "_read_bounded", counting_read)
    chal = jdump(challenge_doc())
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()), chal,
                              "challenge-before_restart")
    assert code == 0 and decode_frame(out) == chal
    assert [limit for _, limit in reads] == [pte.MAX_JOURNAL_BYTES,
                                             pte.MAX_CHALLENGE_BYTES]
    assert seam.leaf_opens == ["journal.json",
                               challenge_name("challenge-before_restart")]
    first_frame = out
    # changed bytes on disk are reflected on the NEXT run (no caching),
    # still with exactly one open and one capture per leaf
    reads[:] = []
    chal2 = jdump(challenge_doc(challenge="m" * 64))
    code2, out2, err2 = run_emit(seam, monkeypatch, jdump(journal_doc()),
                                 chal2, "challenge-before_restart")
    assert code2 == 0 and out2 != first_frame
    assert decode_frame(out2) == chal2
    assert len(reads) == 2
    # the real bounded reader tolerates legitimate short reads (no
    # single-os.read assumption) and still enforces the cap
    payload = os.urandom(200)
    blob = tmp_path / "blob"
    blob.write_bytes(payload)
    fd = os.open(str(blob), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        real_os_read = os.read
        calls = []

        def limited_read(f, n):
            calls.append(n)
            return real_os_read(f, min(n, 7))

        monkeypatch.setattr(os, "read", limited_read)
        assert pte._read_bounded(fd, 4096) == payload
        assert len(calls) > 1
        # the first capture left the fd at EOF; rewind so the oversize
        # arm reads fresh bytes and actually exercises the cap
        os.lseek(fd, 0, os.SEEK_SET)
        assert pte._read_bounded(fd, 100) is None
    finally:
        os.close(fd)


def test_zero_intentional_mutation_snapshot_every_path(
        tmp_path, monkeypatch):
    cases = [
        ("happy-before", jdump(journal_doc()), jdump(challenge_doc()),
         "challenge-before_restart", 0),
        ("happy-after", jdump(after_journal_doc()),
         jdump(challenge_doc(phase="after_restart")),
         "challenge-after_restart", 0),
        ("binding-fail", jdump(journal_doc()),
         jdump(challenge_doc(attempt_id="attempt-" + "q" * 32)),
         "challenge-before_restart", 4),
        ("terminal-fail", jdump(journal_doc(phase="COMPLETE")),
         jdump(challenge_doc()), "challenge-before_restart", 4),
        ("missing-challenge", jdump(journal_doc()), None,
         "challenge-before_restart", 4),
        ("malformed-challenge", jdump(journal_doc()), b"{nope",
         "challenge-before_restart", 4),
        ("oversize-challenge", jdump(journal_doc()),
         b'{"p":"' + b"a" * 4090 + b'"}', "challenge-before_restart", 4),
    ]
    for name, journal, chal, klass, want in cases:
        case_dir = tmp_path / name
        case_dir.mkdir()
        seam = EmitSeam(monkeypatch, case_dir)
        # the helper's inputs are written first; the snapshot covers the
        # settled state the helper must leave untouched
        if journal is not None:
            seam.write("journal.json", journal)
        if chal is not None:
            seam.write(challenge_name(klass), chal)
        before = snapshot_tree(case_dir)
        code, out, err = run_emit(seam, monkeypatch, journal, chal, klass,
                                  write=False)
        assert code == want, (name, code, err)
        assert snapshot_tree(case_dir) == before, name


def test_stderr_fixed_value_free_vocabulary(tmp_path, monkeypatch):
    seam = EmitSeam(monkeypatch, tmp_path)
    seen = []
    battery = [
        (jdump(journal_doc()), jdump(challenge_doc()),
         "challenge-before_restart"),
        (jdump(journal_doc()), jdump(challenge_doc(attempt_id="other")),
         "challenge-before_restart"),
        (jdump(journal_doc(phase="COMPLETE")), jdump(challenge_doc()),
         "challenge-before_restart"),
        (jdump(journal_doc()), None, "challenge-before_restart"),
        (jdump(journal_doc()), b"{nope", "challenge-before_restart"),
        (jdump(journal_doc(persistence_decision="x")), jdump(challenge_doc()),
         "challenge-before_restart"),
    ]
    for index, (journal, chal, klass) in enumerate(battery):
        case_dir = tmp_path / ("stderr-" + str(index))
        case_dir.mkdir()
        case_seam = EmitSeam(monkeypatch, case_dir)
        code, out, err = run_emit(case_seam, monkeypatch, journal, chal, klass)
        assert len(err) <= 4096
        seen.extend(err.splitlines())
    assert seen and set(seen) <= FIXED_CODES
    joined = "\n".join(seen)
    assert "Traceback" not in joined
    assert hashlib.sha256(b"x").hexdigest() not in joined
    assert ATTEMPT not in joined and NONCE not in joined
    assert str(tmp_path).replace("\\", "/") not in joined


def test_exit_code_matrix(tmp_path, monkeypatch):
    # 0: happy path through the seam
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()),
                              jdump(challenge_doc()),
                              "challenge-before_restart")
    assert code == 0 and err == "" and decode_frame(out)
    monkeypatch.undo()
    # 2: usage (real parser, platform-independent)
    assert pte.main(["--journal", "bad"]) == 2
    # 4: real platform guard on unsupported platforms (the validation
    # runner is Windows; on a POSIX host the seam keeps the check honest)
    if os.name == "posix":
        monkeypatch.setattr(pte, "_platform_ok", lambda: False)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "stdout", FakeStdout())
    code = pte.main(["--journal", "/some/root",
                     "--class", "challenge-before_restart"])
    assert code == 4 and err.getvalue() == "platform_refused\n"
    monkeypatch.undo()
    # 8: stdout failure refuses after validation (frame may be partial;
    # nonzero exit means the consumer discards the entire frame)
    seam = EmitSeam(monkeypatch, tmp_path)
    code, out, err = run_emit(seam, monkeypatch, jdump(journal_doc()),
                              jdump(challenge_doc()),
                              "challenge-before_restart", stdout_fail=True)
    assert code == 8 and err == "output_refused\n"
    monkeypatch.undo()
    # 8: unexpected internal condition is sanitized to a fixed code
    def boom(root_fd, klass):
        raise RuntimeError("secret-" + "z" * 99)

    monkeypatch.setattr(pte, "_emit_frame", boom)
    seam = EmitSeam(monkeypatch, tmp_path)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "stdout", FakeStdout())
    code = pte.main(["--journal", "/seam/root",
                     "--class", "challenge-before_restart"])
    assert code == 8 and err.getvalue() == "internal_refused\n"
    assert "secret" not in err.getvalue()
    monkeypatch.undo()


OUTPUT_COUNT_ARMS = [
    ("short-count", lambda: CountingBuffer(_arm_short), 0),
    ("none-count", lambda: CountingBuffer(_arm_none), 0),
    ("bool-count", lambda: CountingBuffer(_arm_bool), 0),
    ("negative-count", lambda: CountingBuffer(_arm_negative), 0),
    ("overlong-count", lambda: CountingBuffer(_arm_overlong), 0),
    ("runtimeerror-count", RuntimeErrorBuffer, 0),
    ("write-prefix-then-raise", PrefixThenRaiseBuffer, 0),
    ("full-count-flush-raises", FlushRaisesBuffer, 1),
]


@pytest.mark.parametrize("label,make_buffer,expected_flushes",
                         OUTPUT_COUNT_ARMS)
def test_output_count_failures_refuse_exit_8(
        tmp_path, monkeypatch, label, make_buffer, expected_flushes):
    # single write, exact count check: the short-count stub stores ONLY
    # the prefix it reports; None, bool, negative and overlong counts are
    # all refused BEFORE the flush runs; a partial write that raises
    # (OSError after the prefix, or an ordinary RuntimeError proving the
    # except Exception routing) and a full-count write whose flush raises
    # are DISTINCT arms. Every arm performs exactly one write; each
    # refuses with output_refused / exit 8, exactly one fixed stderr
    # line, and no payload, digest, id or nonce content in stderr.
    # Partial stdout may legitimately remain and is never asserted
    # absent.
    seam = EmitSeam(monkeypatch, tmp_path)
    chal = jdump(challenge_doc())
    seam.write("journal.json", jdump(journal_doc()))
    seam.write("validator-challenge-before_restart.json", chal)
    out = FakeStdout()
    out.buffer = make_buffer()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = pte.main(["--journal", GOOD_PATH,
                     "--class", "challenge-before_restart"])
    assert code == 8
    assert len(out.buffer.write_calls) == 1, label
    assert out.buffer.flush_calls == expected_flushes, label
    text = err.getvalue()
    assert text == "output_refused\n"
    assert NONCE not in text and ATTEMPT not in text
    assert hashlib.sha256(chal).hexdigest() not in text


def test_module_is_self_contained_stdlib_only():
    # the pre-existing v3.6 suite stays green alongside this module: it is
    # stdlib-only, has no import-time side effects beyond definitions and
    # constants, and touches no sibling deploy module
    source = Path(pte.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_imports = {"__future__", "base64", "errno", "hashlib",
                       "json", "os", "stat", "sys"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module.split(".")[0])
    assert imported <= allowed_imports
    kinds = set(type(node).__name__ for node in tree.body)
    assert kinds <= {"Expr", "Import", "ImportFrom", "Assign",
                     "FunctionDef", "ClassDef", "If"}
    assert sum(1 for node in tree.body if isinstance(node, ast.If)) == 1
    for banned in ("subprocess", "socket", "urllib", "requests",
                   "http.client", "publish_exclusive", "os.replace",
                   "os.rename", "os.write"):
        assert banned not in source


def test_module_codes_match_vocabulary():
    # the module's emitted stderr codes (every _Reject code literal and
    # every _refuse literal) are exactly the fixed vocabulary above; the
    # six leaf codes travel as call-site string literals (bound to
    # variables at the open), so their presence is asserted in the source
    source = Path(pte.__file__).read_text(encoding="utf-8")
    found = set(re.findall(r"_Reject\(\"([a-z0-9_]+)\"\)", source))
    found |= set(re.findall(r"_refuse\(\w+, \"([a-z0-9_]+)\"\)", source))
    leaf_codes = {"journal_missing", "journal_open_failed",
                  "journal_too_large", "challenge_missing",
                  "challenge_open_failed", "challenge_too_large"}
    for code in sorted(leaf_codes):
        assert '"%s"' % code in source
    assert found | leaf_codes == FIXED_CODES
