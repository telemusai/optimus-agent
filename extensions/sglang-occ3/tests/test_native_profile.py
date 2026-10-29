"""Offline stdlib tests; MemoryPosix is a MODEL, not native Linux evidence.

Run: python -m unittest discover -s tests -p test_native_profile.py -v
Real Linux filesystem/procfs/flock tests require a disposable CPU environment and
OCC3_NATIVE_PROFILE_LINUX_TESTS=1 plus OCC3_NATIVE_PROFILE_TEST_NAMESPACE=exclusive-disposable.
Native tests use only a uniquely claimed test directory, never the production basename.
The production loaders read no environment.
"""
from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import importlib
import inspect
import io
import json
import os
import posixpath
import stat
import sys
import tempfile
import threading
import traceback
import types
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from occ3_sglang import native_profile as profile


def static_profile():
    return {"model_fingerprint": "offline-model-fingerprint", "model_id": "offline-model",
            "expected_ranks": 8, "hidden_dim": 6144, "layer_index": 77,
            "max_prefix_tokens": 128, "max_live_bytes": 268435456,
            "workspace_bytes": 134217728, "max_records": 256, "max_active": 1,
            "max_ttl_seconds": 3600, "anchor_alpha": 0.0, "tile_rows": 256,
            "method": "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1"}


def source_attestation():
    return {"schema": "optimus.occ3.native-sources.v1",
            "source_revision": "e00930c5489053f26d86b179cee0d087f846acbb",
            "sglang_version": "0.5.21",
            "manifests": [{"name": "offline-overlay.json", "sha256": "a" * 64}],
            "files": [{"root": "sglang_srt", "path": "managers/scheduler.py", "sha256": "b" * 64},
                      {"root": "occ3_sglang", "path": "native_profile.py", "sha256": "c" * 64}]}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def pid1_stat(start=999, comm=b"init (nested) with ) spaces"):
    # Fields 3..21, followed by starttime at 22 and harmless later fields.
    fields = [b"S"] + [str(index).encode() for index in range(4, 22)]
    return b"1 (" + comm + b") " + b" ".join(fields + [str(start).encode(), b"23", b"24"]) + b"\n"


class ErrorAssertions:
    def error_code(self, code, operation):
        with self.assertRaises(profile.ProfileError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        return caught.exception


class PackagedArtifactTests(ErrorAssertions, unittest.TestCase):
    """Actual platform file IO/hash/JSON tests; not Linux mode enforcement on Windows."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.baked = types.ModuleType("occ3_sglang._native_profile_digest")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(profile, "__file__", str(self.directory / "native_profile.py")))
        self.stack.enter_context(mock.patch.dict(sys.modules, {self.baked.__name__: self.baked}))
        self.store("native_profile.json", encoded(static_profile()), "PROFILE_SHA256")
        self.store("native_sources.json", encoded(source_attestation()), "SOURCE_SHA256")

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def store(self, name, raw, attribute):
        path = self.directory / name
        path.write_bytes(raw)
        if os.name == "posix":
            path.chmod(0o600)  # New private test fixture, never production repair.
        setattr(self.baked, attribute, hashlib.sha256(raw).hexdigest())

    def test_exact_raw_hash_and_no_runtime_fields(self):
        first = profile.load_pinned_profile()
        self.assertEqual(first, static_profile())
        self.assertEqual(set(first), profile._PROFILE_FIELDS)
        first["model_id"] = "mutated-return"
        self.assertEqual(profile.load_pinned_profile(), static_profile())
        path = self.directory / "native_profile.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.error_code("profile_digest", profile.load_pinned_profile)

    def test_static_identifiers_match_existing_runtime_profile_alphabet(self):
        value = static_profile()
        value["model_id"] = "organization/offline-model"
        value["model_fingerprint"] = "x" * 256
        self.store("native_profile.json", encoded(value), "PROFILE_SHA256")
        self.assertEqual(profile.load_pinned_profile(), value)

    def test_actual_platform_hardlink_is_refused(self):
        path = self.directory / "native_profile.json"
        alias = self.directory / "artifact-alias.json"
        os.link(path, alias)
        self.error_code("profile_unsafe", profile.load_pinned_profile)

    def test_source_artifact_is_separate_pinned_metadata(self):
        self.assertEqual(profile.load_source_attestation(), source_attestation())
        self.baked.SOURCE_SHA256 = "0" * 64
        self.error_code("profile_digest", profile.load_source_attestation)
        self.assertEqual(profile.load_pinned_profile(), static_profile())

    def test_missing_artifacts_and_baked_digest_fail_closed(self):
        del self.baked.PROFILE_SHA256
        self.error_code("profile_digest", profile.load_pinned_profile)
        self.baked.PROFILE_SHA256 = "A" * 64
        self.error_code("profile_digest", profile.load_pinned_profile)
        self.baked.PROFILE_SHA256 = "a" * 64
        (self.directory / "native_profile.json").unlink()
        self.error_code("profile_io", profile.load_pinned_profile)

    def test_no_parameters_or_environment_redirect(self):
        for operation in (profile.load_pinned_profile, profile.load_source_attestation,
                          profile.get_runtime_incarnation):
            self.assertEqual(len(inspect.signature(operation).parameters), 0)
        with mock.patch.dict(os.environ, {"OCC3_PROFILE_PATH": "untrusted-client-input",
                                         "OCC3_RUNTIME_DIRECTORY": "untrusted-client-input",
                                         "PROFILE_SHA256": "0" * 64,
                                         "SOURCE_SHA256": "0" * 64}):
            self.assertEqual(profile.load_pinned_profile(), static_profile())
            self.assertEqual(profile.load_source_attestation(), source_attestation())

    def test_bounds_invalid_duplicate_and_nonfinite_json(self):
        for raw in (b"", b" " * (profile._PROFILE_LIMIT + 1)):
            self.store("native_profile.json", raw, "PROFILE_SHA256")
            self.error_code("profile_oversize", profile.load_pinned_profile)
        invalid = [b"{", b"\xff", b"[]", b'{"model_id":"a","model_id":"b"}',
                   b'{"anchor_alpha":NaN}', b'{"anchor_alpha":Infinity}',
                   b'{"n":18446744073709551616}', b'{"n":1e999}',
                   b'{"nested":' + b"[" * 10 + b"0" + b"]" * 10 + b"}"]
        for raw in invalid:
            with self.subTest(raw_length=len(raw)):
                self.store("native_profile.json", raw, "PROFILE_SHA256")
                with self.assertRaises(profile.ProfileError):
                    profile.load_pinned_profile()

    def test_profile_is_exact_flat_typed_shape(self):
        changes = [{key: "forbidden"} for key in
                   ("runtime_epoch", "runtime_generation", "generation", "source_revision", "schema")]
        changes += [{"expected_ranks": True}, {"anchor_alpha": True}, {"model_id": {"nested": 1}},
                    {"hidden_dim": -1}, {"method": "bad method"}]
        for changeset in changes:
            with self.subTest(fields=tuple(changeset)):
                self.store("native_profile.json", encoded({**static_profile(), **changeset}), "PROFILE_SHA256")
                self.error_code("profile_invalid", profile.load_pinned_profile)
        value = static_profile()
        del value["tile_rows"]
        self.store("native_profile.json", encoded(value), "PROFILE_SHA256")
        self.error_code("profile_invalid", profile.load_pinned_profile)

    def test_source_schema_revision_version_roots_and_paths_are_fixed(self):
        for key, value in (("source_revision", "0" * 40), ("sglang_version", "0.5.22"),
                           ("schema", "future"), ("extra", 1)):
            obj = source_attestation()
            obj[key] = value
            self.store("native_sources.json", encoded(obj), "SOURCE_SHA256")
            self.error_code("profile_invalid", profile.load_source_attestation)
        for path in ("../escape.py", "/absolute.py", "a/../b.py", "a//b.py", "C:/file.py",
                     "a\\b.py", "a/./b.py", "file.py.txt", "a/", "a/é.py", "_native_profile_digest.py"):
            obj = source_attestation()
            obj["files"][1]["path"] = path
            self.store("native_sources.json", encoded(obj), "SOURCE_SHA256")
            self.error_code("profile_invalid", profile.load_source_attestation)
        obj = source_attestation()
        obj["files"][0]["root"] = "arbitrary-root"
        self.store("native_sources.json", encoded(obj), "SOURCE_SHA256")
        self.error_code("profile_invalid", profile.load_source_attestation)

    def test_source_duplicate_entries_and_bounds(self):
        for collection in ("files", "manifests"):
            obj = source_attestation()
            obj[collection].append(copy.deepcopy(obj[collection][0]))
            self.store("native_sources.json", encoded(obj), "SOURCE_SHA256")
            self.error_code("profile_invalid", profile.load_source_attestation)
        obj = source_attestation()
        obj["files"] = [{"root": "sglang_srt", "path": f"file{i}.py", "sha256": "c" * 64}
                        for i in range(129)]
        self.store("native_sources.json", encoded(obj), "SOURCE_SHA256")
        self.error_code("profile_invalid", profile.load_source_attestation)
        self.store("native_sources.json", b" " * (profile._SOURCE_LIMIT + 1), "SOURCE_SHA256")
        self.error_code("profile_oversize", profile.load_source_attestation)

    def test_native_windows_runtime_refuses_before_artifact_access(self):
        with mock.patch.object(profile, "sys", types.SimpleNamespace(platform="win32")), \
             mock.patch.object(profile, "_artifact") as artifact:
            self.error_code("runtime_unsupported", profile.get_runtime_incarnation)
            artifact.assert_not_called()
        if sys.platform == "win32":
            self.error_code("runtime_unsupported", profile.get_runtime_incarnation)

    def test_import_and_static_loading_do_not_call_runtime_or_model(self):
        with mock.patch.object(profile, "get_runtime_incarnation", side_effect=AssertionError), \
             mock.patch.object(profile, "_read_parent_digest", side_effect=AssertionError), \
             mock.patch.object(profile, "_open_runtime_directory", side_effect=AssertionError):
            self.assertEqual(profile.load_pinned_profile(), static_profile())
        source = inspect.getsource(profile)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import sglang", source)
        self.assertNotIn("os.environ", source)

    def test_fixed_codes_hide_os_error_paths_and_contents(self):
        output = io.StringIO()
        with mock.patch.object(profile, "_read_packaged", side_effect=OSError("private-sentinel")), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                profile.load_pinned_profile()
            except profile.ProfileError:
                trace = traceback.format_exc()
        self.assertNotIn("private-sentinel", trace)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(str(profile.ProfileError("private-sentinel")), "profile_invalid")


class RegisteredMethodTests(ErrorAssertions, unittest.TestCase):
    """Registry extension: challengers registered, advertised surface unchanged."""

    def test_registered_methods_are_advertised_plus_challengers(self):
        advertised = profile.advertised_methods()
        registered = profile.registered_methods()
        self.assertEqual(advertised, (profile.PER_MESSAGE_METHOD, profile.CALIBRATED_METHOD))
        self.assertEqual(registered, (profile.PER_MESSAGE_METHOD, profile.CALIBRATED_METHOD,
                                     profile.PROCRUSTES_METHOD, profile.PROCRUSTES_METHOD_SO,
                                     profile.PROCRUSTES_METHOD_PLAIN))
        self.assertTrue(set(advertised) < set(registered))
        for method in registered:
            descriptor = profile.method_descriptor(method)
            self.assertEqual(set(descriptor), {"method_id", "representation", "alignment",
                                               "fit_source", "solver",
                                               "predictor_position_offset", "precision",
                                               "requires_calibration_artifact",
                                               "artifact_module", "artifact_schema"})
            self.assertEqual(descriptor["predictor_position_offset"], -1)
            self.assertEqual(descriptor["precision"], "float32")
            self.assertEqual(descriptor["solver"].count("."), 2)
            self.assertEqual(descriptor["representation"], "final_layer_prenorm_predictor")
            if descriptor["requires_calibration_artifact"]:
                module = importlib.import_module(descriptor["artifact_module"])
                self.assertEqual(module.ARTIFACT_SCHEMA, descriptor["artifact_schema"])
                self.assertTrue(callable(module.apply_map))
                self.assertTrue(callable(module.deserialize_artifact))
                self.assertTrue(callable(module.artifact_digest))
                self.assertTrue(callable(module.artifact_bytes))
            else:
                self.assertIsNone(descriptor["artifact_module"])
                self.assertIsNone(descriptor["artifact_schema"])
            self.assertEqual(profile.require_method(method), method)
            receipt = profile.receipt_descriptor(method)
            self.assertEqual(set(receipt), {"method_id", "representation", "alignment",
                                            "predictor_position_offset", "precision"})

    def test_challenger_alignment_labels_are_distinct(self):
        labels = {profile.method_descriptor(method)["alignment"]
                  for method in profile.registered_methods()}
        self.assertEqual(len(labels), len(profile.registered_methods()))

    def test_binding_types_dispatch_through_the_registry(self):
        import dataclasses

        from occ3_sglang.statebridge_calibrated import CalibrationBinding
        from occ3_sglang.statebridge_calibrated_procrustes import ProcrustesBinding
        calibrated = static_profile()
        calibrated["method"] = profile.CALIBRATED_METHOD
        values = dict(CalibrationBinding(
            method_id=profile.CALIBRATED_METHOD, hidden_dim=6144, rows=512, layer_index=77,
            ridge=1.5, ridge_rel=0.02, solve_residual=0.01, hyperparameters={"rho": 0.02},
            capture_span_policy="whole_completed_response_tail_v1",
            model_fingerprint=calibrated["model_fingerprint"], checkpoint_revision="rev-1",
            tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
            capture_method_id="glm53.final-prenorm.capture.v1",
            predictor_position_offset=-1, precision="float32", corpus_digest="c" * 64,
            fit_manifest_digest="d" * 64, validation_record_digest="",
            validation_artifact_digest="").scalars())
        binding = CalibrationBinding(**values)
        self.assertEqual(profile.validate_method_binding(binding, calibrated)["method_id"],
                         profile.CALIBRATED_METHOD)
        # A Procrustes binding class does not belong to the dual-ridge method:
        # the registry refuses a binding built by another method's module.
        foreign = ProcrustesBinding(
            method_id=profile.PROCRUSTES_METHOD, variant="o", r=64, eta=0.01, a0=0.0,
            rotation_defect=1e-6, rotation_determinant=1.0, hidden_dim=6144, rows=512,
            layer_index=77, model_fingerprint=calibrated["model_fingerprint"],
            checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
            embedding_id="glm53-tp8-vocab", capture_method_id="glm53.final-prenorm.capture.v1",
            capture_span_policy="whole_completed_response_tail_v1",
            predictor_position_offset=-1, precision="float32",
            fit_manifest_digest="d" * 64, corpus_digest="c" * 64, rank_hidden=64,
            rank_embedding=64, participation_hidden=2.0, participation_embedding=2.0,
            whitening_gain_hidden=1.0, whitening_gain_embedding=1.0)
        self.error_code("profile_invalid", lambda:
                        profile.validate_method_binding(foreign, calibrated))
        forged = dataclasses.replace(binding)
        object.__setattr__(forged, "method_id", profile.PROCRUSTES_METHOD)
        self.error_code("profile_invalid", lambda:
                        profile.validate_method_binding(forged, calibrated))
        # A calibrated binding under the message method is refused as before.
        message = static_profile()
        self.error_code("profile_invalid", lambda:
                        profile.validate_method_binding(binding, message))

    def test_a_pinned_profile_still_names_only_advertised_methods(self):
        self.assertTrue(static_profile()["method"] == profile.PER_MESSAGE_METHOD
                        or static_profile()["method"] in profile.advertised_methods())
        challenger = static_profile()
        challenger["method"] = profile.PROCRUSTES_METHOD
        self.error_code("profile_invalid", lambda: profile._validate_profile(challenger))


class ParentTokenTests(ErrorAssertions, unittest.TestCase):
    def test_field22_not21_with_nested_comm(self):
        self.assertEqual(profile._parse_pid1_start(pid1_stat(54321)), 54321)
        self.assertEqual(profile._parse_pid1_start(pid1_stat(0, b")")), 0)
        self.assertEqual(profile._parse_pid1_start(pid1_stat((1 << 64) - 1)), (1 << 64) - 1)

    def test_bounded_invalid_stat(self):
        for raw in (b"", b"1 (bad", b"2 (bad) S 3", b"1 (a) S 1", b"x" * 4097,
                    pid1_stat(-1), pid1_stat(1 << 64), pid1_stat(1).replace(b"\n", b"\0")):
            self.error_code("runtime_parent", lambda: profile._parse_pid1_start(raw))

    def test_each_parent_component_changes_digest(self):
        boot = b"11111111-2222-4333-8444-555555555555\n"
        first = profile._parent_digest(boot, 4026532000, pid1_stat(10))
        self.assertEqual(first, profile._parent_digest(boot, 4026532000, pid1_stat(10)))
        values = [profile._parent_digest(boot, 4026532001, pid1_stat(10)),
                  profile._parent_digest(boot, 4026532000, pid1_stat(11)),
                  profile._parent_digest(b"21111111-2222-4333-8444-555555555555\n", 4026532000, pid1_stat(10))]
        self.assertTrue(all(value != first for value in values))
        self.assertTrue(all(len(value) == 64 for value in values))


class MemoryPosix:
    """Explicit in-memory filesystem/flock semantics; NOT an OS implementation."""
    name = "posix"
    path = posixpath
    O_RDONLY, O_WRONLY, O_RDWR = 0, 1, 2
    O_CREAT, O_EXCL, O_NOFOLLOW = 64, 128, 131072
    O_CLOEXEC, O_NONBLOCK, O_DIRECTORY = 524288, 2048, 65536
    LOCK_EX, LOCK_NB = 2, 4

    def __init__(self):
        self.nodes = {}
        self.handles = {}
        self.counter = 10
        self.mutex = threading.RLock()
        self.fail_at = None
        self.busy = False
        self.swap_lock = False
        self.write_limit = 10000
        self.events = []
        self.add("/", stat.S_IFDIR | 0o755, uid=0)
        self.add("/tmp", stat.S_IFDIR | 0o1777, uid=0)

    def add(self, path, mode=stat.S_IFREG | 0o600, *, uid=1000, data=b"", links=1):
        self.counter += 1
        node = types.SimpleNamespace(st_dev=1, st_ino=self.counter, st_mode=mode, st_uid=uid,
                                     st_nlink=links, data=data, st_size=len(data), lock=threading.Lock())
        self.nodes[path] = node
        return node

    def resolve(self, path, dir_fd=None):
        return path if path.startswith("/") else posixpath.join(self.handles[dir_fd].path, path)

    @staticmethod
    def geteuid():
        return 1000

    def mkdir(self, path, mode, dir_fd=None):
        with self.mutex:
            path = self.resolve(path, dir_fd)
            if path in self.nodes:
                raise FileExistsError()
            self.add(path, stat.S_IFDIR | mode)

    def open(self, path, flags, mode=0o777, dir_fd=None):
        with self.mutex:
            path = self.resolve(path, dir_fd)
            self.events.append(("open", path, flags))
            node = self.nodes.get(path)
            if node is not None and flags & self.O_EXCL:
                raise FileExistsError()
            if node is None:
                if not flags & self.O_CREAT:
                    raise FileNotFoundError()
                node = self.add(path, stat.S_IFREG | mode)
            if stat.S_ISLNK(node.st_mode) and flags & self.O_NOFOLLOW:
                raise OSError(errno.ELOOP, "private-sentinel")
            if flags & self.O_DIRECTORY and not stat.S_ISDIR(node.st_mode):
                raise NotADirectoryError()
            self.counter += 1
            fd = self.counter
            self.handles[fd] = types.SimpleNamespace(node=node, offset=0, path=path, held=False)
            return fd

    def close(self, fd):
        with self.mutex:
            handle = self.handles.pop(fd)
            if handle.held:
                handle.node.lock.release()

    def fstat(self, fd):
        with self.mutex:
            return copy.copy(self.handles[fd].node)

    def stat(self, path, *, dir_fd=None, follow_symlinks=True):
        with self.mutex:
            path = self.resolve(path, dir_fd)
            if path not in self.nodes:
                raise FileNotFoundError()
            return copy.copy(self.nodes[path])

    def read(self, fd, count):
        with self.mutex:
            handle = self.handles[fd]
            value = handle.node.data[handle.offset:handle.offset + count]
            handle.offset += len(value)
            self.events.append(("read", handle.path, count))
            return value

    def interrupt(self, stage):
        if self.fail_at == stage:
            self.fail_at = None
            raise OSError("private-sentinel")

    def write(self, fd, data):
        with self.mutex:
            self.interrupt("write")
            handle = self.handles[fd]
            count = min(len(data), self.write_limit)
            handle.node.data += data[:count]
            handle.node.st_size = len(handle.node.data)
            return count

    def fsync(self, fd):
        with self.mutex:
            directory = stat.S_ISDIR(self.handles[fd].node.st_mode)
            self.events.append(("fsync_dir" if directory else "fsync_file", self.handles[fd].path))
            self.interrupt("dir_fsync" if directory else "file_fsync")

    def replace(self, src, dst, *, src_dir_fd, dst_dir_fd):
        with self.mutex:
            self.interrupt("replace")
            src, dst = self.resolve(src, src_dir_fd), self.resolve(dst, dst_dir_fd)
            self.events.append(("replace", src, dst))
            self.nodes[dst] = self.nodes.pop(src)

    def unlink(self, path, *, dir_fd=None):
        with self.mutex:
            self.nodes.pop(self.resolve(path, dir_fd))

    def listdir(self, fd):
        with self.mutex:
            prefix = self.handles[fd].path.rstrip("/") + "/"
            return [path[len(prefix):] for path in self.nodes
                    if path.startswith(prefix) and "/" not in path[len(prefix):]]

    def rmdir(self, path, *, dir_fd=None):
        with self.mutex:
            path = self.resolve(path, dir_fd)
            if any(name.startswith(path.rstrip("/") + "/") for name in self.nodes):
                raise OSError(errno.ENOTEMPTY, "directory not empty")
            self.nodes.pop(path)

    def flock(self, fd, flags):
        with self.mutex:
            handle = self.handles[fd]
            if self.busy or not handle.node.lock.acquire(blocking=False):
                raise BlockingIOError(errno.EAGAIN, "private-sentinel")
            handle.held = True
            if self.swap_lock:
                self.add(handle.path)


class ModeledFilesystemTests(ErrorAssertions, unittest.TestCase):
    """All results in this class are explicit filesystem-model evidence only."""
    private = "/tmp/optimus-occ3-native"
    record_path = private + "/incarnation.json"
    lock_path = private + "/incarnation.lock"

    def setUp(self):
        self.fs = MemoryPosix()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(profile, "os", self.fs))
        self.stack.enter_context(mock.patch.object(profile, "sys", types.SimpleNamespace(platform="linux")))
        self.stack.enter_context(mock.patch.dict(sys.modules, {"fcntl": self.fs}))
        self.parent_digest = "1" * 64
        self.profile_sha = "2" * 64
        self.source_sha = "3" * 64
        self.parent_reader = profile._read_parent_digest
        self.stack.enter_context(mock.patch.object(profile, "_read_parent_digest", side_effect=lambda: self.parent_digest))
        self.artifact_patch = self.stack.enter_context(mock.patch.object(profile, "_artifact", side_effect=self.artifact))

    def tearDown(self):
        self.stack.close()
        self.assertEqual(self.fs.handles, {}, "descriptor leak in filesystem model")

    def artifact(self, name, _attribute, _limit):
        if name == "native_profile.json":
            return static_profile(), self.profile_sha
        return source_attestation(), self.source_sha

    def record(self):
        return json.loads(self.fs.nodes[self.record_path].data)

    def set_record(self, record):
        self.fs.add(self.record_path, data=encoded(record))

    def test_same_token_stable_and_restart_changes_epoch_and_generation(self):
        first = profile.get_runtime_incarnation()
        self.assertEqual(set(first), profile._RUNTIME_FIELDS)
        self.assertEqual(first, profile.get_runtime_incarnation())
        self.assertEqual(first["runtime_generation"], 1)
        self.parent_digest = "4" * 64
        second = profile.get_runtime_incarnation()
        self.assertNotEqual(first["runtime_epoch"], second["runtime_epoch"])
        self.assertEqual(second["runtime_generation"], 2)
        self.assertEqual(second, profile.get_runtime_incarnation())
        self.assertEqual(self.fs.nodes[self.private].st_mode, stat.S_IFDIR | 0o700)
        self.assertEqual(self.fs.nodes[self.record_path].st_mode, stat.S_IFREG | 0o600)
        self.assertEqual(self.fs.nodes[self.lock_path].st_mode, stat.S_IFREG | 0o600)

    def test_each_static_digest_change_refuses_under_same_parent(self):
        first = profile.get_runtime_incarnation()
        for attr in ("profile_sha", "source_sha"):
            old = getattr(self, attr)
            setattr(self, attr, "9" * 64)
            self.error_code("runtime_profile_changed", profile.get_runtime_incarnation)
            setattr(self, attr, old)
            self.assertEqual(first, profile.get_runtime_incarnation())
        self.parent_digest = "5" * 64
        self.profile_sha = "6" * 64
        self.source_sha = "7" * 64
        self.assertEqual(profile.get_runtime_incarnation()["runtime_generation"], 2)

    def test_max_generation_is_stable_but_cannot_advance(self):
        profile.get_runtime_incarnation()
        value = self.record()
        value["runtime_generation"] = (1 << 64) - 1
        self.set_record(value)
        self.assertEqual(profile.get_runtime_incarnation()["runtime_generation"], (1 << 64) - 1)
        self.parent_digest = "8" * 64
        self.error_code("runtime_generation_exhausted", profile.get_runtime_incarnation)
        self.assertEqual(self.record(), value)

    def test_malformed_existing_record_never_repaired_even_after_restart(self):
        profile.get_runtime_incarnation()
        valid = self.record()
        malformed = [b"{", b"", b"x" * (profile._RECORD_LIMIT + 1),
                     encoded({**valid, "runtime_generation": True}),
                     encoded({**valid, "runtime_generation": 0}),
                     encoded({**valid, "runtime_epoch": "invalid"}),
                     encoded({**valid, "source_attestation_sha256": "F" * 64}),
                     encoded({**valid, "extra": 1}),
                     encoded(valid)[:-1] + b',"runtime_generation":1}']
        self.parent_digest = "9" * 64
        for raw in malformed:
            with self.subTest(length=len(raw)):
                self.fs.add(self.record_path, data=raw)
                self.error_code("runtime_record", profile.get_runtime_incarnation)
                self.assertEqual(self.fs.nodes[self.record_path].data, raw)

    def test_private_directory_owner_mode_and_symlink_fail_without_repair(self):
        for mode, uid in ((stat.S_IFDIR | 0o755, 1000), (stat.S_IFDIR | 0o700, 1001),
                          (stat.S_IFLNK | 0o777, 1000)):
            with self.subTest(mode=mode, uid=uid):
                node = self.fs.add(self.private, mode, uid=uid)
                with self.assertRaises(profile.ProfileError):
                    profile.get_runtime_incarnation()
                self.assertEqual((node.st_mode, node.st_uid), (mode, uid))
                self.assertNotIn(self.lock_path, self.fs.nodes)

    def test_unsafe_lock_and_record_refused_without_repair(self):
        for name in (self.lock_path, self.record_path):
            for mode, links, uid in ((stat.S_IFREG | 0o644, 1, 1000),
                                     (stat.S_IFREG | 0o600, 2, 1000),
                                     (stat.S_IFREG | 0o600, 1, 1001),
                                     (stat.S_IFLNK | 0o777, 1, 1000),
                                     (stat.S_IFIFO | 0o600, 1, 1000)):
                with self.subTest(name=name, mode=mode, links=links, uid=uid):
                    self.fs.add(self.private, stat.S_IFDIR | 0o700)
                    node = self.fs.add(name, mode, links=links, uid=uid, data=b"bad")
                    with self.assertRaises(profile.ProfileError):
                        profile.get_runtime_incarnation()
                    self.assertIs(self.fs.nodes[name], node)
                    self.fs.nodes.pop(name)
        self.fs.add(self.lock_path, data=b"nonempty-lock")
        self.error_code("runtime_unsafe", profile.get_runtime_incarnation)

    def test_unsafe_fixed_parent_and_lock_inode_replacement_refuse(self):
        self.fs.nodes["/tmp"].st_mode = stat.S_IFDIR | 0o777
        self.error_code("runtime_unsafe", profile.get_runtime_incarnation)
        self.fs.nodes["/tmp"].st_mode = stat.S_IFDIR | 0o1777
        self.fs.swap_lock = True
        self.error_code("runtime_unsafe", profile.get_runtime_incarnation)

    def test_lock_contention_deadline_is_bounded(self):
        self.fs.busy = True
        clock = [0.0]
        timer = types.SimpleNamespace(monotonic=lambda: clock[0],
                                      sleep=lambda delay: clock.__setitem__(0, clock[0] + delay))
        with mock.patch.object(profile, "time", timer):
            try:
                profile.get_runtime_incarnation()
            except profile.ProfileError as exc:
                self.assertEqual(exc.code, "runtime_busy")
                trace = traceback.format_exc()
            else:
                self.fail("contended lock did not time out")
        self.assertNotIn("private-sentinel", trace)
        self.assertLessEqual(clock[0], profile._LOCK_TIMEOUT + profile._LOCK_INTERVAL)
        self.assertNotIn(self.record_path, self.fs.nodes)

    def test_atomic_interruption_retains_old_or_complete_new_record_only(self):
        first = profile.get_runtime_incarnation()
        orphan_path = self.private + "/.incarnation-unrelated.tmp"
        self.fs.add(orphan_path, data=b"leave-this-alone")
        self.parent_digest = "a" * 64
        for stage in ("write", "file_fsync", "replace"):
            with self.subTest(stage=stage):
                self.fs.fail_at = stage
                self.error_code("runtime_io", profile.get_runtime_incarnation)
                self.assertEqual(self.record()["runtime_epoch"], first["runtime_epoch"])
                self.assertEqual(set(self.fs.nodes) - {"/", "/tmp", self.private},
                                 {self.record_path, self.lock_path, orphan_path})
        self.fs.fail_at = "dir_fsync"
        self.error_code("runtime_io", profile.get_runtime_incarnation)
        complete = profile.get_runtime_incarnation()
        self.assertEqual(complete["runtime_generation"], 2)
        self.assertNotEqual(complete["runtime_epoch"], first["runtime_epoch"])
        self.assertEqual(self.fs.nodes[orphan_path].data, b"leave-this-alone")

    def test_short_writes_and_atomic_fsync_order(self):
        self.fs.write_limit = 7
        result = profile.get_runtime_incarnation()
        self.assertEqual(self.record()["runtime_epoch"], result["runtime_epoch"])
        stages = [event[0] for event in self.fs.events if event[0] in {"fsync_file", "replace", "fsync_dir"}]
        self.assertEqual(stages, ["fsync_file", "replace", "fsync_dir"])
        for event in self.fs.events:
            if event[0] == "open" and event[1] != "/":
                self.assertTrue(event[2] & self.fs.O_NOFOLLOW)

    def test_concurrent_modeled_frontend_and_workers_converge(self):
        barrier = threading.Barrier(9)
        def call(_index):
            barrier.wait(timeout=5)
            return profile.get_runtime_incarnation()
        with ThreadPoolExecutor(max_workers=9) as pool:
            results = list(pool.map(call, range(9)))
        self.assertTrue(all(value == results[0] for value in results))
        self.assertEqual(results[0]["runtime_generation"], 1)
        self.assertEqual(sum(event[0] == "replace" for event in self.fs.events), 1)

    def test_static_posix_file_modes_links_symlinks_and_size_are_modeled(self):
        self.fs.add("/image", stat.S_IFDIR | 0o755, uid=0)
        self.fs.add("/image/occ3_sglang", stat.S_IFDIR | 0o755, uid=0)
        path = "/image/occ3_sglang/native_profile.json"
        raw = encoded(static_profile())
        with mock.patch.object(profile, "__file__", "/image/occ3_sglang/native_profile.py"):
            for mode, links, uid in ((stat.S_IFLNK | 0o777, 1, 0),
                                     (stat.S_IFREG | 0o666, 1, 0),
                                     (stat.S_IFREG | 0o644, 2, 0),
                                     (stat.S_IFREG | 0o644, 1, 1001)):
                self.fs.add(path, mode, links=links, uid=uid, data=raw)
                self.error_code("profile_unsafe", lambda: profile._read_packaged("native_profile.json", 16384))
            self.fs.add(path, stat.S_IFREG | 0o444, uid=0, data=raw)
            self.assertEqual(profile._read_packaged("native_profile.json", 16384), raw)
            self.fs.add(path, stat.S_IFREG | 0o444, uid=0, data=b"x" * 16385)
            reads_before = sum(event[0] == "read" for event in self.fs.events)
            self.error_code("profile_oversize", lambda: profile._read_packaged("native_profile.json", 16384))
            self.assertEqual(sum(event[0] == "read" for event in self.fs.events), reads_before)

    def test_procfs_model_reads_fixed_parents_and_checks_own_namespace(self):
        for directory in ("/proc", "/proc/sys", "/proc/sys/kernel", "/proc/sys/kernel/random",
                          "/proc/1", "/proc/1/ns", "/proc/self", "/proc/self/ns"):
            self.fs.add(directory, stat.S_IFDIR | 0o555, uid=0)
        boot = b"11111111-2222-4333-8444-555555555555\n"
        self.fs.add("/proc/sys/kernel/random/boot_id", stat.S_IFREG | 0o444, uid=0, data=boot)
        self.fs.add("/proc/1/stat", stat.S_IFREG | 0o444, uid=0, data=pid1_stat(12345))
        # The model's /proc/self directory represents the resolved kernel link.
        own = self.fs.add("/proc/self/ns/pid", stat.S_IFREG | 0o444, uid=0)
        init = self.fs.add("/proc/1/ns/pid", stat.S_IFREG | 0o444, uid=0)
        own.st_ino = init.st_ino = 4026532000
        # setUp patches the public parent-reading seam; retain its original callable.
        original = self.parent_reader
        self.assertEqual(original(), profile._parent_digest(boot, 4026532000, pid1_stat(12345)))
        own.st_ino += 1
        self.error_code("runtime_parent", original)
        own.st_ino = init.st_ino
        self.fs.nodes["/proc/sys/kernel/random/boot_id"].data = b"x" * 129
        self.error_code("runtime_parent", original)

    def test_runtime_errors_do_not_log_private_exception_content(self):
        self.fs.fail_at = "write"
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                profile.get_runtime_incarnation()
            except profile.ProfileError:
                trace = traceback.format_exc()
        self.assertNotIn("private-sentinel", trace)
        self.assertEqual(output.getvalue(), "")


class _NativeTestDirectory:
    """Test-only exclusive claim; never delete untracked or replaced state."""
    prefix = "optimus-occ3-native-test-"
    expected_files = {"incarnation.json", "incarnation.lock"}

    def __init__(self, filesystem=os, *, name=None):
        self.name = self.prefix + uuid.uuid4().hex if name is None else name
        self._check(type(self.name) is str and self.name.startswith(self.prefix)
                    and len(self.name) == len(self.prefix) + 32
                    and all(char in "0123456789abcdef" for char in self.name[len(self.prefix):]))
        self.fs = filesystem
        self.parent = self.directory = None
        self.owned = {}
        self.flags = self.fs.O_RDONLY | self.fs.O_DIRECTORY | self.fs.O_NOFOLLOW | self.fs.O_CLOEXEC
        try:
            self.parent = self.fs.open("/tmp", self.flags)
            # Atomic exclusion, not a check-then-later-create window.
            self.fs.mkdir(self.name, mode=0o700, dir_fd=self.parent)
            self.directory = self.fs.open(self.name, self.flags, dir_fd=self.parent)
            self.identity = self.fs.fstat(self.directory)
            self._check(stat.S_ISDIR(self.identity.st_mode)
                        and self.identity.st_uid == self.fs.geteuid()
                        and stat.S_IMODE(self.identity.st_mode) == 0o700)
            self._check_directory()
            self._check(not self.fs.listdir(self.directory))
        except BaseException:
            self._close()
            raise

    @staticmethod
    def _check(condition):
        if not condition:
            raise AssertionError("native_test_state_changed")

    def _check_directory(self):
        named = self.fs.stat(self.name, dir_fd=self.parent, follow_symlinks=False)
        self._check(profile._same_file(self.identity, named)
                    and profile._same_file(self.identity, self.fs.fstat(self.directory))
                    and stat.S_ISDIR(named.st_mode) and named.st_uid == self.fs.geteuid()
                    and stat.S_IMODE(named.st_mode) == 0o700)

    def _check_file(self, name, identity):
        named = self.fs.stat(name, dir_fd=self.directory, follow_symlinks=False)
        self._check(profile._same_file(identity, named) and stat.S_ISREG(named.st_mode)
                    and (named.st_uid, named.st_nlink, stat.S_IMODE(named.st_mode))
                    == (self.fs.geteuid(), 1, 0o600))

    def remember_files(self):
        self._check_directory()
        self._check(not self.owned and set(self.fs.listdir(self.directory)) == self.expected_files)
        for name in sorted(self.expected_files):
            fd = self.fs.open(name, self.fs.O_RDONLY | self.fs.O_NOFOLLOW | self.fs.O_CLOEXEC,
                              dir_fd=self.directory)
            identity = self.fs.fstat(fd)
            self.owned[name] = (fd, identity)
            self._check_file(name, identity)
        # Open descriptors keep the recorded inodes from being reused until cleanup.

    def _close(self):
        for fd, _identity in self.owned.values():
            self.fs.close(fd)
        self.owned.clear()
        if self.directory is not None:
            self.fs.close(self.directory)
            self.directory = None
        if self.parent is not None:
            self.fs.close(self.parent)
            self.parent = None

    def cleanup(self):
        if self.parent is None:
            return
        try:
            self._check_directory()
            self._check(set(self.fs.listdir(self.directory)) == set(self.owned))
            # Validate every file before deleting any, then use retained directory fd.
            for name, (_fd, identity) in self.owned.items():
                self._check_file(name, identity)
            for name, (_fd, identity) in self.owned.items():
                self._check_file(name, identity)
                self.fs.unlink(name, dir_fd=self.directory)
            self._check_directory()
            self.fs.rmdir(self.name, dir_fd=self.parent)
        finally:
            self._close()


class NativeTestCleanupModelTests(unittest.TestCase):
    """NP01 regressions use MemoryPosix, not native Linux cleanup proof."""
    name = "optimus-occ3-native-test-" + "0" * 32
    directory = "/tmp/" + name

    def setUp(self):
        self.fs = MemoryPosix()

    def claim(self):
        return _NativeTestDirectory(self.fs, name=self.name)

    def add_files(self):
        for name in _NativeTestDirectory.expected_files:
            self.fs.add(self.directory + "/" + name, data=b"test-state")

    def test_exclusive_claim_refuses_preexisting_directory_without_touching_it(self):
        self.fs.add(self.directory, stat.S_IFDIR | 0o700)
        self.add_files()
        before = dict(self.fs.nodes)
        with self.assertRaises(FileExistsError):
            self.claim()
        self.assertEqual(self.fs.nodes, before)
        self.assertEqual(self.fs.handles, {})

    def test_explicit_disposable_namespace_gate_precedes_filesystem_claim(self):
        module = sys.modules[__name__]
        with mock.patch.object(module, "sys", types.SimpleNamespace(platform="linux")):
            for enabled, namespace, expected in (("", "", False), ("1", "", False),
                    ("", "exclusive-disposable", False), ("1", "exclusive-disposable", True)):
                with mock.patch.dict(os.environ, {"OCC3_NATIVE_PROFILE_LINUX_TESTS": enabled,
                                     "OCC3_NATIVE_PROFILE_TEST_NAMESPACE": namespace}):
                    self.assertEqual(_native_linux_test_authorized(), expected)
            with mock.patch.dict(os.environ, {"OCC3_NATIVE_PROFILE_LINUX_TESTS": "1",
                                 "OCC3_NATIVE_PROFILE_TEST_NAMESPACE": ""}), \
                 mock.patch.object(module, "_NativeTestDirectory") as claim:
                case = NativeLinuxFilesystemTests("test_actual_flock_contention_times_out")
                with self.assertRaises(unittest.SkipTest):
                    case.setUp()
                claim.assert_not_called()

    def test_production_basename_refused_before_any_filesystem_change(self):
        before = dict(self.fs.nodes)
        with self.assertRaisesRegex(AssertionError, "native_test_state_changed"):
            _NativeTestDirectory(self.fs, name="optimus-occ3-native")
        self.assertEqual(self.fs.nodes, before)
        self.assertEqual(self.fs.handles, {})

    def test_empty_owned_directory_cleanup_is_safe(self):
        claim = self.claim()
        claim.cleanup()
        claim.cleanup()  # Cleanup never reopens a path after relinquishing ownership.
        self.assertNotIn(self.directory, self.fs.nodes)
        self.assertEqual(self.fs.handles, {})

    def test_inert_setup_cleanup_never_adopts_untracked_two_file_state(self):
        claim = self.claim()
        self.add_files()  # Unrelated state appeared without this test creating it.
        before = dict(self.fs.nodes)
        with self.assertRaisesRegex(AssertionError, "native_test_state_changed"):
            claim.cleanup()
        self.assertEqual(self.fs.nodes, before)
        self.assertEqual(self.fs.handles, {})

    def test_replaced_directory_inode_is_not_cleaned(self):
        claim = self.claim()
        self.add_files()
        claim.remember_files()
        self.fs.add(self.directory, stat.S_IFDIR | 0o700)
        self.add_files()
        before = dict(self.fs.nodes)
        with self.assertRaisesRegex(AssertionError, "native_test_state_changed"):
            claim.cleanup()
        self.assertEqual(self.fs.nodes, before)
        self.assertEqual(self.fs.handles, {})

    def test_replaced_file_inode_refuses_all_cleanup_before_first_unlink(self):
        claim = self.claim()
        self.add_files()
        claim.remember_files()
        self.fs.add(self.directory + "/incarnation.json", data=b"unrelated-state")
        before = dict(self.fs.nodes)
        with self.assertRaisesRegex(AssertionError, "native_test_state_changed"):
            claim.cleanup()
        self.assertEqual(self.fs.nodes, before)
        self.assertEqual(self.fs.handles, {})

    def test_cleanup_removes_only_tracked_own_state(self):
        claim = self.claim()
        self.add_files()
        claim.remember_files()
        neighbor = self.fs.add("/tmp/unrelated-state", data=b"leave-alone")
        claim.cleanup()
        self.assertEqual(set(self.fs.nodes), {"/", "/tmp", "/tmp/unrelated-state"})
        self.assertIs(self.fs.nodes["/tmp/unrelated-state"], neighbor)
        self.assertEqual(self.fs.handles, {})


def _linux_child(queue):
    try:
        queue.put((True, profile.get_runtime_incarnation()))
    except profile.ProfileError as exc:
        queue.put((False, exc.code))


def _native_linux_test_authorized():
    return (sys.platform.startswith("linux")
            and os.environ.get("OCC3_NATIVE_PROFILE_LINUX_TESTS") == "1"
            and os.environ.get("OCC3_NATIVE_PROFILE_TEST_NAMESPACE") == "exclusive-disposable")


@unittest.skipUnless(_native_linux_test_authorized(),
                     "requires native Linux and explicit exclusive disposable test namespace")
class NativeLinuxFilesystemTests(ErrorAssertions, unittest.TestCase):
    """Real Linux primitives. Synthetic pinned artifacts are explicitly test-only."""

    def setUp(self):
        if not _native_linux_test_authorized():
            self.skipTest("explicit exclusive disposable Linux namespace not authorized")
        try:
            self.claim = _NativeTestDirectory()
        except FileExistsError:
            self.skipTest("exclusive test directory claim failed; never touch existing state")
        self.addCleanup(self.claim.cleanup)
        self.directory = Path("/tmp") / self.claim.name
        directory_patch = mock.patch.object(profile, "_RUNTIME_DIRECTORY", self.claim.name)
        directory_patch.start()
        self.addCleanup(directory_patch.stop)
        self.artifact_patch = mock.patch.object(profile, "_artifact", side_effect=lambda name, *_:
            (static_profile(), "2" * 64) if name == "native_profile.json" else (source_attestation(), "3" * 64))
        self.artifact_patch.start()
        self.addCleanup(self.artifact_patch.stop)

    def test_actual_procfs_permissions_atomic_record_and_multiprocess_convergence(self):
        import multiprocessing
        first = profile.get_runtime_incarnation()
        self.claim.remember_files()
        self.assertEqual(first, profile.get_runtime_incarnation())
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        workers = [context.Process(target=_linux_child, args=(queue,)) for _ in range(8)]
        try:
            for worker in workers:
                worker.start()
            results = [queue.get(timeout=10) for _ in workers]
            for worker in workers:
                worker.join(timeout=5)
                self.assertEqual(worker.exitcode, 0)
            self.assertTrue(all(ok and value == first for ok, value in results))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            queue.close()
            queue.join_thread()

    def test_actual_flock_contention_times_out(self):
        import fcntl
        profile.get_runtime_incarnation()
        self.claim.remember_files()
        fd = os.open(self.directory / "incarnation.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(profile, "_LOCK_TIMEOUT", 0.05):
                self.error_code("runtime_busy", profile.get_runtime_incarnation)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
