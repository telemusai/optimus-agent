"""Synthetic stdlib exporter tests. No live profile, Cargo, provider, or scorer."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_memory_bench_snapshot", ROOT / "scripts/export_memory_bench_snapshot.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
PROJECT_ID = "project_fixture"
HOST_ID = "00000000-0000-0000-0000-000000000001"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def document(project_id=PROJECT_ID):
    return {"schema": 1, "entries": {kind: {} for kind in ("memory", "prompt", "skill", "subagent")},
            "refinements": [], "memory": {"schema": 1, "projectId": project_id, "revision": 3,
                                           "history": [], "events": {}}}


def harness():
    value = document()
    del value["memory"]
    return value


def entry(entry_id="fact", sources=None, **metadata):
    meta = {"projectId": PROJECT_ID, **metadata}
    if sources is not None:
        meta["sources"] = sources
    return {"id": entry_id, "kind": "memory", "title": "Synthetic note", "content": "Synthetic remembered fact.",
            "path": "general", "metadata": meta, "reference": {}, "arguments": {}, "source": "refine",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z", "version": 1}


def full_settings(**updates):
    result = {"recall": True, "learning": True, "maxRecallChars": 6000, "maxRecallEntries": 6,
              "maxExtractionTokens": 4096, "maxImportBytes": 33554432, "maxImportChunkChars": 40000,
              "maxImportChunksPerRun": 4, "importInstructions": None,
              "recallQueryDistillation": False, "recallRerank": False}
    result.update(updates)
    return result


class SnapshotExportTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="memory-export-synthetic-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.run = self.root / "frozen run"
        self.env = self.run / "envs" / "env_one"
        self.workspace = self.env / "workspace"
        self.workspace.mkdir(parents=True)
        self.project_dir = self.env / "agent" / "memory" / "projects" / PROJECT_ID
        self.memory_path = self.project_dir / "harness_state.json"
        self.global_settings = self.env / "agent" / "settings.json"
        self.local_settings = self.project_dir / "settings.json"
        self.host_path = self.env / "agent" / "memory" / "host-id.json"
        self.output = self.root / "snapshot export"
        self.memory = document()
        self.memory["entries"]["memory"]["fact"] = entry()
        self.write_json(self.memory_path, self.memory)
        self.write_json(self.host_path, {"id": HOST_ID})
        self.write_json(self.global_settings, {"memory": {"recall": True, "recallRerank": True},
                                              "unrelatedSetting": "DO-NOT-EXPORT-THIS-UNRELATED-VALUE"})
        self.write_json(self.local_settings, {"maxRecallEntries": 2})
        # These are synthetic poison files, not credentials or provider configuration.
        for path in (self.env / "agent" / "models.json", self.env / "agent" / "auth.json",
                     self.run / "env.jsonl", self.run / "run.jsonl", self.run / "manifest.json"):
            path.write_bytes(b"\xffDO-NOT-READ")

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))

    def save_memory(self):
        self.write_json(self.memory_path, self.memory)

    def args(self, *extra, output=None, mode=None):
        return ["snapshot", "--run-dir", str(self.run), "--run-id", "synthetic-run", "--env-id", "env_one",
                "--project-id", PROJECT_ID, "--code-revision", "synthetic-revision", "--capture-stage", "final_state",
                "--frozen", *(mode or ["--persisted-settings-for-current-replay"]),
                "--output-dir", str(output or self.output), *map(str, extra)]

    def call(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = exporter.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def capture(self, *extra, **kwargs):
        status, stdout, stderr = self.call(self.args(*extra, **kwargs))
        self.assertEqual(status, 0, stderr)
        self.assertIn("Exported snapshot", stdout)
        output = kwargs.get("output", self.output)
        fixture = json.loads((output / "fixture.json").read_bytes())
        provenance = json.loads((output / "provenance.json").read_bytes())
        return fixture, provenance

    def reject(self, pattern, *extra, args=None, **kwargs):
        status, stdout, stderr = self.call(args or self.args(*extra, **kwargs))
        self.assertEqual(status, 1, stdout)
        self.assertRegex(stderr, pattern)
        self.assertFalse((kwargs.get("output", self.output)).exists())
        return stderr

    def source(self, path=None, *, relative="note.txt", origin="file", expected=b"original", uri=None):
        path = path or self.workspace / "note.txt"
        value = {"id": "synthetic-source", "origin": origin, "sha256": digest(expected),
                 "uri": uri or path.as_uri()}
        if relative is not None:
            value["projectPath"] = relative
        return value

    def tree(self, root):
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def test_exact_probe_schema_and_native_run_layout_without_discovery(self):
        before = self.tree(self.run)
        fixture, provenance = self.capture("--project-root", "original-root-label-not-opened",
                                           "--project-alias", "path:original-synthetic-root")
        self.assertEqual(set(fixture), {"schema", "lineage", "project", "host_id", "memory",
                                       "global_memory_settings", "project_memory_settings", "global_harness",
                                       "session_harness", "shared_cache", "files"})
        self.assertEqual(fixture["schema"], "optimus-memory-recall-fixture/v1")
        self.assertEqual(set(fixture["lineage"]), {"run_id", "env_id", "capture_stage", "code_revision"})
        self.assertEqual(fixture["project"], {"id": PROJECT_ID, "root": "original-root-label-not-opened",
                                              "aliases": ["path:original-synthetic-root"]})
        self.assertEqual(fixture["host_id"], HOST_ID)
        self.assertEqual(fixture["memory"], self.memory)
        self.assertEqual(fixture["files"], [])
        self.assertEqual(fixture["global_memory_settings"], {"recall": True, "recallRerank": True})
        self.assertEqual(fixture["project_memory_settings"], {"maxRecallEntries": 2})
        self.assertEqual(provenance["selection"]["included_corpora"], ["project"])
        self.assertEqual(provenance["selection"]["omitted_corpora"], ["global", "session", "shared"])
        self.assertEqual(provenance["settings"]["authority"], "persisted_requested")
        self.assertEqual(provenance["settings"]["historical_runtime_status"], "historical_runtime_unverified")
        self.assertFalse(provenance["settings"]["historical_runtime_verified"])
        self.assertEqual(provenance["settings"]["probe_current_resolved"], "not_observed_by_exporter")
        self.assertFalse(provenance["historical_per_question_reconstruction"])
        self.assertEqual(before, self.tree(self.run))
        self.assertFalse((self.output / "queries.jsonl").exists())
        self.assertNotIn("DO-NOT-EXPORT-THIS-UNRELATED-VALUE", (self.output / "provenance.json").read_text())
        self.assertNotIn("models.json", json.dumps(provenance))
        self.assertNotIn("auth.json", json.dumps(provenance))
        for item in provenance["inputs"]:
            self.assertEqual(item["status"], "utf8")
            self.assertEqual(item["sha256"], digest(Path(item["path"]).read_bytes()))
        self.assertEqual(provenance["outputs"]["fixture.json"]["sha256"], digest((self.output / "fixture.json").read_bytes()))

    def test_repeated_export_is_byte_reproducible(self):
        self.capture()
        second = self.root / "second export"
        self.capture(output=second)
        self.assertEqual(self.tree(self.output), self.tree(second))

    def test_effective_override_is_complete_explicit_and_not_runtime_proof(self):
        settings = self.root / "observed settings.json"
        effective = full_settings(recall=False, recallRerank=False, maxRecallEntries=5)
        self.write_json(settings, effective)
        fixture, provenance = self.capture(mode=["--effective-memory-settings", str(settings),
                                                  "--settings-provenance", "synthetic same-cut observation"])
        self.assertEqual(fixture["global_memory_settings"], effective)
        self.assertEqual(fixture["project_memory_settings"], {})
        self.assertEqual(provenance["settings"]["authority"], "caller_effective")
        self.assertEqual(provenance["settings"]["caller_provenance"], "synthetic same-cut observation")
        self.assertEqual(provenance["settings"]["persisted_requested"]["project_memory"], {"maxRecallEntries": 2})
        self.assertFalse(provenance["settings"]["historical_runtime_verified"])
        self.assertEqual(provenance["settings"]["probe_current_resolved"], "not_observed_by_exporter")

    def test_effective_override_requires_provenance_and_both_feature_flags(self):
        settings = self.root / "effective.json"
        self.write_json(settings, full_settings())
        self.reject("settings-provenance", mode=["--effective-memory-settings", str(settings)])
        value = full_settings()
        del value["recallRerank"]
        self.write_json(settings, value)
        self.reject("complete", mode=["--effective-memory-settings", str(settings), "--settings-provenance", "synthetic"])

    def test_settings_flags_are_typed_and_limits_are_not_silently_defaulted(self):
        for invalid in ({"recall": 1}, {"maxRecallEntries": True}, {"maxRecallChars": -1},
                        {"maxRecallEntries": 51}, {"unknownFutureSetting": True},
                        {"shared": {"url": "https://user:password@invalid.test", "tokenFile": "not-read"}}):
            with self.subTest(invalid=invalid):
                self.write_json(self.local_settings, invalid)
                self.reject("boolean|invalid|unsupported|unsafe")

    def test_missing_settings_are_explicit_empty_memory_objects(self):
        self.global_settings.unlink()
        self.local_settings.unlink()
        fixture, provenance = self.capture()
        self.assertEqual(fixture["global_memory_settings"], {})
        self.assertEqual(fixture["project_memory_settings"], {})
        self.assertEqual(sum(row["status"] == "missing" for row in provenance["inputs"]), 2)

    def test_missing_required_memory_or_host_fails_without_initializing_a_store(self):
        self.memory_path.unlink()
        before = self.tree(self.run)
        self.reject("required input is missing")
        self.assertEqual(before, self.tree(self.run))
        self.save_memory()
        self.host_path.unlink()
        self.reject("required input is missing")
        self.assertFalse(self.host_path.exists())

    def test_null_corrupt_duplicate_nonfinite_and_non_utf8_json_fail_closed(self):
        for raw in (b"null", b"{broken", b'{"memory":{},"memory":{}}', b'{"memory":{"x":NaN}}', b"\xff"):
            with self.subTest(raw=raw):
                self.global_settings.write_bytes(raw)
                self.reject("null|invalid JSON|not UTF-8")

    def test_no_store_snapshot_fallback_or_cross_project_relabeling(self):
        self.write_json(self.memory_path, self.memory["entries"])
        self.reject("schema")
        self.memory["memory"]["projectId"] = "project_other"
        self.save_memory()
        self.reject("project/schema mismatch")
        self.memory["memory"]["projectId"] = PROJECT_ID
        self.memory["entries"]["memory"]["fact"]["metadata"]["projectId"] = "project_other"
        self.save_memory()
        self.reject("another project")

    def test_scope_inputs_are_opt_in_and_recorded_even_when_missing(self):
        global_path = self.env / "agent" / "harness" / "harness_state.json"
        global_path.parent.mkdir(parents=True)
        global_path.write_bytes(b"\xffnot-selected")
        fixture, provenance = self.capture()
        self.assertIsNone(fixture["global_harness"])
        self.assertFalse(any(row["role"] == "global_harness" for row in provenance["inputs"]))
        output = self.root / "included"
        self.reject("not UTF-8", "--include-scope", "global", output=output)
        global_path.unlink()
        fixture, provenance = self.capture("--include-scope", "global", "--include-scope", "session", output=output)
        self.assertIsNone(fixture["global_harness"])
        self.assertIsNone(fixture["session_harness"])
        self.assertEqual(provenance["selection"]["included_corpora"], ["project", "global", "session"])
        self.assertEqual(provenance["selection"]["omitted_corpora"], ["shared"])
        self.assertEqual([row["status"] for row in provenance["inputs"] if row["role"].endswith("_harness")],
                         ["missing", "missing"])

    def test_legacy_sparse_global_and_session_harnesses_remain_compatible(self):
        sparse = {"schema": 1, "entries": {"memory": {"legacy": entry("legacy")}}, "refinements": []}
        self.write_json(self.env / "agent" / "harness" / "harness_state.json", sparse)
        self.write_json(self.env / "session-artifacts" / "harness" / "harness_state.json", sparse)
        fixture, _ = self.capture("--include-scope", "global", "--include-scope", "session")
        self.assertEqual(fixture["global_harness"], sparse)
        self.assertEqual(fixture["session_harness"], sparse)
        self.assertEqual(fixture["files"], [])
        self.memory["entries"].pop("skill")
        self.save_memory()
        self.reject("all four kind buckets", output=self.root / "invalid-project")

    def test_selected_global_session_shared_states_are_preserved(self):
        global_state, session_state, shared_state = harness(), harness(), document()
        global_state["entries"]["memory"]["g"] = entry("g")
        session_state["entries"]["memory"]["s"] = entry("s")
        shared_state["entries"]["memory"]["remote"] = entry("remote")
        shared = {"schema": 1, "url": "https://synthetic.invalid", "revision": 2,
                  "state": shared_state, "pending": [], "connected": False}
        self.write_json(self.env / "agent" / "harness" / "harness_state.json", global_state)
        self.write_json(self.env / "session-artifacts" / "harness" / "harness_state.json", session_state)
        self.write_json(self.project_dir / "shared.json", shared)
        self.write_json(self.global_settings, {"memory": {"shared": {"url": shared["url"], "tokenFile": "never-open-token"}}})
        fixture, provenance = self.capture("--include-scope", "global", "--include-scope", "session",
                                           "--include-scope", "shared")
        self.assertEqual(fixture["global_harness"], global_state)
        self.assertEqual(fixture["session_harness"], session_state)
        self.assertEqual(fixture["shared_cache"], shared)
        self.assertEqual(provenance["selection"]["omitted_corpora"], [])
        self.assertFalse(any("token" in row["path"] for row in provenance["inputs"]))
        self.assertFalse((self.env / "agent" / "never-open-token").exists())

    def test_source_utf8_missing_stale_crlf_and_duplicate_locations(self):
        raw = "Frozen é 東京 🙂\r\nsecond line\r\n".encode("utf-8")
        (self.workspace / "note.txt").write_bytes(raw)
        source = self.source(expected=b"old-content")
        missing = self.source(self.workspace / "missing" / "note.txt", relative="missing/note.txt")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [source, missing]
        self.memory["entries"]["memory"]["duplicate"] = entry("duplicate", [source])
        self.save_memory()
        before = self.tree(self.run)
        fixture, provenance = self.capture("--allow-source-root", self.workspace)
        self.assertEqual(len(fixture["files"]), 2)
        by_relative = {item["projectPath"]: item for item in fixture["files"]}
        self.assertEqual(by_relative["note.txt"]["content"], raw.decode("utf-8"))
        self.assertIsNone(by_relative["missing/note.txt"]["content"])
        self.assertEqual(set(by_relative["note.txt"]), {"uri", "projectPath", "content"})
        self.assertEqual(len(provenance["sources"]), 3)
        present = [item for item in provenance["sources"] if item["status"] == "utf8"]
        self.assertTrue(all(item["input_sha256"] == digest(raw) for item in present))
        self.assertTrue(all(item["recorded_sha256"] == digest(b"old-content") for item in present))
        self.assertNotIn("freshness", json.dumps(provenance["sources"]))
        self.assertEqual(before, self.tree(self.run))
        self.assertFalse((self.workspace / "missing").exists())

    def test_file_uri_without_project_path_reads_only_explicit_allowed_root(self):
        external = self.root / "selected evidence"
        external.mkdir()
        source_path = external / "snow 東京 #.txt"
        source_path.write_bytes(b"selected current bytes")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(source_path, relative=None)]
        self.save_memory()
        self.reject("outside allowed roots", "--allow-source-root", self.workspace)
        fixture, _ = self.capture("--allow-source-root", external)
        self.assertEqual(fixture["files"], [{"uri": source_path.as_uri(), "content": "selected current bytes"}])

    def test_project_path_maps_to_frozen_workspace_not_original_uri_or_root_label(self):
        (self.workspace / "note.txt").write_text("frozen copy", encoding="utf-8")
        original = self.root / "original-not-authorized" / "note.txt"
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(original)]
        self.save_memory()
        fixture, _ = self.capture("--allow-source-root", self.workspace, "--project-root", str(original.parent))
        self.assertEqual(fixture["files"][0]["content"], "frozen copy")
        self.assertEqual(fixture["files"][0]["uri"], original.as_uri())
        self.assertFalse(original.parent.exists())

    def test_explicit_source_project_root_override(self):
        alternate = self.root / "relocated sources"
        alternate.mkdir()
        (alternate / "note.txt").write_text("relocated frozen bytes", encoding="utf-8")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        fixture, provenance = self.capture("--source-project-root", alternate, "--allow-source-root", alternate)
        self.assertEqual(fixture["files"][0]["content"], "relocated frozen bytes")
        self.assertEqual(provenance["selection"]["source_project_root"], str(alternate))

    def test_no_implicit_source_allowroot_even_for_missing_files(self):
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        self.reject("outside allowed roots")
        fixture, _ = self.capture("--allow-source-root", self.workspace)
        self.assertIsNone(fixture["files"][0]["content"])

    def test_non_file_origins_and_host_project_invisible_refs_are_not_opened(self):
        source = self.source(self.env / "agent" / "models.json", relative=None, origin="user")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [source]
        hidden = self.source(self.root / "outside-not-allowed.txt", relative=None)
        self.memory["entries"]["memory"]["other_host"] = entry("other_host", [hidden], hostId="other-host")
        self.save_memory()
        global_state = harness()
        global_state["entries"]["memory"]["other_project"] = entry("other_project", [hidden], projectId="project_other")
        self.write_json(self.env / "agent" / "harness" / "harness_state.json", global_state)
        fixture, provenance = self.capture("--include-scope", "global")
        self.assertEqual(fixture["files"], [])
        self.assertEqual(len(provenance["sources"]), 1)
        self.assertEqual(provenance["sources"][0]["status"], "not_file_backed_native_unknown")

    def test_unsafe_project_paths_are_rejected_before_any_source_read(self):
        for relative in ("../outside", "nested/../../outside", "/absolute", "C:/outside", "a\\b", "./a", "a//b", ""):
            with self.subTest(relative=relative):
                self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(relative=relative)]
                self.save_memory()
                self.reject("projectPath", "--allow-source-root", self.workspace)

    def test_traversal_encoded_remote_and_unsupported_file_urls_fail_closed(self):
        base = self.workspace.as_uri()
        for uri in (base + "/../outside.txt", base + "/%2e%2e/outside.txt", base + "/%00name",
                    "file://remote-server/share/note.txt", "file:relative", base + "/note?secret=x",
                    base + "/note#fragment", base + "/%GG", base + "/%FF", base + "/line\nname"):
            with self.subTest(uri=uri):
                self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(uri=uri, relative=None)]
                self.save_memory()
                self.reject("traversal|URI", "--allow-source-root", self.workspace)

    def test_environment_and_project_identifiers_cannot_escape(self):
        for flag, value in (("--env-id", "../env_one"), ("--env-id", "env_one/../env_one"),
                            ("--project-id", "project_fixture/../../other")):
            with self.subTest(flag=flag, value=value):
                args = self.args()
                args[args.index(flag) + 1] = value
                self.reject("unsafe|invalid", args=args)

    def test_sensitive_file_sources_are_refused_even_under_allowed_roots(self):
        for name in ("models.json", "auth.json", "settings.json", ".env", ".env.local", "credentials.json", "private.key"):
            with self.subTest(name=name):
                source = self.source(self.workspace / name, relative=name)
                self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [source]
                self.save_memory()
                self.reject("sensitive source", "--allow-source-root", self.workspace)

    def test_configured_shared_token_path_is_never_read_as_source(self):
        token = self.workspace / "custom-auth-name"
        token.write_bytes(b"synthetic-secret-not-to-read")
        self.write_json(self.global_settings, {"memory": {"shared": {"url": "https://synthetic.invalid", "tokenFile": str(token)}}})
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(token, relative=None)]
        self.save_memory()
        self.reject("sensitive source", "--allow-source-root", self.workspace)

    def test_non_utf8_oversize_and_non_regular_sources_are_not_exported_as_missing(self):
        path = self.workspace / "note.txt"
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        path.write_bytes(b"\xff")
        self.reject("not UTF-8", "--allow-source-root", self.workspace)
        path.write_bytes(b"12345")
        self.reject("oversized", "--allow-source-root", self.workspace, "--max-source-bytes", 4)
        path.unlink()
        path.mkdir()
        self.reject("not a regular file", "--allow-source-root", self.workspace)

    def test_unreadable_source_is_not_mislabeled_missing(self):
        path = self.workspace / "note.txt"
        path.write_bytes(b"synthetic")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        real_open = exporter.os.open
        def deny_source(value, flags, *args, **kwargs):
            if Path(value) == path:
                raise PermissionError("synthetic denied read")
            return real_open(value, flags, *args, **kwargs)
        with mock.patch.object(exporter.os, "open", side_effect=deny_source):
            self.reject("unreadable", "--allow-source-root", self.workspace)

    def test_parent_not_directory_is_not_mislabeled_missing(self):
        (self.workspace / "parent").write_bytes(b"not a directory")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(relative="parent/child.txt")]
        self.save_memory()
        self.reject("parent is not a directory", "--allow-source-root", self.workspace)

    def test_symlink_file_parent_and_allowroot_are_refused(self):
        external = self.root / "outside"
        external.mkdir()
        target = external / "note.txt"
        target.write_bytes(b"must not read through link")
        link = self.workspace / "note.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        self.reject("symlink/reparse", "--allow-source-root", self.workspace)
        link.unlink()
        parent_link = self.workspace / "linked"
        parent_link.symlink_to(external, target_is_directory=True)
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source(relative="linked/note.txt")]
        self.save_memory()
        self.reject("symlink/reparse", "--allow-source-root", self.workspace)
        self.reject("symlink/reparse", "--allow-source-root", parent_link)

    def test_hardlinked_source_cannot_hide_an_outside_root_target(self):
        target = self.root / "outside-source.txt"
        target.write_bytes(b"synthetic outside source")
        link = self.workspace / "note.txt"
        try:
            os.link(target, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"hardlink creation unavailable: {error}")
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        self.reject("hard-linked source", "--allow-source-root", self.workspace)
        self.assertEqual(target.read_bytes(), b"synthetic outside source")

    def test_reparse_attribute_is_refused_without_following_target(self):
        actual_lstat = Path.lstat
        target = self.workspace / "note.txt"
        target.write_bytes(b"synthetic")
        def fake_lstat(path, *args, **kwargs):
            value = actual_lstat(path, *args, **kwargs)
            if path == target:
                info = mock.Mock(wraps=value)
                info.st_mode = stat.S_IFREG | 0o600
                info.st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                return info
            return value
        with mock.patch.object(Path, "lstat", fake_lstat):
            with self.assertRaisesRegex(exporter.ExportError, "symlink/reparse"):
                exporter.checked_stat(target)

    def test_metadata_source_and_total_size_limits(self):
        self.reject("oversized", "--max-input-bytes", 8)
        self.reject("total input byte limit", "--max-total-bytes", 8)
        self.reject("source limit", "--max-source-bytes", exporter.SOURCE_LIMIT + 1)
        self.reject("byte limits", "--max-total-bytes", 0)

    def test_missing_freeze_ack_and_settings_authority_are_errors(self):
        args = self.args()
        args.remove("--frozen")
        self.reject("--frozen", args=args)
        args = self.args()
        args.remove("--persisted-settings-for-current-replay")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            exporter.main(args)
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_outputs_never_overwrite_files_or_contaminate_input_roots(self):
        for output in (self.run / "export", self.workspace / "export", self.memory_path):
            with self.subTest(output=output):
                before = self.tree(self.run)
                status, _, stderr = self.call(self.args(output=output))
                self.assertEqual(status, 1)
                self.assertIn("outside input/source roots", stderr)
                self.assertEqual(before, self.tree(self.run))
        self.output.mkdir()
        marker = self.output / "fixture.json"
        marker.write_bytes(b"existing output must survive")
        status, _, stderr = self.call(self.args())
        self.assertEqual(status, 1)
        self.assertIn("already exists", stderr)
        self.assertEqual(marker.read_bytes(), b"existing output must survive")

    def test_ordinary_input_mutation_aborts_and_rolls_back_only_new_bundle(self):
        actual_verify = exporter.Reader.verify
        calls = []
        def mutate_then_verify(reader):
            calls.append(True)
            if len(calls) == 2:
                self.write_json(self.local_settings, {"recall": False, "maxRecallChars": 1000})
            return actual_verify(reader)
        with mock.patch.object(exporter.Reader, "verify", mutate_then_verify):
            self.reject("input changed since capture")
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(self.local_settings.read_bytes())["recall"], False)

    def test_missing_source_created_during_export_aborts(self):
        self.memory["entries"]["memory"]["fact"]["metadata"]["sources"] = [self.source()]
        self.save_memory()
        actual_verify = exporter.Reader.verify
        def create_then_verify(reader):
            (self.workspace / "note.txt").write_bytes(b"created after capture")
            return actual_verify(reader)
        with mock.patch.object(exporter.Reader, "verify", create_then_verify):
            self.reject("input changed since capture", "--allow-source-root", self.workspace)

    def test_output_write_failure_removes_only_new_outputs(self):
        actual_open = Path.open
        def failing_open(path, mode="r", *args, **kwargs):
            if path == self.output / "provenance.json" and mode == "xb":
                raise PermissionError("synthetic output failure")
            return actual_open(path, mode, *args, **kwargs)
        before = self.tree(self.run)
        with mock.patch.object(Path, "open", failing_open):
            self.reject("synthetic output failure")
        self.assertEqual(before, self.tree(self.run))


class QueryExportTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="memory-query-export-synthetic-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.input = self.root / "explicit.jsonl"
        self.output = self.root / "query export"

    def write(self, rows):
        self.input.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))

    def call(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = exporter.main(["queries", "--input", str(self.input), "--output-dir", str(self.output), *extra])
        return result, stderr.getvalue()

    def reject(self, pattern, *extra):
        result, error = self.call(*extra)
        self.assertEqual(result, 1)
        self.assertRegex(error, pattern)
        self.assertFalse(self.output.exists())

    def test_independent_queries_strip_gold_reference_evidence_and_answers(self):
        raw_question = "Exact user text: 東京?\nDo not normalize this."
        self.write([{"qid": "q1", "question": raw_question, "gold": "GOLD-SENTINEL", "answer_text": "ANSWER-SENTINEL",
                     "reference_answer": "REFERENCE-SENTINEL", "ground_truth_refs": ["EVIDENCE-SENTINEL"],
                     "eval": {"answer": "NESTED-GOLD-SENTINEL"}, "answer_instruction": "DO-NOT-PREPEND"},
                    {"qid": "q1", "variant": "budget", "query": "", "options": {"max_recall_chars": 800,
                                                                                      "max_recall_entries": 1,
                                                                                      "scope": "project"}}])
        before = self.input.read_bytes()
        result, error = self.call()
        self.assertEqual(result, 0, error)
        self.assertEqual(before, self.input.read_bytes())
        raw = (self.output / "queries.jsonl").read_bytes()
        queries = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(queries[0], {"qid": "q1", "variant": "base", "query": raw_question})
        self.assertEqual(set(queries[1]), {"qid", "variant", "query", "options"})
        self.assertEqual(queries[1]["query"], "")
        self.assertFalse((self.output / "fixture.json").exists())
        provenance = json.loads((self.output / "provenance.json").read_bytes())
        all_output = raw + (self.output / "provenance.json").read_bytes()
        for sentinel in (b"GOLD-SENTINEL", b"ANSWER-SENTINEL", b"REFERENCE-SENTINEL", b"EVIDENCE-SENTINEL", b"DO-NOT-PREPEND"):
            self.assertNotIn(sentinel, all_output)
        self.assertEqual(provenance["inputs"][0]["sha256"], digest(before))
        self.assertEqual(provenance["outputs"]["queries.jsonl"]["sha256"], digest(raw))
        self.assertEqual(provenance["lines"][0]["input_line_sha256"], digest(before.splitlines()[0]))
        self.assertEqual(provenance["lines"][0]["output_line_sha256"], digest(raw.splitlines()[0]))

    def test_benchmark_rows_require_exact_explicit_run_environment_selection(self):
        self.write([{"run_id": "r1", "env_id": "e1", "qid": "q", "question": "Selected user question", "answer": "gold"},
                    {"run_id": "r1", "env_id": "e2", "qid": "q", "question": "Other environment", "answer": "gold"},
                    {"run_id": "r2", "env_id": "e1", "qid": "q", "question": "Other run", "answer": "gold"}])
        self.reject("--run-id")
        self.reject("--env-id", "--run-id", "r1")
        result, error = self.call("--run-id", "r1", "--env-id", "e1")
        self.assertEqual(result, 0, error)
        rows = (self.output / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["query"], "Selected user question")
        self.assertNotIn("answer", rows[0])

    def test_duplicate_qid_variant_blank_id_and_ambiguous_text_rejected(self):
        for rows in ([{"qid": "q", "query": "a"}, {"qid": "q", "query": "b"}],
                     [{"qid": "", "query": "a"}], [{"qid": "q", "query": "a", "question": "b"}],
                     [{"qid": "q", "query": {"gold": "answer"}}], [{"qid": "q", "query": "a", "variant": ""}]):
            with self.subTest(rows=rows):
                self.write(rows)
                self.reject("duplicate|nonempty|ambiguous|requires a query")

    def test_unknown_or_gold_options_and_invalid_limits_are_rejected(self):
        for options in ({"answer": "gold"}, {"rerank": True}, {"max_recall_chars": -1},
                        {"max_recall_entries": 51}, {"max_recall_entries": 1.5},
                        {"max_recall_entries": True}, {"recall": 1}, {"include_inactive": None}, {"scope": "other"}):
            with self.subTest(options=options):
                self.write([{"qid": "q", "query": "needle", "options": options}])
                self.reject("options|invalid query")

    def test_unicode_line_separators_inside_query_do_not_split_jsonl_records(self):
        query = "Exact Unicode \u2028 line separator and \u0085 next-line char."
        self.write([{"qid": "unicode", "query": query}])
        result, error = self.call()
        self.assertEqual(result, 0, error)
        raw = (self.output / "queries.jsonl").read_bytes()
        self.assertEqual(len(raw.split(b"\n")), 2)
        self.assertEqual(json.loads(raw)["query"], query)

    def test_query_nullable_native_options_and_blank_lines_are_preserved(self):
        row = {"qid": "q", "query": "a", "options": {"scope": None, "recall": None, "max_recall_chars": None,
                                                          "max_recall_entries": 0, "include_inactive": True}}
        self.input.write_bytes(b"\r\n" + json.dumps(row).encode() + b"\r\n\r\n")
        result, error = self.call()
        self.assertEqual(result, 0, error)
        exported = json.loads((self.output / "queries.jsonl").read_bytes())
        self.assertEqual(exported["options"], row["options"])
        provenance = json.loads((self.output / "provenance.json").read_bytes())
        self.assertEqual(provenance["lines"][0]["input_line"], 2)

    def test_query_malformed_duplicate_nonfinite_or_empty_input_fails_without_echoing_gold(self):
        for raw in (b'{"qid":"q","query":"a","gold":"DO-NOT-ECHO"',
                    b'{"qid":"q","query":"a","query":"b"}',
                    b'{"qid":"q","query":"a","gold":NaN}',
                    b'{"qid":"q","query":"a","gold":1e999}', b"\xff", b"\n\n"):
            with self.subTest(raw=raw):
                self.input.write_bytes(raw)
                status, error = self.call()
                self.assertEqual(status, 1)
                self.assertNotIn("DO-NOT-ECHO", error)
                self.assertFalse(self.output.exists())

    def test_no_selected_query_or_missing_identity_is_an_error(self):
        self.write([{"qid": "q", "query": "a", "run_id": "r", "env_id": "e"}])
        self.reject("no selected queries", "--run-id", "other", "--env-id", "e")
        self.write([{"qid": "q", "query": "a"}])
        self.reject("nonempty", "--run-id", "r", "--env-id", "e")

    def test_query_output_never_overwrites_input_or_existing_bundle(self):
        self.write([{"qid": "q", "query": "a"}])
        before = self.input.read_bytes()
        self.output = self.input
        status, error = self.call()
        self.assertEqual(status, 1)
        self.assertIn("outside input/source roots", error)
        self.assertEqual(self.input.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
