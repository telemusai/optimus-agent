"""Offline tests for native bundles and safe installation activation."""

import importlib.util
import json
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rust_release', ROOT / 'scripts/rust_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
NATIVE_QUERY = release.query_build_provenance


def fixture_provenance(source=ROOT):
    target = ("x86_64-pc-windows-msvc" if sys.platform == "win32" else
              "x86_64-apple-darwin" if sys.platform == "darwin" else "x86_64-unknown-linux-gnu")
    receipt = {"schema": release.PROVENANCE_SCHEMA,
               "sourceTreeSha256": release.raw_aggregate(source, release.source_files(source)),
               "payloadSourceSha256": release.raw_aggregate(source, release.payload_files(source)),
               "runtimeSourceSha256": release.runtime_source_sha256(source),
               "version": json.loads((source / "resources/agent/package.json").read_text())["version"],
               "target": target, "profile": "release",
               "rustc": "rustc offline-fixture", "buildOptionsSha256": "1" * 64}
    receipt["buildFingerprint"] = release.build_fingerprint(receipt)
    return receipt


class NativeReleaseTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='optimus-release-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.binary = self.root / release.EXECUTABLE
        self.binary.write_bytes(b'synthetic native executable')
        self.prefix = self.root / 'prefix with spaces'
        self.bin_dir = self.root / 'bin with spaces'
        self.provenance = fixture_provenance()
        patcher = mock.patch.object(release, 'query_build_provenance', return_value=self.provenance)
        self.query = patcher.start()
        self.addCleanup(patcher.stop)

    def stage(self, destination):
        release.stage_release(ROOT, self.binary, destination, 'abcdef0123456789')

    def install(self):
        with mock.patch.object(release.shutil, 'which', return_value='/test/uv'), \
             mock.patch.object(release, 'git_bash', return_value=Path('C:/Program Files/Git/bin/bash.exe')), \
             mock.patch.object(release.subprocess, 'run') as run:
            result = release.install_release(ROOT, self.binary, self.prefix, self.bin_dir, 'abcdef0123456789')
            self.assertEqual(run.call_args.args[0][-1], '--version')
            return result

    def test_bundle_contains_runtime_skills_and_export_assets(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        for relative in ['bin/' + release.EXECUTABLE, 'resources/agent/package.json',
                         'resources/agent/skills/edit/pyproject.toml',
                         'resources/agent/src/core/export-html/template.js',
                         'prime-agent-runtime/src/rlm/repl.py', 'scripts/launch-with-jev-env.py']:
            self.assertTrue((bundle / relative).is_file(), relative)
        self.assertFalse((bundle / 'packages').exists())
        self.assertFalse((bundle / 'package.json').exists())
        self.assertFalse(list(bundle.rglob('*.ts')))
        self.assertFalse(list(bundle.rglob('__pycache__')))

    def test_stage_refuses_overwrite(self):
        self.stage(self.root / 'bundle')
        with self.assertRaises(FileExistsError):
            self.stage(self.root / 'bundle')
        self.assertTrue((self.root / 'bundle/COMMIT').is_file())

    def test_missing_binary_does_not_create_release(self):
        with self.assertRaises(ValueError):
            release.stage_release(ROOT, self.root / 'missing', self.root / 'bundle', 'abcdef0')
        self.assertFalse((self.root / 'bundle').exists())

    def test_installs_and_retains_previous_release(self):
        first = self.install()
        second = self.install()
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        if os.name == 'nt':
            self.assertEqual((self.prefix / 'current.txt').read_text().strip(), second.name)
        else:
            self.assertEqual((self.prefix / 'current').resolve(), second)
        self.assertEqual((self.bin_dir / 'optimus-agent').read_bytes(), (ROOT / 'scripts/optimus-agent').read_bytes())

    def test_failed_version_check_leaves_install_untouched(self):
        first = self.install()
        launcher = self.bin_dir / 'optimus-agent'
        launcher.write_text('original launcher')
        with mock.patch.object(release.shutil, 'which', return_value='/test/uv'), \
             mock.patch.object(release, 'git_bash', return_value=Path('C:/Program Files/Git/bin/bash.exe')), \
             mock.patch.object(release.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'probe')):
            with self.assertRaises(subprocess.CalledProcessError):
                release.install_release(ROOT, self.binary, self.prefix, self.bin_dir, 'abcdef0123456789')
        self.assertEqual(launcher.read_text(), 'original launcher')
        self.assertEqual(list((self.prefix / 'releases').iterdir()), [first])

    def test_failed_activation_restores_previous_launcher(self):
        first = self.install()
        launcher = self.bin_dir / 'optimus-agent'
        launcher.write_text('original launcher')
        replace = os.replace
        pointer = self.prefix / ('current.txt' if os.name == 'nt' else 'current')
        def fail_pointer(source, destination):
            if destination == pointer:
                raise OSError('synthetic activation failure')
            return replace(source, destination)
        with mock.patch.object(release.os, 'replace', side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, 'synthetic activation failure'):
                self.install()
        self.assertEqual(launcher.read_text(), 'original launcher')
        self.assertEqual(list((self.prefix / 'releases').iterdir()), [first])
        self.assertFalse(list(self.prefix.glob('.current-*')))
        if os.name == 'nt':
            self.assertEqual(pointer.read_text().strip(), first.name)
        else:
            self.assertEqual(pointer.resolve(), first)

    def test_windows_launchers_activate_and_roll_back_together(self):
        with mock.patch.object(release, 'WINDOWS', True):
            first = self.install()
            shell = self.bin_dir / 'optimus-agent'
            command = self.bin_dir / 'optimus-agent.cmd'
            self.assertIn('Program Files/Git/bin/bash.exe', command.read_text())
            self.assertIn(' %*', command.read_text())
            shell.write_text('old shell launcher')
            command.write_text('old Windows launcher')
            replace = os.replace
            def fail_pointer(source, destination):
                if destination == self.prefix / 'current.txt':
                    raise OSError('pointer locked')
                return replace(source, destination)
            with mock.patch.object(release.os, 'replace', side_effect=fail_pointer):
                with self.assertRaisesRegex(OSError, 'pointer locked'):
                    self.install()
            self.assertEqual(shell.read_text(), 'old shell launcher')
            self.assertEqual(command.read_text(), 'old Windows launcher')
            self.assertEqual((self.prefix / 'current.txt').read_text().strip(), first.name)
            self.assertEqual(list((self.prefix / 'releases').iterdir()), [first])

    def test_locked_windows_launcher_does_not_activate_or_leave_partial_update(self):
        with mock.patch.object(release, 'WINDOWS', True):
            first = self.install()
            shell = self.bin_dir / 'optimus-agent'
            command = self.bin_dir / 'optimus-agent.cmd'
            shell.write_text('old shell launcher')
            old_command = command.read_bytes()
            replace = os.replace
            def fail_command(source, destination):
                if destination == command:
                    raise PermissionError('command locked')
                return replace(source, destination)
            with mock.patch.object(release.os, 'replace', side_effect=fail_command):
                with self.assertRaises(PermissionError):
                    self.install()
            self.assertEqual(shell.read_text(), 'old shell launcher')
            self.assertEqual(command.read_bytes(), old_command)
            self.assertEqual(list((self.prefix / 'releases').iterdir()), [first])

    def test_windows_launcher_quotes_spaces_and_escapes_percent_paths(self):
        script = release.windows_launcher(Path('C:/Program Files/Git/bash.exe'),
                                          Path('C:/Users/100% real/bin/optimus-agent'))
        self.assertIn('"C:/Program Files/Git/bash.exe"', script)
        self.assertIn('"C:/Users/100%% real/bin/optimus-agent" %*', script)
        self.assertIn('DisableDelayedExpansion', script)

    def test_git_bash_discovery_uses_git_installation(self):
        git = self.root / 'Git installation/cmd/git.exe'
        bash = self.root / 'Git installation/bin/bash.exe'
        git.parent.mkdir(parents=True)
        bash.parent.mkdir(parents=True)
        git.touch()
        bash.touch()
        with mock.patch.object(release.shutil, 'which', return_value=str(git)):
            self.assertEqual(release.git_bash(), bash)


    def test_receipt_is_embedded_query_bound_and_full_bundle_reinstalls(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        self.assertEqual(json.loads((bundle / 'BUILD-PROVENANCE.json').read_text()), self.provenance)
        self.assertEqual(release.verify_build_provenance(bundle, bundle / 'bin' / release.EXECUTABLE), self.provenance)
        copied = self.root / 'second portable bundle'
        release.stage_release(bundle, bundle / 'bin' / release.EXECUTABLE, copied, 'abcdef0')
        self.assertTrue((copied / 'prime-agent-runtime/test/test_lifecycle.py').is_file())
        self.assertEqual(self.query.call_args.args[0], copied / 'bin' / release.EXECUTABLE)

    def test_missing_null_malformed_pins_and_wrong_native_source_fail_before_copy(self):
        for field in ('buildFingerprint', 'runtimeSourceSha256', 'sourceTreeSha256', 'payloadSourceSha256'):
            for invalid in (None, '', 'a' * 63, 'A' * 64):
                with self.subTest(field=field, invalid=invalid):
                    receipt = dict(self.provenance, **{field: invalid})
                    self.query.return_value = receipt
                    with self.assertRaises(ValueError):
                        self.stage(self.root / 'bad')
                    self.assertFalse((self.root / 'bad').exists())
        for field, bad in [('sourceTreeSha256', 'a' * 64), ('runtimeSourceSha256', 'b' * 64),
                           ('payloadSourceSha256', 'c' * 64), ('profile', 'debug'), ('target', 'unknown-platform'),
                           ('version', '0.0.0'), ('rustc', 'not-a-rust-compiler')]:
            with self.subTest(field=field):
                receipt = dict(self.provenance, **{field: bad})
                receipt['buildFingerprint'] = release.build_fingerprint(receipt)
                self.query.return_value = receipt
                with self.assertRaises(ValueError):
                    self.stage(self.root / 'bad')
                self.assertFalse((self.root / 'bad').exists())

    def test_sidecar_cannot_override_a_different_or_unpinned_binary(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        self.query.return_value = dict(self.provenance, buildFingerprint=None)
        with self.assertRaisesRegex(ValueError, 'not pinned'):
            release.verify_build_provenance(bundle, self.binary)
        self.query.return_value = self.provenance
        receipt = dict(self.provenance, target='forged-target')
        (bundle / 'BUILD-PROVENANCE.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'does not match the native binary'):
            release.verify_build_provenance(bundle, self.binary)
        (bundle / 'BUILD-PROVENANCE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'does not match the native binary'):
            release.verify_build_provenance(bundle, self.binary)

    def test_runtime_mutation_addition_and_omission_are_rejected(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        for action in ('mutate', 'add', 'remove'):
            with self.subTest(action=action):
                runtime = bundle / 'prime-agent-runtime/src/rlm'
                path = runtime / ('added.py' if action == 'add' else 'lifecycle.py')
                original = path.read_bytes() if path.exists() else None
                if action == 'remove':
                    path.unlink()
                else:
                    path.write_bytes(b'# changed raw runtime\r\n')
                with self.assertRaisesRegex(ValueError, 'Runtime|runtime'):
                    release.verify_build_provenance(bundle, self.binary)
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_resource_mutation_and_added_file_are_rejected(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        resource = bundle / 'resources/agent/docs/keybindings.md'
        original = resource.read_bytes()
        resource.write_bytes(original + b'\nchanged')
        with self.assertRaisesRegex(ValueError, 'resources/runtime'):
            release.verify_build_provenance(bundle, self.binary)
        resource.write_bytes(original)
        extra = bundle / 'resources/extra.json'
        extra.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'resources/runtime'):
            release.verify_build_provenance(bundle, self.binary)
        extra.unlink()
        release.verify_build_provenance(bundle, self.binary)

    def test_mutation_during_copy_cleans_partial_stage(self):
        copytree = release.shutil.copytree
        def corrupt(source, destination, *args, **kwargs):
            result = copytree(source, destination, *args, **kwargs)
            if Path(destination).name == 'prime-agent-runtime':
                (Path(destination) / 'src/rlm/lifecycle.py').write_text('# synthetic corruption')
            return result
        with mock.patch.object(release.shutil, 'copytree', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'Runtime source'):
                self.stage(self.root / 'partial')
        self.assertFalse((self.root / 'partial').exists())

    def test_raw_runtime_hash_matches_normal_public_wrapper(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'prime-agent-runtime/src'), PYTHONDONTWRITEBYTECODE='1')
        code = ('import asyncio,json,rlm.lifecycle as lifecycle\n'
                'report=json.loads(' + repr(json.dumps({
                    'schema': 'optimus.native-lifecycle.v1', 'capability': 'rlm.stop-retain.v1',
                    'supported': False, 'targetProfile': {'model': 'offline/unproven'},
                    'provenance': {'hostImplementation': 'optimus-rust', 'protocolVersion': 7, 'schemaRevision': 32,
                                   'buildFingerprint': self.provenance['buildFingerprint'],
                                   'runtimeSourceSha256': self.provenance['runtimeSourceSha256']},
                    'activeOnlyMessages': {'supported': True}, 'scriptReports': {'supported': True},
                    'auditResume': {'supported': False}})) + ')\n'
                'async def request(*args): return report\n'
                'lifecycle.host_request=request\n'
                'actual=asyncio.run(lifecycle.lifecycle_capabilities(model="offline/unproven"))\n'
                'assert actual["supported"] is False and actual["auditResume"]["supported"] is False\n'
                'assert actual["activeOnlyMessages"]["supported"] is True\n'
                'print(lifecycle.runtime_source_sha256())')
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(result.stdout.strip(), self.provenance['runtimeSourceSha256'])

    def test_embedded_query_is_bounded_and_rejects_non_json_or_nonobject_output(self):
        for stdout in ('not JSON', 'null', '[]'):
            with mock.patch.object(release.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout)):
                with self.assertRaises(ValueError):
                    NATIVE_QUERY(self.binary)
        with mock.patch.object(release.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(self.provenance))) as run:
            self.assertEqual(NATIVE_QUERY(self.binary), self.provenance)
            self.assertEqual(run.call_args.args[0], [str(self.binary.resolve()), '--build-provenance'])
            self.assertEqual(run.call_args.kwargs['timeout'], 15)

    def test_top_level_payload_symlinks_are_rejected(self):
        bundle = self.root / 'bundle'
        self.stage(bundle)
        script = bundle / 'scripts/optimus-agent'
        script.unlink()
        try:
            script.symlink_to(ROOT / 'scripts/optimus-agent')
        except OSError as error:
            self.skipTest(f'symlink privilege unavailable: {error}')
        with self.assertRaisesRegex(ValueError, 'regular file'):
            release.verify_build_provenance(bundle, self.binary)


if __name__ == '__main__':
    unittest.main()
