"""Offline tests for native bundles and safe installation activation."""

import importlib.util
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


class NativeReleaseTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='optimus-release-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.binary = self.root / release.EXECUTABLE
        self.binary.write_bytes(b'synthetic native executable')
        self.prefix = self.root / 'prefix with spaces'
        self.bin_dir = self.root / 'bin with spaces'

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


if __name__ == '__main__':
    unittest.main()
