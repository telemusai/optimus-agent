"""Offline stable-tag selection and installer failure/success coverage."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('standalone_install', ROOT / 'installers/install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
COMMIT = 'a' * 40


class TagTests(unittest.TestCase):
    def test_numeric_version_order_excludes_prereleases_and_other_refs(self):
        refs = '\n'.join(f'{COMMIT}\trefs/tags/{tag}' for tag in
                         ['v0.9.9', 'v0.10.1', 'v0.10.0', 'v1.0.0-rc.1', 'nightly', 'v00.11.0'])
        self.assertEqual(installer.latest_tag(refs), ('v0.10.1', COMMIT))

    def test_annotated_tag_resolves_to_peeled_commit(self):
        refs = f'{"b" * 40}\trefs/tags/v0.1.1\n{COMMIT}\trefs/tags/v0.1.1^{{}}\n'
        self.assertEqual(installer.latest_tag(refs), ('v0.1.1', COMMIT))

    def test_no_stable_tag_fails_closed(self):
        for refs in ['', f'{COMMIT}\trefs/tags/v1.0.0-beta', 'garbage refs/tags/v1.0.0']:
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                installer.latest_tag(refs)


@unittest.skipIf(os.name == 'nt', 'POSIX shell entry point')
class ShellBootstrapTests(unittest.TestCase):
    def test_download_execution_argument_forwarding_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='optimus-bootstrap-') as temporary:
            root = Path(temporary)
            commands = root / 'commands'
            commands.mkdir()
            download = commands / 'curl'
            download.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
assert 'https://telemus.ai/optimus-agent/install.py' in sys.argv
output = Path(sys.argv[sys.argv.index('-o') + 1])
output.write_text('import json, os, sys; from pathlib import Path; Path(os.environ["MARKER"]).write_text(json.dumps(sys.argv[1:])); sys.exit(int(os.environ["FAIL"]))')
''')
            download.chmod(0o755)
            for name in ['cargo', 'git', 'uv']:
                command = commands / name
                command.write_text('#!/bin/sh\nexit 0\n')
                command.chmod(0o755)
            build_tmp = root / 'temp with spaces'
            build_tmp.mkdir()
            marker = root / 'marker'
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       TMPDIR=str(build_tmp), MARKER=str(marker))
            for failure in ['0', '42']:
                env['FAIL'] = failure
                result = subprocess.run(['sh', str(ROOT / 'installers/install.sh'), '--prefix',
                                         '/example/path with spaces'], env=env, capture_output=True)
                self.assertEqual(result.returncode, int(failure), result.stderr)
                self.assertEqual(json.loads(marker.read_text()), ['--prefix', '/example/path with spaces'])
                self.assertEqual(list(build_tmp.iterdir()), [])


@unittest.skipIf(os.name == 'nt', 'POSIX fixture executable; Windows activation has separate tests')
class SourceInstallTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='optimus-source-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.prefix = self.root / 'prefix with spaces'
        self.bin_dir = self.root / 'bin with spaces'
        self.repo = self.root / 'tagged repository'
        self.repo.mkdir()
        for name in ['resources/agent', 'prime-agent-runtime', 'scripts']:
            (self.repo / name).mkdir(parents=True)
        for name in ['optimus-agent', 'launch-with-jev-env.py', 'rust_release.py']:
            shutil.copy2(ROOT / 'scripts' / name, self.repo / 'scripts' / name)
        for name in ['install.sh', 'LICENSE', 'README.md']:
            shutil.copy2(ROOT / name, self.repo / name)
        (self.repo / 'resources/agent/package.json').write_text('{"version":"0.1.1"}')
        (self.repo / 'prime-agent-runtime/pyproject.toml').write_text('[project]\nversion = "0.1.1"\n')
        runtime = self.repo / 'prime-agent-runtime/src/rlm'
        runtime.mkdir(parents=True)
        (runtime / '__init__.py').write_text('# synthetic runtime')
        (runtime / 'lifecycle.py').write_text('# synthetic lifecycle')
        (self.repo / 'Cargo.toml').write_text('# synthetic Cargo workspace')
        (self.repo / 'Cargo.lock').write_text('# synthetic locked dependencies')
        native = self.repo / 'crates/example/src'
        native.mkdir(parents=True)
        (native.parent / 'Cargo.toml').write_text('# synthetic crate')
        (native / 'lib.rs').write_text('// synthetic native source')
        self.git('init', '--quiet')
        self.git('add', 'resources', 'prime-agent-runtime', 'scripts', 'install.sh', 'LICENSE', 'README.md', 'Cargo.toml', 'Cargo.lock', 'crates')
        self.git('-c', 'user.name=Installer Test', '-c', 'user.email=installer@example.invalid',
                 'commit', '--quiet', '-m', 'Synthetic release')
        self.git('tag', 'v0.1.1')
        self.git('tag', 'v99.0.0-rc.1')
        self.commands = self.root / 'commands'
        self.commands.mkdir()
        cargo = self.commands / 'cargo'
        cargo.write_text('''#!/usr/bin/env python3
import importlib.util, json, os, sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('rust_release', root / 'scripts/rust_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
receipt = {'schema':release.PROVENANCE_SCHEMA, 'version':'0.1.1',
    'sourceTreeSha256':release.raw_aggregate(root, release.source_files(root)),
    'payloadSourceSha256':release.raw_aggregate(root, release.payload_files(root)),
    'runtimeSourceSha256':release.runtime_source_sha256(root),
    'target':'x86_64-apple-darwin' if sys.platform == 'darwin' else 'x86_64-unknown-linux-gnu',
    'profile':'release', 'rustc':'rustc synthetic installer fixture', 'buildOptionsSha256':'1'*64}
receipt['buildFingerprint'] = release.build_fingerprint(receipt)
p = Path(os.environ['CARGO_TARGET_DIR']) / 'release/optimus-rust'
p.parent.mkdir(parents=True)
p.write_text('#!/usr/bin/env python3\\nimport sys\\nprint(' + repr(json.dumps(receipt)) + ' if sys.argv[1:] == ["--build-provenance"] else "0.1.1")\\n')
p.chmod(0o755)
''')
        cargo.chmod(0o755)
        uv = self.commands / 'uv'
        uv.write_text('#!/bin/sh\nexit 0\n')
        uv.chmod(0o755)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(installer, 'REPOSITORY', str(self.repo)).start()
        mock.patch.dict(os.environ, {'PATH': str(self.commands) + os.pathsep + os.environ['PATH'],
                                    'GIT_CONFIG_GLOBAL': os.devnull,
                                    'GIT_CONFIG_NOSYSTEM': '1'}).start()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True).strip()

    def install(self, force=False):
        return installer.install(self.prefix, self.bin_dir, force)

    def test_tagged_source_install_repeat_update_and_cleanup(self):
        first = self.install()
        self.assertEqual((first / 'TAG').read_text().strip(), 'v0.1.1')
        self.assertEqual((first / 'COMMIT').read_text().strip(), self.git('rev-parse', 'HEAD'))
        self.assertFalse(list(self.prefix.glob('.build-*')))
        self.assertEqual(self.install(), first)
        second = self.install(force=True)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        # Check the installed launcher with a custom prefix and paths containing spaces.
        env = dict(os.environ, OPTIMUS_RUST_ROOT=str(self.prefix),
                   PRIME_AGENT_CODING_AGENT_DIR=str(self.root / 'isolated profile'),
                   XDG_RUNTIME_DIR=str(self.root / 'runtime'))
        result = subprocess.check_output([str(self.bin_dir / 'optimus-agent'), '--version'], env=env, text=True)
        self.assertEqual(result.strip(), '0.1.1')
        self.git('tag', 'v0.1.2')
        third = self.install()
        self.assertEqual((third / 'TAG').read_text().strip(), 'v0.1.2')
        self.assertTrue(second.is_dir())

    def test_failed_build_keeps_current_release_and_removes_build_tree(self):
        first = self.install()
        (self.commands / 'cargo').write_text('#!/bin/sh\nexit 42\n')
        self.git('tag', 'v0.1.2')
        with self.assertRaises(subprocess.CalledProcessError):
            self.install()
        self.assertEqual((self.prefix / 'current').resolve(), first)
        self.assertEqual(list((self.prefix / 'releases').iterdir()), [first])
        self.assertFalse(list(self.prefix.glob('.build-*')))

    def test_tag_moved_between_resolution_and_clone_fails_before_build(self):
        original = installer.git_output
        def moved(*args, **kwargs):
            if args == ('rev-parse', 'HEAD'):
                return 'f' * 40
            return original(*args, **kwargs)
        with mock.patch.object(installer, 'git_output', side_effect=moved):
            with self.assertRaisesRegex(ValueError, 'tag changed'):
                self.install()
        self.assertFalse((self.prefix / 'current').exists())
        self.assertFalse(list(self.prefix.glob('.build-*')))


if __name__ == '__main__':
    unittest.main()
