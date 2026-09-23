#!/usr/bin/env python3
"""Build and install the highest stable vMAJOR.MINOR.PATCH tag from Telemus."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = 'https://github.com/telemusai/optimus-agent.git'
STABLE_TAG = re.compile(r'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')


def latest_tag(remote_refs):
    tags, peeled = {}, {}
    for line in remote_refs.splitlines():
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r'[0-9a-f]{40}', parts[0]):
            continue
        commit, ref = parts
        if not ref.startswith('refs/tags/'):
            continue
        tag = ref[len('refs/tags/'):]
        if tag.endswith('^{}'):
            peeled[tag[:-3]] = commit
            continue
        version = STABLE_TAG.fullmatch(tag)
        if version:
            tags[tuple(map(int, version.groups()))] = (tag, commit)
    if not tags:
        raise ValueError('No stable vMAJOR.MINOR.PATCH tag is published. No installation was changed.')
    tag, commit = tags[max(tags)]
    return tag, peeled.get(tag, commit)


def git_output(*arguments, cwd=None):
    return subprocess.check_output(['git', *arguments], cwd=cwd, text=True).strip()


def installed_release(prefix):
    if os.name == 'nt':
        pointer = prefix / 'current.txt'
        if not pointer.is_file():
            return None
        name = pointer.read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
            raise ValueError('Invalid installed release pointer')
        return prefix / 'releases' / name
    current = prefix / 'current'
    return current.resolve() if current.is_dir() else None


def install(prefix, bin_dir, force=False):
    for command in ('git', 'cargo', 'uv'):
        if not shutil.which(command):
            raise ValueError(f'{command} is required. Install the documented prerequisites first.')
    tag, commit = latest_tag(git_output('ls-remote', '--tags', REPOSITORY))
    print(f'Latest stable tag: {tag} ({commit[:12]})', flush=True)
    current = installed_release(prefix)
    executable = 'optimus-rust.exe' if os.name == 'nt' else 'optimus-rust'
    launchers = ['optimus-agent', 'optimus-agent.cmd'] if os.name == 'nt' else ['optimus-agent']
    launchers_present = all((bin_dir / name).is_file() for name in launchers)
    if not force and launchers_present and current and (current / 'COMMIT').is_file() and (current / 'TAG').is_file():
        if (current / 'COMMIT').read_text().strip() == commit and (current / 'TAG').read_text().strip() == tag:
            subprocess.run([str(current / 'bin' / executable), '--version'], check=True)
            print(f'{tag} is already installed. Use --force to rebuild it.')
            return current
    prefix.mkdir(parents=True, exist_ok=True)
    # A private build directory on the installation filesystem avoids small RAM-backed /tmp mounts.
    with tempfile.TemporaryDirectory(prefix='.build-', dir=prefix) as temporary:
        source = Path(temporary) / 'source'
        subprocess.run(['git', 'clone', '--quiet', '--depth', '1', '--single-branch',
                        '--branch', tag, REPOSITORY, str(source)], check=True)
        if git_output('rev-parse', 'HEAD', cwd=source) != commit:
            raise ValueError('The release tag changed during download; rerun the installer.')
        if not (source / 'scripts/rust_release.py').is_file():
            raise ValueError(f'{tag} predates the standalone Rust installer. Publish a newer stable tag first.')
        environment = dict(os.environ)
        environment['CARGO_TARGET_DIR'] = str(Path(temporary) / 'target')
        environment['CARGO_INCREMENTAL'] = '0'
        environment.setdefault('CARGO_BUILD_JOBS', '2')
        environment.pop('CARGO_BUILD_TARGET', None)
        print('Building Optimus locally. Rust and native C/C++ build tools are required; this can take several minutes.', flush=True)
        subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'pi-coding-agent',
                        '--bin', 'optimus-rust'], cwd=source, env=environment, check=True)
        (source / 'TAG').write_text(tag + '\n')
        binary = Path(environment['CARGO_TARGET_DIR']) / 'release' / executable
        subprocess.run([sys.executable, str(source / 'scripts/rust_release.py'), 'install',
                        '--binary', str(binary), '--prefix', str(prefix), '--bin-dir', str(bin_dir)],
                       cwd=source, env=environment, check=True)
    print(f'Installed {tag}. Existing configuration, credentials, sessions, and prior releases were retained.')
    print('Running daemons are unchanged. Close/restart Optimus services when ready, then resume your session.')
    return installed_release(prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/optimus-rust')
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    parser.add_argument('--force', action='store_true', help='Rebuild even if the current tag is installed')
    args = parser.parse_args()
    install(args.prefix.expanduser().resolve(), args.bin_dir.expanduser().resolve(), args.force)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Optimus installation failed: {error}', file=sys.stderr)
        sys.exit(1)
