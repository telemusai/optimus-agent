#!/usr/bin/env python3
"""Stage and install the native application without a Node/npm toolchain."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = "optimus-rust.exe" if os.name == "nt" else "optimus-rust"


def source_commit(source: Path) -> str:
    recorded = source / "COMMIT"
    if recorded.is_file():
        return recorded.read_text().strip()
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def stage_release(source: Path, binary: Path, destination: Path, commit: str) -> None:
    """Create a new bundle; never overwrite a release or copy local credentials."""
    if not binary.is_file():
        raise ValueError(f"Native executable does not exist: {binary}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise ValueError("Expected a Git commit ID")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "bin").mkdir()
        shutil.copy2(binary, destination / "bin" / EXECUTABLE)
        (destination / "bin" / EXECUTABLE).chmod(0o755)
        # This allowlist excludes user configuration, build caches and Git state.
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "*.egg-info")
        for directory in ("resources", "prime-agent-runtime"):
            shutil.copytree(source / directory, destination / directory, ignore=ignore)
        (destination / "scripts").mkdir()
        for name in ("optimus-agent", "launch-with-jev-env.py", "rust_release.py"):
            shutil.copy2(source / "scripts" / name, destination / "scripts" / name)
        for name in ("install.sh", "LICENSE", "README.md"):
            shutil.copy2(source / name, destination / name)
        (destination / "COMMIT").write_text(commit + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise


def resolve_binary(source: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    bundled = source / "bin" / EXECUTABLE
    if bundled.is_file():
        return bundled
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "-p", "pi-coding-agent", "--bin", "optimus-rust"],
        cwd=source, check=True,
    )
    target = Path(os.environ.get("CARGO_TARGET_DIR", source / "target"))
    if not target.is_absolute():
        target = source / target
    triple = os.environ.get("CARGO_BUILD_TARGET")
    if triple:
        target = target / triple
    return target / "release" / EXECUTABLE


def install_release(source: Path, binary: Path, prefix: Path, bin_dir: Path, commit: str) -> Path:
    if shutil.which("uv") is None:
        raise ValueError("Install uv first; Optimus uses it to prepare its Python runtime on first use.")
    release_name = f"{commit[:12]}-{uuid.uuid4().hex[:8]}"
    release = prefix / "releases" / release_name
    stage_release(source, binary, release, commit)
    try:
        environment = dict(os.environ, PI_PACKAGE_DIR=str(release / "resources" / "agent"))
        subprocess.run([str(release / "bin" / EXECUTABLE), "--version"],
                       env=environment, cwd=release, check=True)
        bin_dir.mkdir(parents=True, exist_ok=True)
        pointer = prefix / f".current-{uuid.uuid4().hex}"
        try:
            if os.name == "nt":
                # Git Bash works without Windows symlink privileges.
                pointer.write_text(release_name + "\n")
                current = prefix / "current.txt"
            else:
                pointer.symlink_to(Path("releases") / release_name, target_is_directory=True)
                current = prefix / "current"
            launcher = bin_dir / "optimus-agent"
            with tempfile.TemporaryDirectory(prefix=".optimus-agent-", dir=bin_dir) as temporary:
                candidate = Path(temporary) / "launcher"
                previous = Path(temporary) / "previous"
                had_launcher = launcher.exists() or launcher.is_symlink()
                if had_launcher:
                    shutil.copy2(launcher, previous, follow_symlinks=False)
                shutil.copyfile(release / "scripts" / "optimus-agent", candidate)
                candidate.chmod(0o755)
                os.replace(candidate, launcher)
                try:
                    os.replace(pointer, current)
                except OSError:
                    if had_launcher:
                        os.replace(previous, launcher)
                    else:
                        launcher.unlink()
                    raise
        finally:
            pointer.unlink(missing_ok=True)
    except BaseException:
        # A failed candidate is never selected; previous releases are retained.
        shutil.rmtree(release)
        raise
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="Create a portable native release archive")
    stage.add_argument("--binary", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--name", default="optimus-agent")
    install = commands.add_parser("install", help="Install from a checkout or extracted native bundle")
    install.add_argument("--binary", type=Path)
    install.add_argument("--prefix", type=Path, default=Path.home() / ".local/share/optimus-rust")
    install.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
    args = parser.parse_args()
    commit = source_commit(ROOT)
    if args.command == "stage":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
            parser.error("Archive name must be a simple filename")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        archive_format = "zip" if os.name == "nt" else "gztar"
        suffix = ".zip" if os.name == "nt" else ".tar.gz"
        if (output / (args.name + suffix)).exists():
            parser.error("Refusing to overwrite an existing archive")
        with tempfile.TemporaryDirectory(prefix="optimus-stage-") as temporary:
            bundle = Path(temporary) / args.name
            stage_release(ROOT, args.binary.resolve(), bundle, commit)
            archive = Path(shutil.make_archive(str(output / args.name), archive_format,
                                             root_dir=temporary, base_dir=args.name))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
        print(archive)
    else:
        binary = resolve_binary(ROOT, args.binary)
        prefix = args.prefix.expanduser().resolve()
        release = install_release(ROOT, binary, prefix, args.bin_dir.expanduser().resolve(), commit)
        print(f"Installed {release}")
        print(f"Add {args.bin_dir} to PATH, then run optimus-agent in your project.")
        if prefix != (Path.home() / ".local/share/optimus-rust").resolve():
            print(f"Set OPTIMUS_RUST_ROOT to {prefix} when using this custom prefix.")
        print("The Python environment is prepared by uv on first use. Existing sessions are not restarted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Optimus installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
