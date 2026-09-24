#!/usr/bin/env python3
"""Stage and install the native application without a Node/npm toolchain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
EXECUTABLE = "optimus-rust.exe" if WINDOWS else "optimus-rust"


# Raw-byte manifests are deliberately independent of Git state and machine paths.
PROVENANCE_SCHEMA = "optimus.build-provenance.v1"
FINGERPRINT_FIELDS = ("sourceTreeSha256", "payloadSourceSha256", "runtimeSourceSha256",
                      "version", "target", "profile", "rustc", "buildOptionsSha256")
PAYLOAD_FILES = ("scripts/optimus-agent", "scripts/launch-with-jev-env.py", "scripts/rust_release.py",
                 "install.sh", "LICENSE", "README.md")


def ignored_input(name: str) -> bool:
    return name in ("__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache") or name.endswith((".pyc", ".egg-info"))


def input_tree(root: Path, directory: Path) -> list[str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Build input directory must not be a symlink")
    files = []
    for path in directory.iterdir():
        if ignored_input(path.name):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink build input is not supported: {path}")
        if path.is_dir():
            files.extend(input_tree(root, path))
        elif path.is_file():
            files.append(path.relative_to(root).as_posix())
    return files


def payload_files(source: Path) -> list[str]:
    return sorted(list(PAYLOAD_FILES) + input_tree(source, source / "resources")
                  + input_tree(source, source / "prime-agent-runtime"))


def source_files(source: Path) -> list[str]:
    files = payload_files(source) + ["Cargo.toml", "Cargo.lock"]
    for crate in (source / "crates").iterdir():
        if crate.is_symlink():
            raise ValueError("Symlink crate input is not supported")
        if not crate.is_dir() or not (crate / "Cargo.toml").is_file():
            continue
        for name in ("Cargo.toml", "build.rs", "build_provenance.rs"):
            if (crate / name).is_file():
                files.append((crate / name).relative_to(source).as_posix())
        files.extend(input_tree(source, crate / "src"))
    return sorted(set(files))


def raw_aggregate(root: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Build input must be a regular file")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def runtime_source_sha256(source: Path) -> str:
    # Exact lifecycle.py contract: sorted immediate *.py filenames + NUL + raw digest.
    root = source / "prime-agent-runtime/src/rlm"
    names = sorted(path.name for path in root.glob("*.py"))
    if "lifecycle.py" not in names or "__init__.py" not in names:
        raise ValueError("Runtime lifecycle sources missing")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in names):
        raise ValueError("Runtime source must contain regular files")
    return raw_aggregate(root, names)


def build_fingerprint(receipt: dict) -> str:
    digest = hashlib.sha256((PROVENANCE_SCHEMA + "\0").encode())
    for field in FINGERPRINT_FIELDS:
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing native build provenance: {field}")
        digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def query_build_provenance(binary: Path) -> dict:
    # This native early exit runs before profile/auth/provider/daemon initialization.
    result = subprocess.run([str(binary.resolve()), "--build-provenance"], check=True,
                            capture_output=True, text=True, timeout=15)
    try:
        receipt = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError("Native binary has no valid embedded build provenance; rebuild it") from error
    if not isinstance(receipt, dict):
        raise ValueError("Native build provenance must be an object")
    return receipt


def verify_build_provenance(source: Path, binary: Path) -> dict:
    receipt = query_build_provenance(binary)
    if receipt.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("Native build provenance schema is missing or unsupported")
    for field in ("buildFingerprint", "sourceTreeSha256", "payloadSourceSha256",
                  "runtimeSourceSha256", "buildOptionsSha256"):
        if not isinstance(receipt.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[field]):
            raise ValueError(f"Native build provenance is not pinned: {field}")
    if receipt["buildFingerprint"] != build_fingerprint(receipt):
        raise ValueError("Native build fingerprint does not match its embedded inputs")
    target = receipt.get("target", "")
    platform_matches = ((sys.platform == "win32" and "-windows-" in target)
                        or (sys.platform == "darwin" and target.endswith("-apple-darwin"))
                        or (sys.platform.startswith("linux") and "-linux-" in target))
    if not platform_matches or not receipt["rustc"].startswith("rustc "):
        raise ValueError("Native target/toolchain does not match this packaging platform")
    if receipt.get("profile") != "release":
        raise ValueError("A portable release requires a release-profile native binary")
    identity = json.loads((source / "resources/agent/package.json").read_text())
    runtime_identity = tomllib.loads((source / "prime-agent-runtime/pyproject.toml").read_text())
    if receipt["version"] != identity.get("version") or receipt["version"] != runtime_identity.get("project", {}).get("version"):
        raise ValueError("Native/resource/runtime version identity mismatch")
    if receipt["runtimeSourceSha256"] != runtime_source_sha256(source):
        raise ValueError("Runtime source does not match the native build receipt")
    if receipt["payloadSourceSha256"] != raw_aggregate(source, payload_files(source)):
        raise ValueError("Packaged resources/runtime do not match the native build receipt")
    if (source / "Cargo.toml").is_file() and (source / "crates").is_dir():
        if receipt["sourceTreeSha256"] != raw_aggregate(source, source_files(source)):
            raise ValueError("Native source inputs do not match the binary; rebuild before staging")
    else:
        recorded = source / "BUILD-PROVENANCE.json"
        if not recorded.is_file() or json.loads(recorded.read_text()) != receipt:
            raise ValueError("Bundled install provenance does not match the native binary")
    return receipt


def git_bash() -> Path:
    """Find Git for Windows Bash, never the incompatible WSL bash.exe shim."""
    candidates = []
    git = shutil.which("git")
    if git:
        directory = Path(git).resolve().parent
        candidates.extend((directory / "bash.exe", directory.parent / "bin/bash.exe"))
    for variable, relative in (("ProgramFiles", "Git/bin/bash.exe"),
                               ("ProgramFiles(x86)", "Git/bin/bash.exe"),
                               ("LOCALAPPDATA", "Programs/Git/bin/bash.exe")):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]) / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("Install Git for Windows with Git Bash before installing Optimus.")


def windows_launcher(bash: Path, launcher: Path) -> str:
    # Literal percent signs must survive cmd.exe expansion; quoting handles spaces.
    bash_path = bash.as_posix().replace("%", "%%")
    script_path = launcher.as_posix().replace("%", "%%")
    return ("@echo off\nsetlocal DisableDelayedExpansion\n"
            f'"{bash_path}" -- "{script_path}" %*\nexit /b %errorlevel%\n')


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
    if destination.exists():
        raise FileExistsError(destination)
    provenance = verify_build_provenance(source, binary)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "bin").mkdir()
        shutil.copy2(binary, destination / "bin" / EXECUTABLE)
        (destination / "bin" / EXECUTABLE).chmod(0o755)
        # This allowlist excludes user configuration, build caches and Git state.
        ignore = lambda _directory, names: [name for name in names if ignored_input(name)]
        for directory in ("resources", "prime-agent-runtime"):
            shutil.copytree(source / directory, destination / directory, ignore=ignore)
        (destination / "scripts").mkdir()
        for name in ("optimus-agent", "launch-with-jev-env.py", "rust_release.py"):
            shutil.copy2(source / "scripts" / name, destination / "scripts" / name)
        for name in ("install.sh", "LICENSE", "README.md"):
            shutil.copy2(source / name, destination / name)
        (destination / "COMMIT").write_text(commit + "\n")
        if (source / "TAG").is_file():
            shutil.copy2(source / "TAG", destination / "TAG")
        (destination / "BUILD-PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
        verify_build_provenance(destination, destination / "bin" / EXECUTABLE)
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
    bash = git_bash() if WINDOWS else None
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
            if WINDOWS:
                # Git Bash works without Windows symlink privileges.
                pointer.write_text(release_name + "\n")
                current = prefix / "current.txt"
            else:
                pointer.symlink_to(Path("releases") / release_name, target_is_directory=True)
                current = prefix / "current"
            launcher = bin_dir / "optimus-agent"
            with tempfile.TemporaryDirectory(prefix=".optimus-agent-", dir=bin_dir) as temporary:
                staging = Path(temporary)
                candidate = staging / "optimus-agent"
                shutil.copyfile(release / "scripts" / "optimus-agent", candidate)
                candidate.chmod(0o755)
                candidates = [(candidate, launcher)]
                if WINDOWS:
                    command = staging / "optimus-agent.cmd"
                    command.write_text(windows_launcher(bash, launcher), newline="\r\n")
                    candidates.append((command, bin_dir / command.name))
                backups = {}
                for candidate, destination in candidates:
                    previous = staging / (candidate.name + ".previous")
                    if destination.exists() or destination.is_symlink():
                        shutil.copy2(destination, previous, follow_symlinks=False)
                        backups[destination] = previous
                replaced = []
                try:
                    for candidate, destination in candidates:
                        os.replace(candidate, destination)
                        replaced.append(destination)
                    os.replace(pointer, current)
                except BaseException:
                    for destination in reversed(replaced):
                        if destination in backups:
                            os.replace(backups[destination], destination)
                        else:
                            destination.unlink()
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
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Optimus installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
