"""Offline native derived-image layer driver; runs inside the image build stage.

Pipeline: verify the five root-staged input files (frozen raw-hash input
receipt), apply the exclusive disposable SGLang source tree overlay and
generated packaging artifacts through ``deploy.v3.native_package``, install
the staged plugin package with an offline-only pip command that also refuses
bytecode compilation (``--no-compile``, so the installed tree contains only
attested ``.py``/generated-JSON bytes), then prove the installed result with
an isolated interpreter (``-I``) that cannot see the staged build tree, the
current directory, ``PYTHON*`` variables or any network.

The driver performs no Docker operation itself and grants no runtime
authority. Missing build tools, any need for a network fallback, a digest
mismatch, a staged shadow, an unexpected installed file or an unexpected
installer-generated dist-info entry fails closed with a fixed code and a
nonzero exit, so the enclosing image layer is discarded. Receipts describe
staged and installed bytes only; they are not runtime, GPU, model-identity,
semantic or deployment-readiness evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

if __package__ in (None, ""):  # executed as a script inside the build stage
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deploy.v3 import native_package, source_patch
else:
    from . import native_package, source_patch

SCHEMA = "optimus.occ3.native-image-build.v1"
MANIFEST_NAMES = ("capture-eager-mtp.v1.json", "native-startup.v2.json",
                  "scheduler-cache-ipc.v1.json")
INPUT_NAMES = ("source-baseline-expanded.json", "native-profile-input.json")
STAGED_NAMES = tuple(sorted(MANIFEST_NAMES + INPUT_NAMES))
DIST_NAME = "occ3-bridge"
DIST_VERSION = "0.1.0"
DIST_INFO = "occ3_bridge-0.1.0.dist-info"
PACKAGE_NAME = "occ3_sglang"
ENTRY_GROUP = "sglang.srt.plugins"
ENTRY_NAME = "occ3_bridge"
ENTRY_VALUE = "occ3_sglang.plugin:load"
# Wheel data files plus the three installer-generated files a real
# ``pip install <local-dir>`` writes. Everything else in dist-info is refused.
ALLOWED_DIST_INFO = ("RECORD", "METADATA", "WHEEL", "entry_points.txt", "top_level.txt",
                     "INSTALLER", "REQUESTED", "direct_url.json")
INSTALLER_FILES = ("INSTALLER", "REQUESTED", "direct_url.json")
REQUIRED_DIST_INFO = ("RECORD", "METADATA", "WHEEL", "INSTALLER", "direct_url.json")
# Exact expected installer-generated contents; REQUESTED is optional-if-present.
INSTALLER_CONTENT = b"pip\n"
PIP_INSTALL_FLAGS = ("--no-index", "--no-deps", "--no-cache-dir", "--no-build-isolation",
                     "--no-compile", "--disable-pip-version-check")
MIN_PIP = (21, 0)
MIN_SETUPTOOLS = (64, 0)
MAX_SPEC_BYTES = 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_METADATA_FILE_BYTES = 8192
TOOLS_TIMEOUT = 120
INSTALL_TIMEOUT = 1800
VERIFY_TIMEOUT = 600
_BASE_IMAGE = re.compile(r"[A-Za-z0-9@._:/-]{1,512}\Z")


class BuildError(Exception):
    """Fixed non-sensitive error codes only; no paths or source text escape."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _need(condition, code):
    if not condition:
        raise BuildError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _execute(argv, *, timeout, env=None, cwd=None):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                            env=env, cwd=cwd)
    return result.returncode, result.stdout, result.stderr


def _isolated_env():
    # -I already ignores PYTHON* variables; strip them for the child anyway.
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("PYTHON", "PIP_"))}


TOOLS_PROBE_SOURCE = (
    "import importlib.metadata as metadata, json\n"
    "tools = {}\n"
    "for name in ('pip', 'setuptools'):\n"
    "    try:\n"
    "        tools[name] = metadata.distribution(name).version\n"
    "    except Exception:\n"
    "        tools[name] = None\n"
    "print(json.dumps(tools))\n"
)


def _version_tuple(value):
    parts = []
    for piece in str(value).split("."):
        digits = ""
        for character in piece:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            return None
        parts.append(int(digits))
    return tuple(parts) if parts else None


def precheck_build_tools(executable):
    """Presence and version metadata for pip/setuptools before any install.

    Reads distribution metadata only; pip and setuptools code is never
    imported here. There is no interpreter substitution and no network
    fallback: a missing tool fails the build.
    """
    code, stdout, _ = _execute([executable, "-I", "-c", TOOLS_PROBE_SOURCE],
                               timeout=TOOLS_TIMEOUT, env=_isolated_env())
    _need(code == 0, "build_tools_probe_failed")
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    try:
        tools = json.loads(lines[-1]) if lines else None
    except ValueError:
        tools = None
    _need(isinstance(tools, dict) and set(tools) == {"pip", "setuptools"},
          "build_tools_probe_failed")
    for name in ("pip", "setuptools"):
        version = tools[name]
        _need(isinstance(version, str) and 0 < len(version) <= 32, "build_tools_missing")
        parsed = _version_tuple(version)
        _need(parsed is not None, "build_tool_version_unsupported")
        minimum = MIN_PIP if name == "pip" else MIN_SETUPTOOLS
        _need(parsed >= minimum, "build_tool_version_unsupported")
        tools[name] = version
    tools["python"] = executable
    return tools


def pip_install_command(executable, project):
    """Exact offline pip argv; no index, no isolation, no cache, no bytecode
    compilation and no fallback."""
    return [executable, "-m", "pip", "install", *PIP_INSTALL_FLAGS, str(project)]


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise BuildError("invalid_arguments")

    def exit(self, status=0, message=None):
        raise BuildError("invalid_arguments")


def parse_options(argv):
    parser = _Parser(description=__doc__, add_help=False)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--sglang-srt", required=True)
    parser.add_argument("--occ3-sglang", required=True)
    parser.add_argument("--staged-inputs", required=True)
    parser.add_argument("--experimental-startup", default="false")
    parser.add_argument("--disposable-tree", action="store_true")
    arguments = parser.parse_args(argv)
    _need(_BASE_IMAGE.fullmatch(arguments.base_image) is not None, "invalid_base_image")
    for name in ("sglang_srt", "occ3_sglang", "staged_inputs"):
        value = getattr(arguments, name)
        _need(isinstance(value, str) and os.path.isabs(value)
              and len(value) <= 4096 and "\0" not in value, "invalid_path_argument")
    _need(arguments.disposable_tree is True, "explicit_disposable_tree_required")
    if arguments.experimental_startup == "true":
        arguments.experimental_startup = True
    elif arguments.experimental_startup == "false":
        arguments.experimental_startup = False
    else:
        raise BuildError("invalid_experimental_startup_flag")
    return arguments


def _metadata(info):
    return (info.st_dev, info.st_ino, stat.S_IMODE(info.st_mode), info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_uid, info.st_gid)


def _read_input_file(path, maximum):
    """Bounded fd read: the size bound is enforced before any full read, with
    identity re-checks around it. This narrows read races; it is not a
    TOCTOU-safety claim against a hostile concurrent writer."""
    info = path.lstat()
    _need(not stat.S_ISLNK(info.st_mode)
          and not (getattr(info, "st_file_attributes", 0) & 0x400)
          and stat.S_ISREG(info.st_mode), "staged_input_unsupported")
    _need(info.st_size <= maximum, "staged_input_oversize")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        actual = os.fstat(fd)
        _need(_metadata(info) == _metadata(actual), "staged_input_identity_drift")
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        _need(len(raw) <= maximum and len(raw) == info.st_size, "staged_input_oversize")
        _need(_metadata(after) == _metadata(info), "staged_input_identity_drift")
        return raw
    finally:
        if fd is not None:
            os.close(fd)


def _load_staged_inputs(directory):
    """Read exactly the five root-staged input files; nothing else is accepted."""
    directory = Path(directory)
    staged = {}
    with os.scandir(directory) as children:
        for child in children:
            _need(child.name in STAGED_NAMES, "staged_input_unexpected")
            _need(not child.is_symlink() and child.is_file()
                  and not child.is_dir(follow_symlinks=False), "staged_input_unsupported")
            staged[child.name] = Path(child.path)
    _need(set(staged) == set(STAGED_NAMES), "staged_input_missing")
    limits = {"source-baseline-expanded.json": native_package.MAX_BASELINE_BYTES,
              "native-profile-input.json": native_package.MAX_PROFILE_BYTES}
    for name in MANIFEST_NAMES:
        limits[name] = source_patch.MAX_MANIFEST_BYTES
    raws = {}
    for name, path in staged.items():
        raws[name] = _read_input_file(path, limits[name])
    return staged, raws


def compose_verification_spec(plan, sglang_root):
    """Byte expectations for the isolated child: staged plugin + generated
    artifacts as installed package files, final SGLang source hashes, and the
    trusted staged project root that a local pip install must attest in
    ``direct_url.json``."""
    patched = {item.relative: item.patched for item in plan.overlay.files}
    package_files = {relative: _sha(record.raw) for relative, record in plan.plugin_files}
    for name, raw in plan.artifacts:
        _need(name not in package_files, "spec_composition_invalid")
        package_files[name] = _sha(raw)
    sglang_files = [{"path": relative, "sha256": _sha(patched.get(relative, record.raw))}
                    for relative, record in plan.sg_files]
    return {"dist": {"name": DIST_NAME, "version": DIST_VERSION, "entry_group": ENTRY_GROUP,
                     "entry_name": ENTRY_NAME, "entry_value": ENTRY_VALUE},
            "package_files": package_files,
            "project_root": str(Path(plan.occ3_sglang).parent),
            "sglang_root": str(Path(sglang_root)),
            "sglang_files": sglang_files,
            "sglang_version": native_package.SOURCE_VERSION}


ISOLATED_VERIFY_SOURCE = """
import base64
import hashlib
import json
import os
import stat
import sys
import sysconfig
import importlib.metadata as metadata
from pathlib import Path, PurePosixPath

MAX_SPEC_BYTES = %(max_spec)d
MAX_FILE_BYTES = %(max_file)d
MAX_METADATA_FILE_BYTES = %(max_meta)d
DIST_NAME = %(dist_name)r
DIST_VERSION = %(dist_version)r
DIST_INFO = %(dist_info)r
PACKAGE_NAME = %(package_name)r
ENTRY_GROUP = %(entry_group)r
ENTRY_NAME = %(entry_name)r
ENTRY_VALUE = %(entry_value)r
ALLOWED_DIST_INFO = %(allowed_dist_info)r
REQUIRED_DIST_INFO = %(required_dist_info)r
INSTALLER_FILES = %(installer_files)r
INSTALLER_CONTENT = %(installer_content)r
HEX = "0123456789abcdef"


def emit(value):
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def fail(code):
    emit({"ok": False, "code": code, "mode": "isolated-interpreter"})
    raise SystemExit(1)


def _no_duplicate(code):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(code)
            result[key] = value
        return result
    return hook


def constant(_value):
    fail("spec_invalid_constant")


def _read_fd(path, limit, unreadable, oversize):
    # Bounded fd read: the limit is checked against fstat BEFORE reading, and
    # the file identity is re-checked after the read. This narrows read races;
    # it is not a TOCTOU-safety claim against a hostile concurrent writer.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        fail(unreadable)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > limit):
            fail(oversize)
        chunks = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if (len(raw) > limit or len(raw) != before.st_size
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            fail(oversize)
        return raw
    finally:
        os.close(fd)


def read_bounded(path, limit):
    return _read_fd(path, limit, "read_failed", "file_oversize")


def read_spec(path, limit):
    return _read_fd(path, limit, "spec_unreadable", "spec_size_invalid")


def digest_hex(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        fail("spec_digest_invalid")
    return value


if len(sys.argv) != 2:
    fail("spec_argument_invalid")
spec_raw = read_spec(Path(sys.argv[1]), MAX_SPEC_BYTES)
if not 0 < len(spec_raw) <= MAX_SPEC_BYTES:
    fail("spec_size_invalid")
try:
    spec = json.loads(spec_raw.decode("utf-8"),
                      object_pairs_hook=_no_duplicate("spec_duplicate_key"),
                      parse_constant=constant)
except Exception:
    fail("spec_invalid")
if not isinstance(spec, dict) or set(spec) != {"dist", "package_files", "project_root",
                                               "sglang_root", "sglang_files",
                                               "sglang_version"}:
    fail("spec_fields_invalid")
if spec["dist"] != {"name": DIST_NAME, "version": DIST_VERSION, "entry_group": ENTRY_GROUP,
                    "entry_name": ENTRY_NAME, "entry_value": ENTRY_VALUE}:
    fail("spec_dist_invalid")
if not isinstance(spec["package_files"], dict) or not spec["package_files"]:
    fail("spec_package_files_invalid")
for name, value in spec["package_files"].items():
    if (not isinstance(name, str) or not name.endswith((".py", ".json"))
            or "/" in name or "\\\\" in name or ".." in name or not name.isascii()):
        fail("spec_package_path_invalid")
    digest_hex(value)
project_root = spec["project_root"]
if (not isinstance(project_root, str) or not 1 <= len(project_root) <= 4096
        or "\\0" in project_root or not project_root.isascii()
        or not os.path.isabs(project_root)):
    fail("spec_project_root_invalid")
if not isinstance(spec["sglang_root"], str) or len(spec["sglang_root"]) > 4096:
    fail("spec_sglang_root_invalid")
if not isinstance(spec["sglang_files"], list) or not spec["sglang_files"]:
    fail("spec_sglang_files_invalid")
for entry in spec["sglang_files"]:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        fail("spec_sglang_entry_invalid")
    path = entry["path"]
    if (not isinstance(path, str) or len(path) > 180 or not path.isascii()
            or path.startswith("/") or "\\\\" in path or not path.endswith(".py")):
        fail("spec_sglang_path_invalid")
    parts = PurePosixPath(path).parts
    if (not parts or any(part in (".", "..") for part in parts)
            or str(PurePosixPath(path)) != path):
        fail("spec_sglang_path_invalid")
    digest_hex(entry["sha256"])
if not isinstance(spec["sglang_version"], str) or not 0 < len(spec["sglang_version"]) <= 32:
    fail("spec_sglang_version_invalid")

# -I keeps the staged build tree, the current directory and PYTHON* variables
# off sys.path, so only the interpreter's own installed files can be read.
try:
    distribution = metadata.distribution(DIST_NAME)
except Exception:
    fail("distribution_missing")
try:
    installed_name = distribution.metadata["Name"]
except Exception:
    fail("distribution_metadata_invalid")
if installed_name != DIST_NAME or distribution.version != DIST_VERSION:
    fail("distribution_identity_mismatch")
try:
    matches = [item for item in metadata.distributions()
               if (item.metadata["Name"] or "").lower() == DIST_NAME]
except Exception:
    fail("distribution_enumeration_failed")
if len(matches) != 1:
    fail("duplicate_distribution")
try:
    entries = [item for item in distribution.entry_points if item.group == ENTRY_GROUP]
except Exception:
    fail("entry_points_invalid")
if len(entries) != 1 or entries[0].name != ENTRY_NAME or entries[0].value != ENTRY_VALUE:
    fail("entry_point_mismatch")

purelib = Path(sysconfig.get_paths()["purelib"])
package_dir = purelib / PACKAGE_NAME
dist_info = purelib / DIST_INFO
if not package_dir.is_dir() or not dist_info.is_dir():
    fail("installed_layout_missing")

found = {}
for path in sorted(package_dir.rglob("*")):
    if path.is_dir() and not path.is_symlink():
        continue
    info = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (getattr(info, "st_file_attributes", 0) & 0x400)):
        fail("installed_file_unsupported")
    relative = path.relative_to(package_dir).as_posix()
    if relative in found:
        fail("installed_path_duplicate")
    found[relative] = hashlib.sha256(read_bounded(path, MAX_FILE_BYTES)).hexdigest()
if set(found) != set(spec["package_files"]):
    fail("installed_membership_mismatch")
for name, value in found.items():
    if value != spec["package_files"][name]:
        fail("installed_bytes_mismatch")

# Installer-generated dist-info layout: walk the actual directory (not RECORD)
# and require exactly the allowed, bounded file set.
found_dist = set()
for path in sorted(dist_info.rglob("*")):
    if path.is_dir() and not path.is_symlink():
        continue
    info = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (getattr(info, "st_file_attributes", 0) & 0x400)):
        fail("dist_info_file_unsupported")
    name = path.relative_to(dist_info).as_posix()
    if "/" in name or name not in ALLOWED_DIST_INFO:
        fail("record_unexpected_dist_info_entry")
    found_dist.add(name)
for name in REQUIRED_DIST_INFO:
    if name not in found_dist:
        fail("installer_metadata_missing")

record_raw = read_bounded(dist_info / "RECORD", MAX_FILE_BYTES)
try:
    record_lines = record_raw.decode("utf-8").splitlines()
except UnicodeDecodeError:
    fail("record_invalid")


def _record_path_ok(row):
    if (not row or len(row) > 512 or not row.isascii() or row.startswith("/")
            or "\\\\" in row):
        return False
    parts = PurePosixPath(row).parts
    return (all(part not in (".", "..", "") for part in parts)
            and str(PurePosixPath(row)) == row)


def _check_record_hash(encoded, expected_hex, code):
    algorithm, _, value = encoded.partition("=")
    if algorithm != "sha256" or not value:
        fail("record_digest_unsupported")
    padded = value + "=" * (-len(value) %% 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        fail("record_digest_invalid")
    if len(decoded) != 32 or decoded.hex() != expected_hex:
        fail(code)


seen_rows = set()
recorded = set()
record_dist = {}
for line in record_lines:
    if not line:
        continue
    parts = line.split(",")
    if not parts[0] or len(parts) > 3:
        fail("record_invalid")
    row = parts[0]
    if not _record_path_ok(row):
        fail("record_invalid")
    if row in seen_rows:
        fail("record_duplicate_entry")
    seen_rows.add(row)
    if row.startswith(PACKAGE_NAME + "/"):
        relative = row[len(PACKAGE_NAME) + 1:]
        if relative not in spec["package_files"]:
            fail("record_unexpected_package_entry")
        if len(parts) < 2 or not parts[1]:
            fail("record_hash_missing")
        _check_record_hash(parts[1], found[relative], "record_hash_mismatch")
        recorded.add(relative)
    elif row.startswith(DIST_INFO + "/"):
        name = row[len(DIST_INFO) + 1:]
        if name not in ALLOWED_DIST_INFO:
            fail("record_unexpected_dist_info_entry")
        record_dist[name] = parts
    else:
        fail("record_unexpected_entry")
if recorded != set(spec["package_files"]):
    fail("record_missing_package_entry")
if set(record_dist) != found_dist:
    fail("record_dist_info_membership_mismatch")

dist_bytes = {"RECORD": record_raw}
for name, parts in sorted(record_dist.items()):
    if name == "RECORD":
        # pip writes its own row without hash/size; a hash, if present, is
        # still verified against the actual bytes.
        if len(parts) >= 2 and parts[1]:
            _check_record_hash(parts[1], hashlib.sha256(record_raw).hexdigest(),
                               "record_hash_mismatch")
        continue
    if len(parts) < 2 or not parts[1]:
        fail("record_hash_missing")
    limit = MAX_METADATA_FILE_BYTES if name in INSTALLER_FILES else MAX_FILE_BYTES
    raw = read_bounded(dist_info / name, limit)
    _check_record_hash(parts[1], hashlib.sha256(raw).hexdigest(), "record_hash_mismatch")
    dist_bytes[name] = raw

# Exact installer-generated contents, bound to the trusted staged project root
# from the driver spec (never to a self-asserted RECORD or file content).
if dist_bytes["INSTALLER"] != INSTALLER_CONTENT:
    fail("installer_invalid")
if "REQUESTED" in dist_bytes and dist_bytes["REQUESTED"] != b"":
    fail("requested_invalid")
try:
    direct = json.loads(dist_bytes["direct_url.json"].decode("utf-8"),
                        object_pairs_hook=_no_duplicate("direct_url_invalid"))
except Exception:
    fail("direct_url_invalid")
if direct != {"dir_info": {}, "url": "file://" + project_root}:
    fail("direct_url_invalid")

sglang_root = Path(spec["sglang_root"])
if not sglang_root.is_dir():
    fail("sglang_root_missing")
for entry in spec["sglang_files"]:
    target = sglang_root.joinpath(*PurePosixPath(entry["path"]).parts)
    info = target.lstat()
    if (target.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (getattr(info, "st_file_attributes", 0) & 0x400)):
        fail("sglang_source_unsupported")
    if hashlib.sha256(read_bounded(target, MAX_FILE_BYTES)).hexdigest() != entry["sha256"]:
        fail("sglang_source_mismatch")

try:
    sglang = metadata.distribution("sglang")
except metadata.PackageNotFoundError:
    fail("sglang_distribution_missing")
except Exception:
    fail("sglang_distribution_invalid")
if sglang.version != spec["sglang_version"]:
    fail("sglang_version_mismatch")

emit({"ok": True, "mode": "isolated-interpreter", "isolated": True,
      "installed_package_files": len(found),
      "installed_sglang_files": len(spec["sglang_files"]),
      "sglang_version": spec["sglang_version"],
      "distribution": DIST_NAME + "==" + DIST_VERSION,
      "entry_point": ENTRY_NAME + "=" + ENTRY_VALUE,
      "runtime_imported": False, "gpu_proof": False, "readiness_proof": False})
""" % {"max_spec": MAX_SPEC_BYTES, "max_file": MAX_FILE_BYTES, "max_meta": MAX_METADATA_FILE_BYTES,
       "dist_name": DIST_NAME, "dist_version": DIST_VERSION, "dist_info": DIST_INFO,
       "package_name": PACKAGE_NAME, "entry_group": ENTRY_GROUP, "entry_name": ENTRY_NAME,
       "entry_value": ENTRY_VALUE, "allowed_dist_info": ALLOWED_DIST_INFO,
       "required_dist_info": REQUIRED_DIST_INFO, "installer_files": INSTALLER_FILES,
       "installer_content": INSTALLER_CONTENT}


VERIFY_RECEIPT_FIELDS = frozenset({"ok", "mode", "isolated", "installed_package_files",
                                    "installed_sglang_files", "sglang_version", "distribution",
                                    "entry_point", "runtime_imported", "gpu_proof",
                                    "readiness_proof"})


def _run_verify_child(executable, spec_path, *, cwd=None, env=None):
    argv = [executable, "-I", "-c", ISOLATED_VERIFY_SOURCE, str(spec_path)]
    code, stdout, _ = _execute(argv, timeout=VERIFY_TIMEOUT, env=env or _isolated_env(),
                               cwd=cwd)
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    receipt = None
    if lines:
        try:
            receipt = json.loads(lines[-1])
        except ValueError:
            receipt = None
    _need(code == 0 and isinstance(receipt, dict) and set(receipt) == VERIFY_RECEIPT_FIELDS
          and receipt["ok"] is True and receipt["mode"] == "isolated-interpreter"
          and receipt["isolated"] is True and receipt["runtime_imported"] is False
          and receipt["gpu_proof"] is False and receipt["readiness_proof"] is False,
          "installed_verification_failed")
    return receipt


def run_isolated_verify(executable, spec, *, cwd=None, env=None):
    """Write a bounded temp spec and prove the installed result with -I."""
    raw = _canonical(spec)
    _need(0 < len(raw) <= MAX_SPEC_BYTES, "spec_oversize")
    handle, path = tempfile.mkstemp(prefix="occ3-verify-", suffix=".json")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return _run_verify_child(executable, path, cwd=cwd, env=env)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_layer(argv):
    """One exclusive disposable layer: verify, overlay, install offline, prove."""
    options = parse_options(argv)
    staged, raws = _load_staged_inputs(options.staged_inputs)
    input_receipt = {name: {"sha256": _sha(raw), "bytes": len(raw)}
                     for name, raw in sorted(raws.items())}
    tools = precheck_build_tools(sys.executable)
    plan = native_package.prepare(options.sglang_srt, options.occ3_sglang,
                                  staged["source-baseline-expanded.json"],
                                  staged["native-profile-input.json"],
                                  [staged[name] for name in MANIFEST_NAMES],
                                  experimental_startup=options.experimental_startup)
    spec = compose_verification_spec(plan, options.sglang_srt)
    project_root = Path(plan.occ3_sglang).parent
    package_receipt = native_package.apply(plan, disposable=True)
    _need(package_receipt.get("ok") is True and package_receipt.get("applied") is True,
          "package_apply_failed")
    code, _, _ = _execute(pip_install_command(sys.executable, project_root),
                          timeout=INSTALL_TIMEOUT)
    _need(code == 0, "pip_install_failed")
    verified = run_isolated_verify(sys.executable, spec)
    return {"schema": SCHEMA, "ok": True, "layer_only": True, "docker_operation": False,
            "base_image": options.base_image,
            "experimental_startup_selected": options.experimental_startup,
            "input_receipt": input_receipt,
            "package_receipt": package_receipt,
            "staged": {"sglang_srt_root": str(Path(options.sglang_srt)),
                       "occ3_sglang_root": str(Path(options.occ3_sglang)),
                       "plugin_source_files": len(plan.plugin_files),
                       "sg_source_files": len(plan.sg_files),
                       "patched_source_files": len(plan.overlay.files)},
            "install": {"flags": list(PIP_INSTALL_FLAGS), "offline": True, "index": False,
                        "build_isolation": False, "network_fallback": False,
                        "bytecode_compilation": False, "project_root": str(project_root)},
            "build_tools": tools,
            "installed": verified,
            "runtime_imported": False, "gpu_proof": False, "readiness_proof": False}


def _failure(code):
    return json.dumps({"ok": False, "schema": SCHEMA, "code": code,
                       "runtime_imported": False, "gpu_proof": False,
                       "readiness_proof": False}, sort_keys=True,
                      separators=(",", ":"))


def main(argv=None):
    try:
        receipt = build_layer(sys.argv[1:] if argv is None else list(argv))
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except BuildError as error:
        print(_failure(error.code))
        return 1
    except native_package.PackageError as error:
        print(_failure("package_" + error.code))
        return 1
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        print(_failure("build_failed"))
        return 1
    except BaseException:
        print(_failure("build_failed"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
