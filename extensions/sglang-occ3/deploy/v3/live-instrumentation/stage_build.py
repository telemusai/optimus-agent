#!/usr/bin/env python3
"""Assemble the DGX staging for the OCC-3 iCalib window (P1 build, P2 swap, P3 arms).

Offline and side-effect free until it writes: it reads this worktree, builds the exact file list the
DGX steps need, and emits (a) a chunked-base64 payload list for `dgx_ssh.py run` transfer and (b) a
manifest of every staged file with its sha256. `--dry-run` computes and prints everything without
writing a byte.

Staged layout (live-build-sequence.md section 3, live-swap-checklist.md section 2):

  /data/occ3-bridge/icalib-src-<WIN>/          build context + controller CLI closure
      pyproject.toml, README.md
      occ3_sglang/*.py                         35 flat plugin modules, nothing else
      deploy/Dockerfile
      deploy/v3/*.py                           the five Dockerfile-copied utilities, plus the
                                               controller/lifecycle/supervisor modules P2 and S4 run
      patches/<three>.v1.json                  the three pinned SGLang patch records
      deploy/native-build-inputs/*.json         the two root-staged packaging inputs
      stage-manifest.sha256                     `sha256sum -c` input for the S3 pre-build verify
  /data/occ3-bridge/v3-live-<WIN>/             attempt dir inputs
      plan.json                               the P1/P2 plan (per-message profile)
      arms/arm-order.json                     declared arm order, control path, P5 exclusion
      arms/control-<arm>.txt                  the operator's control-file content per arm
      extras/site-packages/occ3_p3_arms_hook.py, zz_occ3_p3_arms.pth
      stage-manifest.json                     full manifest (this script's evidence)
  /opt/sglang/lib/python3.12/site-packages/   the hook's install target inside the image

The hook extras are NOT part of the attested plugin set: they are copied into the candidate
container's site-packages (docker cp, between S4 and S6) and declared by digest in the window record.

Exit codes: 0 clean, 2 refused (an invariant failed), 3 usage.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parents[3]
INST_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = EXT_ROOT / "occ3_sglang"
DEFAULT_PLAN = EXT_ROOT / "deploy" / "v3" / "plans" / "occ3-icalib-live-p1p2.json"

BUILD_UTILITIES = ("__init__.py", "artifact_preflight.py", "source_patch.py",
                   "native_package.py", "native_image_build.py")
CONTROLLER_MODULES = ("controller.py", "docker_api.py", "docker_config.py", "lifecycle.py",
                      "live_validation.py", "native_validator.py", "rehearsal.py",
                      "rehearsal_ingress.py", "service_checks.py", "supervisor.py")
PATCHES = ("capture-eager-mtp.v1.json", "native-startup.v1.json", "scheduler-cache-ipc.v1.json")
BUILD_INPUTS = ("source-baseline-expanded.json", "native-profile-input.json")
HOOK_FILES = ("occ3_p3_arms_hook.py", "zz_occ3_p3_arms.pth")

# Hard pins from live-build-sequence.md section 3. A mismatch is a refusal, never a warning.
PROFILE_SHA256 = "6e676f1c6fc795ba412c4327168b51c8ef6f65056664638c4954eec58e75348a"
BASELINE_SHA256 = "509345ec366fe775e29412b057f68a0019b8a6829c98d303229f5ebf5387f8bc"
PLUGIN_FILE_COUNT = 35
GENERATED_OUTPUTS = ("native_profile.json", "native_sources.json", "_native_profile_digest.py")
PER_MESSAGE_METHOD = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1"

SITE_PACKAGES = "/opt/sglang/lib/python3.12/site-packages"
CONTROL_PATH = "/tmp/occ3_p3_arm"
CHUNK_CHARS = 6000
MAX_PY_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024

ARM_ORDER = ("N-none", "F-text", "T-exact", "T-norm", "N-shape")
ARM_TOKENS = {"T-exact": b"texact\n", "T-norm": b"tnorm\n", "N-shape": b"nshape\n",
              "N-none": b"", "F-text": b""}
ARM_OPERATOR_STEP = {
    "N-none": "remove the control file before the turn",
    "F-text": "control file absent",
    "T-exact": "write arms/control-T-exact.txt into the control path",
    "T-norm": "write arms/control-T-norm.txt into the control path",
    "N-shape": "write arms/control-N-shape.txt into the control path, remove it after the turn",
}


class Refusal(Exception):
    """One fixed refusal code; nothing is written after a refusal."""

    def __init__(self, code, detail=""):
        self.code, self.detail = code, detail
        super().__init__(code if not detail else code + ": " + detail)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def b64_chunks(data, size=CHUNK_CHARS):
    encoded = base64.b64encode(data).decode("ascii")
    return [encoded[index:index + size] for index in range(0, len(encoded), size)]


def _require(condition, code, detail=""):
    if not condition:
        raise Refusal(code, detail)


IGNORED_LOCAL_DIRS = ("__pycache__", ".pytest_cache")


def plugin_files():
    """(the 35 flat plugin modules, local dirs present but never staged, excluded files).

    Only `*.py` at the top level is staged. A `__pycache__` inside the staged tree is a hard build
    failure (`plugin_flat_source_only`, deploy/v3/native_package.py:530-533), so local caches are
    recorded as excluded rather than shipped; any other subdirectory is a real layout change and
    refuses.
    """
    names = sorted(path.name for path in PLUGIN_DIR.glob("*.py"))
    _require(len(names) == PLUGIN_FILE_COUNT, "plugin_file_count",
             "%d != %d" % (len(names), PLUGIN_FILE_COUNT))
    _require("__init__.py" in names, "plugin_package_missing")
    ignored = []
    for nested in sorted(PLUGIN_DIR.iterdir()):
        if not nested.is_dir():
            continue
        if nested.name in IGNORED_LOCAL_DIRS:
            ignored.append("occ3_sglang/" + nested.name)
            continue
        raise Refusal("plugin_flat_source_only", nested.name)
    for name in GENERATED_OUTPUTS:
        _require(not (PLUGIN_DIR / name).exists(), "generated_output_already_exists", name)
    excluded = sorted("occ3_sglang/" + path.name for path in PLUGIN_DIR.glob("*.pyc"))
    return names, ignored, excluded


def _entry(role, source, dgx_path, content=None):
    return {"role": role, "source": source, "dgx_path": dgx_path, "content": content}


def plan_window_id(plan_path):
    try:
        text = Path(plan_path).read_text(encoding="utf-8")
    except OSError:
        raise Refusal("plan_unreadable", str(plan_path)) from None
    try:
        plan = json.loads(text)
    except ValueError:
        raise Refusal("plan_invalid_json", str(plan_path)) from None
    _require(isinstance(plan, dict), "plan_invalid")
    window = plan.get("window") or plan.get("environment_additions", [{}])[0].get("value", "")
    _require(isinstance(window, str) and window.startswith("icalib-"), "plan_window_missing",
             str(window))
    return plan, window.rstrip("/").split("/")[-1]


def staged_entries(window, plan_path):
    """The full staged file list as (role, source path, absolute DGX path, literal bytes)."""
    src_root = "/data/occ3-bridge/icalib-src-" + window
    live_root = "/data/occ3-bridge/v3-live-" + window
    entries = [_entry("build-context", EXT_ROOT / "pyproject.toml", src_root + "/pyproject.toml"),
               _entry("build-context", EXT_ROOT / "README.md", src_root + "/README.md"),
               _entry("build-context", EXT_ROOT / "deploy" / "Dockerfile",
                      src_root + "/deploy/Dockerfile")]
    for name in plugin_files()[0]:
        entries.append(_entry("build-context", PLUGIN_DIR / name, src_root + "/occ3_sglang/" + name))
    for name in BUILD_UTILITIES:
        entries.append(_entry("build-context", EXT_ROOT / "deploy" / "v3" / name,
                              src_root + "/deploy/v3/" + name))
    for name in CONTROLLER_MODULES:
        entries.append(_entry("controller-cli", EXT_ROOT / "deploy" / "v3" / name,
                              src_root + "/deploy/v3/" + name))
    for name in PATCHES:
        entries.append(_entry("patch-record", EXT_ROOT / "patches" / name, src_root + "/patches/" + name))
    for name in BUILD_INPUTS:
        entries.append(_entry("build-input", EXT_ROOT / "deploy" / "native-build-inputs" / name,
                              src_root + "/deploy/native-build-inputs/" + name))
    entries.append(_entry("plan", Path(plan_path), live_root + "/plan.json"))
    entries.append(_entry("instrumentation", INST_DIR / "occ3_p3_arms_hook.py",
                          live_root + "/extras/site-packages/occ3_p3_arms_hook.py"))
    entries.append(_entry("instrumentation", INST_DIR / "zz_occ3_p3_arms.pth",
                          live_root + "/extras/site-packages/zz_occ3_p3_arms.pth"))
    for arm in ARM_ORDER:
        entries.append(_entry("arm-control", None, live_root + "/arms/control-" + arm + ".txt",
                              ARM_TOKENS[arm]))
    return entries


def _arm_order_document(window, hook_digests):
    return {
        "schema": "occ3.icalib.p3-arm-order.v1",
        "window": window,
        "control_path": CONTROL_PATH,
        "control_path_env": "OCC3_P3_ARM_FILE",
        "control_path_note": ("fixed once at boot; the window uses the default path, and the knob is "
                              "deliberately NOT a plan environment addition"),
        "arm_order": list(ARM_ORDER),
        "tokens": {arm: token.decode("ascii").strip() or None
                   for arm, token in ARM_TOKENS.items()},
        "operator_step": dict(ARM_OPERATOR_STEP),
        "one_attempt_per_arm": True,
        "log_line": "occ3_p3_arm=<token|none> file_sha256=<hex|none> mtime_ns=<int>",
        "p5_exclusion": ("P3 families are excluded from the P5 fit by an explicit turn-id list "
                         "recorded in the window evidence; they are not removed from the corpus."),
        "hook_digests": hook_digests,
    }


def _verify(entries, plan, window):
    _require(len(PATCHES) == 3, "patch_set_size")
    profile = EXT_ROOT / "deploy" / "native-build-inputs" / "native-profile-input.json"
    baseline = EXT_ROOT / "deploy" / "native-build-inputs" / "source-baseline-expanded.json"
    _require(sha256_file(profile) == PROFILE_SHA256, "profile_digest_mismatch", sha256_file(profile))
    _require(sha256_file(baseline) == BASELINE_SHA256, "baseline_digest_mismatch",
             sha256_file(baseline))
    _require(plan["native_validator"]["method_id"] == PER_MESSAGE_METHOD, "plan_method_not_per_message")
    _require("static_input_sha256" not in plan["native_validator"], "plan_carries_static_input_field")
    _require(plan["static_input"]["sha256"] == PROFILE_SHA256, "plan_static_input_mismatch")
    _require(plan["candidate_image_id"] != plan["original_image_id"], "plan_candidate_equals_original")
    additions = [item["name"] for item in plan["environment_additions"]]
    _require(additions == ["OCC3_CALIBRATION_CAPTURE_DIR", "OCC3_CALIBRATION_FIT_TURNS",
                           "OCC3_CALIBRATION_VALIDATION_TURNS", "OCC3_CALIBRATION_TEST_TURNS",
                           "OCC3_MAX_TURN_SECONDS", "SGLANG_USE_PICKLE_IPC"],
             "plan_environment_additions_changed", ",".join(additions))
    corpus = plan["environment_additions"][0]["value"]
    _require(corpus == "/data/occ3-bridge/icalib-corpus/" + window, "plan_corpus_dir_mismatch", corpus)
    for entry in entries:
        if entry["source"] is None:
            _require(isinstance(entry["content"], bytes), "arm_control_missing")
            _require(len(entry["content"]) <= 64, "arm_control_oversize")
            continue
        source = Path(entry["source"])
        _require(source.is_file(), "staged_source_missing", str(source))
        size = source.stat().st_size
        _require(size <= MAX_FILE_BYTES, "staged_file_oversize", str(source))
        if source.suffix == ".py":
            _require(size <= MAX_PY_BYTES, "staged_python_oversize", str(source))
        text = str(source).replace("\\", "/")
        for forbidden in ("/tests/", "/benchmarks/", "/__pycache__/", ".pytest_cache"):
            _require(forbidden not in text, "staged_forbidden_path", text)
        _require(not text.endswith("/nul"), "staged_forbidden_path", text)


def build(window, plan_path):
    """Return (entries, payloads, manifest) with every digest computed."""
    plan, declared_window = plan_window_id(plan_path)
    _require(declared_window == window, "window_mismatch", declared_window)
    entries = staged_entries(window, plan_path)
    _verify(entries, plan, window)
    hook_digests = {name: sha256_file(INST_DIR / name) for name in HOOK_FILES}
    arm_order = _arm_order_document(window, hook_digests)
    entries.append(_entry("arm-control", None,
                          "/data/occ3-bridge/v3-live-" + window + "/arms/arm-order.json",
                          json.dumps(arm_order, indent=1, sort_keys=True).encode() + b"\n"))
    files = []
    for entry in entries:
        data = entry["content"] if entry["source"] is None else Path(entry["source"]).read_bytes()
        files.append({"role": entry["role"],
                      "dgx_path": entry["dgx_path"],
                      "source": None if entry["source"] is None else
                                str(Path(entry["source"]).relative_to(EXT_ROOT)).replace("\\", "/"),
                      "bytes": len(data),
                      "sha256": sha256_bytes(data),
                      "_data": data})
    src_root = "/data/occ3-bridge/icalib-src-" + window
    sha_lines = []
    for item in files:
        if item["dgx_path"].startswith(src_root + "/"):
            sha_lines.append("%s  %s" % (item["sha256"], item["dgx_path"][len(src_root) + 1:]))
    sha_lines.sort(key=lambda line: line.split("  ", 1)[1])
    manifest_sh = ("\n".join(sha_lines) + "\n").encode()
    files.append({"role": "manifest", "dgx_path": src_root + "/stage-manifest.sha256",
                  "source": None, "bytes": len(manifest_sh), "sha256": sha256_bytes(manifest_sh),
                  "_data": manifest_sh})
    payloads = [{"index": index, "role": item["role"], "dgx_path": item["dgx_path"],
                 "bytes": item["bytes"], "sha256": item["sha256"],
                 "chunk_chars": CHUNK_CHARS,
                 "chunks": b64_chunks(item["_data"])}
                for index, item in enumerate(files)]
    manifest_body = {
        "schema": "occ3.icalib.build-staging.v1",
        "window": window,
        "worktree": str(EXT_ROOT).replace("\\", "/"),
        "src_root": src_root,
        "live_root": "/data/occ3-bridge/v3-live-" + window,
        "site_packages": SITE_PACKAGES,
        "instrumentation": {
            "schema": "occ3.icalib.instrumentation.v1",
            "declared": "unattested site-packages helper, not part of the attested plugin set",
            "files": {name: {"sha256": digest,
                             "bytes": (INST_DIR / name).stat().st_size,
                             "install_target": SITE_PACKAGES + "/" + name}
                      for name, digest in hook_digests.items()},
            "install_step": ("docker cp the two extras into the STOPPED candidate container at "
                             "SITE_PACKAGES between S4 (preflight) and S6 (swap); verify both "
                             "digests inside the container before the swap"),
            "control_path": CONTROL_PATH,
        },
        "excluded": {"local_directories": plugin_files()[1],
                     "local_files": plugin_files()[2],
                     "never_staged": ["occ3_sglang/__pycache__", "occ3_sglang/*.pyc",
                                      "tests/", "benchmarks/", "nul"]},
        "counts": {"files": len(files),
                   "plugin_modules": PLUGIN_FILE_COUNT,
                   "patch_records": len(PATCHES),
                   "controller_modules": len(CONTROLLER_MODULES),
                   "build_utilities": len(BUILD_UTILITIES),
                   "arms": len(ARM_ORDER),
                   "bytes": sum(item["bytes"] for item in files),
                   "chunks": sum(len(item["chunks"]) for item in payloads)},
        "pins": {"native_profile_input_sha256": PROFILE_SHA256,
                 "source_baseline_expanded_sha256": BASELINE_SHA256},
        "files": [{key: item[key] for key in ("role", "dgx_path", "source", "bytes", "sha256")}
                  for item in files],
    }
    manifest_digest = sha256_bytes(json.dumps(manifest_body, sort_keys=True,
                                              separators=(",", ":")).encode())
    manifest_body["manifest_sha256"] = manifest_digest
    return files, payloads, manifest_body


def write_staging(files, payloads, manifest, out_dir):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    (out / "live").mkdir(parents=True)
    for item, payload in zip(files, payloads):
        if item["dgx_path"].startswith(manifest["src_root"] + "/"):
            target = out / "src" / item["dgx_path"][len(manifest["src_root"]) + 1:]
        else:
            target = out / "live" / item["dgx_path"][len(manifest["live_root"]) + 1:]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item["_data"])
        _require(sha256_file(target) == item["sha256"], "write_digest_mismatch", str(target))
        _require(b"".join(base64.b64decode(chunk) for chunk in payload["chunks"]) == item["_data"],
                 "payload_roundtrip_failed", item["dgx_path"])
    (out / "payloads.json").write_text(json.dumps(
        {"schema": "occ3.icalib.stage-payloads.v1", "window": manifest["window"],
         "chunk_chars": CHUNK_CHARS, "files": payloads}, indent=1) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")
    return out


def self_check(files, payloads):
    for item, payload in zip(files, payloads):
        _require(payload["sha256"] == item["sha256"], "payload_digest_mismatch", item["dgx_path"])
        rebuilt = b"".join(base64.b64decode(chunk) for chunk in payload["chunks"])
        _require(sha256_bytes(rebuilt) == item["sha256"], "payload_roundtrip_failed",
                 item["dgx_path"])
        _require(all(len(chunk) <= CHUNK_CHARS for chunk in payload["chunks"]),
                 "payload_chunk_too_long", item["dgx_path"])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window-id", default=None,
                        help="icalib-YYYYMMDDTHHMMSSZ (default: read from the plan)")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--out", default=None, help="staging output directory")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args(argv)
    try:
        window = args.window_id or plan_window_id(args.plan)[1]
        files, payloads, manifest = build(window, args.plan)
        self_check(files, payloads)
    except Refusal as error:
        print(json.dumps({"ok": False, "code": error.code, "detail": error.detail}))
        return 2
    report = {"ok": True, "window": window, "dry_run": bool(args.dry_run),
              "counts": manifest["counts"],
              "manifest_sha256": manifest["manifest_sha256"],
              "instrumentation": manifest["instrumentation"]["files"],
              "src_root": manifest["src_root"], "live_root": manifest["live_root"],
              "self_check": "payload chunks reassemble to the staged bytes"}
    if args.dry_run:
        report["files"] = manifest["files"]
    else:
        out_dir = args.out or str(Path(os.environ.get("TEMP", "/tmp")) / "occ3-icalib-stage" / window)
        out = write_staging(files, payloads, manifest, out_dir)
        report["out"] = str(out).replace("\\", "/")
        report["payloads"] = str((out / "payloads.json")).replace("\\", "/")
        report["manifest"] = str((out / "manifest.json")).replace("\\", "/")
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print("window            %s" % report["window"])
        print("dry_run           %s" % report["dry_run"])
        print("files/chunks      %d / %d  (%d bytes)"
              % (report["counts"]["files"], report["counts"]["chunks"], report["counts"]["bytes"]))
        print("manifest_sha256   %s" % report["manifest_sha256"])
        for name, item in sorted(report["instrumentation"].items()):
            print("instrumentation   %-24s %s (%d bytes)"
                  % (name, item["sha256"], item["bytes"]))
        if report["dry_run"]:
            print("")
            for item in report["files"]:
                print("  %-16s %s  %8d  %s" % (item["role"], item["dgx_path"], item["bytes"],
                                               item["sha256"]))
        else:
            print("out               %s" % report["out"])
            print("payloads          %s" % report["payloads"])
            print("manifest          %s" % report["manifest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
