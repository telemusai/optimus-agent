"""Offline adapter for the method-specific native phase-suite validator.

Zero HTTP and zero credential resolution. This module only mints per-phase
challenges, verifies strictly bound harness reports and admits offline pinned
inputs. It never contacts the candidate, never resolves helper credentials,
never executes the Windows harness and never claims GPU proof. The report is
produced by the root-executed native harness under the challenge written here.

Selected only by an explicit reviewed plan key in lifecycle.Services.validator.
Absent key keeps the blocked LiveValidator; unknown key fails closed.

Report publication contract: the final report file must be published atomically
at its exact expected name only when complete; a missing final name waits
bounded, a malformed final name is an immediate protocol violation refusal, and
an existing final target is immutable (no replay). All control/admission/report
JSON is STRICT UTF-8 ONLY: the byte bound is enforced first, then a strict
UTF-8 decode (no autodetection, no BOM, no replacement) precedes the
depth-bounded scan (MAX_JSON_DEPTH) and parsing, so UTF-16/UTF-32 or otherwise
invalid encodings are refused as a fixed value-free malformed_json. A
RecursionError is translated to a fixed refusal. The after-phase restart gate
is the controlled profile: fresh candidate epoch AND advanced
native-incarnation generation with a retained counter; a volatile-reset
runtime is not admitted by this profile.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time
from urllib.parse import urlsplit

VALIDATOR_KEY = "native_handoff_v1"
PHASES = ("before_restart", "after_restart")
REPORT_SCHEMA = "optimus.occ3.native-phase-suite.v1"
CHALLENGE_SCHEMA = "optimus.occ3.validator-phase-challenge.v1"
PIN_SCHEMA = "occ3.admission.pin.v1"
ADMISSION_SCHEMA = "optimus.occ3.native-admission.v1"
CONTROL_SCHEMAS = ("occ3.sessions.enroll.v1", "occ3.publish.v1",
                   "occ3.publication.status.v1", "occ3.retire.v1")
ALLOWED_HARNESS_PLATFORMS = {"windows-x86_64"}
ALLOWED_HEADER_NAMES = {"x-pomerium-authorization"}
FORBIDDEN_HEADER_NAMES = {"authorization", "host", "proxy-authorization",
                          "content-type", "accept-encoding"}
OBSERVED_LAYOUT_FIELDS = ("runtime_epoch", "runtime_generation", "tp_layout_digest",
                          "hidden_dim", "layer_index", "dtype", "expected_ranks", "ready_ranks")
IDENTITY_FIELDS = ("model_id", "architecture", "checkpoint_revision", "weight_fingerprint",
                   "tokenizer_fingerprint", "chat_template_fingerprint", "quantisation",
                   "tensor_geometry", "kv_layout", "latent_method", "runtime")
ADMISSION_FIELDS = ("method_id", "harness_artifact", "source_manifest", "interface_versions",
                    "offline_selftest", "admission_template", "budget_resource_receipt",
                    "max_budget", "layout_strategy", "endpoint", "credential_references",
                    "control_spec")
# A non-default registered method has no in-repo static derivation input file,
# so its plan declares the digest of the per-method static input it was built
# from; the per-phase pins must re-pin exactly that declared digest.
STATIC_INPUT_FIELD = "static_input_sha256"
# The pinned static profile fields every method shares. dtype and
# layout_strategy_digest are admission-time observations and are deliberately
# excluded from the per-boot derivation receipt.
DERIVATION_STATIC_FIELDS = ("model_fingerprint", "model_id", "expected_ranks", "hidden_dim",
                            "layer_index", "max_prefix_tokens", "max_live_bytes",
                            "workspace_bytes", "max_records", "max_active", "max_ttl_seconds",
                            "anchor_alpha", "tile_rows", "method")
CALIBRATION_REFERENCE_FIELDS = ("calibration_path", "calibration_digest")
# The derivation fieldset is the wire fieldset: occ3_sglang.latent_types
# RuntimeProfile always carries the frozen-artifact reference, empty for the
# per-message method, so every method's served profile_digest hashes it. A
# shorter per-method set can never equal the served digest (live defect,
# window icalib-20261029T032600Z: report_profile_derivation_mismatch on every
# honest report).
DERIVATION_PROFILE_FIELDS = DERIVATION_STATIC_FIELDS + CALIBRATION_REFERENCE_FIELDS
# Shared admission source_profile fields for every registered method. The
# registry adds the calibration reference for artifact-requiring methods.
SOURCE_PROFILE_FIELDS = ("dtype", "expected_ranks", "layer_index", "hidden_dim",
                         "layout_strategy_digest", "model_fingerprint", "model_id",
                         "method", "anchor_alpha", "max_prefix_tokens", "max_live_bytes",
                         "workspace_bytes", "max_records", "max_active", "max_ttl_seconds",
                         "tile_rows")
PIN_FIELDS = ("schema", "phase", "attempt_id", "static_input_sha256", "runtime_epoch",
              "runtime_generation", "profile_digest", "written_at")
# Per-phase pin files re-pin the static derivation input by file digest. The
# registry-default per-message method pins the reviewed file
# deploy/native-build-inputs/native-profile-input.json; every other registered
# method re-pins the per-method static input digest its plan declared.
PIN_STATIC_INPUT_SHA256 = "6e676f1c6fc795ba412c4327168b51c8ef6f65056664638c4954eec58e75348a"
REPORT_COMMON_FIELDS = ("schema", "phase", "attempt_id", "candidate_id",
                         "candidate_started_at", "validator_challenge", "harness", "binding",
                         "observed", "suite_nonce_sha256", "cases", "logical_cleanup_confirmed",
                         "gpu_free_proven", "completed_at")
BEFORE_EXTRA_FIELDS = ("negative_control_retained",)
BEFORE_CASES = ("positive", "no_state", "stale_handle_pre")
AFTER_CASES = ("stale_handle_post", "positive", "no_state")
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_CHALLENGE_BYTES = 4096
MAX_PIN_BYTES = 8192
MAX_MANIFEST_FILES = 128
MAX_BINARY_MIRROR_BYTES = 256 * 1024 * 1024
UNKNOWN_HASH = hashlib.sha256(b"UNKNOWN").hexdigest()
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
FINGERPRINT = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
ENV_REFERENCE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")
DTYPES = ("bfloat16", "float16", "float32")
MAX_DEFAULT_TOKENS = 131072
MAX_CEILING_TOKENS = 262144
CONTEXT_WINDOW = 1048576


class _Reject(Exception):
    """Internal control flow: one specific refusal code, never partial state."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _canonical_sha256(value) -> str:
    """Sorted, compact, ASCII-escaped canonical JSON digest (Rust parity)."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Reject("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise _Reject("non_finite_number")


MAX_JSON_DEPTH = 32


def _json_depth_ok(value):
    """Bounded pre-parse depth scan for finite control/admission/report JSON.

    Counts object/array nesting depth outside quoted strings with escape
    handling, over strictly decoded UTF-8 text (a bytes input must itself be
    strict UTF-8; anything else is refused). This is not a heap/RSS bound and
    never applies to model payloads.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
    depth = 0
    in_string = False
    escaped = False
    for char in value:
        code = ord(char)
        if in_string:
            if escaped:
                escaped = False
            elif code == 0x5C:  # backslash
                escaped = True
            elif code == 0x22:  # closing quote
                in_string = False
            continue
        if code == 0x22:  # opening quote
            in_string = True
        elif code in (0x7B, 0x5B):  # { or [
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return False
        elif code in (0x7D, 0x5D):  # } or ]
            depth -= 1
            if depth < 0:
                return False
    return not in_string and depth == 0


def _strict_loads(data, limit):
    """Strict UTF-8-only document loader: byte bound, decode, depth, parse.

    The byte bound is enforced FIRST. The bytes are then decoded as strict
    UTF-8 with no autodetection, no BOM stripping and no replacement, so a
    UTF-16/UTF-32 document (whose JSON structural bytes would desynchronize a
    raw ASCII scan and whose depth could bypass the bound) is refused as a
    fixed value-free malformed_json. Parsing then runs on the decoded text so
    json.loads encoding autodetection cannot re-enter.
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) > limit:
        raise _Reject("file_too_large")
    try:
        text = bytes(data).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _Reject("malformed_json") from None
    if text.startswith("\ufeff"):
        # Explicit fail-closed BOM policy: no BOM is accepted.
        raise _Reject("malformed_json")
    if not _json_depth_ok(text):
        raise _Reject("malformed_json")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates,
                          parse_constant=_reject_constant)
    except _Reject:
        raise
    except (ValueError, RecursionError):
        # RecursionError is translated defensively: the depth guard above bounds
        # the normal case; anything still escaping the parser is a malformed
        # document, never a public-API exception.
        raise _Reject("malformed_json") from None


def _safe_read(path, limit):
    """Bounded, symlink/reparse/hardlink-checked read of one private file."""
    path = Path(path)
    try:
        info = path.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise _Reject("symlink_refused")
    attributes = getattr(info, "st_file_attributes", 0)
    if attributes and attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise _Reject("reparse_point_refused")
    if info.st_nlink > 1:
        raise _Reject("extra_hardlink_refused")
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise _Reject("file_too_large")
    with path.open("rb") as handle:
        return handle.read(info.st_size)


def _require(condition, code):
    if not condition:
        raise _Reject(code)


def _uint(value, code, *, positive=False):
    _require(type(value) is int and not isinstance(value, bool) and value >= 0
             and (value > 0 if positive else True), code)
    return value


def _exact_fields(value, fields, code):
    _require(isinstance(value, dict) and set(value) == set(fields), code)
    return value


def _sha_field(value, code):
    _require(isinstance(value, str) and HEX64.fullmatch(value), code)
    return value


def _fingerprint(value, code):
    _require(isinstance(value, str) and FINGERPRINT.fullmatch(value), code)
    return value


def _method_registry():
    """The shared method registry: the only place a native method id is defined.

    Imported lazily so the offline adapter itself stays import-light; the
    registry module is metadata-only (no torch, no sglang). Any registry
    failure is a fixed refusal, never a bypass to a hardcoded method.
    """
    try:
        from occ3_sglang import native_profile
    except ImportError:
        raise _Reject("method_registry_unavailable") from None
    return native_profile


def _method_descriptor(method_id):
    """Registry descriptor for one admitted method id, or a fixed refusal."""
    registry = _method_registry()
    try:
        descriptor = registry.method_descriptor(method_id)
    except Exception:
        raise _Reject("method_mismatch") from None
    return registry, method_id, descriptor


def _method_profile_fields(descriptor):
    """Exact admission source_profile fieldset for one registered method.

    Shared static fields for every method, plus the frozen-artifact
    reference exactly when the registry says the method requires a
    calibration artifact, plus any extra per-method profile fields the
    registry descriptor declares.
    """
    fields = list(SOURCE_PROFILE_FIELDS)
    if descriptor["requires_calibration_artifact"]:
        fields += list(CALIBRATION_REFERENCE_FIELDS)
    fields += sorted(frozenset(descriptor.get("profile_fields") or ()))
    return tuple(fields)


def _string(value, code, *, limit=256, allow_empty=False):
    _require(isinstance(value, str) and (bool(value) or allow_empty)
             and len(value) <= limit, code)
    return value


def _origin(value, code):
    _require(isinstance(value, str), code)
    parsed = urlsplit(value)
    _require(parsed.scheme == "https" and parsed.hostname
             and not parsed.username and not parsed.password
             and not parsed.query and not parsed.fragment
             and not parsed.path.rstrip("/"), code)
    return "%s://%s" % (parsed.scheme, parsed.netloc)


def _reference(value, code):
    """Reference format only. Helper values are never resolved or logged."""
    _require(isinstance(value, str) and 0 < len(value) <= 512, code)
    _require(value.startswith("!") or ENV_REFERENCE.fullmatch(value), code)
    return value


def _phase_report_path(directory, phase):
    return Path(directory) / ("native-phase-suite-" + phase + ".json")


def _phase_challenge_path(directory, phase):
    return Path(directory) / ("validator-challenge-" + phase + ".json")


def _phase_pin_path(directory, phase):
    return Path(directory) / ("admission-pin-" + phase + ".json")


def _started_at_of(challenge):
    started = challenge.get("expected_candidate_started_at")
    _string(started, "challenge_started_at_invalid", limit=64)
    return started


def _derivation_value(profile, observed):
    """The exact RuntimeProfile fieldset the wire binding digest hashes.

    occ3_sglang.latent_types.RuntimeProfile always carries the frozen-artifact
    reference, so the fieldset is uniform for every registered method: the
    pinned static fields, the reference pair and the observed incarnation. A
    per-message profile names no artifact, which is the empty pair the wire
    carries, never an absent field; any other missing field fails closed.
    """
    value = {}
    for name in DERIVATION_PROFILE_FIELDS:
        if name in profile:
            value[name] = profile[name]
        else:
            _require(name in CALIBRATION_REFERENCE_FIELDS,
                     "profile_derivation_field_missing")
            value[name] = ""
    value["runtime_epoch"] = observed["runtime_epoch"]
    value["runtime_generation"] = observed["runtime_generation"]
    return value


def _derived_profile_digest(profile, observed):
    """Per-boot RuntimeProfile binding digest for one observed incarnation.

    Parity contract (occ3_sglang.attachment_contract.digest over
    dataclasses.asdict(RuntimeProfile)): sha256 of the canonical JSON -
    sorted keys, compact separators, ASCII escapes, NaN refused - of the
    pinned STATIC profile fields plus runtime_epoch/runtime_generation as
    observed in the harness report. The fieldset is the wire fieldset
    (DERIVATION_PROFILE_FIELDS), identical for every registered method: the
    per-message method binds the empty calibration reference and an
    artifact-requiring method binds its own, so a swapped artifact can never
    reuse a readiness record. The wire profile_digest is incarnation-bound
    (fresh per boot), so it is recomputed here as the anti-fabrication
    derivation receipt and never statically pinned.
    """
    _method_descriptor(profile["method"])
    value = _derivation_value(profile, observed)
    try:
        body = json.dumps(value, separators=(",", ":"), sort_keys=True,
                          ensure_ascii=True, allow_nan=False)
    except ValueError:
        raise _Reject("profile_derivation_unserializable") from None
    return hashlib.sha256(body.encode("ascii")).hexdigest()


class NativeHandoffValidator:
    """Offline admission plus per-phase challenge/report supervision."""

    def __init__(self, plan, journal, state, *, timeout=None, clock=time.monotonic,
                 sleep=time.sleep, challenge_factory=lambda: secrets.token_hex(32)):
        self.plan, self.journal = plan, journal
        self.timeout = timeout if timeout is not None else plan.get("validation_timeout_s", 900)
        self._clock, self._sleep = clock, sleep
        self._challenge_factory = challenge_factory
        self._state = state
        self._before_binding = None
        # Validated method identity, set once _pinned_inputs consults the
        # registry; results before validation report the declared value only.
        self._validated_method = None

    def _declared_method(self):
        block = self.plan.get("native_validator") if isinstance(self.plan, dict) else None
        value = block.get("method_id") if isinstance(block, dict) else None
        return value if isinstance(value, str) else None

    def _method(self):
        """The plan's validated registry method (id, registry, descriptor)."""
        if self._validated_method is not None:
            return self._validated_method
        self._validated_method = _method_descriptor(self._declared_method())
        return self._validated_method

    def _expected_static_input(self):
        """Static derivation input digest this plan's per-phase pins re-pin."""
        registry, method_id, _ = self._method()
        block = self.plan["native_validator"]
        if method_id == registry.PER_MESSAGE_METHOD:
            return PIN_STATIC_INPUT_SHA256
        _sha_field(block[STATIC_INPUT_FIELD], "static_input_digest_invalid")
        return block[STATIC_INPUT_FIELD]

    # -- shared result shapes -------------------------------------------------
    def _result(self, ok, phase, status, code=None, **details):
        method = self._validated_method[1] if self._validated_method else self._declared_method()
        result = {"ok": ok is True, "scope": "native_handoff_method_v1",
                  "validator": VALIDATOR_KEY, "method_id": method,
                  "phase": phase, "status": status, "code": code,
                  "gpu_proof": False, "physical_gpu_free_proven": False,
                  "live_success_claimed": False,
                  "http_calls_performed": 0, "credentials_resolved": 0}
        result.update(details)
        return result

    def _blocked(self, phase, code, **details):
        return self._result(False, phase, "blocked", code, **details)

    # -- offline admission ----------------------------------------------------
    def pre_mutation_admission(self):
        """Offline pinned-input admission. No HTTP, no resolution, no GPU claim."""
        try:
            pinned = self._pinned_inputs()
            mirror = self._binary_mirror(pinned)
            return self._result(
                True, "pre_mutation_admission", "admitted",
                admission_schema=ADMISSION_SCHEMA,
                admission_scope="offline_inputs_only",
                gpu_proof="not_asserted_preflight",
                runtime_binding="deferred_to_post_start",
                bindings={
                    "harness_binary_sha256": pinned["harness_artifact"]["binary_sha256"],
                    "source_manifest_files": len(pinned["source_manifest"]["files"]),
                    "identity_fingerprint": pinned["admission_template"]["fingerprint"],
                    "profile_digest_source": "per-phase admission pin files (journal "
                                              "admission-pin-<phase>.json, write-once after the "
                                              "identity derivation receipt; unequal per phase by design)",
                    "budget_receipt_sha256": pinned["budget_resource_receipt"]["receipt_sha256"],
                    "endpoint_origin": pinned["endpoint"]["origin"]},
                binary_mirror_verified=mirror,
                note="real admission stays blocked until root supplies pinned inputs")
        except _Reject as error:
            return self._blocked("pre_mutation_admission", error.code)

    def _pinned_inputs(self):
        block = self.plan.get("native_validator")
        _require(isinstance(block, dict), "native_validator_inputs_missing")
        # Depth-bound the admission document itself before structural checks.
        serialized = json.dumps(block, default=str)
        _require(_json_depth_ok(serialized.encode("utf-8")), "malformed_json")
        # Registry-driven method admission: only ids the installed registry
        # advertises validate; a bogus or missing method string is refused.
        registry, method_id, descriptor = _method_descriptor(block.get("method_id"))
        fields = ADMISSION_FIELDS
        if method_id != registry.PER_MESSAGE_METHOD:
            fields = ADMISSION_FIELDS + (STATIC_INPUT_FIELD,)
        _exact_fields(block, fields, "admission_fieldset_invalid")
        if STATIC_INPUT_FIELD in fields:
            _sha_field(block[STATIC_INPUT_FIELD], "static_input_digest_invalid")
        self._validated_method = (registry, method_id, descriptor)
        self._check_harness(block["harness_artifact"])
        self._check_source_manifest(block["source_manifest"])
        self._check_interface_versions(block["interface_versions"])
        self._check_offline_selftest(block["offline_selftest"], block["harness_artifact"])
        self._check_admission_template(block["admission_template"], method_id, descriptor)
        self._check_budget(block["budget_resource_receipt"], block["max_budget"])
        self._check_layout_strategy(block["layout_strategy"])
        self._check_endpoint(block["endpoint"])
        self._check_credential_references(block["credential_references"])
        self._check_control_spec(block["control_spec"])
        return block

    def _check_harness(self, value):
        value = _exact_fields(value, ("binary_sha256", "platform", "build_manifest_sha256"),
                              "harness_artifact_invalid")
        _sha_field(value["binary_sha256"], "harness_binary_sha256_invalid")
        _sha_field(value["build_manifest_sha256"], "harness_build_manifest_invalid")
        _require(value["platform"] in ALLOWED_HARNESS_PLATFORMS, "harness_platform_unsupported")

    def _check_source_manifest(self, value):
        value = _exact_fields(value, ("files", "cargo_lock_sha256"), "source_manifest_invalid")
        files = value["files"]
        _require(isinstance(files, dict) and 0 < len(files) <= MAX_MANIFEST_FILES,
                 "source_manifest_invalid")
        for name, digest in files.items():
            _require(isinstance(name, str) and name.endswith(".rs")
                     and not name.startswith(("/", "\\")) and ".." not in name
                     and ":" not in name, "source_manifest_path_invalid")
            _sha_field(digest, "source_manifest_digest_invalid")
        _sha_field(value["cargo_lock_sha256"], "cargo_lock_digest_invalid")

    def _check_interface_versions(self, value):
        value = _exact_fields(value, ("report", "challenge", "control"),
                              "interface_versions_invalid")
        _require(value["report"] == REPORT_SCHEMA, "report_schema_version_unknown")
        _require(value["challenge"] == CHALLENGE_SCHEMA, "challenge_schema_version_unknown")
        _require(tuple(value["control"]) == CONTROL_SCHEMAS, "control_schema_version_unknown")

    def _check_offline_selftest(self, value, harness):
        value = _exact_fields(value, ("exit_code", "report_sha256", "binary_sha256"),
                              "offline_selftest_invalid")
        _require(value["exit_code"] == 0, "offline_selftest_not_clean")
        _sha_field(value["report_sha256"], "offline_selftest_digest_invalid")
        _require(value["binary_sha256"] == harness["binary_sha256"],
                 "offline_selftest_binary_mismatch")

    def _check_admission_template(self, value, method_id, descriptor):
        value = _exact_fields(value, ("identity", "fingerprint", "source_profile"),
                              "admission_template_invalid")
        identity = _exact_fields(value["identity"], IDENTITY_FIELDS, "identity_fieldset_invalid")
        for field in IDENTITY_FIELDS:
            _string(identity[field], "identity_field_invalid")
        _require(identity["latent_method"] == method_id, "identity_method_mismatch")
        _require(value["fingerprint"] == "sha256:" + _canonical_sha256(identity),
                 "identity_fingerprint_mismatch")
        _fingerprint(value["fingerprint"], "identity_fingerprint_invalid")
        # Per-method fieldset: every registered method shares the static
        # profile fields; the registry adds the frozen-artifact reference
        # exactly for methods that require a calibration artifact (and never
        # for the per-message method, which consumes none).
        profile = _exact_fields(value["source_profile"],
                                _method_profile_fields(descriptor), "source_profile_invalid")
        _require(profile["dtype"] in DTYPES, "source_dtype_unsupported")
        _require(profile["expected_ranks"] == 8, "expected_ranks_not_tp8")
        _uint(profile["layer_index"], "layer_index_invalid")
        _uint(profile["hidden_dim"], "hidden_dim_invalid", positive=True)
        _sha_field(profile["layout_strategy_digest"], "layout_strategy_digest_invalid")
        # profile_digest is incarnation-bound (fresh per boot) and the live
        # server epoch exists only after the candidate boots, so the per-phase
        # OBSERVED pins moved OUT of this pre-boot-frozen admission: they
        # arrive as write-once attempt-dir pin files (admission-pin-<phase>.json,
        # _load_phase_pin) verified per phase in _verify_report.
        # Static derivation fieldset (per method; the per-message set equals
        # native-build-inputs/native-profile-input.json); recomputed per phase.
        _string(profile["model_fingerprint"], "source_model_fingerprint_invalid", limit=128)
        _string(profile["model_id"], "source_model_id_invalid", limit=128)
        _require(profile["method"] == method_id, "source_method_mismatch")
        if descriptor["requires_calibration_artifact"]:
            path = profile["calibration_path"]
            _require(isinstance(path, str) and 0 < len(path) <= 512 and path.isprintable()
                     and os.path.isabs(path), "source_calibration_path_invalid")
            _sha_field(profile["calibration_digest"], "source_calibration_digest_invalid")
        for name in sorted(frozenset(descriptor.get("profile_fields") or ())):
            extra = profile[name]
            _require(type(extra) in (int, float, str) and not isinstance(extra, bool)
                     and (math.isfinite(extra) if type(extra) is float
                          else len(str(extra)) <= 256), "source_profile_invalid")
        _require(type(profile["anchor_alpha"]) in (int, float)
                 and math.isfinite(profile["anchor_alpha"]), "source_anchor_alpha_invalid")
        _uint(profile["max_prefix_tokens"], "max_prefix_tokens_invalid", positive=True)
        _uint(profile["max_live_bytes"], "max_live_bytes_invalid", positive=True)
        _uint(profile["workspace_bytes"], "workspace_bytes_invalid", positive=True)
        _uint(profile["max_records"], "max_records_invalid", positive=True)
        _uint(profile["max_active"], "max_active_invalid", positive=True)
        _uint(profile["max_ttl_seconds"], "max_ttl_seconds_invalid", positive=True)
        _uint(profile["tile_rows"], "tile_rows_invalid", positive=True)

    def _check_budget(self, receipt, budget):
        receipt = _exact_fields(receipt, ("root_reviewed", "receipt_sha256", "review_digest"),
                                 "budget_receipt_invalid")
        # A self-asserted boolean can never substitute the reviewed receipt.
        _require(receipt["root_reviewed"] is True, "budget_receipt_not_root_reviewed")
        _sha_field(receipt["receipt_sha256"], "budget_receipt_digest_invalid")
        _sha_field(receipt["review_digest"], "budget_review_digest_invalid")
        budget = _exact_fields(budget, ("default_max_tokens", "max_tokens", "context_window",
                                        "reasoning", "thinking_level"), "max_budget_invalid")
        _require(budget["default_max_tokens"] == MAX_DEFAULT_TOKENS, "default_max_tokens_changed")
        _require(budget["max_tokens"] == MAX_CEILING_TOKENS, "max_ceiling_changed")
        _require(budget["context_window"] == CONTEXT_WINDOW, "context_window_changed")
        _require(budget["reasoning"] is True, "reasoning_disabled")
        _require(budget["thinking_level"] == "max", "thinking_level_not_max")

    def _check_layout_strategy(self, value):
        value = _exact_fields(value, ("observed_fields", "bound_at"), "layout_strategy_invalid")
        _require(value["bound_at"] == "post_start_enrollment", "layout_bound_not_post_start")
        fields = value["observed_fields"]
        _require(isinstance(fields, list) and set(OBSERVED_LAYOUT_FIELDS) <= set(fields)
                 and len(fields) == len(set(fields)), "layout_strategy_fields_invalid")

    def _check_endpoint(self, value):
        value = _exact_fields(value, ("origin", "inference_path", "control_root"),
                              "endpoint_invalid")
        origin = _origin(value["origin"], "endpoint_origin_not_https")
        _require(value["inference_path"] == "/v1", "inference_path_invalid")
        _require(value["control_root"] == "/occ3/v1/native/", "control_root_invalid")
        value["origin"] = origin

    def _check_credential_references(self, value):
        value = _exact_fields(value, ("api_key", "headers"), "credential_references_invalid")
        _reference(value["api_key"], "api_key_reference_invalid")
        headers = value["headers"]
        _require(isinstance(headers, dict) and headers, "credential_headers_invalid")
        for name, reference in headers.items():
            _require(isinstance(name, str) and name.lower() in ALLOWED_HEADER_NAMES
                     and name.lower() not in FORBIDDEN_HEADER_NAMES, "header_name_not_allowed")
            _reference(reference, "header_reference_invalid")

    def _check_control_spec(self, value):
        value = _exact_fields(value, ("before_cases", "after_cases", "one_shot", "no_replay",
                                      "max_attempts"), "control_spec_invalid")
        _require(tuple(value["before_cases"]) == BEFORE_CASES, "before_cases_invalid")
        _require(tuple(value["after_cases"]) == AFTER_CASES, "after_cases_invalid")
        _require(value["one_shot"] is True and value["no_replay"] is True, "replay_allowed")
        _require(value["max_attempts"] == 1, "multiple_attempts_allowed")

    def _binary_mirror(self, pinned):
        """Optionally verify mirrored harness bytes. Never a Linux execution claim."""
        data = _safe_read(Path(self.journal.directory) / "harness-binary-mirror",
                          MAX_BINARY_MIRROR_BYTES)
        if data is None:
            return "absent_pinned_digest_only"
        _require(hashlib.sha256(data).hexdigest()
                 == pinned["harness_artifact"]["binary_sha256"], "binary_mirror_digest_mismatch")
        return "verified_mirrored_bytes_no_execution"

    # -- per-phase challenge and report --------------------------------------
    def _fresh_state(self):
        state = self.journal.read()
        _require(isinstance(state, dict) and state.get("attempt_id"), "journal_unreadable")
        return state

    def _require_gates(self, state, phase):
        gates = state.get("gates", {})
        required = ("service_initial", "native_initial") if phase == "before_restart" \
            else ("live_before", "restart")
        for gate in required:
            _require(isinstance(gates.get(gate), dict)
                     and gates[gate].get("ok") is True, "phase_gate_missing:" + gate)

    def _require_candidate(self, state, phase):
        _require(state.get("candidate_id") and state.get("candidate_started") is True,
                 "candidate_not_started")
        started_at = state.get("candidate_epoch")
        _string(started_at, "candidate_epoch_missing", limit=64)
        if phase == "after_restart":
            _require(state.get("restart_done") is True, "restart_not_completed")
            _require(state.get("restart_count") == 1, "restart_count_invalid")
        return started_at

    def _write_challenge(self, phase, state, started_at):
        path = _phase_challenge_path(self.journal.directory, phase)
        _require(not path.exists(), "challenge_already_exists_no_replay")
        document = {
            "schema": CHALLENGE_SCHEMA, "phase": phase,
            "attempt_id": state["attempt_id"], "candidate_id": state["candidate_id"],
            "expected_candidate_started_at": started_at,
            "model_fingerprint": self._pinned_fingerprint(),
            "method_id": self._method()[1],
            "expected_ranks": self._pinned_ranks(),
            "challenge": self._challenge_factory(),
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + int(self.timeout) + 60,
        }
        from .lifecycle import private_json
        private_json(path, document, exclusive=True)
        return document

    def _pinned_fingerprint(self):
        block = self.plan.get("native_validator", {})
        template = block.get("admission_template", {}) if isinstance(block, dict) else {}
        return template.get("fingerprint")

    def _pinned_ranks(self):
        block = self.plan.get("native_validator", {})
        template = block.get("admission_template", {}) if isinstance(block, dict) else {}
        profile = template.get("source_profile", {}) if isinstance(template, dict) else {}
        return profile.get("expected_ranks")

    def _wait_for_report(self, phase):
        """Wait for the COMPLETE final report publication, never a partial one.

        Missing final name: bounded wait. Malformed final name: immediate
        refusal — a published final document is immutable, so retrying or
        ignoring it is a protocol violation, not a transient state.
        """
        deadline = self._clock() + float(self.timeout)
        path = _phase_report_path(self.journal.directory, phase)
        data = None
        while self._clock() < deadline:
            data = _safe_read(path, MAX_REPORT_BYTES)
            if data is not None:
                break
            self._sleep(min(0.2, max(0.0, deadline - self._clock())))
        if data is None:
            raise _Reject("native_phase_suite_report_timeout")
        return _strict_loads(data, MAX_REPORT_BYTES)

    # -- strict report verification ------------------------------------------
    def _verify_report(self, report, phase, challenge, pinned, started_at):
        _require(isinstance(report, dict), "report_not_object")
        fields = REPORT_COMMON_FIELDS + (BEFORE_EXTRA_FIELDS if phase == "before_restart" else ())
        _exact_fields(report, fields, "report_fieldset_invalid")
        _require(report.get("schema") == REPORT_SCHEMA, "report_schema_unknown")
        _require(report.get("phase") == phase, "report_phase_mismatch")
        _require(report.get("attempt_id") == challenge["attempt_id"], "attempt_binding_mismatch")
        _require(report.get("candidate_id") == challenge["candidate_id"], "candidate_binding_mismatch")
        _require(report.get("candidate_started_at") == started_at, "started_at_binding_mismatch")
        _require(report.get("validator_challenge") == challenge["challenge"],
                 "challenge_binding_mismatch")
        harness = _exact_fields(report["harness"],
                                ("binary_sha256", "source_manifest_sha256", "platform"),
                                "report_harness_fieldset_invalid")
        _require(harness["binary_sha256"] == pinned["harness_artifact"]["binary_sha256"],
                 "report_binary_binding_mismatch")
        _require(harness["platform"] == pinned["harness_artifact"]["platform"],
                 "report_platform_mismatch")
        manifest = pinned["source_manifest"]
        expected_manifest = _canonical_sha256(manifest)
        _require(harness["source_manifest_sha256"] == expected_manifest,
                 "report_source_manifest_mismatch")
        binding = _exact_fields(report["binding"],
                                ("identity_fingerprint", "method_id", "profile_digest",
                                 "endpoint_origin"), "report_binding_fieldset_invalid")
        template = pinned["admission_template"]
        _require(binding["identity_fingerprint"] == template["fingerprint"],
                 "report_identity_binding_mismatch")
        _require(binding["method_id"] == self._method()[1], "report_method_mismatch")
        _require(binding["endpoint_origin"] == pinned["endpoint"]["origin"],
                 "report_origin_binding_mismatch")
        observed = self._verify_observed(report["observed"], challenge)
        # profile_digest is incarnation-bound: the binding must equal THIS
        # phase's admission pin file (written into the attempt dir after the
        # identity derivation receipt, never frozen into the pre-boot plan)
        # and must re-derive from the pinned static profile fields plus the
        # phase's observed runtime_epoch/generation.
        profile = template["source_profile"]
        pin = self._load_phase_pin(challenge, phase, observed)
        _require(binding["profile_digest"] == pin["profile_digest"],
                 "report_profile_binding_mismatch")
        _require(_derived_profile_digest(profile, observed) == binding["profile_digest"],
                 "report_profile_derivation_mismatch")
        suite_nonce = _sha_field(report["suite_nonce_sha256"], "suite_nonce_hash_invalid")
        cases = _exact_fields(report["cases"],
                              BEFORE_CASES if phase == "before_restart" else AFTER_CASES,
                              "cases_fieldset_invalid")
        positive = self._verify_positive(cases["positive"], suite_nonce)
        no_state = self._verify_no_state(cases["no_state"], suite_nonce)
        retained = stale_pre = stale_post = None
        if phase == "before_restart":
            stale_pre = self._verify_stale_pre(cases["stale_handle_pre"])
            retained = self._verify_retained(report["negative_control_retained"], observed)
            _require(stale_pre is not None, "stale_pre_case_failed")
        else:
            stale_post = self._verify_stale_post(cases["stale_handle_post"], observed)
        _require(report["logical_cleanup_confirmed"] is True, "logical_cleanup_unconfirmed")
        _require(report["gpu_free_proven"] is False, "gpu_free_overclaimed")
        completed = _uint(report["completed_at"], "completed_at_invalid", positive=True)
        now = int(time.time())
        _require(challenge["created_at"] <= completed <= now + 60, "completion_outside_window")
        _require(completed <= challenge["expires_at"], "completion_after_expiry")
        return {"observed": observed, "suite_nonce_sha256": suite_nonce,
                "positive": positive, "no_state": no_state, "retained": retained,
                "stale_post": stale_post, "pin": pin, "completed_at": completed}

    def _load_phase_pin(self, challenge, phase, observed):
        """Strict-load and bind this phase's admission pin file.

        The admission freezes into the immutable journal plan before the
        candidate boots, but the live server epoch is a fresh uuid4 minted
        only at boot, so per-phase pins can never reach the validator
        through the plan. The executor writes each phase's pin write-once
        into the attempt dir after the identity derivation receipt passes
        and before the harness run; this re-reads it and strictly binds it
        to the challenge attempt, the pinned static input file digest and
        the phase's observed incarnation. Equal before/after pins are
        refused as fabricated pre-restart in _verify_after_chain.
        """
        data = _safe_read(_phase_pin_path(self.journal.directory, phase), MAX_PIN_BYTES)
        _require(data is not None, "pin_file_missing")
        pin = _strict_loads(data, MAX_PIN_BYTES)
        _exact_fields(pin, PIN_FIELDS, "pin_fieldset_invalid")
        _require(pin["schema"] == PIN_SCHEMA, "pin_schema_unknown")
        _require(pin["phase"] == phase, "pin_phase_mismatch")
        _require(pin["attempt_id"] == challenge["attempt_id"], "pin_attempt_binding_mismatch")
        _sha_field(pin["static_input_sha256"], "pin_static_input_invalid")
        _require(pin["static_input_sha256"] == self._expected_static_input(),
                 "pin_static_input_mismatch")
        _string(pin["runtime_epoch"], "pin_epoch_invalid", limit=256)
        _uint(pin["runtime_generation"], "pin_generation_invalid", positive=True)
        _sha_field(pin["profile_digest"], "pin_digest_invalid")
        _uint(pin["written_at"], "pin_written_at_invalid")
        _require(pin["runtime_epoch"] == observed["runtime_epoch"], "pin_epoch_mismatch")
        _require(pin["runtime_generation"] == observed["runtime_generation"],
                 "pin_generation_mismatch")
        return pin

    def _verify_observed(self, value, challenge):
        value = _exact_fields(value, OBSERVED_LAYOUT_FIELDS, "observed_fieldset_invalid")
        _string(value["runtime_epoch"], "runtime_epoch_invalid", limit=256)
        _uint(value["runtime_generation"], "runtime_generation_invalid", positive=True)
        _sha_field(value["tp_layout_digest"], "tp_layout_digest_invalid")
        _uint(value["hidden_dim"], "observed_hidden_dim_invalid", positive=True)
        _uint(value["layer_index"], "observed_layer_index_invalid")
        _require(value["dtype"] in DTYPES, "observed_dtype_unsupported")
        ranks = _uint(value["expected_ranks"], "observed_expected_ranks_invalid", positive=True)
        _require(ranks == challenge["expected_ranks"], "observed_ranks_mismatch")
        _require(value["ready_ranks"] == ranks, "all_ranks_not_ready")
        return value

    def _audit(self, value, code):
        value = _exact_fields(value, ("source_facts_absent", "max_tokens", "request_count"),
                              code)
        _require(value["source_facts_absent"] is True, code + ":source_facts_present")
        _require(value["max_tokens"] == MAX_DEFAULT_TOKENS, code + ":budget_not_default_max")
        _require(value["request_count"] == 1, code + ":request_count_invalid")
        return value

    def _verify_positive(self, value, suite_nonce):
        value = _exact_fields(value, ("case_passed", "semantic_nonce_match", "answer_sha256",
                                      "message_status", "lease_state", "consumed_before_tools",
                                      "receiver_request_audit"), "positive_fieldset_invalid")
        _require(value["case_passed"] is True, "positive_case_failed")
        _require(value["semantic_nonce_match"] is True, "positive_nonce_mismatch")
        # Exact normalized nonce match: the normalized answer hashes to the nonce.
        _require(value["answer_sha256"] == suite_nonce, "positive_answer_not_nonce_hash")
        _require(value["message_status"] == "delivered", "message_not_delivered")
        _require(value["lease_state"] == "Consumed", "lease_not_consumed")
        _require(value["consumed_before_tools"] is True, "consumed_not_before_tools")
        self._audit(value["receiver_request_audit"], "positive_audit_invalid")
        return {"answer_sha256": value["answer_sha256"], "message_status": "delivered",
                "lease_state": "Consumed"}

    def _verify_no_state(self, value, suite_nonce):
        value = _exact_fields(value, ("case_passed", "semantic_nonce_match", "answer_sha256",
                                      "messages", "consumed", "tool_calls",
                                      "receiver_request_audit"), "no_state_fieldset_invalid")
        _require(value["case_passed"] is True, "no_state_case_failed")
        _require(value["semantic_nonce_match"] is False, "no_state_nonce_leaked")
        # The exact reviewed UNKNOWN outcome, not any non-matching answer.
        _require(value["answer_sha256"] == UNKNOWN_HASH, "no_state_not_exact_unknown")
        _require(value["messages"] == 0 and value["consumed"] == 0 and value["tool_calls"] == 0,
                 "no_state_side_effects_present")
        self._audit(value["receiver_request_audit"], "no_state_audit_invalid")
        return {"answer": "UNKNOWN", "nonce_match": False}

    def _verify_stale_pre(self, value):
        value = _exact_fields(value, ("case_passed", "capture_confirmed", "unpublished",
                                      "unretired"), "stale_pre_fieldset_invalid")
        _require(value["case_passed"] is True, "stale_pre_case_failed")
        _require(value["capture_confirmed"] is True, "stale_pre_capture_unconfirmed")
        _require(value["unpublished"] is True and value["unretired"] is True,
                 "stale_pre_not_retained")
        return {"capture_confirmed": True, "unpublished": True, "unretired": True}

    def _verify_retained(self, value, observed):
        value = _exact_fields(value, ("handle_sha256", "runtime_epoch", "runtime_generation",
                                      "expires_at_epoch", "retention_reason", "artifact_sha256"),
                              "retained_fieldset_invalid")
        _sha_field(value["handle_sha256"], "retained_handle_digest_invalid")
        _require(value["runtime_epoch"] == observed["runtime_epoch"], "retained_epoch_mismatch")
        _require(value["runtime_generation"] == observed["runtime_generation"],
                 "retained_generation_mismatch")
        _uint(value["expires_at_epoch"], "retained_expiry_invalid", positive=True)
        _require(value["retention_reason"] == "planned_restart", "retention_reason_invalid")
        _sha_field(value["artifact_sha256"], "retained_artifact_digest_invalid")
        return value

    def _verify_stale_post(self, value, observed):
        value = _exact_fields(value, ("case_passed", "http_status", "error_code", "attempts",
                                      "publication_request_sha256", "artifact_sha256",
                                      "fresh_enrollment_runtime_epoch"),
                              "stale_post_fieldset_invalid")
        _require(value["case_passed"] is True, "stale_post_case_failed")
        _require(value["http_status"] == 409, "stale_refusal_not_409")
        _require(value["error_code"] == "state", "stale_refusal_code_not_state")
        _require(value["attempts"] == 1, "stale_probe_attempted_twice")
        _sha_field(value["publication_request_sha256"], "stale_publication_digest_invalid")
        _sha_field(value["artifact_sha256"], "stale_artifact_digest_invalid")
        _require(value["fresh_enrollment_runtime_epoch"] == observed["runtime_epoch"],
                 "fresh_enrollment_epoch_mismatch")
        return value

    def _load_before_binding(self, pinned):
        if self._before_binding is not None:
            return self._before_binding
        challenge_data = _safe_read(
            _phase_challenge_path(self.journal.directory, "before_restart"), MAX_CHALLENGE_BYTES)
        report_data = _safe_read(
            _phase_report_path(self.journal.directory, "before_restart"), MAX_REPORT_BYTES)
        _require(challenge_data is not None and report_data is not None,
                 "before_phase_binding_unavailable")
        challenge = _strict_loads(challenge_data, MAX_CHALLENGE_BYTES)
        _exact_fields(challenge, CHALLENGE_FIELDS, "challenge_fieldset_invalid")
        report = _strict_loads(report_data, MAX_REPORT_BYTES)
        verified = self._verify_report(report, "before_restart", challenge, pinned,
                                       _started_at_of(challenge))
        self._before_binding = verified
        return verified

    # -- public phase methods -------------------------------------------------
    def before_restart(self):
        try:
            return self._run_phase("before_restart")
        except _Reject as error:
            return self._blocked("before_restart", error.code)

    def after_restart(self):
        try:
            return self._run_phase("after_restart")
        except _Reject as error:
            return self._blocked("after_restart", error.code)

    def _run_phase(self, phase):
        state = self._fresh_state()
        self._require_gates(state, phase)
        started_at = self._require_candidate(state, phase)
        pinned = self._pinned_inputs()
        before = self._load_before_binding(pinned) if phase == "after_restart" else None
        challenge = self._write_challenge(phase, state, started_at)
        report = self._wait_for_report(phase)
        verified = self._verify_report(report, phase, challenge, pinned, started_at)
        if phase == "after_restart":
            self._verify_after_chain(verified, before)
        details = {
            "bindings": {
                "challenge_sha256": hashlib.sha256(
                    challenge["challenge"].encode()).hexdigest(),
                "candidate_started_at": started_at,
                "runtime_epoch": verified["observed"]["runtime_epoch"],
                "runtime_generation": verified["observed"]["runtime_generation"],
                "identity_fingerprint": pinned["admission_template"]["fingerprint"],
                "harness_binary_sha256": pinned["harness_artifact"]["binary_sha256"]},
            "cases": {"positive": verified["positive"], "no_state": verified["no_state"]},
            "ordinary_serving_nonregression": "separate_controller_service_final_gate",
            "cleanup_scope": "positive_and_no_state_only",
            "logical_cleanup_confirmed": True,
        }
        if phase == "before_restart":
            # Exactly one known negative control retained for the planned restart.
            details["negative_control_retained"] = dict(verified["retained"])
            self._before_binding = verified
        else:
            details["stale_handle_refusal"] = {
                "http_status": verified["stale_post"]["http_status"],
                "error_code": verified["stale_post"]["error_code"],
                "attempts": verified["stale_post"]["attempts"],
                "artifact_sha256": verified["stale_post"]["artifact_sha256"]}
            details["stale_binding"] = {
                "before_runtime_epoch": before["observed"]["runtime_epoch"],
                "after_runtime_epoch": verified["observed"]["runtime_epoch"],
                "generation_advanced": verified["observed"]["runtime_generation"]
                > before["observed"]["runtime_generation"],
                "restart_profile": "controlled_same_container_stop_start_retained_counter"}
        return self._result(True, phase, "passed", **details)

    def _verify_after_chain(self, verified, before):
        """Controlled-profile restart gate: fresh epoch AND advanced generation.

        This validation profile deliberately requires the same candidate
        container stopped and started with its retained native-incarnation
        counter, so the generation counter must ADVANCE, not merely differ. A
        volatile-reset runtime (lost /tmp record: generation restarts at 1) is
        NOT admitted by this profile and fails here into ordinary recovery;
        there is no epoch-only fallback. This is a stricter scenario gate, not
        a claim that every valid runtime restart has globally monotonic
        generation.
        """
        _require(before is not None, "before_binding_missing")
        # Per-boot pin anti-fabrication: the after pin must differ from the
        # before pin (equal pins mean the after value was fabricated
        # pre-restart; a fresh boot is a fresh incarnation) and its pinned
        # generation must advance, mirroring the observed-incarnation gates.
        _require(verified["pin"]["profile_digest"]
                 != before["pin"]["profile_digest"],
                 "profile_digest_pins_not_per_boot")
        _require(verified["observed"]["runtime_epoch"]
                 != before["observed"]["runtime_epoch"], "runtime_epoch_unchanged")
        _require(verified["observed"]["runtime_generation"]
                 > before["observed"]["runtime_generation"], "runtime_generation_not_advanced")
        _require(verified["pin"]["runtime_generation"]
                 > before["pin"]["runtime_generation"],
                 "pin_generation_not_advanced")
        _require(verified["stale_post"]["fresh_enrollment_runtime_epoch"]
                 == verified["observed"]["runtime_epoch"], "fresh_enrollment_epoch_mismatch")
        _require(verified["stale_post"]["fresh_enrollment_runtime_epoch"]
                 != before["observed"]["runtime_epoch"], "stale_enrollment_epoch_not_fresh")
        _require(verified["stale_post"]["artifact_sha256"]
                 == before["retained"]["artifact_sha256"], "stale_artifact_binding_mismatch")
        _require(verified["retained"] is None, "retained_control_in_after_phase")


CHALLENGE_FIELDS = ("schema", "phase", "attempt_id", "candidate_id",
                    "expected_candidate_started_at", "model_fingerprint", "method_id",
                    "expected_ranks", "challenge", "created_at", "expires_at")


class UnselectedValidator:
    """Fail-closed selection guard for unknown or unusable validator selections."""

    def __init__(self, selected, *, code="unknown_validator_selection"):
        self.selected = selected
        self.code = code

    def _refused(self, phase):
        return {"ok": False, "status": "blocked", "phase": phase,
                "code": self.code, "selected": self.selected,
                "scope": "validator_selection", "live_success_claimed": False}

    def pre_mutation_admission(self):
        return self._refused("pre_mutation_admission")

    def before_restart(self):
        return self._refused("before_restart")

    def after_restart(self):
        return self._refused("after_restart")
