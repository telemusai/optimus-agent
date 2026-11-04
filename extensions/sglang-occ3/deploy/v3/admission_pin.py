"""Admission-pin writer: host-observed incarnation evidence for the native phase suite.

Three one-shot verbs, no loops, sleeps or retries (design contract
occ3-queue-campaign/admission-pin-contract-reconciled-v3.md):

  fetch-receipt  ONE POST to the FIXED DGX host-local identity endpoint
                 (http://127.0.0.1:30001/occ3/v1/native/runtime/identity)
                 using the existing protected serving api key; every response
                 field is validated against the plan (wrong method, wrong
                 incarnation shape, not-ready readiness, digest or derivation
                 drift all refuse); the receipt is saved write-once next to
                 the attempt.
  write-pin      reads the SAME journal the validator reads (attempt
                 binding), byte-verifies the deployed
                 calibrated_profile_evidence profile, re-derives the profile
                 digest with the validator's own _derived_profile_digest and
                 publishes the per-phase pin file via renameat2
                 RENAME_NOREPLACE (staged 0600 write + fsync, then the
                 kernel-atomic no-replace move; the final name never shows
                 partial content and is never overwritten).
  verify-pins    read-only diagnostic re-running the pin binding checks.

The pin is HOST-OBSERVED INCARNATION EVIDENCE. It is not a signed proof, not
native acceptance, and never sufficient without the harness report: the
validator's _verify_report and _verify_after_chain remain the final
authorities. This module never mints challenges, never waits for reports,
never touches restart permits and never mutates the journal or the plan.

Scope: the calibrated-method campaign plan (calibrated_artifact +
calibrated_profile_evidence + three-way static-input SHA agreement). A
per-message plan or the repo packaging default (613698c4...) is refused, not
silently accepted. The logical build-input path (static_input.path) is a
packaging role and is never read here.

Fail-closed: every anomaly is one fixed refusal code and a nonzero exit with
zero partial writes. The api key exists only inside the Authorization header;
it is never printed, logged or written to any artifact.

Every fetch-receipt invocation performs exactly ONE fresh identity POST - a
read-only identity query, never a replay of a previous request and never
inference. A receipt REJECTED by validation still performed that one POST;
only pre-request failures prove zero HTTP calls. Re-fetching the same
incarnation is idempotent at the receipt file (server_time_epoch drift
tolerated).

Publication statuses: "written" (rename + parent fsync succeeded - the only
success), "idempotent" (EEXIST with the same binding), refusal codes
otherwise. post-rename cleanup/fsync uncertainty is the fixed NONZERO
publish_uncertain (the final stays and is never deleted); "idempotent"
asserts same bytes/binding only and is never proof that an earlier uncertain
fsync became durable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

from . import atomic_publish

from .native_validator import (DTYPES, MAX_CHALLENGE_BYTES, MAX_PIN_BYTES, PIN_FIELDS,
                               PIN_SCHEMA, _Reject, _derived_profile_digest, _exact_fields,
                               _method_descriptor, _method_profile_fields,
                               _phase_challenge_path, _phase_pin_path, _require, _safe_read,
                               _sha_field, _strict_loads, _string, _uint)
from .service_checks import request_json

IDENTITY_ORIGIN = "http://127.0.0.1:30001"
IDENTITY_PATH = "/occ3/v1/native/runtime/identity"
IDENTITY_URL = IDENTITY_ORIGIN + IDENTITY_PATH
REQUEST_SCHEMA = "occ3.runtime.identity.request.v1"
RECEIPT_SCHEMA = "occ3.runtime.identity.v1"
# The exact response fieldset of occ3_sglang.latent_control_api.runtime_identity.
RECEIPT_FIELDS = ("schema", "runtime_id", "runtime_generation", "model_fingerprint",
                  "method_id", "tp_layout_digest", "layer_index", "hidden_dim", "dtype",
                  "expected_ranks", "ready_ranks", "profile_digest", "source_capture",
                  "continuous_prefix", "terminal_journal", "server_time_epoch",
                  "max_ttl_seconds", "max_capture_bytes")
# Write-time metadata, never a binding: a re-write of the SAME binding may
# legitimately carry a fresh timestamp.
PIN_DRIFT_FIELDS = ("written_at",)
# server_time_epoch is server clock data, not a binding input: a re-fetch of
# the SAME incarnation may legitimately carry a fresh one.
RECEIPT_DRIFT_FIELDS = ("server_time_epoch",)
MAX_RECEIPT_BYTES = 8192
REQUEST_TIMEOUT_S = 30.0
DEFAULT_API_KEY_FILE = "/data/neo-glm53/api-key"
PHASES = ("before_restart", "after_restart")
EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_ERROR = 0, 2, 4, 8
RUNTIME_EPOCH_HEX = re.compile(r"\A[0-9a-f]{32}\Z")

# Test-only I/O seam, mirroring docker_config._HOST_PATH_RESOLVER: maps the
# plan-logical POSIX staged-profile path to the real host file when the test
# host is not the production Linux daemon host. Production leaves this None and
# the path is used as-is; the size and digest checks always run on the real
# file, so the map never weakens any rule.
STAGED_PROFILE_RESOLVER = None


def _resolve_staged_path(path):
    resolver = STAGED_PROFILE_RESOLVER
    return resolver(path) if resolver is not None else path


# -- plan contract (read-only journal + plan consumption) ---------------------

def _plan_contract(attempt_dir):
    """Journal, plan and the calibrated pin contract fields.

    The journal is strict-loaded (duplicate keys, BOM, depth and size bounds
    all refuse) but never written. Returns
    (state, evidence, method_id, source_profile).
    """
    data = _safe_read(Path(attempt_dir) / "journal.json", 4 * 1024 * 1024)
    _require(data is not None, "journal_unreadable")
    state = _strict_loads(data, 4 * 1024 * 1024)
    _require(isinstance(state, dict) and isinstance(state.get("attempt_id"), str)
             and state["attempt_id"], "journal_unreadable")
    plan_value = state.get("plan")
    _require(isinstance(plan_value, dict), "plan_missing")
    evidence = plan_value.get("calibrated_profile_evidence")
    _require(isinstance(evidence, dict) and set(evidence) == {"path", "sha256", "bytes"},
             "calibrated_profile_evidence_invalid")
    static = plan_value.get("static_input")
    native = plan_value.get("native_validator")
    _require(isinstance(static, dict) and isinstance(native, dict), "static_input_pins_missing")
    # Three-way static-input SHA agreement. The logical build-input path
    # (static_input.path) is a packaging role and is deliberately NOT read.
    _require(static.get("sha256") == evidence["sha256"]
             == native.get("static_input_sha256"), "calibrated_profile_pin_mismatch")
    artifact = plan_value.get("calibrated_artifact")
    _require(isinstance(artifact, dict) and set(artifact) == {"sha256", "size"},
             "calibrated_artifact_invalid")
    _sha_field(artifact.get("sha256"), "calibrated_artifact_invalid")
    _uint(artifact.get("size"), "calibrated_artifact_invalid", positive=True)
    method_id = native.get("method_id")
    _require(isinstance(method_id, str) and method_id, "method_id_invalid")
    _, _, descriptor = _method_descriptor(method_id)
    block = native.get("admission_template")
    _require(isinstance(block, dict), "admission_template_invalid")
    source_profile = block.get("source_profile")
    _require(isinstance(source_profile, dict), "source_profile_invalid")
    _exact_fields(source_profile, _method_profile_fields(descriptor), "source_profile_invalid")
    _require(source_profile.get("method") == method_id, "source_method_mismatch")
    return state, evidence, method_id, source_profile


def _verified_staged_profile(evidence, source_profile):
    """Byte-verify the deployed profile receipt; strict-parse; light cross-checks.

    The deployed file is the one verified source: size and sha256 must equal
    the plan's calibrated_profile_evidence values. Only fields the staged file
    itself carries are cross-checked against the plan profile.
    """
    path = Path(_resolve_staged_path(evidence["path"]))
    data = _safe_read(path, MAX_RECEIPT_BYTES)
    _require(data is not None, "staged_profile_unreadable")
    _require(len(data) == evidence["bytes"], "staged_profile_size_mismatch")
    _require(hashlib.sha256(data).hexdigest() == evidence["sha256"],
             "staged_profile_digest_mismatch")
    value = _strict_loads(data, MAX_RECEIPT_BYTES)
    _require(isinstance(value, dict), "staged_profile_invalid")
    for name in ("method", "model_id", "model_fingerprint"):
        if name in value:
            _require(value[name] == source_profile.get(name),
                     "staged_profile_" + name + "_mismatch")
    return value


# -- identity receipt ---------------------------------------------------------

def _validate_receipt(receipt, method_id, source_profile):
    """Full field validation of the authenticated identity receipt.

    Every check refuses with one fixed code. The derivation gate re-derives
    profile_digest from the plan's pinned static profile plus THIS receipt's
    incarnation, so a receipt from any other profile, method or incarnation
    can never pass.
    """
    _exact_fields(receipt, RECEIPT_FIELDS, "receipt_fieldset_invalid")
    _require(receipt["schema"] == RECEIPT_SCHEMA, "receipt_schema_unknown")
    _require(isinstance(receipt["runtime_id"], str)
             and RUNTIME_EPOCH_HEX.fullmatch(receipt["runtime_id"]) is not None,
             "receipt_epoch_invalid")
    _uint(receipt["runtime_generation"], "receipt_generation_invalid", positive=True)
    _require(receipt["expected_ranks"] == 8, "receipt_ranks_invalid")
    _require(receipt["ready_ranks"] == 8, "receipt_ranks_not_ready")
    for name in ("source_capture", "continuous_prefix", "terminal_journal"):
        _require(receipt[name] is True, "receipt_not_ready")
    _sha_field(receipt["tp_layout_digest"], "receipt_layout_digest_invalid")
    _sha_field(receipt["profile_digest"], "receipt_profile_digest_invalid")
    _require(receipt["dtype"] in DTYPES, "receipt_dtype_unsupported")
    _uint(receipt["hidden_dim"], "receipt_hidden_dim_invalid", positive=True)
    _uint(receipt["layer_index"], "receipt_layer_index_invalid")
    _uint(receipt["server_time_epoch"], "receipt_time_invalid", positive=True)
    _uint(receipt["max_ttl_seconds"], "receipt_ttl_invalid", positive=True)
    _uint(receipt["max_capture_bytes"], "receipt_capture_bytes_invalid", positive=True)
    # Wrong-method hard refusal: the receipt must be THIS plan's method.
    _require(receipt["method_id"] == method_id, "method_mismatch")
    _require(receipt["model_fingerprint"] == source_profile["model_fingerprint"],
             "receipt_fingerprint_mismatch")
    _require(receipt["hidden_dim"] == source_profile["hidden_dim"],
             "receipt_hidden_dim_mismatch")
    _require(receipt["layer_index"] == source_profile["layer_index"],
             "receipt_layer_index_mismatch")
    observed = {"runtime_epoch": receipt["runtime_id"],
                "runtime_generation": receipt["runtime_generation"]}
    derived = _derived_profile_digest(source_profile, observed)
    _require(derived == receipt["profile_digest"], "derivation_mismatch")
    return receipt


def _load_api_key(path):
    """First line, stripped. Exists only inside the Authorization header."""
    data = _safe_read(Path(path), 4096)
    _require(data is not None, "api_key_unavailable")
    try:
        line = data.split(b"\n", 1)[0].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _Reject("api_key_unavailable") from None
    key = line.strip()
    _require(0 < len(key) <= 4096 and key.isprintable(), "api_key_unavailable")
    return key


# -- challenge and sequencing binding ----------------------------------------

def _require_phase_sequence(state, phase):
    """after_restart only after the controlled restart completed in the journal."""
    if phase == "after_restart":
        _require(state.get("restart_done") is True, "restart_not_completed")
        _require(state.get("restart_count") == 1, "restart_count_invalid")


def _current_challenge(attempt_dir, phase, state, method_id):
    """The CURRENT phase challenge: attempt, candidate and method must bind."""
    data = _safe_read(_phase_challenge_path(attempt_dir, phase), MAX_CHALLENGE_BYTES)
    _require(data is not None, "challenge_missing")
    challenge = _strict_loads(data, MAX_CHALLENGE_BYTES)
    _require(isinstance(challenge, dict), "challenge_invalid")
    _require(challenge.get("phase") == phase, "challenge_phase_mismatch")
    _require(challenge.get("attempt_id") == state["attempt_id"],
             "challenge_attempt_mismatch")
    if state.get("candidate_id"):
        _require(challenge.get("candidate_id") == state["candidate_id"],
                 "challenge_candidate_mismatch")
    _require(challenge.get("method_id") == method_id, "challenge_method_mismatch")
    _require(int(time.time()) <= _uint(challenge.get("expires_at"),
             "challenge_expired_invalid", positive=True), "challenge_expired")
    return challenge


def _load_before_pin(attempt_dir, pin_attempt_id):
    """The before_restart pin, REQUIRED before any after_restart effect.

    Absence is refused early (before_pin_missing) rather than left to the
    validator's later rejection; shape and attempt binding are checked.
    """
    data = _safe_read(_phase_pin_path(attempt_dir, "before_restart"), MAX_PIN_BYTES)
    _require(data is not None, "before_pin_missing")
    before = _strict_loads(data, MAX_PIN_BYTES)
    _require(isinstance(before, dict), "before_pin_invalid")
    _require(before.get("schema") == PIN_SCHEMA, "before_pin_invalid")
    _require(before.get("phase") == "before_restart", "before_pin_invalid")
    _require(before.get("attempt_id") == pin_attempt_id, "cross_attempt_pin")
    _sha_field(before.get("static_input_sha256"), "before_pin_invalid")
    _string(before.get("runtime_epoch"), "before_pin_invalid", limit=256)
    _uint(before.get("runtime_generation"), "before_pin_generation_invalid",
          positive=True)
    _sha_field(before.get("profile_digest"), "before_pin_invalid")
    return before


def _require_cross_pin(attempt_dir, pin):
    """after_restart guards: same attempt, fresh epoch, advanced generation,
    per-boot digest distinction - all before publication."""
    if pin["phase"] != "after_restart":
        return
    before = _load_before_pin(attempt_dir, pin["attempt_id"])
    _require(pin["runtime_epoch"] != before["runtime_epoch"],
             "stale_cross_restart_pin")
    _require(pin["runtime_generation"] > before["runtime_generation"],
             "pin_generation_not_advanced")
    _require(pin["profile_digest"] != before["profile_digest"],
             "profile_digest_pins_not_per_boot")


# -- exclusive publication with idempotency ---------------------------------

def _require_same_binding(existing, document, conflict_code, drift_fields=()):
    """Refusal when an existing document is not the same binding.

    Fields in drift_fields are write-time metadata, never bindings: a fresh
    timestamp on an otherwise identical document stays idempotent.
    """
    _require(isinstance(existing, dict), conflict_code)
    for name in document:
        if name in drift_fields:
            continue
        _require(name in existing and existing[name] == document[name], conflict_code)


def _require_publish_platform():
    """Publication is Linux-only by contract: the primitive is
    renameat2(RENAME_NOREPLACE) via atomic_publish. Non-Linux hosts get
    a fixed refusal BEFORE any staging side effect (no open, no write,
    no rename attempt). Scope, stated honestly: this refusal proves
    zero STAGING operations only. fetch_receipt loads the api key and
    POSTs the identity request BEFORE publication is attempted; this
    guard does not and cannot prove zero key reads or zero HTTP. It is
    the publisher's guard, not an entrypoint preflight."""
    if not sys.platform.startswith("linux"):
        raise _Reject("publish_platform_unsupported")


def _fsync_parent_directory(parent):
    """Open the parent directory read-only and fsync it (rename durability).

    The caller classifies ANY OSError from this helper as the fixed nonzero
    publish_uncertain refusal. Extracted verbatim from the rc0 path so
    non-Linux test hosts can substitute the CAPABILITY (an owned writable
    file fsync) without touching production classification semantics.
    """
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_exclusive(path, document, *, conflict_code, drift_fields=(), limit):
    """JSON facade over the shared staged-byte publisher.

    Observable order (unchanged): platform refusal FIRST, then Path()
    coercion, staging-name construction, json.dumps serialization, and the
    size bound - all before any staging side effect. The EEXIST branch
    compares the parsed existing document against the in-memory document
    with drift-field tolerance (semantic idempotency).
    """
    # Platform refusal FIRST: before the staging open, before any
    # side effect. Linux keeps the real, complete flags below.
    _require_publish_platform()
    path = Path(path)
    parent = path.parent
    staging = parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    _require(len(body) <= limit, "document_too_large")
    return _publish_staged_bytes(
        path, staging, body, conflict_code=conflict_code, limit=limit,
        equality=lambda existing: _require_same_binding(
            _strict_loads(existing, limit), document, conflict_code,
            drift_fields=drift_fields))


def _publish_staged_bytes(path, staging, body, *, conflict_code, limit, equality):
    """The ONE staging/rename/cleanup implementation.

    Receives PREPARED inputs only (caller-owned preconditions, enforced by
    the facades): platform already verified; path already a Path; staging
    name already constructed; body exactly bytes with len(body) <= limit.
    `equality(existing)` decides the EEXIST branch (returns "idempotent"
    or raises); the bounded _safe_read and its not-None refusal stay here.

    Certainty classes (authoritative):
    - written        rename ok + parent-directory fsync ok. The ONLY success.
    - idempotent     EEXIST + equality(existing) satisfied (per-facade
                     semantics: JSON binding equality with tolerated drift
                     fields, or strict byte equality for raw artifacts).
                     Never proof an earlier uncertain fsync became durable.
    - publish_uncertain (NONZERO refusal)  rename ok but the parent fsync
                     failed, or the staging PATHNAME persists after rc == 0
                     in ANY form (our inode, or a foreign actor's recreated
                     file): the anomaly is the name existing at all, and it
                     is never unlinked. The final stays, is never deleted;
                     no retry, no recovery.
    - staging_identity_lost  pre-rename check: the path no longer resolves to
                     the fd-captured inode; refused WITHOUT publishing and
                     WITHOUT unlinking the (foreign) name.
    - staging_fsync_failed   staged durability sync failed: fixed
                     PRE-publication refusal (no rename attempted), with
                     owned-inode cleanup.
    - staging_close_failed   the NORMAL-path close itself failed (the only
                     close mapped to a refusal): fixed pre-publication
                     refusal with owned-inode cleanup. A cleanup close in
                     the fstat-failure branch or the BaseException handler
                     is best-effort and NEVER replaces the primary refusal
                     (staging_identity_unavailable / the original error).
                     Close is attempted at most once per fd in every
                     branch; a failed close does not establish the fd still
                     open or the bytes durable, and the descriptor is
                     abandoned rather than probed or retried.
    - publish_unsupported / publish_refused_policy / publish_unknown /
                     conflict_code: fixed refusals per atomic_publish.
    - publish_platform_unsupported  non-Linux host: refused by the facade
                     before any staging side effect (publisher guard; see
                     _require_publish_platform for the honest scope).

    Honest trust boundary: the pre-rename identity check narrows the TOCTOU
    window between the fd-captured identity and the rename syscall; it does
    NOT eliminate it. This module trusts the attempt-directory owner (the
    executor's own private directory) and claims no safety against a
    hostile directory writer.
    """
    parent = path.parent
    # Owned-identity discipline (BLOCKER fix): cleanup may unlink ONLY the
    # inode captured by fstat on OUR OWN open fd, and ONLY while the name
    # still resolves to that exact regular inode. Ownership is NEVER
    # established from the generated path being unpredictable or the
    # directory being private.
    fd = None
    identity = None
    try:
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
                         0o600)
        except OSError:
            # Open failed (EEXIST foreign file, EACCES, ...): ownership was
            # NEVER established - no fd, no identity, ZERO unlink by name.
            raise _Reject("staging_open_failed") from None
        try:
            info = os.fstat(fd)
        except OSError:
            # Identity capture failed: close OUR fd safely, but never unlink
            # by name without a verified identity (the name may no longer be
            # ours). Uncertainty is reported, not guessed away.
            # Precedence (mandatory): a cleanup-close failure here NEVER
            # replaces the public refusal staging_identity_unavailable; the
            # close is best-effort with a silent swallow, ownership
            # relinquished BEFORE the attempt. No cause chain is attached.
            closing = fd
            fd = None
            try:
                os.close(closing)
            except OSError:
                pass
            raise _Reject("staging_identity_unavailable") from None
        _require(stat.S_ISREG(info.st_mode), "staging_not_regular")
        identity = (info.st_dev, info.st_ino)
        written = 0
        while written < len(body):
            try:
                chunk = os.write(fd, body[written:])
            except OSError:
                # The write itself failed (Linux-reachable EIO/ENOSPC/EDQUOT
                # on a failing or full filesystem): map to the same fixed
                # refusal the short-write case uses. Raised inside this
                # protected block, the owned-inode cleanup still runs.
                raise _Reject("staging_write_failed") from None
            _require(chunk > 0, "staging_write_failed")
            written += chunk
        try:
            os.fsync(fd)
        except OSError:
            # Staged durability sync failed: nothing has been published
            # (no rename attempt). Fixed pre-publication refusal; raised
            # inside the protected block so the owned-inode cleanup runs.
            raise _Reject("staging_fsync_failed") from None
        # Relinquish ownership BEFORE the close attempt: a failed close
        # establishes nothing (neither that the fd is still open nor that
        # the bytes are durable), so the slot must already be empty when
        # it happens. The normal close is the ONLY site mapping
        # staging_close_failed: nothing has gone wrong before it.
        closing = fd
        fd = None
        try:
            os.close(closing)
        except OSError:
            raise _Reject("staging_close_failed") from None
    except BaseException:
        # Single-attempt, relinquish-first handler close: never retries (the
        # normal/fstat paths already cleared the slot, so this fires only
        # for failures BEFORE the normal close - write/fsync rejections),
        # never masks the in-flight original exception (silent swallow), and
        # never prevents the owned-inode cleanup from running. A failed
        # close establishes nothing; the descriptor is abandoned rather
        # than probed or double-closed.
        if fd is not None:
            closing = fd
            fd = None
            try:
                os.close(closing)
            except OSError:
                pass
        _cleanup_owned_inode(staging, identity)
        raise
    # Immediate pre-rename check: the staging path must still resolve to the
    # fd-captured regular inode. A mismatch refuses WITHOUT publishing and
    # WITHOUT unlinking the foreign name (it is not ours). Honest boundary:
    # this check narrows the TOCTOU window between identity capture and the
    # rename syscall, it does NOT eliminate it - a racing adversarial
    # directory writer can still swap the name after this lstat and before
    # renameat2 observes it. This module trusts the attempt-directory owner
    # (the executor's own 0700 directory) and makes no claim of safety
    # against a hostile directory owner.
    if not _name_resolves_to(staging, identity):
        raise _Reject("staging_identity_lost")
    try:
        atomic_publish.rename_noreplace(staging, path)
    except atomic_publish.RenameWouldOverwrite:
        _cleanup_owned_inode(staging, identity)
        data = _safe_read(path, limit)
        _require(data is not None, conflict_code)
        equality(data)
        return "idempotent"
    except atomic_publish.RenameNotSupported:
        _cleanup_owned_inode(staging, identity)
        raise _Reject("publish_unsupported") from None
    except atomic_publish.RenamePolicyRefused:
        _cleanup_owned_inode(staging, identity)
        raise _Reject("publish_refused_policy") from None
    except atomic_publish.RenameUnknown as error:
        # Uncertain outcome: stop. Bounded safe identity facts only; no
        # deletion, no retry, no recovery action - including NO cleanup.
        raise _Reject("publish_unknown") from error
    # rc == 0: the staging name was consumed by the kernel. Never unlink it.
    # Root's original rc0 contract (restored): ANY unexpected staging-path
    # persistence after a successful rename - our captured inode OR any
    # other inode a foreign actor recreated - classifies publish_uncertain,
    # nonzero, never unlinked. The name existing at all after rc0 is the
    # anomaly; what it contains is not ours to decide.
    try:
        os.lstat(staging)
    except OSError:
        pass
    else:
        raise _Reject("publish_uncertain")
    try:
        _fsync_parent_directory(parent)
    except OSError:
        raise _Reject("publish_uncertain") from None
    return "written"


def _name_resolves_to(path, identity):
    """True only when the name currently resolves to the captured inode."""
    if identity is None:
        return False
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity


def _cleanup_owned_inode(staging, identity):
    """Unlink the staging entry ONLY with a captured identity match.

    Requires a NON-None captured identity (from fstat on our own open fd)
    AND a current lstat resolving the name to that exact regular inode.
    Without a captured identity there is NO unlink, ever: an unpredictable
    generated path or a private directory never establishes ownership.
    Never sweeps the directory.
    """
    if identity is None:
        return
    if not _name_resolves_to(staging, identity):
        return
    try:
        os.unlink(staging)
    except OSError:
        pass


def _publish_pin(attempt_dir, phase, pin):
    """Owner-only exclusive publication; identical binding is idempotent."""
    return _publish_exclusive(_phase_pin_path(attempt_dir, phase), pin,
                              conflict_code="pin_conflict",
                              drift_fields=PIN_DRIFT_FIELDS, limit=MAX_PIN_BYTES)


def _publish_receipt(path, receipt):
    """Write-once receipt; the same incarnation re-fetched is idempotent."""
    return _publish_exclusive(path, receipt, conflict_code="receipt_conflict",
                              drift_fields=RECEIPT_DRIFT_FIELDS,
                              limit=MAX_RECEIPT_BYTES)

def publish_artifact(path, body, *, expected_sha256, max_bytes):
    """Publish downloaded RAW PROOF BYTES byte-identically (public API).

    Bytes-only: exactly `bytes` (immutable; str/bytearray/memoryview are
    refused with a TypeError - the raw analogue of the JSON facade's
    unserializable-document TypeError). Strict byte-equality idempotency:
    on EEXIST the existing bytes must equal `body` EXACTLY - no parse, no
    decode, non-UTF8-safe by construction. Derived guarantee: the body is
    digest-verified before any side effect, so a strict-equal existing
    final transitively hashes to expected_sha256.

    Order (guard-first, mirroring the JSON facade): platform refusal, then
    the body-type TypeError, then digest-syntax, limit, size, and digest
    gates - all BEFORE the staging name exists (the raw facade constructs
    its staging name only after its digest/size gates). The EEXIST read
    uses _safe_read (lstat-based symlink/reparse/hardlink/regularity/size
    checks; plain open - the same TOCTOU-bounded trust boundary as the
    publisher: the attempt-directory owner, no claim against a hostile
    directory writer).
    """
    _require_publish_platform()
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    _sha_field(expected_sha256, "expected_digest_invalid")
    _uint(max_bytes, "limit_invalid", positive=True)
    _require(len(body) <= max_bytes, "artifact_too_large")
    _require(hashlib.sha256(body).hexdigest() == expected_sha256,
             "digest_mismatch")
    path = Path(path)
    parent = path.parent
    staging = parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    return _publish_staged_bytes(
        path, staging, body, conflict_code="artifact_conflict",
        limit=max_bytes,
        equality=lambda existing: _require(existing == body,
                                          "artifact_conflict"))


# -- verbs (pure functions: return status dicts, never print) ----------------

def fetch_receipt(attempt_dir, phase, receipt_path, *, api_key_file=DEFAULT_API_KEY_FILE,
                  requester=request_json):
    """One POST; validate; save the receipt write-once. Never retries."""
    _require(phase in PHASES, "phase_invalid")
    state, evidence, method_id, source_profile = _plan_contract(attempt_dir)
    _require_phase_sequence(state, phase)
    _current_challenge(attempt_dir, phase, state, method_id)
    before = _load_before_pin(attempt_dir, state["attempt_id"]) \
        if phase == "after_restart" else None
    key = _load_api_key(api_key_file)
    result = requester("POST", IDENTITY_URL, key, {"schema": REQUEST_SCHEMA},
                        REQUEST_TIMEOUT_S)
    _require(result is not None and result.status == 200 and result.error_code is None,
             "identity_route_unavailable")
    receipt = _validate_receipt(result.body, method_id, source_profile)
    if before is not None:
        _require(receipt["runtime_id"] != before["runtime_epoch"],
                 "stale_cross_restart_pin")
        _require(receipt["runtime_generation"] > before["runtime_generation"],
                 "pin_generation_not_advanced")
        _require(receipt["profile_digest"] != before["profile_digest"],
                 "profile_digest_pins_not_per_boot")
    status = _publish_receipt(receipt_path, receipt)
    return {"verb": "fetch-receipt", "phase": phase, "status": status,
            "attempt_id": state["attempt_id"], "runtime_epoch": receipt["runtime_id"],
            "runtime_generation": receipt["runtime_generation"],
            "profile_digest": receipt["profile_digest"], "method_id": method_id}


def write_pin(attempt_dir, phase, receipt_path):
    """Re-validate the saved receipt and publish the per-phase pin."""
    _require(phase in PHASES, "phase_invalid")
    state, evidence, method_id, source_profile = _plan_contract(attempt_dir)
    _require_phase_sequence(state, phase)
    _current_challenge(attempt_dir, phase, state, method_id)
    _verified_staged_profile(evidence, source_profile)
    data = _safe_read(Path(receipt_path), MAX_RECEIPT_BYTES)
    _require(data is not None, "receipt_missing")
    receipt = _validate_receipt(_strict_loads(data, MAX_RECEIPT_BYTES), method_id,
                                source_profile)
    pin = {
        "schema": PIN_SCHEMA,
        "phase": phase,
        "attempt_id": state["attempt_id"],
        "static_input_sha256": evidence["sha256"],
        "runtime_epoch": receipt["runtime_id"],
        "runtime_generation": receipt["runtime_generation"],
        "profile_digest": receipt["profile_digest"],
        "written_at": int(time.time()),
    }
    _exact_fields(pin, PIN_FIELDS, "pin_fieldset_invalid")
    _require_cross_pin(attempt_dir, pin)
    status = _publish_pin(attempt_dir, phase, pin)
    return {"verb": "write-pin", "phase": phase, "status": status,
            "attempt_id": state["attempt_id"], "runtime_epoch": pin["runtime_epoch"],
            "runtime_generation": pin["runtime_generation"],
            "profile_digest": pin["profile_digest"]}


def verify_pins(attempt_dir):
    """Read-only: re-run every pin binding check, including the derivation."""
    state, evidence, method_id, source_profile = _plan_contract(attempt_dir)
    report = {}
    for phase in PHASES:
        data = _safe_read(_phase_pin_path(attempt_dir, phase), MAX_PIN_BYTES)
        if data is None:
            report[phase] = {"present": False}
            continue
        entry = {"present": True}
        try:
            pin = _strict_loads(data, MAX_PIN_BYTES)
            _exact_fields(pin, PIN_FIELDS, "pin_fieldset_invalid")
            _require(pin["schema"] == PIN_SCHEMA, "pin_schema_unknown")
            _require(pin["phase"] == phase, "pin_phase_mismatch")
            _require(pin["attempt_id"] == state["attempt_id"],
                     "pin_attempt_binding_mismatch")
            _require(pin["static_input_sha256"] == evidence["sha256"],
                     "pin_static_input_mismatch")
            _string(pin["runtime_epoch"], "pin_epoch_invalid", limit=256)
            _uint(pin["runtime_generation"], "pin_generation_invalid", positive=True)
            _sha_field(pin["profile_digest"], "pin_digest_invalid")
            _uint(pin["written_at"], "pin_written_at_invalid", positive=True)
            derived = _derived_profile_digest(source_profile,
                                              {"runtime_epoch": pin["runtime_epoch"],
                                               "runtime_generation": pin["runtime_generation"]})
            _require(derived == pin["profile_digest"], "pin_derivation_mismatch")
            entry.update(ok=True, runtime_epoch=pin["runtime_epoch"],
                         runtime_generation=pin["runtime_generation"])
        except _Reject as error:
            entry.update(ok=False, code=error.code)
        report[phase] = entry
    if report["before_restart"]["present"] and report["after_restart"]["present"]:
        before, after = report["before_restart"], report["after_restart"]
        chain_ok = (before.get("ok") and after.get("ok")
                    and before["runtime_epoch"] != after["runtime_epoch"]
                    and after["runtime_generation"] > before["runtime_generation"])
        report["chain"] = {"ok": bool(chain_ok)}
    return {"verb": "verify-pins", "ok": True, "attempt_id": state["attempt_id"],
            "pins": report}


# -- one-shot CLI ------------------------------------------------------------

def _emit(status):
    print(json.dumps(status, sort_keys=True, separators=(",", ":")))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="admission_pin",
        description="One-shot admission-pin writer (host-observed incarnation evidence).")
    sub = parser.add_subparsers(dest="verb", required=True)
    fetch = sub.add_parser("fetch-receipt", help="one POST + validate + save the receipt")
    fetch.add_argument("--attempt-dir", required=True)
    fetch.add_argument("--phase", required=True, choices=PHASES)
    fetch.add_argument("--receipt", required=True)
    fetch.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE)
    write = sub.add_parser("write-pin", help="publish the per-phase pin file")
    write.add_argument("--attempt-dir", required=True)
    write.add_argument("--phase", required=True, choices=PHASES)
    write.add_argument("--receipt", required=True)
    verify = sub.add_parser("verify-pins", help="read-only binding diagnostic")
    verify.add_argument("--attempt-dir", required=True)
    args = parser.parse_args(argv)
    verbs = {
        "fetch-receipt": lambda: fetch_receipt(args.attempt_dir, args.phase, args.receipt,
                                               api_key_file=args.api_key_file),
        "write-pin": lambda: write_pin(args.attempt_dir, args.phase, args.receipt),
        "verify-pins": lambda: verify_pins(args.attempt_dir),
    }
    try:
        status = verbs[args.verb]()
        status["ok"] = True
        _emit(status)
        return EXIT_OK
    except _Reject as error:
        _emit({"verb": args.verb, "ok": False, "code": error.code})
        return EXIT_REFUSED
    except OSError:
        _emit({"verb": args.verb, "ok": False, "code": "io_error"})
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
