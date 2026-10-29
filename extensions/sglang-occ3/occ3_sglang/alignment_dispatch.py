"""Method dispatch: exactly one aligner per negotiated method id, never a substitute.

The per-message method keeps the shipped union-subspace Procrustes solver. Every
registered calibration method serves the frozen artifact its own module fits:
the registry names that module (artifact_module), and the loader, the device
move, the content digest and the aligner are resolved through it. No artifact, an
artifact fitted for another method id, an artifact whose hidden dimension or
model identity is not this profile's, is refused at load or at dispatch: there
is deliberately no fallback path from a calibrated method to the per-message
solver, so a decline can never be answered with per-message output.

The aligner closure applies the method's apply_map and then its serving
transform with the binding's serving config: N0 (None) returns the raw rows
unchanged, N1 ({"kind": "n1"}) renormalises every row to the fixed vocabulary
norm statistic the worker measures. apply_map itself never renormalises and
never reads the message's reference embeddings.

Tensor-free at import: torch is reached only through the method module that runs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import os

from . import statebridge
from .latent_types import LatentError, RuntimeProfile, require
from .native_profile import method_descriptor

MAX_ARTIFACT_FILE_BYTES = 256 * 1024 * 1024


class AlignmentRefusal(LatentError):
    """Named refusal for a profile and artifact that one method cannot serve.

    Codes are the existing bounded LatentError codes: "unavailable" when no
    artifact is supplied or the file is unreadable, "identity" for a method,
    provenance or digest mismatch, "profile" for any other inconsistency,
    "quota" for a file beyond the artifact bound.
    """


@dataclass(frozen=True)
class CalibratedAlignmentInfo:
    """What the calibrated aligner used, for the capture record. Scalars only."""

    method: str
    rows: int
    hidden_dim: int
    artifact_digest: str


def _refuse(code):
    raise AlignmentRefusal(code)


def _descriptor_or_refuse(method):
    try:
        return method_descriptor(method)
    except Exception:
        return _refuse("profile")


def _method_module(descriptor):
    """The artifact module the registry names for this method; fail closed."""
    module = descriptor["artifact_module"]
    try:
        return importlib.import_module(module)
    except Exception:
        return _refuse("profile")


def frozen_artifact_reference(profile):
    """The validated (path, digest) pair this profile names, or None."""
    require(isinstance(profile, RuntimeProfile), "profile")
    if not profile.calibration_path:
        return None
    return profile.calibration_path, profile.calibration_digest


def load_frozen_artifact(profile, *, provenance=None, torch_module=None):
    """Resolve and validate the frozen artifact this profile is bound to.

    Returns None for a profile that names no reference. A named reference is
    read only after the size bound, and the bytes are accepted only through
    the method module's own deserializer with the profile's method id, hidden
    dimension and expected artifact digest, so a swapped or truncated file
    refuses instead of loading. A caller that negotiated the full live model
    identity passes provenance and every provenance field is compared at
    load; the fields the runtime profile itself carries (model fingerprint,
    layer index) are always compared here as well, so an artifact mis-authored
    for another model or layer is refused on every path.
    """
    reference = frozen_artifact_reference(profile)
    if reference is None:
        return None
    path, expected = reference
    descriptor = _descriptor_or_refuse(profile.method)
    module = _method_module(descriptor)
    try:
        size = os.stat(path).st_size
    except OSError:
        _refuse("unavailable")
    if not 0 < size <= module.MAX_ARTIFACT_BYTES:
        _refuse("quota")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(module.MAX_ARTIFACT_BYTES + 1)
    except OSError:
        _refuse("unavailable")
    if len(raw) != size or len(raw) > module.MAX_ARTIFACT_BYTES:
        _refuse("identity")
    try:
        artifact = module.deserialize_artifact(
            raw, expected_method_id=profile.method,
            expected_hidden_dim=profile.hidden_dim,
            expected_provenance=provenance,
            expected_artifact_digest=expected, torch_module=torch_module)
    except LatentError as error:
        raise AlignmentRefusal(error.code) from None
    # The profile pins the runtime identity it carries; a loaded binding that
    # disagrees has been mis-authored for another model or layer.
    binding = artifact.binding
    if binding.layer_index != profile.layer_index or binding.model_fingerprint != profile.model_fingerprint:
        _refuse("identity")
    return artifact


def _on_device(artifact, device, module):
    """Move the frozen tensors once; the fit parameters are never recomputed."""
    fields = module.TENSOR_NAMES
    if getattr(artifact, fields[0]).device == device:
        return artifact
    return replace(artifact, **{name: getattr(artifact, name).to(device)
                                for name in fields})


def _verified_digest(artifact, module):
    """The artifact's verified content digest, without a second tensor pass.

    A fit or a deserialize sets the content digest after digesting the tensor
    bytes once, and the method module verifies it against those bytes. Only
    an artifact that never went through either path is digested here.
    """
    digest = getattr(artifact, "content_digest", None)
    if digest is not None:
        return digest
    return module.artifact_digest(artifact)


def artifact_resident_bytes(artifact):
    """Resident size of one loaded frozen artifact, for the live-bytes budget.

    The size is the artifact module's own accounting over its tensor set, so
    every registered method is charged through its real field set.
    """
    descriptor = _descriptor_or_refuse(getattr(artifact, "method_id", None))
    module = _method_module(descriptor)
    return int(module.artifact_bytes(artifact))


def _binding_serving(artifact):
    """The serving config the artifact's binding records, None for N0.

    The binding owns the serving choice (N0 or N1). A method whose binding
    schema does not yet record a serving choice serves raw rows: a serving
    renormalisation is only ever applied when it was selected and recorded.
    """
    return getattr(artifact.binding, "serving", None)


def _calibrated_aligner(profile, artifact, descriptor, module):
    digest = _verified_digest(artifact, module)
    serving = _binding_serving(artifact)
    holder = {"artifact": artifact}

    def align(hidden, embeddings, target_norm):
        frozen = _on_device(holder["artifact"], hidden.device, module)
        holder["artifact"] = frozen
        z_raw = module.apply_map(frozen, hidden)
        prefix = module.serving_transform(serving, z_raw, nu=target_norm)
        return prefix, CalibratedAlignmentInfo(method=profile.method, rows=frozen.rows,
                                               hidden_dim=frozen.hidden_dim,
                                               artifact_digest=digest)

    align.artifact_digest = digest
    align.rows = artifact.rows
    align.serving = serving
    return align


def _per_message_aligner():
    def align(hidden, embeddings, target_norm):
        return statebridge.align_hidden_sequence(hidden, embeddings, target_norm)

    # The shipped solver renormalises every row to the vocabulary norm it is
    # handed, so it always needs the statistic the worker measures.
    align.serving = {"kind": "n1"}
    return align


def aligner_for(profile, artifact=None):
    """Return (hidden, embeddings, target_norm) -> (prefix, info) for this method.

    The per-message method keeps the shipped solver and ignores any artifact.
    Every registered calibration method serves the frozen artifact its own
    module fits and refuses a missing artifact, a method-id mismatch, a
    hidden-dimension or profile-identity mismatch, or a digest that is not the
    one this profile pins. The pair is resolved once, so a misconfiguration
    declines before any rank enters the solver path.
    """
    require(isinstance(profile, RuntimeProfile), "profile")
    descriptor = _descriptor_or_refuse(profile.method)
    if not descriptor["requires_calibration_artifact"]:
        return _per_message_aligner()
    if artifact is None:
        _refuse("unavailable")
    module = _method_module(descriptor)
    if artifact.method_id != profile.method:
        _refuse("identity")
    if type(artifact).__module__ != descriptor["artifact_module"]:
        # A foreign artifact type claiming this method's id: the registry
        # decides which module may serve it, and only that module's class does.
        _refuse("profile")
    if artifact.binding is None:
        # Without a binding there is no serving choice and no identity to
        # serve under; an unbound artifact is never served.
        _refuse("profile")
    if artifact.hidden_dim != profile.hidden_dim:
        _refuse("profile")
    if artifact.binding.layer_index != profile.layer_index:
        _refuse("identity")
    if artifact.binding.model_fingerprint != profile.model_fingerprint:
        _refuse("identity")
    if profile.calibration_digest and _verified_digest(artifact, module) != profile.calibration_digest:
        _refuse("identity")
    return _calibrated_aligner(profile, artifact, descriptor, module)
