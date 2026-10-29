"""Frozen low-rank whitened Procrustes alignment fitted on independent rows.

Method B of the independent-calibration campaign (spec sections 10 to 14): the
transform is fitted once on a FIT partition of calibration rows and then frozen.
The per-message StateBridge fit is a different method and is never used here.

Row vectors everywhere. With FIT rows H_f [M, D] and reference embeddings
E_f [M, D]:

    A = H_f - mu_H,  B = E_f - mu_E                    M x D
    A = U_H diag(s_H) V_H^T,  B = U_E diag(s_E) V_E^T   reduced SVD
    P_H = V_H[:, :r],  P_E = V_E[:, :r]                 D x r
    g_H = s_H[:r]^2 / M,  g_E = s_E[:r]^2 / M          r
    tau = eta * mean(g),  d = sqrt(g + tau)             regularised whitening
    X = (A P_H) / d_H,  Y = (B P_E) / d_E               M x r
    J = X^T Y,  R = U V^T from svd(J)                   r x r, O(r)

d is a regularised whitening factor: with positive tau the whitened covariance
is not exactly identity, and nothing here claims exact whitening. R is solved
over O(r): reflections are allowed and det(R) is never forced to +1. The
rotation defect ||R^T R - I_r||_F is verified at fit and recorded in the
binding. Rank policy: r <= min(numerical_rank(A), numerical_rank(B), M - 1, D);
a requested r above that bound is refused instead of completed with null
directions, and the numerical rank and the participation ratio are recorded
separately from the requested r.

Apply to unseen rows H_q [N, D]:

    Q = ((H_q - mu_H) P_H) / d_H                        N x r
    Z_raw = mu_E + ((Q R) * d_E) P_E^T                  N x D

apply_map returns Z_raw: it never reads reference embeddings of the message and
never renormalises. Components of H_q outside span(P_H) are discarded; the
discard ratio ||(H_q - mu_H) - P_H P_H^T (H_q - mu_H)||_F / ||H_q - mu_H||_F is
measured by residual_discard_ratio for the evaluation record. No D x D matrix
is stored or formed anywhere in fit or apply.

Variants (each its own method identity, configuration and artifact identity):

    "o"     whitened O(r) candidate (default)
    "so"    proper rotation R_SO = U diag(1, ..., 1, det(U V^T)) V^T
    "plain" unwhitened comparator: J0 = X0^T Y0 with X0 = A P_H, Y0 = B P_E,
            R0 = U0 V0^T, a0 = trace(R0^T J0) / ||X0||_F^2 fitted on FIT only,
            Z_plain = mu_E + a0 ((H_q - mu_H) P_H R0) P_E^T

The "so" optimum is defined on the fitted PCA coordinates, whose column signs
are a property of the fit SVD output; the O(r) map is invariant under
consistent sign flips of those coordinates while the SO(r) map is not. The
"plain" comparator keeps the same tensor set; a0 is a binding scalar because it
is a scalar fitted statistic, and it is exactly 0.0 for the other variants.

Artifact format. One byte string:

    MAGIC(8) | header_length(8, big-endian) | header(ASCII JSON) | payload

The header is a flat dictionary of scalars plus two bounded shape/digest maps.
It carries the full binding (method and variant identity, rank and whitening
hyperparameters, dimensions, row count, rotation defect and determinant,
numerical ranks and participation ratios, whitening gains, model, checkpoint,
tokenizer, embedding and capture identity, capture span policy, predictor
position offset, precision, fit manifest digest and fit corpus digest) and two
content digests. It is validated in full before any tensor byte is interpreted:
schema, exact field set, identifier and digest syntax, expected method id,
expected provenance, tensor shapes against hidden_dim and r, payload length
and sha256, and the artifact digest. The payload is a torch.save archive of
plain tensors only and is always read with torch.load(weights_only=True,
map_location="cpu"). The payload bytes are little-endian float32, so a
big-endian host is refused instead of reinterpreting them.

Everything here is pure: torch is passed in, nothing is imported at module
load. fit_map has no internal quality gate and no internal split: a no-signal
corpus fits and is recorded honestly, and promotion is decided elsewhere from
an independent validation record bound to the artifact.
"""
from __future__ import annotations

import array
import hashlib
import io
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, replace

from .latent_types import LatentError, require

METHOD_ID = "statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes.v1"
METHOD_ID_SO = "statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-so.v1"
METHOD_ID_PLAIN = "statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-plain.v1"
VARIANTS = ("o", "so", "plain")
VARIANT_METHOD_IDS = {"o": METHOD_ID, "so": METHOD_ID_SO, "plain": METHOD_ID_PLAIN}
HYPERPARAMETER_FIELDS = ("r", "eta", "variant")
DEFAULT_HYPERPARAMETERS = {"r": 128, "eta": 0.01, "variant": "o"}
RANK_GRID = (16, 32, 64, 128, 256)
ETA_GRID = (0.001, 0.01, 0.1)

ARTIFACT_SCHEMA = "optimus.occ3.calibration-artifact.lowrank-procrustes.v1"
CORPUS_DIGEST_DOMAIN = "optimus.occ3.calibration-corpus.v1"
ARTIFACT_MAGIC = b"OCC3PRD1"
PRECISION = "float32"
PREDICTOR_POSITION_OFFSET = -1
MIN_FIT_ROWS = 64
MAX_FIT_ROWS = 2048
MAX_CORPUS_TURNS = 4096
MAX_HEADER_BYTES = 8192
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
RANK_RELATIVE_TOLERANCE = 1e-5
ROTATION_DEFECT_LIMIT = 1e-3
DETERMINANT_LIMIT = 1e-3
NORM_FLOOR = 1e-6
PARTITION_FIELDS = frozenset({"hidden", "embeddings", "row_turn_ids", "turn_ids",
                              "manifest_digest"})
TENSOR_NAMES = ("mean_hidden", "mean_embedding", "basis_hidden", "basis_embedding",
                "scale_hidden", "scale_embedding", "rotation")
BINDING_FIELDS = frozenset({
    "method_id", "variant", "r", "eta", "a0", "rotation_defect", "rotation_determinant",
    "hidden_dim", "rows", "layer_index", "model_fingerprint", "checkpoint_revision",
    "tokenizer_fingerprint", "embedding_id", "capture_method_id", "capture_span_policy",
    "predictor_position_offset", "precision", "fit_manifest_digest", "corpus_digest",
    "rank_hidden", "rank_embedding", "participation_hidden", "participation_embedding",
    "whitening_gain_hidden", "whitening_gain_embedding",
    "validation_record_digest", "validation_artifact_digest", "serving",
})
HEADER_FIELDS = BINDING_FIELDS | frozenset({
    "schema", "byte_order", "tensor_dtype", "tensor_shapes", "tensor_digest",
    "payload_sha256", "payload_bytes", "artifact_digest",
})
DIGEST_EXCLUDED = frozenset({"artifact_digest", "payload_sha256", "payload_bytes"})
PROVENANCE_FIELDS = ("layer_index", "model_fingerprint", "checkpoint_revision",
                     "tokenizer_fingerprint", "embedding_id", "capture_method_id",
                     "capture_span_policy", "predictor_position_offset", "precision")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")


@dataclass(frozen=True)
class ProcrustesProvenance:
    """Caller-supplied identity of the model that produced the FIT partition.

    None of these values is derived from the corpus rows: the caller supplies
    what the runtime negotiates, and deserialization compares the saved copy
    against it. The capture span policy names the semantic-span convention the
    corpus rows were captured under, so an artifact cannot be silently reused
    across span policies.
    """
    layer_index: int
    model_fingerprint: str
    checkpoint_revision: str
    tokenizer_fingerprint: str
    embedding_id: str
    capture_method_id: str
    capture_span_policy: str
    predictor_position_offset: int = PREDICTOR_POSITION_OFFSET
    precision: str = PRECISION

    def __post_init__(self):
        require(_uint(self.layer_index, 0, 2**32 - 1), "profile")
        for name in ("model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
                     "embedding_id", "capture_method_id", "capture_span_policy"):
            require(_identifier(getattr(self, name)), "profile")
        require(self.predictor_position_offset == PREDICTOR_POSITION_OFFSET, "profile")
        require(self.precision == PRECISION, "profile")


@dataclass(frozen=True)
class ProcrustesBinding:
    """Content-independent identity of one fitted transform."""
    method_id: str
    variant: str
    r: int
    eta: float
    a0: float
    rotation_defect: float
    rotation_determinant: float
    hidden_dim: int
    rows: int
    layer_index: int
    model_fingerprint: str
    checkpoint_revision: str
    tokenizer_fingerprint: str
    embedding_id: str
    capture_method_id: str
    capture_span_policy: str
    predictor_position_offset: int
    precision: str
    fit_manifest_digest: str
    corpus_digest: str
    rank_hidden: int
    rank_embedding: int
    participation_hidden: float
    participation_embedding: float
    whitening_gain_hidden: float
    whitening_gain_embedding: float
    validation_record_digest: str = ""
    validation_artifact_digest: str = ""
    serving: object = None

    def __post_init__(self):
        require(self.variant in VARIANTS, "identity")
        require(self.method_id == VARIANT_METHOD_IDS[self.variant], "identity")
        require(_uint(self.hidden_dim, 3, 2**20), "profile")
        require(_uint(self.rows, MIN_FIT_ROWS, MAX_FIT_ROWS), "profile")
        require(type(self.r) is int and self.r in RANK_GRID
                and self.r <= min(self.hidden_dim, self.rows - 1), "profile")
        require(_finite(self.eta) and any(self.eta == value for value in ETA_GRID), "profile")
        require(_finite(self.a0), "numerical")
        require(self.variant == "plain" or self.a0 == 0.0, "identity")
        require(_finite(self.rotation_defect) and 0.0 <= self.rotation_defect
                <= ROTATION_DEFECT_LIMIT, "numerical")
        require(_finite(self.rotation_determinant), "numerical")
        require(abs(abs(self.rotation_determinant) - 1.0) <= DETERMINANT_LIMIT, "numerical")
        require(self.variant != "so" or abs(self.rotation_determinant - 1.0)
                <= DETERMINANT_LIMIT, "identity")
        require(_uint(self.layer_index, 0, 2**32 - 1), "profile")
        for name in ("model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
                     "embedding_id", "capture_method_id", "capture_span_policy"):
            require(_identifier(getattr(self, name)), "identity")
        require(self.predictor_position_offset == PREDICTOR_POSITION_OFFSET, "profile")
        require(self.precision == PRECISION, "profile")
        require(type(self.fit_manifest_digest) is str
                and (self.fit_manifest_digest == "" or _digest(self.fit_manifest_digest)),
                "identity")
        require(_digest(self.corpus_digest), "identity")
        for name in ("validation_record_digest", "validation_artifact_digest"):
            value = getattr(self, name)
            require(type(value) is str and (value == "" or _digest(value)), "identity")
        require(self.serving is None
                or (type(self.serving) is dict and set(self.serving) == {"kind"}
                    and self.serving["kind"] == "n1"), "profile")
        span = min(self.rows, self.hidden_dim)
        require(_uint(self.rank_hidden, 1, span) and _uint(self.rank_embedding, 1, span),
                "profile")
        require(self.r <= min(self.rank_hidden, self.rank_embedding), "profile")
        for name in ("participation_hidden", "participation_embedding"):
            value = getattr(self, name)
            require(_finite(value) and 1.0 <= value <= float(span) + 1e-6, "numerical")
        for name in ("whitening_gain_hidden", "whitening_gain_embedding"):
            value = getattr(self, name)
            require(_finite(value) and value > 0.0, "numerical")

    def scalars(self):
        """The flat scalar header of the binding; no tensor content."""
        return {"method_id": self.method_id, "variant": self.variant, "r": self.r,
                "eta": self.eta, "a0": self.a0,
                "rotation_defect": self.rotation_defect,
                "rotation_determinant": self.rotation_determinant,
                "hidden_dim": self.hidden_dim, "rows": self.rows,
                "layer_index": self.layer_index,
                "model_fingerprint": self.model_fingerprint,
                "checkpoint_revision": self.checkpoint_revision,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "embedding_id": self.embedding_id,
                "capture_method_id": self.capture_method_id,
                "capture_span_policy": self.capture_span_policy,
                "predictor_position_offset": self.predictor_position_offset,
                "precision": self.precision,
                "fit_manifest_digest": self.fit_manifest_digest,
                "corpus_digest": self.corpus_digest,
                "rank_hidden": self.rank_hidden, "rank_embedding": self.rank_embedding,
                "participation_hidden": self.participation_hidden,
                "participation_embedding": self.participation_embedding,
                "whitening_gain_hidden": self.whitening_gain_hidden,
                "whitening_gain_embedding": self.whitening_gain_embedding,
                "validation_record_digest": self.validation_record_digest,
                "validation_artifact_digest": self.validation_artifact_digest,
                "serving": self.serving}


@dataclass(frozen=True)
class ProcrustesArtifact:
    method_id: str
    variant: str
    r: int
    eta: float
    a0: float
    hidden_dim: int
    rows: int
    mean_hidden: object
    mean_embedding: object
    basis_hidden: object
    basis_embedding: object
    scale_hidden: object
    scale_embedding: object
    rotation: object
    binding: object = None
    content_digest: object = None


def _uint(value, minimum=0, maximum=(1 << 64) - 1):
    return type(value) is int and minimum <= value <= maximum


def _finite(value):
    return type(value) is float and math.isfinite(value)


def _digest(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _byte_order():
    require(sys.byteorder == "little", "profile")
    return sys.byteorder


def _torch(_reference):
    import torch
    return torch


def _tensor_bytes(matrix, torch):
    """Little-endian float32 image of one tensor, as bytes.

    torch.storage.UntypedStorage.__bytes__ is a per-byte loop (measured 4.7 s
    per MiB on torch 2.11), so the storage never reaches the digest. numpy
    gives a memcpy and is already part of the serving image; the array fallback
    keeps the digest defined when it is absent. Both paths produce the same
    bytes.
    """
    require(getattr(matrix, "ndim", 0) in (1, 2), "profile")
    flat = matrix.detach().to(dtype=torch.float32, device="cpu", copy=True).reshape(-1)
    require(int(flat.numel()) == int(matrix.numel()), "profile")
    try:
        import numpy
    except ImportError:
        raw = array.array("f", flat.tolist()).tobytes()
    else:
        raw = bytes(memoryview(numpy.ascontiguousarray(flat.numpy())))
    require(len(raw) == int(flat.numel()) * 4, "numerical")
    return raw


def _tensor_digest(matrix, torch, tag):
    value = hashlib.sha256()
    shape = "x".join(str(int(x)) for x in matrix.shape)
    value.update(f"{tag}:{shape}:{PRECISION}:".encode("ascii"))
    value.update(_tensor_bytes(matrix, torch))
    return value.hexdigest()


def _tensor_set_digest(artifact, torch):
    value = hashlib.sha256()
    for name in TENSOR_NAMES:
        value.update(_tensor_digest(getattr(artifact, name), torch, name).encode("ascii"))
    return value.hexdigest()


def corpus_digest(hidden, embeddings, *, turn_ids=None, torch_module=None):
    """Digest of the FIT rows and the ordered turn ids they came from.

    The digest binds content, so a different corpus cannot be presented under a
    saved artifact, and it stores no row and no turn row count that could be
    inverted into a hidden state.
    """
    torch = torch_module or _torch(hidden)
    require(hidden.ndim == 2 and embeddings.ndim == 2 and hidden.shape == embeddings.shape, "profile")
    require(int(hidden.shape[0]) > 0 and int(hidden.shape[1]) > 2, "profile")
    ids = () if turn_ids is None else tuple(turn_ids)
    require(len(ids) <= MAX_CORPUS_TURNS, "profile")
    require(all(_identifier(value) for value in ids), "profile")
    require(len(set(ids)) == len(ids), "profile")
    value = {"domain": CORPUS_DIGEST_DOMAIN, "rows": int(hidden.shape[0]),
             "hidden_dim": int(hidden.shape[1]), "dtype": PRECISION,
             "byte_order": _byte_order(), "turn_ids": list(ids),
             "hidden": _tensor_digest(hidden, torch, "hidden"),
             "embeddings": _tensor_digest(embeddings, torch, "embeddings")}
    return hashlib.sha256(_canonical(value)).hexdigest()


def _expected_shapes(rank, hidden_dim):
    return {"mean_hidden": [1, hidden_dim], "mean_embedding": [1, hidden_dim],
            "basis_hidden": [hidden_dim, rank], "basis_embedding": [hidden_dim, rank],
            "scale_hidden": [rank], "scale_embedding": [rank],
            "rotation": [rank, rank]}


def _require_consistent(artifact, torch, *, check_values=True):
    """Every scalar, the tensor shapes and (optionally) the tensor values agree."""
    require(isinstance(artifact, ProcrustesArtifact), "identity")
    require(artifact.variant in VARIANTS, "identity")
    require(artifact.method_id == VARIANT_METHOD_IDS[artifact.variant], "identity")
    require(_uint(artifact.hidden_dim, 3, 2**20), "profile")
    require(_uint(artifact.rows, MIN_FIT_ROWS, MAX_FIT_ROWS), "profile")
    require(type(artifact.r) is int and artifact.r in RANK_GRID
            and artifact.r <= min(artifact.hidden_dim, artifact.rows - 1), "profile")
    require(_finite(artifact.eta) and any(artifact.eta == value for value in ETA_GRID), "profile")
    require(_finite(artifact.a0), "numerical")
    require(artifact.variant == "plain" or artifact.a0 == 0.0, "identity")
    shapes = _expected_shapes(artifact.r, artifact.hidden_dim)
    for name, shape in shapes.items():
        value = getattr(artifact, name)
        require(tuple(int(x) for x in value.shape) == tuple(shape), "profile")
        require(value.dtype == torch.float32, "profile")
        if check_values:
            require(bool(torch.isfinite(value).all()), "numerical")
    binding = artifact.binding
    if binding is not None:
        require(isinstance(binding, ProcrustesBinding), "identity")
        for name in ("method_id", "variant", "r", "eta", "a0", "hidden_dim", "rows"):
            require(getattr(binding, name) == getattr(artifact, name), "identity")
    require(artifact.content_digest is None or _digest(artifact.content_digest), "identity")
    return artifact


def _binding_header(binding):
    header = dict(binding.scalars())
    require(set(header) == set(BINDING_FIELDS), "profile")
    header["schema"] = ARTIFACT_SCHEMA
    header["byte_order"] = _byte_order()
    header["tensor_dtype"] = PRECISION
    header["tensor_shapes"] = _expected_shapes(binding.r, binding.hidden_dim)
    return header


def _artifact_digest(header):
    body = {name: value for name, value in header.items() if name not in DIGEST_EXCLUDED}
    return hashlib.sha256(_canonical(body)).hexdigest()


def artifact_digest(artifact, *, torch_module=None):
    """Content digest of a bound artifact: binding, shapes and tensor values.

    O(tensor bytes): this reads every tensor once and is not part of describe().
    """
    torch = torch_module or _torch(None)
    _require_consistent(artifact, torch)
    require(artifact.binding is not None, "profile")
    header = _binding_header(artifact.binding)
    header["tensor_digest"] = _tensor_set_digest(artifact, torch)
    return _artifact_digest(header)


def required_provenance_fields(artifact_class):
    """The provenance fields deserialization compares against expected_provenance."""
    require(artifact_class in (ProcrustesProvenance, ProcrustesBinding,
                               ProcrustesArtifact), "profile")
    return PROVENANCE_FIELDS


def _numerical_rank(singular, torch):
    """Count of singular directions above the documented relative tolerance."""
    require(bool(torch.isfinite(singular).all()), "numerical")
    largest = float(singular[0])
    require(largest > 0.0, "numerical")
    floor = largest * RANK_RELATIVE_TOLERANCE
    return int((singular > floor).sum())


def _participation(singular):
    """Participation ratio (sum s^2)^2 / sum s^4 over the full spectrum."""
    values = singular.double()
    energy = values.pow(2).sum()
    quartic = values.pow(4).sum()
    require(float(energy) > 0.0 and float(quartic) > 0.0, "numerical")
    return float(energy * energy / quartic)


def _hyperparameters(hyperparams):
    require(hasattr(hyperparams, "get"), "profile")
    names = tuple(HYPERPARAMETER_FIELDS)
    for name in hyperparams:
        require(name in names, "profile")
    values = dict(DEFAULT_HYPERPARAMETERS)
    values.update({name: hyperparams[name] for name in hyperparams})
    require(type(values["r"]) is int and values["r"] in RANK_GRID, "profile")
    eta = values["eta"]
    require(type(eta) in (int, float) and not isinstance(eta, bool)
            and math.isfinite(eta) and any(float(eta) == value for value in ETA_GRID),
            "profile")
    values["eta"] = float(eta)
    require(values["variant"] in VARIANTS, "profile")
    return values


def _partition_rows(partition):
    """Validate the frozen partition shape and return its checked contents."""
    require(hasattr(partition, "get") and set(partition) == set(PARTITION_FIELDS), "profile")
    hidden, embeddings = partition["hidden"], partition["embeddings"]
    require(hasattr(hidden, "ndim") and hasattr(embeddings, "ndim"), "profile")
    require(hidden.ndim == 2 and embeddings.ndim == 2, "profile")
    require(hidden.shape == embeddings.shape, "profile")
    rows, dim = int(hidden.shape[0]), int(hidden.shape[1])
    require(MIN_FIT_ROWS <= rows <= MAX_FIT_ROWS, "profile")
    require(3 <= dim <= 2**20, "profile")
    require(hidden.device == embeddings.device, "profile")
    row_turn_ids = partition["row_turn_ids"]
    turn_ids = partition["turn_ids"]
    require(isinstance(row_turn_ids, (tuple, list)) and len(row_turn_ids) == rows, "profile")
    require(isinstance(turn_ids, (tuple, list)) and len(turn_ids) <= MAX_CORPUS_TURNS, "profile")
    require(all(_identifier(value) for value in row_turn_ids), "profile")
    require(all(_identifier(value) for value in turn_ids), "profile")
    require(len(set(turn_ids)) == len(turn_ids), "profile")
    require(set(turn_ids) == set(row_turn_ids), "profile")
    manifest_digest = partition["manifest_digest"]
    require(type(manifest_digest) is str
            and (manifest_digest == "" or _digest(manifest_digest)), "profile")
    return (hidden, embeddings, tuple(row_turn_ids), tuple(turn_ids), manifest_digest,
            rows, dim)


def fit_map(partition, hyperparams, *, provenance=None, torch_module=None):
    """Fit and freeze the transform on the FIT partition only.

    No internal quality gate and no internal split: numerical guards stay
    (finite rows, dimensions, rank support, rotation defect), and promotion is
    decided elsewhere from an independent validation record. provenance is
    optional so an offline study can fit without model identity, but such an
    artifact cannot be serialized.
    """
    torch = torch_module or _torch(partition.get("hidden") if hasattr(partition, "get") else None)
    require(provenance is None or isinstance(provenance, ProcrustesProvenance), "profile")
    hyper = _hyperparameters(hyperparams)
    rank_request, eta, variant = hyper["r"], hyper["eta"], hyper["variant"]
    (hidden, embeddings, _row_turn_ids, turn_ids, manifest_digest, rows, dim) = (
        _partition_rows(partition))
    require(rank_request <= min(dim, rows - 1), "profile")
    with torch.no_grad():
        h, e = hidden.float(), embeddings.float()
        require(bool(torch.isfinite(h).all()) and bool(torch.isfinite(e).all()), "numerical")
        mu_h, mu_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
        a, b = h - mu_h, e - mu_e
        energy_h = float(a.pow(2).sum())
        energy_e = float(b.pow(2).sum())
        require(energy_h > 0.0 and energy_e > 0.0, "numerical")
        try:
            _, s_h, vh_h = torch.linalg.svd(a, full_matrices=False)
            _, s_e, vh_e = torch.linalg.svd(b, full_matrices=False)
        except RuntimeError:
            raise LatentError("numerical") from None
        rank_h = _numerical_rank(s_h, torch)
        rank_e = _numerical_rank(s_e, torch)
        # Never retain null directions to reach the requested rank: the request
        # is refused when either centred matrix does not support it.
        require(rank_request <= min(rank_h, rank_e), "numerical")
        p_h = vh_h[:rank_request].T.contiguous()
        p_e = vh_e[:rank_request].T.contiguous()
        g_h = s_h[:rank_request].double().pow(2) / rows
        g_e = s_e[:rank_request].double().pow(2) / rows
        tau_h = eta * float(g_h.mean())
        tau_e = eta * float(g_e.mean())
        require(math.isfinite(tau_h) and tau_h > 0.0, "numerical")
        require(math.isfinite(tau_e) and tau_e > 0.0, "numerical")
        d_h64 = (g_h + tau_h).sqrt()
        d_e64 = (g_e + tau_e).sqrt()
        require(bool(torch.isfinite(d_h64).all()) and bool(torch.isfinite(d_e64).all()),
                "numerical")
        d_h, d_e = d_h64.to(torch.float32), d_e64.to(torch.float32)
        if variant == "plain":
            x, y = a @ p_h, b @ p_e
        else:
            x, y = (a @ p_h) / d_h, (b @ p_e) / d_e
        j = x.T @ y
        try:
            u, _s_j, vh = torch.linalg.svd(j, full_matrices=False)
        except RuntimeError:
            raise LatentError("numerical") from None
        if variant == "so":
            determinant = float(torch.det(u.double() @ vh.double()))
            flip = torch.ones(rank_request, dtype=torch.float32)
            flip[-1] = determinant
            rotation = u @ torch.diag(flip) @ vh
        else:
            rotation = u @ vh
        a0 = 0.0
        if variant == "plain":
            denominator = float(x.double().pow(2).sum())
            require(denominator > 0.0, "numerical")
            a0 = float(torch.trace(rotation.double().T @ j.double()) / denominator)
            require(math.isfinite(a0), "numerical")
        rotation64 = rotation.double()
        eye = torch.eye(rank_request, dtype=torch.float64)
        defect = float((rotation64.T @ rotation64 - eye).norm())
        require(math.isfinite(defect) and 0.0 <= defect <= ROTATION_DEFECT_LIMIT, "numerical")
        determinant = float(torch.det(rotation64))
        require(abs(abs(determinant) - 1.0) <= DETERMINANT_LIMIT, "numerical")
        require(variant != "so" or abs(determinant - 1.0) <= DETERMINANT_LIMIT, "numerical")
        participation_h = _participation(s_h)
        participation_e = _participation(s_e)
        gain_h = float(1.0 / d_h64.min())
        gain_e = float(1.0 / d_e64.min())
    binding = None
    if provenance is not None:
        binding = ProcrustesBinding(
            method_id=VARIANT_METHOD_IDS[variant], variant=variant, r=rank_request,
            eta=eta, a0=a0, rotation_defect=defect, rotation_determinant=determinant,
            hidden_dim=dim, rows=rows, layer_index=provenance.layer_index,
            model_fingerprint=provenance.model_fingerprint,
            checkpoint_revision=provenance.checkpoint_revision,
            tokenizer_fingerprint=provenance.tokenizer_fingerprint,
            embedding_id=provenance.embedding_id,
            capture_method_id=provenance.capture_method_id,
            capture_span_policy=provenance.capture_span_policy,
            predictor_position_offset=provenance.predictor_position_offset,
            precision=provenance.precision,
            fit_manifest_digest=manifest_digest,
            corpus_digest=corpus_digest(h, e, turn_ids=turn_ids, torch_module=torch),
            rank_hidden=rank_h, rank_embedding=rank_e,
            participation_hidden=participation_h, participation_embedding=participation_e,
            whitening_gain_hidden=gain_h, whitening_gain_embedding=gain_e)
    artifact = ProcrustesArtifact(method_id=VARIANT_METHOD_IDS[variant], variant=variant,
                                   r=rank_request, eta=eta, a0=a0, hidden_dim=dim, rows=rows,
                                   mean_hidden=mu_h.contiguous(), mean_embedding=mu_e.contiguous(),
                                   basis_hidden=p_h, basis_embedding=p_e,
                                   scale_hidden=d_h.contiguous(), scale_embedding=d_e.contiguous(),
                                   rotation=rotation.contiguous(), binding=binding)
    if artifact.binding is not None:
        artifact = replace(artifact, content_digest=artifact_digest(artifact, torch_module=torch))
    return artifact


_FACTOR_CACHE_LIMIT = 8
_FACTOR_CACHE = OrderedDict()


def _apply_bundle(artifact, device, torch):
    """Device-resident apply tensors for one artifact, built once per device."""
    key = (id(artifact), str(device))
    entry = _FACTOR_CACHE.get(key)
    if entry is not None and entry[0] is artifact:
        return entry[1]

    def move(value):
        return value.detach().to(device=device, dtype=torch.float32).contiguous()

    bundle = {"mean_hidden": move(artifact.mean_hidden),
              "mean_embedding": move(artifact.mean_embedding),
              "basis_hidden": move(artifact.basis_hidden),
              "basis_hidden_transposed": move(artifact.basis_hidden.T.contiguous()),
              "basis_embedding": move(artifact.basis_embedding),
              "basis_embedding_transposed": move(artifact.basis_embedding.T.contiguous()),
              "scale_hidden": move(artifact.scale_hidden),
              "scale_embedding": move(artifact.scale_embedding),
              "rotation": move(artifact.rotation)}
    _FACTOR_CACHE[key] = (artifact, bundle)
    while len(_FACTOR_CACHE) > _FACTOR_CACHE_LIMIT:
        _FACTOR_CACHE.popitem(last=False)
    return bundle


def apply_map(artifact, hidden, *, torch_module=None):
    """Map new hidden rows into the embedding space with the frozen transform.

    Returns Z_raw: no reference embedding of the message is read and no row is
    renormalised. The applied rows never enter the fit, so replacing them
    changes the output and never changes the artifact.
    """
    torch = torch_module or _torch(hidden)
    _require_consistent(artifact, torch, check_values=False)
    require(hidden.ndim == 2 and hidden.shape[1] == artifact.hidden_dim, "profile")
    require(int(hidden.shape[0]) >= 1, "profile")
    with torch.no_grad():
        h = hidden.float()
        require(bool(torch.isfinite(h).all()), "numerical")
        bundle = _apply_bundle(artifact, hidden.device, torch)
        centred = h - bundle["mean_hidden"]
        if artifact.variant == "plain":
            reduced = centred @ bundle["basis_hidden"]
            z = bundle["mean_embedding"] + artifact.a0 * (
                reduced @ bundle["rotation"] @ bundle["basis_embedding_transposed"])
        else:
            reduced = (centred @ bundle["basis_hidden"]) / bundle["scale_hidden"]
            z = bundle["mean_embedding"] + (
                (reduced @ bundle["rotation"]) * bundle["scale_embedding"]
            ) @ bundle["basis_embedding_transposed"]
        require(bool(torch.isfinite(z).all()), "numerical")
    return z.to(dtype=hidden.dtype).contiguous()


def residual_discard_ratio(artifact, hidden, *, torch_module=None):
    """Frobenius fraction of centred query rows that lies outside span(P_H).

    Components outside span(P_H) are discarded by apply_map; this measures that
    loss for the evaluation record. Zero centred energy is refused rather than
    reported as an undefined ratio.
    """
    torch = torch_module or _torch(hidden)
    _require_consistent(artifact, torch, check_values=False)
    require(hidden.ndim == 2 and hidden.shape[1] == artifact.hidden_dim, "profile")
    with torch.no_grad():
        bundle = _apply_bundle(artifact, hidden.device, torch)
        centred = hidden.float() - bundle["mean_hidden"]
        total = float(centred.norm())
        require(total > 0.0, "numerical")
        projection = (centred @ bundle["basis_hidden"]) @ bundle["basis_hidden_transposed"]
        residual = float((centred - projection).norm())
    return residual / total


def serving_transform(config, z_raw, *, nu=None, torch_module=None):
    """Apply the selected serving normalisation to raw aligned rows.

    config None is N0: the rows are returned unchanged. config {"kind": "n1"}
    renormalises every row to the fixed vocabulary norm statistic nu, which is
    measured from the frozen embedding matrix by the caller and never derived
    from evaluation answers. Norms at or below NORM_FLOOR, and nonfinite norms,
    are refused rather than fabricated into a carrier.
    """
    torch = torch_module or _torch(z_raw)
    require(hasattr(z_raw, "ndim") and z_raw.ndim == 2, "profile")
    if config is None:
        return z_raw
    require(type(config) is dict and set(config) == {"kind"} and config["kind"] == "n1",
            "profile")
    require(type(nu) in (int, float) and not isinstance(nu, bool) and math.isfinite(nu)
            and nu > 0, "profile")
    with torch.no_grad():
        z = z_raw.float()
        norms = z.norm(dim=-1, keepdim=True)
        require(bool(torch.isfinite(norms).all()), "numerical")
        require(bool((norms > NORM_FLOOR).all()), "numerical")
        served = z * (float(nu) / norms)
        require(bool(torch.isfinite(served).all()), "numerical")
    return served.to(dtype=z_raw.dtype).contiguous()


def artifact_bytes(artifact):
    """Resident size of the frozen transform, for the live-bytes budget."""
    total = 0
    for name in TENSOR_NAMES:
        value = getattr(artifact, name)
        total += value.numel() * value.element_size()
    return total


def describe(artifact):
    """Scalar-only summary; safe for a receipt. Never reads tensor content."""
    summary = {"method_id": artifact.method_id, "variant": artifact.variant,
               "r": artifact.r, "eta": artifact.eta, "a0": artifact.a0,
               "rows": artifact.rows, "hidden_dim": artifact.hidden_dim,
               "bytes": artifact_bytes(artifact),
               "bound": 1 if artifact.binding is not None else 0}
    if artifact.binding is not None:
        scalars = {name: value for name, value in artifact.binding.scalars().items()
                   if name != "serving"}
        summary.update(scalars)
        summary["serving_kind"] = ("none" if artifact.binding.serving is None
                                  else artifact.binding.serving["kind"])
    return summary


def _payload(artifact, torch):
    buffer = io.BytesIO()
    torch.save({name: getattr(artifact, name).detach().to("cpu").contiguous()
                for name in TENSOR_NAMES}, buffer)
    return buffer.getvalue()


def serialize_artifact(artifact, *, torch_module=None):
    """Bind, hash and encode one artifact. Returns the exact saved byte string."""
    torch = torch_module or _torch(None)
    _require_consistent(artifact, torch)
    require(artifact.binding is not None, "profile")
    header = _binding_header(artifact.binding)
    header["tensor_digest"] = _tensor_set_digest(artifact, torch)
    payload = _payload(artifact, torch)
    header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    header["payload_bytes"] = len(payload)
    header["artifact_digest"] = _artifact_digest(header)
    if artifact.content_digest is not None:
        require(artifact.content_digest == header["artifact_digest"], "identity")
    require(set(header) == set(HEADER_FIELDS), "profile")
    raw_header = _canonical(header)
    require(len(raw_header) <= MAX_HEADER_BYTES, "quota")
    raw = ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload
    require(len(raw) <= MAX_ARTIFACT_BYTES, "quota")
    return raw


def _decode_header(raw):
    require(0 < len(raw) <= MAX_HEADER_BYTES, "profile")

    def pairs(items):
        result = {}
        for key, value in items:
            require(type(key) is str and key not in result, "profile")
            result[key] = value
        return result

    def constant(_value):
        require(False, "profile")

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs, parse_constant=constant)
    except LatentError:
        raise
    except Exception:
        raise LatentError("profile") from None
    require(type(value) is dict, "profile")
    return value


def _validate_header(header):
    """Every header field, then the shape binding. Touches no tensor."""
    require(set(header) == set(HEADER_FIELDS), "profile")
    require(header["schema"] == ARTIFACT_SCHEMA, "profile")
    require(header["byte_order"] == _byte_order() and header["tensor_dtype"] == PRECISION, "profile")
    require(_identifier(header["method_id"]), "identity")
    require(_uint(header["hidden_dim"], 3, 2**20), "profile")
    require(_uint(header["rows"], MIN_FIT_ROWS, MAX_FIT_ROWS), "profile")
    require(type(header["r"]) is int and header["r"] in RANK_GRID, "profile")
    require(header["tensor_shapes"] == _expected_shapes(header["r"], header["hidden_dim"]),
            "profile")
    for name in ("tensor_digest", "payload_sha256", "artifact_digest", "corpus_digest"):
        require(_digest(header[name]), "identity")
    require(type(header["fit_manifest_digest"]) is str
            and (header["fit_manifest_digest"] == "" or _digest(header["fit_manifest_digest"])),
            "identity")
    require(_uint(header["payload_bytes"], 1, MAX_ARTIFACT_BYTES), "profile")
    return ProcrustesBinding(**{name: header[name] for name in BINDING_FIELDS})


def deserialize_artifact(raw, *, expected_method_id=None, expected_hidden_dim=None,
                         expected_provenance=None, expected_artifact_digest=None,
                         torch_module=None):
    """Validate the header in full, then and only then read the tensor payload.

    Every refusal is LatentError: "identity" for a method, provenance or digest
    mismatch, "profile" for a malformed header or shape, "numerical" for a
    non-finite tensor. Without an expected method id only this module's own
    variant identities are accepted, so a dual-ridge or per-message artifact
    is refused rather than silently reinterpreted.
    """
    torch = torch_module or _torch(None)
    require(isinstance(raw, (bytes, bytearray)), "profile")
    raw = bytes(raw)
    require(len(raw) <= MAX_ARTIFACT_BYTES, "quota")
    require(raw[:len(ARTIFACT_MAGIC)] == ARTIFACT_MAGIC and len(raw) >= 16, "profile")
    header_length = int.from_bytes(raw[len(ARTIFACT_MAGIC):16], "big")
    require(0 < header_length <= MAX_HEADER_BYTES and 16 + header_length < len(raw), "profile")
    header = _decode_header(raw[16:16 + header_length])
    binding = _validate_header(header)
    if expected_method_id is None:
        require(header["method_id"] in set(VARIANT_METHOD_IDS.values()), "identity")
    else:
        require(_identifier(expected_method_id), "profile")
        require(header["method_id"] == expected_method_id, "identity")
    if expected_hidden_dim is not None:
        require(header["hidden_dim"] == expected_hidden_dim, "profile")
    if expected_provenance is not None:
        require(isinstance(expected_provenance, ProcrustesProvenance), "profile")
        for name in required_provenance_fields(type(expected_provenance)):
            require(header[name] == getattr(expected_provenance, name), "identity")
    require(header["artifact_digest"] == _artifact_digest(header), "identity")
    if expected_artifact_digest is not None:
        require(_digest(expected_artifact_digest)
                and header["artifact_digest"] == expected_artifact_digest, "identity")
    payload = raw[16 + header_length:]
    require(len(payload) == header["payload_bytes"], "identity")
    require(hashlib.sha256(payload).hexdigest() == header["payload_sha256"], "identity")
    try:
        loaded = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    except Exception:
        raise LatentError("profile") from None
    require(type(loaded) is dict and set(loaded) == set(TENSOR_NAMES), "profile")
    for name in TENSOR_NAMES:
        value = loaded[name]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32, "profile")
        require([int(x) for x in value.shape] == header["tensor_shapes"][name], "profile")
        require(bool(torch.isfinite(value).all()), "numerical")
    artifact = ProcrustesArtifact(method_id=header["method_id"], variant=header["variant"],
                                  r=header["r"], eta=header["eta"], a0=header["a0"],
                                  hidden_dim=header["hidden_dim"], rows=header["rows"],
                                  mean_hidden=loaded["mean_hidden"],
                                  mean_embedding=loaded["mean_embedding"],
                                  basis_hidden=loaded["basis_hidden"],
                                  basis_embedding=loaded["basis_embedding"],
                                  scale_hidden=loaded["scale_hidden"],
                                  scale_embedding=loaded["scale_embedding"],
                                  rotation=loaded["rotation"], binding=binding,
                                  content_digest=header["artifact_digest"])
    _require_consistent(artifact, torch)
    require(_tensor_set_digest(artifact, torch) == header["tensor_digest"], "identity")
    return artifact
