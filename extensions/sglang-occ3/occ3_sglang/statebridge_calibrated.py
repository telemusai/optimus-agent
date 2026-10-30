"""Frozen dual-ridge alignment fitted on one independent calibration partition.

The per-message StateBridge fit uses the transmitted rows' own embeddings as the
Procrustes target, so it reproduces those embeddings and discards the state
(see PRODUCTION-SCALE-DEGENERACY.md and MATHEMATICAL-DEGENERACY.md). This module
fits the transform on a calibration corpus collected earlier, freezes it, and
applies it to a message whose embeddings were never part of the fit.

Method (dual form of ridge regression from hidden rows to embedding rows):

    A = H_cal - mean(H_cal)                     M x D
    B = E_cal - mean(E_cal)                     M x D
    G = symmetrise(A A^T)                        M x M
    lambda = rho * trace(G) / M
    C = solve(G + lambda I, B)                  M x D
    Z_raw = mean(E_cal) + ((H_msg - mean(H_cal)) A^T) C        N x D

The solve is an M x M Cholesky solve, never an explicit inverse and never a
D x D matrix. The dual coefficient C is derived once per immutable artifact and
device and cached; a message never repeats the M x M factorisation. The full
map would be W = A^T C, which the serving path never materialises.

Separation of concerns (spec section 8): fit_map fits exactly the partition it
is handed and never gates quality; apply_map returns raw rows and never
renormalises; evaluate_map in calibration_corpus.py records metrics on an
independent partition without refusing poor quality; validate_for_activation is
the only promotion gate. A no-signal corpus therefore fits, is measured and is
recorded, and is refused only at promotion.

Method identity: this is NOT the shipped per-message solver. Any use must carry
its own method identifier; see CALIBRATED-METHOD.md for the versioning and
integration plan.

Artifact format. A saved artifact is one byte string:

    MAGIC(8) | header_length(8, big-endian) | header(ASCII JSON) | payload

The header is a flat dictionary of scalars plus bounded nested maps (tensor
shapes, hyperparameters, serving config). It carries the full binding (method
id, dimensions, row count, ridge, solve residual, hyperparameters, span policy,
model/checkpoint/tokenizer/embedding/capture identity, predictor position
offset, precision, corpus digest, fit manifest digest, pinned validation
evidence digests, serving config) and two content digests. It is validated in
full before any tensor byte is interpreted: schema, exact field set, identifier
and digest syntax, expected method id, expected provenance, tensor shapes
against hidden_dim and rows, payload length and sha256, and the artifact
digest. The payload is a torch.save archive of plain tensors only and is
always read with torch.load(weights_only=True, map_location="cpu"); this
module never calls torch.load with pickle weights_only=False. The payload bytes
are little-endian float32, so a big-endian host is refused instead of
reinterpreting them.
"""
from __future__ import annotations

import array
import hashlib
import io
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .latent_types import LatentError, require

METHOD_ID = "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1"
CALIBRATION_ROWS_LIMIT = 2048
MIN_CALIBRATION_ROWS = 64
DEFAULT_RIDGE_REL = 0.02

HYPERPARAMETER_FIELDS = ("rho",)
DEFAULT_HYPERPARAMETERS = {"rho": DEFAULT_RIDGE_REL}

ARTIFACT_SCHEMA = "optimus.occ3.calibration-artifact.v2"
CORPUS_DIGEST_DOMAIN = "optimus.occ3.calibration-corpus.v1"
ARTIFACT_MAGIC = b"OCC3CAL1"
PRECISION = "float32"
PREDICTOR_POSITION_OFFSET = -1
MAX_HEADER_BYTES = 8192
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_CORPUS_TURNS = 4096
DEFAULT_CAPTURE_SPAN_POLICY = "whole_completed_response_tail_v1"
SERVING_NORM_FLOOR = 1e-6
SOLVE_RESIDUAL_FLOOR = 1e-30
PARTITION_FIELDS = frozenset({"hidden", "embeddings", "row_turn_ids", "turn_ids",
                              "manifest_digest"})
TENSOR_NAMES = ("mean_hidden", "mean_embedding", "hidden_residual",
                "embedding_residual", "gram")
BINDING_FIELDS = frozenset({
    "method_id", "hidden_dim", "rows", "layer_index", "ridge", "ridge_rel",
    "solve_residual", "hyperparameters", "capture_span_policy",
    "model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
    "embedding_id", "capture_method_id", "predictor_position_offset", "precision",
    "corpus_digest", "fit_manifest_digest", "validation_record_digest",
    "validation_artifact_digest", "serving",
})
HEADER_FIELDS = BINDING_FIELDS | frozenset({
    "schema", "byte_order", "tensor_dtype", "tensor_shapes", "tensor_digest",
    "payload_sha256", "payload_bytes", "artifact_digest",
})
DIGEST_EXCLUDED = frozenset({"artifact_digest", "payload_sha256", "payload_bytes"})
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")
_SPAN_POLICY = re.compile(r"[A-Za-z0-9._:/_-]{1,256}\Z")
_PROVENANCE_COMPARED = ("layer_index", "model_fingerprint", "checkpoint_revision",
                        "tokenizer_fingerprint", "embedding_id", "capture_method_id",
                        "capture_span_policy", "predictor_position_offset", "precision")

# Cached dual coefficients, at most _SOLVER_CACHE_LIMIT live entries. The key
# holds the artifact object itself in the value, so an id cannot be reused
# while its entry exists. One entry costs one [rows, D] float32 tensor.
_SOLVER_CACHE: dict = {}
_SOLVER_CACHE_LIMIT = 4


@dataclass(frozen=True)
class CalibrationProvenance:
    """Caller-supplied identity of the model that produced the corpus.

    None of these values is derived from the corpus rows: the caller has to
    supply what the runtime negotiates, and deserialization compares the saved
    copy against it. capture_span_policy is the versioned span policy the
    corpus was captured under. precision, predictor_position_offset and the
    default span policy are fixed because the fit and the capture convention
    are fixed.
    """
    layer_index: int
    model_fingerprint: str
    checkpoint_revision: str
    tokenizer_fingerprint: str
    embedding_id: str
    capture_method_id: str
    capture_span_policy: str = DEFAULT_CAPTURE_SPAN_POLICY
    predictor_position_offset: int = PREDICTOR_POSITION_OFFSET
    precision: str = PRECISION

    def __post_init__(self):
        require(_uint(self.layer_index, 0, 2**32 - 1), "profile")
        for name in ("model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
                     "embedding_id", "capture_method_id"):
            require(_identifier(getattr(self, name)), "profile")
        require(_span_policy(self.capture_span_policy), "profile")
        require(self.predictor_position_offset == PREDICTOR_POSITION_OFFSET, "profile")
        require(self.precision == PRECISION, "profile")


@dataclass(frozen=True)
class CalibrationBinding:
    """Content-independent identity of one fitted transform."""
    method_id: str
    hidden_dim: int
    rows: int
    layer_index: int
    ridge: float
    ridge_rel: float
    solve_residual: float
    hyperparameters: dict
    capture_span_policy: str
    model_fingerprint: str
    checkpoint_revision: str
    tokenizer_fingerprint: str
    embedding_id: str
    capture_method_id: str
    predictor_position_offset: int
    precision: str
    corpus_digest: str
    fit_manifest_digest: str
    validation_record_digest: str
    validation_artifact_digest: str
    serving: object = None

    def __post_init__(self):
        require(self.method_id == METHOD_ID, "identity")
        require(_uint(self.hidden_dim, 3, 2**20) and _uint(self.rows, MIN_CALIBRATION_ROWS,
                                                          CALIBRATION_ROWS_LIMIT), "profile")
        require(_uint(self.layer_index, 0, 2**32 - 1), "profile")
        require(_finite(self.ridge) and self.ridge > 0.0, "numerical")
        require(_finite(self.ridge_rel) and 0.0 < self.ridge_rel < 1.0, "profile")
        require(_finite(self.solve_residual) and self.solve_residual >= 0.0, "numerical")
        require(type(self.hyperparameters) is dict
                and set(self.hyperparameters) == set(HYPERPARAMETER_FIELDS), "profile")
        rho = self.hyperparameters.get("rho")
        require(_number(rho) and 0.0 < float(rho) < 1.0 and float(rho) == self.ridge_rel,
                "profile")
        require(_span_policy(self.capture_span_policy), "profile")
        for name in ("model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
                     "embedding_id", "capture_method_id"):
            require(_identifier(getattr(self, name)), "identity")
        require(self.predictor_position_offset == PREDICTOR_POSITION_OFFSET, "profile")
        require(self.precision == PRECISION, "profile")
        require(_digest(self.corpus_digest), "identity")
        require(_digest_or_empty(self.fit_manifest_digest), "identity")
        require(_digest_or_empty(self.validation_record_digest), "identity")
        require(_digest_or_empty(self.validation_artifact_digest), "identity")
        require(self.serving is None
                or (type(self.serving) is dict and set(self.serving) == {"kind"}
                    and self.serving["kind"] == "n1"), "profile")

    def scalars(self):
        """The flat header of the binding; no tensor content."""
        return {"method_id": self.method_id, "hidden_dim": self.hidden_dim,
                "rows": self.rows, "layer_index": self.layer_index,
                "ridge": self.ridge, "ridge_rel": self.ridge_rel,
                "solve_residual": self.solve_residual,
                "hyperparameters": dict(self.hyperparameters),
                "capture_span_policy": self.capture_span_policy,
                "model_fingerprint": self.model_fingerprint,
                "checkpoint_revision": self.checkpoint_revision,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "embedding_id": self.embedding_id,
                "capture_method_id": self.capture_method_id,
                "predictor_position_offset": self.predictor_position_offset,
                "precision": self.precision, "corpus_digest": self.corpus_digest,
                "fit_manifest_digest": self.fit_manifest_digest,
                "validation_record_digest": self.validation_record_digest,
                "validation_artifact_digest": self.validation_artifact_digest,
                "serving": None if self.serving is None else dict(self.serving)}


@dataclass(frozen=True)
class CalibrationArtifact:
    method_id: str
    hidden_dim: int
    rows: int
    ridge: float
    ridge_rel: float
    solve_residual: float
    mean_hidden: object
    mean_embedding: object
    hidden_residual: object
    embedding_residual: object
    gram: object
    capture_span_policy: str = ""
    fit_manifest_digest: str = ""
    binding: object = None
    # Content digest of the bound artifact, computed once at fit or load and
    # never serialized: it is the value a validation record is bound to.
    content_digest: object = None


def _uint(value, minimum=0, maximum=(1 << 64) - 1):
    return type(value) is int and minimum <= value <= maximum


def _finite(value):
    return type(value) is float and math.isfinite(value)


def _number(value):
    return type(value) in (int, float) and not isinstance(value, bool) \
        and math.isfinite(value)


def _digest(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def _digest_or_empty(value):
    return value == "" or _digest(value)


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _span_policy(value):
    return type(value) is str and _SPAN_POLICY.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _byte_order():
    require(sys.byteorder == "little", "profile")
    return sys.byteorder


def _tensor_bytes(matrix, torch):
    """Little-endian float32 image of one matrix, as bytes.

    torch.storage.UntypedStorage.__bytes__ is a per-byte loop (measured 4.7 s
    per MiB on torch 2.11), so the storage never reaches the digest. numpy gives
    a memcpy and is already part of the serving image; the array fallback keeps
    the digest defined when it is absent. Both paths produce the same bytes.
    """
    require(getattr(matrix, "ndim", 0) == 2, "profile")
    flat = matrix.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
    require(int(flat.numel()) == int(matrix.numel()), "profile")
    try:
        import numpy
    except ImportError:
        raw = array.array("f", flat.view(-1).tolist()).tobytes()
    else:
        raw = bytes(memoryview(numpy.ascontiguousarray(flat.numpy())))
    require(len(raw) == int(flat.numel()) * 4, "numerical")
    return raw


def _matrix_digest(matrix, torch, tag):
    value = hashlib.sha256()
    value.update(f"{tag}:{int(matrix.shape[0])}x{int(matrix.shape[1])}:{PRECISION}:".encode("ascii"))
    value.update(_tensor_bytes(matrix, torch))
    return value.hexdigest()


def _tensor_set_digest(artifact, torch):
    value = hashlib.sha256()
    for name in TENSOR_NAMES:
        value.update(_matrix_digest(getattr(artifact, name), torch, name).encode("ascii"))
    return value.hexdigest()


def corpus_digest(hidden, embeddings, *, turn_ids=None, torch_module=None):
    """Digest of the calibration rows and the ordered turn ids they came from.

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
             "hidden": _matrix_digest(hidden, torch, "hidden"),
             "embeddings": _matrix_digest(embeddings, torch, "embeddings")}
    return hashlib.sha256(_canonical(value)).hexdigest()


def _expected_shapes(rows, hidden_dim):
    return {"mean_hidden": [1, hidden_dim], "mean_embedding": [1, hidden_dim],
            "hidden_residual": [rows, hidden_dim],
            "embedding_residual": [rows, hidden_dim], "gram": [rows, rows]}


def _require_consistent(artifact, torch, *, check_values=True):
    """Every scalar, the tensor shapes and (optionally) the tensor values agree."""
    require(isinstance(artifact, CalibrationArtifact) and artifact.method_id == METHOD_ID, "identity")
    require(_uint(artifact.hidden_dim, 3, 2**20), "profile")
    require(_uint(artifact.rows, MIN_CALIBRATION_ROWS, CALIBRATION_ROWS_LIMIT), "profile")
    require(_finite(artifact.ridge) and artifact.ridge > 0.0, "numerical")
    require(_finite(artifact.ridge_rel) and 0.0 < artifact.ridge_rel < 1.0, "profile")
    require(_finite(artifact.solve_residual) and artifact.solve_residual >= 0.0, "numerical")
    require(artifact.capture_span_policy == "" or _span_policy(artifact.capture_span_policy),
            "profile")
    require(_digest_or_empty(artifact.fit_manifest_digest), "identity")
    require(artifact.content_digest is None or _digest(artifact.content_digest), "identity")
    shapes = _expected_shapes(artifact.rows, artifact.hidden_dim)
    devices = set()
    for name, shape in shapes.items():
        value = getattr(artifact, name)
        require(tuple(int(x) for x in value.shape) == tuple(shape), "profile")
        require(value.dtype == torch.float32, "profile")
        devices.add(str(value.device))
        if check_values:
            require(bool(torch.isfinite(value).all()), "numerical")
    require(len(devices) == 1, "profile")
    binding = artifact.binding
    if binding is not None:
        require(isinstance(binding, CalibrationBinding), "identity")
        require(binding.method_id == artifact.method_id, "identity")
        require(binding.hidden_dim == artifact.hidden_dim and binding.rows == artifact.rows, "profile")
        require((binding.ridge, binding.ridge_rel, binding.solve_residual)
                == (artifact.ridge, artifact.ridge_rel, artifact.solve_residual), "identity")
        require(binding.capture_span_policy == artifact.capture_span_policy, "identity")
        require(binding.fit_manifest_digest == artifact.fit_manifest_digest, "identity")
        require(binding.hyperparameters == {"rho": artifact.ridge_rel}, "identity")
    return artifact


def _binding_header(binding):
    header = dict(binding.scalars())
    require(set(header) == set(BINDING_FIELDS), "profile")
    header["schema"] = ARTIFACT_SCHEMA
    header["byte_order"] = _byte_order()
    header["tensor_dtype"] = PRECISION
    header["tensor_shapes"] = _expected_shapes(binding.rows, binding.hidden_dim)
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


def _torch(reference):
    import torch
    return torch


def _require_partition(partition, torch):
    """The contract partition: exactly five keys, whole-turn row identity."""
    require(isinstance(partition, Mapping), "profile")
    require(set(partition) == set(PARTITION_FIELDS), "profile")
    hidden, embeddings = partition["hidden"], partition["embeddings"]
    for matrix in (hidden, embeddings):
        require(hasattr(matrix, "dtype") and matrix.dtype.is_floating_point, "profile")
        require(getattr(matrix, "ndim", None) == 2, "profile")
    require(hidden.shape == embeddings.shape, "profile")
    require(hidden.device == embeddings.device, "profile")
    rows, dim = (int(value) for value in hidden.shape)
    row_turn_ids = partition["row_turn_ids"]
    turn_ids = partition["turn_ids"]
    require(isinstance(row_turn_ids, (tuple, list)), "profile")
    require(isinstance(turn_ids, (tuple, list)), "profile")
    row_turn_ids, turn_ids = tuple(row_turn_ids), tuple(turn_ids)
    require(len(row_turn_ids) == rows, "profile")
    require(all(_identifier(value) for value in row_turn_ids), "profile")
    require(all(_identifier(value) for value in turn_ids), "profile")
    require(len(set(turn_ids)) == len(turn_ids), "profile")
    require(set(turn_ids) == set(row_turn_ids), "profile")
    digest = partition["manifest_digest"]
    require(type(digest) is str and _digest_or_empty(digest), "profile")
    return hidden, embeddings, row_turn_ids, turn_ids, digest


def _hyperparams(hyperparams):
    """Explicit hyperparameters, filled with the method defaults."""
    require(isinstance(hyperparams, Mapping), "profile")
    require(set(hyperparams) <= set(HYPERPARAMETER_FIELDS), "profile")
    values = dict(DEFAULT_HYPERPARAMETERS)
    for name, value in hyperparams.items():
        require(_number(value) and 0.0 < float(value) < 1.0, "profile")
        values[name] = float(value)
    return values


def _cholesky(system, torch):
    """Named numerical failure; no jitter is added by default."""
    try:
        return torch.linalg.cholesky(system)
    except LatentError:
        raise
    except Exception as failure:
        raise LatentError("numerical") from failure


def _seed_solver_cache(artifact, coef):
    key = (id(artifact), str(coef.device))
    _SOLVER_CACHE[key] = (artifact, coef)
    while len(_SOLVER_CACHE) > _SOLVER_CACHE_LIMIT:
        _SOLVER_CACHE.pop(next(iter(_SOLVER_CACHE)))


def _cached_coef(artifact, torch):
    """C = (G + ridge I)^-1 B, derived once per immutable artifact and device."""
    key = (id(artifact), str(artifact.gram.device))
    entry = _SOLVER_CACHE.get(key)
    if entry is not None and entry[0] is artifact:
        return entry[1]
    eye = torch.eye(artifact.rows, dtype=artifact.gram.dtype, device=artifact.gram.device)
    factor = _cholesky(artifact.gram + artifact.ridge * eye, torch)
    coef = torch.cholesky_solve(artifact.embedding_residual, factor)
    _seed_solver_cache(artifact, coef)
    return coef


def fit_map(partition, hyperparams, *, provenance=None, torch_module=None):
    """Fit the frozen transform on exactly one partition of calibration rows.

    No internal split, no quality gate and no refusal on poor quality: the
    partition is the whole fit, and measurement belongs to evaluate_map. The
    numerical guards stay (finite rows, dimensions, row bounds, PSD sanity,
    named solver failures) and the relative solve residual is recorded in the
    binding. provenance is optional so an offline study can fit without model
    identity, but such an artifact cannot be serialized.
    """
    torch = torch_module or _torch(None)
    hidden, embeddings, row_turn_ids, turn_ids, manifest_digest = _require_partition(partition, torch)
    values = _hyperparams(hyperparams)
    rho = values["rho"]
    require(provenance is None or isinstance(provenance, CalibrationProvenance), "profile")
    rows, dim = (int(value) for value in hidden.shape)
    require(MIN_CALIBRATION_ROWS <= rows <= CALIBRATION_ROWS_LIMIT, "profile")
    require(dim > 2, "profile")
    with torch.no_grad():
        h, e = hidden.float(), embeddings.float()
        require(bool(torch.isfinite(h).all()) and bool(torch.isfinite(e).all()), "numerical")
        mu_h, mu_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
        a, b = h - mu_h, e - mu_e
        product = a @ a.mT
        gram = 0.5 * (product + product.mT)
        require(bool(torch.isfinite(gram).all()), "numerical")
        # A rank-deficient Gram is expected when rows exceed the hidden width: its
        # zero eigenvalues carry float32 noise of order eps * lambda_max, which can
        # be far above eps * max|G|. The ridge makes the solve positive definite
        # regardless, so the guard only has to reject a Gram that is PSD-broken.
        eigenvalues = torch.linalg.eigvalsh(gram)
        largest = float(eigenvalues.max())
        require(largest > 0.0 and float(eigenvalues.min()) >= -1e-4 * largest, "numerical")
        ridge = rho * float(gram.diag().mean())
        require(ridge > 0.0, "numerical")
        eye = torch.eye(rows, dtype=gram.dtype, device=gram.device)
        system = gram + ridge * eye
        factor = _cholesky(system, torch)
        coef = torch.cholesky_solve(b, factor)
        numerator = float((system @ coef - b).norm())
        solve_residual = numerator / max(float(b.norm()), SOLVE_RESIDUAL_FLOOR)
        require(math.isfinite(solve_residual), "numerical")
    binding = None
    if provenance is not None:
        binding = CalibrationBinding(
            method_id=METHOD_ID, hidden_dim=dim, rows=rows, layer_index=provenance.layer_index,
            ridge=float(ridge), ridge_rel=float(rho), solve_residual=float(solve_residual),
            hyperparameters={"rho": float(rho)},
            capture_span_policy=provenance.capture_span_policy,
            model_fingerprint=provenance.model_fingerprint,
            checkpoint_revision=provenance.checkpoint_revision,
            tokenizer_fingerprint=provenance.tokenizer_fingerprint,
            embedding_id=provenance.embedding_id,
            capture_method_id=provenance.capture_method_id,
            predictor_position_offset=provenance.predictor_position_offset,
            precision=provenance.precision,
            corpus_digest=corpus_digest(h, e, turn_ids=turn_ids, torch_module=torch),
            fit_manifest_digest=manifest_digest,
            validation_record_digest="", validation_artifact_digest="")
    artifact = CalibrationArtifact(method_id=METHOD_ID, hidden_dim=dim, rows=rows,
                                   ridge=float(ridge), ridge_rel=float(rho),
                                   solve_residual=float(solve_residual),
                                   mean_hidden=mu_h, mean_embedding=mu_e,
                                   hidden_residual=a, embedding_residual=b, gram=gram,
                                   capture_span_policy=(provenance.capture_span_policy
                                                        if provenance is not None else ""),
                                   fit_manifest_digest=manifest_digest, binding=binding)
    if binding is not None:
        artifact = replace(artifact, content_digest=artifact_digest(artifact, torch_module=torch))
    _seed_solver_cache(artifact, coef)
    return artifact


def apply_map(artifact, hidden, *, torch_module=None):
    """Map new hidden rows into the embedding space with the frozen transform.

    Only the frozen parameters are used. The applied rows never enter the fit,
    so replacing them changes the output and never changes the artifact. The
    returned rows are raw: never renormalised. serving_transform is the only
    place a serving scale is applied.
    """
    torch = torch_module or _torch(hidden)
    _require_consistent(artifact, torch, check_values=False)
    require(hidden.ndim == 2 and hidden.shape[1] == artifact.hidden_dim, "profile")
    require(hidden.shape[0] >= 2, "profile")
    require(hidden.device == artifact.mean_hidden.device, "profile")
    with torch.no_grad():
        h = hidden.float()
        require(bool(torch.isfinite(h).all()), "numerical")
        coef = _cached_coef(artifact, torch)
        k = (h - artifact.mean_hidden) @ artifact.hidden_residual.mT
        z = artifact.mean_embedding + k @ coef
        require(bool(torch.isfinite(z).all()), "numerical")
    return z.to(dtype=hidden.dtype).contiguous()


def serving_transform(config, z_raw, *, nu=None):
    """N0 (identity) or N1 (row renorm to the vocabulary norm statistic).

    config is None for N0 or {"kind": "n1"} for N1. nu is the fixed norm
    statistic measured from the frozen embedding matrix; it is never derived
    from evaluation answers. Zero or nonfinite row norms are refused rather
    than fabricating a carrier, for either variant.
    """
    require(getattr(z_raw, "ndim", None) == 2, "profile")
    require(bool(z_raw.isfinite().all()), "numerical")
    norms = z_raw.norm(dim=-1, keepdim=True)
    require(bool(norms.isfinite().all()) and bool((norms > SERVING_NORM_FLOOR).all()),
            "numerical")
    if config is None:
        return z_raw.contiguous()
    require(type(config) is dict and set(config) == {"kind"} and config["kind"] == "n1",
            "profile")
    if getattr(nu, "ndim", None) == 0 and getattr(nu, "numel", lambda: None)() == 1:
        nu = float(nu)
    require(_number(nu) and float(nu) > 0.0, "profile")
    served = z_raw * (float(nu) / norms)
    require(bool(served.isfinite().all()), "numerical")
    return served.contiguous()


def required_provenance_fields(artifact_class=None):
    """Provenance fields compared at load; the span policy is one of them."""
    require(artifact_class is None or isinstance(artifact_class, type), "profile")
    return _PROVENANCE_COMPARED


def artifact_bytes(artifact):
    """Resident size of the frozen transform, for the live-bytes budget."""
    total = 0
    for value in (artifact.mean_hidden, artifact.mean_embedding,
                  artifact.hidden_residual, artifact.embedding_residual, artifact.gram):
        total += value.numel() * value.element_size()
    return total


def describe(artifact):
    """Scalar-only summary; safe for a receipt. Never reads tensor content."""
    summary = {"method_id": artifact.method_id, "rows": artifact.rows,
               "hidden_dim": artifact.hidden_dim, "ridge": artifact.ridge,
               "ridge_rel": artifact.ridge_rel, "solve_residual": artifact.solve_residual,
               "rho": artifact.ridge_rel, "capture_span_policy": artifact.capture_span_policy,
               "bytes": artifact_bytes(artifact),
               "bound": 1 if artifact.binding is not None else 0}
    if artifact.binding is not None:
        scalars = {name: value for name, value in artifact.binding.scalars().items()
                   if name not in ("serving", "hyperparameters")}
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
    require(_uint(header["rows"], MIN_CALIBRATION_ROWS, CALIBRATION_ROWS_LIMIT), "profile")
    require(header["tensor_shapes"] == _expected_shapes(header["rows"], header["hidden_dim"]), "profile")
    for name in ("tensor_digest", "payload_sha256", "artifact_digest", "corpus_digest"):
        require(_digest(header[name]), "identity")
    for name in ("fit_manifest_digest", "validation_record_digest",
                 "validation_artifact_digest"):
        require(_digest_or_empty(header[name]), "identity")
    require(_finite(header["ridge"]) and header["ridge"] > 0.0, "numerical")
    require(_finite(header["ridge_rel"]) and 0.0 < header["ridge_rel"] < 1.0, "profile")
    require(_finite(header["solve_residual"]) and header["solve_residual"] >= 0.0, "numerical")
    require(type(header["hyperparameters"]) is dict
            and set(header["hyperparameters"]) == set(HYPERPARAMETER_FIELDS)
            and header["hyperparameters"]["rho"] == header["ridge_rel"], "profile")
    require(_span_policy(header["capture_span_policy"]), "profile")
    require(header["serving"] is None
            or (type(header["serving"]) is dict and set(header["serving"]) == {"kind"}
                and header["serving"]["kind"] == "n1"), "profile")
    require(_uint(header["payload_bytes"], 1, MAX_ARTIFACT_BYTES), "profile")
    require(_uint(header["layer_index"], 0, 2**32 - 1), "profile")
    return CalibrationBinding(**{name: header[name] for name in BINDING_FIELDS})


def deserialize_artifact(raw, *, expected_method_id=None, expected_hidden_dim=None,
                         expected_provenance=None, expected_artifact_digest=None,
                         torch_module=None):
    """Validate the header in full, then and only then read the tensor payload.

    Every refusal is LatentError: "identity" for a method, provenance or digest
    mismatch, "profile" for a malformed header or shape, "numerical" for a
    non-finite tensor. The method id defaults to this module's METHOD_ID, so a
    per-message artifact is refused rather than silently reinterpreted. The
    loaded artifact carries its verified content digest, which is what a
    validation record binds to.
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
    negotiated = METHOD_ID if expected_method_id is None else expected_method_id
    require(_identifier(negotiated), "profile")
    require(header["method_id"] == negotiated, "identity")
    if expected_hidden_dim is not None:
        require(header["hidden_dim"] == expected_hidden_dim, "profile")
    if expected_provenance is not None:
        require(isinstance(expected_provenance, CalibrationProvenance), "profile")
        for name in _PROVENANCE_COMPARED:
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
        require(list(int(x) for x in value.shape) == header["tensor_shapes"][name], "profile")
        require(bool(torch.isfinite(value).all()), "numerical")
    artifact = CalibrationArtifact(method_id=header["method_id"], hidden_dim=header["hidden_dim"],
                                   rows=header["rows"], ridge=header["ridge"],
                                   ridge_rel=header["ridge_rel"],
                                   solve_residual=header["solve_residual"],
                                   mean_hidden=loaded["mean_hidden"],
                                   mean_embedding=loaded["mean_embedding"],
                                   hidden_residual=loaded["hidden_residual"],
                                   embedding_residual=loaded["embedding_residual"],
                                   gram=loaded["gram"],
                                   capture_span_policy=header["capture_span_policy"],
                                   fit_manifest_digest=header["fit_manifest_digest"],
                                   binding=binding, content_digest=header["artifact_digest"])
    _require_consistent(artifact, torch)
    require(_tensor_set_digest(artifact, torch) == header["tensor_digest"], "identity")
    return artifact
