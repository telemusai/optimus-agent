# OCC-3 calibration equations and metric definitions

Protocol version 2 - pre-registered before final-test data collection.

This document fixes the exact mathematics of both campaign methods, the serving
normalisation, the anchoring ablation, and every evaluation metric. The implementing modules
are `occ3_sglang/statebridge_calibrated.py` (Method A) and
`occ3_sglang/statebridge_calibrated_procrustes.py` (Method B), with the shared evaluation in
`occ3_sglang/calibration_corpus.py`. Line citations are as of 2026-10-28; wave-1 repairs were
in flight and lines may drift, the equations do not.

## 1. Conventions

- Row vectors everywhere. Hidden states are H [M, D]; reference embeddings are E [M, D],
  row-aligned with H (row i of H corresponds to row i of E).
- Subscript f marks FIT quantities (H_f, E_f, M rows); q marks an unseen message (H_q, N rows).
- All fitted statistics come from FIT only. mu_H and mu_E are frozen in the artifact and are
  never recomputed from a message.
- Numerical standards: FP64 CPU fixtures for mathematical reference tests; FP32 for the real
  calibration and serving paths; no FP8 solves. Well-conditioned toy tolerances: FP64
  relative 1e-8, FP32 relative 1e-4. Deterministic tests use their own torch.Generator and
  never the serving model RNG.
- A metric whose denominator is zero is undefined: it is recorded as `None` with an entry in
  `undefined_denominators`, never fabricated as zero and never a run refusal.

## 2. Method A - frozen dual ridge

Identity `statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1`
(`statebridge_calibrated.py:67`); hyperparameter `rho`; grid {0.002, 0.02, 0.2}
(`calibration_corpus.py:56`), selected on VALIDATION only.

Fit (on FIT rows; `statebridge_calibrated.py:479`, `fit_map`):

```
mu_H = mean_rows(H_f)                       [1, D]
mu_E = mean_rows(E_f)                       [1, D]
A = H_f - mu_H                              [M, D]
B = E_f - mu_E                              [M, D]
G = symmetrise(A A^T)                       [M, M]      (0.5 * (X + X^T))

W* = argmin_W  || A W - B ||_F^2  +  lambda || W ||_F^2      (primal objective)

lambda = rho * trace(G) / M
```

The objective is a sum of squares. If an average-loss objective is ever used, the factor of
M must be accounted for in lambda explicitly. In the implementation,
`lambda = rho * gram.diagonal().mean()` (`statebridge_calibrated.py`, ridge line inside
`fit_map`), which is exactly `rho * trace(G) / M`.

Dual solve:

```
C = solve(G + lambda I_M, B)                [M, D]
```

solved by Cholesky (`torch.linalg.cholesky` + `torch.cholesky_solve`); no explicit inverse. A
Cholesky failure is a named numerical failure (`LatentError("numerical")` with the cause
recorded), not a bare exception; no unbounded jitter is added to force it, and any permitted
jitter must be bounded, recorded and included in the effective regularisation provenance.

Solve-residual check at fit (recorded in the binding as `solve_residual`):

```
|| (G + lambda I_M) C - B ||_F  /  max( || B ||_F, 1e-30 )
```

Apply to an unseen message (`statebridge_calibrated.py:552`, `apply_map`):

```
Z_raw = mu_E + ((H_q - mu_H) A^T) C          [N, D]
```

- The full map would be W = A^T C. W is never materialised in the serving path.
- No D x D matrix is ever formed.
- `apply_map` never reads the message's reference embeddings and never renormalises; it
  returns Z_raw only. The Cholesky factor (or the dual coefficient C) is cached once per
  (artifact, device) after validated load (`statebridge_calibrated.py:459`, `:466`); a
  message never repeats the M x M factorisation.

## 3. Method B - frozen low-rank whitened Procrustes

Identities (`statebridge_calibrated_procrustes.py:89-91`):

- variant `o`: `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes.v1`
- variant `so`: `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-so.v1`
- variant `plain`: `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-plain.v1`

Hyperparameters (r, eta, variant); r grid {64, 128, 256} when supported by the fit data with
{16, 32} predeclared for low-rank data, eta grid {0.001, 0.01, 0.1}
(`statebridge_calibrated_procrustes.py:96-97`), selected on VALIDATION only. Artifact schema
`optimus.occ3.calibration-artifact.lowrank-procrustes.v1` (:99). Fitted on FIT only
(`statebridge_calibrated_procrustes.py:497`, `fit_map`).

### 3.1 Separate calibration bases

Same A, B as Method A. Reduced SVD (`torch.linalg.svd(..., full_matrices=False)` returns Vh
= V^T):

```
A = U_H diag(s_H) V_H^T          B = U_E diag(s_E) V_E^T
P_H = V_H[:, :r]  (= vh_h[:r].T)     [D, r]
P_E = V_E[:, :r]  (= vh_e[:r].T)     [D, r]
P_H^T P_H = I_r                    P_E^T P_E = I_r
```

Rank bounds: `r <= min(numerical_rank(A), numerical_rank(B), M - 1, D)`. A requested r above
that bound is refused; null singular directions are never retained merely to reach r. The
numerical rank and the participation ratio / effective rank are recorded separately and are
never treated as the same quantity. P_H and P_E are never derived from a message being
evaluated.

### 3.2 Reduced covariance and (regularised) whitening

```
X0 = A P_H                    [M, r]
Y0 = B P_E                    [M, r]
g_H = s_H[:r]^2 / M           [r]
g_E = s_E[:r]^2 / M           [r]
tau_H = eta * mean(g_H)       tau_E = eta * mean(g_E)
d_H = sqrt(g_H + tau_H)       [r]
d_E = sqrt(g_E + tau_E)       [r]
X = X0 / d_H                  [M, r]      (column-wise division)
Y = Y0 / d_E                  [M, r]
```

These are regularised whitening factors: with positive tau the whitened covariance is not
exactly identity, and no claim of exact whitening is made. Nonfinite factors and unsupported
near-zero retained directions are refused; the maximum whitening gain is reported.

### 3.3 Orthogonal Procrustes over O(r)

```
R* = argmin_R  || X R - Y ||_F^2    subject to  R^T R = I_r

J = X^T Y                    [r, r]
J = U diag(s) V^T
R = U V^T                    [r, r]
```

- The candidate uses O(r): reflections are allowed and `det(R)` is never forced to +1. A
  reflection fixture is required to pass (det < 0 optimum must survive; no hidden SO(r)
  correction).
- The rotation defect `|| R^T R - I_r ||_F` is verified at fit and recorded in the binding
  (limit 1e-3, `statebridge_calibrated_procrustes.py:110`).
- Cross-check on small CPU fixtures against `scipy.linalg.orthogonal_procrustes` is required;
  SciPy's returned "scale" is the sum of the singular values of J and is NOT a fitted
  multiplicative gain.

### 3.4 SO(r) variant (separate identity)

If a proper-rotation variant is explicitly tested:

```
R_SO = U diag(1, ..., 1, det(U V^T)) V^T
```

It gets its own configuration, method identity and artifact identity
(`...calibrated-lowrank-procrustes-so.v1`). Documented caveat, as recorded by the
implementation: the construction flips the LAST singular direction of J when
`det(U V^T) < 0`, not the lowest-energy one. The SO(r) optimum is defined on the fitted PCA
coordinates, whose column signs are a property of the fit SVD output; the O(r) map is
invariant under consistent sign flips of those coordinates while the SO(r) map is not
(`statebridge_calibrated_procrustes.py:46-48`, module docstring). On data where the O(r)
optimum has det < 0, the SO variant is strictly worse by construction; it is an ablation,
not a competing candidate.

### 3.5 Apply to a new message

`statebridge_calibrated_procrustes.py:633`, `apply_map`:

```
Q = ((H_q - mu_H) P_H) / d_H          [N, r]      (column-wise division by d_H)
Z_raw = mu_E + ((Q R) * d_E) P_E^T    [N, D]      (column-wise multiplication by d_E)
```

Equivalent factorised form, which may be stored instead with its exact provenance:

```
F = (P_H / d_H) R diag(d_E)           [D, r]
Z_raw = mu_E + ((H_q - mu_H) F) P_E^T
```

No D x D matrix is stored or formed in either representation. Only R is orthogonal: the
complete map is rank-r and includes projection, whitening, rotation, dewhitening and
translation; it is not a D-dimensional isometry. Components of H_q outside span(P_H) are
discarded, and that loss is measured (`statebridge_calibrated_procrustes.py:662`,
`residual_discard_ratio`):

```
residual_discard = || (H_q - mu_H) - P_H P_H^T (H_q - mu_H) ||_F  /  || H_q - mu_H ||_F
```

No arbitrary "identity completion" of the operator is added, and no completed operator is
claimed to remain orthogonal or semantically justified.

### 3.6 Unwhitened comparator, variant `plain`

Use only to isolate whether whitening causes instability. Kept a separately labelled variant
with its own identity, never a silent change to the whitened candidate:

```
J0 = X0^T Y0                    [r, r]      (X0 = A P_H, Y0 = B P_E)
R0 = U0 V0^T  from SVD(J0)
a0 = trace(R0^T J0) / || X0 ||_F^2
Z_plain = mu_E + a0 * ((H_q - mu_H) P_H R0) P_E^T
```

a0 is fitted on FIT only; a zero or undefined denominator is refused. For the O(r) optimum
the numerator equals the sum of the singular values of J0; SciPy's "scale" is never reused
directly as a0.

## 4. Serving normalisation

Retained Z_raw is the basis of all numerical evaluation; serving transforms are applied and
scored separately.

```
N0:  Z_serv = Z_raw                                  (identity; no row renormalisation)
N1:  Z_serv[i] = nu * Z_raw[i] / || Z_raw[i] ||_2    (vocabulary-norm calibration)
```

- nu is a fixed norm statistic measured from the actual frozen input-embedding matrix (the
  mean row L2 norm, computed with TP all-reduce; `occ3_sglang/statebridge.py:87`,
  `mean_embedding_norm`). It is never derived from evaluation answers.
- Zero or nonfinite row norms are refused rather than fabricating a carrier (norm floor 1e-6).
- N0 or N1 is selected on VALIDATION only. Raw and served metrics are recorded separately in
  every evaluation record.
- Implementation surface: `serving_transform(config, z_raw, *, nu=None)`
  (`statebridge_calibrated.py:575`, `statebridge_calibrated_procrustes.py:682`).

## 5. Anchoring (ablation only)

Primary candidates run with alpha = 0 (no anchoring). Only after readable semantic behaviour
is established, and as a separately identified ablation:

```
v_i = argmax_v  cosine(Z_serv[i], embedding[v])     (global argmax over the vocabulary;
                                                    smallest token id wins ties, as implemented)
P_i = (1 - alpha) Z_serv[i] + alpha * embedding[v_i]      (alpha = 0.3 ablation)
```

- No second unrecorded normalisation is applied after anchoring.
- Norm scaling and anchoring change geometry; they are not information-preserving operations
  and are never described as such.

## 6. Sensitivity

The frozen-map state sensitivity metric is defined in `EVALUATION-PROTOCOL.md` R1:

```
S_wrong = || f(H_correct) - f(H_wrong) ||_F / max( || f(H_correct) ||_F, numerical_floor )
```

with the SAME frozen artifact, normalisation and anchoring in both terms. High sensitivity can
be noise amplification and is always paired with receiver-task results.

## 7. Metrics

All are computed on independent held-out turns, before serving normalisation (raw) and again
after it (served), for the pooled row set and per turn (`calibration_corpus.py:444`
`_row_metrics`, `:459` `_view`, `:477` `_output_stats`, `:492` `evaluate_map`). Z is Z_raw (or
Z_serv for the served view); E_test is the held-out reference embedding; mu_E_fit is the
frozen fit mean from the artifact, never a test-set mean.

```
SSE = sum_i || Z[i] - E_test[i] ||_2^2
SST_test = sum_i || E_test[i] - mean_rows(E_test) ||_2^2
R2_standard = 1 - SSE / SST_test

SSE_mean = sum_i || E_test[i] - mu_E_fit ||_2^2
R2_vs_fit_mean = 1 - SSE / SSE_mean

relative_frobenius = || Z - E_test ||_F / max( || E_test ||_F, 1e-6 )
mean_row_cosine = mean_i  ( Z[i] . E_test[i] ) / ( || Z[i] ||_2 * || E_test[i] ||_2 )
                    (row norms floor-clamped at 1e-6 before the division)
```

- `R2_standard` and `R2_vs_fit_mean` are different metrics and both are named and reported;
  neither is reported as an unqualified "held-out R2". The test-set mean used to define
  SST_test is a metric definition, never a fitted model parameter, and is never used to
  adjust predictions.
- Row norms: raw and served row-norm means are both reported.
- Output variance and effective rank: variance is the mean squared centred row norm of Z;
  effective rank is `exp(-H)` where H is the Shannon entropy of the squared-singular-value
  shares of the centred Z (computed in float64).
- Per-turn metrics and pooled row-weighted metrics are both reported, plus a turn-macro
  average, because a different number of rows per turn can distort pooled estimates. The
  fit/validation/test error gap is reported alongside.
- Negative values are reported as they are; they are never truncated or suppressed, and poor
  quality never prevents the record from existing (`evaluate_map` records, it does not gate).
- Undefined denominators (SST_test = 0, SSE_mean = 0) are recorded as `None` plus an
  `undefined_denominators` entry listing the metric and turn.

## 8. Record binding and promotion

- Every evaluation record names the method id, the artifact content digest, the fit corpus
  digest, the fit and partition manifest digests, the serving config and nu, and the
  undefined-denominator list (`calibration_corpus.py:492`).
- Promotion is decided only by `validate_for_activation(artifact, validation_record, policy)`
  (`calibration_corpus.py:634`) from a policy of fixed `min_<metric>` / `max_<metric>`
  thresholds over a validation record bound to the artifact digest; an undefined metric is a
  refusal reason, not a zero. Research measurement and deployment promotion stay separated.

## 9. Numerical reference tests

The equation-level contracts above are pinned by the campaign's numerical test suite (spec
section 13): dual and primal ridge agreement on FP64 fixtures, O(r) reflection fixtures,
SciPy objective agreement, rank-deficiency refusal, PCA-sign-flip invariance, residual
discard, artifact immutability under evaluation, serialization round trips, wrong-model
refusals, and CPU/GPU parity with conditioning-aware tolerances. CPU-only tests do not
establish CUDA correctness; parity is checked when GPU access exists.
