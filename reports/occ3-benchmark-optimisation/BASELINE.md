# OCC-3 Benchmark & Optimisation Campaign — Baseline Manifest

Campaign: `occ3-bench-opt-20261109` — recorded 2026-11-09T13:40Z

| Field | Value |
|---|---|
| repository_commit | `445facb53d29796724944551a5455b8a36b7fcc8` |
| client_binary_digest | `ee2b9e567e5110086cdd976be3cac08c469c3b973995acb3c961f23cab1f8788` |
| client_build | `optimus-0.1.44-main-20261109-122335 (built from 1a61096d2 = 445facb53 + test-only fixture)` |
| server_image_digest | `sha256:9728d909511b222f688c75f6da800ac4d5c9a86c59df00650835b3bc5d183c80` |
| server_extension_revision | `package fingerprint sha256:cfa2f1bef532c42d53d0b682acd9acfdaaa8381b68cdd194b52263ce07d1bc3a (image-b…` |
| server_extension_version | `0.1.0` |
| runtime_identity | `79019420a14443d68c678528c82891b9 gen 1 TP 8/8` |
| model_fingerprint | `sha256:efef608a8f55388d16e16af8ced7b6936f95c018ee098ae6665c69fcb8f90da8` |
| alignment_method | `statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1` |
| calibration_artifact_digest | `9a7394dea73cf8d33fd651bf8b43d62d739c94d0940bd6f4ae494fd83ba0be90` |
| anchor_alpha (artifact) | `0.0` |
| label_anchor_alpha | `allowed {0, 0.3, 0.5, 0.7, 1.0}; server default 0.0; client default 0.3; request-resolved per handoff` |
| effective max_active | `RuntimeProfile 1 (env unset); frontend armed-turn cap 2; max turn 900s` |
| live memory budget | `256 MiB live / 128 MiB workspace / 128 MiB capture / 256 records / 3600s TTL` |
| SGLang / PyTorch / CUDA | `0.5.21:e00930c / 2.13.0+cu130 / 13.0` |
| hardware | `8x B300 SXM6 AC, TP8 fp8_e4m3` |
| registry_state_at_baseline | `NATIVE_READY; legacy channels disabled (410); 0 channels; 146 lifetime admissions` |
| ordinary_completion_baseline | `133.3ms HTTP; 17+16 tokens` |
| background_load | `GPU util varied 100%→0% across snapshots; 3 sibling containers` |

## Notes
- The deployed runtime was verified on the wire (attestation POST, identity GET, registry status), not inferred from merged source.
- Client 0.1.44 is the running build; campaign branch carries the twin-guard fix (bc4a6375b) pending its own build/cutover.
- Effective per-handoff alpha is receipt-bound at handoff time; Suite 0 canaries pin the per-arm values. alpha=0.0 is the artifact anchor; the client default is 0.3.
- Two distinct capacity caps: RuntimeProfile max_active (native latent capture slots) = 1; frontend BufferProfile armed-turn cap = 2. Do not conflate.
- Historical scores from occ23/occ2opt campaigns are reference-only (different gateway/config); contemporary four-way rerun happens on this stack.

Full machine-readable manifest: campaign workspace `occ3-bench-opt-20261109/BASELINE-MANIFEST.json`; raw probe receipts under `probes/baseline-live-1/`.

---

## Extended manifest detail (appended 2026-11-10 by reports-init worker; source: BASELINE-MANIFEST.json, all fields verbatim)

### Client build provenance (optimus-agent 0.1.44)

| Field | Value |
|---|---|
| buildFingerprint | `5ba4b78fc55db0da4a8aa9547d145e83d31c6ac6539779f857d21d514c560c89` |
| sourceTreeSha256 | `12d66b2bf20b7d06115ba8904234056c19379b98b4832d8b374d712e5ef0f7e6` |
| rustc | `rustc 1.98.1 (48a229cea 2026-09-01)` |
| live client source commit | `1a61096d28c93f82e75651fe2768dc4d54cc9332` |

### Client configuration (build defaults; no campaign overrides)

| Field | Value |
|---|---|
| busy_retry_budget_default | 3 |
| busy_backoff_s | 2 / 4 / 8 |
| capture_retries_default | 3 |
| capture_capacity_wait_s_default | 90 |
| capture_concurrency_default | 1 |
| enroll_transport_retry | 2x250 ms (send-phase) |
| restart_retry_window_s | 3700 |
| label_anchor_alpha_client_default | 0.3 |
| configuration_digest | PENDING: hash after first campaign client build (defaults recorded from source) |

### Server container

| Field | Value |
|---|---|
| container name / id | `neo-glm53` / `262d903ddf9173bb76afe6e31dc05bf8112551c28337ae65eb7ad9e2a7eb6ed9` |
| started_utc | `2026-11-08T03:56:14.412735201Z` |
| server_source_attestation | runtime identity profile_digest `b98f9dc8fb0ce5839241ba93b3750acf72e6b5c4e0acee7bc106d103823838b3`; binding attested in runtime identity |

### Model checkpoint (glm-5.3)

| Field | Value |
|---|---|
| architecture / quantisation | `GlmMoeDsaForCausalLM` / fp8 |
| checkpoint_revision | `aca966e4e02791568aa6a4ced368624b3d897f42` |
| weight_fingerprint | metadata-manifest-sha256 `493a065ae286f3ca2077d36e8d0fd0bb931d7443d6dab309e7e2cf288c32d23e` |
| tokenizer_fingerprint | sha256 `08d2801689875e6819f26919b3e07ecf42693dd58e0527be477de286fb8ef532` |
| chat_template_fingerprint | sha256 `3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c` |

### Calibration artifact (alignment)

| Field | Value |
|---|---|
| path | `/occ3/calibration/calibrated-dual-ridge-9a7394dea73cf8d3.bin` |
| layer_index / hidden_dim / tile_rows | 77 / 6144 / 256 |
| schema | `optimus.occ3.calibration-artifact.v2` |

### Normalisation and anchoring

| Field | Value |
|---|---|
| serving_transform | N0/N1, floor 1e-6, no renormalize on apply_map |
| label_blend | `P_i = (1-alpha) Z_i + alpha E_in[t_i]` (token-anchored, not nearest-vocab) |
| effective per-handoff alpha | request-resolved (env > file > client default); Suite 0 completed-use receipts pin the effective value per arm |

### Capacity and profile policy

| Field | Value |
|---|---|
| OCC3_MAX_ACTIVE env | unset (no override) |
| OCC3_MAX_TURN_SECONDS | 900 |
| max_prefix_tokens | 128 |
| gpu memory per rank | 272499-272699 / 275040 MiB at baseline |

### Sampling and limits (pinned at Suite 0)

| Field | Value |
|---|---|
| reasoning_sampling_config | serving defaults; benchmark task budgets pinned at Suite 0 (matched across arms) |
| context_output_limits | pinned at Suite 0 per task budget; identical across arms |

## Update log
- 2026-11-09T13:40Z — initial manifest table and notes recorded (prior run).
- 2026-11-10 — extended with full manifest detail above by reports-init worker; no existing line modified. Branch head at extension time: `c1790003f` (held-parent fix; twin-guard fix `bc4a6375b` still pending its own build/cutover).
