# OCC-3 Client Binary Verification (rebuilds + live campaign binaries)

## Rebuild at the 13:00Z tip (unchanged record)

1. Commit: 59d41ba207d9e88129a435b960bac606fc04fd9d (tip 59d41ba20, "docs(occ3): completion milestone reports for the retained lifecycle"; worktree clean, only untracked occ3-fullrecovery/ and target-occ3compl/).
2. sha256: 06c9dea81805ec21b097a1205a4c054415c5621ac8d5b0b6cceb655a68311f88
3. Size: 19,156,480 bytes (19,156,480 B; ~18.27 MiB).
4. Binary: C:/Users/openclawuser/optimus-occ3-icalib-20261028/target-occ3compl/release/examples/occ3_native_phase_suite.exe
5. Build command: cargo build --release -p pi-coding-agent --example occ3_native_phase_suite --locked
6. Build env: CARGO_TARGET_DIR=C:/Users/openclawuser/optimus-occ3-icalib-20261028/target-occ3compl; toolchain 1.99.1-x86_64-pc-windows-msvc; CARGO_HOME=C:/Users/openclawuser/optimus-rust-toolchain/cargo; RUSTUP_HOME=C:/Users/openclawuser/optimus-rust-toolchain/rustup; serial cargo (no concurrent cargo).
7. Build result: exit 0, "Finished `release` profile [optimized] target(s) in 8m 40s" (warnings only); completed 2026-11-01T13:33:17Z (UTC).
8. Smoke: `occ3_native_phase_suite.exe --help` -> exit code 1 (expected: no --help flag exists; arg-parser catch-all refuses before any enrollment or HTTP effect).
9. Smoke first output line (stderr): "Usage: occ3_native_phase_suite [--arm ARM] [--source-mode legacy|retained] [--corpus-turn TURN_MANIFEST_JSON] [--task TASK_JSON] [--offline [DIR] | --live PRIVATE_DESCRIPTOR_PATH]"; stdout empty.
10. Recorded: 2026-11-01T13:33:52Z (UTC). No git writes, no server code touched, gate battery not re-run (already green).

## Live campaign binaries (measured; the exe actually fired at the DGX)

- r16 — window icalib-canary2-20261101T1615Z (tree 77d57c64c):
  path `target-occ3compl/release/examples/occ3_native_phase_suite.exe`,
  sha256 7cd02b9c54795cfaa797fe72f5c82b768bb555c5c3fd77b87bef61de9fc595d8,
  18909696 B. Ran the canary-2 arm-A trial (refused by the producer on the arm's
  FAIL; evidence arm-evidence-native-2ec6cdef…, run stdout sha256 0e14ed32…).
- r17 — windows icalib-canary3-20261101T1745Z and icalib-sup-20261101T2003Z
  (tree 109ab15db): same exe path,
  sha256 1525df159a26b9a6c7e4938a972d3e8321133dcdc07ba1e385a6d7348abcc273.
  Ran the canary-3 arm-A fire (18:15:10Z, fire record in the window), the TOKEN_FULL
  diagnostic arm, and the supplemental collection turns (descriptor mint-manifest pins
  this sha). Descriptor sha256 for the arm-A fire: e2996390….
- Env for live fires: OCC3_SOURCE_THINKING=low; OCC3_NATIVE_API_KEY and
  OCC3_NATIVE_AUTH_HEADER supplied from the credential helpers, never printed.

## Pin discrepancy (recorded, not yet re-staged)

- The stage plan's `native_validator.harness_artifact.binary_sha256` and the offline
  self-test receipt still carry the r13 pins generation (160613f7d3320f203fdae4ab2
  0b554fcdc38410e41e26a15762b9d4e621f03c8); no harness-pins file exists for r16/r17
  (newest is harness-pins-r13.json, minted 2026-11-01T08:30:20Z).
- The descriptors are the self-binding truth layer: each names the exe sha actually
  executed (r16 7cd02b9c…, r17 1525df15…), and the phase suite re-verifies that
  digest at runtime. Root decided on 2026-11-01 not to re-stage for these windows;
  a future window can mint harness-pins-r17.json and re-stage if plan-level parity
  is needed (pin-discrepancy.txt, window icalib-canary3-20261101T1745Z).
