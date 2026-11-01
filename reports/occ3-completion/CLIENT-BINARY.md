# OCC-3 Client Binary Verification (offline rebuild at current tip)

1. Commit: 59d41ba207d9e88129a435b960bac606fc04fd9d (tip 59d41ba20, "docs(occ3): completion milestone reports for the retained lifecycle"; worktree clean, only untracked occ3-fullrecovery/ and target-occ3compl/).
2. sha256: 06c9dea81805ec21b097a1205a4c054415c5621ac8d5b0b6cceb655a68311f88
3. Size: 19,156,480 bytes (19,156,480 B; ~18.27 MiB).
4. Binary: C:/Users/openclawuser/optimus-occ3-icalib-20261028/target-occ3compl/release/examples/occ3_native_phase_suite.exe
5. Build command: cargo build --release -p pi-coding-agent --example occ3_native_phase_suite --locked
6. Build env: CARGO_TARGET_DIR=C:/Users/openclawuser/optimus-occ3-icalib-20261028/target-occ3compl; toolchain 1.99.1-x86_64-pc-windows-msvc; CARGO_HOME=C:/Users/openclawuser/optimus-rust-toolchain/cargo; RUSTUP_HOME=C:/Users/openclawuser/optimus-rust-toolchain/rustup; serial cargo (no concurrent cargo).
7. Build result: exit 0, "Finished `release` profile [optimized] target(s) in 8m 40s" (warnings only, no errors); completed 2026-11-01T13:33:17Z (UTC).
8. Smoke: `occ3_native_phase_suite.exe --help` -> exit code 1 (expected: no --help flag exists; arg-parser catch-all refuses before any enrollment or HTTP effect).
9. Smoke first output line (stderr): "Usage: occ3_native_phase_suite [--arm ARM] [--source-mode legacy|retained] [--corpus-turn TURN_MANIFEST_JSON] [--task TASK_JSON] [--offline [DIR] | --live PRIVATE_DESCRIPTOR_PATH]"; stdout empty.
10. Recorded: 2026-11-01T13:33:52Z (UTC). No git writes, no server code touched, gate battery not re-run (already green).
