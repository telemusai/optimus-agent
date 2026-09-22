# Native development

See the root [README](../../../README.md) and [AGENTS.md](../../../AGENTS.md).

## Build and run

Use Rust/Cargo (validated with 1.95.0), Python 3.11+, uv, and Bash. Windows also requires the MSVC build tools and Git Bash.

```bash
cargo build --locked -p pi-coding-agent --bin optimus-rust
./optimus-agent.sh
```

Run the launcher from any working directory. It selects the checkout's binary and resources while preserving the project directory. Set `OPTIMUS_RUST_BINARY` for a different build. Use `PRIME_AGENT_CODING_AGENT_DIR` and `--daemon-socket` to isolate development from running sessions.

The Rust crates are the application source. `prime-agent-runtime/` implements the Python kernel. `resources/agent/` contains bundled skills, documentation, themes, and HTML viewer assets. The resource `package.json` is identity metadata, not an npm package to install. Use `crate::config` helpers for resource paths. Package the executable and resources together with `scripts/rust_release.py`.

## Validation

```bash
bash scripts/check.sh
./test.sh
```

The check runs Cargo checking, packaging/launcher Python tests, and resource validation. The test script runs Rust unit tests and the Python runtime suite. Run affected integration tests explicitly, for example:

```bash
cargo test --locked -p pi-coding-agent --test native_cli -- --test-threads=1
```

See the [Rust-only validation record](../../../docs/RUST_ONLY_VALIDATION.md) for focused CI coverage and wider-suite limitations.

Use local model fixtures and private profiles. Do not call paid providers for regression tests. Preserve daemon protocol negotiation when changing client/worker contracts.

## Releases

Use `./install.sh` for a local release build. The native bundle workflow builds platform-specific archives on tags and manual runs; tagged bundles are published on GitHub. The former npm/R2 release pipeline is removed.

Add a short changelog fragment under `.changes/`. Historical release notes remain in `resources/agent/CHANGELOG.md`; do not rewrite them.
