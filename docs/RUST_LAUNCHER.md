# Running the installed Rust application

The installed command is **`optimus-agent`**. Run it from the project you want to work on:

```bash
cd /path/to/your/project
optimus-agent
```

Use `optimus-agent --help` for command-line options. In the TUI, `/model` opens the searchable picker. Type a model ID, name, or provider; use Up/Down to navigate, Enter to select, and Escape to cancel. `/model <search>` opens a prefilled search, or switches directly when the reference is exact and unambiguous. Alt+S toggles all/scoped models when a model scope is configured; the shortcut follows the `app.model.toggleScope` keybinding. Current and recent models appear first within signed-in providers. Selecting a built-in provider that needs credentials opens its login flow.

Provider login is available through `/login` and the model picker. Native limitations remain listed in [RUST_MAIN_READINESS.md](RUST_MAIN_READINESS.md).

## Linux installation layout

`scripts/optimus-agent` is the maintained launcher for the local installation:

```text
~/.local/bin/optimus-agent
~/.local/share/optimus-rust/
  current -> releases/<commit>
  releases/<commit>/
    COMMIT
    bin/optimus-rust
    resources/agent/
    scripts/
    install.sh
    prime-agent-runtime/
    kernel-venv/
~/.config/optimus-rust/
```

The release contains the built executable and the matching Python runtime, skills, and package resources. The launcher sets their locations, uses a separate Rust profile, and isolates daemon sockets between releases. Each release gets its own Python environment on first tool use, prepared by `uv`, so an upgrade leaves the older environment intact. The launcher preserves the caller's working directory and forwards every argument. It does not copy credentials or sessions automatically.

## Build and install

For installation without a checkout, use the [standalone install/update commands](../README.md#install-and-update). Both platforms resolve the highest stable `vMAJOR.MINOR.PATCH` tag, verify the checked-out commit against that tag, and build its locked Cargo dependencies locally. Annotated and lightweight tags work; prerelease tags and untagged `main` commits are excluded. The first supported standalone tag is `v0.1.1`. GitHub Actions remains disabled.

The source build uses a private `.build-*` directory under the installation prefix, defaults to two Cargo jobs, and removes its source and build artifacts on success or failure. Allow several GB of free space and several minutes to compile. `CARGO_BUILD_JOBS` can override the job count. An unchanged tag/commit with installed launchers is checked with `--version` and not rebuilt.

On Linux/macOS, custom installation options can be passed to the downloaded shell script:

```bash
curl -fsSL https://telemus.ai/optimus-agent/install.sh | sh -s -- \
  --prefix "$HOME/opt/optimus" --bin-dir "$HOME/opt/bin"
```

Use `--force` to rebuild the same tag. Set `OPTIMUS_RUST_ROOT` to your custom prefix when launching. Windows uses the default user installation directories through the PowerShell one-liner.

From a source checkout with Rust/Cargo, Python 3.11+, `uv`, and Bash installed:

```bash
./install.sh
```

This builds the native release binary and installs a complete bundle. To reuse a compiled executable, pass `--binary /path/to/optimus-rust`. Add `~/.local/bin` to PATH and check `optimus-agent --version`. The installer replaces the launcher atomically without following an old symlink, and retains previous releases. It never copies or rewrites credentials, sessions, or memory.

Windows uses Git Bash and the MSVC Rust toolchain. The installer records `current.txt` instead of requiring Windows symlink privileges, and the launcher selects `bin/optimus-rust.exe`. It also installs `optimus-agent.cmd`, which invokes the detected Git Bash with the shared launcher, allowing `optimus-agent` to run directly from PowerShell or Command Prompt. Both launchers are restored if activation fails. The web PowerShell installer adds the bin directory to the user's PATH. Linux/macOS use the `current` symlink shown above. Standalone installations record both `COMMIT` and `TAG` in each release.

To create a distributable archive from a compiled binary:

```bash
python3 scripts/rust_release.py stage \
  --binary target/release/optimus-rust \
  --output dist --name optimus-agent-local
```

The archive contains the executable, resource bundle, Python runtime source, installer, and license, plus a SHA-256 sidecar. The repository retains native release workflows, but GitHub Actions is disabled and the hosted installers build from source. For a manually built archive, extract it and run its `./install.sh`; Rust is only required when compiling from source. Python and `uv` are still required for the execution runtime. Node/npm are not required.

Use `--prefix` and `--bin-dir` for an isolated installation. Set `OPTIMUS_RUST_ROOT` to a custom prefix when invoking its launcher. Copying only the executable is insufficient for Python tools and bundled skills.

The launcher respects `PRIME_AGENT_CODING_AGENT_DIR` and `PRIME_AGENT_KERNEL_VENV` overrides. `OPTIMUS_RUST_TMPDIR` overrides the socket directory; keep it short enough for Unix socket limits. The default is `$XDG_RUNTIME_DIR/optimus-rust/<release>`, falling back to `~/.cache/optimus-rust/<release>`. It is separate from Prime's temporary socket directory.

Closing the TUI disconnects that client; the supervisor and resident sessions can continue in the background. Client-owned temporary sessions have their own disconnect cleanup. Changing the launcher does not restart existing processes.

To reuse an existing Prime OAuth login, share its `auth.json` through a symlink after backing up the Rust credential file. Independent copies of a rotating refresh token can become stale. Rust resolves the credential path before locking, so it coordinates refreshes with Prime's `proper-lockfile` lock on the original file. Settings, sessions, and daemon state can remain in the separate Rust profile. Only share credentials between trusted local profiles.

To use an existing Codex CLI login instead, configure `providers.openai-codex.apiKey` in the Rust profile's `models.json` with a command that reads the current access token:

```json
"apiKey": "!python3 -c 'import json; from pathlib import Path; print(json.loads((Path.home()/\".codex/auth.json\").read_text())[\"tokens\"][\"access_token\"])'"
```

Back up and remove any `openai-codex` entry in the Rust profile's `auth.json` first, because stored credentials take precedence over `models.json`. Keep the existing model definitions. This command is resolved for each request; Codex remains responsible for its login and token refresh. Use the actual Codex auth path if `CODEX_HOME` is customized. Do not symlink the whole Codex auth file: its format differs from Prime's.

Retain previous releases during upgrades. An already-running client keeps its loaded executable; relaunch `optimus-agent` to use an updated client. Do not replace a Python environment while it is executing tools.
