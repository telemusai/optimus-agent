#!/usr/bin/env bash
set -euo pipefail
optimus_source="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
optimus_target="${CARGO_TARGET_DIR:-$optimus_source/target}"
if [[ "$optimus_target" != /* && "$optimus_target" != [A-Za-z]:* ]]; then
  optimus_target="$optimus_source/$optimus_target"
fi
optimus_binary="${OPTIMUS_RUST_BINARY:-$optimus_target/debug/optimus-rust}"
if [[ ! -x "$optimus_binary" && -x "$optimus_binary.exe" ]]; then
  optimus_binary="$optimus_binary.exe"
fi
if [[ ! -x "$optimus_binary" ]]; then
  echo "Build Optimus first: cargo build --locked -p pi-coding-agent --bin optimus-rust" >&2
  echo "For a release or cross-target build, set OPTIMUS_RUST_BINARY to its executable." >&2
  exit 1
fi
export PI_PACKAGE_DIR="$optimus_source/resources/agent"
export PI_CODING_AGENT_MODULE_DIR="$PI_PACKAGE_DIR"
export PRIME_AGENT_LAUNCHER_PATH="$optimus_source/optimus-agent.sh"
if optimus_build_id="$(git -C "$optimus_source" describe --tags --always --dirty 2>/dev/null)"; then
  export PRIME_AGENT_BUILD_ID="$optimus_build_id"
fi
exec "$optimus_binary" "$@"
