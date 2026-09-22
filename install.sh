#!/usr/bin/env bash
set -euo pipefail
optimus_source="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
  optimus_python=python3
else
  optimus_python=python
fi
exec "$optimus_python" "$optimus_source/scripts/rust_release.py" install "$@"
