#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cargo test --locked --workspace --lib -- --test-threads=1
python3 -m unittest discover -s tests -p 'test_*.py'
uv run --locked --project prime-agent-runtime python -m unittest discover -s prime-agent-runtime/test
