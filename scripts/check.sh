#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
git diff --check
cargo check --locked --workspace --all-targets
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check-rust-layout.py
