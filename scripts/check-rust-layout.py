#!/usr/bin/env python3
"""Validate the native bundle and reject accidental legacy source reintroduction."""

import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
).decode().split("\0")
legacy = [name for name in paths if name.endswith((".ts", ".tsx")) and (root / name).is_file()]
assert not legacy, f"Legacy TypeScript sources found: {legacy}"
assert not (root / "package.json").exists(), "The native repository must not require an npm workspace"
resources = root / "resources/agent"
identity = json.loads((resources / "package.json").read_text())
assert not any(key in identity for key in ("dependencies", "devDependencies", "scripts", "bin"))
for name in ("prime", "dark", "light"):
    json.loads((resources / f"src/modes/interactive/theme/{name}.json").read_text())
for name in ("template.html", "template.css", "template.js", "vendor/marked.min.js", "vendor/highlight.min.js"):
    assert (resources / "src/core/export-html" / name).is_file(), name
for name in ("edit", "memory", "compact", "goal", "refine", "agent-message", "agent-observe"):
    assert (resources / "skills" / name / "pyproject.toml").is_file(), name
assert (root / "prime-agent-runtime/src/rlm/repl.py").is_file()
print("Native resource layout verified; no TypeScript implementation or npm workspace.")
