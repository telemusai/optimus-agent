# Native application resources

This directory contains resources loaded by the Rust application: Python skills,
user documentation, themes, terminal images, and the standalone HTML session
viewer. It contains no TypeScript application or npm dependencies.

`package.json` is resource identity metadata consumed by Rust. The `src/` paths
are resource paths retained by the theme and export readers, not a second
application implementation. JavaScript in `src/core/export-html/` runs in the
exported HTML viewer; it is not a Node application.

Release bundles include this directory and `prime-agent-runtime/`. Use the root
README and `scripts/rust_release.py` for development and packaging.
