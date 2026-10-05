@echo off
rem Private Rust toolchain for this port (no machine-wide PATH/registry changes).
set "RUSTUP_HOME=C:\Users\openclawuser\optimus-rust-toolchain\rustup"
set "CARGO_HOME=C:\Users\openclawuser\optimus-rust-toolchain\cargo"
set "PATH=C:\Users\openclawuser\optimus-rust-toolchain\cargo\bin;%PATH%"
