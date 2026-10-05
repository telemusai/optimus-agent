//! Per-request credential resolution cache (`resolveConfigValueOrThrow`).
//!
//! Command-backed models.json credentials previously executed a shell helper on
//! every dispatch (measured as a ~0.6s pre-dispatch floor on every dgx request:
//! wait_ms p50 608ms, p10 515ms, n=42,853, size-independent). This test pins the
//! refresh-window contract: reuse within the TTL, re-execute after invalidation,
//! the env-var escape hatch that restores uncached resolution, and the actual
//! spawn skip on the second call.
//!
//! One sequential test on purpose: the cache and its TTL env var are
//! process-global, so parallel #[test]s mutating them would race each other.

use std::time::{Duration, Instant};

use pi_coding_agent::core::resolve_config_value::{
    invalidate_resolved_command_values, resolve_config_value_or_throw,
};

const TTL_ENV: &str = "PRIME_AGENT_CREDENTIAL_CACHE_TTL_MS";

/// A helper that echoes a file, so the test can rotate the "token" on disk.
fn file_echo_command(directory: &std::path::Path) -> String {
    let path = directory.join("credential.txt");
    if cfg!(windows) {
        let script = directory.join("credential.ps1");
        std::fs::write(
            &script,
            format!(
                "$ErrorActionPreference = 'Stop'\nGet-Content -Raw -LiteralPath '{}'\n",
                path.display().to_string().replace('\'', "''"),
            ),
        )
        .unwrap();
        format!(
            "!powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{}\"",
            script.display()
        )
    } else {
        format!("!cat '{}'", path.display().to_string().replace('\'', "'\\''"))
    }
}

fn write_token(directory: &std::path::Path, token: &str) {
    std::fs::write(directory.join("credential.txt"), token).unwrap();
}

fn resolve(directory: &std::path::Path) -> String {
    let command = file_echo_command(directory);
    resolve_config_value_or_throw(&command, "test credential")
        .unwrap()
        .trim()
        .to_string()
}

#[test]
fn refresh_window_reuse_invalidation_and_env_escape_hatch() {
    std::env::remove_var(TTL_ENV);
    let directory = tempfile::tempdir().unwrap();

    write_token(directory.path(), "first-token\n");
    let first = resolve(directory.path());
    assert_eq!(first, "first-token");

    // Within the TTL the helper must not run again: the rotated file is not read.
    write_token(directory.path(), "second-token\n");
    let second = resolve(directory.path());
    assert_eq!(second, "first-token");

    // A credential event (write, refresh, 401 stale marking) drops the cache and
    // the next resolution re-executes the helper.
    invalidate_resolved_command_values();
    let third = resolve(directory.path());
    assert_eq!(third, "second-token");

    // The fresh value is cached again.
    write_token(directory.path(), "third-token\n");
    let fourth = resolve(directory.path());
    assert_eq!(fourth, "second-token");

    // The measured motivation, asserted directly: the cached lookup skips the
    // process spawn entirely, so it costs a small fraction of one helper run.
    let started = Instant::now();
    let cached = resolve(directory.path());
    let cached_duration = started.elapsed();
    assert_eq!(cached, "second-token");
    invalidate_resolved_command_values();
    let started = Instant::now();
    let fresh = resolve(directory.path());
    let fresh_duration = started.elapsed();
    assert_eq!(fresh, "third-token");
    assert!(
        cached_duration < fresh_duration.max(Duration::from_millis(50)) / 4,
        "cached lookup took {cached_duration:?} vs helper {fresh_duration:?}"
    );

    // TTL 0 restores the pre-cache behavior: every resolution executes the helper.
    std::env::set_var(TTL_ENV, "0");
    write_token(directory.path(), "fourth-token\n");
    assert_eq!(resolve(directory.path()), "fourth-token");
    write_token(directory.path(), "fifth-token\n");
    assert_eq!(resolve(directory.path()), "fifth-token");

    // A huge TTL is accepted and clamped internally; the value stays cached,
    // and a value stored under the earlier TTL=0 escape hatch does not
    // resurface after the TTL is raised again.
    std::env::set_var(TTL_ENV, "999999999");
    write_token(directory.path(), "sixth-token\n");
    assert_eq!(resolve(directory.path()), "sixth-token");
    write_token(directory.path(), "seventh-token\n");
    assert_eq!(resolve(directory.path()), "sixth-token");

    std::env::remove_var(TTL_ENV);
}
