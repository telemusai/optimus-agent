//! Port of packages/coding-agent/src/core/kernel/state-snapshot.ts.
//!
//! Locations and result shapes for the kernel's persisted user namespace, which
//! is revived when a session resumes. The kernel is otherwise spawned fresh on
//! resume, leaving the model believing it still has access to variables/imports
//! it defined earlier.
//!
//! Snapshotting is best-effort and per-variable: each top-level name is pickled
//! with `dill` independently, so a single unpicklable object (open file, socket,
//! GPU tensor, ...) is skipped and reported rather than aborting the whole snapshot.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Default ceiling on a snapshot payload. Over-cap variables are skipped + reported.
pub const DEFAULT_SNAPSHOT_MAX_BYTES: u64 = 256 * 1024 * 1024;
/// Default ceiling for one serialized variable.
pub const DEFAULT_SNAPSHOT_MAX_VARIABLE_BYTES: u64 = 16 * 1024 * 1024;

/// Base filename for the kernel snapshot within a session's artifact directory.
const KERNEL_STATE_BASENAME: &str = "kernel-state";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelSnapshotFormat {
    Legacy,
    CasV2,
}

impl KernelSnapshotFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelSnapshotFormat::Legacy => "legacy",
            KernelSnapshotFormat::CasV2 => "cas-v2",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "legacy" => Some(KernelSnapshotFormat::Legacy),
            "cas-v2" => Some(KernelSnapshotFormat::CasV2),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelRestoreSource {
    Auto,
    Current,
    Previous,
    Legacy,
}

impl KernelRestoreSource {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelRestoreSource::Auto => "auto",
            KernelRestoreSource::Current => "current",
            KernelRestoreSource::Previous => "previous",
            KernelRestoreSource::Legacy => "legacy",
        }
    }
}

/// Numeric-only timings and byte counts returned by the Python serializer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SnapshotPerformanceMetadata {
    pub serialization_wall_ms: Option<f64>,
    pub serialization_cpu_ms: Option<f64>,
    pub serialization_max_variable_ms: Option<f64>,
    pub serialization_slow_variables: Option<f64>,
    pub serialization_saved_ms: Option<f64>,
    pub serialization_skipped_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_native_values: Option<f64>,
    #[serde(skip)]
    pub serialization_dill_values: Option<f64>,
    #[serde(skip)]
    pub serialization_native_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_dill_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_bytes: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_attempts: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_rejected: Option<f64>,
    #[serde(skip)]
    pub serialization_fragment_prepare_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_fragment_write_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_fragment_bytes: Option<f64>,
    #[serde(skip)]
    pub serialization_fragment_segments: Option<f64>,
    #[serde(skip)]
    pub serialization_buffer_reset_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_blob_extract_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_saved_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_native_probe_skipped_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_envelope_count_ms: Option<f64>,
    #[serde(skip)]
    pub serialization_envelope_count_calls: Option<f64>,
    #[serde(skip)]
    pub serialization_envelope_write_ms: Option<f64>,
    #[serde(skip)]
    pub snapshot_cas_captures: Option<f64>,
    #[serde(skip)]
    pub snapshot_legacy_captures: Option<f64>,
    pub serialized_bytes: Option<f64>,
    /// A5: names reused from the per-variable digest cache (not re-pickled).
    pub serialization_reused_names: Option<f64>,
    /// A6: names dropped from a budget-aware partial snapshot.
    pub dropped_names_count: Option<f64>,
    pub write_ms: Option<f64>,
    pub written_bytes: Option<f64>,
    pub total_wall_ms: Option<f64>,
}

/// One skipped (unserializable) name and a short reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkippedVariable {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotResult {
    /// Top-level names successfully serialized into the payload.
    pub saved: Vec<String>,
    /// Names that could not be serialized, with a short reason.
    pub skipped: Vec<SkippedVariable>,
    /// Oversized live variables removed by an explicit compaction snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruned: Option<Vec<String>>,
    /// A6: names dropped by the snapshot time budget (bounded list; the
    /// generation manifest keeps the full list).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped: Option<Vec<String>>,
    /// Legacy-equivalent payload bytes, including outer-container overhead.
    pub bytes: u64,
    /// Sum of retained independent per-name dill blobs.
    pub logical_bytes: u64,
    /// Bytes physically written by this snapshot attempt, including CAS metadata.
    pub written_bytes: u64,
    pub format: KernelSnapshotFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<String>,
    /// False for CAS v2 until an explicit legacy export is completed.
    pub backward_readable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<SnapshotPerformanceMetadata>,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RestoreResult {
    /// Names successfully revived into the kernel namespace.
    pub restored: Vec<String>,
    /// Names present in the snapshot that failed to revive, with a short reason.
    pub failed: Vec<SkippedVariable>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<KernelSnapshotFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<String>,
    /// Previous-generation restore was explicitly requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rolled_back: Option<bool>,
    /// The selected source may omit newer committed or unsaved work.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsaved_work_possible: Option<bool>,
    /// Legacy was explicitly selected despite a v2 state root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_recovery: Option<bool>,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotLegacyExportResult {
    pub exported: Vec<String>,
    pub bytes: u64,
    pub source_generation: String,
    pub source: LegacyExportSource,
    pub backward_readable: bool,
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LegacyExportSource {
    Current,
    Previous,
}

impl LegacyExportSource {
    pub fn as_str(self) -> &'static str {
        match self {
            LegacyExportSource::Current => "current",
            LegacyExportSource::Previous => "previous",
        }
    }
}

/// Absolute path to the legacy dill payload within a session's artifact directory.
pub fn snapshot_path_in(artifact_dir: &str) -> String {
    join_path(artifact_dir, &format!("{KERNEL_STATE_BASENAME}.dill"))
}

/// Absolute path to the legacy JSON manifest within a session's artifact directory.
pub fn manifest_path_in(artifact_dir: &str) -> String {
    join_path(artifact_dir, &format!("{KERNEL_STATE_BASENAME}.json"))
}

/// Session-scoped root for CAS v2 blobs, generations, marker, and pointer.
pub fn cas_snapshot_root_in(artifact_dir: &str) -> String {
    join_path(artifact_dir, &format!("{KERNEL_STATE_BASENAME}.v2"))
}

/// Derive a sibling CAS root for direct manager users that only supply a legacy path.
pub fn cas_snapshot_root_for_legacy_path(snapshot_path: &str) -> String {
    match snapshot_path.strip_suffix(".dill") {
        Some(prefix) => format!("{prefix}.v2"),
        None => format!("{snapshot_path}.v2"),
    }
}

fn join_path(dir: &str, name: &str) -> String {
    PathBuf::from(dir).join(name).to_string_lossy().into_owned()
}

fn path_entry_exists(path: &str) -> bool {
    match std::fs::symlink_metadata(Path::new(path)) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        // Permission and malformed-reparse failures are state, not proof of absence.
        Err(_) => true,
    }
}

/// True for a legacy payload or any v2 root entry, including broken reparse links.
pub fn snapshot_state_exists_in(artifact_dir: &str) -> bool {
    path_entry_exists(&snapshot_path_in(artifact_dir)) || path_entry_exists(&cas_snapshot_root_in(artifact_dir))
}

/// Manager-level v2 detection when only explicit paths are available.
pub fn cas_snapshot_state_exists(root: &str) -> bool {
    path_entry_exists(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_snapshot_stage_metrics_remain_local_and_preserve_result_json() {
        let mut legacy: SnapshotPerformanceMetadata = serde_json::from_value(serde_json::json!({
            "serialization_wall_ms": 12.0,
            "serialized_bytes": 1024.0
        })).unwrap();
        assert_eq!(legacy.serialization_wall_ms, Some(12.0));
        let original = serde_json::to_value(&legacy).unwrap();
        legacy.serialization_native_ms = Some(5.0);
        legacy.snapshot_cas_captures = Some(1.0);
        assert_eq!(serde_json::to_value(&legacy).unwrap(), original);
        let keys = [
            "serialization_native_values",
            "serialization_dill_values",
            "serialization_native_ms",
            "serialization_dill_ms",
            "serialization_native_probe_ms",
            "serialization_native_probe_bytes",
            "serialization_native_probe_attempts",
            "serialization_native_probe_rejected",
            "serialization_fragment_prepare_ms",
            "serialization_fragment_write_ms",
            "serialization_fragment_bytes",
            "serialization_fragment_segments",
            "serialization_buffer_reset_ms",
            "serialization_blob_extract_ms",
            "serialization_native_probe_saved_ms",
            "serialization_native_probe_skipped_ms",
            "serialization_envelope_count_ms",
            "serialization_envelope_count_calls",
            "serialization_envelope_write_ms",
            "snapshot_cas_captures",
            "snapshot_legacy_captures",
        ];
        let mut input = original.clone();
        for key in keys {
            input.as_object_mut().unwrap().insert(key.into(), serde_json::json!(5.0));
        }
        let decoded: SnapshotPerformanceMetadata = serde_json::from_value(input).unwrap();
        assert_eq!(decoded.serialization_native_ms, None);
        assert_eq!(decoded.snapshot_cas_captures, None);
        assert_eq!(serde_json::to_value(decoded).unwrap(), original);
        for key in keys {
            assert!(original.get(key).is_none());
        }
    }

    #[test]
    fn snapshot_paths_are_derived_from_the_artifact_dir() {
        let dir = if cfg!(windows) { "C:\\artifacts" } else { "/artifacts" };
        assert!(snapshot_path_in(dir).ends_with("kernel-state.dill"));
        assert!(manifest_path_in(dir).ends_with("kernel-state.json"));
        assert!(cas_snapshot_root_in(dir).ends_with("kernel-state.v2"));
    }

    #[test]
    fn cas_root_for_legacy_path_strips_the_dill_suffix() {
        assert_eq!(cas_snapshot_root_for_legacy_path("/a/kernel-state.dill"), "/a/kernel-state.v2");
        assert_eq!(cas_snapshot_root_for_legacy_path("/a/kernel-state"), "/a/kernel-state.v2");
    }

    #[test]
    fn missing_entries_are_absent_and_present_entries_exist() {
        let dir = std::env::temp_dir().join(format!("pi-snapshot-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_string = dir.to_string_lossy().into_owned();
        assert!(!snapshot_state_exists_in(&dir_string));
        std::fs::write(snapshot_path_in(&dir_string), b"x").unwrap();
        assert!(snapshot_state_exists_in(&dir_string));
        assert!(cas_snapshot_state_exists(&snapshot_path_in(&dir_string)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_format_round_trips_wire_names() {
        assert_eq!(KernelSnapshotFormat::CasV2.as_str(), "cas-v2");
        assert_eq!(KernelSnapshotFormat::from_str("legacy"), Some(KernelSnapshotFormat::Legacy));
        assert_eq!(KernelSnapshotFormat::from_str("nope"), None);
        assert_eq!(KernelRestoreSource::Previous.as_str(), "previous");
    }
}
