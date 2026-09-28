//! Port of packages/coding-agent/src/modes/daemon/daemon-catalog-entry.ts
//!
//! The entry point is a module-level side effect in TypeScript: run the catalog
//! process and exit(1) with the same stderr line when it fails.

use futures::future::BoxFuture;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::core::session_file_actions::{
    delete_session_file, DeleteSessionFileMethod, DeleteSessionFileOptions, DeleteSessionFileResult,
};
use crate::core::session_manager::{
    read_session_info, CustomMessageEntryContent, SessionInfo, SessionListCallbacks, SessionManager,
};

use super::daemon_catalog_process::{run_daemon_catalog_process, CatalogSessionBackend};

pub const DAEMON_CATALOG_FAILURE_PREFIX: &str = "Optimus Agent daemon catalog failed: ";

/// Run the catalog process as the entry point does, reporting the failure text.
///
/// The TypeScript entry calls `runDaemonCatalogProcess()` with no argument
/// because it reaches the `SessionManager` statics directly. The port routes
/// those calls through `CatalogSessionBackend`, so the backend is forwarded from
/// the caller that owns the session manager.
pub async fn run_daemon_catalog_entry(
    backend: Arc<dyn CatalogSessionBackend>,
) -> Result<(), String> {
    match run_daemon_catalog_process(backend).await {
        Ok(()) => Ok(()),
        Err(message) => {
            eprintln!("{DAEMON_CATALOG_FAILURE_PREFIX}{message}");
            std::process::exit(1);
        }
    }
}

pub(crate) async fn run_native_catalog_process() -> Result<(), String> {
    run_daemon_catalog_process(Arc::new(NativeCatalogBackend)).await
}

struct NativeCatalogBackend;

impl CatalogSessionBackend for NativeCatalogBackend {
    fn list_boxed(
        &self,
        cwd: Option<String>,
        session_dir: Option<String>,
        on_progress: Arc<dyn Fn(u64, u64) + Send + Sync>,
        on_session: Arc<dyn Fn(SessionInfo) + Send + Sync>,
    ) -> BoxFuture<'static, Result<Vec<SessionInfo>, String>> {
        Box::pin(async move {
            let callbacks = SessionListCallbacks {
                on_progress: Some(Box::new(move |done, total| {
                    on_progress(done.max(0) as u64, total.max(0) as u64)
                })),
                on_session: Some(Box::new(move |session| on_session(session.clone()))),
            };
            Ok(match cwd {
                Some(cwd) => {
                    SessionManager::list(&cwd, session_dir.as_deref(), Some(callbacks)).await
                }
                None => SessionManager::list_all(Some(&callbacks), session_dir.as_deref()).await,
            })
        })
    }

    fn open_rename(&self, path: &str, name: &str) -> Result<(), String> {
        SessionManager::open(path, None, None)?
            .append_session_info(name.trim())
            .map(|_| ())
    }

    fn delete_boxed(&self, path: &str) -> BoxFuture<'static, Result<Value, String>> {
        let path = path.to_string();
        Box::pin(async move {
            let result = delete_session_file(&path, &mut DeleteSessionFileOptions::default());
            Ok(match result {
                DeleteSessionFileResult::Ok { method } => {
                    json!({"ok": true, "method": match method {
                        DeleteSessionFileMethod::Trash => "trash", DeleteSessionFileMethod::Unlink => "unlink",
                    }})
                }
                DeleteSessionFileResult::Error { error } => json!({"ok": false, "error": error}),
            })
        })
    }

    fn read_session_info_boxed(
        &self,
        path: &str,
    ) -> BoxFuture<'static, Result<Option<SessionInfo>, String>> {
        let path = path.to_string();
        Box::pin(async move { Ok(read_session_info(&path).await) })
    }

    fn append_session_state(&self, path: &str, status: &str) -> Result<(), String> {
        let state =
            serde_json::from_value(json!({"status": status})).map_err(|error| error.to_string())?;
        SessionManager::open(path, None, None)?
            .append_session_state(&state)
            .map(|_| ())
    }

    fn append_custom_message_entry(
        &self,
        path: &str,
        custom_type: &str,
        content: &str,
        metadata: Value,
    ) -> Result<(), String> {
        SessionManager::open(path, None, None)?
            .append_custom_message_entry(
                custom_type,
                &CustomMessageEntryContent::Text(content.to_string()),
                false,
                Some(metadata),
            )
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_prefix_matches_the_typescript() {
        assert_eq!(
            DAEMON_CATALOG_FAILURE_PREFIX,
            "Optimus Agent daemon catalog failed: "
        );
    }
}
