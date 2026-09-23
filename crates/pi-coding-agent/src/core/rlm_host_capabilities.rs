//! Provenance-bound, target-profile native capability. Not a remote-vendor rollback claim.
use super::*;

pub fn native_lifecycle_capabilities(profile: Option<Value>, ownership_supported: bool) -> Value {
    let sha256 = |value: Option<&str>| value.is_some_and(|value|
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    let provenance_pinned = sha256(option_env!("OPTIMUS_BUILD_FINGERPRINT"))
        && sha256(option_env!("OPTIMUS_RUNTIME_SOURCE_SHA256"));
    let supported = cfg!(windows) && ownership_supported && provenance_pinned;
    serde_json::json!({
        "schema":"optimus.native-lifecycle.v1", "capability":"rlm.stop-retain.v1",
        "supported":supported,
        "reasonCode":if supported { Value::Null } else { Value::String("unsupported_or_unpinned_target_profile".into()) },
        "targetProfile":profile,
        "provenance":{
            "hostImplementation":"optimus-rust",
            "protocolVersion":crate::modes::daemon::daemon_protocol::DAEMON_PROTOCOL_VERSION,
            "schemaRevision":crate::modes::daemon::daemon_protocol::DAEMON_SCHEMA_REVISION,
            "buildFingerprint":option_env!("OPTIMUS_BUILD_FINGERPRINT"),
            "runtimeSourceSha256":option_env!("OPTIMUS_RUNTIME_SOURCE_SHA256")
        },
        "limits":{"stopTimeoutMsMax":10000},
        "activeOnlyMessages":{"capability":"rlm.active-only-message.v1",
            "supported":provenance_pinned,"schema":"optimus.active-message.v1"},
        "auditResume":{"capability":"rlm.audit-resume.v1","supported":supported,
            "schema":"optimus.audit-resume.v1"},
        "scriptReports":{"capability":"rlm.structured-script-result.v1","supported":true,
            "schema":"optimus.script-result.v1","maxReportsPerCell":32,"maxReceiptBytes":16384}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_lifecycle_metadata_is_optional_and_never_grants_unknown_ownership() {
        let value = native_lifecycle_capabilities(None, false);
        assert_eq!(value["supported"], false);
        assert_eq!(value["auditResume"]["supported"], false);
        assert_eq!(value["provenance"]["protocolVersion"], 7);
        assert_eq!(value["provenance"]["schemaRevision"], 32);
        assert_eq!(value["limits"]["stopTimeoutMsMax"], 10_000);
        assert!(value["targetProfile"].is_null());
    }

    #[test]
    fn native_lifecycle_new_client_accepts_old_optional_result_shapes() {
        let tool: pi_agent_core::types::AgentToolResult = serde_json::from_value(
            serde_json::json!({"content":[],"details":{}})).unwrap();
        assert_eq!(tool.is_error, None);
        let result: crate::core::kernel::shared::ExecuteResult = serde_json::from_value(
            serde_json::json!({"stdout":"Traceback example","stderr":"", "status":"ok","durationMs":1.0})).unwrap();
        assert!(result.execution_reports.is_none());
        assert_eq!(result.status, crate::core::kernel::shared::ExecuteStatus::Ok);
    }

    #[test]
    fn native_lifecycle_old_client_ignores_additive_typed_metadata() {
        #[derive(serde::Deserialize)]
        struct LegacyTool { content: Vec<pi_agent_core::types::ContentBlock>, details: Value }
        #[derive(serde::Deserialize)]
        struct LegacyCell { stdout: String, status: String }
        let tool = pi_agent_core::types::AgentToolResult::new(Vec::new(), serde_json::json!({})).with_error(true);
        let old: LegacyTool = serde_json::from_value(serde_json::to_value(tool).unwrap()).unwrap();
        assert!(old.content.is_empty());
        assert_eq!(old.details, serde_json::json!({}));
        let cell: LegacyCell = serde_json::from_value(serde_json::json!({
            "stdout":"legacy view", "status":"error", "executionReports":[]
        })).unwrap();
        assert_eq!(cell.stdout, "legacy view");
        assert_eq!(cell.status, "error");
    }
}
