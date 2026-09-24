//! Native daemon roster transport and unified session workspace entry point.
use std::sync::Arc;
use serde_json::Value;
use super::native_wire::normalize_browser_numbers;
use super::roster_store as roster;
use crate::main_entry::AgentsViewSeamOptions;
use crate::modes::daemon::daemon_client::{self, DaemonClient};

pub(crate) struct NativeTransport(Arc<DaemonClient>);
impl NativeTransport {
    pub(crate) fn new(socket: &str) -> Self {
        Self(DaemonClient::create(socket))
    }
    fn hello_value(&self, hello: daemon_client::DaemonHello) -> roster::DaemonHello {
        roster::DaemonHello {
            socket_path: self.0.socket_path().into(),
            server_capabilities: hello.server_capabilities,
            client_id: hello
                .raw
                .get("clientId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .into(),
            schema_revision: hello.schema_revision.map(i64::from),
        }
    }
}

fn outbound(value: &Value) -> roster::DaemonOutbound {
    match value.get("type").and_then(Value::as_str) {
        Some("roster_update") => roster::DaemonOutbound::RosterUpdate {
            changed: value
                .get("changed")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|entry| serde_json::from_value(normalize_browser_numbers(entry.clone())).ok())
                .collect(),
            removed: value
                .get("removed")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok()),
            resync: value.get("resync").and_then(Value::as_bool),
        },
        Some("heartbeats_changed") => roster::DaemonOutbound::HeartbeatsChanged,
        _ => roster::DaemonOutbound::Other,
    }
}

impl roster::DaemonTransport for NativeTransport {
    fn fresh_transport(&self) -> Option<Arc<dyn roster::DaemonTransport>> {
        Some(Arc::new(Self::new(self.0.socket_path())))
    }
    fn hello(&self) -> Option<roster::DaemonHello> {
        self.0.hello().map(|hello| self.hello_value(hello))
    }
    fn is_connected(&self) -> bool {
        self.0.is_connected()
    }
    fn supports_server_capability(&self, capability: &str) -> bool {
        self.0.supports_server_capability(capability)
    }
    fn wait_for_hello(
        &self,
        timeout_ms: u64,
    ) -> roster::TransportFuture<Result<roster::DaemonHello, String>> {
        let client = self.0.clone();
        Box::pin(async move {
            let hello = client
                .wait_for_hello(timeout_ms)
                .await
                .map_err(|error| error.message())?;
            Ok(NativeTransport(client).hello_value(hello))
        })
    }
    fn request(
        &self,
        command: Value,
        timeout_ms: u64,
        options: roster::DaemonClientRequestOptions,
    ) -> roster::TransportFuture<Result<roster::DaemonResponse, String>> {
        let client = self.0.clone();
        Box::pin(async move {
            let body = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be an object".to_string())?;
            let response = client
                .request(
                    body,
                    Some(timeout_ms),
                    daemon_client::DaemonClientRequestOptions {
                        on_progress: options.on_progress.map(Arc::from),
                        recoverable: options.recoverable,
                        signal: None,
                    },
                )
                .await
                .map_err(|error| error.message())?;
            Ok(roster::DaemonResponse {
                command: response.command,
                success: response.success,
                data: response.data.map(normalize_browser_numbers),
                error: response.error,
            })
        })
    }
    fn on_message(&self, listener: roster::MessageListener) -> Box<dyn Fn() + Send + Sync> {
        self.0
            .on_message(Arc::new(move |value| listener(&outbound(value))))
    }
    fn on_close(&self, listener: Box<dyn Fn(&str) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.0
            .on_close(Arc::new(move |error| listener(&error.message())))
    }
    fn connect(&self, timeout_ms: u64) -> roster::TransportFuture<Result<(), String>> {
        let client = self.0.clone();
        Box::pin(async move {
            client
                .connect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }
    fn reconnect(&self, timeout_ms: u64) -> roster::TransportFuture<Result<(), String>> {
        let client = self.0.clone();
        Box::pin(async move {
            client
                .reconnect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }
    fn close(&self) {
        let client = self.0.clone();
        tokio::spawn(async move {
            client.close().await;
        });
    }
    fn socket_path(&self) -> String {
        self.0.socket_path().into()
    }
}

pub(crate) async fn run_agents_view_mode(options: AgentsViewSeamOptions) -> Result<(), String> {
    crate::modes::interactive::native_host::run_workspace(options).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roster_wire_push_preserves_changes_removals_and_resync() {
        let entry = roster::AgentRosterEntry {
            agent_id: "agent-1".into(),
            summary: super::super::agents_view_state::SessionSummary::new(
                "agent-1",
                "session-1",
                "/work",
            ),
            ..Default::default()
        };
        let event = outbound(&serde_json::json!({
            "type": "roster_update", "changed": [entry.clone()], "removed": ["old-agent"], "resync": true,
        }));
        assert_eq!(
            event,
            roster::DaemonOutbound::RosterUpdate {
                changed: vec![entry],
                removed: Some(vec!["old-agent".into()]),
                resync: Some(true),
            }
        );
        assert_eq!(
            outbound(&serde_json::json!({"type":"heartbeats_changed"})),
            roster::DaemonOutbound::HeartbeatsChanged
        );
    }
}
