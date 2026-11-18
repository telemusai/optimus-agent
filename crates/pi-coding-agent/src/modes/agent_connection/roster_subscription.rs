//! Connect the shared roster store to the interactive connection's accessory bar.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::modes::agents_view::native_host::NativeTransport;
use crate::modes::agents_view::roster_store::{
    AgentsViewRosterStore, DaemonTransport, DaemonTransportClient as RosterClient,
};

#[derive(Clone)]
pub(super) struct RosterSubscription {
    store: AgentsViewRosterStore,
    summaries: Arc<Mutex<Vec<Value>>>,
    listeners: Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>>,
    transport: Arc<Mutex<Option<Arc<dyn DaemonTransport>>>>,
    attach_lock: Arc<tokio::sync::Mutex<()>>,
    cancelled: tokio_util::sync::CancellationToken,
    initialized: Arc<Mutex<bool>>,
    observation_epoch: Arc<AtomicU64>,
}

impl RosterSubscription {
    pub(super) fn new() -> Self {
        Self {
            store: AgentsViewRosterStore::new(),
            summaries: Arc::new(Mutex::new(Vec::new())),
            listeners: Arc::new(Mutex::new(Vec::new())),
            transport: Arc::new(Mutex::new(None)),
            attach_lock: Arc::new(tokio::sync::Mutex::new(())),
            cancelled: tokio_util::sync::CancellationToken::new(),
            initialized: Arc::new(Mutex::new(false)),
            observation_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    fn activity_supported(&self) -> bool {
        self.transport.lock().unwrap().as_ref().is_some_and(|transport| {
            activity_supported(transport.is_connected(), transport.hello().as_ref())
        })
    }

    fn throughput_supported(&self) -> bool {
        self.transport.lock().unwrap().as_ref().is_some_and(|transport| {
            throughput_supported(transport.is_connected(), transport.hello().as_ref())
        })
    }

    fn queue_supported(&self) -> bool {
        self.transport.lock().unwrap().as_ref().is_some_and(|transport| {
            queue_supported(transport.is_connected(), transport.hello().as_ref())
        })
    }

    async fn refresh(&self) {
        let epoch = self.observation_epoch.load(Ordering::SeqCst);
        let mut rows: Vec<Value> = self
            .store
            .summaries()
            .await
            .into_iter()
            .map(|row| serde_json::to_value(row).expect("roster row serializes"))
            .collect();
        if epoch != self.observation_epoch.load(Ordering::SeqCst) { return; }
        project_activity(&mut rows, self.activity_supported(), epoch);
        project_throughput(&mut rows, self.throughput_supported());
        *self.summaries.lock().unwrap() = rows;
        let listeners = self.listeners.lock().unwrap().clone();
        for listener in listeners {
            listener();
        }
    }
}

impl AgentConnectionRosterStore for RosterSubscription {
    fn attach(&self, client: Arc<dyn DaemonTransportClient>) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        Box::pin(async move {
            let _guard = this.attach_lock.lock().await;
            if this.cancelled.is_cancelled() {
                return Ok(false);
            }
            let control = client.control_plane_transport();
            if !control.supports_server_capability("agent_roster") {
                return Ok(false);
            }
            let socket = control
                .hello_socket_path()
                .ok_or("Roster daemon endpoint is unavailable")?;
            let existing = this.transport.lock().unwrap().clone();
            let transport = match existing {
                Some(transport) if transport.socket_path() == socket => transport,
                Some(old) => {
                    old.close();
                    Arc::new(NativeTransport::new(&socket)) as Arc<dyn DaemonTransport>
                }
                None => Arc::new(NativeTransport::new(&socket)) as Arc<dyn DaemonTransport>,
            };
            if !transport.is_connected() {
                transport.connect(3000).await?;
            }
            transport.wait_for_hello(3000).await?;
            *this.transport.lock().unwrap() = Some(transport.clone());
            this.observation_epoch.fetch_add(1, Ordering::SeqCst);
            if !this
                .store
                .attach(RosterClient::new(transport.clone()))
                .await?
            {
                return Ok(false);
            }
            this.refresh().await;
            let start = {
                let mut initialized = this.initialized.lock().unwrap();
                let start = !*initialized;
                *initialized = true;
                start
            };
            if start {
                let store = this.clone();
                this.store
                    .on_update(Arc::new(move || {
                        let store = store.clone();
                        tokio::spawn(async move {
                            if !store.cancelled.is_cancelled() {
                                store.refresh().await;
                            }
                        });
                    }))
                    .await;
                let store = this.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            _ = store.cancelled.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                        }
                        let _guard = store.attach_lock.lock().await;
                        if store.cancelled.is_cancelled() {
                            break;
                        }
                        let transport = store.transport.lock().unwrap().clone();
                        if let Some(transport) = transport {
                            if !transport.is_connected() {
                                store.observation_epoch.fetch_add(1, Ordering::SeqCst);
                                store.refresh().await;
                                tokio::select! {
                                    _ = store.cancelled.cancelled() => break,
                                    _ = async {
                                        if transport.reconnect(3000).await.is_ok() {
                                            let _ = store.store.attach(RosterClient::new(transport)).await;
                                            store.refresh().await;
                                        }
                                    } => {}
                                }
                            }
                        }
                    }
                });
            }
            this.refresh().await;
            Ok(true)
        })
    }

    fn on_update(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.listeners.lock().unwrap().push(listener.clone());
        let listeners = self.listeners.clone();
        Box::new(move || {
            listeners
                .lock()
                .unwrap()
                .retain(|candidate| !Arc::ptr_eq(candidate, &listener));
        })
    }
    fn summaries(&self) -> Vec<Value> {
        let mut rows = self.summaries.lock().unwrap().clone();
        if !self.activity_supported() {
            project_activity(&mut rows, false, self.observation_epoch.load(Ordering::SeqCst));
        }
        project_throughput(&mut rows, self.throughput_supported());
        project_queue(&mut rows, self.queue_supported());
        rows
    }
    fn dispose(&self) -> BoxFuture<()> {
        let this = self.clone();
        this.cancelled.cancel();
        Box::pin(async move {
            let _guard = this.attach_lock.lock().await;
            this.store.dispose().await;
            let transport = this.transport.lock().unwrap().take();
            if let Some(transport) = transport {
                transport.close();
            }
            this.listeners.lock().unwrap().clear();
        })
    }
}

// Optional metadata must never turn an older daemon into a startup failure.
fn activity_supported(connected: bool, hello: Option<&crate::modes::agents_view::roster_store::DaemonHello>) -> bool {
    connected && hello.is_some_and(|hello| hello.schema_revision.is_some_and(|revision| revision >= 39)
        && hello.server_capabilities.iter().any(|capability| capability == "agent_communication_activity"))
}

fn project_activity(rows: &mut [Value], supported: bool, epoch: u64) {
    for row in rows {
        if let Some(object) = row.as_object_mut() {
            if !supported { object.remove("communicationActivity"); }
            object.insert("communicationObservationEpoch".into(), epoch.into());
        }
    }
}

fn throughput_supported(connected: bool, hello: Option<&crate::modes::agents_view::roster_store::DaemonHello>) -> bool {
    connected && hello.is_some_and(|hello| hello.schema_revision.is_some_and(|revision| revision >= 44)
        && hello.server_capabilities.iter().any(|capability| capability == "streaming_throughput"))
}

// Schema-45 queue metadata is advisory: same negotiation discipline as the
// activity field, and the row stays valid without it.
fn queue_supported(connected: bool, hello: Option<&crate::modes::agents_view::roster_store::DaemonHello>) -> bool {
    connected && hello.is_some_and(|hello| hello.schema_revision.is_some_and(|revision| revision >= 45)
        && hello.server_capabilities.iter().any(|capability| capability == "agent_communication_queue"))
}

fn project_queue(rows: &mut [Value], supported: bool) {
    for row in rows {
        if let Some(object) = row.as_object_mut() {
            if !supported { object.remove("communicationQueue"); }
        }
    }
}

fn project_throughput(rows: &mut [Value], supported: bool) {
    for row in rows {
        if let Some(row) = row.as_object_mut() {
            // Legacy request-duration totals must never become streaming-rate input.
            row.remove("generationThroughput");
            if !supported { row.remove("streamingThroughput"); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::agents_view::roster_store::DaemonHello;

    #[test]
    fn communication_queue_requires_negotiation_and_degrades_on_disconnect() {
        let mut hello = DaemonHello { schema_revision: Some(45), server_capabilities: vec!["agent_communication_queue".into()], ..Default::default() };
        assert!(queue_supported(true, Some(&hello)));
        assert!(!queue_supported(false, Some(&hello)));
        hello.schema_revision = Some(44);
        assert!(!queue_supported(true, Some(&hello)));
        hello.schema_revision = None;
        assert!(!queue_supported(true, Some(&hello)));
        hello.schema_revision = Some(45);
        hello.server_capabilities.clear();
        assert!(!queue_supported(true, Some(&hello)));
        assert!(!queue_supported(true, None));
        let mut rows = vec![serde_json::json!({
            "sessionId":"p",
            "communicationQueue":{"generation":"g","entries":[{"queueId":"q1"}]},
            "communicationActivity":{"deliveries":[]},
        })];
        // Projection is a destructive filter: the supported pass must be
        // asserted on the intact row BEFORE the unsupported pass strips
        // the field (a re-projection can never restore it).
        project_queue(&mut rows, true);
        assert!(rows[0].get("communicationQueue").is_some());
        project_queue(&mut rows, false);
        assert!(rows[0].get("communicationQueue").is_none());
        // The activity field keeps its own, independent gate.
        assert!(rows[0].get("communicationActivity").is_some());
    }

    #[test]
    fn streaming_throughput_requires_negotiation_and_degrades_on_disconnect() {
        let mut hello = DaemonHello { schema_revision: Some(44), server_capabilities: vec!["streaming_throughput".into()], ..Default::default() };
        assert!(throughput_supported(true, Some(&hello)));
        assert!(!throughput_supported(false, Some(&hello)));
        hello.schema_revision = Some(43);
        assert!(!throughput_supported(true, Some(&hello)));
        hello.schema_revision = None;
        assert!(!throughput_supported(true, Some(&hello)));
        hello.schema_revision = Some(44);
        hello.server_capabilities.clear();
        assert!(!throughput_supported(true, Some(&hello)));
        assert!(!throughput_supported(true, None));
        let mut rows = vec![serde_json::json!({
            "sessionId":"p", "streamingThroughput":{"streamingMs":1000, "outputTokens":100, "responses":1},
            "generationThroughput":{"generationMs":2000, "outputTokens":100, "responses":1}
        })];
        project_throughput(&mut rows, true);
        assert!(rows[0].get("streamingThroughput").is_some());
        assert!(rows[0].get("generationThroughput").is_none());
        project_throughput(&mut rows, false);
        assert!(rows[0].get("streamingThroughput").is_none());
        assert_eq!(rows[0]["sessionId"], "p");
    }

    #[test]
    fn new_client_never_projects_legacy_throughput_as_streaming() {
        for revision in [None, Some(42), Some(43), Some(44)] {
            let hello = DaemonHello { schema_revision: revision, server_capabilities: vec!["generation_throughput".into()], ..Default::default() };
            let supported = throughput_supported(true, Some(&hello));
            assert!(!supported);
            let mut rows = vec![serde_json::json!({
                "sessionId":"old", "generationThroughput":{"generationMs":2000, "responses":1},
                "streamingThroughput":{"streamingMs":1000, "responses":1}
            })];
            project_throughput(&mut rows, supported);
            assert_eq!(rows, vec![serde_json::json!({"sessionId":"old"})]);
        }
        let mut rows = vec![serde_json::json!({"sessionId":"old", "generationThroughput":{"generationMs":2000}})];
        project_throughput(&mut rows, true);
        assert_eq!(rows, vec![serde_json::json!({"sessionId":"old"})]);
    }

    #[test]
    fn communication_metadata_degrades_with_old_daemons_missing_capabilities_and_disconnects() {
        let mut hello = DaemonHello { schema_revision: Some(39), server_capabilities: vec!["agent_communication_activity".into()], ..Default::default() };
        assert!(activity_supported(true, Some(&hello)));
        assert!(!activity_supported(false, Some(&hello)));
        hello.schema_revision = Some(38);
        assert!(!activity_supported(true, Some(&hello)));
        hello.schema_revision = Some(39);
        hello.server_capabilities.clear();
        assert!(!activity_supported(true, Some(&hello)));
        assert!(!activity_supported(true, None));
        let mut rows = vec![serde_json::json!({"sessionId":"p", "communicationActivity":{"deliveries":[]}})];
        project_activity(&mut rows, true, 1);
        assert!(rows[0].get("communicationActivity").is_some());
        assert_eq!(rows[0]["communicationObservationEpoch"], 1);
        project_activity(&mut rows, false, 2);
        assert!(rows[0].get("communicationActivity").is_none());
        assert_eq!(rows[0]["sessionId"], "p");
        assert_eq!(rows[0]["communicationObservationEpoch"], 2);
    }
}
