//! Native socket adapter for daemon-mode.ts's createServer and handleConnection.

use super::*;
use futures::Future;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio_util::sync::CancellationToken;
use crate::modes::rpc::jsonl::{JsonlLineReaderOptions, StringDecoder};

pub(super) async fn bind(daemon: &Arc<AgentDaemon>) -> Result<(), String> {
    #[cfg(unix)]
    {
        let listener = tokio::net::UnixListener::bind(&daemon.socket_path).map_err(|error| error.to_string())?;
        let daemon = daemon.clone();
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    _ = daemon.server_stopped.cancelled() => break,
                    result = listener.accept() => result,
                };
                match accepted {
                    Ok((socket, _)) => { spawn_connection(daemon.clone(), socket); }
                    Err(error) => { daemon.log(&format!("Daemon socket accept failed: {error}")); daemon.shutdown(1, None).await; break; }
                }
            }
        });
        Ok(())
    }
    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ServerOptions;
        let mut server = ServerOptions::new().first_pipe_instance(true).create(&daemon.socket_path).map_err(|error| error.to_string())?;
        let daemon = daemon.clone();
        tokio::spawn(async move {
            loop {
                let connected = tokio::select! {
                    _ = daemon.server_stopped.cancelled() => break,
                    result = server.connect() => result,
                };
                if let Err(error) = connected { daemon.log(&format!("Daemon pipe accept failed: {error}")); daemon.shutdown(1, None).await; break; }
                let next = match ServerOptions::new().create(&daemon.socket_path) {
                    Ok(next) => next,
                    Err(error) => { daemon.log(&format!("Daemon pipe creation failed: {error}")); daemon.shutdown(1, None).await; break; }
                };
                spawn_connection(daemon.clone(), server);
                server = next;
            }
        });
        Ok(())
    }
}

fn spawn_connection<S>(daemon: Arc<AgentDaemon>, socket: S)
where S: AsyncRead + AsyncWrite + Unpin + Send + 'static {
    tokio::spawn(async move {
        if let Err(error) = handle_connection(daemon.clone(), socket).await {
            daemon.log(&format!("Daemon client connection failed: {error}"));
        }
    });
}

async fn handle_connection<S>(daemon: Arc<AgentDaemon>, socket: S) -> Result<(), String>
where S: AsyncRead + AsyncWrite + Unpin + Send + 'static {
    let (mut input, mut output) = tokio::io::split(socket);
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let writer = Arc::new(DaemonClientWriter::new(sender));
    let cancelled = CancellationToken::new();
    let detach = cancelled.clone();
    let client = Arc::new(DaemonClientHandle::new(create_active_session_id(None), writer.clone(), Arc::new(move || detach.cancel())));
    {
        let mut state = client.state.lock().expect("daemon client poisoned");
        state.authenticated = Some(!daemon.is_worker());
        state.transport = Some(if daemon.is_worker() { "private-framed" } else { "jsonl" }.to_string());
        state.backpressured = Some(false);
        state.catchup_active_session_ids = Some(HashSet::new());
        state.capabilities = normalize_client_capabilities(None, None);
    }
    daemon.clients.lock().expect("clients poisoned").push(client.clone());
    daemon.write(&client, &DaemonOutbound::Raw(serde_json::json!({
        "type": "daemon_hello",
        "socketPath": daemon.socket_path,
        "protocol": crate::modes::daemon::daemon_protocol::daemon_protocol_info(),
        "schemaId": DAEMON_SCHEMA_ID,
        "schemaRevision": DAEMON_SCHEMA_REVISION,
        "appVersion": crate::config::VERSION,
        "runtime": crate::modes::daemon::daemon_runtime_identity::get_daemon_runtime_identity_from_process(),
        "clientId": client.id(),
        // A fixed 33-element array exceeds serde's const-generic array
        // Serialize impls (32 max), so the wire serializes the same
        // capability list as a slice. The capability set itself is
        // unchanged (DAEMON_DEFAULT_SERVER_CAPABILITIES).
        "serverCapabilities": DAEMON_DEFAULT_SERVER_CAPABILITIES.as_slice(),
    })));
    let write_cancelled = cancelled.clone();
    let write_writer = writer.clone();
    let write_task = tokio::spawn(async move {
        let _completion = WriterCompletion(write_writer.clone());
        loop {
            if write_writer.destroyed() {
                while let Ok(bytes) = receiver.try_recv() { output.write_all(&bytes).await?; }
                break;
            }
            tokio::select! {
                _ = write_cancelled.cancelled() => break,
                _ = write_writer.ended.cancelled() => continue,
                bytes = receiver.recv() => match bytes {
                    Some(bytes) => output.write_all(&bytes).await?,
                    None => break,
                },
            }
        }
        output.shutdown().await
    });
    let dispatch_daemon = daemon.clone();
    let dispatch_client = client.clone();
    let dispatch = Arc::new(move |line: String| dispatch_line(dispatch_daemon.clone(), dispatch_client.clone(), line));
    let mut lines = JsonlLineReader::new(dispatch.clone(), JsonlLineReaderOptions::default());
    let mut decoder = StringDecoder::new();
    let mut frames = PrivateFrameDecoder::new();
    let private_framed = daemon.is_worker();
    let mut buffer = vec![0; 8192];
    let result = loop {
        let read = tokio::select! {
            _ = cancelled.cancelled() => break Ok(()),
            _ = writer.ended.cancelled() => break Ok(()),
            read = input.read(&mut buffer) => read,
        };
        match read {
            Ok(0) => {
                if private_framed { break frames.finish(); }
                lines.push(&decoder.end());
                lines.end();
                break Ok(());
            }
            Ok(size) if private_framed => match frames.push(&buffer[..size]) {
                Ok(frames) => for frame in frames {
                    if frame.header.get("kind").and_then(Value::as_str) == Some("command") {
                        dispatch(String::from_utf8_lossy(&frame.payload).into_owned());
                    }
                },
                Err(error) => break Err(error),
            },
            Ok(size) => lines.push(&decoder.write(&buffer[..size])),
            Err(error) => break Err(error.to_string()),
        }
    };
    writer.end();
    // Flush replies queued before end(), as Node socket.end() does.
    let _ = write_task.await;
    cleanup(&daemon, &client);
    result
}

struct WriterCompletion(Arc<DaemonClientWriter>);

impl Drop for WriterCompletion {
    fn drop(&mut self) {
        self.0.end();
        self.0.drained.cancel();
    }
}

fn dispatch_line(daemon: Arc<AgentDaemon>, client: Arc<DaemonClientHandle>, line: String) {
    let mut future = Box::pin(async move { daemon.handle_line(client, line).await });
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    // JavaScript runs the admission-registration prologue before yielding its promise.
    if future.as_mut().poll(&mut context).is_pending() { tokio::spawn(future); }
}

fn cleanup(daemon: &Arc<AgentDaemon>, client: &Arc<DaemonClientHandle>) {
    daemon.clear_client_catchup_retry(client);
    daemon.release_session_input_pauses_for(client, None);
    daemon.detach_client(client);
    (client.detach_input)();
    daemon.clients.lock().expect("clients poisoned").retain(|current| !Arc::ptr_eq(current, client));
    let key = Arc::as_ptr(client) as usize;
    daemon.peer_claims.lock().expect("peer claims poisoned").remove(&key);
    let was_supervisor = client.state.lock().expect("daemon client poisoned").authentication_role.as_deref() == Some("supervisor");
    daemon.revoke_supervisor_claim(key, None);
    if daemon.is_worker() && was_supervisor {
        if let Some(socket_path) = daemon.supervisor_socket_path_from_env() {
            daemon.schedule_supervisor_availability_check(socket_path, 100);
        }
    }
}

pub(super) fn register_signal_handlers(daemon: &Arc<AgentDaemon>) {
    #[cfg(unix)]
    for (kind, code) in [(tokio::signal::unix::SignalKind::interrupt(), 130), (tokio::signal::unix::SignalKind::terminate(), 143), (tokio::signal::unix::SignalKind::hangup(), 129)] {
        match tokio::signal::unix::signal(kind) {
            Ok(mut signal) => {
                let daemon = daemon.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = daemon.server_stopped.cancelled() => {},
                        value = signal.recv() => { if value.is_some() { daemon.shutdown(code, None).await; } }
                    }
                });
            }
            Err(error) => daemon.log(&format!("Could not install daemon signal handler: {error}")),
        }
    }
    #[cfg(windows)]
    {
        let daemon = daemon.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = daemon.server_stopped.cancelled() => {},
                result = tokio::signal::ctrl_c() => { if result.is_ok() { daemon.shutdown(130, None).await; } }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daemon(directory: &std::path::Path, worker: bool) -> Arc<AgentDaemon> {
        let path = directory.join("daemon.sock").to_string_lossy().into_owned();
        AgentDaemon::new(path.clone(), DaemonModeOptions {
            socket_path: Some(path),
            default_session_config: AgentSessionRuntimeConfig {
                cwd: Some(directory.to_string_lossy().into_owned()),
                agent_dir: Some(directory.join("agent").to_string_lossy().into_owned()),
                ..Default::default()
            },
            create_runtime: Arc::new(|_| Box::pin(async { panic!("transport test must not create a session") })),
            worker: worker.then(|| DaemonWorkerOptions { authentication_token: "test-token".to_string(), ..Default::default() }),
        })
    }

    #[tokio::test]
    async fn public_socket_serves_handshake_and_commands() {
        let directory = tempfile::tempdir().unwrap();
        let daemon = daemon(directory.path(), false);
        let (socket, peer) = tokio::io::duplex(65536);
        let task = tokio::spawn(handle_connection(daemon.clone(), socket));
        let (input, mut output) = tokio::io::split(peer);
        let mut lines = tokio::io::BufReader::new(input);
        let mut line = String::new();
        tokio::io::AsyncBufReadExt::read_line(&mut lines, &mut line).await.unwrap();
        let hello: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(hello["type"], "daemon_hello");
        assert_eq!(hello["schemaRevision"], 44);
        let capabilities = hello["serverCapabilities"].as_array().unwrap();
        assert!(capabilities.iter().any(|capability| capability == "streaming_throughput"));
        assert!(!capabilities.iter().any(|capability| capability == "generation_throughput"));
        output.write_all(b"{\"type\":\"does_not_exist\",\"id\":\"request-1\"}\n").await.unwrap();
        line.clear();
        tokio::io::AsyncBufReadExt::read_line(&mut lines, &mut line).await.unwrap();
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], "request-1");
        assert_eq!(response["success"], false);
        assert_eq!(response["error"], "Unknown daemon command: does_not_exist");
        output.shutdown().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap().unwrap();
        assert!(daemon.clients.lock().unwrap().is_empty());
    }

    #[test]
    fn sequenced_worker_events_keep_nested_metadata_and_routing_headers() {
        let directory = tempfile::tempdir().unwrap();
        let daemon = daemon(directory.path(), true);
        let state = Arc::new(StdMutex::new(ActiveSessionState::new("active-1", Default::default())));
        let event = daemon.add_session_event_meta(&state, DaemonOutbound::SessionEvent { active_session_id: "active-1".to_string(), event: serde_json::json!({"type":"agent_end"}) });
        let value = event.to_value();
        assert_eq!(value["meta"]["sequence"], 1);
        assert!(value.get("sequence").is_none());
        assert_eq!(event.active_session_id(), Some("active-1"));
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let client = Arc::new(DaemonClientHandle::new("client-1".to_string(), Arc::new(DaemonClientWriter::new(sender)), Arc::new(|| {})));
        client.state.lock().unwrap().transport = Some("private-framed".to_string());
        assert!(daemon.write(&client, &event));
        let bytes = receiver.try_recv().unwrap();
        let frame = PrivateFrameDecoder::new().push(&bytes).unwrap().remove(0);
        assert_eq!(frame.header["activeSessionId"], "active-1");
        assert_eq!(frame.header["sessionEventType"], "agent_end");
        assert!(frame.header.get("requestId").is_none());
        assert_eq!(serde_json::from_slice::<Value>(&frame.payload).unwrap(), value);
    }

    #[tokio::test]
    async fn private_socket_uses_binary_frames_for_hello_and_auth_errors() {
        let directory = tempfile::tempdir().unwrap();
        let daemon = daemon(directory.path(), true);
        let (socket, mut peer) = tokio::io::duplex(65536);
        let task = tokio::spawn(handle_connection(daemon.clone(), socket));
        let mut decoder = PrivateFrameDecoder::new();
        let mut buffer = [0; 8192];
        let hello = loop {
            let size = peer.read(&mut buffer).await.unwrap();
            let frames = decoder.push(&buffer[..size]).unwrap();
            if let Some(frame) = frames.into_iter().next() { break frame; }
        };
        assert_eq!(hello.header["kind"], "outbound");
        let hello: Value = serde_json::from_slice(&hello.payload).unwrap();
        assert_eq!(hello["type"], "daemon_hello");
        let capabilities = hello["serverCapabilities"].as_array().unwrap();
        assert!(capabilities.iter().any(|capability| capability == "streaming_throughput"));
        assert!(!capabilities.iter().any(|capability| capability == "generation_throughput"));
        let command = encode_private_frame(&serde_json::json!({"kind":"command","requestId":"auth-1","commandType":"worker_auth"}), b"{\"type\":\"worker_auth\",\"id\":\"auth-1\",\"token\":\"wrong\"}").unwrap();
        peer.write_all(&command).await.unwrap();
        let response = loop {
            let size = peer.read(&mut buffer).await.unwrap();
            assert_ne!(size, 0);
            let frames = decoder.push(&buffer[..size]).unwrap();
            if let Some(frame) = frames.into_iter().next() { break frame; }
        };
        let response: Value = serde_json::from_slice(&response.payload).unwrap();
        assert_eq!(response["id"], "auth-1");
        assert_eq!(response["success"], false);
        tokio::time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap().unwrap();
    }
}
