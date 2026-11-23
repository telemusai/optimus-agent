//! Ordered delivery for the native terminal. The command reader must remain free to
//! receive RPC replies even when the terminal is waiting for one with a full UI queue.
use super::*;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use tokio::sync::Notify;

const MEMORY_FRAMES: usize = 128;
const MEMORY_BYTES: usize = 4 * 1024 * 1024;
const SEGMENT_BYTES: u64 = 4 * 1024 * 1024;
const SPILL_BYTES: u64 = 128 * 1024 * 1024;

#[derive(PartialEq)]
enum UpdateKey {
    Heartbeat,
    Assistant {
        session: String,
        generation: String,
        timestamp: u64,
        sequence: i64,
    },
}

impl UpdateKey {
    fn from_frame(frame: &Value) -> Option<Self> {
        if frame["type"] == "heartbeats_changed" {
            return Some(Self::Heartbeat);
        }
        let event = &frame["event"];
        if frame["type"] != "session_event"
            || event["type"] != "message_update"
            || event["message"]["role"] != "assistant"
        {
            return None;
        }
        let cursor = &frame["meta"]["cursor"];
        Some(Self::Assistant {
            session: frame["activeSessionId"].as_str()?.into(),
            generation: cursor["generation"].as_str()?.into(),
            timestamp: event["message"]["timestamp"].as_u64()?,
            sequence: cursor["sequence"].as_i64()?,
        })
    }

    fn supersedes(&self, previous: &Self) -> bool {
        match (self, previous) {
            (Self::Heartbeat, Self::Heartbeat) => true,
            (
                Self::Assistant {
                    session,
                    generation,
                    timestamp,
                    sequence,
                },
                Self::Assistant {
                    session: old_session,
                    generation: old_gen,
                    timestamp: old_time,
                    sequence: old_seq,
                },
            ) => {
                session == old_session
                    && generation == old_gen
                    && timestamp == old_time
                    && old_seq.checked_add(1) == Some(*sequence)
            }
            _ => false,
        }
    }
}

struct Frame {
    bytes: Vec<u8>,
    key: Option<UpdateKey>,
}

// Anonymous files are removed on close, including process exit. Rotating segments
// release consumed disk space even if a slow viewer never completely catches up.
struct Segment {
    file: File,
    read_at: u64,
    written: u64,
}

#[derive(Default)]
struct Inbox {
    memory: VecDeque<Vec<u8>>,
    memory_bytes: usize,
    segments: VecDeque<Segment>,
    disk_bytes: u64,
    tail: Option<Frame>,
    failure: Option<String>,
}

impl Inbox {
    fn push(&mut self, bytes: &[u8], key: Option<UpdateKey>) -> io::Result<()> {
        if let Some(mut old) = self.tail.take() {
            if key
                .as_ref()
                .zip(old.key.as_ref())
                .is_some_and(|(new, old)| new.supersedes(old))
            {
                if bytes.len() <= MEMORY_BYTES {
                    old.bytes.clear();
                    if bytes.len() > old.bytes.capacity() && old.bytes.capacity() > MEMORY_BYTES / 2
                    {
                        old.bytes.reserve_exact(bytes.len());
                    }
                    old.bytes.extend_from_slice(bytes);
                    old.key = key;
                    self.tail = Some(old);
                    return Ok(());
                }
            } else if self.segments.is_empty()
                && self.memory.len() < MEMORY_FRAMES
                && self.memory_bytes + old.bytes.capacity() <= MEMORY_BYTES
            {
                self.memory_bytes += old.bytes.capacity();
                self.memory.push_back(old.bytes);
            } else {
                self.spill(&old.bytes)?;
            }
        }
        if bytes.len() > MEMORY_BYTES {
            return self.spill(bytes);
        }
        self.tail = Some(Frame {
            bytes: bytes.into(),
            key,
        });
        Ok(())
    }

    fn spill(&mut self, bytes: &[u8]) -> io::Result<()> {
        let length = bytes.len() as u64;
        let size = length + 8;
        if self.disk_bytes + size > SPILL_BYTES {
            return Err(io::Error::other(
                "pending viewer updates exceed 128 MiB of temporary storage",
            ));
        }
        if self
            .segments
            .back()
            .is_none_or(|s| s.written + size > SEGMENT_BYTES)
        {
            self.segments.push_back(Segment {
                file: tempfile::tempfile()?,
                read_at: 0,
                written: 0,
            });
        }
        let segment = self.segments.back_mut().unwrap();
        segment.file.seek(SeekFrom::Start(segment.written))?;
        segment.file.write_all(&length.to_le_bytes())?;
        segment.file.write_all(bytes)?;
        segment.written += size;
        self.disk_bytes += size;
        Ok(())
    }

    fn pop(&mut self) -> io::Result<Option<Vec<u8>>> {
        if let Some(bytes) = self.memory.pop_front() {
            self.memory_bytes -= bytes.capacity();
            return Ok(Some(bytes));
        }
        if let Some(segment) = self.segments.front_mut() {
            segment.file.seek(SeekFrom::Start(segment.read_at))?;
            let mut header = [0; 8];
            segment.file.read_exact(&mut header)?;
            let length = u64::from_le_bytes(header);
            if length > segment.written - segment.read_at - 8 {
                return Err(io::Error::other("invalid pending viewer frame length"));
            }
            let mut bytes = vec![0; length as usize];
            segment.file.read_exact(&mut bytes)?;
            segment.read_at += length + 8;
            if segment.read_at == segment.written {
                self.disk_bytes -= segment.written;
                self.segments.pop_front();
            }
            return Ok(Some(bytes));
        }
        Ok(self.tail.take().map(|tail| tail.bytes))
    }

    fn fail(&mut self, error: io::Error) {
        *self = Self {
            failure: Some(format!("The terminal could not buffer live chat updates: {error}. Reopen this chat to load the saved conversation.")),
            ..Self::default()
        };
    }
}

fn for_viewer(frame: &Value, active: &str) -> bool {
    match frame["type"].as_str() {
        Some("heartbeats_changed" | "daemon_closing") => true,
        Some(
            "session_event"
            | "side_question_event"
            | "session_status"
            | "session_resynced"
            | "session_replaced"
            | "session_closed"
            | "extension_ui_request"
            | "extension_error"
            | "session_snapshot_begin"
            | "session_snapshot_chunk"
            | "session_snapshot_end"
            | "session_snapshot_failed",
        ) => frame["activeSessionId"].as_str() == Some(active),
        _ => false,
    }
}

pub(super) fn bind(connection: &Arc<DaemonAgentConnection>) -> Option<Box<dyn Fn() + Send + Sync>> {
    let inbox = Arc::new(Mutex::new(Inbox::default()));
    let ready = Arc::new(Notify::new());
    let weak = Arc::downgrade(connection);
    let incoming = inbox.clone();
    let wake = ready.clone();
    let unsubscribe = connection
        .client
        .on_viewer_frame(Arc::new(move |frame, line| {
            let Some(connection) = weak.upgrade() else {
                return;
            };
            if !for_viewer(frame, &connection.active_session_id.lock().unwrap()) {
                return;
            }
            if frame["type"] == "daemon_closing" {
                if let Some(DaemonOutbound::DaemonClosing { reason }) =
                    crate::main_entry::main_entry_outbound_from_owned_wire(frame.clone())
                {
                    connection.observe_daemon_closing(&reason);
                }
                return;
            }
            let mut inbox = incoming.lock().unwrap();
            if inbox.failure.is_none() {
                if let Err(error) = inbox.push(line.as_bytes(), UpdateKey::from_frame(frame)) {
                    inbox.fail(error);
                }
            }
            wake.notify_one();
        }))?;
    let worker = tokio::spawn(run_viewer(Arc::downgrade(connection), inbox, ready));
    Some(Box::new(move || {
        unsubscribe();
        worker.abort();
    }))
}

fn needs_reader(connection: &DaemonAgentConnection, message: &DaemonOutbound) -> bool {
    match message {
        DaemonOutbound::SessionClosed { reason, .. } => {
            matches!(reason.as_str(), "shutdown" | "killed")
                && *connection.shutdown_restart_pending.lock().unwrap()
        }
        DaemonOutbound::SessionSnapshotFailed { snapshot_id, .. } => {
            let assembly = connection
                .snapshot_assemblies
                .lock()
                .unwrap()
                .get(snapshot_id)
                .cloned();
            assembly.is_some_and(|a| {
                a.lock()
                    .unwrap()
                    .begin
                    .as_ref()
                    .is_some_and(|begin| matches!(begin.purpose.as_str(), "replacement" | "resync"))
            })
        }
        _ => false,
    }
}

async fn run_viewer(
    weak: Weak<DaemonAgentConnection>,
    inbox: Arc<Mutex<Inbox>>,
    ready: Arc<Notify>,
) {
    let mut recoveries = tokio::task::JoinSet::new();
    loop {
        let next = {
            let mut pending = inbox.lock().unwrap();
            if pending.failure.is_none() {
                match pending.pop() {
                    Ok(bytes) => Ok(bytes),
                    Err(error) => {
                        pending.fail(error);
                        Err(pending.failure.clone().unwrap())
                    }
                }
            } else {
                Err(pending.failure.clone().unwrap())
            }
        };
        let Some(connection) = weak.upgrade() else {
            break;
        };
        if *connection.disposed.lock().unwrap() {
            break;
        }
        let bytes = match next {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                drop(connection);
                ready.notified().await;
                continue;
            }
            Err(error) => {
                connection.deferred_session_events.lock().unwrap().failure = Some(error.clone());
                *connection.terminal_close_emitted.lock().unwrap() = true;
                append_rotating_log(
                    &get_agent_log_path(),
                    &format!("[{}] viewer delivery: {error}", now_iso()),
                );
                connection
                    .emit(AgentConnectionEvent::Closed { error: Some(error) })
                    .await;
                break;
            }
        };
        tokio::task::consume_budget().await;
        let frame = match serde_json::from_slice(&bytes) {
            Ok(frame) => frame,
            Err(error) => {
                inbox
                    .lock()
                    .unwrap()
                    .fail(io::Error::new(io::ErrorKind::InvalidData, error));
                continue;
            }
        };
        let Some(message) = crate::main_entry::main_entry_outbound_from_owned_wire(frame) else {
            continue;
        };
        drop(bytes);
        // A restart or failed snapshot may await frames from this same socket.
        // Poll its state transition in order, then let only that recovery run independently.
        if needs_reader(&connection, &message) {
            while recoveries.try_join_next().is_some() {}
            if recoveries.len() >= 8 {
                inbox.lock().unwrap().fail(io::Error::other(
                    "too many simultaneous snapshot or restart recoveries",
                ));
                continue;
            }
            let mut delivery = Box::pin(deliver(connection, message));
            let first = futures::future::poll_fn(|cx| {
                std::task::Poll::Ready(std::future::Future::poll(delivery.as_mut(), cx))
            })
            .await;
            if first.is_pending() {
                recoveries.spawn(delivery);
            }
        } else {
            deliver(connection, message).await;
        }
    }
}

async fn deliver(connection: Arc<DaemonAgentConnection>, message: DaemonOutbound) {
    if let Err(error) = connection.handle_daemon_message(message).await {
        append_rotating_log(
            &get_agent_log_path(),
            &format!("[{}] daemon-message: ignored failure: {error}", now_iso()),
        );
    }
}

#[cfg(test)]
#[path = "viewer_delivery_tests.rs"]
mod tests;
