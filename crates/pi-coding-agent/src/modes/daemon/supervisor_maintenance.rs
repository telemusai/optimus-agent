//! Recovery cleanup and existing optional heartbeat/snapshot contracts.
use super::*;
use std::io::Read;
use crate::core::cron_jobs::{AgentCronJobStore, CancelJobsForSessionInput, SESSION_SCHEDULED_JOBS_FILENAME};
use crate::core::orphan_process_journal::{clear_orphan_process_journal, is_orphan_process_identity_current, kill_orphan_process, read_active_orphan_processes, should_reap_orphan_process, ActiveOrphanProcess, OrphanProcessRecord};
use crate::modes::daemon::snapshot_transcript_cache::{SnapshotTranscriptCache, SnapshotTranscriptCacheOptions, SNAPSHOT_TARGET_CHUNK_BYTES};
use crate::modes::daemon::worker_recovery_journal::{WorkerRecoveryJournal, WorkerRecoveryRecordInput};

#[derive(Default)]
pub(super) struct HeartbeatSnapshot {
    pub epoch: u64,
    rows: Option<Vec<Value>>,
    stale: bool,
}

impl HeartbeatSnapshot {
    pub fn invalidate(&mut self) { self.epoch = self.epoch.wrapping_add(1); self.stale = true; }
    pub fn store_if_current(&mut self, epoch: u64, rows: Vec<Value>) {
        if self.epoch == epoch { self.rows = Some(rows); self.stale = false; }
    }
    pub fn fresh(&self) -> Option<Vec<Value>> { (!self.stale).then(|| self.rows.clone()).flatten() }
}

/// Durable evidence for unresolved ownership. Capacity is a refusal boundary,
/// never permission to discard an older unresolved process record.
const ORPHANS_UNREAPABLE_MAX_RECORDS: usize = 64;
const ORPHANS_UNREAPABLE_MAX_BYTES: u64 = 256 * 1024;
const DEFERRED_ORPHAN_PERSISTENCE_ERROR: &str = "Deferred orphan evidence could not be persisted; recovery journal retained";

fn record_deferred_orphans(descriptor_dir: &Path, worker_id: &str, orphans: &[ActiveOrphanProcess]) -> Result<(), String> {
    let path = descriptor_dir.join(format!("{worker_id}.orphans.unreapable.jsonl"));
    let mut raw = Vec::new();
    match std::fs::File::open(&path) {
        Ok(file) => {
            file.take(ORPHANS_UNREAPABLE_MAX_BYTES + 1).read_to_end(&mut raw)
                .map_err(|error| format!("could not read {}: {error}", path.display()))?;
            if raw.len() as u64 > ORPHANS_UNREAPABLE_MAX_BYTES {
                return Err("deferred orphan evidence exceeds its byte limit".into());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    }
    let contents = String::from_utf8(raw).map_err(|_| "deferred orphan evidence is not UTF-8".to_string())?;
    let mut prior: Vec<Value> = Vec::new();
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line).map_err(|_| "deferred orphan evidence contains malformed JSON".to_string())?;
        if record.get("workerId").and_then(Value::as_str) != Some(worker_id)
            || !record.get("pid").and_then(Value::as_i64).is_some_and(|pid| pid > 0) {
            return Err("deferred orphan evidence has an unverified owner or pid".into());
        }
        prior.push(record);
        if prior.len() > ORPHANS_UNREAPABLE_MAX_RECORDS {
            return Err("deferred orphan evidence exceeds its record limit".into());
        }
    }
    let stamp = iso_from_ms(supervisor_now_ms() as f64);
    for orphan in orphans {
        let record = json!({
            "at": stamp,
            "workerId": worker_id,
            "pid": orphan.pid,
            "kernelPid": orphan.kernel_pid,
            "processStartId": orphan.process_start_id,
            "reason": "cleanup could not prove the live pid is the journaled process; no bare-pid kill was attempted and cleanup remains unverified",
        });
        // A crash after side-record persistence but before journal removal must
        // be safe to retry without consuming capacity a second time.
        if prior.iter().any(|existing| existing.get("pid") == record.get("pid")
            && existing.get("kernelPid") == record.get("kernelPid")
            && existing.get("processStartId") == record.get("processStartId")) { continue; }
        if prior.len() >= ORPHANS_UNREAPABLE_MAX_RECORDS {
            return Err("deferred orphan evidence exceeds its record limit".into());
        }
        prior.push(record);
    }
    let payload = prior.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?.join("\n");
    let payload = format!("{payload}\n");
    if payload.len() as u64 > ORPHANS_UNREAPABLE_MAX_BYTES {
        return Err("deferred orphan evidence exceeds its byte limit".into());
    }
    write_file_atomic_sync(&path.to_string_lossy(), &payload, WriteFileAtomicOptions {
        mode: Some(0o600), fsync: true, fsync_dir: true, ..Default::default()
    }).map_err(|error| format!("could not persist {}: {error}", path.display()))?;
    eprintln!("[{stamp}] Session worker {worker_id} deferred unproven orphan cleanup for {} record(s) {orphans:?}; recorded in {} (the journal is superseded; the unproven pids stay observable)", orphans.len(), path.display());
    Ok(())
}

fn same_registration(left: &DaemonWorkerDescriptor, right: &DaemonWorkerDescriptor) -> bool {
    left.worker_id == right.worker_id && left.pid == right.pid
        && left.process_start_id == right.process_start_id
        && left.worker_instance_id == right.worker_instance_id
        && left.authentication_token == right.authentication_token
        && left.stop_requested_at == right.stop_requested_at
        && left.owner_client_id == right.owner_client_id
}

pub(super) fn permanent_stop_cleanup_error(error: &str) -> bool {
    matches!(error,
        "Uncertain operation has no saved transcript; recovery journal retained"
        | "Malformed orphan record; recovery journal retained"
        | "Unverified orphan record; recovery journal retained"
        | DEFERRED_ORPHAN_PERSISTENCE_ERROR)
}

impl Supervisor {
    pub(super) fn invalidate_worker_input_pauses(&self, worker: &Arc<Worker>) {
        let mut owners = HashSet::new();
        self.pauses.lock().unwrap().retain(|_, pause| {
            if Arc::ptr_eq(&pause.worker, worker) { owners.insert(pause.connection_id.clone()); false } else { true }
        });
        for owner in owners {
            if let Some(client) = self.clients.lock().unwrap().get(&owner) {
                client.pause_epoch.fetch_add(1, Ordering::SeqCst);
                // Existing clients invalidate their local pause fence on transport close.
                client.stopped.cancel();
            }
        }
    }

    pub(super) fn broadcast_heartbeats_changed(self: &Arc<Self>) {
        self.schedule_scheduled_session_wake_recompute();
        for client in self.clients.lock().unwrap().values() {
            // This is the existing heartbeat_catalog outbound (no new wire shape).
            client.write(&json!({"type":"heartbeats_changed"}));
        }
    }

    async fn assert_dead_registration(&self, worker: &Arc<Worker>, expected: &DaemonWorkerDescriptor) -> Result<(), String> {
        self.ownership.assert_current().await.map_err(|error| error.to_string())?;
        let current = worker.descriptor.lock().unwrap().clone();
        if !same_registration(&current, expected)
            || !self.workers.lock().unwrap().get(&expected.worker_id).is_some_and(|registered| Arc::ptr_eq(registered, worker)) {
            return Err("Worker registration changed during recovery cleanup".into());
        }
        if is_stopping_process_alive(&ProcessIdentity { pid: expected.pid as i64, process_start_id: expected.process_start_id.clone() }) {
            return Err("Cannot recover uncertain operations of a live or unverified worker".into());
        }
        Ok(())
    }

    pub(super) async fn recover_uncertain_worker_operations(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<(), String> {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        self.assert_dead_registration(worker, &descriptor).await?;
        let mut journal = WorkerRecoveryJournal::new(&descriptor.recovery_journal_path);
        let latest = journal.get_latest();
        for record in latest.iter().filter(|record| record.busy) {
            let file = record.session_file.as_ref().or_else(||
                (record.active_session_id == descriptor.root_active_session_id).then_some(descriptor.session_file.as_ref()).flatten());
            let Some(file) = file else { return Err("Uncertain operation has no saved transcript; recovery journal retained".into()); };
            self.assert_dead_registration(worker, &descriptor).await?;
            self.catalog.mark_interrupted(file, &record.active_session_id, &[record.operation.clone()]).await?;
        }
        self.assert_dead_registration(worker, &descriptor).await?;
        if let Some(path) = &descriptor.orphan_process_journal_path {
            // The permissive journal reader tolerates crash-truncated appends. That
            // is useful for inspection, but not evidence that the entire journal
            // can be erased. Preserve malformed/unknown records for later repair.
            match std::fs::read_to_string(path) {
                Ok(contents) => for line in contents.lines().filter(|line| !line.trim().is_empty()) {
                    let record: OrphanProcessRecord = serde_json::from_str(line)
                        .map_err(|_| "Malformed orphan record; recovery journal retained".to_string())?;
                    if record.version != 1 || record.pid <= 0 || record.owner_pid != descriptor.pid as i64 {
                        return Err("Unverified orphan record; recovery journal retained".into());
                    }
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(error.to_string()),
            }
            let orphans = read_active_orphan_processes(path, descriptor.pid as i64).map_err(|error| error.to_string())?;
            let mut failed = false;
            let mut deferred = Vec::new();
            for orphan in orphans {
                self.assert_dead_registration(worker, &descriptor).await?;
                let identity = ProcessIdentity { pid: orphan.pid, process_start_id: orphan.process_start_id.clone() };
                if should_reap_orphan_process(&orphan) {
                    // Only a provably live record needs a kill; a gone pid is
                    // nothing to reap, not a failed cleanup.
                    if is_stopping_process_alive(&identity) {
                        let killed = tokio::time::timeout(Duration::from_secs(6), tokio::task::spawn_blocking(move || {
                            kill_orphan_process(orphan.pid)
                        })).await;
                        if !matches!(killed, Ok(Ok(true))) { failed = true; }
                    }
                    continue;
                }
                // The reaper intentionally declines this record. A recorded start
                // id that provably no longer matches the pid proves the journaled
                // process is gone (or the pid moved on), so there is nothing left
                // to reap; a start id that cannot be observed at all is unknown,
                // like a pid-only record.
                if orphan.process_start_id.as_deref().is_some_and(|start| !start.is_empty())
                    && !is_orphan_process_identity_current(&orphan)
                {
                    if get_process_start_id(orphan.pid).is_some() { continue; }
                    // The pid is gone entirely: the journaled process is provably
                    // dead, so there is nothing to defer (same gate as the
                    // pid-only branch below).
                    if !is_stopping_process_alive(&ProcessIdentity { pid: orphan.pid, process_start_id: None }) {
                        continue;
                    }
                    deferred.push(orphan.clone());
                    continue;
                }
                // A pid-only record can never prove identity: win32 relies on the
                // kernel's kill-on-close job for those, and a bare pid is never
                // kill authority. A provably dead pid needs no reap.
                if !is_stopping_process_alive(&ProcessIdentity { pid: orphan.pid, process_start_id: None }) {
                    continue;
                }
                // A live pid the cleanup cannot prove is ours: never kill by bare
                // pid and never report the stop failed for it. Record bounded,
                // truthful, recoverable state instead of erasing the uncertainty.
                deferred.push(orphan);
            }
            if !deferred.is_empty() {
                if let Err(error) = record_deferred_orphans(&self.descriptor_dir, &descriptor.worker_id, &deferred) {
                    eprintln!("[{}] Deferred orphan evidence for {} was not safely persisted: {error}; authoritative journal retained", iso_from_ms(supervisor_now_ms() as f64), descriptor.worker_id);
                    return Err(DEFERRED_ORPHAN_PERSISTENCE_ERROR.into());
                }
            }
            if failed { return Err("Orphan cleanup incomplete; recovery journal retained".into()); }
            self.assert_dead_registration(worker, &descriptor).await?;
            clear_orphan_process_journal(path);
        }
        for record in latest.iter().filter(|record| record.busy) {
            journal.record(WorkerRecoveryRecordInput {
                active_session_id: record.active_session_id.clone(), session_id: record.session_id.clone(),
                session_file: record.session_file.clone(), busy: false, operation: "recovery_hold".into(),
            });
        }
        if WorkerRecoveryJournal::read_latest(&descriptor.recovery_journal_path).iter().any(|record| record.busy) {
            return Err("Could not persist uncertain-operation recovery; journal retained".into());
        }
        Ok(())
    }

    pub(super) fn park_worker_recovery_failure(&self, worker: &Arc<Worker>, error: &str) {
        let id = worker.descriptor.lock().unwrap().worker_id.clone();
        let descriptor = {
            let workers = self.workers.lock().unwrap();
            if !workers.get(&id).is_some_and(|current| Arc::ptr_eq(current, worker)) { return; }
            let mut descriptor = worker.descriptor.lock().unwrap();
            // A failed relaunch may already have installed a newer generation.
            // Never let the retired Arc overwrite its replacement's descriptor.
            descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
            descriptor.last_error = Some(error.into());
            descriptor.last_failure_at = Some(chrono::Utc::now().to_rfc3339());
            if let Err(error) = self.persist_worker(&descriptor) { eprintln!("Could not persist failed recovery {}: {error}", descriptor.worker_id); }
            descriptor.clone()
        };
        self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
        eprintln!("Session worker {} recovery parked: {error}", descriptor.worker_id);
    }

    pub(super) fn park_worker_stop_cleanup_failure(&self, worker: &Arc<Worker>, error: &str) {
        let id = worker.descriptor.lock().unwrap().worker_id.clone();
        {
            let workers = self.workers.lock().unwrap();
            if !workers.get(&id).is_some_and(|current| Arc::ptr_eq(current, worker)) { return; }
            let mut descriptor = worker.descriptor.lock().unwrap();
            if descriptor.stop_requested_at.is_none() { return; }
            descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
            descriptor.last_error = Some(format!("Stop cleanup parked; retained descriptor and journals require attention: {error}"));
            descriptor.last_failure_at = Some(chrono::Utc::now().to_rfc3339());
            if let Err(error) = self.persist_worker(&descriptor) { eprintln!("Could not persist parked stop cleanup {id}: {error}"); }
        }
        self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
        eprintln!("Session worker {id} stop cleanup parked; recovery evidence retained: {error}");
    }

    pub(super) async fn reclaim_stale_worker_registration(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<bool, String> {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if worker.client.lock().unwrap().as_ref().is_some_and(|client| client.is_connected()) || worker.recovery.load(Ordering::SeqCst) { return Ok(false); }
        if descriptor.stop_requested_at.is_none() && (descriptor.lifecycle != DAEMON_WORKER_LIFECYCLE_FAILED || descriptor.owner_client_id.is_some()) { return Ok(false); }
        if stop_cleanup_is_parked(&descriptor) { return Err("Session worker stop cleanup is parked; retained recovery journals require attention".into()); }
        if is_stopping_process_alive(&ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() }) { return Ok(false); }
        if descriptor.stop_requested_at.is_some() {
            self.stop_worker(worker, descriptor.archive_on_stop == Some(true), false).await?;
            return Ok(true);
        }
        self.recover_uncertain_worker_operations(worker).await?;
        self.assert_dead_registration(worker, &descriptor).await?;
        self.invalidate_worker_input_pauses(worker);
        self.flip_worker_roster_entries_inactive(worker);
        remove_file_durably(&self.descriptor_dir.join(format!("{}.json", descriptor.worker_id)).to_string_lossy(), RemoveFileDurablyOptions { fsync_dir: true, platform: None }).await.map_err(|error| error.to_string())?;
        self.retire_worker_journals(&descriptor).await;
        self.workers.lock().unwrap().remove(&descriptor.worker_id);
        Ok(true)
    }

    /// Retirement of a stopped generation's journals (audit STALE-JOURNALS-01):
    /// a successful stop has resolved the recovery evidence, so this worker's own
    /// recovery and orphan journals are superseded instead of accumulating. The
    /// deferred-orphan side record (unproven pids) is intentionally kept, and
    /// journals of other runs or workers are never touched.
    pub(super) async fn retire_worker_journals(&self, descriptor: &DaemonWorkerDescriptor) {
        // RemoveFileDurablyOptions is not Copy; each durable removal gets its own value.
        if let Err(error) = remove_file_durably(
            &descriptor.recovery_journal_path,
            RemoveFileDurablyOptions { fsync_dir: true, platform: None },
        ).await {
            eprintln!("Could not retire recovery journal for {}: {error}", descriptor.worker_id);
        }
        if let Some(orphan_path) = &descriptor.orphan_process_journal_path {
            if let Err(error) = remove_file_durably(
                orphan_path,
                RemoveFileDurablyOptions { fsync_dir: true, platform: None },
            ).await {
                eprintln!("Could not retire orphan journal for {}: {error}", descriptor.worker_id);
            }
        }
    }

    async fn cancel_session_tree(self: &Arc<Self>, descriptor: &DaemonWorkerDescriptor, exclude: Option<&Arc<Worker>>) -> Result<(), String> {
        let Some(root_file) = descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref()) else { return Ok(()); };
        let Some(root_id) = &descriptor.root_session_id else { return Ok(()); };
        let infos = self.rlm_spawn_ledger().await?.family().await;
        let parents: HashMap<String, String> = infos.iter().filter_map(|info| info.parent_session_path.as_ref().map(|parent| (canonical_session_path(&info.path), canonical_session_path(parent)))).collect();
        let roots = HashSet::from([canonical_session_path(root_file)]);
        let mut sessions = vec![(root_id.clone(), root_file.clone())];
        sessions.extend(infos.into_iter().filter(|info| info.id != *root_id && roster_path_descends_from(&parents, &canonical_session_path(&info.path), &roots)).map(|info| (info.id, info.path)));
        // A replacement worker (including any child) owns its schedules again.
        for (_, file) in &sessions { if self.find_worker_by_session_file(file, exclude).is_some() { return Ok(()); } }
        if let Some(worker) = exclude {
            self.assert_dead_registration(worker, descriptor).await?;
            if worker.descriptor.lock().unwrap().owner_client_id.is_none() && descriptor.owner_client_id.is_some() { return Ok(()); }
        } else { self.ownership.assert_current().await.map_err(|error| error.to_string())?; }
        let store = AgentCronJobStore::for_session_artifacts();
        for (id, file) in &sessions {
            let artifacts = crate::core::session_manager::get_session_artifact_path_for_file(file, Some(id));
            if Path::new(&artifacts).join(SESSION_SCHEDULED_JOBS_FILENAME).exists() { store.register_session_artifact(id, &artifacts); }
        }
        for (_, file) in sessions {
            store.try_cancel_jobs_for_session(&CancelJobsForSessionInput { session_file: Some(file), ..Default::default() }, supervisor_now_ms() as f64)?;
        }
        Ok(())
    }

    pub(super) async fn finish_worker_schedules(self: &Arc<Self>, worker: &Arc<Worker>, archive: bool) -> Result<(), String> {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if descriptor.owner_client_id.is_some() { self.cancel_session_tree(&descriptor, Some(worker)).await?; }
        if archive {
            if let (Some(id), Some(file)) = (&descriptor.root_session_id, descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref())) {
                let artifacts = crate::core::session_manager::get_session_artifact_path_for_file(file, Some(id));
                let store = AgentCronJobStore::for_session_artifacts();
                store.register_session_artifact(id, &artifacts);
                store.try_cancel_jobs_for_session(&CancelJobsForSessionInput { session_file: Some(file.clone()), ..Default::default() }, supervisor_now_ms() as f64)?;
                self.assert_dead_registration(worker, &descriptor).await?;
                self.catalog.archive(file, id).await?;
            }
        }
        Ok(())
    }

    pub(super) async fn settle_ephemeral_cancel_intents(self: &Arc<Self>) -> HashSet<String> {
        let mut blocked = HashSet::new();
        let Ok(entries) = std::fs::read_dir(&self.descriptor_dir) else { return blocked; };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") { continue; }
            let Some(descriptor) = std::fs::read(&path).ok().and_then(|bytes| serde_json::from_slice::<DaemonWorkerDescriptor>(&bytes).ok()) else { continue; };
            if descriptor.owner_client_id.is_none() || descriptor.stop_requested_at.is_none()
                || normalize_socket_path_for_daemon(&descriptor.supervisor_socket_path, None) != self.socket_path { continue; }
            let Some(file) = descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref()) else { continue; };
            blocked.insert(canonical_session_path(file));
            if self.workers.lock().unwrap().contains_key(&descriptor.worker_id) { continue; }
            if is_stopping_process_alive(&ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() }) { continue; }
            if let Err(error) = self.cancel_session_tree(&descriptor, None).await {
                eprintln!("Retaining cancelled-owner schedule cleanup {}: {error}", descriptor.worker_id);
                continue;
            }
            let current = std::fs::read(&path).ok().and_then(|bytes| serde_json::from_slice::<DaemonWorkerDescriptor>(&bytes).ok());
            if current.as_ref().is_some_and(|current| same_registration(current, &descriptor)) && self.ownership.assert_current().await.is_ok() {
                let _ = remove_file_durably(&path.to_string_lossy(), RemoveFileDurablyOptions { fsync_dir: true, platform: None }).await;
            }
        }
        blocked
    }

    pub(super) async fn stream_cached_attach(&self, public: &Arc<PublicClient>, active: &str, response: &mut DaemonResponse) -> Result<(), String> {
        let data = response.data.as_mut().and_then(Value::as_object_mut).ok_or("Missing attach result")?;
        let snapshot = data.get_mut("snapshot").and_then(Value::as_object_mut).ok_or("Missing attach snapshot")?;
        // Older Rust workers emitted JS-style whole floats (0.0). The public
        // snapshot protocol requires an integer; otherwise the UI silently drops
        // both begin/end records and times out. Preserve the exact sequence.
        let sequence = snapshot.get("lastEventSequence").and_then(|value| {
            value.as_u64().or_else(|| value.as_f64().filter(|number|
                number.is_finite() && *number >= 0.0 && *number <= 9_007_199_254_740_991.0 && number.fract() == 0.0
            ).map(|number| number as u64))
        }).ok_or("Invalid attach snapshot event sequence")?;
        snapshot.insert("lastEventSequence".into(), json!(sequence));
        let messages = match snapshot.remove("messages") { Some(Value::Array(messages)) => messages, _ => return Err("Invalid attach transcript".into()) };
        let count = messages.len();
        let snapshot_id = format!("supervisor-{}", uuid::Uuid::new_v4());
        let cache = Arc::new(SnapshotTranscriptCache::new(SnapshotTranscriptCacheOptions {
            active_session_id: active.into(), snapshot_id: snapshot_id.clone(), messages: None,
            cache_root: self.descriptor_dir.join("snapshot-cache").to_string_lossy().into_owned(), target_chunk_bytes: None, memory_cache_bytes: None,
        }));
        let retained = cache.retain();
        cache.dispose(); // RAII cleanup on success, disconnect, failure or cancellation.
        cache.append_messages(messages)?;
        snapshot.insert("messages".into(), json!([]));
        let metadata = Value::Object(snapshot.clone());
        data.insert("snapshotStream".into(), json!({"id": snapshot_id}));
        let send = |mut bytes: Vec<u8>| async move {
            if bytes.last() != Some(&b'\n') { bytes.push(b'\n'); }
            // Drain before each chunk; otherwise a 1024-entry event queue retains
            // hundreds of MiB of encoded transcript despite the disk cache.
            tokio::time::timeout(Duration::from_secs(30), async {
                while public.output.capacity() < public.output.max_capacity() {
                    tokio::select! { _ = public.stopped.cancelled() => return Err("Snapshot client disconnected".to_string()), _ = tokio::time::sleep(Duration::from_millis(1)) => {} }
                }
                tokio::select! { _ = public.stopped.cancelled() => Err("Snapshot client disconnected".to_string()), sent = public.output.send(bytes) => sent.map_err(|_| "Snapshot client disconnected".to_string()) }
            }).await.map_err(|_| "Snapshot client did not drain within 30 seconds".to_string())?
        };
        let result: Result<(), String> = async {
        send(serde_json::to_vec(response).map_err(|error| error.to_string())?).await?;
        send(serde_json::to_vec(&json!({"type":"session_snapshot_begin","activeSessionId":active,"snapshotId":snapshot_id,"snapshot":metadata,"messageCount":count,"targetChunkBytes":SNAPSHOT_TARGET_CHUNK_BYTES,"purpose":"attach"})).unwrap()).await?;
        for index in 0..cache.chunk_count() { send(cache.read_chunk(index)?).await?; }
        send(serde_json::to_vec(&json!({"type":"session_snapshot_end","activeSessionId":active,"snapshotId":snapshot_id,"chunkCount":cache.chunk_count(),"lastEventSequence":metadata["lastEventSequence"],"lastEventCursor":metadata["lastEventCursor"]})).unwrap()).await?;
        Ok(())
        }.await;
        if let Err(error) = result {
            if !public.write(&json!({"type":"session_snapshot_failed","activeSessionId":active,"snapshotId":snapshot_id,"error":error})) { public.stopped.cancel(); }
        }
        drop(retained);
        Ok(())
    }
}
