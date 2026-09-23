//! Refinement work owns progress and settlement, but not prompt admission during planning.
use super::*;
use std::time::Instant;

pub(super) struct RefinementCurrentness {
    epoch: u64,
    branch_version: u64,
    pub(super) signal: CancellationToken,
}

impl RefinementCurrentness {
    pub(super) fn capture(session: &AgentSession, epoch: Option<u64>) -> Result<Self, String> {
        let _admission = session.explicit_stop_admission.lock().unwrap();
        let current = Self {
            epoch: epoch.unwrap_or_else(|| session.session_input_pump_epoch.load(Ordering::SeqCst)),
            branch_version: session.auto_refine_branch_version.load(Ordering::SeqCst),
            signal: session.refinement_signal(),
        };
        current.check(session)?;
        Ok(current)
    }

    pub(super) fn check(&self, session: &AgentSession) -> Result<(), String> {
        if self.signal.is_cancelled()
            || session.explicitly_stopped()
            || session.disposed.load(Ordering::SeqCst)
            || session.disposing.load(Ordering::SeqCst)
            || self.epoch != session.session_input_pump_epoch.load(Ordering::SeqCst)
            || self.branch_version != session.auto_refine_branch_version.load(Ordering::SeqCst)
        {
            return Err(
                "Refinement was aborted because the session changed or stopped.".to_string(),
            );
        }
        Ok(())
    }
}

pub(super) struct RefinementProgress {
    session: Arc<AgentSession>,
    generation: u64,
    started: Instant,
    stage_started: Mutex<Instant>,
    finished: AtomicBool,
}

impl RefinementProgress {
    pub(super) fn new(session: &Arc<AgentSession>) -> Self {
        let progress = Self {
            session: session.clone(),
            generation: session
                .refinement_progress_generation
                .fetch_add(1, Ordering::SeqCst)
                + 1,
            started: Instant::now(),
            stage_started: Mutex::new(Instant::now()),
            finished: AtomicBool::new(false),
        };
        progress.stage("Refinement queued");
        progress
    }

    pub(super) fn stage(&self, reason: &'static str) {
        if !self.finished.load(Ordering::SeqCst) {
            self.emit(true, reason);
        }
    }

    fn emit(&self, active: bool, reason: &'static str) {
        let elapsed_ms = self.started.elapsed().as_millis();
        let stage_ms = {
            let mut started = self.stage_started.lock().unwrap();
            let elapsed = started.elapsed().as_millis();
            *started = Instant::now();
            elapsed
        };
        // Fixed stage labels and monotonic durations only. Never log planner input/output or auth.
        eprintln!("refinement_progress active={active} stage={reason:?} elapsed_ms={elapsed_ms} previous_stage_ms={stage_ms}");
        if self
            .session
            .refinement_progress_generation
            .load(Ordering::SeqCst)
            == self.generation
        {
            self.session.emit(AgentSessionEvent::RefinementUpdate {
                active,
                reason: Some(reason.to_string()),
            });
        }
    }

    pub(super) fn finish(&self, reason: &'static str) {
        if !self.finished.swap(true, Ordering::SeqCst) {
            self.emit(false, reason);
        }
    }

    pub(super) fn finish_result(&self, result: &Result<RefinementResult, String>) {
        self.finish(match result {
            Ok(result) if result.applied_edits.iter().any(|edit| edit.applied) => {
                "Refinement saved"
            }
            Ok(_) => "No refinement changes needed",
            Err(error) if is_refinement_skipped_error(error) => "Refinement skipped",
            Err(error) if error.contains("aborted") || error.contains("cancelled") => {
                "Refinement cancelled"
            }
            Err(_) => "Refinement failed",
        });
    }

    pub(super) fn finish_plan(&self, result: &Result<RefinementPlan, String>) {
        self.finish(match result {
            Ok(_) => "Refinement ready to save",
            Err(error) if is_refinement_skipped_error(error) => "Refinement skipped",
            Err(error) if error.contains("aborted") || error.contains("cancelled") => {
                "Refinement cancelled"
            }
            Err(_) => "Refinement failed",
        });
    }
}

impl Drop for RefinementProgress {
    fn drop(&mut self) {
        self.finish("Refinement cancelled");
    }
}

pub(super) struct RefinementFlight(pub(super) Arc<dyn Fn() + Send + Sync>);

impl Drop for RefinementFlight {
    fn drop(&mut self) {
        (self.0)();
    }
}

pub(super) struct RefinementCommitFence(pub(super) CommitFence);

impl Drop for RefinementCommitFence {
    fn drop(&mut self) {
        self.0.release();
    }
}
