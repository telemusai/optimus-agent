use super::*;

pub(super) type SharedStop = futures::future::Shared<BoxFuture<'static, Result<(), KernelError>>>;
pub(super) enum StartupAdmission {
    Ready(Arc<StartupHandle>, bool),
    Stopping(SharedStop),
}

impl IpythonKernelProvisioner {
    pub(super) fn admit_startup(&self) -> Result<StartupAdmission, KernelError> {
        let mut lifecycle = self.lifecycle.lock().unwrap();
        let mut ownership = self.ownership.lock().unwrap();
        if ownership.fenced {
            return Err(KernelError::new("Kernel provisioner retained-stop fence"));
        }
        if self.dispose_controller.is_aborted() {
            return Err(KernelError::new("Kernel provisioner disposed"));
        }
        if let Some(stop) = lifecycle.as_ref() {
            match stop.peek() {
                Some(Ok(())) => *lifecycle = None,
                Some(Err(error)) => return Err(error.clone()),
                None => return Ok(StartupAdmission::Stopping(stop.clone())),
            }
        }
        let mut memo = self.manager_promise.lock().unwrap();
        if self.manager().is_some_and(|manager| manager.is_defunct()) {
            *memo = None;
            *self.started_manager.lock().unwrap() = None;
        }
        if let Some(handle) = memo.as_ref() {
            return Ok(StartupAdmission::Ready(handle.clone(), false));
        }
        self.settle_startup();
        *self.last_restore.lock().unwrap() = None;
        let handle = Arc::new(StartupHandle::new());
        *memo = Some(handle.clone());
        ownership.startups.push(handle.clone());
        Ok(StartupAdmission::Ready(handle, true))
    }

    pub(super) fn complete_startup(
        &self,
        handle: &Arc<StartupHandle>,
        outcome: &Result<Arc<dyn KernelClient>, KernelError>,
    ) {
        let _lifecycle = self.lifecycle.lock().unwrap();
        let mut memo = self.manager_promise.lock().unwrap();
        if memo
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, handle))
        {
            match outcome {
                Ok(manager)
                    if !self.dispose_controller.is_aborted()
                        && !self.ownership.lock().unwrap().fenced =>
                {
                    *self.started_manager.lock().unwrap() = Some(manager.clone());
                }
                Err(_) => *memo = None,
                _ => {}
            }
            self.settle_startup();
        }
    }

    pub(super) fn is_current_startup(&self, handle: &Arc<StartupHandle>) -> bool {
        let _lifecycle = self.lifecycle.lock().unwrap();
        !self.dispose_controller.is_aborted()
            && self
                .manager_promise
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, handle))
    }

    pub(super) fn begin_stop(&self, snapshot: Option<bool>) -> SharedStop {
        let mut lifecycle = self.lifecycle.lock().unwrap();
        if let Some(snapshot) = snapshot {
            *self.dispose_snapshot.lock().unwrap() = snapshot;
            self.dispose_controller.abort(None);
        }
        if let Some(stop) = lifecycle.as_ref() {
            return stop.clone();
        }
        let pending = self.manager_promise.lock().unwrap().take();
        *self.started_manager.lock().unwrap() = None;
        self.settle_startup();
        let gate = self
            .options
            .as_ref()
            .and_then(|options| options.ready_gate.clone());
        let stop = async move {
            if let Some(gate) = gate {
                gate().await;
            }
            if let Some(pending) = pending {
                // Failed startup tears down before settling; successful startup
                // hands its retired manager to this single teardown owner.
                if let Ok(manager) = pending.wait().await {
                    match snapshot {
                        Some(snapshot) => manager.shutdown(snapshot, true).await?,
                        None => manager.kill().await?,
                    }
                }
            }
            Ok(())
        };
        let stop = std::panic::AssertUnwindSafe(stop)
            .catch_unwind()
            .map(|outcome| {
                outcome.unwrap_or_else(|_| Err(KernelError::new("Kernel shutdown panicked")))
            })
            .boxed()
            .shared();
        *lifecycle = Some(stop.clone());
        // A cancelled caller cannot abandon the stop and let a new kernel race its flush.
        let _ = self.owned_tasks.spawn(stop.clone(), false, true);
        stop
    }
}

#[cfg(test)]
#[path = "ipython_lifecycle_tests.rs"]
mod tests;
