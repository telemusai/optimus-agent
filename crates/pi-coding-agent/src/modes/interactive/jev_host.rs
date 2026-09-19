//! `/jev` handler: the mode / menu / key / status command surface.
//!
//! Everything the command needs lives here, so `native_host_commands.rs` keeps
//! only a small dispatch arm, two `Dialog` variants and two mount branches.
//!
//! Boundaries this file obeys (DESIGN.md sections 11 and 12):
//!
//! * NO model control. There is no primary-model, scoped-model, thinking-level
//!   or service-tier call here. The user's primary model, provider and effort
//!   stay authoritative, and category 6 stays an advisory record that is never
//!   executed.
//! * NO subagent control. There is no create / delete / cancel / pause / resume,
//!   no child model / effort / task / message / depth / concurrency / budget
//!   call, and no way for a configuration flag to grant one. This file cannot
//!   reach a mutating child handle.
//! * Connection calls read session identity and optional worker telemetry. A
//!   `/jev` command never mutates the session.
//!
//! Off semantics: Off is written to the settings file and the footer is
//! republished in the same turn, so it takes effect immediately. The lane owns no
//! scheduler or client, so there is nothing in flight to drain here; the
//! integration hook that must drop in-flight comparison work is documented in
//! `reports/ui/LANE_REPORT.md`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;

use pi_tui::tui::Component as TuiComponent;
use tokio::sync::mpsc as async_mpsc;

use crate::modes::agent_connection::types as wire;

use super::Dialog;
use super::jev_footer::JevFooterSnapshot;
use super::jev_key_input::JevKeyInputComponent;
use super::jev_menu::{
    clear_secret, is_on_shorthand, is_submit_key, jev_usage, mode_change_message,
    parse_jev_request, render_help, render_status, CredentialStatus, JevMenuAction,
    JevModeBridge, JevRequest, JevSecret, JevStatusReport, KeyInputState,
    JEV_BOUNDARY_NOTICE, JEV_ON_COMPARE_NOTICE,
};
use super::jev_menu_component::JevMenuComponent;
// `CommandOutput` and `HostEvent` are declared by the host module that owns this
// one (`native_host.rs`); the explicit path keeps this file independent of the
// host's own glob imports.
use super::super::{CommandOutput, HostEvent};
use pi_jev::credential::{default_credential_store, CredentialStore, CredentialWriteGate};
use pi_jev::types::JevMode;

/// One menu action, sent from the overlay component back to this handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum JevMenuEvent {
    Action(JevMenuAction),
}

/// One key-entry event. `Submitted` carries the secret inside [`JevSecret`],
/// which has a redacting `Debug` and no `Display`.
pub(super) enum JevKeyEvent {
    Submitted(JevSecret),
    Cancelled,
}

/// The `/jev` menu overlay. It routes keys and reports actions; all decisions
/// live in the pure [`JevMenuState`](super::jev_menu::JevMenuState).
pub(super) struct JevMenuOverlay {
    component: JevMenuComponent,
    send: mpsc::Sender<HostEvent>,
}

impl JevMenuOverlay {
    pub(super) fn new(
        mode: JevMode,
        events: async_mpsc::UnboundedSender<JevMenuEvent>,
        send: mpsc::Sender<HostEvent>,
    ) -> Self {
        let component = JevMenuComponent::new(
            mode,
            Box::new(move |action| {
                let _ = events.send(JevMenuEvent::Action(action));
            }),
        );
        Self { component, send }
    }
}

impl TuiComponent for JevMenuOverlay {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.component.render(width)
    }

    fn handle_input(&mut self, data: &str) {
        self.component.handle_input(data);
        if self.component.state().closed {
            // The overlay closes itself; the owner loop hides it on this event.
            let _ = self.send.send(HostEvent::CloseCommandDialog);
        }
    }

    fn invalidate(&mut self) {
        self.component.invalidate();
    }
}

/// The masked key-entry overlay.
pub(super) struct JevKeyOverlay {
    component: JevKeyInputComponent,
    events: Rc<RefCell<Option<async_mpsc::UnboundedSender<JevKeyEvent>>>>,
    send: mpsc::Sender<HostEvent>,
}

impl JevKeyOverlay {
    pub(super) fn new(
        events: async_mpsc::UnboundedSender<JevKeyEvent>,
        send: mpsc::Sender<HostEvent>,
    ) -> Self {
        Self {
            component: JevKeyInputComponent::new(),
            events: Rc::new(RefCell::new(Some(events))),
            send,
        }
    }
}

impl TuiComponent for JevKeyOverlay {
    fn render(&mut self, width: f64) -> Vec<String> {
        self.component.render(width)
    }

    fn handle_input(&mut self, data: &str) {
        let state = self.component.handle_key(data);
        let submit = is_submit_key(data);
        let events = self.events.borrow_mut();
        // The sender stays alive for the whole dialog so a cancel AFTER a
        // submit (e.g. Escape while validation is in flight) still delivers
        // `Cancelled` to the key-dialog handler. Double-submit is already
        // impossible: `take_for_validation` moves the buffer out exactly once.
        // Submit moves the buffer out of the component exactly once; the secret
        // then lives only inside the handler's validation future.
        let event = match (state, submit) {
            (KeyInputState::Cancelled, _) => Some(JevKeyEvent::Cancelled),
            (_, true) => self
                .component
                .input_mut()
                .take_for_validation()
                .map(|value| JevKeyEvent::Submitted(JevSecret::new(value))),
            _ => None,
        };
        if let Some(event) = event {
            if let Some(sender) = events.as_ref() {
                let _ = sender.send(event);
            }
        }
        drop(events);
        if self.component.closed() {
            // The dialog is done: consume the sender exactly once, then close.
            *self.events.borrow_mut() = None;
            let _ = self.send.send(HostEvent::CloseCommandDialog);
        }
    }

    fn invalidate(&mut self) {
        self.component.invalidate();
    }
}

/// The credential-presence snapshot. Presence booleans only: no secret is read,
/// formatted or returned anywhere in this file.
///
/// The SAVED credential is DESIGN.md section 6 (lane A's platform credential
/// store), read through `CredentialStore::exists("typesafe")`. Presence only:
/// no secret is read, and a store that is unavailable reports `false` (with
/// `credentialPresenceKnown: false` in the daemon view) instead of a guess.
pub(super) fn credential_status(saved_present: bool) -> CredentialStatus {
    let env = super::jev_menu::process_env_presence();
    CredentialStatus::resolve(saved_present, env.typesafe_api_key, env.jev_api_key)
}

/// The agent dir that holds both the settings file and the credential store.
pub(super) fn agent_dir() -> PathBuf {
    PathBuf::from(crate::config::get_agent_dir())
}

/// Saved-credential presence through the OWNER's store. Presence only: the
/// `CredentialStore` is never asked for the secret here, and a store that is
/// unavailable reports `false` rather than a guess.
pub(super) fn saved_credential_present(store: &dyn CredentialStore) -> bool {
    store.is_available() && store.exists(pi_jev::config::DEFAULT_KEY_ID).unwrap_or(false)
}

/// The footer snapshot for a mode. Pure local reads.
pub(super) fn footer_snapshot(mode: JevMode, credential: CredentialStatus) -> JevFooterSnapshot {
    JevFooterSnapshot {
        mode,
        credential,
        pipeline: Default::default(),
    }
}

/// Publish the footer segment through the EXISTING extension status surface
/// (`ExtensionUiRequest { method: "setStatus" }`), which renders directly under
/// the model/effort tray. No `native_host.rs` edit is needed for this, and the
/// host's status row truncates the text to the live width.
fn publish_footer(
    send: &mpsc::Sender<HostEvent>,
    mode: JevMode,
    credential: CredentialStatus,
) {
    let event = footer_snapshot(mode, credential).published_event();
    let _ = send.send(HostEvent::Connection(event));
}

/// The `/jev status` panel, built from local state only: no network call and no
/// secret. The `applied` figure is zero in Compare because Compare applies
/// nothing; in Active the panel renders the worker's real Active counters
/// (applied boundaries, accepted-with-no-effect, refused, unavailable), and an
/// absent block is reported as unknown rather than as zero.
pub(super) async fn status_panel(
    connection: &Arc<dyn wire::AgentConnection>,
    settings: &pi_jev::config::JevSettings,
    session_id: &str,
    credential: CredentialStatus,
) -> String {
    let payload = tokio::time::timeout(std::time::Duration::from_secs(2), connection.get_jev_status())
        .await.ok().and_then(Result::ok).flatten();
    let snapshot = payload.as_ref().and_then(|value| value.get("pipeline")).filter(|value| value.is_object());
    render_status(&JevStatusReport::local_only(
        settings.effective_mode(session_id),
        settings.effective_mode_with_scope(session_id).scope,
        credential,
    ).with_snapshot(snapshot))
}

/// Run `/jev`.
///
/// The dispatch task holds no UI handle (the host's established rule: compare
/// `HostEvent::ContextTree`, which is formatted on the owner loop for the same
/// reason), so the footer width is resolved by the host's status row, which
/// truncates to the live terminal width when it renders. `footer_segment` still
/// implements the narrow-terminal rule for a caller that can measure first.
pub(super) async fn run(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    args: &str,
) -> Result<CommandOutput, String> {
    let state = connection.get_state().await?;
    let session_id = state.session_id.clone();
    let bridge = JevModeBridge::new(&agent_dir());
    let store = default_credential_store(agent_dir());
    let credential = credential_status(saved_credential_present(store.as_ref()));

    match parse_jev_request(args) {
        JevRequest::Unknown(argument) => Ok(CommandOutput::Error(format!(
            "Unknown /jev option: {argument}\n{}\n{JEV_BOUNDARY_NOTICE}",
            jev_usage()
        ))),
        JevRequest::Menu => menu_dialog(connection, &bridge, &session_id, credential, send).await,
        // Every mode is a real mode, so `active` takes the same write path as Off
        // and Compare and simply reports what was written.
        JevRequest::SetMode(mode) => {
            let change = bridge.set_session_mode(&session_id, mode)?;
            crate::core::jev_bridge::invalidate_settings_cache();
            // A mode change must take effect immediately: the footer is republished
            // in the same turn as the write, so nothing can render the old state.
            publish_footer(send, bridge.effective_mode(&session_id), credential);
            let message = mode_change_message(&change);
            // `/jev active` needs no extra wording: `mode_change_message` already
            // appends the exact Active notice and the permanent boundary for it.
            // `/jev on` stays Compare and adds the one line that explains why the
            // shorthand is not the request-changing mode.
            if mode == JevMode::Compare && is_on_shorthand(args) {
                return Ok(CommandOutput::Status(format!(
                    "{message}\n{JEV_ON_COMPARE_NOTICE}"
                )));
            }
            Ok(CommandOutput::Status(message))
        }
        JevRequest::Status => {
            let settings = bridge.settings();
            Ok(CommandOutput::Panel(status_panel(
                connection,
                &settings,
                &session_id,
                credential,
            ).await))
        }
        JevRequest::InputKey => key_dialog(store, credential, &session_id, send).await,
        JevRequest::ClearKey => {
            // The inverse of the only credential write. It reports presence only and
            // never reads the stored value.
            let before = saved_credential_present(store.as_ref());
            clear_secret(store.as_ref()).map_err(|error| error.log_line())?;
            let credential = credential_status(false);
            publish_footer(
                send,
                bridge.effective_mode(&session_id),
                credential,
            );
            Ok(CommandOutput::Status(format!(
                "Jev API key removed from the credential store (was {before}). Mode is unchanged: {}.",
                super::jev_menu::mode_label(bridge.effective_mode(&session_id))
            )))
        }
        JevRequest::Help => Ok(CommandOutput::Status(render_help())),
    }
}

/// The interactive menu. Actions stream in from the overlay component.
async fn menu_dialog(
    connection: &Arc<dyn wire::AgentConnection>,
    bridge: &JevModeBridge,
    session_id: &str,
    credential: CredentialStatus,
    send: &mpsc::Sender<HostEvent>,
) -> Result<CommandOutput, String> {
    let mode = bridge.effective_mode(session_id);
    let (events, mut receive) = async_mpsc::unbounded_channel();
    send.send(HostEvent::CommandDialog(Dialog::Jev(mode, events)))
        .map_err(|error| error.to_string())?;
    while let Some(JevMenuEvent::Action(action)) = receive.recv().await {
        match action {
            JevMenuAction::None => {}
            JevMenuAction::Cancel => {
                let _ = send.send(HostEvent::CloseCommandDialog);
                return Ok(CommandOutput::Status("Jev: cancelled".to_string()));
            }
            JevMenuAction::SetMode(mode) => {
                let change = bridge.set_session_mode(session_id, mode)?;
                crate::core::jev_bridge::invalidate_settings_cache();
                publish_footer(send, bridge.effective_mode(session_id), credential);
                let _ = send.send(HostEvent::CloseCommandDialog);
                return Ok(CommandOutput::Status(mode_change_message(&change)));
            }
            JevMenuAction::ShowStatus => {
                let _ = send.send(HostEvent::CloseCommandDialog);
                let settings = bridge.settings();
                return Ok(CommandOutput::Panel(status_panel(
                    connection, &settings, session_id, credential,
                ).await));
            }
            JevMenuAction::InputKey => {
                let _ = send.send(HostEvent::CloseCommandDialog);
                let store = default_credential_store(agent_dir());
                return key_dialog(store, credential, session_id, send).await;
            }
        }
    }
    let _ = send.send(HostEvent::CloseCommandDialog);
    Ok(CommandOutput::Nothing)
}

/// Key encryption runs off the event loop; cancellation fences the final commit.
///
/// The store is the OWNER's `default_credential_store` (lane A): the platform
/// store on Windows, or an explicit "unavailable" store. There is no plaintext
/// fallback anywhere: [`store_secret`] refuses an unavailable store before the
/// write, and the UI reports that refusal instead of degrading.
async fn key_dialog(
    store: Arc<dyn CredentialStore>,
    credential: CredentialStatus,
    session_id: &str,
    send: &mpsc::Sender<HostEvent>,
) -> Result<CommandOutput, String> {
    let (events, mut receive) = async_mpsc::unbounded_channel();
    send.send(HostEvent::CommandDialog(Dialog::JevKey(events)))
        .map_err(|error| error.to_string())?;
    let Some(event) = receive.recv().await else {
        return Ok(CommandOutput::Status("Jev key entry cancelled".to_string()));
    };
    let JevKeyEvent::Submitted(secret) = event else {
        let _ = send.send(HostEvent::CloseCommandDialog);
        return Ok(CommandOutput::Status(
            "Jev key entry cancelled; nothing was stored.".to_string(),
        ));
    };
    let backend = store.backend_name();
    if secret.is_empty() || secret.looks_like_placeholder() || secret.looks_malformed() {
        let _ = send.send(HostEvent::CloseCommandDialog);
        return Ok(CommandOutput::Warning("Key not stored: invalid key format.".to_string()));
    }
    let gate = Arc::new(CredentialWriteGate::default());
    struct CancelOnDrop(Arc<CredentialWriteGate>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) { self.0.cancel(); }
    }
    let _cancel_on_drop = CancelOnDrop(gate.clone());
    let worker_gate = gate.clone();
    let mut validation = tokio::task::spawn_blocking(move || {
        if !store.is_available() {
            return Err(pi_jev::error::JevError::Unavailable { reason: "secure credential store unavailable".to_string() });
        }
        let secret = secret.into_secret_string();
        store.store_cancellable(pi_jev::config::DEFAULT_KEY_ID, secret.expose(), &worker_gate)
    });
    let result = tokio::select! {
        biased;
        _ = receive.recv() => None,
        result = &mut validation => Some(result.unwrap_or_else(|_| Err(pi_jev::error::JevError::Internal { detail: "credential worker failed".to_string() }))),
        _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => None,
    };
    let _ = send.send(HostEvent::CloseCommandDialog);
    match result {
        None => Ok(CommandOutput::Status(if gate.cancel() {
            "Key entry cancelled. Nothing was stored.".to_string()
        } else {
            "Key entry closed after saving began; the key may have been stored. Check /jev status or use /jev key clear.".to_string()
        })),
        Some(Ok(())) => {
            // The key now exists, so the footer must stop reading "unavailable".
            // The MODE is re-read from the store: a stored key never changes it.
            let now = credential_status(true);
            let mode = JevModeBridge::new(&agent_dir()).effective_mode(session_id);
            let _ = send.send(HostEvent::Connection(
                footer_snapshot(mode, now).published_event(),
            ));
            Ok(CommandOutput::Status(format!(
                "API key stored in the {backend} credential store. Live validation against TypeSafe is not performed in this build."
            )))
        }
        Some(Err(error)) => Ok(CommandOutput::Warning(format!(
            "Key not stored: {error}. {}",
            credential.describe()
        ))),
    }
}
