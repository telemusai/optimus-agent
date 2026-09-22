//! `/jev` handler: the mode / menu / key / status command surface.
//!
//! Everything the command needs lives here, so `native_host_commands.rs` keeps
//! only a small dispatch arm, two `Dialog` variants and two mount branches.
//!
//! * NO PRIMARY-model control (DESIGN.md section 11; ROOT-CONTRACT v9 adds
//!   exactly one bounded Jev-model surface: `/jev models` is the single
//!   explicit networked catalog command, and `/jev model status|set|reset`
//!   writes only the agent-dir-persistent requested JEV SystemOne model with
//!   zero network, zero probe and no budget effect). The user's primary
//!   model, provider and effort stay authoritative, and category 6 stays an
//!   advisory record that is never executed. There is no primary-model,
//!   scoped-model, thinking-level or service-tier call here.
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
use std::time::Duration;

use pi_tui::tui::Component as TuiComponent;
use tokio::sync::mpsc as async_mpsc;

use crate::modes::agent_connection::types as wire;

use super::Dialog;
use super::jev_footer::JevFooterSnapshot;
use super::jev_key_input::JevKeyInputComponent;
use super::jev_menu::{
    clear_secret, compaction_state, is_on_shorthand, is_submit_key, jev_usage, mode_change_message,
    parse_jev_request, render_compaction_settings, render_full_jev_status, render_help,
    render_model_catalog, render_model_status, render_status, require_feature_support,
    CredentialStatus, FullJevChange, JevCompactionState, JevMenuAction, JevModeBridge,
    JevModelStatusReport, JevRequest, JevSecret, JevStatusReport, KeyInputState,
    JEV_ACTIVE_NOTICE, JEV_BOUNDARY_NOTICE, JEV_DISCLOSURE_NOTICE, JEV_FULL_JEV_ALREADY_OFF_NOTICE,
    JEV_FULL_JEV_OFF_NOTICE, JEV_FULL_JEV_ON_NOTICE, JEV_ON_COMPARE_NOTICE,
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

/// The footer snapshot for a session. Pure local reads.
pub(super) fn footer_snapshot(
    mode: JevMode,
    credential: CredentialStatus,
    compaction: JevCompactionState,
) -> JevFooterSnapshot {
    JevFooterSnapshot {
        mode,
        credential,
        pipeline: Default::default(),
        compaction,
    }
}

/// Publish BOTH footer segments (decision + independent compaction) through the
/// EXISTING extension status surface (`ExtensionUiRequest { method: "setStatus"
/// }`). `native_host_extensions::Statuses` renders them ON the model/effort tray
/// row and truncates to the live width. No new host surface, no RPC.
fn publish_footer(
    send: &mpsc::Sender<HostEvent>,
    bridge: &JevModeBridge,
    session_id: &str,
    credential: CredentialStatus,
) {
    let settings = bridge.settings();
    publish_footer_from_settings(send, &settings, session_id, credential);
}

/// Publish from one already-loaded settings snapshot, so the decision mode and
/// the compaction state always come from the same read.
fn publish_footer_from_settings(
    send: &mpsc::Sender<HostEvent>,
    settings: &pi_jev::config::JevSettings,
    session_id: &str,
    credential: CredentialStatus,
) {
    let snapshot = footer_snapshot(
        settings.effective_mode(session_id),
        credential,
        compaction_state(settings, session_id),
    );
    for event in snapshot.published_events() {
        let _ = send.send(HostEvent::Connection(event));
    }
}

/// Publish the CURRENT effective segments for a session: the interactive host
/// calls this on startup and on a session change, so the row shows the actual
/// per-session settings (overrides included) before any `/jev` command runs.
/// One local settings read plus one credential-presence probe; no RPC, no
/// network, and no settings change.
pub(crate) fn publish_session_footer(send: &mpsc::Sender<HostEvent>, session_id: &str) {
    let bridge = JevModeBridge::new(&agent_dir());
    let store = default_credential_store(agent_dir());
    let credential = credential_status(saved_credential_present(store.as_ref()));
    publish_footer(send, &bridge, session_id, credential);
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
    let mut text = render_status(&JevStatusReport::local_only(
        settings.effective_mode(session_id),
        settings.effective_mode_with_scope(session_id).scope,
        credential,
    ).with_settings(settings, session_id).with_snapshot(snapshot));
    if !connection.supports_jev_features() {
        text.push_str("\nAttached worker lacks Jev System One capability; new feature settings are local configuration only.\n");
    }
    text
}

const DISCLOSURE_DURATION: Duration = Duration::from_secs(5);

fn automatic_status(send: &mpsc::Sender<HostEvent>, message: String) -> CommandOutput {
    if message.contains(JEV_DISCLOSURE_NOTICE) || message.contains(JEV_ACTIVE_NOTICE) {
        show_disclosure(send, message, DISCLOSURE_DURATION);
        CommandOutput::Nothing
    } else {
        CommandOutput::Status(message)
    }
}

fn disclosure_widget(key: &str, lines: Option<Vec<String>>) -> HostEvent {
    HostEvent::Connection(wire::AgentConnectionEvent::ExtensionUiRequest {
        request: wire::AgentConnectionExtensionUiRequest {
            id: key.to_string(),
            method: "setWidget".to_string(),
            payload: serde_json::json!({
                "widgetKey": key,
                "widgetLines": lines,
                "widgetPlacement": "aboveEditor",
            }),
        },
    })
}

fn show_disclosure(send: &mpsc::Sender<HostEvent>, message: String, duration: Duration) {
    // Each expiry owns only its notice, even across another command or session switch.
    let key = format!("jev-disclosure-{}", uuid::Uuid::new_v4());
    let lines = message
        .lines()
        .map(|line| crate::modes::interactive::theme::theme::theme().fg("dim", line))
        .collect();
    if send.send(disclosure_widget(&key, Some(lines))).is_err() {
        return;
    }
    let send = send.clone();
    tokio::spawn(async move {
        tokio::time::sleep(duration).await;
        let _ = send.send(disclosure_widget(&key, None));
    });
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
            let change = bridge.set_session_mode_supported(&session_id, mode, connection.supports_jev_features())?;
            crate::core::jev_bridge::invalidate_settings_cache();
            // A mode change must take effect immediately: the footer is republished
            // in the same turn as the write, so nothing can render the old state.
            publish_footer(send, &bridge, &session_id, credential);
            let message = mode_change_message(&change);
            // `/jev active` needs no extra wording: `mode_change_message` already
            // appends the exact Active notice and the permanent boundary for it.
            // `/jev on` stays Compare and adds the one line that explains why the
            // shorthand is not the request-changing mode.
            if mode == JevMode::Compare && is_on_shorthand(args) {
                return Ok(automatic_status(send, format!(
                    "{message}\n{JEV_ON_COMPARE_NOTICE}"
                )));
            }
            Ok(automatic_status(send, message))
        }
        JevRequest::SetDefaultMode(mode) => {
            let change = bridge.set_global_default_supported(mode, connection.supports_jev_features())?;
            crate::core::jev_bridge::invalidate_settings_cache();
            publish_footer(send, &bridge, &session_id, credential);
            Ok(automatic_status(send, mode_change_message(&change)))
        }
        JevRequest::SetFeature(feature, enabled) => {
            require_feature_support(connection.supports_jev_features())?;
            bridge.set_feature(&session_id, feature, enabled)?;
            crate::core::jev_bridge::invalidate_settings_cache();
            Ok(CommandOutput::Status(format!("Jev feature {}: {} (this chat). Mode and compaction are unchanged.",
                feature.as_str(), if enabled { "on" } else { "off" })))
        }
        JevRequest::SetCompaction(enabled) => {
            require_feature_support(connection.supports_jev_features())?;
            bridge.set_compaction(&session_id, enabled)?;
            crate::core::jev_bridge::invalidate_settings_cache();
            // The compaction dot refreshes in the same turn as the write, so the
            // row can never render the previous state. The decision segment is
            // republished from the same read and stays unchanged.
            publish_footer(send, &bridge, &session_id, credential);
            Ok(automatic_status(send, render_compaction_settings(&bridge.settings(), &session_id)))
        }
        JevRequest::SetDefaultCompaction(enabled) => {
            require_feature_support(connection.supports_jev_features())?;
            bridge.set_default_compaction(enabled)?;
            crate::core::jev_bridge::invalidate_settings_cache();
            // Sessions without an override see the new default immediately.
            publish_footer(send, &bridge, &session_id, credential);
            Ok(automatic_status(send, format!("Jev compaction default: {} (sessions without a compaction override).\n{}",
                if enabled { "on" } else { "off" }, render_compaction_settings(&bridge.settings(), &session_id))))
        }
        JevRequest::CompactionStatus => {
            let mut text = render_compaction_settings(&bridge.settings(), &session_id);
            if !connection.supports_jev_features() {
                text.push_str("Attached worker lacks Jev System One capability; this is local configuration only.\n");
            }
            Ok(CommandOutput::Panel(text))
        }
        JevRequest::Models => {
            // ROOT-CONTRACT v9: the EXPLICIT operator-initiated read-only
            // catalog query — the ONLY /jev command that may touch the
            // network, and only here: one bounded single-attempt GET
            // /v1/models through the existing transport/limits abstractions.
            // No prompt/history/tool data leaves; no selection; no settings
            // write; no mode/feature/compaction change; no budget effect. A
            // missing credential reports honest unavailability with NO fetch.
            let settings = bridge.settings();
            let Some((transport, limits)) =
                crate::core::jev_bridge::catalog_transport_for_command(&settings)
            else {
                return Ok(CommandOutput::Error(format!(
                    "Model catalog unavailable: no Jev credential is configured. Set one with /jev key. Nothing was fetched and no settings were changed.\n{JEV_BOUNDARY_NOTICE}"
                )));
            };
            match pi_jev::models::fetch_model_catalog(transport.as_ref(), &limits).await {
                Ok(catalog) => Ok(CommandOutput::Panel(render_model_catalog(&catalog))),
                Err(error) => Ok(CommandOutput::Error(format!(
                    "Model catalog unavailable ({}). Nothing was changed and nothing was selected.\n{JEV_BOUNDARY_NOTICE}",
                    error.log_line()
                ))),
            }
        }
        JevRequest::ModelStatus => {
            // LOCAL only: settings truth plus the in-process comparison
            // snapshot when this process holds it. No RPC, no catalog, no
            // network of any kind (ROOT-CONTRACT v9).
            let settings = bridge.settings();
            let reported = crate::core::jev_bridge::session_status_snapshot(&session_id)
                .as_ref()
                .and_then(|snapshot| snapshot.get("response_model"))
                .and_then(|value| value.as_str())
                .map(str::to_string);
            Ok(CommandOutput::Panel(render_model_status(&JevModelStatusReport {
                requested: settings.requested_model_or_default().to_string(),
                explicit: settings.requested_model.clone(),
                write_revision: settings.write_revision,
                reported,
            })))
        }
        JevRequest::ModelSet(id) => {
            if id.is_empty() {
                return Ok(CommandOutput::Error(format!(
                    "Usage: /jev model set <id>\n{JEV_BOUNDARY_NOTICE}"
                )));
            }
            // The effective credential is loaded ONLY for the overlap refusal
            // check; it is never logged, echoed or persisted (ROOT-CONTRACT
            // v9). No network call, no probe, no availability claim.
            let overlap_secret = crate::core::jev_bridge::credential_for_model_overlap();
            let outcome = bridge.set_requested_model(&id, overlap_secret.as_ref())?;
            crate::core::jev_bridge::invalidate_settings_cache();
            Ok(CommandOutput::Status(if outcome.written {
                format!(
                    "Requested Jev model: {}. Native SystemOne requests now carry it; the primary chat model/provider stays authoritative. No network call was made and no budget was changed.\n{JEV_BOUNDARY_NOTICE}",
                    outcome.requested
                )
            } else {
                format!(
                    "Requested Jev model is already {}; nothing was written (the durable write revision did not move).\n{JEV_BOUNDARY_NOTICE}",
                    outcome.requested
                )
            }))
        }
        JevRequest::ModelReset => {
            let outcome = bridge.reset_requested_model()?;
            crate::core::jev_bridge::invalidate_settings_cache();
            Ok(CommandOutput::Status(if outcome.written {
                format!(
                    "Requested Jev model reset to the native default {}. Nothing was probed, no network call was made and no budget was changed.\n{JEV_BOUNDARY_NOTICE}",
                    outcome.requested
                )
            } else {
                format!(
                    "Requested Jev model was already the native default {}; nothing was written.\n{JEV_BOUNDARY_NOTICE}",
                    outcome.requested
                )
            }))
        }
        JevRequest::SetFullJev(enabled) => {
            let change = bridge.set_full_jev(enabled)?;
            crate::core::jev_bridge::invalidate_settings_cache();
            // Same-turn refresh: the footer must render the new effective
            // state in the same turn as the write, exactly like every other
            // /jev write. Other chats refresh at their next footer boundary.
            publish_footer(send, &bridge, &session_id, credential);
            Ok(automatic_status(send, match change {
                FullJevChange::Installed { already_active: false } => format!(
                    "{JEV_FULL_JEV_ON_NOTICE}\n{JEV_DISCLOSURE_NOTICE}\n{JEV_ACTIVE_NOTICE}\n{JEV_BOUNDARY_NOTICE}{}",
                    if credential.present() {
                        String::new()
                    } else {
                        "\nNo API key is configured: the footer shows unavailable and every decision fails closed until /jev key.".to_string()
                    }
                ),
                FullJevChange::Installed { already_active: true } => format!(
                    "Full-jev is already active; nothing was changed. Use /jev full-jev off to remove it.\n{JEV_DISCLOSURE_NOTICE}"
                ),
                FullJevChange::Removed { was_active: true } => JEV_FULL_JEV_OFF_NOTICE.to_string(),
                FullJevChange::Removed { was_active: false } => {
                    JEV_FULL_JEV_ALREADY_OFF_NOTICE.to_string()
                }
            }))
        }
        JevRequest::FullJevStatus => {
            let settings = bridge.settings();
            Ok(CommandOutput::Panel(render_full_jev_status(
                &settings,
                &session_id,
                credential,
            )))
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
            publish_footer(send, &bridge, &session_id, credential);
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
                let change = bridge.set_session_mode_supported(session_id, mode, connection.supports_jev_features())?;
                crate::core::jev_bridge::invalidate_settings_cache();
                publish_footer(send, bridge, session_id, credential);
                let _ = send.send(HostEvent::CloseCommandDialog);
                return Ok(automatic_status(send, mode_change_message(&change)));
            }
            JevMenuAction::SetCompaction(enabled) => {
                require_feature_support(connection.supports_jev_features())?;
                bridge.set_compaction(session_id, enabled)?;
                crate::core::jev_bridge::invalidate_settings_cache();
                // Same-turn refresh, identical to the command path.
                publish_footer(send, bridge, session_id, credential);
                let _ = send.send(HostEvent::CloseCommandDialog);
                return Ok(automatic_status(send, render_compaction_settings(&bridge.settings(), session_id)));
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
            let bridge = JevModeBridge::new(&agent_dir());
            publish_footer(send, &bridge, session_id, now);
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

#[cfg(test)]
mod disclosure_tests {
    use super::super::super::native_extensions::{Surfaces, Widgets};
    use super::*;

    fn apply_widget(event: HostEvent, surfaces: &Rc<RefCell<Surfaces>>) {
        let HostEvent::Connection(wire::AgentConnectionEvent::ExtensionUiRequest { request }) =
            event
        else {
            panic!("expected a display-only widget event");
        };
        assert_eq!(request.method, "setWidget");
        let lines = request.payload["widgetLines"].as_array().map(|lines| {
            lines
                .iter()
                .map(|line| line.as_str().unwrap().to_string())
                .collect()
        });
        surfaces.borrow_mut().set_widget(
            request.payload["widgetKey"].as_str().unwrap().to_string(),
            lines,
            false,
        );
    }

    fn render(surfaces: &Rc<RefCell<Surfaces>>) -> String {
        Widgets(surfaces.clone(), false).render(160.0).join("\n")
    }

    async fn expiry(receive: &mpsc::Receiver<HostEvent>) -> HostEvent {
        tokio::time::timeout(Duration::from_secs(6), async {
            loop {
                match receive.try_recv() {
                    Ok(event) => return event,
                    Err(mpsc::TryRecvError::Empty) => {
                        tokio::time::sleep(Duration::from_millis(5)).await
                    }
                    Err(error) => panic!("notice channel closed before expiry: {error}"),
                }
            }
        })
        .await
        .expect("notice must expire without keyboard input")
    }

    #[tokio::test]
    async fn automatic_disclosure_disappears_after_five_seconds_without_input() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let (send, receive) = mpsc::channel();
        let surfaces = Rc::new(RefCell::new(Surfaces::default()));
        surfaces.borrow_mut().set_widget(
            "unrelated".into(),
            Some(vec!["Unrelated widget".into()]),
            false,
        );
        let started = std::time::Instant::now();
        let output = automatic_status(
            &send,
            format!("{JEV_FULL_JEV_ON_NOTICE}\n{JEV_DISCLOSURE_NOTICE}"),
        );
        assert!(
            matches!(output, CommandOutput::Nothing),
            "disclosure must not enter permanent chat status"
        );
        apply_widget(receive.try_recv().unwrap(), &surfaces);
        assert!(render(&surfaces).contains("Disclosure: Jev"));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(matches!(receive.try_recv(), Err(mpsc::TryRecvError::Empty)));
        apply_widget(expiry(&receive).await, &surfaces);
        assert!(started.elapsed() >= Duration::from_secs(5));
        assert!(!render(&surfaces).contains("Disclosure: Jev"));
        assert!(render(&surfaces).contains("Unrelated widget"));
    }

    #[tokio::test]
    async fn older_expiry_does_not_remove_a_newer_notice_after_session_reset() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let (send, receive) = mpsc::channel();
        let surfaces = Rc::new(RefCell::new(Surfaces::default()));
        show_disclosure(&send, "Old notice".into(), Duration::from_millis(20));
        apply_widget(receive.try_recv().unwrap(), &surfaces);
        surfaces.borrow_mut().reset();
        show_disclosure(&send, "New notice".into(), Duration::from_millis(150));
        apply_widget(receive.try_recv().unwrap(), &surfaces);
        apply_widget(expiry(&receive).await, &surfaces);
        assert!(render(&surfaces).contains("New notice"));
        apply_widget(expiry(&receive).await, &surfaces);
        assert!(render(&surfaces).is_empty());
    }

    #[test]
    fn brief_status_messages_keep_their_existing_lifetime() {
        let (send, receive) = mpsc::channel();
        let output = automatic_status(&send, "Jev mode: Off".into());
        assert!(matches!(output, CommandOutput::Status(message) if message == "Jev mode: Off"));
        assert!(matches!(receive.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }
}
