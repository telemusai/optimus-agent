//! Chat-local OCC controls. The host that owns the session persists them:
//! daemons expose the capability-gated `occ_get_preferences` /
//! `occ_set_preferences` commands (schema revision 41), so a client never
//! writes OCC-2 storage the daemon may not read. OCC-3 retains its existing
//! chat-local preference; native activation is configured separately.

use super::*;

#[derive(Debug, PartialEq)]
enum Choice {
    Menu,
    Status,
    Set(bool),
    SetOcc3(bool),
    /// `/occ queue`: the same queue view the status panel carries.
    Queue,
    /// `/occ queue cancel <queueId>`: manual resolution of one uncertain
    /// entry as terminal `failed` (manual_cancel). Never re-sends.
    QueueCancel(String),
    /// `/occ queue reemit <queueId>`: manual re-emit of one uncertain entry
    /// through ordinary OCC-1 delivery. Requires the explicit duplicate
    /// risk confirmation before the send.
    QueueReemit(String),
}

fn parse(args: &str) -> Result<Choice, String> {
    let words: Vec<_> = args.split_whitespace().collect();
    match words.as_slice() {
        [] => Ok(Choice::Menu),
        ["status"] => Ok(Choice::Status),
        ["queue"] => Ok(Choice::Queue),
        ["queue", "cancel", queue_id] => Ok(Choice::QueueCancel(queue_id.to_string())),
        ["queue", "reemit", queue_id] => Ok(Choice::QueueReemit(queue_id.to_string())),
        ["1"] | ["occ1"] | ["off"] | ["2", "off"] | ["occ2", "off"] => Ok(Choice::Set(false)),
        ["2"] | ["occ2"] | ["on"] | ["2", "on"] | ["occ2", "on"] => Ok(Choice::Set(true)),
        ["3", "on"] | ["occ3", "on"] => Ok(Choice::SetOcc3(true)),
        ["3", "off"] | ["occ3", "off"] | ["3"] | ["occ3"] => Ok(Choice::SetOcc3(false)),
        _ => Err("Usage: /occ [status|queue|1|2 on|2 off|3 on|3 off|queue cancel <id>|queue reemit <id>]. OCC-1 remains enabled as the fallback; OCC-2 is enabled by default; OCC-3 requires a live connector handshake. queue cancel/reemit resolve one uncertain native handoff; reemit states the duplicate risk before sending.".into()),
    }
}

fn status(enabled: bool) -> String {
    status_occ3(enabled, false)
}

fn status_occ3(enabled: bool, occ3_enabled: bool) -> String {
    status_occ3_native(enabled, occ3_enabled, false)
}

/// `live_native`: this chat currently has an installed native OCC-3
/// enrollment (the occ3_handoff tool is active). Then OCC-3 is not merely
/// allowed - it is the preferred channel for compatible-family fact
/// exchange, and the panel says so.
fn status_occ3_native(enabled: bool, occ3_enabled: bool, live_native: bool) -> String {
    let occ3_state = if live_native {
        "PREFERRED (live native enrollment; occ3_handoff is the preferred path for compatible family fact exchange)"
    } else if occ3_enabled {
        "ON"
    } else {
        "OFF"
    };
    format!(
        "Agent communication (this chat)\n\nOCC-1: ON (fallback)\nOCC-2: {} (adaptive symbolic encoding; default ON)\nOCC-3: {} (chat preference; same-exact-model native handoffs)\n\n/occ 2 on enables OCC-2; /occ 2 off returns to OCC-1; /occ 3 on saves the OCC-3 preference; /occ 3 off blocks native handoffs for this chat.\nNative OCC-3 still requires agentCommunication.occ3Enabled and occ3Experimental, a configured model endpoint, and a successful live connector handshake. An explicit chat preference gates native actions dynamically; it does not install a missing connector. Without a chat override, an installed native connector follows the existing global opt-in. Agents use occ3_handoff only when the runtime exposes it; ordinary messages retain their OCC-2/OCC-1/TXT path.\nSaved for this chat and inherited by its subagents unless they have an override. Other chats are unchanged.\nTXT / OCC1 / OCC2 indicators reflect delivered messages; OCC3 requires a completed native-use receipt, not just enablement or a queued handoff.",
        if enabled { "ON" } else { "OFF" },
        occ3_state
    )
}

const UNSUPPORTED: &str = "This daemon does not support OCC-2 controls. Restart it with Optimus Agent v0.1.34 or later; OCC-1 remains available.";

/// How /occ reaches the preference storage for the attached host.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    /// The host owns the get/set (v0.1.34+ daemons; in-process hosts).
    Remote,
    /// A v0.1.33 daemon: it has OCC-2 encoding but no session-preference
    /// commands. Keep the historical local write keyed on the daemon-reported
    /// session file, and never claim silent success.
    LegacyLocal,
    /// No OCC-2 support at all.
    Unsupported,
}

fn route(supports_session_preferences: bool, supports_occ2_communication: bool) -> Route {
    if supports_session_preferences {
        Route::Remote
    } else if supports_occ2_communication {
        Route::LegacyLocal
    } else {
        Route::Unsupported
    }
}

/// The effective toggle plus its provenance as the host reports it: the full
/// `occ_get_preferences` payload for Remote routes (so `/occ status` can show
/// inheritance), just the value for legacy hosts.
async fn read_preference(
    connection: &Arc<dyn wire::AgentConnection>,
    route: &Route,
) -> Result<serde_json::Value, String> {
    match route {
        Route::Remote => connection.get_occ_preferences().await,
        Route::LegacyLocal => {
            let state = connection.get_state().await?;
            let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
            let resolution = crate::core::occ::session_settings::resolve(session_file)?;
            let occ2_enabled = legacy_enabled(&resolution);
            Ok(serde_json::json!({ "occ2Enabled": occ2_enabled }))
        }
        Route::Unsupported => Err(UNSUPPORTED.to_string()),
    }
}

// A v0.1.33 daemon still defaults off until a preference is explicitly saved.
fn legacy_enabled(resolution: &crate::core::occ::session_settings::Resolution) -> bool {
    resolution.source_session_id.is_some() && resolution.occ2_enabled
}

/// The effective toggle from a host preference payload.
fn enabled_of(payload: &serde_json::Value) -> Result<bool, String> {
    payload
        .get("occ2Enabled")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| "Daemon returned an invalid OCC-2 preference".to_string())
}

/// One bounded queue-table row: position, receiver, state, attempt count
/// and both stable ids. The handoff message id stays an explicit
/// placeholder until the daemon mints it; it is never fabricated.
fn queue_table_row(entry: &crate::core::occ::queue::CommunicationQueueEntry) -> String {
    let mut row = format!(
        "  {}. {} \u{00b7} {} \u{00b7} attempts {} \u{00b7} queue {} \u{00b7} msg {}",
        entry.position,
        entry.receiver,
        crate::core::occ::queue::display_state(&entry.state),
        entry.attempts,
        entry.queue_id,
        if entry.message_id.is_empty() { "\u{2014}" } else { entry.message_id.as_str() }
    );
    // The fallback transport verdict, stated honestly: a queued ordinary
    // delivery is an acknowledgment, not acceptance.
    if let Some(ordinary_delivery) = entry.ordinary_delivery.as_deref() {
        row.push_str(&format!(" \u{00b7} ordinary delivery {ordinary_delivery}"));
    }
    // The manual resolution that drained an uncertain entry, when the
    // notice names one (whitelisted upstream; anything else is absent).
    if let Some(reason) = entry.reason.as_deref() {
        let label = match reason {
            "manual_reemit" => "manual re-emit",
            "manual_cancel" => "manual cancel",
            other => other,
        };
        row.push_str(&format!(" \u{00b7} {label}"));
    }
    row
}

/// Host-authoritative OCC-2 status and provenance, alongside the existing
/// chat-local OCC-3 preference. `queue` is this chat's own native handoff
/// queue, projected from its notices; the family-wide count lives in the
/// subagents bar badge.
fn status_of(
    payload: &serde_json::Value,
    occ3_enabled: bool,
    queue: Option<&crate::core::occ::queue::CommunicationQueueSnapshot>,
) -> Result<String, String> {
    status_of_native(payload, occ3_enabled, false, queue)
}

/// `status_of` with the chat's native-enrollment liveness, so the panel
/// reports OCC-3 as the preferred channel when the connector is live.
fn status_of_native(
    payload: &serde_json::Value,
    occ3_enabled: bool,
    live_native: bool,
    queue: Option<&crate::core::occ::queue::CommunicationQueueSnapshot>,
) -> Result<String, String> {
    let mut panel = status_occ3_native(enabled_of(payload)?, occ3_enabled, live_native);
    if let Some(queue) = queue {
        // The whole-queue blocked state first: a queue whose persisted
        // artifact could not be loaded safely refuses new handoffs, and the
        // user must see that before any entry detail.
        if let Some(reason) = queue.blocked.as_deref() {
            panel.push_str(&format!(
                "\nNative handoff queue BLOCKED ({reason}): the persisted queue file could not be read safely. New handoffs are refused until the model starts a fresh queue; the file is left untouched for manual or future-build recovery."
            ));
        }
        // The newest enqueue refusal, when it is still the newest event:
        // a handoff was refused (full or blocked), stated without claiming
        // it still holds beyond what the record proves.
        if let Some(refused) = &queue.refused {
            panel.push_str(&format!(
                "\nEnqueue refused ({}): a native handoff was not queued. The banner clears once later activity proves admission resumed.",
                refused.reason
            ));
        }
        // The durable notice outbox hit its bound: some queue notices are
        // undeliverable until it drains. Visible fail state, never silent.
        if queue.notice_overflow {
            panel.push_str(
                "\nNotice outbox overflow: the durable queue-notice outbox is full, so some state notices cannot be delivered until it drains. The queue state itself is unaffected.",
            );
        }
        if !queue.entries.is_empty() {
            panel.push_str(&format!(
                "\nNative handoff queue: {} waiting (pending/armed; the blue QUEUED marker flashes while waiting, independently of the OCC3 completed-use delivery pulse).",
                queue.pending_total
            ));
            for entry in &queue.entries {
                panel.push('\n');
                panel.push_str(&queue_table_row(entry));
            }
            if queue.omitted > 0 {
                panel.push_str(&format!("\n  +{} more queued", queue.omitted));
            }
        }
        // Actionable guidance where the user looks: uncertain entries
        // resolve ONLY through explicit manual resolution, so the panel
        // names the commands instead of leaving a dead queue unexplained.
        let uncertain: Vec<&crate::core::occ::queue::CommunicationQueueEntry> = queue
            .entries
            .iter()
            .filter(|entry| entry.state == "uncertain" || entry.state == "reconciling" || entry.state == "blocked")
            .collect();
        if !uncertain.is_empty() {
            panel.push_str(&format!(
                "\n{} uncertain entr{} awaiting explicit resolution: /occ queue reemit <id> re-sends via ordinary OCC-1 delivery (the receiver may have already consumed it - duplicates are possible) or /occ queue cancel <id> abandons it (never re-sent). No automatic retry or duplicate will occur.",
                uncertain.len(),
                if uncertain.len() == 1 { "y is" } else { "ies are" }
            ));
        }
    }
    if payload.get("inherited").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        if let Some(source) = payload.get("sourceSessionId").and_then(serde_json::Value::as_str) {
            panel.push_str(&format!(
                "\nInherited from chat {source}: this chat has no own override, so /occ 2 off here saves one."
            ));
        }
    }
    Ok(panel)
}

async fn write_preference(
    connection: &Arc<dyn wire::AgentConnection>,
    route: &Route,
    enabled: bool,
) -> Result<CommandOutput, String> {
    match route {
        Route::Remote => {
            let value = connection.set_occ_preferences(enabled).await?;
            // The daemon-authoritative echo, rendered with any provenance it
            // reports; an invalid echo is an error, never a success panel.
            Ok(CommandOutput::TransientStatus(status_of(&value, false, None)?))
        }
        Route::LegacyLocal => {
            let state = connection.get_state().await?;
            let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
            crate::core::occ::session_settings::save(session_file, enabled)?;
            // Never a silent success: a v0.1.33 daemon may read session
            // preferences (same machine) or may not (older storage); the
            // warning keeps both outcomes honest while OCC-1 stays available.
            Ok(CommandOutput::Warning(format!(
                "Saved the OCC-2 preference for this chat locally, but this daemon does not advertise OCC-2 session controls (Optimus Agent v0.1.34 or later). The preference takes effect only when the daemon reads session preferences; OCC-1 remains the default and fallback.\n\n{}",
                status(enabled)
            )))
        }
        Route::Unsupported => Ok(CommandOutput::Warning(UNSUPPORTED.into())),
    }
}

const MENU_CHOICES: [&str; 4] = [
    "OCC-1 only",
    "Enable OCC-2 (default)",
    "Enable OCC-3 preference (experimental)",
    "Disable OCC-3 preference",
];
const OCC3_UNAVAILABLE: &str = "OCC-3 unavailable for this model";
const OCC3_MODEL_REQUIRED: &str = "OCC-3 requires a model with an enabled, supported occ3 configuration in models.json. Select an OCC-3 model, such as your DGX MTP (OCC-3) entry. Both sender and receiver subagents must use compatible OCC-3 models and pass the live exact-model/runtime handshake; a matching model name alone is insufficient. Unsupported pairs use ordinary OCC-2/OCC-1/TXT communication.";

fn model_supports_occ3(model: Option<&pi_ai::types::Model>) -> bool {
    use crate::core::occ3_connectors::activation::{configured_connector, Occ3ActivationRequest};
    // Configuration eligibility only; the daemon still owns live activation.
    model.is_some_and(|model| {
        configured_connector(&Occ3ActivationRequest::new(model, true, true)).is_ok()
    })
}

fn menu_choice(value: &str) -> Result<Choice, String> {
    if value == OCC3_UNAVAILABLE {
        return Ok(Choice::SetOcc3(true));
    }
    match MENU_CHOICES.iter().position(|choice| *choice == value) {
        Some(0) => Ok(Choice::Set(false)),
        Some(1) => Ok(Choice::Set(true)),
        Some(2) => Ok(Choice::SetOcc3(true)),
        Some(3) => Ok(Choice::SetOcc3(false)),
        _ => Err("Unknown OCC selection".into()),
    }
}

async fn read_occ3_preference(connection: &Arc<dyn wire::AgentConnection>) -> Result<(bool, bool), String> {
    read_occ3_preference_native(connection).await.map(|(enabled, supported, _live)| (enabled, supported))
}

/// The chat's OCC-3 resolution with liveness: `(preference, model
/// supported, live native enrollment)`. `live` is true only when the
/// occ3_handoff tool is active in this chat - the connector enrolled and
/// the runtime exposes it - which is what makes OCC-3 the preferred
/// channel rather than merely allowed.
async fn read_occ3_preference_native(connection: &Arc<dyn wire::AgentConnection>) -> Result<(bool, bool, bool), String> {
    // Match the existing chat-local OCC-3 write path; the OCC-2 daemon
    // preference payload does not include OCC-3.
    let state = connection.get_state().await?;
    // The tool name lingers on a dead enrollment until the next turn removes
    // it, so liveness additionally requires the permit to be alive.
    let live = state.active_tool_names.iter().any(|name| name == "occ3_handoff")
        && !state.occ3_native_dead;
    let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
    let preference = crate::core::occ::session_settings::resolve(session_file)?;
    let enabled = if preference.source_session_id.is_some() {
        preference.occ3_enabled
    } else {
        live
    };
    Ok((enabled, model_supports_occ3(state.model.as_ref()), live))
}

/// This chat's queue entry with the given id, from the notice projection.
/// Best effort (the bounded display list): a miss does NOT prove the entry
/// is gone; the daemon's resolve API stays authoritative.
async fn queue_entry_of(
    connection: &Arc<dyn wire::AgentConnection>,
    queue_id: &str,
) -> Option<crate::core::occ::queue::CommunicationQueueEntry> {
    let messages = connection.get_messages().await.ok()?;
    if messages.is_empty() {
        return None;
    }
    crate::core::occ::queue::from_queue_notices(&messages, "", "")
        .entries
        .into_iter()
        .find(|entry| entry.queue_id == queue_id)
}

/// Renders the daemon's content-free resolve echo as the user-facing line.
fn resolution_summary(value: &serde_json::Value, queue_id: &str, reemit: bool, receiver: &str) -> String {
    let for_receiver = if receiver.is_empty() {
        String::new()
    } else {
        format!(" for {receiver}")
    };
    let state = value.get("state").and_then(serde_json::Value::as_str).unwrap_or("unknown");
    match (state, reemit) {
        ("failed", _) => format!(
            "Cancelled handoff {queue_id}{for_receiver}: marked failed (manual cancel). Nothing was re-sent."
        ),
        ("fallback_pending", true) => format!(
            "Re-emit armed for handoff {queue_id}{for_receiver}: the ordinary OCC-1 re-emission is pending and the model will be prompted for fresh source facts. The receiver may have already consumed the original handoff, so a duplicate delivery is possible."
        ),
        ("unchanged", _) => format!("No change for handoff {queue_id}: the entry had already resolved."),
        (state, _) => format!("Handoff {queue_id}{for_receiver} is now {state}."),
    }
}

/// One manual resolution of an uncertain queue entry. `reemit=false`
/// cancels (terminal `failed`, never re-sent); `reemit=true` re-arms the
/// ordinary OCC-1 re-emission and REQUIRES the explicit duplicate-risk
/// acknowledgment. The confirmation is a dialog the user must actively
/// choose; a dismissed dialog resolves nothing.
async fn queue_resolution(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    queue_id: &str,
    reemit: bool,
) -> Result<CommandOutput, String> {
    if !connection.supports_occ_queue_resolution() {
        return Ok(CommandOutput::Warning(
            "This session host does not support OCC-3 queue resolution (agent_communication_queue_control, schema revision 46). Update the daemon; until then, uncertain entries stay visible and unresolved - no automatic retry or duplicate will occur.".into(),
        ));
    }
    // The dialog states what it can prove from the projection; the
    // daemon's resolve API remains the authority on the entry's state.
    let entry = queue_entry_of(connection, queue_id).await;
    if let Some(entry) = &entry {
        if !matches!(entry.state.as_str(), "uncertain" | "reconciling" | "blocked") {
            return Ok(CommandOutput::Warning(format!(
                "Handoff {} is {} (not uncertain); only uncertain entries resolve manually. No change was made.",
                queue_id,
                crate::core::occ::queue::display_state(&entry.state)
            )));
        }
    }
    let receiver = entry.as_ref().map(|entry| entry.receiver.as_str()).unwrap_or("");
    let unknown_entry = entry.is_none().then(|| " The entry is not in the bounded queue view; the daemon will verify it.").unwrap_or("");
    let (title, decline, proceed) = if reemit {
        (
            format!(
                "Re-emit handoff {queue_id}{for_receiver} via ordinary OCC-1 delivery? The receiver may have already consumed it; re-emitting can duplicate the handoff.{unknown_entry}",
                for_receiver = if receiver.is_empty() { String::new() } else { format!(" to {receiver}") },
            ),
            "Don't re-emit",
            "Re-emit anyway",
        )
    } else {
        (
            format!(
                "Cancel handoff {queue_id}{for_receiver}? It will be marked failed (manual cancel) and never re-sent.{unknown_entry}",
                for_receiver = if receiver.is_empty() { String::new() } else { format!(" to {receiver}") },
            ),
            "Keep it",
            "Cancel handoff",
        )
    };
    let Some(choice) = select(send, &title, &[decline, proceed]).await else {
        return Ok(CommandOutput::Nothing);
    };
    if choice != proceed {
        return Ok(CommandOutput::Nothing);
    }
    match connection.resolve_occ_queue_entry(queue_id, reemit).await {
        Ok(value) => Ok(CommandOutput::TransientStatus(resolution_summary(&value, queue_id, reemit, receiver))),
        Err(error) => Ok(CommandOutput::Warning(format!("Queue resolution refused: {error}"))),
    }
}

pub(super) async fn run(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    args: &str,
) -> Result<CommandOutput, String> {
    let choice = parse(args)?;
    // Manual queue resolution is gated on its own connection capability,
    // not on the OCC-2 preference route: resolve before any preference
    // read. `/occ queue` itself is the status panel's queue view.
    let choice = match choice {
        Choice::QueueCancel(queue_id) => {
            return queue_resolution(connection, send, &queue_id, false).await;
        }
        Choice::QueueReemit(queue_id) => {
            return queue_resolution(connection, send, &queue_id, true).await;
        }
        Choice::Queue => Choice::Status,
        other => other,
    };
    let route = route(
        connection.supports_occ_session_preferences(),
        connection.supports_occ2_communication(),
    );
    if route == Route::Unsupported {
        return Ok(CommandOutput::Warning(UNSUPPORTED.into()));
    }
    let payload = read_preference(connection, &route).await?;
    let enabled = enabled_of(&payload)?;
    let choice = match choice {
        Choice::Status => {
            let (occ3_enabled, supported, live_native) = read_occ3_preference_native(connection).await?;
            // This chat's own native handoff queue, projected from its
            // notices; a failed read renders the panel without the queue
            // section instead of failing the status command. The state read
            // awaits before the closure so no async block hides inside
            // and_then (E0728).
            let session_id = connection.get_state().await.ok()
                .and_then(|state| (!state.session_id.is_empty()).then_some(state.session_id))
                .unwrap_or_default();
            let queue = match connection.get_messages().await {
                Ok(messages) if !messages.is_empty() => Some(
                    crate::core::occ::queue::from_queue_notices(&messages, &session_id, ""),
                ),
                _ => None,
            };
            let mut panel = status_of_native(&payload, occ3_enabled, live_native, queue.as_ref())?;
            panel.push_str(if supported {
                "\nSelected model: OCC-3 configured. The receiver must also be OCC-3 configured with matching verified model/runtime identity."
            } else {
                "\nSelected model: OCC-3 unavailable. Use an explicitly configured OCC-3 model for both subagents."
            });
            return Ok(CommandOutput::Panel(panel));
        }
        Choice::Menu => {
            // The menu title intentionally shows the saved ON/OFF preference,
            // not the live PREFERRED state: the menu toggles the preference,
            // while the status panel reports liveness. The two labels can
            // differ for the same chat by design (preference ON + tool
            // present = PREFERRED in the panel, ON in the menu).
            let (occ3_enabled, supported) = read_occ3_preference(connection).await?;
            let title = format!(
                "OCC-1 ON · OCC-2 {} · OCC-3 {} (preference)",
                if enabled { "ON" } else { "OFF" },
                if occ3_enabled { "ON" } else { "OFF" },
            );
            let mut choices = MENU_CHOICES;
            if !supported {
                choices[2] = OCC3_UNAVAILABLE;
            }
            match select(send, &title, &choices).await {
                Some(value) => menu_choice(&value)?,
                None => return Ok(CommandOutput::Nothing),
            }
        }
        choice => choice,
    };
    // OCC-3 keeps its existing chat-local storage path. This does not
    // change the daemon protocol or install a native connector.
    if let Choice::SetOcc3(value) = choice {
        let state = connection.get_state().await?;
        if value && !model_supports_occ3(state.model.as_ref()) {
            return Ok(CommandOutput::Warning(OCC3_MODEL_REQUIRED.into()));
        }
        let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
        let resolution = crate::core::occ::session_settings::resolve(session_file)?;
        crate::core::occ::session_settings::save_occ3(
            session_file,
            resolution.occ2_enabled,
            value,
        )?;
        return Ok(CommandOutput::TransientStatus(status_occ3(enabled, value)));
    }
    match choice {
        Choice::Set(value) => write_preference(connection, &route, value).await,
        _ => unreachable!("menu, status and OCC-3 handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ_control_accepts_modes_and_rejects_unsupported_or_ambiguous_input() {
        assert_eq!(parse("").unwrap(), Choice::Menu);
        assert_eq!(parse("status").unwrap(), Choice::Status);
        for args in ["1", "off", "2 off", "occ2 off"] {
            assert_eq!(parse(args).unwrap(), Choice::Set(false));
        }
        for args in ["2", "on", "2 on", "occ2 on"] {
            assert_eq!(parse(args).unwrap(), Choice::Set(true));
        }
        for args in ["1 off", "2 maybe", "2 on off"] {
            assert!(parse(args).is_err());
        }
        for args in ["3 on", "occ3 on"] {
            assert_eq!(parse(args).unwrap(), Choice::SetOcc3(true));
        }
        for args in ["3 off", "occ3 off", "3", "occ3"] {
            assert_eq!(parse(args).unwrap(), Choice::SetOcc3(false));
        }
    }

    #[test]
    fn occ_control_accepts_queue_grammar_and_rejects_broken_forms() {
        assert_eq!(parse("queue").unwrap(), Choice::Queue);
        assert_eq!(
            parse("queue cancel occ3q_1").unwrap(),
            Choice::QueueCancel("occ3q_1".into())
        );
        assert_eq!(
            parse("queue reemit occ3q_2").unwrap(),
            Choice::QueueReemit("occ3q_2".into())
        );
        // Missing or extra words never resolve anything.
        for args in ["queue cancel", "queue reemit", "queue cancel a b", "queue maybe x", "queue x cancel"] {
            assert!(parse(args).is_err(), "args {args:?} must not parse");
        }
    }

    #[test]
    fn queue_resolution_summary_renders_each_post_state_honestly() {
        let cancelled = serde_json::json!({"queueId": "occ3q_1", "state": "failed", "reason": "manual_cancel"});
        let text = resolution_summary(&cancelled, "occ3q_1", false, "child:receiver-a");
        assert!(text.contains("Cancelled handoff occ3q_1 for child:receiver-a"));
        assert!(text.contains("Nothing was re-sent."));
        // The re-emit summary states the duplicate risk explicitly.
        let armed = serde_json::json!({"queueId": "occ3q_2", "state": "fallback_pending", "reason": "manual_reemit"});
        let text = resolution_summary(&armed, "occ3q_2", true, "sibling:worker-b");
        assert!(text.contains("Re-emit armed for handoff occ3q_2 for sibling:worker-b"));
        assert!(text.contains("duplicate delivery is possible"));
        // An already-resolved entry reports no change, never a fake success.
        let unchanged = serde_json::json!({"queueId": "occ3q_3", "state": "unchanged"});
        let text = resolution_summary(&unchanged, "occ3q_3", false, "");
        assert!(text.contains("No change for handoff occ3q_3"));
        assert!(!text.contains("Cancelled"));
        // An unknown receiver stays absent, never fabricated.
        let text = resolution_summary(&cancelled, "occ3q_1", false, "");
        assert!(!text.contains(" for "));
    }

    #[test]
    fn queue_status_names_the_manual_resolution_commands_for_uncertain_entries() {
        let own = serde_json::json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "chat-1"});
        let queue = crate::core::occ::queue::CommunicationQueueSnapshot {
            generation: "g".into(),
            pending_total: 1,
            omitted: 0,
            entries: vec![crate::core::occ::queue::CommunicationQueueEntry {
                queue_id: "occ3q_unc".into(),
                message_id: String::new(),
                position: 1,
                receiver: "child:receiver-a".into(),
                state: "uncertain".into(),
                attempts: 1,
                timestamp: 1000,
                ordinary_delivery: None,
                reason: None,
            }],
            blocked: None,
            refused: None,
            notice_overflow: false,
        };
        let panel = status_of(&own, true, Some(&queue)).unwrap();
        assert!(panel.contains("uncertain (pending receipt)"));
        assert!(panel.contains("1 uncertain entry is awaiting explicit resolution"));
        assert!(panel.contains("/occ queue reemit <id>"));
        assert!(panel.contains("/occ queue cancel <id>"));
        assert!(panel.contains("No automatic retry or duplicate will occur."));
    }

    #[test]
    fn queue_status_surfaces_blocked_and_refused_banners_before_the_entries() {
        let own = serde_json::json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "chat-1"});
        let queue = crate::core::occ::queue::CommunicationQueueSnapshot {
            generation: "g".into(),
            pending_total: 0,
            omitted: 0,
            entries: Vec::new(),
            blocked: Some("unsupported_schema".into()),
            refused: Some(crate::core::occ::queue::CommunicationQueueRefusal {
                reason: "queue_full".into(),
                timestamp: 1500,
            }),
            notice_overflow: false,
        };
        let panel = status_of(&own, true, Some(&queue)).unwrap();
        assert!(panel.contains("Native handoff queue BLOCKED (unsupported_schema)"));
        assert!(panel.contains("New handoffs are refused until the model starts a fresh queue"));
        assert!(panel.contains("Enqueue refused (queue_full)"));
        // No entry rows, no queue count line, no uncertain guidance.
        assert!(!panel.contains("Native handoff queue: 0 pending"));
        assert!(!panel.contains("uncertain"));

        // The outbox overflow banner rides the same projection.
        let overflow = crate::core::occ::queue::CommunicationQueueSnapshot {
            notice_overflow: true,
            ..queue.clone()
        };
        let panel = status_of(&own, true, Some(&overflow)).unwrap();
        assert!(panel.contains("Notice outbox overflow"));
        let recovered = crate::core::occ::queue::CommunicationQueueSnapshot {
            notice_overflow: false,
            ..overflow
        };
        assert!(!status_of(&own, true, Some(&recovered)).unwrap().contains("Notice outbox overflow"));
    }

    #[test]
    fn legacy_daemon_keeps_its_default_until_an_explicit_preference() {
        use crate::core::occ::session_settings::Resolution;
        assert!(!legacy_enabled(&Resolution::default()));
        for enabled in [false, true] {
            assert_eq!(legacy_enabled(&Resolution {
                occ2_enabled: enabled,
                occ3_enabled: false,
                source_session_id: Some("chat".into()),
            }), enabled);
        }
    }

    #[test]
    fn live_native_enrollment_reports_occ3_as_preferred() {
        // The panel distinguishes a live native enrollment (preferred
        // channel) from a merely-saved chat preference (ON) and an explicit
        // opt-out (OFF).
        let preferred = status_occ3_native(true, true, true);
        assert!(preferred.contains("OCC-3: PREFERRED (live native enrollment"));
        assert!(preferred.contains("occ3_handoff is the preferred path for compatible family fact exchange"));
        // Without liveness the old wording is preserved exactly.
        assert_eq!(status_occ3_native(true, true, false), status_occ3(true, true));
        assert!(status_occ3_native(true, false, false).contains("OCC-3: OFF"));
    }

    #[test]
    fn occ_status_renders_host_truth_with_provenance() {
        // A chat's own preference: no provenance line.
        let own = serde_json::json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "chat-1"});
        let queue = crate::core::occ::queue::CommunicationQueueSnapshot {
            generation: "g".into(),
            pending_total: 2,
            omitted: 1,
            entries: vec![
                crate::core::occ::queue::CommunicationQueueEntry {
                    queue_id: "occ3q_1".into(),
                    message_id: String::new(),
                    position: 1,
                    receiver: "child:receiver-a".into(),
                    state: "armed".into(),
                    attempts: 1,
                    timestamp: 1000,
                    ordinary_delivery: None,
                    reason: None,
                },
                crate::core::occ::queue::CommunicationQueueEntry {
                    queue_id: "occ3q_2".into(),
                    message_id: "m-2".into(),
                    position: 2,
                    receiver: "sibling:worker-b".into(),
                    state: "fallback_pending".into(),
                    attempts: 0,
                    timestamp: 2000,
                    ordinary_delivery: Some("queued".into()),
                    reason: Some("manual_reemit".into()),
                },
            ],
            blocked: None,
            refused: None,
            notice_overflow: false,
        };
        let panel = status_of(&own, true, Some(&queue)).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(panel.contains("OCC-3: ON"));
        assert!(panel.contains("An explicit chat preference gates native actions dynamically"));
        assert!(!panel.contains("Inherited from chat"));
        // Count line uses the full pending total, not the bounded list.
        assert!(panel.contains("Native handoff queue: 2 waiting"));
        // Per-entry rows: position, receiver, state (sender's UI labels,
        // corrected: admitted is an honest transport acknowledgment, never
        // consumed; unfinished re-emission renders "fallback (pending)"),
        // attempts, both ids.
        assert!(panel.contains("1. child:receiver-a"));
        assert!(panel.contains("armed"));
        assert!(panel.contains("attempts 1"));
        assert!(panel.contains("queue occ3q_1"));
        // Unminted message id stays an explicit placeholder, never fabricated.
        assert!(panel.contains("msg \u{2014}"));
        assert!(panel.contains("2. sibling:worker-b"));
        assert!(panel.contains("fallback (pending)"));
        assert!(panel.contains("msg m-2"));
        // The fallback row states its ordinary-delivery verdict honestly,
        // plus the manual resolution that drained its uncertain entry.
        assert!(panel.contains("ordinary delivery queued"));
        assert!(panel.contains("manual re-emit"));
        // The display bound is explicit, never silent.
        assert!(panel.contains("+1 more queued"));
        // The count badge line still appears with the queue table.
        assert!(panel.contains("QUEUED"));
        // A subagent chat: the daemon reports the ancestor as the source.
        let inherited = serde_json::json!({"occ2Enabled": true, "inherited": true, "sourceSessionId": "parent-7"});
        let panel = status_of(&inherited, false, None).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(panel.contains("OCC-3: OFF"));
        assert!(panel.contains("Inherited from chat parent-7"));
        // A host response without the toggle is an error, never a silent value.
        assert!(status_of(&serde_json::json!({}), false, None).is_err());
    }

    #[test]
    fn occ_menu_exposes_both_occ3_actions_without_changing_occ2_selection() {
        assert_eq!(menu_choice(MENU_CHOICES[0]).unwrap(), Choice::Set(false));
        assert_eq!(menu_choice(MENU_CHOICES[1]).unwrap(), Choice::Set(true));
        assert_eq!(menu_choice(MENU_CHOICES[2]).unwrap(), Choice::SetOcc3(true));
        assert_eq!(menu_choice(MENU_CHOICES[3]).unwrap(), Choice::SetOcc3(false));
        assert!(menu_choice("unknown").is_err());
    }

    #[test]
    fn occ3_menu_requires_explicit_supported_model_configuration() {
        use pi_ai::types::{Model, Occ3Capability, Occ3Validation, OCC3_API_VERSION_V1, OCC3_PROTOCOL_V1};

        assert!(!model_supports_occ3(None));
        let mut model = Model {
            id: "glm-5.3".into(),
            name: "GLM-5.3 - DGX MTP (OCC-3)".into(),
            ..Default::default()
        };
        // Even the expected display name cannot opt an ordinary model in.
        assert!(!model_supports_occ3(Some(&model)));
        model.occ3 = Some(Occ3Capability {
            protocol: OCC3_PROTOCOL_V1.into(),
            api_version: OCC3_API_VERSION_V1.into(),
            connector: "sglang".into(),
            endpoint: "http://127.0.0.1:30001".into(),
            same_model_only: true,
            enabled: true,
            validation: Occ3Validation::LiveVerified,
            fallback: Vec::new(),
            method: None,
            artifact_digest: None,
        });
        assert!(model_supports_occ3(Some(&model)));
        model.occ3.as_mut().unwrap().enabled = false;
        assert!(!model_supports_occ3(Some(&model)));
        model.occ3.as_mut().unwrap().enabled = true;
        model.occ3.as_mut().unwrap().same_model_only = false;
        assert!(!model_supports_occ3(Some(&model)));
        model.occ3.as_mut().unwrap().same_model_only = true;
        model.occ3.as_mut().unwrap().connector = "unsupported".into();
        assert!(!model_supports_occ3(Some(&model)));
    }

    #[test]
    fn occ_control_routes_by_host_capability_and_never_succeeds_silently_on_old_daemons() {
        // v0.1.34+ daemon: the daemon owns get/set.
        assert_eq!(route(true, true), Route::Remote);
        assert_eq!(route(true, false), Route::Remote);
        // v0.1.33 daemon: OCC-2 encoding without session-preference commands.
        assert_eq!(route(false, true), Route::LegacyLocal);
        // Older daemon: no OCC-2 at all.
        assert_eq!(route(false, false), Route::Unsupported);
    }

}
