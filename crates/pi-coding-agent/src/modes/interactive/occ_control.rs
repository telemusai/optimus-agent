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
}

fn parse(args: &str) -> Result<Choice, String> {
    let words: Vec<_> = args.split_whitespace().collect();
    match words.as_slice() {
        [] => Ok(Choice::Menu),
        ["status"] => Ok(Choice::Status),
        ["1"] | ["occ1"] | ["off"] | ["2", "off"] | ["occ2", "off"] => Ok(Choice::Set(false)),
        ["2"] | ["occ2"] | ["on"] | ["2", "on"] | ["occ2", "on"] => Ok(Choice::Set(true)),
        ["3", "on"] | ["occ3", "on"] => Ok(Choice::SetOcc3(true)),
        ["3", "off"] | ["occ3", "off"] | ["3"] | ["occ3"] => Ok(Choice::SetOcc3(false)),
        _ => Err("Usage: /occ [status|1|2 on|2 off|3 on|3 off]. OCC-1 remains enabled as the fallback; OCC-2 is enabled by default; OCC-3 requires a live connector handshake.".into()),
    }
}

fn status(enabled: bool) -> String {
    status_occ3(enabled, false)
}

fn status_occ3(enabled: bool, occ3_enabled: bool) -> String {
    format!(
        "Agent communication (this chat)\n\nOCC-1: ON (fallback)\nOCC-2: {} (adaptive symbolic encoding; default ON)\nOCC-3: {} (chat preference; same-exact-model native handoffs)\n\n/occ 2 on enables OCC-2; /occ 2 off returns to OCC-1; /occ 3 on saves the OCC-3 preference; /occ 3 off blocks native handoffs for this chat.\nNative OCC-3 still requires agentCommunication.occ3Enabled and occ3Experimental, a configured model endpoint, and a successful live connector handshake. An explicit chat preference gates native actions dynamically; it does not install a missing connector. Without a chat override, an installed native connector follows the existing global opt-in. Agents use occ3_handoff only when the runtime exposes it; ordinary messages retain their OCC-2/OCC-1/TXT path.\nSaved for this chat and inherited by its subagents unless they have an override. Other chats are unchanged.\nTXT / OCC1 / OCC2 indicators reflect delivered messages; OCC3 requires a completed native-use receipt, not just enablement or a queued handoff.",
        if enabled { "ON" } else { "OFF" },
        if occ3_enabled { "ON" } else { "OFF" }
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

/// Host-authoritative OCC-2 status and provenance, alongside the existing
/// chat-local OCC-3 preference.
fn status_of(payload: &serde_json::Value, occ3_enabled: bool) -> Result<String, String> {
    let mut panel = status_occ3(enabled_of(payload)?, occ3_enabled);
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
            Ok(CommandOutput::Status(status_of(&value, false)?))
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
    // Match the existing chat-local OCC-3 write path; the OCC-2 daemon
    // preference payload does not include OCC-3.
    let state = connection.get_state().await?;
    let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
    let preference = crate::core::occ::session_settings::resolve(session_file)?;
    let enabled = if preference.source_session_id.is_some() {
        preference.occ3_enabled
    } else {
        state.active_tool_names.iter().any(|name| name == "occ3_handoff")
    };
    Ok((enabled, model_supports_occ3(state.model.as_ref())))
}

pub(super) async fn run(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &mpsc::Sender<HostEvent>,
    args: &str,
) -> Result<CommandOutput, String> {
    let choice = parse(args)?;
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
            let (occ3_enabled, supported) = read_occ3_preference(connection).await?;
            let mut panel = status_of(&payload, occ3_enabled)?;
            panel.push_str(if supported {
                "\nSelected model: OCC-3 configured. The receiver must also be OCC-3 configured with matching verified model/runtime identity."
            } else {
                "\nSelected model: OCC-3 unavailable. Use an explicitly configured OCC-3 model for both subagents."
            });
            return Ok(CommandOutput::Panel(panel));
        }
        Choice::Menu => {
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
        return Ok(CommandOutput::Status(status_occ3(enabled, value)));
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
    fn occ_status_renders_host_truth_with_provenance() {
        // A chat's own preference: no provenance line.
        let own = serde_json::json!({"occ2Enabled": true, "inherited": false, "sourceSessionId": "chat-1"});
        let panel = status_of(&own, true).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(panel.contains("OCC-3: ON"));
        assert!(panel.contains("An explicit chat preference gates native actions dynamically"));
        assert!(!panel.contains("Inherited from chat"));
        // A subagent chat: the daemon reports the ancestor as the source.
        let inherited = serde_json::json!({"occ2Enabled": true, "inherited": true, "sourceSessionId": "parent-7"});
        let panel = status_of(&inherited, false).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(panel.contains("OCC-3: OFF"));
        assert!(panel.contains("Inherited from chat parent-7"));
        // A host response without the toggle is an error, never a silent value.
        assert!(status_of(&serde_json::json!({}), false).is_err());
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
