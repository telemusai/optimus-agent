//! Chat-local OCC controls. The host that owns the session persists them:
//! daemons expose the capability-gated `occ_get_preferences` /
//! `occ_set_preferences` commands (schema revision 41), so a client never
//! writes storage the daemon may not read. Workers read session artifacts at
//! each send either way.

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
        "Agent communication (this chat)\n\nOCC-1: ON (fallback)\nOCC-2: {} (adaptive symbolic encoding; default ON)\nOCC-3: {} (latent, same-exact-model only; requires a live connector handshake)\n\n/occ 2 on enables OCC-2; /occ 2 off returns to OCC-1; /occ 3 on enables OCC-3 (falls back automatically when the receiver, model or connector is ineligible).\nSaved for this chat and inherited by its subagents unless they have an override. Other chats are unchanged. Applies to subsequent structured messages. Plain-text messages remain TXT.\nNeon-blue TXT / OCC1 / OCC2 indicators flash on delivered messages.",
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

/// The status panel from a host preference payload, with provenance when the
/// host reports the value was inherited from an ancestor chat (daemon truth,
/// not a client-local read).
fn status_of(payload: &serde_json::Value) -> Result<String, String> {
    let mut panel = status(enabled_of(payload)?);
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
            Ok(CommandOutput::Status(status_of(&value)?))
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

pub(super) async fn run(
    connection: &Arc<dyn wire::AgentConnection>,
    send: &HostEventSender,
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
    // OCC-3 writes are chat-local for v1: the daemon preference command
    // set is unchanged (capability-gated), so the toggle persists via
    // the session artifact the worker reads at send time.
    if let Choice::SetOcc3(value) = choice {
        let state = connection.get_state().await?;
        let session_file = state.session_file.as_deref().ok_or("This chat has no saved session identity")?;
        let resolution = crate::core::occ::session_settings::resolve(session_file)?;
        crate::core::occ::session_settings::save_occ3(
            session_file,
            resolution.occ2_enabled,
            value,
        )?;
        return Ok(CommandOutput::Status(status_occ3(enabled, value)));
    }
    let requested = match choice {
        Choice::Status => return Ok(CommandOutput::Panel(status_of(&payload)?)),
        Choice::Set(value) => value,
        Choice::SetOcc3(_) => unreachable!("handled above"),
        Choice::Menu => {
            let title = format!("OCC-1 ON · OCC-2 {}", if enabled { "ON" } else { "OFF" });
            match select(send, &title, &["OCC-1 only", "Enable OCC-2 (default)"]).await {
                Some(choice) => choice == "Enable OCC-2 (default)",
                None => return Ok(CommandOutput::Nothing),
            }
        }
    };
    write_preference(connection, &route, requested).await
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
        let panel = status_of(&own).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(!panel.contains("Inherited from chat"));
        // A subagent chat: the daemon reports the ancestor as the source.
        let inherited = serde_json::json!({"occ2Enabled": true, "inherited": true, "sourceSessionId": "parent-7"});
        let panel = status_of(&inherited).unwrap();
        assert!(panel.contains("OCC-2: ON"));
        assert!(panel.contains("Inherited from chat parent-7"));
        // A host response without the toggle is an error, never a silent value.
        assert!(status_of(&serde_json::json!({})).is_err());
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
