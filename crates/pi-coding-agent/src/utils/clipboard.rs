//! Clipboard delivery distinguishes local backend acceptance from unacknowledged terminal forwarding.

use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::time::Duration;

use base64::Engine as _;
use tokio::io::AsyncWriteExt;

use super::pi_user_agent::process_platform;

const MAX_OSC52_ENCODED_LENGTH: usize = 100_000;
const COPY_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardOutcome {
    LocalBackendAccepted,
    TerminalForwarded,
}

impl ClipboardOutcome {
    pub fn status(self) -> &'static str {
        match self {
            Self::LocalBackendAccepted => "Copied to local clipboard",
            Self::TerminalForwarded => "Clipboard request sent to terminal (unconfirmed; requires OSC 52 support)",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClipboardError {
    #[error("Failed to copy: no local clipboard backend succeeded and terminal forwarding is unavailable")]
    Failed,
}

pub fn is_remote_session_with(env: &HashMap<String, String>) -> bool {
    ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "MOSH_CONNECTION"]
        .iter()
        .any(|key| env.get(*key).is_some_and(|value| !value.is_empty()))
}

fn osc52_sequence(text: &str, tmux: bool) -> Option<String> {
    // Bound before allocating the encoded payload.
    if text.len() > MAX_OSC52_ENCODED_LENGTH / 4 * 3 {
        return None;
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(text.as_bytes());
    let osc = format!("\x1b]52;c;{encoded}\x07");
    Some(if tmux {
        format!("\x1bPtmux;{}\x1b\\", osc.replace('\x1b', "\x1b\x1b"))
    } else {
        osc
    })
}

fn write_osc52(writer: &mut impl Write, text: &str, tmux: bool) -> bool {
    let Some(sequence) = osc52_sequence(text, tmux) else { return false };
    writer.write_all(sequence.as_bytes()).and_then(|_| writer.flush()).is_ok()
}

/// Success means bytes were flushed, not that a terminal accepted the clipboard request.
pub fn emit_osc52(text: &str) -> bool {
    let stdout = std::io::stdout();
    if !stdout.is_terminal() {
        return false;
    }
    write_osc52(&mut stdout.lock(), text, std::env::var("TMUX").is_ok_and(|v| !v.is_empty()))
}

fn local_commands(platform: &str, env: &HashMap<String, String>) -> Vec<(&'static str, Vec<String>)> {
    // Never populate the remote host's clipboard instead of the SSH client's.
    if is_remote_session_with(env) {
        return Vec::new();
    }
    let has = |key: &str| env.get(key).is_some_and(|value| !value.is_empty());
    let mut commands = Vec::new();
    if platform == "darwin" {
        commands.push(("pbcopy", vec![]));
    } else if platform == "win32" {
        commands.push(("clip.exe", vec![]));
    } else {
        if has("TERMUX_VERSION") {
            commands.push(("termux-clipboard-set", vec![]));
        }
        if has("WAYLAND_DISPLAY") {
            commands.push(("wl-copy", vec![]));
        }
        if has("DISPLAY") {
            commands.push(("xclip", vec!["-selection".into(), "clipboard".into()]));
            commands.push(("xsel", vec!["--clipboard".into(), "--input".into()]));
        }
    }
    commands
}

// clip.exe otherwise interprets piped UTF-8 through the active Windows code page.
fn clipboard_input<'a>(program: &str, text: &'a str) -> std::borrow::Cow<'a, [u8]> {
    if program == "clip.exe" {
        let bytes = [0xff, 0xfe].into_iter()
            .chain(text.encode_utf16().flat_map(u16::to_le_bytes)).collect();
        std::borrow::Cow::Owned(bytes)
    } else {
        std::borrow::Cow::Borrowed(text.as_bytes())
    }
}

/// Bound both pipe writes and process exit. Background clipboard owners keep null stdio.
async fn copy_with_command(program: &str, args: &[String], text: &str, timeout: Duration) -> bool {
    let mut command = tokio::process::Command::new(program);
    command.args(args).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let Ok(mut child) = command.spawn() else { return false };
    let result = tokio::time::timeout(timeout, async {
        let mut stdin = child.stdin.take().ok_or_else(|| std::io::Error::other("missing stdin"))?;
        stdin.write_all(&clipboard_input(program, text)).await?;
        stdin.shutdown().await?;
        drop(stdin);
        child.wait().await
    }).await;
    match result {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    }
}

pub async fn copy_to_clipboard(text: &str) -> Result<ClipboardOutcome, ClipboardError> {
    let env = std::env::vars().collect();
    copy_with_backends(text, local_commands(process_platform(), &env), emit_osc52).await
}

async fn copy_with_backends(
    text: &str,
    commands: Vec<(&str, Vec<String>)>,
    forward: impl FnOnce(&str) -> bool,
) -> Result<ClipboardOutcome, ClipboardError> {
    for (program, args) in commands {
        if copy_with_command(program, &args, text, COPY_TIMEOUT).await {
            return Ok(ClipboardOutcome::LocalBackendAccepted);
        }
    }
    if forward(text) {
        Ok(ClipboardOutcome::TerminalForwarded)
    } else {
        Err(ClipboardError::Failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn remote_sessions_never_write_host_clipboard_even_with_forwarded_display() {
        for remote in ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "MOSH_CONNECTION"] {
            let vars = env(&[(remote, "remote"), ("DISPLAY", ":10"), ("WAYLAND_DISPLAY", "wayland-0")]);
            assert!(is_remote_session_with(&vars));
            for platform in ["linux", "darwin", "win32"] {
                assert!(local_commands(platform, &vars).is_empty());
            }
        }
        assert!(!is_remote_session_with(&env(&[("SSH_CLIENT", "")])));
    }

    #[test]
    fn linux_backends_prefer_wayland_then_x11_and_do_not_need_session_type() {
        let commands = local_commands("linux", &env(&[("WAYLAND_DISPLAY", "wayland-0"), ("DISPLAY", ":0")]));
        assert_eq!(commands.iter().map(|(p, _)| *p).collect::<Vec<_>>(), ["wl-copy", "xclip", "xsel"]);
        assert!(local_commands("linux", &env(&[])).is_empty());
    }

    #[test]
    fn clipboard_input_preserves_unicode_and_uses_bom_utf16_only_for_windows_clip() {
        let text = "A雪🦀\n";
        assert_eq!(clipboard_input("clip.exe", text).as_ref(),
            &[0xff, 0xfe, 0x41, 0, 0xea, 0x96, 0x3e, 0xd8, 0x80, 0xdd, 0x0a, 0]);
        assert_eq!(clipboard_input("clip.exe", "").as_ref(), &[0xff, 0xfe]);
        for program in ["wl-copy", "xclip", "xsel", "pbcopy", "termux-clipboard-set"] {
            assert_eq!(clipboard_input(program, text).as_ref(), text.as_bytes());
        }
    }

    #[test]
    fn osc52_is_bounded_and_tmux_escapes_the_inner_sequence() {
        assert_eq!(osc52_sequence("hi", false).unwrap(), "\x1b]52;c;aGk=\x07");
        assert_eq!(osc52_sequence("hi", true).unwrap(), "\x1bPtmux;\x1b\x1b]52;c;aGk=\x07\x1b\\");
        assert!(osc52_sequence(&"a".repeat(75_000), false).is_some());
        assert!(osc52_sequence(&"a".repeat(75_001), true).is_none());
        assert!(osc52_sequence(&"雪".repeat(25_001), false).is_none());
    }

    #[test]
    fn osc52_io_failure_is_not_success_and_forwarded_status_is_unconfirmed() {
        struct Broken(bool);
        impl Write for Broken {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                if self.0 { Ok(data.len()) } else { Err(std::io::ErrorKind::BrokenPipe.into()) }
            }
            fn flush(&mut self) -> std::io::Result<()> { Err(std::io::ErrorKind::BrokenPipe.into()) }
        }
        assert!(!write_osc52(&mut Broken(false), "hi", false));
        assert!(!write_osc52(&mut Broken(true), "hi", true));
        let mut bytes = Vec::new();
        assert!(write_osc52(&mut bytes, "hi", false));
        assert_eq!(bytes, b"\x1b]52;c;aGk=\x07");
        assert!(ClipboardOutcome::TerminalForwarded.status().contains("unconfirmed"));
        assert!(!ClipboardOutcome::TerminalForwarded.status().contains("Copied"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn clipboard_commands_require_complete_input_and_successful_exit() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("clipboard");
        let script = "cat > \"$1\"".to_string();
        assert!(copy_with_command("sh", &["-c".into(), script, "copy".into(), output.to_string_lossy().into_owned()], "雪\nhi", COPY_TIMEOUT).await);
        assert_eq!(std::fs::read_to_string(output).unwrap(), "雪\nhi");
        assert!(!copy_with_command("sh", &["-c".into(), "cat >/dev/null; exit 7".into()], "hi", COPY_TIMEOUT).await);
        assert!(!copy_with_command("/nonexistent/clipboard-test", &[], "hi", COPY_TIMEOUT).await);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn clipboard_falls_through_failed_backends_without_claiming_terminal_confirmation() {
        let fail = || ("sh", vec!["-c".into(), "cat >/dev/null; exit 7".into()]);
        let accept = ("sh", vec!["-c".into(), "cat >/dev/null".into()]);
        assert_eq!(copy_with_backends("hi", vec![fail(), accept], |_| panic!("local success must not forward")).await.unwrap(), ClipboardOutcome::LocalBackendAccepted);
        assert_eq!(copy_with_backends("hi", vec![fail()], |text| text == "hi").await.unwrap(), ClipboardOutcome::TerminalForwarded);
        assert!(copy_with_backends("hi", vec![fail()], |_| false).await.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn clipboard_timeout_covers_blocked_input_and_exit() {
        for text in ["hi".to_string(), "x".repeat(1_000_000)] {
            let started = std::time::Instant::now();
            assert!(!copy_with_command("sh", &["-c".into(), "while :; do :; done".into()], &text, Duration::from_millis(30)).await);
            assert!(started.elapsed() < Duration::from_secs(2));
        }
    }
}
