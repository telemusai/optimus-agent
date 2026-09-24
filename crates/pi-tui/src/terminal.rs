//! Port of packages/tui/src/terminal.ts.

use crate::keys::set_kitty_protocol_active;
use crate::stdin_buffer::{StdinBuffer, StdinBufferEvent, StdinBufferOptions};
use crate::terminal_colors::{
    parse_osc_color_response, set_default_terminal_colors, DefaultTerminalColors, OscColorKind, Rgb,
};
#[cfg(not(windows))]
use crate::terminal_colors::{QUERY_DEFAULT_BACKGROUND, QUERY_DEFAULT_FOREGROUND};
use once_cell::sync::Lazy;
use std::cell::RefCell;
use std::io::{Read, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

#[path = "terminal_image_probe.rs"]
mod image_probe;
use image_probe::{ImageProbe, CELL_QUERY, IMAGE_QUERY};

const TERMINAL_PROGRESS_KEEPALIVE_MS: u64 = 1000;
const TERMINAL_PROGRESS_ACTIVE_SEQUENCE: &str = "\x1b]9;4;3\x07";
const TERMINAL_PROGRESS_CLEAR_SEQUENCE: &str = "\x1b]9;4;0;\x07";

/// A preserved alternate screen is adopted by the next ProcessTerminal during in-process handoff.
static PENDING_ALT_SCREEN_HANDOFF: AtomicBool = AtomicBool::new(false);

struct PendingInputHandoff {
    token: u64,
    was_raw: bool,
}

// Keep stdin raw and drain input while a preserved fullscreen frame waits for
// the next in-process TUI. Worker-backed session attach can make this handoff
// noticeably longer; restoring cooked mode during the gap makes arrow escape
// sequences echo into the preserved frame.
static PENDING_INPUT_HANDOFF: Lazy<Mutex<Option<PendingInputHandoff>>> = Lazy::new(|| Mutex::new(None));
static HANDOFF_TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);

fn consume_alt_screen_handoff() -> bool {
    PENDING_ALT_SCREEN_HANDOFF.swap(false, Ordering::SeqCst)
}

fn begin_input_handoff(token: u64, was_raw: bool) {
    let mut slot = PENDING_INPUT_HANDOFF.lock().unwrap();
    let inherited_was_raw = slot.as_ref().map(|h| h.was_raw).unwrap_or(was_raw);
    *slot = Some(PendingInputHandoff {
        token,
        was_raw: inherited_was_raw,
    });
    drop(slot);
    // Keep stdin flowing while the handoff is pending so the gap data is
    // delivered and drained instead of staying buffered in the tty until the
    // next TUI reads it. Mirrors the non-blocking read pump that keeps stdin
    // flowing while the handoff record is armed (terminal.ts:51 `stdin.resume()`).
    let _ = read_available_input();
}

fn consume_input_handoff() -> Option<bool> {
    PENDING_INPUT_HANDOFF.lock().unwrap().take().map(|h| h.was_raw)
}

fn cancel_input_handoff(token: u64) {
    let handoff = {
        let mut slot = PENDING_INPUT_HANDOFF.lock().unwrap();
        let matches = slot.as_ref().map(|h| h.token == token).unwrap_or(false);
        if !matches {
            return;
        }
        slot.take().unwrap()
    };
    let _ = set_raw_mode(handoff.was_raw);
}

/// Port of the `discardHandler` installed by `beginInputHandoff()`
/// (packages/tui/src/terminal.ts:48-51) and removed by `consumeInputHandoff()`
/// (terminal.ts:59): while a preserved fullscreen frame waits for the next
/// in-process TUI, keys typed in the handoff gap are received and thrown away
/// instead of being replayed into the next TUI's first `pollInput()` calls.
///
/// The pending record existing is the token match: `beginInputHandoff()` sets it
/// and only `consumeInputHandoff()` or `cancelInputHandoff(token)` clear it.
/// Returns true when handoff data was drained.
fn drain_pending_handoff_input() -> bool {
    if PENDING_INPUT_HANDOFF.lock().unwrap().is_none() {
        return false;
    }
    match read_available_input() {
        Ok(NativeInput::Bytes(_)) | Ok(NativeInput::Ignored) => true,
        _ => false,
    }
}

pub(crate) fn stdout_write(data: &str) {
    #[cfg(windows)]
    ensure_windows_utf8();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let conpty = cfg!(windows) || std::env::var_os("WSL_INTEROP").is_some()
        || std::env::var_os("WSL_DISTRO_NAME").is_some();
    let _ = write_terminal_output(&mut lock, data, conpty);
    let _ = lock.flush();
}

fn write_terminal_output(writer: &mut impl Write, mut data: &str, conpty: bool) -> std::io::Result<()> {
    // OMP uses 16 KiB writes to avoid ConPTY losing viewport tracking on
    // large frames. UTF-8 boundaries matter when writing to a Win32 console.
    while conpty && data.len() > 16 * 1024 {
        let mut end = 16 * 1024;
        while !data.is_char_boundary(end) { end -= 1; }
        if let Some(newline) = data[..end].rfind('\n') { end = newline + 1; }
        writer.write_all(data[..end].as_bytes())?;
        data = &data[end..];
    }
    writer.write_all(data.as_bytes())
}

#[cfg(windows)]
fn ensure_windows_utf8() {
    use windows_sys::Win32::System::Console::{GetConsoleCP, GetConsoleOutputCP, SetConsoleCP, SetConsoleOutputCP};
    // A console-sharing child can change these. Invalid/non-console handles
    // return zero; leave redirected streams alone.
    unsafe {
        let input = GetConsoleCP();
        let output = GetConsoleOutputCP();
        if input != 0 && input != 65001 { let _ = SetConsoleCP(65001); }
        if output != 0 && output != 65001 { let _ = SetConsoleOutputCP(65001); }
    }
}

fn set_raw_mode(raw: bool) -> std::io::Result<()> {
    if raw {
        crossterm::terminal::enable_raw_mode()
    } else {
        crossterm::terminal::disable_raw_mode()
    }
}

enum NativeInput { Pending, Ignored, Closed, Bytes(Vec<u8>) }

#[cfg(unix)]
fn read_available_input() -> std::io::Result<NativeInput> {
    let mut fd = libc::pollfd { fd: libc::STDIN_FILENO, events: libc::POLLIN, revents: 0 };
    // The descriptor is borrowed; polling never changes ownership or flags.
    let ready = unsafe { libc::poll(&mut fd, 1, 0) };
    if ready < 0 {
        let error = std::io::Error::last_os_error();
        return if error.kind() == std::io::ErrorKind::Interrupted { Ok(NativeInput::Pending) } else { Err(error) };
    }
    if ready == 0 { return Ok(NativeInput::Pending); }
    if fd.revents & libc::POLLNVAL != 0 { return Ok(NativeInput::Closed); }
    let mut bytes = [0u8; 4096];
    match std::io::stdin().read(&mut bytes) {
        Ok(0) => Ok(NativeInput::Closed),
        Ok(count) => Ok(NativeInput::Bytes(bytes[..count].to_vec())),
        Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => Ok(NativeInput::Pending),
        Err(error) if error.raw_os_error() == Some(libc::EIO) => Ok(NativeInput::Closed),
        Err(error) => Err(error),
    }
}

#[cfg(windows)]
static WINDOWS_PENDING_SURROGATE: Mutex<Option<u16>> = Mutex::new(None);

#[cfg(windows)]
fn read_available_input() -> std::io::Result<NativeInput> {
    use windows_sys::Win32::System::Console::{
        GetNumberOfConsoleInputEvents, GetStdHandle, ReadConsoleInputW, INPUT_RECORD,
        KEY_EVENT, STD_INPUT_HANDLE,
    };
    unsafe {
        let handle = GetStdHandle(STD_INPUT_HANDLE);
        let mut count = 0;
        if GetNumberOfConsoleInputEvents(handle, &mut count) == 0 {
            return Err(std::io::Error::last_os_error());
        }
        if count == 0 { return Ok(NativeInput::Pending); }
        let mut record: INPUT_RECORD = std::mem::zeroed();
        if ReadConsoleInputW(handle, &mut record, 1, &mut count) == 0 {
            return Err(std::io::Error::last_os_error());
        }
        if count == 0 || record.EventType != KEY_EVENT as u16 {
            return Ok(NativeInput::Ignored);
        }
        let key = record.Event.KeyEvent;
        let mut surrogate = WINDOWS_PENDING_SURROGATE.lock().unwrap();
        Ok(match windows_record_sequence(
            key.bKeyDown != 0, key.wVirtualKeyCode, key.uChar.UnicodeChar,
            key.dwControlKeyState, &mut surrogate,
        ) {
            Some(sequence) => NativeInput::Bytes(sequence.into_bytes()),
            None => NativeInput::Ignored,
        })
    }
}

#[cfg(not(any(unix, windows)))]
fn read_available_input() -> std::io::Result<NativeInput> {
    use crossterm::event;
    if !event::poll(std::time::Duration::ZERO)? { return Ok(NativeInput::Pending); }
    Ok(match native_event_sequence(event::read()?) {
        Some(sequence) => NativeInput::Bytes(sequence.into_bytes()),
        None => NativeInput::Ignored,
    })
}

#[cfg(any(windows, test))]
fn windows_record_sequence(
    down: bool,
    virtual_key: u16,
    code_unit: u16,
    control: u32,
    pending_surrogate: &mut Option<u16>,
) -> Option<String> {
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    if !down { return None; }
    let mut decoded = String::new();
    if let Some(high) = pending_surrogate.take() {
        if (0xdc00..=0xdfff).contains(&code_unit) {
            decoded.push(char::from_u32(0x10000 + ((u32::from(high) - 0xd800) << 10)
                + u32::from(code_unit) - 0xdc00).unwrap());
        } else {
            decoded.push(char::REPLACEMENT_CHARACTER);
        }
    }
    if decoded.chars().next().is_none_or(|ch| ch == char::REPLACEMENT_CHARACTER) {
        if (0xd800..=0xdbff).contains(&code_unit) {
            *pending_surrogate = Some(code_unit);
            return if decoded.is_empty() { None } else { Some(decoded) };
        }
        decoded.push(char::from_u32(u32::from(code_unit)).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
    // In VT input mode ConPTY writes exact characters as VK=0 records. Do not
    // reinterpret ESC/LF as keyboard shortcuts or strip paste framing. Decode
    // UTF-16 here: crossterm's record reader drops these controls and can pair
    // surrogate key releases rather than the two key-down code units.
    if virtual_key == 0 { return Some(decoded); }

    let mut modifiers = KeyModifiers::NONE;
    if control & 0x10 != 0 { modifiers |= KeyModifiers::SHIFT; }
    if control & 0x03 != 0 { modifiers |= KeyModifiers::ALT; }
    if control & 0x0c != 0 { modifiers |= KeyModifiers::CONTROL; }
    let key = match virtual_key {
        0x08 => KeyCode::Backspace,
        0x09 if modifiers.contains(KeyModifiers::SHIFT) => KeyCode::BackTab,
        0x09 => KeyCode::Tab,
        0x0d => KeyCode::Enter,
        0x1b => KeyCode::Esc,
        0x21 => KeyCode::PageUp, 0x22 => KeyCode::PageDown,
        0x23 => KeyCode::End, 0x24 => KeyCode::Home,
        0x25 => KeyCode::Left, 0x26 => KeyCode::Up,
        0x27 => KeyCode::Right, 0x28 => KeyCode::Down,
        0x2d => KeyCode::Insert, 0x2e => KeyCode::Delete,
        0x70..=0x87 => KeyCode::F((virtual_key - 0x6f) as u8),
        _ => {
            let mut ch = decoded.chars().last()?;
            if ch.is_control() && modifiers.contains(KeyModifiers::CONTROL) {
                ch = match virtual_key {
                    0x41..=0x5a => char::from_u32(u32::from(virtual_key) + 32)?,
                    0x20 => ' ',
                    _ if (1..=31).contains(&code_unit) => char::from_u32(u32::from(code_unit) + 64)?,
                    _ => return None,
                };
            } else if ch == '\0' {
                return None;
            }
            KeyCode::Char(ch)
        }
    };
    native_event_sequence(Event::Key(KeyEvent::new(key, modifiers)))
}

#[cfg(any(not(unix), test))]
fn native_event_sequence(event: crossterm::event::Event) -> Option<String> {
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
    Some(match event {
        Event::Paste(text) => format!("\x1b[200~{text}\x1b[201~"),
        Event::Key(key) if key.kind != KeyEventKind::Release => {
            let modifier = 1 + u8::from(key.modifiers.contains(KeyModifiers::SHIFT))
                + 2 * u8::from(key.modifiers.contains(KeyModifiers::ALT))
                + 4 * u8::from(key.modifiers.contains(KeyModifiers::CONTROL));
            // Use the same VT forms as the byte-oriented Unix reader. The
            // editor's functional-key parser does not accept synthetic PUA CSI-u codes.
            let suffix = match key.code {
                KeyCode::Up => Some("A"), KeyCode::Down => Some("B"),
                KeyCode::Right => Some("C"), KeyCode::Left => Some("D"),
                KeyCode::Home => Some("H"), KeyCode::End => Some("F"),
                _ => None,
            };
            if let Some(suffix) = suffix { return Some(format!("\x1b[1;{modifier}{suffix}")); }
            let functional = match key.code {
                KeyCode::Insert => Some(2), KeyCode::Delete => Some(3),
                KeyCode::PageUp => Some(5), KeyCode::PageDown => Some(6),
                KeyCode::F(n @ 1..=4) => return Some(format!("\x1b[1;{modifier}{}", char::from(b'P' + n - 1))),
                KeyCode::F(n @ 5..=12) => Some([15, 17, 18, 19, 20, 21, 23, 24][usize::from(n - 5)]),
                _ => None,
            };
            if let Some(code) = functional { return Some(format!("\x1b[{code};{modifier}~")); }
            let code = match key.code {
                KeyCode::Char(ch) if !key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL | KeyModifiers::SUPER) => return Some(ch.to_string()),
                KeyCode::Char(ch) => ch as u32,
                KeyCode::Enter if modifier == 1 => return Some("\r".into()),
                KeyCode::Tab if modifier == 1 => return Some("\t".into()),
                KeyCode::Esc if modifier == 1 => return Some("\x1b".into()),
                KeyCode::Backspace if modifier == 1 => return Some("\x7f".into()),
                KeyCode::BackTab => return Some(format!("\x1b[9;{}u", ((modifier - 1) | 1) + 1)),
                KeyCode::Enter => 13, KeyCode::Tab => 9, KeyCode::Backspace => 127, KeyCode::Esc => 27,
                KeyCode::F(n) => 57363 + u32::from(n),
                _ => return None,
            };
            format!("\x1b[{code};{modifier}u")
        }
        _ => return None,
    })
}

/// Minimal terminal interface for TUI
pub trait Terminal {
    /// Pump available native input on the UI thread. False means EOF.
    fn poll_input(&mut self) -> std::io::Result<bool> { Ok(true) }
    // Start the terminal with input and resize handlers
    fn start(&mut self, on_input: Box<dyn Fn(String)>, on_resize: Box<dyn Fn()>);

    // Stop the terminal and restore state
    fn stop(&mut self, options: TerminalStopOptions);

    /// Drain stdin before exiting to prevent Kitty key release events from
    /// leaking to the parent shell over slow SSH connections.
    fn drain_input(&mut self, max_ms: u64, idle_ms: u64);

    // Write output to terminal
    fn write(&mut self, data: &str);

    fn columns(&self) -> usize;
    fn rows(&self) -> usize;

    /// Whether Kitty keyboard protocol is active
    fn kitty_protocol_active(&self) -> bool;

    /// Move cursor up (negative) or down (positive) by N lines.
    fn move_by(&mut self, lines: i64);

    fn hide_cursor(&mut self);
    fn show_cursor(&mut self);

    fn clear_line(&mut self);
    fn clear_from_cursor(&mut self);
    fn clear_screen(&mut self);

    fn enter_alt_screen(&mut self);
    fn leave_alt_screen(&mut self);
    fn alt_screen_active(&self) -> bool;

    /// SGR mouse tracking (?1000 + ?1006); motion tracking is deliberately never enabled.
    fn set_mouse_tracking(&mut self, enabled: bool);
    fn mouse_tracking_active(&self) -> bool;

    fn set_title(&mut self, title: &str);

    /// Progress indicator (OSC 9;4)
    fn set_progress(&mut self, active: bool);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalStopOptions {
    pub preserve_alt_screen: bool,
}

struct DefaultColorProbe {
    foreground: Option<Rgb>,
    background: Option<Rgb>,
}

/// State shared with the stdin dispatcher closure.
struct Shared {
    image_probe: ImageProbe,
    input_handler: Option<Box<dyn Fn(String)>>,
    resize_handler: Option<Box<dyn Fn()>>,
    kitty_protocol_active: bool,
    modify_other_keys_active: bool,
    keyboard_protocol_fallback_timer: Option<u64>,
    default_color_probe: Option<DefaultColorProbe>,
}

/// Real terminal using process stdin/stdout
pub struct ProcessTerminal {
    shared: Rc<RefCell<Shared>>,
    was_raw: bool,
    started: bool,
    alt_screen_handoff_token: u64,
    alt_screen_active: bool,
    mouse_tracking_active: bool,
    stdin_buffer: Option<Rc<RefCell<StdinBuffer>>>,
    stdin_dispatcher: Option<Rc<dyn Fn(String)>>,
    progress_interval: Option<u64>,
    /// Next re-emit deadline for the armed OSC 9;4 keepalive. The reference uses
    /// `setInterval(..., TERMINAL_PROGRESS_KEEPALIVE_MS)` (terminal.ts:596-600);
    /// this port drives the same period from the input pump, like the other
    /// timers (`setTimeout` -> `pollInput` checkpoints).
    progress_keepalive_at: Option<std::time::Instant>,
    write_log_path: String,
    started_at: Option<std::time::Instant>,
    last_input_at: Option<std::time::Instant>,
    last_size: Option<(usize, usize)>,
    pending_native_input: Vec<u8>,
    #[cfg(windows)]
    windows_input_mode: Option<u32>,
}

fn timestamp_for_log() -> String {
    chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string()
}

fn compute_write_log_path() -> String {
    let env = std::env::var("PI_TUI_WRITE_LOG").unwrap_or_default();
    if env.is_empty() {
        return String::new();
    }
    let path = std::path::PathBuf::from(&env);
    if path.is_dir() {
        let ts = timestamp_for_log();
        return path
            .join(format!("tui-{ts}-{}.log", std::process::id()))
            .to_string_lossy()
            .to_string();
    }
    env
}

/// Maximum characters kept in a terminal title so long titles truncate instead
/// of corrupting narrow tab bars.
pub const MAX_TITLE_CHARS: usize = 256;

/// Strip C0/C1 control characters (including ESC and BEL) plus zero-width and
/// bidi-formatting characters from terminal title text, so a title containing
/// escape sequences cannot smuggle terminal commands through the OSC 0 write.
/// Stops scanning as soon as MAX_TITLE_CHARS accepted characters are stored, so
/// an arbitrarily long title neither allocates nor scans unboundedly.
pub fn sanitize_title_text(title: &str) -> String {
    let mut out = String::with_capacity(title.len().min(MAX_TITLE_CHARS * 4));
    let mut accepted = 0usize;
    for character in title.chars() {
        if accepted == MAX_TITLE_CHARS {
            break;
        }
        if matches!(
            character,
            '\u{0}'..='\u{1f}'
                | '\u{7f}'
                | '\u{80}'..='\u{9f}'
                | '\u{200b}'..='\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2066}'..='\u{2069}'
                | '\u{feff}'
        ) {
            continue;
        }
        out.push(character);
        accepted += 1;
    }
    out
}

/// OSC 0 window-title sequence for `title` with the sanitized payload.
pub fn title_osc_sequence(title: &str) -> String {
    format!("\x1b]0;{}\x07", sanitize_title_text(title))
}

impl ProcessTerminal {
    pub fn new() -> Self {
        Self {
            shared: Rc::new(RefCell::new(Shared {
                image_probe: ImageProbe::default(),
                input_handler: None,
                resize_handler: None,
                kitty_protocol_active: false,
                modify_other_keys_active: false,
                keyboard_protocol_fallback_timer: None,
                default_color_probe: None,
            })),
            was_raw: false,
            started: false,
            alt_screen_handoff_token: HANDOFF_TOKEN_COUNTER.fetch_add(1, Ordering::SeqCst),
            alt_screen_active: consume_alt_screen_handoff(),
            mouse_tracking_active: false,
            stdin_buffer: None,
            stdin_dispatcher: None,
            progress_interval: None,
            progress_keepalive_at: None,
            write_log_path: compute_write_log_path(),
            started_at: None,
            last_input_at: None,
            last_size: None,
            pending_native_input: Vec::new(),
            #[cfg(windows)]
            windows_input_mode: None,
        }
    }

    pub fn kitty_protocol_active(&self) -> bool {
        self.shared.borrow().kitty_protocol_active
    }

    /// Port of `setupStdinBuffer()`.
    fn setup_stdin_buffer(&mut self) {
        let buffer = Rc::new(RefCell::new(StdinBuffer::new(StdinBufferOptions {
            timeout: Some(10),
        })));
        self.stdin_buffer = Some(buffer.clone());

        let shared = self.shared.clone();
        let dispatcher = Rc::new(move |sequence: String| {
            let result = shared.borrow_mut().image_probe.filter(&sequence, std::time::Instant::now());
            if result.query_cells { stdout_write(CELL_QUERY); }
            if result.changed {
                if let Some(handler) = &shared.borrow().resize_handler { handler(); }
            }
            let sequence = result.input;
            if sequence.is_empty() { return; }
            // Check for Kitty protocol response (only if not already enabled).
            {
                let mut s = shared.borrow_mut();
                if handle_default_color_probe_response(&mut s, &sequence) {
                    return;
                }
                if !s.kitty_protocol_active && is_kitty_protocol_response(&sequence) {
                    s.keyboard_protocol_fallback_timer = None;
                    s.kitty_protocol_active = true;
                    set_kitty_protocol_active(true);

                    // Enable Kitty keyboard protocol (push flags)
                    // Flag 1 = disambiguate escape codes
                    // Flag 2 = report event types (press/repeat/release)
                    // Flag 4 = report alternate keys (shifted key, base layout key)
                    stdout_write("\x1b[>7u");
                    return;
                }
            }
            let handler = shared.borrow_mut().input_handler.take();
            if let Some(handler) = handler {
                handler(sequence);
                shared.borrow_mut().input_handler = Some(handler);
            }
        });
        self.stdin_dispatcher = Some(dispatcher);
    }

    /// Port of `queryAndEnableKittyProtocol()`.
    fn query_and_enable_kitty_protocol(&mut self) {
        self.setup_stdin_buffer();
        self.query_default_terminal_colors();
        // Win32 supplies key/modifier records directly; VT reply parsing and
        // keyboard protocol negotiation belong only to the byte reader.
        if cfg!(windows) { return; }
        stdout_write("\x1b[?u");
        self.clear_keyboard_protocol_fallback_timer();
        self.shared.borrow_mut().keyboard_protocol_fallback_timer = Some(150);
    }

    fn clear_keyboard_protocol_fallback_timer(&mut self) {
        self.shared.borrow_mut().keyboard_protocol_fallback_timer = None;
    }

    /// Fallback timer body: enable xterm modifyOtherKeys mode 2 when Kitty never answers.
    pub fn apply_keyboard_protocol_fallback(&mut self) {
        let mut s = self.shared.borrow_mut();
        if s.keyboard_protocol_fallback_timer.is_none() {
            return;
        }
        s.keyboard_protocol_fallback_timer = None;
        if !s.kitty_protocol_active && !s.modify_other_keys_active {
            stdout_write("\x1b[>4;2m");
            s.modify_other_keys_active = true;
        }
    }

    fn query_default_terminal_colors(&mut self) {
        #[cfg(windows)]
        {
            // OSC replies lose their framing through ReadConsoleInput. Read
            // colours through the console API without injecting input instead.
            use windows_sys::Win32::System::Console::{
                GetConsoleScreenBufferInfoEx, GetStdHandle, CONSOLE_SCREEN_BUFFER_INFOEX,
                STD_OUTPUT_HANDLE,
            };
            let mut info: CONSOLE_SCREEN_BUFFER_INFOEX = unsafe { std::mem::zeroed() };
            info.cbSize = std::mem::size_of_val(&info) as u32;
            if unsafe { GetConsoleScreenBufferInfoEx(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) } != 0 {
                let rgb = |value: u32| Rgb {
                    r: i64::from(value & 255), g: i64::from((value >> 8) & 255),
                    b: i64::from((value >> 16) & 255),
                };
                set_default_terminal_colors(Some(DefaultTerminalColors {
                    foreground: rgb(info.ColorTable[usize::from(info.wAttributes & 15)]),
                    background: rgb(info.ColorTable[usize::from((info.wAttributes >> 4) & 15)]),
                }));
            }
            return;
        }
        #[cfg(not(windows))]
        {
        if !crossterm::tty::IsTty::is_tty(&std::io::stdin()) || !crossterm::tty::IsTty::is_tty(&std::io::stdout())
        {
            return;
        }
        self.finish_default_color_probe();
        self.shared.borrow_mut().default_color_probe = Some(DefaultColorProbe {
            foreground: None,
            background: None,
        });
        stdout_write(QUERY_DEFAULT_FOREGROUND);
        stdout_write(QUERY_DEFAULT_BACKGROUND);
        }
    }

    /// Called by the owner after 100 ms to close an unanswered colour probe.
    pub fn apply_default_color_probe_timeout(&mut self) {
        self.finish_default_color_probe();
    }

    fn finish_default_color_probe(&mut self) {
        let probe = self.shared.borrow_mut().default_color_probe.take();
        let probe = match probe {
            Some(p) => p,
            None => return,
        };
        if let (Some(foreground), Some(background)) = (probe.foreground, probe.background) {
            set_default_terminal_colors(Some(DefaultTerminalColors {
                foreground,
                background,
            }));
            let handler = self.shared.borrow_mut().resize_handler.take();
            if let Some(handler) = handler {
                handler();
                self.shared.borrow_mut().resize_handler = Some(handler);
            }
        }
    }

    /// Our Windows record reader preserves VT characters (including controls
    /// and UTF-16 pairs) instead of feeding them through crossterm's key parser.
    fn configure_windows_event_input(&mut self) {
        #[cfg(windows)]
        {
            const STD_INPUT_HANDLE: u32 = -10i32 as u32;
            const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;
            unsafe {
                use windows_sys::Win32::System::Console::{GetConsoleMode, GetStdHandle, SetConsoleMode};
                let handle = GetStdHandle(STD_INPUT_HANDLE);
                let mut mode: u32 = 0;
                if GetConsoleMode(handle, &mut mode) != 0 {
                    if SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_INPUT) != 0 {
                        self.windows_input_mode = Some(mode);
                    }
                }
            }
            *WINDOWS_PENDING_SURROGATE.lock().unwrap() = None;
        }
    }

    fn windows_vt_input_ready(&self) -> bool {
        #[cfg(windows)]
        { self.windows_input_mode.is_some() }
        #[cfg(not(windows))]
        { true }
    }

    fn restore_windows_event_input(&mut self) {
        #[cfg(windows)]
        if let Some(mode) = self.windows_input_mode.take() {
            unsafe {
                use windows_sys::Win32::System::Console::{GetStdHandle, SetConsoleMode, STD_INPUT_HANDLE};
                let _ = SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), mode);
            }
        }
    }

    /// Feed raw bytes through the stdin buffer and dispatch the emitted sequences.
    pub fn process_input_bytes(&mut self, data: &[u8]) {
        let dispatcher = match &self.stdin_dispatcher {
            Some(d) => d.clone(),
            None => return,
        };
        let buffer = match &self.stdin_buffer {
            Some(b) => b.clone(),
            None => return,
        };
        let events = {
            let mut buffer = buffer.borrow_mut();
            buffer.process(data);
            buffer.take_events()
        };
        for event in events {
            match event {
                StdinBufferEvent::Data(sequence) => dispatcher(sequence),
                // Re-wrap paste content with bracketed paste markers for existing editor handling.
                StdinBufferEvent::Paste(content) => dispatcher(format!("\x1b[200~{content}\x1b[201~")),
            }
        }
    }

    fn flush_native_input(&mut self) {
        if !self.pending_native_input.is_empty() {
            let bytes = std::mem::take(&mut self.pending_native_input);
            self.process_input_bytes(&bytes);
        }
    }

    fn poll_input_from(
        &mut self,
        mut read: impl FnMut() -> std::io::Result<NativeInput>,
        coalesce_native: bool,
    ) -> std::io::Result<bool> {
        // Win32 pastes arrive as press/release records, not Event::Paste. Keep
        // their text together until the native queue is empty, including when
        // a large paste spans several bounded polls. Otherwise pasted Enter
        // reaches the editor as Submit instead of part of the pasted text.
        let budget = if coalesce_native { 4096 } else { 64 };
        for _ in 0..budget {
            match read()? {
                NativeInput::Pending => {
                    self.flush_native_input();
                    break;
                }
                NativeInput::Ignored => continue,
                NativeInput::Closed => {
                    self.flush_native_input();
                    return Ok(false);
                }
                NativeInput::Bytes(bytes) => {
                    if coalesce_native && !bytes.starts_with(b"\x1b") {
                        self.pending_native_input.extend_from_slice(&bytes);
                    } else {
                        self.flush_native_input();
                        self.process_input_bytes(&bytes);
                    }
                    self.last_input_at = Some(std::time::Instant::now());
                }
            }
        }
        Ok(true)
    }

    /// Flush a pending partial sequence (the TypeScript flush timer).
    pub fn flush_pending_input(&mut self) {
        let dispatcher = match &self.stdin_dispatcher {
            Some(d) => d.clone(),
            None => return,
        };
        let buffer = match &self.stdin_buffer {
            Some(b) => b.clone(),
            None => return,
        };
        let events = {
            let mut buffer = buffer.borrow_mut();
            buffer.flush_events();
            buffer.take_events()
        };
        for event in events {
            match event {
                StdinBufferEvent::Data(sequence) => dispatcher(sequence),
                StdinBufferEvent::Paste(content) => dispatcher(format!("\x1b[200~{content}\x1b[201~")),
            }
        }
    }

    fn clear_progress_interval(&mut self) -> bool {
        self.progress_keepalive_at = None;
        if self.progress_interval.is_none() {
            return false;
        }
        self.progress_interval = None;
        true
    }

    /// Port of the `setInterval` body in `setProgress(true)`
    /// (packages/tui/src/terminal.ts:596-600): re-write OSC 9;4;3 every
    /// TERMINAL_PROGRESS_KEEPALIVE_MS while progress stays active, so terminals
    /// that clear unrefreshed OSC 9;4 state keep showing the indicator.
    fn apply_progress_keepalive(&mut self) {
        if self.progress_interval.is_none() {
            self.progress_keepalive_at = None;
            return;
        }
        if self.progress_keepalive_at.is_some_and(|deadline| std::time::Instant::now() < deadline) {
            return;
        }
        self.progress_keepalive_at =
            Some(std::time::Instant::now() + std::time::Duration::from_millis(TERMINAL_PROGRESS_KEEPALIVE_MS));
        stdout_write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
    }

    fn release_alt_screen(&mut self) {
        let owns_pending_handoff = self.owns_pending_alt_screen_handoff();
        if !self.alt_screen_active && !owns_pending_handoff {
            return;
        }
        self.alt_screen_active = false;
        if owns_pending_handoff {
            PENDING_ALT_SCREEN_HANDOFF.store(false, Ordering::SeqCst);
            cancel_input_handoff(self.alt_screen_handoff_token);
        }
        self.write("\x1b[?1049l");
    }

    fn owns_pending_alt_screen_handoff(&self) -> bool {
        PENDING_ALT_SCREEN_HANDOFF.load(Ordering::SeqCst)
    }
}

impl Default for ProcessTerminal {
    fn default() -> Self {
        Self::new()
    }
}

/// `process.stdout.columns || Number(process.env.COLUMNS) || 80`
/// (terminal.ts:502-508). JavaScript `||` treats `0`, `NaN` and `""` as absent,
/// so a reported size of 0 is unknown and falls through to the environment
/// variable and finally to the 80x24 default.
fn resolve_dimension(reported: Option<usize>, env_value: Option<&str>, default: usize) -> usize {
    if let Some(value) = reported {
        if value > 0 {
            return value;
        }
    }
    env_value
        .map(str::trim)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn is_kitty_protocol_response(sequence: &str) -> bool {
    // Kitty protocol response pattern: \x1b[?<flags>u
    let bytes = sequence.as_bytes();
    if !sequence.starts_with("\x1b[?") || !sequence.ends_with('u') {
        return false;
    }
    let inner = &sequence[3..sequence.len() - 1];
    !inner.is_empty() && bytes.len() > 4 && inner.chars().all(|c| c.is_ascii_digit())
}

fn handle_default_color_probe_response(shared: &mut Shared, sequence: &str) -> bool {
    let response = match parse_osc_color_response(sequence) {
        Some(r) => r,
        None => return false,
    };
    if shared.default_color_probe.is_none() {
        return true;
    }
    let probe = shared.default_color_probe.as_mut().unwrap();
    match response.kind {
        OscColorKind::Foreground => probe.foreground = Some(response.rgb),
        OscColorKind::Background => probe.background = Some(response.rgb),
    }
    if probe.foreground.is_some() && probe.background.is_some() {
        let done = shared.default_color_probe.take();
        if let Some(done) = done {
            if let (Some(foreground), Some(background)) = (done.foreground, done.background) {
                set_default_terminal_colors(Some(DefaultTerminalColors {
                    foreground,
                    background,
                }));
                let handler = shared.resize_handler.take();
                if let Some(handler) = handler {
                    handler();
                    shared.resize_handler = Some(handler);
                }
            }
        }
    }
    true
}

impl Terminal for ProcessTerminal {
    fn poll_input(&mut self) -> std::io::Result<bool> {
        if !self.started {
            // The preserved-fullscreen handoff gap: `beginInputHandoff()` keeps
            // stdin attached to a discard handler (terminal.ts:48-51) so gap keys
            // are consumed and dropped here instead of staying buffered for the
            // next TUI's first `pollInput()` calls.
            while drain_pending_handoff_input() {}
            return Ok(true);
        }
        if !self.poll_input_from(read_available_input, cfg!(windows))? {
            return Ok(false);
        }
        if self.pending_native_input.is_empty()
            && self.last_input_at.is_some_and(|last| last.elapsed().as_millis() >= 10)
        {
            self.flush_pending_input();
            self.last_input_at = None;
        }
        if let Some(start) = self.started_at {
            if start.elapsed().as_millis() >= 100 { self.apply_default_color_probe_timeout(); }
            if start.elapsed().as_millis() >= 150 { self.apply_keyboard_protocol_fallback(); }
        }
        self.apply_progress_keepalive();
        let size = (self.columns(), self.rows());
        if self.last_size != Some(size) {
            self.last_size = Some(size);
            let shared = self.shared.borrow();
            if let Some(handler) = &shared.resize_handler { handler(); }
        }
        Ok(true)
    }

    fn start(&mut self, on_input: Box<dyn Fn(String)>, on_resize: Box<dyn Fn()>) {
        self.started = true;
        self.started_at = Some(std::time::Instant::now());
        self.last_size = Some((self.columns(), self.rows()));

        // Adopt the handoff before installing the handlers: the discardHandler is
        // removed by `consumeInputHandoff()` (terminal.ts:54-62), so keys typed
        // during the gap can never reach this TUI's `onInput`.
        self.was_raw = consume_input_handoff()
            .unwrap_or_else(|| crossterm::terminal::is_raw_mode_enabled().unwrap_or(false));

        {
            let mut s = self.shared.borrow_mut();
            s.input_handler = Some(on_input);
            s.resize_handler = Some(on_resize);
        }

        // Save previous state and enable raw mode
        let _ = set_raw_mode(true);

        // Enable bracketed paste mode - terminal will wrap pastes in \x1b[200~ ... \x1b[201~
        stdout_write("\x1b[?2004h");

        self.configure_windows_event_input();

        // Query and enable Kitty keyboard protocol.
        self.query_and_enable_kitty_protocol();
        use crate::terminal_image::{get_capabilities, image_passthrough_blocked, image_protocol_forced, reset_capabilities_cache};
        reset_capabilities_cache();
        crate::terminal_image::reset_cell_dimensions();
        if get_capabilities().images.is_none() && !image_passthrough_blocked() && !image_protocol_forced()
            && crossterm::tty::IsTty::is_tty(&std::io::stdin())
            && crossterm::tty::IsTty::is_tty(&std::io::stdout())
            && (!cfg!(windows) || self.windows_vt_input_ready())
        {
            self.shared.borrow_mut().image_probe.start(
                std::time::Instant::now(), std::env::var_os("WT_SESSION").is_some(),
            );
            stdout_write(IMAGE_QUERY);
        }
    }

    fn stop(&mut self, options: TerminalStopOptions) {
        let was_started = self.started;
        self.started = false;
        self.shared.borrow_mut().image_probe = ImageProbe::default();
        self.finish_default_color_probe();
        self.clear_keyboard_protocol_fallback_timer();

        if self.clear_progress_interval() {
            stdout_write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
        }

        if self.mouse_tracking_active {
            stdout_write("\x1b[?1006l\x1b[?1002l");
            self.mouse_tracking_active = false;
        }
        if self.alt_screen_active {
            if options.preserve_alt_screen {
                PENDING_ALT_SCREEN_HANDOFF.store(true, Ordering::SeqCst);
                self.alt_screen_active = false;
            } else {
                self.release_alt_screen();
            }
        } else if !options.preserve_alt_screen {
            self.release_alt_screen();
        }

        // Disable bracketed paste mode
        stdout_write("\x1b[?2004l");

        // Disable Kitty keyboard protocol if not already done by drainInput()
        {
            let mut s = self.shared.borrow_mut();
            if s.kitty_protocol_active {
                stdout_write("\x1b[<u");
                s.kitty_protocol_active = false;
                set_kitty_protocol_active(false);
            }
            if s.modify_other_keys_active {
                stdout_write("\x1b[>4;0m");
                s.modify_other_keys_active = false;
            }
            s.input_handler = None;
            s.resize_handler = None;
        }

        // Clean up StdinBuffer
        self.stdin_buffer = None;
        self.stdin_dispatcher = None;
        self.pending_native_input.clear();
        self.restore_windows_event_input();

        if options.preserve_alt_screen && was_started {
            begin_input_handoff(self.alt_screen_handoff_token, self.was_raw);
        } else {
            // Pause stdin to prevent any buffered input (e.g., Ctrl+D) from being
            // re-interpreted after raw mode is disabled.
            let _ = set_raw_mode(self.was_raw);
        }
    }

    fn drain_input(&mut self, max_ms: u64, idle_ms: u64) {
        {
            let mut s = self.shared.borrow_mut();
            if s.kitty_protocol_active {
                // Disable Kitty keyboard protocol first so any late key releases
                // do not generate new Kitty escape sequences.
                stdout_write("\x1b[<u");
                s.kitty_protocol_active = false;
                set_kitty_protocol_active(false);
            }
            if s.modify_other_keys_active {
                stdout_write("\x1b[>4;0m");
                s.modify_other_keys_active = false;
            }
            s.input_handler = None;
        }

        let start = std::time::Instant::now();
        let mut last_data_time = std::time::Instant::now();
        loop {
            let now = std::time::Instant::now();
            if now.duration_since(start).as_millis() as u64 >= max_ms {
                break;
            }
            if now.duration_since(last_data_time).as_millis() as u64 >= idle_ms {
                break;
            }
            match read_available_input() {
                Ok(NativeInput::Closed) => break,
                Ok(NativeInput::Bytes(_)) | Ok(NativeInput::Ignored) => {
                    last_data_time = std::time::Instant::now();
                }
                Ok(NativeInput::Pending) | Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(idle_ms.min(10)));
                }
            }
        }
    }

    fn write(&mut self, data: &str) {
        stdout_write(data);
        if !self.write_log_path.is_empty() {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.write_log_path)
            {
                let _ = file.write_all(data.as_bytes());
            }
        }
    }

    fn columns(&self) -> usize {
        let reported = crossterm::terminal::size()
            .ok()
            .map(|(cols, _)| cols as usize);
        resolve_dimension(reported, std::env::var("COLUMNS").ok().as_deref(), 80)
    }

    fn rows(&self) -> usize {
        let reported = crossterm::terminal::size()
            .ok()
            .map(|(_, rows)| rows as usize);
        resolve_dimension(reported, std::env::var("LINES").ok().as_deref(), 24)
    }

    fn kitty_protocol_active(&self) -> bool {
        self.shared.borrow().kitty_protocol_active
    }

    fn move_by(&mut self, lines: i64) {
        if lines > 0 {
            self.write(&format!("\x1b[{lines}B"));
        } else if lines < 0 {
            self.write(&format!("\x1b[{}A", -lines));
        }
        // lines === 0: no movement
    }

    fn hide_cursor(&mut self) {
        self.write("\x1b[?25l");
    }

    fn show_cursor(&mut self) {
        self.write("\x1b[?25h");
    }

    fn clear_line(&mut self) {
        self.write("\x1b[K");
    }

    fn clear_from_cursor(&mut self) {
        self.write("\x1b[J");
    }

    fn clear_screen(&mut self) {
        self.write("\x1b[2J\x1b[H"); // Clear screen and move to home (1,1)
    }

    fn enter_alt_screen(&mut self) {
        if self.alt_screen_active {
            return;
        }
        if self.owns_pending_alt_screen_handoff() {
            PENDING_ALT_SCREEN_HANDOFF.store(false, Ordering::SeqCst);
            self.alt_screen_active = true;
            return;
        }
        self.alt_screen_active = true;
        self.write("\x1b[?1049h");
    }

    fn leave_alt_screen(&mut self) {
        self.release_alt_screen();
    }

    fn alt_screen_active(&self) -> bool {
        self.alt_screen_active
    }

    fn set_mouse_tracking(&mut self, enabled: bool) {
        if enabled == self.mouse_tracking_active {
            return;
        }
        self.mouse_tracking_active = enabled;
        // ?1002 (button-event tracking) reports drag motion for in-app selection
        // but not hover, keeping passive mouse movement unreported.
        let seq = if enabled {
            "\x1b[?1002h\x1b[?1006h"
        } else {
            "\x1b[?1006l\x1b[?1002l"
        };
        self.write(seq);
    }

    fn mouse_tracking_active(&self) -> bool {
        self.mouse_tracking_active
    }

    fn set_title(&mut self, title: &str) {
        // OSC 0;title BEL - set terminal window title
        self.write(&title_osc_sequence(title));
    }

    fn set_progress(&mut self, active: bool) {
        if active {
            // OSC 9;4;3 - indeterminate progress
            stdout_write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
            if self.progress_interval.is_none() {
                self.progress_interval = Some(TERMINAL_PROGRESS_KEEPALIVE_MS);
            }
            self.progress_keepalive_at =
                Some(std::time::Instant::now() + std::time::Duration::from_millis(TERMINAL_PROGRESS_KEEPALIVE_MS));
        } else {
            self.clear_progress_interval();
            // OSC 9;4;0 - clear progress
            stdout_write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conpty_image_writes_are_utf8_safe_bounded_and_stop_on_error() {
        #[derive(Default)]
        struct Writer { chunks: Vec<Vec<u8>>, fail_at: Option<usize> }
        impl Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.fail_at == Some(self.chunks.len()) { return Err(std::io::ErrorKind::BrokenPipe.into()); }
                self.chunks.push(bytes.to_vec()); Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let data = format!("{}\n\x1bP0;1;0q{}\x1b\\", "界😀".repeat(5000), "~".repeat(50000));
        let mut writer = Writer::default();
        write_terminal_output(&mut writer, &data, true).unwrap();
        assert_eq!(writer.chunks.concat(), data.as_bytes());
        assert!(writer.chunks.iter().all(|chunk| chunk.len() <= 16 * 1024 && std::str::from_utf8(chunk).is_ok()));
        writer = Writer { fail_at: Some(1), ..Default::default() };
        assert!(write_terminal_output(&mut writer, &data, true).is_err());
        assert_eq!(writer.chunks.len(), 1);
    }

    #[test]
    fn windows_vt_image_reply_survives_native_records_and_does_not_type_into_prompt() {
        use crate::terminal_image::*;
        set_capabilities(TerminalCapabilities { images: None, true_color: true, hyperlinks: true });
        let (mut terminal, received) = native_input_fixture();
        terminal.shared.borrow_mut().image_probe.start(std::time::Instant::now(), false);
        let mut surrogate = None;
        let mut records = vt_records("\x1b[?2;0;4096;4096S", &mut surrogate);
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        assert!(received.borrow().is_empty());
        assert_eq!(get_capabilities().images, Some(ImageProtocol::Sixel));
        reset_capabilities_cache();
    }

    #[test]
    fn windows_terminal_da1_and_cell_records_enable_full_color_without_xterm_replies() {
        use crate::terminal_image::*;
        reset_capabilities_cache();
        reset_cell_dimensions();
        set_capabilities(TerminalCapabilities { images: None, true_color: true, hyperlinks: true });
        let (mut terminal, received) = native_input_fixture();
        terminal.shared.borrow_mut().image_probe.start(std::time::Instant::now(), true);
        let mut surrogate = None;
        // Microsoft's DA1 response advertises feature 4; XTSMGRAPHICS is unsupported.
        for reply in ["\x1b[?61;4;6;7;14;21;22;23;24;28;32;42c", "\x1b[6;20;10t"] {
            let mut records = vt_records(reply, &mut surrogate);
            terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        }
        assert!(received.borrow().is_empty());
        assert_eq!(get_capabilities().images, Some(ImageProtocol::Sixel));
        assert!(cell_dimensions_known());
        assert_eq!(get_cell_dimensions(), CellDimensions { width_px: 10, height_px: 20 });
        assert_eq!(get_sixel_palette_size(), 256);
        let rendered = render_image(&test_palette_png(), &ImageDimensions { width_px: 96, height_px: 12 },
            &ImageRenderOptions::default()).unwrap();
        let highest = regex::Regex::new(r"#(\d+)").unwrap().captures_iter(&rendered.sequence)
            .map(|capture| capture[1].parse::<u16>().unwrap()).max().unwrap();
        assert!((16..256).contains(&highest));
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        reset_capabilities_cache();
    }

    fn native_input_fixture() -> (ProcessTerminal, Rc<RefCell<Vec<String>>>) {
        let mut terminal = ProcessTerminal::new();
        let received = Rc::new(RefCell::new(Vec::new()));
        let output = received.clone();
        terminal.shared.borrow_mut().input_handler = Some(Box::new(move |data| {
            output.borrow_mut().push(data);
        }));
        terminal.setup_stdin_buffer();
        (terminal, received)
    }

    fn native_paste_records(text: &str) -> std::collections::VecDeque<NativeInput> {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
        let mut records = std::collections::VecDeque::new();
        for ch in text.chars() {
            let code = if ch == '\r' { KeyCode::Enter } else { KeyCode::Char(ch) };
            for kind in [KeyEventKind::Press, KeyEventKind::Release] {
                let sequence = native_event_sequence(Event::Key(KeyEvent::new_with_kind(
                    code, KeyModifiers::NONE, kind,
                )));
                records.push_back(match sequence {
                    Some(sequence) => NativeInput::Bytes(sequence.into_bytes()),
                    None => NativeInput::Ignored,
                });
            }
        }
        records
    }

    fn vt_records(text: &str, surrogate: &mut Option<u16>) -> std::collections::VecDeque<NativeInput> {
        text.encode_utf16().map(|unit| match windows_record_sequence(true, 0, unit, 0, surrogate) {
            Some(sequence) => NativeInput::Bytes(sequence.into_bytes()),
            None => NativeInput::Ignored,
        }).collect()
    }

    #[test]
    fn windows_vt_records_preserve_split_paste_framing_and_utf16() {
        let (mut terminal, received) = native_input_fixture();
        let mut surrogate = None;
        let mut records = vt_records("\x1b[200~first\n世界", &mut surrogate);
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        terminal.flush_pending_input();
        assert!(received.borrow().is_empty(), "a gap inside a bracketed paste must not dispatch text");
        let mut records = vt_records("😀second\nlast\x1b[201~", &mut surrogate);
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        assert_eq!(*received.borrow(), vec!["\x1b[200~first\n世界😀second\nlast\x1b[201~"]);
    }

    #[test]
    fn windows_utf16_surrogate_releases_do_not_consume_the_pair() {
        let mut surrogate = None;
        assert_eq!(windows_record_sequence(true, 0, 0xd83d, 0, &mut surrogate), None);
        assert_eq!(windows_record_sequence(false, 0, 0xd83d, 0, &mut surrogate), None);
        assert_eq!(windows_record_sequence(true, 0, 0xde00, 0, &mut surrogate), Some("😀".into()));
        assert_eq!(windows_record_sequence(false, 0, 0xde00, 0, &mut surrogate), None);
        assert_eq!(surrogate, None);
    }

    #[test]
    fn windows_vt_controls_and_legacy_navigation_remain_distinct() {
        let mut surrogate = None;
        for unit in [0, 3, 8, 10, 13, 27, 127] {
            assert_eq!(windows_record_sequence(true, 0, unit, 0, &mut surrogate), Some(char::from_u32(u32::from(unit)).unwrap().to_string()));
        }
        for (virtual_key, name) in [(0x25, "left"), (0x26, "up"), (0x27, "right"), (0x28, "down"), (0x0d, "enter"), (0x1b, "escape")] {
            let sequence = windows_record_sequence(true, virtual_key, 0, 0, &mut surrogate).unwrap();
            assert!(crate::keys::matches_key(&sequence, name));
        }
        for (virtual_key, unit, modifiers, name) in [
            (0x43, 3, 8, "ctrl+c"), (0x48, 8, 8, "ctrl+h"),
            (0x4d, 13, 8, "ctrl+m"), (0x4c, 12, 24, "ctrl+shift+l"),
        ] {
            let sequence = windows_record_sequence(true, virtual_key, unit, modifiers, &mut surrogate).unwrap();
            assert!(crate::keys::matches_key(&sequence, name), "{name}: {sequence:?}");
            assert!(crate::tui::is_unambiguous_ctrl_combo(&sequence));
        }
        for unit in [8, 13, 127] {
            let raw = windows_record_sequence(true, 0, unit, 0, &mut surrogate).unwrap();
            assert!(!crate::tui::is_unambiguous_ctrl_combo(&raw), "VK=0 keeps editing bytes");
        }
        let shift_enter = windows_record_sequence(true, 0x0d, 13, 16, &mut surrogate).unwrap();
        assert!(crate::keys::matches_key(&shift_enter, "shift+enter"));
    }

    #[test]
    fn windows_native_paste_ignores_releases_and_keeps_newlines_in_one_paste() {
        let (mut terminal, received) = native_input_fixture();
        let text = "Do not build\rDo not run tests\rReview only: café 世界";
        let mut records = native_paste_records(text);
        assert!(terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap());
        assert!(records.is_empty(), "a release record must not end the poll");
        assert_eq!(*received.borrow(), vec![format!("\x1b[200~{text}\x1b[201~")]);
    }

    #[test]
    fn windows_native_paste_survives_poll_budget_without_submitting_fragments() {
        let (mut terminal, received) = native_input_fixture();
        let text = format!("{}\r{}\rFinal line", "a".repeat(5000), "界".repeat(5000));
        let mut records = native_paste_records(&text);
        let initial_count = records.len();
        assert!(terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap());
        assert_eq!(initial_count - records.len(), 4096, "each poll remains bounded");
        assert!(received.borrow().is_empty(), "do not dispatch an unfinished native batch");
        while !records.is_empty() {
            assert!(terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap());
        }
        terminal.poll_input_from(|| Ok(NativeInput::Pending), true).unwrap();
        assert_eq!(*received.borrow(), vec![format!("\x1b[200~{text}\x1b[201~")]);
    }

    #[test]
    fn windows_native_paste_preserves_following_navigation_and_escape() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let (mut terminal, received) = native_input_fixture();
        let mut records = native_paste_records("first\rsecond");
        let left = native_event_sequence(Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE))).unwrap();
        records.push_back(NativeInput::Bytes(left.as_bytes().to_vec()));
        records.push_back(NativeInput::Bytes(vec![0x1b]));
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        terminal.flush_pending_input();
        assert_eq!(*received.borrow(), vec!["\x1b[200~first\rsecond\x1b[201~".to_string(), left, "\x1b".to_string()]);
    }

    #[test]
    fn windows_native_bracketed_paste_and_ordinary_enter_are_preserved() {
        let (mut terminal, received) = native_input_fixture();
        let mut records = native_paste_records("\x1b[200~first\rsecond\x1b[201~");
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        assert_eq!(*received.borrow(), vec!["\x1b[200~first\rsecond\x1b[201~"]);
        received.borrow_mut().clear();
        let mut records = native_paste_records("ok\r");
        terminal.poll_input_from(|| Ok(records.pop_front().unwrap_or(NativeInput::Pending)), true).unwrap();
        assert_eq!(*received.borrow(), vec!["o", "k", "\r"]);
    }

    #[test]
    fn windows_control_and_navigation_events_match_editor_bindings() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let keys = [(KeyCode::Up,"up"), (KeyCode::Down,"down"),
            (KeyCode::Left,"left"), (KeyCode::Right,"right"), (KeyCode::Home,"home"),
            (KeyCode::End,"end"), (KeyCode::PageUp,"pageup"), (KeyCode::PageDown,"pagedown"),
            (KeyCode::Insert,"insert"), (KeyCode::Delete,"delete"),
            (KeyCode::Enter,"enter"), (KeyCode::Tab,"tab"),
            (KeyCode::Backspace,"backspace"), (KeyCode::Esc,"escape")];
        for (code, name) in keys {
            for (modifier, prefix) in [(KeyModifiers::NONE,""), (KeyModifiers::SHIFT,"shift+"),
                (KeyModifiers::CONTROL,"ctrl+"), (KeyModifiers::ALT,"alt+")] {
                if code == KeyCode::Esc && modifier != KeyModifiers::NONE { continue; }
                let encoded = native_event_sequence(Event::Key(KeyEvent::new(code,modifier))).unwrap();
                assert!(crate::keys::matches_key(&encoded, &format!("{prefix}{name}")), "{name} {modifier:?}: {encoded:?}");
            }
        }
        let backtab = native_event_sequence(Event::Key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT))).unwrap();
        assert!(crate::keys::matches_key(&backtab,"shift+tab"));
    }

    #[test]
    fn windows_text_paste_and_releases_are_preserved() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
        for ch in ['/', 'A', '?', 'é'] {
            assert_eq!(native_event_sequence(Event::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::SHIFT))), Some(ch.to_string()));
        }
        assert_eq!(native_event_sequence(Event::Paste("/login\ntext".into())), Some("\x1b[200~/login\ntext\x1b[201~".into()));
        assert!(native_event_sequence(Event::Key(KeyEvent::new_with_kind(KeyCode::Enter,KeyModifiers::NONE,KeyEventKind::Release))).is_none());
    }

    #[test]
    fn kitty_response_pattern() {
        assert!(is_kitty_protocol_response("\x1b[?1u"));
        assert!(is_kitty_protocol_response("\x1b[?7u"));
        assert!(!is_kitty_protocol_response("\x1b[?u"));
        assert!(!is_kitty_protocol_response("\x1b[A"));
    }

    #[test]
    fn handoff_drain_requires_a_pending_token() {
        // Only the handoff gap drains stdin (terminal.ts:48-51); once
        // consumeInputHandoff() runs (terminal.ts:54-62) the discard handler is
        // gone and the next TUI receives user input normally.
        let _ = consume_input_handoff();
        begin_input_handoff(7, true);
        // The handoff record is armed: data arriving in the gap is discarded.
        assert!(PENDING_INPUT_HANDOFF.lock().unwrap().is_some());
        assert!(!drain_pending_handoff_input()); // nothing available right now
        assert_eq!(consume_input_handoff(), Some(true));
        // consumeInputHandoff() removed the discard handler (terminal.ts:59).
        assert!(!drain_pending_handoff_input());
    }

    #[test]
    fn progress_keepalive_rearms_after_the_interval() {
        let mut terminal = ProcessTerminal::new();
        terminal.set_progress(true);
        assert_eq!(terminal.progress_interval, Some(TERMINAL_PROGRESS_KEEPALIVE_MS));
        let armed = terminal.progress_keepalive_at.expect("keepalive armed after set_progress(true)");
        // Not due yet: no re-emit, deadline unchanged.
        terminal.apply_progress_keepalive();
        assert_eq!(terminal.progress_keepalive_at, Some(armed));
        // Deadline reached: re-emit and arm the next 1000ms window.
        terminal.progress_keepalive_at = Some(std::time::Instant::now() - std::time::Duration::from_millis(1));
        terminal.apply_progress_keepalive();
        assert!(terminal.progress_keepalive_at.expect("re-armed") > std::time::Instant::now());
        terminal.set_progress(false);
        assert_eq!(terminal.progress_interval, None);
        assert_eq!(terminal.progress_keepalive_at, None);
    }

    /// `process.stdout.columns || Number(process.env.COLUMNS) || 80`
    /// (terminal.ts:502-508) and the `rows` twin (terminal.ts:506-508). Teeth:
    /// restoring the old `crossterm::terminal::size().map(..).unwrap_or_else(..)`
    /// shape makes the 0-valued cases return 0 instead of the env/default value.
    #[test]
    fn dimension_falls_back_like_a_javascript_truthiness_chain() {
        assert_eq!(resolve_dimension(Some(120), Some("40"), 80), 120);
        assert_eq!(
            resolve_dimension(Some(0), Some("40"), 80),
            40,
            "0 columns is unknown"
        );
        assert_eq!(
            resolve_dimension(Some(0), Some("0"), 80),
            80,
            "0 env is unknown too"
        );
        assert_eq!(resolve_dimension(None, Some("40"), 80), 40);
        assert_eq!(
            resolve_dimension(None, Some("nonsense"), 80),
            80,
            "NaN -> default"
        );
        assert_eq!(resolve_dimension(None, Some(""), 80), 80, "'' -> default");
        assert_eq!(resolve_dimension(None, None, 80), 80);
        assert_eq!(
            resolve_dimension(Some(0), None, 24),
            24,
            "the rows default is 24"
        );
    }

    #[test]
    fn write_log_path_falls_back_to_env() {
        std::env::remove_var("PI_TUI_WRITE_LOG");
        assert_eq!(compute_write_log_path(), "");
    }

    #[test]
    fn title_sanitization_removes_control_and_escape_payloads() {
        assert_eq!(sanitize_title_text("Optimus - Agent"), "Optimus - Agent");
        let cleaned = sanitize_title_text("App\x1b]0;evil\x07name\u{202e}spoof\x00\x1b[31m\u{feff}");
        assert_eq!(cleaned, "App]0;evilnamespoof[31m");
        assert!(!cleaned.contains('\x1b'));
        assert!(!cleaned.contains('\x07'));
    }

    #[test]
    fn title_osc_sequence_frames_the_sanitized_title() {
        assert_eq!(title_osc_sequence("Optimus - Agent"), "\x1b]0;Optimus - Agent\x07");
        let sequence = title_osc_sequence("bad\x1b]2;inject\x07");
        let payload = sequence
            .trim_start_matches("\x1b]0;")
            .trim_end_matches('\x07');
        assert_eq!(payload, "bad]2;inject");
        assert!(!payload.contains('\x1b') && !payload.contains('\x07'));
    }

    #[test]
    fn title_sanitization_caps_length() {
        let long = "x".repeat(MAX_TITLE_CHARS + 50);
        assert_eq!(sanitize_title_text(&long).chars().count(), MAX_TITLE_CHARS);
    }

    #[test]
    fn title_sanitization_counts_unicode_characters_not_bytes() {
        // Four-byte emoji count as one character each: the cap is characters,
        // not bytes, and the result keeps whole emoji.
        let emoji = "\u{1f600}".repeat(MAX_TITLE_CHARS + 50);
        let cleaned = sanitize_title_text(&emoji);
        assert_eq!(cleaned.chars().count(), MAX_TITLE_CHARS);
        assert_eq!(cleaned.len(), MAX_TITLE_CHARS * 4);
        assert!(cleaned.starts_with("\u{1f600}"));
        assert!(cleaned.ends_with("\u{1f600}"));
    }

    #[test]
    fn title_sanitization_strips_control_and_bidi_around_unicode_text() {
        // ESC, BEL, NUL, bidi override, zero-width, and FEFF are removed; plain
        // text and printable Unicode (including the pi letter) are kept.
        let mixed = "a\u{202e}b\x1b[31m\u{3c0}\u{200b}c\x07d\u{feff}\u{1f600}";
        assert_eq!(sanitize_title_text(mixed), "ab[31m\u{3c0}cd\u{1f600}");
        let trailing = "Optimus - Agent\u{1f600}\u{202e}\u{200b}";
        assert_eq!(sanitize_title_text(trailing), "Optimus - Agent\u{1f600}");
    }
}
