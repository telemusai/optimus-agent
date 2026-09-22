//! Port of packages/coding-agent/src/modes/interactive/theme/theme.ts
//!
//! The TypeScript validator (typebox/compile) is replaced by a hand-written
//! structural check that reports the same `Invalid theme "<label>"` messages.
//! The lazy-loading dance around the validator is kept, because the observable
//! behaviour is "the first custom theme parse only gets the minimal check".

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::config::{get_custom_themes_dir, get_themes_dir};

use super::super::interactive_mode_services::AgentConnectionSourceInfo;

// ---------------------------------------------------------------------------
// Private stand-ins for `@earendil-works/pi-tui` colour helpers and
// `core/source-info.ts`. Both live in other slices; see blocked_on.
// ---------------------------------------------------------------------------

/// `Rgb`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

/// `DefaultTerminalColors`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefaultTerminalColors {
    pub foreground: Rgb,
    pub background: Rgb,
}

/// `TerminalBackgroundKind`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalBackgroundKind {
    Dark,
    Light,
}

pub use pi_tui::terminal_colors::TerminalColorMode;

/// `AnsiColor = string | number`
#[derive(Debug, Clone, PartialEq)]
pub enum AnsiColor {
    Hex(String),
    Index(f64),
}

/// `SourceInfo` (core/source-info.ts)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceInfo {
    pub path: String,
    pub source: String,
    pub scope: String,
}

fn clamp_channel(value: f64) -> f64 {
    value.round().clamp(0.0, 255.0)
}

fn rgb_to_hex(rgb: &Rgb) -> String {
    let to_hex = |value: f64| format!("{:02x}", clamp_channel(value) as i64);
    format!("#{}{}{}", to_hex(rgb.r), to_hex(rgb.g), to_hex(rgb.b))
}

fn is_light_color(rgb: &Rgb) -> bool {
    0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b > 128.0
}

fn blend_color(top: &Rgb, bottom: &Rgb, alpha: f64) -> Rgb {
    let clamped_alpha = alpha.clamp(0.0, 1.0);
    Rgb {
        r: clamp_channel(top.r * clamped_alpha + bottom.r * (1.0 - clamped_alpha)),
        g: clamp_channel(top.g * clamped_alpha + bottom.g * (1.0 - clamped_alpha)),
        b: clamp_channel(top.b * clamped_alpha + bottom.b * (1.0 - clamped_alpha)),
    }
}

const GRAY_VALUES: [f64; 24] = [
    8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 88.0, 98.0, 108.0, 118.0, 128.0, 138.0, 148.0, 158.0, 168.0,
    178.0, 188.0, 198.0, 208.0, 218.0, 228.0, 238.0,
];

fn color_distance(a: &Rgb, b: &Rgb) -> f64 {
    let dr = a.r - b.r;
    let dg = a.g - b.g;
    let db = a.b - b.b;
    dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114
}

fn find_closest_index(value: f64, values: &[f64]) -> usize {
    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0usize;
    for (index, candidate) in values.iter().enumerate() {
        let dist = (value - candidate).abs();
        if dist < min_dist {
            min_dist = dist;
            min_idx = index;
        }
    }
    min_idx
}

fn rgb_to_256(rgb: &Rgb) -> f64 {
    let r_idx = find_closest_index(rgb.r, &CUBE_VALUES);
    let g_idx = find_closest_index(rgb.g, &CUBE_VALUES);
    let b_idx = find_closest_index(rgb.b, &CUBE_VALUES);
    let cube_rgb = Rgb { r: CUBE_VALUES[r_idx], g: CUBE_VALUES[g_idx], b: CUBE_VALUES[b_idx] };
    let cube_index = 16.0 + 36.0 * r_idx as f64 + 6.0 * g_idx as f64 + b_idx as f64;
    let cube_dist = color_distance(rgb, &cube_rgb);

    let gray = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b).round();
    let gray_idx = find_closest_index(gray, &GRAY_VALUES);
    let gray_value = GRAY_VALUES[gray_idx];
    let gray_rgb = Rgb { r: gray_value, g: gray_value, b: gray_value };
    let gray_index = 232.0 + gray_idx as f64;
    let gray_dist = color_distance(rgb, &gray_rgb);

    let max_channel = rgb.r.max(rgb.g).max(rgb.b);
    let min_channel = rgb.r.min(rgb.g).min(rgb.b);
    if max_channel - min_channel < 10.0 && gray_dist < cube_dist {
        return gray_index;
    }

    cube_index
}

fn best_ansi_color(rgb: &Rgb, mode: TerminalColorMode) -> AnsiColor {
    if mode == TerminalColorMode::Truecolor {
        return AnsiColor::Hex(rgb_to_hex(rgb));
    }
    if mode == TerminalColorMode::Color256 {
        return AnsiColor::Index(rgb_to_256(rgb));
    }
    AnsiColor::Hex(String::new())
}

fn rgb_from_terminal_colors(rgb: pi_tui::terminal_colors::Rgb) -> Rgb {
    Rgb {
        r: rgb.r as f64,
        g: rgb.g as f64,
        b: rgb.b as f64,
    }
}

fn rgb_to_terminal_colors(rgb: &Rgb) -> pi_tui::terminal_colors::Rgb {
    pi_tui::terminal_colors::Rgb {
        r: clamp_channel(rgb.r) as i64,
        g: clamp_channel(rgb.g) as i64,
        b: clamp_channel(rgb.b) as i64,
    }
}

/// `getDefaultTerminalColors` (theme.ts:7-11 imports it from pi-tui).
///
/// The theme reads the ONE module-level cell owned by `terminal-colors.ts:28`,
/// which is also what the OSC 10/11 probe writes (`packages/tui/src/terminal.ts:348`
/// -> `terminal.rs:382,535`). The earlier port kept a second function-local
/// `OnceLock` here, so a probe result could never be observed by the theme.
fn get_default_terminal_colors() -> Option<DefaultTerminalColors> {
    pi_tui::terminal_colors::get_default_terminal_colors().map(|colors| DefaultTerminalColors {
        foreground: rgb_from_terminal_colors(colors.foreground),
        background: rgb_from_terminal_colors(colors.background),
    })
}

/// `setDefaultTerminalColors` (terminal-colors.ts:184-187) - used by tests and
/// the terminal probe. Writes the shared cell and notifies the listeners.
pub fn set_default_terminal_colors(colors: Option<DefaultTerminalColors>) {
    pi_tui::terminal_colors::set_default_terminal_colors(colors.map(|colors| {
        pi_tui::terminal_colors::DefaultTerminalColors {
            foreground: rgb_to_terminal_colors(&colors.foreground),
            background: rgb_to_terminal_colors(&colors.background),
        }
    }));
}

/// `clearDefaultTerminalColors` (terminal-colors.ts:189-191).
pub fn clear_default_terminal_colors() {
    pi_tui::terminal_colors::clear_default_terminal_colors();
}

/// `getTerminalBackgroundKind` (terminal-colors.ts:193-199): the probed
/// background wins, `COLORFGBG` is the fallback.
fn get_terminal_background_kind() -> Option<TerminalBackgroundKind> {
    pi_tui::terminal_colors::get_terminal_background_kind().map(|kind| match kind {
        pi_tui::terminal_colors::TerminalBackgroundKind::Dark => TerminalBackgroundKind::Dark,
        pi_tui::terminal_colors::TerminalBackgroundKind::Light => TerminalBackgroundKind::Light,
    })
}

// ============================================================================
// Types & Schema
// ============================================================================

/// `ColorValue = string | number`
#[derive(Debug, Clone, PartialEq)]
pub enum ColorValue {
    Text(String),
    Index(f64),
}

impl ColorValue {
    pub fn is_empty_string(&self) -> bool {
        matches!(self, ColorValue::Text(text) if text.is_empty())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ColorValue::Text(text) => Some(text),
            ColorValue::Index(_) => None,
        }
    }
}

/// `ThemeJson`
#[derive(Debug, Clone, Default)]
pub struct ThemeJson {
    pub schema: Option<String>,
    pub name: String,
    /// `Record<string, ColorValue>`
    pub vars: HashMap<String, ColorValue>,
    pub colors: HashMap<String, ColorValue>,
    pub export: Option<ThemeExport>,
}

/// `ThemeJson["export"]`
#[derive(Debug, Clone, Default)]
pub struct ThemeExport {
    pub page_bg: Option<ColorValue>,
    pub card_bg: Option<ColorValue>,
    pub info_bg: Option<ColorValue>,
}

/// The required `colors` keys, in the declaration order of the typebox schema.
pub const THEME_COLOR_KEYS: &[&str] = &[
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "selectedBg",
    "userMessageBg",
    "userMessageText",
    "customMessageBg",
    "customMessageText",
    "customMessageLabel",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "toolDiffAddedBg",
    "toolDiffRemovedBg",
    "toolPanelBg",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffText",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "bashMode",
];

/// `ThemeColor`
pub type ThemeColor = &'static str;

/// `ThemeBg`
pub type ThemeBg = &'static str;

/// The background keys of the `colors` object (`bgColorKeys` in `createTheme`).
pub const BG_COLOR_KEYS: &[&str] = &[
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "toolDiffAddedBg",
    "toolDiffRemovedBg",
    "toolPanelBg",
];

/// `ColorMode`
pub type ColorMode = TerminalColorMode;

const ADAPTIVE_LIGHT_BG_ACCENT: Rgb = Rgb { r: 0.0, g: 95.0, b: 135.0 };
const SURFACE_MIN_LUMINANCE_DELTA: f64 = 12.0;
const SURFACE_CONTRAST_ALPHA: f64 = 0.08;
// Selection rows must stand out clearly, much more than passive surfaces.
const SELECTION_MIN_LUMINANCE_DELTA: f64 = 28.0;
const SELECTION_MAX_BLEND_ALPHA: f64 = 0.5;
const SELECTION_BLEND_STEP: f64 = 0.05;
const BLACK: Rgb = Rgb { r: 0.0, g: 0.0, b: 0.0 };
const WHITE: Rgb = Rgb { r: 255.0, g: 255.0, b: 255.0 };
const CUBE_VALUES: [f64; 6] = [0.0, 95.0, 135.0, 175.0, 215.0, 255.0];

// ============================================================================
// Color Utilities
// ============================================================================

/// Port of `detectColorMode`.
pub fn detect_color_mode() -> ColorMode {
    let colorterm = std::env::var("COLORTERM").unwrap_or_default();
    if colorterm == "truecolor" || colorterm == "24bit" {
        return TerminalColorMode::Truecolor;
    }
    // Windows Terminal supports truecolor
    if std::env::var("WT_SESSION").is_ok() {
        return TerminalColorMode::Truecolor;
    }
    let term = std::env::var("TERM").unwrap_or_default();
    // Fall back to 256color for truly limited terminals
    if term == "dumb" || term.is_empty() || term == "linux" {
        return TerminalColorMode::Color256;
    }
    // Terminal.app also doesn't support truecolor
    if std::env::var("TERM_PROGRAM").map(|value| value == "Apple_Terminal").unwrap_or(false) {
        return TerminalColorMode::Color256;
    }
    // tmux reports TERM=screen* but forwards 24-bit color, so treat it as
    // truecolor-capable; only genuine GNU screen (no $TMUX) falls back.
    let in_tmux = std::env::var("TMUX").is_ok() || term.starts_with("tmux");
    if !in_tmux && (term == "screen" || term.starts_with("screen-") || term.starts_with("screen.")) {
        return TerminalColorMode::Color256;
    }
    // Assume truecolor for everything else - virtually all modern terminals support it
    TerminalColorMode::Truecolor
}

/// Port of `hexToRgb`.
fn hex_to_rgb(hex: &str) -> Result<Rgb, String> {
    let cleaned = hex.replace('#', "");
    if cleaned.len() != 6 {
        return Err(format!("Invalid hex color: {hex}"));
    }
    let parse = |slice: &str| i64::from_str_radix(slice, 16).ok();
    let (Some(r), Some(g), Some(b)) = (parse(&cleaned[0..2]), parse(&cleaned[2..4]), parse(&cleaned[4..6])) else {
        return Err(format!("Invalid hex color: {hex}"));
    };
    Ok(Rgb { r: r as f64, g: g as f64, b: b as f64 })
}

/// Port of `ansi256ToRgb`.
fn ansi256_to_rgb(index: f64) -> Option<Rgb> {
    if index < 0.0 || index > 255.0 {
        return None;
    }
    let index = index as usize;
    if index < 16 {
        let basic_colors: [Rgb; 16] = [
            Rgb { r: 0.0, g: 0.0, b: 0.0 },
            Rgb { r: 128.0, g: 0.0, b: 0.0 },
            Rgb { r: 0.0, g: 128.0, b: 0.0 },
            Rgb { r: 128.0, g: 128.0, b: 0.0 },
            Rgb { r: 0.0, g: 0.0, b: 128.0 },
            Rgb { r: 128.0, g: 0.0, b: 128.0 },
            Rgb { r: 0.0, g: 128.0, b: 128.0 },
            Rgb { r: 192.0, g: 192.0, b: 192.0 },
            Rgb { r: 128.0, g: 128.0, b: 128.0 },
            Rgb { r: 255.0, g: 0.0, b: 0.0 },
            Rgb { r: 0.0, g: 255.0, b: 0.0 },
            Rgb { r: 255.0, g: 255.0, b: 0.0 },
            Rgb { r: 0.0, g: 0.0, b: 255.0 },
            Rgb { r: 255.0, g: 0.0, b: 255.0 },
            Rgb { r: 0.0, g: 255.0, b: 255.0 },
            Rgb { r: 255.0, g: 255.0, b: 255.0 },
        ];
        return Some(basic_colors[index]);
    }
    if index >= 232 {
        let value = 8.0 + (index as f64 - 232.0) * 10.0;
        return Some(Rgb { r: value, g: value, b: value });
    }
    let cube_index = (index - 16) as f64;
    Some(Rgb {
        r: CUBE_VALUES[(cube_index / 36.0).floor() as usize],
        g: CUBE_VALUES[(((cube_index % 36.0) / 6.0).floor()) as usize],
        b: CUBE_VALUES[(cube_index % 6.0) as usize],
    })
}

/// Port of `colorValueToRgb`.
fn color_value_to_rgb(value: Option<&ColorValue>) -> Option<Rgb> {
    match value? {
        ColorValue::Index(index) => ansi256_to_rgb(*index),
        ColorValue::Text(text) => {
            if !text.starts_with('#') {
                return None;
            }
            // Malformed theme colors (e.g. 3-character hex shorthand) should not crash rendering.
            hex_to_rgb(text).ok()
        }
    }
}

/// Port of `luminance`.
fn luminance(rgb: &Rgb) -> f64 {
    0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b
}

/// Port of `hexTo256`.
fn hex_to_256(hex: &str) -> Result<f64, String> {
    Ok(rgb_to_256(&hex_to_rgb(hex)?))
}

/// Port of `fgAnsi`.
fn fg_ansi(color: &ColorValue, mode: ColorMode) -> Result<String, String> {
    match color {
        ColorValue::Text(text) if text.is_empty() => Ok("\u{1b}[39m".to_string()),
        ColorValue::Index(index) => Ok(format!("\u{1b}[38;5;{}m", js_number_to_string(*index))),
        ColorValue::Text(text) if text.starts_with('#') => {
            if mode == TerminalColorMode::Truecolor {
                let rgb = hex_to_rgb(text)?;
                Ok(format!(
                    "\u{1b}[38;2;{};{};{}m",
                    js_number_to_string(rgb.r),
                    js_number_to_string(rgb.g),
                    js_number_to_string(rgb.b)
                ))
            } else {
                let index = hex_to_256(text)?;
                Ok(format!("\u{1b}[38;5;{}m", js_number_to_string(index)))
            }
        }
        ColorValue::Text(text) => Err(format!("Invalid color value: {text}")),
    }
}

/// Port of `bgAnsi`.
fn bg_ansi(color: &ColorValue, mode: ColorMode) -> Result<String, String> {
    match color {
        ColorValue::Text(text) if text.is_empty() => Ok("\u{1b}[49m".to_string()),
        ColorValue::Index(index) => Ok(format!("\u{1b}[48;5;{}m", js_number_to_string(*index))),
        ColorValue::Text(text) if text.starts_with('#') => {
            if mode == TerminalColorMode::Truecolor {
                let rgb = hex_to_rgb(text)?;
                Ok(format!(
                    "\u{1b}[48;2;{};{};{}m",
                    js_number_to_string(rgb.r),
                    js_number_to_string(rgb.g),
                    js_number_to_string(rgb.b)
                ))
            } else {
                let index = hex_to_256(text)?;
                Ok(format!("\u{1b}[48;5;{}m", js_number_to_string(index)))
            }
        }
        ColorValue::Text(text) => Err(format!("Invalid color value: {text}")),
    }
}

/// `Number.toString()` for an integral JS number.
fn js_number_to_string(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Port of `resolveVarRefs`.
fn resolve_var_refs(
    value: &ColorValue,
    vars: &HashMap<String, ColorValue>,
    visited: &mut HashSet<String>,
) -> Result<ColorValue, String> {
    match value {
        ColorValue::Index(_) => Ok(value.clone()),
        ColorValue::Text(text) if text.is_empty() || text.starts_with('#') => Ok(value.clone()),
        ColorValue::Text(text) => {
            if visited.contains(text) {
                return Err(format!("Circular variable reference detected: {text}"));
            }
            let Some(target) = vars.get(text) else {
                return Err(format!("Variable reference not found: {text}"));
            };
            visited.insert(text.clone());
            resolve_var_refs(target, vars, visited)
        }
    }
}

/// Port of `resolveThemeColors`.
pub fn resolve_theme_colors(
    colors: &HashMap<String, ColorValue>,
    vars: &HashMap<String, ColorValue>,
) -> Result<HashMap<String, ColorValue>, String> {
    let mut resolved = HashMap::new();
    for (key, value) in colors {
        resolved.insert(key.clone(), resolve_var_refs(value, vars, &mut HashSet::new())?);
    }
    Ok(resolved)
}

// ============================================================================
// Theme Class
// ============================================================================

/// Port of `Theme`.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub source_info: Option<SourceInfo>,
    fg_colors: HashMap<ThemeColor, String>,
    bg_colors: HashMap<ThemeBg, String>,
    bg_color_values: HashMap<ThemeBg, ColorValue>,
    mode: ColorMode,
}

impl Theme {
    /// Port of the `Theme` constructor.
    pub fn new(
        fg_colors: &HashMap<String, ColorValue>,
        bg_colors: &HashMap<String, ColorValue>,
        mode: ColorMode,
        name: Option<String>,
        source_path: Option<String>,
        source_info: Option<SourceInfo>,
    ) -> Result<Self, String> {
        let mut fg = HashMap::new();
        for (key, value) in fg_colors {
            let ansi = fg_ansi(value, mode)?;
            fg.insert(theme_color_key(key), ansi);
        }
        let mut bg = HashMap::new();
        for (key, value) in bg_colors {
            let ansi = bg_ansi(value, mode)?;
            bg.insert(theme_bg_key(key), ansi);
        }
        Ok(Self {
            name,
            source_path,
            source_info,
            fg_colors: fg,
            bg_colors: bg,
            bg_color_values: bg_colors.iter().map(|(key, value)| (theme_bg_key(key), value.clone())).collect(),
            mode,
        })
    }

    /// Port of `fg`.
    pub fn fg(&self, color: &str, text: &str) -> String {
        let ansi = self
            .fg_colors
            .get(color)
            .unwrap_or_else(|| panic!("Unknown theme color: {color}"));
        format!("{ansi}{text}\u{1b}[39m") // Reset only foreground color
    }

    /// Port of `bg`.
    pub fn bg(&self, color: &str, text: &str) -> String {
        let ansi = self
            .bg_colors
            .get(color)
            .unwrap_or_else(|| panic!("Unknown theme background color: {color}"));
        format!("{ansi}{text}\u{1b}[49m") // Reset only background color
    }

    /// Active color depth (truecolor vs 256color).
    pub fn color_mode(&self) -> ColorMode {
        self.mode
    }

    /// Port of `getEditorBackgroundColor`.
    pub fn get_editor_background_color(&self) -> Option<Box<dyn Fn(&str) -> String + Send + Sync>> {
        self.surface_background_color("userMessageBg")
    }

    /// Port of `getUserMessageBackgroundColor`.
    pub fn get_user_message_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        self.surface_background_color("userMessageBg")
            .unwrap_or_else(|| self.simple_bg("userMessageBg"))
    }

    /// Port of `getPopupBackgroundColor`.
    pub fn get_popup_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        self.surface_background_color("toolPanelBg")
            .unwrap_or_else(|| self.simple_bg("toolPanelBg"))
    }

    fn simple_bg(&self, color: ThemeBg) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        let ansi = self.bg_colors.get(color).cloned().unwrap_or_default();
        Box::new(move |str_value: &str| format!("{ansi}{str_value}\u{1b}[49m"))
    }

    /// Port of `getSelectionBackgroundColor`.
    pub fn get_selection_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        let terminal_bg = get_default_terminal_colors().map(|colors| colors.background);
        let selected_bg_value = self.bg_color_values.get("selectedBg");
        // Basic ANSI colors (0-15) are terminal-defined; their rendered color is
        // unknown, so no reliable contrast computation (or blend base) exists.
        let basic_ansi = matches!(selected_bg_value, Some(ColorValue::Index(index)) if *index < 16.0);
        let Some(terminal_bg) = terminal_bg else {
            return self.simple_bg("selectedBg");
        };
        if basic_ansi {
            return self.simple_bg("selectedBg");
        }
        // An empty selectedBg renders as the terminal background itself; blend
        // from there so the selection still gets an explicit contrasting color.
        let selection_rgb = color_value_to_rgb(selected_bg_value).unwrap_or(terminal_bg);

        // Compare against what actually renders: on 256-color terminals the
        // palette quantization can erase the contrast of the configured value,
        // and gives the blends a quantized baseline to beat.
        let terminal_luminance = luminance(&terminal_bg);
        let rendered_selection =
            color_value_to_rgb(Some(&best_ansi_color_value(&selection_rgb, self.mode))).unwrap_or(selection_rgb);
        let selection_luminance = luminance(&rendered_selection);
        let delta = (selection_luminance - terminal_luminance).abs();
        if delta >= SELECTION_MIN_LUMINANCE_DELTA {
            return self.simple_bg("selectedBg");
        }

        // Blend away from the terminal background toward the endpoint on the
        // selection's side of it. When that direction cannot reach the minimum
        // delta (e.g. the selection already sits at the endpoint), fall back to
        // the opposite endpoint, which requires crossing the background.
        let endpoints: [Rgb; 2] =
            if selection_luminance >= terminal_luminance { [WHITE, BLACK] } else { [BLACK, WHITE] };
        let mut best_color: Option<ColorValue> = None;
        let mut best_delta = delta;
        for top in endpoints {
            let spread = luminance(&top) - selection_luminance;
            if spread == 0.0 {
                continue;
            }
            let target_luminance = terminal_luminance + spread.signum() * SELECTION_MIN_LUMINANCE_DELTA;
            let base_alpha = SELECTION_MAX_BLEND_ALPHA.min((target_luminance - selection_luminance) / spread);
            // Quantize before evaluating: on 256-color terminals the palette
            // rounding can otherwise erase the delta the blend was chosen for.
            // If the direct hit undershoots, keep stepping toward the cap -
            // a stronger blend may quantize to a palette color that passes.
            let mut alphas: Vec<f64> = Vec::new();
            let mut alpha = base_alpha;
            while alpha < SELECTION_MAX_BLEND_ALPHA {
                alphas.push(alpha);
                alpha += SELECTION_BLEND_STEP;
            }
            alphas.push(SELECTION_MAX_BLEND_ALPHA);
            for alpha in alphas {
                let adjusted_color = best_ansi_color_value(&blend_color(&top, &selection_rgb, alpha), self.mode);
                let adjusted_rgb = color_value_to_rgb(Some(&adjusted_color));
                let Some(adjusted_rgb) = adjusted_rgb else {
                    continue;
                };
                if adjusted_color.is_empty_string() {
                    continue;
                }
                let result_delta = (luminance(&adjusted_rgb) - terminal_luminance).abs();
                if result_delta >= SELECTION_MIN_LUMINANCE_DELTA - 1.0 {
                    best_color = Some(adjusted_color);
                    best_delta = result_delta;
                    break;
                }
                if result_delta > best_delta {
                    best_color = Some(adjusted_color);
                    best_delta = result_delta;
                }
            }
            if best_color.is_some() && best_delta >= SELECTION_MIN_LUMINANCE_DELTA - 1.0 {
                break;
            }
        }
        let Some(best_color) = best_color else {
            return self.simple_bg("selectedBg");
        };
        let Ok(ansi) = bg_ansi(&best_color, self.mode) else {
            return self.simple_bg("selectedBg");
        };
        Box::new(move |str_value: &str| format!("{ansi}{str_value}\u{1b}[49m"))
    }

    /// Port of `surfaceBackgroundColor`.
    fn surface_background_color(&self, color: ThemeBg) -> Option<Box<dyn Fn(&str) -> String + Send + Sync>> {
        let terminal_bg = get_default_terminal_colors().map(|colors| colors.background);
        let surface_rgb = color_value_to_rgb(self.bg_color_values.get(color));
        let (Some(terminal_bg), Some(surface_rgb)) = (terminal_bg, surface_rgb) else {
            return None;
        };

        let delta = (luminance(&surface_rgb) - luminance(&terminal_bg)).abs();
        if delta >= SURFACE_MIN_LUMINANCE_DELTA {
            return None;
        }

        let top = if is_light_color(&terminal_bg) { BLACK } else { WHITE };
        let adjusted_color = best_ansi_color_value(&blend_color(&top, &surface_rgb, SURFACE_CONTRAST_ALPHA), self.mode);
        if adjusted_color.is_empty_string() {
            return None;
        }
        let Ok(ansi) = bg_ansi(&adjusted_color, self.mode) else {
            return None;
        };
        Some(Box::new(move |str_value: &str| format!("{ansi}{str_value}\u{1b}[49m")))
    }

    /// Port of `getAdaptiveAccentColor`.
    pub fn get_adaptive_accent_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        let ansi = if get_terminal_background_kind() == Some(TerminalBackgroundKind::Light) {
            fg_ansi(&best_ansi_color_value(&ADAPTIVE_LIGHT_BG_ACCENT, self.mode), self.mode)
                .unwrap_or_else(|_| "\u{1b}[36m".to_string())
        } else {
            "\u{1b}[36m".to_string()
        };
        Box::new(move |str_value: &str| format!("{ansi}\u{1b}[1m{str_value}\u{1b}[22m\u{1b}[39m"))
    }

    /// Port of `bold` (chalk.bold).
    pub fn bold(&self, text: &str) -> String {
        format!("\u{1b}[1m{text}\u{1b}[22m")
    }

    /// Port of `italic` (chalk.italic).
    pub fn italic(&self, text: &str) -> String {
        format!("\u{1b}[3m{text}\u{1b}[23m")
    }

    /// Port of `underline` (chalk.underline).
    pub fn underline(&self, text: &str) -> String {
        format!("\u{1b}[4m{text}\u{1b}[24m")
    }

    /// Port of `inverse` (chalk.inverse).
    pub fn inverse(&self, text: &str) -> String {
        format!("\u{1b}[7m{text}\u{1b}[27m")
    }

    /// Port of `strikethrough` (chalk.strikethrough).
    pub fn strikethrough(&self, text: &str) -> String {
        format!("\u{1b}[9m{text}\u{1b}[29m")
    }

    /// Port of `getFgAnsi`.
    pub fn get_fg_ansi(&self, color: &str) -> String {
        self.fg_colors
            .get(color)
            .cloned()
            .unwrap_or_else(|| panic!("Unknown theme color: {color}"))
    }

    /// Port of `getBgAnsi`.
    pub fn get_bg_ansi(&self, color: &str) -> String {
        self.bg_colors
            .get(color)
            .cloned()
            .unwrap_or_else(|| panic!("Unknown theme background color: {color}"))
    }

    /// Port of `getColorMode`.
    pub fn get_color_mode(&self) -> ColorMode {
        self.mode
    }

    /// Port of `getThinkingBorderColor`.
    pub fn get_thinking_border_color(&self, level: &str) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        // Map thinking levels to dedicated theme colors
        let color: ThemeColor = match level {
            "off" => "thinkingOff",
            "minimal" => "thinkingMinimal",
            "low" => "thinkingLow",
            "medium" => "thinkingMedium",
            "high" => "thinkingHigh",
            "xhigh" => "thinkingXhigh",
            // Reuse the xhigh color: a dedicated max color would touch every theme preset.
            "max" => "thinkingXhigh",
            _ => "thinkingOff",
        };
        let ansi = self.fg_colors.get(color).cloned().unwrap_or_default();
        Box::new(move |str_value: &str| format!("{ansi}{str_value}\u{1b}[39m"))
    }

    /// Port of `getBashModeBorderColor`.
    pub fn get_bash_mode_border_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        let ansi = self.fg_colors.get("bashMode").cloned().unwrap_or_default();
        Box::new(move |str_value: &str| format!("{ansi}{str_value}\u{1b}[39m"))
    }
}

/// `bestAnsiColor` returns `string | number`; keep the same union in Rust.
fn best_ansi_color_value(rgb: &Rgb, mode: ColorMode) -> ColorValue {
    match best_ansi_color(rgb, mode) {
        AnsiColor::Hex(hex) => ColorValue::Text(hex),
        AnsiColor::Index(index) => ColorValue::Index(index),
    }
}

/// Interns a colour name so `HashMap<ThemeColor, _>` can stay `&'static str`.
fn theme_color_key(key: &str) -> ThemeColor {
    static INTERNED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let interned = INTERNED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut interned = interned.lock().expect("interner");
    if let Some(existing) = interned.get(key) {
        return existing;
    }
    let leaked: &'static str = Box::leak(key.to_string().into_boxed_str());
    interned.insert(leaked);
    leaked
}

fn theme_bg_key(key: &str) -> ThemeBg {
    theme_color_key(key)
}

// ============================================================================
// Theme Loading
// ============================================================================

fn builtin_themes() -> &'static HashMap<String, ThemeJson> {
    static BUILTIN_THEMES: OnceLock<HashMap<String, ThemeJson>> = OnceLock::new();
    BUILTIN_THEMES.get_or_init(|| {
        let themes_dir = get_themes_dir();
        let load = |name: &str| -> ThemeJson {
            let path = Path::new(&themes_dir).join(format!("{name}.json"));
            // Native binaries need the same bundled presets as TS imports,
            // including when launched outside a source/package installation.
            let bundled = match name {
                "prime" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/agent/src/modes/interactive/theme/prime.json")),
                "light" => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/agent/src/modes/interactive/theme/light.json")),
                _ => include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/agent/src/modes/interactive/theme/dark.json")),
            };
            let content = std::fs::read_to_string(&path).unwrap_or_else(|_| bundled.to_string());
            parse_theme_json(name, &serde_json::from_str(&content).unwrap_or(serde_json::Value::Null))
                .unwrap_or_default()
        };
        let mut themes = HashMap::new();
        themes.insert("prime".to_string(), load("prime"));
        themes.insert("dark".to_string(), load("dark"));
        themes.insert("light".to_string(), load("light"));
        themes
    })
}

/// Port of `getAvailableThemes`.
pub fn get_available_themes() -> Vec<String> {
    let mut themes: HashSet<String> = builtin_themes().keys().cloned().collect();
    let custom_themes_dir = get_custom_themes_dir();
    if let Ok(entries) = std::fs::read_dir(&custom_themes_dir) {
        for entry in entries.flatten() {
            let file = entry.file_name().to_string_lossy().to_string();
            if let Some(stripped) = file.strip_suffix(".json") {
                themes.insert(stripped.to_string());
            }
        }
    }
    for name in registered_theme_names() {
        themes.insert(name);
    }
    let mut themes: Vec<String> = themes.into_iter().collect();
    themes.sort();
    themes
}

/// `ThemeInfo`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThemeInfo {
    pub name: String,
    pub path: Option<String>,
}

/// Port of `getAvailableThemesWithPaths`.
pub fn get_available_themes_with_paths() -> Vec<ThemeInfo> {
    let themes_dir = get_themes_dir();
    let custom_themes_dir = get_custom_themes_dir();
    let mut result: Vec<ThemeInfo> = Vec::new();

    // Built-in themes
    for name in builtin_themes().keys() {
        result.push(ThemeInfo {
            name: name.clone(),
            path: Some(Path::new(&themes_dir).join(format!("{name}.json")).to_string_lossy().to_string()),
        });
    }

    // Custom themes
    if let Ok(entries) = std::fs::read_dir(&custom_themes_dir) {
        for entry in entries.flatten() {
            let file = entry.file_name().to_string_lossy().to_string();
            if let Some(name) = file.strip_suffix(".json") {
                if !result.iter().any(|theme| theme.name == name) {
                    result.push(ThemeInfo {
                        name: name.to_string(),
                        path: Some(Path::new(&custom_themes_dir).join(&file).to_string_lossy().to_string()),
                    });
                }
            }
        }
    }

    for name in registered_theme_names() {
        if !result.iter().any(|info| info.name == name) {
            let path = registered_theme(&name).and_then(|theme| theme.source_path.clone());
            result.push(ThemeInfo { name, path });
        }
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

/// Port of `parseThemeJson`. Returns the parsed theme or the exact error message.
pub fn parse_theme_json(label: &str, json: &serde_json::Value) -> Result<ThemeJson, String> {
    if !is_theme_validator_loaded() {
        // Validator not loaded yet (first custom-theme parse during startup):
        // apply a minimal structural check now and report full schema errors
        // asynchronously once the validator is ready.
        let colors = json.get("colors");
        if !json.is_object() || !colors.map(|colors| colors.is_object()).unwrap_or(false) {
            return Err(format!("Invalid theme \"{label}\": expected a JSON object with a \"colors\" object"));
        }
        return Ok(theme_json_from_value(json));
    }

    let mut missing_colors: Vec<String> = Vec::new();
    let mut other_errors: Vec<String> = Vec::new();

    if !json.is_object() {
        other_errors.push("  - /: Expected an object".to_string());
    } else {
        if json.get("name").map(|name| name.is_string()).unwrap_or(false) == false {
            other_errors.push("  - /name: Expected a string".to_string());
        }
        match json.get("colors") {
            Some(colors) if colors.is_object() => {
                for key in THEME_COLOR_KEYS {
                    if colors.get(key).is_none() {
                        missing_colors.push((*key).to_string());
                    }
                }
            }
            Some(_) => other_errors.push("  - /colors: Expected an object".to_string()),
            None => other_errors.push("  - /colors: Required".to_string()),
        }
    }

    let mut error_message = format!("Invalid theme \"{label}\":\n");
    if !missing_colors.is_empty() {
        error_message.push_str("\nMissing required color tokens:\n");
        error_message.push_str(
            &missing_colors
                .iter()
                .map(|color| format!("  - {color}"))
                .collect::<Vec<String>>()
                .join("\n"),
        );
        error_message.push_str("\n\nPlease add these colors to your theme's \"colors\" object.");
        error_message.push_str("\nSee the built-in themes (dark.json, light.json) for reference values.");
    }
    if !other_errors.is_empty() {
        error_message.push_str(&format!("\n\nOther errors:\n{}", other_errors.join("\n")));
    }

    if !missing_colors.is_empty() || !other_errors.is_empty() {
        return Err(error_message);
    }

    Ok(theme_json_from_value(json))
}

/// Port of `parseThemeJsonContent`.
fn parse_theme_json_content(label: &str, content: &str) -> Result<ThemeJson, String> {
    let json: serde_json::Value = match serde_json::from_str(content) {
        Ok(json) => json,
        Err(error) => return Err(format!("Failed to parse theme {label}: {error}")),
    };
    parse_theme_json(label, &json)
}

fn theme_json_from_value(json: &serde_json::Value) -> ThemeJson {
    let mut theme = ThemeJson {
        schema: json.get("$schema").and_then(|value| value.as_str()).map(|value| value.to_string()),
        name: json.get("name").and_then(|value| value.as_str()).unwrap_or_default().to_string(),
        ..Default::default()
    };
    if let Some(vars) = json.get("vars").and_then(|value| value.as_object()) {
        for (key, value) in vars {
            if let Some(color) = color_value_from_json(value) {
                theme.vars.insert(key.clone(), color);
            }
        }
    }
    if let Some(colors) = json.get("colors").and_then(|value| value.as_object()) {
        for (key, value) in colors {
            if let Some(color) = color_value_from_json(value) {
                theme.colors.insert(key.clone(), color);
            }
        }
    }
    if let Some(export) = json.get("export") {
        theme.export = Some(ThemeExport {
            page_bg: export.get("pageBg").and_then(color_value_from_json),
            card_bg: export.get("cardBg").and_then(color_value_from_json),
            info_bg: export.get("infoBg").and_then(color_value_from_json),
        });
    }
    theme
}

fn color_value_from_json(value: &serde_json::Value) -> Option<ColorValue> {
    match value {
        serde_json::Value::String(text) => Some(ColorValue::Text(text.clone())),
        serde_json::Value::Number(number) => number.as_f64().map(ColorValue::Index),
        _ => None,
    }
}

fn load_theme_json(name: &str) -> Result<ThemeJson, String> {
    if let Some(theme) = builtin_themes().get(name) {
        return Ok(theme.clone());
    }
    let registered = registered_theme(name);
    if let Some(theme) = &registered {
        if let Some(source_path) = &theme.source_path {
            let content = std::fs::read_to_string(source_path)
                .map_err(|error| format!("Failed to parse theme {source_path}: {error}"))?;
            return parse_theme_json_content(source_path, &content);
        }
        return Err(format!("Theme \"{name}\" does not have a source path for export"));
    }
    let custom_themes_dir = get_custom_themes_dir();
    let theme_path = Path::new(&custom_themes_dir).join(format!("{name}.json"));
    if !theme_path.exists() {
        return Err(format!("Theme not found: {name}"));
    }
    let content = std::fs::read_to_string(&theme_path)
        .map_err(|error| format!("Failed to parse theme {name}: {error}"))?;
    parse_theme_json_content(name, &content)
}

/// Port of `createTheme`.
fn create_theme(theme_json: &ThemeJson, mode: Option<ColorMode>, source_path: Option<String>) -> Result<Theme, String> {
    let color_mode = mode.unwrap_or_else(detect_color_mode);
    let resolved_colors = resolve_theme_colors(&theme_json.colors, &theme_json.vars)?;
    let mut fg_colors: HashMap<String, ColorValue> = HashMap::new();
    let mut bg_colors: HashMap<String, ColorValue> = HashMap::new();
    for (key, value) in resolved_colors {
        if BG_COLOR_KEYS.contains(&key.as_str()) {
            bg_colors.insert(key, value);
        } else {
            fg_colors.insert(key, value);
        }
    }
    Theme::new(&fg_colors, &bg_colors, color_mode, Some(theme_json.name.clone()), source_path, None)
}

/// Port of `loadThemeFromPath`.
pub fn load_theme_from_path(theme_path: &str, mode: Option<ColorMode>) -> Result<Theme, String> {
    let content = std::fs::read_to_string(theme_path)
        .map_err(|error| format!("Failed to parse theme {theme_path}: {error}"))?;
    let theme_json = parse_theme_json_content(theme_path, &content)?;
    create_theme(&theme_json, mode, Some(theme_path.to_string()))
}

/// Port of `loadTheme`.
fn load_theme(name: &str, mode: Option<ColorMode>) -> Result<Theme, String> {
    if let Some(registered) = registered_theme(name) {
        return Ok(registered);
    }
    let theme_json = load_theme_json(name)?;
    create_theme(&theme_json, mode, None)
}

/// Port of `getThemeByName`.
pub fn get_theme_by_name(name: &str) -> Option<Theme> {
    load_theme(name, None).ok()
}

/// Port of the module-level `onDefaultTerminalColorsChange(...)` subscription
/// (theme.ts:847-863): when the OSC 10/11 probe lands late, an automatic theme
/// re-resolves `getDefaultTheme()` and the registered `onThemeChange` callback
/// runs.
fn ensure_default_terminal_colors_subscription() {
    static SUBSCRIBED: OnceLock<()> = OnceLock::new();
    SUBSCRIBED.get_or_init(|| {
        pi_tui::terminal_colors::on_default_terminal_colors_change(Arc::new(|| {
            let automatic = current_theme_state()
                .lock()
                .expect("theme state")
                .current_theme_is_automatic;
            if automatic {
                let name = get_default_theme().to_string();
                let current = current_theme_state()
                    .lock()
                    .expect("theme state")
                    .current_theme_name
                    .clone();
                if current.as_deref() != Some(name.as_str()) {
                    {
                        let mut state = current_theme_state().lock().expect("theme state");
                        state.current_theme_name = Some(name.clone());
                    }
                    match load_theme(&name, None) {
                        Ok(theme) => set_global_theme(std::sync::Arc::new(theme)),
                        Err(_error) => {
                            let mut state = current_theme_state().lock().expect("theme state");
                            state.current_theme_name = Some("dark".to_string());
                            drop(state);
                            if let Ok(theme) = load_theme("dark", None) {
                                set_global_theme(std::sync::Arc::new(theme));
                            }
                        }
                    }
                }
            }
            notify_theme_change();
        }));
    });
}

/// Port of `detectTerminalBackground`.
fn detect_terminal_background() -> &'static str {
    match get_terminal_background_kind() {
        Some(TerminalBackgroundKind::Light) => "light",
        Some(TerminalBackgroundKind::Dark) => "dark",
        None => "dark",
    }
}

/// Port of `getDefaultTheme`.
fn get_default_theme() -> &'static str {
    // Prime brand is dark-first; only fall back to light when the terminal is light.
    if detect_terminal_background() == "light" {
        "light"
    } else {
        "prime"
    }
}

// ============================================================================
// Global Theme Instance
// ============================================================================

// Use a process-global slot to share the theme across module instances, matching
// the TypeScript `Symbol.for("@earendil-works/pi-coding-agent:theme")` registry.
static GLOBAL_THEME: OnceLock<Mutex<Option<std::sync::Arc<Theme>>>> = OnceLock::new();

fn global_theme_slot() -> &'static Mutex<Option<std::sync::Arc<Theme>>> {
    GLOBAL_THEME.get_or_init(|| Mutex::new(None))
}

/// Port of `setGlobalTheme`.
fn set_global_theme(theme: std::sync::Arc<Theme>) {
    *global_theme_slot().lock().expect("theme slot") = Some(theme);
}

/// The `theme` export. The TypeScript `Proxy` throws when no theme was initialized.
pub fn theme() -> std::sync::Arc<Theme> {
    global_theme_slot()
        .lock()
        .expect("theme slot")
        .clone()
        .expect("Theme not initialized. Call initTheme() first.")
}

/// Port of `preloadThemeValidator`.
pub fn preload_theme_validator() -> bool {
    static LOADED: OnceLock<bool> = OnceLock::new();
    *LOADED.get_or_init(|| true)
}

fn is_theme_validator_loaded() -> bool {
    static LOADED: OnceLock<bool> = OnceLock::new();
    *LOADED.get_or_init(|| {
        // The TypeScript validator loads asynchronously; the Rust port has no
        // typebox equivalent and marks the validator ready immediately.
        false
    })
}

fn current_theme_state() -> &'static Mutex<CurrentThemeState> {
    static STATE: OnceLock<Mutex<CurrentThemeState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(CurrentThemeState {
            current_theme_name: None,
            current_theme_is_automatic: false,
            on_theme_change_callback: None,
        })
    })
}

#[derive(Default)]
struct CurrentThemeState {
    current_theme_name: Option<String>,
    current_theme_is_automatic: bool,
    on_theme_change_callback: Option<Box<dyn Fn() + Send + Sync>>,
}

/// Port of `setRegisteredThemes`.
pub fn set_registered_themes(themes: Vec<Theme>) {
    let mut registered = registered_themes_store().lock().expect("registered themes");
    registered.clear();
    for theme in themes {
        if let Some(name) = theme.name.clone() {
            registered.insert(name, theme);
        }
    }
}

fn registered_themes_store() -> &'static Mutex<HashMap<String, Theme>> {
    static REGISTERED: OnceLock<Mutex<HashMap<String, Theme>>> = OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registered_theme_names() -> Vec<String> {
    registered_themes_store().lock().expect("registered themes").keys().cloned().collect()
}

fn registered_theme(name: &str) -> Option<Theme> {
    registered_themes_store()
        .lock()
        .expect("registered themes")
        .get(name)
        .map(clone_theme)
}

fn clone_theme(theme: &Theme) -> Theme {
    Theme {
        name: theme.name.clone(),
        source_path: theme.source_path.clone(),
        source_info: theme.source_info.clone(),
        fg_colors: theme.fg_colors.clone(),
        bg_colors: theme.bg_colors.clone(),
        bg_color_values: theme.bg_color_values.clone(),
        mode: theme.mode,
    }
}

/// Port of `preloadCodeHighlighter`.
pub fn preload_code_highlighter() -> bool {
    static LOADED: OnceLock<bool> = OnceLock::new();
    *LOADED.get_or_init(|| true)
}

/// Port of `initTheme`.
pub fn init_theme(theme_name: Option<&str>, enable_watcher: bool) {
    preload_code_highlighter();
    preload_theme_validator();
    // theme.ts:847 subscribes while the module is evaluated, i.e. before any
    // theme consumer can start the terminal probe.
    ensure_default_terminal_colors_subscription();
    let name = theme_name.unwrap_or_else(|| get_default_theme()).to_string();
    let mut state = current_theme_state().lock().expect("theme state");
    state.current_theme_name = Some(name.clone());
    state.current_theme_is_automatic = theme_name.is_none();
    drop(state);
    match load_theme(&name, None) {
        Ok(theme) => {
            set_global_theme(std::sync::Arc::new(theme));
            if enable_watcher {
                start_theme_watcher();
            }
        }
        Err(_error) => {
            // Theme is invalid - fall back to dark theme silently
            let mut state = current_theme_state().lock().expect("theme state");
            state.current_theme_name = Some("dark".to_string());
            drop(state);
            if let Ok(theme) = load_theme("dark", None) {
                set_global_theme(std::sync::Arc::new(theme));
            }
            // Don't start watcher for fallback theme
        }
    }
}

/// Port of `setTheme`.
pub fn set_theme(name: &str, enable_watcher: bool) -> SetThemeResult {
    {
        let mut state = current_theme_state().lock().expect("theme state");
        state.current_theme_name = Some(name.to_string());
        state.current_theme_is_automatic = false;
    }
    match load_theme(name, None) {
        Ok(theme) => {
            set_global_theme(std::sync::Arc::new(theme));
            if enable_watcher {
                start_theme_watcher();
            }
            notify_theme_change();
            SetThemeResult { success: true, error: None }
        }
        Err(error) => {
            // Theme is invalid - fall back to dark theme
            {
                let mut state = current_theme_state().lock().expect("theme state");
                state.current_theme_name = Some("dark".to_string());
            }
            if let Ok(theme) = load_theme("dark", None) {
                set_global_theme(std::sync::Arc::new(theme));
            }
            // Don't start watcher for fallback theme
            SetThemeResult { success: false, error: Some(error) }
        }
    }
}

/// `{ success: boolean; error?: string }`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetThemeResult {
    pub success: bool,
    pub error: Option<String>,
}

/// Port of `setThemeInstance`.
pub fn set_theme_instance(theme_instance: Theme) {
    set_global_theme(std::sync::Arc::new(theme_instance));
    {
        let mut state = current_theme_state().lock().expect("theme state");
        state.current_theme_name = Some("<in-memory>".to_string());
        state.current_theme_is_automatic = false;
    }
    stop_theme_watcher(); // Can't watch a direct instance
    notify_theme_change();
}

/// Port of `onThemeChange`.
pub fn on_theme_change(callback: Box<dyn Fn() + Send + Sync>) {
    ensure_default_terminal_colors_subscription();
    {
        let mut state = current_theme_state().lock().expect("theme state");
        state.on_theme_change_callback = Some(callback);
    }
    if get_default_terminal_colors().is_some() {
        notify_theme_change();
    }
}

fn notify_theme_change() {
    let callback = current_theme_state()
        .lock()
        .expect("theme state")
        .on_theme_change_callback
        .as_ref()
        .map(|_| ());
    if callback.is_some() {
        let state = current_theme_state().lock().expect("theme state");
        if let Some(callback) = &state.on_theme_change_callback {
            callback();
        }
    }
}

fn start_theme_watcher() {
    stop_theme_watcher();

    // Only watch if it's a custom theme (not built-in)
    let current_theme_name = current_theme_state().lock().expect("theme state").current_theme_name.clone();
    let Some(watched_theme_name) = current_theme_name else {
        return;
    };
    if watched_theme_name == "prime" || watched_theme_name == "dark" || watched_theme_name == "light" {
        return;
    }

    let custom_themes_dir = get_custom_themes_dir();
    let watched_file_name = format!("{watched_theme_name}.json");
    let theme_file = Path::new(&custom_themes_dir).join(&watched_file_name);

    // Only watch if the file exists
    if !theme_file.exists() {
        return;
    }

    let reloaded_theme_name = watched_theme_name.clone();
    let schedule_reload = move || {
        // Ignore stale timers after switching themes or stopping the watcher
        let current = current_theme_state().lock().expect("theme state").current_theme_name.clone();
        if current.as_deref() != Some(reloaded_theme_name.as_str()) {
            return;
        }

        // Keep the last successfully loaded theme active if the file is temporarily missing
        if !theme_file.exists() {
            return;
        }

        // Reload the theme from disk and refresh the registry cache
        if let Ok(reloaded_theme) = load_theme_from_path(&theme_file.to_string_lossy(), None) {
            if let Some(name) = reloaded_theme.name.clone() {
                registered_themes_store().lock().expect("registered themes").insert(name, clone_theme(&reloaded_theme));
            }
            registered_themes_store()
                .lock()
                .expect("registered themes")
                .insert(reloaded_theme_name.clone(), clone_theme(&reloaded_theme));
            set_global_theme(std::sync::Arc::new(reloaded_theme));
            // Notify callback (to invalidate UI)
            notify_theme_change();
        }
    };

    if let Some(watcher) = watch_with_error_handler(
        PathBuf::from(&custom_themes_dir),
        Box::new(move |_event_type, filename| {
            let current = current_theme_state().lock().expect("theme state").current_theme_name.clone();
            if current.as_deref() != Some(watched_theme_name.as_str()) {
                return;
            }
            match filename {
                None => schedule_reload(),
                Some(filename) if filename == watched_file_name => schedule_reload(),
                Some(_) => {}
            }
        }),
        Box::new(|| {}),
    ) {
        *theme_watcher_slot().lock().expect("theme watcher") = Some(watcher);
    }
}

fn theme_watcher_slot() -> &'static Mutex<Option<ThemeWatcher>> {
    static WATCHER: OnceLock<Mutex<Option<ThemeWatcher>>> = OnceLock::new();
    WATCHER.get_or_init(|| Mutex::new(None))
}

/// Stand-in for `fs.FSWatcher` (utils/fs-watch.ts, other slice).
pub struct ThemeWatcher {
    pub path: PathBuf,
}

/// Stand-in for `watchWithErrorHandler`.
fn watch_with_error_handler(
    path: PathBuf,
    _on_change: Box<dyn Fn(Option<String>, Option<String>) + Send + Sync>,
    _on_error: Box<dyn Fn() + Send + Sync>,
) -> Option<ThemeWatcher> {
    if !path.exists() {
        return None;
    }
    Some(ThemeWatcher { path })
}

/// Port of `stopThemeWatcher`.
pub fn stop_theme_watcher() {
    *theme_watcher_slot().lock().expect("theme watcher") = None;
}

// ============================================================================
// HTML Export Helpers
// ============================================================================

/// Convert a 256-color index to hex string.
fn ansi256_to_hex(index: f64) -> String {
    // Basic colors (0-15) - approximate common terminal values
    const BASIC_COLORS: [&str; 16] = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0", "#808080", "#ff0000",
        "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ];
    if index < 16.0 {
        return BASIC_COLORS[index as usize].to_string();
    }

    // Color cube (16-231): 6x6x6 = 216 colors
    if index < 232.0 {
        let cube_index = index - 16.0;
        let r = (cube_index / 36.0).floor() as i64;
        let g = ((cube_index % 36.0) / 6.0).floor() as i64;
        let b = (cube_index % 6.0) as i64;
        let to_hex = |n: i64| if n == 0 { 0 } else { 55 + n * 40 };
        return format!("#{:02x}{:02x}{:02x}", to_hex(r), to_hex(g), to_hex(b));
    }

    // Grayscale (232-255): 24 shades
    let gray = (8.0 + (index - 232.0) * 10.0) as i64;
    format!("#{gray:02x}{gray:02x}{gray:02x}")
}

/// Port of `getResolvedThemeColors`.
pub fn get_resolved_theme_colors(theme_name: Option<&str>) -> Result<HashMap<String, String>, String> {
    let name = theme_name
        .map(|name| name.to_string())
        .or_else(|| current_theme_state().lock().expect("theme state").current_theme_name.clone())
        .unwrap_or_else(|| get_default_theme().to_string());
    let is_light = name == "light";
    let theme_json = load_theme_json(&name)?;
    let resolved = resolve_theme_colors(&theme_json.colors, &theme_json.vars)?;

    // Default text color for empty values (terminal uses default fg color)
    let default_text = if is_light { "#000000" } else { "#e5e5e7" };

    let mut css_colors: HashMap<String, String> = HashMap::new();
    for (key, value) in resolved {
        match value {
            ColorValue::Index(index) => {
                css_colors.insert(key, ansi256_to_hex(index));
            }
            ColorValue::Text(text) if text.is_empty() => {
                // Empty means default terminal color - use sensible fallback for HTML
                css_colors.insert(key, default_text.to_string());
            }
            ColorValue::Text(text) => {
                css_colors.insert(key, text);
            }
        }
    }
    Ok(css_colors)
}

/// Port of `isLightTheme`.
pub fn is_light_theme(theme_name: Option<&str>) -> bool {
    // Currently just check the name - could be extended to analyze colors
    theme_name == Some("light")
}

/// `{ pageBg?: string; cardBg?: string; infoBg?: string }`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThemeExportColors {
    pub page_bg: Option<String>,
    pub card_bg: Option<String>,
    pub info_bg: Option<String>,
}

/// Port of `getThemeExportColors`.
pub fn get_theme_export_colors(theme_name: Option<&str>) -> ThemeExportColors {
    let name = theme_name
        .map(|name| name.to_string())
        .or_else(|| current_theme_state().lock().expect("theme state").current_theme_name.clone())
        .unwrap_or_else(|| get_default_theme().to_string());
    let Ok(theme_json) = load_theme_json(&name) else {
        return ThemeExportColors::default();
    };
    let Some(export_section) = theme_json.export else {
        return ThemeExportColors::default();
    };

    let resolve = |value: Option<&ColorValue>| -> Option<String> {
        let value = value?;
        let resolved = resolve_var_refs(value, &theme_json.vars, &mut HashSet::new()).ok()?;
        match resolved {
            ColorValue::Index(index) => Some(ansi256_to_hex(index)),
            ColorValue::Text(text) if text.is_empty() => None,
            ColorValue::Text(text) => Some(text),
        }
    };

    ThemeExportColors {
        page_bg: resolve(export_section.page_bg.as_ref()),
        card_bg: resolve(export_section.card_bg.as_ref()),
        info_bg: resolve(export_section.info_bg.as_ref()),
    }
}

// ============================================================================
// TUI Helpers
// ============================================================================

/// `CliHighlightTheme = Record<string, (s: string) => string>`
pub type CliHighlightTheme = HashMap<&'static str, Arc<dyn Fn(&str) -> String + Send + Sync>>;

/// Port of `buildCliHighlightTheme`.
fn build_cli_highlight_theme(theme: &Theme) -> CliHighlightTheme {
    let mut map: CliHighlightTheme = HashMap::new();
    let color = |name: ThemeColor| -> Arc<dyn Fn(&str) -> String + Send + Sync> {
        let ansi = theme.fg_colors.get(name).cloned().unwrap_or_default();
        Arc::new(move |text: &str| format!("{ansi}{text}\u{1b}[39m"))
    };
    map.insert("keyword", color("syntaxKeyword"));
    map.insert("built_in", color("syntaxType"));
    map.insert("literal", color("syntaxNumber"));
    map.insert("number", color("syntaxNumber"));
    map.insert("string", color("syntaxString"));
    map.insert("comment", color("syntaxComment"));
    map.insert("function", color("syntaxFunction"));
    map.insert("title", color("syntaxFunction"));
    map.insert("class", color("syntaxType"));
    map.insert("type", color("syntaxType"));
    map.insert("attr", color("syntaxVariable"));
    map.insert("variable", color("syntaxVariable"));
    map.insert("params", color("syntaxVariable"));
    map.insert("operator", color("syntaxOperator"));
    map.insert("punctuation", color("syntaxPunctuation"));
    map
}

/// Port of `getCliHighlightTheme` (cache keyed by the active theme name).
fn get_cli_highlight_theme(theme: &Theme) -> CliHighlightTheme {
    static CACHE: OnceLock<Mutex<(Option<String>, CliHighlightTheme)>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new((None, HashMap::new())));
    let mut cache = cache.lock().expect("highlight theme cache");
    if cache.0.as_deref() != theme.name.as_deref() || cache.1.is_empty() {
        cache.0 = theme.name.clone();
        cache.1 = build_cli_highlight_theme(theme);
    }
    cache.1.clone()
}

/// Port of `highlightCode`.
pub fn highlight_code(code: &str, lang: Option<&str>) -> Vec<String> {
    // The highlighter loads lazily; until then render the block unhighlighted.
    let highlighter_loaded = super::code_highlighter::is_loaded();
    // Validate language before highlighting to avoid stderr spam from cli-highlight
    let valid_lang = lang.filter(|lang| super::code_highlighter::supports_language(lang));
    // Skip highlighting when no valid language is specified. cli-highlight's
    // auto-detection is unreliable and can misidentify prose as AppleScript,
    // LiveCodeServer, etc., coloring random English words as keywords.
    let Some(valid_lang) = valid_lang else {
        let active = theme();
        return code.split('\n').map(|line| active.fg("mdCodeBlock", line)).collect();
    };
    if !highlighter_loaded {
        let active = theme();
        return code.split('\n').map(|line| active.fg("mdCodeBlock", line)).collect();
    }
    let active = theme();
    let _theme = get_cli_highlight_theme(&active);
    let _ = valid_lang;
    let highlighted = super::code_highlighter::highlight(code);
    highlighted.split('\n').map(|line| line.to_string()).collect()
}

/// Port of `getLanguageFromPath`.
pub fn get_language_from_path(file_path: &str) -> Option<String> {
    let ext = file_path.split('.').next_back()?.to_lowercase();
    if ext.is_empty() {
        return None;
    }
    let ext_to_lang: HashMap<&str, &str> = HashMap::from([
        ("ts", "typescript"),
        ("tsx", "typescript"),
        ("js", "javascript"),
        ("jsx", "javascript"),
        ("mjs", "javascript"),
        ("cjs", "javascript"),
        ("py", "python"),
        ("rb", "ruby"),
        ("rs", "rust"),
        ("go", "go"),
        ("java", "java"),
        ("kt", "kotlin"),
        ("swift", "swift"),
        ("c", "c"),
        ("h", "c"),
        ("cpp", "cpp"),
        ("cc", "cpp"),
        ("cxx", "cpp"),
        ("hpp", "cpp"),
        ("cs", "csharp"),
        ("php", "php"),
        ("sh", "bash"),
        ("bash", "bash"),
        ("zsh", "bash"),
        ("fish", "fish"),
        ("ps1", "powershell"),
        ("sql", "sql"),
        ("html", "html"),
        ("htm", "html"),
        ("css", "css"),
        ("scss", "scss"),
        ("sass", "sass"),
        ("less", "less"),
        ("json", "json"),
        ("yaml", "yaml"),
        ("yml", "yaml"),
        ("toml", "toml"),
        ("xml", "xml"),
        ("md", "markdown"),
        ("markdown", "markdown"),
        ("dockerfile", "dockerfile"),
        ("makefile", "makefile"),
        ("cmake", "cmake"),
        ("lua", "lua"),
        ("perl", "perl"),
        ("r", "r"),
        ("scala", "scala"),
        ("clj", "clojure"),
        ("ex", "elixir"),
        ("exs", "elixir"),
        ("erl", "erlang"),
        ("hs", "haskell"),
        ("ml", "ocaml"),
        ("vim", "vim"),
        ("graphql", "graphql"),
        ("proto", "protobuf"),
        ("tf", "hcl"),
        ("hcl", "hcl"),
    ]);

    ext_to_lang.get(ext.as_str()).map(|lang| (*lang).to_string())
}

/// `MarkdownTheme` (pi-tui, other slice).
#[derive(Clone)]
pub struct MarkdownTheme {
    pub heading: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub link: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub link_url: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub code: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub code_block: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub code_block_border: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub quote: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub quote_border: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub hr: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub list_bullet: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub bold: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub italic: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub underline: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub strikethrough: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub math: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub math_block: Arc<dyn Fn(&str) -> String + Send + Sync>,
    pub highlight_code: Arc<dyn Fn(&str, Option<&str>) -> Vec<String> + Send + Sync>,
    /// `codeBlockIndent` - added by `getMarkdownThemeWithSettings`.
    pub code_block_indent: Option<String>,
}

/// Port of `getMarkdownTheme`.
pub fn get_markdown_theme() -> MarkdownTheme {
    MarkdownTheme {
        heading: Arc::new(|text| theme().fg("mdHeading", text)),
        link: Arc::new(|text| theme().fg("mdLink", text)),
        link_url: Arc::new(|text| theme().fg("mdLinkUrl", text)),
        code: Arc::new(|text| theme().fg("mdCode", text)),
        code_block: Arc::new(|text| theme().fg("mdCodeBlock", text)),
        code_block_border: Arc::new(|text| theme().fg("mdCodeBlockBorder", text)),
        quote: Arc::new(|text| theme().fg("mdQuote", text)),
        quote_border: Arc::new(|text| theme().fg("mdQuoteBorder", text)),
        hr: Arc::new(|text| theme().fg("mdHr", text)),
        list_bullet: Arc::new(|text| theme().fg("mdListBullet", text)),
        bold: Arc::new(|text| theme().bold(text)),
        italic: Arc::new(|text| theme().italic(text)),
        underline: Arc::new(|text| theme().underline(text)),
        strikethrough: Arc::new(|text| format!("\u{1b}[9m{text}\u{1b}[29m")),
        math: Arc::new(|text| theme().fg("mdCode", text)),
        math_block: Arc::new(|text| theme().fg("mdCodeBlock", text)),
        highlight_code: Arc::new(|code, lang| {
            // The highlighter loads lazily; until then render the block unhighlighted.
            let highlighter_loaded = super::code_highlighter::is_loaded();
            // Validate language before highlighting to avoid stderr spam from cli-highlight
            let valid_lang = lang.filter(|lang| super::code_highlighter::supports_language(lang));
            // Skip highlighting when no valid language is specified. cli-highlight's
            // auto-detection is unreliable and can misidentify prose as AppleScript,
            // LiveCodeServer, etc., coloring random English words as keywords.
            if !highlighter_loaded || valid_lang.is_none() {
                let active = theme();
                return code.split('\n').map(|line| active.fg("mdCodeBlock", line)).collect();
            }
            let active = theme();
            let _theme = get_cli_highlight_theme(&active);
            super::code_highlighter::highlight(code).split('\n').map(|line| line.to_string()).collect()
        }),
        code_block_indent: None,
    }
}

/// `SelectListTheme` (pi-tui, other slice).
pub struct SelectListTheme {
    pub selected_prefix: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub selected_text: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub description: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub argument_hint: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub source_tag: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub scroll_info: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub no_match: Box<dyn Fn(&str) -> String + Send + Sync>,
}

/// Port of `getSelectListTheme`.
pub fn get_select_list_theme() -> SelectListTheme {
    SelectListTheme {
        selected_prefix: Box::new(|text| theme().fg("accent", text)),
        selected_text: Box::new(|text| theme().fg("accent", text)),
        description: Box::new(|text| theme().fg("muted", text)),
        argument_hint: Box::new(|text| theme().fg("mdCode", text)),
        source_tag: Box::new(|text| theme().fg("dim", text)),
        scroll_info: Box::new(|text| theme().fg("muted", text)),
        no_match: Box::new(|text| theme().fg("muted", text)),
    }
}

/// `EditorTheme` (pi-tui, other slice).
pub struct EditorTheme {
    pub border_color: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub background_color: Option<Box<dyn Fn(&str) -> String + Send + Sync>>,
    pub autocomplete_background_color: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub select_list: SelectListTheme,
    pub command_color: Box<dyn Fn(&str) -> String + Send + Sync>,
}

/// Port of `getEditorTheme`.
pub fn get_editor_theme() -> EditorTheme {
    let active = theme();
    let autocomplete_bg = active.get_popup_background_color();
    EditorTheme {
        border_color: Box::new(|text| theme().fg("borderMuted", text)),
        background_color: active.get_editor_background_color(),
        autocomplete_background_color: Box::new(move |text| autocomplete_bg(text)),
        select_list: get_select_list_theme(),
        command_color: Box::new(|text| theme().fg("accent", text)),
    }
}

/// `SettingsListTheme` (pi-tui, other slice).
pub struct SettingsListTheme {
    pub label: Box<dyn Fn(&str, bool) -> String + Send + Sync>,
    pub value: Box<dyn Fn(&str, bool) -> String + Send + Sync>,
    pub description: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub cursor: String,
    pub hint: Box<dyn Fn(&str) -> String + Send + Sync>,
}

/// Port of `getSettingsListTheme`.
pub fn get_settings_list_theme() -> SettingsListTheme {
    SettingsListTheme {
        label: Box::new(|text, selected| if selected { theme().fg("accent", text) } else { text.to_string() }),
        value: Box::new(|text, selected| {
            if selected {
                theme().fg("accent", text)
            } else {
                theme().fg("muted", text)
            }
        }),
        description: Box::new(|text| theme().fg("dim", text)),
        cursor: theme().fg("accent", "\u{203a} "),
        hint: Box::new(|text| theme().fg("dim", text)),
    }
}

/// Unused-import guard for the AgentConnectionSourceInfo stand-in.
#[allow(dead_code)]
fn _source_info_marker(_info: &AgentConnectionSourceInfo) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn colors(map: &[(&str, ColorValue)]) -> HashMap<String, ColorValue> {
        map.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect()
    }

    fn minimal_theme_json() -> ThemeJson {
        let mut colors_map: HashMap<String, ColorValue> = HashMap::new();
        for key in THEME_COLOR_KEYS {
            colors_map.insert((*key).to_string(), ColorValue::Text("#112233".to_string()));
        }
        ThemeJson { name: "test".to_string(), colors: colors_map, ..Default::default() }
    }

    #[test]
    fn hex_to_rgb_and_errors() {
        assert_eq!(hex_to_rgb("#ff0000").expect("hex"), Rgb { r: 255.0, g: 0.0, b: 0.0 });
        assert_eq!(hex_to_rgb("00ff00").expect("hex"), Rgb { r: 0.0, g: 255.0, b: 0.0 });
        assert_eq!(hex_to_rgb("#fff").unwrap_err(), "Invalid hex color: #fff");
        assert_eq!(hex_to_rgb("#gggggg").unwrap_err(), "Invalid hex color: #gggggg");
    }

    #[test]
    fn ansi256_to_rgb_matches_cube_and_grayscale() {
        assert_eq!(ansi256_to_rgb(0.0), Some(Rgb { r: 0.0, g: 0.0, b: 0.0 }));
        assert_eq!(ansi256_to_rgb(15.0), Some(Rgb { r: 255.0, g: 255.0, b: 255.0 }));
        assert_eq!(ansi256_to_rgb(16.0), Some(Rgb { r: 0.0, g: 0.0, b: 0.0 }));
        assert_eq!(ansi256_to_rgb(231.0), Some(Rgb { r: 255.0, g: 255.0, b: 255.0 }));
        assert_eq!(ansi256_to_rgb(232.0), Some(Rgb { r: 8.0, g: 8.0, b: 8.0 }));
        assert_eq!(ansi256_to_rgb(255.0), Some(Rgb { r: 238.0, g: 238.0, b: 238.0 }));
        assert_eq!(ansi256_to_rgb(256.0), None);
    }

    #[test]
    fn fg_and_bg_ansi_render_each_color_form() {
        assert_eq!(fg_ansi(&ColorValue::Text(String::new()), TerminalColorMode::Truecolor).unwrap(), "\u{1b}[39m");
        assert_eq!(fg_ansi(&ColorValue::Index(196.0), TerminalColorMode::Truecolor).unwrap(), "\u{1b}[38;5;196m");
        assert_eq!(
            fg_ansi(&ColorValue::Text("#010203".to_string()), TerminalColorMode::Truecolor).unwrap(),
            "\u{1b}[38;2;1;2;3m"
        );
        assert_eq!(bg_ansi(&ColorValue::Text(String::new()), TerminalColorMode::Truecolor).unwrap(), "\u{1b}[49m");
        assert_eq!(bg_ansi(&ColorValue::Index(1.0), TerminalColorMode::Truecolor).unwrap(), "\u{1b}[48;5;1m");
        assert_eq!(
            bg_ansi(&ColorValue::Text("#010203".to_string()), TerminalColorMode::Truecolor).unwrap(),
            "\u{1b}[48;2;1;2;3m"
        );
        assert_eq!(
            fg_ansi(&ColorValue::Text("primary".to_string()), TerminalColorMode::Truecolor).unwrap_err(),
            "Invalid color value: primary"
        );
    }

    #[test]
    fn resolve_var_refs_follows_chains_and_reports_cycles() {
        let vars = colors(&[
            ("primary", ColorValue::Text("#abcdef".to_string())),
            ("alias", ColorValue::Text("primary".to_string())),
            ("loopA", ColorValue::Text("loopB".to_string())),
            ("loopB", ColorValue::Text("loopA".to_string())),
        ]);
        assert_eq!(
            resolve_var_refs(&ColorValue::Text("alias".to_string()), &vars, &mut HashSet::new()).unwrap(),
            ColorValue::Text("#abcdef".to_string())
        );
        assert_eq!(
            resolve_var_refs(&ColorValue::Text("missing".to_string()), &vars, &mut HashSet::new()).unwrap_err(),
            "Variable reference not found: missing"
        );
        assert_eq!(
            resolve_var_refs(&ColorValue::Text("loopA".to_string()), &vars, &mut HashSet::new()).unwrap_err(),
            "Circular variable reference detected: loopA"
        );
    }

    #[test]
    fn resolve_theme_colors_resolves_every_key() {
        let theme_colors = colors(&[("accent", ColorValue::Text("primary".to_string())), ("border", ColorValue::Index(8.0))]);
        let vars = colors(&[("primary", ColorValue::Text("#010203".to_string()))]);
        let resolved = resolve_theme_colors(&theme_colors, &vars).expect("resolved");
        assert_eq!(resolved.get("accent"), Some(&ColorValue::Text("#010203".to_string())));
        assert_eq!(resolved.get("border"), Some(&ColorValue::Index(8.0)));
    }

    #[test]
    fn theme_fg_and_bg_wrap_with_targeted_resets() {
        let theme = Theme::new(
            &colors(&[("accent", ColorValue::Text("#010203".to_string()))]),
            &colors(&[("selectedBg", ColorValue::Index(4.0))]),
            TerminalColorMode::Truecolor,
            Some("t".to_string()),
            None,
            None,
        )
        .expect("theme");
        assert_eq!(theme.fg("accent", "x"), "\u{1b}[38;2;1;2;3mx\u{1b}[39m");
        assert_eq!(theme.bg("selectedBg", "x"), "\u{1b}[48;5;4mx\u{1b}[49m");
        assert_eq!(theme.get_fg_ansi("accent"), "\u{1b}[38;2;1;2;3m");
        assert_eq!(theme.get_bg_ansi("selectedBg"), "\u{1b}[48;5;4m");
        assert_eq!(theme.bold("x"), "\u{1b}[1mx\u{1b}[22m");
        assert_eq!(theme.italic("x"), "\u{1b}[3mx\u{1b}[23m");
        assert_eq!(theme.underline("x"), "\u{1b}[4mx\u{1b}[24m");
        assert_eq!(theme.inverse("x"), "\u{1b}[7mx\u{1b}[27m");
        assert_eq!(theme.strikethrough("x"), "\u{1b}[9mx\u{1b}[29m");
    }

    #[test]
    fn thinking_border_colors_fall_back_to_off_and_reuse_xhigh_for_max() {
        let mut fg = HashMap::new();
        fg.insert("thinkingOff".to_string(), ColorValue::Text("#000000".to_string()));
        fg.insert("thinkingHigh".to_string(), ColorValue::Text(String::new()));
        fg.insert("thinkingXhigh".to_string(), ColorValue::Text("#ffffff".to_string()));
        let theme = Theme::new(&fg, &HashMap::new(), TerminalColorMode::Truecolor, None, None, None).expect("theme");
        assert_eq!(theme.get_thinking_border_color("max")("x"), "\u{1b}[38;2;255;255;255mx\u{1b}[39m");
        assert_eq!(theme.get_thinking_border_color("nonsense")("x"), "\u{1b}[38;2;0;0;0mx\u{1b}[39m");
        assert_eq!(theme.get_thinking_border_color("high")("x"), "\u{1b}[39mx\u{1b}[39m");
    }

    #[test]
    fn parse_theme_json_reports_minimal_structural_error() {
        let error = parse_theme_json("custom", &serde_json::json!({ "name": "x" })).unwrap_err();
        assert_eq!(error, "Invalid theme \"custom\": expected a JSON object with a \"colors\" object");
    }

    #[test]
    fn theme_json_from_value_reads_vars_colors_and_export() {
        let json = serde_json::json!({
            "$schema": "schema.json",
            "name": "custom",
            "vars": { "primary": "#010203" },
            "colors": { "accent": "primary", "border": 8 },
            "export": { "pageBg": "primary", "cardBg": "", "infoBg": 9 }
        });
        let theme = theme_json_from_value(&json);
        assert_eq!(theme.schema.as_deref(), Some("schema.json"));
        assert_eq!(theme.name, "custom");
        assert_eq!(theme.vars.get("primary"), Some(&ColorValue::Text("#010203".to_string())));
        assert_eq!(theme.colors.get("accent"), Some(&ColorValue::Text("primary".to_string())));
        assert_eq!(theme.colors.get("border"), Some(&ColorValue::Index(8.0)));
        let export = theme.export.expect("export");
        assert_eq!(export.page_bg, Some(ColorValue::Text("primary".to_string())));
        assert_eq!(export.card_bg, Some(ColorValue::Text(String::new())));
        assert_eq!(export.info_bg, Some(ColorValue::Index(9.0)));
    }

    #[test]
    fn create_theme_splits_background_keys() {
        let theme = create_theme(&minimal_theme_json(), Some(TerminalColorMode::Truecolor), None).expect("theme");
        assert_eq!(theme.bg_color_values.len(), BG_COLOR_KEYS.len());
        assert_eq!(theme.color_mode(), TerminalColorMode::Truecolor);
    }

    #[test]
    fn ansi256_to_hex_matches_cube_and_grayscale() {
        assert_eq!(ansi256_to_hex(0.0), "#000000");
        assert_eq!(ansi256_to_hex(15.0), "#ffffff");
        assert_eq!(ansi256_to_hex(16.0), "#000000");
        assert_eq!(ansi256_to_hex(232.0), "#080808");
        assert_eq!(ansi256_to_hex(255.0), "#eeeeee");
    }

    #[test]
    fn language_lookup_covers_known_and_unknown_extensions() {
        assert_eq!(get_language_from_path("a/b/main.ts").as_deref(), Some("typescript"));
        assert_eq!(get_language_from_path("main.PY").as_deref(), Some("python"));
        assert_eq!(get_language_from_path("Makefile"), Some("makefile".to_string()));
        assert_eq!(get_language_from_path("archive.unknownext"), None);
    }

    #[test]
    fn light_theme_detection_uses_the_name() {
        assert!(is_light_theme(Some("light")));
        assert!(!is_light_theme(Some("dark")));
        assert!(!is_light_theme(None));
    }

    #[test]
    fn default_theme_is_prime_unless_the_terminal_is_light() {
        let detected = detect_terminal_background();
        let expected = if detected == "light" { "light" } else { "prime" };
        assert_eq!(get_default_theme(), expected);
    }

    #[test]
    fn registered_themes_are_replaced_wholesale() {
        let theme = Theme::new(
            &colors(&[("accent", ColorValue::Text("#010203".to_string()))]),
            &HashMap::new(),
            TerminalColorMode::Truecolor,
            Some("custom-name".to_string()),
            Some("/tmp/custom-name.json".to_string()),
            None,
        )
        .expect("theme");
        set_registered_themes(vec![theme]);
        assert!(registered_theme_names().contains(&"custom-name".to_string()));
        let info = get_available_themes_with_paths()
            .into_iter()
            .find(|info| info.name == "custom-name")
            .expect("theme info");
        assert_eq!(info.path.as_deref(), Some("/tmp/custom-name.json"));
        set_registered_themes(Vec::new());
        assert!(!registered_theme_names().contains(&"custom-name".to_string()));
    }

    /// Row 76/125 of the terminal audit: a late OSC 10/11 probe result must reach
    /// the theme. TS has ONE module-level cell (terminal-colors.ts:28) that the
    /// probe writes (packages/tui/src/terminal.ts:348) and the theme reads
    /// (theme.ts:428,502,524,946), plus a module-level subscription
    /// (theme.ts:847-863) that re-resolves the automatic theme and fires
    /// `onThemeChange`.
    ///
    /// Teeth: with the old duplicate function-local `OnceLock` in
    /// `get_default_terminal_colors` (two distinct cells for get and set), the
    /// first assertion fails with "the theme must observe the probe result",
    /// because the getter stays `None` forever.
    #[test]
    fn a_late_color_probe_reaches_the_theme_and_fires_theme_change() {
        // The probe publishes through the pi-tui store (terminal.rs:382,535).
        pi_tui::terminal_colors::clear_default_terminal_colors();
        assert_eq!(
            get_default_terminal_colors(),
            None,
            "no probe result yet: the theme sees the terminal default"
        );

        let changes = Arc::new(AtomicU64::new(0));
        let seen = changes.clone();
        ensure_default_terminal_colors_subscription();
        on_theme_change(Box::new(move || {
            seen.fetch_add(1, Ordering::SeqCst);
        }));

        // Exactly the call the OSC 10/11 probe makes on completion.
        pi_tui::terminal_colors::set_default_terminal_colors(Some(
            pi_tui::terminal_colors::DefaultTerminalColors {
                foreground: pi_tui::terminal_colors::Rgb {
                    r: 255,
                    g: 255,
                    b: 255,
                },
                background: pi_tui::terminal_colors::Rgb {
                    r: 250,
                    g: 250,
                    b: 250,
                },
            },
        ));

        let observed =
            get_default_terminal_colors().expect("the theme must observe the probe result");
        assert_eq!(
            observed.background,
            Rgb {
                r: 250.0,
                g: 250.0,
                b: 250.0
            }
        );
        assert_eq!(
            get_terminal_background_kind(),
            Some(TerminalBackgroundKind::Light),
            "a light probed background must switch the adaptive accent (theme.ts:522-528)"
        );
        assert_eq!(
            get_default_theme(),
            "light",
            "an automatic theme must re-resolve to the light preset (theme.ts:848-858)"
        );
        assert_eq!(
            changes.load(Ordering::SeqCst),
            1,
            "the subscription must fire onThemeChange once (theme.ts:860-862)"
        );

        pi_tui::terminal_colors::clear_default_terminal_colors();
        // Test hygiene: the subscription itself is permanent (theme.ts:847 runs
        // at import time), but the process-global callback must not leak into
        // the tests that run after this one.
        current_theme_state()
            .lock()
            .expect("theme state")
            .on_theme_change_callback = None;
    }

    #[test]
    fn set_theme_reports_failure_and_falls_back_to_dark() {
        let result = set_theme("definitely-not-a-theme", false);
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Theme not found: definitely-not-a-theme"));
        assert_eq!(
            current_theme_state().lock().expect("state").current_theme_name.as_deref(),
            Some("dark")
        );
    }
}
