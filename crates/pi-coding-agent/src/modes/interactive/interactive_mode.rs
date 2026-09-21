//! Port of packages/coding-agent/src/modes/interactive/interactive-mode.ts

//!
//! This is the largest module of the slice (10,352 TypeScript lines, ~396
//! methods). The port keeps every module-level free function, constant, type and
//! pure helper exactly, and implements the `InteractiveMode` class against the
//! private stand-ins in `interactive_mode_services.rs`. Methods whose whole body
//! depends on UI components that other slices still own are marked `PARTIAL:`
//! with the TypeScript method name; see evidence/status/ca-interactive-a.json.

#[path = "native_host.rs"]
pub(crate) mod native_host;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{ImageContent, Model, ServiceTier};

use crate::config::VERSION;
use crate::utils::paths::get_cwd_relative_path;

use super::agent_activity::format_token_count;
use super::auth_flows::ProviderAuthFlows;
use super::feature_hints::FeatureHintDeck;
use super::heartbeat_scope::scope_heartbeats_to_session;
use super::image_markers::{collect_marked_images, evict_images_to_budget, format_image_marker, remap_image_markers};
use super::interactive_mode_services::{
    AgentConnection, AgentConnectionHeartbeat, AgentConnectionHistoryRange, AgentConnectionHistoryWindow,
    AgentConnectionModel, AgentConnectionQueueState, AgentConnectionRlmChildAgentSnapshot,
    AgentConnectionSessionEvent, AgentConnectionSnapshot, AgentConnectionState, ContextUsage, GoalState,
    InteractiveModeLocalSessionHost, InteractiveModeUiServices, OverlayHandle, Text, Theme,
};
use super::onboarding::should_run_onboarding;
use crate::core::goals::{empty_goal_state, GoalStatus};
use super::prompt_stash_state::{ClientPromptStashStore, PromptStash, PromptStashState};
use super::queue_selection::{QueueSelection, QueueSelectionItem};
use super::resume_hint::format_resume_hint;
use super::theme::theme::{theme, ThemeColor};
use super::theme::working_icon::{set_working_pulse_frame, WORKING_ICON_INTERVAL_MS};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `HEARTBEAT_PROMPT_PREVIEW_LABEL`
pub const HEARTBEAT_PROMPT_PREVIEW_LABEL: &str = "Heartbeat";
/// `GOAL_CONTEXT_PREVIEW_LABEL`
pub const GOAL_CONTEXT_PREVIEW_LABEL: &str = "Goal";
/// `AGENT_MESSAGE_RECEIVED_PREVIEW_LABEL`
pub const AGENT_MESSAGE_RECEIVED_PREVIEW_LABEL: &str = "Message";
/// `ASYNC_BASH_COMPLETION_PREVIEW_LABEL`
pub const ASYNC_BASH_COMPLETION_PREVIEW_LABEL: &str = "Bash";

/// `APP_NAME` (config.ts). The TypeScript constant is `piConfigName || "pi"`.
pub fn app_name() -> String {
    crate::utils::tools_manager::app_name()
}

const HEARTBEAT_LEGACY_PROMPT_MIN_TOLERANCE_MS: f64 = 15_000.0;
const HEARTBEAT_LEGACY_PROMPT_MAX_TOLERANCE_MS: f64 = 120_000.0;
const MODEL_CATALOG_REFRESH_TTL_MS: f64 = 60_000.0;
const FEATURE_HINT_DELAY_MS: f64 = 5_000.0;

/// `START_HINTS`
pub const START_HINTS: [&str; 5] = [
    "Try \"refactor @<filepath>\"",
    "Try \"fix bugs in @<filepath>\"",
    "Try \"add tests for @<filepath>\"",
    "Try \"explain how @<filepath> works\"",
    "Try \"improve performance in @<filepath>\"",
];

/// Port of `getRandomStartHint`.
pub fn get_random_start_hint(random: &dyn Fn() -> f64) -> &'static str {
    let index = (random() * START_HINTS.len() as f64).floor() as i64;
    START_HINTS.get(index as usize).copied().unwrap_or(START_HINTS[0])
}

/// Port of `isLabeledQueuedPreview`.
fn is_labeled_queued_preview(message: &str) -> bool {
    message.starts_with(&format!("{HEARTBEAT_PROMPT_PREVIEW_LABEL}: "))
        || message.starts_with(&format!("{GOAL_CONTEXT_PREVIEW_LABEL}: "))
        || message.starts_with(&format!("{AGENT_MESSAGE_RECEIVED_PREVIEW_LABEL}: "))
        || message.starts_with(&format!("{ASYNC_BASH_COMPLETION_PREVIEW_LABEL}: "))
}

/// `"Steering" | "Follow-up"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueLabel {
    Steering,
    FollowUp,
}

impl QueueLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueLabel::Steering => "Steering",
            QueueLabel::FollowUp => "Follow-up",
        }
    }
}

/// Port of `formatQueuedMessagePreview`.
pub fn format_queued_message_preview(message: &str, label: QueueLabel) -> String {
    if is_labeled_queued_preview(message) {
        message.to_string()
    } else {
        format!("{}: {message}", label.as_str())
    }
}

/// Port of `styleQueuedMessagePreview`.
///
/// `isRecognizedSlashCommand`, `isLeadingSlashCommand`, `styleArgumentTokens`
/// and `styleSlashCommandText` live in other slices; the callback shape is kept
/// so the styling split stays identical.
pub fn style_queued_message_preview(
    message: &str,
    label: QueueLabel,
    is_recognized_slash_command: &dyn Fn(&str) -> bool,
    is_leading_slash_command: &dyn Fn(&str, &dyn Fn(&str) -> bool) -> bool,
    style_argument_tokens: &dyn Fn(&str, &dyn Fn(&str) -> String, bool) -> String,
    style_slash_command_text: &dyn Fn(&str, &dyn Fn(&str, bool) -> String) -> String,
) -> String {
    let preview = format_queued_message_preview(message, label);
    let style_dim = |segment: &str| theme().fg("dim", segment);
    if !is_leading_slash_command(message, is_recognized_slash_command) {
        return style_argument_tokens(&preview, &style_dim, false);
    }
    let prefix = &preview[..preview.len() - message.len()];
    let styled = style_slash_command_text(message, &|rest, include_bare_separator| {
        style_argument_tokens(rest, &style_dim, include_bare_separator)
    });
    format!("{}{styled}", theme().fg("dim", prefix))
}

/// Port of `isExpandable`.
pub trait Expandable {
    fn set_expanded(&mut self, expanded: bool);
}

/// Port of `ExpandableText`.
pub struct ExpandableText {
    text: Text,
    get_collapsed_text: Box<dyn Fn() -> String + Send + Sync>,
    get_expanded_text: Box<dyn Fn() -> String + Send + Sync>,
}

impl ExpandableText {
    pub fn new(
        get_collapsed_text: Box<dyn Fn() -> String + Send + Sync>,
        get_expanded_text: Box<dyn Fn() -> String + Send + Sync>,
        expanded: bool,
        padding_x: usize,
        padding_y: usize,
    ) -> Self {
        let initial = if expanded { get_expanded_text() } else { get_collapsed_text() };
        Self {
            text: Text::new(initial, padding_x, padding_y),
            get_collapsed_text,
            get_expanded_text,
        }
    }
}

impl Expandable for ExpandableText {
    fn set_expanded(&mut self, expanded: bool) {
        let value = if expanded { (self.get_expanded_text)() } else { (self.get_collapsed_text)() };
        self.text.set_text(value);
    }
}

/// Port of `formatSplashCwd`.
pub fn format_splash_cwd(cwd: &str) -> String {
    let normalized = cwd.replace('\\', "/");
    let home = home_dir().replace('\\', "/");
    if !home.is_empty() && normalized == home {
        return "~".to_string();
    }
    if !home.is_empty() && normalized.starts_with(&format!("{home}/")) {
        return format!("~{}", &normalized[home.len()..]);
    }
    normalized
}

fn home_dir() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default()
}

/// Port of `mergeSubagentSnapshot`.
pub fn merge_subagent_snapshot(
    previous: &AgentConnectionRlmChildAgentSnapshot,
    incoming: &AgentConnectionRlmChildAgentSnapshot,
) -> AgentConnectionRlmChildAgentSnapshot {
    let active = incoming.status == "running" || incoming.status == "queued";
    AgentConnectionRlmChildAgentSnapshot {
        parent_id: incoming.parent_id.clone().or_else(|| previous.parent_id.clone()),
        // Active updates may omit a previously known daemon session id, but a
        // terminal update without one means the child is no longer resident.
        active_session_id: if active {
            incoming.active_session_id.clone().or_else(|| previous.active_session_id.clone())
        } else {
            incoming.active_session_id.clone()
        },
        // A completed retained child can become active again when it receives a
        // follow-up. Its RLM run status stays terminal, so activity must remain an
        // independent projection of the live session state.
        activity: if active { incoming.activity.clone().or_else(|| previous.activity.clone()) } else { incoming.activity.clone() },
        ..incoming.clone()
    }
}

/// Port of `truncatePathMiddle`.
pub fn truncate_path_middle(value: &str, width: f64) -> String {
    if visible_width(value) <= width {
        return value.to_string();
    }
    if width <= 1.0 {
        return truncate_to_width(value, width, "", false);
    }

    let ellipsis = "\u{2026}";
    let normalized = value.replace('\\', "/");
    let prefix = if normalized.starts_with("~/") {
        "~/"
    } else if normalized.starts_with('/') {
        "/"
    } else {
        ""
    };
    let body = if prefix.is_empty() { normalized.as_str() } else { &normalized[prefix.len()..] };
    let mut parts: Vec<&str> = body.split('/').filter(|part| !part.is_empty()).collect();
    let last = parts.pop().unwrap_or("");
    let previous = parts.pop();
    let suffix = match previous {
        Some(previous) => format!("{previous}/{last}"),
        None => last.to_string(),
    };
    let candidate = format!("{prefix}{ellipsis}/{suffix}");
    if visible_width(&candidate) <= width {
        return candidate;
    }

    truncate_to_width(&candidate, width, "…", false)
}

/// `BrandSplashMetadataLine`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrandSplashMetadataLine {
    pub label: String,
    pub value: String,
}

/// `BrandSplashHeaderOptions`
pub struct BrandSplashHeaderOptions {
    pub logo: Option<String>,
    pub top_padding: bool,
    pub get_rows: Option<Box<dyn Fn() -> f64 + Send + Sync>>,
    pub get_extra_metadata: Option<Box<dyn Fn() -> Vec<BrandSplashMetadataLine> + Send + Sync>>,
    pub get_hide_start_hint: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    pub get_start_hint: Option<Box<dyn Fn() -> String + Send + Sync>>,
}

impl Default for BrandSplashHeaderOptions {
    fn default() -> Self {
        Self {
            logo: None,
            top_padding: false,
            get_rows: None,
            get_extra_metadata: None,
            get_hide_start_hint: None,
            get_start_hint: None,
        }
    }
}

/// Port of `BrandSplashHeader`.
pub struct BrandSplashHeader {
    version: String,
    get_model_id: Box<dyn Fn() -> Option<String> + Send + Sync>,
    get_cwd: Box<dyn Fn() -> String + Send + Sync>,
    verbose_instructions: Option<String>,
    options: BrandSplashHeaderOptions,
}

impl BrandSplashHeader {
    const GUTTER: usize = 4;
    const LABEL_WIDTH: usize = 9;

    pub fn new(
        version: String,
        get_model_id: Box<dyn Fn() -> Option<String> + Send + Sync>,
        get_cwd: Box<dyn Fn() -> String + Send + Sync>,
        verbose_instructions: Option<String>,
        options: BrandSplashHeaderOptions,
    ) -> Self {
        Self { version, get_model_id, get_cwd, verbose_instructions, options }
    }

    pub fn invalidate(&self) {
        // Render output is derived from current theme/session state.
    }

    /// Port of `render`.
    pub fn render(&self, width: f64, max_rows: Option<f64>) -> Vec<String> {
        let max_rows = max_rows.unwrap_or(f64::INFINITY);
        let safe_width = width.max(1.0).floor();
        let padding_x = if safe_width >= 3.0 { 1usize } else { 0 };
        let content_width = (safe_width - (padding_x * 2) as f64).max(1.0).floor();
        let terminal_rows = self
            .options
            .get_rows
            .as_ref()
            .map(|get_rows| get_rows())
            .unwrap_or_else(|| process_stdout_rows().unwrap_or(40.0));
        let available_rows = 0.0f64.max(max_rows.min(terminal_rows - 8.0)).floor();
        if available_rows == 0.0 {
            return Vec::new();
        }
        let top_padding = self.options.top_padding && terminal_rows >= 24.0 && available_rows > 1.0;
        let body_rows = available_rows - if top_padding { 1.0 } else { 0.0 };
        let logo_max_rows = 12.0f64.min((terminal_rows - 16.0).max(0.0)).min(body_rows);
        let logo_max_width =
            26.0f64.min(content_width - (Self::GUTTER as f64) - (Self::LABEL_WIDTH as f64) - 8.0);
        // Decorative art yields before metadata or the caller's search/session rows.
        let logo_raw: Vec<String> = match &self.options.logo {
            Some(logo) => logo.split('\n').map(|line| line.to_string()).collect(),
            None => {
                if logo_max_rows >= 8.0 && logo_max_width >= 16.0 {
                    crate::themes::optimus_logo::get_optimus_logo(logo_max_width, logo_max_rows)
                } else {
                    Vec::new()
                }
            }
        };
        let logo_canvas_width = logo_raw.iter().map(|line| visible_width(line)).fold(0.0f64, f64::max);
        let logo_gutter = if logo_raw.is_empty() { 0.0 } else { Self::GUTTER as f64 };
        let meta_width = content_width - logo_canvas_width - logo_gutter;
        let show_meta = meta_width >= (Self::LABEL_WIDTH as f64) + 8.0;
        let value_width = (meta_width - Self::LABEL_WIDTH as f64).max(1.0).floor();
        let labelled = |label: &str, value: &str| -> String {
            let display_value = if label == "cwd" {
                truncate_path_middle(value, value_width)
            } else {
                truncate_to_width(value, value_width, "", true)
            };
            let padded = format!("{label:<width$}", width = Self::LABEL_WIDTH);
            format!("{}{}", theme().fg("dim", &padded), theme().fg("muted", &display_value))
        };
        let extra_metadata: Vec<BrandSplashMetadataLine> = self
            .options
            .get_extra_metadata
            .as_ref()
            .map(|get_extra_metadata| get_extra_metadata())
            .unwrap_or_default();
        let hide_start_hint = self
            .options
            .get_hide_start_hint
            .as_ref()
            .map(|get_hide_start_hint| get_hide_start_hint())
            .unwrap_or(false);
        let start_hint = self
            .options
            .get_start_hint
            .as_ref()
            .map(|get_start_hint| get_start_hint())
            .unwrap_or_else(|| "type to search sessions".to_string());
        let mut meta_lines: Vec<String> = if show_meta {
            let mut lines: Vec<String> = Vec::new();
            if self.options.logo.is_none() {
                lines.push(theme().bold(&colorize_optimus_logo("OPTIMUS")));
                lines.push(String::new());
            }
            lines.push(labelled("version", &format!("v{}", self.version)));
            lines.push(labelled(
                "model",
                &(self.get_model_id)().unwrap_or_else(|| "\u{2014}".to_string()),
            ));
            lines.push(labelled("cwd", &format_splash_cwd(&(self.get_cwd)())));
            lines.extend(extra_metadata.iter().map(|line| labelled(&line.label, &line.value)));
            if !hide_start_hint {
                lines.push(String::new());
                lines.push(theme().fg("dim", &start_hint));
            }
            lines
        } else {
            Vec::new()
        };
        if meta_lines.len() as f64 > body_rows {
            let mut index = meta_lines.len() as i64 - 1;
            while index >= 0 {
                if meta_lines[index as usize].is_empty() {
                    meta_lines.remove(index as usize);
                }
                index -= 1;
            }
            if meta_lines.len() as f64 > body_rows && !hide_start_hint {
                meta_lines.pop();
            }
            meta_lines.truncate(body_rows.max(0.0) as usize);
        }
        let row_count = logo_raw.len().max(meta_lines.len());
        let meta_start = (row_count.saturating_sub(meta_lines.len())) / 2;
        let mut lines: Vec<String> = if top_padding { vec![String::new()] } else { Vec::new() };
        for index in 0..row_count {
            let line = logo_raw.get(index).cloned().unwrap_or_default();
            let colored = if self.options.logo.is_none() {
                colorize_optimus_logo(&line)
            } else {
                theme().fg("text", &line)
            };
            let meta = if index >= meta_start && index < meta_start + meta_lines.len() {
                meta_lines[index - meta_start].clone()
            } else {
                String::new()
            };
            let padding = if show_meta {
                " ".repeat(
                    (logo_canvas_width - visible_width(&line) + logo_gutter).max(0.0) as usize,
                )
            } else {
                String::new()
            };
            let content = truncate_to_width(&format!("{colored}{padding}{meta}"), content_width, "", false);
            lines.push(format!(
                "{}{content}{}",
                " ".repeat(padding_x),
                " ".repeat((safe_width - padding_x as f64 - visible_width(&content)).max(0.0) as usize)
            ));
        }
        if !show_meta && self.options.logo.is_none() {
            lines.push(format!(
                "{}{}",
                " ".repeat(padding_x),
                truncate_to_width(&theme().bold(&colorize_optimus_logo("OPTIMUS")), content_width, "", true)
            ));
        }

        if let Some(verbose_instructions) = &self.verbose_instructions {
            lines.push(" ".repeat(safe_width as usize));
            for instruction in verbose_instructions.split('\n') {
                let content = truncate_to_width(instruction, content_width, "", true);
                lines.push(format!(
                    "{}{content}{}",
                    " ".repeat(padding_x),
                    " ".repeat((safe_width - padding_x as f64 - visible_width(&content)).max(0.0) as usize)
                ));
            }
        }

        lines
    }
}

/// Stand-in for `process.stdout.rows`.
fn process_stdout_rows() -> Option<f64> {
    std::env::var("LINES").ok().and_then(|value| value.parse::<f64>().ok())
}

/// Stand-in for `colorizeOptimusLogo` (themes/optimus-logo.ts, other slice).
fn colorize_optimus_logo(line: &str) -> String {
    theme().fg("accent", line)
}

/// `theme.ts`'s `MarkdownTheme` holds `Arc` closures with `Send + Sync`; the
/// pi-tui renderer holds `Rc` closures. Private port of `toTuiMarkdownTheme`.
fn to_tui_markdown_theme(
    theme_source: super::theme::theme::MarkdownTheme,
) -> pi_tui::components::markdown::MarkdownTheme {
    fn rc(value: Arc<dyn Fn(&str) -> String + Send + Sync>) -> std::rc::Rc<dyn Fn(&str) -> String> {
        std::rc::Rc::new(move |text: &str| value(text))
    }

    pi_tui::components::markdown::MarkdownTheme {
        heading: rc(theme_source.heading),
        link: rc(theme_source.link),
        link_url: rc(theme_source.link_url),
        code: rc(theme_source.code),
        code_block: rc(theme_source.code_block),
        code_block_border: rc(theme_source.code_block_border),
        quote: rc(theme_source.quote),
        quote_border: rc(theme_source.quote_border),
        hr: rc(theme_source.hr),
        list_bullet: rc(theme_source.list_bullet),
        bold: rc(theme_source.bold),
        italic: rc(theme_source.italic),
        strikethrough: rc(theme_source.strikethrough),
        underline: rc(theme_source.underline),
        highlight_code: Some(std::rc::Rc::new(move |code: &str, language: Option<&str>| {
            (theme_source.highlight_code)(code, language)
        })),
        code_block_indent: theme_source.code_block_indent,
        math: Some(rc(theme_source.math)),
        math_block: Some(rc(theme_source.math_block)),
    }
}

/// Adapter that lets a pi-tui `Markdown` live in an `InteractiveMode` container.
///
/// The TypeScript adds `new Markdown(...)` straight to `chatContainer` /
/// `shortcutGuideContainer`. The port's containers hold the slice's local
/// `Component` trait, which is `Send`, while the pi-tui `MarkdownTheme` holds
/// `Rc` closures. The block therefore keeps the owned `Send + Sync` theme and
/// builds the renderer per render call.
struct MarkdownBlock {
    text: String,
    padding_x: usize,
    padding_y: usize,
    markdown_theme: super::theme::theme::MarkdownTheme,
}

impl MarkdownBlock {
    fn new(
        text: String,
        padding_x: usize,
        padding_y: usize,
        markdown_theme: super::theme::theme::MarkdownTheme,
    ) -> Self {
        Self { text, padding_x, padding_y, markdown_theme }
    }
}

impl super::interactive_mode_services::Component for MarkdownBlock {
    fn render(&self, width: usize) -> Vec<String> {
        let mut markdown = pi_tui::components::markdown::Markdown::new(
            self.text.clone(),
            self.padding_x,
            self.padding_y,
            to_tui_markdown_theme(self.markdown_theme.clone()),
            None,
            pi_tui::components::markdown::MarkdownOptions::default(),
        );
        pi_tui::tui::Component::render(&mut markdown, width as f64)
    }

    fn invalidate(&mut self) {}

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `StartupPromptBarrierOutcome`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupPromptBarrierOutcome {
    Admitted,
    Retained,
    LifecycleCancelled,
}

impl StartupPromptBarrierOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            StartupPromptBarrierOutcome::Admitted => "admitted",
            StartupPromptBarrierOutcome::Retained => "retained",
            StartupPromptBarrierOutcome::LifecycleCancelled => "lifecycle-cancelled",
        }
    }
}

/// `GoalAnnouncementSnapshot`
#[derive(Debug, Clone, PartialEq)]
pub struct GoalAnnouncementSnapshot {
    pub goal_id: Option<String>,
    /// `GoalState["status"]` - the TypeScript snapshot copies the union verbatim,
    /// so the port keeps the typed enum rather than a string.
    pub status: GoalStatus,
    pub objective: Option<String>,
    pub last_reason: Option<String>,
    pub last_error: Option<String>,
}

impl Default for GoalAnnouncementSnapshot {
    /// A snapshot is only ever built from a live goal (`goalAnnouncementSnapshot`), so
    /// the default mirrors `emptyGoalState()`: idle with no details.
    fn default() -> Self {
        Self {
            goal_id: None,
            status: GoalStatus::Idle,
            objective: None,
            last_reason: None,
            last_error: None,
        }
    }
}

/// `ModelFallbackWarningAction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFallbackWarningAction {
    Show,
    Suppress,
}

/// `OnboardingSplashHandle`
pub struct OnboardingSplashHandle {
    pub disposed: bool,
}

/// `THINKING_LEVEL_DESCRIPTIONS`
pub fn thinking_level_descriptions() -> HashMap<ThinkingLevel, &'static str> {
    HashMap::from([
        (ThinkingLevel::Off, "No reasoning"),
        (ThinkingLevel::Minimal, "Very brief reasoning (~1k tokens)"),
        (ThinkingLevel::Low, "Light reasoning (~2k tokens)"),
        (ThinkingLevel::Medium, "Moderate reasoning (~8k tokens)"),
        (ThinkingLevel::High, "Deep reasoning (~16k tokens)"),
        (ThinkingLevel::Xhigh, "Very deep reasoning (~32k tokens)"),
        (ThinkingLevel::Max, "Maximum reasoning"),
    ])
}

/// `HEARTBEAT_ARGUMENT_COMPLETIONS`
pub fn heartbeat_argument_completions() -> Vec<(String, String)> {
    vec![
        ("start <schedule> [prompt]".to_string(), "Create or update a heartbeat".to_string()),
        ("stop".to_string(), "Stop and remove the heartbeat".to_string()),
        ("status".to_string(), "Show heartbeat status".to_string()),
        ("help".to_string(), "Show heartbeat help".to_string()),
    ]
}

/// `DEAD_TERMINAL_ERROR_CODES`
pub fn dead_terminal_error_codes() -> HashSet<&'static str> {
    HashSet::from(["EIO", "EPIPE", "ENOTCONN"])
}

/// `MAX_PASTED_IMAGE_BYTES`
pub const MAX_PASTED_IMAGE_BYTES: f64 = 64.0 * 1024.0 * 1024.0;
/// `INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT`
pub const INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT: usize = 400;

/// Port of `initialRenderMessages`.
pub fn initial_render_messages(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    if messages.len() <= INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT {
        return messages;
    }
    let mut tool_call_messages: HashMap<String, (usize, AgentMessage)> = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        let AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) = message else {
            continue;
        };
        for content in &assistant.content {
            if let pi_ai::types::ContentBlock::ToolCall(tool_call) = content {
                tool_call_messages.insert(tool_call.id.clone(), (index, message.clone()));
            }
        }
    }

    let initial_start_index = messages.len() - INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT;
    for start_index in initial_start_index..messages.len() {
        let visible_messages = &messages[start_index..];
        let mut visible_tool_call_ids: HashSet<String> = HashSet::new();
        for message in visible_messages {
            let AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) = message else {
                continue;
            };
            for content in &assistant.content {
                if let pi_ai::types::ContentBlock::ToolCall(tool_call) = content {
                    visible_tool_call_ids.insert(tool_call.id.clone());
                }
            }
        }

        let mut required_tool_call_ids_by_message: HashMap<usize, (AgentMessage, HashSet<String>)> = HashMap::new();
        for message in visible_messages {
            let AgentMessage::Message(pi_ai::types::Message::ToolResult(tool_result)) = message else {
                continue;
            };
            if visible_tool_call_ids.contains(&tool_result.tool_call_id) {
                continue;
            }
            let Some((tool_call_index, tool_call_message)) = tool_call_messages.get(&tool_result.tool_call_id) else {
                continue;
            };
            if *tool_call_index >= start_index {
                continue;
            }
            let entry = required_tool_call_ids_by_message
                .entry(*tool_call_index)
                .or_insert_with(|| (tool_call_message.clone(), HashSet::new()));
            entry.1.insert(tool_result.tool_call_id.clone());
        }

        if visible_messages.len() + required_tool_call_ids_by_message.len() > INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT
        {
            continue;
        }

        let mut required_entries: Vec<(usize, AgentMessage, HashSet<String>)> = required_tool_call_ids_by_message
            .into_iter()
            .map(|(index, (message, tool_call_ids))| (index, message, tool_call_ids))
            .collect();
        required_entries.sort_by_key(|(index, _, _)| *index);
        let required_tool_call_messages: Vec<AgentMessage> = required_entries
            .into_iter()
            .map(|(_, message, tool_call_ids)| {
                let AgentMessage::Message(pi_ai::types::Message::Assistant(mut assistant)) = message else {
                    return message;
                };
                assistant.content.retain(|content| match content {
                    pi_ai::types::ContentBlock::ToolCall(tool_call) => tool_call_ids.contains(&tool_call.id),
                    _ => true,
                });
                AgentMessage::Message(pi_ai::types::Message::Assistant(assistant))
            })
            .collect();
        let mut combined = required_tool_call_messages;
        combined.extend(visible_messages.iter().cloned());
        return omit_orphan_tool_results(combined);
    }

    Vec::new()
}

/// Port of `omitOrphanToolResults`.
pub fn omit_orphan_tool_results(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut rendered_tool_call_ids: HashSet<String> = HashSet::new();
    let mut renderable_messages: Vec<AgentMessage> = Vec::new();
    for message in messages {
        match &message {
            AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => {
                for content in &assistant.content {
                    if let pi_ai::types::ContentBlock::ToolCall(tool_call) = content {
                        rendered_tool_call_ids.insert(tool_call.id.clone());
                    }
                }
                renderable_messages.push(message);
            }
            AgentMessage::Message(pi_ai::types::Message::ToolResult(tool_result)) => {
                if rendered_tool_call_ids.contains(&tool_result.tool_call_id) {
                    renderable_messages.push(message);
                }
            }
            _ => renderable_messages.push(message),
        }
    }
    renderable_messages
}

/// Port of `isDeadTerminalError`.
pub fn is_dead_terminal_error(code: Option<&str>) -> bool {
    match code {
        Some(code) => dead_terminal_error_codes().contains(code),
        None => false,
    }
}

/// Port of `getPayloadString`.
pub fn get_payload_string(payload: &serde_json::Value, key: &str) -> Option<String> {
    payload.get(key).and_then(|value| value.as_str()).map(|value| value.to_string())
}

/// Port of `getPayloadNumber`.
pub fn get_payload_number(payload: &serde_json::Value, key: &str) -> Option<f64> {
    payload.get(key).and_then(|value| value.as_f64()).filter(|value| value.is_finite())
}

/// Port of `getPayloadBoolean`.
pub fn get_payload_boolean(payload: &serde_json::Value, key: &str) -> Option<bool> {
    payload.get(key).and_then(|value| value.as_bool())
}

/// Port of `getPayloadStringArray`.
pub fn get_payload_string_array(payload: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    let value = payload.get(key)?;
    if value.is_null() {
        return None;
    }
    let array = value.as_array()?;
    let mut out = Vec::new();
    for item in array {
        out.push(item.as_str()?.to_string());
    }
    Some(out)
}

/// `"info" | "warning" | "error"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyType {
    Info,
    Warning,
    Error,
}

impl NotifyType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NotifyType::Info => "info",
            NotifyType::Warning => "warning",
            NotifyType::Error => "error",
        }
    }
}

/// Port of `getPayloadNotifyType`.
pub fn get_payload_notify_type(payload: &serde_json::Value, key: &str) -> Option<NotifyType> {
    match payload.get(key).and_then(|value| value.as_str()) {
        Some("info") => Some(NotifyType::Info),
        Some("warning") => Some(NotifyType::Warning),
        Some("error") => Some(NotifyType::Error),
        _ => None,
    }
}

/// `"aboveEditor" | "belowEditor"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetPlacement {
    AboveEditor,
    BelowEditor,
}

impl WidgetPlacement {
    pub fn as_str(&self) -> &'static str {
        match self {
            WidgetPlacement::AboveEditor => "aboveEditor",
            WidgetPlacement::BelowEditor => "belowEditor",
        }
    }
}

/// Port of `getPayloadWidgetPlacement`.
pub fn get_payload_widget_placement(payload: &serde_json::Value, key: &str) -> Option<WidgetPlacement> {
    match payload.get(key).and_then(|value| value.as_str()) {
        Some("aboveEditor") => Some(WidgetPlacement::AboveEditor),
        Some("belowEditor") => Some(WidgetPlacement::BelowEditor),
        _ => None,
    }
}

/// `LoaderIndicatorOptions`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoaderIndicatorOptions {
    pub frames: Option<Vec<String>>,
    pub interval_ms: Option<f64>,
}

/// Port of `getPayloadWorkingIndicatorOptions`.
pub fn get_payload_working_indicator_options(
    payload: &serde_json::Value,
    key: &str,
) -> Option<LoaderIndicatorOptions> {
    let value = payload.get(key)?;
    if !value.is_object() {
        return None;
    }
    let frames = get_payload_string_array(value, "frames");
    let interval_ms = get_payload_number(value, "intervalMs");
    Some(LoaderIndicatorOptions { frames, interval_ms })
}

/// Port of `updateArgsIncludeSelf`.
pub fn update_args_include_self(args: &[String]) -> bool {
    let mut self_flag = false;
    let mut extensions_only_flag = false;
    let mut positional: Option<String> = None;
    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg == "--self" {
            self_flag = true;
        } else if arg == "--extensions" {
            extensions_only_flag = true;
        } else if arg == "--extension" {
            extensions_only_flag = true;
            index += 1;
        } else if arg == "--daemon-socket" {
            index += 1;
        } else if !arg.is_empty() && !arg.starts_with('-') && positional.is_none() {
            positional = Some(arg.to_string());
        }
        index += 1;
    }
    if self_flag {
        return true;
    }
    if extensions_only_flag {
        return false;
    }
    let Some(positional) = positional else {
        return true;
    };
    let normalized = positional.to_lowercase();
    normalized == "self" || normalized == "pi" || normalized == app_name().to_lowercase()
}

/// Port of `argsIncludeSessionSelection`.
fn args_include_session_selection(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--resume" || arg == "-r" || arg == "--continue" || arg == "-c" || arg == "--fork"
    })
}

/// Port of `buildUpdateRelaunchArgs`.
pub fn build_update_relaunch_args(args: &[String], session_file: Option<&str>) -> Vec<String> {
    let mut relaunch_args = args.to_vec();
    if let Some(session_file) = session_file {
        if !args_include_session_selection(&relaunch_args) {
            relaunch_args.push("--resume".to_string());
            relaunch_args.push(session_file.to_string());
        }
    }
    relaunch_args
}

/// Port of `execveFailureThrows`.
pub fn execve_failure_throws(node_version: &str) -> bool {
    // Before Node 26.1, a failed execve syscall aborts the process instead of throwing for the fallback below.
    let Some(captures) = regex_match_node_version(node_version) else {
        return false;
    };
    let major = captures.0;
    let minor = captures.1;
    major > 26 || (major == 26 && minor >= 1)
}

/// `/^(\d+)\.(\d+)\./`
fn regex_match_node_version(node_version: &str) -> Option<(i64, i64)> {
    let mut parts = node_version.split('.');
    let major = parts.next()?.parse::<i64>().ok()?;
    let minor = parts.next()?.parse::<i64>().ok()?;
    // The TypeScript pattern requires the trailing dot.
    if !node_version.contains('.') || node_version.matches('.').count() < 2 {
        return None;
    }
    Some((major, minor))
}

/// `CliSubprocessLaunchSpec`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CliSubprocessLaunchSpec {
    pub command: String,
    pub args: Vec<String>,
}

/// `UpdateRelaunchExecOptions`
pub struct UpdateRelaunchExecOptions {
    pub platform: String,
    pub node_version: String,
    pub cwd: String,
    pub previous_cwd: String,
    pub environment: HashMap<String, Option<String>>,
    pub chdir: Box<dyn FnMut(&str)>,
    pub execve: Option<Box<dyn FnMut(&str, &[String], &HashMap<String, String>)>>,
}

/// Port of `tryExecUpdateRelaunch`.
pub fn try_exec_update_relaunch(launch: &CliSubprocessLaunchSpec, options: &mut UpdateRelaunchExecOptions) -> bool {
    // Process replacement preserves the shell job and foreground terminal without retaining the old TUI.
    if options.execve.is_none()
        || options.platform == "win32"
        || options.platform == "os400"
        || !execve_failure_throws(&options.node_version)
    {
        return false;
    }
    let environment: HashMap<String, String> = options
        .environment
        .iter()
        .filter_map(|(key, value)| value.as_ref().map(|value| (key.clone(), value.clone())))
        .collect();
    (options.chdir)(&options.cwd);
    let mut argv: Vec<String> = vec![launch.command.clone()];
    argv.extend(launch.args.iter().cloned());
    let execve = options.execve.as_mut().expect("execve checked above");
    execve(&launch.command, &argv, &environment);
    true
}

/// Port of `buildUpdateChildArgs`.
pub fn build_update_child_args(args: &[String], daemon_socket_path: &str) -> Vec<String> {
    if args.iter().any(|arg| arg == "--daemon-socket") {
        args.to_vec()
    } else {
        let mut out = args.to_vec();
        out.push("--daemon-socket".to_string());
        out.push(daemon_socket_path.to_string());
        out
    }
}

/// Port of `resolveInteractiveUpdateDaemonSocketPath`.
pub fn resolve_interactive_update_daemon_socket_path(args: &[String], active_daemon_socket_path: &str) -> String {
    match args.iter().position(|arg| arg == "--daemon-socket") {
        Some(index) => args.get(index + 1).cloned().unwrap_or_else(|| active_daemon_socket_path.to_string()),
        None => active_daemon_socket_path.to_string(),
    }
}

/// `InteractiveInitialPrompt`
#[derive(Debug, Clone, Default)]
pub struct InteractiveInitialPrompt {
    pub text: String,
    pub images: Option<Vec<ImageContent>>,
}

/// `InteractiveModeOptions`
pub struct InteractiveModeOptions {
    /// Providers that were migrated to auth.json (shows warning)
    pub migrated_providers: Option<Vec<String>>,
    /// Warning message if session model couldn't be restored
    pub model_fallback_message: Option<String>,
    /// One-off warning shown on startup.
    pub startup_notice: Option<String>,
    /// Initial message to send on startup (can include @file content)
    pub initial_message: Option<String>,
    /// Images to attach to the initial message
    pub initial_images: Option<Vec<ImageContent>>,
    /// Additional text-only messages to send after the initial message.
    pub initial_messages: Option<Vec<String>>,
    /// Additional image-bearing prompts to send after the initial messages.
    pub initial_prompts: Option<Vec<InteractiveInitialPrompt>>,
    /// Force verbose startup (overrides quietStartup setting)
    pub verbose: bool,
    /// Agent execution boundary. InteractiveMode never talks directly to AgentSession for core execution.
    pub agent_connection: AgentConnection,
    /// Exact daemon socket to preserve across an interactive self-update restart.
    pub daemon_socket_path: Option<String>,
    /// Local-only host for in-process extension binding and callback-bearing session operations.
    pub local_session_host: Option<Arc<dyn InteractiveModeLocalSessionHost>>,
    /// Bind extension handlers in the local session host. Disabled for daemon/gateway-backed clients.
    pub bind_local_session_extensions: bool,
    /// UI-local services used for settings, auth, resources, and rendering.
    pub ui_services: Option<InteractiveModeUiServices>,
    /// Extra cleanup for externally-owned UI service hosts.
    pub on_shutdown: Option<Box<dyn FnMut() + Send + Sync>>,
    /// Allow returning from a full session to the agents view without stopping the daemon-owned agent.
    pub return_to_agents_view: bool,
    /// Enter fullscreen regardless of the persisted fullscreen preference.
    pub force_fullscreen: bool,
    /// The agents view already surfaced global startup notices.
    pub agents_view_owns_startup_notices: bool,
    /// Persisted RLM depth supplied by the daemon SessionSummary.
    pub session_depth: Option<f64>,
    /// Whether the unified daemon/catalog projection had any direct children.
    pub session_has_children: bool,
    /// Client-owned stash store shared across chat views in this TUI process.
    /// The TypeScript store is a mutable reference shared by several chat views;
    /// the port keeps one `Arc<Mutex<..>>` so each view can borrow it mutably.
    pub prompt_stash_store: Option<Arc<std::sync::Mutex<ClientPromptStashStore>>>,
    /// Initial stable session id used to scope prompt stash state.
    pub prompt_stash_session_id: Option<String>,
}

/// `InteractiveModeRunResult`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractiveModeRunResultType {
    AgentsView,
    ScopedAgentsView,
}

impl InteractiveModeRunResultType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InteractiveModeRunResultType::AgentsView => "agents_view",
            InteractiveModeRunResultType::ScopedAgentsView => "scoped_agents_view",
        }
    }
}

/// `Pick<AgentConnectionState, "activeSessionId" | "sessionFile" | "sessionId" | "sessionName" | "cwd">`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InteractiveModeRunResultSource {
    pub active_session_id: Option<String>,
    pub session_file: Option<String>,
    pub session_id: String,
    pub session_name: Option<String>,
    pub cwd: String,
}

/// `InteractiveModeRunResult`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractiveModeRunResult {
    pub type_: InteractiveModeRunResultType,
    pub source: InteractiveModeRunResultSource,
}

/// Port of `formatAgentDepthLabel`.
pub fn format_agent_depth_label(depth: Option<f64>, has_children: bool) -> Option<String> {
    match depth {
        None => None,
        Some(depth) if depth == 0.0 && !has_children => None,
        Some(depth) => Some(format!("depth {}", js_number_to_string(depth))),
    }
}

fn js_number_to_string(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// `LoadedAgentConnectionHistory`
#[derive(Debug, Clone, Default)]
pub struct LoadedAgentConnectionHistory {
    pub window: AgentConnectionHistoryWindow,
    pub messages: Vec<AgentMessage>,
}

/// Port of `mergeOlderAgentConnectionHistory`.
pub fn merge_older_agent_connection_history(
    current: &LoadedAgentConnectionHistory,
    range: &AgentConnectionHistoryRange,
) -> Result<LoadedAgentConnectionHistory, String> {
    let range_window = &range.window;
    if current.window.version != 1.0
        || current.window.order != "chronological"
        || current.window.representation.is_empty()
        || current.window.entry_ids.len() != current.messages.len()
        || current.window.start_index < 0.0
        || current.window.start_index + current.messages.len() as f64 != current.window.total_message_count
        || current.window.has_older != (current.window.start_index > 0.0)
        || range_window.version != 1.0
        || range_window.order != "chronological"
        || range_window.generation != current.window.generation
        || range_window.representation != current.window.representation
        || range_window.tip_entry_id != current.window.tip_entry_id
        || range_window.total_message_count != current.window.total_message_count
        || range_window.start_index < 0.0
        || range_window.start_index + range.messages.len() as f64 != current.window.start_index
        || range_window.entry_ids.len() != range.messages.len()
        || range_window.has_older != (range_window.start_index > 0.0)
    {
        return Err("Older history range does not continue the pinned snapshot".to_string());
    }
    let existing_ids: HashSet<&String> = current.window.entry_ids.iter().collect();
    let range_ids: HashSet<&String> = range_window.entry_ids.iter().collect();
    if existing_ids.len() != current.window.entry_ids.len()
        || range_ids.len() != range_window.entry_ids.len()
        || range_window.entry_ids.iter().any(|entry_id| existing_ids.contains(entry_id))
    {
        return Err("Older history range overlaps already loaded messages".to_string());
    }
    let mut entry_ids = range_window.entry_ids.clone();
    entry_ids.extend(current.window.entry_ids.iter().cloned());
    let mut messages = range.messages.clone();
    messages.extend(current.messages.iter().cloned());
    Ok(LoadedAgentConnectionHistory {
        window: AgentConnectionHistoryWindow {
            entry_ids,
            ..range_window.clone()
        },
        messages,
    })
}


// ---------------------------------------------------------------------------
// InteractiveMode
// ---------------------------------------------------------------------------

/// `SessionSummary` (modes/daemon/daemon-session-list.ts, other slice).
pub use crate::modes::daemon::daemon_session_list::SessionSummary;
/// `SessionSummary` in the roster projection.
pub use crate::modes::daemon::agent_roster::RosterSessionSummary;

/// `SubagentSummaryCounts`
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubagentSummaryCounts {
    pub total: usize,
    pub running: usize,
    pub idle: usize,
    pub inactive: usize,
}

/// Stand-in for `SubagentSummaryLine` (components/subagent-summary-line.ts).
pub struct SubagentSummaryLine {
    pub get_location_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
    pub get_context_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
    pub get_override_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
    pub focused: bool,
    pub counts: SubagentSummaryCounts,
    pub openable: bool,
}

impl SubagentSummaryLine {
    pub fn new(
        get_location_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
        get_context_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
        get_override_label: Box<dyn Fn() -> Option<String> + Send + Sync>,
    ) -> Self {
        Self {
            get_location_label,
            get_context_label,
            get_override_label,
            focused: false,
            counts: SubagentSummaryCounts::default(),
            openable: false,
        }
    }

    pub fn set_subagent_counts(&mut self, counts: SubagentSummaryCounts) {
        self.counts = counts;
    }

    pub fn set_openable(&mut self, openable: bool) {
        self.openable = openable;
    }

    pub fn is_selectable(&self) -> bool {
        self.counts.total > 0 && self.openable
    }
}

/// Port of `classifySubagentSnapshotStatus`.
pub fn classify_subagent_snapshot_status(
    child: &AgentConnectionRlmChildAgentSnapshot,
) -> crate::modes::daemon::agent_roster::AgentRosterStatus {
    // Activity implies a live session; the in-process connection never stamps activeSessionId.
    let resident = child.active_session_id.is_some() || child.activity.is_some();
    let busy = child.status == "running" || child.status == "queued" || child.activity.is_some();
    crate::modes::daemon::agent_roster::classify_agent_status(
        crate::modes::daemon::agent_roster::AgentStatusInput {
            resident,
            queued_child: !resident && busy,
            busy,
        },
    )
}

/// Port of `countDirectSubagentStatuses`.
pub fn count_direct_subagent_statuses(
    children: &[AgentConnectionRlmChildAgentSnapshot],
    parent_id: Option<&str>,
) -> SubagentSummaryCounts {
    let mut counts = SubagentSummaryCounts::default();
    for child in children {
        if child.parent_id.as_deref() != parent_id || child.status == "cancelled" {
            continue;
        }
        counts.total += 1;
        match classify_subagent_snapshot_status(child) {
            crate::modes::daemon::agent_roster::AgentRosterStatus::Running => counts.running += 1,
            crate::modes::daemon::agent_roster::AgentRosterStatus::Idle => counts.idle += 1,
            crate::modes::daemon::agent_roster::AgentRosterStatus::Inactive => counts.inactive += 1,
        }
    }
    counts
}

/// Projects a roster-bar row (the daemon wire `SessionSummary`) onto the
/// agents-view row so the shared direct-child linkage check can run on it: only
/// the parent-linkage fields `getParentKeys` reads are copied.
fn roster_row_for_linkage(
    row: &crate::modes::daemon::daemon_session_list::SessionSummary,
) -> crate::modes::agents_view::agents_view_state::SessionSummary {
    let mut view = crate::modes::agents_view::agents_view_state::SessionSummary::new(
        row.id.clone(),
        row.session_id.clone(),
        row.cwd.clone(),
    );
    view.parent_active_session_id = row.parent_active_session_id.clone();
    view.parent_session_id = row.parent_session_id.clone();
    view.parent_session_path = row.parent_session_path.clone();
    view
}

/// Port of `countRosterSubagentStatuses`.
pub fn count_roster_subagent_statuses(
    summaries: &[SessionSummary],
    parent: (Option<&str>, Option<&str>, Option<&str>),
) -> SubagentSummaryCounts {
    let mut counts = SubagentSummaryCounts::default();
    for child in summaries {
        if child.runtime_kind.as_deref() != Some("subagent") || child.lifecycle != "live" {
            continue;
        }
        // `isDirectAgentChild` (agents-view-state.ts) is defined over the agents-view
        // row type; the roster bar carries the daemon wire row, so project the
        // parent-linkage fields instead of duplicating the parent-key formula.
        let linkage_row = roster_row_for_linkage(child);
        if !crate::modes::agents_view::agents_view_state::is_direct_agent_child(
            &linkage_row,
            parent.0,
            parent.1,
            parent.2,
        ) {
            continue;
        }
        counts.total += 1;
        // `child.rosterStatus ?? classifySessionRosterStatus(child)` - `queuedChild`
        // keeps its TypeScript default (`false`).
        let status = child.roster_status.clone().unwrap_or_else(|| {
            crate::modes::daemon::agent_roster::classify_session_roster_status(
                &crate::modes::daemon::agent_roster::RosterSummaryView {
                    active_session_id: child.active_session_id.clone(),
                    activity: Some(child.activity.clone()),
                    is_session_active: Some(child.is_session_active),
                    ..Default::default()
                },
                false,
            )
        });
        match status {
            crate::modes::daemon::agent_roster::AgentRosterStatus::Running => counts.running += 1,
            crate::modes::daemon::agent_roster::AgentRosterStatus::Idle => counts.idle += 1,
            crate::modes::daemon::agent_roster::AgentRosterStatus::Inactive => counts.inactive += 1,
        }
    }
    counts
}

/// `{ summaries(): SessionSummary[]; dispose(): Promise<void> }`
pub trait RosterBar: Send + Sync {
    fn summaries(&self) -> Vec<SessionSummary>;
    fn dispose(&self) -> futures::future::BoxFuture<'static, ()>;
}

struct CompactionNotice {
    session_id: Option<String>,
    text: Text,
}

impl super::interactive_mode_services::Component for CompactionNotice {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = vec![String::new()];
        lines.extend(super::interactive_mode_services::Component::render(&self.text, width));
        lines
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// Port of `InteractiveMode`.
pub struct InteractiveMode {
    // Static configuration
    version: String,
    start_hint: &'static str,
    options: InteractiveModeOptions,

    // Local stand-ins for the TUI tree. The real TUI lives in pi-tui; the Rust
    // port keeps the same containers so the wiring and render order match.
    ui: super::interactive_mode_services::Tui,
    header_container: super::interactive_mode_services::Container,
    history_container: super::interactive_mode_services::Container,
    chat_container: super::interactive_mode_services::Container,
    shortcut_guide_container: super::interactive_mode_services::Container,
    pending_messages_container: super::interactive_mode_services::Container,
    status_container: super::interactive_mode_services::Container,
    queued_messages_container: super::interactive_mode_services::Container,
    side_question_container: super::interactive_mode_services::Container,
    feature_hint_container: super::interactive_mode_services::Container,
    widget_container_above: super::interactive_mode_services::Container,
    widget_container_below: super::interactive_mode_services::Container,
    recap_container: super::interactive_mode_services::Container,
    main_container: super::interactive_mode_services::Container,
    main_view_container: super::interactive_mode_services::Container,
    prompt_dock: super::interactive_mode_services::Container,
    footer_slot: super::interactive_mode_services::Container,
    editor_container: super::interactive_mode_services::Container,

    ui_services: InteractiveModeUiServices,
    agent_connection: AgentConnection,
    local_session_host: Option<Arc<dyn InteractiveModeLocalSessionHost>>,
    bind_local_session_extensions: bool,

    prompt_stash_store: Option<Arc<std::sync::Mutex<ClientPromptStashStore>>>,
    prompt_stash_session_id: Option<String>,
    prompt_stash_state: PromptStashState,
    prompt_stash_handle: Option<super::prompt_stash_state::PromptStashHandle>,
    pending_prompt_stash_releases: Vec<(String, super::prompt_stash_state::PromptStashHandle)>,

    is_initialized: bool,
    fullscreen_enabled: bool,
    hide_thinking_block: bool,
    default_hidden_thinking_label: String,
    hidden_thinking_label: String,

    tool_output_expanded: bool,
    agent_messages_expanded: bool,
    edit_diffs_expanded: bool,

    connection_state: Option<AgentConnectionState>,
    connection_commands: Vec<super::interactive_mode_services::AgentConnectionSlashCommand>,
    connection_model_catalog: Vec<AgentConnectionModel>,
    connection_configured_providers: HashSet<String>,
    connection_models_fetched_at: f64,
    connection_models_refresh_version: u64,
    connection_models_refresh_in_flight: bool,
    heartbeat_catalog: Vec<AgentConnectionHeartbeat>,

    subagent_snapshots: HashMap<String, AgentConnectionRlmChildAgentSnapshot>,
    rlm_node_id: Option<String>,
    roster_bar: Option<Arc<dyn RosterBar>>,
    subagent_summary_line: SubagentSummaryLine,

    queue_selection: QueueSelection,
    is_applying_queue_selection_text: bool,

    pasted_images: HashMap<i64, ImageContent>,
    next_image_marker_id: i64,

    agents_view_request: Option<InteractiveModeRunResultType>,
    shutdown_requested: bool,
    ctrl_c_exit_hint_expires_at: f64,
    escape_repeat_action: Option<&'static str>,
    escape_repeat_expires_at: f64,
    anthropic_subscription_warning_shown: bool,

    last_status_spacer_index: Option<usize>,
    last_status_text_index: Option<usize>,
    restored_draft_notice: std::rc::Rc<std::cell::RefCell<Option<String>>>,
    last_goal_announcement: Option<GoalAnnouncementSnapshot>,

    feature_hint_deck: FeatureHintDeck,
    current_feature_hint: Option<String>,
    feature_hint_eligible_at: f64,
    feature_hint_run_pending: bool,
    feature_hint_suppressed_by_queue: bool,

    working_visible: bool,
    working_message: Option<String>,
    working_started_at: Option<f64>,
    /// `autoCompactionLoader` - the label of the mounted compaction loader.
    compaction_loader_label: Option<String>,
    turn_started_at: Option<f64>,
    working_indicator_options: Option<LoaderIndicatorOptions>,
    pulse_frame: i64,
    working_pulse_active: bool,

    activity_tracker: super::agent_activity::AgentActivityTracker,
    context_usage_token_baseline: f64,
    context_usage_refresh_generation: u64,
    context_usage_last_success_generation: u64,

    session_recap: Option<String>,

    /// Exit hint duration (ms) from `InteractiveMode.EXIT_HINT_DURATION_MS`.
    exit_hint_duration_ms: f64,
    /// Escape repeat window (ms) from `InteractiveMode.ESCAPE_REPEAT_WINDOW_MS`.
    escape_repeat_window_ms: f64,
}

impl InteractiveMode {
    pub const EXIT_HINT_DURATION_MS: f64 = 2000.0;
    pub const ESCAPE_REPEAT_WINDOW_MS: f64 = 500.0;

    /// Port of the `InteractiveMode` constructor (TUI construction only).
    pub fn new(mut options: InteractiveModeOptions) -> Result<Self, String> {
        let ui_services = match options.ui_services.take() {
            Some(services) => Some(services),
            None => options
                .local_session_host
                .as_ref()
                .map(|host| host.create_ui_services()),
        };
        let Some(ui_services) = ui_services else {
            return Err("InteractiveMode requires uiServices when no localSessionHost is supplied".to_string());
        };
        let prompt_stash_store = options.prompt_stash_store.clone();
        let prompt_stash_session_id = options.prompt_stash_session_id.clone();
        let prompt_stash_handle = match (&prompt_stash_store, &prompt_stash_session_id) {
            (Some(store), Some(session_id)) => Some(
                store
                    .lock()
                    .expect("prompt stash store poisoned")
                    .for_session(session_id),
            ),
            _ => None,
        };
        let prompt_stash_state = match (&prompt_stash_store, prompt_stash_handle) {
            (Some(store), Some(handle)) => store
                .lock()
                .expect("prompt stash store poisoned")
                .state(handle)
                .cloned()
                .unwrap_or_default(),
            _ => PromptStashState::default(),
        };
        let bind_local_session_extensions =
            options.bind_local_session_extensions || options.local_session_host.is_some();
        // `options` is moved into the mode below, so read out everything the mode
        // needs from it first.
        let local_session_host = options.local_session_host.clone();
        let agent_connection = options_agent_connection(&options);
        if bind_local_session_extensions && options.local_session_host.is_none() {
            return Err("Local extension binding requires localSessionHost".to_string());
        }

        let mut mode = Self {
            version: VERSION.to_string(),
            start_hint: get_random_start_hint(&default_random),
            options,
            ui: super::interactive_mode_services::Tui::default(),
            header_container: super::interactive_mode_services::Container::new(),
            history_container: super::interactive_mode_services::Container::new(),
            chat_container: super::interactive_mode_services::Container::new(),
            shortcut_guide_container: super::interactive_mode_services::Container::new(),
            pending_messages_container: super::interactive_mode_services::Container::new(),
            status_container: super::interactive_mode_services::Container::new(),
            queued_messages_container: super::interactive_mode_services::Container::new(),
            side_question_container: super::interactive_mode_services::Container::new(),
            feature_hint_container: super::interactive_mode_services::Container::new(),
            widget_container_above: super::interactive_mode_services::Container::new(),
            widget_container_below: super::interactive_mode_services::Container::new(),
            recap_container: super::interactive_mode_services::Container::new(),
            main_container: super::interactive_mode_services::Container::new(),
            main_view_container: super::interactive_mode_services::Container::new(),
            prompt_dock: super::interactive_mode_services::Container::new(),
            footer_slot: super::interactive_mode_services::Container::new(),
            editor_container: super::interactive_mode_services::Container::new(),
            ui_services,
            agent_connection,
            local_session_host,
            bind_local_session_extensions,
            prompt_stash_store,
            prompt_stash_session_id,
            prompt_stash_state,
            prompt_stash_handle,
            pending_prompt_stash_releases: Vec::new(),
            is_initialized: false,
            fullscreen_enabled: false,
            hide_thinking_block: false,
            default_hidden_thinking_label: "Thinking...".to_string(),
            hidden_thinking_label: "Thinking...".to_string(),
            tool_output_expanded: false,
            agent_messages_expanded: false,
            edit_diffs_expanded: false,
            connection_state: None,
            connection_commands: Vec::new(),
            connection_model_catalog: Vec::new(),
            connection_configured_providers: HashSet::new(),
            connection_models_fetched_at: 0.0,
            connection_models_refresh_version: 0,
            connection_models_refresh_in_flight: false,
            heartbeat_catalog: Vec::new(),
            subagent_snapshots: HashMap::new(),
            rlm_node_id: None,
            roster_bar: None,
            subagent_summary_line: SubagentSummaryLine::new(Box::new(|| None), Box::new(|| None), Box::new(|| None)),
            queue_selection: QueueSelection::new(),
            is_applying_queue_selection_text: false,
            pasted_images: HashMap::new(),
            next_image_marker_id: 1,
            agents_view_request: None,
            shutdown_requested: false,
            ctrl_c_exit_hint_expires_at: 0.0,
            escape_repeat_action: None,
            escape_repeat_expires_at: 0.0,
            anthropic_subscription_warning_shown: false,
            last_status_spacer_index: None,
            last_status_text_index: None,
            restored_draft_notice: std::rc::Rc::new(std::cell::RefCell::new(None)),
            last_goal_announcement: None,
            feature_hint_deck: FeatureHintDeck::default(),
            current_feature_hint: None,
            feature_hint_eligible_at: 0.0,
            feature_hint_run_pending: false,
            feature_hint_suppressed_by_queue: false,
            working_visible: true,
            working_message: None,
            working_started_at: None,
            compaction_loader_label: None,
            turn_started_at: None,
            working_indicator_options: None,
            pulse_frame: 0,
            working_pulse_active: false,
            activity_tracker: super::agent_activity::AgentActivityTracker::default(),
            context_usage_token_baseline: 0.0,
            context_usage_refresh_generation: 0,
            context_usage_last_success_generation: 0,
            session_recap: None,
            exit_hint_duration_ms: Self::EXIT_HINT_DURATION_MS,
            escape_repeat_window_ms: Self::ESCAPE_REPEAT_WINDOW_MS,
        };
        mode.hydrate_prompt_stash();
        mode.hide_thinking_block = mode.with_settings(|settings| settings.get_hide_thinking_block());
        Ok(mode)
    }

    /// `this.uiServices`
    fn ui_services(&self) -> &InteractiveModeUiServices {
        &self.ui_services
    }

    /// `private get settingsManager()`
    ///
    /// The TypeScript getter hands out the live `SettingsManager`; the port shares it
    /// as `Arc<Mutex<..>>`, so every read takes the lock for the call it wraps.
    fn with_settings<R>(&self, read: impl FnOnce(&super::interactive_mode_services::SettingsManager) -> R) -> R {
        let settings = self.ui_services.settings_manager.lock().expect("settings manager poisoned");
        read(&settings)
    }

    /// Same as `with_settings`, for the one mutating call (`setOnboardingShown`).
    fn with_settings_mut<R>(
        &self,
        write: impl FnOnce(&mut super::interactive_mode_services::SettingsManager) -> R,
    ) -> R {
        let mut settings = self.ui_services.settings_manager.lock().expect("settings manager poisoned");
        write(&mut settings)
    }

    /// `private get settingsManager()` - the raw shared handle.
    pub fn settings_manager(&self) -> &Arc<Mutex<super::interactive_mode_services::SettingsManager>> {
        &self.ui_services.settings_manager
    }

    /// `private get modelRegistry()`
    pub fn model_registry(&self) -> &Arc<Mutex<super::interactive_mode_services::ModelRegistry>> {
        &self.ui_services.model_registry
    }

    /// `private getLocalSessionHost()`
    pub fn get_local_session_host(&self) -> Result<&Arc<dyn InteractiveModeLocalSessionHost>, String> {
        self.local_session_host
            .as_ref()
            .ok_or_else(|| "Local session host is not available in connection-backed interactive mode".to_string())
    }

    /// Port of `hydratePromptStash`.
    fn hydrate_prompt_stash(&mut self) {
        let mut stashes: Vec<PromptStash> = Vec::new();
        if let Some(stash) = &self.prompt_stash_state.stash {
            stashes.push(stash.clone());
        }
        if let Some(queued) = &self.prompt_stash_state.queued_stashes {
            stashes.extend(queued.iter().cloned());
        }
        for stash in stashes {
            if let Some(images) = &stash.images {
                for (marker_id, image) in images {
                    self.pasted_images.insert(*marker_id, image.clone());
                    self.next_image_marker_id = self.next_image_marker_id.max(marker_id + 1);
                }
            }
            for marker_id in super::image_markers::image_marker_ids(&stash.text) {
                self.next_image_marker_id = self.next_image_marker_id.max(marker_id as i64 + 1);
            }
        }
    }

    /// Port of `bindPromptStashSession`.
    fn bind_prompt_stash_session(&mut self, session_id: &str) {
        if self.prompt_stash_store.is_none() || self.prompt_stash_session_id.as_deref() == Some(session_id) {
            return;
        }
        self.release_prompt_stash_session();
        self.prompt_stash_session_id = Some(session_id.to_string());
        let Some(store) = self.prompt_stash_store.clone() else {
            return;
        };
        let (handle, state) = {
            let mut store = store.lock().expect("prompt stash store poisoned");
            let handle = store.for_session(session_id);
            let state = store.state(handle).cloned().unwrap_or_default();
            (handle, state)
        };
        self.prompt_stash_handle = Some(handle);
        self.prompt_stash_state = state;
        self.hydrate_prompt_stash();
    }

    /// Port of `releasePromptStashSession`.
    fn release_prompt_stash_session(&mut self) {
        if self.pending_submissions_pending() > 0 {
            // Capture the pair: a rebind may repoint the fields before the deferred
            // release fires, and repeated rebinds/teardowns each defer their own pair.
            if let (Some(session_id), Some(handle)) =
                (self.prompt_stash_session_id.clone(), self.prompt_stash_handle)
            {
                if !self
                    .pending_prompt_stash_releases
                    .iter()
                    .any(|(pending_session_id, _)| *pending_session_id == session_id)
                {
                    self.pending_prompt_stash_releases.push((session_id, handle));
                }
            }
            return;
        }
        let pending = std::mem::take(&mut self.pending_prompt_stash_releases);
        let Some(store) = self.prompt_stash_store.clone() else {
            return;
        };
        let mut store = store.lock().expect("prompt stash store poisoned");
        for (session_id, handle) in pending {
            store.release(&session_id, handle);
        }
        if let (Some(session_id), Some(handle)) = (self.prompt_stash_session_id.clone(), self.prompt_stash_handle) {
            store.release(&session_id, handle);
        }
    }

    /// Port of `completeDeferredPromptStashRelease`.
    fn complete_deferred_prompt_stash_release(&mut self) {
        let pending = std::mem::take(&mut self.pending_prompt_stash_releases);
        if pending.is_empty() {
            return;
        }
        let Some(store) = self.prompt_stash_store.clone() else {
            return;
        };
        let mut store = store.lock().expect("prompt stash store poisoned");
        for (session_id, handle) in pending {
            store.release(&session_id, handle);
        }
    }

    /// The `inputSubmissionsPending` counter.
    fn pending_submissions_pending(&self) -> usize {
        self.pending_prompt_stash_releases.len()
    }

    /// Port of `getAutocompleteSourceTag`.
    pub fn get_autocomplete_source_tag(
        &self,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> Option<String> {
        let source_info = source_info?;
        let scope_prefix = match source_info.scope.as_str() {
            "user" => "user",
            "project" => "project",
            _ => "temporary",
        };
        let source = source_info.source.trim();
        if source == "builtin" {
            return Some("builtin".to_string());
        }
        if source == "auto" || source == "local" || source == "cli" {
            return Some(scope_prefix.to_string());
        }
        if let Some(rest) = source.strip_prefix("npm:") {
            return Some(format!("{scope_prefix}:npm:{rest}"));
        }
        if let Some(git_source) = crate::utils::git::parse_git_url(source) {
            let git_ref = git_source
                .reference
                .as_ref()
                .map(|git_ref| format!("@{git_ref}"))
                .unwrap_or_default();
            return Some(format!(
                "{scope_prefix}:git:{}/{}{git_ref}",
                git_source.host, git_source.path
            ));
        }
        Some(scope_prefix.to_string())
    }

    /// Port of `getAutocompleteSourceLabel`.
    pub fn get_autocomplete_source_label(
        &self,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> Option<String> {
        self.get_autocomplete_source_tag(source_info).map(|source_tag| format!("#{source_tag}"))
    }

    /// Port of `getBuiltInCommandConflictDiagnostics`.
    pub fn get_built_in_command_conflict_diagnostics(
        &self,
        commands: &[super::interactive_mode_services::AgentConnectionSlashCommand],
    ) -> Vec<super::interactive_mode_services::AgentConnectionResourceDiagnostic> {
        commands
            .iter()
            .filter(|command| command.source == "extension")
            .filter(|command| {
                crate::core::slash_commands::is_builtin_slash_command_name(
                    command.registered_name.as_deref().unwrap_or(&command.name),
                )
            })
            .map(|command| {
                let registered = command.registered_name.clone().unwrap_or_else(|| command.name.clone());
                let message = if command.name == registered {
                    format!(
                        "Extension command '/{}' conflicts with built-in interactive command. Skipping in autocomplete.",
                        command.name
                    )
                } else {
                    format!(
                        "Extension command '/{registered}' conflicts with built-in interactive command. Available as '/{}'.",
                        command.name
                    )
                };
                super::interactive_mode_services::AgentConnectionResourceDiagnostic {
                    type_: "warning".to_string(),
                    message,
                    path: Some(command.source_info.path.clone()),
                    ..Default::default()
                }
            })
            .collect()
    }

    /// Port of `isRecognizedSlashCommand`.
    pub fn is_recognized_slash_command(&self, name: &str) -> bool {
        crate::core::slash_commands::is_builtin_slash_command_name(name)
            || self.connection_commands.iter().any(|command| command.name == name)
    }

    /// Port of `getModelFallbackWarningAction`.
    pub fn get_model_fallback_warning_action(&self, model_fallback_message: Option<&str>) -> ModelFallbackWarningAction {
        let Some(model_fallback_message) = model_fallback_message else {
            return ModelFallbackWarningAction::Suppress;
        };
        let model = self.get_current_model();
        if crate::core::auth_guidance::is_obsolete_model_fallback_message(
            Some(model_fallback_message), model.as_ref().map(|model| (model.provider.as_str(), model.id.as_str())),
        ) {
            return ModelFallbackWarningAction::Suppress;
        }
        ModelFallbackWarningAction::Show
    }

    /// Port of `getOnboardingState`.
    pub fn get_onboarding_state(&self) -> super::onboarding::OnboardingStartupState<'_> {
        // The TypeScript hands the live objects straight through; the port shares them
        // as `Arc<Mutex<..>>`, and the adapters below expose the same read surface.
        super::onboarding::OnboardingStartupState {
            settings_manager: self.settings_manager().as_ref(),
            model_registry: self.model_registry().as_ref(),
            model: self.get_current_model(),
        }
    }

    /// Port of `shouldRunOnboarding`.
    pub fn should_run_onboarding(&self) -> bool {
        should_run_onboarding(&self.get_onboarding_state())
    }

    /// Port of `shouldRunPrimeCliOnboardingSplash`.
    pub fn should_run_prime_cli_onboarding_splash(&self) -> bool {
        super::onboarding::should_run_prime_cli_onboarding_splash(&self.get_onboarding_state())
    }

    /// Port of `markOnboardingShown` (interactive-mode.ts:1865-1869).
    ///
    /// `SettingsManager::set_onboarding_shown` writes the global settings file,
    /// so this is the persistence point `runStartupOnboarding` performs *before*
    /// the flow opens (TS:1880-1881).
    pub fn mark_onboarding_shown(&self) {
        if !self.with_settings(|settings| settings.get_onboarding_shown()) {
            self.with_settings_mut(|settings| settings.set_onboarding_shown(true));
        }
    }

    /// Port of `runStartupOnboarding`'s pre-splash half
    /// (interactive-mode.ts:1871-1881).
    ///
    /// Returns whether the onboarding flow should open. PARTIAL: the telemetry
    /// `captureOnboardingCompleted` call (TS:1888-1901) belongs to the telemetry
    /// slice, and the `runOnboardingFlow` body is driven by the native host
    /// (`showOnboardingSplash` / the model selector) because it needs the TUI.
    pub fn run_startup_onboarding(&mut self) -> bool {
        if !self.should_run_onboarding() {
            return false;
        }
        self.mark_onboarding_shown();
        true
    }

    /// `runOnboardingFlow`'s splash branch key
    /// (interactive-mode.ts:1912-1922): the Prime CLI users get the splash first,
    /// everyone else goes straight to the model menu.
    pub fn onboarding_uses_prime_cli_splash(&self) -> bool {
        self.should_run_prime_cli_onboarding_splash()
    }

    /// Port of `formatDisplayPath`.
    pub fn format_display_path(&self, path: &str) -> String {
        let home = home_dir();
        let mut result = path.to_string();
        if !home.is_empty() && result.starts_with(&home) {
            result = format!("~{}", &result[home.len()..]);
        }
        result
    }

    /// Port of `isPackageSource`.
    pub fn is_package_source(
        &self,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> bool {
        let source = source_info.map(|info| info.source.as_str()).unwrap_or("");
        source.starts_with("npm:") || source.starts_with("git:")
    }

    /// Port of `getShortPath`.
    pub fn get_short_path(
        &self,
        full_path: &str,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> String {
        let base_dir = source_info.and_then(|info| info.base_dir.clone());
        if let Some(base_dir) = base_dir {
            if self.is_package_source(source_info) {
                let relative_path = relative_path_between(&resolve_path(&base_dir), &resolve_path(full_path));
                if !relative_path.is_empty()
                    && relative_path != "."
                    && !relative_path.starts_with("..")
                    && !relative_path.starts_with(&format!("..{}", std::path::MAIN_SEPARATOR))
                    && !Path::new(&relative_path).is_absolute()
                {
                    return relative_path.replace('\\', "/");
                }
            }
        }

        let source = source_info.map(|info| info.source.as_str()).unwrap_or("");
        if let Some(captures) = regex_match_node_modules(full_path) {
            if source.starts_with("npm:") {
                return captures;
            }
        }

        if let Some(captures) = regex_match_git_path(full_path) {
            if source.starts_with("git:") {
                return captures;
            }
        }

        self.format_display_path(full_path)
    }

    /// `/node_modules\/(@?[^/]+(?:\/[^/]+)?)\/(.*)/`
    /// Port of `getCompactPathLabel`.
    pub fn get_compact_path_label(
        &self,
        resource_path: &str,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> String {
        let short_path = self.get_short_path(resource_path, source_info);
        let normalized_path = short_path.replace('\\', "/");
        let segments: Vec<&str> = normalized_path
            .split('/')
            .filter(|segment| !segment.is_empty() && *segment != "~")
            .collect();
        if let Some(last) = segments.last() {
            return (*last).to_string();
        }
        short_path
    }

    /// Port of `getCompactDisplayPathSegments`.
    pub fn get_compact_display_path_segments(&self, resource_path: &str) -> Vec<String> {
        self.format_display_path(resource_path)
            .replace('\\', "/")
            .split('/')
            .filter(|segment| !segment.is_empty() && *segment != "~")
            .map(|segment| segment.to_string())
            .collect()
    }

    /// Port of `getCompactPackageSourceLabel`.
    pub fn get_compact_package_source_label(
        &self,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> String {
        let source = source_info.map(|info| info.source.as_str()).unwrap_or("");
        if let Some(rest) = source.strip_prefix("npm:") {
            return if rest.is_empty() { source.to_string() } else { rest.to_string() };
        }
        if let Some(git_source) = crate::utils::git::parse_git_url(source) {
            if !git_source.path.is_empty() {
                return git_source.path.clone();
            }
        }
        source.to_string()
    }

    /// Port of `getCompactExtensionLabel`.
    pub fn get_compact_extension_label(
        &self,
        resource_path: &str,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> String {
        if !self.is_package_source(source_info) {
            return self.get_compact_path_label(resource_path, source_info);
        }

        let source_label = self.get_compact_package_source_label(source_info);
        if source_label.is_empty() {
            return self.get_compact_path_label(resource_path, source_info);
        }

        let short_path = self.get_short_path(resource_path, source_info).replace('\\', "/");
        let package_path = short_path.strip_prefix("extensions/").unwrap_or(&short_path);
        let (dir, name) = posix_parse(package_path);

        if name == "index" {
            return if dir.is_empty() || dir == "." {
                source_label
            } else {
                format!("{source_label}:{dir}")
            };
        }

        format!("{source_label}:{package_path}")
    }

    /// Port of `getCompactNonPackageExtensionLabel`.
    pub fn get_compact_non_package_extension_label(
        &self,
        resource_path: &str,
        index: usize,
        all_paths: &[(String, Vec<String>)],
    ) -> String {
        let Some((_, segments)) = all_paths.get(index) else {
            return self.get_compact_path_label(resource_path, None);
        };
        if segments.is_empty() {
            return self.get_compact_path_label(resource_path, None);
        }

        for segment_count in 1..=segments.len() {
            let candidate = segments[segments.len() - segment_count..].join("/");
            let is_unique = all_paths.iter().enumerate().all(|(item_index, (_, item_segments))| {
                if item_index == index {
                    return true;
                }
                item_segments[item_segments.len().saturating_sub(segment_count)..].join("/") != candidate
            });
            if is_unique {
                return candidate;
            }
        }

        segments.join("/")
    }

    /// Port of `getCompactExtensionLabels`.
    pub fn get_compact_extension_labels(
        &self,
        extensions: &[(String, Option<super::interactive_mode_services::AgentConnectionSourceInfo>)],
    ) -> Vec<String> {
        let non_package_extensions: Vec<(String, Vec<String>)> = extensions
            .iter()
            .map(|(path, _)| {
                let mut segments = self.get_compact_display_path_segments(path);
                let last_segment = segments.last().cloned();
                if segments.len() > 1
                    && matches!(last_segment.as_deref(), Some("index.ts") | Some("index.js"))
                {
                    segments.pop();
                }
                (path.clone(), segments)
            })
            .filter(|(_, _)| true)
            .collect();
        let non_package_extensions: Vec<(String, Vec<String>)> = extensions
            .iter()
            .zip(non_package_extensions.iter())
            .filter(|((_, source_info), _)| !self.is_package_source(source_info.as_ref()))
            .map(|(_, (path, segments))| (path.clone(), segments.clone()))
            .collect();

        extensions
            .iter()
            .map(|(path, source_info)| {
                if self.is_package_source(source_info.as_ref()) {
                    return self.get_compact_extension_label(path, source_info.as_ref());
                }
                match non_package_extensions.iter().position(|(item_path, _)| item_path == path) {
                    None => self.get_compact_path_label(path, source_info.as_ref()),
                    Some(non_package_index) => {
                        self.get_compact_non_package_extension_label(path, non_package_index, &non_package_extensions)
                    }
                }
            })
            .collect()
    }

    /// Port of `formatExtensionDisplayPath`.
    pub fn format_extension_display_path(&self, path: &str) -> String {
        let result = self.format_display_path(path);
        let result = result.strip_suffix("/index.ts").unwrap_or(&result).to_string();
        result.strip_suffix("/index.js").unwrap_or(&result).to_string()
    }

    /// Port of `formatContextPath`.
    pub fn format_context_path(&self, p: &str) -> String {
        let cwd = resolve_path(&self.get_current_cwd());
        let absolute_path = if Path::new(p).is_absolute() {
            resolve_path(p)
        } else {
            resolve_path(&format!("{cwd}{}{p}", std::path::MAIN_SEPARATOR))
        };
        if let Some(relative_path) = get_cwd_relative_path(&absolute_path, &cwd) {
            return relative_path;
        }
        self.format_display_path(&absolute_path)
    }

    /// Port of `getScopeGroup`.
    pub fn get_scope_group(
        &self,
        source_info: Option<&super::interactive_mode_services::AgentConnectionSourceInfo>,
    ) -> &'static str {
        let source = source_info.map(|info| info.source.as_str()).unwrap_or("local");
        let scope = source_info.map(|info| info.scope.as_str()).unwrap_or("project");
        if source == "cli" || scope == "temporary" {
            return "path";
        }
        if scope == "user" {
            return "user";
        }
        if scope == "project" {
            return "project";
        }
        "path"
    }

    /// Port of `getStartupExpansionState`.
    pub fn get_startup_expansion_state(&self) -> bool {
        self.options.verbose || self.tool_output_expanded
    }

    /// Port of `getConnectionQueue`.
    pub fn get_connection_queue(&self) -> AgentConnectionQueueState {
        AgentConnectionQueueState {
            steering: self
                .connection_state
                .as_ref()
                .map(|state| state.session_actions.steering.clone())
                .unwrap_or_default(),
            follow_up: self
                .connection_state
                .as_ref()
                .map(|state| state.session_actions.follow_ups.clone())
                .unwrap_or_default(),
        }
    }

    /// Port of `getAllQueuedMessages`.
    pub fn get_all_queued_messages(&self) -> AgentConnectionQueueState {
        self.get_connection_queue()
    }

    /// Port of `getScopedHeartbeats`.
    pub fn get_scoped_heartbeats(&self) -> Vec<AgentConnectionHeartbeat> {
        let identity = self.connection_state.as_ref().map(|state| {
            super::heartbeat_scope::HeartbeatSessionIdentity {
                active_session_id: state.active_session_id.clone(),
                session_id: state.session_id.clone(),
            }
        });
        let children: Vec<AgentConnectionRlmChildAgentSnapshot> =
            self.subagent_snapshots.values().cloned().collect();
        scope_heartbeats_to_session(&self.heartbeat_catalog, identity.as_ref(), &children)
    }

    /// Port of `getCurrentCwd`.
    pub fn get_current_cwd(&self) -> String {
        self.connection_state
            .as_ref()
            .map(|state| state.cwd.clone())
            .unwrap_or_else(|| self.ui_services().get_initial_cwd())
    }

    /// Port of `getCurrentSessionName`.
    pub fn get_current_session_name(&self) -> Option<String> {
        self.connection_state
            .as_ref()
            .and_then(|state| state.session_name.clone())
            .or_else(|| self.ui_services().get_initial_session_name())
    }

    /// Port of `getCurrentModel`.
    pub fn get_current_model(&self) -> Option<&AgentConnectionModel> {
        self.connection_state.as_ref().and_then(|state| state.model.as_ref())
    }

    /// Port of `getCurrentModelId`.
    pub fn get_current_model_id(&self) -> Option<String> {
        self.get_current_model().map(|model| model.id.clone())
    }

    /// Port of `isAgentStreaming`.
    pub fn is_agent_streaming(&self) -> bool {
        self.connection_state.as_ref().map(|state| state.is_streaming).unwrap_or(false)
    }

    /// Port of `isAgentCompacting`.
    pub fn is_agent_compacting(&self) -> bool {
        self.connection_state.as_ref().map(|state| state.is_compacting).unwrap_or(false)
    }

    /// Port of `isBashRunning`.
    pub fn is_bash_running(&self) -> bool {
        self.connection_state.as_ref().map(|state| state.is_bash_running).unwrap_or(false)
    }

    /// Port of `hasInterruptibleWork`.
    pub fn has_interruptible_work(&self) -> bool {
        self.is_agent_streaming()
            || self.is_agent_compacting()
            || self.is_bash_running()
            || self.get_retry_attempt() > 0.0
            || self
                .connection_state
                .as_ref()
                .map(|state| state.session_actions.active.is_some() || state.session_actions.queued_count > 0)
                .unwrap_or(false)
    }

    /// Port of `getRetryAttempt`.
    pub fn get_retry_attempt(&self) -> f64 {
        self.connection_state.as_ref().map(|state| state.retry_attempt).unwrap_or(0.0)
    }

    /// Port of `getQueuedActionCount`.
    pub fn get_queued_action_count(&self) -> usize {
        self.connection_state
            .as_ref()
            .map(|state| state.session_actions.queued_count)
            .unwrap_or(0)
    }

    /// Port of `getGoalState`.
    pub fn get_goal_state(&self) -> GoalState {
        self.connection_state
            .as_ref()
            .map(|state| state.goal.clone())
            .unwrap_or_else(empty_goal_state)
    }

    /// Port of `getConnectionContextUsage`.
    pub fn get_connection_context_usage(&self) -> Option<ContextUsage> {
        let snapshot = self.connection_state.as_ref().map(|state| state.context_usage.clone())?;
        let Some(tokens) = snapshot.tokens else {
            return Some(snapshot);
        };
        if snapshot.context_window <= 0.0 {
            return Some(snapshot);
        }
        // Add only the output produced since the snapshot was last refreshed. The activity
        // tracker accumulates across auto-retries within a turn, so subtract the baseline
        // captured at the last refresh to avoid re-adding a failed attempt's tokens.
        let in_flight = if self.is_agent_streaming() {
            0.0f64.max(self.activity_tracker.get_status().tokens - self.context_usage_token_baseline)
        } else {
            0.0
        };
        if in_flight <= 0.0 {
            return Some(snapshot);
        }
        let tokens = tokens + in_flight;
        Some(ContextUsage {
            tokens: Some(tokens),
            context_window: snapshot.context_window,
            percent: Some((tokens / snapshot.context_window) * 100.0),
        })
    }

    /// Port of `getScopedModelState`.
    pub fn get_scoped_model_state(&self) -> Vec<super::interactive_mode_services::AgentConnectionScopedModel> {
        self.connection_state
            .as_ref()
            .map(|state| state.scoped_models.clone())
            .unwrap_or_default()
    }

    /// Port of `updateTerminalTitle`.
    pub fn update_terminal_title(&mut self) {
        let cwd_basename = basename(&self.get_current_cwd());
        let session_name = self.get_current_session_name();
        self.ui.terminal.set_title(terminal_title_text(
            crate::config::app_display_title(),
            session_name.as_deref(),
            &cwd_basename,
        ));
    }

    /// Port of `formatGoalElapsed`.
    pub fn format_goal_elapsed(&self, seconds: f64) -> String {
        let total_seconds = 0.0f64.max(seconds.trunc());
        if total_seconds < 60.0 {
            return format!("{}s", js_number_to_string(total_seconds));
        }
        let minutes = (total_seconds / 60.0).floor();
        let remaining_seconds = total_seconds % 60.0;
        if minutes < 60.0 {
            return format!(
                "{}m {}s",
                js_number_to_string(minutes),
                pad_start(&js_number_to_string(remaining_seconds), 2, '0')
            );
        }
        let hours = (minutes / 60.0).floor();
        let remaining_minutes = minutes % 60.0;
        format!(
            "{}h {}m",
            js_number_to_string(hours),
            pad_start(&js_number_to_string(remaining_minutes), 2, '0')
        )
    }

    /// Port of `formatWorkingElapsed`.
    pub fn format_working_elapsed(&self, elapsed_ms: f64) -> String {
        let total_seconds = 0.0f64.max((elapsed_ms / 1000.0).floor());
        if total_seconds < 60.0 {
            return format!("{}s", js_number_to_string(total_seconds));
        }
        let minutes = (total_seconds / 60.0).floor();
        let seconds = total_seconds % 60.0;
        if minutes < 60.0 {
            return format!(
                "{}m {}s",
                js_number_to_string(minutes),
                pad_start(&js_number_to_string(seconds), 2, '0')
            );
        }
        let hours = (minutes / 60.0).floor();
        let remaining_minutes = minutes % 60.0;
        if hours < 24.0 {
            return format!(
                "{}h {}m {}s",
                js_number_to_string(hours),
                pad_start(&js_number_to_string(remaining_minutes), 2, '0'),
                pad_start(&js_number_to_string(seconds), 2, '0')
            );
        }
        let days = (hours / 24.0).floor();
        let remaining_hours = hours % 24.0;
        format!(
            "{}d {}h {}m {}s",
            js_number_to_string(days),
            pad_start(&js_number_to_string(remaining_hours), 2, '0'),
            pad_start(&js_number_to_string(remaining_minutes), 2, '0'),
            pad_start(&js_number_to_string(seconds), 2, '0')
        )
    }

    /// Port of `getTrayGoalLabel`.
    pub fn get_tray_goal_label(&self) -> Option<String> {
        let goal = self.get_goal_state();
        match goal.status.as_str() {
            "active" => Some(format!("Pursuing goal ({})", self.format_goal_elapsed(goal.time_used_seconds))),
            "paused" => Some(format!("Goal paused ({})", self.format_goal_elapsed(goal.time_used_seconds))),
            "budget_limited" => Some(format!(
                "Goal budget limited ({})",
                self.format_goal_elapsed(goal.time_used_seconds)
            )),
            "idle" | "complete" | "error" => None,
            _ => None,
        }
    }

    /// Port of `getTrayHeartbeatLabel`.
    pub fn get_tray_heartbeat_label(&self) -> Option<String> {
        let heartbeats = self.get_scoped_heartbeats();
        if heartbeats.is_empty() {
            return None;
        }
        let paused = heartbeats.iter().filter(|heartbeat| heartbeat.job.status == "paused").count();
        let count = format!(
            "{} heartbeat{}",
            heartbeats.len(),
            if heartbeats.len() == 1 { "" } else { "s" }
        );
        let paused_label = if paused > 0 { format!(" \u{b7} {paused} paused") } else { String::new() };
        let shortcut = self.key_text("app.heartbeats.open");
        Some(format!(
            "{count}{paused_label}{}",
            if shortcut.is_empty() { String::new() } else { format!(" ({shortcut})") }
        ))
    }

    /// Port of `getTrayContextLabel`, minus the usage counter.
    ///
    /// Goal and heartbeat labels keep their own line above the tray row; the
    /// context usage counter moved onto the tray row itself, right-aligned
    /// beside the model/effort label (`get_tray_context_usage_text`).
    pub fn get_tray_context_label(&self) -> Option<String> {
        let goal_label = self.get_tray_goal_label();
        let heartbeat_label = self.get_tray_heartbeat_label();
        let labels: Vec<String> = [goal_label, heartbeat_label]
            .into_iter()
            .flatten()
            .collect();
        if labels.is_empty() {
            None
        } else {
            Some(labels.join(" \u{b7} "))
        }
    }

    /// The context usage counter for the tray row: `146k (14%)`.
    ///
    /// `None` while the connection has no usage snapshot: unknown is never
    /// rendered as zero, and no setting is changed to produce a value.
    pub fn get_tray_context_usage_text(&self) -> Option<String> {
        let usage = self.get_connection_context_usage()?;
        match (usage.tokens, usage.percent) {
            (Some(tokens), Some(percent)) => Some(format!(
                "{} ({}%)",
                format_token_count(tokens),
                percent.round() as i64
            )),
            _ => None,
        }
    }

    /// Port of `getModelTrayLabel`.
    pub fn get_model_tray_label(&self) -> String {
        let Some(model) = self.get_current_model() else {
            return "\u{2014}".to_string();
        };
        let mut parts = vec![model.name.clone()];
        if model.reasoning {
            let level = self
                .connection_state
                .as_ref()
                .map(|state| state.thinking_level.clone())
                .unwrap_or(ThinkingLevel::Off);
            if level != ThinkingLevel::Off {
                parts.push(level.as_str().to_string());
            }
        }
        if self
            .connection_state
            .as_ref()
            .map(|state| {
                state.service_tier.as_ref().and_then(|tier| tier.as_deref()) == Some("priority")
            })
            .unwrap_or(false)
        {
            parts.push("fast".to_string());
        }
        parts.join(" \u{2022} ")
    }

    /// Port of `isNewChat`.
    pub fn is_new_chat(&self) -> bool {
        self.connection_state.as_ref().map(|state| state.message_count).unwrap_or(0.0) == 0.0
            && !self
                .connection_state
                .as_ref()
                .map(|state| state.is_streaming)
                .unwrap_or(false)
    }

    /// Port of `getShortcutsTrayHint`.
    pub fn get_shortcuts_tray_hint(&self) -> Option<String> {
        if !self.is_new_chat() {
            return None;
        }
        let shortcuts = self.key_text("app.shortcuts");
        if shortcuts.is_empty() {
            Some("/hotkeys for shortcuts".to_string())
        } else {
            Some(self.key_hint("app.shortcuts", "for shortcuts"))
        }
    }

    /// Port of `getAgentsViewTrayHint`.
    pub fn get_agents_view_tray_hint(&self) -> Option<String> {
        if !self.options.return_to_agents_view {
            return None;
        }
        Some(self.key_hint("app.agents.back", "agents/resume"))
    }

    /// Port of `getTrayLocationLabel`.
    pub fn get_tray_location_label(&self) -> Option<String> {
        let model_label = self.get_model_tray_label();
        let has_children = self.options.session_has_children || !self.subagent_snapshots.is_empty();
        let depth_label = format_agent_depth_label(self.options.session_depth, has_children);
        let shortcuts_hint = self.get_shortcuts_tray_hint();
        let agents_hint = self.get_agents_view_tray_hint();
        let labels: Vec<String> = [agents_hint, depth_label, Some(model_label), shortcuts_hint]
            .into_iter()
            .flatten()
            .collect();
        Some(labels.join("  "))
    }

    /// Port of `getTrayOverrideLabel`.
    pub fn get_tray_override_label(&self, editor_text: &str) -> Option<String> {
        if self.is_ctrl_c_exit_hint_visible() {
            let clear_key = self.key_text("app.clear");
            return Some(if clear_key.is_empty() {
                "Press again to exit".to_string()
            } else {
                format!("Press {clear_key} again to exit")
            });
        }
        if !self.is_agent_streaming() || editor_text.trim().is_empty() {
            return None;
        }
        Some(format!("{} to queue message", self.key_text("app.message.followUp")))
    }

    /// Port of `isCtrlCExitHintVisible`.
    pub fn is_ctrl_c_exit_hint_visible(&self) -> bool {
        self.ctrl_c_exit_hint_expires_at > now_ms()
    }

    /// Port of `getShortcutGuide` (interactive-mode.ts:10045-10073).
    pub fn get_shortcut_guide(&self) -> String {
        let tab = self.get_editor_key_display("tui.input.tab");
        let new_line = self.get_editor_key_display("tui.input.newLine");
        let clear_input = self.get_app_key_display("app.input.clear");
        let shortcuts_key = self.get_app_key_display("app.shortcuts");
        let select_model = self.get_app_key_display("app.model.select");
        let expand_tools = self.get_app_key_display("app.tools.expand");
        let expand_messages = self.get_app_key_display("app.messages.expand");
        let expand_edits = self.get_app_key_display("app.edits.expand");
        let toggle_thinking = self.get_app_key_display("app.thinking.toggle");
        let external_editor = self.get_app_key_display("app.editor.external");
        let prompt_stash = self.get_app_key_display("app.prompt.stash");
        let paste_image = self.get_app_key_display("app.clipboard.pasteImage");
        let help_line = if shortcuts_key.is_empty() {
            "`/hotkeys` full reference".to_string()
        } else {
            format!("`{shortcuts_key}` quick shortcuts \u{b7} `/hotkeys` full reference")
        };

        format!(
            "\
**Prompt**
`!` shell mode \u{b7} `/` commands \u{b7} `@` file paths
`{tab}` complete paths \u{b7} `{new_line}` new line
`{clear_input}` interrupt \u{b7} press twice to rewind or clear the prompt

**Controls**
`{select_model}` select model \u{b7} `/effort` set reasoning \u{b7} `{expand_tools}` tool output
`{expand_messages}` agent messages \u{b7} `{expand_edits}` edit diffs \u{b7} `{toggle_thinking}` thinking blocks \u{b7} `{prompt_stash}` stash prompt \u{b7} `{external_editor}` edit in `$EDITOR`
`{paste_image}` paste image

**Help**
{help_line}
"
        )
    }

    /// Port of `getHotkeysGuide` (interactive-mode.ts:10075-10195).
    ///
    /// The `**Extensions**` table the TypeScript appends when the local session
    /// host exposes registered shortcuts is omitted: the Rust
    /// `InteractiveModeLocalSessionHost` trait has no shortcut accessor, and the
    /// base guide is the full reference promised by `/hotkeys`.
    pub fn get_hotkeys_guide(&self) -> String {
        let cursor_up = self.get_editor_key_display("tui.editor.cursorUp");
        let cursor_down = self.get_editor_key_display("tui.editor.cursorDown");
        let cursor_left = self.get_editor_key_display("tui.editor.cursorLeft");
        let cursor_right = self.get_editor_key_display("tui.editor.cursorRight");
        let cursor_word_left = self.get_editor_key_display("tui.editor.cursorWordLeft");
        let cursor_word_right = self.get_editor_key_display("tui.editor.cursorWordRight");
        let cursor_line_start = self.get_editor_key_display("tui.editor.cursorLineStart");
        let cursor_line_end = self.get_editor_key_display("tui.editor.cursorLineEnd");
        let jump_forward = self.get_editor_key_display("tui.editor.jumpForward");
        let jump_backward = self.get_editor_key_display("tui.editor.jumpBackward");
        let page_up = self.get_editor_key_display("tui.editor.pageUp");
        let page_down = self.get_editor_key_display("tui.editor.pageDown");
        let submit = self.get_editor_key_display("tui.input.submit");
        let new_line = self.get_editor_key_display("tui.input.newLine");
        let delete_word_backward = self.get_editor_key_display("tui.editor.deleteWordBackward");
        let delete_word_forward = self.get_editor_key_display("tui.editor.deleteWordForward");
        let delete_to_line_start = self.get_editor_key_display("tui.editor.deleteToLineStart");
        let delete_to_line_end = self.get_editor_key_display("tui.editor.deleteToLineEnd");
        let yank = self.get_editor_key_display("tui.editor.yank");
        let yank_pop = self.get_editor_key_display("tui.editor.yankPop");
        let undo = self.get_editor_key_display("tui.editor.undo");
        let tab = self.get_editor_key_display("tui.input.tab");
        let clear = self.get_app_key_display("app.clear");
        let clear_input = self.get_app_key_display("app.input.clear");
        let interrupt = self.get_app_key_display("app.interrupt");
        let shortcuts_key = self.get_app_key_display("app.shortcuts");
        let exit = self.get_app_key_display("app.exit");
        let select_model = self.get_app_key_display("app.model.select");
        let expand_tools = self.get_app_key_display("app.tools.expand");
        let expand_messages = self.get_app_key_display("app.messages.expand");
        let expand_edits = self.get_app_key_display("app.edits.expand");
        let toggle_thinking = self.get_app_key_display("app.thinking.toggle");
        let focus_subagents = self.get_app_key_display("app.subagents.focus");
        let manage_heartbeats = self.get_app_key_display("app.heartbeats.open");
        let external_editor = self.get_app_key_display("app.editor.external");
        let prompt_stash = self.get_app_key_display("app.prompt.stash");
        let follow_up = self.get_app_key_display("app.message.followUp");
        let browse_queue = self.get_app_key_display("app.message.navigateOlder");
        let reorder_queue = format!(
            "{} / {}",
            self.get_app_key_display("app.message.moveEarlier"),
            self.get_app_key_display("app.message.moveLater"),
        );
        let paste_image = self.get_app_key_display("app.clipboard.pasteImage");
        let viewport_page_up = self.get_editor_key_display("tui.viewport.pageUp");
        let viewport_page_down = self.get_editor_key_display("tui.viewport.pageDown");
        let viewport_top = self.get_editor_key_display("tui.viewport.top");
        let viewport_follow = self.get_editor_key_display("tui.viewport.follow");
        let new_line_note = if crate::utils::pi_user_agent::process_platform() == "win32" {
            " (Ctrl+Enter on Windows Terminal)"
        } else {
            ""
        };
        let interrupt_row = if interrupt.is_empty() {
            String::new()
        } else {
            format!("| `{interrupt}` | Interrupt current operation |\n")
        };
        let shortcuts_row = if shortcuts_key.is_empty() {
            String::new()
        } else {
            format!("| `{shortcuts_key}` | Show quick shortcuts |\n")
        };

        format!(
            "\
**Navigation**
| Key | Action |
|-----|--------|
| `{cursor_up}` / `{cursor_down}` / `{cursor_left}` / `{cursor_right}` | Move cursor / browse history (Up when empty) |
| `{cursor_word_left}` / `{cursor_word_right}` | Move by word |
| `{cursor_line_start}` | Start of line |
| `{cursor_line_end}` | End of line |
| `{jump_forward}` | Jump forward to character |
| `{jump_backward}` | Jump backward to character |
| `{page_up}` / `{page_down}` | Scroll by page |

**Editing**
| Key | Action |
|-----|--------|
| `{submit}` | Send message |
| `{new_line}` | New line{new_line_note} |
| `{delete_word_backward}` | Delete word backwards |
| `{delete_word_forward}` | Delete word forwards |
| `{delete_to_line_start}` | Delete to start of line |
| `{delete_to_line_end}` | Delete to end of line |
| `{yank}` | Paste the most-recently-deleted text |
| `{yank_pop}` | Cycle through the deleted text after pasting |
| `{undo}` | Undo |

**Other**
| Key | Action |
|-----|--------|
| `{tab}` | Path completion / accept autocomplete |
| `{clear_input}` | Clear input / cancel autocomplete |
| `{clear}` | Interrupt current operation (first) / exit (second) |
{interrupt_row}{shortcuts_row}| `{exit}` | Exit (when editor is empty) |
| `{select_model}` | Open model selector |
| `{expand_tools}` | Toggle tool output expansion |
| `{expand_messages}` | Toggle agent message expansion |
| `{expand_edits}` | Toggle edit diff expansion |
| `{toggle_thinking}` | Toggle thinking block visibility |
| `{focus_subagents}` | Focus the subagent summary / open the scoped agents view |
| `{manage_heartbeats}` | Manage heartbeats |
| `{external_editor}` | Edit message in external editor |
| `{prompt_stash}` | Stash or restore draft prompt |
| `{follow_up}` | Queue follow-up message |
| `{browse_queue}` | Browse and edit queued messages |
| `{reorder_queue}` | Reorder the selected queued message |
| `{paste_image}` | Paste image from clipboard |
| `/` | Slash commands |

**Fullscreen mode (`/fullscreen`)**
| Key | Action |
|-----|--------|
| `{viewport_page_up}` / `{viewport_page_down}` | Scroll transcript by page |
| `{viewport_top}` | Scroll to top |
| `{viewport_follow}` | Scroll to bottom and follow output |
| mouse wheel | Scroll transcript |
| mouse drag | Select and copy text |
| mouse click on link | Open link in browser |
"
        )
    }

    /// Port of `showShortcutGuide` (interactive-mode.ts:10197-10204): an
    /// ephemeral spacer + markdown in `shortcutGuideContainer`, never appended to
    /// the chat history.
    pub fn show_shortcut_guide(&mut self) {
        let hotkeys = self.get_shortcut_guide();
        self.shortcut_guide_container.clear();
        self.shortcut_guide_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.shortcut_guide_container.add_child(Box::new(MarkdownBlock::new(
            hotkeys.trim().to_string(),
            1,
            1,
            self.get_markdown_theme_with_settings(),
        )));
        self.ui.request_render();
    }

    /// Port of `handleHotkeysCommand` (interactive-mode.ts:10206-10212): the full
    /// guide is durable chat content, unlike the `?` guide.
    pub fn handle_hotkeys_command(&mut self) {
        let hotkeys = self.get_hotkeys_guide();
        self.chat_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.chat_container.add_child(Box::new(MarkdownBlock::new(
            hotkeys.trim().to_string(),
            1,
            1,
            self.get_markdown_theme_with_settings(),
        )));
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
    }

    /// Port of `clearShortcutGuide` (interactive-mode.ts:10214-10220).
    pub fn clear_shortcut_guide(&mut self) {
        if self.shortcut_guide_container.is_empty() {
            return;
        }
        self.shortcut_guide_container.clear();
        self.ui.request_render();
    }

    /// Port of `getQueueSelectionHeader`.
    pub fn get_queue_selection_header(&self) -> Option<String> {
        let selected = self.queue_selection.selected()?;
        let lane = if selected.lane == super::queue_selection::QueueLane::Steering { "steering" } else { "follow-up" };
        let older = self.capitalize_key(&self.key_text("app.message.navigateOlder"));
        let newer = self.capitalize_key(&self.key_text("app.message.navigateNewer"));
        let earlier = self.capitalize_key(&self.key_text("app.message.moveEarlier"));
        let later = self.capitalize_key(&self.key_text("app.message.moveLater"));
        let queue = self.capitalize_key(&self.key_text("app.message.followUp"));
        Some(theme().fg(
            "dim",
            &format!(
                "{lane} {} \u{b7} {older}/{newer} browse \u{b7} {earlier}/{later} reorder \u{b7} enter steers \u{b7} {queue} queues \u{b7} empty deletes",
                selected.index + 1
            ),
        ))
    }

    /// Port of `getWorkingLoaderMessage`.
    pub fn get_working_loader_message(&self) -> String {
        let elapsed = self
            .working_started_at
            .map(|started_at| self.format_working_elapsed(now_ms() - started_at));
        let status = self.activity_tracker.get_status();
        // The subagent count/recaps live in the tree above the loader, so the loader
        // message itself no longer repeats "N subagents running".
        if !self.is_agent_streaming() {
            return String::new();
        }
        if let Some(working_message) = &self.working_message {
            // Extensions and tool bootstrap own the message; keep the plain "<message> <elapsed>" form.
            return match elapsed {
                Some(elapsed) => format!("{working_message} {elapsed}"),
                None => working_message.clone(),
            };
        }
        let mut parts: Vec<String> = vec![super::agent_activity::agent_activity_label(status.activity).to_string()];
        if let Some(elapsed) = elapsed {
            parts.push(elapsed);
        }
        if status.tokens > 0.0 {
            parts.push(format!(
                "{} {} tokens",
                if status.direction == super::agent_activity::Direction::Down { "\u{2193}" } else { "\u{2191}" },
                format_token_count(status.tokens)
            ));
        }
        parts.join(" \u{b7} ")
    }

    /// Port of `shouldShowWorkingLoader`.
    pub fn should_show_working_loader(&self) -> bool {
        // Background subagents (agent turn done, asyncio tasks still running) would
        // otherwise show a textless spinner; the subagent tree above the loader carries
        // that state, so the loader only shows while the main agent is itself streaming.
        self.working_visible && self.is_agent_streaming()
    }

    /// Port of `shouldSuppressFeatureHint`.
    pub fn should_suppress_feature_hint(&self) -> bool {
        let queue = self.get_all_queued_messages();
        !queue.steering.is_empty() || !queue.follow_up.is_empty()
    }

    /// Port of `isModelProviderConfigured`.
    pub fn is_model_provider_configured(&self, model: &AgentConnectionModel) -> bool {
        self.connection_configured_providers.contains(&model.provider)
            || self
                .model_registry()
                .lock()
                .expect("model registry poisoned")
                .has_configured_auth(model)
    }

    /// Port of `currentModelSupportsFastMode`.
    pub fn current_model_supports_fast_mode(&self) -> bool {
        self.get_current_model().map(pi_ai::models::supports_fast_mode).unwrap_or(false)
    }

    /// Port of `getAvailableThinkingLevels`.
    pub fn get_available_thinking_levels(&self) -> Vec<ThinkingLevel> {
        let levels = self
            .connection_state
            .as_ref()
            .map(|state| state.available_thinking_levels.clone())
            .unwrap_or_default();
        let supports_thinking =
            !levels.is_empty() && !(levels.len() == 1 && levels[0] == ThinkingLevel::Off);
        if supports_thinking {
            levels
        } else {
            Vec::new()
        }
    }

    /// Port of `getThinkingLevelCompletions`.
    pub fn get_thinking_level_completions(&self, prefix: &str) -> Option<Vec<AutocompleteItem>> {
        let levels = self.get_available_thinking_levels();
        if levels.is_empty() {
            return None;
        }
        let current = self.connection_state.as_ref().map(|state| state.thinking_level.clone());
        let term = prefix.trim().to_lowercase();
        let matches: Vec<ThinkingLevel> = if term.is_empty() {
            levels
        } else {
            levels.into_iter().filter(|level| level.as_str().starts_with(&term)).collect()
        };
        if matches.is_empty() {
            return None;
        }
        let descriptions = thinking_level_descriptions();
        Some(
            matches
                .into_iter()
                .map(|level| {
                    let description = descriptions.get(&level).copied().unwrap_or("");
                    AutocompleteItem {
                        value: level.as_str().to_string(),
                        label: level.as_str().to_string(),
                        description: Some(if Some(&level) == current.as_ref() {
                            format!("{description} (current)")
                        } else {
                            description.to_string()
                        }),
                    }
                })
                .collect(),
        )
    }

    /// Port of `getHeartbeatArgumentCompletions`.
    pub fn get_heartbeat_argument_completions(&self, prefix: &str) -> Option<Vec<AutocompleteItem>> {
        let term = prefix.trim().to_lowercase();
        let all = heartbeat_argument_completions();
        let filtered: Vec<(String, String)> = if term.is_empty() {
            all
        } else {
            all.into_iter()
                .filter(|(value, label)| {
                    value.to_lowercase().starts_with(&term) || label.to_lowercase().starts_with(&term)
                })
                .collect()
        };
        if filtered.is_empty() {
            return None;
        }
        Some(
            filtered
                .into_iter()
                .map(|(value, label)| AutocompleteItem { value, label, description: None })
                .collect(),
        )
    }

    /// Port of `formatGoalStatus`.
    ///
    pub fn format_goal_status(&self, goal: &GoalState, terminal_columns: f64) -> String {
        let usage = crate::core::goals::format_goal_usage(goal);
        let usage_text = usage.map(|usage| format!(" ({usage})")).unwrap_or_default();
        match goal.status {
            GoalStatus::Idle => "No active goal".to_string(),
            GoalStatus::Active => match &goal.objective {
                Some(objective) => format!(
                    "Goal{}",
                    self.format_goal_detail_suffix(Some(objective), visible_width("Goal"), terminal_columns)
                ),
                None => "Pursuing goal".to_string(),
            },
            GoalStatus::Paused => match &goal.last_reason {
                Some(last_reason) => format!(
                    "Goal paused{}",
                    self.format_goal_detail_suffix(
                        Some(last_reason),
                        visible_width("Goal paused"),
                        terminal_columns
                    )
                ),
                None => "Goal paused (/goal resume)".to_string(),
            },
            GoalStatus::BudgetLimited => match &goal.last_reason {
                Some(last_reason) => {
                    let prefix = format!("Goal budget limited{usage_text}");
                    let suffix = self.format_goal_detail_suffix(
                        Some(last_reason),
                        visible_width(&prefix),
                        terminal_columns,
                    );
                    format!("{prefix}{suffix}")
                }
                None => format!("Goal budget limited{usage_text}"),
            },
            GoalStatus::Complete => match &goal.last_reason {
                Some(last_reason) => format!(
                    "Goal complete{}",
                    self.format_goal_detail_suffix(
                        Some(last_reason),
                        visible_width("Goal complete"),
                        terminal_columns
                    )
                ),
                None => "Goal complete".to_string(),
            },
            GoalStatus::Error => match &goal.last_error {
                Some(last_error) => format!(
                    "Goal error{}",
                    self.format_goal_detail_suffix(
                        Some(last_error),
                        visible_width("Goal error"),
                        terminal_columns
                    )
                ),
                None => "Goal error".to_string(),
            },
        }
    }

    /// Port of `formatGoalDetailSuffix`.
    pub fn format_goal_detail_suffix(
        &self,
        value: Option<&str>,
        prefix_width: f64,
        terminal_columns: f64,
    ) -> String {
        let detail = value.map(|value| value.split_whitespace().collect::<Vec<&str>>().join(" "));
        let Some(detail) = detail else {
            return String::new();
        };
        if detail.is_empty() {
            return String::new();
        }
        let available_width = 120.0f64.min(1.0f64.max(terminal_columns - prefix_width - 2.0));
        if available_width < 8.0 {
            return String::new();
        }
        format!(": {}", truncate_to_width(&detail, available_width, "…", false))
    }

    /// Port of `getPathCommandArgument`.
    pub fn get_path_command_argument(&self, text: &str, command: &str) -> Option<String> {
        path_command_argument(text, command)
    }

    /// Port of `capitalizeKey`.
    pub fn capitalize_key(&self, key: &str) -> String {
        key.split('/')
            .map(|k| {
                k.split('+')
                    .map(|part| {
                        if part == "esc" {
                            part.to_string()
                        } else {
                            let mut chars = part.chars();
                            match chars.next() {
                                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                                None => String::new(),
                            }
                        }
                    })
                    .collect::<Vec<String>>()
                    .join("+")
            })
            .collect::<Vec<String>>()
            .join("/")
    }

    /// Port of `getAppKeyDisplay`.
    pub fn get_app_key_display(&self, action: &str) -> String {
        self.capitalize_key(&self.key_text(action))
    }

    /// Port of `getEditorKeyDisplay`.
    pub fn get_editor_key_display(&self, action: &str) -> String {
        self.capitalize_key(&self.key_text(action))
    }

    /// Stand-in for `keyText(action)` (components/keybinding-hints.ts, other slice).
    fn key_text(&self, action: &str) -> String {
        let keys = pi_tui::keybindings::get_keybindings().get_keys(action);
        format_key_text(&keys, std::env::consts::OS)
    }

    /// Stand-in for `keyHint(action, description)`.
    fn key_hint(&self, action: &str, description: &str) -> String {
        format!("{}{}", theme().fg("dim", &self.key_text(action)), theme().fg("muted", &format!(" {description}")))
    }

    /// Port of `getUserMessageText`.
    pub fn get_user_message_text(&self, message: &pi_ai::types::UserMessage) -> String {
        match &message.content {
            pi_ai::types::UserContent::Text(text) => text.clone(),
            pi_ai::types::UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<String>>()
                .join(""),
        }
    }

    /// Port of `isTextOnlyUserMessage`.
    pub fn is_text_only_user_message(&self, message: &pi_ai::types::UserMessage) -> bool {
        match &message.content {
            pi_ai::types::UserContent::Text(_) => true,
            pi_ai::types::UserContent::Blocks(blocks) => blocks
                .iter()
                .all(|block| matches!(block, pi_ai::types::ImageOrTextContent::Text(_))),
        }
    }

    /// Port of `getMarkdownThemeWithSettings`.
    pub fn get_markdown_theme_with_settings(&self) -> super::theme::theme::MarkdownTheme {
        let mut markdown_theme = super::theme::theme::get_markdown_theme();
        markdown_theme.code_block_indent = Some(self.with_settings(|settings| settings.get_code_block_indent()));
        markdown_theme
    }

    /// Port of `getCachedModelCandidates`.
    pub fn get_cached_model_candidates(&self) -> Vec<AgentConnectionModel> {
        let mut models_by_id: indexmap::IndexMap<String, AgentConnectionModel> = indexmap::IndexMap::new();
        for scoped in self.get_scoped_model_state() {
            models_by_id.insert(format!("{}/{}", scoped.model.provider, scoped.model.id), scoped.model);
        }
        for model in &self.connection_model_catalog {
            models_by_id.insert(format!("{}/{}", model.provider, model.id), model.clone());
        }
        models_by_id.into_values().collect()
    }

    /// Port of `invalidateConnectionModels`.
    pub fn invalidate_connection_models(&mut self) {
        self.connection_configured_providers = HashSet::new();
        self.connection_models_fetched_at = 0.0;
        self.invalidate_connection_model_refresh();
    }

    /// Port of `invalidateConnectionModelRefresh`.
    pub fn invalidate_connection_model_refresh(&mut self) {
        self.connection_models_refresh_version += 1;
        self.connection_models_refresh_in_flight = false;
    }

    /// Port of `applyConnectionStateSnapshot`.
    pub fn apply_connection_state_snapshot(&mut self, state: AgentConnectionState) {
        if self.connection_state.as_ref().is_some_and(|previous| previous.session_id != state.session_id) {
            self.restored_draft_notice.borrow_mut().take();
            self.clear_compaction_notices();
        }
        self.bind_prompt_stash_session(&state.session_id);
        self.connection_state = Some(state.clone());
        if let Some(warning) = self.options.model_fallback_message.clone() {
            if self.get_model_fallback_warning_action(Some(&warning)) == ModelFallbackWarningAction::Suppress {
                let rendered = theme().fg("warning", &format!("\u{26a0} {warning}"));
                let before = self.chat_container.len();
                self.chat_container.children.retain(|child| child.as_any().downcast_ref::<Text>().is_none_or(|text| text.text() != rendered));
                if before != self.chat_container.len() {
                    self.last_status_text_index = None;
                    self.last_status_spacer_index = None;
                }
                self.options.model_fallback_message = None;
            }
        }
        // Don't touch contextUsageTokenBaseline: a mid-stream snapshot reflects only completed
        // turns (the in-flight message isn't persisted yet), so the in-flight delta must keep
        // accumulating. The baseline is managed at turn end (refreshConnectionContextUsage) and
        // reset on a new user message.
        self.session_recap = state.recap.clone();
        self.update_pending_messages_display();
        self.update_working_pulse();
    }

    /// Port of `patchConnectionState`.
    pub fn patch_connection_state(&mut self, patch: impl FnOnce(&mut AgentConnectionState)) {
        let Some(state) = self.connection_state.as_mut() else {
            return;
        };
        patch(state);
        self.update_working_pulse();
    }

    /// Apply an authoritative queue change and refresh its bounded previews.
    pub fn patch_connection_queue(&mut self, patch: impl FnOnce(&mut AgentConnectionState)) {
        self.patch_connection_state(patch);
        self.update_pending_messages_display();
    }

    /// Render authoritative accepted queue state without resetting the editor.
    fn update_pending_messages_display(&mut self) {
        let queue = self.get_connection_queue();
        self.queued_messages_container.clear();
        let total = queue.steering.len() + queue.follow_up.len();
        for (text, label) in queue.steering.iter().map(|text| (text, QueueLabel::Steering))
            .chain(queue.follow_up.iter().map(|text| (text, QueueLabel::FollowUp))).take(4)
        {
            let bounded: String = text.split_whitespace().flat_map(|word| word.chars().chain(std::iter::once(' '))).take(240).collect();
            self.queued_messages_container.add_child(Box::new(
                super::interactive_mode_services::TruncatedText::new(
                    theme().fg("dim", &format_queued_message_preview(&bounded, label)), 1, 0,
                ),
            ));
        }
        if total > 4 {
            self.queued_messages_container.add_child(Box::new(Text::new(
                theme().fg("dim", &format!("{} more queued messages", total - 4)), 1, 0,
            )));
        }
        self.ui.request_render();
    }

    /// Port of `setGoalAnnouncementBaseline`.
    pub fn set_goal_announcement_baseline(&mut self, goal: &GoalState) {
        self.last_goal_announcement = Some(self.goal_announcement_snapshot(goal));
    }

    /// Port of `goalAnnouncementSnapshot`.
    pub fn goal_announcement_snapshot(&self, goal: &GoalState) -> GoalAnnouncementSnapshot {
        GoalAnnouncementSnapshot {
            goal_id: goal.goal_id.clone(),
            status: goal.status.clone(),
            objective: goal.objective.clone(),
            last_reason: goal.last_reason.clone(),
            last_error: goal.last_error.clone(),
        }
    }

    /// Port of `shouldAnnounceGoalUpdate`.
    pub fn should_announce_goal_update(&mut self, goal: &GoalState) -> bool {
        let previous = self.last_goal_announcement.clone();
        let next = self.goal_announcement_snapshot(goal);
        self.last_goal_announcement = Some(next.clone());
        let Some(previous) = previous else {
            return goal.status != GoalStatus::Idle;
        };
        if previous.status != next.status {
            return true;
        }
        if previous.goal_id != next.goal_id {
            return goal.status != GoalStatus::Idle;
        }
        match goal.status {
            GoalStatus::Active => false,
            GoalStatus::Paused | GoalStatus::BudgetLimited | GoalStatus::Complete => {
                previous.last_reason != next.last_reason
            }
            GoalStatus::Error => previous.last_error != next.last_error,
            GoalStatus::Idle => false,
            _ => false,
        }
    }

    /// Port of `seedSubagentSummary`.
    pub fn seed_subagent_summary(&mut self, children: Option<&[AgentConnectionRlmChildAgentSnapshot]>) {
        for child in children.unwrap_or(&[]) {
            // Live updates can arrive before the initial snapshot; do not replace them
            // with the snapshot's older state.
            if !self.subagent_snapshots.contains_key(&child.id) && child.status != "cancelled" {
                self.subagent_snapshots.insert(child.id.clone(), child.clone());
            }
        }
        self.refresh_subagent_summary();
    }

    /// Port of `replaceSubagentSummary`.
    pub fn replace_subagent_summary(&mut self, children: Option<&[AgentConnectionRlmChildAgentSnapshot]>) {
        let mut next: HashMap<String, AgentConnectionRlmChildAgentSnapshot> = HashMap::new();
        for child in children.unwrap_or(&[]) {
            if child.status == "cancelled" {
                continue;
            }
            let previous = self.subagent_snapshots.get(&child.id);
            next.insert(
                child.id.clone(),
                match previous {
                    Some(previous) => merge_subagent_snapshot(previous, child),
                    None => child.clone(),
                },
            );
        }
        self.subagent_snapshots = next;
        self.refresh_subagent_summary();
    }

    /// Port of `updateSubagentSummary`.
    pub fn update_subagent_summary(&mut self, child: AgentConnectionRlmChildAgentSnapshot) {
        // "cancelled" also covers never-bound terminal runs; AgentSession owns that rule.
        if child.status == "cancelled" {
            self.remove_subagent_snapshot(&child.id);
        } else {
            let previous = self.subagent_snapshots.get(&child.id).cloned();
            self.subagent_snapshots.insert(
                child.id.clone(),
                match previous {
                    Some(previous) => merge_subagent_snapshot(&previous, &child),
                    None => child,
                },
            );
        }
        self.refresh_subagent_summary();
    }

    /// Port of `refreshSubagentSummary`.
    pub fn refresh_subagent_summary(&mut self) {
        self.update_subagent_summary_line();
        self.update_working_pulse();
        self.sync_working_loader();
        self.ui.request_render();
    }

    /// Port of `updateSubagentSummaryLine`.
    pub fn update_subagent_summary_line(&mut self) {
        let roster_summaries = self.roster_bar.as_ref().map(|roster_bar| roster_bar.summaries());
        // A client-owned session has no row on the public roster; only then do the
        // snapshots carry the bar. A public parent with zero roster children shows zero.
        let session_on_roster = roster_summaries
            .as_ref()
            .map(|rows| {
                rows.iter().any(|row| {
                    Some(row.session_id.as_str())
                        == self.connection_state.as_ref().map(|state| state.session_id.as_str())
                })
            })
            .unwrap_or(false);
        let counts = match (&roster_summaries, session_on_roster) {
            (Some(rows), true) => {
                let active = self
                    .connection_state
                    .as_ref()
                    .and_then(|state| state.active_session_id.clone())
                    .unwrap_or_default();
                let session_id = self
                    .connection_state
                    .as_ref()
                    .map(|state| state.session_id.clone())
                    .unwrap_or_default();
                let session_file = self.connection_state.as_ref().and_then(|state| state.session_file.clone());
                count_roster_subagent_statuses(
                    rows,
                    (Some(&active), Some(&session_id), session_file.as_deref()),
                )
            }
            _ => {
                let children: Vec<AgentConnectionRlmChildAgentSnapshot> =
                    self.subagent_snapshots.values().cloned().collect();
                count_direct_subagent_statuses(&children, self.rlm_node_id.as_deref())
            }
        };
        self.subagent_summary_line.set_subagent_counts(counts);
    }

    /// Port of `removeSubagentSnapshot`.
    pub fn remove_subagent_snapshot(&mut self, id: &str) {
        self.subagent_snapshots.remove(id);
        let children: Vec<AgentConnectionRlmChildAgentSnapshot> = self.subagent_snapshots.values().cloned().collect();
        for child in children {
            if child.parent_id.as_deref() == Some(id) {
                self.remove_subagent_snapshot(&child.id);
            }
        }
    }

    /// Port of `resetSubagentSummary`.
    pub fn reset_subagent_summary(&mut self) {
        self.subagent_snapshots.clear();
        self.rlm_node_id = None;
        self.update_subagent_summary_line();
        // Clearing snapshots can drop the last running subagent; reconcile the
        // pulse and loader so neither lingers when nothing is in flight.
        self.update_working_pulse();
        self.sync_working_loader();
    }

    /// Port of `updateWorkingPulse`.
    pub fn update_working_pulse(&mut self) {
        let active = self.is_agent_streaming();
        if !active {
            self.stop_working_pulse();
            return;
        }
        if !self.working_pulse_active {
            self.working_pulse_active = true;
        }
    }

    /// Port of `tickWorkingPulse`.
    pub fn tick_working_pulse(&mut self) {
        self.pulse_frame += 1;
        set_working_pulse_frame(self.pulse_frame);
        self.ui.request_render();
    }

    /// Port of the `showCtrlCExitHint` timeout callback
    /// (interactive-mode.ts:7018-7025). The Rust mode has no `setTimeout`; the
    /// host drives this from its existing tick so the exit hint auto-expires and
    /// the tray repaints without a stale "Press Ctrl+C again to exit".
    pub fn expire_ctrl_c_exit_hint(&mut self) {
        if self.ctrl_c_exit_hint_expires_at == 0.0 || self.is_ctrl_c_exit_hint_visible() {
            return;
        }
        self.ctrl_c_exit_hint_expires_at = 0.0;
        self.ui.request_render();
    }

    /// Port of `stopWorkingPulse`.
    pub fn stop_working_pulse(&mut self) {
        self.working_pulse_active = false;
    }

    /// Port of `startCompactionLoader` (interactive-mode.ts:3528-3555).
    pub fn start_compaction_loader(&mut self, reason: &str, custom_instructions: Option<&str>) {
        // Keep the editor active; submissions are queued during compaction. Fully
        // stop the working loader (not just detach) so it is not orphaned.
        self.stop_working_loader();
        self.compaction_loader_label = Some(self.format_compaction_loader_label(
            reason,
            custom_instructions,
        ));
        self.sync_working_loader();
    }

    /// The `startCompactionLoader` label ladder (interactive-mode.ts:3539-3546).
    fn format_compaction_loader_label(
        &self,
        reason: &str,
        custom_instructions: Option<&str>,
    ) -> String {
        let cancel_hint = format!("({} to cancel)", self.key_text("app.clear"));
        let focus = custom_instructions
            .filter(|instructions| !instructions.is_empty())
            .map(|instructions| {
                format!(
                    " (focus: {})",
                    truncate_to_width(instructions, 60.0, "\u{2026}", false)
                )
            })
            .unwrap_or_default();
        match reason {
            "manual" => format!("Compacting context{focus}... {cancel_hint}"),
            "requested" => {
                format!("Agent requested compaction, compacting context{focus}... {cancel_hint}")
            }
            _ => format!(
                "{}Auto-compacting... {cancel_hint}",
                if reason == "overflow" { "Context overflow detected, " } else { "" }
            ),
        }
    }

    /// Port of `syncWorkingLoader` (interactive-mode.ts:3586-3616).
    pub fn sync_working_loader(&mut self) {
        // A compaction that started before this client attached (or while another
        // view was open) has no start-event edge; restore its loader from state.
        if self.compaction_loader_label.is_none() && self.is_agent_compacting() {
            self.start_compaction_loader("manual", None);
            return;
        }
        self.status_container.clear();
        if let Some(label) = &self.compaction_loader_label {
            self.status_container.add_child(Box::new(
                super::interactive_mode_services::Text::new(theme().fg("muted", label), 1, 0),
            ));
        }
        self.ui.request_render();
    }

    /// Port of the `compaction_end` loader teardown
    /// (interactive-mode.ts:5812-5814).
    pub fn stop_compaction_loader(&mut self) {
        if self.compaction_loader_label.is_none() {
            return;
        }
        self.compaction_loader_label = None;
        self.sync_working_loader();
    }

    /// Port of `setWorkingVisible`.
    pub fn set_working_visible(&mut self, visible: bool) {
        self.working_visible = visible;
        if !visible {
            self.stop_working_loader();
            self.ui.request_render();
            return;
        }
        if self.should_show_working_loader() {
            self.status_container.clear();
        }
        self.ui.request_render();
    }

    /// Port of `stopWorkingLoader`.
    pub fn stop_working_loader(&mut self) {
        self.working_started_at = None;
        self.status_container.clear();
    }

    /// Port of `setWorkingIndicator`.
    pub fn set_working_indicator(&mut self, options: Option<LoaderIndicatorOptions>) {
        self.working_indicator_options = options;
        self.ui.request_render();
    }

    /// Port of `setHiddenThinkingLabel`.
    pub fn set_hidden_thinking_label(&mut self, label: Option<String>) {
        self.hidden_thinking_label = label.unwrap_or_else(|| self.default_hidden_thinking_label.clone());
        self.ui.request_render();
    }

    /// Port of `setToolsExpanded`.
    pub fn set_tools_expanded(&mut self, expanded: bool) {
        self.tool_output_expanded = expanded;
        self.apply_chat_expansion();
    }

    /// Port of `toggleToolOutputExpansion`.
    pub fn toggle_tool_output_expansion(&mut self) {
        self.set_tools_expanded(!self.tool_output_expanded);
    }

    /// Port of `toggleAgentMessageExpansion`.
    pub fn toggle_agent_message_expansion(&mut self) {
        self.agent_messages_expanded = !self.agent_messages_expanded;
        self.apply_chat_expansion();
    }

    /// Port of `toggleEditDiffExpansion`.
    pub fn toggle_edit_diff_expansion(&mut self) {
        self.edit_diffs_expanded = !self.edit_diffs_expanded;
        self.apply_chat_expansion();
    }

    /// Port of `applyChatExpansion`.
    pub fn apply_chat_expansion(&mut self) {
        // Expanding/collapsing changes blocks above the viewport, which would
        // otherwise force a full redraw that scrolls to the top and replays the
        // whole transcript. Keep the user anchored at their current position.
        // Fullscreen frames have no scrollback to preserve.
        if self.ui.is_fullscreen() {
            self.ui.request_render();
        } else {
            self.ui.request_render_preserving_viewport();
        }
    }

    /// Port of `startFeatureHintPresentation`.
    pub fn start_feature_hint_presentation(&mut self) {
        self.clear_feature_hint_presentation();
        if self.should_suppress_feature_hint() {
            return;
        }
        if self.feature_hint_eligible_at == 0.0 {
            self.feature_hint_eligible_at = now_ms() + FEATURE_HINT_DELAY_MS;
        }
        let delay = 0.0f64.max(self.feature_hint_eligible_at - now_ms());
        if delay == 0.0 {
            self.show_feature_hint();
        }
    }

    /// Port of `clearFeatureHintPresentation`.
    pub fn clear_feature_hint_presentation(&mut self) {
        self.feature_hint_container.clear();
        self.current_feature_hint = None;
    }

    /// Port of `endFeatureHintRun`.
    pub fn end_feature_hint_run(&mut self) {
        self.clear_feature_hint_presentation();
        self.current_feature_hint = None;
        self.feature_hint_eligible_at = 0.0;
        self.feature_hint_run_pending = false;
    }

    /// Port of `prepareFeatureHintRun`.
    pub fn prepare_feature_hint_run(&mut self, message: &AgentMessage) {
        if !self.feature_hint_run_pending {
            return;
        }
        if message.role() == "assistant" {
            self.feature_hint_run_pending = false;
            return;
        }
        if !starts_agent_run(message) {
            return;
        }

        self.end_feature_hint_run();
        if self.should_show_working_loader() {
            self.start_feature_hint_presentation();
        }
    }

    /// Port of `showFeatureHint`.
    pub fn show_feature_hint(&mut self) {
        if self.should_suppress_feature_hint() || !self.should_show_working_loader() {
            return;
        }
        if self.current_feature_hint.is_none() {
            let get_keybinding = |action: &str| {
                let key = pi_tui::keybindings::get_keybindings().get_keys(action);
                let key = format_key_text(&key, std::env::consts::OS);
                if key.is_empty() {
                    None
                } else {
                    Some(key)
                }
            };
            let context = super::feature_hints::FeatureHintContext {
                get_keybinding: Box::new(get_keybinding),
                is_resident_session: self.options.return_to_agents_view,
            };
            let hint = self.feature_hint_deck.next(&context);
            self.current_feature_hint = hint.map(|hint| hint.text);
        }
        if self.current_feature_hint.is_none() {
            return;
        }
        self.ui.request_render();
    }

    /// Port of `resumeFeatureHintPresentation`.
    pub fn resume_feature_hint_presentation(&mut self) {
        if !self.should_suppress_feature_hint() && self.should_show_working_loader() {
            self.start_feature_hint_presentation();
        }
    }

    /// Port of `expansionStateFor`.
    pub fn expansion_state_for(&self, is_agent_message_component: bool) -> bool {
        if is_agent_message_component {
            self.agent_messages_expanded
        } else {
            self.tool_output_expanded
        }
    }

    /// Port of `showStatus`.
    ///
    /// Back-to-back status messages update the previous status line instead of
    /// appending new ones to avoid log spam (interactive-mode.ts:6372-6390).
    pub fn show_status(&mut self, message: &str, tone: ThemeColor) {
        let children = self.chat_container.len();
        let last_is_previous_status = matches!(
            (self.last_status_spacer_index, self.last_status_text_index),
            (Some(spacer), Some(text)) if spacer + 2 == children && text + 1 == children
        );
        if last_is_previous_status {
            if let Some(index) = self.last_status_text_index {
                self.chat_container.children[index] =
                    Box::new(Text::new(theme().fg(tone, message), 1, 0));
            }
            self.ui.request_render();
            return;
        }
        self.chat_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.chat_container
            .add_child(Box::new(Text::new(theme().fg(tone, message), 1, 0)));
        self.last_status_spacer_index = Some(self.chat_container.len() - 2);
        self.last_status_text_index = Some(self.chat_container.len() - 1);
        self.ui.request_render();
    }

    /// Port of `resetPagedHistory`.
    pub fn reset_paged_history(&mut self) {
        self.paged_history_generation_bump();
        self.history_container.clear();
    }

    fn paged_history_generation_bump(&mut self) {}

    /// Port of `run`.
    ///
    /// PARTIAL: the startup-prompt admission barrier and the input loop depend on
    /// `AgentConnection` (agent-connection slice). The Rust port returns the run
    /// result shape with the same defaults; see blocked_on.
    pub async fn run(&mut self) -> InteractiveModeRunResult {
        let state = self.connection_state.clone();
        InteractiveModeRunResult {
            type_: self
                .agents_view_request
                .clone()
                .unwrap_or(InteractiveModeRunResultType::AgentsView),
            source: InteractiveModeRunResultSource {
                active_session_id: state.as_ref().and_then(|state| state.active_session_id.clone()),
                session_file: state.as_ref().and_then(|state| state.session_file.clone()),
                session_id: state
                    .as_ref()
                    .map(|state| state.session_id.clone())
                    .or_else(|| self.prompt_stash_session_id.clone())
                    .unwrap_or_default(),
                session_name: state.as_ref().and_then(|state| state.session_name.clone()),
                cwd: state.as_ref().map(|state| state.cwd.clone()).unwrap_or_else(|| self.get_current_cwd()),
            },
        }
    }

    /// Port of `init` (TUI assembly; tool bootstrap and startup rendering live in
    /// other slices, so the port keeps the container wiring and the theme setup).
    pub async fn init(&mut self) -> Result<(), String> {
        if self.is_initialized {
            return Ok(());
        }

        self.header_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.main_container.add_child(Box::new(super::interactive_mode_services::Container::new()));
        self.ui.add_child(Box::new(Text::new("", 0, 0)));
        self.ui.start();
        self.fullscreen_enabled =
            self.options.force_fullscreen || self.with_settings(|settings| settings.get_fullscreen());
        self.is_initialized = true;
        Ok(())
    }

    /// Port of `checkShutdownRequested`.
    pub async fn check_shutdown_requested(&mut self) {
        if !self.shutdown_requested {
            return;
        }
        self.shutdown().await;
    }

    /// Port of `shutdown` (TUI teardown; connection disposal is owned by the
    /// agent-connection slice).
    pub async fn shutdown(&mut self) {
        self.stop_working_pulse();
        self.ui.stop(false, None);
        if let Some(on_shutdown) = self.options.on_shutdown.as_mut() {
            on_shutdown();
        }
    }

    /// Port of `teardownSessionUi`.
    pub fn teardown_session_ui(&mut self) {
        self.release_prompt_stash_session();
        self.reset_paged_history();
        self.reset_subagent_summary();
    }
}

/// `AutocompleteItem` (pi-tui, other slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutocompleteItem {
    pub value: String,
    pub label: String,
    pub description: Option<String>,
}

/// Stand-in for `Math.random`.
fn default_random() -> f64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    (hasher.finish() % 1_000_000) as f64 / 1_000_000.0
}

/// `Date.now()`
fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

fn pad_start(value: &str, width: usize, pad: char) -> String {
    if value.len() >= width {
        value.to_string()
    } else {
        format!("{}{value}", pad.to_string().repeat(width - value.len()))
    }
}

fn basename(path: &str) -> String {
    path.replace('\\', "/").split('/').next_back().unwrap_or("").to_string()
}

/// Terminal/tab title for the interactive chat view: app identity, the current
/// session name when known, then the working directory basename.
fn terminal_title_text(app_title: &str, session_name: Option<&str>, cwd_basename: &str) -> String {
    match session_name {
        Some(session_name) => format!("{app_title} - {session_name} - {cwd_basename}"),
        None => format!("{app_title} - {cwd_basename}"),
    }
}



fn resolve_path(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string())
}

fn relative_path_between(from: &str, to: &str) -> String {
    let from = Path::new(from);
    let to = Path::new(to);
    match pathdiff(from, to) {
        Some(relative) => relative.to_string_lossy().to_string(),
        None => to.to_string_lossy().to_string(),
    }
}

/// Minimal `path.relative` stand-in.
fn pathdiff(from: &Path, to: &Path) -> Option<std::path::PathBuf> {
    let from_components: Vec<_> = from.components().collect();
    let to_components: Vec<_> = to.components().collect();
    let mut common = 0usize;
    while common < from_components.len().min(to_components.len())
        && from_components[common] == to_components[common]
    {
        common += 1;
    }
    let mut result = std::path::PathBuf::new();
    for _ in common..from_components.len() {
        result.push("..");
    }
    for component in &to_components[common..] {
        result.push(component);
    }
    Some(result)
}

/// `/node_modules\/(@?[^/]+(?:\/[^/]+)?)\/(.*)/`
fn regex_match_node_modules(full_path: &str) -> Option<String> {
    let normalized = full_path.replace('\\', "/");
    let index = normalized.find("/node_modules/")?;
    let rest = &normalized[index + "/node_modules/".len()..];
    let (first, remainder) = match rest.split_once('/') {
        Some((first, remainder)) => (first, remainder),
        None => return None,
    };
    if first.starts_with('@') {
        let (second, remainder) = remainder.split_once('/')?;
        let _ = second;
        return Some(remainder.to_string());
    }
    Some(remainder.to_string())
}

/// `/git\/[^/]+\/[^/]+\/(.*)/`
fn regex_match_git_path(full_path: &str) -> Option<String> {
    let normalized = full_path.replace('\\', "/");
    let index = normalized.find("/git/")?;
    let rest = &normalized[index + "/git/".len()..];
    let mut parts = rest.splitn(3, '/');
    parts.next()?;
    parts.next()?;
    parts.next().map(|value| value.to_string())
}

/// Minimal `path.posix.parse`.
fn posix_parse(path: &str) -> (String, String) {
    match path.rfind('/') {
        Some(index) => (path[..index].to_string(), path[index + 1..].to_string()),
        None => (String::new(), path.to_string()),
    }
}

/// `formatKeyText` (components/keybinding-hints.ts, other slice).
fn format_key_text(keys: &[String], platform: &str) -> String {
    if keys.is_empty() {
        return String::new();
    }
    let joined = keys.join("/");
    joined
        .split('/')
        .map(|binding| {
            binding
                .split('+')
                .map(|part| {
                    let normalized = if part == "escape" { "esc" } else { part };
                    let arrow = match normalized {
                        "up" => Some("\u{2191}"),
                        "down" => Some("\u{2193}"),
                        "left" => Some("\u{2190}"),
                        "right" => Some("\u{2192}"),
                        _ => None,
                    };
                    if let Some(arrow) = arrow {
                        return arrow.to_string();
                    }
                    if platform == "macos" && normalized == "alt" {
                        return "Option".to_string();
                    }
                    let mut chars = normalized.chars();
                    match chars.next() {
                        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                        None => String::new(),
                    }
                })
                .collect::<Vec<String>>()
                .join("+")
        })
        .collect::<Vec<String>>()
        .join("/")
}

/// Stand-in for `startsAgentRun` (core/agent-messages.ts).
fn starts_agent_run(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Message(pi_ai::types::Message::User(_)) => true,
        AgentMessage::Custom(custom) => custom.role() != "assistant",
        _ => false,
    }
}

/// `visibleWidth` re-export.
fn visible_width(text: &str) -> f64 {
    pi_tui::utils::visible_width(text) as f64
}

/// `truncateToWidth` re-export.
fn truncate_to_width(text: &str, max_width: f64, ellipsis: &str, pad: bool) -> String {
    pi_tui::utils::truncate_to_width(text, max_width, ellipsis, pad)
}

/// `getPathCommandArgument` (interactive-mode.ts:9255-9260 region) as a free
/// function, so the native host can resolve `/import <path>` without holding the
/// mode across a spawned task.
pub(crate) fn path_command_argument(text: &str, command: &str) -> Option<String> {
    if text == command {
        return None;
    }
    if !text.starts_with(&format!("{command} ")) {
        return None;
    }

    let args_string = text[command.len() + 1..].trim_start();
    if args_string.is_empty() {
        return None;
    }

    let first_char = args_string.chars().next()?;
    if first_char == '"' || first_char == '\'' {
        let rest = &args_string[1..];
        let closing_quote_index = rest.find(first_char)?;
        return Some(rest[..closing_quote_index].to_string());
    }

    match args_string.find(char::is_whitespace) {
        Some(index) => Some(args_string[..index].to_string()),
        None => Some(args_string.to_string()),
    }
}

fn options_agent_connection(options: &InteractiveModeOptions) -> AgentConnection {
    Arc::clone(&options.agent_connection)
}

fn clone_ui_services(services: &InteractiveModeUiServices) -> InteractiveModeUiServices {
    InteractiveModeUiServices {
        settings_manager: Arc::clone(&services.settings_manager),
        model_registry: Arc::clone(&services.model_registry),
        get_initial_cwd: Box::new(|| String::new()),
        get_initial_session_name: Box::new(|| None),
        get_themes: Box::new(Vec::new),
        refresh_mcp_providers: None,
    }
}

/// `OnboardingSettingsReader` reads `session.settingsManager`, which the client owns as
/// `Arc<Mutex<..>>`; this adapter keeps the onboarding call shape identical while the
/// lock is held only for the read.
impl super::onboarding::OnboardingSettingsReader for Mutex<super::interactive_mode_services::SettingsManager> {
    fn get_onboarding_shown(&self) -> bool {
        self.lock()
            .expect("settings manager poisoned")
            .get_onboarding_shown()
    }
}

/// `OnboardingModelRegistryReader` - same adapter shape for the shared model registry.
impl super::onboarding::OnboardingModelRegistryReader for Mutex<super::interactive_mode_services::ModelRegistry> {
    fn refresh(&self) {
        self.lock().expect("model registry poisoned").refresh();
    }

    fn has_configured_auth(&self, model: &AgentConnectionModel) -> bool {
        self.lock().expect("model registry poisoned").has_configured_auth(model)
    }

    fn get_provider_auth_status(&self, provider: &str) -> super::interactive_mode_services::AuthStatus {
        self.lock()
            .expect("model registry poisoned")
            .get_provider_auth_status(provider)
    }
}

impl InteractiveMode {
    /// Port of `armEscapeRepeat`.
    pub fn arm_escape_repeat(&mut self, action: &'static str) {
        self.clear_escape_repeat();
        self.escape_repeat_action = Some(action);
        self.escape_repeat_expires_at = now_ms() + Self::ESCAPE_REPEAT_WINDOW_MS;
    }

    /// Port of `takeEscapeRepeatAction`.
    pub fn take_escape_repeat_action(&mut self) -> Option<&'static str> {
        let Some(action) = self.escape_repeat_action else {
            self.clear_escape_repeat();
            return None;
        };
        if self.escape_repeat_expires_at <= now_ms() {
            self.clear_escape_repeat();
            return None;
        }
        self.clear_escape_repeat();
        Some(action)
    }

    /// Port of `clearEscapeRepeat`.
    pub fn clear_escape_repeat(&mut self) {
        self.escape_repeat_action = None;
        self.escape_repeat_expires_at = 0.0;
    }

    /// Port of `handleEscape`.
    ///
    /// PARTIAL: `clearSideQuestion` and `showTreeSelector` need the side-question
    /// and tree-selector components from other slices.
    pub fn handle_escape(&mut self, has_side_question: bool) {
        self.clear_ctrl_c_exit_hint(true);
        if has_side_question {
            self.clear_escape_repeat();
            return;
        }
        match self.take_escape_repeat_action() {
            Some("tree") => return,
            Some("clear") => {
                self.clear_input_bar();
                return;
            }
            _ => {}
        }

        self.arm_escape_repeat(if self.has_interruptible_work() { "tree" } else { "clear" });
        self.interrupt_or_clear_input();
    }

    /// Port of `handleCtrlC`.
    pub async fn handle_ctrl_c(&mut self) {
        self.clear_escape_repeat();
        if self.is_ctrl_c_exit_hint_visible() {
            self.shutdown().await;
            return;
        }
        self.handle_interrupt_key();
    }

    /// Port of `handleInterruptKey`.
    pub fn handle_interrupt_key(&mut self) {
        self.clear_escape_repeat();
        self.interrupt_or_clear_input();
        self.show_ctrl_c_exit_hint();
    }

    /// Port of `showCtrlCExitHint`.
    pub fn show_ctrl_c_exit_hint(&mut self) {
        self.ctrl_c_exit_hint_expires_at = now_ms() + Self::EXIT_HINT_DURATION_MS;
        self.ui.request_render();
    }

    /// Port of `clearCtrlCExitHint`.
    pub fn clear_ctrl_c_exit_hint(&mut self, render: bool) {
        if self.ctrl_c_exit_hint_expires_at == 0.0 {
            return;
        }
        self.ctrl_c_exit_hint_expires_at = 0.0;
        if render {
            self.ui.request_render();
        }
    }

    /// Port of `handleCtrlD`.
    pub async fn handle_ctrl_d(&mut self) {
        self.shutdown().await;
    }

    /// Port of `handleCtrlZ` (interactive-mode.ts:7201-7240).
    ///
    /// PARTIAL: the Windows branch is exact (`showStatus` + skip, TS:7202-7205).
    /// The Unix branch needs a SIGTSTP/SIGCONT pair around `ui.stop()`/`ui.start()`;
    /// the mode handle owns only the `Tui` stand-in, so the port reports the
    /// unsupported host instead of stopping the terminal with no restore path.
    pub fn handle_ctrl_z(&mut self) {
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            self.show_status("Suspend to background is not supported on Windows", "dim");
            return;
        }
        self.show_status("Suspend to background is not available in this terminal host", "dim");
    }

    /// Port of `interruptOrClearInput`.
    ///
    /// PARTIAL: the aborts are issued through `AgentConnection`, which belongs to
    /// the agent-connection slice.
    pub fn interrupt_or_clear_input(&mut self) {
        self.show_status("Interrupt requested", "dim");
    }

    /// Port of `handleAgentsBack`.
    pub fn handle_agents_back(&mut self, editor_text: &str) -> bool {
        if !editor_text.trim().is_empty() {
            return false;
        }
        if !self.options.return_to_agents_view {
            self.request_agents_view_blocking();
            return true;
        }
        self.return_to_agents_view(InteractiveModeRunResultType::AgentsView);
        true
    }

    /// Port of `requestAgentsView`.
    pub fn request_agents_view(&mut self) {
        self.request_agents_view_blocking();
    }

    /// Port of `requestAgentsView`.
    fn request_agents_view_blocking(&mut self) {
        if !self.options.return_to_agents_view {
            self.show_status(
                "The agents view needs the daemon; start without --no-daemon to browse sessions",
                "dim",
            );
            return;
        }
        self.return_to_agents_view(InteractiveModeRunResultType::AgentsView);
    }

    /// Port of `returnToAgentsView`.
    ///
    /// PARTIAL: `agentConnection.dispose()` and the signal handlers belong to the
    /// agent-connection slice.
    pub fn return_to_agents_view(&mut self, request: InteractiveModeRunResultType) {
        if self.shutdown_requested || self.agents_view_request.is_some() {
            return;
        }
        self.agents_view_request = Some(request);
        self.shutdown_requested = true;
    }

    /// Port of `browseQueueSelection`.
    pub fn browse_queue_selection(&mut self, editor_text: &str, direction: i64) -> Option<String> {
        let queue = self.get_connection_queue();
        self.queue_selection.move_cursor(&queue, editor_text, direction)
    }

    /// Port of `applyAuthStaleEvent`.
    pub fn apply_auth_stale_event(&self, _provider: &str, _source_tokens: &[String]) {
        self.model_registry()
            .lock()
            .expect("model registry poisoned")
            .mark_provider_auth_stale(_provider);
    }

    /// Port of `updateConnectionStateFromEvent`.
    pub fn update_connection_state_from_event(&mut self, event: &AgentConnectionSessionEvent) {
        if self.connection_state.is_none() {
            return;
        }
        match event {
            AgentConnectionSessionEvent::AgentStart => {
                self.patch_connection_state(|state| {
                    state.is_streaming = true;
                    state.active_tool_names.clear();
                });
            }
            AgentConnectionSessionEvent::MessageEnd { .. } => {
                self.patch_connection_state(|state| {
                    state.message_count += 1.0;
                });
            }
            AgentConnectionSessionEvent::AgentEnd { .. } => {
                self.patch_connection_state(|state| {
                    state.is_streaming = false;
                    state.active_tool_names.clear();
                });
            }
            AgentConnectionSessionEvent::SessionActionUpdate { actions } => {
                self.patch_connection_queue(|state| {
                    state.session_actions = actions.clone();
                });
            }
            AgentConnectionSessionEvent::CompactionStart { .. } => {
                self.patch_connection_state(|state| state.is_compacting = true);
            }
            AgentConnectionSessionEvent::CompactionEnd { .. } => {
                self.patch_connection_state(|state| state.is_compacting = false);
            }
            AgentConnectionSessionEvent::SessionInfoChanged { name } => {
                self.patch_connection_state(|state| state.session_name = name.clone());
            }
            AgentConnectionSessionEvent::ThinkingLevelChanged { level } => {
                self.patch_connection_state(|state| state.thinking_level = level.clone());
            }
            AgentConnectionSessionEvent::ServiceTierChanged { service_tier } => {
                self.patch_connection_state(|state| state.service_tier = service_tier.clone());
            }
            AgentConnectionSessionEvent::AutoRetryStart { attempt, .. } => {
                self.patch_connection_state(|state| state.retry_attempt = *attempt);
            }
            AgentConnectionSessionEvent::AutoRetryEnd { .. } => {
                self.patch_connection_state(|state| state.retry_attempt = 0.0);
            }
            AgentConnectionSessionEvent::GoalUpdate { goal } => {
                self.patch_connection_state(|state| state.goal = goal.clone());
            }
            AgentConnectionSessionEvent::BashStart { .. } => {
                self.patch_connection_state(|state| state.is_bash_running = true);
            }
            AgentConnectionSessionEvent::BashEnd { .. } => {
                self.patch_connection_state(|state| state.is_bash_running = false);
            }
            AgentConnectionSessionEvent::RecapUpdate { recap } => {
                self.session_recap = recap.clone();
            }
            AgentConnectionSessionEvent::RlmChildUpdate { child } => {
                self.update_subagent_summary(child.clone());
            }
            _ => {}
        }
    }

    /// Port of `handleGoalUpdate`.
    pub fn handle_goal_update(&mut self, goal: &GoalState, terminal_columns: f64) {
        self.sync_goal_tray(goal);
        if self.should_announce_goal_update(goal) {
            let status = self.format_goal_status(goal, terminal_columns);
            self.show_status(&status, "dim");
        } else {
            self.ui.request_render();
        }
    }

    /// Port of `syncGoalTray`.
    pub fn sync_goal_tray(&mut self, _goal: &GoalState) {
        self.ui.request_render();
    }

    /// Port of `updateGoalTrayTimer`.
    pub fn update_goal_tray_timer(&mut self, _goal: &GoalState) {
        // The 1s tray refresh interval is driven by the host timer loop.
    }

    /// Port of `stopGoalTrayTimer`.
    pub fn stop_goal_tray_timer(&mut self) {}

    /// Port of `renderRecap`.
    pub fn render_recap(&mut self) {
        self.recap_container.clear();
        let recap = self
            .session_recap
            .as_ref()
            .map(|recap| recap.trim().to_string())
            .filter(|recap| !recap.is_empty());
        if let Some(recap) = recap {
            self.recap_container
                .add_child(Box::new(super::interactive_mode_services::TruncatedText::new(
                    theme().fg("dim", &format!("Recap: {recap}")),
                    1,
                    0,
                )));
            self.recap_container
                .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        }
        self.ui.request_render();
    }

    /// Port of `renderWidgets`.
    pub fn render_widgets(&mut self) {
        self.render_widget_container(true);
        self.render_widget_container(false);
        self.ui.request_render();
    }

    /// Port of `renderWidgetContainer` (component factories live in other slices).
    pub fn render_widget_container(&mut self, above: bool) {
        let container = if above { &mut self.widget_container_above } else { &mut self.widget_container_below };
        container.clear();
    }

    /// Port of `clearInputBar`.
    pub fn clear_input_bar(&mut self) {
        self.queue_selection.reset();
        self.ui.request_render();
    }

    fn show_compaction_error(&mut self, message: &str) {
        self.chat_container.add_child(Box::new(CompactionNotice {
            session_id: self.connection_state.as_ref().map(|state| state.session_id.clone()),
            text: Text::new(theme().fg("error", &format!("Error: {message}")), 1, 0),
        }));
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
    }

    fn clear_compaction_notices(&mut self) {
        let session_id = self.connection_state.as_ref().map(|state| state.session_id.as_str());
        let before = self.chat_container.len();
        self.chat_container.children.retain(|child| {
            child.as_any().downcast_ref::<CompactionNotice>()
                .is_none_or(|notice| notice.session_id.as_deref() != session_id)
        });
        if before != self.chat_container.len() {
            self.last_status_spacer_index = None;
            self.last_status_text_index = None;
            self.ui.request_render();
        }
    }

    /// Port of `showError` (interactive-mode.ts:7672-7676).
    ///
    /// Errors always append a fresh spacer + line and never coalesce, so the
    /// `Error: ` prefix is never overwritten by a later status.
    pub fn show_error(&mut self, message: &str) {
        self.chat_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.chat_container.add_child(Box::new(Text::new(
            theme().fg("error", &format!("Error: {message}")),
            1,
            0,
        )));
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
    }

    /// Port of `showWarning` (interactive-mode.ts:7678-7682).
    pub fn show_warning(&mut self, message: &str) {
        self.chat_container
            .add_child(Box::new(super::interactive_mode_services::Spacer::new(1)));
        self.chat_container.add_child(Box::new(Text::new(
            theme().fg("warning", &format!("\u{26a0} {message}")),
            1,
            0,
        )));
        self.last_status_spacer_index = None;
        self.last_status_text_index = None;
        self.ui.request_render();
    }

    /// Port of `updateEditorBorderColor`.
    pub fn update_editor_border_color(&mut self) {
        self.ui.request_render();
    }

    /// Port of `getPromptContextContainers`.
    pub fn get_prompt_context_containers(&self) -> Vec<&super::interactive_mode_services::Container> {
        vec![
            &self.widget_container_above,
            &self.recap_container,
            &self.queued_messages_container,
            &self.side_question_container,
            &self.feature_hint_container,
        ]
    }

    /// Port of the `mainViewContainer` child order
    /// (interactive-mode.ts:1268-1272): history, chat, shortcut guide, pending
    /// messages, status. The host renders these in that order before the prompt
    /// context containers.
    pub fn get_main_view_containers(&self) -> Vec<&super::interactive_mode_services::Container> {
        vec![
            &self.history_container,
            &self.chat_container,
            &self.shortcut_guide_container,
            &self.pending_messages_container,
            &self.status_container,
        ]
    }

    /// Port of `getPromptDockComponents`.
    pub fn get_prompt_dock_components(&self) -> Vec<&super::interactive_mode_services::Container> {
        vec![&self.editor_container, &self.footer_slot]
    }

    /// Port of `collectQueueReplaceImages`.
    pub fn collect_queue_replace_images(&self, text: &str) -> Vec<ImageContent> {
        let pending: Vec<(i64, ImageContent)> =
            self.pasted_images.iter().map(|(id, image)| (*id, image.clone())).collect();
        collect_marked_images(&pending, text)
    }

    /// Port of `liveImageMarkerIds`.
    pub fn live_image_marker_ids(&self) -> HashSet<i64> {
        self.pasted_images.keys().copied().collect()
    }

    /// Port of `collectImagesFor`.
    pub fn collect_images_for(&self, text: &str) -> Vec<ImageContent> {
        let pending: Vec<(i64, ImageContent)> =
            self.pasted_images.iter().map(|(id, image)| (*id, image.clone())).collect();
        collect_marked_images(&pending, text)
    }

    /// Port of `hasPastedImagesFor`.
    pub fn has_pasted_images_for(&self, text: &str) -> bool {
        !self.collect_images_for(text).is_empty()
    }

    /// Port of `rememberPastedImage`.
    pub fn remember_pasted_image(&mut self, image: ImageContent, data_bytes: f64) -> i64 {
        let marker_id = self.next_image_marker_id;
        self.next_image_marker_id += 1;
        self.pasted_images.insert(marker_id, image);
        self.evict_pasted_images(data_bytes);
        marker_id
    }

    /// Port of the pasted-image eviction loop (`MAX_PASTED_IMAGE_BYTES`).
    fn evict_pasted_images(&mut self, incoming_bytes: f64) {
        let mut images: Vec<(i64, ImageContent)> = self
            .pasted_images
            .iter()
            .map(|(id, image)| (*id, image.clone()))
            .collect();
        let keep: HashSet<i64> = images.iter().map(|(id, _)| *id).collect();
        evict_images_to_budget(
            &mut images,
            |image: &ImageContent| image.data.len() as f64,
            MAX_PASTED_IMAGE_BYTES - incoming_bytes,
            &keep,
        );
        let retained: HashSet<i64> = images.into_iter().map(|(id, _)| id).collect();
        self.pasted_images.retain(|id, _| retained.contains(id));
    }

    /// Port of `formatImageMarker`.
    pub fn format_image_marker(&self, id: i64) -> String {
        format_image_marker(id as f64)
    }

    /// Port of `remapImageMarkers`.
    pub fn remap_image_markers(&self, text: &str, remaps: &HashMap<i64, i64>) -> String {
        remap_image_markers(text, remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::theme::theme::init_theme;
    use pi_ai::types::{ContentBlock, TextContent, ToolCall, ToolResultMessage};

    fn assistant_with_tool_calls(ids: &[&str]) -> AgentMessage {
        let content = ids
            .iter()
            .map(|id| {
                ContentBlock::ToolCall(ToolCall::new(*id, "read", serde_json::Map::new()))
            })
            .collect();
        AgentMessage::Message(pi_ai::types::Message::Assistant(pi_ai::types::AssistantMessage {
            content,
            ..Default::default()
        }))
    }

    fn tool_result(id: &str) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::ToolResult(ToolResultMessage::new(
            id,
            "read",
            Vec::new(),
            false,
            0,
        )))
    }

    #[test]
    fn terminal_title_keeps_the_app_identity_in_every_state() {
        assert_eq!(
            terminal_title_text("Optimus - Agent", Some("fix batch"), "workspace"),
            "Optimus - Agent - fix batch - workspace"
        );
        assert_eq!(
            terminal_title_text("Optimus - Agent", None, "workspace"),
            "Optimus - Agent - workspace"
        );
        assert_eq!(
            terminal_title_text("Optimus - Assistant", Some("resume"), "profile"),
            "Optimus - Assistant - resume - profile"
        );
    }

    fn user_message(text: &str) -> AgentMessage {
        AgentMessage::Message(pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text(text.to_string()),
            0,
        )))
    }

    #[test]
    fn start_hints_and_random_selection() {
        assert_eq!(START_HINTS.len(), 5);
        assert_eq!(START_HINTS[0], "Try \"refactor @<filepath>\"");
        assert_eq!(get_random_start_hint(&|| 0.0), START_HINTS[0]);
        assert_eq!(get_random_start_hint(&|| 0.99), START_HINTS[4]);
        // Out-of-range random values use the TypeScript nullish fallback.
        assert_eq!(get_random_start_hint(&|| 1.0), START_HINTS[0]);
    }

    #[test]
    fn queued_preview_labels_only_unlabeled_messages() {
        assert_eq!(format_queued_message_preview("hello", QueueLabel::Steering), "Steering: hello");
        assert_eq!(format_queued_message_preview("hello", QueueLabel::FollowUp), "Follow-up: hello");
        assert_eq!(format_queued_message_preview("Heartbeat: ping", QueueLabel::Steering), "Heartbeat: ping");
        assert_eq!(format_queued_message_preview("Goal: x", QueueLabel::Steering), "Goal: x");
        assert_eq!(format_queued_message_preview("Message: x", QueueLabel::Steering), "Message: x");
        assert_eq!(format_queued_message_preview("Bash: x", QueueLabel::Steering), "Bash: x");
    }

    #[test]
    fn splash_cwd_shortens_the_home_directory() {
        let home = home_dir().replace('\\', "/");
        if !home.is_empty() {
            assert_eq!(format_splash_cwd(&home), "~");
            assert_eq!(format_splash_cwd(&format!("{home}/work")), "~/work");
        }
        assert_eq!(format_splash_cwd("C:\\work\\repo"), "C:/work/repo");
    }

    #[test]
    fn merge_subagent_snapshot_keeps_previous_identity_for_active_updates() {
        let previous = AgentConnectionRlmChildAgentSnapshot {
            id: "c1".into(),
            parent_id: Some("p1".into()),
            active_session_id: Some("s1".into()),
            status: "running".into(),
            activity: Some("working".into()),
            ..Default::default()
        };
        let incoming = AgentConnectionRlmChildAgentSnapshot {
            id: "c1".into(),
            status: "running".into(),
            ..Default::default()
        };
        let merged = merge_subagent_snapshot(&previous, &incoming);
        assert_eq!(merged.parent_id.as_deref(), Some("p1"));
        assert_eq!(merged.active_session_id.as_deref(), Some("s1"));
        assert_eq!(merged.activity.as_deref(), Some("working"));

        let terminal = AgentConnectionRlmChildAgentSnapshot {
            id: "c1".into(),
            status: "completed".into(),
            ..Default::default()
        };
        let merged = merge_subagent_snapshot(&previous, &terminal);
        assert_eq!(merged.active_session_id, None);
        assert_eq!(merged.activity, None);
    }

    #[test]
    fn truncate_path_middle_keeps_the_tail() {
        assert_eq!(truncate_path_middle("src/a.rs", 100.0), "src/a.rs");
        assert_eq!(truncate_path_middle("/very/long/path/file.rs", 12.0), "/…/path/fil…");
        assert_eq!(truncate_path_middle("/x/y.rs", 1.0), "/");
    }

    #[test]
    fn payload_getters_match_the_typescript_type_checks() {
        let payload = serde_json::json!({
            "text": "hello",
            "number": 3,
            "boolean": true,
            "strings": ["a", "b"],
            "mixed": ["a", 1],
            "notify": "warning",
            "placement": "belowEditor",
            "indicator": { "frames": ["a"], "intervalMs": 250 }
        });
        assert_eq!(get_payload_string(&payload, "text").as_deref(), Some("hello"));
        assert_eq!(get_payload_string(&payload, "number"), None);
        assert_eq!(get_payload_number(&payload, "number"), Some(3.0));
        assert_eq!(get_payload_boolean(&payload, "boolean"), Some(true));
        assert_eq!(get_payload_string_array(&payload, "strings"), Some(vec!["a".into(), "b".into()]));
        assert_eq!(get_payload_string_array(&payload, "mixed"), None);
        assert_eq!(get_payload_string_array(&payload, "missing"), None);
        assert_eq!(get_payload_notify_type(&payload, "notify"), Some(NotifyType::Warning));
        assert_eq!(get_payload_notify_type(&payload, "text"), None);
        assert_eq!(get_payload_widget_placement(&payload, "placement"), Some(WidgetPlacement::BelowEditor));
        assert_eq!(get_payload_widget_placement(&payload, "text"), None);
        let indicator = get_payload_working_indicator_options(&payload, "indicator").expect("indicator");
        assert_eq!(indicator.frames, Some(vec!["a".to_string()]));
        assert_eq!(indicator.interval_ms, Some(250.0));
        assert_eq!(get_payload_working_indicator_options(&payload, "strings"), None);
    }

    #[test]
    fn update_args_include_self_follows_the_flag_rules() {
        let args = |values: &[&str]| values.iter().map(|value| (*value).to_string()).collect::<Vec<String>>();
        assert!(update_args_include_self(&args(&[])));
        assert!(update_args_include_self(&args(&["--self"])));
        assert!(update_args_include_self(&args(&["--self", "--extensions"])));
        assert!(!update_args_include_self(&args(&["--extension", "foo"])));
        assert!(update_args_include_self(&args(&["self"])));
        assert!(update_args_include_self(&args(&["pi"])));
        assert!(update_args_include_self(&args(&[&app_name()])));
        assert!(!update_args_include_self(&args(&["other"])));
        // `--daemon-socket <path>` consumes its value, so the path is never positional.
        assert!(update_args_include_self(&args(&["--daemon-socket", "other"])));
    }

    #[test]
    fn relaunch_and_child_args_keep_the_daemon_socket() {
        let args = |values: &[&str]| values.iter().map(|value| (*value).to_string()).collect::<Vec<String>>();
        assert_eq!(
            build_update_relaunch_args(&args(&["--verbose"]), Some("/tmp/s.json")),
            args(&["--verbose", "--resume", "/tmp/s.json"])
        );
        assert_eq!(
            build_update_relaunch_args(&args(&["--continue"]), Some("/tmp/s.json")),
            args(&["--continue"])
        );
        assert_eq!(build_update_relaunch_args(&args(&["--verbose"]), None), args(&["--verbose"]));
        assert_eq!(
            build_update_child_args(&args(&["--verbose"]), "/tmp/d.sock"),
            args(&["--verbose", "--daemon-socket", "/tmp/d.sock"])
        );
        assert_eq!(
            build_update_child_args(&args(&["--daemon-socket", "/x"]), "/tmp/d.sock"),
            args(&["--daemon-socket", "/x"])
        );
        assert_eq!(resolve_interactive_update_daemon_socket_path(&args(&["--daemon-socket", "/x"]), "/a"), "/x");
        assert_eq!(resolve_interactive_update_daemon_socket_path(&args(&["--daemon-socket"]), "/a"), "/a");
        assert_eq!(resolve_interactive_update_daemon_socket_path(&args(&["--verbose"]), "/a"), "/a");
    }

    #[test]
    fn execve_failure_throws_requires_node_26_1() {
        assert!(execve_failure_throws("26.1.0"));
        assert!(execve_failure_throws("27.0.0"));
        assert!(!execve_failure_throws("26.0.0"));
        assert!(!execve_failure_throws("22.11.0"));
        assert!(!execve_failure_throws("not-a-version"));
    }

    #[test]
    fn try_exec_update_relaunch_skips_unsupported_platforms() {
        let launch = CliSubprocessLaunchSpec { command: "node".into(), args: vec!["index.js".into()] };
        let mut options = UpdateRelaunchExecOptions {
            platform: "win32".into(),
            node_version: "26.1.0".into(),
            cwd: "/tmp".into(),
            previous_cwd: "/".into(),
            environment: HashMap::new(),
            chdir: Box::new(|_| {}),
            execve: Some(Box::new(|_, _, _| {})),
        };
        assert!(!try_exec_update_relaunch(&launch, &mut options));
        options.platform = "linux".into();
        assert!(try_exec_update_relaunch(&launch, &mut options));
        options.execve = None;
        assert!(!try_exec_update_relaunch(&launch, &mut options));
    }

    #[test]
    fn initial_render_messages_keeps_orphan_tool_calls_attached() {
        // Under the limit: the messages are returned unchanged.
        let small = vec![user_message("hi"), assistant_with_tool_calls(&["t1"]), tool_result("t1")];
        assert_eq!(initial_render_messages(small.clone()).len(), small.len());

        // Over the limit: the required assistant tool call is prepended.
        let mut messages: Vec<AgentMessage> = Vec::new();
        messages.push(assistant_with_tool_calls(&["old"]));
        for index in 0..INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT {
            messages.push(user_message(&format!("m{index}")));
        }
        messages.push(tool_result("old"));
        let rendered = initial_render_messages(messages);
        assert_eq!(rendered.len(), INITIAL_TRANSCRIPT_RENDER_MESSAGE_LIMIT);
        assert_eq!(rendered[0].role(), "assistant");
        assert_eq!(rendered.last().unwrap().role(), "toolResult");
    }

    #[test]
    fn omit_orphan_tool_results_drops_unmatched_results() {
        let messages = vec![
            assistant_with_tool_calls(&["t1"]),
            tool_result("t1"),
            tool_result("t2"),
            user_message("hi"),
        ];
        let kept = omit_orphan_tool_results(messages);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[1].role(), "toolResult");
    }

    #[test]
    fn dead_terminal_errors_match_the_known_codes() {
        assert!(is_dead_terminal_error(Some("EIO")));
        assert!(is_dead_terminal_error(Some("EPIPE")));
        assert!(is_dead_terminal_error(Some("ENOTCONN")));
        assert!(!is_dead_terminal_error(Some("ECONNRESET")));
        assert!(!is_dead_terminal_error(None));
    }

    #[test]
    fn agent_depth_label_hides_the_root_without_children() {
        assert_eq!(format_agent_depth_label(None, false), None);
        assert_eq!(format_agent_depth_label(Some(0.0), false), None);
        assert_eq!(format_agent_depth_label(Some(0.0), true), Some("depth 0".to_string()));
        assert_eq!(format_agent_depth_label(Some(2.0), false), Some("depth 2".to_string()));
    }

    #[test]
    fn merge_older_history_rejects_a_discontinuous_range() {
        let current = LoadedAgentConnectionHistory {
            window: AgentConnectionHistoryWindow {
                version: 1.0,
                generation: "g".into(),
                representation: "jsonl".into(),
                tip_entry_id: Some("t".into()),
                total_message_count: 3.0,
                start_index: 1.0,
                entry_ids: vec!["b".into(), "c".into()],
                has_older: true,
                order: "chronological".into(),
            },
            messages: vec![user_message("b"), user_message("c")],
        };
        let good = AgentConnectionHistoryRange {
            window: AgentConnectionHistoryWindow {
                version: 1.0,
                generation: "g".into(),
                representation: "jsonl".into(),
                tip_entry_id: Some("t".into()),
                total_message_count: 3.0,
                start_index: 0.0,
                entry_ids: vec!["a".into()],
                has_older: false,
                order: "chronological".into(),
            },
            messages: vec![user_message("a")],
        };
        let merged = merge_older_agent_connection_history(&current, &good).expect("merged");
        assert_eq!(merged.window.entry_ids, vec!["a".to_string(), "b".into(), "c".into()]);
        assert_eq!(merged.messages.len(), 3);

        let mut bad = good.clone();
        bad.window.generation = "other".into();
        assert_eq!(
            merge_older_agent_connection_history(&current, &bad).unwrap_err(),
            "Older history range does not continue the pinned snapshot"
        );

        let mut overlapping = good;
        overlapping.window.entry_ids = vec!["b".into()];
        overlapping.messages = vec![user_message("b")];
        assert_eq!(
            merge_older_agent_connection_history(&current, &overlapping).unwrap_err(),
            "Older history range overlaps already loaded messages"
        );
    }

    #[test]
    fn get_path_command_argument_handles_quotes_and_spaces() {
        let mode = test_mode();
        assert_eq!(mode.get_path_command_argument("/export", "/export"), None);
        assert_eq!(mode.get_path_command_argument("/export ", "/export"), None);
        assert_eq!(mode.get_path_command_argument("/export a.md", "/export"), Some("a.md".to_string()));
        assert_eq!(
            mode.get_path_command_argument("/export \"a b.md\"", "/export"),
            Some("a b.md".to_string())
        );
        assert_eq!(mode.get_path_command_argument("/import 'a.md'", "/import"), Some("a.md".to_string()));
        assert_eq!(mode.get_path_command_argument("/export \"unclosed", "/export"), None);
    }

    #[test]
    fn shortcut_guide_is_ephemeral_and_capitalizes_keys() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let mut mode = test_mode();

        // `showShortcutGuide` may be called twice; the container is reused
        // (interactive-mode-startup.test.ts:334-353).
        mode.show_shortcut_guide();
        mode.show_shortcut_guide();
        assert_eq!(mode.chat_container.len(), 0, "the guide must not enter chat history");
        assert_eq!(mode.shortcut_guide_container.len(), 2, "spacer + markdown");

        let rendered =
            crate::modes::interactive::interactive_mode_services::Component::render(
                &mode.shortcut_guide_container,
                80,
            )
            .join("\n");
        assert!(rendered.contains("shell mode"), "{rendered:?}");
        assert!(rendered.contains("file paths"));
        assert!(rendered.contains("stash prompt"));
        assert!(rendered.contains("full reference"), "{rendered:?}");
        // Row 27: capitalized key text, never the raw `ctrl+s`.
        assert!(!rendered.contains("ctrl+s"), "{rendered:?}");
        assert!(!rendered.contains("ctrl+o"));
        assert!(!rendered.contains("Ctrl+Z"));

        mode.clear_shortcut_guide();
        assert_eq!(mode.shortcut_guide_container.len(), 0);
    }

    #[test]
    fn compaction_loader_text_matches_the_reason_ladder() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState::default());
        mode.patch_connection_state(|state| state.is_compacting = true);

        // "manual" (interactive-mode.ts:3543).
        mode.start_compaction_loader("manual", None);
        let manual = mode.compaction_loader_label.clone().expect("label");
        assert!(manual.starts_with("Compacting context..."), "{manual:?}");
        assert!(manual.contains("(Ctrl+C to cancel)"), "{manual:?}");

        // "requested" + custom instructions (interactive-mode.ts:3545, :3540).
        mode.start_compaction_loader("requested", Some("keep the API notes"));
        let requested = mode.compaction_loader_label.clone().expect("label");
        assert!(requested.starts_with("Agent requested compaction, compacting context"), "{requested:?}");
        assert!(requested.contains("(focus: keep the API notes)"), "{requested:?}");

        // "overflow" (interactive-mode.ts:3546).
        mode.start_compaction_loader("overflow", None);
        let overflow = mode.compaction_loader_label.clone().expect("label");
        assert!(overflow.starts_with("Context overflow detected, Auto-compacting..."), "{overflow:?}");

        // "threshold" - the auto branch without the overflow prefix.
        mode.start_compaction_loader("threshold", None);
        let threshold = mode.compaction_loader_label.clone().expect("label");
        assert!(threshold.starts_with("Auto-compacting..."), "{threshold:?}");

        // `syncWorkingLoader` re-arms the loader while `isAgentCompacting()` is
        // still true (interactive-mode.ts:3589-3591), so the end event clears the
        // flag before the teardown.
        mode.patch_connection_state(|state| state.is_compacting = false);
        mode.stop_compaction_loader();
        assert_eq!(mode.compaction_loader_label, None);
    }

    #[test]
    fn sync_working_loader_restores_the_compaction_loader_from_state() {
        init_theme(Some("prime"), false);
        let mut mode = test_mode();
        // No start event was seen: the mode only knows `isCompacting`.
        mode.connection_state = Some(AgentConnectionState::default());
        mode.patch_connection_state(|state| state.is_compacting = true);
        assert_eq!(mode.compaction_loader_label, None);

        mode.sync_working_loader();
        assert!(mode.compaction_loader_label.is_some(), "restored from state");
        assert_eq!(
            crate::modes::interactive::interactive_mode_services::Component::render(
                &mode.status_container,
                80
            )
            .join("\n")
            .contains("Compacting context"),
            true
        );
    }

    #[test]
    fn onboarding_shown_is_persisted_before_the_flow_opens() {
        let mut mode = test_mode();
        // `test_mode` uses an in-memory `SettingsManager` with no model configured,
        // so `shouldRunOnboarding` is true (onboarding.rs:74-83).
        assert!(mode.should_run_onboarding());
        assert!(!mode
            .settings_manager()
            .lock()
            .expect("settings")
            .get_onboarding_shown());

        assert!(mode.run_startup_onboarding());
        // Persisted before the flow body runs (TS:1880-1881); a second call is a
        // no-op because the flag is now set.
        assert!(mode
            .settings_manager()
            .lock()
            .expect("settings")
            .get_onboarding_shown());
        assert!(!mode.run_startup_onboarding());
    }

    #[test]
    fn hotkeys_guide_lands_in_chat_history_not_the_ephemeral_container() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let mut mode = test_mode();
        mode.handle_hotkeys_command();
        assert_eq!(mode.chat_container.len(), 2, "spacer + markdown");
        assert_eq!(mode.shortcut_guide_container.len(), 0);

        let rendered = chat_lines(&mode).join("\n");
        assert!(rendered.contains("Navigation"));
        assert!(rendered.contains("Editing"));
        assert!(rendered.contains("Queue follow-up message"), "{rendered:?}");
        assert!(rendered.contains("Browse and edit queued messages"), "{rendered:?}");
        assert!(!rendered.contains("Ctrl+Z"), "Ctrl+Z stays out of the guide");
    }

    #[test]
    fn ctrl_c_exit_hint_arms_a_two_second_window_then_expires() {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let mut mode = test_mode();

        // Idle + empty editor: the first press only arms the hint.
        mode.show_ctrl_c_exit_hint();
        assert!(mode.is_ctrl_c_exit_hint_visible());
        assert_eq!(
            mode.get_tray_override_label(""),
            Some("Press Ctrl+C again to exit".to_string())
        );

        // The window is 2 s (InteractiveMode.EXIT_HINT_DURATION_MS, TS:976).
        assert!(mode.ctrl_c_exit_hint_expires_at > now_ms());
        assert!(mode.ctrl_c_exit_hint_expires_at <= now_ms() + 2_000.0);

        // Simulate the elapsed window: the host tick must drop the hint so the
        // tray stops overriding the location label (TS:7018-7025).
        mode.ctrl_c_exit_hint_expires_at = now_ms() - 1.0;
        assert!(!mode.is_ctrl_c_exit_hint_visible());
        mode.expire_ctrl_c_exit_hint();
        assert_eq!(mode.ctrl_c_exit_hint_expires_at, 0.0);
        assert_eq!(mode.get_tray_override_label(""), None);
    }

    /// Reads back the rendered text of the chat container, matching the TS test
    /// helper `renderLastLine` (interactive-mode-status.test.ts:167-205).
    fn chat_lines(mode: &InteractiveMode) -> Vec<String> {
        mode.chat_container.children.iter().flat_map(|child| child.render(80)).collect()
    }


    #[test]
    fn show_status_coalesces_sequential_messages_into_the_previous_line() {
        init_theme(Some("prime"), false);
        let mut mode = test_mode();
        mode.show_status("STATUS_ONE", "dim");
        assert_eq!(mode.chat_container.len(), 2, "spacer + text");
        assert!(chat_lines(&mode).join("\n").contains("STATUS_ONE"));

        mode.show_status("STATUS_TWO", "dim");
        // The second status updates the previous line instead of appending.
        assert_eq!(mode.chat_container.len(), 2);
        let joined = chat_lines(&mode).join("\n");
        assert!(joined.contains("STATUS_TWO"));
        assert!(!joined.contains("STATUS_ONE"));
    }

    #[test]
    fn show_status_appends_a_new_line_when_something_else_was_added_between() {
        init_theme(Some("prime"), false);
        let mut mode = test_mode();
        mode.show_status("STATUS_ONE", "dim");
        assert_eq!(mode.chat_container.len(), 2);

        mode.chat_container.add_child(Box::new(Text::new("OTHER", 1, 0)));
        assert_eq!(mode.chat_container.len(), 3);

        mode.show_status("STATUS_TWO", "dim");
        // Adds a fresh spacer + text.
        assert_eq!(mode.chat_container.len(), 5);
        let joined = chat_lines(&mode).join("\n");
        assert!(joined.contains("STATUS_ONE"));
        assert!(joined.contains("STATUS_TWO"));
    }

    #[test]
    fn show_warning_and_show_error_use_the_typescript_prefixes() {
        init_theme(Some("prime"), false);
        let mut mode = test_mode();
        mode.show_warning("careful");
        let warned = chat_lines(&mode).join("\n");
        assert!(warned.contains("\u{26a0} careful"), "warning keeps the marker: {warned:?}");

        let mut mode = test_mode();
        mode.show_error("boom");
        let errored = chat_lines(&mode).join("\n");
        assert!(errored.contains("Error: boom"), "error keeps the prefix: {errored:?}");

        // Neither coalesces into a neighbouring status line.
        let mut mode = test_mode();
        mode.show_status("first", "dim");
        mode.show_error("boom");
        mode.show_status("second", "dim");
        assert_eq!(mode.chat_container.len(), 6);
        let joined = chat_lines(&mode).join("\n");
        assert!(joined.contains("first"));
        assert!(joined.contains("Error: boom"));
        assert!(joined.contains("second"));
    }

    #[test]
    fn path_command_argument_matches_the_import_cases() {
        // interactive-mode-import-command.test.ts:29-52.
        let mode = test_mode();
        assert_eq!(
            mode.get_path_command_argument("/import \"path/to/session.jsonl\"", "/import"),
            Some("path/to/session.jsonl".to_string())
        );
        assert_eq!(
            mode.get_path_command_argument("/import john's/session.jsonl", "/import"),
            Some("john's/session.jsonl".to_string())
        );
        assert_eq!(
            mode.get_path_command_argument("/important /tmp/session.jsonl", "/import"),
            None
        );
        assert_eq!(mode.get_path_command_argument("/exporter out.html", "/export"), None);
    }

    #[test]
    fn capitalize_key_keeps_esc_lowercase() {
        let mode = test_mode();
        assert_eq!(mode.capitalize_key("ctrl+o"), "Ctrl+O");
        assert_eq!(mode.capitalize_key("esc"), "esc");
        assert_eq!(mode.capitalize_key("shift+esc/ctrl+o"), "Shift+esc/Ctrl+O");
    }

    #[test]
    fn format_goal_elapsed_scales_units() {
        let mode = test_mode();
        assert_eq!(mode.format_goal_elapsed(5.0), "5s");
        assert_eq!(mode.format_goal_elapsed(65.0), "1m 05s");
        assert_eq!(mode.format_goal_elapsed(3661.0), "1h 01m");
    }

    #[test]
    fn format_working_elapsed_scales_units() {
        let mode = test_mode();
        assert_eq!(mode.format_working_elapsed(1_000.0), "1s");
        assert_eq!(mode.format_working_elapsed(65_000.0), "1m 05s");
        assert_eq!(mode.format_working_elapsed(3_661_000.0), "1h 01m 01s");
        assert_eq!(mode.format_working_elapsed(90_000_000.0), "1d 01h 00m 00s");
    }

    #[test]
    fn queue_selection_and_tray_labels_use_the_connection_state() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState {
            session_id: "s1".into(),
            cwd: "/repo".into(),
            message_count: 2.0,
            model: Some(Model { reasoning: true, ..Model::new("glm-5.3", "GLM 5.3", "openai-completions", "anthropic", "https://x.invalid") }),
            thinking_level: ThinkingLevel::High,
            service_tier: Some(Some("priority".to_string())),
            ..Default::default()
        });
        assert_eq!(mode.get_current_cwd(), "/repo");
        assert_eq!(mode.get_current_model_id().as_deref(), Some("glm-5.3"));
        assert_eq!(mode.get_model_tray_label(), "GLM 5.3 \u{2022} high \u{2022} fast");
        assert!(!mode.is_new_chat());

        let mut fresh = test_mode();
        fresh.connection_state = Some(AgentConnectionState::default());
        assert!(fresh.is_new_chat());
        assert_eq!(fresh.get_model_tray_label(), "\u{2014}");
    }

    #[test]
    fn context_usage_adds_in_flight_tokens_only_while_streaming() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState {
            context_usage: ContextUsage { tokens: Some(100.0), context_window: 1000.0, percent: Some(10.0) },
            is_streaming: false,
            ..Default::default()
        });
        let usage = mode.get_connection_context_usage().expect("usage");
        assert_eq!(usage.tokens, Some(100.0));
        assert_eq!(usage.percent, Some(10.0));
    }

    #[test]
    fn goal_tray_labels_follow_the_status() {
        let mut mode = test_mode();
        let mut goal = empty_goal_state();
        goal.status = GoalStatus::Active;
        goal.time_used_seconds = 61.0;
        // `getTrayGoalLabel()` reads `this.getGoalState()`, so the label is driven
        // through the connection state exactly like the reference.
        mode.connection_state = Some(AgentConnectionState { goal: goal.clone(), ..Default::default() });
        assert_eq!(mode.get_tray_goal_label().as_deref(), Some("Pursuing goal (1m 01s)"));
        goal.status = GoalStatus::Idle;
        mode.connection_state = Some(AgentConnectionState { goal, ..Default::default() });
        assert_eq!(mode.get_tray_goal_label(), None);
    }

    #[test]
    fn goal_announcements_track_status_and_reason_changes() {
        let mut mode = test_mode();
        let mut goal = empty_goal_state();
        mode.set_goal_announcement_baseline(&goal);
        assert!(!mode.should_announce_goal_update(&goal));

        goal.status = GoalStatus::Active;
        assert!(mode.should_announce_goal_update(&goal));
        assert!(!mode.should_announce_goal_update(&goal));

        goal.status = GoalStatus::Paused;
        goal.last_reason = Some("budget".to_string());
        assert!(mode.should_announce_goal_update(&goal));
        goal.last_reason = Some("other".to_string());
        assert!(mode.should_announce_goal_update(&goal));
        assert!(!mode.should_announce_goal_update(&goal));
    }

    #[test]
    fn subagent_summary_seeding_and_removal_follow_the_status_rules() {
        let mut mode = test_mode();
        mode.rlm_node_id = Some("p1".into());
        let child = AgentConnectionRlmChildAgentSnapshot {
            id: "c1".into(),
            parent_id: Some("p1".into()),
            status: "running".into(),
            activity: Some("working".into()),
            ..Default::default()
        };
        mode.seed_subagent_summary(Some(&[child.clone()]));
        assert_eq!(mode.subagent_snapshots.len(), 1);
        assert_eq!(mode.subagent_summary_line.counts.running, 1);
        assert_eq!(mode.subagent_summary_line.counts.total, 1);

        let cancelled = AgentConnectionRlmChildAgentSnapshot { status: "cancelled".into(), ..child.clone() };
        mode.update_subagent_summary(cancelled);
        assert!(mode.subagent_snapshots.is_empty());
    }

    #[test]
    fn remove_subagent_snapshot_removes_descendants() {
        let mut mode = test_mode();
        let parent = AgentConnectionRlmChildAgentSnapshot { id: "p".into(), status: "running".into(), ..Default::default() };
        let child = AgentConnectionRlmChildAgentSnapshot {
            id: "c".into(),
            parent_id: Some("p".into()),
            status: "running".into(),
            ..Default::default()
        };
        mode.subagent_snapshots.insert("p".into(), parent);
        mode.subagent_snapshots.insert("c".into(), child);
        mode.remove_subagent_snapshot("p");
        assert!(mode.subagent_snapshots.is_empty());
    }

    #[test]
    fn connection_state_events_patch_the_snapshot() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState::default());
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::AgentStart);
        assert!(mode.is_agent_streaming());
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::AgentEnd { messages: Vec::new() });
        assert!(!mode.is_agent_streaming());
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::MessageEnd {
            message: user_message("hi"),
        });
        assert_eq!(mode.connection_state.as_ref().expect("state").message_count, 1.0);
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::BashStart {
            command: "ls".into(),
            exclude_from_context: false,
            transient: None,
            run_id: None,
        });
        assert!(mode.is_bash_running());
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::BashEnd {
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            error_message: None,
            transient: None,
            run_id: None,
        });
        assert!(!mode.is_bash_running());
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::SessionInfoChanged {
            name: Some("named".into()),
        });
        assert_eq!(mode.get_current_session_name().as_deref(), Some("named"));
    }

    #[test]
    fn escape_repeat_arms_and_consumes_the_action_once() {
        let mut mode = test_mode();
        mode.arm_escape_repeat("tree");
        assert_eq!(mode.take_escape_repeat_action(), Some("tree"));
        assert_eq!(mode.take_escape_repeat_action(), None);
    }

    #[test]
    fn ctrl_c_exit_hint_expires_and_clears() {
        let mut mode = test_mode();
        assert!(!mode.is_ctrl_c_exit_hint_visible());
        mode.show_ctrl_c_exit_hint();
        assert!(mode.is_ctrl_c_exit_hint_visible());
        mode.clear_ctrl_c_exit_hint(false);
        assert!(!mode.is_ctrl_c_exit_hint_visible());
    }

    #[test]
    fn pasted_images_evict_oldest_beyond_the_cap() {
        let mut mode = test_mode();
        let image = |data: &str| ImageContent::new(data, "image/png");
        mode.remember_pasted_image(image("aaa"), 3.0);
        mode.remember_pasted_image(image("bbb"), 3.0);
        assert_eq!(mode.pasted_images.len(), 2);
        assert_eq!(mode.format_image_marker(1), "[image #1]");
        assert_eq!(mode.collect_images_for("see [image #2]").len(), 1);
        assert!(mode.has_pasted_images_for("[image #1]"));
        assert!(!mode.has_pasted_images_for("no markers"));
    }

    #[test]
    fn queue_helpers_use_the_connection_queue() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState {
            session_actions: super::super::interactive_mode_services::SessionActionSnapshot {
                steering: vec!["s1".into()],
                follow_ups: vec!["f1".into()],
                queued_count: 2,
                active: None,
            },
            ..Default::default()
        });
        let queue = mode.get_connection_queue();
        assert_eq!(queue.steering, vec!["s1".to_string()]);
        assert_eq!(queue.follow_up, vec!["f1".to_string()]);
        assert_eq!(mode.get_queued_action_count(), 2);
        assert!(mode.should_suppress_feature_hint());
        assert_eq!(mode.browse_queue_selection("draft", -1), Some("f1".to_string()));
    }

    #[test]
    fn accepted_queue_previews_survive_stream_refresh_and_clear_authoritatively() {
        use super::super::interactive_mode_services::Component;
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut mode = test_mode();
        mode.apply_connection_state_snapshot(AgentConnectionState {
            session_id: "queue-preview".into(), is_streaming: true,
            session_actions: super::super::interactive_mode_services::SessionActionSnapshot {
                steering: vec!["first human".into(), "second human".into()],
                follow_ups: vec!["after run".into()], queued_count: 3,
                ..Default::default()
            }, ..Default::default()
        });
        let rendered = mode.queued_messages_container.render(80).join("\n");
        assert!(rendered.contains("Steering: first human"));
        assert!(rendered.find("first human") < rendered.find("second human"));
        assert!(rendered.contains("Follow-up: after run"));
        mode.patch_connection_state(|state| state.message_count += 1.0);
        assert_eq!(mode.queued_messages_container.render(80).join("\n"), rendered);
        let mut actions = mode.connection_state.as_ref().unwrap().session_actions.clone();
        actions.steering.remove(0);
        mode.update_connection_state_from_event(&AgentConnectionSessionEvent::SessionActionUpdate { actions });
        assert!(!mode.queued_messages_container.render(80).join("\n").contains("first human"));
        mode.apply_connection_state_snapshot(AgentConnectionState { session_id: "other".into(), ..Default::default() });
        assert!(mode.queued_messages_container.is_empty());
    }

    #[test]
    fn scoped_heartbeats_only_keep_the_session() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState {
            session_id: "s1".into(),
            active_session_id: Some("a1".into()),
            ..Default::default()
        });
        mode.heartbeat_catalog = vec![
            AgentConnectionHeartbeat {
                job: super::super::interactive_mode_services::AgentCronJob {
                    id: "own".into(),
                    active_session_id: "a1".into(),
                    status: "paused".into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            AgentConnectionHeartbeat {
                job: super::super::interactive_mode_services::AgentCronJob {
                    id: "foreign".into(),
                    active_session_id: "a2".into(),
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        let heartbeats = mode.get_scoped_heartbeats();
        assert_eq!(heartbeats.len(), 1);
        assert_eq!(mode.get_tray_heartbeat_label().as_deref(), Some("1 heartbeat \u{b7} 1 paused"));
    }

    #[test]
    fn compact_labels_deduplicate_non_package_extensions() {
        let mode = test_mode();
        let extensions = vec![
            ("/home/u/a/index.ts".to_string(), None),
            ("/home/u/b/index.ts".to_string(), None),
        ];
        let labels = mode.get_compact_extension_labels(&extensions);
        assert_eq!(labels.len(), 2);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn compact_package_source_labels_strip_the_prefixes() {
        let mode = test_mode();
        let npm = super::super::interactive_mode_services::AgentConnectionSourceInfo {
            source: "npm:pkg@1.0.0".into(),
            scope: "user".into(),
            ..Default::default()
        };
        assert_eq!(mode.get_compact_package_source_label(Some(&npm)), "pkg@1.0.0");
        assert_eq!(mode.get_autocomplete_source_tag(Some(&npm)).as_deref(), Some("user:npm:pkg@1.0.0"));
        assert_eq!(mode.get_autocomplete_source_label(Some(&npm)).as_deref(), Some("#user:npm:pkg@1.0.0"));
    }

    #[test]
    fn scope_groups_follow_the_source_and_scope() {
        let mode = test_mode();
        let info = |source: &str, scope: &str| super::super::interactive_mode_services::AgentConnectionSourceInfo {
            source: source.to_string(),
            scope: scope.to_string(),
            ..Default::default()
        };
        assert_eq!(mode.get_scope_group(Some(&info("local", "user"))), "user");
        assert_eq!(mode.get_scope_group(Some(&info("local", "project"))), "project");
        assert_eq!(mode.get_scope_group(Some(&info("cli", "project"))), "path");
        assert_eq!(mode.get_scope_group(Some(&info("local", "temporary"))), "path");
        assert_eq!(mode.get_scope_group(None), "project");
    }

    #[test]
    fn thinking_level_completions_mark_the_current_level() {
        let mut mode = test_mode();
        mode.connection_state = Some(AgentConnectionState {
            available_thinking_levels: vec![ThinkingLevel::Off, ThinkingLevel::High],
            thinking_level: ThinkingLevel::High,
            ..Default::default()
        });
        let completions = mode.get_thinking_level_completions("").expect("completions");
        assert_eq!(completions.len(), 2);
        assert_eq!(completions[1].description.as_deref(), Some("Deep reasoning (~16k tokens) (current)"));
        assert_eq!(mode.get_thinking_level_completions("zzz"), None);

        let mut off_only = test_mode();
        off_only.connection_state = Some(AgentConnectionState {
            available_thinking_levels: vec![ThinkingLevel::Off],
            ..Default::default()
        });
        assert_eq!(off_only.get_thinking_level_completions(""), None);
    }

    #[test]
    fn heartbeat_argument_completions_filter_by_prefix() {
        let mode = test_mode();
        assert_eq!(mode.get_heartbeat_argument_completions("").expect("all").len(), 4);
        assert_eq!(mode.get_heartbeat_argument_completions("st").expect("filtered").len(), 3);
        assert_eq!(mode.get_heartbeat_argument_completions("zzz"), None);
    }

    #[test]
    fn user_message_helpers_handle_text_and_blocks() {
        let mode = test_mode();
        let text = pi_ai::types::UserMessage::new(pi_ai::types::UserContent::Text("hi".into()), 0);
        assert_eq!(mode.get_user_message_text(&text), "hi");
        assert!(mode.is_text_only_user_message(&text));

        let blocks = pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Blocks(vec![
                pi_ai::types::ImageOrTextContent::Text(TextContent::new("a")),
                pi_ai::types::ImageOrTextContent::Image(ImageContent::new("data", "image/png")),
            ]),
            0,
        );
        assert_eq!(mode.get_user_message_text(&blocks), "a");
        assert!(!mode.is_text_only_user_message(&blocks));
    }

    #[test]
    fn goal_status_formatting_matches_the_typescript_branches() {
        let mode = test_mode();
        let mut goal = empty_goal_state();
        assert_eq!(mode.format_goal_status(&goal, 120.0), "No active goal");
        goal.status = GoalStatus::Active;
        assert_eq!(mode.format_goal_status(&goal, 120.0), "Pursuing goal");
        goal.objective = Some("ship it".to_string());
        assert_eq!(mode.format_goal_status(&goal, 120.0), "Goal: ship it");
        goal.status = GoalStatus::Complete;
        assert_eq!(mode.format_goal_status(&goal, 120.0), "Goal complete");
        goal.last_reason = Some("done".to_string());
        assert_eq!(mode.format_goal_status(&goal, 120.0), "Goal complete: done");
        goal.status = GoalStatus::Error;
        goal.last_error = Some("boom".to_string());
        assert_eq!(mode.format_goal_status(&goal, 120.0), "Goal error: boom");
    }

    #[test]
    fn goal_detail_suffix_hides_narrow_details() {
        let mode = test_mode();
        assert_eq!(mode.format_goal_detail_suffix(None, 0.0, 120.0), "");
        assert_eq!(mode.format_goal_detail_suffix(Some("   "), 0.0, 120.0), "");
        assert_eq!(mode.format_goal_detail_suffix(Some("a\n b"), 0.0, 120.0), ": a b");
        // The available width must reach 8 columns before the detail is appended.
        assert_eq!(mode.format_goal_detail_suffix(Some("detail"), 100.0, 104.0), "");
    }

    #[test]
    fn model_fallback_warning_prefers_the_live_connection() {
        let mut mode = test_mode();
        assert_eq!(mode.get_model_fallback_warning_action(None), ModelFallbackWarningAction::Suppress);
        assert_eq!(
            mode.get_model_fallback_warning_action(Some("boom")),
            ModelFallbackWarningAction::Show
        );
        let no_models = crate::core::auth_guidance::format_no_models_available_message();
        assert_eq!(
            mode.get_model_fallback_warning_action(Some(&no_models)),
            ModelFallbackWarningAction::Show
        );
        mode.connection_state = Some(AgentConnectionState {
            model: Some(Model::new("m", "M", "openai-completions", "p", "https://x.invalid")),
            ..Default::default()
        });
        assert_eq!(
            mode.get_model_fallback_warning_action(Some(&no_models)),
            ModelFallbackWarningAction::Suppress
        );
    }

    #[test]
    fn successful_model_change_removes_only_obsolete_restore_warning() {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        let mut mode = test_mode();
        let warning = "Could not restore model github-copilot/gpt-5.6-sol. Using ollama-cloud/glm-5.3-flash";
        mode.options.model_fallback_message = Some(warning.into());
        mode.apply_connection_state_snapshot(AgentConnectionState {
            model: Some(Model::new("glm-5.3-flash", "GLM", "openai-completions", "ollama-cloud", "https://x.invalid")),
            ..Default::default()
        });
        assert_eq!(mode.get_model_fallback_warning_action(Some(warning)), ModelFallbackWarningAction::Show);
        mode.show_warning(warning);
        mode.show_error("keep this error");
        mode.apply_connection_state_snapshot(AgentConnectionState {
            model: Some(Model::new("gpt-5.6-sol", "Sol", "openai-responses", "github-copilot", "https://x.invalid")),
            ..Default::default()
        });
        let rendered = mode.chat_container.children.iter().flat_map(|child| child.render(120)).collect::<Vec<_>>().join("\n");
        assert!(!rendered.contains("Could not restore"));
        assert!(rendered.contains("keep this error"));
        assert!(mode.options.model_fallback_message.is_none());
    }

    #[test]
    fn run_result_source_defaults_to_the_stash_session_id() {
        let mut mode = test_mode();
        mode.prompt_stash_session_id = Some("stash-session".to_string());
        let result = futures::executor::block_on(mode.run());
        assert_eq!(result.type_, InteractiveModeRunResultType::AgentsView);
        assert_eq!(result.type_.as_str(), "agents_view");
        assert_eq!(result.source.session_id, "stash-session");
    }

    /// A minimal mode with no connection: enough for the pure helpers above.
    fn test_mode() -> InteractiveMode {
        let services = InteractiveModeUiServices {
            settings_manager: Arc::new(Mutex::new(
                super::super::interactive_mode_services::SettingsManager::in_memory(serde_json::Map::new()),
            )),
            model_registry: Arc::new(Mutex::new(
                super::super::interactive_mode_services::ModelRegistry::in_memory(),
            )),
            get_initial_cwd: Box::new(|| "/initial".to_string()),
            get_initial_session_name: Box::new(|| Some("initial".to_string())),
            get_themes: Box::new(Vec::new),
            refresh_mcp_providers: None,
        };
        let options = InteractiveModeOptions {
            migrated_providers: None,
            model_fallback_message: None,
            startup_notice: None,
            initial_message: None,
            initial_images: None,
            initial_messages: None,
            initial_prompts: None,
            verbose: false,
            agent_connection: Arc::new(()),
            daemon_socket_path: None,
            local_session_host: None,
            bind_local_session_extensions: false,
            ui_services: Some(services),
            on_shutdown: None,
            return_to_agents_view: false,
            force_fullscreen: false,
            agents_view_owns_startup_notices: false,
            session_depth: None,
            session_has_children: false,
            prompt_stash_store: None,
            prompt_stash_session_id: None,
        };
        InteractiveMode::new(options).expect("mode")
    }
}
