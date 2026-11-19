//! Port of packages/tui/src/components/editor.ts

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::components::select_list::{
    SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme,
};
use crate::editor_component::EditorPasteSnapshot;
use crate::keybindings::{get_keybindings, KeybindingsManager};
use crate::keys::{decode_printable_key, matches_key};
use crate::kill_ring::KillRing;
use crate::slash_command_context::{get_slash_command_context, SlashCommandContext};
use crate::terminal::Terminal;
use crate::tui::{Component, Focusable, OverlayHandle, OverlayOptions, SizeValue, TUI, CURSOR_MARKER};
use crate::undo_stack::UndoStack;
use crate::utils::{graphemes, is_punctuation_char, is_whitespace_char, truncate_to_width, visible_width};

/// Regex matching paste markers like `[paste #1 +123 lines]` or `[paste #2 1234 chars]`.
const PASTE_MARKER_PREFIX: &str = "[paste #";
const IMAGE_MARKER_PREFIX: &str = "[image #";

/// Non-global version for single-segment testing.
fn is_paste_marker_single(segment: &str) -> bool {
    let Some(rest) = segment.strip_prefix("[paste #") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(']') else {
        return false;
    };
    // `\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]`
    let (digits, suffix) = match rest.find(' ') {
        Some(index) => (&rest[..index], &rest[index + 1..]),
        None => (rest, ""),
    };
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    if suffix.is_empty() {
        return true;
    }
    if let Some(lines) = suffix.strip_prefix('+') {
        return lines.ends_with(" lines") && lines[..lines.len() - 6].chars().all(|ch| ch.is_ascii_digit());
    }
    if let Some(chars) = suffix.strip_suffix(" chars") {
        return !chars.is_empty() && chars.chars().all(|ch| ch.is_ascii_digit());
    }
    false
}

fn is_image_marker_single(segment: &str) -> bool {
    let Some(rest) = segment.strip_prefix("[image #") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(']') else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
}

/// Check if a segment is an atomic marker (paste or image) merged by segmentWithMarkers.
fn is_atomic_marker(segment: &str) -> bool {
    segment.chars().count() >= 10 && (is_paste_marker_single(segment) || is_image_marker_single(segment))
}

/// Port of `segmentWithMarkers`. Returns `(segment, byte index)` pairs.
///
/// A segmenter that wraps the grapheme segmenter and merges graphemes that fall
/// within paste or image markers into single atomic segments. This makes cursor
/// movement, deletion, word-wrap, etc. treat the markers as single units.
///
/// Paste markers are only merged when their numeric ID exists in `valid_paste_ids`
/// (so a stale `[paste #N]` typed by the user isn't treated as atomic). Image
/// markers are self-contained and always merged.
pub fn segment_with_markers(text: &str, valid_paste_ids: &[i64]) -> Vec<(String, usize)> {
    let has_paste = !valid_paste_ids.is_empty() && text.contains(PASTE_MARKER_PREFIX);
    let has_image = text.contains(IMAGE_MARKER_PREFIX);

    if !has_paste && !has_image {
        return graphemes_with_indices(text);
    }

    let mut markers: Vec<(usize, usize)> = Vec::new();
    if has_paste {
        for (start, end, id) in find_markers(text, PASTE_MARKER_PREFIX) {
            if valid_paste_ids.contains(&id) {
                markers.push((start, end));
            }
        }
    }
    if has_image {
        for (start, end, _) in find_markers(text, IMAGE_MARKER_PREFIX) {
            markers.push((start, end));
        }
    }
    if markers.is_empty() {
        return graphemes_with_indices(text);
    }
    markers.sort_by_key(|(start, _)| *start);

    let base_segments = graphemes_with_indices(text);
    let mut result: Vec<(String, usize)> = Vec::new();
    let mut marker_idx = 0usize;

    for (segment, index) in base_segments {
        while marker_idx < markers.len() && markers[marker_idx].1 <= index {
            marker_idx += 1;
        }

        let marker = markers.get(marker_idx).copied();

        match marker {
            Some((start, end)) if index >= start && index < end => {
                if index == start {
                    let marker_text = text[start..end].to_string();
                    result.push((marker_text, start));
                }
            }
            _ => result.push((segment, index)),
        }
    }

    result
}

/// Finds marker spans for a prefix like `[paste #` / `[image #`.
/// Returns `(start, end, id)` where `id` is parsed from the digits after the prefix.
fn find_markers(text: &str, prefix: &str) -> Vec<(usize, usize, i64)> {
    let mut markers: Vec<(usize, usize, i64)> = Vec::new();
    let mut search_from = 0usize;
    while let Some(relative) = text[search_from..].find(prefix) {
        let start = search_from + relative;
        let digits_start = start + prefix.len();
        let mut digits_end = digits_start;
        while digits_end < text.len()
            && text.as_bytes()[digits_end].is_ascii_digit()
        {
            digits_end += 1;
        }
        if digits_end == digits_start {
            search_from = digits_start;
            continue;
        }
        let id: i64 = text[digits_start..digits_end].parse().unwrap_or(0);
        // Optional " (+N lines|N chars)" suffix, then `]`.
        let mut cursor = digits_end;
        let mut end: Option<usize> = None;
        if text[cursor..].starts_with(']') {
            end = Some(cursor + 1);
        } else if text[cursor..].starts_with(' ') {
            // `( (\+\d+ lines|\d+ chars))?` - a space, then either "+N lines" or "N chars".
            cursor += 1;
            let has_plus = text[cursor..].starts_with('+');
            if has_plus {
                cursor += 1;
            }
            let number_start = cursor;
            while cursor < text.len() && text.as_bytes()[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor > number_start {
                if has_plus && text[cursor..].starts_with(" lines]") {
                    end = Some(cursor + 7);
                } else if !has_plus && text[cursor..].starts_with(" chars]") {
                    end = Some(cursor + 7);
                }
            }
        }
        match end {
            Some(end) => {
                markers.push((start, end, id));
                search_from = end;
            }
            None => search_from = digits_start,
        }
    }
    markers
}

/// Port of `[...segmenter.segment(text)]` returning `(segment, index)` pairs.
fn graphemes_with_indices(text: &str) -> Vec<(String, usize)> {
    let mut result: Vec<(String, usize)> = Vec::new();
    let mut byte_index = 0usize;
    for segment in graphemes(text) {
        result.push((segment.clone(), byte_index));
        byte_index += segment.len();
    }
    result
}

/// Port of the `TextChunk` interface.
#[derive(Clone, Debug, PartialEq)]
pub struct TextChunk {
    pub text: String,
    pub start_index: usize,
    pub end_index: usize,
}

/// Split a line into word-wrapped chunks.
/// Wraps at word boundaries when possible, falling back to character-level
/// wrapping for words longer than the available width.
pub fn word_wrap_line(line: &str, max_width: usize, pre_segmented: Option<&[(String, usize)]>) -> Vec<TextChunk> {
    if line.is_empty() || max_width == 0 {
        return vec![TextChunk {
            text: String::new(),
            start_index: 0,
            end_index: 0,
        }];
    }

    let line_width = visible_width(line);
    if line_width <= max_width {
        return vec![TextChunk {
            text: line.to_string(),
            start_index: 0,
            end_index: line.len(),
        }];
    }

    let mut chunks: Vec<TextChunk> = Vec::new();
    let owned_segments = graphemes_with_indices(line);
    let segments: &[(String, usize)] = pre_segmented.unwrap_or(&owned_segments);

    let mut current_width = 0usize;
    let mut chunk_start = 0usize;

    // Wrap opportunity: the position after the last whitespace before a non-whitespace
    // grapheme, i.e. where a line break is allowed.
    let mut wrap_opp_index: Option<usize> = None;
    let mut wrap_opp_width = 0usize;

    for i in 0..segments.len() {
        let (grapheme, char_index) = &segments[i];
        let g_width = visible_width(grapheme);
        let is_ws = !is_atomic_marker(grapheme) && is_whitespace_char(grapheme);

        // Overflow check before advancing.
        if current_width + g_width > max_width {
            match wrap_opp_index {
                Some(opp_index) if current_width + g_width - wrap_opp_width <= max_width => {
                    // Backtrack to last wrap opportunity (the remaining content
                    // plus the current grapheme still fits within maxWidth).
                    chunks.push(TextChunk {
                        text: line[chunk_start..opp_index].to_string(),
                        start_index: chunk_start,
                        end_index: opp_index,
                    });
                    chunk_start = opp_index;
                    current_width -= wrap_opp_width;
                }
                _ if chunk_start < *char_index => {
                    // No viable wrap opportunity: force-break at current position.
                    // This also handles the case where backtracking to a word
                    // boundary wouldn't help because the remaining content plus
                    // the current grapheme (e.g. a wide character) still exceeds
                    // maxWidth.
                    chunks.push(TextChunk {
                        text: line[chunk_start..*char_index].to_string(),
                        start_index: chunk_start,
                        end_index: *char_index,
                    });
                    chunk_start = *char_index;
                    current_width = 0;
                }
                _ => {}
            }
            wrap_opp_index = None;
        }

        if g_width > max_width {
            // Single atomic segment wider than maxWidth (e.g. paste marker
            // in a narrow terminal). Re-wrap it at grapheme granularity.

            // The segment remains logically atomic for cursor
            // movement / editing - the split is purely visual for word-wrap layout.
            let sub_chunks = word_wrap_line(grapheme, max_width, None);
            for sub_chunk in sub_chunks.iter().take(sub_chunks.len().saturating_sub(1)) {
                chunks.push(TextChunk {
                    text: sub_chunk.text.clone(),
                    start_index: char_index + sub_chunk.start_index,
                    end_index: char_index + sub_chunk.end_index,
                });
            }
            let last = sub_chunks.last().unwrap();
            chunk_start = char_index + last.start_index;
            current_width = visible_width(&last.text);
            wrap_opp_index = None;
            continue;
        }

        // Advance.
        current_width += g_width;

        // Record wrap opportunity: whitespace followed by non-whitespace.
        // Multiple spaces join (no break between them); the break point is
        // after the last space before the next word.
        if is_ws {
            if let Some((next_segment, next_index)) = segments.get(i + 1) {
                if is_atomic_marker(next_segment) || !is_whitespace_char(next_segment) {
                    wrap_opp_index = Some(*next_index);
                    wrap_opp_width = current_width;
                }
            }
        }
    }

    chunks.push(TextChunk {
        text: line[chunk_start..].to_string(),
        start_index: chunk_start,
        end_index: line.len(),
    });

    chunks
}

/// Port of the `EditorState` interface.
#[derive(Clone, Default)]
struct EditorState {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_col: usize,
}

/// Port of the `EditorUndoSnapshot` interface.
#[derive(Clone, Default)]
struct EditorUndoSnapshot {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_col: usize,
    pastes: Vec<(i64, String)>,
    paste_counter: i64,
}

/// Port of the `LayoutLine` interface.
#[derive(Clone, Debug, PartialEq)]
struct LayoutLine {
    text: String,
    has_cursor: bool,
    cursor_pos: Option<usize>,
    /// Logical source line index this layout line renders.
    source_line: usize,
    /// Start offset of this layout line's text within the source line.
    source_start: usize,
}

/// Port of the `lastAction` union: "kill" | "yank" | "type-word" | null.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LastAction {
    Kill,
    Yank,
    TypeWord,
}

/// Port of `EditorTheme`.
pub struct EditorTheme {
    pub border_color: Rc<dyn Fn(&str) -> String>,
    pub background_color: Option<Rc<dyn Fn(&str) -> String>>,
    pub autocomplete_background_color: Option<Rc<dyn Fn(&str) -> String>>,
    pub select_list: SelectListTheme,
    pub command_color: Option<Rc<dyn Fn(&str) -> String>>,
}

/// Port of `EditorOptions`.
#[derive(Default)]
pub struct EditorOptions {
    pub padding_x: Option<f64>,
    pub autocomplete_max_visible: Option<f64>,
    pub prompt_prefix: Option<String>,
}

/// Port of `AutocompleteSuggestions` from `packages/tui/src/autocomplete.ts`.
/// Defined locally because `crate::autocomplete` is owned by another slice; see
/// `blocked_on` in evidence/status/tui-components.json.
#[derive(Clone, Default)]
pub struct AutocompleteSuggestions {
    pub items: Vec<SelectItem>,
    pub prefix: String,
    /// "slash-command" | "file" | "attachment" | undefined
    pub kind: Option<String>,
}

/// The autocomplete contract the editor needs. Declared here so the editor can be
/// ported before `crate::autocomplete` lands; see blocked_on for the slice handoff.
pub trait EditorAutocompleteProvider {
    fn get_suggestions(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        force: bool,
    ) -> Option<AutocompleteSuggestions>;
    fn apply_completion(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &SelectItem,
        prefix: &str,
    ) -> ApplyCompletionResult;
    fn should_trigger_file_completion(&self, lines: &[String], cursor_line: usize, cursor_col: usize) -> bool {
        let _ = (lines, cursor_line, cursor_col);
        true
    }
}

/// Port of the `applyCompletion` result shape.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyCompletionResult {
    pub lines: Vec<String>,
    pub cursor_line: usize,
    pub cursor_col: usize,
}

/// Port of `SLASH_COMMAND_SELECT_LIST_LAYOUT`.
fn slash_command_select_list_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        truncate_primary: None,
        show_item_metadata: true,
        show_directional_scroll_info: true,
        show_selected_description: true,
    }
}

const ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS: u64 = 20;

thread_local! {
    static AUTOCOMPLETE_ANCHOR_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn next_autocomplete_anchor_id() -> u64 {
    AUTOCOMPLETE_ANCHOR_ID.with(|id| {
        let next = id.get() + 1;
        id.set(next);
        next
    })
}

/// Port of the `autocompleteState` union: "regular" | "force" | null.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AutocompleteState {
    Regular,
    Force,
}

pub struct Editor {
    state: EditorState,

    focused: bool,

    tui: Rc<RefCell<TUI>>,
    terminal_rows: usize,
    theme: EditorTheme,
    padding_x: usize,
    prompt_prefix: String,

    last_width: usize,

    scroll_offset: usize,

    pub border_color: Rc<dyn Fn(&str) -> String>,
    pub background_color: Option<Rc<dyn Fn(&str) -> String>>,
    pub autocomplete_background_color: Option<Rc<dyn Fn(&str) -> String>>,
    pub command_color: Option<Rc<dyn Fn(&str) -> String>>,

    autocomplete_provider: Option<Rc<RefCell<dyn EditorAutocompleteProvider>>>,
    autocomplete_list: Option<SelectList>,
    autocomplete_state: Option<AutocompleteState>,
    autocomplete_prefix: String,
    autocomplete_kind: Option<String>,
    autocomplete_max_visible: usize,
    autocomplete_aborted: bool,
    autocomplete_start_token: u64,
    autocomplete_request_id: u64,
    autocomplete_overlay: Option<OverlayHandle>,
    /// Backs the overlay's `visible` predicate:
    /// `() => this.focused && this.autocompleteState !== null`
    /// (packages/tui/src/components/editor.ts:2353). The `TUI` evaluates the
    /// predicate on every frame, so the editor cannot pass `&self`; this cell is
    /// the shared truth it reads.
    autocomplete_visibility: Rc<Cell<bool>>,
    /// Buffer the overlay component returns from `render`. The editor fills it
    /// during its own render pass; see [`EditorOverlayComponent`].
    autocomplete_overlay_lines: Rc<RefCell<Vec<String>>>,
    pending_autocomplete: Option<PendingAutocomplete>,
    autocomplete_anchor_marker: String,

    pastes: Vec<(i64, String)>,
    paste_counter: i64,

    paste_buffer: String,
    is_in_paste: bool,

    history: Vec<String>,
    history_index: i64, // -1 = not browsing, 0 = most recent, 1 = older, etc.

    kill_ring: KillRing,
    last_action: Option<LastAction>,

    jump_mode: Option<bool>, // true = forward, false = backward

    preferred_visual_col: Option<usize>,

    // When the cursor is snapped to the start of an atomic segment, e.g. a
    // paste marker, cursorCol no longer reflects where the cursor would have
    // landed. This field stores the pre-snap cursorCol so that the next
    // vertical move can resolve it to a visual column on whatever VL it belongs
    // to.
    snapped_from_cursor_col: Option<usize>,

    undo_stack: UndoStack<EditorUndoSnapshot>,

    pub on_submit: Option<Box<dyn FnMut(&str)>>,
    pub on_change: Option<Box<dyn FnMut(&str)>>,
    pub disable_submit: bool,
}

impl Editor {
    pub fn new(tui: Rc<RefCell<TUI>>, theme: EditorTheme, options: EditorOptions) -> Self {
        let terminal_rows = tui.borrow().terminal_rows();
        let border_color = Rc::clone(&theme.border_color);
        let background_color = theme.background_color.clone();
        let autocomplete_background_color = theme.autocomplete_background_color.clone();
        let command_color = theme.command_color.clone();
        let padding_x = match options.padding_x {
            Some(padding) if padding.is_finite() => padding.max(0.0).floor() as usize,
            _ => 0,
        };
        let prompt_prefix = options.prompt_prefix.unwrap_or_default();
        let max_visible = options.autocomplete_max_visible.unwrap_or(5.0);
        let autocomplete_max_visible = if max_visible.is_finite() {
            max_visible.max(3.0).min(20.0).floor() as usize
        } else {
            5
        };

        Self {
            state: EditorState {
                lines: vec![String::new()],
                cursor_line: 0,
                cursor_col: 0,
            },
            focused: false,
            tui,
            terminal_rows,
            theme,
            padding_x,
            prompt_prefix,
            last_width: 80,
            scroll_offset: 0,
            border_color,
            background_color,
            autocomplete_background_color,
            command_color,
            autocomplete_provider: None,
            autocomplete_list: None,
            autocomplete_state: None,
            autocomplete_prefix: String::new(),
            autocomplete_kind: None,
            autocomplete_max_visible,
            autocomplete_aborted: false,
            autocomplete_start_token: 0,
            autocomplete_request_id: 0,
            autocomplete_overlay: None,
            autocomplete_visibility: Rc::new(Cell::new(false)),
            autocomplete_overlay_lines: Rc::new(RefCell::new(Vec::new())),
            pending_autocomplete: None,
            autocomplete_anchor_marker: format!("\x1b_pi:autocomplete:{}\x07", next_autocomplete_anchor_id()),
            pastes: Vec::new(),
            paste_counter: 0,
            paste_buffer: String::new(),
            is_in_paste: false,
            history: Vec::new(),
            history_index: -1,
            kill_ring: KillRing::new(),
            last_action: None,
            jump_mode: None,
            preferred_visual_col: None,
            snapped_from_cursor_col: None,
            undo_stack: UndoStack::new(),
            on_submit: None,
            on_change: None,
            disable_submit: false,
        }
    }

    /// Set of currently valid paste IDs, for marker-aware segmentation.
    fn valid_paste_ids(&self) -> Vec<i64> {
        self.pastes.iter().map(|(id, _)| *id).collect()
    }

    /// Segment text with paste-marker awareness, only merging markers with valid IDs.
    fn segment(&self, text: &str) -> Vec<(String, usize)> {
        segment_with_markers(text, &self.valid_paste_ids())
    }

    pub fn get_padding_x(&self) -> usize {
        self.padding_x
    }

    /// The UI host updates this on resize, before borrowing the TUI to render.
    /// Rendering must not re-borrow the owner that is traversing its children.
    pub fn set_terminal_rows(&mut self, rows: usize) {
        self.terminal_rows = rows.max(1);
    }

    fn current_terminal_rows(&self) -> usize {
        self.tui.try_borrow().map(|ui| ui.terminal_rows()).unwrap_or(self.terminal_rows)
    }

    pub fn set_padding_x(&mut self, padding: f64) {
        let new_padding = if padding.is_finite() {
            padding.max(0.0).floor() as usize
        } else {
            0
        };
        if self.padding_x != new_padding {
            self.padding_x = new_padding;
            self.tui.borrow_mut().request_render();
        }
    }

    pub fn get_autocomplete_max_visible(&self) -> usize {
        self.autocomplete_max_visible
    }

    pub fn set_autocomplete_max_visible(&mut self, max_visible: f64) {
        let new_max_visible = if max_visible.is_finite() {
            max_visible.max(3.0).min(20.0).floor() as usize
        } else {
            5
        };
        if self.autocomplete_max_visible != new_max_visible {
            self.autocomplete_max_visible = new_max_visible;
            self.tui.borrow_mut().request_render();
        }
    }

    fn get_prompt_prefix(&self) -> String {
        self.prompt_prefix.clone()
    }

    fn format_prompt_prefix(&self, prefix: &str) -> String {
        prefix.to_string()
    }

    /// Extension hook: number of hidden characters at the start of a line.
    fn get_hidden_text_prefix_length(&self, _line_index: usize, _line: &str) -> f64 {
        0.0
    }

    /// Extension hook: style the display text of one layout line.
    #[allow(clippy::too_many_arguments)]
    fn style_display_text(
        &self,
        display_text: String,
        _layout_line_index: usize,
        _line_text: &str,
        _cursor_col: Option<usize>,
        _source_line: Option<usize>,
        _source_start: Option<usize>,
    ) -> String {
        display_text
    }

    fn get_line_hidden_text_prefix_length(&self, line_index: usize, line: &str) -> usize {
        let hidden_length = self.get_hidden_text_prefix_length(line_index, line);
        if !hidden_length.is_finite() {
            return 0;
        }
        (hidden_length.max(0.0).floor() as usize).min(line.len())
    }

    pub fn set_autocomplete_provider(&mut self, provider: Rc<RefCell<dyn EditorAutocompleteProvider>>) {
        self.cancel_autocomplete();
        self.autocomplete_provider = Some(provider);
    }

    /// Add a prompt to history for up/down arrow navigation.
    /// Called after successful submission.
    pub fn add_to_history(&mut self, text: &str) {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        if !self.history.is_empty() && self.history[0] == trimmed {
            return;
        }
        self.history.insert(0, trimmed);
        if self.history.len() > 100 {
            self.history.pop();
        }
    }

    /// Prompt history entries (most recent first).
    pub fn get_history(&self) -> &[String] {
        &self.history
    }

    /// Clear prompt history (e.g. when switching to a different session).
    pub fn clear_history(&mut self) {
        self.history = Vec::new();
        self.history_index = -1;
    }

    fn is_editor_empty(&self) -> bool {
        self.state.lines.len() == 1 && self.state.lines[0].is_empty()
    }

    fn is_on_first_visual_line(&self) -> bool {
        let visual_lines = self.build_visual_line_map(self.last_width);
        let current_visual_line = self.find_current_visual_line(&visual_lines);
        current_visual_line == 0
    }

    fn is_on_last_visual_line(&self) -> bool {
        let visual_lines = self.build_visual_line_map(self.last_width);
        let current_visual_line = self.find_current_visual_line(&visual_lines);
        current_visual_line + 1 == visual_lines.len()
    }

    pub fn is_history_navigation_active(&self) -> bool {
        self.history_index > -1
    }

    fn navigate_history(&mut self, direction: i64) {
        self.last_action = None;
        if self.history.is_empty() {
            return;
        }

        let new_index = self.history_index - direction; // Up(-1) increases index, Down(1) decreases
        if new_index < -1 || new_index >= self.history.len() as i64 {
            return;
        }

        if self.history_index == -1 && new_index >= 0 {
            self.push_undo_snapshot();
        }

        self.history_index = new_index;

        if self.history_index == -1 {
            self.set_text_internal("");
        } else {
            let text = self
                .history
                .get(self.history_index as usize)
                .cloned()
                .unwrap_or_default();
            self.set_text_internal(&text);
        }
    }

    /// Internal setText that doesn't reset history state - used by navigateHistory
    fn set_text_internal(&mut self, text: &str) {
        let lines: Vec<String> = text.split('\n').map(|line| line.to_string()).collect();
        self.state.lines = if lines.is_empty() { vec![String::new()] } else { lines };
        self.state.cursor_line = self.state.lines.len() - 1;
        let col = self.state.lines[self.state.cursor_line].len();
        self.set_cursor_col(col);
        self.scroll_offset = 0;

        self.emit_change();
    }

    fn emit_change(&mut self) {
        if self.on_change.is_none() {
            return;
        }
        let text = self.get_text();
        if let Some(callback) = self.on_change.as_mut() {
            callback(&text);
        }
    }

    fn emit_submit(&mut self, text: &str) {
        if let Some(callback) = self.on_submit.as_mut() {
            callback(text);
        }
    }

    fn get_render_metrics(&self, width: usize) -> RenderMetrics {
        let max_padding = (width as i64 - 1).max(0) as usize / 2;
        let use_background_surface = self.background_color.is_some();
        let configured_padding_x = self.padding_x.min(max_padding);
        let padding_x = if use_background_surface {
            configured_padding_x.max(2).min(max_padding)
        } else {
            configured_padding_x
        };
        let content_width = width.saturating_sub(padding_x * 2).max(1);
        let prompt_prefix_text = self.get_prompt_prefix();
        let prompt_prefix_width = visible_width(&prompt_prefix_text).min(content_width.saturating_sub(1));

        RenderMetrics {
            use_background_surface,
            padding_x,
            prompt_prefix_text,
            prompt_prefix_width,
            input_width: content_width.saturating_sub(prompt_prefix_width).max(1),
        }
    }

    fn get_autocomplete_anchor_marker(&self) -> String {
        if self.autocomplete_overlay.is_some() && self.focused {
            self.autocomplete_anchor_marker.clone()
        } else {
            String::new()
        }
    }

    /// Keep the overlay's `visible` predicate in sync with
    /// `this.focused && this.autocompleteState !== null`
    /// (packages/tui/src/components/editor.ts:2353).
    fn sync_autocomplete_visibility(&self) {
        self.autocomplete_visibility
            .set(self.focused && self.autocomplete_state.is_some());
    }

    fn render_autocomplete_overlay(&mut self, width: usize) -> Vec<String> {
        if !(self.autocomplete_state.is_some() && self.autocomplete_list.is_some()) {
            return Vec::new();
        }

        let metrics = self.get_render_metrics(width);
        let left_padding = " ".repeat(metrics.padding_x + metrics.prompt_prefix_width);
        let right_padding = " ".repeat(metrics.padding_x);
        let background_color = self
            .autocomplete_background_color
            .clone()
            .or_else(|| {
                if metrics.use_background_surface {
                    self.background_color.clone()
                } else {
                    None
                }
            });

        let list_lines = self
            .autocomplete_list
            .as_mut()
            .map(|list| list.render(metrics.input_width as f64))
            .unwrap_or_default();

        let mut source: Vec<String> = vec![String::new()];
        source.extend(list_lines);
        source.push(String::new());

        source
            .into_iter()
            .map(|line| {
                let line_padding = " ".repeat(metrics.input_width.saturating_sub(visible_width(&line)));
                let content_line = format!("{left_padding}{line}{line_padding}{right_padding}");
                match &background_color {
                    Some(background_color) => background_color(&content_line),
                    None => content_line,
                }
            })
            .collect()
    }
}

/// Port of the `getRenderMetrics` return shape.
struct RenderMetrics {
    use_background_surface: bool,
    padding_x: usize,
    prompt_prefix_text: String,
    prompt_prefix_width: usize,
    input_width: usize,
}

impl Editor {
    fn layout_text(&self, content_width: usize) -> Vec<LayoutLine> {
        let mut layout_lines: Vec<LayoutLine> = Vec::new();

        if self.state.lines.is_empty() || (self.state.lines.len() == 1 && self.state.lines[0].is_empty()) {
            layout_lines.push(LayoutLine {
                text: String::new(),
                has_cursor: true,
                cursor_pos: Some(0),
                source_line: 0,
                source_start: 0,
            });
            return layout_lines;
        }

        for i in 0..self.state.lines.len() {
            let line = self.state.lines[i].clone();
            let hidden_prefix_length = self.get_line_hidden_text_prefix_length(i, &line);
            let display_line = line[hidden_prefix_length.min(line.len())..].to_string();
            let is_current_line = i == self.state.cursor_line;
            let line_visible_width = visible_width(&display_line);

            if display_line.is_empty() {
                layout_lines.push(LayoutLine {
                    text: String::new(),
                    has_cursor: is_current_line,
                    cursor_pos: if is_current_line { Some(0) } else { None },
                    source_line: i,
                    source_start: hidden_prefix_length,
                });
                continue;
            }

            if line_visible_width <= content_width {
                if is_current_line {
                    layout_lines.push(LayoutLine {
                        text: display_line,
                        has_cursor: true,
                        cursor_pos: Some(self.state.cursor_col.saturating_sub(hidden_prefix_length)),
                        source_line: i,
                        source_start: hidden_prefix_length,
                    });
                } else {
                    layout_lines.push(LayoutLine {
                        text: display_line,
                        has_cursor: false,
                        cursor_pos: None,
                        source_line: i,
                        source_start: hidden_prefix_length,
                    });
                }
            } else {
                let segments = self.segment(&display_line);
                let chunks = word_wrap_line(&display_line, content_width, Some(&segments));

                for (chunk_index, chunk) in chunks.iter().enumerate() {
                    let cursor_pos = self.state.cursor_col.saturating_sub(hidden_prefix_length);
                    let is_last_chunk = chunk_index == chunks.len() - 1;

                    // For word-wrapped chunks, we need to handle the case where
                    // cursor might be in trimmed whitespace at end of chunk
                    let mut has_cursor_in_chunk = false;
                    let mut adjusted_cursor_pos = 0usize;

                    if is_current_line {
                        if is_last_chunk {
                            has_cursor_in_chunk = cursor_pos >= chunk.start_index;
                            adjusted_cursor_pos = cursor_pos.saturating_sub(chunk.start_index);
                        } else {
                            has_cursor_in_chunk = cursor_pos >= chunk.start_index && cursor_pos < chunk.end_index;
                            if has_cursor_in_chunk {
                                adjusted_cursor_pos = cursor_pos.saturating_sub(chunk.start_index);
                                if adjusted_cursor_pos > chunk.text.len() {
                                    adjusted_cursor_pos = chunk.text.len();
                                }
                            }
                        }
                    }

                    layout_lines.push(LayoutLine {
                        text: chunk.text.clone(),
                        has_cursor: has_cursor_in_chunk,
                        cursor_pos: if has_cursor_in_chunk {
                            Some(adjusted_cursor_pos)
                        } else {
                            None
                        },
                        source_line: i,
                        source_start: hidden_prefix_length + chunk.start_index,
                    });
                }
            }
        }

        layout_lines
    }

    pub fn get_text(&self) -> String {
        self.state.lines.join("\n")
    }

    fn expand_paste_markers(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (paste_id, paste_content) in &self.pastes {
            // `new RegExp(\\[paste #<id>( (\\+N lines|N chars))?\\], "g")`
            result = replace_paste_marker(&result, *paste_id, paste_content);
        }
        result
    }

    /// Get text with paste markers expanded to their actual content.
    /// Use this when you need the full content (e.g., for external editor).
    pub fn get_expanded_text(&self) -> String {
        self.expand_paste_markers(&self.state.lines.join("\n"))
    }

    pub fn get_paste_snapshot(&self) -> EditorPasteSnapshot {
        EditorPasteSnapshot {
            pastes: self.pastes.clone(),
            paste_counter: self.paste_counter,
        }
    }

    pub fn restore_paste_snapshot(&mut self, snapshot: EditorPasteSnapshot) {
        self.pastes = snapshot.pastes;
        self.paste_counter = snapshot.paste_counter;
    }

    pub fn get_lines(&self) -> Vec<String> {
        self.state.lines.clone()
    }

    /// Port of `getCursor(): { line, col }`.
    pub fn get_cursor(&self) -> (usize, usize) {
        (self.state.cursor_line, self.state.cursor_col)
    }

    fn char_before_cursor(&self) -> Option<char> {
        let current_line = self.state.lines.get(self.state.cursor_line)?;
        let prefix = current_line.get(..self.state.cursor_col.min(current_line.len()))?;
        prefix.chars().next_back()
    }

    pub fn set_text(&mut self, text: &str) {
        self.cancel_autocomplete();
        self.last_action = None;
        self.history_index = -1; // Exit history browsing mode
        let normalized = self.normalize_text(text);
        if self.get_text() != normalized {
            self.push_undo_snapshot();
        }
        self.set_text_internal(&normalized);
    }

    /// Insert text at the current cursor position.
    /// Used for programmatic insertion (e.g., clipboard image markers).
    /// This is atomic for undo - single undo restores entire pre-insert state.
    pub fn insert_text_at_cursor(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.cancel_autocomplete();
        self.push_undo_snapshot();
        self.last_action = None;
        self.history_index = -1;
        self.insert_text_at_cursor_internal(text);
    }

    /// Normalize text for editor storage:
    /// - Normalize line endings (\r\n and \r -> \n)
    /// - Expand tabs to 4 spaces
    fn normalize_text(&self, text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n").replace('\t', "    ")
    }

    /// Internal text insertion at cursor. Handles single and multi-line text.
    /// Does not push undo snapshots or trigger autocomplete - caller is responsible.
    /// Normalizes line endings and calls onChange once at the end.
    fn insert_text_at_cursor_internal(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        let normalized = self.normalize_text(text);
        let inserted_lines: Vec<String> = normalized.split('\n').map(|line| line.to_string()).collect();

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let before_cursor = current_line[..self.state.cursor_col.min(current_line.len())].to_string();
        let after_cursor = current_line[self.state.cursor_col.min(current_line.len())..].to_string();

        if inserted_lines.len() == 1 {
            self.state.lines[self.state.cursor_line] = format!("{before_cursor}{normalized}{after_cursor}");
            let col = self.state.cursor_col + normalized.len();
            self.set_cursor_col(col);
        } else {
            let mut lines: Vec<String> = Vec::new();
            lines.extend(self.state.lines[..self.state.cursor_line].iter().cloned());
            lines.push(format!("{before_cursor}{}", inserted_lines[0]));
            lines.extend(
                inserted_lines[1..inserted_lines.len() - 1]
                    .iter()
                    .cloned(),
            );
            lines.push(format!(
                "{}{after_cursor}",
                inserted_lines[inserted_lines.len() - 1]
            ));
            lines.extend(self.state.lines[self.state.cursor_line + 1..].iter().cloned());
            self.state.lines = lines;

            self.state.cursor_line += inserted_lines.len() - 1;
            let col = inserted_lines[inserted_lines.len() - 1].len();
            self.set_cursor_col(col);
        }

        self.emit_change();
    }

    fn insert_character(&mut self, char: &str, skip_undo_coalescing: bool) {
        self.history_index = -1; // Exit history browsing mode

        // Undo coalescing (fish-style):
        // - Consecutive word chars coalesce into one undo unit
        // - Space captures state before itself (so undo removes space+following word together)
        // - Each space is separately undoable
        // Skip coalescing when called from atomic operations (e.g., handlePaste)
        if !skip_undo_coalescing {
            if is_whitespace_char(char) || self.last_action != Some(LastAction::TypeWord) {
                self.push_undo_snapshot();
            }
            self.last_action = Some(LastAction::TypeWord);
        }

        let line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        let col = self.state.cursor_col.min(line.len());
        let before = &line[..col];
        let after = &line[col..];

        self.state.lines[self.state.cursor_line] = format!("{before}{char}{after}");
        self.set_cursor_col(self.state.cursor_col + char.len());

        self.emit_change();

        if self.autocomplete_state.is_none() {
            let slash_context = self.get_current_slash_command_context();
            if char == "/" && matches!(slash_context, Some(SlashCommandContext::Name { .. })) {
                self.try_trigger_autocomplete(false);
            } else if char == "@" || char == "#" {
                let current_line = self
                    .state
                    .lines
                    .get(self.state.cursor_line)
                    .cloned()
                    .unwrap_or_default();
                let text_before_cursor = &current_line[..self.state.cursor_col.min(current_line.len())];
                let chars: Vec<char> = text_before_cursor.chars().collect();
                let char_before_symbol = if chars.len() >= 2 {
                    Some(chars[chars.len() - 2])
                } else {
                    None
                };
                if chars.len() == 1 || char_before_symbol == Some(' ') || char_before_symbol == Some('\t') {
                    self.try_trigger_autocomplete(false);
                }
            } else if is_autocomplete_word_char(char) {
                let current_line = self
                    .state
                    .lines
                    .get(self.state.cursor_line)
                    .cloned()
                    .unwrap_or_default();
                let text_before_cursor = &current_line[..self.state.cursor_col.min(current_line.len())];
                if slash_context.is_some() {
                    self.try_trigger_autocomplete(false);
                } else if matches_symbol_context(text_before_cursor) {
                    self.try_trigger_autocomplete(false);
                }
            }
        } else {
            self.refresh_autocomplete_after_edit(false);
        }
    }
}

/// Single-pass port of the global paste-marker replacement for one paste ID.
fn replace_paste_marker(text: &str, paste_id: i64, content: &str) -> String {
    let prefix = format!("[paste #{paste_id}");
    let mut out = String::new();
    let mut search_from = 0usize;
    while let Some(relative) = text[search_from..].find(&prefix) {
        let start = search_from + relative;
        let after_prefix = start + prefix.len();
        let mut cursor = after_prefix;
        let mut end: Option<usize> = None;
        if text[cursor..].starts_with(']') {
            end = Some(cursor + 1);
        } else if text[cursor..].starts_with(' ') {
            // `( (\+\d+ lines|\d+ chars))?` - a space, then either "+N lines" or "N chars".
            cursor += 1;
            let has_plus = text[cursor..].starts_with('+');
            if has_plus {
                cursor += 1;
            }
            let number_start = cursor;
            while cursor < text.len() && text.as_bytes()[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor > number_start {
                if has_plus && text[cursor..].starts_with(" lines]") {
                    end = Some(cursor + 7);
                } else if !has_plus && text[cursor..].starts_with(" chars]") {
                    end = Some(cursor + 7);
                }
            }
        }
        match end {
            Some(end) => {
                out.push_str(&text[search_from..start]);
                out.push_str(content);
                search_from = end;
            }
            None => {
                out.push_str(&text[search_from..after_prefix]);
                search_from = after_prefix;
            }
        }
    }
    out.push_str(&text[search_from..]);
    out
}

/// Port of `/[a-zA-Z0-9.\-_]/.test(char)`.
fn is_autocomplete_word_char(char: &str) -> bool {
    char.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_')
        && !char.is_empty()
}

/// Port of `/(?:^|[\s])[@#][^\s]*$/`.
fn matches_symbol_context(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut index = chars.len();
    while index > 0 {
        index -= 1;
        match chars[index] {
            // `[^\s]*` may be empty, so the `@`/`#` itself can be the last character.
            '@' | '#' => {
                return index == 0 || chars[index - 1].is_whitespace();
            }
            ch if ch.is_whitespace() => return false,
            _ => {}
        }
    }
    false
}

/// Port of `/(?:^|[ \t])(?:@(?:"[^"]*|[^\s]*)|#[^\s]*)$/`.
fn matches_attachment_context(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut index = chars.len();
    while index > 0 {
        index -= 1;
        let ch = chars[index];
        if ch == '#' && index > 0 && chars[index - 1] == '@' {
            // `@(?:"[^"]*|[^\s]*)`: after an `@` the rest is `[^\s]*`, so the
            // `#` is part of the same token and never its own match.
            return false;
        }
        if ch == '@' || ch == '#' {
            return index == 0 || chars[index - 1] == ' ' || chars[index - 1] == '\t';
        }
        if ch.is_whitespace() {
            return false;
        }
    }
    false
}

impl Editor {
    fn handle_paste(&mut self, pasted_text: &str) {
        self.cancel_autocomplete();
        self.history_index = -1; // Exit history browsing mode
        self.last_action = None;

        self.push_undo_snapshot();

        // Some terminals (e.g. tmux popups with extended-keys-format=csi-u) re-encode
        // control bytes inside bracketed paste as CSI-u Ctrl+<letter> sequences
        // (ESC [ <codepoint> ; 5 u). Decode those back to their literal byte so the
        // per-char filter below preserves newlines instead of stripping ESC and
        // leaking the printable tail (e.g. "[106;5u") into the editor.
        let decoded_text = decode_csi_u_ctrl(pasted_text);

        let clean_text = self.normalize_text(&decoded_text);

        let mut filtered_text: String = clean_text
            .chars()
            .filter(|char| *char == '\n' || (*char as u32) >= 32)
            .collect();

        // If pasting a file path (starts with /, ~, or .) and the character before
        // the cursor is a word character, prepend a space for better readability
        if filtered_text.starts_with('/') || filtered_text.starts_with('~') || filtered_text.starts_with('.') {
            if self
                .char_before_cursor()
                .map(|character| is_word_char(&character.to_string()))
                .unwrap_or(false)
            {
                filtered_text = format!(" {filtered_text}");
            }
        }

        // Split into lines to check for large paste
        let pasted_lines: Vec<&str> = filtered_text.split('\n').collect();

        // Check if this is a large paste (> 10 lines or > 1000 characters)
        let total_chars = filtered_text.chars().count();
        if pasted_lines.len() > 10 || total_chars > 1000 {
            self.paste_counter += 1;
            let paste_id = self.paste_counter;
            self.pastes.retain(|(id, _)| *id != paste_id);
            self.pastes.push((paste_id, filtered_text.clone()));

            let marker = if pasted_lines.len() > 10 {
                format!("[paste #{paste_id} +{} lines]", pasted_lines.len())
            } else {
                format!("[paste #{paste_id} {total_chars} chars]")
            };
            self.insert_text_at_cursor_internal(&marker);
            return;
        }

        if pasted_lines.len() == 1 {
            self.insert_text_at_cursor_internal(&filtered_text);
            return;
        }

        // Multi-line paste - use direct state manipulation
        self.insert_text_at_cursor_internal(&filtered_text);
    }

    fn add_new_line(&mut self) {
        self.cancel_autocomplete();
        self.history_index = -1; // Exit history browsing mode
        self.last_action = None;

        self.push_undo_snapshot();

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        let col = self.state.cursor_col.min(current_line.len());
        let before = current_line[..col].to_string();
        let after = current_line[col..].to_string();

        self.state.lines[self.state.cursor_line] = before;
        self.state.lines.insert(self.state.cursor_line + 1, after);

        self.state.cursor_line += 1;
        self.set_cursor_col(0);

        self.emit_change();
    }

    fn should_submit_on_backslash_enter(&self, data: &str, kb: &KeybindingsManager) -> bool {
        if self.disable_submit {
            return false;
        }
        if !matches_key(data, "enter") {
            return false;
        }
        let submit_keys = kb.get_keys("tui.input.submit");
        let has_shift_enter = submit_keys
            .iter()
            .any(|key| key == "shift+enter" || key == "shift+return");
        if !has_shift_enter {
            return false;
        }

        self.char_before_cursor() == Some('\\')
    }

    fn submit_value(&mut self) {
        self.cancel_autocomplete();
        let result = self.expand_paste_markers(&self.state.lines.join("\n")).trim().to_string();

        self.state = EditorState {
            lines: vec![String::new()],
            cursor_line: 0,
            cursor_col: 0,
        };
        self.pastes.clear();
        self.paste_counter = 0;
        self.history_index = -1;
        self.scroll_offset = 0;
        self.undo_stack.clear();
        self.last_action = None;

        self.emit_change();
        self.emit_submit(&result);
    }

    fn handle_backspace(&mut self) {
        self.history_index = -1; // Exit history browsing mode
        self.last_action = None;

        let line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let line_start_col = self.get_line_hidden_text_prefix_length(self.state.cursor_line, &line);

        if self.state.cursor_col > line_start_col {
            self.push_undo_snapshot();

            let before_cursor = line[line_start_col..self.state.cursor_col.min(line.len())].to_string();

            let segments = self.segment(&before_cursor);
            let grapheme_length = segments
                .last()
                .map(|(segment, _)| segment.len())
                .unwrap_or(1);

            let cut = self.state.cursor_col.saturating_sub(grapheme_length);
            let before = &line[..cut];
            let after = &line[self.state.cursor_col.min(line.len())..];

            self.state.lines[self.state.cursor_line] = format!("{before}{after}");
            self.set_cursor_col(cut);
        } else if line_start_col > 0 && line.len() == line_start_col {
            self.push_undo_snapshot();

            self.state.lines[self.state.cursor_line] = String::new();
            self.set_cursor_col(0);
        } else if self.state.cursor_line > 0 {
            self.push_undo_snapshot();

            let current_line = self
                .state
                .lines
                .get(self.state.cursor_line)
                .cloned()
                .unwrap_or_default();
            let previous_line = self
                .state
                .lines
                .get(self.state.cursor_line - 1)
                .cloned()
                .unwrap_or_default();

            self.state.lines[self.state.cursor_line - 1] = format!("{previous_line}{current_line}");
            self.state.lines.remove(self.state.cursor_line);

            self.state.cursor_line -= 1;
            self.set_cursor_col(previous_line.len());
        }

        self.emit_change();

        self.refresh_autocomplete_after_edit(true);
    }

    /// Set cursor column and clear preferredVisualCol.
    /// Use this for all non-vertical cursor movements to reset sticky column behavior.
    fn set_cursor_col(&mut self, col: usize) {
        self.state.cursor_col = col;
        self.preferred_visual_col = None;
        self.snapped_from_cursor_col = None;
    }

    fn visual_line_bounds(&self, visual_line: &VisualLine) -> (usize, usize) {
        let line = self
            .state
            .lines
            .get(visual_line.logical_line)
            .map(String::as_str)
            .unwrap_or("");
        let start = visual_line.start_col.min(line.len());
        let end = start.saturating_add(visual_line.length).min(line.len());
        (start, end)
    }

    fn visual_column_at_byte(&self, visual_line: &VisualLine, byte_col: usize) -> usize {
        let line = self
            .state
            .lines
            .get(visual_line.logical_line)
            .map(String::as_str)
            .unwrap_or("");
        let (start, end) = self.visual_line_bounds(visual_line);
        let cursor = byte_col.clamp(start, end);
        line.get(start..cursor).map(visible_width).unwrap_or(0)
    }

    fn visual_line_max_cursor_column(
        &self,
        visual_line: &VisualLine,
        is_last_segment: bool,
    ) -> usize {
        let line = self
            .state
            .lines
            .get(visual_line.logical_line)
            .map(String::as_str)
            .unwrap_or("");
        let (start, end) = self.visual_line_bounds(visual_line);
        let Some(text) = line.get(start..end) else {
            return 0;
        };
        let width = visible_width(text);
        if is_last_segment {
            return width;
        }
        // A wrap boundary belongs to the next visual line. Exclude the final
        // grapheme's cells instead of assuming that one byte equals one column.
        let last_width = self
            .segment(text)
            .last()
            .map(|(segment, _)| visible_width(segment))
            .unwrap_or(0);
        width.saturating_sub(last_width)
    }

    fn byte_at_visual_column(&self, visual_line: &VisualLine, visual_col: usize) -> usize {
        let line = self
            .state
            .lines
            .get(visual_line.logical_line)
            .map(String::as_str)
            .unwrap_or("");
        let (start, end) = self.visual_line_bounds(visual_line);
        let Some(text) = line.get(start..end) else {
            return start;
        };
        if visual_col == 0 {
            return start;
        }
        let mut width = 0usize;
        for (segment, index) in graphemes_with_indices(text) {
            width += visible_width(&segment);
            if width >= visual_col {
                return start + index + segment.len();
            }
        }
        end
    }

    /// Move cursor to a target visual line, applying sticky column logic.
    /// Shared by moveCursor() and pageScroll().
    fn move_to_visual_line(
        &mut self,
        visual_lines: &[VisualLine],
        current_visual_line: usize,
        target_visual_line: usize,
    ) {
        let current_vl = match visual_lines.get(current_visual_line) {
            Some(vl) => vl.clone(),
            None => return,
        };
        let target_vl = match visual_lines.get(target_visual_line) {
            Some(vl) => vl.clone(),
            None => return,
        };

        // When the cursor was snapped to a segment start, resolve the pre-snap
        // position against the VL it belongs to. This gives the correct visual
        // column even after a resize reshuffles VLs.
        let current_visual_col: usize;
        if let Some(snapped) = self.snapped_from_cursor_col {
            let vl_index = self.find_visual_line_at(visual_lines, current_vl.logical_line, snapped);
            current_visual_col = self.visual_column_at_byte(&visual_lines[vl_index], snapped);
        } else {
            current_visual_col = self.visual_column_at_byte(&current_vl, self.state.cursor_col);
        }

        // For non-last segments, clamp to length-1 to stay within the segment
        let is_last_source_segment = current_visual_line == visual_lines.len() - 1
            || visual_lines
                .get(current_visual_line + 1)
                .map(|vl| vl.logical_line != current_vl.logical_line)
                .unwrap_or(true);
        let source_max_visual_col =
            self.visual_line_max_cursor_column(&current_vl, is_last_source_segment);

        let is_last_target_segment = target_visual_line == visual_lines.len() - 1
            || visual_lines
                .get(target_visual_line + 1)
                .map(|vl| vl.logical_line != target_vl.logical_line)
                .unwrap_or(true);
        let target_max_visual_col =
            self.visual_line_max_cursor_column(&target_vl, is_last_target_segment);

        let move_to_visual_col = self.compute_vertical_move_column(
            current_visual_col,
            source_max_visual_col,
            target_max_visual_col,
        );

        self.state.cursor_line = target_vl.logical_line;
        let logical_line = self
            .state
            .lines
            .get(target_vl.logical_line)
            .cloned()
            .unwrap_or_default();
        self.state.cursor_col = self.byte_at_visual_column(&target_vl, move_to_visual_col);
        // A display column can fall inside a wide grapheme. Preserve the desired
        // cell so a later ASCII line can restore it instead of drifting.
        if self.visual_column_at_byte(&target_vl, self.state.cursor_col) != move_to_visual_col {
            self.preferred_visual_col = Some(move_to_visual_col);
        }

        // Snap cursor to atomic segment boundary (e.g. paste markers)
        // so the cursor never lands in the middle of a multi-grapheme unit.
        // Single-grapheme segments don't need snapping.
        let segments = self.segment(&logical_line);
        for (segment, index) in segments {
            if index > self.state.cursor_col {
                break;
            }
            if segment.len() <= 1 {
                continue;
            }
            if self.state.cursor_col < index + segment.len() {
                let is_continuation = index < target_vl.start_col;
                let is_moving_down = target_visual_line > current_visual_line;

                if is_continuation && is_moving_down {
                    // The segment started on a previous visual line, and we
                    // already visited it on the way down. Skip all remaining
                    // continuation VLs and land on the first VL past it.
                    let seg_end = index + segment.len();
                    let mut next = target_visual_line + 1;
                    while next < visual_lines.len()
                        && visual_lines[next].logical_line == target_vl.logical_line
                        && visual_lines[next].start_col < seg_end
                    {
                        next += 1;
                    }
                    if next < visual_lines.len() {
                        self.move_to_visual_line(visual_lines, current_visual_line, next);
                        return;
                    }
                }

                // Snap to the start of the segment so it gets highlighted.
                // Store the pre-snap position so the next vertical move can
                // resolve it to the correct visual column.
                self.snapped_from_cursor_col = Some(self.state.cursor_col);
                self.state.cursor_col = index;
                return;
            }
        }

        // No snap occurred - we moved out of the atomic segment.
        self.snapped_from_cursor_col = None;
    }

    /// Compute the target visual column for vertical cursor movement.
    /// Implements the sticky column decision table:
    ///
    /// | P | S | T | U | Scenario                                             | Set Preferred | Move To     |
    /// |---|---|---|---| ---------------------------------------------------- |---------------|-------------|
    /// | 0 | * | 0 | - | Start nav, target fits                               | null          | current     |
    /// | 0 | * | 1 | - | Start nav, target shorter                            | current       | target end  |
    /// | 1 | 0 | 0 | 0 | Clamped, target fits preferred                       | null          | preferred   |
    /// | 1 | 0 | 0 | 1 | Clamped, target longer but still can't fit preferred | keep          | target end  |
    /// | 1 | 0 | 1 | - | Clamped, target even shorter                         | keep          | target end  |
    /// | 1 | 1 | 0 | - | Rewrapped, target fits current                       | null          | current     |
    /// | 1 | 1 | 1 | - | Rewrapped, target shorter than current               | current       | target end  |
    ///
    /// Where:
    /// - P = preferred col is set
    /// - S = cursor in middle of source line (not clamped to end)
    /// - T = target line shorter than current visual col
    /// - U = target line shorter than preferred col
    fn compute_vertical_move_column(
        &mut self,
        current_visual_col: usize,
        source_max_visual_col: usize,
        target_max_visual_col: usize,
    ) -> usize {
        let has_preferred = self.preferred_visual_col.is_some(); // P
        let cursor_in_middle = current_visual_col < source_max_visual_col; // S
        let target_too_short = target_max_visual_col < current_visual_col; // T

        if !has_preferred || cursor_in_middle {
            if target_too_short {
                // Cases 2 and 7
                self.preferred_visual_col = Some(current_visual_col);
                return target_max_visual_col;
            }

            // Cases 1 and 6
            self.preferred_visual_col = None;
            return current_visual_col;
        }

        let preferred = self.preferred_visual_col.unwrap();
        let target_cant_fit_preferred = target_max_visual_col < preferred; // U
        if target_too_short || target_cant_fit_preferred {
            // Cases 4 and 5
            return target_max_visual_col;
        }

        // Case 3
        let result = preferred;
        self.preferred_visual_col = None;
        result
    }
}

/// Port of the visual line map entry: `{ logicalLine, startCol, length }`.
#[derive(Clone, Debug, PartialEq)]
struct VisualLine {
    logical_line: usize,
    start_col: usize,
    length: usize,
}

/// Port of `/\x1b\[(\d+);5u/g` -> literal control byte.
fn decode_csi_u_ctrl(text: &str) -> String {
    let mut result = String::new();
    let mut index = 0usize;
    while index < text.len() {
        let rest = &text[index..];
        if rest.starts_with("\x1b[") {
            let after = &rest[2..];
            let digits_len = after
                .as_bytes()
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
            if digits_len > 0 && after[digits_len..].starts_with(";5u") {
                let code: u32 = after[..digits_len].parse().unwrap_or(0);
                if (97..=122).contains(&code) {
                    result.push(char::from_u32(code - 96).unwrap_or('?'));
                    index += 2 + digits_len + 3;
                    continue;
                }
                if (65..=90).contains(&code) {
                    result.push(char::from_u32(code - 64).unwrap_or('?'));
                    index += 2 + digits_len + 3;
                    continue;
                }
            }
        }
        let ch = rest.chars().next().unwrap();
        result.push(ch);
        index += ch.len_utf8();
    }
    result
}

/// Port of `/\w/.test(charBeforeCursor)`.
fn is_word_char(char: &str) -> bool {
    char.chars()
        .next()
        .map(|ch| ch.is_alphanumeric() || ch == '_')
        .unwrap_or(false)
}

impl Editor {
    fn move_to_line_start(&mut self) {
        self.last_action = None;
        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let col = self.get_line_hidden_text_prefix_length(self.state.cursor_line, &current_line);
        self.set_cursor_col(col);
    }

    fn move_to_line_end(&mut self) {
        self.last_action = None;
        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        self.set_cursor_col(current_line.len());
    }

    fn delete_to_start_of_line(&mut self) {
        self.history_index = -1; // Exit history browsing mode

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let line_start_col = self.get_line_hidden_text_prefix_length(self.state.cursor_line, &current_line);

        if self.state.cursor_col > line_start_col {
            self.push_undo_snapshot();

            // Calculate text to be deleted and save to kill ring (backward deletion = prepend)
            let deleted_text =
                current_line[line_start_col..self.state.cursor_col.min(current_line.len())].to_string();
            self.kill_ring.push(
                &deleted_text,
                true,
                self.last_action == Some(LastAction::Kill),
            );
            self.last_action = Some(LastAction::Kill);

            self.state.lines[self.state.cursor_line] = format!(
                "{}{}",
                &current_line[..line_start_col],
                &current_line[self.state.cursor_col.min(current_line.len())..]
            );
            self.set_cursor_col(line_start_col);
        } else if self.state.cursor_line > 0 {
            self.push_undo_snapshot();

            // At start of line - merge with previous line, treating newline as deleted text
            self.kill_ring
                .push("\n", true, self.last_action == Some(LastAction::Kill));
            self.last_action = Some(LastAction::Kill);

            let previous_line = self
                .state
                .lines
                .get(self.state.cursor_line - 1)
                .cloned()
                .unwrap_or_default();
            self.state.lines[self.state.cursor_line - 1] = format!("{previous_line}{current_line}");
            self.state.lines.remove(self.state.cursor_line);
            self.state.cursor_line -= 1;
            self.set_cursor_col(previous_line.len());
        }

        self.emit_change();
        self.refresh_autocomplete_after_edit(false);
    }

    fn delete_to_end_of_line(&mut self) {
        self.history_index = -1; // Exit history browsing mode

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        if self.state.cursor_col < current_line.len() {
            self.push_undo_snapshot();

            // Calculate text to be deleted and save to kill ring (forward deletion = append)
            let deleted_text = current_line[self.state.cursor_col.min(current_line.len())..].to_string();
            self.kill_ring.push(
                &deleted_text,
                false,
                self.last_action == Some(LastAction::Kill),
            );
            self.last_action = Some(LastAction::Kill);

            self.state.lines[self.state.cursor_line] =
                current_line[..self.state.cursor_col.min(current_line.len())].to_string();
        } else if self.state.cursor_line + 1 < self.state.lines.len() {
            self.push_undo_snapshot();

            // At end of line - merge with next line, treating newline as deleted text
            self.kill_ring
                .push("\n", false, self.last_action == Some(LastAction::Kill));
            self.last_action = Some(LastAction::Kill);

            let next_line = self
                .state
                .lines
                .get(self.state.cursor_line + 1)
                .cloned()
                .unwrap_or_default();
            self.state.lines[self.state.cursor_line] = format!("{current_line}{next_line}");
            self.state.lines.remove(self.state.cursor_line + 1);
        }

        self.emit_change();
        self.refresh_autocomplete_after_edit(false);
    }

    fn delete_word_backwards(&mut self) {
        self.history_index = -1; // Exit history browsing mode

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        // If at start of line, behave like backspace at column 0 (merge with previous line)
        if self.state.cursor_col == 0 {
            if self.state.cursor_line > 0 {
                self.push_undo_snapshot();

                // Treat newline as deleted text (backward deletion = prepend)
                self.kill_ring
                    .push("\n", true, self.last_action == Some(LastAction::Kill));
                self.last_action = Some(LastAction::Kill);

                let previous_line = self
                    .state
                    .lines
                    .get(self.state.cursor_line - 1)
                    .cloned()
                    .unwrap_or_default();
                self.state.lines[self.state.cursor_line - 1] = format!("{previous_line}{current_line}");
                self.state.lines.remove(self.state.cursor_line);
                self.state.cursor_line -= 1;
                self.set_cursor_col(previous_line.len());
            }
        } else {
            self.push_undo_snapshot();

            // Save lastAction before cursor movement (moveWordBackwards resets it)
            let was_kill = self.last_action == Some(LastAction::Kill);

            let old_cursor_col = self.state.cursor_col;
            self.move_word_backwards();
            let delete_from = self.state.cursor_col;
            self.set_cursor_col(old_cursor_col);

            let deleted_text =
                current_line[delete_from.min(current_line.len())..self.state.cursor_col.min(current_line.len())]
                    .to_string();
            self.kill_ring.push(&deleted_text, true, was_kill);
            self.last_action = Some(LastAction::Kill);

            self.state.lines[self.state.cursor_line] = format!(
                "{}{}",
                &current_line[..delete_from.min(current_line.len())],
                &current_line[self.state.cursor_col.min(current_line.len())..]
            );
            self.set_cursor_col(delete_from);
        }

        self.emit_change();
        self.refresh_autocomplete_after_edit(false);
    }

    fn delete_word_forward(&mut self) {
        self.history_index = -1; // Exit history browsing mode

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        // If at end of line, merge with next line (delete the newline)
        if self.state.cursor_col >= current_line.len() {
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.push_undo_snapshot();

                // Treat newline as deleted text (forward deletion = append)
                self.kill_ring
                    .push("\n", false, self.last_action == Some(LastAction::Kill));
                self.last_action = Some(LastAction::Kill);

                let next_line = self
                    .state
                    .lines
                    .get(self.state.cursor_line + 1)
                    .cloned()
                    .unwrap_or_default();
                self.state.lines[self.state.cursor_line] = format!("{current_line}{next_line}");
                self.state.lines.remove(self.state.cursor_line + 1);
            }
        } else {
            self.push_undo_snapshot();

            // Save lastAction before cursor movement (moveWordForwards resets it)
            let was_kill = self.last_action == Some(LastAction::Kill);

            let old_cursor_col = self.state.cursor_col;
            self.move_word_forwards();
            let delete_to = self.state.cursor_col;
            self.set_cursor_col(old_cursor_col);

            let deleted_text =
                current_line[self.state.cursor_col.min(current_line.len())..delete_to.min(current_line.len())]
                    .to_string();
            self.kill_ring.push(&deleted_text, false, was_kill);
            self.last_action = Some(LastAction::Kill);

            self.state.lines[self.state.cursor_line] = format!(
                "{}{}",
                &current_line[..self.state.cursor_col.min(current_line.len())],
                &current_line[delete_to.min(current_line.len())..]
            );
        }

        self.emit_change();
        self.refresh_autocomplete_after_edit(false);
    }

    fn handle_forward_delete(&mut self) {
        self.history_index = -1; // Exit history browsing mode
        self.last_action = None;

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        if self.state.cursor_col < current_line.len() {
            self.push_undo_snapshot();

            // Delete grapheme at cursor position (handles emojis, combining characters, etc.)
            let after_cursor = current_line[self.state.cursor_col.min(current_line.len())..].to_string();

            let segments = self.segment(&after_cursor);
            let grapheme_length = segments.first().map(|(segment, _)| segment.len()).unwrap_or(1);

            let before = &current_line[..self.state.cursor_col.min(current_line.len())];
            let after = &current_line[(self.state.cursor_col + grapheme_length).min(current_line.len())..];
            self.state.lines[self.state.cursor_line] = format!("{before}{after}");
        } else if self.state.cursor_line + 1 < self.state.lines.len() {
            self.push_undo_snapshot();

            // At end of line - merge with next line
            let next_line = self
                .state
                .lines
                .get(self.state.cursor_line + 1)
                .cloned()
                .unwrap_or_default();
            self.state.lines[self.state.cursor_line] = format!("{current_line}{next_line}");
            self.state.lines.remove(self.state.cursor_line + 1);
        }

        self.emit_change();

        self.refresh_autocomplete_after_edit(true);
    }

    /// Build a mapping from visual lines to logical positions.
    /// Returns an array where each element represents a visual line with:
    /// - logicalLine: index into this.state.lines
    /// - startCol: starting column in the logical line
    /// - length: length of this visual line segment
    fn build_visual_line_map(&self, width: usize) -> Vec<VisualLine> {
        let mut visual_lines: Vec<VisualLine> = Vec::new();

        for i in 0..self.state.lines.len() {
            let line = self.state.lines[i].clone();
            let hidden_prefix_length = self.get_line_hidden_text_prefix_length(i, &line);
            let display_line = line[hidden_prefix_length.min(line.len())..].to_string();
            let line_vis_width = visible_width(&display_line);
            if display_line.is_empty() {
                visual_lines.push(VisualLine {
                    logical_line: i,
                    start_col: hidden_prefix_length,
                    length: 0,
                });
            } else if line_vis_width <= width {
                visual_lines.push(VisualLine {
                    logical_line: i,
                    start_col: hidden_prefix_length,
                    length: display_line.len(),
                });
            } else {
                let segments = self.segment(&display_line);
                for chunk in word_wrap_line(&display_line, width, Some(&segments)) {
                    visual_lines.push(VisualLine {
                        logical_line: i,
                        start_col: hidden_prefix_length + chunk.start_index,
                        length: chunk.end_index - chunk.start_index,
                    });
                }
            }
        }

        visual_lines
    }

    /// Find the visual line index that contains the given logical position.
    fn find_visual_line_at(&self, visual_lines: &[VisualLine], line: usize, col: usize) -> usize {
        let hidden_prefix_length = self.get_line_hidden_text_prefix_length(
            line,
            self.state.lines.get(line).map(|s| s.as_str()).unwrap_or(""),
        );
        for (i, vl) in visual_lines.iter().enumerate() {
            if vl.logical_line != line {
                continue;
            }
            if hidden_prefix_length > 0 && col < hidden_prefix_length && vl.start_col == hidden_prefix_length {
                return i;
            }
            let offset = col as i64 - vl.start_col as i64;
            // Cursor is in this segment if it's within range. For the last
            // segment of a logical line, cursor can be at length (end position)
            let is_last_segment_of_line = i == visual_lines.len() - 1
                || visual_lines[i + 1].logical_line != vl.logical_line;
            if offset >= 0 && (offset < vl.length as i64 || (is_last_segment_of_line && offset == vl.length as i64))
            {
                return i;
            }
        }
        visual_lines.len() - 1
    }

    /// Find the visual line index for the current cursor position.
    fn find_current_visual_line(&self, visual_lines: &[VisualLine]) -> usize {
        self.find_visual_line_at(visual_lines, self.state.cursor_line, self.state.cursor_col)
    }
}

impl Editor {
    fn move_cursor(&mut self, delta_line: i64, delta_col: i64) {
        self.last_action = None;
        let visual_lines = self.build_visual_line_map(self.last_width);
        let current_visual_line = self.find_current_visual_line(&visual_lines);

        if delta_line != 0 {
            let target_visual_line = current_visual_line as i64 + delta_line;

            if target_visual_line >= 0 && (target_visual_line as usize) < visual_lines.len() {
                self.move_to_visual_line(&visual_lines, current_visual_line, target_visual_line as usize);
            }
        }

        if delta_col != 0 {
            let current_line = self
                .state
                .lines
                .get(self.state.cursor_line)
                .cloned()
                .unwrap_or_default();
            let line_start_col = self.get_line_hidden_text_prefix_length(self.state.cursor_line, &current_line);

            if delta_col > 0 {
                // Moving right - move by one grapheme (handles emojis, combining characters, etc.)
                if self.state.cursor_col < current_line.len() {
                    let after_cursor = current_line[self.state.cursor_col.min(current_line.len())..].to_string();
                    let segments = self.segment(&after_cursor);
                    let step = segments.first().map(|(segment, _)| segment.len()).unwrap_or(1);
                    let col = self.state.cursor_col + step;
                    self.set_cursor_col(col);
                } else if self.state.cursor_line + 1 < self.state.lines.len() {
                    self.state.cursor_line += 1;
                    self.set_cursor_col(0);
                } else if let Some(current_vl) = visual_lines.get(current_visual_line) {
                    self.preferred_visual_col = Some(self.visual_column_at_byte(current_vl, self.state.cursor_col));
                }
            } else {
                // Moving left - move by one grapheme (handles emojis, combining characters, etc.)
                if self.state.cursor_col > line_start_col {
                    let before_cursor =
                        current_line[line_start_col..self.state.cursor_col.min(current_line.len())].to_string();
                    let segments = self.segment(&before_cursor);
                    let step = segments.last().map(|(segment, _)| segment.len()).unwrap_or(1);
                    let previous_col = self.state.cursor_col.saturating_sub(step);
                    self.set_cursor_col(previous_col.max(line_start_col));
                } else if self.state.cursor_line > 0 {
                    self.state.cursor_line -= 1;
                    let prev_line = self
                        .state
                        .lines
                        .get(self.state.cursor_line)
                        .cloned()
                        .unwrap_or_default();
                    self.set_cursor_col(prev_line.len());
                }
            }
        }
    }

    /// Scroll by a page (direction: -1 for up, 1 for down).
    /// Moves cursor by the page size while keeping it in bounds.
    fn page_scroll(&mut self, direction: i64) {
        self.last_action = None;
        let terminal_rows = self.current_terminal_rows();
        let page_size = ((terminal_rows as f64 * 0.3).floor() as i64).max(5);

        let visual_lines = self.build_visual_line_map(self.last_width);
        let current_visual_line = self.find_current_visual_line(&visual_lines);
        let target_visual_line = (current_visual_line as i64 + direction * page_size)
            .max(0)
            .min(visual_lines.len() as i64 - 1) as usize;

        self.move_to_visual_line(&visual_lines, current_visual_line, target_visual_line);
    }

    fn move_word_backwards(&mut self) {
        self.last_action = None;
        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let line_start_col = self.get_line_hidden_text_prefix_length(self.state.cursor_line, &current_line);

        if self.state.cursor_col <= line_start_col {
            if self.state.cursor_line > 0 {
                self.state.cursor_line -= 1;
                let prev_line = self
                    .state
                    .lines
                    .get(self.state.cursor_line)
                    .cloned()
                    .unwrap_or_default();
                self.set_cursor_col(prev_line.len());
            }
            return;
        }

        let text_before_cursor =
            current_line[line_start_col..self.state.cursor_col.min(current_line.len())].to_string();
        let mut segments = self.segment(&text_before_cursor);
        let mut new_col = self.state.cursor_col;

        while let Some((segment, _)) = segments.last() {
            if is_atomic_marker(segment) || !is_whitespace_char(segment) {
                break;
            }
            let (segment, _) = segments.pop().unwrap();
            new_col -= segment.len();
        }

        if let Some((last_grapheme, _)) = segments.last().cloned() {
            if is_atomic_marker(&last_grapheme) {
                let (segment, _) = segments.pop().unwrap();
                new_col -= segment.len();
            } else if is_punctuation_char(&last_grapheme) {
                while let Some((segment, _)) = segments.last() {
                    if !is_punctuation_char(segment) || is_atomic_marker(segment) {
                        break;
                    }
                    let (segment, _) = segments.pop().unwrap();
                    new_col -= segment.len();
                }
            } else {
                while let Some((segment, _)) = segments.last() {
                    if is_whitespace_char(segment) || is_punctuation_char(segment) || is_atomic_marker(segment) {
                        break;
                    }
                    let (segment, _) = segments.pop().unwrap();
                    new_col -= segment.len();
                }
            }
        }

        self.set_cursor_col(new_col.max(line_start_col));
    }

    /// Yank (paste) the most recent kill ring entry at cursor position.
    fn yank(&mut self) {
        if self.kill_ring.len() == 0 {
            return;
        }

        self.push_undo_snapshot();

        let text = self.kill_ring.peek().unwrap_or_default();
        self.insert_yanked_text(&text);

        self.last_action = Some(LastAction::Yank);
        self.refresh_autocomplete_after_edit(false);
    }

    /// Cycle through kill ring (only works immediately after yank or yank-pop).
    /// Replaces the last yanked text with the previous entry in the ring.
    fn yank_pop(&mut self) {
        if self.last_action != Some(LastAction::Yank) || self.kill_ring.len() <= 1 {
            return;
        }

        self.push_undo_snapshot();

        self.delete_yanked_text();

        self.kill_ring.rotate();

        let text = self.kill_ring.peek().unwrap_or_default();
        self.insert_yanked_text(&text);

        self.last_action = Some(LastAction::Yank);
        self.refresh_autocomplete_after_edit(false);
    }

    /// Insert text at cursor position (used by yank operations).
    fn insert_yanked_text(&mut self, text: &str) {
        self.history_index = -1; // Exit history browsing mode
        let lines: Vec<String> = text.split('\n').map(|line| line.to_string()).collect();

        if lines.len() == 1 {
            let current_line = self
                .state
                .lines
                .get(self.state.cursor_line)
                .cloned()
                .unwrap_or_default();
            let col = self.state.cursor_col.min(current_line.len());
            let before = &current_line[..col];
            let after = &current_line[col..];
            self.state.lines[self.state.cursor_line] = format!("{before}{text}{after}");
            self.set_cursor_col(self.state.cursor_col + text.len());
        } else {
            let current_line = self
                .state
                .lines
                .get(self.state.cursor_line)
                .cloned()
                .unwrap_or_default();
            let col = self.state.cursor_col.min(current_line.len());
            let before = &current_line[..col];
            let after = &current_line[col..];

            // First line merges with text before cursor
            self.state.lines[self.state.cursor_line] = format!("{before}{}", lines[0]);

            // Insert middle lines
            for i in 1..lines.len().saturating_sub(1) {
                self.state
                    .lines
                    .insert(self.state.cursor_line + i, lines[i].clone());
            }

            // Last line merges with text after cursor
            let last_line_index = self.state.cursor_line + lines.len() - 1;
            self.state
                .lines
                .insert(last_line_index, format!("{}{after}", lines[lines.len() - 1]));

            // Update cursor position
            self.state.cursor_line = last_line_index;
            self.set_cursor_col(lines[lines.len() - 1].len());
        }

        self.emit_change();
    }

    /// Delete the previously yanked text (used by yank-pop).
    /// The yanked text is derived from killRing[end] since it hasn't been rotated yet.
    fn delete_yanked_text(&mut self) {
        let yanked_text = match self.kill_ring.peek() {
            Some(text) if !text.is_empty() => text,
            _ => return,
        };

        let yank_lines: Vec<String> = yanked_text.split('\n').map(|line| line.to_string()).collect();

        if yank_lines.len() == 1 {
            let current_line = self
                .state
                .lines
                .get(self.state.cursor_line)
                .cloned()
                .unwrap_or_default();
            let delete_len = yanked_text.len();
            let start = self.state.cursor_col.saturating_sub(delete_len);
            let before = &current_line[..start.min(current_line.len())];
            let after = &current_line[self.state.cursor_col.min(current_line.len())..];
            self.state.lines[self.state.cursor_line] = format!("{before}{after}");
            self.set_cursor_col(start);
        } else {
            let start_line = self.state.cursor_line.saturating_sub(yank_lines.len() - 1);
            let start_col = self
                .state
                .lines
                .get(start_line)
                .map(|line| line.len())
                .unwrap_or(0)
                .saturating_sub(yank_lines[0].len());

            // Get text after cursor on current line
            let after_cursor = self
                .state
                .lines
                .get(self.state.cursor_line)
                .map(|line| line[self.state.cursor_col.min(line.len())..].to_string())
                .unwrap_or_default();

            // Get text before yank start position
            let before_yank = self
                .state
                .lines
                .get(start_line)
                .map(|line| line[..start_col.min(line.len())].to_string())
                .unwrap_or_default();

            // Remove all lines from startLine to cursorLine and replace with merged line
            self.state
                .lines
                .splice(start_line..start_line + yank_lines.len(), [format!("{before_yank}{after_cursor}")]);

            // Update cursor
            self.state.cursor_line = start_line;
            self.set_cursor_col(start_col);
        }

        self.emit_change();
    }

    fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(&EditorUndoSnapshot {
            lines: self.state.lines.clone(),
            cursor_line: self.state.cursor_line,
            cursor_col: self.state.cursor_col,
            pastes: self.pastes.clone(),
            paste_counter: self.paste_counter,
        });
    }

    fn undo(&mut self) {
        self.history_index = -1; // Exit history browsing mode
        let snapshot = match self.undo_stack.pop() {
            Some(snapshot) => snapshot,
            None => return,
        };
        self.state = EditorState {
            lines: snapshot.lines,
            cursor_line: snapshot.cursor_line,
            cursor_col: snapshot.cursor_col,
        };
        self.pastes = snapshot.pastes;
        self.paste_counter = snapshot.paste_counter;
        self.last_action = None;
        self.preferred_visual_col = None;
        self.emit_change();
        self.refresh_autocomplete_after_edit(false);
    }

    /// Jump to the first occurrence of a character in the specified direction.
    /// Multi-line search. Case-sensitive. Skips the current cursor position.
    fn jump_to_char(&mut self, char: &str, direction: bool) {
        self.last_action = None;
        let is_forward = direction;
        let lines = self.state.lines.clone();

        if is_forward {
            for line_idx in self.state.cursor_line..lines.len() {
                let line = lines[line_idx].clone();
                let is_current_line = line_idx == self.state.cursor_line;
                let search_from = if is_current_line {
                    Some(self.state.cursor_col + 1)
                } else {
                    None
                };
                if let Some(index) = index_of_from(&line, char, search_from.unwrap_or(0)) {
                    self.state.cursor_line = line_idx;
                    self.set_cursor_col(index);
                    return;
                }
            }
        } else {
            let mut line_idx = self.state.cursor_line as i64;
            while line_idx >= 0 {
                let line = lines[line_idx as usize].clone();
                let is_current_line = line_idx as usize == self.state.cursor_line;
                let search_from = if is_current_line {
                    self.state.cursor_col.checked_sub(1)
                } else {
                    None
                };
                if let Some(index) = last_index_of_from(&line, char, search_from) {
                    self.state.cursor_line = line_idx as usize;
                    self.set_cursor_col(index);
                    return;
                }
                line_idx -= 1;
            }
        }
    }

    fn move_word_forwards(&mut self) {
        self.last_action = None;
        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();

        if self.state.cursor_col >= current_line.len() {
            if self.state.cursor_line + 1 < self.state.lines.len() {
                self.state.cursor_line += 1;
                self.set_cursor_col(0);
            }
            return;
        }

        let text_after_cursor = current_line[self.state.cursor_col.min(current_line.len())..].to_string();
        let segments = self.segment(&text_after_cursor);
        let mut index = 0usize;
        let mut new_col = self.state.cursor_col;

        while index < segments.len()
            && !is_atomic_marker(&segments[index].0)
            && is_whitespace_char(&segments[index].0)
        {
            new_col += segments[index].0.len();
            index += 1;
        }

        if index < segments.len() {
            let first_grapheme = segments[index].0.clone();
            if is_atomic_marker(&first_grapheme) {
                new_col += first_grapheme.len();
            } else if is_punctuation_char(&first_grapheme) {
                while index < segments.len()
                    && is_punctuation_char(&segments[index].0)
                    && !is_atomic_marker(&segments[index].0)
                {
                    new_col += segments[index].0.len();
                    index += 1;
                }
            } else {
                while index < segments.len()
                    && !is_whitespace_char(&segments[index].0)
                    && !is_punctuation_char(&segments[index].0)
                    && !is_atomic_marker(&segments[index].0)
                {
                    new_col += segments[index].0.len();
                    index += 1;
                }
            }
        }

        self.set_cursor_col(new_col);
    }

    fn get_current_slash_command_context(&self) -> Option<SlashCommandContext> {
        get_slash_command_context(&self.state.lines, self.state.cursor_line, self.state.cursor_col)
    }
}

/// Move a byte offset that may sit inside a multi-byte character back to the
/// start of that character, so slicing at the offset can never panic.
///
/// Port of the `from` argument of `String.prototype.indexOf` / `lastIndexOf`:
/// TS (`packages/tui/src/components/editor.ts:2081`) passes `cursorCol + 1`, and
/// the JS engine accepts it because string indices share the same UTF-16 unit as
/// the cursor. Rust cursor columns are byte offsets (`insert_character` does
/// `set_cursor_col(self.state.cursor_col + char.len())`), so an offset derived
/// from a cursor that sits just before a CJK/emoji character can point into the
/// middle of that character. `str::floor_char_boundary` is still unstable, so the
/// same rounding is done here.
fn floor_to_char_boundary(line: &str, index: usize) -> usize {
    let mut safe = index.min(line.len());
    while safe > 0 && !line.is_char_boundary(safe) {
        safe -= 1;
    }
    safe
}

/// Port of `line.indexOf(char, from)`.
fn index_of_from(line: &str, needle: &str, from: usize) -> Option<usize> {
    if from > line.len() {
        return None;
    }
    let from = floor_to_char_boundary(line, from);
    line[from..].find(needle).map(|index| index + from)
}

/// Port of `line.lastIndexOf(char, from)`; `None` searches the whole line.
fn last_index_of_from(line: &str, needle: &str, from: Option<usize>) -> Option<usize> {
    match from {
        Some(from) => {
            let end = floor_to_char_boundary(line, from + needle.len());
            line[..end].rfind(needle)
        }
        None => line.rfind(needle),
    }
}

impl Editor {
    /// Find the best autocomplete item index for the given prefix.
    /// Returns -1 if no match is found.
    ///
    /// Match priority:
    /// 1. Exact match (prefix === item.value) -> always selected
    /// 2. Prefix match -> first item whose value starts with prefix
    /// 3. No match -> -1 (keep default highlight)
    ///
    /// Matching is case-sensitive and checks item.value only.
    fn get_best_autocomplete_match_index(items: &[SelectItem], prefix: &str) -> i64 {
        if prefix.is_empty() {
            return -1;
        }

        let mut first_prefix_index: i64 = -1;

        for (i, item) in items.iter().enumerate() {
            if item.value == prefix {
                return i as i64; // Exact match always wins
            }
            if first_prefix_index == -1 && item.value.starts_with(prefix) {
                first_prefix_index = i as i64;
            }
        }

        first_prefix_index
    }

    fn create_autocomplete_list(&self, suggestions: &AutocompleteSuggestions) -> SelectList {
        let layout = if suggestions.kind.as_deref() == Some("slash-command")
            || (suggestions.kind.is_none() && suggestions.prefix.starts_with('/'))
        {
            Some(slash_command_select_list_layout())
        } else {
            None
        };
        SelectList::new(
            suggestions.items.clone(),
            self.autocomplete_max_visible,
            SelectListTheme {
                selected_prefix: Box::new(|text: &str| text.to_string()),
                selected_text: Box::new(|text: &str| text.to_string()),
                description: Box::new(|text: &str| text.to_string()),
                argument_hint: None,
                source_tag: None,
                scroll_info: Box::new(|text: &str| text.to_string()),
                no_match: Box::new(|text: &str| text.to_string()),
            },
            layout.unwrap_or_default(),
        )
    }

    fn try_trigger_autocomplete(&mut self, explicit_tab: bool) {
        self.request_autocomplete(false, explicit_tab);
    }

    fn handle_tab_completion(&mut self) {
        if self.autocomplete_provider.is_none() {
            return;
        }

        if matches!(
            self.get_current_slash_command_context(),
            Some(SlashCommandContext::Name { .. })
        ) {
            self.handle_slash_command_completion();
        } else {
            self.force_file_autocomplete(true);
        }
    }

    fn handle_slash_command_completion(&mut self) {
        self.request_autocomplete(false, true);
    }

    fn force_file_autocomplete(&mut self, explicit_tab: bool) {
        self.request_autocomplete(true, explicit_tab);
    }

    fn request_autocomplete(&mut self, force: bool, explicit_tab: bool) {
        if self.autocomplete_provider.is_none() {
            return;
        }

        if force {
            let should_trigger = {
                let provider = self.autocomplete_provider.as_ref().unwrap().clone();
                let provider = provider.borrow();
                provider.should_trigger_file_completion(
                    &self.state.lines,
                    self.state.cursor_line,
                    self.state.cursor_col,
                )
            };
            if !should_trigger {
                return;
            }
        }

        self.cancel_autocomplete_request();
        self.autocomplete_start_token += 1;
        let start_token = self.autocomplete_start_token;

        let debounce_ms = self.get_autocomplete_debounce_ms(force, explicit_tab);
        if debounce_ms > 0 {
            // Port of `setTimeout(() => startAutocompleteRequest(...), debounceMs)`:
            // the TUI calls `poll_autocomplete()` on its tick and the due request runs then.
            self.pending_autocomplete = Some(PendingAutocomplete {
                start_token,
                force,
                explicit_tab,
                due: std::time::Instant::now() + std::time::Duration::from_millis(debounce_ms),
            });
            return;
        }

        self.start_autocomplete_request(start_token, force, explicit_tab);
    }

    /// Port of the debounce timer firing. Call from the TUI tick while autocomplete
    /// debounce is pending; a no-op otherwise.
    #[allow(dead_code)]
    pub fn poll_autocomplete(&mut self) {
        let pending = match &self.pending_autocomplete {
            Some(pending) if pending.due <= std::time::Instant::now() => pending.clone(),
            _ => return,
        };
        self.pending_autocomplete = None;
        self.start_autocomplete_request(pending.start_token, pending.force, pending.explicit_tab);
    }

    /// True while a debounced autocomplete request is still waiting for its timer.
    pub fn has_pending_autocomplete(&self) -> bool {
        self.pending_autocomplete.is_some()
    }

    fn start_autocomplete_request(&mut self, start_token: u64, force: bool, explicit_tab: bool) {
        if start_token != self.autocomplete_start_token || self.autocomplete_provider.is_none() {
            return;
        }

        self.autocomplete_aborted = false;
        self.autocomplete_request_id += 1;
        let request_id = self.autocomplete_request_id;
        let snapshot_text = self.get_text();
        let snapshot_line = self.state.cursor_line;
        let snapshot_col = self.state.cursor_col;

        self.run_autocomplete_request(
            request_id,
            snapshot_text,
            snapshot_line,
            snapshot_col,
            force,
            explicit_tab,
        );
    }

    fn get_autocomplete_debounce_ms(&self, force: bool, explicit_tab: bool) -> u64 {
        if explicit_tab || force {
            return 0;
        }

        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let text_before_cursor = &current_line[..self.state.cursor_col.min(current_line.len())];
        if matches_attachment_context(text_before_cursor) {
            ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS
        } else {
            0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_autocomplete_request(
        &mut self,
        request_id: u64,
        snapshot_text: String,
        snapshot_line: usize,
        snapshot_col: usize,
        force: bool,
        explicit_tab: bool,
    ) {
        if self.autocomplete_provider.is_none() {
            return;
        }

        let suggestions = {
            let provider = self.autocomplete_provider.as_ref().unwrap().clone();
            let mut provider = provider.borrow_mut();
            provider.get_suggestions(
                &self.state.lines,
                self.state.cursor_line,
                self.state.cursor_col,
                force,
            )
        };

        if !self.is_autocomplete_request_current(request_id, &snapshot_text, snapshot_line, snapshot_col) {
            return;
        }

        self.autocomplete_aborted = false;

        let suggestions = match suggestions {
            Some(suggestions) if !suggestions.items.is_empty() => suggestions,
            _ => {
                self.cancel_autocomplete();
                self.tui.borrow_mut().request_render();
                return;
            }
        };

        if force && explicit_tab && suggestions.items.len() == 1 {
            let item = suggestions.items[0].clone();
            self.push_undo_snapshot();
            self.last_action = None;
            let result = {
                let provider = self.autocomplete_provider.as_ref().unwrap().clone();
                let mut provider = provider.borrow_mut();
                provider.apply_completion(
                    &self.state.lines,
                    self.state.cursor_line,
                    self.state.cursor_col,
                    &item,
                    &suggestions.prefix,
                )
            };
            self.state.lines = result.lines;
            self.state.cursor_line = result.cursor_line;
            self.set_cursor_col(result.cursor_col);
            self.emit_change();
            self.tui.borrow_mut().request_render();
            return;
        }

        self.apply_autocomplete_suggestions(
            suggestions,
            if force {
                AutocompleteState::Force
            } else {
                AutocompleteState::Regular
            },
        );
        self.tui.borrow_mut().request_render();
    }

    fn is_autocomplete_request_current(
        &self,
        request_id: u64,
        snapshot_text: &str,
        snapshot_line: usize,
        snapshot_col: usize,
    ) -> bool {
        !self.autocomplete_aborted
            && request_id == self.autocomplete_request_id
            && self.get_text() == snapshot_text
            && self.state.cursor_line == snapshot_line
            && self.state.cursor_col == snapshot_col
    }

    fn apply_autocomplete_suggestions(&mut self, suggestions: AutocompleteSuggestions, state: AutocompleteState) {
        self.autocomplete_prefix = suggestions.prefix.clone();
        self.autocomplete_kind = suggestions.kind.clone();
        let mut list = self.create_autocomplete_list(&suggestions);

        let matching_prefix = if suggestions.kind.as_deref() == Some("slash-command") {
            suggestions.prefix.chars().skip(1).collect::<String>()
        } else {
            suggestions.prefix.clone()
        };
        let best_match_index = Self::get_best_autocomplete_match_index(&suggestions.items, &matching_prefix);
        if best_match_index >= 0 {
            list.set_selected_index(best_match_index as usize);
        }
        self.autocomplete_list = Some(list);

        self.autocomplete_state = Some(state);
        self.sync_autocomplete_visibility();
        if self.autocomplete_overlay.is_none() {
            let anchor = self.autocomplete_anchor_marker.clone();
            let visible_state = Rc::clone(&self.autocomplete_visibility);
            let overlay = self.tui.borrow_mut().show_overlay(
                Rc::new(RefCell::new(EditorOverlayComponent {
                    lines: Rc::clone(&self.autocomplete_overlay_lines),
                })) as Rc<RefCell<dyn Component>>,
                OverlayOptions {
                    width: Some(SizeValue::Percent("100%".to_string())),
                    above_marker: Some(anchor),
                    offset_y: Some(-1),
                    non_capturing: true,
                    visible: Some(Rc::new(move |_columns, _rows| visible_state.get())),
                    ..OverlayOptions::default()
                },
            );
            self.autocomplete_overlay = Some(overlay);
        }
    }

    fn cancel_autocomplete_request(&mut self) {
        self.autocomplete_start_token += 1;
        self.pending_autocomplete = None;
        self.autocomplete_aborted = true;
    }

    fn clear_autocomplete_ui(&mut self) {
        if let Some(overlay) = self.autocomplete_overlay.as_mut() {
            overlay.hide();
        }
        self.autocomplete_overlay = None;
        self.autocomplete_state = None;
        self.sync_autocomplete_visibility();
        self.autocomplete_overlay_lines.borrow_mut().clear();
        self.autocomplete_list = None;
        self.autocomplete_prefix = String::new();
        self.autocomplete_kind = None;
    }

    pub fn cancel_autocomplete(&mut self) {
        self.cancel_autocomplete_request();
        self.clear_autocomplete_ui();
    }

    pub fn is_showing_autocomplete(&self) -> bool {
        self.autocomplete_state.is_some()
    }

    fn refresh_autocomplete_after_edit(&mut self, retrigger: bool) {
        let current_line = self
            .state
            .lines
            .get(self.state.cursor_line)
            .cloned()
            .unwrap_or_default();
        let text_before_cursor = &current_line[..self.state.cursor_col.min(current_line.len())];
        let has_completion_context =
            self.get_current_slash_command_context().is_some() || matches_symbol_context_end(text_before_cursor);

        if self.autocomplete_state.is_some() {
            if self.get_text().trim().is_empty()
                || (self.autocomplete_state == Some(AutocompleteState::Regular) && !has_completion_context)
            {
                self.cancel_autocomplete();
                return;
            }
            self.update_autocomplete();
            return;
        }

        if retrigger && has_completion_context {
            self.try_trigger_autocomplete(false);
        }
    }

    fn update_autocomplete(&mut self) {
        if self.autocomplete_state.is_none() || self.autocomplete_provider.is_none() {
            return;
        }
        self.request_autocomplete(self.autocomplete_state == Some(AutocompleteState::Force), false);
    }
}

/// Port of `/(?:^|[\s])[@#][^\s]*$/`.
fn matches_symbol_context_end(text: &str) -> bool {
    matches_symbol_context(text)
}

/// Port of `autocompleteOverlayComponent`.
///
/// The TypeScript component closes over the editor instance
/// (packages/tui/src/components/editor.ts:287-290) and calls
/// `renderAutocompleteOverlay` from its own `render`. Rust cannot alias the
/// owning editor, so the editor paints the dropdown into this shared buffer from
/// its render pass and the overlay component returns that buffer. The buffer is
/// only ever produced by `Editor::render_autocomplete_overlay`, so the overlay
/// and the editor cannot diverge.
struct EditorOverlayComponent {
    lines: Rc<RefCell<Vec<String>>>,
}

impl Component for EditorOverlayComponent {
    fn render(&mut self, _width: f64) -> Vec<String> {
        self.lines.borrow().clone()
    }

    /// Port of `invalidate: () => this.autocompleteList?.invalidate()`
    /// (packages/tui/src/components/editor.ts:289): `SelectList::invalidate` is
    /// a no-op, and the editor refills `lines` on every render pass.
    fn invalidate(&mut self) {}
}

/// Port of the pending `setTimeout` for autocomplete debounce.
#[derive(Clone)]
struct PendingAutocomplete {
    start_token: u64,
    force: bool,
    explicit_tab: bool,
    due: std::time::Instant,
}

impl Focusable for Editor {
    fn focused(&self) -> bool {
        self.focused
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.sync_autocomplete_visibility();
    }
}

impl Component for Editor {
    /// `isFocusable(component)` gate (`packages/tui/src/tui.ts:418-421`): without
    /// this override `TUI::set_focus` finds no `Focusable` and the editor's
    /// `focused` flag is never set, so focus-dependent behaviour (the autocomplete
    /// overlay's `visible` predicate, the anchor marker, the hardware cursor)
    /// silently stays off.
    fn as_focusable(&mut self) -> Option<&mut dyn super::super::tui::Focusable> {
        Some(self)
    }

    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let metrics = self.get_render_metrics(width);
        let prompt_prefix = if metrics.prompt_prefix_width > 0 {
            self.format_prompt_prefix(&truncate_to_width(
                &metrics.prompt_prefix_text,
                metrics.prompt_prefix_width as f64,
                "",
                false,
            ))
        } else {
            String::new()
        };

        // Layout width: with padding the cursor can overflow into it,
        // without padding we reserve 1 column for the cursor.
        let layout_width = metrics
            .input_width
            .saturating_sub(if metrics.padding_x > 0 { 0 } else { 1 })
            .max(1);

        self.last_width = layout_width;

        let horizontal = (self.border_color)("─");

        let layout_lines = self.layout_text(layout_width);

        let terminal_rows = self.current_terminal_rows();
        let max_visible_lines = ((terminal_rows as f64 * 0.3).floor() as usize).max(5);

        let cursor_line_index = layout_lines
            .iter()
            .position(|line| line.has_cursor)
            .unwrap_or(0);

        if cursor_line_index < self.scroll_offset {
            self.scroll_offset = cursor_line_index;
        } else if cursor_line_index >= self.scroll_offset + max_visible_lines {
            self.scroll_offset = cursor_line_index - max_visible_lines + 1;
        }

        let max_scroll_offset = layout_lines.len().saturating_sub(max_visible_lines);
        self.scroll_offset = self.scroll_offset.min(max_scroll_offset);

        let visible_lines: Vec<LayoutLine> = layout_lines
            .iter()
            .skip(self.scroll_offset)
            .take(max_visible_lines)
            .cloned()
            .collect();

        let mut result: Vec<String> = Vec::new();
        let left_padding = " ".repeat(metrics.padding_x);
        let right_padding = left_padding.clone();
        let prompt_prefix_inset = if metrics.prompt_prefix_width > 0 {
            metrics.padding_x.min(1)
        } else {
            0
        };
        let prompt_leading_padding = " ".repeat(prompt_prefix_inset);
        let prompt_trailing_padding = " ".repeat(metrics.padding_x.saturating_sub(prompt_prefix_inset));
        let cursor_reset = if metrics.use_background_surface {
            "\x1b[27m"
        } else {
            "\x1b[0m"
        };
        let background_color = self.background_color.clone();
        let render_surface_line = |line: &str| -> String {
            let padded = format!("{line}{}", " ".repeat(width.saturating_sub(visible_width(line))));
            match &background_color {
                Some(background_color) => background_color(&padded),
                None => padded,
            }
        };

        if !metrics.use_background_surface {
            if self.scroll_offset > 0 {
                let indicator = format!("─── ↑ {} more ", self.scroll_offset);
                let remaining = width as i64 - visible_width(&indicator) as i64;
                if remaining >= 0 {
                    result.push((self.border_color)(&format!(
                        "{indicator}{}",
                        "─".repeat(remaining as usize)
                    )));
                } else {
                    result.push((self.border_color)(&truncate_to_width(&indicator, width as f64, "", false)));
                }
            } else {
                result.push(horizontal.repeat(width));
            }
        } else {
            let line = if self.scroll_offset > 0 {
                (self.border_color)(&format!(" ↑ {} more", self.scroll_offset))
            } else {
                String::new()
            };
            result.push(render_surface_line(&truncate_to_width(&line, width as f64, "", false)));
        }

        // Emit hardware cursor marker only when focused and not showing autocomplete
        let emit_cursor_marker = self.focused && self.autocomplete_state.is_none();

        for (visible_line_index, layout_line) in visible_lines.iter().enumerate() {
            let absolute_line_index = self.scroll_offset + visible_line_index;
            let line_prompt_prefix = if absolute_line_index == 0 {
                prompt_prefix.clone()
            } else {
                " ".repeat(metrics.prompt_prefix_width)
            };
            let mut display_text = layout_line.text.clone();
            let mut line_visible_width = visible_width(&layout_line.text);
            let mut cursor_in_padding = false;

            if layout_line.has_cursor {
                if let Some(cursor_pos) = layout_line.cursor_pos {
                    let cursor_pos = cursor_pos.min(display_text.len());
                    let before = display_text[..cursor_pos].to_string();
                    let after = display_text[cursor_pos..].to_string();

                    // Hardware cursor marker (zero-width, emitted before fake cursor for IME positioning)
                    let marker = if emit_cursor_marker { CURSOR_MARKER } else { "" };

                    if !after.is_empty() {
                        let after_segments = self.segment(&after);
                        let first_grapheme = after_segments
                            .first()
                            .map(|(segment, _)| segment.clone())
                            .unwrap_or_default();
                        let rest_after = after[first_grapheme.len().min(after.len())..].to_string();
                        let cursor = format!("\x1b[7m{first_grapheme}{cursor_reset}");
                        display_text = format!("{before}{marker}{cursor}{rest_after}");
                    } else {
                        let cursor = format!("\x1b[7m {cursor_reset}");
                        display_text = format!("{before}{marker}{cursor}");
                        line_visible_width += 1;
                        if line_visible_width > metrics.input_width && metrics.padding_x > 0 {
                            cursor_in_padding = true;
                        }
                    }
                }
            }

            display_text = self.style_display_text(
                display_text,
                absolute_line_index,
                &layout_line.text,
                if layout_line.has_cursor {
                    layout_line.cursor_pos
                } else {
                    None
                },
                Some(layout_line.source_line),
                Some(layout_line.source_start),
            );

            let padding = " ".repeat(metrics.input_width.saturating_sub(line_visible_width));
            let line_right_padding = if cursor_in_padding {
                right_padding.get(1..).unwrap_or("").to_string()
            } else {
                right_padding.clone()
            };

            let content_line = format!(
                "{prompt_leading_padding}{line_prompt_prefix}{prompt_trailing_padding}{display_text}{padding}{line_right_padding}"
            );
            let anchor_marker = if layout_line.has_cursor {
                self.get_autocomplete_anchor_marker()
            } else {
                String::new()
            };
            result.push(format!(
                "{anchor_marker}{}",
                if metrics.use_background_surface {
                    render_surface_line(&content_line)
                } else {
                    content_line
                }
            ));
        }

        // Render bottom border (with scroll indicator if more content below)
        let lines_below = layout_lines.len() - (self.scroll_offset + visible_lines.len());
        if !metrics.use_background_surface {
            if lines_below > 0 {
                let indicator = format!("─── ↓ {lines_below} more ");
                let remaining = width as i64 - visible_width(&indicator) as i64;
                result.push((self.border_color)(&format!(
                    "{indicator}{}",
                    "─".repeat(remaining.max(0) as usize)
                )));
            } else {
                result.push(horizontal.repeat(width));
            }
        } else {
            let line = if lines_below > 0 {
                (self.border_color)(&format!(" ↓ {lines_below} more"))
            } else {
                String::new()
            };
            result.push(render_surface_line(&truncate_to_width(&line, width as f64, "", false)));
        }

        // The TypeScript overlay component renders itself by calling back into
        // the editor (`packages/tui/src/components/editor.ts:287-290`). Rust
        // cannot alias the owner, so the editor paints the same lines into the
        // overlay's buffer here, before `TUI::composite_overlays` reads it. The
        // composite pass always runs after the container (and therefore this
        // editor) has rendered (`crates/pi-tui/src/tui.rs:2168-2172`).
        let overlay_lines = self.render_autocomplete_overlay(width);
        *self.autocomplete_overlay_lines.borrow_mut() = overlay_lines;

        result
    }

    fn handle_input(&mut self, data: &str) {
        let kb = get_keybindings();

        if let Some(direction) = self.jump_mode {
            if kb.matches(data, "tui.editor.jumpForward") || kb.matches(data, "tui.editor.jumpBackward") {
                self.jump_mode = None;
                return;
            }

            let printable = decode_printable_key(data).or_else(|| {
                if data.chars().next().map(|c| c as u32 >= 32).unwrap_or(false) {
                    Some(data.to_string())
                } else {
                    None
                }
            });
            if let Some(printable) = printable {
                self.jump_mode = None;
                self.jump_to_char(&printable, direction);
                return;
            }

            self.jump_mode = None;
        }

        let mut data = data.to_string();
        if data.contains("\x1b[200~") {
            self.is_in_paste = true;
            self.paste_buffer = String::new();
            data = data.replace("\x1b[200~", "");
        }

        if self.is_in_paste {
            self.paste_buffer.push_str(&data);
            if let Some(end_index) = self.paste_buffer.find("\x1b[201~") {
                let paste_content = self.paste_buffer[..end_index].to_string();
                if !paste_content.is_empty() {
                    self.handle_paste(&paste_content);
                }
                self.is_in_paste = false;
                let remaining = self.paste_buffer[end_index + 6..].to_string();
                self.paste_buffer = String::new();
                if !remaining.is_empty() {
                    self.handle_input(&remaining);
                }
                return;
            }
            return;
        }

        if kb.matches(&data, "tui.input.copy") {
            return;
        }

        if kb.matches(&data, "tui.editor.undo") {
            self.undo();
            return;
        }

        if self.autocomplete_state.is_some() && self.autocomplete_list.is_some() {
            if kb.matches(&data, "tui.select.cancel") {
                self.cancel_autocomplete();
                return;
            }

            if kb.matches(&data, "tui.select.up") || kb.matches(&data, "tui.select.down") {
                if let Some(list) = self.autocomplete_list.as_mut() {
                    list.handle_input(&data);
                }
                return;
            }

            if kb.matches(&data, "tui.input.tab") {
                let selected = self
                    .autocomplete_list
                    .as_ref()
                    .and_then(|list| list.get_selected_item());
                if let Some(selected) = selected {
                    if self.autocomplete_provider.is_some() {
                        self.push_undo_snapshot();
                        self.last_action = None;
                        let result = {
                            let provider = self.autocomplete_provider.as_ref().unwrap().clone();
                            let mut provider = provider.borrow_mut();
                            provider.apply_completion(
                                &self.state.lines,
                                self.state.cursor_line,
                                self.state.cursor_col,
                                &selected,
                                &self.autocomplete_prefix.clone(),
                            )
                        };
                        self.state.lines = result.lines;
                        self.state.cursor_line = result.cursor_line;
                        self.set_cursor_col(result.cursor_col);
                        self.cancel_autocomplete();
                        self.emit_change();
                    }
                }
                return;
            }

            if kb.matches(&data, "tui.select.confirm") {
                let selected = self
                    .autocomplete_list
                    .as_ref()
                    .and_then(|list| list.get_selected_item());
                if let Some(selected) = selected {
                    if self.autocomplete_provider.is_some() {
                        let slash_context = self.get_current_slash_command_context();
                        let is_slash_command_completion = self.autocomplete_kind.as_deref()
                            == Some("slash-command")
                            || (self.autocomplete_kind.is_none()
                                && self.autocomplete_state == Some(AutocompleteState::Regular)
                                && self.autocomplete_prefix.starts_with('/'));
                        let should_submit_slash_command = is_slash_command_completion
                            && matches!(
                                slash_context.as_ref(),
                                Some(SlashCommandContext::Name {
                                    is_at_prompt_start: true,
                                    ..
                                })
                            );
                        self.push_undo_snapshot();
                        self.last_action = None;
                        let result = {
                            let provider = self.autocomplete_provider.as_ref().unwrap().clone();
                            let mut provider = provider.borrow_mut();
                            provider.apply_completion(
                                &self.state.lines,
                                self.state.cursor_line,
                                self.state.cursor_col,
                                &selected,
                                &self.autocomplete_prefix.clone(),
                            )
                        };
                        self.state.lines = result.lines;
                        self.state.cursor_line = result.cursor_line;
                        self.set_cursor_col(result.cursor_col);

                        if is_slash_command_completion {
                            self.cancel_autocomplete();
                            if !should_submit_slash_command || selected.takes_argument.unwrap_or(false) {
                                self.emit_change();
                                return;
                            }
                            self.emit_change();
                            self.submit_value();
                            return;
                        }
                        self.cancel_autocomplete();
                        self.emit_change();
                        return;
                    }
                }
            }
        }

        if kb.matches(&data, "tui.input.tab") && self.autocomplete_state.is_none() {
            self.handle_tab_completion();
            return;
        }

        if kb.matches(&data, "tui.editor.deleteToLineEnd") {
            self.delete_to_end_of_line();
            return;
        }
        if kb.matches(&data, "tui.editor.deleteToLineStart") {
            self.delete_to_start_of_line();
            return;
        }
        if kb.matches(&data, "tui.editor.deleteWordBackward") {
            self.delete_word_backwards();
            return;
        }
        if kb.matches(&data, "tui.editor.deleteWordForward") {
            self.delete_word_forward();
            return;
        }
        if kb.matches(&data, "tui.editor.deleteCharBackward") || matches_key(&data, "shift+backspace") {
            self.handle_backspace();
            return;
        }
        if kb.matches(&data, "tui.editor.deleteCharForward") || matches_key(&data, "shift+delete") {
            self.handle_forward_delete();
            return;
        }

        if kb.matches(&data, "tui.editor.yank") {
            self.yank();
            return;
        }
        if kb.matches(&data, "tui.editor.yankPop") {
            self.yank_pop();
            return;
        }

        if kb.matches(&data, "tui.editor.cursorLineStart") {
            self.move_to_line_start();
            return;
        }
        if kb.matches(&data, "tui.editor.cursorLineEnd") {
            self.move_to_line_end();
            return;
        }
        if kb.matches(&data, "tui.editor.cursorWordLeft") {
            self.move_word_backwards();
            return;
        }
        if kb.matches(&data, "tui.editor.cursorWordRight") {
            self.move_word_forwards();
            return;
        }

        if kb.matches(&data, "tui.input.newLine")
            || (data.chars().next().map(|c| c as u32 == 10).unwrap_or(false) && data.len() > 1)
            || data == "\x1b\r"
            || data == "\x1b[13;2~"
            || (data.len() > 1 && data.contains('\x1b') && data.contains('\r'))
            || (data == "\n" && data.len() == 1)
        {
            if self.should_submit_on_backslash_enter(&data, &kb) {
                self.handle_backspace();
                self.submit_value();
                return;
            }
            self.add_new_line();
            return;
        }

        if kb.matches(&data, "tui.input.submit") {
            if self.disable_submit {
                return;
            }

            // Workaround for terminals without Shift+Enter support:
            // If char before cursor is \, delete it and insert newline instead of submitting.
            if self.char_before_cursor() == Some('\\') {
                self.handle_backspace();
                self.add_new_line();
                return;
            }

            self.submit_value();
            return;
        }

        if kb.matches(&data, "tui.editor.cursorUp") {
            if self.is_editor_empty() {
                self.navigate_history(-1);
            } else if self.history_index > -1 && self.is_on_first_visual_line() {
                self.navigate_history(-1);
            } else if self.is_on_first_visual_line() {
                self.move_to_line_start();
            } else {
                self.move_cursor(-1, 0);
            }
            return;
        }
        if kb.matches(&data, "tui.editor.cursorDown") {
            if self.history_index > -1 && self.is_on_last_visual_line() {
                self.navigate_history(1);
            } else if self.is_on_last_visual_line() {
                self.move_to_line_end();
            } else {
                self.move_cursor(1, 0);
            }
            return;
        }
        if kb.matches(&data, "tui.editor.cursorRight") {
            self.move_cursor(0, 1);
            return;
        }
        if kb.matches(&data, "tui.editor.cursorLeft") {
            self.move_cursor(0, -1);
            return;
        }

        if kb.matches(&data, "tui.editor.pageUp") {
            self.page_scroll(-1);
            return;
        }
        if kb.matches(&data, "tui.editor.pageDown") {
            self.page_scroll(1);
            return;
        }

        if kb.matches(&data, "tui.editor.jumpForward") {
            self.jump_mode = Some(true);
            return;
        }
        if kb.matches(&data, "tui.editor.jumpBackward") {
            self.jump_mode = Some(false);
            return;
        }

        if matches_key(&data, "shift+space") {
            self.insert_character(" ", false);
            return;
        }

        if let Some(printable) = decode_printable_key(&data) {
            self.insert_character(&printable, false);
            return;
        }

        if data.chars().next().map(|c| c as u32 >= 32).unwrap_or(false) {
            self.insert_character(&data, false);
        }
    }

    fn invalidate(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_while_the_owning_tui_is_borrowed() {
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(crate::terminal::ProcessTerminal::new()), None)));
        let mut editor = Editor::new(ui.clone(), EditorTheme {
            border_color: Rc::new(str::to_string), background_color: None,
            autocomplete_background_color: None, command_color: None,
            select_list: SelectListTheme {
                selected_prefix: Box::new(str::to_string), selected_text: Box::new(str::to_string),
                description: Box::new(str::to_string), argument_hint: None, source_tag: None,
                scroll_info: Box::new(str::to_string), no_match: Box::new(str::to_string),
            },
        }, EditorOptions::default());
        editor.set_text("one\ntwo\nthree\nfour\nfive\nsix\nseven");
        editor.set_terminal_rows(10);
        let _render_owner = ui.borrow_mut();
        let lines = editor.render(30.0);
        assert_eq!(lines.len(), 7, "five visible lines and two borders");
        assert!(lines.iter().any(|line| line.contains("seven")));
    }

    #[test]
    fn paste_marker_is_atomic_only_with_valid_id() {
        assert!(is_atomic_marker("[paste #1 +12 lines]"));
        assert!(is_atomic_marker("[paste #2 1234 chars]"));
        assert!(is_atomic_marker("[image #3]"));
        assert!(!is_atomic_marker("[paste #1"));
        assert!(!is_atomic_marker("[image #]"));
        assert!(!is_atomic_marker("[paste #abc]"));
    }

    #[test]
    fn segment_with_markers_merges_valid_paste_ids_only() {
        let text = "a[paste #1 +2 lines]b";
        let merged = segment_with_markers(text, &[1]);
        assert!(merged.iter().any(|(segment, _)| segment == "[paste #1 +2 lines]"));

        let stale = segment_with_markers(text, &[9]);
        assert!(!stale.iter().any(|(segment, _)| segment == "[paste #1 +2 lines]"));
    }

    #[test]
    fn segment_with_markers_merges_image_markers_always() {
        let text = "x[image #4]y";
        let merged = segment_with_markers(text, &[]);
        assert!(merged.iter().any(|(segment, _)| segment == "[image #4]"));
    }

    #[test]
    fn word_wrap_line_returns_single_chunk_when_it_fits() {
        let chunks = word_wrap_line("hello", 10, None);
        assert_eq!(
            chunks,
            vec![TextChunk {
                text: "hello".to_string(),
                start_index: 0,
                end_index: 5
            }]
        );
    }

    #[test]
    fn word_wrap_line_empty_input() {
        let chunks = word_wrap_line("", 10, None);
        assert_eq!(
            chunks,
            vec![TextChunk {
                text: String::new(),
                start_index: 0,
                end_index: 0
            }]
        );
        assert_eq!(word_wrap_line("x", 0, None).len(), 1);
    }

    #[test]
    fn word_wrap_line_wraps_at_word_boundary() {
        let chunks = word_wrap_line("hello world", 6, None);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "hello ");
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks[0].end_index, 6);
        assert_eq!(chunks[1].text, "world");
        assert_eq!(chunks[1].start_index, 6);
        assert_eq!(chunks[1].end_index, 11);
    }

    #[test]
    fn word_wrap_line_force_breaks_long_words() {
        let chunks = word_wrap_line("abcdefgh", 3, None);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "abc");
        assert_eq!(chunks[1].text, "def");
        assert_eq!(chunks[2].text, "gh");
    }

    #[test]
    fn word_wrap_line_keeps_atomic_marker_together() {
        let text = "[paste #1 1234 chars] tail";
        let segments = segment_with_markers(text, &[1]);
        let chunks = word_wrap_line(text, 30, Some(&segments));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn word_wrap_line_rewraps_atomic_marker_wider_than_max() {
        let text = "[paste #1 1234 chars]";
        let segments = segment_with_markers(text, &[1]);
        let chunks = word_wrap_line(text, 5, Some(&segments));
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| visible_width(&chunk.text) <= 5));
    }

    #[test]
    fn decode_csi_u_ctrl_maps_letters() {
        assert_eq!(decode_csi_u_ctrl("\x1b[106;5u"), "\n");
        assert_eq!(decode_csi_u_ctrl("a\x1b[65;5ub"), "a\u{1}b");
        assert_eq!(decode_csi_u_ctrl("\x1b[9;5u"), "\x1b[9;5u");
    }

    #[test]
    fn symbol_and_attachment_contexts() {
        assert!(matches_symbol_context("@"));
        assert!(matches_symbol_context("hi @foo"));
        // `/(?:^|[\s])[@#][^\s]*$/` needs whitespace (or start) before the symbol.
        assert!(!matches_symbol_context("hi#bar"));
        assert!(!matches_symbol_context("hi @foo bar"));
        assert!(matches_attachment_context("@"));
        assert!(matches_attachment_context("say @foo"));
        assert!(!matches_attachment_context("say @foo bar"));
    }

    #[test]
    fn is_word_char_matches_backslash_w() {
        assert!(is_word_char("a"));
        assert!(is_word_char("_"));
        assert!(is_word_char("7"));
        assert!(!is_word_char("-"));
        assert!(!is_word_char(""));
    }

    #[test]
    fn autocomplete_word_char_class() {
        assert!(is_autocomplete_word_char("a"));
        assert!(is_autocomplete_word_char("."));
        assert!(is_autocomplete_word_char("-"));
        assert!(is_autocomplete_word_char("_"));
        assert!(!is_autocomplete_word_char("@"));
        assert!(!is_autocomplete_word_char(""));
    }

    #[test]
    fn paste_marker_replacement_expands_matching_id() {
        assert_eq!(
            replace_paste_marker("a[paste #1 +2 lines]b", 1, "X"),
            "aXb"
        );
        assert_eq!(
            replace_paste_marker("a[paste #1 12 chars]b", 1, "X"),
            "aXb"
        );
        assert_eq!(replace_paste_marker("a[paste #2]b", 1, "X"), "a[paste #2]b");
        assert_eq!(replace_paste_marker("[paste #1]", 1, "X"), "X");
    }

    #[test]
    fn best_match_prefers_exact_then_first_prefix() {
        let items = vec![
            SelectItem {
                value: "alpha".to_string(),
                ..SelectItem::default()
            },
            SelectItem {
                value: "al".to_string(),
                ..SelectItem::default()
            },
        ];
        assert_eq!(Editor::get_best_autocomplete_match_index(&items, "al"), 1);
        assert_eq!(Editor::get_best_autocomplete_match_index(&items, "alp"), 0);
        assert_eq!(Editor::get_best_autocomplete_match_index(&items, "zzz"), -1);
        assert_eq!(Editor::get_best_autocomplete_match_index(&items, ""), -1);
    }

    #[test]
    fn index_of_from_and_last_index_of_from() {
        assert_eq!(index_of_from("abcabc", "b", 0), Some(1));
        assert_eq!(index_of_from("abcabc", "b", 2), Some(4));
        assert_eq!(index_of_from("abc", "b", 9), None);
        assert_eq!(last_index_of_from("abcabc", "b", None), Some(4));
        assert_eq!(last_index_of_from("abcabc", "b", Some(3)), Some(1));
    }

    /// The cursor column is a byte offset, so `cursor_col + 1` can point inside a
    /// multi-byte character. `line.indexOf(char, from)` in TS
    /// (packages/tui/src/components/editor.ts:2081) tolerates any offset; the Rust
    /// slice must not panic there.
    #[test]
    fn index_of_from_clamps_offsets_inside_multibyte_characters() {
        // "x" + 😀(4 bytes) + "y": offset 2 is inside the emoji.
        let line = "x\u{1F600}y";
        assert_eq!(index_of_from(line, "y", 2), Some(5));
        assert_eq!(index_of_from(line, "x", 2), None);
        assert_eq!(last_index_of_from(line, "x", Some(2)), Some(0));
        assert_eq!(last_index_of_from(line, "y", Some(2)), None);

        let cjk = "\u{4F60}\u{597D}";
        assert_eq!(index_of_from(cjk, "\u{597D}", 1), Some(3));
        assert_eq!(index_of_from(cjk, "\u{597D}", 3), Some(3));
        // `lastIndexOf(char, from)` searches backwards from `from` inclusive, so a
        // mid-character `from` clamps back to the start of that character.
        assert_eq!(last_index_of_from(cjk, "\u{4F60}", Some(2)), Some(0));
        assert_eq!(last_index_of_from("abcabc", "b", Some(9)), Some(4));
    }

    /// Ctrl+] (jumpForward) must never abort the process when the cursor sits
    /// before a multi-byte character: `jump_to_char` passes `cursor_col + 1`,
    /// which is a byte offset one past `a` and therefore inside the emoji.
    #[test]
    fn jump_to_char_survives_a_cursor_before_a_multibyte_character() {
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(crate::terminal::ProcessTerminal::new()), None)));
        let mut editor = Editor::new(ui.clone(), test_theme(), EditorOptions::default());
        editor.set_text("a\u{1F600}y");
        editor.set_cursor_col(1);
        assert_eq!(editor.get_cursor(), (0, 1));

        editor.jump_to_char("y", true);
        assert_eq!(editor.get_cursor(), (0, 5));

        // Backward jump from the same mid-character offset: `cursor_col - 1 == 0`.
        editor.set_cursor_col(1);
        editor.jump_to_char("a", false);
        assert_eq!(editor.get_cursor(), (0, 0));

        // "\u{4F60}\u{597D}x": byte 5 is inside \u{597D}, so the old
        // `line[from..]` slice aborted the process here.
        let cjk_ui = Rc::new(RefCell::new(TUI::new(Box::new(crate::terminal::ProcessTerminal::new()), None)));
        let mut cjk = Editor::new(cjk_ui, test_theme(), EditorOptions::default());
        cjk.set_text("\u{4F60}\u{597D}x");
        cjk.set_cursor_col(4);
        cjk.jump_to_char("x", true);
        assert_eq!(cjk.get_cursor(), (0, 6));
    }

    /// Test provider that answers every request with fixed slash-command items.
    struct FixedAutocompleteProvider {
        items: Vec<SelectItem>,
        prefix: String,
        kind: Option<String>,
    }

    impl EditorAutocompleteProvider for FixedAutocompleteProvider {
        fn get_suggestions(
            &mut self,
            _lines: &[String],
            _cursor_line: usize,
            _cursor_col: usize,
            _force: bool,
        ) -> Option<AutocompleteSuggestions> {
            Some(AutocompleteSuggestions {
                items: self.items.clone(),
                prefix: self.prefix.clone(),
                kind: self.kind.clone(),
            })
        }

        fn apply_completion(
            &mut self,
            lines: &[String],
            cursor_line: usize,
            _cursor_col: usize,
            item: &SelectItem,
            _prefix: &str,
        ) -> ApplyCompletionResult {
            let mut next = lines.to_vec();
            next[cursor_line] = item.value.clone();
            ApplyCompletionResult {
                lines: next,
                cursor_line,
                cursor_col: item.value.len(),
            }
        }
    }

    fn item(value: &str) -> SelectItem {
        SelectItem {
            value: value.to_string(),
            label: value.to_string(),
            description: None,
            argument_hint: None,
            source_tag: None,
            takes_argument: None,
        }
    }

    /// A focused editor with a provider and two suggestions already applied.
    fn editor_with_autocomplete_with_terminal<T: Terminal + 'static>(
        terminal: T,
    ) -> (Rc<RefCell<TUI>>, Rc<RefCell<Editor>>) {
        let ui = Rc::new(RefCell::new(TUI::new(Box::new(terminal), Some(false))));
        // The dropdown composites ABOVE the cursor row and is truncated to the rows
        // available there (`above_marker` handling: `TUI::composite_overlays`,
        // packages/tui/src/tui.ts:1265-1274). In the real app the editor sits under a
        // transcript, so give the test the same headroom instead of one blank row.
        ui.borrow_mut()
            .add_child(Rc::new(RefCell::new(crate::components::text::Text::new(
                "transcript\nline\nline\nline\nline\nline\nline\nline\nline\nline".to_string(),
                0,
                0,
                None,
            ))));
        let editor = Rc::new(RefCell::new(Editor::new(
            ui.clone(),
            test_theme(),
            EditorOptions::default(),
        )));
        ui.borrow_mut().add_child(editor.clone());
        ui.borrow_mut().set_focus(Some(editor.clone()));
        {
            let mut editor = editor.borrow_mut();
            editor.set_autocomplete_provider(Rc::new(RefCell::new(FixedAutocompleteProvider {
                items: vec![item("help"), item("hotkeys")],
                prefix: "/".to_string(),
                kind: Some("slash-command".to_string()),
            })));
            editor.request_autocomplete(false, true);
        }
        (ui, editor)
    }

    /// Records everything the TUI paints so a test can assert on the frame.
    #[derive(Clone)]
    struct RecordingTerminal {
        written: Rc<RefCell<String>>,
    }

    impl Terminal for RecordingTerminal {
        fn start(&mut self, _on_input: Box<dyn Fn(String)>, _on_resize: Box<dyn Fn()>) {}

        fn stop(&mut self, _options: crate::terminal::TerminalStopOptions) {}

        fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {}

        fn write(&mut self, data: &str) {
            self.written.borrow_mut().push_str(data);
        }

        fn columns(&self) -> usize {
            40
        }

        fn rows(&self) -> usize {
            12
        }

        fn kitty_protocol_active(&self) -> bool {
            false
        }

        fn move_by(&mut self, _lines: i64) {}
        fn hide_cursor(&mut self) {}
        fn show_cursor(&mut self) {}
        fn clear_line(&mut self) {}
        fn clear_from_cursor(&mut self) {}
        fn clear_screen(&mut self) {}
        fn enter_alt_screen(&mut self) {}
        fn leave_alt_screen(&mut self) {}
        fn alt_screen_active(&self) -> bool {
            false
        }
        fn set_mouse_tracking(&mut self, _enabled: bool) {}
        fn mouse_tracking_active(&self) -> bool {
            false
        }
        fn set_title(&mut self, _title: &str) {}
        fn set_progress(&mut self, _active: bool) {}
    }

    /// Teeth: reverting `EditorOverlayComponent::render` to `Vec::new()` (or
    /// dropping the editor-side buffer fill added in `Editor::render`) makes this
    /// fail with a frame that never mentions the suggestions.
    #[test]
    fn autocomplete_overlay_paints_the_suggestion_list() {
        let written = Rc::new(RefCell::new(String::new()));
        let (ui, editor) = editor_with_autocomplete_with_terminal(RecordingTerminal {
            written: Rc::clone(&written),
        });
        assert!(editor.borrow().is_showing_autocomplete());

        ui.borrow_mut().do_render();
        let frame = String::from_utf8_lossy(&written.borrow().clone().into_bytes()).to_string();
        assert!(
            frame.contains("help"),
            "the dropdown must paint the suggestion rows, frame={frame:?}"
        );
        assert!(frame.contains("hotkeys"));
    }

    /// Teeth: `visible: None` (the pre-fix state) keeps the overlay painted after
    /// focus leaves; `focused() && autocompleteState !== null` hides it
    /// (packages/tui/src/components/editor.ts:2353).
    #[test]
    fn autocomplete_overlay_hides_when_focus_leaves() {
        let written = Rc::new(RefCell::new(String::new()));
        let (ui, editor) = editor_with_autocomplete_with_terminal(RecordingTerminal {
            written: Rc::clone(&written),
        });
        ui.borrow_mut().do_render();
        assert!(written.borrow().contains("help"));

        written.borrow_mut().clear();
        editor.borrow_mut().set_focused(false);
        ui.borrow_mut().do_render();
        assert!(
            !written.borrow().contains("help"),
            "an unfocused editor must not paint the dropdown"
        );
        assert!(
            editor.borrow().is_showing_autocomplete(),
            "focus alone clears nothing"
        );
    }

    fn vertical_test_editor(text: &str, width: usize, line: usize, col: usize) -> Editor {
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(crate::terminal::ProcessTerminal::new()),
            None,
        )));
        let mut editor = Editor::new(ui, test_theme(), EditorOptions::default());
        editor.set_text(text);
        editor.last_width = width;
        editor.state.cursor_line = line;
        editor.set_cursor_col(col);
        editor
    }

    #[test]
    fn vertical_movement_maps_accented_bytes_to_ascii_display_columns() {
        let mut editor = vertical_test_editor("éx\nab", 80, 0, "é".len());
        editor.move_cursor(1, 0);
        assert_eq!(
            editor.get_cursor(),
            (1, 1),
            "one display cell right is one ASCII byte"
        );
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (0, "é".len()));
    }

    #[test]
    fn wrapped_cjk_vertical_moves_keep_display_columns_and_sticky_position() {
        let mut editor = vertical_test_editor("abcdefgh\n界界界界\nijklmnop", 4, 0, 3);

        editor.move_cursor(1, 0);
        assert_eq!(editor.get_cursor(), (0, 7));
        editor.move_cursor(1, 0);
        assert_eq!(
            editor.get_cursor(),
            (1, "界".len()),
            "wide target clamps to two cells"
        );
        assert_eq!(editor.preferred_visual_col, Some(3));
        editor.move_cursor(1, 0);
        assert_eq!(editor.get_cursor(), (1, 4 * "界".len()));
        assert_eq!(editor.preferred_visual_col, Some(3));
        editor.move_cursor(1, 0);
        assert_eq!(
            editor.get_cursor(),
            (2, 3),
            "the original three-cell column remains sticky"
        );
        assert_eq!(editor.preferred_visual_col, Some(3));
        editor.move_cursor(1, 0);
        assert_eq!(editor.get_cursor(), (2, 7));
        assert_eq!(editor.preferred_visual_col, None);
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (2, 3));
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (1, 4 * "界".len()));
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (1, "界".len()));
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (0, 7));
        editor.move_cursor(-1, 0);
        assert_eq!(editor.get_cursor(), (0, 3));
    }

    #[test]
    fn page_movement_from_wrapped_cjk_uses_display_columns_without_bad_byte_slices() {
        let mut editor = vertical_test_editor("abcdefgh\n界界界界\nijklmnop", 4, 0, 7);
        editor.move_cursor(1, 0);
        assert_eq!(editor.get_cursor(), (1, "界".len()));
        assert_eq!(editor.preferred_visual_col, Some(3));
        editor.page_scroll(1);
        assert_eq!(editor.get_cursor(), (2, 7));
        assert_eq!(editor.preferred_visual_col, None);
        editor.page_scroll(-1);
        assert_eq!(editor.get_cursor(), (0, 3));
    }

    #[test]
    fn unicode_backslash_enter_inserts_newline_instead_of_submitting() {
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(crate::terminal::ProcessTerminal::new()),
            None,
        )));
        let mut editor = Editor::new(ui, test_theme(), EditorOptions::default());
        editor.set_text("é\\");
        let submitted = Rc::new(RefCell::new(None::<String>));
        let submitted_slot = Rc::clone(&submitted);
        editor.on_submit = Some(Box::new(move |text| {
            *submitted_slot.borrow_mut() = Some(text.to_string());
        }));

        editor.handle_input("\r");

        assert_eq!(editor.get_text(), "é\n");
        assert!(submitted.borrow().is_none());
    }

    #[test]
    fn absolute_paste_is_separated_after_a_multibyte_word_character() {
        let ui = Rc::new(RefCell::new(TUI::new(
            Box::new(crate::terminal::ProcessTerminal::new()),
            None,
        )));
        let mut editor = Editor::new(ui, test_theme(), EditorOptions::default());
        editor.set_text("é");

        editor.handle_input("\x1b[200~/tmp/café\x1b[201~");

        assert_eq!(editor.get_text(), "é /tmp/café");
    }

    /// Shared theme builder for the editor tests.
    fn test_theme() -> EditorTheme {
        EditorTheme {
            border_color: Rc::new(str::to_string), background_color: None,
            autocomplete_background_color: None, command_color: None,
            select_list: SelectListTheme {
                selected_prefix: Box::new(str::to_string), selected_text: Box::new(str::to_string),
                description: Box::new(str::to_string), argument_hint: None, source_tag: None,
                scroll_info: Box::new(str::to_string), no_match: Box::new(str::to_string),
            },
        }
    }
}
