//! Port of packages/tui/src/fullscreen.ts.
//!
//! Fullscreen (alternate-screen) viewport: a scrollable window over the
//! transcript with a dock (editor/footer) pinned to the bottom rows. Frames
//! are a fixed grid painted with absolute addressing and diffed row-by-row;
//! scroll position is application state, not terminal scrollback.

use crate::selection_metadata::TableCellSelectionRegion;
use crate::terminal_image::is_image_line;
use crate::utils::{slice_by_column, strip_ansi, url_at_column, visible_width};
use std::rc::Rc;

pub const FULLSCREEN_MIN_TRANSCRIPT_ROWS: usize = 3;

pub fn clipped_fullscreen_dock_height(dock_length: usize, height: usize) -> usize {
    let max_dock = height.saturating_sub(FULLSCREEN_MIN_TRANSCRIPT_ROWS);
    dock_length.min(max_dock)
}

/// Kitty images span multiple physical rows and cannot be clipped to a window.
const IMAGE_PLACEHOLDER: &str = "\x1b[2m[image \u{2014} view in inline mode]\x1b[0m";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollInfo {
    pub following: bool,
    pub lines_below: usize,
    pub lines_above: usize,
}

/// Stable component identity and content offset for one rendered transcript line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewportAnchor {
    pub key: Rc<str>,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionScrollDirection {
    Up,
    Down,
}

impl SelectionScrollDirection {
    fn delta(self) -> i64 {
        match self {
            SelectionScrollDirection::Up => -1,
            SelectionScrollDirection::Down => 1,
        }
    }
}

/// Transcript-anchored selection endpoint (line index + visible column), so
/// streaming appends and scrolling never shift what is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionPoint {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSelectionRegion {
    pub line: usize,
    pub col: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColumnSpan {
    from: usize,
    to: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameSelectionSnapshot {
    frame: Vec<String>,
    regions: Vec<FrameSelectionRegion>,
    visible_start: usize,
    visible_height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableCellPosition {
    pub row: i64,
    pub column: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveTableSelection {
    table: usize,
    anchor: TableCellPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TableSelectionRange {
    from_row: i64,
    to_row: i64,
    from_column: i64,
    to_column: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionMode {
    Transcript,
    Table,
    Frame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TranscriptBounds {
    first_row: usize,
    last_row: usize,
    visible_start: usize,
    visible_height: usize,
    transcript_start: usize,
    transcript_end: usize,
}

pub struct FullscreenViewport {
    scroll_top: usize,
    following: bool,
    prev_frame: Vec<String>,
    prev_width: usize,
    prev_height: usize,
    last_max_scroll: usize,
    last_window_height: usize,
    last_header_height: usize,
    selection_columns: Vec<Option<(usize, usize)>>,
    padding_line: String,
    last_transcript: Vec<String>,
    last_anchors: Vec<Option<ViewportAnchor>>,
    last_top_padding: usize,
    last_bottom_aligned: bool,
    last_frame: Vec<String>,
    last_frame_visible_start: usize,
    last_frame_visible_height: usize,
    frame_selection_regions: Vec<FrameSelectionRegion>,
    table_cell_selection_regions: Vec<TableCellSelectionRegion>,
    active_frame_selection: Option<FrameSelectionSnapshot>,
    active_table_selection: Option<ActiveTableSelection>,
    selection_anchor: Option<SelectionPoint>,
    selection_head: Option<SelectionPoint>,
    selection_mode: Option<SelectionMode>,
}

impl Default for FullscreenViewport {
    fn default() -> Self {
        Self::new()
    }
}

impl FullscreenViewport {
    pub fn new() -> Self {
        Self {
            scroll_top: 0,
            following: true,
            prev_frame: Vec::new(),
            prev_width: 0,
            prev_height: 0,
            last_max_scroll: 0,
            last_window_height: 0,
            last_header_height: 0,
            selection_columns: Vec::new(),
            padding_line: String::new(),
            last_transcript: Vec::new(),
            last_anchors: Vec::new(),
            last_top_padding: 0,
            last_bottom_aligned: false,
            last_frame: Vec::new(),
            last_frame_visible_start: 0,
            last_frame_visible_height: 0,
            frame_selection_regions: Vec::new(),
            table_cell_selection_regions: Vec::new(),
            active_frame_selection: None,
            active_table_selection: None,
            selection_anchor: None,
            selection_head: None,
            selection_mode: None,
        }
    }

    /// Compose a frame of exactly `height` lines: scrolled transcript window on
    /// top, dock pinned to the bottom. Following pins the window to the
    /// transcript end; otherwise it stays frozen while content appends.
    pub fn compose_frame(
        &mut self,
        transcript: &[String],
        dock: &[String],
        height: usize,
        table_cell_selection_regions: &[TableCellSelectionRegion],
    ) -> Vec<String> {
        self.compose_frame_anchored(transcript, dock, height, table_cell_selection_regions, &[], false)
    }

    pub fn compose_frame_anchored(
        &mut self,
        transcript: &[String],
        dock: &[String],
        height: usize,
        table_cell_selection_regions: &[TableCellSelectionRegion],
        anchors: &[Option<ViewportAnchor>],
        bottom_aligned: bool,
    ) -> Vec<String> {
        self.compose_frame_with_header(transcript, dock, height, table_cell_selection_regions, anchors, bottom_aligned, &[])
    }

    /// Content column bounds exclude decorative gutters from selection and copying.
    pub fn set_transcript_presentation(&mut self, columns: Vec<Option<(usize, usize)>>, padding: String) {
        self.selection_columns = columns;
        self.padding_line = padding;
    }

    /// Header chrome yields before the editor and minimum transcript allocation.
    pub fn compose_frame_with_header(
        &mut self,
        transcript: &[String],
        dock: &[String],
        height: usize,
        table_cell_selection_regions: &[TableCellSelectionRegion],
        anchors: &[Option<ViewportAnchor>],
        bottom_aligned: bool,
        header: &[String],
    ) -> Vec<String> {
        let dock_height = clipped_fullscreen_dock_height(dock.len(), height);
        let header_height = header.len().min(height.saturating_sub(dock_height + FULLSCREEN_MIN_TRANSCRIPT_ROWS));
        let dock_lines: Vec<String> = if dock.len() > dock_height {
            dock[dock.len() - dock_height..].to_vec()
        } else {
            dock.to_vec()
        };
        let window_height = height - dock_lines.len() - header_height;
        let max_scroll = transcript.len().saturating_sub(window_height);

        if self.following {
            self.scroll_top = max_scroll;
        } else {
            // The first anchored visible row remains at the same screen row.
            // Its content offset survives wrapping changes within that message.
            let old_end = (self.scroll_top + self.last_window_height).min(self.last_anchors.len());
            if let Some((old_line, anchor)) = (self.scroll_top..old_end)
                .find_map(|line| self.last_anchors[line].as_ref().map(|anchor| (line, anchor)))
            {
                if let Some((new_line, _)) = anchors.iter().enumerate()
                    .filter_map(|(line, candidate)| candidate.as_ref().map(|candidate| (line, candidate)))
                    .filter(|(_, candidate)| candidate.key == anchor.key && candidate.offset <= anchor.offset)
                    .max_by_key(|(_, candidate)| candidate.offset)
                {
                    let screen_row = self.last_top_padding + old_line - self.scroll_top;
                    self.scroll_top = new_line.saturating_sub(screen_row);
                }
            }
            self.scroll_top = self.scroll_top.min(max_scroll);
        }
        self.last_max_scroll = max_scroll;
        self.last_window_height = window_height;
        self.last_header_height = header_height;
        self.last_transcript = transcript.to_vec();
        self.last_anchors = anchors.to_vec();
        self.last_bottom_aligned = bottom_aligned;
        self.table_cell_selection_regions = table_cell_selection_regions.to_vec();

        let end = (self.scroll_top + window_height).min(transcript.len());
        let mut window: Vec<String> = transcript[self.scroll_top..end].to_vec();
        for line in window.iter_mut() {
            if is_image_line(line) {
                *line = IMAGE_PLACEHOLDER.to_string();
            }
        }
        self.highlight_selection(&mut window);
        self.last_top_padding = if bottom_aligned { window_height - window.len() } else { 0 };
        if self.last_top_padding > 0 {
            window.splice(0..0, std::iter::repeat_n(self.padding_line.clone(), self.last_top_padding));
        }
        while window.len() < window_height {
            window.push(self.padding_line.clone());
        }
        window.splice(0..0, header[..header_height].iter().cloned());
        window.extend(dock_lines);
        window
    }

    fn ordered_selection(&self) -> Option<(SelectionPoint, SelectionPoint)> {
        let a = self.selection_anchor?;
        let b = self.selection_head?;
        if a.line == b.line && a.col == b.col {
            return None;
        }
        let flipped = a.line > b.line || (a.line == b.line && a.col > b.col);
        Some(if flipped { (b, a) } else { (a, b) })
    }

    fn selection_span(&self, line_index: usize, sel: (SelectionPoint, SelectionPoint)) -> Option<ColumnSpan> {
        let (start, end) = sel;
        if line_index < start.line || line_index > end.line {
            return None;
        }
        let mut span = ColumnSpan {
            from: if line_index == start.line { start.col } else { 0 },
            to: if line_index == end.line {
                end.col
            } else {
                usize::MAX
            },
        };
        if matches!(self.selection_mode, Some(SelectionMode::Transcript | SelectionMode::Table)) {
            if let Some(Some((from, to))) = self.selection_columns.get(line_index) {
                span.from = span.from.max(*from);
                span.to = span.to.min(*to);
            }
        }
        (span.to >= span.from).then_some(span)
    }

    fn highlight_selection(&mut self, window: &mut [String]) {
        if self.selection_mode != Some(SelectionMode::Transcript)
            && self.selection_mode != Some(SelectionMode::Table)
        {
            return;
        }
        let sel = match self.ordered_selection() {
            Some(sel) => sel,
            None => return,
        };
        for i in 0..window.len() {
            let line_index = self.scroll_top + i;
            let spans = if self.selection_mode == Some(SelectionMode::Table) {
                self.selected_table_spans(line_index, sel)
            } else {
                self.selection_span(line_index, sel).into_iter().collect()
            };
            for span in spans.iter().rev() {
                window[i] = Self::highlight_line(&window[i], *span);
            }
        }
    }

    /// Begin a selection at a screen position; false when outside the transcript window.
    pub fn begin_selection(&mut self, screen_row: i64, screen_col: i64) -> bool {
        let line = match self.transcript_line_for_screen_row(screen_row, false) {
            Some(line) => line,
            None => {
                self.clear_selection();
                return false;
            }
        };
        let point = SelectionPoint {
            line,
            col: screen_col.max(0) as usize,
        };
        self.selection_anchor = Some(point);
        self.selection_head = Some(point);
        let table = self.table_at_point(point);
        let table_cell = table.and_then(|table| self.closest_table_cell(table, point));
        if let (Some(table), Some(cell)) = (table, table_cell) {
            self.active_table_selection = Some(ActiveTableSelection {
                table,
                anchor: cell,
            });
            self.selection_mode = Some(SelectionMode::Table);
        } else {
            self.active_table_selection = None;
            self.selection_mode = Some(SelectionMode::Transcript);
        }
        true
    }

    pub fn extend_selection(&mut self, screen_row: i64, screen_col: i64) {
        if self.selection_anchor.is_none()
            || (self.selection_mode != Some(SelectionMode::Transcript)
                && self.selection_mode != Some(SelectionMode::Table))
        {
            return;
        }
        let line = match self.transcript_line_for_screen_row(screen_row, true) {
            Some(line) => line,
            None => return,
        };
        self.selection_head = Some(SelectionPoint {
            line,
            col: screen_col.max(0) as usize,
        });
    }

    pub fn selection_auto_scroll_direction(&self, screen_row: i64) -> Option<SelectionScrollDirection> {
        let anchor = self.selection_anchor?;
        let head = self.selection_head?;
        if self.selection_mode != Some(SelectionMode::Transcript)
            && self.selection_mode != Some(SelectionMode::Table)
        {
            return None;
        }
        let bounds = self.transcript_screen_bounds()?;
        if head.line < anchor.line && screen_row <= bounds.first_row as i64 && self.scroll_top > 0 {
            return Some(SelectionScrollDirection::Up);
        }
        if head.line > anchor.line
            && screen_row >= bounds.last_row as i64
            && self.scroll_top < self.last_max_scroll
        {
            return Some(SelectionScrollDirection::Down);
        }
        None
    }

    pub fn scroll_selection(&mut self, direction: SelectionScrollDirection, screen_col: i64) -> bool {
        if self.selection_anchor.is_none()
            || (self.selection_mode != Some(SelectionMode::Transcript)
                && self.selection_mode != Some(SelectionMode::Table))
        {
            return false;
        }
        let previous_scroll_top = self.scroll_top;
        self.scroll_by(direction.delta());
        if self.scroll_top == previous_scroll_top {
            return false;
        }
        let bounds = match self.transcript_screen_bounds() {
            Some(bounds) => bounds,
            None => return false,
        };
        let row = match direction {
            SelectionScrollDirection::Up => bounds.first_row,
            SelectionScrollDirection::Down => bounds.last_row,
        };
        self.extend_selection(row as i64, screen_col);
        true
    }

    /// Finish the selection and return its plain text (null when empty).
    pub fn end_selection(&mut self) -> Option<String> {
        if self.selection_mode != Some(SelectionMode::Transcript)
            && self.selection_mode != Some(SelectionMode::Table)
        {
            self.clear_selection();
            return None;
        }
        let sel = self.ordered_selection();
        let transcript = self.last_transcript.clone();
        let mode = self.selection_mode;
        let text = match sel {
            None => None,
            Some(sel) => {
                if mode == Some(SelectionMode::Table) {
                    self.extract_table_selection_text(&transcript, sel)
                } else {
                    Self::extract_selection_text(&transcript, sel, &|line, span| self.selection_span(line, span))
                }
            }
        };
        self.clear_selection();
        text
    }

    pub fn extend_active_selection(&mut self, screen_row: i64, screen_col: i64) {
        match self.selection_mode {
            Some(SelectionMode::Frame) => self.extend_frame_selection(screen_row, screen_col),
            Some(SelectionMode::Transcript) | Some(SelectionMode::Table) => {
                self.extend_selection(screen_row, screen_col)
            }
            None => {}
        }
    }

    pub fn end_active_selection(&mut self) -> Option<String> {
        match self.selection_mode {
            Some(SelectionMode::Frame) => self.end_frame_selection(),
            Some(SelectionMode::Transcript) | Some(SelectionMode::Table) => self.end_selection(),
            None => {
                self.clear_selection();
                None
            }
        }
    }

    /// Snapshot and highlight the final screen frame for overlay/dock selection.
    pub fn apply_frame_selection(
        &mut self,
        frame: &mut [String],
        height: usize,
        selectable_regions: &[FrameSelectionRegion],
    ) {
        self.last_frame = frame.to_vec();
        self.last_frame_visible_height = height.min(frame.len());
        self.last_frame_visible_start = frame.len() - self.last_frame_visible_height;
        self.frame_selection_regions = selectable_regions.to_vec();
        if self.selection_mode != Some(SelectionMode::Frame) {
            return;
        }
        let sel = match self.ordered_selection() {
            Some(sel) => sel,
            None => return,
        };
        for line_index in sel.0.line..=sel.1.line {
            if line_index >= frame.len() {
                continue;
            }
            let mut line = frame[line_index].clone();
            let spans = self.selected_frame_spans(line_index, sel, &self.frame_selection_regions);
            for i in (0..spans.len()).rev() {
                line = Self::highlight_line(&line, spans[i]);
            }
            frame[line_index] = line;
        }
    }

    pub fn begin_frame_selection(&mut self, screen_row: i64, screen_col: i64) -> bool {
        let point = match self.frame_point(screen_row, screen_col, None) {
            Some(point) => point,
            None => {
                self.clear_selection();
                return false;
            }
        };
        if !self.is_frame_selectable(point) {
            self.clear_selection();
            return false;
        }
        self.active_frame_selection = Some(FrameSelectionSnapshot {
            frame: self.last_frame.clone(),
            regions: self.frame_selection_regions.clone(),
            visible_start: self.last_frame_visible_start,
            visible_height: self.last_frame_visible_height,
        });
        self.selection_anchor = Some(point);
        self.selection_head = Some(point);
        self.selection_mode = Some(SelectionMode::Frame);
        true
    }

    pub fn extend_frame_selection(&mut self, screen_row: i64, screen_col: i64) {
        if self.selection_anchor.is_none() || self.selection_mode != Some(SelectionMode::Frame) {
            return;
        }
        let snapshot = self.active_frame_selection.clone();
        let point = match self.frame_point(screen_row, screen_col, snapshot) {
            Some(point) => point,
            None => return,
        };
        let clamped = match self.clamp_frame_selection_point(point) {
            Some(clamped) => clamped,
            None => return,
        };
        self.selection_head = Some(clamped);
    }

    pub fn end_frame_selection(&mut self) -> Option<String> {
        if self.selection_mode != Some(SelectionMode::Frame) {
            self.clear_selection();
            return None;
        }
        let sel = self.ordered_selection();
        let source_lines = self
            .active_frame_selection
            .as_ref()
            .map(|active| active.frame.clone())
            .unwrap_or_else(|| self.last_frame.clone());
        let regions = self
            .active_frame_selection
            .as_ref()
            .map(|active| active.regions.clone())
            .unwrap_or_else(|| self.frame_selection_regions.clone());
        self.clear_selection();
        let sel = match sel {
            Some(sel) => sel,
            None => return None,
        };
        self.extract_frame_selection_text(&source_lines, &regions, sel)
    }

    fn highlight_line(line: &str, span: ColumnSpan) -> String {
        let width = visible_width(line);
        let from = span.from.min(width);
        let to = span.to.min(width);
        if to <= from {
            return line.to_string();
        }
        let before = slice_by_column(line, 0, from, false);
        let selected = strip_ansi(&slice_by_column(line, from, to - from, false));
        let after = slice_by_column(line, to, width.saturating_sub(to), false);
        format!("{before}\x1b[0m\x1b[7m{selected}\x1b[27m{after}")
    }

    fn frame_point(
        &self,
        screen_row: i64,
        screen_col: i64,
        snapshot: Option<FrameSelectionSnapshot>,
    ) -> Option<SelectionPoint> {
        let visible_height = snapshot
            .as_ref()
            .map(|s| s.visible_height)
            .unwrap_or(self.last_frame_visible_height);
        if visible_height == 0 {
            return None;
        }
        let visible_start = snapshot
            .as_ref()
            .map(|s| s.visible_start)
            .unwrap_or(self.last_frame_visible_start);
        let frame_length = snapshot
            .as_ref()
            .map(|s| s.frame.len())
            .unwrap_or(self.last_frame.len());
        let row = screen_row.clamp(0, visible_height as i64 - 1) as usize;
        let line = visible_start + row;
        if line >= frame_length {
            return None;
        }
        Some(SelectionPoint {
            line,
            col: screen_col.max(0) as usize,
        })
    }

    fn transcript_screen_bounds(&self) -> Option<TranscriptBounds> {
        if self.last_window_height == 0 {
            return None;
        }
        let visible_height = if self.last_frame_visible_height > 0 {
            self.last_frame_visible_height
        } else {
            self.last_header_height + self.last_window_height
        };
        if visible_height == 0 {
            return None;
        }
        let visible_start = if self.last_frame_visible_height > 0 {
            self.last_frame_visible_start
        } else {
            0
        };
        let visible_end = visible_start + visible_height - 1;
        let transcript_start = visible_start.max(self.last_header_height + self.last_top_padding);
        let transcript_end = (self.last_header_height + self.last_window_height - 1).min(visible_end);
        if transcript_start > transcript_end {
            return None;
        }
        Some(TranscriptBounds {
            first_row: transcript_start - visible_start,
            last_row: transcript_end - visible_start,
            visible_start,
            visible_height,
            transcript_start,
            transcript_end,
        })
    }

    fn transcript_line_for_screen_row(&self, screen_row: i64, clamp: bool) -> Option<usize> {
        let bounds = self.transcript_screen_bounds()?;
        if !clamp && (screen_row < 0 || screen_row >= bounds.visible_height as i64) {
            return None;
        }
        let row = if clamp {
            screen_row.clamp(0, bounds.visible_height as i64 - 1) as usize
        } else {
            screen_row as usize
        };
        let frame_line = bounds.visible_start + row;
        if !clamp && (frame_line < bounds.transcript_start || frame_line > bounds.transcript_end) {
            return None;
        }
        Some(self.scroll_top + frame_line.clamp(bounds.transcript_start, bounds.transcript_end) - self.last_top_padding - self.last_header_height)
    }

    fn is_frame_selectable(&self, point: SelectionPoint) -> bool {
        self.frame_selection_regions.iter().any(|region| {
            region.line == point.line
                && point.col >= region.col
                && point.col < region.col + region.width
        })
    }

    fn table_regions(&self, table: usize) -> Vec<TableCellSelectionRegion> {
        self.table_cell_selection_regions
            .iter()
            .filter(|region| region.table == table)
            .cloned()
            .collect()
    }

    fn table_at_point(&self, point: SelectionPoint) -> Option<usize> {
        let mut tables: Vec<usize> = Vec::new();
        for region in &self.table_cell_selection_regions {
            if !tables.contains(&region.table) {
                tables.push(region.table);
            }
        }
        for table in tables {
            let regions = self.table_regions(table);
            if let Some(region) = regions.first() {
                if point.line >= region.table_top
                    && point.line <= region.table_bottom
                    && point.col >= region.table_left.saturating_sub(1)
                    && point.col <= region.table_right
                {
                    return Some(table);
                }
            }
        }
        None
    }

    fn closest_table_cell(&self, table: usize, point: SelectionPoint) -> Option<TableCellPosition> {
        let mut closest: Option<TableCellPosition> = None;
        let mut closest_line_distance = i64::MAX;
        let mut closest_column_distance = i64::MAX;
        for region in self.table_regions(table) {
            let line_distance = (point.line as i64 - region.line as i64).abs();
            let end = region.col + region.width;
            let point_col = point.col as i64;
            let column_distance = if point_col < region.col as i64 {
                region.col as i64 - point_col
            } else if point_col > end as i64 {
                point_col - end as i64
            } else {
                0
            };
            if line_distance < closest_line_distance
                || (line_distance == closest_line_distance && column_distance < closest_column_distance)
            {
                closest = Some(TableCellPosition {
                    row: region.row,
                    column: region.column,
                });
                closest_line_distance = line_distance;
                closest_column_distance = column_distance;
            }
        }
        closest
    }

    fn active_table_range(&self) -> Option<TableSelectionRange> {
        let active = self.active_table_selection?;
        let head = self.selection_head?;
        let head_cell = self.closest_table_cell(active.table, head)?;
        Some(TableSelectionRange {
            from_row: active.anchor.row.min(head_cell.row),
            to_row: active.anchor.row.max(head_cell.row),
            from_column: active.anchor.column.min(head_cell.column),
            to_column: active.anchor.column.max(head_cell.column),
        })
    }

    fn selected_table_spans(&self, line_index: usize, sel: (SelectionPoint, SelectionPoint)) -> Vec<ColumnSpan> {
        let active = match self.active_table_selection {
            Some(active) => active,
            None => return Vec::new(),
        };
        let range = match self.active_table_range() {
            Some(range) => range,
            None => return Vec::new(),
        };
        let single_cell = range.from_row == range.to_row && range.from_column == range.to_column;
        let selection_span = if single_cell {
            self.selection_span(line_index, sel)
        } else {
            None
        };
        if single_cell && selection_span.is_none() {
            return Vec::new();
        }

        let mut spans: Vec<ColumnSpan> = Vec::new();
        for region in &self.table_cell_selection_regions {
            if region.line != line_index
                || region.table != active.table
                || region.row < range.from_row
                || region.row > range.to_row
                || region.column < range.from_column
                || region.column > range.to_column
            {
                continue;
            }
            let from = selection_span
                .map(|span| span.from.max(region.col))
                .unwrap_or(region.col);
            let to = selection_span
                .map(|span| span.to.min(region.col + region.width))
                .unwrap_or(region.col + region.width);
            if to > from {
                spans.push(ColumnSpan { from, to });
            }
        }
        spans.sort_by_key(|span| span.from);
        spans
    }

    fn frame_regions_for_line(&self, line: usize, regions: &[FrameSelectionRegion]) -> Vec<FrameSelectionRegion> {
        let mut filtered: Vec<FrameSelectionRegion> = regions
            .iter()
            .filter(|region| region.line == line && region.width > 0)
            .cloned()
            .collect();
        filtered.sort_by_key(|region| region.col);
        filtered
    }

    fn clamp_frame_selection_point(&self, point: SelectionPoint) -> Option<SelectionPoint> {
        let regions = self.frame_regions_for_line(point.line, &self.frame_selection_regions);
        if regions.is_empty() {
            return None;
        }
        let mut closest = regions[0].col;
        let mut distance = usize::MAX;
        for region in &regions {
            let start = region.col;
            let end = region.col + region.width;
            if point.col >= start && point.col <= end {
                return Some(SelectionPoint {
                    line: point.line,
                    col: point.col.clamp(start, end),
                });
            }
            for col in [start, end] {
                let next_distance = point.col.abs_diff(col);
                if next_distance < distance {
                    closest = col;
                    distance = next_distance;
                }
            }
        }
        Some(SelectionPoint {
            line: point.line,
            col: closest,
        })
    }

    fn selected_frame_spans(
        &self,
        line_index: usize,
        sel: (SelectionPoint, SelectionPoint),
        regions: &[FrameSelectionRegion],
    ) -> Vec<ColumnSpan> {
        let span = match self.selection_span(line_index, sel) {
            Some(span) => span,
            None => return Vec::new(),
        };
        let mut spans: Vec<ColumnSpan> = Vec::new();
        for region in self.frame_regions_for_line(line_index, regions) {
            let from = span.from.max(region.col);
            let to = span.to.min(region.col + region.width);
            if to > from {
                spans.push(ColumnSpan { from, to });
            }
        }
        spans
    }

    fn extract_frame_selection_text(
        &self,
        source_lines: &[String],
        regions: &[FrameSelectionRegion],
        sel: (SelectionPoint, SelectionPoint),
    ) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        for line_index in sel.0.line..=sel.1.line {
            let line = source_lines.get(line_index).cloned().unwrap_or_default();
            let spans = self.selected_frame_spans(line_index, sel, regions);
            if spans.is_empty() {
                continue;
            }
            let mut parts: Vec<String> = Vec::new();
            for span in spans {
                parts.push(strip_ansi(&slice_by_column(
                    &line,
                    span.from,
                    span.to.saturating_sub(span.from),
                    false,
                )));
            }
            lines.push(trim_end(&parts.join("")));
        }
        let text = lines.join("\n");
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn extract_selection_text(
        source_lines: &[String],
        sel: (SelectionPoint, SelectionPoint),
        span_for: &dyn Fn(usize, (SelectionPoint, SelectionPoint)) -> Option<ColumnSpan>,
    ) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        for line_index in sel.0.line..=sel.1.line {
            let line = source_lines.get(line_index).cloned().unwrap_or_default();
            let span = match span_for(line_index, sel) {
                Some(span) => span,
                None => continue,
            };
            let width = visible_width(&line);
            let from = span.from.min(width);
            let to = span.to.min(width);
            lines.push(trim_end(&strip_ansi(&slice_by_column(
                &line,
                from,
                to.saturating_sub(from),
                false,
            ))));
        }
        let text = lines.join("\n");
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn compare_selection_points(a: SelectionPoint, b: SelectionPoint) -> i64 {
        if a.line == b.line {
            a.col as i64 - b.col as i64
        } else {
            a.line as i64 - b.line as i64
        }
    }

    fn extract_table_selection_text(
        &self,
        source_lines: &[String],
        sel: (SelectionPoint, SelectionPoint),
    ) -> Option<String> {
        let active = match self.active_table_selection {
            Some(active) => active,
            None => return None,
        };
        let range = match self.active_table_range() {
            Some(range) => range,
            None => return None,
        };

        if range.from_row != range.to_row || range.from_column != range.to_column {
            let mut contents: Vec<(String, String)> = Vec::new();
            for region in self.table_regions(active.table) {
                contents.push((format!("{}:{}", region.row, region.column), region.content));
            }
            let lookup = |row: i64, column: i64| -> String {
                contents
                    .iter()
                    .find(|(key, _)| *key == format!("{row}:{column}"))
                    .map(|(_, value)| value.clone())
                    .unwrap_or_default()
            };
            let mut rows: Vec<String> = Vec::new();
            for row in range.from_row..=range.to_row {
                let mut cells: Vec<String> = Vec::new();
                for column in range.from_column..=range.to_column {
                    cells.push(lookup(row, column));
                }
                rows.push(cells.join("\t"));
            }
            let text = rows.join("\n");
            return if text.trim().is_empty() { None } else { Some(text) };
        }

        let mut cell_regions: Vec<TableCellSelectionRegion> = self
            .table_regions(active.table)
            .into_iter()
            .filter(|region| region.row == range.from_row && region.column == range.from_column)
            .collect();
        cell_regions.sort_by_key(|region| (region.line, region.segment));
        let first = cell_regions.first().cloned();
        let last = cell_regions.last().cloned();
        if let (Some(first), Some(last)) = (first, last) {
            let cell_start = SelectionPoint {
                line: first.line,
                col: first.col,
            };
            let cell_end = SelectionPoint {
                line: last.line,
                col: last.col + last.width,
            };
            if Self::compare_selection_points(sel.0, cell_start) <= 0
                && Self::compare_selection_points(sel.1, cell_end) >= 0
            {
                return if first.content.trim().is_empty() {
                    None
                } else {
                    Some(first.content)
                };
            }
        }

        let mut lines: Vec<String> = Vec::new();
        for line_index in sel.0.line..=sel.1.line {
            let line = source_lines.get(line_index).cloned().unwrap_or_default();
            let spans = self.selected_table_spans(line_index, sel);
            if spans.is_empty() {
                continue;
            }
            let parts: Vec<String> = spans
                .iter()
                .map(|span| {
                    strip_ansi(&slice_by_column(
                        &line,
                        span.from,
                        span.to.saturating_sub(span.from),
                        false,
                    ))
                })
                .collect();
            lines.push(trim_end(&parts.join("")));
        }
        let text = lines.join("\n");
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    pub fn clear_selection(&mut self) {
        self.selection_anchor = None;
        self.selection_head = None;
        self.selection_mode = None;
        self.active_frame_selection = None;
        self.active_table_selection = None;
    }

    pub fn has_selection(&self) -> bool {
        self.ordered_selection().is_some()
    }

    /// OSC 8 hyperlink URL at a screen position in the last painted frame, or
    /// null when the position is not over a hyperlink. Covers the transcript
    /// window, the dock, and composited overlays.
    pub fn hyperlink_at(&self, screen_row: i64, screen_col: i64) -> Option<String> {
        if screen_row < 0 || screen_col < 0 || self.last_frame_visible_height == 0 {
            return None;
        }
        if screen_row >= self.last_frame_visible_height as i64 {
            return None;
        }
        let line = self
            .last_frame
            .get(self.last_frame_visible_start + screen_row as usize)?;
        if is_image_line(line) {
            return None;
        }
        url_at_column(line, screen_col)
    }

    /// Row-diff a composed frame against the previous one with absolute addressing.
    pub fn paint(
        &mut self,
        write: &mut dyn FnMut(&str),
        frame: &[String],
        width: usize,
        height: usize,
        cursor_pos: Option<(usize, usize)>,
    ) {
        let frame: Vec<String> = if frame.len() > height {
            frame[frame.len() - height..].to_vec()
        } else {
            frame.to_vec()
        };

        let mut buffer = String::from("\x1b[?2026h");
        if width != self.prev_width || height != self.prev_height || self.prev_frame.is_empty() {
            buffer.push_str("\x1b[2J\x1b[H");
            self.prev_frame = Vec::new();
        }
        for row in 0..height {
            let line = frame.get(row).cloned().unwrap_or_default();
            if self.prev_frame.get(row) == Some(&line) {
                continue;
            }
            buffer.push_str(&format!("\x1b[{};1H\x1b[2K", row + 1));
            // an overwide line would wrap and shear the grid; clamp instead of crash
            buffer.push_str(&if visible_width(&line) > width {
                slice_by_column(&line, 0, width, true)
            } else {
                line
            });
        }
        if let Some((row, col)) = cursor_pos {
            buffer.push_str(&format!(
                "\x1b[{};{}H",
                row.min(height - 1) + 1,
                col + 1
            ));
        }
        buffer.push_str("\x1b[?2026l");
        write(&buffer);

        self.prev_frame = frame;
        self.prev_width = width;
        self.prev_height = height;
    }

    /// Force the next paint to clear and repaint the whole screen.
    pub fn reset(&mut self) {
        self.prev_frame = Vec::new();
    }

    /// Scrolling up pauses following; reaching the bottom resumes it.
    pub fn scroll_by(&mut self, delta: i64) {
        let base = if self.following {
            self.last_max_scroll
        } else {
            self.scroll_top
        };
        let next = base as i64 + delta;
        self.scroll_top = next.clamp(0, self.last_max_scroll as i64) as usize;
        self.following = (delta >= 0 || !self.last_bottom_aligned) && self.scroll_top >= self.last_max_scroll;
    }

    pub fn scroll_to_top(&mut self) {
        self.scroll_top = 0;
        self.following = !self.last_bottom_aligned && self.last_max_scroll == 0;
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_top = self.last_max_scroll;
        self.following = true;
    }

    pub fn page_size(&self) -> usize {
        self.last_window_height.saturating_sub(1).max(1)
    }

    pub fn window_height(&self) -> usize {
        self.last_window_height
    }

    pub fn header_height(&self) -> usize {
        self.last_header_height
    }

    pub fn is_following(&self) -> bool {
        self.following
    }

    pub fn scroll_info(&self) -> ScrollInfo {
        ScrollInfo {
            following: self.following,
            lines_below: self.last_max_scroll.saturating_sub(self.scroll_top),
            lines_above: self.scroll_top,
        }
    }
}

fn trim_end(s: &str) -> String {
    s.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn dock_height_is_clipped_to_leave_transcript_rows() {
        assert_eq!(clipped_fullscreen_dock_height(5, 10), 5);
        assert_eq!(clipped_fullscreen_dock_height(20, 10), 7);
        assert_eq!(clipped_fullscreen_dock_height(4, 2), 0);
    }

    #[test]
    fn neon_header_stays_fixed_and_selection_uses_transcript_coordinates() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["first", "second", "third", "fourth", "fifth"]);
        let header = lines(&["OPTIMUS", "session"]);
        let dock = lines(&["input", "status"]);
        let frame = viewport.compose_frame_with_header(&transcript, &dock, 7, &[], &[], false, &header);
        assert_eq!(frame, lines(&["OPTIMUS", "session", "third", "fourth", "fifth", "input", "status"]));
        viewport.apply_frame_selection(&mut frame.clone(), 7, &[]);
        assert!(!viewport.begin_selection(0, 0));
        assert!(!viewport.begin_selection(1, 0));
        assert!(!viewport.begin_selection(5, 0));
        assert!(viewport.begin_selection(2, 0));
        viewport.extend_selection(3, 6);
        assert_eq!(viewport.end_selection().as_deref(), Some("third\nfourth"));
        viewport.scroll_by(-1);
        let frame = viewport.compose_frame_with_header(&transcript, &dock, 7, &[], &[], false, &header);
        assert_eq!(&frame[..3], &lines(&["OPTIMUS", "session", "second"]));
        assert_eq!(viewport.page_size(), 2);
    }

    #[test]
    fn neon_header_yields_to_input_on_tiny_terminals() {
        let mut viewport = FullscreenViewport::new();
        for height in 0..12 {
            let frame = viewport.compose_frame_with_header(&lines(&["message"]), &lines(&["input", "status"]), height, &[], &[], true, &lines(&["brand", "art", "session"]));
            assert_eq!(frame.len(), height);
            assert!(viewport.header_height() <= height.saturating_sub(5));
            if height >= 5 {
                assert_eq!(&frame[height - 2..], &lines(&["input", "status"]));
            }
        }
    }

    #[test]
    fn neon_header_resize_preserves_follow_and_bottom_aligned_selection() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["hello", "world"]);
        let frame = viewport.compose_frame_with_header(&transcript, &lines(&["input"]), 8, &[], &[], true, &lines(&["brand", "session"]));
        assert_eq!(frame, lines(&["brand", "session", "", "", "", "hello", "world", "input"]));
        assert!(!viewport.begin_selection(4, 0));
        assert!(viewport.begin_selection(5, 0));
        viewport.extend_selection(6, 5);
        assert_eq!(viewport.end_selection().as_deref(), Some("hello\nworld"));
        let frame = viewport.compose_frame_with_header(&transcript, &lines(&["input"]), 4, &[], &[], true, &lines(&["brand", "session"]));
        assert_eq!(frame, lines(&["", "hello", "world", "input"]));
        assert!(viewport.is_following());
    }

    #[test]
    fn following_pins_window_to_transcript_end() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["1", "2", "3", "4", "5"]);
        let dock = lines(&["dock"]);
        let frame = viewport.compose_frame(&transcript, &dock, 4, &[]);
        assert_eq!(frame, lines(&["3", "4", "5", "dock"]));
        assert_eq!(viewport.scroll_info().lines_below, 0);
        assert!(viewport.is_following());
    }

    #[test]
    fn bottom_aligned_short_tail_keeps_selection_coordinates_and_up_scroll_intent() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["recent", "live"]);
        let frame = viewport.compose_frame_anchored(&transcript, &lines(&["dock"]), 6, &[], &[], true);
        assert_eq!(frame, lines(&["", "", "", "recent", "live", "dock"]));
        assert!(!viewport.begin_selection(0, 0));
        assert!(viewport.begin_selection(3, 0));
        viewport.extend_selection(4, 4);
        assert_eq!(viewport.end_selection(), Some("recent\nlive".into()));
        viewport.scroll_by(-1);
        assert!(!viewport.is_following(), "scrolling up a short tail must pause follow before backfill");
        viewport.scroll_to_bottom();
        assert!(viewport.is_following());
    }

    #[test]
    fn message_anchor_survives_prepend_reflow_and_async_backfill_while_following() {
        let mut viewport = FullscreenViewport::new();
        let anchor = |key: &str, offset| Some(ViewportAnchor { key: Rc::from(key), offset });
        let transcript = lines(&["header", "first", "second", "third", "live"]);
        let anchors = vec![None, anchor("message", 0), anchor("message", 5), anchor("message", 11), anchor("live", 0)];
        viewport.compose_frame_anchored(&transcript, &[], 3, &[], &anchors, true);
        viewport.scroll_by(-1);
        assert_eq!(viewport.compose_frame_anchored(&transcript, &[], 3, &[], &anchors, true)[0], "first");
        let prepended = lines(&["older 1", "older 2", "header", "first", "second", "third", "live update", "later 1", "later 2"]);
        let prepended_anchors = vec![anchor("older", 0), anchor("older", 7), None, anchor("message", 0), anchor("message", 5), anchor("message", 11), anchor("live", 0), anchor("later", 0), anchor("later", 7)];
        assert_eq!(viewport.compose_frame_anchored(&prepended, &[], 3, &[], &prepended_anchors, true)[0], "first");
        viewport.scroll_by(1);
        viewport.compose_frame_anchored(&prepended, &[], 3, &[], &prepended_anchors, true);
        let reflowed = lines(&["old", "er 1", "old", "er 2", "header", "fir", "stse", "cond", "third", "live update", "later 1", "later 2"]);
        let reflowed_anchors = vec![anchor("older", 0), anchor("older", 3), anchor("older", 7), anchor("older", 10), None, anchor("message", 0), anchor("message", 3), anchor("message", 7), anchor("message", 11), anchor("live", 0), anchor("later", 0), anchor("later", 7)];
        assert_eq!(viewport.compose_frame_anchored(&reflowed, &[], 3, &[], &reflowed_anchors, true)[0], "stse");
        viewport.scroll_to_bottom();
        let mut new_tail = reflowed.clone();
        new_tail.push("latest".into());
        assert_eq!(viewport.compose_frame_anchored(&new_tail, &[], 3, &[], &reflowed_anchors, true)[2], "latest");
    }

    #[test]
    fn scroll_by_pauses_and_resumes_following() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["1", "2", "3", "4", "5"]);
        viewport.compose_frame(&transcript, &[], 3, &[]);
        viewport.scroll_by(-1);
        assert_eq!(viewport.scroll_info().lines_above, 1);
        assert!(!viewport.is_following());
        viewport.scroll_by(1);
        assert!(viewport.is_following());
    }

    #[test]
    fn image_lines_are_replaced_by_placeholder() {
        let mut viewport = FullscreenViewport::new();
        let transcript = vec!["\x1b_Ga=T,f=100\x1b\\".to_string()];
        let frame = viewport.compose_frame(&transcript, &[], 2, &[]);
        assert_eq!(frame[0], IMAGE_PLACEHOLDER);
        assert_eq!(frame[1], "");
    }

    #[test]
    fn paint_diffs_rows_and_addresses_absolutely() {
        let mut viewport = FullscreenViewport::new();
        let output = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
        {
            let output = output.clone();
            let mut write = move |data: &str| output.borrow_mut().push_str(data);
            viewport.paint(&mut write, &lines(&["a", "b"]), 10, 2, None);
        }
        let first = output.borrow().clone();
        assert!(first.starts_with("\x1b[?2026h\x1b[2J\x1b[H"));
        assert!(first.contains("\x1b[1;1H\x1b[2Ka"));
        assert!(first.ends_with("\x1b[?2026l"));

        output.borrow_mut().clear();
        {
            let output = output.clone();
            let mut write = move |data: &str| output.borrow_mut().push_str(data);
            viewport.paint(&mut write, &lines(&["a", "c"]), 10, 2, None);
        }
        let second = output.borrow().clone();
        assert!(!second.contains("\x1b[2J"));
        assert!(second.contains("\x1b[2;1H\x1b[2Kc"));
        assert!(!second.contains("\x1b[1;1H"));
    }

    #[test]
    fn transcript_selection_extracts_plain_text() {
        let mut viewport = FullscreenViewport::new();
        let transcript = lines(&["hello", "world"]);
        viewport.compose_frame(&transcript, &[], 2, &[]);
        assert!(viewport.begin_selection(0, 0));
        viewport.extend_selection(1, 3);
        assert_eq!(viewport.end_selection().as_deref(), Some("hello\nwor"));
        assert!(!viewport.has_selection());
    }
}
