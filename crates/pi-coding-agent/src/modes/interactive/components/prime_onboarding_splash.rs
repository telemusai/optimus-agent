//! Port of packages/coding-agent/src/modes/interactive/components/prime-onboarding-splash.ts

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use pi_tui::keybindings::get_keybindings;
use pi_tui::tui::Component;
use pi_tui::utils::visible_width;

use crate::modes::interactive::theme::theme::theme;
use crate::themes::optimus_logo::{colorize_optimus_logo, get_optimus_logo};

/// `ANIMATION_INTERVAL_MS`
pub const ANIMATION_INTERVAL_MS: u64 = 120;
/// `LAB_FIELD_HEIGHT`
pub const LAB_FIELD_HEIGHT: usize = 14;
/// `LAB_FIELD_MIN_WIDTH`
pub const LAB_FIELD_MIN_WIDTH: usize = 42;
/// `LAB_FIELD_MAX_WIDTH`
pub const LAB_FIELD_MAX_WIDTH: usize = 78;

/// Port of `PrimeOnboardingSplashOptions`.
#[derive(Default)]
pub struct PrimeOnboardingSplashOptions {
    pub get_rows: Option<Box<dyn Fn() -> f64>>,
    pub request_render: Option<Box<dyn Fn()>>,
    pub animation_interval_ms: Option<u64>,
    pub continue_action_label: Option<String>,
}

/// Port of `SplashTone`.
pub type SplashTone = &'static str;

/// Port of `SplashCell`.
#[derive(Debug, Clone, PartialEq)]
struct SplashCell {
    char: String,
    tone: SplashTone,
    priority: i64,
    bold: bool,
    italic: bool,
}

impl SplashCell {
    fn blank() -> Self {
        Self {
            char: " ".to_string(),
            tone: "dim",
            priority: 0,
            bold: false,
            italic: false,
        }
    }
}

/// Port of `StyledText`.
#[derive(Debug, Clone, PartialEq)]
struct StyledText {
    text: String,
    tone: SplashTone,
    bold: bool,
    italic: bool,
    transparent_spaces: bool,
}

fn styled(text: &str, tone: SplashTone) -> StyledText {
    StyledText {
        text: text.to_string(),
        tone,
        bold: false,
        italic: false,
        transparent_spaces: false,
    }
}

fn styled_bold(text: &str, tone: SplashTone) -> StyledText {
    StyledText {
        bold: true,
        ..styled(text, tone)
    }
}

/// Port of `PanelTextLine`.
type PanelTextLine = Vec<StyledText>;

/// Port of `PanelLine`.
struct PanelLine {
    left: usize,
    parts: PanelTextLine,
}

/// Port of `QuietZone`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuietZone {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

/// Port of `PrimeOnboardingSplashComponent`.
pub struct PrimeOnboardingSplashComponent {
    frame: Arc<AtomicI64>,
    /// `ReturnType<typeof setInterval>` for the animation timer.
    animation_task: Option<tokio::task::JoinHandle<()>>,
    progress_message: Option<String>,
    on_select: Box<dyn FnMut()>,
    on_cancel: Box<dyn FnMut()>,
    options: PrimeOnboardingSplashOptions,
}

impl PrimeOnboardingSplashComponent {
    pub fn new(
        on_select: Box<dyn FnMut()>,
        on_cancel: Box<dyn FnMut()>,
        options: PrimeOnboardingSplashOptions,
    ) -> Self {
        let frame = Arc::new(AtomicI64::new(0));
        let mut component = Self {
            frame,
            animation_task: None,
            progress_message: None,
            on_select,
            on_cancel,
            options,
        };
        if component.options.request_render.is_some() {
            let interval_ms = component
                .options
                .animation_interval_ms
                .unwrap_or(ANIMATION_INTERVAL_MS);
            let frame = Arc::clone(&component.frame);
            // Port of `setInterval(() => { this.frame++; options.requestRender?.(); }, ms)`.
            // The `requestRender` callback is not `Send`, so the timer task only
            // advances the shared frame counter; the TUI reads it while rendering,
            // exactly like the pi-tui `Loader` port does for its animation.
            component.animation_task = Some(tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_millis(interval_ms));
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                ticker.tick().await;
                loop {
                    ticker.tick().await;
                    frame.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }
        component
    }

    /// Port of `dispose`.
    pub fn dispose(&mut self) {
        if self.animation_task.is_none() {
            return;
        }
        if let Some(task) = self.animation_task.take() {
            task.abort();
        }
    }

    /// Port of `showProgress`.
    pub fn show_progress(&mut self, message: &str) {
        self.progress_message = Some(message.to_string());
        self.dispose();
        if let Some(request_render) = self.options.request_render.as_ref() {
            request_render();
        }
    }

    /// `this.frame` as read by the render path.
    pub fn frame(&self) -> i64 {
        self.frame.load(Ordering::SeqCst)
    }
}

impl PrimeOnboardingSplashComponent {
    /// Port of `formatContinueHint`.
    fn format_continue_hint(&self) -> PanelTextLine {
        if let Some(progress_message) = &self.progress_message {
            return vec![styled(progress_message, "muted")];
        }
        let action_label = self
            .options
            .continue_action_label
            .clone()
            .unwrap_or_else(|| "continue".to_string());
        vec![
            styled("Press ", "muted"),
            styled_bold("Enter", "accent"),
            styled(&format!(" to {action_label}"), "muted"),
        ]
    }

    /// Port of `formatBrandLine`.
    fn format_brand_line(&self) -> PanelTextLine {
        vec![
            styled("Welcome to ", "text"),
            styled_bold("OPTIMUS", "brand"),
        ]
    }

    /// Port of `renderPanel`.
    fn render_panel(&self, width: usize, logo: &[String]) -> Vec<PanelLine> {
        let mut lines: Vec<PanelLine> = Vec::new();

        for line in self.render_logo_block(logo) {
            lines.push(self.center_parts(&line, width));
        }
        lines.push(PanelLine {
            left: 0,
            parts: Vec::new(),
        });
        lines.push(PanelLine {
            left: 0,
            parts: Vec::new(),
        });
        lines.push(self.center_parts(&self.format_brand_line(), width));
        lines.push(self.center_parts(
            &vec![styled("by ", "muted"), styled_bold("Telemus AI", "text")],
            width,
        ));
        lines.push(self.center_parts(&self.format_continue_hint(), width));

        lines
    }

    /// Port of `renderLogoBlock`.
    fn render_logo_block(&self, logo: &[String]) -> Vec<PanelTextLine> {
        let logo_width = logo
            .iter()
            .map(|line| visible_width(line))
            .max()
            .unwrap_or(0);
        logo.iter()
            .map(|line| {
                let padded_line = format!(
                    "{line}{}",
                    " ".repeat(logo_width.saturating_sub(visible_width(line)))
                );
                vec![StyledText {
                    text: padded_line,
                    tone: "brand",
                    bold: false,
                    italic: false,
                    transparent_spaces: true,
                }]
            })
            .collect()
    }

    /// Port of `renderBackdrop`.
    fn render_backdrop(
        &self,
        width: usize,
        rows: usize,
        quiet_zone: Option<QuietZone>,
    ) -> Vec<Vec<SplashCell>> {
        let mut canvas: Vec<Vec<SplashCell>> = (0..rows)
            .map(|_| vec![SplashCell::blank(); width])
            .collect();

        self.draw_lab_field(&mut canvas, width, rows, quiet_zone);

        canvas
    }

    /// Port of `drawLabField`.
    fn draw_lab_field(
        &self,
        canvas: &mut [Vec<SplashCell>],
        width: usize,
        rows: usize,
        quiet_zone: Option<QuietZone>,
    ) {
        if width < LAB_FIELD_MIN_WIDTH || rows < LAB_FIELD_HEIGHT + 2 {
            return;
        }

        let lab_width = LAB_FIELD_MIN_WIDTH.max(LAB_FIELD_MAX_WIDTH.min(width.saturating_sub(6)));
        let left = width.saturating_sub(lab_width) / 2;
        let field_top = match quiet_zone {
            Some(zone) => zone.top.max(0).saturating_sub(2),
            None => (rows.saturating_sub(LAB_FIELD_HEIGHT)) / 2,
        };
        let top = field_top.min(rows.saturating_sub(LAB_FIELD_HEIGHT));

        for lab_y in 0..LAB_FIELD_HEIGHT {
            for lab_x in 0..lab_width {
                let x = left + lab_x;
                let y = top + lab_y;
                let cell = match self.lab_cell(lab_x, lab_y, lab_width) {
                    Some(cell) => cell,
                    None => continue,
                };
                if is_inside_quiet_zone(x, y, quiet_zone) {
                    continue;
                }
                put(
                    canvas,
                    x,
                    y,
                    &cell.char,
                    cell.tone,
                    cell.priority,
                    cell.bold,
                    cell.italic,
                );
            }
        }
    }

    /// Port of `labCell`.
    fn lab_cell(&self, x: usize, y: usize, width: usize) -> Option<SplashCell> {
        let height = LAB_FIELD_HEIGHT;
        let frame = self.frame();
        let mut cell: Option<SplashCell> = None;
        let mut set_cell = |glyph: &str, tone: SplashTone, priority: i64| {
            if cell.is_none() || priority >= cell.as_ref().map(|c| c.priority).unwrap_or(i64::MIN) {
                cell = Some(SplashCell {
                    char: glyph.to_string(),
                    tone,
                    priority,
                    bold: false,
                    italic: false,
                });
            }
        };

        let x_i = x as i64;
        let y_i = y as i64;
        let width_i = width as i64;
        let height_i = height as i64;

        let hash = mod_js(x_i * 37 + y_i * 53 + frame * 11 + x_i * y_i * 3, 101);
        if hash < 3 {
            set_cell("\u{00b7}", "dim", 1);
        }

        let center_x = (width_i * 36) / 100;
        let center_y = (height_i * 54) / 100;
        let contour = (x_i - center_x).abs() + (y_i - center_y).abs() * 4 + x_i / 6 - frame;
        if x_i < (width_i * 82) / 100 && mod_js(contour, 24) == 12 {
            set_cell(
                if (x_i + y_i) % 5 == 0 {
                    "\u{254c}"
                } else {
                    "\u{00b7}"
                },
                "borderMuted",
                2,
            );
        }

        let horizon_y = (height_i * 58) / 100;
        if y_i == horizon_y && x_i % 2 == 0 && mod_js(x_i + frame, 13) < 2 {
            set_cell(
                "\u{2500}",
                if x_i > (width_i * 60) / 100 {
                    "accent"
                } else {
                    "dim"
                },
                3,
            );
        }

        let scan_start = 0.max(width_i / 2 - 5);
        if x_i >= scan_start {
            let scan_offset = x_i - scan_start;
            if scan_offset % 4 == 0 {
                let scan_index = scan_offset / 4;
                let segment = mod_js(y_i + scan_index * 2 + frame / 2, 6);
                if y_i > 0 && y_i < height_i - 1 && segment < 2 {
                    set_cell(
                        if (scan_index + y_i) % 4 == 0 {
                            "\u{2503}"
                        } else {
                            "\u{254e}"
                        },
                        "mdLink",
                        4,
                    );
                }
            }
        }

        for trace_index in 0..3i64 {
            let base = if trace_index == 0 {
                (height_i * 30) / 100
            } else if trace_index == 1 {
                (height_i * 49) / 100
            } else {
                (height_i * 72) / 100
            };
            let mut wave = mod_js(x_i * 2 + frame + trace_index * 7, 16);
            if wave > 7 {
                wave = 15 - wave;
            }
            // `Math.trunc((wave - 3) / 2)`
            let trace_y = base + trunc_div(wave - 3, 2);
            if y_i != trace_y {
                continue;
            }

            if mod_js(x_i + frame + trace_index * 13, 41) == 0 {
                set_cell("\u{25c6}", "warning", 5);
            } else if mod_js(x_i + frame, 12) == 0 {
                set_cell("\u{2022}", "accent", 5);
            } else {
                set_cell("\u{00b7}", "accent", 3);
            }
        }

        cell
    }

    /// Port of `drawStyledText`.
    fn draw_styled_text(
        &self,
        canvas: &mut [Vec<SplashCell>],
        left: usize,
        y: usize,
        parts: &PanelTextLine,
        priority: i64,
    ) {
        let mut x = left;
        for part in parts {
            for char in part.text.chars() {
                let width = visible_width(&char.to_string());
                if char != ' ' || !part.transparent_spaces {
                    put(
                        canvas,
                        x,
                        y,
                        &char.to_string(),
                        part.tone,
                        priority,
                        part.bold,
                        part.italic,
                    );
                }
                x += width.max(1);
            }
        }
    }

    /// Port of `renderCells`.
    fn render_cells(&self, cells: &[SplashCell]) -> String {
        let mut rendered = String::new();
        let mut current_tone: Option<SplashTone> = None;
        let mut current_bold = false;
        let mut current_italic = false;
        let mut segment = String::new();

        fn flush(
            rendered: &mut String,
            segment: &mut String,
            current_tone: Option<SplashTone>,
            current_bold: bool,
            current_italic: bool,
        ) {
            if segment.is_empty() || current_tone.is_none() {
                return;
            }
            let tone = current_tone.unwrap();
            let mut styled = if tone == "brand" {
                colorize_optimus_logo(segment)
            } else {
                theme().fg(tone, segment)
            };
            if current_italic {
                styled = theme().italic(&styled);
            }
            if current_bold {
                styled = theme().bold(&styled);
            }
            rendered.push_str(&styled);
            segment.clear();
        }

        for cell in cells {
            let bold = cell.bold;
            let italic = cell.italic;
            if Some(cell.tone) != current_tone || bold != current_bold || italic != current_italic {
                flush(
                    &mut rendered,
                    &mut segment,
                    current_tone,
                    current_bold,
                    current_italic,
                );
                current_tone = Some(cell.tone);
                current_bold = bold;
                current_italic = italic;
            }
            segment.push_str(&cell.char);
        }
        flush(
            &mut rendered,
            &mut segment,
            current_tone,
            current_bold,
            current_italic,
        );
        rendered
    }

    /// Port of `visiblePartsWidth`.
    fn visible_parts_width(&self, parts: &PanelTextLine) -> usize {
        parts.iter().map(|part| visible_width(&part.text)).sum()
    }

    /// Port of `logoQuietZone`.
    fn logo_quiet_zone(
        &self,
        width: usize,
        top_padding: usize,
        rows: usize,
        logo: &[String],
    ) -> Option<QuietZone> {
        let logo_width = logo
            .iter()
            .map(|line| visible_width(line))
            .max()
            .unwrap_or(0);
        if logo_width < 1 || rows < 1 {
            return None;
        }
        let left = width.saturating_sub(logo_width) / 2;
        Some(QuietZone {
            left,
            right: (width.saturating_sub(1)).min(left + logo_width - 1),
            top: top_padding.max(0),
            bottom: (rows - 1).min(top_padding + logo.len() - 1),
        })
    }

    /// Port of `centerParts`.
    fn center_parts(&self, parts: &PanelTextLine, width: usize) -> PanelLine {
        let left = width.saturating_sub(self.visible_parts_width(parts)) / 2;
        PanelLine {
            left,
            parts: parts.clone(),
        }
    }

    /// Port of `getTargetRows`.
    fn get_target_rows(&self, content_line_count: usize) -> usize {
        let requested_rows = self.options.get_rows.as_ref().map(|get_rows| get_rows());
        match requested_rows {
            Some(rows) if rows.is_finite() => content_line_count.max(rows.floor() as usize),
            _ => content_line_count,
        }
    }
}

fn mod_js(value: i64, divisor: i64) -> i64 {
    value.rem_euclid(divisor)
}

/// `Math.trunc(a / b)` for a signed dividend.
fn trunc_div(value: i64, divisor: i64) -> i64 {
    value / divisor
}

/// Port of `isInsideQuietZone`.
fn is_inside_quiet_zone(x: usize, y: usize, quiet_zone: Option<QuietZone>) -> bool {
    match quiet_zone {
        Some(zone) => x >= zone.left && x <= zone.right && y >= zone.top && y <= zone.bottom,
        None => false,
    }
}

/// Port of `put`.
fn put(
    canvas: &mut [Vec<SplashCell>],
    x: usize,
    y: usize,
    char: &str,
    tone: SplashTone,
    priority: i64,
    bold: bool,
    italic: bool,
) {
    if y >= canvas.len() {
        return;
    }
    let row = &mut canvas[y];
    if x >= row.len() {
        return;
    }
    if row[x].priority > priority {
        return;
    }
    row[x] = SplashCell {
        char: char.to_string(),
        tone,
        priority,
        bold,
        italic,
    };
}

impl Component for PrimeOnboardingSplashComponent {
    fn render(&mut self, width: f64) -> Vec<String> {
        let safe_width = (width.max(1.0).floor() as usize).max(1);
        let get_rows = self.options.get_rows.as_ref().map(|get_rows| get_rows());
        let rows_for_logo = (get_rows.unwrap_or(36.0) - 9.0).max(1.0);
        let logo = get_optimus_logo(safe_width as f64, rows_for_logo);
        let panel_lines = self.render_panel(safe_width, &logo);
        let target_rows = self.get_target_rows(panel_lines.len());
        let top_padding = target_rows.saturating_sub(panel_lines.len()) / 2;
        let logo_zone = self.logo_quiet_zone(safe_width, top_padding, target_rows, &logo);
        let mut canvas = self.render_backdrop(safe_width, target_rows, logo_zone);
        for (index, line) in panel_lines.iter().enumerate() {
            self.draw_styled_text(&mut canvas, line.left, top_padding + index, &line.parts, 8);
        }
        canvas.iter().map(|line| self.render_cells(line)).collect()
    }

    fn handle_input(&mut self, key_data: &str) {
        if self.progress_message.is_some() {
            return;
        }
        let kb = get_keybindings();
        if kb.matches(key_data, "tui.select.confirm") {
            (self.on_select)();
            return;
        }
        if kb.matches(key_data, "tui.select.cancel") {
            (self.on_cancel)();
        }
    }

    fn invalidate(&mut self) {
        // Render output is derived from current theme.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn component(options: PrimeOnboardingSplashOptions) -> PrimeOnboardingSplashComponent {
        crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
        PrimeOnboardingSplashComponent::new(Box::new(|| {}), Box::new(|| {}), options)
    }

    #[test]
    fn animation_constants_match_typescript() {
        assert_eq!(ANIMATION_INTERVAL_MS, 120);
        assert_eq!(LAB_FIELD_HEIGHT, 14);
        assert_eq!(LAB_FIELD_MIN_WIDTH, 42);
        assert_eq!(LAB_FIELD_MAX_WIDTH, 78);
    }

    #[test]
    fn mod_matches_javascript_remainder_for_negatives() {
        assert_eq!(mod_js(-1, 24), 23);
        assert_eq!(mod_js(25, 24), 1);
        assert_eq!(trunc_div(-3, 2), -1);
    }

    #[test]
    fn narrow_renders_keep_telemus_branding_inside_the_panel() {
        let mut splash = component(PrimeOnboardingSplashOptions {
            get_rows: Some(Box::new(|| 36.0)),
            ..Default::default()
        });
        for width in [24, 40, 80, 120] {
            let lines = splash.render(width as f64);
            assert_eq!(lines.len(), 36);
            assert!(lines.iter().all(|line| visible_width(line) <= width));
            let text = lines
                .iter()
                .map(|line| pi_tui::utils::strip_ansi(line))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("Welcome to OPTIMUS"));
            assert!(text.contains("by Telemus AI"));
            assert!(text.contains("Press Enter to continue"));
            assert!(!text.contains("Prime Intellect"));
        }
    }

    #[test]
    fn the_lab_field_is_skipped_below_its_minimum_width() {
        let splash = component(PrimeOnboardingSplashOptions::default());
        let canvas = splash.render_backdrop(20, 20, None);
        assert!(canvas.iter().flatten().all(|cell| cell.char == " "));
    }

    #[test]
    fn logo_quiet_zone_covers_the_logo_block() {
        let splash = component(PrimeOnboardingSplashOptions::default());
        let logo = vec!["abcd".to_string()];
        let zone = splash.logo_quiet_zone(10, 2, 20, &logo).unwrap();
        assert_eq!(zone.left, 3);
        assert_eq!(zone.right, 6);
        assert_eq!(zone.top, 2);
        assert_eq!(zone.bottom, 2);
        assert!(splash.logo_quiet_zone(10, 0, 0, &logo).is_none());
        assert!(splash.logo_quiet_zone(10, 0, 5, &[]).is_none());
    }

    #[test]
    fn target_rows_use_the_requested_row_count_when_finite() {
        let splash = component(PrimeOnboardingSplashOptions {
            get_rows: Some(Box::new(|| 40.9)),
            ..Default::default()
        });
        assert_eq!(splash.get_target_rows(5), 40);
        assert_eq!(splash.get_target_rows(100), 100);
        let splash = component(PrimeOnboardingSplashOptions::default());
        assert_eq!(splash.get_target_rows(7), 7);
    }

    #[test]
    fn progress_message_replaces_the_continue_hint_and_blocks_input() {
        let selects = Arc::new(AtomicUsize::new(0));
        let cancels = Arc::new(AtomicUsize::new(0));
        let select_flag = Arc::clone(&selects);
        let cancel_flag = Arc::clone(&cancels);
        let mut splash = PrimeOnboardingSplashComponent::new(
            Box::new(move || {
                select_flag.fetch_add(1, Ordering::SeqCst);
            }),
            Box::new(move || {
                cancel_flag.fetch_add(1, Ordering::SeqCst);
            }),
            PrimeOnboardingSplashOptions::default(),
        );
        splash.handle_input("\r");
        assert_eq!(selects.load(Ordering::SeqCst), 1);
        splash.handle_input("\u{1b}");
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
        splash.show_progress("working");
        splash.handle_input("\r");
        splash.handle_input("\u{1b}");
        assert_eq!(selects.load(Ordering::SeqCst), 1);
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
        assert_eq!(splash.format_continue_hint()[0].text, "working");
    }

    #[test]
    fn the_continue_hint_uses_the_configured_action_label() {
        let splash = component(PrimeOnboardingSplashOptions {
            continue_action_label: Some("pick a team".to_string()),
            ..Default::default()
        });
        let hint = splash.format_continue_hint();
        assert_eq!(hint[2].text, " to pick a team");
        assert!(hint[1].bold);
    }

    #[test]
    fn render_cells_batches_runs_by_tone_bold_and_italic() {
        let splash = component(PrimeOnboardingSplashOptions::default());
        let cells = vec![
            SplashCell {
                char: "a".to_string(),
                tone: "text",
                priority: 8,
                bold: false,
                italic: false,
            },
            SplashCell {
                char: "b".to_string(),
                tone: "text",
                priority: 8,
                bold: true,
                italic: false,
            },
        ];
        let rendered = splash.render_cells(&cells);
        assert_eq!(
            rendered,
            format!(
                "{}{}",
                theme().fg("text", "a"),
                theme().bold(&theme().fg("text", "b"))
            )
        );
    }

    #[test]
    fn dispose_is_idempotent_without_a_timer() {
        let mut splash = component(PrimeOnboardingSplashOptions::default());
        splash.dispose();
        splash.dispose();
        assert_eq!(splash.frame(), 0);
    }
}
