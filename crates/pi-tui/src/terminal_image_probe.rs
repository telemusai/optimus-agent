//! Nonblocking terminal graphics negotiation. Replies are never editor input.
use crate::terminal_image::{
    get_capabilities, get_sixel_palette_size, set_capabilities, set_cell_dimensions,
    set_sixel_palette_size, CellDimensions, ImageProtocol,
};
use std::time::{Duration, Instant};

// XTerm reports palette and raster limits; Windows Terminal advertises SIXEL in DA1.
pub(super) const IMAGE_QUERY: &str = "\x1b[?1;1;0S\x1b[?2;1;0S\x1b[c";
pub(super) const CELL_QUERY: &str = "\x1b[16t";

#[derive(Default)]
pub(super) struct ImageProbe {
    deadline: Option<Instant>,
    partial: String,
    discard_reply: bool,
    cells_requested: bool,
    windows_terminal: bool,
    palette_reported: bool,
}

#[derive(Default)]
pub(super) struct ProbeResult {
    pub input: String,
    pub changed: bool,
    pub query_cells: bool,
}

impl ImageProbe {
    pub fn start(&mut self, now: Instant, windows_terminal: bool) {
        self.deadline = Some(now + Duration::from_millis(250));
        self.partial.clear();
        self.discard_reply = false;
        self.cells_requested = false;
        self.windows_terminal = windows_terminal;
        self.palette_reported = false;
        set_sixel_palette_size(16);
    }

    fn enable_sixel(&mut self) -> ProbeResult {
        let mut caps = get_capabilities();
        let changed = caps.images.is_none();
        if changed {
            caps.images = Some(ImageProtocol::Sixel);
            set_capabilities(caps);
        }
        let query_cells = caps.images == Some(ImageProtocol::Sixel) && !self.cells_requested;
        self.cells_requested |= query_cells;
        ProbeResult { changed, query_cells, ..Default::default() }
    }

    pub fn filter(&mut self, data: &str, now: Instant) -> ProbeResult {
        // StdinBuffer normally delivers complete CSI sequences. Preserve a
        // reply split by its flush timer as well, including late replies.
        if data.starts_with('\x1b') {
            self.partial.clear();
            self.discard_reply = false;
        }
        if self.discard_reply {
            if data.bytes().any(|b| (0x40..=0x7e).contains(&b)) {
                self.discard_reply = false;
            }
            return ProbeResult::default();
        }
        if self.partial.is_empty()
            && !data.starts_with("\x1b[?")
            && !data.starts_with("\x1b[6;")
            && data != "\x1b["
        {
            return ProbeResult {
                input: data.into(),
                ..Default::default()
            };
        }
        if self.partial.len().saturating_add(data.len()) > 256 {
            self.partial.clear();
            self.discard_reply = !data
                .as_bytes()
                .last()
                .is_some_and(|b| (0x40..=0x7e).contains(b));
            return ProbeResult::default();
        }
        self.partial.push_str(data);
        let body = self.partial.strip_prefix("\x1b[").unwrap_or("");
        if body.is_empty()
            || body
                .bytes()
                .all(|b| b.is_ascii_digit() || b == b';' || b == b'?')
        {
            return ProbeResult::default();
        }
        let sequence = std::mem::take(&mut self.partial);
        let timely = self.deadline.is_some_and(|deadline| now <= deadline);
        if let Some(reply) = sequence.strip_prefix("\x1b[?1;").and_then(|s| s.strip_suffix('S')) {
            if let Some(colors) = reply.strip_prefix("0;").and_then(|s| s.parse::<u32>().ok()) {
                if timely && colors >= 2 {
                    let previous = get_sixel_palette_size();
                    self.palette_reported = true;
                    set_sixel_palette_size(colors);
                    return ProbeResult { changed: previous != get_sixel_palette_size(), ..Default::default() };
                }
            }
            return ProbeResult::default();
        }
        if let Some(reply) = sequence.strip_prefix("\x1b[?").and_then(|s| s.strip_suffix('c')) {
            let fields = reply.split(';').map(str::parse::<u32>).collect::<Result<Vec<_>, _>>();
            if let Ok(fields) = fields {
                if timely && fields.iter().skip(1).any(|&feature| feature == 4) {
                    let previous = get_sixel_palette_size();
                    // Microsoft's level-1 DA1 advertises SIXEL, but its palette
                    // is the modern 256-register default, not a VT125 palette.
                    if self.windows_terminal && fields.first() == Some(&61) && !self.palette_reported {
                        set_sixel_palette_size(256);
                    }
                    let mut result = self.enable_sixel();
                    result.changed |= previous != get_sixel_palette_size();
                    return result;
                }
            }
            return ProbeResult::default();
        }
        if let Some(reply) = sequence
            .strip_prefix("\x1b[?2;")
            .and_then(|s| s.strip_suffix('S'))
        {
            let fields: Vec<_> = reply.split(';').collect();
            let geometry = if fields.len() == 3 && fields[0] == "0" {
                fields[1]
                    .parse::<u32>()
                    .ok()
                    .zip(fields[2].parse::<u32>().ok())
                    .filter(|(w, h)| *w > 0 && *h > 0)
            } else {
                None
            };
            if let Some(geometry) = geometry.filter(|_| timely) {
                crate::terminal_image::set_sixel_limits(geometry);
                let mut result = self.enable_sixel();
                result.changed = true;
                return result;
            }
            return ProbeResult::default();
        }
        if let Some(reply) = sequence
            .strip_prefix("\x1b[6;")
            .and_then(|s| s.strip_suffix('t'))
        {
            if let Some((height, width)) = reply.split_once(';') {
                if let (Ok(height_px), Ok(width_px)) = (height.parse(), width.parse()) {
                    let previous = crate::terminal_image::get_cell_dimensions();
                    let known = crate::terminal_image::cell_dimensions_known();
                    set_cell_dimensions(CellDimensions {
                        width_px,
                        height_px,
                    });
                    return ProbeResult {
                        changed: previous != crate::terminal_image::get_cell_dimensions()
                            || known != crate::terminal_image::cell_dimensions_known(),
                        ..Default::default()
                    };
                }
            }
            return ProbeResult::default();
        }
        ProbeResult {
            input: sequence,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal_image::{reset_capabilities_cache, TerminalCapabilities};

    fn fixture() -> (ImageProbe, Instant) {
        set_capabilities(TerminalCapabilities {
            images: None,
            true_color: true,
            hyperlinks: true,
        });
        let now = Instant::now();
        let mut probe = ImageProbe::default();
        probe.start(now, false);
        (probe, now)
    }

    #[test]
    fn fragmented_sixel_reply_enables_images_and_keeps_keyboard_input() {
        let (mut probe, now) = fixture();
        assert_eq!(probe.filter("hello", now).input, "hello");
        for part in ["\x1b[", "?2;", "0;4096;"] {
            let result = probe.filter(part, now);
            assert!(result.input.is_empty() && !result.changed);
        }
        let result = probe.filter("4096S", now);
        assert!(result.changed && result.query_cells && result.input.is_empty());
        assert_eq!(get_capabilities().images, Some(ImageProtocol::Sixel));
        assert_eq!(probe.filter("\x1b[A", now).input, "\x1b[A");
        assert_eq!(probe.filter("\x1b[?1u", now).input, "\x1b[?1u");
        reset_capabilities_cache();
    }

    #[test]
    fn invalid_zero_and_late_replies_leave_images_disabled() {
        for reply in [
            "\x1b[?2;0;0;0S",
            "\x1b[?2;1;4096;4096S",
            "\x1b[?2;0;999999999999999;9S",
        ] {
            let (mut probe, now) = fixture();
            assert!(!probe.filter(reply, now).changed);
            assert_eq!(get_capabilities().images, None);
        }
        let (mut probe, now) = fixture();
        let late = now + Duration::from_millis(300);
        assert_eq!(probe.filter("x", late).input, "x");
        assert!(probe.filter("\x1b[?2;0;4096;4096S", late).input.is_empty());
        assert_eq!(get_capabilities().images, None);
        reset_capabilities_cache();
    }

    #[test]
    fn palette_and_raster_replies_work_in_either_order_after_da1() {
        for palette_first in [true, false] {
            let (mut probe, now) = fixture();
            let da = probe.filter("\x1b[?62;4;22c", now);
            assert!(da.changed && da.query_cells);
            let palette = "\x1b[?1;0;256S";
            let raster = "\x1b[?2;0;4096;4096S";
            for reply in if palette_first { [palette, raster] } else { [raster, palette] } {
                let result = probe.filter(reply, now);
                assert!(result.changed && !result.query_cells && result.input.is_empty());
            }
            assert_eq!(get_sixel_palette_size(), 256);
            assert!(probe.filter("\x1b[?1;", now).input.is_empty());
            assert!(probe.filter("0;16S", now).changed);
            assert_eq!(get_sixel_palette_size(), 16);
            reset_capabilities_cache();
        }
    }

    #[test]
    fn invalid_and_late_palette_or_da1_replies_are_consumed_without_enabling_images() {
        let (mut probe, now) = fixture();
        for reply in ["\x1b[?1;1;256S", "\x1b[?1;0;0S", "\x1b[?1;0;1S",
            "\x1b[?1;0;99999999999999999S", "\x1b[?1;0;256;7S",
            "\x1b[?61;6;22c", "\x1b[?4c", "\x1b[?61;4;badc"] {
            let result = probe.filter(reply, now);
            assert!(!result.changed && !result.query_cells && result.input.is_empty());
        }
        let late = now + Duration::from_millis(300);
        assert!(!probe.filter("\x1b[?1;0;256S", late).changed);
        assert!(!probe.filter("\x1b[?61;4;22c", late).changed);
        assert_eq!(get_sixel_palette_size(), 16);
        assert_eq!(get_capabilities().images, None);
        assert_eq!(probe.filter("hello", late).input, "hello");
        reset_capabilities_cache();
    }

    #[test]
    fn windows_palette_default_requires_sixel_and_never_overrides_a_measured_limit() {
        let (mut probe, now) = fixture();
        probe.start(now, true);
        assert!(!probe.filter("\x1b[?61;6;22c", now).changed);
        assert_eq!(get_sixel_palette_size(), 16);
        assert_eq!(get_capabilities().images, None);
        assert!(probe.filter("\x1b[?1;0;4S", now).changed);
        assert!(probe.filter("\x1b[?61;4;6;22c", now).changed);
        assert_eq!(get_sixel_palette_size(), 4);
        reset_capabilities_cache();
    }

    #[test]
    fn cell_replies_are_bounded_and_never_reach_the_editor() {
        let (mut probe, now) = fixture();
        set_cell_dimensions(CellDimensions {
            width_px: 9,
            height_px: 18,
        });
        assert!(probe.filter("\x1b[6;", now).input.is_empty());
        assert!(probe.filter("20;10t", now).changed);
        assert_eq!(crate::terminal_image::get_cell_dimensions().width_px, 10);
        assert!(!probe.filter("\x1b[6;999999;999999t", now).changed);
        assert_eq!(crate::terminal_image::get_cell_dimensions().width_px, 10);
        assert!(probe
            .filter(&format!("\x1b[?2;{}", "9".repeat(512)), now)
            .input
            .is_empty());
        assert!(probe.partial.is_empty());
        assert!(probe.filter("123S", now).input.is_empty());
        assert_eq!(probe.filter("z", now).input, "z");
        assert_eq!(
            probe.filter("\x1b[200~paste\x1b[201~", now).input,
            "\x1b[200~paste\x1b[201~"
        );
        set_cell_dimensions(CellDimensions {
            width_px: 9,
            height_px: 18,
        });
        reset_capabilities_cache();
    }
}
