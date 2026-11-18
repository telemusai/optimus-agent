//! Port of packages/tui/src/render-cache.ts.

use std::{borrow::Borrow, cell::{Cell, OnceCell}, fmt, ops::Deref, rc::Rc};

/// Immutable transcript text shared across component, wrapping and viewport caches.
/// Cloning a line retains its allocation; graphics metadata is scanned only once.
#[derive(Clone, Debug, Default)]
pub struct RenderedLine(Rc<RenderedLineData>);

#[derive(Debug, Default)]
struct RenderedLineData {
    text: String,
    image: OnceCell<bool>,
    image_rows: OnceCell<Option<usize>>,
    terminal_plain: OnceCell<RenderedLine>,
    terminal_hyperlinks: OnceCell<RenderedLine>,
    terminal_normalizations: Cell<u8>,
    #[cfg(test)]
    text_comparisons: Cell<u32>,
}

impl RenderedLine {
    pub fn as_str(&self) -> &str {
        &self.0.text
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn is_image(&self) -> bool {
        *self
            .0
            .image
            .get_or_init(|| crate::terminal_image::is_image_line(self))
    }

    pub fn image_rows(&self) -> Option<usize> {
        *self
            .0
            .image_rows
            .get_or_init(|| crate::terminal_image::image_row_count(self))
    }

    /// Return shared, terminal-normalized text for the inline differ. The
    /// original line owns the normalized allocation, so an unchanged line is
    /// never formatted or copied again on editor-only repaints.
    pub fn terminal_normalized(&self, hyperlinks: bool, reset: &str) -> Self {
        if self.is_image() {
            return self.clone();
        }
        let slot = if hyperlinks {
            &self.0.terminal_hyperlinks
        } else {
            &self.0.terminal_plain
        };
        slot.get_or_init(|| {
            self.0
                .terminal_normalizations
                .set(self.0.terminal_normalizations.get().saturating_add(1));
            format!("{}{reset}", crate::utils::normalize_terminal_output(self.as_str())).into()
        })
        .clone()
    }

    #[cfg(test)]
    pub fn terminal_normalization_count(&self) -> u8 {
        self.0.terminal_normalizations.get()
    }

    #[cfg(test)]
    pub fn text_comparison_count(&self) -> u32 {
        self.0.text_comparisons.get()
    }
}

impl From<String> for RenderedLine {
    fn from(text: String) -> Self {
        Self(Rc::new(RenderedLineData {
            text,
            ..Default::default()
        }))
    }
}

impl From<&str> for RenderedLine {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl Deref for RenderedLine {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for RenderedLine {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for RenderedLine {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for RenderedLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}

impl PartialEq for RenderedLine {
    fn eq(&self, other: &Self) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        #[cfg(test)]
        self.0
            .text_comparisons
            .set(self.0.text_comparisons.get().saturating_add(1));
        self.as_str() == other.as_str()
    }
}

impl Eq for RenderedLine {}

/// Cache of rendered lines keyed by (width, revision, style epoch).
///
/// Native addition for transcript row memoization (A11): unlike
/// [`VersionedRenderCache`], the content revision and the global style epoch
/// (theme + keybindings revisions) are separate key parts, so no bit-packing
/// of two independent counters into one `u64` can collide.
#[derive(Debug, Default)]
pub struct StyledRenderCache {
    cached_width: Option<usize>,
    cached_revision: Option<u64>,
    cached_style: Option<u64>,
    cached_lines: Option<Vec<String>>,
}

impl StyledRenderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, width: usize, revision: u64, style: u64) -> Option<Vec<String>> {
        if self.cached_width == Some(width)
            && self.cached_revision == Some(revision)
            && self.cached_style == Some(style)
        {
            return self.cached_lines.clone();
        }
        None
    }

    pub fn set(
        &mut self,
        width: usize,
        revision: u64,
        style: u64,
        lines: Vec<String>,
    ) -> Vec<String> {
        self.cached_width = Some(width);
        self.cached_revision = Some(revision);
        self.cached_style = Some(style);
        self.cached_lines = Some(lines.clone());
        lines
    }

    pub fn invalidate(&mut self) {
        self.cached_width = None;
        self.cached_revision = None;
        self.cached_style = None;
        self.cached_lines = None;
    }
}

/// Cache of rendered lines keyed by (width, version).
#[derive(Debug, Default)]
pub struct VersionedRenderCache {
    cached_width: Option<usize>,
    cached_version: Option<u64>,
    cached_lines: Option<Vec<String>>,
}

impl VersionedRenderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, width: usize, version: u64) -> Option<Vec<String>> {
        if self.cached_width == Some(width) && self.cached_version == Some(version) {
            return self.cached_lines.clone();
        }
        None
    }

    pub fn set(&mut self, width: usize, version: u64, lines: Vec<String>) -> Vec<String> {
        self.cached_width = Some(width);
        self.cached_version = Some(version);
        self.cached_lines = Some(lines.clone());
        lines
    }

    pub fn invalidate(&mut self) {
        self.cached_width = None;
        self.cached_version = None;
        self.cached_lines = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_lines_retain_text_and_graphics_metadata() {
        let text = String::from("café 世界");
        let allocation = text.as_ptr();
        let line = RenderedLine::from(text);
        assert_eq!(line.as_str().as_ptr(), allocation);
        assert!(!line.is_image());
        let shared = line.clone();
        assert!(line.ptr_eq(&shared));
        assert_eq!(shared.0.image.get(), Some(&false));
        assert_eq!(line, RenderedLine::from("café 世界"));
        let image = RenderedLine::from(crate::terminal_image::position_image(
            "\x1bP0;1;0q~\x1b\\",
            3,
        ));
        assert!(image.is_image());
        assert_eq!(image.image_rows(), Some(3));
        assert_eq!(image.clone().0.image_rows.get(), Some(&Some(3)));
    }

    #[test]
    fn cache_hits_only_for_same_width_and_version() {
        let mut cache = VersionedRenderCache::new();
        assert_eq!(cache.get(80, 1), None);
        cache.set(80, 1, vec!["a".to_string()]);
        assert_eq!(cache.get(80, 1), Some(vec!["a".to_string()]));
        assert_eq!(cache.get(81, 1), None);
        assert_eq!(cache.get(80, 2), None);
        cache.invalidate();
        assert_eq!(cache.get(80, 1), None);
    }

    #[test]
    fn styled_cache_keeps_revision_and_style_epoch_exact() {
        let mut cache = StyledRenderCache::new();
        assert_eq!(cache.get(80, 1, 7), None);
        cache.set(80, 1, 7, vec!["a".to_string()]);
        assert_eq!(cache.get(80, 1, 7), Some(vec!["a".to_string()]));
        // Every key part is compared exactly: width, revision, style.
        assert_eq!(cache.get(81, 1, 7), None);
        assert_eq!(cache.get(80, 2, 7), None);
        assert_eq!(cache.get(80, 1, 8), None);
        // Style epochs live in the full u64 range; high bits must not be
        // truncated away by packing revision and style into one word.
        let high_style = u64::MAX - 3;
        cache.set(80, 1, high_style, vec!["b".to_string()]);
        assert_eq!(cache.get(80, 1, high_style), Some(vec!["b".to_string()]));
        assert_eq!(cache.get(80, 1, high_style ^ (1 << 63)), None);
        cache.invalidate();
        assert_eq!(cache.get(80, 1, high_style), None);
    }
}
