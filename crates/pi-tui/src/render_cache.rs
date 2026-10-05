//! Port of packages/tui/src/render-cache.ts.

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

    pub fn set(&mut self, width: usize, revision: u64, style: u64, lines: Vec<String>) -> Vec<String> {
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
