//! Port of packages/tui/src/components/image.ts

use std::cell::Cell;

use crate::terminal_image::{
    allocate_image_id, cell_dimensions_known, get_capabilities, get_cell_dimensions, get_image_dimensions, image_fallback,
    position_image, render_image, CellDimensions, ImageDimensions, ImageProtocol, ImageRenderOptions,
    get_sixel_limits, SixelLimits, MAX_IMAGE_ROWS,
};
use crate::tui::Component;

thread_local! {
    /// Port of the module-level `fullscreenFallback` flag.
    static FULLSCREEN_FALLBACK: Cell<bool> = const { Cell::new(false) };
    static HEIGHT_LIMIT: Cell<usize> = const { Cell::new(MAX_IMAGE_ROWS) };
}

pub fn with_image_height_limit<T>(height: usize, render: impl FnOnce() -> T) -> T {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) { HEIGHT_LIMIT.with(|flag| flag.set(self.0)); }
    }
    let _restore = Restore(HEIGHT_LIMIT.with(|flag| flag.replace(height.min(MAX_IMAGE_ROWS))));
    render()
}

pub fn with_fullscreen_image_fallback<T>(render: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) { FULLSCREEN_FALLBACK.with(|flag| flag.set(self.0)); }
    }
    let _restore = Restore(FULLSCREEN_FALLBACK.with(|flag| flag.replace(true)));
    render()
}

fn fullscreen_fallback() -> bool {
    FULLSCREEN_FALLBACK.with(|flag| flag.get())
}

/// Port of `ImageTheme`.
pub struct ImageTheme {
    pub fallback_color: Box<dyn Fn(&str) -> String>,
}

/// Port of `ImageOptions`.
#[derive(Clone, Default)]
pub struct ImageOptions {
    pub max_width_cells: Option<usize>,
    pub max_height_cells: Option<usize>,
    pub filename: Option<String>,
    /// Renders textual image metadata instead of terminal graphics.
    pub fallback_only: bool,
    /// Prefix prepended to textual fallback metadata.
    pub fallback_prefix: Option<String>,
    /// Kitty image ID to reuse across updates or animations.
    pub image_id: Option<u32>,
}

pub struct Image {
    base64_data: String,
    mime_type: String,
    dimensions: ImageDimensions,
    theme: ImageTheme,
    options: ImageOptions,
    image_id: Option<u32>,

    cached_lines: Option<Vec<String>>,
    cached_width: Option<usize>,
    cached_fullscreen_fallback: Option<bool>,
    cached_geometry: Option<(Option<ImageProtocol>, CellDimensions, bool, usize, SixelLimits)>,
}

impl Image {
    pub fn new(
        base64_data: String,
        mime_type: String,
        theme: ImageTheme,
        options: ImageOptions,
        dimensions: Option<ImageDimensions>,
    ) -> Self {
        let dimensions = dimensions
            .or_else(|| get_image_dimensions(&base64_data, &mime_type))
            .unwrap_or(ImageDimensions {
                width_px: 800,
                height_px: 600,
            });
        let image_id = options.image_id;
        Self {
            base64_data,
            mime_type,
            dimensions,
            theme,
            options,
            image_id,
            cached_lines: None,
            cached_width: None,
            cached_fullscreen_fallback: None,
            cached_geometry: None,
        }
    }

    /// Returns the Kitty image ID allocated or supplied for this image.
    pub fn get_image_id(&self) -> Option<u32> {
        self.image_id
    }

    pub fn dimensions(&self) -> &ImageDimensions {
        &self.dimensions
    }
}

impl Component for Image {
    fn render(&mut self, width: f64) -> Vec<String> {
        let width = width.max(0.0).floor() as usize;
        let fallback_flag = fullscreen_fallback();
        let caps = get_capabilities();
        let height = HEIGHT_LIMIT.with(|limit| limit.get())
            .min(self.options.max_height_cells.unwrap_or(MAX_IMAGE_ROWS));
        let geometry = (caps.images, get_cell_dimensions(), cell_dimensions_known(), height, get_sixel_limits());
        if let (Some(lines), Some(cached_width), Some(cached_fallback)) = (
            &self.cached_lines,
            self.cached_width,
            self.cached_fullscreen_fallback,
        ) {
            if cached_width == width && cached_fallback == fallback_flag && self.cached_geometry == Some(geometry) {
                return lines.clone();
            }
        }

        // `Math.min(width - 2, maxWidthCells ?? 60)`; the subtraction stays in i64
        // because TypeScript allows a negative result for width < 2.
        let max_width = (width as i64 - 2).min(self.options.max_width_cells.unwrap_or(60) as i64);

        let lines: Vec<String>;

        if self.options.fallback_only {
            let mut parts: Vec<String> = vec![self.mime_type.clone()];
            parts.push(format!("{}×{}", self.dimensions.width_px, self.dimensions.height_px));
            if let Some(filename) = &self.options.filename {
                parts.insert(0, filename.clone());
            }
            lines = vec![(self.theme.fallback_color)(&format!(
                "{}[{}]",
                self.options.fallback_prefix.clone().unwrap_or_default(),
                parts.join(" · ")
            ))];
        } else if !fallback_flag && caps.images.is_some() {
            if caps.images == Some(ImageProtocol::Kitty) && self.image_id.is_none() {
                self.image_id = Some(allocate_image_id());
            }
            let result = render_image(
                &self.base64_data,
                &self.dimensions,
                &ImageRenderOptions {
                    max_width_cells: Some(max_width.max(0)),
                    max_height_cells: Some(height as i64),
                    preserve_aspect_ratio: Some(true),
                    image_id: self.image_id,
                    move_cursor: Some(false),
                },
            );

            match result {
                Some(result) => {
                    if let Some(image_id) = result.image_id {
                        self.image_id = Some(image_id);
                    }

                    let mut rendered = vec![String::new(); result.rows.saturating_sub(1)];
                    rendered.push(position_image(&result.sequence, result.rows));
                    lines = rendered;
                }
                None => {
                    let fallback = image_fallback(
                        &self.mime_type,
                        Some(&self.dimensions),
                        self.options.filename.as_deref(),
                    );
                    lines = vec![(self.theme.fallback_color)(&fallback)];
                }
            }
        } else {
            let fallback = image_fallback(
                &self.mime_type,
                Some(&self.dimensions),
                self.options.filename.as_deref(),
            );
            lines = vec![(self.theme.fallback_color)(&fallback)];
        }

        let lines: Vec<_> = lines.into_iter().map(|line| {
            if crate::terminal_image::is_image_line(&line) { line } else {
                crate::utils::truncate_to_width(&line, width as f64, "…", false)
            }
        }).collect();
        self.cached_lines = Some(lines.clone());
        self.cached_width = Some(width);
        self.cached_fullscreen_fallback = Some(fallback_flag);
        self.cached_geometry = Some(geometry);

        lines
    }

    fn invalidate(&mut self) {
        self.cached_lines = None;
        self.cached_width = None;
        self.cached_fullscreen_fallback = None;
        self.cached_geometry = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_reply_reencodes_an_already_cached_image() {
        use crate::terminal_image::*;
        reset_capabilities_cache();
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        let mut image = Image::new(test_palette_png(), "image/png".into(),
            ImageTheme { fallback_color: Box::new(str::to_string) }, ImageOptions::default(), None);
        let low_color = image.render(80.0);
        set_sixel_palette_size(256);
        let full_color = image.render(80.0);
        assert_ne!(low_color, full_color);
        set_sixel_palette_size(16);
        assert_eq!(image.render(80.0), low_color);
        // A DA1 reply may enable images before the raster-limit reply arrives.
        set_sixel_limits((24, 12));
        let bounded = image.render(80.0);
        assert_ne!(bounded, low_color);
        let line = bounded.last().unwrap();
        let start = line.find("\x1bP").unwrap();
        let decoded = icy_sixel::SixelImage::decode(line[start..].as_bytes()).unwrap();
        assert!(decoded.width <= 24 && decoded.height <= 12);
        reset_capabilities_cache();
    }

    #[test]
    fn image_cache_tracks_protocol_cells_and_viewport_height() {
        use crate::terminal_image::*;
        let caps = |images| TerminalCapabilities { images, true_color: true, hyperlinks: true };
        set_capabilities(caps(None));
        let mut image = Image::new(test_png(72, 144), "image/png".into(),
            ImageTheme { fallback_color: Box::new(str::to_string) }, ImageOptions::default(), None);
        assert!(image.render(40.0)[0].contains("Cannot display image"));
        set_capabilities(caps(Some(ImageProtocol::Sixel)));
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        let lines = with_image_height_limit(3, || image.render(40.0));
        assert!(lines.len() <= 3 && is_image_line(lines.last().unwrap()));
        let fewer = with_image_height_limit(2, || image.render(40.0));
        assert!(fewer.len() <= 2);
        assert_ne!(fewer, lines);
        set_cell_dimensions(CellDimensions { width_px: 8, height_px: 12 });
        assert_ne!(with_image_height_limit(2, || image.render(40.0)), fewer);
        assert!(with_fullscreen_image_fallback(|| image.render(40.0))[0].contains("Cannot display image"));
        assert!(is_image_line(image.render(40.0).last().unwrap()));
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        reset_capabilities_cache();
    }

    #[test]
    fn invalid_image_fallback_is_safe_in_a_narrow_terminal() {
        use crate::terminal_image::*;
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        let mut image = Image::new("bad".into(), "image/png".into(),
            ImageTheme { fallback_color: Box::new(str::to_string) },
            ImageOptions { filename: Some("bad\x1b[2J\nname".into()), ..Default::default() }, None);
        for width in [0, 1, 8, 40] {
            let rendered = image.render(width as f64);
            assert!(!is_image_line(&rendered[0]));
            assert!(crate::utils::visible_width(&rendered[0]) <= width);
            assert!(!rendered[0].contains('\x1b') && !rendered[0].contains('\n'));
        }
        reset_capabilities_cache();
    }

    #[test]
    fn fullscreen_fallback_flag_is_restored() {
        assert!(!fullscreen_fallback());
        let inner = with_fullscreen_image_fallback(|| fullscreen_fallback());
        assert!(inner);
        assert!(!fullscreen_fallback());
    }

    #[test]
    fn fallback_line_uses_metadata_parts() {
        let image = Image::new(
            String::new(),
            "image/png".to_string(),
            ImageTheme {
                fallback_color: Box::new(|text: &str| text.to_string()),
            },
            ImageOptions {
                filename: Some("cat.png".to_string()),
                fallback_only: true,
                fallback_prefix: Some("> ".to_string()),
                ..ImageOptions::default()
            },
            Some(ImageDimensions {
                width_px: 10,
                height_px: 20,
            }),
        );
        let mut image = image;
        let lines = image.render(40.0);
        assert_eq!(lines, vec!["> [cat.png · image/png · 10×20]".to_string()]);
    }
}
