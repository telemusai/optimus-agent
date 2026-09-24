//! Port of packages/tui/src/terminal-image.ts.

use base64::Engine;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use rand::Rng;
use std::cell::RefCell;
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageProtocol {
    Kitty,
    Iterm2,
    Sixel,
}

/// `null` in the TypeScript protocol union.
pub type ImageProtocolOption = Option<ImageProtocol>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalCapabilities {
    pub images: ImageProtocolOption,
    pub true_color: bool,
    pub hyperlinks: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellDimensions {
    pub width_px: i64,
    pub height_px: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDimensions {
    pub width_px: i64,
    pub height_px: i64,
}

#[derive(Debug, Clone, Default)]
pub struct ImageRenderOptions {
    pub max_width_cells: Option<i64>,
    pub max_height_cells: Option<i64>,
    pub preserve_aspect_ratio: Option<bool>,
    /// Kitty image ID. If provided, reuses/replaces existing image with this ID.
    pub image_id: Option<u32>,
    /// Whether Kitty should apply its default cursor movement after placement.
    pub move_cursor: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedImage {
    pub sequence: String,
    pub rows: usize,
    pub image_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SixelLimits {
    pub width: u32,
    pub height: u32,
    pub colors: u16,
}

thread_local! {
    static CELL_DIMENSIONS_KNOWN: RefCell<bool> = const { RefCell::new(false) };
    static SIXEL_LIMITS: RefCell<SixelLimits> = const {
        RefCell::new(SixelLimits { width: 1024, height: 1024, colors: 16 })
    };
    static CACHED_CAPABILITIES: RefCell<Option<TerminalCapabilities>> = const { RefCell::new(None) };
    static CELL_DIMENSIONS: RefCell<CellDimensions> = const {
        RefCell::new(CellDimensions { width_px: 9, height_px: 18 })
    };
}

pub(crate) fn set_sixel_limits(limits: (u32, u32)) {
    SIXEL_LIMITS.with(|slot| {
        let mut current = slot.borrow_mut();
        current.width = limits.0;
        current.height = limits.1;
    });
}

pub(crate) fn get_sixel_limits() -> SixelLimits {
    SIXEL_LIMITS.with(|slot| *slot.borrow())
}

pub(crate) fn get_sixel_palette_size() -> u16 {
    get_sixel_limits().colors
}

pub(crate) fn set_sixel_palette_size(colors: u32) {
    if colors >= 2 {
        SIXEL_LIMITS.with(|slot| slot.borrow_mut().colors = colors.min(256) as u16);
    }
}

pub fn get_cell_dimensions() -> CellDimensions {
    CELL_DIMENSIONS.with(|c| *c.borrow())
}

pub fn cell_dimensions_known() -> bool {
    CELL_DIMENSIONS_KNOWN.with(|known| *known.borrow())
}

pub(crate) fn reset_cell_dimensions() {
    CELL_DIMENSIONS_KNOWN.with(|known| *known.borrow_mut() = false);
}

pub fn set_cell_dimensions(dims: CellDimensions) {
    if (1..=256).contains(&dims.width_px) && (1..=256).contains(&dims.height_px) {
        CELL_DIMENSIONS.with(|c| *c.borrow_mut() = dims);
        CELL_DIMENSIONS_KNOWN.with(|known| *known.borrow_mut() = true);
    }
}

pub const MAX_IMAGE_ROWS: usize = 24;
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_SOURCE_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_ENCODED_BYTES: usize = 2 * 1024 * 1024;
const MAX_RENDER_EDGE: i64 = 1024;

pub fn image_protocol_forced() -> bool {
    let value = env_lower("PI_FORCE_IMAGE_PROTOCOL");
    !value.is_empty() && value != "auto"
}

pub fn image_passthrough_blocked() -> bool {
    let term = env_lower("TERM");
    std::env::var_os("TMUX").is_some() || term.starts_with("tmux") || term.starts_with("screen")
        || term == "dumb"
}

fn env_lower(name: &str) -> String {
    std::env::var(name).unwrap_or_default().to_lowercase()
}

pub fn detect_capabilities() -> TerminalCapabilities {
    let mut caps = detect_environment_capabilities();
    if image_passthrough_blocked() {
        caps.images = None;
    } else if image_protocol_forced() {
        caps.images = match env_lower("PI_FORCE_IMAGE_PROTOCOL").as_str() {
            "kitty" => Some(ImageProtocol::Kitty),
            "iterm2" => Some(ImageProtocol::Iterm2),
            "sixel" => Some(ImageProtocol::Sixel),
            _ => None,
        };
    }
    caps
}

fn detect_environment_capabilities() -> TerminalCapabilities {
    let term_program = env_lower("TERM_PROGRAM");
    let term = env_lower("TERM");
    let color_term = env_lower("COLORTERM");

    // tmux and screen swallow OSC 8 by default (passthrough is opt-in and wraps
    // sequences differently). Force hyperlinks off whenever we detect them.
    let in_tmux_or_screen = std::env::var("TMUX").is_ok()
        || term.starts_with("tmux")
        || term.starts_with("screen");
    if in_tmux_or_screen {
        let true_color = color_term == "truecolor" || color_term == "24bit";
        return TerminalCapabilities {
            images: None,
            true_color,
            hyperlinks: false,
        };
    }

    if std::env::var("KITTY_WINDOW_ID").is_ok() || term_program == "kitty" {
        return TerminalCapabilities {
            images: Some(ImageProtocol::Kitty),
            true_color: true,
            hyperlinks: true,
        };
    }

    if term_program == "ghostty" || term.contains("ghostty") || std::env::var("GHOSTTY_RESOURCES_DIR").is_ok() {
        return TerminalCapabilities {
            images: Some(ImageProtocol::Kitty),
            true_color: true,
            hyperlinks: true,
        };
    }

    if std::env::var("WEZTERM_PANE").is_ok() || term_program == "wezterm" {
        return TerminalCapabilities {
            images: Some(ImageProtocol::Kitty),
            true_color: true,
            hyperlinks: true,
        };
    }

    if std::env::var("ITERM_SESSION_ID").is_ok() || term_program == "iterm.app" {
        return TerminalCapabilities {
            images: Some(ImageProtocol::Iterm2),
            true_color: true,
            hyperlinks: true,
        };
    }

    if term_program == "vscode" {
        return TerminalCapabilities {
            images: None,
            true_color: true,
            hyperlinks: true,
        };
    }

    if term_program == "alacritty" {
        return TerminalCapabilities {
            images: None,
            true_color: true,
            hyperlinks: true,
        };
    }

    let true_color = color_term == "truecolor" || color_term == "24bit";
    TerminalCapabilities {
        images: None,
        true_color,
        hyperlinks: false,
    }
}

pub fn get_capabilities() -> TerminalCapabilities {
    CACHED_CAPABILITIES.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() {
            *slot = Some(detect_capabilities());
        }
        slot.unwrap()
    })
}

pub fn reset_capabilities_cache() {
    CACHED_CAPABILITIES.with(|c| *c.borrow_mut() = None);
    set_sixel_limits((1024, 1024));
    set_sixel_palette_size(16);
}

/// Override the cached capabilities. Useful in tests to exercise both code paths.
pub fn set_capabilities(caps: TerminalCapabilities) {
    CACHED_CAPABILITIES.with(|c| *c.borrow_mut() = Some(caps));
}

const KITTY_PREFIX: &str = "\x1b_G";
const ITERM2_PREFIX: &str = "\x1b]1337;File=";

pub fn is_image_line(line: &str) -> bool {
    // Fast path: sequence at line start (single-row images)
    if line.starts_with(KITTY_PREFIX) || line.starts_with(ITERM2_PREFIX) {
        return true;
    }
    // Slow path: sequence elsewhere (multi-row images have cursor-up prefix)
    line.contains(KITTY_PREFIX) || line.contains(ITERM2_PREFIX) || line.contains("\x1bP")
}

/// Managed images save the cursor at their last reserved row before moving up.
/// Unknown/custom graphics have no trustworthy bounds and stay placeholders.
pub fn image_row_count(line: &str) -> Option<usize> {
    if !is_image_line(line) { return None; }
    let (_, saved) = line.split_once("\x1b[s")?;
    let (body, _) = saved.split_once("\x1b[u")?;
    if let Some(up) = body.strip_prefix("\x1b[") {
        let (count, _) = up.split_once('A')?;
        let rows = count.parse::<usize>().ok()?.checked_add(1)?;
        return (rows <= MAX_IMAGE_ROWS).then_some(rows);
    }
    Some(1)
}

pub fn position_image(sequence: &str, rows: usize) -> String {
    let up = if rows > 1 { format!("\x1b[{}A", rows - 1) } else { String::new() };
    format!("\x1b[s{up}{sequence}\x1b[u")
}

pub(crate) fn image_line_for_viewport(line: &str, available_rows: usize, width: usize) -> std::borrow::Cow<'_, str> {
    if image_row_count(line).is_some_and(|rows| rows > available_rows) {
        crate::utils::truncate_to_width("[Cannot display image in this viewport]", width as f64, "…", false).into()
    } else { line.into() }
}

/// Generate a random image ID for Kitty graphics protocol.
pub fn allocate_image_id() -> u32 {
    // Use random ID in range [1, 0xffffffff] to avoid collisions
    rand::thread_rng().gen_range(1..=0xffff_fffeu32)
}

#[derive(Debug, Clone, Default)]
pub struct KittyEncodeOptions {
    pub columns: Option<i64>,
    pub rows: Option<usize>,
    pub image_id: Option<u32>,
    /// Whether Kitty should apply its default cursor movement after placement. Default: true.
    pub move_cursor: Option<bool>,
}

pub fn encode_kitty(base64_data: &str, options: &KittyEncodeOptions) -> String {
    const CHUNK_SIZE: usize = 4096;

    let mut params: Vec<String> = vec!["a=T".to_string(), "f=100".to_string(), "q=2".to_string()];

    if options.move_cursor == Some(false) {
        params.push("C=1".to_string());
    }
    if let Some(columns) = options.columns {
        if columns != 0 {
            params.push(format!("c={columns}"));
        }
    }
    if let Some(rows) = options.rows {
        if rows != 0 {
            params.push(format!("r={rows}"));
        }
    }
    if let Some(image_id) = options.image_id {
        if image_id != 0 {
            params.push(format!("i={image_id}"));
        }
    }

    if base64_data.len() <= CHUNK_SIZE {
        return format!("\x1b_G{};{}\x1b\\", params.join(","), base64_data);
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut offset = 0usize;
    let mut is_first = true;

    while offset < base64_data.len() {
        let end = (offset + CHUNK_SIZE).min(base64_data.len());
        let chunk = &base64_data[offset..end];
        let is_last = offset + CHUNK_SIZE >= base64_data.len();

        if is_first {
            chunks.push(format!("\x1b_G{},m=1;{}\x1b\\", params.join(","), chunk));
            is_first = false;
        } else if is_last {
            chunks.push(format!("\x1b_Gm=0;{chunk}\x1b\\"));
        } else {
            chunks.push(format!("\x1b_Gm=1;{chunk}\x1b\\"));
        }

        offset += CHUNK_SIZE;
    }

    chunks.join("")
}

/// Delete a Kitty graphics image by ID.
pub fn delete_kitty_image(image_id: u32) -> String {
    format!("\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")
}

/// Delete all visible Kitty graphics images.
pub fn delete_all_kitty_images() -> String {
    "\x1b_Ga=d,d=A,q=2\x1b\\".to_string()
}

#[derive(Debug, Clone, Default)]
pub struct Iterm2EncodeOptions {
    pub width: Option<String>,
    pub height: Option<String>,
    pub name: Option<String>,
    pub preserve_aspect_ratio: Option<bool>,
    pub inline: Option<bool>,
}

pub fn encode_iterm2(base64_data: &str, options: &Iterm2EncodeOptions) -> String {
    let mut params: Vec<String> = vec![format!("inline={}", if options.inline != Some(false) { 1 } else { 0 })];

    if let Some(width) = &options.width {
        params.push(format!("width={width}"));
    }
    if let Some(height) = &options.height {
        params.push(format!("height={height}"));
    }
    if let Some(name) = &options.name {
        if !name.is_empty() {
            let name_base64 = base64::engine::general_purpose::STANDARD.encode(name);
            params.push(format!("name={name_base64}"));
        }
    }
    if options.preserve_aspect_ratio == Some(false) {
        params.push("preserveAspectRatio=0".to_string());
    }

    format!("\x1b]1337;File={}:{}\x07", params.join(";"), base64_data)
}

pub fn calculate_image_rows(image_dimensions: &ImageDimensions, target_width_cells: i64) -> usize {
    let cell_dimensions = get_cell_dimensions();
    calculate_image_rows_with_cells(image_dimensions, target_width_cells, &cell_dimensions)
}

pub fn calculate_image_rows_with_cells(
    image_dimensions: &ImageDimensions,
    target_width_cells: i64,
    cell_dimensions: &CellDimensions,
) -> usize {
    let target_width_px = (target_width_cells.saturating_mul(cell_dimensions.width_px)) as f64;
    let scale = if image_dimensions.width_px == 0 {
        0.0
    } else {
        target_width_px / image_dimensions.width_px as f64
    };
    let scaled_height_px = image_dimensions.height_px as f64 * scale;
    let rows = if cell_dimensions.height_px == 0 {
        0.0
    } else {
        (scaled_height_px / cell_dimensions.height_px as f64).ceil()
    };
    (rows as i64).max(1) as usize
}

fn decode_base64(base64_data: &str) -> Option<Vec<u8>> {
    if base64_data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 { return None; }
    base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .ok()
}

pub fn get_png_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_base64(base64_data)?;
    if buffer.len() < 24 {
        return None;
    }
    if buffer[0] != 0x89 || buffer[1] != 0x50 || buffer[2] != 0x4e || buffer[3] != 0x47 {
        return None;
    }
    let width = u32::from_be_bytes([buffer[16], buffer[17], buffer[18], buffer[19]]);
    let height = u32::from_be_bytes([buffer[20], buffer[21], buffer[22], buffer[23]]);
    Some(ImageDimensions {
        width_px: width as i64,
        height_px: height as i64,
    })
}

pub fn get_jpeg_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_base64(base64_data)?;
    if buffer.len() < 2 {
        return None;
    }
    if buffer[0] != 0xff || buffer[1] != 0xd8 {
        return None;
    }

    let mut offset = 2usize;
    while offset + 9 < buffer.len() {
        if buffer[offset] != 0xff {
            offset += 1;
            continue;
        }
        let marker = buffer[offset + 1];
        if (0xc0..=0xc2).contains(&marker) {
            let height = u16::from_be_bytes([buffer[offset + 5], buffer[offset + 6]]);
            let width = u16::from_be_bytes([buffer[offset + 7], buffer[offset + 8]]);
            return Some(ImageDimensions {
                width_px: width as i64,
                height_px: height as i64,
            });
        }
        if offset + 3 >= buffer.len() {
            return None;
        }
        let length = u16::from_be_bytes([buffer[offset + 2], buffer[offset + 3]]) as usize;
        if length < 2 {
            return None;
        }
        offset += 2 + length;
    }
    None
}

pub fn get_gif_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_base64(base64_data)?;
    if buffer.len() < 10 {
        return None;
    }
    let sig = String::from_utf8_lossy(&buffer[0..6]).to_string();
    if sig != "GIF87a" && sig != "GIF89a" {
        return None;
    }
    let width = u16::from_le_bytes([buffer[6], buffer[7]]);
    let height = u16::from_le_bytes([buffer[8], buffer[9]]);
    Some(ImageDimensions {
        width_px: width as i64,
        height_px: height as i64,
    })
}

pub fn get_webp_dimensions(base64_data: &str) -> Option<ImageDimensions> {
    let buffer = decode_base64(base64_data)?;
    if buffer.len() < 30 {
        return None;
    }
    let riff = String::from_utf8_lossy(&buffer[0..4]).to_string();
    let webp = String::from_utf8_lossy(&buffer[8..12]).to_string();
    if riff != "RIFF" || webp != "WEBP" {
        return None;
    }

    let chunk = String::from_utf8_lossy(&buffer[12..16]).to_string();
    if chunk == "VP8 " {
        if buffer.len() < 30 {
            return None;
        }
        let width = u16::from_le_bytes([buffer[26], buffer[27]]) & 0x3fff;
        let height = u16::from_le_bytes([buffer[28], buffer[29]]) & 0x3fff;
        return Some(ImageDimensions {
            width_px: width as i64,
            height_px: height as i64,
        });
    } else if chunk == "VP8L" {
        if buffer.len() < 25 {
            return None;
        }
        let bits = u32::from_le_bytes([buffer[21], buffer[22], buffer[23], buffer[24]]);
        let width = (bits & 0x3fff) + 1;
        let height = ((bits >> 14) & 0x3fff) + 1;
        return Some(ImageDimensions {
            width_px: width as i64,
            height_px: height as i64,
        });
    } else if chunk == "VP8X" {
        if buffer.len() < 30 {
            return None;
        }
        let width = (buffer[24] as i64 | (buffer[25] as i64) << 8 | (buffer[26] as i64) << 16) + 1;
        let height = (buffer[27] as i64 | (buffer[28] as i64) << 8 | (buffer[29] as i64) << 16) + 1;
        return Some(ImageDimensions {
            width_px: width,
            height_px: height,
        });
    }

    None
}

pub fn get_image_dimensions(base64_data: &str, mime_type: &str) -> Option<ImageDimensions> {
    match mime_type {
        "image/png" => get_png_dimensions(base64_data),
        "image/jpeg" => get_jpeg_dimensions(base64_data),
        "image/gif" => get_gif_dimensions(base64_data),
        "image/webp" => get_webp_dimensions(base64_data),
        _ => None,
    }
}

pub fn render_image(
    base64_data: &str,
    _image_dimensions: &ImageDimensions,
    options: &ImageRenderOptions,
) -> Option<RenderedImage> {
    let caps = get_capabilities();
    let protocol = caps.images?;
    // A guessed cell size can make SIXEL spill out of its reserved rows.
    if protocol == ImageProtocol::Sixel && !cell_dimensions_known() { return None; }

    let source = decode_bounded_image(base64_data)?;
    let cells = get_cell_dimensions();
    let max_width = options.max_width_cells.unwrap_or(60).clamp(0, 256);
    let max_height = options.max_height_cells.unwrap_or(MAX_IMAGE_ROWS as i64).clamp(0, MAX_IMAGE_ROWS as i64);
    if max_width == 0 || max_height == 0 { return None; }
    let (protocol_width, protocol_height) = if protocol == ImageProtocol::Sixel {
        let limits = get_sixel_limits();
        (limits.width, limits.height)
    } else { (1024, 1024) };
    let width_limit = (max_width * cells.width_px).min(MAX_RENDER_EDGE).min(i64::from(protocol_width)) as f64;
    let height_limit = (max_height * cells.height_px).min(MAX_RENDER_EDGE).min(i64::from(protocol_height)) as f64;
    let scale = (width_limit / f64::from(source.width()))
        .min(height_limit / f64::from(source.height())).min(1.0);
    let mut width_px = (f64::from(source.width()) * scale).floor().max(1.0) as u32;
    let mut height_px = (f64::from(source.height()) * scale).floor().max(1.0) as u32;
    if protocol == ImageProtocol::Sixel {
        // SIXEL allocates six-pixel bands, including transparent padding.
        // Keep that padding inside the row budget instead of overwriting text.
        if height_limit < 6.0 { return None; }
        if height_px >= 6 {
            let aligned = height_px / 6 * 6;
            width_px = (u64::from(width_px) * u64::from(aligned) / u64::from(height_px)).max(1) as u32;
            height_px = aligned;
        }
    }
    let raster = source.resize_exact(width_px, height_px, image::imageops::FilterType::Lanczos3);
    let allocated_height = if protocol == ImageProtocol::Sixel { height_px.div_ceil(6) * 6 } else { height_px };
    let rows = (i64::from(allocated_height) + cells.height_px - 1) / cells.height_px;
    if rows > max_height { return None; }
    let rows = rows as usize;
    let columns = (i64::from(width_px) + cells.width_px - 1) / cells.width_px;
    let png = if protocol != ImageProtocol::Sixel {
        let mut output = Cursor::new(Vec::new());
        raster.write_to(&mut output, ImageFormat::Png).ok()?;
        Some(base64::engine::general_purpose::STANDARD.encode(output.into_inner()))
    } else { None };

    let result = match protocol {
        ImageProtocol::Kitty => {
            let sequence = encode_kitty(
                png.as_deref()?,
                &KittyEncodeOptions {
                    columns: Some(columns),
                    rows: Some(rows),
                    image_id: options.image_id,
                    move_cursor: options.move_cursor,
                },
            );
            Some(RenderedImage {
                sequence,
                rows,
                image_id: options.image_id,
            })
        }
        ImageProtocol::Iterm2 => {
            let sequence = encode_iterm2(
                png.as_deref()?,
                &Iterm2EncodeOptions {
                    width: Some(columns.to_string()),
                    height: Some("auto".to_string()),
                    name: None,
                    preserve_aspect_ratio: Some(options.preserve_aspect_ratio.unwrap_or(true)),
                    inline: None,
                },
            );
            Some(RenderedImage {
                sequence,
                rows,
                image_id: None,
            })
        }
        ImageProtocol::Sixel => {
            let rgba = raster.to_rgba8();
            let sequence = icy_sixel::sixel_encode(rgba.as_raw(), width_px as usize, height_px as usize,
                &icy_sixel::EncodeOptions {
                    max_colors: get_sixel_palette_size(),
                    ..Default::default()
                }).ok()?;
            Some(RenderedImage { sequence, rows, image_id: None })
        }
    }?;
    (result.sequence.len() <= MAX_ENCODED_BYTES).then_some(result)
}

fn decode_bounded_image(base64_data: &str) -> Option<DynamicImage> {
    let bytes = decode_base64(base64_data)?;
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().ok()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let decoder = reader.into_decoder().ok()?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_SOURCE_PIXELS {
        return None;
    }
    DynamicImage::from_decoder(decoder).ok()
}

/// Wrap text in an OSC 8 hyperlink sequence.
pub fn hyperlink(text: &str, url: &str) -> String {
    format!("\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\")
}

pub fn image_fallback(
    mime_type: &str,
    dimensions: Option<&ImageDimensions>,
    filename: Option<&str>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(filename) = filename {
        parts.push(crate::terminal::sanitize_title_text(filename));
    }
    parts.push(format!("[{}]", crate::terminal::sanitize_title_text(mime_type)));
    if let Some(dimensions) = dimensions {
        parts.push(format!("{}x{}", dimensions.width_px, dimensions.height_px));
    }
    format!("[Cannot display image: {}]", parts.join(" "))
}

#[cfg(test)]
pub(crate) fn test_png(width: u32, height: u32) -> String {
    let image = image::RgbaImage::from_pixel(width, height, image::Rgba([0, 244, 119, 255]));
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, ImageFormat::Png).unwrap();
    base64::engine::general_purpose::STANDARD.encode(png.into_inner())
}

#[cfg(test)]
pub(crate) fn test_palette_png() -> String {
    let image = image::RgbaImage::from_fn(96, 12, |x, _| {
        image::Rgba(match x {
            0..=15 => [0, 0, 0, 255],
            16..=31 => [255, 255, 255, 255],
            32..=47 => [255, 220, 0, 255],
            _ => [(x * 2) as u8, (x * 3 % 256) as u8, (x * 5 % 256) as u8, 255],
        })
    });
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, ImageFormat::Png).unwrap();
    base64::engine::general_purpose::STANDARD.encode(png.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn sixel_palette_stays_within_terminal_registers_and_preserves_primary_colors() {
        reset_capabilities_cache();
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        let registers = regex::Regex::new(r"#(\d+)").unwrap();
        for colors in [2, 4, 16, 256] {
            set_sixel_palette_size(colors);
            let rendered = render_image(&test_palette_png(), &ImageDimensions { width_px: 96, height_px: 12 },
                &ImageRenderOptions::default()).unwrap();
            let highest = registers.captures_iter(&rendered.sequence)
                .map(|capture| capture[1].parse::<u32>().unwrap()).max().unwrap();
            assert!(highest < colors, "{highest} exceeds {colors} color registers");
            if colors == 256 { assert!(highest >= 16); }
            if colors >= 16 {
                let decoded = icy_sixel::SixelImage::decode(rendered.sequence.as_bytes()).unwrap();
                for (x, expected) in [(8, [0u8, 0, 0]), (24, [255, 255, 255]), (40, [255, 220, 0])] {
                    let offset = (6 * decoded.width + x) * 4;
                    for channel in 0..3 {
                        assert!(decoded.pixels[offset + channel].abs_diff(expected[channel]) <= 6);
                    }
                }
            }
        }
        reset_capabilities_cache();
        assert_eq!(get_sixel_palette_size(), 16);
    }

    #[test]
    fn sixel_round_trip_respects_height_band_and_cursor_bounds() {
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 17 });
        let result = render_image(&test_png(200, 400), &ImageDimensions { width_px: 1, height_px: 1 },
            &ImageRenderOptions { max_width_cells: Some(60), max_height_cells: Some(5), ..Default::default() }).unwrap();
        let decoded = icy_sixel::SixelImage::decode(result.sequence.as_bytes()).unwrap();
        assert!(decoded.width <= 540 && decoded.height <= 85);
        assert_eq!(decoded.height % 6, 0);
        assert!((decoded.width as f64 / decoded.height as f64 - 0.5).abs() < 0.02);
        assert!(result.rows <= 5);
        assert_eq!(image_row_count(&position_image(&result.sequence, result.rows)), Some(result.rows));
        assert!(is_image_line(&result.sequence));
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        reset_capabilities_cache();
    }

    #[test]
    fn corrupt_oversize_and_zero_room_images_fail_without_graphics() {
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        let dims = ImageDimensions { width_px: i64::MAX, height_px: i64::MAX };
        for data in ["not-base64".to_string(), base64::engine::general_purpose::STANDARD.encode(b"not an image"),
            "A".repeat(MAX_IMAGE_BYTES.div_ceil(3) * 4 + 1)] {
            assert!(render_image(&data, &dims, &ImageRenderOptions::default()).is_none());
        }
        let png = test_png(12, 12);
        for options in [ImageRenderOptions { max_width_cells: Some(0), ..Default::default() },
            ImageRenderOptions { max_height_cells: Some(0), ..Default::default() }] {
            assert!(render_image(&png, &dims, &options).is_none());
        }
        // Reject the declared dimensions before allocating a decompressed raster.
        let mut bytes = base64::engine::general_purpose::STANDARD.decode(&png).unwrap();
        bytes[16..20].copy_from_slice(&100_000u32.to_be_bytes());
        assert!(decode_bounded_image(&base64::engine::general_purpose::STANDARD.encode(bytes)).is_none());
        reset_capabilities_cache();
    }

    #[test]
    fn sixel_waits_for_cell_geometry_and_honors_the_terminal_raster_limit() {
        set_capabilities(TerminalCapabilities { images: Some(ImageProtocol::Sixel), true_color: true, hyperlinks: true });
        reset_cell_dimensions();
        let png = test_png(90, 90);
        let dims = ImageDimensions { width_px: 90, height_px: 90 };
        assert!(render_image(&png, &dims, &ImageRenderOptions::default()).is_none());
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        set_sixel_limits((30, 30));
        let rendered = render_image(&png, &dims, &ImageRenderOptions::default()).unwrap();
        let decoded = icy_sixel::SixelImage::decode(rendered.sequence.as_bytes()).unwrap();
        assert!(decoded.width <= 30 && decoded.height <= 30);
        reset_capabilities_cache();
    }

    #[test]
    fn inline_repaint_does_not_draw_above_the_visible_viewport() {
        let line = position_image("\x1bP0;1;0q~\x1b\\", 3);
        assert_eq!(image_line_for_viewport(&line, 3, 80), line);
        assert!(image_line_for_viewport(&line, 2, 80).contains("Cannot display image"));
        assert!(!is_image_line(&image_line_for_viewport(&line, 0, 80)));
        assert!(crate::utils::visible_width(&image_line_for_viewport(&line, 0, 4)) <= 4);
    }

    #[test]
    fn all_protocols_still_render_valid_images_and_respect_height_limits() {
        set_cell_dimensions(CellDimensions { width_px: 9, height_px: 18 });
        let dims = ImageDimensions { width_px: 24, height_px: 48 };
        for protocol in [ImageProtocol::Kitty, ImageProtocol::Iterm2, ImageProtocol::Sixel] {
            set_capabilities(TerminalCapabilities { images: Some(protocol), true_color: true, hyperlinks: true });
            let image = render_image(&test_png(24, 48), &dims, &ImageRenderOptions {
                max_height_cells: Some(2), ..Default::default()
            }).unwrap();
            assert!(image.rows <= 2 && is_image_line(&image.sequence));
        }
        reset_capabilities_cache();
    }

    #[test]
    fn detects_png_dimensions() {
        let dims = get_png_dimensions(PNG_1X1).unwrap();
        assert_eq!(dims.width_px, 1);
        assert_eq!(dims.height_px, 1);
        assert!(get_png_dimensions("not-base64!!").is_none());
    }

    #[test]
    fn kitty_encoding_chunks_and_params() {
        let single = encode_kitty(
            "AAAA",
            &KittyEncodeOptions {
                columns: Some(4),
                rows: Some(2),
                image_id: Some(7),
                move_cursor: None,
            },
        );
        assert_eq!(single, "\x1b_Ga=T,f=100,q=2,c=4,r=2,i=7;AAAA\x1b\\");

        let big = "A".repeat(5000);
        let chunked = encode_kitty(&big, &KittyEncodeOptions::default());
        assert!(chunked.contains(",m=1;"));
        assert!(chunked.contains("\x1b_Gm=0;"));
    }

    #[test]
    fn iterm2_encoding_includes_name() {
        let encoded = encode_iterm2("AAAA", &Iterm2EncodeOptions {
            width: Some("10".to_string()),
            height: Some("auto".to_string()),
            name: Some("x".to_string()),
            preserve_aspect_ratio: Some(false),
            inline: None,
        });
        assert_eq!(
            encoded,
            "\x1b]1337;File=inline=1;width=10;height=auto;name=eA==;preserveAspectRatio=0:AAAA\x07"
        );
    }

    #[test]
    fn image_rows_use_cell_dimensions() {
        let dims = ImageDimensions {
            width_px: 100,
            height_px: 50,
        };
        assert_eq!(
            calculate_image_rows_with_cells(&dims, 10, &CellDimensions { width_px: 10, height_px: 10 }),
            5
        );
    }

    #[test]
    fn image_line_detection_and_fallback() {
        assert!(is_image_line("\x1b_Ga=T;AAAA\x1b\\"));
        assert!(is_image_line("\x1b]1337;File=inline=1:AAAA\x07"));
        assert!(!is_image_line("plain"));
        assert_eq!(
            image_fallback("image/png", Some(&ImageDimensions { width_px: 2, height_px: 3 }), Some("a.png")),
            "[Cannot display image: a.png [image/png] 2x3]"
        );
    }

    #[test]
    fn hyperlink_wraps_text() {
        assert_eq!(
            hyperlink("x", "https://e.test"),
            "\x1b]8;;https://e.test\x1b\\x\x1b]8;;\x1b\\"
        );
    }
}
