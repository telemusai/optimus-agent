//! Port of packages/tui/src/index.ts - the crate's public surface.
//!
//! The TypeScript module re-exports symbols from the individual files; the port
//! re-exports the same items from the corresponding modules.

pub use crate::autocomplete::{
    AutocompleteItem, AutocompleteProvider, AutocompleteSuggestions, CombinedAutocompleteProvider, SlashCommand,
};
pub use crate::components::r#box::Box_;
pub use crate::components::cancellable_loader::CancellableLoader;
pub use crate::components::editor::{Editor, EditorOptions, EditorTheme};
pub use crate::components::image::{Image, ImageOptions, ImageTheme};
pub use crate::components::input::Input;
pub use crate::components::loader::{Loader, LoaderIndicatorOptions};
pub use crate::components::markdown::{DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme};
pub use crate::components::select_list::{
    SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme, SelectListTruncatePrimaryContext,
};
pub use crate::components::settings_list::{SettingItem, SettingsList, SettingsListTheme};
pub use crate::components::spacer::Spacer;
pub use crate::components::text::Text;
pub use crate::components::truncated_text::TruncatedText;
pub use crate::editor_component::{EditorComponent, EditorPasteSnapshot};
pub use crate::fullscreen::{clipped_fullscreen_dock_height, FullscreenViewport, ScrollInfo, FULLSCREEN_MIN_TRANSCRIPT_ROWS};
pub use crate::fuzzy::{fuzzy_filter, fuzzy_filter_scored, fuzzy_match, FuzzyMatch, ScoredItem};
pub use crate::keybindings::{
    get_keybindings, keybindings_revision, set_keybindings, Keybinding, KeybindingConflict, KeybindingDefinition,
    KeybindingDefinitions, Keybindings, KeybindingsConfig, KeybindingsManager, TUI_KEYBINDINGS,
};
pub use crate::keys::{
    decode_kitty_printable, is_key_release, is_key_repeat, is_kitty_protocol_active, key, matches_key, parse_key,
    set_kitty_protocol_active, KeyEventType, KeyId,
};
pub use crate::latex::latex_to_unicode;
pub use crate::mouse::{
    is_mouse_sequence, is_wheel_down, is_wheel_up, parse_sgr_mouse_event, MouseEvent, MOUSE_WHEEL_DOWN,
    MOUSE_WHEEL_UP,
};
pub use crate::render_cache::VersionedRenderCache;
pub use crate::selection_metadata::TableCellSelectionRegion;
pub use crate::stdin_buffer::{StdinBuffer, StdinBufferEvent, StdinBufferEventMap, StdinBufferOptions};
pub use crate::terminal::{ProcessTerminal, Terminal, TerminalStopOptions};
pub use crate::terminal_colors::{
    best_ansi_color, blend_color, clear_default_terminal_colors, detect_background_from_color_fg_bg,
    get_default_terminal_colors, get_terminal_background_kind, is_light_color, on_default_terminal_colors_change,
    parse_osc_color_response, rgb_to_256, rgb_to_hex, set_default_terminal_colors, DefaultTerminalColors,
    OscColorKind, OscColorResponse, Rgb, TerminalBackgroundKind, TerminalColorMode, QUERY_DEFAULT_BACKGROUND,
    QUERY_DEFAULT_FOREGROUND,
};
pub use crate::terminal_image::{
    allocate_image_id, calculate_image_rows, delete_all_kitty_images, delete_kitty_image, detect_capabilities,
    encode_iterm2, encode_kitty, get_capabilities, get_cell_dimensions, get_gif_dimensions, get_image_dimensions,
    get_jpeg_dimensions, get_png_dimensions, get_webp_dimensions, hyperlink, image_fallback, render_image,
    reset_capabilities_cache, set_capabilities, set_cell_dimensions, CellDimensions, ImageDimensions, ImageProtocol,
    ImageRenderOptions, TerminalCapabilities,
};
pub use crate::tui::{
    is_focusable, Component, Container, Focusable, FullscreenOptions, OverlayAnchor, OverlayHandle, OverlayMargin,
    OverlayOptions, RenderPhaseTimings, SizeValue, TuiStopOptions, CURSOR_MARKER, TUI,
};
pub use crate::utils::{truncate_to_width, visible_width, wrap_text_with_ansi};

#[cfg(test)]
mod tests {
    #[test]
    fn re_exports_resolve() {
        // The TypeScript module is a pure re-export surface; this asserts the
        // names exist at the same path after the port.
        let _ = super::latex_to_unicode(r"\alpha");
        let _ = super::visible_width("ab");
        assert_eq!(super::FULLSCREEN_MIN_TRANSCRIPT_ROWS, 3);
        assert_eq!(super::CURSOR_MARKER, "\x1b_pi:c\x07");
    }
}
