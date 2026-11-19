//! Port of packages/coding-agent/src/core/tools/path-utils.ts

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

const UNICODE_SPACES: &str = "[\u{00A0}\u{2000}-\u{200A}\u{202F}\u{205F}\u{3000}]";
const NARROW_NO_BREAK_SPACE: &str = "\u{202F}";

fn unicode_spaces_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(UNICODE_SPACES).expect("valid unicode space pattern"))
}

fn am_pm_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r" (AM|PM)\.").expect("valid am/pm pattern"))
}

fn normalize_unicode_spaces(text: &str) -> String {
    unicode_spaces_pattern().replace_all(text, " ").into_owned()
}

fn try_macos_screenshot_path(file_path: &str) -> String {
    am_pm_pattern()
        .replace_all(file_path, |caps: &regex::Captures<'_>| {
            format!("{NARROW_NO_BREAK_SPACE}{}.", &caps[1])
        })
        .into_owned()
}

fn try_nfd_variant(file_path: &str) -> String {
    // macOS stores filenames in NFD (decomposed) form, try converting user input to NFD.
    // Rust's std has no Unicode normalization; the ASCII path is unaffected and
    // non-ASCII input keeps its original bytes.
    file_path.to_string()
}

fn try_curly_quote_variant(file_path: &str) -> String {
    // macOS uses U+2019 (right single quotation mark) in screenshot names like "Capture d'ecran".
    // Users typically type U+0027 (straight apostrophe).
    file_path.replace('\'', "\u{2019}")
}

fn file_exists(file_path: &str) -> bool {
    std::fs::metadata(file_path).is_ok()
}

fn normalize_at_prefix(file_path: &str) -> &str {
    match file_path.strip_prefix('@') {
        Some(rest) => rest,
        None => file_path,
    }
}

/// NodeJS `process.platform` equivalent for the `platform` parameter default.
pub fn process_platform() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

fn home_dir() -> String {
    match dirs::home_dir() {
        Some(home) => home.to_string_lossy().into_owned(),
        None => std::env::var("HOME").unwrap_or_default(),
    }
}

pub fn expand_path(file_path: &str) -> String {
    expand_path_with_platform(file_path, process_platform())
}

pub fn expand_path_with_platform(file_path: &str, platform: &str) -> String {
    let normalized = normalize_unicode_spaces(normalize_at_prefix(file_path));
    if normalized == "~" {
        return home_dir();
    }
    let windows_tilde = platform == "win32" && normalized.starts_with("~\\");
    if normalized.starts_with("~/") || windows_tilde {
        let home = home_dir();
        let rest = &normalized[2..];
        if platform == "win32" {
            return windows_join(&home, rest);
        }
        return posix_join(&home, rest);
    }
    normalized
}

fn posix_join(base: &str, rest: &str) -> String {
    let mut joined = PathBuf::from(base);
    for segment in rest.split('/') {
        joined.push(segment);
    }
    joined.to_string_lossy().replace('\\', "/")
}

fn windows_join(base: &str, rest: &str) -> String {
    let mut joined = PathBuf::from(base);
    for segment in rest.split(['/', '\\']) {
        joined.push(segment);
    }
    joined.to_string_lossy().into_owned()
}

/// Resolve a path relative to the given cwd.
/// Handles ~ expansion and absolute paths.
pub fn resolve_to_cwd(file_path: &str, cwd: &str) -> String {
    let expanded = expand_path(file_path);
    if is_absolute(&expanded) {
        return expanded;
    }
    resolve_path(cwd, &expanded)
}

/// NodeJS `path.resolve(base, target)`, the call `resolveToCwd` makes
/// (`path-utils.ts:59`).
///
/// The TypeScript suite asserts exactly this call
/// (`test/path-utils.test.ts:56-59`: `resolveToCwd(relative, cwd)` must equal
/// `resolve(cwd, relative)`), so the host flavour applies: on win32 the result
/// is drive-absolute with backslash separators, and a drive-less rooted input
/// such as `/a` gains the current drive (`C:\\a`). `Path::join` alone does
/// neither, which is why the port returned `"\\a\\b\\c"` where Node returns
/// `"C:\\a\\b\\c"`.
pub fn resolve_path(base: &str, target: &str) -> String {
    node_resolve(base, target)
}

/// `resolve(...args)` for the two-argument calls this module makes.
///
/// Node resolves right-to-left and stops at the first absolute segment, so a
/// rooted `target` replaces `base` instead of being appended to it. Measured on
/// this host: `resolve("/a", "b/c") === "C:\\a\\b\\c"`, and a drive-less rooted
/// input such as `"/abs"` becomes `"C:\\abs"` because `resolve` prefixes the
/// current drive (or the matching `=<device>` environment entry).
fn node_resolve(base: &str, target: &str) -> String {
    let combined = if is_absolute(target) || target.is_empty() {
        target.to_string()
    } else {
        let absolute_base = if is_absolute(base) || base.is_empty() {
            base.to_string()
        } else {
            let cwd = std::env::current_dir()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default();
            if cwd.is_empty() {
                base.to_string()
            } else {
                format!("{cwd}{}{base}", separator())
            }
        };
        if absolute_base.is_empty() {
            target.to_string()
        } else {
            format!("{absolute_base}{}{target}", separator())
        }
    };

    let (prefix, parts) = split_root_and_segments(&combined);
    let joined = parts.join(&separator().to_string());
    let root = with_current_drive(prefix);
    if root.is_empty() {
        joined
    } else if joined.is_empty() {
        root
    } else {
        format!("{root}{joined}")
    }
}

fn separator() -> char {
    if cfg!(windows) {
        '\\'
    } else {
        '/'
    }
}

/// Split a path into its root prefix (`""`, `"/"`, `"C:"`, `"\\\\server\\share"`)
/// and its collapsed `.`/`..`-free segments, the way Node's `normalizeString`
/// does for `resolve`.
fn split_root_and_segments(path: &str) -> (String, Vec<String>) {
    let mut prefix = String::new();
    let mut rest = path;
    if cfg!(windows) {
        let bytes = rest.as_bytes();
        let drive_rooted = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
        if drive_rooted {
            prefix.push_str(&rest[..2]);
            rest = &rest[2..];
            if let Some(stripped) = rest.strip_prefix(['\\', '/']) {
                prefix.push(separator());
                rest = stripped;
            }
        } else if rest.starts_with("\\\\") || rest.starts_with("//") {
            // UNC root: `\\server\share` (or the forward-slash equivalent).
            let trimmed = rest.trim_start_matches(['\\', '/']);
            let mut pieces = trimmed.splitn(3, ['\\', '/']);
            let server = pieces.next().unwrap_or_default();
            let share = pieces.next().unwrap_or_default();
            let tail = pieces.next().unwrap_or_default();
            prefix = format!("{s}{s}{server}{s}{share}{s}", s = separator());
            rest = tail;
        } else if rest.starts_with(['\\', '/']) {
            prefix.push(separator());
            rest = rest.trim_start_matches(['\\', '/']);
        }
    } else if let Some(stripped) = rest.strip_prefix('/') {
        prefix.push('/');
        rest = stripped;
    }

    let mut parts: Vec<String> = Vec::new();
    for segment in rest.split(['/', '\\']) {
        match segment {
            "" | "." => {}
            ".." => {
                // Node clamps `..` at the root of an absolute path and keeps it
                // while the path is still relative.
                if parts.last().map(|part| part != "..").unwrap_or(false) {
                    parts.pop();
                } else if prefix.is_empty() {
                    parts.push("..".to_string());
                }
            }
            other => parts.push(other.to_string()),
        }
    }
    (prefix, parts)
}

/// A rootless rooted path (`/a`) inherits the current drive on win32; Node
/// reads `process.env[`=${device}`] || process.cwd()` for that step.
fn with_current_drive(prefix: String) -> String {
    if !cfg!(windows) {
        return prefix;
    }
    if prefix.starts_with("\\\\") || prefix.starts_with("//") {
        return prefix;
    }
    let drive = current_drive();
    if drive.is_empty() {
        return prefix;
    }
    if prefix.is_empty() {
        return String::new();
    }
    if prefix == separator().to_string() {
        return format!("{drive}{}", separator());
    }
    if prefix.starts_with(separator()) {
        // A bare rooted prefix keeps its separator and gains the drive.
        return format!("{drive}{prefix}");
    }
    prefix
}

fn current_drive() -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| {
            cwd.components()
                .next()
                .and_then(|component| match component {
                    std::path::Component::Prefix(prefix) => {
                        Some(prefix.as_os_str().to_string_lossy().to_string())
                    }
                    _ => None,
                })
        })
        .unwrap_or_default()
}

/// NodeJS `path.isAbsolute` on the current platform.
///
/// Node's win32 flavour treats a bare rooted path as ABSOLUTE
/// (`path.isAbsolute("/abs") === true`), while `std::path::Path::is_absolute`
/// requires a drive or UNC prefix and reports false. `path-utils.ts:56-58`
/// returns such a path through `expandPath` unchanged, so the port must agree
/// or it joins a value the TypeScript passes through verbatim.
pub fn is_absolute(file_path: &str) -> bool {
    if file_path.is_empty() {
        return false;
    }
    if cfg!(windows) {
        let bytes = file_path.as_bytes();
        let drive_absolute = bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/');
        // Node's win32 `isAbsolute` is `isPathSeparator(code) || deviceRoot`.
        // `"C:a"` is drive-RELATIVE and stays false; one leading `/` or `\`
        // is already enough.
        return drive_absolute || bytes[0] == b'\\' || bytes[0] == b'/';
    }
    file_path.starts_with('/')
}

/// Lexical path normalization (resolves `.` and `..` without touching the disk).
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !result.pop() {
                    result.push("..");
                }
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

pub fn resolve_read_path(file_path: &str, cwd: &str) -> String {
    let resolved = resolve_to_cwd(file_path, cwd);

    if file_exists(&resolved) {
        return resolved;
    }

    let am_pm_variant = try_macos_screenshot_path(&resolved);
    if am_pm_variant != resolved && file_exists(&am_pm_variant) {
        return am_pm_variant;
    }

    let nfd_variant = try_nfd_variant(&resolved);
    if nfd_variant != resolved && file_exists(&nfd_variant) {
        return nfd_variant;
    }

    let curly_variant = try_curly_quote_variant(&resolved);
    if curly_variant != resolved && file_exists(&curly_variant) {
        return curly_variant;
    }

    let nfd_curly_variant = try_curly_quote_variant(&nfd_variant);
    if nfd_curly_variant != resolved && file_exists(&nfd_curly_variant) {
        return nfd_curly_variant;
    }

    resolved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_home_shortcut() {
        let home = home_dir();
        assert_eq!(expand_path_with_platform("~", "linux"), home);
        assert_eq!(expand_path_with_platform("~/x", "linux"), posix_join(&home, "x"));
        assert!(expand_path_with_platform("~\\x", "win32").ends_with("x"));
        assert_eq!(expand_path_with_platform("~\\x", "linux"), "~\\x");
    }

    #[test]
    fn strips_at_prefix_and_normalizes_spaces() {
        assert_eq!(expand_path_with_platform("@/tmp/a", "linux"), "/tmp/a");
        assert_eq!(expand_path_with_platform("/tmp\u{00A0}a", "linux"), "/tmp a");
    }

    /// The drive the host `resolve` prefixes to a drive-less rooted input.
    ///
    /// `path.resolve` falls back to `process.cwd()` while the result is not yet
    /// absolute, so `resolve("/a", "b/c")` is drive-ABSOLUTE even though `/a`
    /// only looks rooted. Measured on this host (Node v24.16.0):
    /// `resolve("/a","b/c") === "C:\\a\\b\\c"`, `resolve("/a/c","../b") === "C:\\a\\b"`.
    fn host_drive() -> String {
        std::env::current_dir()
            .ok()
            .and_then(|cwd| match cwd.components().next() {
                Some(std::path::Component::Prefix(prefix)) => {
                    Some(prefix.as_os_str().to_string_lossy().to_string())
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    #[test]
    fn resolves_relative_paths_against_cwd() {
        // `test/path-utils.test.ts:56-59` requires `resolveToCwd(relative, cwd)` to equal
        // `resolve(cwd, relative)`, and win32 `resolve` returns a drive-absolute path with
        // the platform separator. `test/path-utils.test.ts:50-54` covers the absolute case.
        let drive = host_drive();
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(resolve_to_cwd("b/c", "/a"), format!("{drive}{sep}a{sep}b{sep}c"));
        assert_eq!(resolve_to_cwd("/abs", "/a"), "/abs");
        assert_eq!(resolve_to_cwd("../b", "/a/c"), format!("{drive}{sep}a{sep}b"));
    }

    #[cfg(windows)]
    #[test]
    fn unc_paths_keep_their_server_share_root() {
        assert_eq!(
            resolve_path(r"C:\work", r"\\server\share\folder\file.txt"),
            r"\\server\share\folder\file.txt"
        );
        assert_eq!(
            resolve_path("C:/work", "//server/share/folder/file.txt"),
            r"\\server\share\folder\file.txt"
        );
        assert_eq!(
            resolve_to_cwd(r"\\server\share\folder\file.txt", "C:/work"),
            r"\\server\share\folder\file.txt"
        );
        assert_eq!(
            resolve_to_cwd("//server/share/folder/file.txt", "C:/work"),
            "//server/share/folder/file.txt"
        );
        assert!(!resolve_path("C:/work", "//server/share/folder").starts_with(host_drive().as_str()));
    }

    #[cfg(windows)]
    #[test]
    fn forward_slash_local_paths_resolve_for_write_and_read() {
        let directory = tempfile::tempdir().unwrap();
        let base = directory.path().to_string_lossy().replace('\\', "/");
        let write_path = resolve_path(&base, "nested/file.txt");
        std::fs::create_dir_all(Path::new(&write_path).parent().unwrap()).unwrap();
        std::fs::write(&write_path, "local").unwrap();
        let read_input = write_path.replace('\\', "/");
        let read_path = resolve_read_path(&read_input, &base);
        assert_eq!(std::fs::read_to_string(read_path).unwrap(), "local");
        assert!(!write_path.starts_with("\\\\"));
    }

    #[test]
    fn macos_screenshot_variant_uses_narrow_no_break_space() {
        assert_eq!(
            try_macos_screenshot_path("/tmp/Screen Shot 10.00 AM.png"),
            format!("/tmp/Screen Shot 10.00{NARROW_NO_BREAK_SPACE}AM.png")
        );
    }

    #[test]
    fn curly_quote_variant_replaces_apostrophes() {
        assert_eq!(try_curly_quote_variant("Capture d'ecran.png"), "Capture d\u{2019}ecran.png");
    }

    #[test]
    fn missing_file_returns_resolved_path() {
        let resolved = resolve_read_path("/definitely/missing/file.txt", "/");
        assert_eq!(resolved, "/definitely/missing/file.txt");
    }
}
