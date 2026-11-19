//! Port of packages/coding-agent/src/core/tools/truncate.ts
//!
//! Shared truncation utilities for tool outputs.
//!
//! Truncation is based on two independent limits - whichever is hit first wins:
//! - Line limit (default: 2000 lines)
//! - Byte limit (default: 50KB)
//!
//! Never returns partial lines (except bash tail truncation edge case).

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024; // 50KB
pub const GREP_MAX_LINE_LENGTH: usize = 500; // Max chars per grep match line

/// Which limit was hit: "lines", "bytes", or null if not truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruncationResult {
    /// The truncated content
    pub content: String,
    /// Whether truncation occurred
    pub truncated: bool,
    /// Which limit was hit: "lines", "bytes", or null if not truncated
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content
    pub total_lines: usize,
    /// Total number of bytes in the original content
    pub total_bytes: usize,
    /// Number of complete lines in the truncated output
    pub output_lines: usize,
    /// Number of bytes in the truncated output
    pub output_bytes: usize,
    /// Whether the last line was partially truncated (only for tail truncation edge case)
    pub last_line_partial: bool,
    /// Whether the first line exceeded the byte limit (for head truncation)
    pub first_line_exceeds_limit: bool,
    /// The max lines limit that was applied
    pub max_lines: usize,
    /// The max bytes limit that was applied
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TruncationOptions {
    pub max_lines: Option<usize>,
    pub max_bytes: Option<usize>,
}

/// Format bytes as human-readable size.
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// JavaScript `String.prototype.split("\n")` length for the same text.
pub(crate) fn split_newline_len(content: &str) -> usize {
    content.split('\n').count()
}

/// `Buffer.byteLength(content, "utf-8")`.
pub(crate) fn byte_length(text: &str) -> usize {
    text.len()
}

/// Truncate content from the head (keep first N lines/bytes).
/// Suitable for file reads where you want to see the beginning.
///
/// Never returns partial lines. If first line exceeds byte limit,
/// returns empty content with firstLineExceedsLimit=true.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

    let total_bytes = byte_length(content);
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let first_line_bytes = byte_length(lines[0]);
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines,
            max_bytes,
        };
    }

    let mut output_lines_arr: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (i, line) in lines.iter().enumerate() {
        if i >= max_lines {
            break;
        }
        let line_bytes = byte_length(line) + if i > 0 { 1 } else { 0 }; // +1 for newline

        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }

        output_lines_arr.push(line);
        output_bytes_count += line_bytes;
    }

    if output_lines_arr.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines_arr.join("\n");
    let final_output_bytes = byte_length(&output_content);

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines_arr.len(),
        output_bytes: final_output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate content from the tail (keep last N lines/bytes).
/// Suitable for bash output where you want to see the end (errors, final results).
///
/// May return partial first line if the last line of original content exceeds byte limit.
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

    let total_bytes = byte_length(content);
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let mut output_lines_arr: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;
    let mut partial_line: Option<String> = None;

    let mut i = lines.len();
    while i > 0 && output_lines_arr.len() < max_lines {
        i -= 1;
        let line = lines[i];
        let line_bytes = byte_length(line) + if !output_lines_arr.is_empty() { 1 } else { 0 }; // +1 for newline

        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Preserve the tail of an oversized first collected line together
            // with as many later lines as the byte budget can fit. Each retained
            // later line also needs its rejoined newline separator.
            let mut trailing_cost: usize = output_lines_arr
                .iter()
                .map(|collected| 1 + byte_length(collected))
                .sum();
            let mut first_trailing = 0usize;
            while first_trailing < output_lines_arr.len() && trailing_cost + 1 > max_bytes {
                trailing_cost -= 1 + byte_length(output_lines_arr[first_trailing]);
                first_trailing += 1;
            }
            if first_trailing > 0 {
                output_lines_arr.drain(..first_trailing);
            }
            let truncated_line =
                truncate_string_to_bytes_from_end(line, max_bytes.saturating_sub(trailing_cost));
            output_bytes_count = byte_length(&truncated_line) + trailing_cost;
            partial_line = Some(truncated_line);
            last_line_partial = true;
            break;
        }

        output_lines_arr.insert(0, line);
        output_bytes_count += line_bytes;
    }

    if output_lines_arr.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_lines_count = output_lines_arr.len() + usize::from(partial_line.is_some());
    let mut collected: Vec<String> = Vec::with_capacity(output_lines_count);
    if let Some(partial) = partial_line {
        collected.push(partial);
    }
    collected.extend(output_lines_arr.iter().map(|line| (*line).to_string()));
    let output_content = collected.join("\n");
    let final_output_bytes = byte_length(&output_content);

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines_count,
        output_bytes: final_output_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate a string to fit within a byte limit (from the end).
/// Handles multi-byte UTF-8 characters correctly.
fn truncate_string_to_bytes_from_end(text: &str, max_bytes: usize) -> String {
    let buf = text.as_bytes();
    if buf.len() <= max_bytes {
        return text.to_string();
    }

    let mut start = buf.len() - max_bytes;

    // Find a valid UTF-8 boundary (start of a character)
    while start < buf.len() && (buf[start] & 0xc0) == 0x80 {
        start += 1;
    }

    String::from_utf8_lossy(&buf[start..]).into_owned()
}

/// Truncate a single line to max characters, adding [truncated] suffix.
/// Used for grep match lines.
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
    if line.chars().count() <= max_chars {
        return TruncatedLine {
            text: line.to_string(),
            was_truncated: false,
        };
    }
    TruncatedLine {
        text: format!("{}... [truncated]", line.chars().take(max_chars).collect::<String>()),
        was_truncated: true,
    }
}

/// TypeScript `truncateLine` returns `{ text, wasTruncated }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncatedLine {
    pub text: String,
    pub was_truncated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_matches_js_to_fixed() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(1023), "1023B");
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(50 * 1024), "50.0KB");
        assert_eq!(format_size(1024 * 1024), "1.0MB");
        assert_eq!(format_size(1536 * 1024), "1.5MB");
    }

    #[test]
    fn truncate_head_keeps_short_content() {
        let result = truncate_head("a\nb", TruncationOptions::default());
        assert!(!result.truncated);
        assert_eq!(result.truncated_by, None);
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.output_lines, 2);
        assert_eq!(result.content, "a\nb");
    }

    #[test]
    fn truncate_head_reports_first_line_exceeds_limit() {
        let content = "x".repeat(20);
        let result = truncate_head(&content, TruncationOptions { max_lines: None, max_bytes: Some(10) });
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert!(result.first_line_exceeds_limit);
        assert_eq!(result.content, "");
        assert_eq!(result.output_lines, 0);
    }

    #[test]
    fn truncate_head_hits_line_limit() {
        let content = "a\nb\nc\nd";
        let result = truncate_head(&content, TruncationOptions { max_lines: Some(2), max_bytes: None });
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.content, "a\nb");
        assert_eq!(result.total_lines, 4);
        assert_eq!(result.output_lines, 2);
    }

    #[test]
    fn truncate_head_hits_byte_limit() {
        let content = "aaa\nbbb\nccc";
        let result = truncate_head(&content, TruncationOptions { max_lines: None, max_bytes: Some(7) });
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(result.content, "aaa\nbbb");
        assert_eq!(result.output_lines, 2);
        assert_eq!(result.output_bytes, 7);
    }

    #[test]
    fn truncate_tail_hits_line_limit_from_end() {
        let content = "a\nb\nc\nd";
        let result = truncate_tail(&content, TruncationOptions { max_lines: Some(2), max_bytes: None });
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.content, "c\nd");
        assert_eq!(result.output_lines, 2);
        assert!(!result.last_line_partial);
    }

    #[test]
    fn truncate_tail_keeps_partial_line_for_oversized_last_line() {
        let content = "a\nb\ncdefghij";
        let result = truncate_tail(&content, TruncationOptions { max_lines: None, max_bytes: Some(5) });
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert!(result.last_line_partial);
        assert_eq!(result.content, "fghij");
        assert_eq!(result.output_lines, 1);
    }

    #[test]
    fn truncate_tail_rescues_oversized_line_with_trailing_blanks() {
        let content = "abcdefghij\n\n";
        let result = truncate_tail(&content, TruncationOptions { max_lines: None, max_bytes: Some(6) });
        assert!(result.last_line_partial);
        // keptBlanks = min(2, 6 - 1) = 2 -> truncated line budget = 6 - 2 = 4
        assert_eq!(result.content, "ghij\n\n");
        assert_eq!(result.output_bytes, 6);
    }

    #[test]
    fn truncate_tail_keeps_a_partial_oversized_line_before_a_nonblank_tail() {
        let content = format!("{}\ntail", "x".repeat(90));
        let result = truncate_tail(
            &content,
            TruncationOptions {
                max_lines: Some(100),
                max_bytes: Some(16),
            },
        );
        assert_eq!(result.content, format!("{}\ntail", "x".repeat(11)));
        assert_eq!(result.output_lines, 2);
        assert_eq!(result.output_bytes, 16);
        assert!(result.last_line_partial);
    }

    #[test]
    fn truncate_tail_handles_multibyte_boundary() {
        let content = "ééé";
        let result = truncate_tail(&content, TruncationOptions { max_lines: None, max_bytes: Some(3) });
        // 3 bytes of a 2-byte-per-char string lands mid-character; the scan advances to a boundary.
        assert!(result.last_line_partial);
        assert_eq!(result.content, "é");
    }

    #[test]
    fn truncate_line_suffix_matches_typescript() {
        assert_eq!(truncate_line("abc", 5), TruncatedLine { text: "abc".to_string(), was_truncated: false });
        assert_eq!(
            truncate_line("abcdef", 3),
            TruncatedLine { text: "abc... [truncated]".to_string(), was_truncated: true }
        );
        assert_eq!(truncate_line("abcdef", GREP_MAX_LINE_LENGTH).was_truncated, false);
    }
}
