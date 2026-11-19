//! Port of packages/coding-agent/src/core/tools/output-accumulator.ts

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use rand::Rng;

use super::truncate::{byte_length, truncate_tail, TruncatedBy, TruncationOptions, TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};

#[derive(Debug, Clone, Default)]
pub struct OutputAccumulatorOptions {
    pub max_lines: Option<usize>,
    pub max_bytes: Option<usize>,
    pub temp_file_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutputSnapshot {
    pub content: String,
    pub truncation: TruncationResult,
    pub full_output_path: Option<String>,
}

fn default_temp_file_path(prefix: &str) -> String {
    let id: [u8; 8] = rand::thread_rng().gen();
    let hex = id.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    let mut path = std::env::temp_dir();
    path.push(format!("{prefix}-{hex}.log"));
    path.to_string_lossy().into_owned()
}

/// One spill lifecycle with exactly two terminal states: a COMPLETE file whose
/// path finalize() resolves, or a DEGRADED spill (failure at open, write, or
/// final flush) whose path is never advertised. finalize() never rejects; the
/// caller keeps its bounded in-memory tail either way.
pub struct OutputSpill {
    prefix: String,
    path: Option<String>,
    stream: Option<File>,
    failed: bool,
    partial_path: Option<PathBuf>,
}

impl OutputSpill {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            path: None,
            stream: None,
            failed: false,
            partial_path: None,
        }
    }

    pub fn is_open(&self) -> bool {
        self.stream.is_some()
    }

    /// Advertisable path; `None` once the spill degraded.
    pub fn current_path(&self) -> Option<String> {
        self.path.clone()
    }

    /// Open once, writing `replay` first; a degraded spill never reopens.
    pub fn open<'a, I>(&mut self, replay: I)
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        if self.stream.is_some() || self.failed {
            return;
        }
        let path = default_temp_file_path(&self.prefix);
        match File::create(&path) {
            Ok(mut file) => {
                self.path = Some(path);
                for chunk in replay {
                    if let Err(error) = file.write_all(chunk) {
                        self.degrade(Some(error));
                        return;
                    }
                }
                self.stream = Some(file);
            }
            Err(error) => {
                // An unwritable tmpdir degrades the spill instead of crashing.
                self.degrade(Some(error));
            }
        }
    }

    pub fn write(&mut self, chunk: &[u8]) {
        if let Some(stream) = self.stream.as_mut() {
            if let Err(error) = stream.write_all(chunk) {
                self.degrade(Some(error));
            }
        }
    }

    /// Flush and settle: the complete file's path, or `None` when degraded.
    pub fn finalize(&mut self) -> Option<String> {
        let stream = self.stream.take();
        if let Some(mut stream) = stream {
            if let Err(error) = stream.flush() {
                self.degrade(Some(error));
                return None;
            }
        }
        if self.failed {
            None
        } else {
            self.path.clone()
        }
    }

    fn degrade(&mut self, _error: Option<std::io::Error>) {
        self.failed = true;
        let partial = self.path.take();
        self.stream = None;
        // The partial file would squat on the disk pressure that degraded the spill.
        if let Some(partial) = partial {
            self.partial_path = Some(PathBuf::from(&partial));
            let _ = std::fs::remove_file(&partial);
        }
    }
}

/// Incrementally tracks streaming output with bounded memory.
///
/// Appends decode chunks with a streaming UTF-8 decoder, keeps only a decoded
/// tail for display snapshots, and opens a temp file when the full output needs
/// to be preserved.
pub struct OutputAccumulator {
    max_lines: usize,
    max_bytes: usize,
    max_rolling_bytes: usize,
    temp_file_prefix: String,
    decoder: Utf8StreamDecoder,

    raw_chunks: Vec<Vec<u8>>,
    tail_text: String,
    tail_bytes: usize,
    // Bounded real suffix of text trimmed from the first retained logical
    // line. Empty means the tail starts at a line boundary.
    tail_first_line_removed_suffix: String,
    total_raw_bytes: usize,
    total_decoded_bytes: usize,
    total_lines: usize,
    current_line_bytes: usize,
    finished: bool,

    spill: OutputSpill,
}

impl OutputAccumulator {
    pub fn new(options: OutputAccumulatorOptions) -> Self {
        let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
        let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        let max_rolling_bytes = std::cmp::max(max_bytes * 2, 1);
        let temp_file_prefix = options
            .temp_file_prefix
            .clone()
            .unwrap_or_else(|| "pi-output".to_string());
        Self {
            max_lines,
            max_bytes,
            max_rolling_bytes,
            temp_file_prefix: temp_file_prefix.clone(),
            decoder: Utf8StreamDecoder::new(),
            raw_chunks: Vec::new(),
            tail_text: String::new(),
            tail_bytes: 0,
            tail_first_line_removed_suffix: String::new(),
            total_raw_bytes: 0,
            total_decoded_bytes: 0,
            total_lines: 1,
            current_line_bytes: 0,
            finished: false,
            spill: OutputSpill::new(&temp_file_prefix),
        }
    }

    /// TypeScript `new OutputAccumulator({ tempFilePrefix })`.
    pub fn with_temp_file_prefix(mut options: OutputAccumulatorOptions, temp_file_prefix: &str) -> Self {
        options.temp_file_prefix = Some(temp_file_prefix.to_string());
        Self::new(options)
    }

    pub fn append(&mut self, data: &[u8]) -> Result<(), String> {
        if self.finished {
            return Err("Cannot append to a finished output accumulator".to_string());
        }

        self.total_raw_bytes += data.len();
        let decoded = self.decoder.decode(data, true);
        self.append_decoded_text(&decoded);

        if self.spill.is_open() || self.should_use_temp_file() {
            self.ensure_temp_file();
            self.spill.write(data);
        } else if !data.is_empty() {
            self.raw_chunks.push(data.to_vec());
        }
        Ok(())
    }

    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let decoded = self.decoder.decode(&[], false);
        self.append_decoded_text(&decoded);
        if self.should_use_temp_file() {
            self.ensure_temp_file();
        }
    }

    pub fn snapshot(&self) -> OutputSnapshot {
        let tail_truncation = truncate_tail(
            &self.get_snapshot_text(),
            TruncationOptions {
                max_lines: Some(self.max_lines),
                max_bytes: Some(self.max_bytes),
            },
        );
        let truncated = self.total_lines > self.max_lines || self.total_decoded_bytes > self.max_bytes;
        let truncated_by = if truncated {
            tail_truncation.truncated_by.or(Some(if self.total_decoded_bytes > self.max_bytes {
                TruncatedBy::Bytes
            } else {
                TruncatedBy::Lines
            }))
        } else {
            None
        };
        let truncation = TruncationResult {
            truncated,
            truncated_by,
            total_lines: self.total_lines,
            total_bytes: self.total_decoded_bytes,
            max_lines: self.max_lines,
            max_bytes: self.max_bytes,
            ..tail_truncation
        };

        OutputSnapshot {
            content: truncation.content.clone(),
            truncation,
            full_output_path: self.spill.current_path(),
        }
    }

    /// Settle the spill; never rejects. Afterwards snapshot().fullOutputPath is
    /// terminal: a complete file, or `None` when the spill degraded.
    pub fn close_temp_file(&mut self) {
        self.spill.finalize();
    }

    pub fn get_last_line_bytes(&self) -> usize {
        self.current_line_bytes
    }

    fn append_decoded_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        let bytes = byte_length(text);
        self.total_decoded_bytes += bytes;
        self.tail_text.push_str(text);
        self.tail_bytes += bytes;
        if self.tail_bytes > self.max_rolling_bytes * 2 {
            self.trim_tail();
        }

        let mut newlines = 0usize;
        let mut last_newline: Option<usize> = None;
        for (index, character) in text.char_indices() {
            if character == '\n' {
                newlines += 1;
                last_newline = Some(index);
            }
        }
        if newlines == 0 {
            self.current_line_bytes += bytes;
        } else {
            self.total_lines += newlines;
            let last_newline = last_newline.expect("newline counted");
            self.current_line_bytes = byte_length(&text[last_newline + 1..]);
        }
    }

    fn trim_tail(&mut self) {
        let buffer = self.tail_text.as_bytes().to_vec();
        if buffer.len() <= self.max_rolling_bytes {
            self.tail_bytes = buffer.len();
            return;
        }

        let mut start = buffer.len() - self.max_rolling_bytes;
        while start < buffer.len() && (buffer[start] & 0xc0) == 0x80 {
            start += 1;
        }

        if start > 0 {
            let removed = match buffer[..start].iter().rposition(|byte| *byte == b'\n') {
                Some(newline) => String::from_utf8_lossy(&buffer[newline + 1..start]).into_owned(),
                None => {
                    let mut removed = std::mem::take(&mut self.tail_first_line_removed_suffix);
                    removed.push_str(&String::from_utf8_lossy(&buffer[..start]));
                    removed
                }
            };
            self.tail_first_line_removed_suffix =
                retain_removed_line_suffix(&removed, self.max_bytes);
        }
        self.tail_text = String::from_utf8_lossy(&buffer[start..]).into_owned();
        self.tail_bytes = byte_length(&self.tail_text);
    }

    fn get_snapshot_text(&self) -> String {
        if self.tail_first_line_removed_suffix.is_empty() {
            self.tail_text.clone()
        } else {
            let mut snapshot = self.tail_first_line_removed_suffix.clone();
            snapshot.push_str(&self.tail_text);
            snapshot
        }
    }

    fn should_use_temp_file(&self) -> bool {
        self.total_raw_bytes > self.max_bytes
            || self.total_decoded_bytes > self.max_bytes
            || self.total_lines > self.max_lines
    }

    fn ensure_temp_file(&mut self) {
        let replay = std::mem::take(&mut self.raw_chunks);
        self.spill
            .open(replay.iter().map(|chunk| chunk.as_slice()));
        self.raw_chunks = Vec::new();
    }
}

/// Keep enough real removed-prefix bytes that tail truncation never needs the
/// unbounded history. If older bytes exist, this suffix exceeds max_bytes;
/// otherwise it is the complete removed prefix.
fn retain_removed_line_suffix(text: &str, max_bytes: usize) -> String {
    let cap = max_bytes.saturating_add(1);
    if text.len() <= cap {
        return text.to_string();
    }
    let mut start = text.len() - cap;
    while start > 0 && (text.as_bytes()[start] & 0xc0) == 0x80 {
        start -= 1;
    }
    text[start..].to_string()
}

/// Streaming UTF-8 decoder mirroring `new TextDecoder()` with `{ stream: true }`.
struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    fn new() -> Self {
        Self { pending: Vec::new() }
    }

    fn decode(&mut self, chunk: &[u8], stream: bool) -> String {
        self.pending.extend_from_slice(chunk);
        let mut out = String::new();
        let mut index = 0usize;
        let bytes = self.pending.clone();
        while index < bytes.len() {
            match std::str::from_utf8(&bytes[index..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    index = bytes.len();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        out.push_str(std::str::from_utf8(&bytes[index..index + valid_up_to]).expect("valid prefix"));
                        index += valid_up_to;
                    }
                    let incomplete = error.error_len().is_none();
                    if incomplete && stream {
                        // Keep the incomplete trailing sequence for the next chunk.
                        break;
                    }
                    if error.error_len().is_none() {
                        // Flush: an unterminated sequence at the end is replaced.
                        out.push('\u{FFFD}');
                        index = bytes.len();
                        break;
                    }
                    let error_len = error.error_len().expect("error length");
                    out.push('\u{FFFD}');
                    index += error_len;
                }
            }
        }
        self.pending = bytes[index.min(bytes.len())..].to_vec();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_output_never_spills() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions::default());
        accumulator.append(b"hello\nworld\n").expect("append");
        accumulator.finish();
        let snapshot = accumulator.snapshot();
        assert_eq!(snapshot.content, "hello\nworld\n");
        assert!(!snapshot.truncation.truncated);
        assert_eq!(snapshot.full_output_path, None);
        assert_eq!(accumulator.get_last_line_bytes(), 0);
    }

    #[test]
    fn tracks_total_lines_and_last_line_bytes() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions::default());
        accumulator.append(b"a\nbc").expect("append");
        accumulator.finish();
        let snapshot = accumulator.snapshot();
        assert_eq!(snapshot.truncation.total_lines, 2);
        assert_eq!(accumulator.get_last_line_bytes(), 2);
    }

    #[test]
    fn large_output_spills_to_temp_file_and_reports_path() {
        let mut accumulator = OutputAccumulator::with_temp_file_prefix(
            OutputAccumulatorOptions {
                max_lines: Some(2),
                max_bytes: Some(4),
                ..Default::default()
            },
            "pi-test",
        );
        accumulator.append(b"1\n2\n3\n4\n").expect("append");
        accumulator.finish();
        let snapshot = accumulator.snapshot();
        assert!(snapshot.truncation.truncated);
        // split("\n") counts the terminal empty line against maxLines.
        assert_eq!(snapshot.content, "4\n");
        assert_eq!(snapshot.truncation.output_lines, 2);
        let path = snapshot.full_output_path.clone().expect("spill path");
        accumulator.close_temp_file();
        let final_path = accumulator.snapshot().full_output_path.expect("terminal path");
        assert_eq!(path, final_path);
        assert_eq!(std::fs::read_to_string(&final_path).expect("spilled content"), "1\n2\n3\n4\n");
        let _ = std::fs::remove_file(final_path);
    }

    #[test]
    fn append_after_finish_is_rejected() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions::default());
        accumulator.finish();
        let error = accumulator.append(b"x").expect_err("must reject");
        assert_eq!(error, "Cannot append to a finished output accumulator");
    }

    #[test]
    fn split_multibyte_chunk_decodes_across_appends() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions::default());
        let text = "héllo";
        let bytes = text.as_bytes();
        accumulator.append(&bytes[..2]).expect("append");
        accumulator.append(&bytes[2..]).expect("append");
        accumulator.finish();
        assert_eq!(accumulator.snapshot().content, text);
    }

    #[test]
    fn invalid_utf8_is_replaced() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions::default());
        accumulator.append(&[0xff, 0xfe]).expect("append");
        accumulator.finish();
        assert_eq!(accumulator.snapshot().content, "\u{FFFD}\u{FFFD}");
    }

    fn assert_snapshot_matches_full_output(
        accumulator: &OutputAccumulator,
        input: &str,
        options: &OutputAccumulatorOptions,
    ) {
        let expected = truncate_tail(
            input,
            TruncationOptions {
                max_lines: options.max_lines,
                max_bytes: options.max_bytes,
            },
        );
        let actual = accumulator.snapshot();
        let expected_snapshot = OutputSnapshot {
            content: expected.content.clone(),
            truncation: expected,
            full_output_path: actual.full_output_path.clone(),
        };
        assert_eq!(actual, expected_snapshot);
    }

    #[test]
    fn rolling_snapshot_preserves_multibyte_oob_boundaries_and_partial_metadata() {
        for (input, max_bytes, expected_content, expected_lines) in
            [("🙂\n", 1, "", 1), ("xxxxxxxxxx界b\n", 2, "b\n", 2)]
        {
            let options = OutputAccumulatorOptions {
                max_lines: Some(2),
                max_bytes: Some(max_bytes),
                ..Default::default()
            };
            let mut accumulator =
                OutputAccumulator::with_temp_file_prefix(options.clone(), "pi-test-boundary");
            accumulator.append(input.as_bytes()).expect("append");
            accumulator.finish();
            assert_snapshot_matches_full_output(&accumulator, input, &options);
            let actual = accumulator.snapshot();
            assert_eq!(actual.content, expected_content);
            assert_eq!(actual.truncation.output_lines, expected_lines);
            assert!(actual.truncation.last_line_partial);
            accumulator.close_temp_file();
            if let Some(path) = accumulator.snapshot().full_output_path {
                assert_eq!(
                    std::fs::read_to_string(&path).expect("spilled output"),
                    input
                );
                let _ = std::fs::remove_file(path);
            }
        }
    }

    #[test]
    fn rolling_snapshot_matches_full_tail_with_a_retained_partial_line() {
        let input = format!("{}\ntail", "x".repeat(90));
        let options = OutputAccumulatorOptions {
            max_lines: Some(100),
            max_bytes: Some(16),
            ..Default::default()
        };
        let mut accumulator =
            OutputAccumulator::with_temp_file_prefix(options.clone(), "pi-test-rolling");
        accumulator.append(input.as_bytes()).expect("append");
        accumulator.finish();

        let expected = truncate_tail(
            &input,
            TruncationOptions {
                max_lines: options.max_lines,
                max_bytes: options.max_bytes,
            },
        );
        let actual = accumulator.snapshot();
        let expected_snapshot = OutputSnapshot {
            content: expected.content.clone(),
            truncation: expected,
            full_output_path: actual.full_output_path.clone(),
        };
        assert_eq!(actual, expected_snapshot);
        assert_eq!(actual.content, format!("{}\ntail", "x".repeat(11)));
        assert!(actual.truncation.last_line_partial);

        accumulator.close_temp_file();
        let path = accumulator
            .snapshot()
            .full_output_path
            .expect("complete spill");
        assert_eq!(std::fs::read_to_string(&path).expect("full output"), input);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bounded_rolling_snapshot_preserves_the_default_final_error_and_spill() {
        let final_error = "FINAL ERROR: the operation failed";
        let input = format!("{}{}", "x".repeat(220 * 1024), final_error);
        let mut accumulator = OutputAccumulator::with_temp_file_prefix(
            OutputAccumulatorOptions::default(),
            "pi-test-default-rolling",
        );
        for chunk in input.as_bytes().chunks(4096) {
            accumulator.append(chunk).expect("append");
            assert!(accumulator.tail_bytes <= accumulator.max_rolling_bytes * 2);
        }
        accumulator.finish();

        let expected = truncate_tail(&input, TruncationOptions::default());
        let actual = accumulator.snapshot();
        let expected_snapshot = OutputSnapshot {
            content: expected.content.clone(),
            truncation: expected,
            full_output_path: actual.full_output_path.clone(),
        };
        assert_eq!(actual, expected_snapshot);
        assert!(actual.content.ends_with(final_error));
        assert!(accumulator.tail_bytes <= accumulator.max_rolling_bytes * 2);

        accumulator.close_temp_file();
        let path = accumulator
            .snapshot()
            .full_output_path
            .expect("complete spill");
        assert_eq!(std::fs::read(&path).expect("full output"), input.as_bytes());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn snapshot_uses_total_counters_for_truncation_flags() {
        let mut accumulator = OutputAccumulator::with_temp_file_prefix(
            OutputAccumulatorOptions {
                max_lines: Some(10),
                max_bytes: Some(1024),
                ..Default::default()
            },
            "pi-test",
        );
        for index in 0..40 {
            accumulator.append(format!("line {index}\n").as_bytes()).expect("append");
        }
        accumulator.finish();
        let snapshot = accumulator.snapshot();
        assert!(snapshot.truncation.truncated);
        assert_eq!(snapshot.truncation.total_lines, 41);
        assert!(snapshot.truncation.output_lines <= 10);
        accumulator.close_temp_file();
        if let Some(path) = accumulator.snapshot().full_output_path {
            let _ = std::fs::remove_file(path);
        }
    }
}
