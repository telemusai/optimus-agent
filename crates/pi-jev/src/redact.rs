//! Secret redaction and bounded excerpt construction for shadow observations.
//!
//! Jev shadow observations leave the machine inside a SystemOne request, so two
//! rules hold at every call site:
//!
//! 1. **Bound first.** An excerpt is cut from at most [`MAX_SCAN_CHARS`] leading
//!    characters of the source, so a multi-megabyte tool result or transcript is
//!    never fully copied just to keep 400 characters of it.
//! 2. **Redact before the payload exists.** Credential-shaped material is
//!    replaced with [`REDACTED`] while the excerpt is built, never after the
//!    value is already inside a request.
//!
//! Redaction is deliberately pattern-based: it names the credential shapes this
//! product emits or receives (authorization headers, provider key prefixes,
//! private-key blocks, secret-named assignments, URL userinfo, JWT-shaped
//! strings). It is NOT a general classifier for arbitrary private text, and it
//! does not make free-form task text safe to disclose. Raw tool arguments are
//! not observed at all by the bridge, and Compare stays opt-in per session.

use std::borrow::Cow;

/// Characters kept from an untrusted source before redaction scanning.
///
/// This is the hard cap that keeps excerpt construction from copying a whole
/// multi-megabyte payload: callers pass the full string, this module reads only
/// the leading window.
pub const MAX_SCAN_CHARS: usize = 4_096;

/// Marker written in place of redacted material.
pub const REDACTED: &str = "[redacted]";

/// Secret-named keys whose assigned value is always redacted.
///
/// Matching is on the normalized key (lower-case, `-` folded to `_`): a hint
/// matches when the key equals it or ends with `_<hint>`, so `x-api-key` and
/// `openai_api_key` match while `monkey` does not. Bare `key` is intentionally
/// absent: `sort_key`/`public_key` are common non-secrets, and a literal key
/// value almost always carries a self-identifying token prefix that the
/// [`TOKEN_PREFIXES`] rule catches instead.
const SECRET_KEY_HINTS: &[&str] = &[
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "pwd",
    "token",
    "credential",
    "authorization",
    "auth",
    "bearer",
    "private_key",
    "access_key",
    "secret_key",
    "signing_key",
    "session_key",
    "client_secret",
    "refresh_token",
    "id_token",
    "session_token",
    "access_token",
];

/// Self-identifying credential prefixes. A run starting with one of these is
/// redacted to its end, so a truncated copy can never leak a usable prefix.
const TOKEN_PREFIXES: &[&str] = &[
    "sk-", "sk_", "rk-", "pk-", "ghp_", "gho_", "ghu_", "ghs_", "github_pat_", "xoxb-",
    "xoxp-", "xoxa-", "xoxr-", "AKIA", "ASIA", "AIza", "ya29.", "eyJ", "hf_", "glpat-",
    "npm_", "pypi-", "dop_v1_", "-----BEGIN",
];

/// Cache the character window rather than the borrowed tail so callers can
/// pass any string without lifetime juggling.
fn char_prefix(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte, _)) => &text[..byte],
        None => text,
    }
}

fn char_take(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((byte, _)) => text[..byte].to_string(),
        None => text.to_string(),
    }
}

/// Bounded, redacted excerpt of untrusted text.
///
/// Reads at most [`MAX_SCAN_CHARS`] leading characters, redacts credential
/// shapes, then truncates to `limit` characters on char boundaries.
pub fn bounded_excerpt(text: &str, limit: usize) -> String {
    let window = char_prefix(text, MAX_SCAN_CHARS);
    match redact_window(window) {
        Cow::Borrowed(clean) => char_take(clean, limit),
        Cow::Owned(redacted) => char_take(&redacted, limit),
    }
}

/// Redact credential shapes in untrusted text. Copies only when it redacts.
pub fn redact_text(text: &str) -> String {
    match redact_window(char_prefix(text, MAX_SCAN_CHARS)) {
        Cow::Borrowed(clean) => clean.to_string(),
        Cow::Owned(redacted) => redacted,
    }
}

fn redact_window(window: &str) -> Cow<'_, str> {
    let chars: Vec<char> = window.chars().collect();
    let mut out: Option<String> = None;
    let mut i = 0usize;
    let mut copied = 0usize;

    macro_rules! flush_to {
        ($until:expr) => {
            if let Some(buffer) = out.as_mut() {
                buffer.extend(&chars[copied..$until]);
            }
        };
    }
    macro_rules! mark {
        ($at:expr) => {
            if out.is_none() {
                let mut buffer = String::with_capacity(window.len() + REDACTED.len());
                buffer.extend(&chars[..$at]);
                out = Some(buffer);
            }
        };
    }

    while i < chars.len() {
        // PEM-style private key blocks: redact the whole block.
        if starts_with(chars.as_slice(), i, "-----BEGIN") {
            let mut end = find_sub(&chars, i, "-----END")
                .map(|start| start + "-----END".len())
                .unwrap_or(chars.len());
            while end < chars.len() && chars[end] != '\n' {
                end += 1;
            }
            mark!(i);
            out.as_mut().unwrap().push_str(REDACTED);
            i = end;
            copied = i;
            continue;
        }
        // Authorization headers: `Bearer <token>` / `Basic <base64>`.
        if let Some(prefix_len) = auth_scheme_at(&chars, i) {
            mark!(i);
            let buffer = out.as_mut().unwrap();
            buffer.push_str(&chars[i..i + prefix_len].iter().collect::<String>());
            i += prefix_len;
            while i < chars.len() && chars[i].is_whitespace() {
                buffer.push(chars[i]);
                i += 1;
            }
            i = redact_run(buffer, &chars, i);
            copied = i;
            continue;
        }
        // URL userinfo: `scheme://user:password@host`.
        if starts_with(chars.as_slice(), i, "://") {
            if let Some(end) = url_userinfo_end(&chars, i + 3) {
                mark!(i);
                let buffer = out.as_mut().unwrap();
                buffer.extend(&chars[i..i + 3]);
                buffer.push_str(REDACTED);
                buffer.push('@');
                i = end + 1;
                copied = i;
                continue;
            }
        }
        // Self-identifying token prefixes. No word-boundary requirement: a
        // prefix glued to other characters (`thekeyissk-live-...`) must still be
        // redacted, and over-redacting is the safe direction.
        if token_prefix_at(&chars, i) {
            mark!(i);
            let buffer = out.as_mut().unwrap();
            i = redact_run(buffer, &chars, i);
            copied = i;
            continue;
        }
        // `secret_key = "value"` / `"api_key": "value"` assignments.
        if let Some((next, was_redacted)) = assignment_at(&chars, i) {
            if was_redacted {
                mark!(i);
                let buffer = out.as_mut().unwrap();
                buffer.extend(&chars[copied..next.0]);
                buffer.push_str(REDACTED);
                if let Some(quote) = next.2 {
                    buffer.push(quote);
                }
                i = next.1;
                copied = i;
                continue;
            }
        }
        // Default path: emit through the current character.
        i += 1;
        flush_to!(i);
        copied = i;
    }

    match out {
        Some(buffer) => Cow::Owned(buffer),
        None => Cow::Borrowed(window),
    }
}

/// Consume a credential-looking run and emit the marker instead.
fn redact_run(buffer: &mut String, chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && !is_value_delim(chars[i]) {
        i += 1;
    }
    buffer.push_str(REDACTED);
    i
}

/// Length of a leading authorization scheme, when one starts at `i`.
fn auth_scheme_at(chars: &[char], i: usize) -> Option<usize> {
    if !is_word_boundary(chars, i) {
        return None;
    }
    for scheme in ["Bearer", "bearer", "BEARER", "Basic", "basic"] {
        if starts_with(chars, i, scheme) {
            let end = i + scheme.len();
            if end < chars.len() && chars[end].is_whitespace() {
                return Some(scheme.len());
            }
        }
    }
    None
}

/// Index of the `@` when `start` begins URL userinfo carrying a password-like
/// separator (`user:pass@host`). Bare `user@host` is left alone.
fn url_userinfo_end(chars: &[char], start: usize) -> Option<usize> {
    let mut saw_separator = false;
    let mut i = start;
    while i < chars.len() {
        let c = chars[i];
        if c == '@' {
            return if saw_separator { Some(i) } else { None };
        }
        if matches!(c, '/' | '?' | '#' | '\\') || c.is_whitespace() {
            return None;
        }
        if c == ':' {
            saw_separator = true;
        }
        i += 1;
    }
    None
}

/// Detect a secret-named assignment at `i`.
///
/// Returns `(consumed, redacted)` where `consumed` is `(content_end, resume_index,
/// closing_quote)`: the span of key/separator text to copy verbatim, the index
/// the caller resumes from, and a closing quote character when the value was
/// quoted.
fn assignment_at(chars: &[char], i: usize) -> Option<((usize, usize, Option<char>), bool)> {
    let quote = match chars[i] {
        '"' | '\'' => Some(chars[i]),
        _ if is_key_char(chars[i]) => None,
        _ => return None,
    };
    let key_start = i + usize::from(quote.is_some());
    let mut j = key_start;
    while j < chars.len() && is_key_char(chars[j]) {
        j += 1;
    }
    if j == key_start {
        return None;
    }
    let key: String = chars[key_start..j].iter().collect();
    let mut k = j;
    if let Some(quote) = quote {
        if chars.get(k) != Some(&quote) {
            return None;
        }
        k += 1;
    }
    while k < chars.len() && chars[k] == ' ' {
        k += 1;
    }
    match chars.get(k) {
        Some(':') if chars.get(k + 1) != Some(&':') => {}
        Some('=') if chars.get(k + 1) != Some(&'=') => {}
        _ => return None,
    }
    if !key_is_secret(&key) {
        // Not secret: report "seen but not redacted" so the caller keeps the
        // fast path and simply copies the character.
        return Some(((j, j, None), false));
    }
    let separator_end = k + 1;
    let mut value = separator_end;
    while value < chars.len() && chars[value] == ' ' {
        value += 1;
    }
    match chars.get(value) {
        // Quoted value: copy through the opening quote so the marker sits inside
        // a balanced pair and the excerpt stays valid JSON where it started so.
        Some(value_quote @ ('"' | '\'')) => {
            let mut end = value + 1;
            while end < chars.len() && chars[end] != *value_quote {
                end += 1;
            }
            let resume = if end < chars.len() { end + 1 } else { end };
            Some(((value + 1, resume, Some(*value_quote)), true))
        }
        Some(_) => {
            let mut end = value;
            while end < chars.len() && !is_value_delim(chars[end]) {
                end += 1;
            }
            // `Authorization: Basic <base64>` / `Authorization: Bearer <token>`:
            // the scheme word is the whole first run, so the credential is the
            // run after it. Redact both.
            let run: String = chars[value..end].iter().collect();
            if is_auth_scheme_word(&run) {
                let mut next = end;
                while next < chars.len() && matches!(chars[next], ' ' | '\t') {
                    next += 1;
                }
                while next < chars.len() && !is_value_delim(chars[next]) {
                    next += 1;
                }
                end = next;
            }
            Some(((separator_end, end, None), true))
        }
        None => Some(((separator_end, value, None), true)),
    }
}

fn is_auth_scheme_word(run: &str) -> bool {
    matches!(run.to_ascii_lowercase().as_str(), "bearer" | "basic")
}

fn key_is_secret(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .map(|c| if c == '-' { '_' } else { c.to_ascii_lowercase() })
        .collect();
    SECRET_KEY_HINTS
        .iter()
        .any(|hint| normalized == *hint || normalized.ends_with(&format!("_{hint}")))
}

fn token_prefix_at(chars: &[char], i: usize) -> bool {
    TOKEN_PREFIXES.iter().any(|prefix| starts_with(chars, i, prefix))
}

fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn is_word_boundary(chars: &[char], i: usize) -> bool {
    i == 0 || !is_key_char(chars[i - 1])
}

fn is_value_delim(c: char) -> bool {
    c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ',' | ';' | ')' | ']' | '}' | '>' | '|' | '\\')
}

fn starts_with(chars: &[char], i: usize, prefix: &str) -> bool {
    let prefix: Vec<char> = prefix.chars().collect();
    i + prefix.len() <= chars.len() && chars[i..i + prefix.len()] == prefix[..]
}

fn find_sub(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let needle: Vec<char> = needle.chars().collect();
    if needle.is_empty() || chars.len() < needle.len() {
        return None;
    }
    (from..=chars.len() - needle.len()).find(|&i| chars[i..i + needle.len()] == needle[..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_authorization_header_values() {
        let text = "curl -H 'Authorization: Bearer sk-live-1234567890abcdef' https://api.example.test";
        let redacted = redact_text(text);
        assert!(!redacted.contains("1234567890abcdef"), "{redacted}");
        assert!(redacted.contains(REDACTED));
        assert!(redacted.contains("https://api.example.test"));
    }

    #[test]
    fn redacts_basic_authorization_header_values() {
        let redacted = redact_text("Authorization: Basic dXNlcjpwYXNzd29yZA==");
        assert!(!redacted.contains("dXNlcjpwYXNzd29yZA"), "{redacted}");
        assert!(redacted.contains(REDACTED));
    }

    #[test]
    fn redacts_json_secret_assignments() {
        let text = r#"{"api_key":"abc123","x-api-key":"def456","password":"hunter2","client_secret":"shh"}"#;
        let redacted = redact_text(text);
        for secret in ["abc123", "def456", "hunter2", "shh"] {
            assert!(!redacted.contains(secret), "{secret} survived: {redacted}");
        }
        assert!(redacted.contains("\"api_key\":\"[redacted]\"") || redacted.contains("api_key"), "{redacted}");
    }

    #[test]
    fn redacts_shell_style_secret_assignments() {
        let redacted = redact_text("export OPENAI_API_KEY=sk-proj-abcdef\nGITHUB_TOKEN=ghp_abcdef");
        assert!(!redacted.contains("sk-proj-abcdef"), "{redacted}");
        assert!(!redacted.contains("ghp_abcdef"), "{redacted}");
    }

    #[test]
    fn redacts_self_identifying_token_prefixes() {
        let text = "keys: ghp_abcdefghijklmnop AKIAIOSFODNN7EXAMPLE xoxb-1234-5678 eyJhbGciOiJIUzI1NiJ9";
        let redacted = redact_text(text);
        for secret in ["ghp_abcdefghijklmnop", "AKIAIOSFODNN7EXAMPLE", "xoxb-1234-5678", "eyJhbGciOiJIUzI1NiJ9"] {
            assert!(!redacted.contains(secret), "{secret} survived: {redacted}");
        }
    }

    #[test]
    fn redacts_pem_private_key_blocks() {
        let text = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----\nafter";
        let redacted = redact_text(text);
        assert!(!redacted.contains("MIIEowIBAAKCAQEA"), "{redacted}");
        assert!(redacted.contains("before") && redacted.contains("after"), "{redacted}");
    }

    #[test]
    fn redacts_url_userinfo_with_password() {
        let redacted = redact_text("clone https://alice:s3cr3t@github.example.test/repo.git now");
        assert!(!redacted.contains("s3cr3t"), "{redacted}");
        assert!(redacted.contains("github.example.test/repo.git"), "{redacted}");
        // A bare user name is not a credential.
        let plain = redact_text("ssh://git@github.example.test/repo.git");
        assert!(plain.contains("git@github.example.test"), "{plain}");
    }

    #[test]
    fn keeps_ordinary_prose_and_code() {
        let text = "fn main() { let count = 3; println!(\"hello {}\", count); } // fix the parser bug";
        assert_eq!(redact_text(text), text);
        assert_eq!(bounded_excerpt(text, 400), text);
    }

    #[test]
    fn windows_edge_redacts_a_secret_cut_by_the_scan_limit() {
        let mut text = "a".repeat(MAX_SCAN_CHARS - 20);
        text.push_str("sk-live-abcdefghijklmnopqrstuvwxyz");
        let excerpt = bounded_excerpt(&text, MAX_SCAN_CHARS);
        assert!(!excerpt.contains("sk-live-abcdefghijklmnopqrstuvwxyz"), "secret survived");
        assert!(!excerpt.contains("sk-live"), "prefix survived: {}", &excerpt[excerpt.len() - 40..]);
        assert!(excerpt.contains(REDACTED));
    }

    #[test]
    fn bounds_multi_megabyte_input_without_copying_it() {
        // 4 MiB of text: the excerpt is built from the leading window only.
        let big = format!("{}{}", "x".repeat(4 * 1024 * 1024), "sk-live-SECRETTAIL");
        let excerpt = bounded_excerpt(&big, 400);
        assert!(excerpt.chars().count() <= 400);
        assert!(!excerpt.contains("SECRETTAIL"));
        assert!(!excerpt.contains("sk-live"));
    }

    #[test]
    fn excerpt_limit_is_char_safe_for_multibyte_text() {
        let text = "é".repeat(100);
        let excerpt = bounded_excerpt(&text, 10);
        assert_eq!(excerpt.chars().count(), 10);
    }
}
