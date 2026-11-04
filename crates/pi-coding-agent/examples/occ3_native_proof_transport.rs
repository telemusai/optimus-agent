//! Offline Windows component of the OCC-3 proof transport (PHASE 1 ONLY).
//!
//! Scope of this phase: bounded framed-stdin verification and protected
//! local materialization. The binary reads ONE transfer frame from stdin,
//! verifies it against REQUIRED trusted caller inputs (expected length and
//! digest), and materializes the decoded bytes into a FIXED class/name
//! mapping at a validated local journal root through the shared owner-only
//! no-replace Windows publisher (`examples/support/owner_publish.rs`,
//! reused unchanged: staged owner-DACL write, atomic no-replace finalization,
//! existing finals refused, never rewritten).
//!
//! EXPLICITLY OUT OF SCOPE (future launcher/helper, separate gates): SSH
//! transport, network, identity/incarnation checks, current-state gates
//! (journal phase/terminal/persistence/restart predicates), the DGX-side
//! placement helper, the staging receipt, and any orchestration. The
//! expected length/digest are CALLER ASSERTIONS in this offline component;
//! the header is framing, never authorization (the trusted values must match
//! the header AND the decoded bytes: three-way equality).
//!
//! Frame contract (exact): one ASCII header line
//! `N=<canonical unsigned decimal><SP>H=<exactly 64 lowercase hex><LF>`
//! followed by exactly `4*((N+2)/3)` unwrapped base64 characters, then EOF:
//! no payload newline, whitespace, marker or trailing byte of any kind.
//! Header cap 77 bytes (7-digit maximum N); per-class decoded caps are
//! enforced BEFORE the payload is read or allocated; N=0 is syntactically
//! valid with a zero-byte payload and every class in this component refuses
//! it semantically (empty documents are not materializable).
//!
//! Destination discipline: the journal root is a validated absolute
//! drive-letter Windows path (ASCII printable, no traversal components);
//! each class maps to exactly ONE fixed filename component at that root (no
//! caller-supplied names, no subpaths). The staging-path budget (root +
//! separator + staging name <= 230 UTF-16 units, staging name being
//! `.{name}.{32hex}.staging` = name + 42 units) is checked after argument
//! parsing and BEFORE any filesystem open or publication. A digest verifies
//! CONTENT, not destination: no claim is made that hash equality detects
//! misplacement. Bytes are materialized verbatim (no JSON reserialization).
//! Non-Windows targets fail closed.
//!
//! Exit codes: 0 ok; 2 usage; 4 verified refusal (fixed codes); 8 local
//! error. Stdout on success is a sanitized value-free status (class, name,
//! byte count, digest); stderr on refusal carries the fixed code only.

#[path = "support/owner_publish.rs"]
mod owner_publish;

use std::io::{self, Read};
use std::path::Path;

use pi_ai::source_capture::sha256;
use serde_json::json;

/// Header wire cap: `N=` + at most 7 canonical decimal digits + ` ` + `H=`
/// + 64 hex + LF = 2 + 7 + 1 + 2 + 64 + 1 = 77 bytes.
const HEADER_CAP_BYTES: usize = 77;
/// Total staging-path budget in UTF-16 units (conservative Windows bound).
const PATH_BUDGET_UTF16: usize = 230;
/// Staging-name overhead: `.` + name + `.` + 32 hex + `.staging`.
const STAGING_OVERHEAD_UTF16: usize = 42;

/// Machine-generated wire cap: 77 + 4*((n+2)/3) for a decoded cap `n`.
/// Test-support constant: the streaming reader is bounded by construction
/// (header cap, then the exact payload length), so the wire cap is asserted
/// by tests rather than consulted on the hot path.
#[cfg_attr(not(test), allow(dead_code))]
const fn wire_cap(decoded_cap: usize) -> usize {
    HEADER_CAP_BYTES + 4 * ((decoded_cap + 2) / 3)
}

/// The materialization classes of this offline component (downlink inputs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    ChallengeBeforeRestart,
    ChallengeAfterRestart,
    CheckpointInput,
    DescriptorInput,
}

impl Class {
    /// Fixed selector vocabulary; nothing else is accepted.
    fn from_selector(value: &str) -> Option<Class> {
        match value {
            "challenge-before_restart" => Some(Class::ChallengeBeforeRestart),
            "challenge-after_restart" => Some(Class::ChallengeAfterRestart),
            "checkpoint-input" => Some(Class::CheckpointInput),
            "descriptor-input" => Some(Class::DescriptorInput),
            _ => None,
        }
    }

    fn selector(self) -> &'static str {
        match self {
            Class::ChallengeBeforeRestart => "challenge-before_restart",
            Class::ChallengeAfterRestart => "challenge-after_restart",
            Class::CheckpointInput => "checkpoint-input",
            Class::DescriptorInput => "descriptor-input",
        }
    }

    /// Per-class decoded byte cap (source constants of the frozen consumers).
    fn decoded_cap(self) -> usize {
        match self {
            Class::ChallengeBeforeRestart | Class::ChallengeAfterRestart => 4096,
            Class::CheckpointInput => 8192,
            Class::DescriptorInput => 65536,
        }
    }

    /// The ONE fixed destination name per class (suite-expected challenge
    /// names; producer input names for the private documents).
    fn fixed_name(self) -> &'static str {
        match self {
            Class::ChallengeBeforeRestart => "validator-challenge-before_restart.json",
            Class::ChallengeAfterRestart => "validator-challenge-after_restart.json",
            Class::CheckpointInput => "checkpoint-proof-input.json",
            Class::DescriptorInput => "producer-descriptor-input.json",
        }
    }

    /// Explicit per-class semantic policy for N=0: every class in this
    /// component refuses empty documents.
    fn allows_empty(self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Frame parsing (strict, bounded, fail-closed).
// ---------------------------------------------------------------------------

/// Read the header line: bytes until LF, at most HEADER_CAP_BYTES in total
/// (LF included). Oversized or truncated headers refuse.
fn read_header(reader: &mut dyn Read) -> Result<Vec<u8>, &'static str> {
    let mut header: Vec<u8> = Vec::with_capacity(HEADER_CAP_BYTES);
    let mut one = [0u8; 1];
    loop {
        match reader.read(&mut one) {
            Ok(0) => return Err("header_truncated"),
            Ok(_) => {
                if one[0] == b'\n' {
                    header.push(b'\n');
                    if header.len() > HEADER_CAP_BYTES {
                        return Err("header_oversized");
                    }
                    return Ok(header);
                }
                if header.len() >= HEADER_CAP_BYTES {
                    return Err("header_oversized");
                }
                header.push(one[0]);
            }
            Err(_) => return Err("stream_read_failed"),
        }
    }
}

/// Parse the header line into (N, 64-byte lowercase-hex digest field).
fn parse_header(header: &[u8]) -> Result<(usize, [u8; 64]), &'static str> {
    if header.last() != Some(&b'\n') {
        return Err("header_malformed");
    }
    let line = &header[..header.len() - 1];
    if line.len() < 6 || &line[..2] != b"N=" {
        return Err("header_malformed");
    }
    let Some(sp) = line[2..].iter().position(|b| *b == b' ') else {
        return Err("header_malformed");
    };
    let digits = &line[2..2 + sp];
    // Canonical unsigned decimal: non-empty, ASCII digits only, no leading
    // zero unless the value itself is zero, no sign, no grouping, no other
    // characters inside the integer field.
    if digits.is_empty() || !digits.iter().all(|b| b.is_ascii_digit()) {
        return Err("header_integer_noncanonical");
    }
    if digits.len() > 1 && digits[0] == b'0' {
        return Err("header_integer_noncanonical");
    }
    let n: usize = std::str::from_utf8(digits)
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .ok_or("header_integer_noncanonical")?;
    let rest = &line[2 + sp + 1..];
    if rest.len() != 66 || &rest[..2] != b"H=" {
        return Err("header_malformed");
    }
    let mut digest = [0u8; 64];
    digest.copy_from_slice(&rest[2..]);
    // Exactly 64 lowercase hex characters.
    if !digest
        .iter()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
    {
        return Err("header_digest_noncanonical");
    }
    Ok((n, digest))
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_value(byte: u8) -> Option<u8> {
    BASE64_ALPHABET.iter().position(|b| *b == byte).map(|p| p as u8)
}

/// Strict canonical base64 decode: length must be a multiple of 4, the
/// standard alphabet plus trailing '=' padding only in the final group, the
/// pad count must match the decoded remainder, and the dropped low bits of
/// the final real characters must be zero (canonical form). The empty
/// input decodes to the empty byte vector. No whitespace is accepted.
fn decode_base64_strict(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() {
        return Some(Vec::new());
    }
    if input.len() % 4 != 0 {
        return None;
    }
    let groups = input.len() / 4;
    let mut out = Vec::with_capacity(groups * 3);
    for (group, chunk) in input.chunks_exact(4).enumerate() {
        let last = group == groups - 1;
        let pad = chunk.iter().rev().take_while(|b| **b == b'=').count();
        if pad > 2 {
            return None;
        }
        if !last && pad != 0 {
            return None;
        }
        if chunk[..4 - pad].iter().any(|b| *b == b'=') {
            return None;
        }
        let mut vals = [0u8; 4];
        for i in 0..4 - pad {
            vals[i] = base64_value(chunk[i])?;
        }
        if last {
            // Canonical padding: the dropped low bits of the final real
            // character(s) must be zero.
            if pad == 1 && (vals[2] & 0x03) != 0 {
                return None;
            }
            if pad == 2 && (vals[1] & 0x0f) != 0 {
                return None;
            }
        }
        let triple = ((vals[0] as u32) << 18)
            | ((vals[1] as u32) << 12)
            | ((vals[2] as u32) << 6)
            | vals[3] as u32;
        let bytes = [(triple >> 16) as u8, (triple >> 8) as u8, triple as u8];
        let keep = match pad {
            0 => 3,
            1 => 2,
            _ => 1,
        };
        out.extend_from_slice(&bytes[..keep]);
    }
    Some(out)
}

/// Read and verify exactly one transfer frame from `reader`.
///
/// Verification order (each step fail-closed with a fixed code): header
/// read/parse; class-cap check BEFORE the payload is read or allocated;
/// exact payload length; EOF immediately after the payload; strict base64
/// decode; three-way trust comparison (header N vs expected length, header
/// H vs expected digest, decoded sha256 vs header H); per-class empty
/// policy.
fn read_frame(
    reader: &mut dyn Read,
    class: Class,
    expected_len: usize,
    expected_sha256: &str,
) -> Result<Vec<u8>, &'static str> {
    let header = read_header(reader)?;
    let (n, digest) = parse_header(&header)?;
    // Class-cap violation refuses BEFORE the payload is read or allocated.
    if n > class.decoded_cap() {
        return Err("class_cap_exceeded");
    }
    let payload_len = 4 * ((n + 2) / 3);
    let mut payload = vec![0u8; payload_len];
    let mut filled = 0usize;
    while filled < payload_len {
        match reader.read(&mut payload[filled..]) {
            Ok(0) => return Err("payload_truncated"),
            Ok(k) => filled += k,
            Err(_) => return Err("stream_read_failed"),
        }
    }
    // EOF must follow the payload immediately: any extra byte refuses.
    let mut one = [0u8; 1];
    match reader.read(&mut one) {
        Ok(0) => {}
        Ok(_) => return Err("payload_trailing_data"),
        Err(_) => return Err("stream_read_failed"),
    }
    let decoded = decode_base64_strict(&payload).ok_or("base64_malformed")?;
    if decoded.len() != n {
        return Err("base64_malformed");
    }
    if n != expected_len {
        return Err("trusted_length_mismatch");
    }
    if &digest[..] != expected_sha256.as_bytes() {
        return Err("trusted_digest_mismatch");
    }
    if sha256(&decoded) != expected_sha256 {
        return Err("content_digest_mismatch");
    }
    if n == 0 && !class.allows_empty() {
        return Err("empty_payload_refused");
    }
    Ok(decoded)
}

// ---------------------------------------------------------------------------
// Destination grammar and budget.
// ---------------------------------------------------------------------------

/// Validated absolute drive-letter Windows path: ASCII printable only
/// (non-ASCII refuses), exactly one root, no traversal, self or empty
/// interior components, no trailing separator.
fn validate_journal(journal: &str) -> Result<(), &'static str> {
    let bytes = journal.as_bytes();
    if bytes.len() < 4 {
        return Err("destination_invalid");
    }
    if !bytes[0].is_ascii_alphabetic() || bytes[1] != b':' {
        return Err("destination_invalid");
    }
    if bytes[2] != b'\\' && bytes[2] != b'/' {
        return Err("destination_invalid");
    }
    if !bytes.iter().all(|b| (0x20..=0x7e).contains(b)) {
        return Err("destination_invalid");
    }
    for component in journal[3..].split(['\\', '/']) {
        if component.is_empty() || component == ".." || component == "." {
            return Err("destination_invalid");
        }
    }
    Ok(())
}

/// Staging-path budget in UTF-16 units: root + separator + (name + 42).
fn path_budget_ok(journal: &str, name: &str) -> bool {
    journal.encode_utf16().count() + 1 + name.encode_utf16().count() + STAGING_OVERHEAD_UTF16
        <= PATH_BUDGET_UTF16
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

struct Args {
    journal: String,
    class: Class,
    expected_len: usize,
    expected_sha256: String,
}

fn usage() -> (i32, &'static str) {
    (2i32, "usage")
}

fn parse_args(args: &[String]) -> Result<Args, (i32, &'static str)> {
    let mut journal: Option<String> = None;
    let mut class: Option<String> = None;
    let mut expected_len: Option<String> = None;
    let mut expected_sha256: Option<String> = None;
    let mut index = 0usize;
    while index < args.len() {
        let slot = match args[index].as_str() {
            "--journal" => &mut journal,
            "--class" => &mut class,
            "--expected-len" => &mut expected_len,
            "--expected-sha256" => &mut expected_sha256,
            _ => return Err(usage()),
        };
        let Some(value) = args.get(index + 1) else {
            return Err(usage());
        };
        if slot.replace(value.clone()).is_some() {
            return Err(usage());
        }
        index += 2;
    }
    let (Some(journal), Some(class), Some(expected_len), Some(expected_sha256)) =
        (journal, class, expected_len, expected_sha256)
    else {
        return Err(usage());
    };
    let class = Class::from_selector(&class).ok_or((2i32, "usage"))?;
    // The expected length is a canonical decimal.
    if expected_len.is_empty()
        || !expected_len.bytes().all(|b| b.is_ascii_digit())
        || (expected_len.len() > 1 && expected_len.starts_with('0'))
    {
        return Err(usage());
    }
    let expected_len: usize = expected_len.parse().map_err(|_| usage())?;
    // The expected digest is exactly 64 lowercase hex characters.
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(usage());
    }
    Ok(Args {
        journal,
        class,
        expected_len,
        expected_sha256,
    })
}

/// Everything after argument parsing: destination grammar, path budget
/// (both pure checks before any filesystem access), journal presence, the
/// bounded stdin frame read, verification, and protected no-replace
/// materialization through the shared primitive (existing finals are
/// refused by the primitive, never rewritten; a failed publication leaves
/// no staging residue).
fn execute(args: &Args, reader: &mut dyn Read) -> Result<serde_json::Value, (i32, String)> {
    validate_journal(&args.journal).map_err(|text| (4i32, text.to_string()))?;
    if !path_budget_ok(&args.journal, args.class.fixed_name()) {
        return Err((4i32, "path_budget_exceeded".to_string()));
    }
    let journal_path = Path::new(&args.journal);
    if !journal_path.is_dir() {
        return Err((8i32, "journal_unavailable".to_string()));
    }
    let bytes = read_frame(reader, args.class, args.expected_len, &args.expected_sha256)
        .map_err(|text| (4i32, text.to_string()))?;
    owner_publish::publish_exclusive(journal_path, args.class.fixed_name(), &bytes)
        .map_err(|text| (4i32, text))?;
    Ok(json!({
        "ok": true,
        "component": "occ3_native_proof_transport_phase1",
        "class": args.class.selector(),
        "name": args.class.fixed_name(),
        "bytes": bytes.len(),
        "sha256": sha256(&bytes),
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match parse_args(&args) {
        Ok(parsed) => parsed,
        Err((code, text)) => {
            eprintln!("proof_transport_refused: {text}");
            std::process::exit(code);
        }
    };
    match execute(&parsed, &mut io::stdin().lock()) {
        Ok(status) => {
            println!("{}", serde_json::to_string(&status).unwrap_or_default());
        }
        Err((code, text)) => {
            eprintln!("proof_transport_refused: {text}");
            std::process::exit(code);
        }
    }
}

// ---------------------------------------------------------------------------
// Embedded offline tests: no network, no fixtures beyond temporary local
// directories; file-system tests are Windows-only (the publication
// primitive is Windows-only; non-Windows fails closed by construction).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Deterministic test bytes: a fixed byte pattern, no randomness.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn encode_base64(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 * ((input.len() + 2) / 3));
        for chunk in input.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(BASE64_ALPHABET[((triple >> 18) & 63) as usize]);
            out.push(BASE64_ALPHABET[((triple >> 12) & 63) as usize]);
            out.push(BASE64_ALPHABET[((triple >> 6) & 63) as usize]);
            out.push(BASE64_ALPHABET[(triple & 63) as usize]);
        }
        match input.len() % 3 {
            1 => {
                out.truncate(out.len() - 2);
                out.push(b'=');
                out.push(b'=');
            }
            2 => {
                out.truncate(out.len() - 1);
                out.push(b'=');
            }
            _ => {}
        }
        out
    }

    /// Build one canonical frame for `bytes`.
    fn build_frame(bytes: &[u8]) -> Vec<u8> {
        let mut frame = format!("N={} H={}\n", bytes.len(), sha256(bytes)).into_bytes();
        frame.extend_from_slice(&encode_base64(bytes));
        frame
    }

    /// A reader that serves exactly the header bytes and then errors on any
    /// further read: proves the class-cap check happens BEFORE the payload
    /// is read.
    struct HeaderThenTrapReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl Read for HeaderThenTrapReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.data.len() {
                return Err(io::Error::new(io::ErrorKind::Other, "payload_read_attempt"));
            }
            let n = buf.len().min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    fn frame_of(bytes: &[u8]) -> Cursor<Vec<u8>> {
        Cursor::new(build_frame(bytes))
    }

    const CH: Class = Class::ChallengeBeforeRestart;

    #[test]
    fn frame_round_trip_exact() {
        for len in [1usize, 2, 3, 4, 57, 58, 59, 100, 4095, 4096] {
            let bytes = pattern(len);
            let digest = sha256(&bytes);
            let mut cursor = frame_of(&bytes);
            let decoded = read_frame(&mut cursor, CH, len, &digest).unwrap();
            assert_eq!(decoded, bytes, "round trip failed at len {len}");
        }
    }

    #[test]
    fn cap_table_is_machine_generated() {
        assert_eq!(wire_cap(4096), 5541);
        assert_eq!(wire_cap(8192), 11001);
        assert_eq!(wire_cap(65536), 87461);
        assert_eq!(wire_cap(262144), 349605);
        assert_eq!(wire_cap(2 * 1024 * 1024), 2796281);
        assert_eq!(CH.decoded_cap(), 4096);
        assert_eq!(Class::CheckpointInput.decoded_cap(), 8192);
        assert_eq!(Class::DescriptorInput.decoded_cap(), 65536);
    }

    #[test]
    fn class_cap_boundary_accepts_cap_and_refuses_overflow_before_read() {
        // At the cap: accepted.
        let bytes = pattern(CH.decoded_cap());
        let digest = sha256(&bytes);
        let mut cursor = frame_of(&bytes);
        assert!(read_frame(&mut cursor, CH, bytes.len(), &digest).is_ok());
        // One over the cap: refused BEFORE the payload is read. The trap
        // reader errors if any payload byte is requested; the refusal code
        // proves the payload was never touched.
        let over = CH.decoded_cap() + 1;
        let header = format!("N={over} H={}\n", sha256(&pattern(over)));
        let mut trap = HeaderThenTrapReader {
            data: header.into_bytes(),
            pos: 0,
        };
        assert_eq!(
            read_frame(&mut trap, CH, over, &sha256(&pattern(over))),
            Err("class_cap_exceeded")
        );
    }

    #[test]
    fn oversized_header_refused() {
        // 78 bytes of header without LF.
        let long = format!("N=9999999 H={}\rx", "a".repeat(64));
        let mut cursor = Cursor::new(long.into_bytes());
        assert_eq!(read_frame(&mut cursor, CH, 7, "a".repeat(64).as_str()), Err("header_oversized"));
    }

    #[test]
    fn eight_digit_n_refused_by_header_cap() {
        // A canonical 8-digit N makes a 78-byte header: refused as oversized
        // before the value is even considered.
        let header = format!("N=10000000 H={}\n", "0".repeat(64));
        let mut cursor = Cursor::new(header.into_bytes());
        assert_eq!(
            read_frame(&mut cursor, CH, 10000000, &sha256(&pattern(1))),
            Err("header_oversized")
        );
    }

    #[test]
    fn malformed_integers_refused() {
        for bad in ["N=07 H=", "N=+7 H=", "N= H=", "N=7a H="] {
            let header = format!("{bad}{}\n", "0".repeat(64));
            let mut cursor = Cursor::new(header.into_bytes());
            let result = read_frame(&mut cursor, CH, 7, &sha256(&pattern(7)));
            assert!(matches!(
                result,
                Err("header_integer_noncanonical") | Err("header_malformed")
            ));
        }
    }

    #[test]
    fn malformed_hex_refused() {
        // Uppercase digest.
        let header = format!("N=7 H={}\n", "A".repeat(64));
        let mut cursor = Cursor::new(header.into_bytes());
        assert_eq!(
            read_frame(&mut cursor, CH, 7, &sha256(&pattern(7))),
            Err("header_digest_noncanonical")
        );
        // 63 characters.
        let header = format!("N=7 H={}\n", "a".repeat(63));
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&pattern(7))).is_err());
        // 65 characters.
        let header = format!("N=7 H={}\n", "a".repeat(65));
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&pattern(7))).is_err());
        // Non-hex character.
        let header = format!("N=7 H={}g{}\n", "a".repeat(63), "");
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&pattern(7))).is_err());
    }

    #[test]
    fn noncanonical_whitespace_refused() {
        // CR before LF.
        let bytes = pattern(7);
        let mut frame = build_frame(&bytes);
        frame.insert(frame.len() - 1, b'\r');
        let mut cursor = Cursor::new(frame);
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&bytes)).is_err());
        // Double space.
        let header = format!("N=7  H={}\n", "0".repeat(64));
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&bytes)).is_err());
        // Tab instead of space.
        let header = format!("N=7\tH={}\n", "0".repeat(64));
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&bytes)).is_err());
    }

    #[test]
    fn non_utf8_header_bytes_refused() {
        let mut header = format!("N=7 H={}\n", "0".repeat(64)).into_bytes();
        header[5] = 0xff;
        let mut cursor = Cursor::new(header);
        assert!(read_frame(&mut cursor, CH, 7, &sha256(&pattern(7))).is_err());
    }

    #[test]
    fn malformed_base64_refused() {
        let bytes = pattern(57);
        let digest = sha256(&bytes);
        let mut frame = build_frame(&bytes);
        // Corrupt one payload character into a non-alphabet byte.
        let header_len = frame.iter().position(|b| *b == b'\n').unwrap() + 1;
        frame[header_len] = b'-';
        let mut cursor = Cursor::new(frame);
        assert_eq!(
            read_frame(&mut cursor, CH, bytes.len(), &digest),
            Err("base64_malformed")
        );
        // Noncanonical trailing bits: "QR==" decodes to the same byte as
        // "QQ==" but the second character carries non-zero dropped bits,
        // which canonical base64 never produces.
        let header = "N=1 H=".to_string() + &sha256(b"\x41") + "\n";
        let mut frame = header.into_bytes();
        frame.extend_from_slice(b"QR==");
        let mut cursor = Cursor::new(frame);
        assert_eq!(
            read_frame(&mut cursor, CH, 1, &sha256(b"\x41")),
            Err("base64_malformed")
        );
    }

    #[test]
    fn wrong_trusted_values_refused() {
        let bytes = pattern(100);
        let mut cursor = frame_of(&bytes);
        assert_eq!(
            read_frame(&mut cursor, CH, 101, &sha256(&bytes)),
            Err("trusted_length_mismatch")
        );
        let mut cursor = frame_of(&bytes);
        assert_eq!(
            read_frame(&mut cursor, CH, 100, &sha256(&pattern(101))),
            Err("trusted_digest_mismatch")
        );
        // Payload tampered while the header claims the true digest. The
        // first encoded character of pattern(100) is 'A'; the mutation to
        // the valid-alphabet 'B' genuinely changes the decoded bytes (the
        // assert below proves the original byte and the mutation differ),
        // so the content digest check fires. The trusted length and the
        // expected/header digests stay the ORIGINAL values.
        let mut frame = build_frame(&bytes);
        let header_len = frame.iter().position(|b| *b == b'\n').unwrap() + 1;
        assert_eq!(frame[header_len], b'A');
        frame[header_len] = b'B';
        assert_ne!(frame[header_len], b'A');
        let mut cursor = Cursor::new(frame);
        assert_eq!(
            read_frame(&mut cursor, CH, 100, &sha256(&bytes)),
            Err("content_digest_mismatch")
        );
    }

    #[test]
    fn truncation_and_trailing_data_refused() {
        let bytes = pattern(50);
        let digest = sha256(&bytes);
        // Header only.
        let frame = build_frame(&bytes);
        let header = &frame[..frame.iter().position(|b| *b == b'\n').unwrap() + 1];
        let mut cursor = Cursor::new(header.to_vec());
        assert_eq!(
            read_frame(&mut cursor, CH, 50, &digest),
            Err("payload_truncated")
        );
        // Short payload.
        let mut short = frame.clone();
        short.truncate(frame.len() - 1);
        let mut cursor = Cursor::new(short);
        assert_eq!(
            read_frame(&mut cursor, CH, 50, &digest),
            Err("payload_truncated")
        );
        // Extra byte after the payload.
        let mut padded = frame.clone();
        padded.push(b'X');
        let mut cursor = Cursor::new(padded);
        assert_eq!(
            read_frame(&mut cursor, CH, 50, &digest),
            Err("payload_trailing_data")
        );
    }

    #[test]
    fn n_zero_semantics() {
        // N=0 is syntactically valid: header + zero payload chars + EOF.
        let frame = format!("N=0 H={}\n", sha256(&[]));
        let mut cursor = Cursor::new(frame.into_bytes());
        assert_eq!(
            read_frame(&mut cursor, CH, 0, &sha256(&[])),
            Err("empty_payload_refused")
        );
        // The same frame against a non-zero trusted length: length mismatch.
        let frame = format!("N=0 H={}\n", sha256(&[]));
        let mut cursor = Cursor::new(frame.into_bytes());
        assert_eq!(
            read_frame(&mut cursor, CH, 5, &sha256(&[])),
            Err("trusted_length_mismatch")
        );
    }

    #[test]
    fn destination_grammar_refuses_invalid_paths() {
        assert!(validate_journal("C:\\data\\journal").is_ok());
        assert!(validate_journal("C:/data/journal").is_ok());
        for bad in [
            "C:\\data\\..\\journal",
            "data\\journal",
            "\\data\\journal",
            "C:data\\journal",
            "C:",
            "C:\\",
            "C:\\data\\journal\\",
            "C:\\data\\.",
            "C:\\data\\journal\u{00fc}",
            "",
            "D:\\data\tjournal",
        ] {
            assert!(validate_journal(bad).is_err(), "must refuse {bad:?}");
        }
    }

    fn args_for(journal: &str, class: Class, len: usize, digest: &str) -> Args {
        Args {
            journal: journal.to_string(),
            class,
            expected_len: len,
            expected_sha256: digest.to_string(),
        }
    }

    #[test]
    fn path_budget_enforced_before_open() {
        // A journal long enough to violate the staging budget for the
        // descriptor class: refused as path_budget_exceeded even though the
        // directory does not exist (the budget check precedes the is_dir
        // open).
        let long_root = format!("C:\\{}{}", "j".repeat(200), "");
        let args = args_for(&long_root, Class::DescriptorInput, 1, &sha256(&pattern(1)));
        let mut cursor = Cursor::new(Vec::new());
        match execute(&args, &mut cursor) {
            Err((4, text)) => assert_eq!(text, "path_budget_exceeded"),
            other => panic!("expected budget refusal, got {other:?}"),
        }
    }

    #[cfg(windows)]
    fn temp_journal(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "occ3-transport-test-{}-{}",
            tag,
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(windows)]
    fn dir_entries(dir: &std::path::Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    #[cfg(windows)]
    #[test]
    fn execute_publishes_fresh_file_and_reports_status() {
        let journal = temp_journal("fresh");
        let bytes = pattern(300);
        let digest = sha256(&bytes);
        let journal_str = journal.to_string_lossy().into_owned();
        let args = args_for(&journal_str, Class::CheckpointInput, bytes.len(), &digest);
        let mut cursor = frame_of(&bytes);
        let status = execute(&args, &mut cursor).expect("publish");
        assert_eq!(status["ok"], json!(true));
        assert_eq!(status["class"], json!("checkpoint-input"));
        assert_eq!(status["bytes"], json!(bytes.len()));
        assert_eq!(status["sha256"], json!(digest));
        let read = owner_publish::read_file_strict(
            &journal.join(Class::CheckpointInput.fixed_name()),
            8192,
        )
        .unwrap();
        assert_eq!(read, bytes);
        std::fs::remove_dir_all(&journal).ok();
    }

    #[cfg(windows)]
    #[test]
    fn execute_refuses_existing_final_without_rewriting() {
        let journal = temp_journal("existing");
        let first = pattern(300);
        let journal_str = journal.to_string_lossy().into_owned();
        let args = args_for(
            &journal_str,
            Class::CheckpointInput,
            first.len(),
            &sha256(&first),
        );
        let mut cursor = frame_of(&first);
        execute(&args, &mut cursor).expect("first publish");
        // The SAME bytes: the primitive refuses existing finals; it does
        // not provide byte-idempotency.
        let mut cursor = frame_of(&first);
        match execute(&args, &mut cursor) {
            Err((4, text)) => assert_eq!(text, "publish_refused_no_replay"),
            other => panic!("expected no-replay refusal, got {other:?}"),
        }
        // DIFFERENT bytes: also refused.
        let second = pattern(301);
        let args_two = args_for(
            &journal_str,
            Class::CheckpointInput,
            second.len(),
            &sha256(&second),
        );
        let mut cursor = frame_of(&second);
        match execute(&args_two, &mut cursor) {
            Err((4, text)) => assert_eq!(text, "publish_refused_no_replay"),
            other => panic!("expected no-replay refusal, got {other:?}"),
        }
        // The prior bytes are unchanged.
        let read = owner_publish::read_file_strict(
            &journal.join(Class::CheckpointInput.fixed_name()),
            8192,
        )
        .unwrap();
        assert_eq!(read, first);
        std::fs::remove_dir_all(&journal).ok();
    }

    #[cfg(windows)]
    #[test]
    fn refused_republication_leaves_private_ineligible_staging_residue() {
        // The shared primitive's documented contract: on the no-replay
        // refusal (existing final) the staging file is DELIBERATELY kept
        // ("stays private and ineligible" - preserve-on-uncertainty; a
        // post-failure unlink could delete a name a foreign actor
        // recreated). The residue is NOT a final: existence/no-replace
        // refusal never publishes it. No deletion, retry, finalization or
        // cleanup happens anywhere here.
        let journal = temp_journal("residue");
        let first = pattern(50);
        let journal_str = journal.to_string_lossy().into_owned();
        let args = args_for(
            &journal_str,
            Class::ChallengeBeforeRestart,
            first.len(),
            &sha256(&first),
        );
        let mut cursor = frame_of(&first);
        execute(&args, &mut cursor).expect("first publish");
        // ONE refused second publish with the same bytes.
        let mut cursor = frame_of(&first);
        let refused = execute(&args, &mut cursor);
        let Err((4, text)) = refused else {
            panic!("expected no-replay refusal, got {refused:?}");
        };
        assert_eq!(text, "publish_refused_no_replay");
        // EXACTLY two directory entries: the unchanged final and the one
        // staging residue.
        let mut entries = dir_entries(&journal);
        entries.sort();
        assert_eq!(entries.len(), 2, "entries: {entries:?}");
        let final_name = Class::ChallengeBeforeRestart.fixed_name();
        // (a) the final exists exactly once, bytes unchanged, read through
        // the strict reader (bounded, reparse-refusing discipline).
        assert!(entries.iter().filter(|e| e.as_str() == final_name).count() == 1);
        assert_eq!(
            owner_publish::read_file_strict(&journal.join(final_name), 8192).unwrap(),
            first
        );
        // (b) EXACTLY one residue matching the EXACT staging grammar
        // `.{final}.{32 lowercase hex}.staging` - no broad prefix/suffix
        // acceptance.
        let residues: Vec<&String> = entries
            .iter()
            .filter(|e| e.as_str() != final_name)
            .collect();
        assert_eq!(residues.len(), 1);
        let residue = residues[0];
        let rest = residue
            .strip_prefix(&format!(".{final_name}."))
            .and_then(|r| r.strip_suffix(".staging"));
        let hex = rest.unwrap_or_else(|| panic!("non-grammar residue name: {residue}"));
        assert_eq!(hex.len(), 32);
        assert!(hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        // (d) the residue bytes equal the published bytes (same strict
        // reader; the residue is a regular owner-only staged file).
        assert_eq!(
            owner_publish::read_file_strict(&journal.join(residue), 8192).unwrap(),
            first
        );
        // (c) no other files of any kind: enforced by the exact count and
        // the two matched names above.
        std::fs::remove_dir_all(&journal).ok();
    }

    #[cfg(windows)]
    #[test]
    fn validation_failures_have_no_side_effects() {
        let journal = temp_journal("nosideeffects");
        let bytes = pattern(200);
        let digest = sha256(&bytes);
        let journal_str = journal.to_string_lossy().into_owned();
        // Wrong trusted length: no file.
        let args = args_for(&journal_str, CH, 999, &digest);
        let mut cursor = frame_of(&bytes);
        assert!(execute(&args, &mut cursor).is_err());
        assert!(dir_entries(&journal).is_empty());
        // Wrong trusted digest: no file.
        let args = args_for(&journal_str, CH, bytes.len(), &sha256(&pattern(1)));
        let mut cursor = frame_of(&bytes);
        assert!(execute(&args, &mut cursor).is_err());
        assert!(dir_entries(&journal).is_empty());
        // Corrupted payload (mid-stream '=' is never valid base64): no file.
        let mut corrupted = build_frame(&bytes);
        let header_len = corrupted.iter().position(|b| *b == b'\n').unwrap() + 1;
        corrupted[header_len + 10] = b'=';
        let args = args_for(&journal_str, CH, bytes.len(), &digest);
        let mut cursor = Cursor::new(corrupted);
        assert!(execute(&args, &mut cursor).is_err());
        assert!(dir_entries(&journal).is_empty());
        // Class-cap overflow: no file.
        let over = CH.decoded_cap() + 1;
        let header = format!("N={over} H={}\n", sha256(&pattern(over)));
        let args = args_for(&journal_str, CH, over, &sha256(&pattern(over)));
        let mut cursor = Cursor::new(header.into_bytes());
        assert!(execute(&args, &mut cursor).is_err());
        assert!(dir_entries(&journal).is_empty());
        std::fs::remove_dir_all(&journal).ok();
    }
}
