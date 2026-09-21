//! Explicit, read-only model catalog (`GET /v1/models`).
//!
//! ROOT-CONTRACT v9: the catalog is fetched ONLY when a caller invokes
//! [`fetch_model_catalog`] on purpose. Nothing in this module is called by decide paths,
//! startup, status, or background work, and nothing here selects a model. The native
//! default stays [`crate::types::DEFAULT_MODEL`] = `jev-latest`.
//!
//! Documented wire shape (docs "Listing models" + `ModelCard`): `GET /v1/models` with
//! `Authorization: Bearer` returns an object whose required `models` field is an array;
//! every entry carries the required string fields `name` ("The model ID or alias, as
//! accepted by the `model` field"), `description` ("What the model is for"), and
//! `release_date` ("When the model or alias was released"). Extra fields are ignored.
//! The list currently names the aliases; versioned IDs are accepted by the `model`
//! field whether or not they appear in the list.
//!
//! Untrusted-input policy: names are IDENTIFIERS, so unsafe names (control characters,
//! bidi controls, ambiguous/invisible whitespace, credential echoes, empty, over-cap)
//! REJECT the entry and safe names are preserved EXACTLY - never silently normalized
//! into a different selectable ID. Prose fields (description/release_date) get
//! presentation sanitizing: control characters are stripped, bidi controls and
//! credential echoes reject, over-long text is capped by the HOST POLICY caps
//! documented below (these are not API guarantees). Unknowns are preserved: rejected
//! entries are disclosed with bounded reasons in [`ModelCatalog::rejected`], never
//! dropped silently. The same identifier rule backs
//! [`validate_requested_model_id`] for the explicit `/jev model set` path.

use serde::Serialize;

use crate::client::{classify_kind, JevLimits};
use crate::credential::SecretString;
use crate::error::{looks_like_credential_echo, sanitize_detail, JevError};
use crate::types::Transport;

/// Documented catalog path, appended to the configured base URL.
pub const MODEL_CATALOG_PATH: &str = "/v1/models";

/// HOST POLICY cap on catalog entries kept. The documented list is tiny; 256 leaves
/// margin. Entries beyond the cap are dropped and disclosed in `ModelCatalog::rejected`.
pub const MAX_MODEL_CATALOG_ENTRIES: usize = 256;

/// HOST POLICY cap on a model name/ID. Over-cap names REJECT the entry and are never
/// truncated: a truncated ID echoed back into a `model` field could select the wrong
/// model.
pub const MAX_MODEL_NAME_CHARS: usize = 64;

/// HOST POLICY cap on a description. Over-cap descriptions are truncated to the cap.
pub const MAX_MODEL_DESCRIPTION_CHARS: usize = 512;

/// HOST POLICY cap on a release-date string. Over-cap dates are truncated to the cap.
pub const MAX_MODEL_DATE_CHARS: usize = 64;

/// One documented catalog entry: `name`, `description`, `release_date` (all required
/// strings per the docs). Unknown fields are ignored; nothing is invented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelCard {
    /// The model ID or alias, as accepted by the `model` field.
    pub name: String,
    /// What the model is for.
    pub description: String,
    /// When the model or alias was released.
    pub release_date: String,
}

/// A parsed, sanitized catalog plus the bounded disclosure of rejected entries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ModelCatalog {
    /// Sanitized entries that passed validation, in wire order.
    pub models: Vec<ModelCard>,
    /// One bounded reason string per rejected entry (or dropped overflow), in wire
    /// order. A silently-dropped entry never exists: unknowns are preserved.
    pub rejected: Vec<String>,
}

/// Bidi/format characters that can silently reorder displayed text. A catalog field
/// carrying any of these REJECTS the entry; the marker is not stripped, because the
/// marker itself is the finding (ROOT-CONTRACT v9: reject bidi surprises before
/// user-visible metadata).
fn has_bidi_control(raw: &str) -> bool {
    raw.chars().any(|c| {
        matches!(
            c as u32,
            0x202A..=0x202E   // LRE, RLE, PDF, LRO, RLR
                | 0x2066..=0x2069 // LRI, RLI, FSI, PDI
                | 0x200E | 0x200F // LRM, RLM
                | 0x061C          // ALM (Arabic letter mark)
                | 0xFEFF          // BOM / zero-width no-break space
        )
    })
}

/// Presentation sanitizer for PROSE fields (description/release_date): strips control
/// characters (including CR/LF header-injection) and surrounding space. NOT used for
/// names - identifiers are validated exactly, never rewritten.
fn clean_field(raw: &str) -> String {
    raw.chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

/// Zero-width / invisible format characters that can make two different names render
/// identically. Bidi controls are handled separately (they can REORDER text).
fn is_invisible_format(c: char) -> bool {
    matches!(c as u32, 0x200B..=0x200D | 0x00AD | 0x2060 | 0xFEFF)
}

/// Core identifier rule shared by catalog parsing and requested-model validation
/// (ROOT-CONTRACT v9 follow-up): remote model NAMES are identifiers, so unsafe content
/// is REJECTED, never silently normalized into a different selectable ID. Returns the
/// bounded rejection reason (which never contains the supplied text), or `None` when
/// the identifier is safe and must be preserved EXACTLY.
///
/// Rejected: bidi controls; control characters (including CR/LF); ANY whitespace
/// (including ordinary spaces, NBSP and tabs); invisible format characters (zero-width
/// space, soft hyphen, word joiner, ...); credential echoes; empty names; names over
/// `cap`. Identifiers are preserved exactly when safe and are never trimmed or normalized.
fn identifier_rejection_reason(text: &str, cap: usize) -> Option<String> {
    if has_bidi_control(text) {
        return Some("name carries a bidi control character".to_string());
    }
    if text.chars().any(|c| c.is_control()) {
        return Some("name carries a control character".to_string());
    }
    if text.chars().any(|c| c.is_whitespace() || is_invisible_format(c)) {
        return Some("name carries whitespace or an invisible character".to_string());
    }
    if looks_like_credential_echo(text) {
        return Some("name carries a credential echo".to_string());
    }
    if text.trim().is_empty() {
        return Some("name is empty".to_string());
    }
    if text.chars().count() > cap {
        return Some(format!(
            "name exceeds {cap} characters (rejected, never truncated)"
        ));
    }
    None
}

/// How one documented field treats emptiness and over-cap values.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldMode {
    /// Required name: an IDENTIFIER, validated exactly by
    /// `identifier_rejection_reason` (unsafe content rejects the entry, safe names are
    /// preserved verbatim, never truncated - see `MAX_MODEL_NAME_CHARS`).
    Name,
    /// Required free text: empty values are kept as sent (preserve unknowns, no
    /// invention); over-cap values are truncated to the cap.
    Free,
}

/// Validates and sanitizes one documented field of one entry. `Err(())` means the ENTRY
/// is rejected; the bounded reason was already pushed to `rejected`.
fn clean_catalog_field(
    index: usize,
    field: &'static str,
    raw: Option<&serde_json::Value>,
    cap: usize,
    mode: FieldMode,
    rejected: &mut Vec<String>,
) -> Result<String, ()> {
    let Some(raw) = raw else {
        rejected.push(format!("entry {index}: {field} is missing"));
        return Err(());
    };
    let Some(text) = raw.as_str() else {
        rejected.push(format!("entry {index}: {field} is not a string"));
        return Err(());
    };
    match mode {
        FieldMode::Name => {
            // Names are IDENTIFIERS: unsafe content rejects the entry and safe names
            // are preserved EXACTLY (no stripping, no trimming, no case changes).
            if let Some(reason) = identifier_rejection_reason(text, cap) {
                rejected.push(format!("entry {index}: {reason}"));
                return Err(());
            }
            Ok(text.to_string())
        }
        FieldMode::Free => {
            // Prose fields get presentation sanitizing: bidi/credential echoes still
            // reject, control characters are stripped, over-cap text is truncated.
            if has_bidi_control(text) {
                rejected.push(format!(
                    "entry {index}: {field} carries a bidi control character"
                ));
                return Err(());
            }
            if looks_like_credential_echo(text) {
                rejected.push(format!("entry {index}: {field} carries a credential echo"));
                return Err(());
            }
            let cleaned = clean_field(text);
            if cleaned.chars().count() > cap {
                return Ok(cleaned.chars().take(cap).collect());
            }
            Ok(cleaned)
        }
    }
}

/// Parses a raw `GET /v1/models` body with the PRODUCTION parser.
///
/// Structural failures (not JSON, not an object, `models` missing or not an array) fail
/// the whole body as `MalformedResponse`. A hostile ENTRY (credential echo, bidi
/// control, missing/non-string required field, empty or over-cap name) is rejected
/// individually with a bounded reason while every other entry still parses.
pub fn parse_model_catalog(bytes: &[u8]) -> Result<ModelCatalog, JevError> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        JevError::MalformedResponse {
            detail: sanitize_detail(&format!(
                "response is not a valid model catalog payload: {}",
                classify_kind(&error)
            )),
        }
    })?;
    let Some(object) = value.as_object() else {
        return Err(JevError::malformed(
            "response is not a valid model catalog payload: not an object",
        ));
    };
    let Some(entries) = object.get("models").and_then(serde_json::Value::as_array) else {
        return Err(JevError::malformed(
            "response is not a valid model catalog payload: `models` is missing or not an array",
        ));
    };
    let mut models = Vec::new();
    let mut rejected: Vec<String> = Vec::new();
    if entries.len() > MAX_MODEL_CATALOG_ENTRIES {
        rejected.push(format!(
            "catalog carries {} entries; entries beyond the host-policy cap of {MAX_MODEL_CATALOG_ENTRIES} are dropped",
            entries.len()
        ));
    }
    for (index, raw) in entries.iter().enumerate().take(MAX_MODEL_CATALOG_ENTRIES) {
        let Some(obj) = raw.as_object() else {
            rejected.push(format!("entry {index}: not an object"));
            continue;
        };
        let name = match clean_catalog_field(
            index,
            "name",
            obj.get("name"),
            MAX_MODEL_NAME_CHARS,
            FieldMode::Name,
            &mut rejected,
        ) {
            Ok(name) => name,
            Err(()) => continue,
        };
        let description = match clean_catalog_field(
            index,
            "description",
            obj.get("description"),
            MAX_MODEL_DESCRIPTION_CHARS,
            FieldMode::Free,
            &mut rejected,
        ) {
            Ok(description) => description,
            Err(()) => continue,
        };
        let release_date = match clean_catalog_field(
            index,
            "release_date",
            obj.get("release_date"),
            MAX_MODEL_DATE_CHARS,
            FieldMode::Free,
            &mut rejected,
        ) {
            Ok(release_date) => release_date,
            Err(()) => continue,
        };
        models.push(ModelCard {
            name,
            description,
            release_date,
        });
    }
    Ok(ModelCatalog { models, rejected })
}

/// Fetches the model catalog through the given transport.
///
/// EXPLICIT-ONLY (ROOT-CONTRACT v9): nothing in pi-jev calls this automatically. No
/// decide path, startup, status, turn, or background task may reach it, and nothing here
/// selects a model or changes the native default.
///
/// Single bounded attempt: no retries and no backoff; a plain failure surfaces as the
/// honest `JevError` it is (kind codes via `JevError::kind()`). The deadline is
/// `limits.timeout` per attempt; callers that need cooperative cancellation wrap this
/// call in their own `tokio::time::timeout` / cancellation select, exactly like the
/// decide path does in the scheduler.
///
/// The credential stays inside the transport. The URL is
/// `JevLimits::models_endpoint()` (`base_url + MODEL_CATALOG_PATH`) and never carries a
/// query string or a key.
pub async fn fetch_model_catalog(
    transport: &dyn Transport,
    limits: &JevLimits,
) -> Result<ModelCatalog, JevError> {
    let bytes = transport.get_models(limits.timeout).await?;
    parse_model_catalog(&bytes)
}

/// Validates a caller-supplied model identifier (the `/jev model set <id>` path,
/// ROOT-CONTRACT v9 follow-up) with the SAME exact-identifier rule the catalog parser
/// uses for server-listed names: unsafe content (control characters, bidi controls,
/// ambiguous/invisible whitespace, credential-echo shapes, empty text, over-cap length)
/// is REFUSED, never normalized into a different selectable ID; a safe identifier is
/// accepted exactly as given.
///
/// The returned reason is bounded and NEVER contains the supplied text, so refusal
/// diagnostics cannot echo a pasted secret.
///
/// CREDENTIAL ECHO PROTECTION: marker-based refusal alone cannot catch a full pasted
/// key. A caller holding the effective credential MUST additionally refuse ids where
/// [`id_overlaps_credential`] is true, surfacing only a generic reason (for example
/// "the supplied id looks like a credential") and never persisting or displaying the
/// value. This function performs no network I/O and no settings write.
pub fn validate_requested_model_id(raw: &str) -> Result<(), String> {
    match identifier_rejection_reason(raw, MAX_MODEL_NAME_CHARS) {
        Some(reason) => Err(reason),
        None => Ok(()),
    }
}

/// Effective-credential echo protection for caller-supplied model ids. True when the id
/// IS the credential, the credential CONTAINS the id (partial paste), or the id CONTAINS
/// the credential (paste with extra characters). Both sides are trimmed first, so a key
/// pasted with surrounding whitespace still matches; the forward-containment check keeps
/// the >=8-character guard so trivial one/two-letter hits do not fire. Callers turn a
/// `true` result into a GENERIC refusal reason - never echo the supplied value into a
/// diagnostic, a setting, or a log line.
pub fn id_overlaps_credential(id: &str, credential: &SecretString) -> bool {
    let secret = credential.expose();
    let trimmed_secret = secret.trim();
    let trimmed_id = id.trim();
    if trimmed_secret.is_empty() || trimmed_id.is_empty() {
        return false;
    }
    if trimmed_id == secret || trimmed_id == trimmed_secret {
        return true;
    }
    if trimmed_id.chars().count() >= 8 && secret.contains(trimmed_id) {
        return true;
    }
    trimmed_id.contains(trimmed_secret)
}
