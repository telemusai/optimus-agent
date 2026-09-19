//! Credential storage (DESIGN.md section 6).
//!
//! Production store: Windows DPAPI (`CryptProtectData`/`CryptUnprotectData`), encrypted file
//! under the agent dir, current-user scope (`CRYPTPROTECT_UI_FORBIDDEN` only, never
//! `CRYPTPROTECT_LOCAL_MACHINE`). The store FAILS CLOSED when DPAPI is unavailable: it returns
//! `JevError::Unavailable` and never falls back to plaintext.
//!
//! Test store: `InMemoryCredentialStore` with synthetic keys.
//! Keys never appear in argv, in logs, in errors, or in the settings file.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use base64::Engine as _;
use parking_lot::Mutex;

use crate::error::{sanitize_detail, JevError};

/// Maximum accepted key length. Bounded so a paste-bomb cannot reach the store.
pub const MAX_SECRET_LEN: usize = 4096;

/// Wrapper that keeps a secret out of `Debug` output and out of logs.
///
/// `Display`/`Debug` print a redaction marker, so a `SecretString` can never leak through
/// `{:?}` formatting, `unwrap()` panic messages or tracing macros.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wraps a secret.
    pub fn new(secret: impl Into<String>) -> Self {
        SecretString(secret.into())
    }

    /// Borrows the secret. Call sites must not format the result.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Length of the secret (safe metadata for status output).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when the secret is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// A `Bearer` header value. The result carries the key; never log it.
    pub fn bearer_header(&self) -> String {
        format!("Bearer {}", self.0)
    }

    /// One-way fingerprint of the key: first 8 hex chars of its sha256.
    ///
    /// Used to notice credential version changes without storing or printing the key.
    pub fn key_fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(self.0.as_bytes());
        let hex = format!("{digest:x}");
        hex[..8].to_string()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Key ids are opaque file names; keep them strict so a key id cannot escape the agent dir.
pub fn validate_key_id(key_id: &str) -> Result<(), JevError> {
    let valid = !key_id.is_empty()
        && key_id.len() <= 64
        && key_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        Ok(())
    } else {
        Err(JevError::InvalidKeyId)
    }
}

/// Cancellation competes with the final atomic commit, never with encryption.
#[derive(Debug, Default)]
pub struct CredentialWriteGate(AtomicU8);

impl CredentialWriteGate {
    /// True proves that no later credential commit can start.
    pub fn cancel(&self) -> bool {
        self.0.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok()
            || self.0.load(Ordering::Acquire) == 1
    }

    fn commit(&self, write: impl FnOnce() -> Result<(), JevError>) -> Result<(), JevError> {
        self.0.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| JevError::Cancelled)?;
        write()
    }
}

/// Storage abstraction shared by the UI lane and the client lane.
pub trait CredentialStore: Send + Sync {
    fn store(&self, key_id: &str, secret: &str) -> Result<(), JevError>;
    fn store_cancellable(&self, key_id: &str, secret: &str, gate: &CredentialWriteGate) -> Result<(), JevError> {
        gate.commit(|| self.store(key_id, secret))
    }
    fn get(&self, key_id: &str) -> Result<Option<String>, JevError>;
    fn delete(&self, key_id: &str) -> Result<(), JevError>;
    fn exists(&self, key_id: &str) -> Result<bool, JevError>;

    /// Availability of the backing store. A store that is not available must never be
    /// asked to persist a plaintext fallback.
    fn is_available(&self) -> bool {
        true
    }

    /// Human-readable store name for status output. Contains no secret.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }
}

/// In-memory store for tests and for an explicitly isolated in-process run.
///
/// Not a production fallback: it keeps the secret in process memory only and writes nothing.
#[derive(Debug, Default)]
pub struct InMemoryCredentialStore {
    entries: Mutex<BTreeMap<String, String>>,
}

impl InMemoryCredentialStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
        }
    }

    /// Number of stored entries (metadata only).
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// True when nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
}

impl CredentialStore for InMemoryCredentialStore {
    fn store(&self, key_id: &str, secret: &str) -> Result<(), JevError> {
        validate_key_id(key_id)?;
        if secret.trim().is_empty() {
            return Err(JevError::CredentialStore {
                detail: "refusing to store an empty secret".to_string(),
            });
        }
        if secret.len() > MAX_SECRET_LEN {
            return Err(JevError::CredentialStore {
                detail: format!("secret length {} exceeds limit {MAX_SECRET_LEN}", secret.len()),
            });
        }
        self.entries.lock().insert(key_id.to_string(), secret.to_string());
        Ok(())
    }

    fn get(&self, key_id: &str) -> Result<Option<String>, JevError> {
        validate_key_id(key_id)?;
        Ok(self.entries.lock().get(key_id).cloned())
    }

    fn delete(&self, key_id: &str) -> Result<(), JevError> {
        validate_key_id(key_id)?;
        self.entries.lock().remove(key_id);
        Ok(())
    }

    fn exists(&self, key_id: &str) -> Result<bool, JevError> {
        validate_key_id(key_id)?;
        Ok(self.entries.lock().contains_key(key_id))
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }
}

/// Store that always reports unavailable. Used when the platform cannot provide DPAPI,
/// and in tests that assert the fail-closed path. Never stores anything.
#[derive(Debug, Default)]
pub struct UnavailableCredentialStore {
    reason: String,
}

impl UnavailableCredentialStore {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: sanitize_detail(&reason.into()),
        }
    }

    /// The sanitized reason.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl CredentialStore for UnavailableCredentialStore {
    fn store(&self, _key_id: &str, _secret: &str) -> Result<(), JevError> {
        Err(JevError::Unavailable {
            reason: self.reason.clone(),
        })
    }

    fn get(&self, _key_id: &str) -> Result<Option<String>, JevError> {
        Err(JevError::Unavailable {
            reason: self.reason.clone(),
        })
    }

    fn delete(&self, _key_id: &str) -> Result<(), JevError> {
        Err(JevError::Unavailable {
            reason: self.reason.clone(),
        })
    }

    fn exists(&self, _key_id: &str) -> Result<bool, JevError> {
        // Presence is reported as `false` rather than an error so status stays renderable.
        Ok(false)
    }

    fn is_available(&self) -> bool {
        false
    }

    fn backend_name(&self) -> &'static str {
        "unavailable"
    }
}

/// File name (inside the agent dir) of the DPAPI-protected credential envelope.
pub const CREDENTIAL_FILE_NAME: &str = "jev-credential.json";

/// Envelope schema version. Bumping it invalidates old blobs instead of misreading them.
pub const CREDENTIAL_ENVELOPE_VERSION: u32 = 1;

/// On-disk envelope: ciphertext only, never the key itself.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CredentialEnvelope {
    version: u32,
    backend: String,
    scope: String,
    /// Base64 of the DPAPI ciphertext. Storing text keeps the file safe to scan and diff.
    blob: String,
}

impl CredentialEnvelope {
    fn wrap(ciphertext: &[u8]) -> Self {
        Self {
            version: CREDENTIAL_ENVELOPE_VERSION,
            backend: "dpapi".to_string(),
            scope: "current_user".to_string(),
            blob: base64::engine::general_purpose::STANDARD.encode(ciphertext),
        }
    }

    fn ciphertext(&self) -> Result<Vec<u8>, JevError> {
        if self.version != CREDENTIAL_ENVELOPE_VERSION {
            return Err(JevError::CredentialStore {
                detail: format!(
                    "credential envelope version {} is not supported (expected {CREDENTIAL_ENVELOPE_VERSION})",
                    self.version
                ),
            });
        }
        base64::engine::general_purpose::STANDARD
            .decode(self.blob.as_bytes())
            .map_err(|_| JevError::CredentialStore {
                detail: "credential envelope is not valid base64".to_string(),
            })
    }
}

/// Windows DPAPI store: ciphertext only, current-user scope, atomic replace.
///
/// Fail-closed rules:
/// - If DPAPI is missing or the call fails, operations return `Unavailable`/`CredentialStore`.
/// - The file is written through a temp file plus rename, so no partial blob is ever visible.
/// - No plaintext is ever written, and no key is passed in argv.
#[derive(Debug)]
pub struct DpapiCredentialStore {
    dir: PathBuf,
}

impl DpapiCredentialStore {
    /// Creates a store rooted at `dir` (typically `<agent_dir>/jev`).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Root directory of the store.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, key_id: &str) -> Result<PathBuf, JevError> {
        validate_key_id(key_id)?;
        Ok(self.dir.join(format!("{key_id}.{CREDENTIAL_FILE_NAME}")))
    }

    fn ensure_dir(&self) -> Result<(), JevError> {
        std::fs::create_dir_all(&self.dir).map_err(|error| JevError::CredentialStore {
            detail: sanitize_detail(&format!("cannot create credential dir: {}", error.kind())),
        })
    }
}

impl CredentialStore for DpapiCredentialStore {
    fn store(&self, key_id: &str, secret: &str) -> Result<(), JevError> {
        self.store_cancellable(key_id, secret, &CredentialWriteGate::default())
    }

    fn store_cancellable(&self, key_id: &str, secret: &str, gate: &CredentialWriteGate) -> Result<(), JevError> {
        let path = self.path_for(key_id)?;
        if secret.trim().is_empty() {
            return Err(JevError::CredentialStore {
                detail: "refusing to store an empty secret".to_string(),
            });
        }
        if secret.len() > MAX_SECRET_LEN {
            return Err(JevError::CredentialStore {
                detail: format!("secret length {} exceeds limit {MAX_SECRET_LEN}", secret.len()),
            });
        }
        let ciphertext = platform::protect(secret.as_bytes())?;
        let envelope = CredentialEnvelope::wrap(&ciphertext);
        let serialized = serde_json::to_vec(&envelope).map_err(|_| JevError::CredentialStore {
            detail: "cannot serialize credential envelope".to_string(),
        })?;
        self.ensure_dir()?;
        let temp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true)
                .open(&temp).map_err(|error| JevError::CredentialStore {
                    detail: format!("cannot create encrypted credential temporary: {}", error.kind()),
                })?;
            file.write_all(&serialized).and_then(|_| file.sync_all()).map_err(|error| JevError::CredentialStore {
                detail: format!("cannot persist encrypted credential: {}", error.kind()),
            })?;
            drop(file);
            gate.commit(|| std::fs::rename(&temp, &path).map_err(|error| JevError::CredentialStore {
                detail: format!("cannot replace credential blob: {}", error.kind()),
            }))
        })();
        if result.is_err() { let _ = std::fs::remove_file(&temp); }
        result
    }

    fn get(&self, key_id: &str) -> Result<Option<String>, JevError> {
        let path = self.path_for(key_id)?;
        let blob = match std::fs::read(&path) {
            Ok(blob) => blob,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(JevError::CredentialStore {
                    detail: sanitize_detail(&format!("cannot read credential blob: {}", error.kind())),
                })
            }
        };
        let envelope: CredentialEnvelope = serde_json::from_slice(&blob).map_err(|_| {
            JevError::CredentialStore {
                detail: "credential envelope is not valid json".to_string(),
            }
        })?;
        if envelope.backend != "dpapi" {
            return Err(JevError::CredentialStore {
                detail: format!("credential envelope backend {} is not supported", envelope.backend),
            });
        }
        if envelope.scope != "current_user" {
            // A machine-scope or unknown blob is refused rather than silently decrypted.
            return Err(JevError::CredentialStore {
                detail: format!("credential envelope scope {} is not supported", envelope.scope),
            });
        }
        let ciphertext = envelope.ciphertext()?;
        let plaintext = platform::unprotect(&ciphertext)?;
        let secret = String::from_utf8(plaintext).map_err(|_| JevError::CredentialStore {
            detail: "credential plaintext is not valid utf-8".to_string(),
        })?;
        Ok(Some(secret))
    }

    fn delete(&self, key_id: &str) -> Result<(), JevError> {
        let path = self.path_for(key_id)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(JevError::CredentialStore {
                detail: sanitize_detail(&format!("cannot delete credential blob: {}", error.kind())),
            }),
        }
    }

    fn exists(&self, key_id: &str) -> Result<bool, JevError> {
        let path = self.path_for(key_id)?;
        Ok(path.is_file())
    }

    fn is_available(&self) -> bool {
        secure_store_available()
    }

    fn backend_name(&self) -> &'static str {
        if secure_store_available() {
            "dpapi"
        } else {
            "unavailable"
        }
    }
}

/// One-time availability probe for the secure store.
///
/// A tiny protect/unprotect round trip of a synthetic probe string is performed once per process
/// and cached. This lets `/jev status` report "unavailable" BEFORE the user types a key, instead
/// of failing at the first save. It never touches a real credential and writes nothing.
pub fn secure_store_available() -> bool {
    static AVAILABLE: once_cell::sync::OnceCell<bool> = once_cell::sync::OnceCell::new();
    *AVAILABLE.get_or_init(|| {
        if !platform::is_supported() {
            return false;
        }
        const PROBE: &[u8] = b"pi-jev-availability-probe";
        match platform::protect(PROBE) {
            Ok(ciphertext) => match platform::unprotect(&ciphertext) {
                Ok(plaintext) => plaintext == PROBE,
                Err(_) => false,
            },
            Err(_) => false,
        }
    })
}

/// Platform backends. Only Windows has a verified secure store in this build.
#[cfg(windows)]
mod platform {
    // Feature path verified against the vendored windows-sys 0.61.2 source:
    // src/Windows/Win32/Security/Cryptography/mod.rs is gated by
    // `Win32_Security_Cryptography` and declares CryptProtectData/CryptUnprotectData,
    // CRYPT_INTEGER_BLOB and CRYPTPROTECT_UI_FORBIDDEN. LocalFree comes from
    // Win32/Foundation, which that feature enables transitively.
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    use crate::error::JevError;

    /// DPAPI is compiled in on Windows.
    pub fn is_supported() -> bool {
        true
    }

    fn blob_from(bytes: &[u8]) -> CRYPT_INTEGER_BLOB {
        CRYPT_INTEGER_BLOB {
            cbData: bytes.len() as u32,
            pbData: bytes.as_ptr() as *mut u8,
        }
    }

    fn take_output(out: CRYPT_INTEGER_BLOB) -> Vec<u8> {
        // DPAPI returns a LocalAlloc block; copy it out and free it exactly once.
        if out.pbData.is_null() {
            return Vec::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(out.pbData, out.cbData as usize) };
        let owned = slice.to_vec();
        unsafe {
            LocalFree(out.pbData as *mut core::ffi::c_void);
        }
        owned
    }

    /// Encrypts for the CURRENT USER scope. Never uses `CRYPTPROTECT_LOCAL_MACHINE`.
    pub fn protect(plaintext: &[u8]) -> Result<Vec<u8>, JevError> {
        let input = blob_from(plaintext);
        let mut output = CRYPT_INTEGER_BLOB::default();
        let ok = unsafe {
            CryptProtectData(
                &input,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut output,
            )
        };
        if ok == 0 {
            return Err(JevError::CredentialStore {
                detail: "CryptProtectData failed (current-user scope)".to_string(),
            });
        }
        Ok(take_output(output))
    }

    /// Decrypts a blob produced by `protect` for this user.
    pub fn unprotect(blob: &[u8]) -> Result<Vec<u8>, JevError> {
        if blob.is_empty() {
            return Err(JevError::CredentialStore {
                detail: "credential blob is empty".to_string(),
            });
        }
        let input = blob_from(blob);
        let mut output = CRYPT_INTEGER_BLOB::default();
        let ok = unsafe {
            CryptUnprotectData(
                &input,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut output,
            )
        };
        if ok == 0 {
            return Err(JevError::CredentialStore {
                detail: "CryptUnprotectData failed (wrong user or corrupt blob)".to_string(),
            });
        }
        Ok(take_output(output))
    }
}

#[cfg(not(windows))]
mod platform {
    use crate::error::JevError;

    /// No verified secure store on this platform in this build.
    pub fn is_supported() -> bool {
        false
    }

    pub fn protect(_plaintext: &[u8]) -> Result<Vec<u8>, JevError> {
        Err(JevError::Unavailable {
            reason: "secure credential storage is only implemented for Windows DPAPI".to_string(),
        })
    }

    pub fn unprotect(_blob: &[u8]) -> Result<Vec<u8>, JevError> {
        Err(JevError::Unavailable {
            reason: "secure credential storage is only implemented for Windows DPAPI".to_string(),
        })
    }
}

/// Chooses the production store for `agent_dir`.
///
/// Returns the DPAPI store on Windows and an `UnavailableCredentialStore` elsewhere, so the
/// fail-closed path is the default and there is never a plaintext fallback.
pub fn default_credential_store(agent_dir: impl AsRef<Path>) -> Arc<dyn CredentialStore> {
    if secure_store_available() {
        Arc::new(DpapiCredentialStore::new(agent_dir.as_ref().join("jev")))
    } else {
        Arc::new(UnavailableCredentialStore::new(
            "secure credential storage (Windows DPAPI, current user) is unavailable",
        ))
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn cancelled_credential_write_cannot_commit() {
        let store = InMemoryCredentialStore::new();
        let gate = CredentialWriteGate::default();
        assert!(gate.cancel());
        assert_eq!(store.store_cancellable("typesafe", "synthetic-only", &gate), Err(JevError::Cancelled));
        assert!(store.is_empty());
    }

    #[test]
    fn committed_credential_is_not_reported_as_cancelled() {
        let store = InMemoryCredentialStore::new();
        let gate = CredentialWriteGate::default();
        store.store_cancellable("typesafe", "synthetic-only", &gate).unwrap();
        assert!(!gate.cancel());
        assert!(store.exists("typesafe").unwrap());
    }

    #[cfg(windows)]
    #[test]
    fn cancelled_dpapi_write_preserves_existing_ciphertext() {
        let dir = tempfile::tempdir().unwrap();
        let store = DpapiCredentialStore::new(dir.path());
        store.store("typesafe", "synthetic-original").unwrap();
        let path = store.path_for("typesafe").unwrap();
        let before = std::fs::read(&path).unwrap();
        let gate = CredentialWriteGate::default();
        assert!(gate.cancel());
        assert_eq!(store.store_cancellable("typesafe", "synthetic-replacement", &gate), Err(JevError::Cancelled));
        assert_eq!(std::fs::read(path).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
