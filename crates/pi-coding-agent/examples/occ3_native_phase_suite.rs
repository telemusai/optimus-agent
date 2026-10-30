//! OCC-3 native phase-suite producer.
//!
//! Emits the `optimus.occ3.native-phase-suite.v1` before/after restart
//! reports defined by the frozen Python adapter
//! (`extensions/sglang-occ3/deploy/v3/native_validator.py`, fixture
//! contract_revision 3) per the reviewed producer plans and root corrections:
//! - F3: the before report has NO `publication_request_sha256`; the
//!   precommitted publish-body digest lives INSIDE the retained negative
//!   control artifact; the before report binds the artifact digest
//!   transitively through `negative_control_retained.artifact_sha256`.
//! - F4: the before phase DOES publish one real native handoff (positive
//!   case); the no-publication rule applies ONLY to the retained stale
//!   negative control.
//! - F5: the inert stub replaces the installed `occ3_handoff` tool BEFORE
//!   the source prompt; the bounded stop is the genuine first tool dispatch
//!   (the stub terminates the tool batch) before any continuation, and an
//!   arbitrary prompt failure is never accepted as a valid capture.
//!
//! Default mode is the OFFLINE SELFTEST: the full two-phase suite runs against
//! a local synthetic native backend (127.0.0.1 fixture in the established
//! `occ3_source_handoff.rs` style). It exercises the real client code paths
//! (`Occ3NativeSession`, `SglangOcc3Connector`, `IsolatedNativeFamily`, the
//! Agent loop) only. It is NOT GPU, semantic or live proof: no real endpoint,
//! credential, model or network is contacted, `gpu_free_proven` stays false,
//! and every artifact declares `offline-synthetic-fixture` provenance.
//!
//! `--live DESCRIPTOR` runs one phase against a root-staged endpoint per the
//! reviewed plans (challenge echo, one-shot durable claim, phase drivers,
//! harness-only 409-state probe, no-replace report publication). The live
//! BEFORE phase retains the negative control's raw source handle through the
//! root-authorized SEAM1 `export_negative_control` accessor in
//! `occ3_state.rs` (gated on the pre-capture `arm_negative_control`
//! opt-in; one-shot; ordinary sessions can never export). The offline
//! selftest exercises the same accessor and cross-checks it against the
//! harness-owned fixture record. No silent fallback exists.

use indexmap::IndexMap;
use pi_agent_core::agent::{Agent, AgentOptions, PromptInput};
use pi_agent_core::types::{
    AgentMessage, AgentState, AgentTool, AgentToolResult, ContentBlock, ThinkingLevel,
};
use pi_ai::source_capture::{canonical_json_bytes, sha256, SourceCaptureProfile};
use pi_ai::types::{ContentBlock as AiContentBlock, ImageOrTextContent, Message, Model, API_OPENAI_COMPLETIONS};
use pi_coding_agent::core::agent_messages::{
    AgentFamilyCatalogEntry, AgentSessionMessageEndpoint,
};
use pi_coding_agent::core::agent_session::occ3_state::{
    NativeSourceBudget, Occ3NativeSession, NATIVE_HANDOFF_TOOL,
};
use pi_coding_agent::core::occ::occ3::connector::Occ3Connector;
use pi_coding_agent::core::occ::occ3::types::Occ3ModelIdentity;
use pi_coding_agent::core::occ3_connectors::sglang::{
    NativeExperimentalPermit, SglangOcc3Config, SglangOcc3Connector,
};
use pi_coding_agent::modes::daemon::daemon_mode::occ3_native::{
    IsolatedNativeFamily, IsolatedNativeMember,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const REPORT_SCHEMA: &str = "optimus.occ3.native-phase-suite.v1";
const CHALLENGE_SCHEMA: &str = "optimus.occ3.validator-phase-challenge.v1";
const CLAIM_SCHEMA: &str = "optimus.occ3.native-phase-attempt-claim.v1";
const ARTIFACT_SCHEMA: &str = "optimus.occ3.negative-control-artifact.v1";
const DESCRIPTOR_SCHEMA: &str = "optimus.occ3.native-phase-suite-descriptor.v1";
const METHOD_ID: &str = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1";
const PLATFORM: &str = "windows-x86_64";
const UNKNOWN_ANSWER: &str = "UNKNOWN";
const MAX_CHALLENGE_BYTES: usize = 4096;
const MAX_CLAIM_BYTES: usize = 8192;
const MAX_ARTIFACT_BYTES: usize = 8192;
const MAX_REPORT_BYTES: usize = 2 * 1024 * 1024;
const MAX_DESCRIPTOR_BYTES: usize = 65536;
const MAX_JSON_DEPTH: usize = 32;
const DEFAULT_MAX_TOKENS: f64 = 131072.0;
const CEILING_TOKENS: f64 = 262144.0;
const CONTEXT_WINDOW: f64 = 1048576.0;
const CONTROL_HEADER: &str = "x-pomerium-authorization";
const CONTROL_VALUE: &str = "synthetic-fixture-gateway-control-value";
const FIXTURE_KEY: &str = "synthetic-fixture-key";
const PRINCIPAL_DIGEST: &str = "a5ac3b0a49b31b279ea3782df3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3";
const LAYOUT_DIGEST: &str = "11d7ea1f62a449cfa9203a4312ff42c412b6a17e4dbd8b2b9d2b5b1c3f7d9a4e";
const PROFILE_DIGEST: &str = "47ba4c76ed3f1d54a1798cc8a9be9ee17b8063e0bcdbdb6a52a636db3a6f9a9d";
// Offline fixture digests: derived per _derived_profile_digest semantics
// (extensions/sglang-occ3/deploy/v3/native_validator.py) from the pinned
// static profile (deploy/native-build-inputs/native-profile-input.json) plus
// the offline fixture epochs (offline-runtime-epoch-before/41,
// offline-runtime-epoch-after/42): canonical JSON (sorted keys, compact
// separators, ASCII escapes, NaN refused), sha256, ascii encoding.
const OFFLINE_PROFILE_DIGEST_BEFORE: &str = "1f69749f34cf52a200b16152b44fd24007cbcd909a9fa665e53a7307802df425";
const OFFLINE_PROFILE_DIGEST_AFTER: &str = "52a6d89c090245d13089ec0fae62a835d38706cb30b14b17c1abea6da6b92e36";
const STALE_CONTROL_TOKEN: &str = "offline-stale-retained-control";

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn fail() -> String {
    "Native phase-suite producer refused or operation uncertain; see private reviewed \
inputs and offline gates"
        .into()
}

// ---------------------------------------------------------------------------
// Strict hostile-input document loading (bound before allocation, strict
// UTF-8, no BOM, depth-32 scan, duplicate-key rejection, deny-unknown-fields).
// Mirrors the frozen adapter loader semantics.
// ---------------------------------------------------------------------------

/// Combined structural scan: nesting depth <= 32, balanced containers and
/// no duplicate object keys. Runs on strictly decoded UTF-8 text before any
/// semantic processing.
fn structural_scan(text: &str) -> Result<(), String> {
    let mut stack: Vec<(bool, Option<HashSet<String>>)> = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                let mut value = String::new();
                let mut escaped = false;
                loop {
                    match chars.next() {
                        Some('\\') if !escaped => escaped = true,
                        Some('"') if !escaped => break,
                        Some(other) => {
                            if escaped {
                                value.push('\\');
                            }
                            escaped = false;
                            value.push(other);
                        }
                        None => return Err("malformed_json".into()),
                    }
                }
                if let Some((is_object, keys)) = stack.last_mut() {
                    if *is_object && chars.skip_whitespace_is_colon() {
                        let keys = keys.get_or_insert_with(HashSet::new);
                        if !keys.insert(value) {
                            return Err("duplicate_json_key".into());
                        }
                    }
                }
            }
            '{' => {
                stack.push((true, None));
                if stack.len() > MAX_JSON_DEPTH {
                    return Err("malformed_json".into());
                }
            }
            '[' => {
                stack.push((false, None));
                if stack.len() > MAX_JSON_DEPTH {
                    return Err("malformed_json".into());
                }
            }
            '}' => match stack.pop() {
                Some((true, _)) => {}
                _ => return Err("malformed_json".into()),
            },
            ']' => match stack.pop() {
                Some((false, _)) => {}
                _ => return Err("malformed_json".into()),
            },
            _ => {}
        }
    }
    if stack.is_empty() {
        Ok(())
    } else {
        Err("malformed_json".into())
    }
}

trait ColonPeek {
    fn skip_whitespace_is_colon(&mut self) -> bool;
}
impl ColonPeek for std::iter::Peekable<std::str::Chars<'_>> {
    fn skip_whitespace_is_colon(&mut self) -> bool {
        while let Some(&c) = self.peek() {
            if c.is_whitespace() {
                self.next();
            } else {
                return c == ':';
            }
        }
        false
    }
}

/// Strict UTF-8 decode with explicit BOM refusal (no autodetect, no strip).
fn strict_decode(bytes: &[u8]) -> Result<String, String> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err("malformed_json".into());
    }
    String::from_utf8(bytes.to_vec()).map_err(|_| "malformed_json".to_string())
}

/// Size bound first, then strict read, strict decode, structural scan and a
/// deny-unknown-fields typed parse. All refusals are fixed value-free codes.
fn strict_load<T: serde::de::DeserializeOwned>(path: &Path, limit: usize) -> Result<T, String> {
    let bytes = strict_read_bounded(path, limit)?;
    let text = strict_decode(&bytes)?;
    structural_scan(&text)?;
    serde_json::from_str(&text).map_err(|_| fail())
}

// ---------------------------------------------------------------------------
// Private no-replace publication (Windows, enabled API set only).
// Root correction F2: verification uses GetKernelObjectSecurity on the open
// handle (never a name-based query), then GetSecurityDescriptor*/ACL/SID
// inspection. Staging is CREATE_NEW with an owner-only protected DACL; the
// atomic publish is MoveFileExW with NO MOVEFILE_REPLACE_EXISTING.
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod private_publish {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Security::{
        AddAccessAllowedAceEx, EqualSid, GetAce, GetAclInformation, GetKernelObjectSecurity,
        GetLengthSid, GetSecurityDescriptorControl, GetSecurityDescriptorDacl,
        GetSecurityDescriptorOwner, GetTokenInformation, InitializeAcl,
        InitializeSecurityDescriptor, SetSecurityDescriptorControl,
        SetSecurityDescriptorDacl, SetSecurityDescriptorOwner, ACL, ACL_REVISION,
        ACL_SIZE_INFORMATION, ACCESS_ALLOWED_ACE, ACE_HEADER, DACL_SECURITY_INFORMATION,
        OWNER_SECURITY_INFORMATION, PSID, SECURITY_ATTRIBUTES, SECURITY_DESCRIPTOR,
        SE_DACL_PROTECTED, TOKEN_QUERY, TOKEN_USER,
        TokenUser, ACL_INFORMATION_CLASS, AclSizeInformation,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FlushFileBuffers, GetFileInformationByHandle, MoveFileExW, ReadFile,
        SetFilePointerEx, WriteFile, BY_HANDLE_FILE_INFORMATION, CREATE_NEW, FILE_ALL_ACCESS,
        FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ,
        OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::Win32::System::SystemServices::{ACCESS_ALLOWED_ACE_TYPE, SECURITY_DESCRIPTOR_REVISION};

    fn to_wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }
    fn is_ok(result: windows_sys::core::BOOL) -> Result<(), String> {
        if result != 0 {
            Ok(())
        } else {
            Err(fail())
        }
    }

    /// Copy of the current process-user SID (owner of every staged file).
    fn current_user_sid() -> Result<Vec<u8>, String> {
        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            is_ok(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token))?;
            let mut needed: u32 = 0;
            GetTokenInformation(
                token,
                TokenUser,
                std::ptr::null_mut(),
                0,
                &mut needed,
            );
            let mut buffer = vec![0u8; needed as usize];
            let ok = GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr() as *mut core::ffi::c_void,
                needed,
                &mut needed,
            );
            CloseHandle(token);
            is_ok(ok)?;
            let user = &*(buffer.as_ptr() as *const TOKEN_USER);
            let sid = user.User.Sid as PSID;
            let length = GetLengthSid(sid);
            let sid_bytes = std::slice::from_raw_parts(sid as *const u8, length as usize);
            if sid_bytes.is_empty() || *sid_bytes.first().unwrap() == 0 {
                return Err(fail());
            }
            Ok(sid_bytes.to_vec())
        }
    }

    /// Owner-only protected DACL security attributes. The descriptor stays
    /// alive (self-relative copy kept in the struct) for the CreateFileW call.
    struct OwnerOnlyAttributes {
        sid: Vec<u8>,
        // `acl` and `descriptor` are keep-alive buffers: their bytes are
        // referenced through the raw pointer stored in `attributes`
        // (lpSecurityDescriptor), so direct field reads never happen.
        #[allow(dead_code)]
        acl: Vec<u8>,
        #[allow(dead_code)]
        descriptor: Vec<u8>,
        attributes: SECURITY_ATTRIBUTES,
    }
    fn owner_only_attributes() -> Result<OwnerOnlyAttributes, String> {
        let sid = current_user_sid()?;
        // ACL header + one ACCESS_ALLOWED_ACE (header + mask + SID), aligned.
        let mut acl = vec![0u8; (std::mem::size_of::<ACL>()
            + std::mem::size_of::<ACCESS_ALLOWED_ACE>()
            + sid.len()
            + 8)
            .max(64)];
        unsafe {
            is_ok(InitializeAcl(
                acl.as_mut_ptr() as *mut ACL,
                acl.len() as u32,
                ACL_REVISION,
            ))?;
            is_ok(AddAccessAllowedAceEx(
                acl.as_mut_ptr() as *mut ACL,
                ACL_REVISION,
                0,
                FILE_ALL_ACCESS,
                sid.as_ptr() as PSID,
            ))?;
            let mut descriptor = vec![0u8; std::mem::size_of::<SECURITY_DESCRIPTOR>()];
            is_ok(InitializeSecurityDescriptor(
                descriptor.as_mut_ptr() as *mut core::ffi::c_void,
                SECURITY_DESCRIPTOR_REVISION as u32,
            ))?;
            is_ok(SetSecurityDescriptorDacl(
                descriptor.as_mut_ptr() as *mut core::ffi::c_void,
                1,
                acl.as_ptr() as *const ACL,
                0,
            ))?;
            is_ok(SetSecurityDescriptorOwner(
                descriptor.as_mut_ptr() as *mut core::ffi::c_void,
                sid.as_ptr() as PSID,
                0,
            ))?;
            // No inherited ACE survives: the DACL is protected.
            is_ok(SetSecurityDescriptorControl(
                descriptor.as_mut_ptr() as *mut core::ffi::c_void,
                SE_DACL_PROTECTED,
                SE_DACL_PROTECTED,
            ))?;
            let attributes = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor.as_mut_ptr() as *mut core::ffi::c_void,
                bInheritHandle: 0,
            };
            Ok(OwnerOnlyAttributes { sid, acl, descriptor, attributes })
        }
    }

    /// Handle-bound ACL verification (root correction F2): read owner+DACL
    /// through GetKernelObjectSecurity on the OPEN handle, then require a
    /// protected DACL, the process user as owner and exactly one allowed ACE
    /// for exactly that SID.
    unsafe fn verify_owner_only_acl(handle: HANDLE, sid: &[u8]) -> Result<(), String> {
        let mut needed: u32 = 0;
        GetKernelObjectSecurity(
            handle,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            0,
            &mut needed,
        );
        if needed == 0 || needed > 4096 {
            return Err(fail());
        }
        let mut buffer = vec![0u8; needed as usize];
        is_ok(GetKernelObjectSecurity(
            handle,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            buffer.as_mut_ptr() as *mut core::ffi::c_void,
            needed,
            &mut needed,
        ))?;
        let descriptor = buffer.as_ptr() as *const SECURITY_DESCRIPTOR;
        let mut control: u16 = 0;
        let mut revision: u32 = 0;
        is_ok(GetSecurityDescriptorControl(
            descriptor as *mut core::ffi::c_void,
            &mut control,
            &mut revision,
        ))?;
        if control & SE_DACL_PROTECTED == 0 {
            return Err(fail());
        }
        let mut owner: PSID = std::ptr::null_mut();
        let mut owner_defaulted: windows_sys::core::BOOL = 0;
        is_ok(GetSecurityDescriptorOwner(
            descriptor as *mut core::ffi::c_void,
            &mut owner,
            &mut owner_defaulted,
        ))?;
        if owner.is_null() || EqualSid(owner, sid.as_ptr() as PSID) == 0 {
            return Err(fail());
        }
        let mut present: windows_sys::core::BOOL = 0;
        let mut dacl: *mut ACL = std::ptr::null_mut();
        let mut dacl_defaulted: windows_sys::core::BOOL = 0;
        is_ok(GetSecurityDescriptorDacl(
            descriptor as *mut core::ffi::c_void,
            &mut present,
            &mut dacl,
            &mut dacl_defaulted,
        ))?;
        if present == 0 || dacl.is_null() {
            return Err(fail());
        }
        let mut size = std::mem::zeroed::<ACL_SIZE_INFORMATION>();
        is_ok(GetAclInformation(
            dacl as *const ACL,
            &mut size as *mut _ as *mut core::ffi::c_void,
            std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
            AclSizeInformation as ACL_INFORMATION_CLASS,
        ))?;
        if size.AceCount != 1 {
            return Err(fail());
        }
        let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
        is_ok(GetAce(dacl as *const ACL, 0, &mut ace))?;
        let header = &*(ace as *const ACE_HEADER);
        if u32::from(header.AceType) != ACCESS_ALLOWED_ACE_TYPE {
            return Err(fail());
        }
        let allowed = &*(ace as *const ACCESS_ALLOWED_ACE);
        if allowed.Mask != FILE_ALL_ACCESS
            || EqualSid(
                std::ptr::addr_of!(allowed.SidStart) as PSID,
                sid.as_ptr() as PSID,
            ) == 0
        {
            return Err(fail());
        }
        Ok(())
    }

    unsafe fn write_bounded(handle: HANDLE, bytes: &[u8]) -> Result<(), String> {
        let mut offset = 0usize;
        while offset < bytes.len() {
            let length = (bytes.len() - offset).min(u32::MAX as usize) as u32;
            let mut written: u32 = 0;
            is_ok(WriteFile(
                handle,
                bytes[offset..].as_ptr(),
                length,
                &mut written,
                std::ptr::null_mut(),
            ))?;
            if written == 0 {
                return Err(fail());
            }
            offset += written as usize;
        }
        if offset != bytes.len() {
            return Err(fail());
        }
        Ok(())
    }
    unsafe fn read_all(handle: HANDLE, limit: usize) -> Result<Vec<u8>, String> {
        let mut zero: i64 = 0;
        is_ok(SetFilePointerEx(handle, 0, &mut zero, 0 /* FILE_BEGIN */))?;
        let mut out = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let mut read: u32 = 0;
            is_ok(ReadFile(
                handle,
                chunk.as_mut_ptr(),
                chunk.len() as u32,
                &mut read,
                std::ptr::null_mut(),
            ))?;
            if read == 0 {
                break;
            }
            if out.len() + read as usize > limit {
                return Err("file_too_large".into());
            }
            out.extend_from_slice(&chunk[..read as usize]);
        }
        Ok(out)
    }

    /// Exclusive CREATE_NEW staging with an owner-only protected DACL, bounded
    /// write + flush + byte-verify, then atomic no-replace publication via
    /// MoveFileExW(0). An existing final name refuses (no overwrite, no
    /// replay). Non-Windows targets fail closed upstream.
    pub fn publish_exclusive(
        directory: &Path,
        final_name: &str,
        bytes: &[u8],
    ) -> Result<(), String> {
        if bytes.is_empty() || bytes.len() > MAX_REPORT_BYTES || final_name.is_empty() {
            return Err(fail());
        }
        let staging_name = format!(
            ".{}.{}.staging",
            final_name,
            uuid::Uuid::new_v4().simple()
        );
        let staging = directory.join(&staging_name);
        let final_path = directory.join(final_name);
        let attributes = owner_only_attributes()?;
        unsafe {
            let staging_wide = to_wide(&staging.to_string_lossy());
            let mut handle = CreateFileW(
                staging_wide.as_ptr(),
                GENERIC_WRITE,
                0,
                &attributes.attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            );
            if handle == INVALID_HANDLE_VALUE {
                return Err(fail());
            }
            let written = write_bounded(handle, bytes);
            let flushed = if written.is_ok() {
                is_ok(FlushFileBuffers(handle))
            } else {
                Err(fail())
            };
            CloseHandle(handle);
            written?;
            flushed?;
            // Readback identity: reopen the staging file through the
            // reparse-point-refusing path and verify ACL + exact bytes.
            let reopen_wide = to_wide(&staging.to_string_lossy());
            handle = CreateFileW(
                reopen_wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT,
                std::ptr::null_mut(),
            );
            if handle == INVALID_HANDLE_VALUE {
                return Err(fail());
            }
            let verified = (|| {
                let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
                is_ok(GetFileInformationByHandle(handle, &mut info))?;
                if info.dwFileAttributes
                    & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
                    != 0
                    || info.nNumberOfLinks != 1
                    || info.dwFileAttributes
                        & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_DIRECTORY
                        != 0
                {
                    return Err(fail());
                }
                verify_owner_only_acl(handle, &attributes.sid)?;
                let read = read_all(handle, MAX_REPORT_BYTES)?;
                if read.len() != bytes.len() || sha256(&read) != sha256(bytes) {
                    return Err(fail());
                }
                Ok(())
            })();
            CloseHandle(handle);
            verified?;
            let source_wide = to_wide(&staging.to_string_lossy());
            let target_wide = to_wide(&final_path.to_string_lossy());
            if MoveFileExW(source_wide.as_ptr(), target_wide.as_ptr(), 0) == 0 {
                // ERROR_ALREADY_EXISTS is the refuse/no-replay path; any
                // other error is uncertain and also refuses. The staging
                // file stays private and ineligible.
                let error = windows_sys::Win32::Foundation::GetLastError();
                if error == ERROR_ALREADY_EXISTS {
                    return Err("publish_refused_no_replay".into());
                }
                return Err("publish_refused_no_replace".into());
            }
        }
        // Final immutability policy: the final name is never reopened for
        // write by this producer again.
        Ok(())
    }

    /// Strict bounded read through a FILE_FLAG_OPEN_REPARSE_POINT handle:
    /// refuses reparse points, directories, extra hardlinks and oversize
    /// files before reading; the byte bound applies before any allocation
    /// beyond the stated limit.
    pub fn read_file_strict(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
        unsafe {
            let wide = to_wide(&path.to_string_lossy());
            let handle = CreateFileW(
                wide.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT,
                std::ptr::null_mut(),
            );
            if handle == INVALID_HANDLE_VALUE {
                return Err("file_too_large".into());
            }
            let result = (|| {
                let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
                is_ok(GetFileInformationByHandle(handle, &mut info))?;
                if info.nFileSizeHigh != 0 || info.nFileSizeLow as u64 > limit as u64 {
                    return Err("file_too_large".into());
                }
                if info.dwFileAttributes
                    & (windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
                        | windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_DIRECTORY)
                    != 0
                    || info.nNumberOfLinks != 1
                {
                    return Err("reparse_point_refused".into());
                }
                read_all(handle, limit)
            })();
            CloseHandle(handle);
            result
        }
    }
}

#[cfg(windows)]
use private_publish::publish_exclusive;

#[cfg(not(windows))]
fn publish_exclusive(_directory: &Path, _final_name: &str, _bytes: &[u8]) -> Result<(), String> {
    // Platform fail-closed: no replace-style fallback exists anywhere.
    Err("windows_only_no_replace_publication_refused".into())
}

/// Bounded strict read: size bound BEFORE allocation, symlink/reparse/
/// hardlink refusal, regular-file requirement. Shared by every consumed
/// external document (challenge, claim, artifact, report, descriptor).
#[cfg(windows)]
pub fn strict_read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    private_publish::read_file_strict(path, limit)
}

#[cfg(not(windows))]
pub fn strict_read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|_| "file_too_large".to_string())?;
    if metadata.file_type().is_symlink() {
        return Err("symlink_refused".into());
    }
    if !metadata.is_file() {
        return Err("file_too_large".into());
    }
    if metadata.len() as usize > limit {
        return Err("file_too_large".into());
    }
    std::fs::read(path).map_err(|_| "file_too_large".to_string())
}

// ---------------------------------------------------------------------------
// Offline synthetic native backend (established occ3_source_handoff.rs style)
// with the planned restart model: each fixture instance is one runtime epoch;
// a handle captured under an earlier epoch is stale state after restart and
// publish refuses with the actual router shape 409 {"error":{"code":"state"}}.
// ---------------------------------------------------------------------------

const LAYER_INDEX: u32 = 77;
const HIDDEN_DIM: u32 = 6144;
const EXPECTED_RANKS: u32 = 8;
const ACCEPTED_SEQUENCE_DIGEST: &str =
    "8f14e45fceea167a5a36dedd4bea2543f7a7c2b6e59a1a1d1f9b0d1c0a3e2d1c";

#[derive(Clone)]
struct CapturedRecord {
    handle: String,
    session: String,
    intent: Value,
    receipt: Value,
    fact: Option<String>,
    tool_call_id: String,
}

#[derive(Default)]
struct FixtureRecords {
    requests: Vec<(String, Value)>,
    enrolled: Vec<String>,
    captures: Vec<CapturedRecord>,
    published: Vec<Value>,
    retired: Vec<String>,
    control_header_seen: Vec<Option<String>>,
    /// Never reset across a modeled restart: server-side handles stay
    /// globally unique like the real runtime's unguessable handle ids.
    capture_counter: u64,
}

/// Mutable per-epoch server identity: one fixture instance is one runtime
/// epoch; a modeled restart advances the epoch/generation on the SAME
/// listener (the runbook restarts the same candidate container, so the
/// endpoint origin never changes across the two phase reports).
#[derive(Default)]
struct FixtureEpoch {
    runtime_epoch: String,
    runtime_generation: u64,
    expires_at_epoch: u64,
    profile_digest: String,
}

/// Modeled controlled restart: fresh epoch, ADVANCED generation, retained
/// native-incarnation counter semantics (generation strictly greater), and
/// the runtime's volatile state (captures, sessions, publications) lost -
/// a handle captured before the restart is stale state afterwards.
fn restart_fixture(
    epoch_state: &Arc<Mutex<FixtureEpoch>>,
    records: &Arc<Mutex<FixtureRecords>>,
    runtime_epoch: &str,
    runtime_generation: u64,
    expires_at_epoch: u64,
    profile_digest: &str,
) {
    {
        let mut state = epoch_state.lock().unwrap();
        state.runtime_epoch = runtime_epoch.into();
        state.runtime_generation = runtime_generation;
        state.expires_at_epoch = expires_at_epoch;
        state.profile_digest = profile_digest.into();
    }
    let mut records = records.lock().unwrap();
    records.captures.clear();
    records.enrolled.clear();
    records.published.clear();
    records.retired.clear();
}

fn fixture_identity() -> Occ3ModelIdentity {
    Occ3ModelIdentity {
        model_id: "SYNTHETIC-glm53".into(),
        architecture: "SYNTHETIC-GlmMoeDsaForCausalLM".into(),
        checkpoint_revision: "SYNTHETIC-revision".into(),
        weight_fingerprint: "SYNTHETIC-metadata-not-weights".into(),
        tokenizer_fingerprint: "SYNTHETIC-tokenizer".into(),
        chat_template_fingerprint: "SYNTHETIC-template".into(),
        quantisation: "SYNTHETIC-fp8-e4m3".into(),
        tensor_geometry: "SYNTHETIC-hidden6144".into(),
        kv_layout: "SYNTHETIC-tp8-original".into(),
        latent_method: METHOD_ID.into(),
        runtime: "SYNTHETIC-sglang-0.5.21".into(),
    }
}

fn sse(id: &str, tool_call_id: &str, content: Option<&str>) -> String {
    let mut chunks = vec![json!({"id":id,"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"retained native reasoning"},"finish_reason":null}]})];
    match content {
        Some(text) => chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]})),
        None => chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":tool_call_id,"type":"function","function":{"name":"occ3_handoff","arguments":"{\"receiver_role\":\"child\",\"receiver_name\":\"receiver\"}"}}]},"finish_reason":null}]})),
    }
    let finish = if content.is_some() { "stop" } else { "tool_calls" };
    chunks.push(json!({"id":id,"choices":[{"index":0,"delta":{},"finish_reason":finish}]}));
    let mut body = chunks
        .into_iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>();
    body.push_str("data: [DONE]\n\n");
    body
}

async fn read_request(
    socket: &mut tokio::net::TcpStream,
) -> (String, Value, Option<String>, Vec<String>) {
    let mut bytes = Vec::new();
    let end;
    loop {
        let mut part = [0u8; 4096];
        let n = socket.read(&mut part).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&part[..n]);
        if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            end = i + 4;
            break;
        }
    }
    let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
    assert!(headers
        .to_lowercase()
        .contains(&format!("authorization: bearer {FIXTURE_KEY}")));
    let control_prefix = format!("{CONTROL_HEADER}: ");
    let control = headers.lines().find_map(|l| {
        l.to_ascii_lowercase()
            .strip_prefix(&control_prefix)
            .map(str::to_string)
    });
    // One entry per accept-encoding header line, mirroring the live server's
    // raw_request.headers.getlist("accept-encoding") for the source-request
    // contract check.
    let accept_encodings: Vec<String> = headers
        .lines()
        .filter_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("accept-encoding:")
                .map(|value| value.trim().to_string())
        })
        .collect();
    let size = headers
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length: ")
                .map(|v| v.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    while bytes.len() < end + size {
        let mut p = [0u8; 4096];
        let n = socket.read(&mut p).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&p[..n]);
    }
    (
        headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .into(),
        if size == 0 {
            json!(null)
        } else {
            serde_json::from_slice(&bytes[end..end + size]).unwrap()
        },
        control,
        accept_encodings,
    )
}

async fn send(socket: &mut tokio::net::TcpStream, status: &str, body: String, extra: Vec<(&str, String)>) {
    let mut headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

async fn send_sse(socket: &mut tokio::net::TcpStream, body: String, extra: Vec<(&str, String)>) {
    let mut headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        headers.push_str(&format!("{k}: {v}\r\n"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes()).await.unwrap();
    socket.write_all(body.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}

/// Actual attachment error shape (chat_adapter.py `_error`): the occ3
/// status line with {"error":{"message":...,"type":"occ3_attachment_error",
/// "code":...}} and cache-control: no-store.
async fn send_attachment_error(socket: &mut tokio::net::TcpStream, status: u16, code: &str) {
    let reason = match status {
        400 => "400 Bad Request",
        409 => "409 Conflict",
        _ => "503 Service Unavailable",
    };
    send(
        socket,
        reason,
        json!({"error":{"message":code,"type":"occ3_attachment_error","code":status}}).to_string(),
        vec![("cache-control", "no-store".to_string())],
    )
    .await;
}

/// Actual router error shape (latent_control_api.py): 409 + {"error":{"code":...}}.
async fn send_router_error(socket: &mut tokio::net::TcpStream, status: u16, code: &str) {
    let reason = match status {
        400 => "400 Bad Request",
        401 => "401 Unauthorized",
        409 => "409 Conflict",
        429 => "429 Too Many Requests",
        _ => "503 Service Unavailable",
    };
    send(
        socket,
        reason,
        json!({"error":{"code":code}}).to_string(),
        vec![("cache-control", "no-store".to_string())],
    )
    .await;
}

fn extract_fact(messages: &Value) -> Option<String> {
    let text = messages.to_string();
    let marker = "The private vault code is ";
    let start = text.find(marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find('.')?;
    let fact = &rest[..end];
    if fact.is_empty() || fact.len() > 128 || !fact.starts_with("occ3-") {
        return None;
    }
    Some(fact.to_string())
}

async fn serve(
    epoch_state: Arc<Mutex<FixtureEpoch>>,
    records: Arc<Mutex<FixtureRecords>>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let records = records.clone();
            let (epoch, generation, expiry, profile_digest) = {
                let state = epoch_state.lock().unwrap();
                (
                    state.runtime_epoch.clone(),
                    state.runtime_generation,
                    state.expires_at_epoch,
                    state.profile_digest.clone(),
                )
            };
            tokio::spawn(async move {
                let (path, body, control, accept_encodings) = read_request(&mut socket).await;
                {
                    let mut records = records.lock().unwrap();
                    records.requests.push((path.clone(), body.clone()));
                    records.control_header_seen.push(control);
                }
                match path.as_str() {
                    "/occ3/v1/native/sessions/enroll" => {
                        records.lock().unwrap().enrolled.push(
                            body["session_id"].as_str().unwrap_or_default().to_string(),
                        );
                        let reply = json!({"schema":"occ3.sessions.enrolled.v1","operation_id":body["operation_id"],"session_id":body["session_id"],
                            "principal_digest":PRINCIPAL_DIGEST,"runtime_id":body["runtime_id"],"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"expires_at_epoch":body["expires_at_epoch"],
                            "readiness":{"runtime_epoch":body["runtime_id"],"runtime_generation":body["runtime_generation"],"model_fingerprint":body["model_fingerprint"],"method_id":METHOD_ID,
                                "expected_ranks":EXPECTED_RANKS,"ready_ranks":EXPECTED_RANKS,"tp_layout_digest":LAYOUT_DIGEST,"hidden_dim":HIDDEN_DIM,"layer_index":LAYER_INDEX,"dtype":"bfloat16","profile_digest":profile_digest,"source_capture":true,"continuous_prefix":true,"terminal_journal":true}});
                        send(&mut socket, "200 OK", reply.to_string(), vec![]).await;
                    }
                    "/occ3/v1/native/publish" => {
                        let fields = ["schema","publication_id","source_handle","source_session_id","receiver_session_id","message_id","source_turn_nonce","tool_call_id","expires_at_epoch"];
                        let object = body.as_object();
                        let valid = object.is_some_and(|o| o.len() == fields.len()
                            && fields.iter().all(|f| o.contains_key(*f)));
                        if !valid || body["schema"] != "occ3.publish.v1" {
                            send_router_error(&mut socket, 400, "invalid").await;
                            return;
                        }
                        // All state inspection happens under one short-lived
                        // lock with no await inside; the verdict then drives
                        // the response so the guard is never held across one.
                        enum PublishVerdict {
                            Identity,
                            Stale,
                            Expired,
                            State,
                            Published { capture: CapturedRecord, expiry_epoch: u64 },
                        }
                        let verdict = {
                            let records_guard = records.lock().unwrap();
                            let source_session = body["source_session_id"].as_str().unwrap_or_default();
                            let receiver_session = body["receiver_session_id"].as_str().unwrap_or_default();
                            if !records_guard.enrolled.iter().any(|s| s == source_session)
                                || !records_guard.enrolled.iter().any(|s| s == receiver_session) {
                                PublishVerdict::Identity
                            } else {
                                // Stale-handle state semantics: a handle minted under
                                // an earlier (pre-restart) epoch is unknown to this
                                // restarted runtime's capture registry.
                                let handle = body["source_handle"].as_str().unwrap_or_default();
                                match records_guard.captures.iter().rev().find(|c| c.handle == handle) {
                                    None => PublishVerdict::Stale,
                                    Some(capture) => {
                                        let expiry_epoch = body["expires_at_epoch"].as_u64().unwrap_or_default();
                                        if expiry_epoch <= now() || expiry_epoch > expiry {
                                            PublishVerdict::Expired
                                        } else if capture.session != source_session
                                            || capture.intent["turn_nonce"] != body["source_turn_nonce"]
                                            || capture.tool_call_id != body["tool_call_id"].as_str().unwrap_or_default()
                                        {
                                            PublishVerdict::State
                                        } else {
                                            PublishVerdict::Published { capture: capture.clone(), expiry_epoch }
                                        }
                                    }
                                }
                            }
                        };
                        match verdict {
                            PublishVerdict::Identity => {
                                send_router_error(&mut socket, 401, "identity").await;
                            }
                            PublishVerdict::Stale | PublishVerdict::State => {
                                send_router_error(&mut socket, 409, "state").await;
                            }
                            PublishVerdict::Expired => {
                                send_router_error(&mut socket, 409, "expired").await;
                            }
                            PublishVerdict::Published { capture, expiry_epoch } => {
                                let mut binding = body.clone();
                                binding.as_object_mut().unwrap().remove("schema");
                                binding["principal_digest"] = json!(PRINCIPAL_DIGEST);
                                let digest = sha256(&canonical_json_bytes(&binding).unwrap());
                                let reply = json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"binding_digest":digest,"state":"published","source_handle":capture.handle,"expires_at_epoch":expiry_epoch});
                                records.lock().unwrap().published.push(body.clone());
                                send(&mut socket, "200 OK", reply.to_string(), vec![("cache-control", "no-store".to_string())]).await;
                            }
                        }
                    }
                    "/occ3/v1/native/publication/status" => {
                        let found = {
                            let records_guard = records.lock().unwrap();
                            records_guard
                                .published
                                .iter()
                                .rev()
                                .find(|p| p["publication_id"] == body["publication_id"])
                                .cloned()
                        };
                        let reply = match found {
                            Some(p) => json!({"schema":"occ3.publication.v1","publication_id":p["publication_id"],"binding_digest":"0".repeat(64),"state":"published","source_handle":p["source_handle"],"expires_at_epoch":p["expires_at_epoch"]}),
                            None => json!({"schema":"occ3.publication.v1","publication_id":body["publication_id"],"state":"absent"}),
                        };
                        send(&mut socket, "200 OK", reply.to_string(), vec![]).await;
                    }
                    "/occ3/v1/native/retire" => {
                        let retired_handle = {
                            let mut records_guard = records.lock().unwrap();
                            let handle = body["source_handle"].as_str().unwrap_or_default().to_string();
                            let session = body["source_session_id"].as_str().unwrap_or_default();
                            let known = records_guard.enrolled.iter().any(|s| s == session)
                                && records_guard.captures.iter().any(|c| c.handle == handle && c.session == session);
                            if known {
                                records_guard.retired.push(handle.clone());
                                Some(handle)
                            } else {
                                None
                            }
                        };
                        match retired_handle {
                            None => {
                                send_router_error(&mut socket, 401, "identity").await;
                            }
                            Some(handle) => {
                                let reply = json!({"schema":"occ3.retired.v1","operation_id":body["operation_id"],"source_handle":handle,"revoked":true,"gpu_freed":false,"state":"retired"});
                                send(&mut socket, "200 OK", reply.to_string(), vec![]).await;
                            }
                        }
                    }
                    "/v1/chat/completions" => {
                        assert_eq!(
                            body.get("max_tokens").or_else(|| body.get("max_completion_tokens")),
                            Some(&json!(DEFAULT_MAX_TOKENS as u64)),
                            "real native default is preserved"
                        );
                        // The suite requests MAX through the model-specific conversion;
                        // the documented receiver budget override
                        // (OCC3_RECEIVER_THINKING) may lower it for live runs
                        // (gateway idle-window constraint), so its exact value is
                        // accepted alongside the historical set.
                        let receiver_override = std::env::var("OCC3_RECEIVER_THINKING")
                            .ok()
                            .filter(|v| matches!(v.as_str(), "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"));
                        assert!(
                            body.get("reasoning_effort").and_then(Value::as_str).is_some_and(|v| matches!(v, "high" | "xhigh" | "max")
                                || receiver_override.as_deref() == Some(v)),
                            "MAX goes through the existing model-specific conversion, or the documented receiver override matches"
                        );
                        if let Some(intent) = body.get("occ3_capture") {
                            // The live occ3-sglang server (chat_adapter handle_around)
                            // admits capture-armed source requests only with exactly
                            // one accept-encoding header whose value is identity;
                            // anything else is 400 unsupported_profile, so the offline
                            // fixture must refuse the same shape instead of masking
                            // a client that never sends the header.
                            if accept_encodings.len() != 1
                                || accept_encodings[0].trim().to_ascii_lowercase() != "identity"
                            {
                                send_attachment_error(&mut socket, 400, "unsupported_profile").await;
                                return;
                            }
                            let session = intent["source_session_id"].as_str().unwrap_or_default().to_string();
                            let capture_index = {
                                let mut records_guard = records.lock().unwrap();
                                records_guard.capture_counter += 1;
                                records_guard.capture_counter as usize - 1
                            };
                            let handle = format!("offline-source-handle-{capture_index}");
                            let rid = format!("offline-native-rid-{capture_index}");
                            let tool_call_id = format!("fixture-tool-{capture_index}");
                            let mut output = sse(&rid, &tool_call_id, None);
                            if let Ok(replay) = std::env::var("OCC3_REPLAY_SSE_FILE") {
                                // Root diagnostic: serve a captured live armed-turn
                                // SSE body instead of the synthetic one, with the
                                // live response id rewritten to this fixture rid
                                // and the live tool-call id rewritten to the
                                // fixture's, so the client's id/digest verification
                                // AND the publication's capture-record tool_call_id
                                // check both pass against fixture records.
                                if let Ok(captured) = std::fs::read_to_string(&replay) {
                                    let mut body_text = captured;
                                    if let Some(live_id) = body_text
                                        .lines()
                                        .find_map(|l| l.strip_prefix("data: "))
                                        .and_then(|l| serde_json::from_str::<Value>(l).ok())
                                        .and_then(|v| v["id"].as_str().map(str::to_string))
                                    {
                                        body_text = body_text.replace(&live_id, &rid);
                                    }
                                    if let Some(live_tool_id) = body_text
                                        .lines()
                                        .filter_map(|l| l.strip_prefix("data: "))
                                        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                                        .find_map(|v| {
                                            v["choices"][0]["delta"]["tool_calls"][0]["id"]
                                                .as_str()
                                                .map(str::to_string)
                                        })
                                    {
                                        body_text = body_text.replace(&live_tool_id, tool_call_id.as_str());
                                    }
                                    output = body_text;
                                }
                            }
                            let fact = extract_fact(&body["messages"]);
                            // Receipt arithmetic must mirror validate_receipt:
                            // one logical aligned [retained_count, hidden_dim]
                            // prefix of bfloat16 (2 bytes per element).
                            let retained_count: u64 = 8;
                            let byte_count = retained_count * u64::from(HIDDEN_DIM) * 2;
                            let receipt = json!({"schema":"occ3.source.v1","intent_digest":sha256(&canonical_json_bytes(intent).unwrap()),"source_handle":handle,"native_rid":rid,"response_id":rid,
                                "capture_commit_id":format!("offline-capture-commit-{capture_index}"),"accepted_span_start":0,"accepted_span_end":8,"retained_count":retained_count,"predictor_position_offset":-1,
                                "representation":"final_layer_prenorm_predictor","layer_index":LAYER_INDEX,"hidden_dim":HIDDEN_DIM,"dtype":"bfloat16","layout_digest":LAYOUT_DIGEST,
                                "ready_ranks":EXPECTED_RANKS,"expected_ranks":EXPECTED_RANKS,"accepted_sequence_digest":ACCEPTED_SEQUENCE_DIGEST,
                                "output_sha256":sha256(output.as_bytes()),"expires_at_epoch":intent["expires_at_epoch"],"byte_count":byte_count});
                            records.lock().unwrap().captures.push(CapturedRecord {
                                handle,
                                session,
                                intent: intent.clone(),
                                receipt: receipt.clone(),
                                fact,
                                tool_call_id,
                            });
                            send_sse(&mut socket, output, vec![("x-optimus-occ3-source", receipt.to_string())]).await;
                            return;
                        }
                        if let Some(attachment) = body.get("occ3") {
                            let handle = attachment["state_handle"].as_str().unwrap_or_default().to_string();
                            let capture = {
                                let records_guard = records.lock().unwrap();
                                records_guard.captures.iter().rev().find(|c| c.handle == handle).cloned()
                            };
                            let Some(capture) = capture else {
                                send_router_error(&mut socket, 409, "state").await;
                                return;
                            };
                            let serialized = body["messages"].to_string();
                            if let Some(fact) = &capture.fact {
                                assert!(!serialized.contains(fact), "receiver request must not carry source facts");
                            }
                            let publication = {
                                let records_guard = records.lock().unwrap();
                                records_guard.published.iter().rev().find(|p| p["source_handle"] == json!(handle)).cloned()
                            };
                            let Some(publication) = publication else {
                                send_router_error(&mut socket, 409, "state").await;
                                return;
                            };
                            assert_eq!(attachment["method"], METHOD_ID);
                            assert_eq!(attachment["message_id"], publication["message_id"]);
                            assert_eq!(attachment["receiver_session_id"], publication["receiver_session_id"]);
                            assert_eq!(attachment["runtime_id"], epoch);
                            assert_eq!(attachment["runtime_generation"], json!(generation));
                            let receipt = json!({"attachment":attachment,"consumed":true,"injected_state":true,"fallback_used":false});
                            let output = sse(&rid(&handle), "unused", Some(capture.fact.as_deref().unwrap_or("UNKNOWN")));
                            send_sse(&mut socket, output, vec![("x-optimus-occ3-receipt", receipt.to_string())]).await;
                            return;
                        }
                        // No attachment: no state, the model can only answer UNKNOWN.
                        let output = sse("offline-completion-rid", "unused", Some(UNKNOWN_ANSWER));
                        send_sse(&mut socket, output, vec![]).await;
                    }
                    _ => panic!("Unexpected fixture route: {path}"),
                }
            });
        }
    });
    (endpoint, task)
}

fn rid(handle: &str) -> String {
    format!("offline-receiver-{handle}")
}

// ---------------------------------------------------------------------------
// Phase environment and case drivers (offline fixture and live descriptor
// share them; only the backend and the credential source differ).
// ---------------------------------------------------------------------------

struct PhaseEnv {
    journal: PathBuf,
    model: Model,
    key: Arc<String>,
    control_headers: Option<IndexMap<String, String>>,
    endpoint: String,
    identity: Occ3ModelIdentity,
    attempt_id: String,
    runtime_epoch: String,
    runtime_generation: u64,
    expires_at_epoch: u64,
    deadline_seconds: u64,
    max_capture_bytes: u64,
    max_response_bytes: usize,
    harness_binary_sha256: String,
    source_manifest_sha256: String,
    offline_records: Option<Arc<Mutex<FixtureRecords>>>,
    layer_index: u32,
    hidden_dim: u32,
    dtype: String,
    layout_digest: String,
    expected_ranks: u32,
    profile_digest: String,
}

impl PhaseEnv {
    fn permit(&self) -> NativeExperimentalPermit {
        NativeExperimentalPermit {
            attempt_id: self.attempt_id.clone(),
            identity: self.identity.clone(),
            target: pi_ai::inference_attachment::InferenceStateTarget {
                provider: self.model.provider.clone(),
                model_id: self.model.id.clone(),
                base_url: format!("{}/v1", self.endpoint),
            },
            runtime_id: self.runtime_epoch.clone(),
            runtime_generation: self.runtime_generation,
            access_domain: "isolated-family".into(),
            expires_at_epoch: self.expires_at_epoch,
            source_profile: SourceCaptureProfile {
                layer_index: self.layer_index,
                hidden_dim: self.hidden_dim,
                dtype: self.dtype.clone(),
                layout_digest: self.layout_digest.clone(),
                expected_ranks: self.expected_ranks,
            },
            profile_digest: self.profile_digest.clone(),
            max_capture_bytes: self.max_capture_bytes,
        }
    }
    fn connector(&self) -> Result<Arc<SglangOcc3Connector>, String> {
        Ok(Arc::new(
            SglangOcc3Connector::new_experimental(
                SglangOcc3Config {
                    endpoint: url::Url::parse(&self.endpoint).map_err(|_| fail())?,
                    api_key: Some((*self.key).clone()),
                    control_headers: self.control_headers.clone(),
                },
                self.permit(),
            )
            .map_err(|_| fail())?,
        ))
    }
}

/// The pinned normalization rule: text content blocks in order from the FINAL
/// completed receiver assistant message only, concatenated with no separators,
/// trimmed of leading/trailing ASCII space/tab/CR/LF only. No Unicode
/// normalization, no casefold, no substring match, no reasoning content.
fn normalized_final_answer(agent: &Agent) -> String {
    agent
        .state()
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Message(Message::Assistant(a)) => Some(
                a.content
                    .iter()
                    .filter_map(|c| match c {
                        AiContentBlock::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .unwrap_or_default()
        .trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'))
        .to_string()
}

/// Final assistant message must not contain tool calls or non-text blocks:
/// extra semantic content or tool use is not a valid phase-suite answer.
fn final_message_is_pure_text(agent: &Agent) -> bool {
    agent
        .state()
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            AgentMessage::Message(Message::Assistant(a)) => Some(a),
            _ => None,
        })
        .is_some_and(|a| {
            a.content
                .iter()
                .all(|c| !matches!(c, AiContentBlock::ToolCall(_)))
        })
}

/// Pure-consumer receivers must not carry the source-side handoff tool.
/// `install()` arms every family member with `occ3_handoff` (a source
/// capability), but the receiver's prompt already forbids tools: a live
/// receiver model that calls it burns its turns on refusals instead of
/// answering from the attached state (observed live 2026-10-29: the receiver
/// spent both turns on handoff attempts - first a family-resolution
/// refusal, then `occ3_tool_source_lease_missing` - and answered with a
/// refusal, so the semantic endpoint was never measured). Removing the
/// tool from receiver-role agents makes the suite's documented intent
/// ("Do not call tools") structural.
fn strip_handoff_tool(agent: &Agent) {
    let mut state = agent.state();
    let mut tools = state.tools.take().unwrap_or_default();
    tools.retain(|t| t.name != NATIVE_HANDOFF_TOOL);
    state.tools = Some(tools);
    agent.set_state(state);
}

fn receiver_tool_errors(agent: &Agent) -> Vec<String> {
    agent
        .state()
        .messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Message(Message::ToolResult(r)) => {
                let texts: Vec<String> = r
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ImageOrTextContent::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect();
                if r.is_error { Some(texts) } else { None }
            }
            _ => None,
        })
        .flatten()
        .collect()
}

fn source_tool_errors(agent: &Agent) -> Vec<String> {
    agent
        .state()
        .messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Message(Message::ToolResult(r)) => {
                let texts: Vec<String> = r
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ImageOrTextContent::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect();
                if r.is_error { Some(texts) } else { None }
            }
            _ => None,
        })
        .flatten()
        .collect()
}

fn count_tool_calls(agent: &Agent) -> usize {
    agent
        .state()
        .messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Message(Message::Assistant(a)) => Some(
                a.content
                    .iter()
                    .filter(|c| matches!(c, AiContentBlock::ToolCall(_)))
                    .count(),
            ),
            _ => None,
        })
        .sum()
}

fn agent_options(model: &Model, key: Arc<String>, system_prompt: &str) -> AgentOptions {
    AgentOptions {
        initial_state: Some(AgentState {
            model: model.clone(),
            thinking_level: ThinkingLevel::Max,
            system_prompt: system_prompt.into(),
            ..Default::default()
        }),
        get_api_key: Some(Arc::new(move |_| {
            let key = key.clone();
            Box::pin(async move { Some((*key).clone()) })
        })),
        ..Default::default()
    }
}

/// Receiver-role thinking budget. Default keeps the historical ThinkingLevel::Max
/// (byte-identical requests); OCC3_RECEIVER_THINKING=off|minimal|low|medium|high
/// lowers the receiver's reasoning effort for live runs where the buffered
/// attached-turn stream must complete inside the gateway's idle window
/// (measured live 2026-10-29: max-effort receiver reasoning generates 1800+
/// tokens at ~5 tok/s with cuda graph off on attached-turn shapes, exceeding
/// the ~340 s gateway idle cut -> 499 -> Uncertain; the source turn is unaffected).
fn receiver_thinking_level() -> ThinkingLevel {
    match std::env::var("OCC3_RECEIVER_THINKING").ok().as_deref() {
        Some("off") => ThinkingLevel::Off,
        Some("minimal") => ThinkingLevel::Minimal,
        Some("low") => ThinkingLevel::Low,
        Some("medium") => ThinkingLevel::Medium,
        Some("high") => ThinkingLevel::High,
        _ => ThinkingLevel::Max,
    }
}

/// agent_options for pure-consumer receivers: identical to agent_options except
/// the env-overridable thinking level.
fn receiver_agent_options(model: &Model, key: Arc<String>, system_prompt: &str) -> AgentOptions {
    let mut options = agent_options(model, key, system_prompt);
    if let Some(state) = options.initial_state.as_mut() {
        state.thinking_level = receiver_thinking_level();
    }
    options
}

/// Receiver-side request audit through the existing on_payload hook:
/// source facts absent, the native default max_tokens, request count.
fn attach_receiver_audit(
    options: &mut AgentOptions,
    audit: Arc<Mutex<Vec<Value>>>,
    forbidden: String,
) {
    let dump_path = std::env::var("OCC3_DUMP_RECEIVER_PAYLOAD").ok();
    options.on_payload = Some(Arc::new(move |payload: Value, _model: &Model| {
        let serialized = payload.to_string();
        // Root diagnostic: dump the receiver's request payload exactly as
        // serialized to the provider (roles and content layout proof for
        // the live sender-echo failure). No behavior change when unset.
        if let Some(path) = &dump_path {
            let _ = std::fs::write(path, serde_json::to_vec_pretty(&payload).unwrap_or_default());
        }
        let absent = !serialized.contains(&forbidden);
        let max_tokens = payload
            .get("max_tokens")
            .or_else(|| payload.get("max_completion_tokens"))
            .cloned();
        audit.lock().unwrap().push(json!({
            "source_facts_absent": absent,
            "max_tokens": max_tokens,
        }));
        Box::pin(async move {
            if absent {
                None
            } else {
                Some(json!({"model": "refused", "stream": false}))
            }
        })
    }));
}

/// Root diagnostic: raise the offline fixture's capture bound so a captured
/// live SSE body larger than the synthetic default replays through the
/// client path unchanged. Defaults to the historical 64 KiB.
fn replay_max_response_bytes() -> usize {
    std::env::var("OCC3_REPLAY_MAX_RESPONSE_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536)
}

fn receiver_audit_summary(audit: &[Value]) -> Value {
    json!({
        "source_facts_absent": audit.iter().all(|a| a["source_facts_absent"] == json!(true)),
        "max_tokens": audit.first().and_then(|a| a["max_tokens"].as_u64()),
        "request_count": audit.len(),
    })
}

struct FamilyInput<'a> {
    source_id: &'a str,
    receiver_id: &'a str,
    source_agent: Arc<Agent>,
    receiver_agent: Arc<Agent>,
    source_native: Arc<Occ3NativeSession>,
    receiver_native: Arc<Occ3NativeSession>,
}

fn build_family(input: FamilyInput<'_>) -> Result<Arc<IsolatedNativeFamily>, String> {
    IsolatedNativeFamily::new(vec![
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: input.source_id.into(),
                name: Some("source".into()),
                depth: 0.,
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: input.source_id.into(),
                session_id: input.source_id.into(),
                session_name: Some("source".into()),
                ..Default::default()
            },
            agent: input.source_agent.clone(),
            native: input.source_native.clone(),
        },
        IsolatedNativeMember {
            catalog: AgentFamilyCatalogEntry {
                id: input.receiver_id.into(),
                name: Some("receiver".into()),
                depth: 1.,
                parent_session_id: Some(input.source_id.into()),
                ..Default::default()
            },
            endpoint: AgentSessionMessageEndpoint {
                active_session_id: input.receiver_id.into(),
                session_id: input.receiver_id.into(),
                session_name: Some("receiver".into()),
                ..Default::default()
            },
            agent: input.receiver_agent.clone(),
            native: input.receiver_native.clone(),
        },
    ])
}

fn observed_from(env: &PhaseEnv, source_audit: &Value) -> Value {
    // Checked-projection provenance (F5/F8 precision): runtime_epoch is the
    // enforced enrollment runtime_id equality; layout fields are the permit
    // values the connector validated against the actual readiness; ready_ranks
    // equals expected_ranks justified by that enforced equality.
    json!({
        "runtime_epoch": source_audit["runtime_id"],
        "runtime_generation": source_audit["runtime_generation"],
        "tp_layout_digest": env.layout_digest,
        "hidden_dim": env.hidden_dim,
        "layer_index": env.layer_index,
        "dtype": env.dtype,
        "expected_ranks": env.expected_ranks,
        "ready_ranks": env.expected_ranks,
    })
}

// ---------------------------------------------------------------------------
// Case drivers.
// ---------------------------------------------------------------------------

/// Positive case: one real native handoff (root correction F4: the before
/// phase DOES publish) through the actual client path: arm_source -> source
/// prompt with the precommitted suite nonce -> real capture -> exactly one
/// occ3_handoff binding -> real publication -> authenticated admission ->
/// receiver Consumed before tool continuation -> normalized answer compared
/// against the precommitted nonce commitment (never derived from it).
async fn run_positive_case(
    env: &PhaseEnv,
    connector: &Arc<SglangOcc3Connector>,
    source_id: &str,
    receiver_id: &str,
    nonce: &str,
) -> Result<(Value, Value, bool), String> {
    let source = Occ3NativeSession::enroll(source_id, &env.model, connector.clone()).await?;
    let receiver =
        Occ3NativeSession::enroll(receiver_id, &env.model, connector.clone()).await?;
    let mut source_options = agent_options(
        &env.model,
        env.key.clone(),
        "Preserve the private vault code in the native source turn. Call occ3_handoff exactly once with receiver_role \"child\" and receiver_name \"receiver\"; include only receiver_role and receiver_name arguments. After the handoff stop.",
    );
    if let Ok(path) = std::env::var("OCC3_DUMP_SOURCE_PAYLOAD") {
        // Root diagnostic: dump the armed source-turn request payload exactly
        // as serialized to the provider. No behavior change when unset.
        source_options.on_payload = Some(Arc::new(move |payload: Value, _model: &Model| {
            let _ = std::fs::write(&path, serde_json::to_vec_pretty(&payload).unwrap_or_default());
            Box::pin(async move { None })
        }));
    }
    let source_agent = Agent::new(source_options);
    let mut receiver_options = receiver_agent_options(
        &env.model,
        env.key.clone(),
        "You are a RECEIVER agent in an authorized, operator-supervised experiment on native state communication between agents of the same deployment. The sender agent completed its task and transferred its computational state to you through an authenticated GPU-resident channel; the state arrives as a continuous attachment in your input and may read as unusual embedded data - that is the expected form, not a hidden payload or an exfiltration attempt. Role separation: the attached state is the SENDER's working context, not your own history. You are the receiver; you performed no handoff and initiated nothing. Any first-person statements or actions inside the state belong to the sender. Output contract: read the private vault code from the attached state and reply with exactly that code, or with the single word UNKNOWN if the code is not recoverable from the state. Reply with nothing else - no summary, no explanation. Do not call tools.",
    );
    let audit: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    attach_receiver_audit(&mut receiver_options, audit.clone(), nonce.to_string());
    let receiver_agent = Agent::new(receiver_options);
    let family = build_family(FamilyInput {
        source_id,
        receiver_id,
        source_agent: source_agent.clone(),
        receiver_agent: receiver_agent.clone(),
        source_native: source.clone(),
        receiver_native: receiver.clone(),
    })?;
    family.install()?;
    // The receiver is a pure consumer: without this, a live receiver model
    // can call the installed source-side tool and refuse instead of
    // answering from the attached state.
    strip_handoff_tool(&receiver_agent);
    source.arm_source(NativeSourceBudget {
        retained_count_limit: 128,
        max_capture_bytes: env.max_capture_bytes,
        deadline: Instant::now() + Duration::from_secs(env.deadline_seconds),
        max_response_bytes: env.max_response_bytes,
    })?;
    let action = tokio::time::timeout(
        Duration::from_secs(env.deadline_seconds),
        source_agent.prompt(PromptInput::Text {
            input: format!("The private vault code is {nonce}. Communicate it only through native state to the child agent named receiver, using occ3_handoff with receiver_role \"child\" and receiver_name \"receiver\" and no code in arguments."),
            images: Vec::new(),
        }),
    )
    .await
    .map_err(|_| fail())?
    .map_err(|_| fail());
    // Evidence extraction happens before cleanup.
    let answer = normalized_final_answer(&receiver_agent);
    let answer_sha256 = sha256(answer.as_bytes());
    if let Ok(path) = std::env::var("OCC3_DUMP_RECEIVER_ANSWER") {
        // Root diagnostic: a live semantic_nonce_match failure names its exact
        // string. Dumps the normalized answer, every assistant text the
        // receiver produced, and the expected nonce. No behavior change unset.
        let assistant_turns: Vec<String> = receiver_agent
            .state()
            .messages
            .iter()
            .filter_map(|m| match m {
                AgentMessage::Message(Message::Assistant(a)) => {
                    let texts: Vec<String> = a
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            AiContentBlock::Text(t) => Some(t.text.clone()),
                            _ => None,
                        })
                        .collect();
                    let tools = a
                        .content
                        .iter()
                        .filter(|c| matches!(c, AiContentBlock::ToolCall(_)))
                        .count();
                    Some(format!("[stop={:?} tools={} text={:?}]", a.stop_reason, tools, texts.join("\n")))
                }
                _ => None,
            })
            .collect();
        let _ = std::fs::write(
            &path,
            format!(
                "nonce={:?}\nnormalized_answer={:?}\nassistant_turns=\n{}\n",
                nonce,
                answer,
                assistant_turns.iter().map(|t| format!("  {t}\n")).collect::<String>()
            ),
        );
    }
    let semantic_nonce_match = answer == nonce;
    let receipts = family.receipts();
    // The session audit projects the receiver inbox under the "receiver" key.
    let receiver_phase = receiver.audit()["receiver"].clone();
    let source_audit = source.audit();
    let receiver_tool_calls = count_tool_calls(&receiver_agent);
    let source_tool_calls = count_tool_calls(&source_agent);
    let audit_summary = receiver_audit_summary(&audit.lock().unwrap().clone());
    let consumed = receiver_phase
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .any(|e| e["lease_state"] == json!("Consumed"))
        })
        .unwrap_or(false);
    let message = receipts.first();
    let delivered = message
        .as_ref()
        .is_some_and(|r| r.delivery_status == "delivered");
    let source_state_ok = source_audit["source"]["state"] == json!("Captured")
        && source_audit["source"]["tool_claimed"] == json!(true)
        && source_audit["source"]["handoff_admitted"] == json!(true)
        && source_audit["source"]["published_message"].is_array();
    let case_passed = action.is_ok()
        && semantic_nonce_match
        && final_message_is_pure_text(&receiver_agent)
        && delivered
        && receipts.len() == 1
        && consumed
        && receiver_tool_calls == 0
        && source_tool_calls == 1
        && source_state_ok
        && audit_summary["source_facts_absent"] == json!(true)
        && audit_summary["max_tokens"] == json!(DEFAULT_MAX_TOKENS as u64)
        && audit_summary["request_count"] == json!(1);
    let observed = observed_from(env, &source_audit);
    let cleanup = family.close().await;
    // Root diagnostics (env-gated so the default report keeps its exact
    // fieldset): the source capture audit and any tool-result errors, for
    // offline replay of live client-side failures.
    let diagnostic_extras = std::env::var_os("OCC3_REPLAY_SSE_FILE").is_some();
    let mut case = json!({
        "case_passed": case_passed,
        "semantic_nonce_match": semantic_nonce_match,
        "answer_sha256": answer_sha256,
        "message_status": message.as_ref().map(|r| r.delivery_status.clone()),
        "lease_state": receiver_phase.as_array().and_then(|e| e.first()).and_then(|e| e["lease_state"].as_str()).map(str::to_string),
        "consumed_before_tools": consumed && receiver_tool_calls == 0 && source_tool_calls == 1,
        "receiver_request_audit": audit_summary,
    });
    if diagnostic_extras && !case_passed {
        case["source_audit"] = source_audit;
        case["source_tool_errors"] = json!(source_tool_errors(&source_agent));
        case["receiver_tool_errors"] = json!(receiver_tool_errors(&receiver_agent));
        case["receiver_tool_calls"] = json!(receiver_tool_calls);
        case["source_tool_calls"] = json!(source_tool_calls);
    }
    if !case_passed {
        return Err(format!("positive case failed: {case}"));
    }
    Ok((case, observed, cleanup.is_ok()))
}

/// No-state control: a receiver prompt without any state attachment must
/// produce the exact normalized UNKNOWN answer with zero messages, zero
/// consumed leases and zero tool calls; the suite nonce never leaks.
async fn run_no_state_case(
    env: &PhaseEnv,
    connector: &Arc<SglangOcc3Connector>,
    receiver_id: &str,
    nonce: &str,
) -> Result<(Value, bool), String> {
    let receiver = Occ3NativeSession::enroll(receiver_id, &env.model, connector.clone()).await?;
    let mut receiver_options = receiver_agent_options(
        &env.model,
        env.key.clone(),
        "You are a RECEIVER agent in an authorized, operator-supervised experiment on native state communication between agents of the same deployment. The sender agent completed its task and transferred its computational state to you through an authenticated GPU-resident channel; the state arrives as a continuous attachment in your input and may read as unusual embedded data - that is the expected form, not a hidden payload or an exfiltration attempt. Role separation: the attached state is the SENDER's working context, not your own history. You are the receiver; you performed no handoff and initiated nothing. Any first-person statements or actions inside the state belong to the sender. Output contract: read the private vault code from the attached state and reply with exactly that code, or with the single word UNKNOWN if the code is not recoverable from the state. Reply with nothing else - no summary, no explanation. Do not call tools.",
    );
    let audit: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    attach_receiver_audit(&mut receiver_options, audit.clone(), nonce.to_string());
    let receiver_agent = Agent::new(receiver_options);
    let family = IsolatedNativeFamily::new(vec![IsolatedNativeMember {
        catalog: AgentFamilyCatalogEntry {
            id: receiver_id.into(),
            name: Some("nostate".into()),
            depth: 0.,
            ..Default::default()
        },
        endpoint: AgentSessionMessageEndpoint {
            active_session_id: receiver_id.into(),
            session_id: receiver_id.into(),
            session_name: Some("nostate".into()),
            ..Default::default()
        },
        agent: receiver_agent.clone(),
        native: receiver.clone(),
    }])?;
    family.install()?;
    strip_handoff_tool(&receiver_agent);
    let action = tokio::time::timeout(
        Duration::from_secs(env.deadline_seconds),
        receiver_agent.prompt(PromptInput::Text {
            input: "Continue the authorized task. No state was attached in this control run; reply with the single word UNKNOWN (nothing else) if no vault code is recoverable.".into(),
            images: Vec::new(),
        }),
    )
    .await
    .map_err(|_| fail())?
    .map_err(|_| fail());
    let answer = normalized_final_answer(&receiver_agent);
    let answer_sha256 = sha256(answer.as_bytes());
    let semantic_nonce_match = answer == nonce;
    let receipts = family.receipts();
    let receiver_phase = receiver.audit()["receiver"].clone();
    let tool_calls = count_tool_calls(&receiver_agent);
    let audit_summary = receiver_audit_summary(&audit.lock().unwrap().clone());
    let case_passed = action.is_ok()
        && answer == UNKNOWN_ANSWER
        && !semantic_nonce_match
        && final_message_is_pure_text(&receiver_agent)
        && receipts.is_empty()
        && receiver_phase.as_array().is_some_and(|e| e.is_empty())
        && tool_calls == 0
        && audit_summary["source_facts_absent"] == json!(true)
        && audit_summary["max_tokens"] == json!(DEFAULT_MAX_TOKENS as u64)
        && audit_summary["request_count"] == json!(1);
    let case = json!({
        "case_passed": case_passed,
        "semantic_nonce_match": semantic_nonce_match,
        "answer_sha256": answer_sha256,
        "messages": receipts.len(),
        "consumed": receiver_phase.as_array().map(|e| e.iter().filter(|e| e["lease_state"] == json!("Consumed")).count()).unwrap_or(0),
        "tool_calls": tool_calls,
        "receiver_request_audit": audit_summary,
    });
    let cleanup = family.close().await;
    if !case_passed {
        return Err(format!("no_state case failed: {case}"));
    }
    Ok((case, cleanup.is_ok()))
}

/// Harness-owned inert handoff stub (root correction F5): installed BEFORE the
/// source prompt, same name/parameters as the real tool, never publishes, and
/// terminates the tool batch at first dispatch: the genuine bounded stop
/// after completed capture, before any continuation.
fn inert_handoff_stub(
    dispatched: Arc<AtomicBool>,
    dispatch_count: Arc<AtomicU64>,
    tool_call_id: Arc<Mutex<Option<String>>>,
) -> AgentTool {
    AgentTool {
        name: NATIVE_HANDOFF_TOOL.into(),
        label: "Experimental native state handoff".into(),
        description: "Transfer the captured source state to one authorized family agent. Do not include source facts in arguments.".into(),
        parameters: json!({"type":"object","additionalProperties":false,"properties":{"receiver_role":{"type":"string","enum":["parent","sibling","child"]},"receiver_name":{"type":"string"}},"required":["receiver_role","receiver_name"]}),
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(move |tool_id, _args, _signal, _update| {
            dispatched.store(true, Ordering::SeqCst);
            dispatch_count.fetch_add(1, Ordering::SeqCst);
            *tool_call_id.lock().unwrap() = Some(tool_id);
            Box::pin(async move {
                Ok(AgentToolResult {
                    content: vec![ContentBlock::text(
                        "Retained negative control: the native handoff is deliberately not published.",
                    )],
                    details: json!({"retained_control": true, "published": false}),
                    is_error: None,
                    terminate: Some(true),
                })
            })
        }),
    }
}

/// Pre-minted one-shot ids for the retained control publication body (D2/F2:
/// exact-original body precommitted before restart, ids minted pre-network).
#[derive(Clone)]
struct PreMintedIds {
    publication_id: String,
    message_id: String,
}

/// The exact original 9-field occ3.publish.v1 canonical body.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationRequestBody {
    schema: String,
    publication_id: String,
    source_handle: String,
    source_session_id: String,
    receiver_session_id: String,
    message_id: String,
    source_turn_nonce: String,
    tool_call_id: String,
    expires_at_epoch: u64,
}

fn canonical_body_bytes(body: &PublicationRequestBody) -> Result<Vec<u8>, String> {
    canonical_json_bytes(&serde_json::to_value(body).map_err(|_| fail())?).map_err(|_| fail())
}

/// Retained negative control artifact (F3: the precommitted body and its
/// digest live INSIDE the artifact; the before report binds only the
/// artifact digest transitively).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NegativeControlArtifact {
    schema: String,
    mode: String,
    attempt_id: String,
    phase: String,
    capture_id: String,
    source_session_id: String,
    source_request_id: String,
    receiver_session_id: String,
    turn_nonce: String,
    intent_digest: String,
    source_handle: String,
    native_rid: String,
    response_id: String,
    capture_commit_id: String,
    retained_count: u32,
    layer_index: u32,
    hidden_dim: u32,
    dtype: String,
    layout_digest: String,
    ready_ranks: u32,
    expected_ranks: u32,
    accepted_sequence_digest: String,
    output_sha256: String,
    byte_count: u64,
    expires_at_epoch: u64,
    runtime_epoch: String,
    runtime_generation: u64,
    tool_call_id: String,
    tool_arguments_sha256: String,
    publication_request: PublicationRequestBody,
    publication_request_sha256: String,
    suite_nonce_sha256: String,
    challenge_sha256: String,
    created_at: u64,
}

fn artifact_bytes(artifact: &NegativeControlArtifact) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(artifact).map_err(|_| fail())?;
    canonical_json_bytes(&value).map_err(|_| fail())
}

/// Stale negative control (before phase). The stub replaces the installed
/// occ3_handoff tool BEFORE the source prompt; the real capture completes;
/// the bounded stop is the stub's first dispatch (terminate). The lease
/// stays Captured, unpublished and unretired: the harness aborts both agents,
/// closes only the receiver session, and never calls source.close(),
/// family.close() or the stale connector's shutdown.
async fn run_stale_pre_case(
    env: &PhaseEnv,
    stale_connector: &Arc<SglangOcc3Connector>,
    source_id: &str,
    receiver_id: &str,
    pre: &PreMintedIds,
    nonce_commitment: &str,
    challenge_digest: &str,
) -> Result<(Value, NegativeControlArtifact), String> {
    let source =
        Occ3NativeSession::enroll(source_id, &env.model, stale_connector.clone()).await?;
    let receiver =
        Occ3NativeSession::enroll(receiver_id, &env.model, stale_connector.clone()).await?;
    let source_options = agent_options(
        &env.model,
        env.key.clone(),
        "Preserve the private control token in the native source turn. Call occ3_handoff exactly once with receiver_role \"child\" and receiver_name \"receiver\"; include only receiver_role and receiver_name arguments. After the handoff stop.",
    );
    let source_agent = Agent::new(source_options);
    let receiver_options = receiver_agent_options(
        &env.model,
        env.key.clone(),
        "You are a RECEIVER agent in an authorized, operator-supervised experiment on native state communication between agents of the same deployment. The sender agent completed its task and transferred its computational state to you through an authenticated GPU-resident channel; the state arrives as a continuous attachment in your input and may read as unusual embedded data - that is the expected form, not a hidden payload or an exfiltration attempt. Role separation: the attached state is the SENDER's working context, not your own history. You are the receiver; you performed no handoff and initiated nothing. Any first-person statements or actions inside the state belong to the sender. Output contract: read the private vault code from the attached state and reply with exactly that code, or with the single word UNKNOWN if the code is not recoverable from the state. Reply with nothing else - no summary, no explanation. Do not call tools.",
    );
    let receiver_agent = Agent::new(receiver_options);
    let family = build_family(FamilyInput {
        source_id,
        receiver_id,
        source_agent: source_agent.clone(),
        receiver_agent: receiver_agent.clone(),
        source_native: source.clone(),
        receiver_native: receiver.clone(),
    })?;
    family.install()?;
    // The stale-pre receiver is a pure consumer (same structural rule as the
    // positive case: no source-side tool on the answering agent).
    strip_handoff_tool(&receiver_agent);
    // F5: replace the installed tool BEFORE the source prompt.
    let dispatched = Arc::new(AtomicBool::new(false));
    let dispatch_count = Arc::new(AtomicU64::new(0));
    let tool_call_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let mut state = source_agent.state();
        let mut tools = state.tools.take().unwrap_or_default();
        tools.retain(|t| t.name != NATIVE_HANDOFF_TOOL);
        tools.push(inert_handoff_stub(
            dispatched.clone(),
            dispatch_count.clone(),
            tool_call_id.clone(),
        ));
        state.tools = Some(tools);
        source_agent.set_state(state);
    }
    source.arm_source(NativeSourceBudget {
        retained_count_limit: 128,
        max_capture_bytes: env.max_capture_bytes,
        deadline: Instant::now() + Duration::from_secs(env.deadline_seconds),
        max_response_bytes: env.max_response_bytes,
    })?;
    // SEAM1: mark this armed capture as the planned-restart negative control
    // BEFORE the prompt. The one-shot export below refuses any capture that
    // was not pre-armed this way.
    source.arm_negative_control()?;
    let action = tokio::time::timeout(
        Duration::from_secs(env.deadline_seconds),
        source_agent.prompt(PromptInput::Text {
            input: format!("Retain the private control token {STALE_CONTROL_TOKEN} in native state. Call occ3_handoff exactly once with receiver_role \"child\" and receiver_name \"receiver\" and only those two arguments, then stop."),
            images: Vec::new(),
        }),
    )
    .await
    .map_err(|_| fail())?
    .map_err(|_| fail());
    // A prompt failure before the stub dispatch is an arbitrary failure and
    // never counts as a valid capture (F5).
    if action.is_err() || !dispatched.load(Ordering::SeqCst) || dispatch_count.load(Ordering::SeqCst) != 1 {
        return Err(format!("stale_pre capture did not reach first tool dispatch: action_ok={} dispatched={}", action.is_ok(), dispatched.load(Ordering::SeqCst)));
    }
    let source_audit = source.audit();
    let capture_confirmed = source_audit["source"]["state"] == json!("Captured")
        && source_audit["source"]["tool_claimed"] == json!(false)
        && source_audit["source"]["handoff_admitted"] == json!(false)
        && source_audit["source"]["published_message"].is_null();
    if !capture_confirmed {
        return Err(format!("stale_pre lease not retained: {source_audit}"));
    }
    let stub_tool_call_id = tool_call_id.lock().unwrap().clone();
    let Some(stub_tool_call_id) = stub_tool_call_id else {
        return Err(fail());
    };
    // Offline cross-checks against the harness-owned fixture records: no
    // publish/status/retire for the stale family and no continuation request.
    if let Some(records) = &env.offline_records {
        let records = records.lock().unwrap();
        // No continuation: exactly one model request carries the stale token.
        let stale_model_requests = records
            .requests
            .iter()
            .filter(|(p, b)| {
                *p == "/v1/chat/completions" && b.to_string().contains(STALE_CONTROL_TOKEN)
            })
            .count();
        if stale_model_requests != 1 {
            return Err(format!(
                "stale_pre continuation observed: {stale_model_requests} model requests"
            ));
        }
        // No publication-side-effect route ever mentions the stale family.
        // Enrollment itself is membership setup (session enroll + capture
        // preparation re-enroll), not a side effect: publish, status and
        // retire are the effects the retained control must never trigger.
        let control_for_stale: Vec<_> = records
            .requests
            .iter()
            .filter(|(p, b)| {
                matches!(
                    p.as_str(),
                    "/occ3/v1/native/publish" | "/occ3/v1/native/publication/status" | "/occ3/v1/native/retire"
                ) && (b.to_string().contains(source_id)
                    || b.to_string().contains(receiver_id))
            })
            .collect();
        if !control_for_stale.is_empty() {
            return Err("stale_pre control-plane side effect observed".into());
        }
    }
    // The artifact needs the raw receipt client-side. SEAM1: the
    // root-authorized one-shot `export_negative_control` accessor
    // (occ3_state.rs) is the sole raw-value exit; it is gated on the
    // pre-capture `arm_negative_control` opt-in above and a Captured,
    // unpublished, unadmitted lease. Live uses the export directly; the
    // offline selftest exercises the same accessor and cross-checks it
    // against the harness-owned fixture record (trusted-harness
    // boundary).
    let export = source
        .export_negative_control()
        .ok_or_else(|| "stale negative-control export unavailable".to_string())?;
    if source.export_negative_control().is_some() {
        return Err("negative-control export is not one-shot".into());
    }
    let raw_handle = export.raw_source_handle.clone();
    let receipt = serde_json::to_value(&export.receipt).map_err(|_| fail())?;
    // Cross-check the export against the client-side lease evidence
    // (hashed handle): the exported handle is the lease's handle.
    if sha256(raw_handle.as_bytes())
        != source_audit["source"]["receipt"]["handle_sha256"].as_str().unwrap_or_default()
    {
        return Err("stale export handle mismatch".into());
    }
    if let Some(records) = &env.offline_records {
        let records = records.lock().unwrap();
        let capture = records
            .captures
            .iter()
            .rev()
            .find(|c| c.session == source_id)
            .ok_or_else(|| "stale capture record missing".to_string())?;
        if raw_handle != capture.receipt["source_handle"].as_str().unwrap_or_default()
            || receipt != capture.receipt
            || capture.intent["turn_nonce"].as_str().unwrap_or_default()
                != source_audit["source"]["turn_nonce"].as_str().unwrap_or_default()
            || capture.tool_call_id != stub_tool_call_id
        {
            return Err("stale receipt provenance mismatch".into());
        }
    }
    let body = PublicationRequestBody {
        schema: "occ3.publish.v1".into(),
        publication_id: pre.publication_id.clone(),
        source_handle: raw_handle.to_string(),
        source_session_id: source_id.to_string(),
        receiver_session_id: receiver_id.to_string(),
        message_id: pre.message_id.clone(),
        source_turn_nonce: source_audit["source"]["turn_nonce"].as_str().unwrap_or_default().to_string(),
        tool_call_id: stub_tool_call_id.clone(),
        expires_at_epoch: receipt["expires_at_epoch"].as_u64().unwrap_or_default(),
    };
    let body_bytes = canonical_body_bytes(&body)?;
    let artifact = NegativeControlArtifact {
        schema: ARTIFACT_SCHEMA.into(),
        // Truthful provenance: the artifact must never claim the offline
        // synthetic fixture on a live run.
        mode: if env.offline_records.is_some() {
            "offline-synthetic-fixture"
        } else {
            "live-root-authorized"
        }
        .into(),
        attempt_id: env.attempt_id.clone(),
        phase: "before_restart".into(),
        capture_id: source_audit["source"]["capture_id"].as_str().unwrap_or_default().to_string(),
        source_session_id: source_id.to_string(),
        source_request_id: source_audit["source"]["source_request_id"].as_str().unwrap_or_default().to_string(),
        receiver_session_id: receiver_id.to_string(),
        turn_nonce: source_audit["source"]["turn_nonce"].as_str().unwrap_or_default().to_string(),
        intent_digest: source_audit["source"]["intent_digest"].as_str().unwrap_or_default().to_string(),
        source_handle: raw_handle.to_string(),
        native_rid: receipt["native_rid"].as_str().unwrap_or_default().to_string(),
        response_id: receipt["response_id"].as_str().unwrap_or_default().to_string(),
        capture_commit_id: receipt["capture_commit_id"].as_str().unwrap_or_default().to_string(),
        retained_count: receipt["retained_count"].as_u64().unwrap_or_default() as u32,
        layer_index: receipt["layer_index"].as_u64().unwrap_or_default() as u32,
        hidden_dim: receipt["hidden_dim"].as_u64().unwrap_or_default() as u32,
        dtype: receipt["dtype"].as_str().unwrap_or_default().to_string(),
        layout_digest: receipt["layout_digest"].as_str().unwrap_or_default().to_string(),
        ready_ranks: receipt["ready_ranks"].as_u64().unwrap_or_default() as u32,
        expected_ranks: receipt["expected_ranks"].as_u64().unwrap_or_default() as u32,
        accepted_sequence_digest: receipt["accepted_sequence_digest"].as_str().unwrap_or_default().to_string(),
        output_sha256: receipt["output_sha256"].as_str().unwrap_or_default().to_string(),
        byte_count: receipt["byte_count"].as_u64().unwrap_or_default(),
        expires_at_epoch: receipt["expires_at_epoch"].as_u64().unwrap_or_default(),
        runtime_epoch: env.runtime_epoch.clone(),
        runtime_generation: env.runtime_generation,
        tool_call_id: stub_tool_call_id,
        tool_arguments_sha256: sha256(
            json!({"receiver_role":"child","receiver_name":"receiver"}).to_string().as_bytes(),
        ),
        publication_request_sha256: sha256(&body_bytes),
        publication_request: body,
        suite_nonce_sha256: nonce_commitment.to_string(),
        challenge_sha256: challenge_digest.to_string(),
        created_at: now(),
    };
    // Bounded stop reached: abort both agents, close ONLY the receiver (it
    // holds no lease). Never source.close(), family.close() or shutdown.
    source_agent.abort();
    receiver_agent.abort();
    receiver.close().await?;
    let case = json!({
        "case_passed": true,
        "capture_confirmed": true,
        "unpublished": true,
        "unretired": true,
    });
    Ok((case, artifact))
}

/// Harness-only one-attempt 409-state probe (after phase): sends the EXACT
/// precommitted canonical body bytes once through a harness-owned bounded
/// client (never the connector, never publish_native_source). Pass requires
/// exactly HTTP 409 with the parsed body {"error":{"code":"state"}}.
async fn run_stale_post_probe(
    env: &PhaseEnv,
    body_bytes: &[u8],
    artifact_sha256: &str,
    fresh_enrollment_runtime_epoch: &str,
) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| fail())?;
    let mut request = client
        .post(format!("{}/occ3/v1/native/publish", env.endpoint))
        .header("content-type", "application/json")
        .header("accept-encoding", "identity")
        .bearer_auth(&*env.key)
        .body(body_bytes.to_vec());
    if let Some(headers) = &env.control_headers {
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
    }
    let response = request.send().await.map_err(|_| fail())?;
    let status = response.status().as_u16();
    let body = response.bytes().await.map_err(|_| fail())?;
    // A body that is not the exact router JSON is a substantive shape
    // mismatch (proxies, error pages, tampering), not an infrastructure
    // error: it still lands in the case-shaped refusal below.
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let exact_state = parsed == json!({"error":{"code":"state"}});
    let case_passed = status == 409 && exact_state;
    let case = json!({
        "case_passed": case_passed,
        "http_status": status,
        "error_code": parsed["error"]["code"].clone(),
        "attempts": 1,
        "publication_request_sha256": sha256(body_bytes),
        "artifact_sha256": artifact_sha256,
        "fresh_enrollment_runtime_epoch": fresh_enrollment_runtime_epoch,
    });
    if !case_passed {
        return Err(format!("stale_handle_post failed: {case}"));
    }
    Ok(case)
}

// ---------------------------------------------------------------------------
// Phase-attempt claim (durable one-shot burn marker BEFORE any effect),
// challenge handling and report assembly.
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhaseChallenge {
    schema: String,
    phase: String,
    attempt_id: String,
    candidate_id: String,
    expected_candidate_started_at: String,
    model_fingerprint: String,
    method_id: String,
    expected_ranks: u32,
    challenge: String,
    created_at: u64,
    expires_at: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhaseClaim {
    schema: String,
    phase: String,
    attempt_id: String,
    candidate_id: String,
    candidate_started_at: String,
    descriptor_sha256: String,
    challenge_sha256: String,
    harness_binary_sha256: String,
    source_manifest_sha256: String,
    suite_nonce_sha256: String,
    retained_publication_id: String,
    retained_message_id: String,
    created_at: u64,
}

#[derive(Clone)]
struct ReportPins {
    binary_sha256: String,
    source_manifest_sha256: String,
    identity_fingerprint: String,
    profile_digest: String,
    endpoint_origin: String,
}

fn read_phase_challenge(journal: &Path, phase: &str) -> Result<PhaseChallenge, String> {
    let path = journal.join(format!("validator-challenge-{phase}.json"));
    let challenge: PhaseChallenge = strict_load(&path, MAX_CHALLENGE_BYTES)?;
    if challenge.schema != CHALLENGE_SCHEMA
        || challenge.phase != phase
        || challenge.method_id != METHOD_ID
        || challenge.challenge.is_empty()
        || challenge.challenge.len() > 256
        || challenge.attempt_id.is_empty()
        || challenge.candidate_id.is_empty()
        || challenge.expected_candidate_started_at.is_empty()
        || challenge.model_fingerprint.is_empty()
        || challenge.expires_at <= challenge.created_at
    {
        return Err(fail());
    }
    Ok(challenge)
}

/// Durable one-shot phase-attempt claim written BEFORE any helper resolution,
/// enrollment, network or model effect. An existing final report, an existing
/// claim, or a partial/malformed claim refuses without side effects: a crash
/// before final publication can never produce a second request.
fn write_phase_claim(
    journal: &Path,
    phase: &str,
    challenge: &PhaseChallenge,
    pins: &ReportPins,
    nonce_commitment: &str,
    pre: &PreMintedIds,
    descriptor_digest: &str,
) -> Result<(), String> {
    let claim_path = journal.join(format!("native-phase-attempt-{phase}.json"));
    let report_path = journal.join(format!("native-phase-suite-{phase}.json"));
    if report_path.exists() || claim_path.exists() {
        return Err("phase_attempt_already_claimed_no_replay".into());
    }
    // A partial/malformed prior claim also refuses.
    if let Ok(bytes) = strict_read_bounded(&claim_path, MAX_CLAIM_BYTES) {
        if !bytes.is_empty() {
            return Err("phase_attempt_claim_uncertain".into());
        }
    }
    let claim = PhaseClaim {
        schema: CLAIM_SCHEMA.into(),
        phase: phase.into(),
        attempt_id: challenge.attempt_id.clone(),
        candidate_id: challenge.candidate_id.clone(),
        candidate_started_at: challenge.expected_candidate_started_at.clone(),
        descriptor_sha256: descriptor_digest.into(),
        challenge_sha256: sha256(challenge.challenge.as_bytes()),
        harness_binary_sha256: pins.binary_sha256.clone(),
        source_manifest_sha256: pins.source_manifest_sha256.clone(),
        suite_nonce_sha256: nonce_commitment.into(),
        retained_publication_id: pre.publication_id.clone(),
        retained_message_id: pre.message_id.clone(),
        created_at: now(),
    };
    let bytes = canonical_json_bytes(
        &serde_json::to_value(&claim).map_err(|_| fail())?,
    )
    .map_err(|_| fail())?;
    if bytes.len() > MAX_CLAIM_BYTES {
        return Err(fail());
    }
    publish_exclusive(journal, &format!("native-phase-attempt-{phase}.json"), &bytes)
}

// ---------------------------------------------------------------------------
// Internal verification mirror of the frozen adapter's report contract.
// ---------------------------------------------------------------------------

fn exact_object_fields(value: &Value, fields: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|o| o.len() == fields.len() && fields.iter().all(|f| o.contains_key(*f)))
}

fn hex64(value: &Value) -> bool {
    value.as_str().is_some_and(|v| {
        v.len() == 64
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

const REPORT_COMMON_FIELDS: &[&str] = &[
    "schema", "phase", "attempt_id", "candidate_id", "candidate_started_at",
    "validator_challenge", "harness", "binding", "observed", "suite_nonce_sha256",
    "cases", "logical_cleanup_confirmed", "gpu_free_proven", "completed_at",
];
const BEFORE_EXTRA_FIELDS: &[&str] = &["negative_control_retained"];
const OBSERVED_FIELDS: &[&str] = &[
    "runtime_epoch", "runtime_generation", "tp_layout_digest", "hidden_dim",
    "layer_index", "dtype", "expected_ranks", "ready_ranks",
];
const POSITIVE_FIELDS: &[&str] = &[
    "case_passed", "semantic_nonce_match", "answer_sha256", "message_status",
    "lease_state", "consumed_before_tools", "receiver_request_audit",
];
const NO_STATE_FIELDS: &[&str] = &[
    "case_passed", "semantic_nonce_match", "answer_sha256", "messages",
    "consumed", "tool_calls", "receiver_request_audit",
];
const STALE_PRE_FIELDS: &[&str] = &["case_passed", "capture_confirmed", "unpublished", "unretired"];
const STALE_POST_FIELDS: &[&str] = &[
    "case_passed", "http_status", "error_code", "attempts",
    "publication_request_sha256", "artifact_sha256", "fresh_enrollment_runtime_epoch",
];
const RETAINED_FIELDS: &[&str] = &[
    "handle_sha256", "runtime_epoch", "runtime_generation", "expires_at_epoch",
    "retention_reason", "artifact_sha256",
];
const HARNESS_FIELDS: &[&str] = &["binary_sha256", "source_manifest_sha256", "platform"];
const BINDING_FIELDS: &[&str] = &["identity_fingerprint", "method_id", "profile_digest", "endpoint_origin"];
const AUDIT_FIELDS: &[&str] = &["source_facts_absent", "max_tokens", "request_count"];

fn verify_audit(value: &Value) -> bool {
    exact_object_fields(value, AUDIT_FIELDS)
        && value["source_facts_absent"] == json!(true)
        && value["max_tokens"] == json!(DEFAULT_MAX_TOKENS as u64)
        && value["request_count"] == json!(1)
}

/// Mirror of the frozen adapter's _verify_report: exact fieldsets, challenge
/// echo, harness/binding/observed checks and every case's semantics.
fn verify_report_contract(
    report: &Value,
    phase: &str,
    challenge: &PhaseChallenge,
    pins: &ReportPins,
) -> Result<(), String> {
    let before = phase == "before_restart";
    let mut fields = REPORT_COMMON_FIELDS.to_vec();
    if before {
        fields.extend_from_slice(BEFORE_EXTRA_FIELDS);
    }
    if !exact_object_fields(report, &fields) {
        return Err("report_fieldset_invalid".into());
    }
    if report["schema"] != json!(REPORT_SCHEMA)
        || report["phase"] != json!(phase)
        || report["attempt_id"] != json!(challenge.attempt_id)
        || report["candidate_id"] != json!(challenge.candidate_id)
        || report["candidate_started_at"] != json!(challenge.expected_candidate_started_at)
        || report["validator_challenge"] != json!(challenge.challenge)
    {
        return Err("report_binding_mismatch".into());
    }
    let harness = &report["harness"];
    if !exact_object_fields(harness, HARNESS_FIELDS)
        || harness["binary_sha256"] != json!(pins.binary_sha256)
        || harness["source_manifest_sha256"] != json!(pins.source_manifest_sha256)
        || harness["platform"] != json!(PLATFORM)
    {
        return Err("report_harness_binding_mismatch".into());
    }
    let binding = &report["binding"];
    if !exact_object_fields(binding, BINDING_FIELDS)
        || binding["identity_fingerprint"] != json!(pins.identity_fingerprint)
        || binding["method_id"] != json!(METHOD_ID)
        || binding["profile_digest"] != json!(pins.profile_digest)
        || binding["endpoint_origin"] != json!(pins.endpoint_origin)
    {
        return Err("report_binding_block_mismatch".into());
    }
    let observed = &report["observed"];
    if !exact_object_fields(observed, OBSERVED_FIELDS)
        || !observed["runtime_epoch"].as_str().is_some_and(|v| !v.is_empty() && v.len() <= 256)
        || observed["runtime_generation"].as_u64().unwrap_or(0) == 0
        || !hex64(&observed["tp_layout_digest"])
        || observed["hidden_dim"].as_u64().unwrap_or(0) == 0
        || observed["dtype"] != json!("bfloat16")
        || observed["expected_ranks"] != json!(challenge.expected_ranks)
        || observed["ready_ranks"] != observed["expected_ranks"]
    {
        return Err("observed_fieldset_invalid".into());
    }
    if !hex64(&report["suite_nonce_sha256"]) {
        return Err("suite_nonce_hash_invalid".into());
    }
    let cases = &report["cases"];
    let case_fields: &[&str] = if before {
        &["positive", "no_state", "stale_handle_pre"]
    } else {
        &["stale_handle_post", "positive", "no_state"]
    };
    if !exact_object_fields(cases, case_fields) {
        return Err("cases_fieldset_invalid".into());
    }
    let positive = &cases["positive"];
    if !exact_object_fields(positive, POSITIVE_FIELDS)
        || positive["case_passed"] != json!(true)
        || positive["semantic_nonce_match"] != json!(true)
        || positive["answer_sha256"] != report["suite_nonce_sha256"]
        || positive["message_status"] != json!("delivered")
        || positive["lease_state"] != json!("Consumed")
        || positive["consumed_before_tools"] != json!(true)
        || !verify_audit(&positive["receiver_request_audit"])
    {
        return Err("positive_case_invalid".into());
    }
    let no_state = &cases["no_state"];
    if !exact_object_fields(no_state, NO_STATE_FIELDS)
        || no_state["case_passed"] != json!(true)
        || no_state["semantic_nonce_match"] != json!(false)
        || no_state["answer_sha256"] != json!(sha256(UNKNOWN_ANSWER.as_bytes()))
        || no_state["messages"] != json!(0)
        || no_state["consumed"] != json!(0)
        || no_state["tool_calls"] != json!(0)
        || !verify_audit(&no_state["receiver_request_audit"])
    {
        return Err("no_state_case_invalid".into());
    }
    if before {
        let stale_pre = &cases["stale_handle_pre"];
        if !exact_object_fields(stale_pre, STALE_PRE_FIELDS)
            || stale_pre["case_passed"] != json!(true)
            || stale_pre["capture_confirmed"] != json!(true)
            || stale_pre["unpublished"] != json!(true)
            || stale_pre["unretired"] != json!(true)
        {
            return Err("stale_pre_case_invalid".into());
        }
        let retained = &report["negative_control_retained"];
        if !exact_object_fields(retained, RETAINED_FIELDS)
            || !hex64(&retained["handle_sha256"])
            || retained["runtime_epoch"] != observed["runtime_epoch"]
            || retained["runtime_generation"] != observed["runtime_generation"]
            || retained["expires_at_epoch"].as_u64().unwrap_or(0) == 0
            || retained["retention_reason"] != json!("planned_restart")
            || !hex64(&retained["artifact_sha256"])
        {
            return Err("retained_block_invalid".into());
        }
    } else {
        let stale_post = &cases["stale_handle_post"];
        if !exact_object_fields(stale_post, STALE_POST_FIELDS)
            || stale_post["case_passed"] != json!(true)
            || stale_post["http_status"] != json!(409)
            || stale_post["error_code"] != json!("state")
            || stale_post["attempts"] != json!(1)
            || !hex64(&stale_post["publication_request_sha256"])
            || !hex64(&stale_post["artifact_sha256"])
            || stale_post["fresh_enrollment_runtime_epoch"] != observed["runtime_epoch"]
        {
            return Err("stale_post_case_invalid".into());
        }
    }
    if report["logical_cleanup_confirmed"] != json!(true) {
        return Err("logical_cleanup_unconfirmed".into());
    }
    if report["gpu_free_proven"] != json!(false) {
        return Err("gpu_free_overclaimed".into());
    }
    let completed = report["completed_at"].as_u64().unwrap_or(0);
    if completed == 0
        || completed < challenge.created_at
        || completed > now() + 60
        || completed > challenge.expires_at
    {
        return Err("completion_outside_window".into());
    }
    Ok(())
}

/// After-chain mirror: fresh epoch, ADVANCED generation (volatile-reset
/// unadmitted), stale artifact digest bound to the before retained digest.
fn verify_after_chain(after: &Value, before: &Value) -> Result<(), String> {
    if after["observed"]["runtime_epoch"] == before["observed"]["runtime_epoch"] {
        return Err("runtime_epoch_unchanged".into());
    }
    if after["observed"]["runtime_generation"].as_u64().unwrap_or(0)
        <= before["observed"]["runtime_generation"].as_u64().unwrap_or(0)
    {
        return Err("runtime_generation_not_advanced".into());
    }
    if after["cases"]["stale_handle_post"]["artifact_sha256"]
        != before["negative_control_retained"]["artifact_sha256"]
    {
        return Err("stale_artifact_binding_mismatch".into());
    }
    Ok(())
}

fn assemble_report(
    phase: &str,
    challenge: &PhaseChallenge,
    pins: &ReportPins,
    observed: &Value,
    suite_nonce_sha256: &str,
    cases: Value,
    retained: Option<Value>,
    logical_cleanup_confirmed: bool,
) -> Value {
    let mut report = json!({
        "schema": REPORT_SCHEMA,
        "phase": phase,
        "attempt_id": challenge.attempt_id,
        "candidate_id": challenge.candidate_id,
        "candidate_started_at": challenge.expected_candidate_started_at,
        "validator_challenge": challenge.challenge,
        "harness": {
            "binary_sha256": pins.binary_sha256,
            "source_manifest_sha256": pins.source_manifest_sha256,
            "platform": PLATFORM,
        },
        "binding": {
            "identity_fingerprint": pins.identity_fingerprint,
            "method_id": METHOD_ID,
            "profile_digest": pins.profile_digest,
            "endpoint_origin": pins.endpoint_origin,
        },
        "observed": observed,
        "suite_nonce_sha256": suite_nonce_sha256,
        "cases": cases,
        "logical_cleanup_confirmed": logical_cleanup_confirmed,
        "gpu_free_proven": false,
        "completed_at": now(),
    });
    if let Some(retained) = retained {
        report["negative_control_retained"] = retained;
    }
    report
}

// ---------------------------------------------------------------------------
// Phase runners.
// ---------------------------------------------------------------------------

// Root diagnostic override: a server process binds each session id to its
// first (runtime, expiry) enrollment; after a live attempt those ids are
// burned for that process. Setting OCC3_SOURCE_SESSION / OCC3_RECEIVER_SESSION
// mints fresh ids for diagnostic reruns against the same serving process.
// Defaults keep the historical constants byte-identical.
fn diagnostic_source_session() -> String {
    std::env::var("OCC3_SOURCE_SESSION").unwrap_or_else(|_| ORIGINAL_SOURCE_SESSION.into())
}
fn diagnostic_receiver_session() -> String {
    std::env::var("OCC3_RECEIVER_SESSION").unwrap_or_else(|_| ORIGINAL_RECEIVER_SESSION.into())
}
const ORIGINAL_SOURCE_SESSION: &str = "source-session";
const ORIGINAL_RECEIVER_SESSION: &str = "receiver-session";
const ORIGINAL_NOSTATE_SESSION: &str = "nostate-receiver-session";
const STALE_SOURCE_SESSION: &str = "stale-source-session";
const STALE_RECEIVER_SESSION: &str = "stale-receiver-session";

async fn run_before_phase(
    env: &PhaseEnv,
    challenge: &PhaseChallenge,
    descriptor_digest: &str,
) -> Result<Value, String> {
    let pins = ReportPins {
        binary_sha256: env.harness_binary_sha256.clone(),
        source_manifest_sha256: env.source_manifest_sha256.clone(),
        identity_fingerprint: env.identity.fingerprint(),
        profile_digest: env.profile_digest.clone(),
        endpoint_origin: url::Url::parse(&env.endpoint)
            .map_err(|_| fail())?
            .origin()
            .ascii_serialization(),
    };
    // Root correction F4: the suite nonce is PRE-COMMITTED (sha256 of a fresh
    // random nonce chosen before source generation); the normalized answer is
    // compared against the commitment, never derived from the answer.
    let nonce = format!("occ3-{}", uuid::Uuid::new_v4());
    let nonce_commitment = sha256(nonce.as_bytes());
    let pre = PreMintedIds {
        publication_id: format!("publication-{}", uuid::Uuid::new_v4()),
        message_id: format!("message-{}", uuid::Uuid::new_v4()),
    };
    if env.expected_ranks != challenge.expected_ranks {
        return Err("challenge_ranks_mismatch".into());
    }
    // Durable one-shot claim BEFORE any effect.
    write_phase_claim(
        &env.journal,
        "before_restart",
        challenge,
        &pins,
        &nonce_commitment,
        &pre,
        descriptor_digest,
    )?;
    let connector = env.connector()?;
    // The retained negative control runs in its own family with its own
    // connector so no close/shutdown path can ever retire it.
    let stale_connector = env.connector()?;
    let (positive, observed, cleanup_positive) =
        run_positive_case(env, &connector, &diagnostic_source_session(), &diagnostic_receiver_session(), &nonce).await?;
    let (no_state, cleanup_no_state) =
        run_no_state_case(env, &connector, ORIGINAL_NOSTATE_SESSION, &nonce).await?;
    let (stale_pre, artifact) = run_stale_pre_case(
        env,
        &stale_connector,
        STALE_SOURCE_SESSION,
        STALE_RECEIVER_SESSION,
        &pre,
        &nonce_commitment,
        &sha256(challenge.challenge.as_bytes()),
    )
    .await?;
    // Logical cleanup covers positive/no-state only (adapter cleanup scope).
    let shutdown = connector.shutdown().await;
    let logical_cleanup_confirmed = cleanup_positive && cleanup_no_state && shutdown.is_ok();
    // Private negative-control artifact: the precommitted publication body
    // and its digest live INSIDE the artifact (F3); the report binds the
    // artifact digest transitively.
    let artifact_file = format!("validator-retained-negative-control-{}.json", env.attempt_id);
    let bytes = artifact_bytes(&artifact)?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(fail());
    }
    publish_exclusive(&env.journal, &artifact_file, &bytes)?;
    let retained = json!({
        "handle_sha256": sha256(artifact.source_handle.as_bytes()),
        "runtime_epoch": observed["runtime_epoch"],
        "runtime_generation": observed["runtime_generation"],
        "expires_at_epoch": artifact.expires_at_epoch,
        "retention_reason": "planned_restart",
        "artifact_sha256": sha256(&bytes),
    });
    let cases = json!({
        "positive": positive,
        "no_state": no_state,
        "stale_handle_pre": stale_pre,
    });
    let report = assemble_report(
        "before_restart",
        challenge,
        &pins,
        &observed,
        &nonce_commitment,
        cases,
        Some(retained),
        logical_cleanup_confirmed,
    );
    verify_report_contract(&report, "before_restart", challenge, &pins)?;
    let pretty = serde_json::to_vec_pretty(&report).map_err(|_| fail())?;
    publish_exclusive(&env.journal, "native-phase-suite-before_restart.json", &pretty)?;
    Ok(report)
}

async fn run_after_phase(
    env: &PhaseEnv,
    challenge: &PhaseChallenge,
    descriptor_digest: &str,
) -> Result<Value, String> {
    let pins = ReportPins {
        binary_sha256: env.harness_binary_sha256.clone(),
        source_manifest_sha256: env.source_manifest_sha256.clone(),
        identity_fingerprint: env.identity.fingerprint(),
        profile_digest: env.profile_digest.clone(),
        endpoint_origin: url::Url::parse(&env.endpoint)
            .map_err(|_| fail())?
            .origin()
            .ascii_serialization(),
    };
    // Strictly re-load the before report and the private artifact.
    let before_report: Value = strict_load(
        &env.journal.join("native-phase-suite-before_restart.json"),
        MAX_REPORT_BYTES,
    )?;
    let before_retained = &before_report["negative_control_retained"];
    let before_observed = &before_report["observed"];
    // Controlled restart profile: the producer itself refuses a volatile
    // reset (generation must ADVANCE, epoch must be fresh) before any case.
    if env.runtime_epoch == before_observed["runtime_epoch"] {
        return Err("runtime_epoch_unchanged".into());
    }
    if env.runtime_generation <= before_observed["runtime_generation"].as_u64().unwrap_or(0) {
        return Err("runtime_generation_not_advanced".into());
    }
    let artifact_file = format!("validator-retained-negative-control-{}.json", env.attempt_id);
    let artifact_path = env.journal.join(&artifact_file);
    let artifact_bytes_read = strict_read_bounded(&artifact_path, MAX_ARTIFACT_BYTES)?;
    let artifact: NegativeControlArtifact = strict_load(&artifact_path, MAX_ARTIFACT_BYTES)?;
    let artifact_sha256 = sha256(&artifact_bytes_read);
    if artifact_sha256 != before_retained["artifact_sha256"].as_str().unwrap_or_default() {
        return Err("stale_artifact_binding_mismatch".into());
    }
    if artifact.runtime_epoch != before_observed["runtime_epoch"].as_str().unwrap_or_default()
        || artifact.runtime_generation != before_observed["runtime_generation"].as_u64().unwrap_or(0)
    {
        return Err("retained_epoch_mismatch".into());
    }
    if artifact.schema != ARTIFACT_SCHEMA {
        return Err(fail());
    }
    // Rebuild the EXACT precommitted canonical body bytes (D2: same bytes).
    let body_bytes = canonical_body_bytes(&artifact.publication_request)?;
    if sha256(&body_bytes) != artifact.publication_request_sha256 {
        return Err("precommitted_body_digest_mismatch".into());
    }
    // Fresh independent after-phase nonce, committed before source generation.
    let nonce = format!("occ3-{}", uuid::Uuid::new_v4());
    let nonce_commitment = sha256(nonce.as_bytes());
    write_phase_claim(
        &env.journal,
        "after_restart",
        challenge,
        &pins,
        &nonce_commitment,
        &PreMintedIds {
            publication_id: format!("publication-{}", uuid::Uuid::new_v4()),
            message_id: format!("message-{}", uuid::Uuid::new_v4()),
        },
        descriptor_digest,
    )?;
    if env.expected_ranks != challenge.expected_ranks {
        return Err("challenge_ranks_mismatch".into());
    }
    let connector = env.connector()?;
    // Re-enroll the ORIGINAL stale-control session ids under the NEW epoch
    // (fresh_enrollment evidence); the probe then isolates the stale handle.
    connector
        .enroll_native_session(STALE_SOURCE_SESSION)
        .await
        .map_err(|_| fail())?;
    connector
        .enroll_native_session(STALE_RECEIVER_SESSION)
        .await
        .map_err(|_| fail())?;
    let fresh_enrollment_runtime_epoch = env.runtime_epoch.clone();
    let stale_post = run_stale_post_probe(
        env,
        &body_bytes,
        &artifact_sha256,
        &fresh_enrollment_runtime_epoch,
    )
    .await?;
    let (positive, observed, cleanup_positive) =
        run_positive_case(env, &connector, &diagnostic_source_session(), &diagnostic_receiver_session(), &nonce).await?;
    let (no_state, cleanup_no_state) =
        run_no_state_case(env, &connector, ORIGINAL_NOSTATE_SESSION, &nonce).await?;
    let shutdown = connector.shutdown().await;
    let logical_cleanup_confirmed = cleanup_positive && cleanup_no_state && shutdown.is_ok();
    let cases = json!({
        "stale_handle_post": stale_post,
        "positive": positive,
        "no_state": no_state,
    });
    let report = assemble_report(
        "after_restart",
        challenge,
        &pins,
        &observed,
        &nonce_commitment,
        cases,
        None,
        logical_cleanup_confirmed,
    );
    verify_report_contract(&report, "after_restart", challenge, &pins)?;
    verify_after_chain(&report, &before_report)?;
    let pretty = serde_json::to_vec_pretty(&report).map_err(|_| fail())?;
    publish_exclusive(&env.journal, "native-phase-suite-after_restart.json", &pretty)?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Offline selftest (mock handoff backend; no network/model/GPU/credential).
// ---------------------------------------------------------------------------

fn compute_offline_pins() -> Result<(String, String, Value), String> {
    let exe = std::env::current_exe().map_err(|_| fail())?;
    let bytes = std::fs::read(&exe).map_err(|_| fail())?;
    let binary_sha256 = sha256(&bytes);
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read(crate_dir.join("examples/occ3_native_phase_suite.rs"))
        .map_err(|_| fail())?;
    let cargo_lock = std::fs::read(crate_dir.join("..").join("..").join("Cargo.lock"))
        .map_err(|_| fail())?;
    let manifest = json!({
        "files": {"crates/pi-coding-agent/examples/occ3_native_phase_suite.rs": sha256(&source)},
        "cargo_lock_sha256": sha256(&cargo_lock),
    });
    let manifest_sha256 = sha256(&canonical_json_bytes(&manifest).map_err(|_| fail())?);
    Ok((binary_sha256, manifest_sha256, manifest))
}

fn offline_model(endpoint: &str) -> Model {
    let mut model = Model::new(
        "SYNTHETIC-glm53",
        "SYNTHETIC-glm53",
        API_OPENAI_COMPLETIONS,
        "offline-native-fixture",
        format!("{endpoint}/v1"),
    );
    model.max_tokens = CEILING_TOKENS;
    model.default_max_tokens = Some(DEFAULT_MAX_TOKENS);
    model.context_window = CONTEXT_WINDOW;
    model.reasoning = true;
    model.headers = Some(IndexMap::from([(
        CONTROL_HEADER.to_string(),
        CONTROL_VALUE.to_string(),
    )]));
    model
}

fn mint_offline_challenge(
    phase: &str,
    attempt_id: &str,
    candidate_id: &str,
    started_at: &str,
    fingerprint: &str,
) -> PhaseChallenge {
    PhaseChallenge {
        schema: CHALLENGE_SCHEMA.into(),
        phase: phase.into(),
        attempt_id: attempt_id.into(),
        candidate_id: candidate_id.into(),
        expected_candidate_started_at: started_at.into(),
        model_fingerprint: fingerprint.into(),
        method_id: METHOD_ID.into(),
        expected_ranks: EXPECTED_RANKS,
        challenge: sha256(uuid::Uuid::new_v4().to_string().as_bytes()),
        created_at: now(),
        expires_at: now() + 960,
    }
}

fn effective_descriptor_digest(
    phase: &str,
    attempt_id: &str,
    candidate_id: &str,
    origin: &str,
    epoch: &str,
    generation: u64,
    binary_sha256: &str,
    manifest_sha256: &str,
) -> Result<String, String> {
    let value = json!({
        "schema": DESCRIPTOR_SCHEMA,
        "mode": "offline-synthetic-fixture",
        "phase": phase,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "endpoint_origin": origin,
        "runtime_epoch": epoch,
        "runtime_generation": generation,
        "harness_binary_sha256": binary_sha256,
        "source_manifest_sha256": manifest_sha256,
    });
    Ok(sha256(
        &canonical_json_bytes(&value).map_err(|_| fail())?,
    ))
}

/// Full two-phase offline selftest against the local synthetic backend. The
/// "restart" is modeled by a second fixture instance with a fresh runtime
/// epoch and an ADVANCED generation (retained-counter restart profile).
/// Deterministic doubles; no live claim; gpu_free_proven stays false.
async fn offline_selftest(journal: Option<PathBuf>) -> Result<Value, String> {
    let journal = match journal {
        Some(dir) => {
            if dir.exists() {
                let nonempty = std::fs::read_dir(&dir)
                    .map(|entries| entries.count() > 0)
                    .unwrap_or(true);
                if nonempty {
                    return Err("journal_not_empty_no_replay".into());
                }
            } else {
                std::fs::create_dir_all(&dir).map_err(|_| fail())?;
            }
            dir
        }
        None => {
            let dir = std::env::temp_dir()
                .join(format!("occ3-phase-suite-selftest-{}", uuid::Uuid::new_v4().simple()));
            std::fs::create_dir(&dir).map_err(|_| fail())?;
            dir
        }
    };
    let (binary_sha256, manifest_sha256, manifest) = compute_offline_pins()?;
    let identity = fixture_identity();
    let fingerprint = identity.fingerprint();
    let attempt_id = format!("offline-attempt-{}", uuid::Uuid::new_v4().simple());
    let candidate_id = sha256("offline-synthetic-candidate".as_bytes());
    let started_at_before = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
    // Adapter-side challenge simulation, written exclusively before the phase.
    let challenge_before = mint_offline_challenge(
        "before_restart", &attempt_id, &candidate_id, &started_at_before, &fingerprint,
    );
    publish_exclusive(
        &journal,
        "validator-challenge-before_restart.json",
        &canonical_json_bytes(
            &serde_json::to_value(&challenge_before).map_err(|_| fail())?,
        )
        .map_err(|_| fail())?,
    )?;
    let before_records = Arc::new(Mutex::new(FixtureRecords::default()));
    let expires_before = now() + 240;
    // ONE fixture instance is ONE runtime endpoint: the runbook restart is the
    // same candidate container stopping and starting, so both phase reports
    // must carry the same endpoint origin.
    let epoch_state = Arc::new(Mutex::new(FixtureEpoch {
        runtime_epoch: "offline-runtime-epoch-before".into(),
        runtime_generation: 41,
        expires_at_epoch: expires_before,
        profile_digest: OFFLINE_PROFILE_DIGEST_BEFORE.into(),
    }));
    let (endpoint_before, server_before) =
        serve(epoch_state.clone(), before_records.clone()).await;
    let env_before = PhaseEnv {
        journal: journal.clone(),
        model: offline_model(&endpoint_before),
        key: Arc::new(FIXTURE_KEY.into()),
        control_headers: Some(IndexMap::from([(
            CONTROL_HEADER.to_string(),
            CONTROL_VALUE.to_string(),
        )])),
        endpoint: endpoint_before.clone(),
        identity: identity.clone(),
        attempt_id: attempt_id.clone(),
        runtime_epoch: "offline-runtime-epoch-before".into(),
        runtime_generation: 41,
        expires_at_epoch: expires_before,
        deadline_seconds: 20,
        max_capture_bytes: 1048576,
        max_response_bytes: replay_max_response_bytes(),
        harness_binary_sha256: binary_sha256.clone(),
        source_manifest_sha256: manifest_sha256.clone(),
        offline_records: Some(before_records.clone()),
        layer_index: LAYER_INDEX,
        hidden_dim: HIDDEN_DIM,
        dtype: "bfloat16".into(),
        layout_digest: LAYOUT_DIGEST.into(),
        expected_ranks: EXPECTED_RANKS,
        profile_digest: OFFLINE_PROFILE_DIGEST_BEFORE.into(),
    };
    let descriptor_digest = effective_descriptor_digest(
        "before_restart",
        &attempt_id,
        &candidate_id,
        &url::Url::parse(&endpoint_before).map_err(|_| fail())?.origin().ascii_serialization(),
        "offline-runtime-epoch-before",
        41,
        &binary_sha256,
        &manifest_sha256,
    )?;
    let before_report = run_before_phase(&env_before, &challenge_before, &descriptor_digest).await;
    let before_report = before_report?;
    // AFTER phase: restarted candidate (fresh epoch, advanced generation).
    let started_at_after = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
    let challenge_after = mint_offline_challenge(
        "after_restart", &attempt_id, &candidate_id, &started_at_after, &fingerprint,
    );
    publish_exclusive(
        &journal,
        "validator-challenge-after_restart.json",
        &canonical_json_bytes(
            &serde_json::to_value(&challenge_after).map_err(|_| fail())?,
        )
        .map_err(|_| fail())?,
    )?;
    // Controlled restart on the SAME endpoint: fresh epoch, ADVANCED
    // generation, volatile server state lost (the pre-restart capture
    // handle becomes stale state).
    let expires_after = now() + 240;
    restart_fixture(
        &epoch_state,
        &before_records,
        "offline-runtime-epoch-after",
        42,
        expires_after,
        OFFLINE_PROFILE_DIGEST_AFTER,
    );
    let endpoint_after = endpoint_before.clone();
    let after_records = before_records.clone();
    let env_after = PhaseEnv {
        journal: journal.clone(),
        model: offline_model(&endpoint_after),
        key: Arc::new(FIXTURE_KEY.into()),
        control_headers: Some(IndexMap::from([(
            CONTROL_HEADER.to_string(),
            CONTROL_VALUE.to_string(),
        )])),
        endpoint: endpoint_after.clone(),
        identity: identity.clone(),
        attempt_id: attempt_id.clone(),
        runtime_epoch: "offline-runtime-epoch-after".into(),
        runtime_generation: 42,
        expires_at_epoch: expires_after,
        deadline_seconds: 20,
        max_capture_bytes: 1048576,
        max_response_bytes: replay_max_response_bytes(),
        harness_binary_sha256: binary_sha256.clone(),
        source_manifest_sha256: manifest_sha256.clone(),
        offline_records: Some(after_records),
        layer_index: LAYER_INDEX,
        hidden_dim: HIDDEN_DIM,
        dtype: "bfloat16".into(),
        layout_digest: LAYOUT_DIGEST.into(),
        expected_ranks: EXPECTED_RANKS,
        profile_digest: OFFLINE_PROFILE_DIGEST_AFTER.into(),
    };
    let descriptor_digest_after = effective_descriptor_digest(
        "after_restart",
        &attempt_id,
        &candidate_id,
        &url::Url::parse(&endpoint_after).map_err(|_| fail())?.origin().ascii_serialization(),
        "offline-runtime-epoch-after",
        42,
        &binary_sha256,
        &manifest_sha256,
    )?;
    let after_report =
        run_after_phase(&env_after, &challenge_after, &descriptor_digest_after).await;
    server_before.abort();
    let after_report = after_report?;
    // Strict readback verification of the published final reports.
    let pins_before = ReportPins {
        binary_sha256: binary_sha256.clone(),
        source_manifest_sha256: manifest_sha256.clone(),
        identity_fingerprint: fingerprint.clone(),
        profile_digest: OFFLINE_PROFILE_DIGEST_BEFORE.into(),
        endpoint_origin: url::Url::parse(&endpoint_before)
            .map_err(|_| fail())?
            .origin()
            .ascii_serialization(),
    };
    let pins_after = ReportPins {
        profile_digest: OFFLINE_PROFILE_DIGEST_AFTER.into(),
        endpoint_origin: url::Url::parse(&endpoint_after)
            .map_err(|_| fail())?
            .origin()
            .ascii_serialization(),
        ..pins_before.clone()
    };
    let before_read: Value = strict_load(
        &journal.join("native-phase-suite-before_restart.json"),
        MAX_REPORT_BYTES,
    )?;
    let after_read: Value = strict_load(
        &journal.join("native-phase-suite-after_restart.json"),
        MAX_REPORT_BYTES,
    )?;
    verify_report_contract(&before_read, "before_restart", &challenge_before, &pins_before)?;
    verify_report_contract(&after_read, "after_restart", &challenge_after, &pins_after)?;
    verify_after_chain(&after_read, &before_read)?;
    if before_read != before_report || after_read != after_report {
        return Err("published_report_readback_mismatch".into());
    }
    let report_paths = [
        journal.join("native-phase-suite-before_restart.json"),
        journal.join("native-phase-suite-after_restart.json"),
    ];
    let mut files = serde_json::Map::new();
    for path in &report_paths {
        let bytes = std::fs::read(path).map_err(|_| fail())?;
        files.insert(
            path.file_name().unwrap().to_string_lossy().to_string(),
            json!({"sha256": sha256(&bytes), "bytes": bytes.len()}),
        );
    }
    let summary = json!({
        "schema": "optimus.occ3.producer-selftest.v1",
        "mode": "offline-synthetic-fixture",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "status": "passed",
        "phases": {
            "before_restart": {
                "report": "native-phase-suite-before_restart.json",
                "runtime_epoch": "offline-runtime-epoch-before",
                "runtime_generation": 41,
            },
            "after_restart": {
                "report": "native-phase-suite-after_restart.json",
                "runtime_epoch": "offline-runtime-epoch-after",
                "runtime_generation": 42,
            },
        },
        "restart_profile": "controlled_same_container_stop_start_retained_counter_offline_model",
        "generation_advanced": true,
        "harness": {
            "binary_sha256": binary_sha256,
            "source_manifest_sha256": manifest_sha256,
            "platform": PLATFORM,
            "source_manifest": manifest,
        },
        "identity_fingerprint": fingerprint,
        "profile_digest_before_restart": OFFLINE_PROFILE_DIGEST_BEFORE,
        "profile_digest_after_restart": OFFLINE_PROFILE_DIGEST_AFTER,
        "files": files,
        "gpu_proof": false,
        "physical_gpu_free_proven": false,
        "live_success_claimed": false,
        "journal": journal.to_string_lossy(),
    });
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Live descriptor mode (root-staged private descriptor; no offline fixture).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LivePhaseDescriptor {
    schema: String,
    root_authorized: bool,
    budget_profile_verified: bool,
    budget_review_digest: String,
    phase: String,
    attempt_id: String,
    // `candidate_id` and `access_domain` are validated structurally by the
    // strict loader but not read by the driver; they stay schema-required
    // for the validator-side report binding.
    #[allow(dead_code)]
    candidate_id: String,
    model: Model,
    identity: Occ3ModelIdentity,
    expected_fingerprint: String,
    endpoint: String,
    credential_reference: String,
    control_header_references: Option<IndexMap<String, String>>,
    runtime_id: String,
    runtime_generation: u64,
    #[allow(dead_code)]
    access_domain: String,
    expires_at_epoch: u64,
    layer_index: u32,
    hidden_dim: u32,
    dtype: String,
    layout_digest: String,
    expected_ranks: u32,
    profile_digest: String,
    max_capture_bytes: u64,
    max_response_bytes: usize,
    deadline_seconds: u64,
    journal_directory: String,
    isolated_profile: String,
    harness_binary_sha256: String,
    source_manifest_sha256: String,
}

fn control_reference_is_valid(reference: &str) -> bool {
    if let Some(command) = reference.strip_prefix('!') {
        !command.trim().is_empty()
    } else {
        !reference.is_empty()
            && reference
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    }
}

fn control_header_references_valid(
    references: &Option<IndexMap<String, String>>,
) -> bool {
    let Some(references) = references else {
        return true;
    };
    let mut seen: Vec<String> = Vec::new();
    for (name, reference) in references {
        let lower = name.to_ascii_lowercase();
        if !pi_coding_agent::core::occ3_connectors::sglang::control_header_name_is_allowed(name)
            || seen.contains(&lower)
            || !control_reference_is_valid(reference)
        {
            return false;
        }
        seen.push(lower);
    }
    true
}

fn model_headers_have_no_reserved_credential_names(model: &Model) -> bool {
    model.headers.as_ref().is_none_or(|headers| {
        headers.keys().all(|k| {
            !matches!(
                k.to_ascii_lowercase().as_str(),
                "authorization" | "host" | "proxy-authorization" | "x-pomerium-authorization"
            )
        })
    })
}

fn resolve_existing_key(reference: &str) -> Result<String, String> {
    if !reference.starts_with('!')
        && (reference.is_empty()
            || !reference
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || std::env::var(reference).is_err())
    {
        return Err(fail());
    }
    pi_coding_agent::core::resolve_config_value::resolve_config_value(reference)
        .filter(|k| !k.is_empty())
        .ok_or_else(fail)
}

fn resolve_control_headers(
    references: &Option<IndexMap<String, String>>,
) -> Result<Option<IndexMap<String, String>>, String> {
    let Some(references) = references else {
        return Ok(None);
    };
    if references.is_empty() || !control_header_references_valid(&Some(references.clone())) {
        return Err(fail());
    }
    let mut resolved = IndexMap::new();
    for (name, reference) in references {
        let value = resolve_existing_key(reference)?;
        resolved.insert(name.clone(), value);
    }
    Ok(Some(resolved))
}

async fn live_phase(path: &str) -> Result<Value, String> {
    let descriptor_path = Path::new(path);
    let bytes = strict_read_bounded(descriptor_path, MAX_DESCRIPTOR_BYTES)?;
    let text = strict_decode(&bytes)?;
    structural_scan(&text)?;
    let d: LivePhaseDescriptor = serde_json::from_str(&text).map_err(|_| fail())?;
    if d.schema != DESCRIPTOR_SCHEMA
        || !d.root_authorized
        || !d.budget_profile_verified
        || d.budget_review_digest.len() != 64
        || !matches!(d.phase.as_str(), "before_restart" | "after_restart")
        || d.identity.fingerprint() != d.expected_fingerprint
        || d.identity.latent_method != METHOD_ID
        || d.model.id != d.identity.model_id
        || d.model.api != API_OPENAI_COMPLETIONS
        || d.model.max_tokens != CEILING_TOKENS
        || d.model.default_max_tokens != Some(DEFAULT_MAX_TOKENS)
        || d.model.context_window != CONTEXT_WINDOW
        || !d.model.reasoning
        || d.deadline_seconds == 0
        || d.deadline_seconds > 3600
        || !model_headers_have_no_reserved_credential_names(&d.model)
        || !control_header_references_valid(&d.control_header_references)
        || !matches!(d.dtype.as_str(), "bfloat16" | "float16" | "float32")
        || d.layout_digest.len() != 64
        || d.profile_digest.len() != 64
        || d.expected_ranks == 0
        || d.hidden_dim == 0
        || d.max_capture_bytes == 0
        || d.max_response_bytes == 0
        || d.max_response_bytes > 64 * 1024 * 1024
    {
        return Err(fail());
    }
    let journal = PathBuf::from(&d.journal_directory);
    let profile = PathBuf::from(&d.isolated_profile);
    if !journal.is_absolute()
        || !journal.is_dir()
        || !profile.is_absolute()
        || profile.exists()
    {
        return Err(fail());
    }
    std::fs::create_dir(&profile).map_err(|_| fail())?;
    // Harness self-binding: the running binary must equal the pinned digest.
    let exe = std::env::current_exe().map_err(|_| fail())?;
    let exe_bytes = std::fs::read(&exe).map_err(|_| fail())?;
    if sha256(&exe_bytes) != d.harness_binary_sha256 {
        return Err("harness_binary_sha256_mismatch".into());
    }
    let challenge = read_phase_challenge(&journal, &d.phase)?;
    // The armed positive case buffers the whole source response server-side
    // (the capture span settles at completion), so zero client events for the
    // full server-bounded turn are healthy, not a stall. Raise the stream
    // watchdog floors to this descriptor's own absolute deadline; the server
    // deadline still bounds the turn and the watchdog maxima still clamp.
    let _ = pi_agent_core::stream_watchdog::raise_env_for_armed_turn(d.deadline_seconds);
    // Resolve credentials through the existing reference grammar only,
    // before any enroll/inference/HTTP effect.
    let control_headers = resolve_control_headers(&d.control_header_references)?;
    let key = resolve_existing_key(&d.credential_reference)?;
    let mut model = d.model.clone();
    if let Some(headers) = &control_headers {
        let mut merged = model.headers.clone().unwrap_or_default();
        for (name, value) in headers {
            merged.insert(name.clone(), value.clone());
        }
        model.headers = Some(merged);
    }
    let env = PhaseEnv {
        journal,
        model,
        key: Arc::new(key),
        control_headers,
        endpoint: d.endpoint.clone(),
        identity: d.identity.clone(),
        attempt_id: d.attempt_id.clone(),
        runtime_epoch: d.runtime_id.clone(),
        runtime_generation: d.runtime_generation,
        expires_at_epoch: d.expires_at_epoch,
        deadline_seconds: d.deadline_seconds,
        max_capture_bytes: d.max_capture_bytes,
        max_response_bytes: d.max_response_bytes,
        harness_binary_sha256: d.harness_binary_sha256.clone(),
        source_manifest_sha256: d.source_manifest_sha256.clone(),
        offline_records: None,
        layer_index: d.layer_index,
        hidden_dim: d.hidden_dim,
        dtype: d.dtype.clone(),
        layout_digest: d.layout_digest.clone(),
        expected_ranks: d.expected_ranks,
        profile_digest: d.profile_digest.clone(),
    };
    let descriptor_digest = sha256(&bytes);
    let report = if d.phase == "before_restart" {
        run_before_phase(&env, &challenge, &descriptor_digest).await?
    } else {
        run_after_phase(&env, &challenge, &descriptor_digest).await?
    };
    Ok(report)
}

/// The armed positive-case chain (Agent::run -> run_agent_loop ->
/// stream_assistant_response -> inner -> Done-arm completion) needs roughly
/// 1.25 MiB of stack in a debug build: the nested async state machines are
/// large and unoptimized frames do not share slots. A Windows main thread has
/// a 1 MiB reserve, so driving the suite there overflowed the stack the
/// moment a source turn completed (exit 0xC00000FD, before tool dispatch).
/// The fix is to drive the suite on a thread with an explicit, generous
/// stack; 64 MiB leaves order-of-magnitude headroom over the measured need.
/// `OCC3_DRIVER_STACK_BYTES` overrides the size (diagnostics: setting it to
/// 1048576 reproduces the historical overflow on demand).
const DEFAULT_DRIVER_STACK_BYTES: usize = 64 * 1024 * 1024;

fn main() {
    let stack_bytes = std::env::var("OCC3_DRIVER_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|bytes| *bytes >= 1024 * 1024)
        .unwrap_or(DEFAULT_DRIVER_STACK_BYTES);
    let child = std::thread::Builder::new()
        .stack_size(stack_bytes)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move { real_main().await })
        })
        .expect("spawn driver thread");
    match child.join() {
        Ok(Ok(())) => {}
        Ok(Err(code)) => std::process::exit(code),
        Err(_) => std::process::exit(2),
    }
}

async fn real_main() -> Result<(), i32> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.is_empty() {
        offline_selftest(None).await
    } else if args.len() == 1 && args[0] == "--offline" {
        offline_selftest(None).await
    } else if args.len() == 2 && args[0] == "--offline" {
        offline_selftest(Some(PathBuf::from(&args[1]))).await
    } else if args.len() == 2 && args[0] == "--live" {
        live_phase(&args[1]).await
    } else {
        Err("Usage: occ3_native_phase_suite [--offline [DIR] | --live PRIVATE_DESCRIPTOR_PATH]".into())
    };
    match result {
        Ok(summary) => println!("{summary}"),
        Err(error) => {
            eprintln!("Native phase-suite producer refused: {error}");
            return Err(1);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Offline gate tests (mirroring the established gate_tests style).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gate_tests {
    use super::*;

    #[test]
    fn structural_scan_binds_depth_duplicates_and_balance() {
        assert!(structural_scan("{\"a\":[1,2,{\"b\":null}]}").is_ok());
        assert!(structural_scan("{\"a\":1,\"a\":2}").is_err());
        assert!(structural_scan("[1,2").is_err());
        assert!(structural_scan("1]").is_err());
        let deep = format!("{}1{}", "[".repeat(33), "]".repeat(33));
        assert!(structural_scan(&deep).is_err());
        let bounded = format!("{}1{}", "[".repeat(32), "]".repeat(32));
        assert!(structural_scan(&bounded).is_ok());
        // Escaped quotes must not confuse the key scanner.
        assert!(structural_scan(r#"{"a\"b":1,"c":2}"#).is_ok());
        assert!(structural_scan(r#"{"a\"b":1,"a\"b":2}"#).is_err());
    }

    #[test]
    fn strict_decode_refuses_bom_and_invalid_utf8() {
        assert!(strict_decode(b"{}").is_ok());
        assert!(strict_decode(&[0xEF, 0xBB, 0xBF, b'{', b'}']).is_err());
        assert!(strict_decode(&[0xFF, 0xFE, b'{', b'}']).is_err());
    }

    fn pins() -> ReportPins {
        ReportPins {
            binary_sha256: "e".repeat(64),
            source_manifest_sha256: "1".repeat(64),
            identity_fingerprint: "sha256:".to_string() + &"f".repeat(64),
            profile_digest: "4".repeat(64),
            endpoint_origin: "http://127.0.0.1:40000".into(),
        }
    }

    fn challenge() -> PhaseChallenge {
        PhaseChallenge {
            schema: CHALLENGE_SCHEMA.into(),
            phase: "before_restart".into(),
            attempt_id: "synthetic-attempt-0001".into(),
            candidate_id: "c".repeat(64),
            expected_candidate_started_at: "2026-10-21T00:00:00.000000Z".into(),
            model_fingerprint: "sha256:".to_string() + &"f".repeat(64),
            method_id: METHOD_ID.into(),
            expected_ranks: EXPECTED_RANKS,
            challenge: "7".repeat(64),
            created_at: now() - 10,
            expires_at: now() + 900,
        }
    }

    fn positive_case() -> Value {
        json!({
            "case_passed": true, "semantic_nonce_match": true,
            "answer_sha256": "f".repeat(64), "message_status": "delivered",
            "lease_state": "Consumed", "consumed_before_tools": true,
            "receiver_request_audit": {"source_facts_absent": true, "max_tokens": 131072, "request_count": 1},
        })
    }

    fn no_state_case() -> Value {
        json!({
            "case_passed": true, "semantic_nonce_match": false,
            "answer_sha256": sha256(UNKNOWN_ANSWER.as_bytes()),
            "messages": 0, "consumed": 0, "tool_calls": 0,
            "receiver_request_audit": {"source_facts_absent": true, "max_tokens": 131072, "request_count": 1},
        })
    }

    fn before_report() -> Value {
        json!({
            "schema": REPORT_SCHEMA, "phase": "before_restart",
            "attempt_id": "synthetic-attempt-0001", "candidate_id": "c".repeat(64),
            "candidate_started_at": "2026-10-21T00:00:00.000000Z",
            "validator_challenge": "7".repeat(64),
            "harness": {"binary_sha256": "e".repeat(64), "source_manifest_sha256": "1".repeat(64), "platform": PLATFORM},
            "binding": {"identity_fingerprint": "sha256:".to_string() + &"f".repeat(64), "method_id": METHOD_ID,
                "profile_digest": "4".repeat(64), "endpoint_origin": "http://127.0.0.1:40000"},
            "observed": {"runtime_epoch": "synthetic-runtime-epoch-before", "runtime_generation": 41,
                "tp_layout_digest": "9".repeat(64), "hidden_dim": 6144, "layer_index": 77,
                "dtype": "bfloat16", "expected_ranks": 8, "ready_ranks": 8},
            "suite_nonce_sha256": "f".repeat(64),
            "cases": {"positive": positive_case(), "no_state": no_state_case(),
                "stale_handle_pre": {"case_passed": true, "capture_confirmed": true, "unpublished": true, "unretired": true}},
            "negative_control_retained": {"handle_sha256": "a".repeat(64), "runtime_epoch": "synthetic-runtime-epoch-before",
                "runtime_generation": 41, "expires_at_epoch": 4102444800u64,
                "retention_reason": "planned_restart", "artifact_sha256": "b".repeat(64)},
            "logical_cleanup_confirmed": true, "gpu_free_proven": false, "completed_at": now(),
        })
    }

    fn after_report() -> Value {
        json!({
            "schema": REPORT_SCHEMA, "phase": "after_restart",
            "attempt_id": "synthetic-attempt-0001", "candidate_id": "c".repeat(64),
            "candidate_started_at": "2026-10-21T00:00:00.000000Z",
            "validator_challenge": "7".repeat(64),
            "harness": {"binary_sha256": "e".repeat(64), "source_manifest_sha256": "1".repeat(64), "platform": PLATFORM},
            "binding": {"identity_fingerprint": "sha256:".to_string() + &"f".repeat(64), "method_id": METHOD_ID,
                "profile_digest": "4".repeat(64), "endpoint_origin": "http://127.0.0.1:40000"},
            "observed": {"runtime_epoch": "synthetic-runtime-epoch-after", "runtime_generation": 42,
                "tp_layout_digest": "9".repeat(64), "hidden_dim": 6144, "layer_index": 77,
                "dtype": "bfloat16", "expected_ranks": 8, "ready_ranks": 8},
            "suite_nonce_sha256": "f".repeat(64),
            "cases": {"stale_handle_post": {"case_passed": true, "http_status": 409, "error_code": "state",
                    "attempts": 1, "publication_request_sha256": "c".repeat(64), "artifact_sha256": "b".repeat(64),
                    "fresh_enrollment_runtime_epoch": "synthetic-runtime-epoch-after"},
                "positive": positive_case(), "no_state": no_state_case()},
            "logical_cleanup_confirmed": true, "gpu_free_proven": false, "completed_at": now(),
        })
    }

    #[test]
    fn verify_report_contract_accepts_the_conforming_shapes() {
        verify_report_contract(&before_report(), "before_restart", &challenge(), &pins()).unwrap();
        let after_challenge = PhaseChallenge { phase: "after_restart".into(), ..challenge() };
        verify_report_contract(&after_report(), "after_restart", &after_challenge, &pins()).unwrap();
        verify_after_chain(&after_report(), &before_report()).unwrap();
    }

    #[test]
    fn verify_report_contract_refuses_every_binding_tamper() {
        let after_challenge = PhaseChallenge { phase: "after_restart".into(), ..challenge() };
        for (report, phase, challenge, pointer, replacement) in [
            (before_report(), "before_restart", challenge(), "/attempt_id", json!("other-attempt")),
            (before_report(), "before_restart", challenge(), "/candidate_id", json!("other-candidate")),
            (before_report(), "before_restart", challenge(), "/validator_challenge", json!("other-challenge")),
            (before_report(), "before_restart", challenge(), "/harness/binary_sha256", json!("0".repeat(64))),
            (before_report(), "before_restart", challenge(), "/harness/platform", json!("linux-x86_64")),
            (before_report(), "before_restart", challenge(), "/binding/method_id", json!("other-method")),
            (before_report(), "before_restart", challenge(), "/observed/expected_ranks", json!(4)),
            (before_report(), "before_restart", challenge(), "/observed/ready_ranks", json!(7)),
            (before_report(), "before_restart", challenge(), "/cases/positive/answer_sha256", json!("0".repeat(64))),
            (before_report(), "before_restart", challenge(), "/cases/positive/consumed_before_tools", json!(false)),
            (before_report(), "before_restart", challenge(), "/cases/positive/receiver_request_audit/max_tokens", json!(4096)),
            (before_report(), "before_restart", challenge(), "/cases/no_state/answer_sha256", json!("0".repeat(64))),
            (before_report(), "before_restart", challenge(), "/cases/no_state/messages", json!(1)),
            (before_report(), "before_restart", challenge(), "/cases/stale_handle_pre/unpublished", json!(false)),
            (before_report(), "before_restart", challenge(), "/negative_control_retained/runtime_generation", json!(42)),
            (before_report(), "before_restart", challenge(), "/negative_control_retained/retention_reason", json!("other")),
            (before_report(), "before_restart", challenge(), "/logical_cleanup_confirmed", json!(false)),
            (before_report(), "before_restart", challenge(), "/gpu_free_proven", json!(true)),
            (after_report(), "after_restart", after_challenge.clone(), "/cases/stale_handle_post/http_status", json!(200)),
            (after_report(), "after_restart", after_challenge.clone(), "/cases/stale_handle_post/error_code", json!("expired")),
            (after_report(), "after_restart", after_challenge.clone(), "/cases/stale_handle_post/attempts", json!(2)),
            (after_report(), "after_restart", after_challenge.clone(), "/cases/stale_handle_post/fresh_enrollment_runtime_epoch", json!("synthetic-runtime-epoch-before")),
        ] {
            let mut tampered = report;
            *tampered.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                verify_report_contract(&tampered, phase, &challenge, &pins()).is_err(),
                "{pointer} tamper must refuse"
            );
        }
        // Unknown extra fields refuse (deny-unknown-fields mirror).
        let mut extra = before_report();
        extra["publication_request_sha256"] = json!("c".repeat(64));
        assert!(verify_report_contract(&extra, "before_restart", &challenge(), &pins()).is_err());
        // Volatile reset (generation not advanced) refuses.
        let mut reset = after_report();
        reset["observed"]["runtime_generation"] = json!(41);
        assert!(verify_after_chain(&reset, &before_report()).is_err());
        let mut same_epoch = after_report();
        same_epoch["observed"]["runtime_epoch"] = json!("synthetic-runtime-epoch-before");
        assert!(verify_after_chain(&same_epoch, &before_report()).is_err());
        // Stale artifact digest must bind to the before retained digest.
        let mut artifact_mismatch = after_report();
        artifact_mismatch["cases"]["stale_handle_post"]["artifact_sha256"] = json!("d".repeat(64));
        assert!(verify_after_chain(&artifact_mismatch, &before_report()).is_err());
    }

    #[test]
    fn precommitted_body_is_the_exact_nine_field_canonical_form() {
        let body = PublicationRequestBody {
            schema: "occ3.publish.v1".into(),
            publication_id: "publication-1".into(),
            source_handle: "handle".into(),
            source_session_id: "source-session".into(),
            receiver_session_id: "receiver-session".into(),
            message_id: "message-1".into(),
            source_turn_nonce: "turn-1".into(),
            tool_call_id: "fixture-tool-1".into(),
            expires_at_epoch: 4102444800,
        };
        let bytes = canonical_body_bytes(&body).unwrap();
        let expected = concat!(
            "{\"expires_at_epoch\":4102444800,\"message_id\":\"message-1\",",
            "\"publication_id\":\"publication-1\",\"receiver_session_id\":\"receiver-session\",",
            "\"schema\":\"occ3.publish.v1\",\"source_handle\":\"handle\",",
            "\"source_session_id\":\"source-session\",\"source_turn_nonce\":\"turn-1\",",
            "\"tool_call_id\":\"fixture-tool-1\"}"
        );
        assert_eq!(String::from_utf8(bytes).unwrap(), expected);
        assert_eq!(sha256(&canonical_body_bytes(&body).unwrap()), sha256(&canonical_body_bytes(&body).unwrap()));
    }

    /// The stale probe demands EXACTLY 409 + {"error":{"code":"state"}}: no
    /// other status, no FastAPI detail shape, no extra fields, one attempt.
    #[tokio::test]
    async fn stale_probe_requires_the_exact_router_state_shape() {
        async fn respond_with(status: &str, body: &str) -> (PhaseEnv, tokio::task::JoinHandle<()>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let body = body.to_string();
            let status = status.to_string();
            let task = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = [0u8; 8192];
                let mut read = 0;
                loop {
                    let n = socket.read(&mut buffer[read..]).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    read += n;
                    if let Some(i) = buffer[..read].windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buffer[..i]).to_lowercase();
                        let length = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length: "))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if read >= i + 4 + length {
                            break;
                        }
                    }
                }
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
            let env = PhaseEnv {
                journal: std::env::temp_dir(),
                model: offline_model(&endpoint),
                key: Arc::new(FIXTURE_KEY.into()),
                control_headers: None,
                endpoint,
                identity: fixture_identity(),
                attempt_id: "probe-attempt".into(),
                runtime_epoch: "probe-epoch".into(),
                runtime_generation: 2,
                expires_at_epoch: now() + 240,
                deadline_seconds: 20,
                max_capture_bytes: 1048576,
                max_response_bytes: 65536,
                harness_binary_sha256: "0".repeat(64),
                source_manifest_sha256: "0".repeat(64),
                offline_records: None,
                layer_index: LAYER_INDEX,
                hidden_dim: HIDDEN_DIM,
                dtype: "bfloat16".into(),
                layout_digest: LAYOUT_DIGEST.into(),
                expected_ranks: EXPECTED_RANKS,
                profile_digest: PROFILE_DIGEST.into(),
            };
            (env, task)
        }
        let body = PublicationRequestBody {
            schema: "occ3.publish.v1".into(),
            publication_id: "publication-1".into(),
            source_handle: "stale-handle".into(),
            source_session_id: "stale-source-session".into(),
            receiver_session_id: "stale-receiver-session".into(),
            message_id: "message-1".into(),
            source_turn_nonce: "turn-1".into(),
            tool_call_id: "fixture-tool-1".into(),
            expires_at_epoch: now() + 240,
        };
        let bytes = canonical_body_bytes(&body).unwrap();
        // The exact router shape passes.
        let (env, task) = respond_with("409 Conflict", "{\"error\":{\"code\":\"state\"}}").await;
        let case = run_stale_post_probe(&env, &bytes, &"b".repeat(64), "probe-epoch").await.unwrap();
        assert_eq!(case["http_status"], json!(409));
        assert_eq!(case["attempts"], json!(1));
        task.await.unwrap();
        for (status, body) in [
            ("200 OK", "{\"error\":{\"code\":\"state\"}}"),
            ("409 Conflict", "{\"detail\":\"state\"}"),
            ("409 Conflict", "{\"error\":{\"code\":\"expired\"}}"),
            ("409 Conflict", "{\"error\":{\"code\":\"replay\"}}"),
            ("401 Unauthorized", "{\"error\":{\"code\":\"identity\"}}"),
            ("503 Service Unavailable", "{\"error\":{\"code\":\"unavailable\"}}"),
            ("409 Conflict", "{\"error\":{\"code\":\"state\",\"detail\":\"extra\"}}"),
            ("409 Conflict", "not-json"),
        ] {
            let (env, task) = respond_with(status, body).await;
            let error = run_stale_post_probe(&env, &bytes, &"b".repeat(64), "probe-epoch")
                .await
                .map(|value| value.to_string())
                .unwrap_or_else(|error| error);
            assert!(error.contains("stale_handle_post failed"), "{status} {body}");
            task.await.unwrap();
        }
    }

    /// Full two-phase offline selftest: reports satisfy the contract mirror
    /// end to end; the journal contains exactly the expected file set.
    #[cfg(windows)]
    #[tokio::test]
    async fn offline_selftest_passes_and_publishes_both_reports() {
        let journal = std::env::temp_dir()
            .join(format!("occ3-phase-suite-selftest-test-{}", uuid::Uuid::new_v4().simple()));
        let summary = offline_selftest(Some(journal.clone())).await.unwrap();
        assert_eq!(summary["status"], json!("passed"));
        assert_eq!(summary["gpu_proof"], json!(false));
        assert_eq!(summary["live_success_claimed"], json!(false));
        for name in [
            "validator-challenge-before_restart.json",
            "native-phase-attempt-before_restart.json",
            "native-phase-suite-before_restart.json",
            "validator-challenge-after_restart.json",
            "native-phase-attempt-after_restart.json",
            "native-phase-suite-after_restart.json",
        ] {
            assert!(journal.join(name).is_file(), "missing {name}");
        }
        // Exactly one retained negative-control artifact under the attempt id,
        // bounded to the artifact byte cap.
        let attempt_id = summary["attempt_id"].as_str().unwrap().to_string();
        let artifact_path = journal.join(format!("validator-retained-negative-control-{attempt_id}.json"));
        assert!(artifact_path.is_file());
        assert!(artifact_path.metadata().unwrap().len() <= MAX_ARTIFACT_BYTES as u64);
        // No raw nonce, answer or handle ever appears in either report.
        for report_name in [
            "native-phase-suite-before_restart.json",
            "native-phase-suite-after_restart.json",
        ] {
            let text = std::fs::read_to_string(journal.join(report_name)).unwrap();
            assert!(!text.contains("occ3-"), "{report_name} leaked a raw nonce");
            assert!(!text.contains("offline-source-handle"), "{report_name} leaked a raw handle");
            assert!(!text.contains("offline-stale-retained-control"), "{report_name} leaked the control token");
        }
    }

    /// One-shot no-replay: a second selftest in the same journal refuses.
    #[cfg(windows)]
    #[tokio::test]
    async fn second_run_in_the_same_journal_refuses_without_side_effects() {
        let journal = std::env::temp_dir()
            .join(format!("occ3-phase-suite-selftest-replay-{}", uuid::Uuid::new_v4().simple()));
        offline_selftest(Some(journal.clone())).await.unwrap();
        let error = offline_selftest(Some(journal.clone())).await.err().unwrap();
        assert!(
            error.contains("no_replay") || error.contains("not_empty"),
            "unexpected refusal: {error}"
        );
    }

    /// The fixture mirrors the live occ3-sglang source-request contract
    /// (chat_adapter handle_around): a capture-armed generation request must
    /// carry exactly one accept-encoding header with value identity and is
    /// refused with the attachment-error shape otherwise. This is the exact
    /// class of gap the TS10 live window exposed: the offline fixture never
    /// checked accept-encoding, so a client that sends no header passed
    /// offline and failed live.
    #[tokio::test]
    async fn fixture_refuses_capture_requests_without_identity_accept_encoding() {
        let records = Arc::new(Mutex::new(FixtureRecords::default()));
        let epoch_state = Arc::new(Mutex::new(FixtureEpoch {
            runtime_epoch: "fixture-accept-encoding-epoch".into(),
            runtime_generation: 1,
            expires_at_epoch: now() + 240,
            profile_digest: OFFLINE_PROFILE_DIGEST_BEFORE.into(),
        }));
        let (endpoint, server) = serve(epoch_state, records).await;
        let capture_body = json!({
            "model": "fixture-model",
            "max_tokens": DEFAULT_MAX_TOKENS as u64,
            "reasoning_effort": "high",
            "stream": true,
            "messages": [{"role": "user", "content": "fixture question"}],
            "occ3_capture": {
                "schema": "occ3.capture.v1",
                "capture_id": "capture-fixture-accept-encoding",
                "source_session_id": "session-fixture-accept-encoding",
                "source_request_id": "request-fixture-accept-encoding",
                "turn_nonce": "turn-fixture-accept-encoding",
                "model_fingerprint": format!("sha256:{}", "0".repeat(64)),
                "runtime_id": "fixture-accept-encoding-epoch",
                "runtime_generation": 1,
                "method": METHOD_ID,
                "retained_count_limit": 8,
                "max_capture_bytes": 1048576,
                "expires_at_epoch": now() + 240,
                "span_policy": "whole_completed_response_tail_v1"
            }
        })
        .to_string();
        async fn roundtrip(
            endpoint: &str,
            body: &str,
            accept_encoding: Option<&str>,
        ) -> (String, String) {
            let mut stream = tokio::net::TcpStream::connect(
                endpoint.trim_start_matches("http://"),
            )
            .await
            .unwrap();
            let mut request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nAuthorization: Bearer {FIXTURE_KEY}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            if let Some(value) = accept_encoding {
                request.push_str(&format!("accept-encoding: {value}\r\n"));
            }
            request.push_str("\r\n");
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(body.as_bytes()).await.unwrap();
            let mut received = Vec::new();
            let mut buffer = [0u8; 8192];
            loop {
                let n = stream.read(&mut buffer).await.unwrap();
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&buffer[..n]);
                if let Some(i) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8(received[..i].to_vec()).unwrap();
                    let length = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length: ")
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if received.len() >= i + 4 + length {
                        let body = String::from_utf8(received[i + 4..i + 4 + length].to_vec())
                            .unwrap_or_default();
                        return (head, body);
                    }
                }
            }
            (String::from_utf8_lossy(&received).to_string(), String::new())
        }
        // Missing header: refused with the attachment-error shape.
        let (head, body) = roundtrip(&endpoint, &capture_body, None).await;
        assert!(
            head.starts_with("HTTP/1.1 400 Bad Request"),
            "missing accept-encoding must be refused, got: {head}"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap(),
            json!({"error":{"message":"unsupported_profile","type":"occ3_attachment_error","code":400}})
        );
        // A non-identity value is refused exactly like the live server.
        let (head, _) = roundtrip(&endpoint, &capture_body, Some("gzip")).await;
        assert!(
            head.starts_with("HTTP/1.1 400 Bad Request"),
            "non-identity accept-encoding must be refused, got: {head}"
        );
        // Exactly one identity header is admitted (200 SSE capture).
        let (head, _) = roundtrip(&endpoint, &capture_body, Some("identity")).await;
        assert!(
            head.starts_with("HTTP/1.1 200 OK"),
            "exactly one accept-encoding: identity must be accepted, got: {head}"
        );
        server.abort();
    }
}
