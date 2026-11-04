//! Owner-only no-replace local publication for the native checkpoint proof
//! producer. This is a self-contained copy of the phase-suite example's
//! `private_publish` primitive (same staged exclusive create, protected
//! owner DACL, bounded bytes, flush/readback, reparse/single-link checks and
//! atomic no-overwrite finalization), placed under `examples/support/` so it
//! is NOT auto-registered as an executable example target and the pinned
//! harness example source stays untouched (its source-manifest digests pin
//! those exact bytes). Do not shrink these guarantees.
//!
//! Non-Windows targets fail closed: no replace-style fallback exists.

#[cfg(windows)]
mod imp {
    use pi_ai::source_capture::sha256;
    use std::path::Path;
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

    /// Publication byte bound (same magnitude as the suite's report bound).
    const MAX_PUBLISH_BYTES: usize = 2 * 1024 * 1024;

    fn fail() -> String {
        "owner-only no-replace publication refused or uncertain".into()
    }

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
        if bytes.is_empty() || bytes.len() > MAX_PUBLISH_BYTES || final_name.is_empty() {
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
                let read = read_all(handle, MAX_PUBLISH_BYTES)?;
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
pub use imp::{publish_exclusive, read_file_strict};

#[cfg(not(windows))]
/// Platform fail-closed: no replace-style fallback exists anywhere.
pub fn publish_exclusive(_directory: &Path, _final_name: &str, _bytes: &[u8]) -> Result<(), String> {
    Err("windows_only_no_replace_publication_refused".into())
}

#[cfg(not(windows))]
pub fn read_file_strict(_path: &Path, _limit: usize) -> Result<Vec<u8>, String> {
    Err("windows_only_strict_read_refused".into())
}

use std::path::Path;

/// Bounded strict read: delegates to the owner-only strict reader on Windows;
/// the non-Windows arm mirrors the suite's std fallback (symlink/refusal,
/// regular-file requirement, size bound before allocation). Shared by every
/// consumed external document (descriptor, checkpoint, claim).
pub fn strict_read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    #[cfg(windows)]
    {
        read_file_strict(path, limit)
    }
    #[cfg(not(windows))]
    {
        let metadata = std::fs::symlink_metadata(path).map_err(|_| "file_too_large".to_string())?;
        if metadata.file_type().is_symlink() {
            return Err("symlink_refused".to_string());
        }
        if !metadata.is_file() {
            return Err("file_too_large".to_string());
        }
        if metadata.len() as usize > limit {
            return Err("file_too_large".to_string());
        }
        std::fs::read(path).map_err(|_| "file_too_large".to_string())
    }
}
