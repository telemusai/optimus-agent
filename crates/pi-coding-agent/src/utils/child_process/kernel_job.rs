//! Native kernel-only containment. Handles never come from a PID lookup.
//! JOB_LIST makes containment part of CreateProcess, before any interpreter code.
use super::SpawnOptions;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io;
use std::mem::{size_of, size_of_val};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::ExitStatusExt;
use std::ptr::{null, null_mut};
use std::sync::Arc;
use std::time::Duration;
use windows_sys::Win32::Foundation::{
    SetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::System::JobObjects::{
    CreateJobObjectW, JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
    QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess,
    InitializeProcThreadAttributeList, ResumeThread, UpdateProcThreadAttribute, WaitForSingleObject,
    CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT,
    LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
    PROC_THREAD_ATTRIBUTE_JOB_LIST, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

/// Both handles are non-inheritable and stay alive across root exit and repeated stop.
pub struct KernelJob {
    job: OwnedHandle,
    process: OwnedHandle,
}

impl KernelJob {
    /// Root exit is a signaled *owned process handle*, not closed stdio or PID absence.
    pub fn probe(&self) -> io::Result<(bool, bool)> {
        let exited = match unsafe { WaitForSingleObject(self.process.as_raw_handle(), 0) } {
            WAIT_OBJECT_0 => true,
            WAIT_TIMEOUT => false,
            _ => return Err(io::Error::last_os_error()),
        };
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        check(unsafe {
            QueryInformationJobObject(
                self.job.as_raw_handle(), JobObjectBasicAccountingInformation,
                (&mut info as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                size_of_val(&info) as u32, null_mut(),
            )
        })?;
        Ok((exited, info.ActiveProcesses == 0))
    }

    pub fn terminate(&self) -> io::Result<()> {
        // Job closure is a crash backstop, not an acknowledgement. Keep it open
        // until the caller has queried the actual active process count.
        check(unsafe { TerminateJobObject(self.job.as_raw_handle(), 1) })
    }

    pub async fn wait_settled(&self, deadline: tokio::time::Instant) -> io::Result<()> {
        loop {
            if self.probe()? == (true, true) {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "Kernel Job did not settle before deadline"));
            }
            tokio::time::sleep_until((tokio::time::Instant::now() + Duration::from_millis(10)).min(deadline)).await;
        }
    }

    async fn wait_root(&self) -> io::Result<std::process::ExitStatus> {
        loop {
            match unsafe { WaitForSingleObject(self.process.as_raw_handle(), 0) } {
                WAIT_OBJECT_0 => {
                    let mut code = 0;
                    check(unsafe { GetExitCodeProcess(self.process.as_raw_handle(), &mut code) })?;
                    return Ok(std::process::ExitStatus::from_raw(code));
                }
                WAIT_TIMEOUT => tokio::time::sleep(Duration::from_millis(10)).await,
                _ => return Err(io::Error::last_os_error()),
            }
        }
    }
}

pub struct KernelProcess {
    pub stdin: Option<tokio::process::ChildStdin>,
    pub stdout: Option<tokio::process::ChildStdout>,
    pub stderr: Option<tokio::process::ChildStderr>,
    pub containment: Arc<KernelJob>,
    pid: u32,
}

impl KernelProcess {
    pub fn id(&self) -> Option<u32> { Some(self.pid) }
    pub async fn wait(&mut self) -> io::Result<std::process::ExitStatus> {
        self.containment.wait_root().await
    }
}

fn check(ok: i32) -> io::Result<()> {
    if ok == 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

fn wide(value: &OsStr) -> io::Result<Vec<u16>> {
    let mut result: Vec<u16> = value.encode_wide().collect();
    if result.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "NUL in process argument"));
    }
    result.push(0);
    Ok(result)
}

// The exact CommandLineToArgv/CRT quoting rule, including backslashes before
// quotes and the closing quote. Never a shell, so shell metacharacters are inert.
fn quote(value: &str) -> String {
    let mut out = String::from("\"");
    let mut slashes = 0;
    for ch in value.chars() {
        if ch == '\\' { slashes += 1; continue; }
        if ch == '"' {
            out.extend(std::iter::repeat_n('\\', slashes * 2 + 1));
        } else {
            out.extend(std::iter::repeat_n('\\', slashes));
        }
        slashes = 0;
        out.push(ch);
    }
    out.extend(std::iter::repeat_n('\\', slashes * 2));
    out.push('"');
    out
}

fn environment(options: &SpawnOptions) -> io::Result<Vec<u16>> {
    let mut vars = BTreeMap::new();
    if !options.replace_env {
        for (key, value) in std::env::vars_os() {
            // Keep original UTF-16 values; only the lookup/sort key is folded.
            vars.insert(key.to_string_lossy().to_uppercase(), (key, value));
        }
    }
    if let Some(extra) = &options.env {
        for (key, value) in extra {
            // Native Windows blocks may contain hidden names with one leading
            // '=' (for example =C: and cmd's =ExitCode). std::env::vars includes
            // them, so preserve them when a manager forwards its full env.
            // Only the next '=' is the name/value delimiter; embedded '=' and
            // NUL remain invalid names, and NUL remains invalid in values.
            let name = key.strip_prefix('=').unwrap_or(key);
            if name.is_empty() || name.contains('=')
                || key.contains('\0') || value.contains('\0')
            {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid process environment"));
            }
            vars.insert(key.to_uppercase(), (key.into(), value.into()));
        }
    }
    let mut block = Vec::new();
    for (_, (key, value)) in vars {
        block.extend(key.encode_wide());
        block.push('=' as u16);
        block.extend(value.encode_wide());
        block.push(0);
    }
    if block.is_empty() { block.push(0); }
    block.push(0);
    Ok(block)
}

fn pipe(child_reads: bool) -> io::Result<(OwnedHandle, OwnedHandle)> {
    let mut read = null_mut();
    let mut write = null_mut();
    check(unsafe { CreatePipe(&mut read, &mut write, null(), 0) })?;
    // Every success path immediately puts handles under RAII ownership.
    let read = unsafe { OwnedHandle::from_raw_handle(read) };
    let write = unsafe { OwnedHandle::from_raw_handle(write) };
    let (child, parent) = if child_reads { (read, write) } else { (write, read) };
    check(unsafe { SetHandleInformation(child.as_raw_handle(), HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) })?;
    Ok((child, parent))
}

struct Attributes {
    // usize storage guarantees native pointer alignment; zero-sized byte buffers do not.
    _storage: Vec<usize>,
    ptr: LPPROC_THREAD_ATTRIBUTE_LIST,
}
impl Attributes {
    fn new() -> io::Result<Self> {
        let mut bytes = 0;
        unsafe { InitializeProcThreadAttributeList(null_mut(), 2, 0, &mut bytes); }
        if bytes == 0 { return Err(io::Error::last_os_error()); }
        let mut storage = vec![0usize; bytes.div_ceil(size_of::<usize>())];
        let ptr = storage.as_mut_ptr().cast();
        check(unsafe { InitializeProcThreadAttributeList(ptr, 2, 0, &mut bytes) })?;
        Ok(Self { _storage: storage, ptr })
    }
    fn handles(&mut self, kind: u32, handles: &[HANDLE]) -> io::Result<()> {
        check(unsafe { UpdateProcThreadAttribute(self.ptr, 0, kind as usize,
            handles.as_ptr().cast(), size_of_val(handles), null_mut(), null()) })
    }
}
impl Drop for Attributes {
    fn drop(&mut self) { unsafe { DeleteProcThreadAttributeList(self.ptr); } }
}

pub fn spawn_kernel(command: &str, args: &[String], options: SpawnOptions) -> io::Result<KernelProcess> {
    if options.shell || !options.stdin_piped || !options.capture_stdout || !options.capture_stderr {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Native kernel requires direct argv and three pipes"));
    }
    let mut command_line = wide(OsStr::new(&std::iter::once(command).chain(args.iter().map(String::as_str)).map(quote).collect::<Vec<_>>().join(" ")))?;
    let env = environment(&options)?;
    let cwd = options.cwd.as_ref().map(|value| wide(OsStr::new(value))).transpose()?;
    let raw_job = unsafe { CreateJobObjectW(null(), null()) };
    if raw_job.is_null() { return Err(io::Error::last_os_error()); }
    let job = unsafe { OwnedHandle::from_raw_handle(raw_job) };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    check(unsafe { SetInformationJobObject(job.as_raw_handle(), JobObjectExtendedLimitInformation,
        (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), size_of_val(&limits) as u32) })?;
    let (child_in, parent_in) = pipe(true)?;
    let (child_out, parent_out) = pipe(false)?;
    let (child_err, parent_err) = pipe(false)?;
    let jobs = [job.as_raw_handle()];
    let inherited = [child_in.as_raw_handle(), child_out.as_raw_handle(), child_err.as_raw_handle()];
    let mut attrs = Attributes::new()?;
    attrs.handles(PROC_THREAD_ATTRIBUTE_JOB_LIST, &jobs)?;
    attrs.handles(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &inherited)?;
    let mut startup = STARTUPINFOEXW::default();
    startup.StartupInfo.cb = size_of_val(&startup) as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inherited[0];
    startup.StartupInfo.hStdOutput = inherited[1];
    startup.StartupInfo.hStdError = inherited[2];
    startup.lpAttributeList = attrs.ptr;
    let mut info = PROCESS_INFORMATION::default();
    check(unsafe { CreateProcessW(null(), command_line.as_mut_ptr(), null(), null(), 1,
        EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        env.as_ptr().cast(), cwd.as_ref().map_or(null(), |value| value.as_ptr()),
        &startup.StartupInfo, &mut info) })?;
    let process = unsafe { OwnedHandle::from_raw_handle(info.hProcess) };
    let thread = unsafe { OwnedHandle::from_raw_handle(info.hThread) };
    let containment = Arc::new(KernelJob { job, process });
    // The process is already in the Job even if any conversion/resume fails.
    // Dropping the last containment handle then kills the still-suspended process.
    let stdin = tokio::process::ChildStdin::from_std(parent_in.into())?;
    let stdout = tokio::process::ChildStdout::from_std(parent_out.into())?;
    let stderr = tokio::process::ChildStderr::from_std(parent_err.into())?;
    if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
        return Err(io::Error::last_os_error());
    }
    Ok(KernelProcess { stdin: Some(stdin), stdout: Some(stdout), stderr: Some(stderr), containment, pid: info.dwProcessId })
}

#[cfg(test)]
mod containment_tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    fn python() -> String {
        std::env::var("KERNEL_CONTAINMENT_TEST_PYTHON").expect("Set KERNEL_CONTAINMENT_TEST_PYTHON to the controlled native Python interpreter")
    }
    fn options() -> SpawnOptions {
        SpawnOptions { stdin_piped: true, capture_stdout: true, capture_stderr: true,
            env: Some(vec![("PYTHONUTF8".into(), "1".into())]), ..Default::default() }
    }
    fn spawn(code: &str) -> KernelProcess {
        spawn_kernel(&python(), &["-u".into(), "-c".into(), code.into()], options()).unwrap()
    }
    async fn line(reader: &mut BufReader<tokio::process::ChildStdout>) -> String {
        let mut out = String::new();
        tokio::time::timeout(Duration::from_secs(10), reader.read_line(&mut out)).await.unwrap().unwrap();
        out.trim().into()
    }
    async fn settled(job: &KernelJob) {
        job.terminate().unwrap();
        job.wait_settled(tokio::time::Instant::now() + Duration::from_secs(5)).await.unwrap();
        assert_eq!(job.probe().unwrap(), (true, true));
    }

    #[tokio::test]
    async fn native_job_exact_argv_env_and_protocol_pipes() {
        let values = ["", "space and tab\t", "a\\\"b", "trailing\\", "雪", "&|<>%!"];
        let code = "import os,sys; print('|'.join(sys.argv[1:]),flush=True); print(os.environ['CONTAINMENT_FIXTURE'],flush=True); print(sys.stdin.readline().strip(),flush=True); print('stderr-ok',file=sys.stderr,flush=True)";
        let mut opts = options();
        opts.env.as_mut().unwrap().push(("CONTAINMENT_FIXTURE".into(), "kept".into()));
        let mut args = vec!["-u".into(), "-c".into(), code.into()];
        args.extend(values.map(String::from));
        let mut child = spawn_kernel(&python(), &args, opts).unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(b"message\n").await.unwrap();
        drop(stdin);
        let mut out = String::new();
        tokio::time::timeout(Duration::from_secs(10), child.stdout.take().unwrap().read_to_string(&mut out)).await.unwrap().unwrap();
        assert_eq!(out.replace("\r\n", "\n"), format!("{}\nkept\nmessage\n", values.join("|")));
        let mut err = String::new();
        child.stderr.take().unwrap().read_to_string(&mut err).await.unwrap();
        assert_eq!(err.trim(), "stderr-ok");
        assert!(child.wait().await.unwrap().success());
        settled(&child.containment).await;
        settled(&child.containment).await;
    }

    #[tokio::test]
    async fn native_job_immediate_detached_grandchild_survives_root_but_not_job() {
        // Both children spawn at their first opportunity. No post-spawn Job
        // assignment can race this fixture. A natural root exit reparents them.
        let code = r#"import subprocess,sys
subprocess.Popen([sys.executable,'-u','-c',"import subprocess,sys,threading; subprocess.Popen([sys.executable,'-u','-c',\"import threading; print('grandchild',flush=True); threading.Event().wait()\"],creationflags=subprocess.DETACHED_PROCESS,stdout=sys.stdout,stderr=sys.stderr); print('child',flush=True); threading.Event().wait()"],creationflags=subprocess.DETACHED_PROCESS,stdout=sys.stdout,stderr=sys.stderr)
print('root',flush=True)
"#;
        let mut child = spawn(code);
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut lines = Vec::new();
        for _ in 0..3 { lines.push(line(&mut stdout).await); }
        lines.sort();
        assert_eq!(lines, ["child", "grandchild", "root"]);
        assert!(tokio::time::timeout(Duration::from_secs(5), child.wait()).await.unwrap().unwrap().success());
        assert_eq!(child.containment.probe().unwrap(), (true, false));
        settled(&child.containment).await;
        assert_eq!(line(&mut stdout).await, "");
    }

    #[tokio::test]
    async fn native_job_breakaway_cannot_escape_outer_membership() {
        use windows_sys::Win32::System::JobObjects::IsProcessInJob;
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        let code = r#"import subprocess,sys,threading
try:
    p=subprocess.Popen([sys.executable,'-u','-c',"import os,threading; print('pid:'+str(os.getpid()),flush=True); threading.Event().wait(10)"],creationflags=subprocess.CREATE_BREAKAWAY_FROM_JOB,stdout=sys.stdout,stderr=sys.stderr)
    print('pid:'+str(p.pid),flush=True)
    p.wait()
except OSError as e:
    print('denied:'+str(e.winerror),flush=True)
"#;
        let mut child = spawn(code);
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let response = line(&mut stdout).await;
        let mut held = Vec::new();
        if response != "denied:5" {
            // Check BOTH the venv launcher and the real interpreter, regardless
            // of which reports first. A successful spawn is not proof of escape.
            let responses = [response, line(&mut stdout).await];
            for response in responses {
                let pid = response.strip_prefix("pid:").unwrap().parse::<u32>().unwrap();
                let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | 0x0010_0000, 0, pid) };
                assert!(!handle.is_null());
                let handle = unsafe { OwnedHandle::from_raw_handle(handle) };
                let mut member = 0;
                check(unsafe { IsProcessInJob(handle.as_raw_handle(), child.containment.job.as_raw_handle(), &mut member) }).unwrap();
                assert_eq!(member, 1, "attempted breakaway escaped the kernel's outer Job");
                held.push(handle);
            }
        }
        settled(&child.containment).await;
        for handle in held {
            assert_eq!(unsafe { WaitForSingleObject(handle.as_raw_handle(), 0) }, WAIT_OBJECT_0);
        }
    }

    #[tokio::test]
    async fn native_job_stubborn_root_timeout_is_not_settled() {
        let mut child = spawn("import threading; print('ready',flush=True); threading.Event().wait()");
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        assert_eq!(line(&mut stdout).await, "ready");
        let error = child.containment.wait_settled(tokio::time::Instant::now()).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_eq!(child.containment.probe().unwrap(), (false, false));
        settled(&child.containment).await;
    }

    #[tokio::test]
    async fn native_job_nested_runtime_bash_and_worker_job() {
        let code = r#"import asyncio,os,sys,threading
from rlm import _winjob
from rlm.bash import bash
async def check():
    result=await bash('printf bash-ok')
    assert result.exit_code==0 and result.output=='bash-ok',repr(result)
asyncio.run(check())
job=_winjob.create_job()
assert job is not None
p=_winjob.spawn_in_job(job,[sys.executable,'-u','-c',"import threading; print('inner',flush=True); threading.Event().wait()"],os.getcwd(),dict(os.environ))
assert p.resume()
assert p.stdout.readline().strip()==b'inner'
assert _winjob.is_empty(job) is False
print('nested-ready',flush=True)
threading.Event().wait()
"#;
        let mut opts = options();
        opts.env.as_mut().unwrap().extend([
            ("PYTHONPATH".into(), std::env::var("KERNEL_CONTAINMENT_TEST_RUNTIME_SRC").unwrap()),
            ("PRIME_AGENT_BASH_SHELL".into(), std::env::var("KERNEL_CONTAINMENT_TEST_BASH").unwrap()),
            ("PRIME_AGENT_INTERNAL_ORPHAN_PROCESS_JOURNAL".into(), std::env::var("KERNEL_CONTAINMENT_TEST_JOURNAL").unwrap()),
        ]);
        let mut child = spawn_kernel(&python(), &["-u".into(), "-c".into(), code.into()], opts).unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let message = line(&mut stdout).await;
        if message != "nested-ready" {
            let mut stderr = String::new();
            let _ = tokio::time::timeout(Duration::from_secs(1), child.stderr.take().unwrap().read_to_string(&mut stderr)).await;
            panic!("nested fixture: {message:?}; {stderr}");
        }
        settled(&child.containment).await;
    }

    #[tokio::test]
    async fn native_job_spawn_failure_is_closed_and_no_target_runs() {
        assert!(spawn_kernel("Z:\\nonexistent-kernel-fixture\\python.exe", &["-c".into(), "pass".into()], options()).is_err());
        let mut opts = options();
        opts.env.as_mut().unwrap().push(("BAD".into(), "embedded\0value".into()));
        assert!(spawn_kernel(&python(), &["-c".into(), "pass".into()], opts).is_err());
    }


    #[tokio::test]
    async fn native_job_failed_query_and_termination_are_errors() {
        let child = spawn("import threading; threading.Event().wait()");
        let wrong = KernelJob {
            job: child.containment.process.try_clone().unwrap(),
            process: child.containment.process.try_clone().unwrap(),
        };
        assert!(wrong.terminate().is_err());
        assert!(wrong.probe().is_err());
        assert!(wrong.wait_settled(tokio::time::Instant::now()).await.is_err());
        settled(&child.containment).await;
    }

    #[tokio::test]
    async fn native_job_only_stdio_handles_are_inherited() {
        use windows_sys::Win32::Foundation::GetHandleInformation;
        use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
        use windows_sys::Win32::System::Threading::CreateEventW;
        let attrs = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: null_mut(), bInheritHandle: 1,
        };
        let raw = unsafe { CreateEventW(&attrs, 1, 0, null()) };
        assert!(!raw.is_null());
        let sentinel = unsafe { OwnedHandle::from_raw_handle(raw) };
        let code = format!(r#"import ctypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.SetEvent.argtypes=[ctypes.c_void_p]
assert not k.SetEvent({})
assert ctypes.get_last_error()==6
print('restricted',flush=True)
"#, sentinel.as_raw_handle() as usize);
        let mut child = spawn(&code);
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        assert_eq!(line(&mut stdout).await, "restricted");
        for handle in [&child.containment.job, &child.containment.process] {
            let mut flags = 0;
            check(unsafe { GetHandleInformation(handle.as_raw_handle(), &mut flags) }).unwrap();
            assert_eq!(flags & HANDLE_FLAG_INHERIT, 0);
        }
        assert_eq!(unsafe { WaitForSingleObject(sentinel.as_raw_handle(), 0) }, WAIT_TIMEOUT);
        settled(&child.containment).await;
    }


    #[test]
    fn native_job_environment_preserves_hidden_keys_but_rejects_invalid_names() {
        for hidden_name in ["=C:", "=z:", "=ExitCode", "=ExitCodeAscii"] {
            let opts = SpawnOptions {
                replace_env: true,
                env: Some(vec![
                    (hidden_name.into(), "Z:\\synthetic directory".into()),
                    ("FIXTURE".into(), "left=right".into()),
                    ("Path".into(), "first".into()),
                    ("PATH".into(), "second".into()),
                ]), ..Default::default()
            };
            let block = String::from_utf16(&environment(&opts).unwrap()).unwrap();
            assert_eq!(block, format!("{hidden_name}=Z:\\synthetic directory\0FIXTURE=left=right\0PATH=second\0\0"));
        }
        for key in ["", "BAD=KEY", "=", "==", "=Exit=Code", "=C:=", "NUL\0KEY", "=NUL\0KEY"] {
            let opts = SpawnOptions { replace_env: true,
                env: Some(vec![(key.into(), "synthetic".into())]), ..Default::default() };
            assert_eq!(environment(&opts).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        }
        let opts = SpawnOptions { replace_env: true,
            env: Some(vec![("=C:".into(), "embedded\0value".into())]), ..Default::default() };
        assert_eq!(environment(&opts).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn native_job_manager_style_full_environment_spawns_and_settles() {
        let inherited: Vec<_> = std::env::vars().collect();
        let rejected_by_old_rule = inherited.iter().filter(|(key, _)| key.contains('=')).count();
        let hidden_keys = inherited.iter().filter(|(key, _)| {
            key.strip_prefix('=').is_some_and(|name| !name.is_empty() && !name.contains('='))
        }).count();
        // Only counts are logged: never environment values or nonstandard names.
        eprintln!("environment-name evidence: old-rule-rejects={rejected_by_old_rule}, hidden-native-keys={hidden_keys}; values omitted");
        assert_eq!(rejected_by_old_rule, hidden_keys);
        let mut opts = options();
        opts.env = Some(inherited);
        opts.env.as_mut().unwrap().extend([
            ("=Q:".into(), "Q:\\contained-synthetic-directory".into()),
            ("=ExitCode".into(), "37".into()),
            ("CONTAINMENT_ENV_FIXTURE".into(), "manager-env-ok".into()),
        ]);
        let code = r#"import ctypes,os
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetEnvironmentStringsW.restype=ctypes.c_void_p
k.FreeEnvironmentStringsW.argtypes=[ctypes.c_void_p]
block=k.GetEnvironmentStringsW()
assert block
cursor=block
found_drive=False
found_exitcode=False
try:
    while True:
        entry=ctypes.wstring_at(cursor)
        if not entry:
            break
        found_drive |= entry == '=Q:=Q:\\contained-synthetic-directory'
        found_exitcode |= entry == '=ExitCode=37'
        cursor += len(entry.encode('utf-16-le')) + 2
finally:
    k.FreeEnvironmentStringsW(block)
assert found_drive and found_exitcode
assert os.environ['CONTAINMENT_ENV_FIXTURE']=='manager-env-ok'
print('manager-env-ok',flush=True)
"#;
        let mut child = spawn_kernel(&python(), &["-u".into(), "-c".into(), code.into()], opts).unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        assert_eq!(line(&mut stdout).await, "manager-env-ok");
        assert!(child.wait().await.unwrap().success());
        settled(&child.containment).await;
    }
}
