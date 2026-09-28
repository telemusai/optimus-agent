//! A lazy, persistent Clang-Repl process owned by one session.
use super::ToolDefinition;
use crate::utils::child_process::{spawn_kernel, KernelProcess, SpawnOptions};
use pi_agent_core::types::{
    AgentToolResult, AgentToolUpdateCallback, ContentBlock, ToolExecutionMode,
};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

const LIMIT: usize = 64 * 1024;

#[derive(Default)]
pub struct ClangRuntime {
    process: tokio::sync::Mutex<Option<ClangProcess>>,
    shutdown: CancellationToken,
}

struct ClangProcess {
    child: KernelProcess,
    input: tokio::process::ChildStdin,
    output: tokio::sync::mpsc::Receiver<(usize, Vec<u8>)>,
    readers: Vec<tokio::task::JoinHandle<()>>,
    source: tempfile::TempDir,
}

fn executable() -> String {
    if let Some(path) = std::env::var_os("OPTIMUS_CLANG_REPL") {
        return path.to_string_lossy().into_owned();
    }
    for name in std::iter::once("clang-repl".to_owned())
        .chain((13..=23).rev().map(|v| format!("clang-repl-{v}")))
    {
        for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
            let path = directory.join(if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.clone()
            });
            if path.is_file() {
                return path.to_string_lossy().into_owned();
            }
        }
    }
    // Distribution compiler symlinks often point into an LLVM bin directory
    // that contains clang-repl but is not itself on PATH.
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let compiler = directory.join(if cfg!(windows) { "clang.exe" } else { "clang" });
        if let Ok(compiler) = compiler.canonicalize() {
            if let Some(directory) = compiler.parent() {
                let repl = directory.join(if cfg!(windows) {
                    "clang-repl.exe"
                } else {
                    "clang-repl"
                });
                if repl.is_file() {
                    return repl.to_string_lossy().into_owned();
                }
            }
        }
    }
    "clang-repl".into()
}

fn reader(
    mut input: impl AsyncRead + Unpin + Send + 'static,
    channel: tokio::sync::mpsc::Sender<(usize, Vec<u8>)>,
    stream: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut buffer = [0; 4096];
        loop {
            match input.read(&mut buffer).await {
                Ok(0) | Err(_) => {
                    let _ = channel.send((stream, Vec::new())).await;
                    break;
                }
                Ok(n) => {
                    if channel.send((stream, buffer[..n].to_vec())).await.is_err() {
                        break;
                    }
                }
            }
        }
    })
}

impl ClangProcess {
    fn spawn(cwd: &str) -> Result<Self, String> {
        let source = tempfile::Builder::new()
            .prefix("optimus-clang-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let mut child = spawn_kernel(&executable(), &["--Xcc=-std=c++17".into(), "--Xcc=-fno-color-diagnostics".into(), format!("--Xcc=-I{cwd}")], SpawnOptions {
            cwd: Some(cwd.into()), detached: true, stdin_piped: true, capture_stdout: true, capture_stderr: true, ..Default::default()
        }).map_err(|e| format!("Cannot start Clang-Repl: {e}. Install LLVM's clang-repl and C++ development headers, or set OPTIMUS_CLANG_REPL to its executable."))?;
        let input = child.stdin.take().ok_or("Clang-Repl stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("Clang-Repl stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("Clang-Repl stderr unavailable")?;
        let (send, output) = tokio::sync::mpsc::channel(16);
        let readers = vec![reader(stdout, send.clone(), 0), reader(stderr, send, 1)];
        Ok(Self {
            child,
            input,
            output,
            readers,
            source,
        })
    }

    fn terminate(&mut self) {
        #[cfg(windows)]
        let _ = self.child.containment.terminate();
        #[cfg(not(windows))]
        if let Some(pid) = self.child.id() {
            crate::utils::shell::kill_process_tree(pid as i32);
            let _ = self.child.start_kill();
        }
    }

    async fn stop(&mut self) {
        self.terminate();
        let _ = tokio::time::timeout(Duration::from_secs(3), self.child.wait()).await;
        for reader in &self.readers {
            reader.abort();
        }
    }

    async fn cell(
        &mut self,
        code: &str,
        cwd: &str,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let path = self.source.path().join(format!("cell_{id}.cpp"));
        // Each complete fragment is parsed together, preserving multiline C++,
        // comments and preprocessor directives without an interactive PTY.
        tokio::fs::write(&path, code)
            .await
            .map_err(|e| e.to_string())?;
        let include = path.to_string_lossy().replace('\\', "/");
        if include.contains(['"', '\n', '\r']) {
            return Err("Clang-Repl temporary path contains unsupported characters".into());
        }
        let marker = format!("\nOPTIMUS_CLANG_DONE_{id}\n");
        let escaped = serde_json::to_string(&marker).map_err(|e| e.to_string())?;
        let request = format!("#include <cstdio>\n#include \"{include}\"\nint optimus_done_{id} = (std::fputs({escaped}, stdout), std::fflush(stdout), std::fputs({escaped}, stderr), std::fflush(stderr), 0);\n");
        self.input
            .write_all(request.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        self.input.flush().await.map_err(|e| e.to_string())?;
        let mut pending = [Vec::new(), Vec::new()];
        let mut done = [false; 2];
        let mut outputs = [Vec::new(), Vec::new()];
        let mut last_update = tokio::time::Instant::now();
        while !done.iter().all(|v| *v) {
            let (stream, bytes) = self
                .output
                .recv()
                .await
                .ok_or("Clang-Repl exited before completing the cell")?;
            if bytes.is_empty() {
                return Err(format!(
                    "Clang-Repl exited before completing the cell. {}",
                    String::from_utf8_lossy(&outputs[1])
                ));
            }
            if done[stream] {
                continue;
            }
            pending[stream].extend(bytes);
            let found = pending[stream]
                .windows(marker.len())
                .position(|window| window == marker.as_bytes());
            let count = found.unwrap_or_else(|| pending[stream].len().saturating_sub(marker.len()));
            let retain = count.min(LIMIT.saturating_sub(outputs[stream].len()));
            outputs[stream].extend_from_slice(&pending[stream][..retain]);
            pending[stream].drain(..count);
            if found.is_some() {
                pending[stream].clear();
                done[stream] = true;
            }
            if last_update.elapsed() >= Duration::from_millis(100) {
                if let Some(update) = &on_update {
                    update(cell_result(&outputs, cwd));
                }
                last_update = tokio::time::Instant::now();
            }
        }
        Ok(cell_result(&outputs, cwd))
    }
}

impl Drop for ClangProcess {
    fn drop(&mut self) {
        self.terminate();
        for reader in &self.readers {
            reader.abort();
        }
    }
}

fn cell_result(output: &[Vec<u8>; 2], cwd: &str) -> AgentToolResult {
    let stdout = String::from_utf8_lossy(&output[0]);
    let stderr = String::from_utf8_lossy(&output[1]);
    let mut text = format!("{stdout}{stderr}");
    if output.iter().any(|s| s.len() == LIMIT) {
        text.push_str("\n[Clang-Repl output truncated at 64 KiB per stream]");
    }
    if text.trim().is_empty() {
        text = "(no output)".into();
    }
    let mut result = AgentToolResult::new(
        vec![ContentBlock::text(text)],
        json!({"language":"cpp","cwd":cwd}),
    );
    result.is_error = Some(stderr.contains("error:") || stderr.contains("Parsing failed"));
    result
}

impl ClangRuntime {
    pub fn cancel(&self) {
        self.shutdown.cancel();
        if let Ok(mut guard) = self.process.try_lock() {
            if let Some(mut process) = guard.take() {
                process.terminate();
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(async move {
                        process.stop().await;
                    });
                }
            }
        }
    }

    pub async fn dispose(&self) {
        self.shutdown.cancel();
        if let Some(mut process) = self.process.lock().await.take() {
            process.stop().await;
        }
    }

    pub async fn execute(
        &self,
        cwd: &str,
        code: &str,
        timeout: f64,
        signal: Option<CancellationToken>,
        update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        if !timeout.is_finite() || timeout <= 0.0 || timeout > 3600.0 {
            return Err(
                "Clang-Repl timeout must be greater than zero and at most 3600 seconds".into(),
            );
        }
        if code.len() > 1_000_000 {
            return Err("Clang-Repl code exceeds the 1 MB cell limit".into());
        }
        let signal = signal.unwrap_or_default();
        let mut guard = tokio::select! {
            guard = self.process.lock() => guard,
            _ = signal.cancelled() => return Err("Clang-Repl cell cancelled before execution".into()),
            _ = self.shutdown.cancelled() => return Err("Clang-Repl workspace disposed".into()),
        };
        if signal.is_cancelled() || self.shutdown.is_cancelled() {
            return Err("Clang-Repl cell cancelled before execution".into());
        }
        let mut process = match guard.take() {
            Some(process) => process,
            None => ClangProcess::spawn(cwd)?,
        };
        let outcome = tokio::select! {
            result = process.cell(code, cwd, update) => result,
            _ = tokio::time::sleep(Duration::from_secs_f64(timeout)) => Err(format!("Clang-Repl cell timed out after {timeout} seconds")),
            _ = signal.cancelled() => Err("Clang-Repl cell cancelled".into()),
            _ = self.shutdown.cancelled() => Err("Clang-Repl workspace disposed".into()),
        };
        match outcome {
            Ok(result) => {
                *guard = Some(process);
                Ok(result)
            }
            Err(error) => {
                process.stop().await;
                Err(format!("{error}. Clang-Repl workspace reset; previous C++ state is no longer available."))
            }
        }
    }
}

pub fn create_clang_tool(cwd: &str, runtime: Arc<ClangRuntime>) -> ToolDefinition<Value> {
    let cwd = PathBuf::from(cwd).to_string_lossy().into_owned();
    ToolDefinition {
        name: "clang".into(), label: "Clang-Repl".into(),
        description: "Execute C++17 fragments in a persistent Clang-Repl workspace. Includes, global variables, functions and classes persist. Execute statements using an immediately invoked lambda, e.g. auto result = [] { std::printf(\"hello\\n\"); return 0; }();. Print results explicitly. No main(), Markdown fences or percent commands. Timeout/cancellation resets this workspace. Requires clang-repl and C++ headers.".into(),
        parameters: json!({"type":"object","properties":{"code":{"type":"string"},"timeout":{"type":"number","exclusiveMinimum":0,"maximum":3600}},"required":["code"]}),
        execution_mode: Some(ToolExecutionMode::Sequential),
        execute: Arc::new(move |_, args, signal, update, _| {
            let runtime = runtime.clone(); let cwd = cwd.clone();
            Box::pin(async move {
                let code = args["code"].as_str().ok_or_else(|| anyhow::anyhow!("code must be a string"))?;
                runtime.execute(&cwd, code, args["timeout"].as_f64().unwrap_or(60.0), signal, update).await.map_err(anyhow::Error::msg)
            })
        }),
        ..Default::default()
    }
}
