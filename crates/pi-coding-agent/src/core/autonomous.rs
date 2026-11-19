//! Port of packages/coding-agent/src/core/autonomous.ts

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pi_ai::types::{AssistantMessage, ImageOrTextContent, Usage, UserContent, UserMessage};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use crate::utils::child_process::{spawn_hidden, wait_for_child_process, SpawnOptions};
use crate::utils::shell::{
    kill_owned_child_process_tree, track_detached_child, untrack_detached_child, OwnedChildProcess,
};

/// `interface AgentAutonomousConfig`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentAutonomousConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_continuations: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuation_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gates: Option<AgentAutonomousGateConfig>,
}

/// `interface AgentAutonomousGateConfig`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentAutonomousGateConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commands: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<f64>,
}

/// `interface AgentAutonomousGateFailure`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAutonomousGateFailure {
    pub command: String,
    pub attempt: f64,
    pub exit_text: String,
    pub output: String,
}

/// `Required<Omit<AgentAutonomousConfig, "enabled" | "continuationPrompt" | "gates">>`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAutonomousLimits {
    pub max_continuations: f64,
    pub max_turns: f64,
    pub max_tokens: f64,
    pub timeout_ms: f64,
}

/// `Required<AgentAutonomousGateConfig>`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAutonomousGates {
    pub commands: Vec<String>,
    pub max_retries: f64,
    pub timeout_ms: f64,
}

/// `interface AgentAutonomousStatus`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAutonomousStatus {
    pub enabled: bool,
    pub continuations_used: f64,
    pub turns_used: f64,
    pub tokens_used: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<f64>,
    pub limits: AgentAutonomousLimits,
    pub gates: AgentAutonomousGates,
    pub gate_attempts: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_gate_failure: Option<AgentAutonomousGateFailure>,
}

pub const DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT: &str = "No human input is available in autonomous mode. Continue working until the host evaluator, verifier, or configured autonomous limits stop the run. If you were asking the user a question, make a reasonable assumption and verify it. If you believe you are blocked, prove it with host-observable evidence, preserve that evidence, and keep looking for safe progress while budget remains. Do not end the session yourself; the verifier/evaluator decides completion when configured gates pass.";

/// `DEFAULT_AUTONOMOUS_LIMITS`.
pub fn default_autonomous_limits() -> AgentAutonomousLimits {
    AgentAutonomousLimits {
        max_continuations: 3.0,
        max_turns: 12.0,
        max_tokens: 80_000.0,
        timeout_ms: 30.0 * 60.0 * 1000.0,
    }
}

/// `DEFAULT_AUTONOMOUS_GATES`.
pub fn default_autonomous_gates() -> AgentAutonomousGates {
    AgentAutonomousGates {
        commands: Vec::new(),
        max_retries: 3.0,
        timeout_ms: 5.0 * 60.0 * 1000.0,
    }
}

const MAX_GATE_OUTPUT_CHARS: usize = 6000;
const MAX_CHILD_PROCESS_OUTPUT_CHARS: usize = 1024 * 1024;

/// `interface GitWorktreeSnapshot`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GitWorktreeSnapshot {
    status: String,
    diff: String,
    untracked_hash: String,
}

/// `interface AutonomousRuntimeState`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRuntimeState {
    pub enabled: bool,
    pub continuations_used: f64,
    pub turns_used: f64,
    pub tokens_used: f64,
    pub started_at: Option<f64>,
    pub limits: AgentAutonomousLimits,
    pub continuation_prompt: String,
    pub gates: AgentAutonomousGates,
    pub gate_attempts: BTreeMap<String, f64>,
    pub last_gate_failure: Option<GateFailure>,
    pub last_gate_failure_snapshot: Option<GitWorktreeSnapshot>,
}

/// `type GateFailure = AgentAutonomousGateFailure`.
type GateFailure = AgentAutonomousGateFailure;

/// `type AutonomousLimitReason`.
pub const AUTONOMOUS_LIMIT_MAX_CONTINUATIONS: &str = "maxContinuations";
pub const AUTONOMOUS_LIMIT_MAX_TURNS: &str = "maxTurns";
pub const AUTONOMOUS_LIMIT_MAX_TOKENS: &str = "maxTokens";
pub const AUTONOMOUS_LIMIT_TIMEOUT_MS: &str = "timeoutMs";

/// `type AutonomousGateResult`.
pub const GATE_RESULT_PASSED: &str = "passed";
pub const GATE_RESULT_FAILED: &str = "failed";
pub const GATE_RESULT_RETRY_EXHAUSTED: &str = "retry_exhausted";

/// `type AutonomousLimitState = Pick<AgentAutonomousStatus, ...>`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousLimitState {
    pub continuations_used: f64,
    pub turns_used: f64,
    pub tokens_used: f64,
    pub started_at: Option<f64>,
    pub limits: AgentAutonomousLimits,
}

impl From<&AutonomousRuntimeState> for AutonomousLimitState {
    fn from(state: &AutonomousRuntimeState) -> Self {
        Self {
            continuations_used: state.continuations_used,
            turns_used: state.turns_used,
            tokens_used: state.tokens_used,
            started_at: state.started_at,
            limits: state.limits.clone(),
        }
    }
}

/// `interface AutonomousDecision`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousDecision {
    pub should_continue: bool,
    pub reason: String,
}

pub const DECISION_MISSING_TERMINAL_EVIDENCE: &str = "missing_terminal_evidence";
pub const DECISION_GATE_FAILED: &str = "gate_failed";
pub const DECISION_NOT_NEEDED: &str = "not_needed";
pub const DECISION_LIMIT_REACHED: &str = "limit_reached";

/// `interface AutonomousOperationOptions`.
#[derive(Debug, Clone, Default)]
pub struct AutonomousOperationOptions {
    pub cwd: Option<String>,
    /// `signal?: AbortSignal`.
    pub signal: Option<CancellationToken>,
}

/// `createAutonomousRuntimeState(config, _options = {})`.
pub fn create_autonomous_runtime_state(config: Option<&AgentAutonomousConfig>) -> AutonomousRuntimeState {
    let enabled = config.and_then(|config| config.enabled).unwrap_or(false);
    let defaults = default_autonomous_limits();
    let gate_defaults = default_autonomous_gates();
    AutonomousRuntimeState {
        enabled,
        continuations_used: 0.0,
        turns_used: 0.0,
        tokens_used: 0.0,
        started_at: if enabled { Some(now_millis()) } else { None },
        limits: AgentAutonomousLimits {
            max_continuations: normalize_limit(
                config.and_then(|config| config.max_continuations),
                defaults.max_continuations,
            ),
            max_turns: normalize_limit(config.and_then(|config| config.max_turns), defaults.max_turns),
            max_tokens: normalize_limit(config.and_then(|config| config.max_tokens), defaults.max_tokens),
            timeout_ms: normalize_limit(config.and_then(|config| config.timeout_ms), defaults.timeout_ms),
        },
        continuation_prompt: config
            .and_then(|config| config.continuation_prompt.as_deref())
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT.to_string()),
        gates: AgentAutonomousGates {
            commands: config
                .and_then(|config| config.gates.as_ref())
                .and_then(|gates| gates.commands.clone())
                .unwrap_or_else(|| gate_defaults.commands.clone()),
            max_retries: normalize_limit(
                config
                    .and_then(|config| config.gates.as_ref())
                    .and_then(|gates| gates.max_retries),
                gate_defaults.max_retries,
            ),
            timeout_ms: normalize_limit(
                config
                    .and_then(|config| config.gates.as_ref())
                    .and_then(|gates| gates.timeout_ms),
                gate_defaults.timeout_ms,
            ),
        },
        gate_attempts: BTreeMap::new(),
        last_gate_failure: None,
        last_gate_failure_snapshot: None,
    }
}

/// `setAutonomousEnabled(state, enabled, _options = {})`.
pub fn set_autonomous_enabled(state: &mut AutonomousRuntimeState, enabled: bool) {
    state.enabled = enabled;
    if enabled {
        state.continuations_used = 0.0;
        state.turns_used = 0.0;
        state.tokens_used = 0.0;
        state.started_at = Some(now_millis());
    } else {
        state.started_at = None;
    }
    state.gate_attempts = BTreeMap::new();
    state.last_gate_failure = None;
    state.last_gate_failure_snapshot = None;
}

/// `autonomousStatus(state)`.
pub fn autonomous_status(state: &AutonomousRuntimeState) -> AgentAutonomousStatus {
    AgentAutonomousStatus {
        enabled: state.enabled,
        continuations_used: state.continuations_used,
        turns_used: state.turns_used,
        tokens_used: state.tokens_used,
        started_at: state.started_at,
        limits: state.limits.clone(),
        gates: state.gates.clone(),
        gate_attempts: state.gate_attempts.clone(),
        last_gate_failure: state.last_gate_failure.clone(),
    }
}

/// `addAutonomousUsage(state, usage)`.
pub fn add_autonomous_usage(state: &mut AutonomousRuntimeState, usage: Option<&Usage>) {
    if !state.enabled {
        return;
    }
    state.turns_used += 1.0;
    state.tokens_used += autonomous_token_delta(usage);
}

/// `addAutonomousContinuation(state)`.
pub fn add_autonomous_continuation(state: &mut AutonomousRuntimeState) {
    if !state.enabled {
        return;
    }
    state.continuations_used += 1.0;
}

/// `autonomousTokenDelta(usage)`.
fn autonomous_token_delta(usage: Option<&Usage>) -> f64 {
    let Some(usage) = usage else {
        return 0.0;
    };
    // Cache-read tokens are repeated context served from provider cache. Counting them
    // cumulatively makes long autonomous verifier loops exhaust their host-side token
    // budget far before the non-cached work reaches the configured cap.
    usage.input + usage.output + usage.cache_write
}

/// `nextAutonomousContinuation(state, message, options = {}, now = Date.now())`.
pub async fn next_autonomous_continuation(
    state: &mut AutonomousRuntimeState,
    message: &AssistantMessage,
    options: AutonomousOperationOptions,
    now: f64,
) -> Result<Option<UserMessage>, AutonomousAbortedError> {
    throw_if_aborted(options.signal.as_ref())?;
    if !state.enabled {
        return Ok(None);
    }
    let decision = should_autonomously_continue(state, message, options.clone(), now).await?;
    throw_if_aborted(options.signal.as_ref())?;
    if !decision.should_continue {
        return Ok(None);
    }
    state.continuations_used += 1.0;
    let text = if decision.reason == DECISION_GATE_FAILED {
        build_gate_failure_continuation(state, now).unwrap_or_else(|| state.continuation_prompt.clone())
    } else {
        state.continuation_prompt.clone()
    };
    Ok(Some(UserMessage {
        role: pi_ai::types::ROLE_USER.to_string(),
        content: UserContent::Blocks(vec![ImageOrTextContent::Text(pi_ai::types::TextContent::new(text))]),
        provider_context: None,
        timestamp: now as i64,
    }))
}

/// `shouldAutonomouslyContinue(state, message, options = {}, now = Date.now())`.
pub async fn should_autonomously_continue(
    state: &mut AutonomousRuntimeState,
    message: &AssistantMessage,
    options: AutonomousOperationOptions,
    now: f64,
) -> Result<AutonomousDecision, AutonomousAbortedError> {
    throw_if_aborted(options.signal.as_ref())?;
    if !state.enabled
        || message.stop_reason == pi_ai::types::STOP_REASON_ERROR
        || message.stop_reason == pi_ai::types::STOP_REASON_ABORTED
    {
        return Ok(AutonomousDecision {
            should_continue: false,
            reason: DECISION_NOT_NEEDED.to_string(),
        });
    }
    let gate_result = refresh_autonomous_quality_gates(state, options.clone()).await?;
    throw_if_aborted(options.signal.as_ref())?;
    if let Some(gate_result) = gate_result {
        if gate_result == GATE_RESULT_PASSED {
            return Ok(AutonomousDecision {
                should_continue: false,
                reason: DECISION_NOT_NEEDED.to_string(),
            });
        }
        if gate_result == GATE_RESULT_RETRY_EXHAUSTED || autonomous_limit_reason(state, now).is_some() {
            return Ok(AutonomousDecision {
                should_continue: false,
                reason: DECISION_LIMIT_REACHED.to_string(),
            });
        }
        return Ok(AutonomousDecision {
            should_continue: true,
            reason: DECISION_GATE_FAILED.to_string(),
        });
    }
    if autonomous_limit_reason(state, now).is_some() {
        return Ok(AutonomousDecision {
            should_continue: false,
            reason: DECISION_LIMIT_REACHED.to_string(),
        });
    }
    Ok(AutonomousDecision {
        should_continue: true,
        reason: DECISION_MISSING_TERMINAL_EVIDENCE.to_string(),
    })
}

/// `signal?.throwIfAborted()`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousAbortedError {
    pub message: String,
}

impl std::fmt::Display for AutonomousAbortedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AutonomousAbortedError {}

fn throw_if_aborted(signal: Option<&CancellationToken>) -> Result<(), AutonomousAbortedError> {
    if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
        return Err(AutonomousAbortedError {
            message: "This operation was aborted".to_string(),
        });
    }
    Ok(())
}

/// `autonomousLimitReason(state, now = Date.now())`.
pub fn autonomous_limit_reason(state: &AutonomousRuntimeState, now: f64) -> Option<&'static str> {
    autonomous_limit_reason_for(&AutonomousLimitState::from(state), now)
}

/// `autonomousLimitReason(state, now)` for the projected limit state.
pub fn autonomous_limit_reason_for(state: &AutonomousLimitState, now: f64) -> Option<&'static str> {
    if state.continuations_used >= state.limits.max_continuations {
        return Some(AUTONOMOUS_LIMIT_MAX_CONTINUATIONS);
    }
    if state.turns_used >= state.limits.max_turns {
        return Some(AUTONOMOUS_LIMIT_MAX_TURNS);
    }
    if state.tokens_used >= state.limits.max_tokens {
        return Some(AUTONOMOUS_LIMIT_MAX_TOKENS);
    }
    if let Some(started_at) = state.started_at {
        if now - started_at >= state.limits.timeout_ms {
            return Some(AUTONOMOUS_LIMIT_TIMEOUT_MS);
        }
    }
    None
}

/// `refreshAutonomousQualityGates(state, options = {})`.
pub async fn refresh_autonomous_quality_gates(
    state: &mut AutonomousRuntimeState,
    options: AutonomousOperationOptions,
) -> Result<Option<String>, AutonomousAbortedError> {
    throw_if_aborted(options.signal.as_ref())?;
    if !state.enabled || state.gates.commands.is_empty() {
        return Ok(None);
    }
    let result = run_autonomous_quality_gates(state, options.cwd.clone(), options.signal.clone()).await?;
    Ok(Some(result))
}

/// `buildAutonomousGateFailureContinuation(failure, maxRetries, timestamp)`.
pub fn build_autonomous_gate_failure_continuation(
    failure: &AgentAutonomousGateFailure,
    max_retries: f64,
    timestamp: f64,
) -> String {
    format!(
        "Autonomous quality gate failed (attempt {}/{}): `{}` {}.\n{}\nContinue working. Fix the failure, then produce terminal evidence. Timestamp: {}.",
        format_number(failure.attempt),
        format_number(max_retries),
        failure.command,
        failure.exit_text,
        if failure.output.is_empty() {
            "\n".to_string()
        } else {
            format!("\nOutput:\n{}\n", failure.output)
        },
        iso_string(timestamp)
    )
}

/// `buildGateFailureContinuation(state, timestamp)`.
fn build_gate_failure_continuation(state: &AutonomousRuntimeState, timestamp: f64) -> Option<String> {
    let failure = state.last_gate_failure.as_ref()?;
    Some(build_autonomous_gate_failure_continuation(
        failure,
        state.gates.max_retries,
        timestamp,
    ))
}

/// `gitWorktreeSnapshotsEqual(a, b)`.
fn git_worktree_snapshots_equal(
    a: Option<&GitWorktreeSnapshot>,
    b: Option<&GitWorktreeSnapshot>,
) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.status == b.status && a.diff == b.diff && a.untracked_hash == b.untracked_hash,
        _ => false,
    }
}

/// `interface ChildProcessResult`.
#[derive(Debug, Clone, Default)]
struct ChildProcessResult {
    status: Option<i32>,
    /// `signal: NodeJS.Signals | null` - the port uses the numeric signal name.
    signal: Option<String>,
    stdout: String,
    stderr: String,
    error: Option<String>,
    timed_out: Option<bool>,
    output_truncated: bool,
}

/// `runAutonomousQualityGates(state, cwd, signal)`.
async fn run_autonomous_quality_gates(
    state: &mut AutonomousRuntimeState,
    cwd: Option<String>,
    signal: Option<CancellationToken>,
) -> Result<String, AutonomousAbortedError> {
    throw_if_aborted(signal.as_ref())?;
    let Some(cwd) = cwd else {
        return Ok(GATE_RESULT_FAILED.to_string());
    };
    for command in state.gates.commands.clone() {
        let current_snapshot = capture_git_worktree_snapshot(Some(cwd.clone()), signal.clone()).await?;
        throw_if_aborted(signal.as_ref())?;
        let reuse_previous_failure = state
            .last_gate_failure
            .as_ref()
            .map(|failure| failure.command == command)
            .unwrap_or(false)
            && git_worktree_snapshots_equal(current_snapshot.as_ref(), state.last_gate_failure_snapshot.as_ref());
        if reuse_previous_failure {
            let previous = state.last_gate_failure.clone().expect("checked above");
            let attempt = state
                .gate_attempts
                .get(&command)
                .copied()
                .unwrap_or(previous.attempt)
                + 1.0;
            state.gate_attempts.insert(command.clone(), attempt);
            state.last_gate_failure = Some(AgentAutonomousGateFailure {
                command: previous.command,
                attempt,
                exit_text: "not rerun: workspace unchanged since previous failed gate".to_string(),
                output: "The autonomous gate was not rerun because the workspace has not changed since this failure. Edit source files, tests, or a blocker artifact before attempting to finish again.".to_string(),
            });
            return Ok(if attempt > state.gates.max_retries {
                GATE_RESULT_RETRY_EXHAUSTED.to_string()
            } else {
                GATE_RESULT_FAILED.to_string()
            });
        }
        let result = run_child_process(
            &command,
            &[],
            ChildProcessOptions {
                cwd: Some(cwd.clone()),
                shell: true,
                timeout_ms: Some(state.gates.timeout_ms),
                max_output_chars: Some(MAX_GATE_OUTPUT_CHARS),
                signal: signal.clone(),
            },
        )
        .await;
        throw_if_aborted(signal.as_ref())?;
        let post_run_snapshot = capture_git_worktree_snapshot(Some(cwd.clone()), signal.clone()).await?;
        throw_if_aborted(signal.as_ref())?;
        if result.status == Some(0) && result.error.is_none() && result.timed_out != Some(true) {
            state.gate_attempts.insert(command.clone(), 0.0);
            if state
                .last_gate_failure
                .as_ref()
                .map(|failure| failure.command == command)
                .unwrap_or(false)
            {
                state.last_gate_failure = None;
                state.last_gate_failure_snapshot = None;
            }
            continue;
        }
        let attempt = state.gate_attempts.get(&command).copied().unwrap_or(0.0) + 1.0;
        state.gate_attempts.insert(command.clone(), attempt);
        let exit_text = format_process_exit(&result);
        state.last_gate_failure = Some(AgentAutonomousGateFailure {
            command: command.clone(),
            attempt,
            exit_text,
            output: truncate_gate_output(
                &[result.stdout.clone(), result.stderr.clone()]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<String>>()
                    .join("\n")
                    .trim()
                    .to_string(),
                result.output_truncated,
                None,
            ),
        });
        state.last_gate_failure_snapshot = post_run_snapshot;
        return Ok(if attempt > state.gates.max_retries {
            GATE_RESULT_RETRY_EXHAUSTED.to_string()
        } else {
            GATE_RESULT_FAILED.to_string()
        });
    }
    state.last_gate_failure = None;
    state.last_gate_failure_snapshot = None;
    Ok(GATE_RESULT_PASSED.to_string())
}

/// `captureGitWorktreeSnapshot(cwd, signal)`.
async fn capture_git_worktree_snapshot(
    cwd: Option<String>,
    signal: Option<CancellationToken>,
) -> Result<Option<GitWorktreeSnapshot>, AutonomousAbortedError> {
    throw_if_aborted(signal.as_ref())?;
    let Some(cwd) = cwd else {
        return Ok(None);
    };
    let pathspec = [
        "--".to_string(),
        ".".to_string(),
        ":(exclude)verification".to_string(),
        ":(exclude)target".to_string(),
        ":(exclude).vf-prime-agent".to_string(),
        ":(exclude)Cargo.lock".to_string(),
        ":(exclude)submission.tar.gz".to_string(),
        ":(exclude)runner_args.log".to_string(),
    ];
    let mut status_args = vec![
        "--no-optional-locks".to_string(),
        "status".to_string(),
        "--porcelain=v1".to_string(),
        "-z".to_string(),
        "-uall".to_string(),
        "--no-renames".to_string(),
    ];
    status_args.extend(pathspec.iter().cloned());
    let status = run_child_process(
        "git",
        &status_args,
        ChildProcessOptions {
            cwd: Some(cwd.clone()),
            timeout_ms: Some(10_000.0),
            signal: signal.clone(),
            ..Default::default()
        },
    )
    .await;
    throw_if_aborted(signal.as_ref())?;
    if status.status != Some(0) || status.error.is_some() || status.timed_out == Some(true) || status.output_truncated {
        return Ok(None);
    }
    let mut diff_args = vec![
        "--no-optional-locks".to_string(),
        "diff".to_string(),
        "--no-ext-diff".to_string(),
        "--binary".to_string(),
        "HEAD".to_string(),
    ];
    diff_args.extend(pathspec.iter().cloned());
    let diff = run_child_process(
        "git",
        &diff_args,
        ChildProcessOptions {
            cwd: Some(cwd.clone()),
            timeout_ms: Some(10_000.0),
            signal: signal.clone(),
            ..Default::default()
        },
    )
    .await;
    throw_if_aborted(signal.as_ref())?;
    if diff.status != Some(0) || diff.error.is_some() || diff.timed_out == Some(true) || diff.output_truncated {
        return Ok(None);
    }
    Ok(Some(GitWorktreeSnapshot {
        status: status.stdout.clone(),
        diff: diff.stdout.clone(),
        untracked_hash: hash_untracked_files(&cwd, &status.stdout, signal.clone()).await?,
    }))
}

/// `untrackedPathsFromStatus(status)`.
fn untracked_paths_from_status(status: &str) -> Vec<String> {
    let mut paths: Vec<String> = status
        .split('\0')
        .filter(|entry| entry.starts_with("?? "))
        .map(|entry| entry[3..].to_string())
        .collect();
    paths.sort();
    paths
}

/// `hashUntrackedFiles(cwd, status, signal)`.
async fn hash_untracked_files(
    cwd: &str,
    status: &str,
    signal: Option<CancellationToken>,
) -> Result<String, AutonomousAbortedError> {
    let mut aggregate = Sha256::new();
    for path in untracked_paths_from_status(status) {
        throw_if_aborted(signal.as_ref())?;
        aggregate.update(path.as_bytes());
        aggregate.update([0u8]);
        aggregate.update(
            hash_untracked_path(&resolve_path(cwd, &path), signal.clone())
                .await?
                .as_bytes(),
        );
        aggregate.update([0u8]);
    }
    throw_if_aborted(signal.as_ref())?;
    Ok(format!("{:x}", aggregate.finalize()))
}

/// `hashUntrackedPath(path, signal)`.
async fn hash_untracked_path(
    path: &str,
    signal: Option<CancellationToken>,
) -> Result<String, AutonomousAbortedError> {
    throw_if_aborted(signal.as_ref())?;
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) => {
            throw_if_aborted(signal.as_ref())?;
            return Ok(format!("error:{}", error));
        }
    };
    throw_if_aborted(signal.as_ref())?;
    if metadata.file_type().is_symlink() {
        let target = match std::fs::read_link(path) {
            Ok(target) => target.to_string_lossy().to_string(),
            Err(error) => {
                throw_if_aborted(signal.as_ref())?;
                return Ok(format!("error:{}", error));
            }
        };
        throw_if_aborted(signal.as_ref())?;
        return Ok(format!("symlink:{}", target));
    }
    if !metadata.is_file() {
        // `stat.mode`: the Unix permission bits Node reports; zero on Windows.
        let mode = file_mode(&metadata);
        return Ok(format!(
            "other:{}:{}:{}",
            mode,
            metadata.len(),
            metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|elapsed| elapsed.as_millis() as f64)
                .unwrap_or(0.0)
                .to_string()
        ));
    }
    let mut hasher = Sha256::new();
    let mut file = match tokio::fs::File::open(path).await {
        Ok(file) => file,
        Err(error) => {
            throw_if_aborted(signal.as_ref())?;
            return Ok(format!("error:{}", error));
        }
    };
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        match file.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => hasher.update(&buffer[..read]),
            Err(error) => {
                throw_if_aborted(signal.as_ref())?;
                return Ok(format!("error:{}", error));
            }
        }
    }
    throw_if_aborted(signal.as_ref())?;
    Ok(format!("file:{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn file_mode(metadata: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    metadata.mode()
}

#[cfg(not(unix))]
fn file_mode(_metadata: &std::fs::Metadata) -> u32 {
    0
}

/// `interface options` of `runChildProcess`.
#[derive(Debug, Clone, Default)]
struct ChildProcessOptions {
    cwd: Option<String>,
    shell: bool,
    timeout_ms: Option<f64>,
    max_output_chars: Option<usize>,
    signal: Option<CancellationToken>,
}

/// `runChildProcess(command, args, options)`.
async fn run_child_process(
    command: &str,
    args: &[String],
    options: ChildProcessOptions,
) -> ChildProcessResult {
    let Some(_) = check_signal(&options.signal) else {
        return ChildProcessResult::default();
    };
    let handle = match spawn_hidden(
        command,
        args,
        SpawnOptions {
            cwd: options.cwd.clone(),
            detached: !cfg!(windows),
            shell: options.shell,
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
    ) {
        Ok(handle) => handle,
        Err(error) => {
            return ChildProcessResult {
                error: Some(error.to_string()),
                ..Default::default()
            }
        }
    };
    let process = handle.child.owned_process().cloned();
    if let Some(process) = process.as_ref() {
        track_detached_child(process);
    }
    let mut child = handle.child;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let max_output_chars = options.max_output_chars.unwrap_or(MAX_CHILD_PROCESS_OUTPUT_CHARS);
    let truncated = Arc::new(AtomicBool::new(false));
    let stdout_output = Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_output = Arc::new(std::sync::Mutex::new(String::new()));

    let stdout_task = stdout.map(|stream| {
        let truncated = truncated.clone();
        let output = stdout_output.clone();
        tokio::spawn(async move { read_stream(stream, max_output_chars, truncated, output).await })
    });
    let stderr_task = stderr.map(|stream| {
        let truncated = truncated.clone();
        let output = stderr_output.clone();
        tokio::spawn(async move { read_stream(stream, max_output_chars, truncated, output).await })
    });

    let timed_out = Arc::new(AtomicBool::new(false));
    let mut signal_task = None;
    if let Some(signal) = options.signal.clone() {
        let timed_out = timed_out.clone();
        let process_for_signal = process.clone();
        signal_task = Some(tokio::spawn(async move {
            signal.cancelled().await;
            let _ = timed_out;
            process_for_signal
                .as_ref()
                .map(kill_autonomous_process_tree)
                .unwrap_or(true)
        }));
    }
    let timeout_task = match options.timeout_ms {
        Some(timeout_ms) => {
            let timed_out = timed_out.clone();
            let process_for_timeout = process.clone();
            Some(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(timeout_ms.max(0.0) as u64)).await;
                timed_out.store(true, Ordering::SeqCst);
                process_for_timeout
                    .as_ref()
                    .map(kill_autonomous_process_tree)
                    .unwrap_or(true)
            }))
        }
        None => None,
    };

    let status_result = wait_for_child_process(child).await;
    let timeout_cleanup_complete = match timeout_task {
        Some(task) => stop_autonomous_cleanup_task(task).await,
        None => true,
    };
    let signal_cleanup_complete = match signal_task {
        Some(task) => stop_autonomous_cleanup_task(task).await,
        None => true,
    };

    // The outer readers own the pipes taken above. The direct child can exit
    // while a descendant retains either write end, so join both readers under
    // one post-exit deadline instead of waiting for EOF without a bound.
    let output_drain_deadline = tokio::time::Instant::now()
        + Duration::from_millis(crate::utils::child_process::EXIT_STDIO_GRACE_MS);
    let reader_tasks = [stdout_task, stderr_task].into_iter().flatten().collect::<Vec<_>>();
    let reader_cleanup_complete =
        finish_stream_readers(reader_tasks, output_drain_deadline, process.clone()).await;
    let cleanup_complete =
        timeout_cleanup_complete && signal_cleanup_complete && reader_cleanup_complete;
    let stdout_text = take_stream_output(&stdout_output);
    let stderr_text = take_stream_output(&stderr_output);
    if let Some(process) = process.as_ref() {
        untrack_autonomous_child_with(process, cleanup_complete, untrack_detached_child);
    }
    let (status, signal_name, error) = match status_result {
        Ok(status) => (status, None, None),
        Err(error) => (None, None, Some(error.to_string())),
    };
    ChildProcessResult {
        status,
        signal: signal_name,
        stdout: stdout_text,
        stderr: stderr_text,
        error,
        timed_out: Some(timed_out.load(Ordering::SeqCst)),
        output_truncated: truncated.load(Ordering::SeqCst),
    }
}

/// Stop a timeout/signal task after the direct wait settles. A task cancelled
/// before firing made no cleanup attempt; a completed false result or panic
/// fails closed and retains the registration for retry.
async fn stop_autonomous_cleanup_task(task: tokio::task::JoinHandle<bool>) -> bool {
    task.abort();
    match task.await {
        Ok(cleanup_complete) => cleanup_complete,
        Err(error) if error.is_cancelled() => true,
        Err(_) => false,
    }
}

/// Kill through live-tree cleanup, including group members that outlive the leader.
fn kill_autonomous_process_tree(process: &OwnedChildProcess) -> bool {
    kill_owned_child_process_tree(process)
}

/// The production path supplies the registry untrack operation. Tests inject an
/// observer so a false cleanup outcome is proven to retain the exact generation
/// without signalling or registering a real process.
fn untrack_autonomous_child_with(
    process: &OwnedChildProcess,
    cleanup_complete: bool,
    untrack: impl FnOnce(&OwnedChildProcess),
) -> bool {
    if !cleanup_complete {
        return false;
    }
    untrack(process);
    true
}

/// Finish every outer reader against the same remaining gate. If a pipe holder
/// prevents EOF, stop its process group and settle every JoinHandle before
/// returning the output captured so far.
async fn finish_stream_readers(
    tasks: Vec<tokio::task::JoinHandle<()>>,
    deadline: tokio::time::Instant,
    process: Option<OwnedChildProcess>,
) -> bool {
    finish_stream_readers_with(tasks, deadline, process, kill_autonomous_process_tree).await
}

async fn finish_stream_readers_with(
    mut tasks: Vec<tokio::task::JoinHandle<()>>,
    deadline: tokio::time::Instant,
    process: Option<OwnedChildProcess>,
    kill: impl FnOnce(&OwnedChildProcess) -> bool,
) -> bool {
    let mut drain_timed_out = false;
    let mut completed_tasks = 0;
    for task in &mut tasks {
        if tokio::time::timeout_at(deadline, task).await.is_err() {
            drain_timed_out = true;
            break;
        }
        // Await through &mut consumes the JoinHandle's output. Do not poll
        // completed handles again during final abort/settlement below.
        completed_tasks += 1;
    }
    let cleanup_complete = if drain_timed_out && tasks.iter().any(|task| !task.is_finished()) {
        process.as_ref().map(kill).unwrap_or(true)
    } else {
        true
    };
    for (index, task) in tasks.into_iter().enumerate() {
        if index < completed_tasks {
            continue;
        }
        if !task.is_finished() {
            task.abort();
        }
        let _ = task.await;
    }
    cleanup_complete
}

fn take_stream_output(output: &Arc<std::sync::Mutex<String>>) -> String {
    let mut output = output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::take(&mut *output)
}

fn check_signal(signal: &Option<CancellationToken>) -> Option<()> {
    if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
        return None;
    }
    Some(())
}

/// Reads a child stream, capping stored characters at `max_output_chars`.
/// The shared buffer retains partial output if the owner has to abort the read.
async fn read_stream(
    mut stream: impl tokio::io::AsyncRead + Unpin,
    max_output_chars: usize,
    truncated: Arc<AtomicBool>,
    output: Arc<std::sync::Mutex<String>>,
) {
    let mut buffer = vec![0u8; 8192];
    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                let mut output = output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                let remaining = max_output_chars.saturating_sub(output.chars().count());
                if remaining > 0 {
                    output.extend(chunk.chars().take(remaining));
                }
                if chunk.chars().count() > remaining {
                    truncated.store(true, Ordering::SeqCst);
                }
            }
            Err(_) => break,
        }
    }
}

/// `formatProcessExit(result)`.
fn format_process_exit(result: &ChildProcessResult) -> String {
    if result.timed_out == Some(true) {
        return "timed out".to_string();
    }
    if let Some(error) = &result.error {
        return error.clone();
    }
    match &result.signal {
        Some(signal) => format!("terminated by {signal}"),
        None => format!(
            "exited {}",
            result
                .status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
    }
}

/// `truncateGateOutput(output, outputAlreadyTruncated = false, maxChars = MAX_GATE_OUTPUT_CHARS)`.
fn truncate_gate_output(output: &str, output_already_truncated: bool, max_chars: Option<usize>) -> String {
    let max_chars = max_chars.unwrap_or(MAX_GATE_OUTPUT_CHARS);
    if output.chars().count() <= max_chars && !output_already_truncated {
        return output.to_string();
    }
    let head: String = output.chars().take(max_chars).collect();
    format!("{head}\n... [truncated]")
}

/// `normalizeLimit(value, fallback)`.
fn normalize_limit(value: Option<f64>, fallback: f64) -> f64 {
    match value {
        Some(value) if value.is_finite() && value > 0.0 => value.trunc(),
        _ => fallback,
    }
}

/// `Date.now()`.
fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

/// `new Date(timestamp).toISOString()`.
fn iso_string(millis: f64) -> String {
    let seconds = (millis / 1000.0).floor() as i64;
    let nanos = ((millis - (seconds as f64) * 1000.0) * 1_000_000.0).round() as u32;
    match chrono::DateTime::from_timestamp(seconds, nanos.min(999_999_999)) {
        Some(datetime) => datetime.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        None => "Invalid Date".to_string(),
    }
}

/// `String(number)` for the template interpolations above.
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// `path.resolve(cwd, path)`.
fn resolve_path(cwd: &str, path: &str) -> String {
    let candidate = std::path::PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    std::path::PathBuf::from(cwd)
        .join(candidate)
        .components()
        .collect::<std::path::PathBuf>()
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AgentAutonomousConfig {
        AgentAutonomousConfig {
            enabled: Some(true),
            ..Default::default()
        }
    }

    #[test]
    fn disabled_config_leaves_the_state_disabled_with_default_limits() {
        let state = create_autonomous_runtime_state(None);
        assert!(!state.enabled);
        assert!(state.started_at.is_none());
        assert_eq!(state.limits, default_autonomous_limits());
        assert_eq!(state.gates, default_autonomous_gates());
        assert_eq!(state.continuation_prompt, DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT);
    }

    #[test]
    fn limits_are_normalized_and_blank_prompts_fall_back() {
        let state = create_autonomous_runtime_state(Some(&AgentAutonomousConfig {
            enabled: Some(true),
            max_continuations: Some(0.0),
            max_turns: Some(2.9),
            max_tokens: Some(f64::NAN),
            timeout_ms: Some(-5.0),
            continuation_prompt: Some("   ".to_string()),
            gates: Some(AgentAutonomousGateConfig {
                commands: Some(vec!["one".to_string()]),
                max_retries: Some(1.0),
                timeout_ms: None,
            }),
        }));
        assert_eq!(state.limits.max_continuations, 3.0);
        assert_eq!(state.limits.max_turns, 2.0);
        assert_eq!(state.limits.max_tokens, 80_000.0);
        assert_eq!(state.limits.timeout_ms, 30.0 * 60.0 * 1000.0);
        assert_eq!(state.continuation_prompt, DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT);
        assert_eq!(state.gates.commands, vec!["one".to_string()]);
        assert_eq!(state.gates.max_retries, 1.0);
        assert_eq!(state.gates.timeout_ms, 5.0 * 60.0 * 1000.0);
    }

    #[test]
    fn set_enabled_resets_counters_and_gate_state() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        state.continuations_used = 2.0;
        state.turns_used = 4.0;
        state.tokens_used = 100.0;
        state.gate_attempts.insert("npm test".to_string(), 1.0);
        state.last_gate_failure = Some(AgentAutonomousGateFailure {
            command: "npm test".to_string(),
            attempt: 1.0,
            exit_text: "exit 1".to_string(),
            output: String::new(),
        });
        set_autonomous_enabled(&mut state, true);
        assert_eq!(state.continuations_used, 0.0);
        assert_eq!(state.turns_used, 0.0);
        assert_eq!(state.tokens_used, 0.0);
        assert!(state.started_at.is_some());
        assert!(state.gate_attempts.is_empty());
        assert!(state.last_gate_failure.is_none());

        set_autonomous_enabled(&mut state, false);
        assert!(state.started_at.is_none());
    }

    #[test]
    fn usage_counts_turns_and_skips_cache_reads() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        let usage = Usage {
            input: 10.0,
            output: 5.0,
            cache_read: 1000.0,
            cache_write: 2.0,
            total_tokens: 1017.0,
            ..Default::default()
        };
        add_autonomous_usage(&mut state, Some(&usage));
        assert_eq!(state.turns_used, 1.0);
        assert_eq!(state.tokens_used, 17.0);

        add_autonomous_usage(&mut state, None);
        assert_eq!(state.turns_used, 2.0);
        assert_eq!(state.tokens_used, 17.0);
    }

    #[test]
    fn usage_and_continuations_are_ignored_while_disabled() {
        let mut state = create_autonomous_runtime_state(None);
        add_autonomous_usage(&mut state, Some(&Usage::zero()));
        add_autonomous_continuation(&mut state);
        assert_eq!(state.turns_used, 0.0);
        assert_eq!(state.continuations_used, 0.0);
    }

    #[test]
    fn limit_reasons_are_checked_in_the_typescript_order() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        state.started_at = Some(0.0);
        assert_eq!(autonomous_limit_reason(&state, 0.0), None);
        state.continuations_used = 3.0;
        assert_eq!(autonomous_limit_reason(&state, 0.0), Some("maxContinuations"));
        state.continuations_used = 0.0;
        state.turns_used = 12.0;
        assert_eq!(autonomous_limit_reason(&state, 0.0), Some("maxTurns"));
        state.turns_used = 0.0;
        state.tokens_used = 80_000.0;
        assert_eq!(autonomous_limit_reason(&state, 0.0), Some("maxTokens"));
        state.tokens_used = 0.0;
        assert_eq!(
            autonomous_limit_reason(&state, 30.0 * 60.0 * 1000.0),
            Some("timeoutMs")
        );
    }

    #[test]
    fn status_projection_copies_limits_gates_and_failures() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        state.gate_attempts.insert("cmd".to_string(), 2.0);
        state.last_gate_failure = Some(AgentAutonomousGateFailure {
            command: "cmd".to_string(),
            attempt: 2.0,
            exit_text: "exit 1".to_string(),
            output: "boom".to_string(),
        });
        let status = autonomous_status(&state);
        assert!(status.enabled);
        assert_eq!(status.gate_attempts.get("cmd"), Some(&2.0));
        assert_eq!(status.last_gate_failure.as_ref().unwrap().output, "boom");
        assert_eq!(status.gates, state.gates);
    }

    #[test]
    fn gate_failure_continuation_matches_the_typescript_text() {
        let text = build_autonomous_gate_failure_continuation(
            &AgentAutonomousGateFailure {
                command: "npm test".to_string(),
                attempt: 2.0,
                exit_text: "exit 1".to_string(),
                output: "failure output".to_string(),
            },
            3.0,
            0.0,
        );
        assert_eq!(
            text,
            "Autonomous quality gate failed (attempt 2/3): `npm test` exit 1.\n\nOutput:\nfailure output\n\nContinue working. Fix the failure, then produce terminal evidence. Timestamp: 1970-01-01T00:00:00.000Z."
        );

        let text = build_autonomous_gate_failure_continuation(
            &AgentAutonomousGateFailure {
                command: "npm test".to_string(),
                attempt: 1.0,
                exit_text: "exit 1".to_string(),
                output: String::new(),
            },
            3.0,
            0.0,
        );
        assert!(text.contains("`npm test` exit 1.\n\n\nContinue working."));
    }

    #[test]
    fn truncate_gate_output_appends_the_marker() {
        assert_eq!(truncate_gate_output("short", false, None), "short");
        let long = "x".repeat(7000);
        assert!(truncate_gate_output(&long, false, None).ends_with("\n... [truncated]"));
        assert!(truncate_gate_output("short", true, None).ends_with("\n... [truncated]"));
    }

    #[test]
    fn untracked_paths_are_parsed_from_porcelain_v1_z_output() {
        let status = "?? b.txt\0 M a.txt\0?? a.txt\0";
        assert_eq!(untracked_paths_from_status(status), vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn format_process_exit_prefers_timeout_then_error_then_signal() {
        assert_eq!(
            format_process_exit(&ChildProcessResult {
                timed_out: Some(true),
                ..Default::default()
            }),
            "timed out"
        );
        assert_eq!(
            format_process_exit(&ChildProcessResult {
                error: Some("spawn failed".to_string()),
                ..Default::default()
            }),
            "spawn failed"
        );
        assert_eq!(
            format_process_exit(&ChildProcessResult {
                signal: Some("SIGKILL".to_string()),
                ..Default::default()
            }),
            "terminated by SIGKILL"
        );
        assert_eq!(format_process_exit(&ChildProcessResult::default()), "exited unknown");
        assert_eq!(
            format_process_exit(&ChildProcessResult {
                status: Some(1),
                ..Default::default()
            }),
            "exited 1"
        );
    }

    #[tokio::test]
    async fn completed_reader_task_is_not_polled_twice() {
        let task = tokio::spawn(async {});
        tokio::task::yield_now().await;
        assert!(finish_stream_readers_with(
            vec![task],
            tokio::time::Instant::now() + Duration::from_secs(1),
            None,
            |_| panic!("completed readers need no process cleanup"),
        )
        .await);
    }

    #[test]
    fn failed_autonomous_cleanup_retains_the_exact_generation_for_retry() {
        let process = crate::utils::shell::register_owned_child_process(i32::MAX)
            .expect("synthetic process registration");
        assert!(!untrack_autonomous_child_with(
            &process,
            false,
            |_| panic!("failed cleanup must retain its generation"),
        ));

        let mut untracked = Vec::new();
        assert!(untrack_autonomous_child_with(
            &process,
            true,
            |candidate| untracked.push(candidate.clone()),
        ));
        assert_eq!(untracked, vec![process.clone()]);
        crate::utils::shell::release_owned_child_process(&process);
    }

    #[tokio::test]
    async fn timeout_or_signal_cleanup_outcome_is_not_discarded() {
        let task = tokio::spawn(async { false });
        while !task.is_finished() {
            tokio::task::yield_now().await;
        }
        assert!(!stop_autonomous_cleanup_task(task).await);

        // A task aborted before it fired did not attempt a failed kill and must
        // not spuriously retain a successfully settled direct child.
        let task = tokio::spawn(futures::future::pending::<bool>());
        assert!(stop_autonomous_cleanup_task(task).await);
    }

    #[tokio::test]
    async fn failed_reader_cleanup_is_returned_after_bounded_task_settlement() {
        let process = crate::utils::shell::register_owned_child_process(i32::MAX)
            .expect("synthetic process registration");
        let task = tokio::spawn(futures::future::pending::<()>());
        let task_handle = task.abort_handle();
        let mut kill_attempts = Vec::new();
        let cleanup_complete = tokio::time::timeout(
            Duration::from_secs(1),
            finish_stream_readers_with(
                vec![task],
                tokio::time::Instant::now(),
                Some(process.clone()),
                |candidate| {
                    kill_attempts.push(candidate.clone());
                    false
                },
            ),
        )
        .await
        .expect("reader abort and join must stay bounded");
        assert!(!cleanup_complete);
        assert!(task_handle.is_finished());
        assert_eq!(kill_attempts, vec![process.clone()]);
        crate::utils::shell::release_owned_child_process(&process);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn direct_early_eof_returns_without_waiting_for_the_drain_deadline() {
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            run_child_process(
                "printf early-direct-eof; exit 0",
                &[],
                ChildProcessOptions {
                    shell: true,
                    timeout_ms: Some(5_000.0),
                    ..Default::default()
                },
            ),
        )
        .await
        .expect("ordinary direct EOF must not wait for an inherited-pipe timeout");
        assert_eq!(result.status, Some(0));
        assert_eq!(result.stdout, "early-direct-eof");
        assert_eq!(result.timed_out, Some(false));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn inherited_pipe_holder_is_killed_after_the_direct_leader_exits() {
        let directory = tempfile::tempdir().unwrap();
        let pid_path = directory.path().join("holder.pid");
        let command = format!(
            "sleep 30 & holder=$!; printf '%s' \"$holder\" > '{}'; printf leader-exited; exit 0",
            pid_path.to_string_lossy()
        );
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            run_child_process(
                &command,
                &[],
                ChildProcessOptions {
                    shell: true,
                    timeout_ms: Some(5_000.0),
                    ..Default::default()
                },
            ),
        )
        .await
        .expect("a descendant holding stdout/stderr must not hang the gate");
        assert_eq!(result.status, Some(0));
        assert_eq!(result.stdout, "leader-exited");
        assert_eq!(result.timed_out, Some(false));

        let holder_pid = std::fs::read_to_string(&pid_path).unwrap().parse::<i32>().unwrap();
        let cleanup_deadline = std::time::Instant::now() + Duration::from_secs(2);
        while crate::utils::child_process::is_process_alive(holder_pid)
            && std::time::Instant::now() < cleanup_deadline
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !crate::utils::child_process::is_process_alive(holder_pid),
            "the pipe-holding descendant survived group cleanup"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_returns_partial_output_and_settles_the_reader_tasks() {
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            run_child_process(
                "printf timeout-partial; sleep 30",
                &[],
                ChildProcessOptions {
                    shell: true,
                    timeout_ms: Some(50.0),
                    ..Default::default()
                },
            ),
        )
        .await
        .expect("timeout cleanup and partial-output collection must be bounded");
        assert_eq!(result.timed_out, Some(true));
        assert_eq!(result.stdout, "timeout-partial");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn abort_returns_partial_output_and_settles_the_reader_tasks() {
        let directory = tempfile::tempdir().unwrap();
        let ready_path = directory.path().join("ready");
        let command = format!(
            "printf abort-partial; : > '{}'; sleep 30",
            ready_path.to_string_lossy()
        );
        let signal = CancellationToken::new();
        let task_signal = signal.clone();
        let mut task = tokio::spawn(async move {
            run_child_process(
                &command,
                &[],
                ChildProcessOptions {
                    shell: true,
                    signal: Some(task_signal),
                    ..Default::default()
                },
            )
            .await
        });
        let ready_deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !ready_path.exists() && std::time::Instant::now() < ready_deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(ready_path.exists(), "the child did not reach its partial-output marker");
        signal.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), &mut task)
            .await
            .expect("abort cleanup and partial-output collection must be bounded")
            .unwrap();
        assert_eq!(result.timed_out, Some(false));
        assert_eq!(result.stdout, "abort-partial");
    }

    #[tokio::test]
    async fn passing_gates_clear_the_recorded_failure() {
        let mut state = create_autonomous_runtime_state(Some(&AgentAutonomousConfig {
            enabled: Some(true),
            gates: Some(AgentAutonomousGateConfig {
                commands: Some(vec!["exit 0".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        }));
        state.last_gate_failure = Some(AgentAutonomousGateFailure {
            command: "exit 0".to_string(),
            attempt: 1.0,
            exit_text: "exited 1".to_string(),
            output: String::new(),
        });
        let directory = tempfile::tempdir().unwrap();
        let result = refresh_autonomous_quality_gates(
            &mut state,
            AutonomousOperationOptions {
                cwd: Some(directory.path().to_string_lossy().to_string()),
                signal: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.as_deref(), Some(GATE_RESULT_PASSED));
        assert!(state.last_gate_failure.is_none());
    }

    #[tokio::test]
    async fn a_missing_cwd_fails_the_gates_without_running_them() {
        let mut state = create_autonomous_runtime_state(Some(&AgentAutonomousConfig {
            enabled: Some(true),
            gates: Some(AgentAutonomousGateConfig {
                commands: Some(vec!["exit 0".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        }));
        let result = refresh_autonomous_quality_gates(&mut state, AutonomousOperationOptions::default())
            .await
            .unwrap();
        assert_eq!(result.as_deref(), Some(GATE_RESULT_FAILED));
    }

    #[tokio::test]
    async fn disabled_state_never_consults_the_gates() {
        let mut state = create_autonomous_runtime_state(Some(&AgentAutonomousConfig {
            enabled: Some(false),
            gates: Some(AgentAutonomousGateConfig {
                commands: Some(vec!["exit 1".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        }));
        let result = refresh_autonomous_quality_gates(&mut state, AutonomousOperationOptions::default())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn should_not_continue_for_error_or_aborted_stop_reasons() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        for stop_reason in [pi_ai::types::STOP_REASON_ERROR, pi_ai::types::STOP_REASON_ABORTED] {
            let decision = should_autonomously_continue(
                &mut state,
                &AssistantMessage {
                    stop_reason: stop_reason.to_string(),
                    ..Default::default()
                },
                AutonomousOperationOptions::default(),
                0.0,
            )
            .await
            .unwrap();
            assert_eq!(decision.reason, DECISION_NOT_NEEDED);
        }
    }

    #[tokio::test]
    async fn missing_terminal_evidence_requests_a_continuation_and_counts_it() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        let message = AssistantMessage {
            stop_reason: pi_ai::types::STOP_REASON_STOP.to_string(),
            ..Default::default()
        };
        let continuation = next_autonomous_continuation(
            &mut state,
            &message,
            AutonomousOperationOptions::default(),
            1234.0,
        )
        .await
        .unwrap()
        .expect("continuation");
        assert_eq!(state.continuations_used, 1.0);
        assert_eq!(continuation.timestamp, 1234);
        match &continuation.content {
            UserContent::Blocks(blocks) => match &blocks[0] {
                ImageOrTextContent::Text(text) => {
                    assert_eq!(text.text, DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT)
                }
                _ => panic!("expected text"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[tokio::test]
    async fn exhausted_limits_stop_the_continuation() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        state.continuations_used = 3.0;
        let message = AssistantMessage {
            stop_reason: pi_ai::types::STOP_REASON_STOP.to_string(),
            ..Default::default()
        };
        assert!(next_autonomous_continuation(
            &mut state,
            &message,
            AutonomousOperationOptions::default(),
            0.0
        )
        .await
        .unwrap()
        .is_none());
    }

    #[tokio::test]
    async fn an_aborted_signal_throws_before_any_work() {
        let mut state = create_autonomous_runtime_state(Some(&config()));
        let signal = CancellationToken::new();
        signal.cancel();
        let error = next_autonomous_continuation(
            &mut state,
            &AssistantMessage::default(),
            AutonomousOperationOptions {
                cwd: None,
                signal: Some(signal),
            },
            0.0,
        )
        .await
        .unwrap_err();
        assert_eq!(error.message, "This operation was aborted");
    }
}
