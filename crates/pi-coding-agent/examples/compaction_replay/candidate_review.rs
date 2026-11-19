//! R2-candidate-only admission controls. The shared projection module is frozen.
use super::*;
use pi_jev::compaction::CompactionSkip;

const REVIEWED_PRODUCTION_BASE: &str = "ad7a98add92a9a15626e72f79e8eb9950004280b";
const REVIEWED_PROJECTION_SHA256: &str = "3c242bcaca016f0f25a96930a32bbb4d712013eb651df84c81e121cf6d694844";
const PROJECTION_SCHEMA_VERSION: u64 = 1;
const NORMALIZATION_VERSION: &str = "reviewed-local-parity-v1";
const CAPTURE_IMPLEMENTATION_ID: &str = "reviewed-shared-local-projection-v1";

// Auxiliary comparison contract, copied outside the frozen projection.
const PARITY_SECTIONS: &[(&str, &str)] = &[
    ("/case", "fixture-identity"),
    ("/fixture_manifest", "fixture-identity"),
    ("/token_outputs", "token-output-parity"),
    ("/cut_output", "cut-output-parity"),
    ("/prepare_compaction", "prepare-compaction-parity"),
    ("/conversion", "conversion-parity"),
    ("/serialization", "serialization-parity"),
    ("/replicated_chunk_slicing", "replicated-chunk-parity"),
    ("/session_manager/raw_branch_semantic", "branch-parity"),
    ("/session_manager/normalized_restored_messages", "normalized-restored-parity"),
    ("/session_manager/context", "restored-context-parity"),
    ("/session_manager/checkpoint_state", "checkpoint-parity"),
    ("/session_manager/protected_custom_state", "protected-custom-state-parity"),
    ("/jev_prepare_context", "jev-output-parity"),
];

fn candidate_parity_key(pointer: &str) -> String {
    pointer.trim_start_matches('/').replace('/', "__")
}

fn candidate_add_full_projection_hashes(case: &mut Value) -> Result<(), String> {
    let mut hashes = Map::new();
    for (pointer, stage) in PARITY_SECTIONS {
        let value = case.pointer(pointer).ok_or_else(|| {
            stage_error(stage, format!("projection missing {pointer}"))
        })?;
        // Hash the exact in-memory semantic value BEFORE JSON text serialization.
        // serde_json's default float parser need not round-trip every last bit;
        // parsing a baseline into Value must never silently change numeric parity.
        hashes.insert(candidate_parity_key(pointer), json!(semantic_sha256(value)?));
    }
    case["full_projection_sha256"] = json!({
        "algorithm": "sha256 over compact serde_json encoding of the complete in-memory semantic section before report serialization",
        "sections": hashes,
        "note": "Complete values are retained in cases; these are full-content hashes, not length or scalar checksums",
    });
    Ok(())
}


fn file_identity(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| stage_error("capture-source-identity", format!("read {}: {error}", path.display())))?;
    Ok(json!({"path": path, "bytes": bytes.len(), "sha256": sha256_hex(&bytes)}))
}

fn git_text(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| stage_error("production-source-identity", error.to_string()))?;
    ensure_stage(output.status.success(), "production-source-identity", format!("git {:?} failed: {}", args, String::from_utf8_lossy(&output.stderr)))?;
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| stage_error("production-source-identity", error.to_string()))
}

fn production_file_set_hash(crate_root: &Path) -> Result<(String, usize), String> {
    let git_root = crate_root.ancestors().nth(2).map(Path::to_path_buf).ok_or_else(|| stage_error("production-source-identity", "cannot derive repository root"))?;
    let crates_dir = crate_root.parent().ok_or_else(|| stage_error("production-source-identity", "cannot derive crates root"))?;
    let roots = [crate_root.join("Cargo.toml"), crate_root.join("src"), crates_dir.join("pi-jev/Cargo.toml"), crates_dir.join("pi-jev/src")];
    let mut identities = Vec::new();
    for root in roots {
        if root.is_file() {
            let bytes = std::fs::read(&root).map_err(|error| stage_error("production-source-identity", error.to_string()))?;
            identities.push((root.strip_prefix(&git_root).unwrap_or(&root).to_string_lossy().replace('\\', "/"), sha256_hex(&bytes)));
            continue;
        }
        let mut pending = vec![root];
        while let Some(directory) = pending.pop() {
            let mut entries = std::fs::read_dir(&directory)
                .map_err(|error| stage_error("production-source-identity", error.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| stage_error("production-source-identity", error.to_string()))?;
            entries.sort_by_key(|entry| entry.path());
            for entry in entries {
                let path = entry.path();
                let file_type = entry.file_type().map_err(|error| stage_error("production-source-identity", error.to_string()))?;
                if file_type.is_dir() {
                    pending.push(path);
                } else if file_type.is_file() {
                    let bytes = std::fs::read(&path).map_err(|error| stage_error("production-source-identity", error.to_string()))?;
                    identities.push((path.strip_prefix(&git_root).unwrap_or(&path).to_string_lossy().replace('\\', "/"), sha256_hex(&bytes)));
                }
            }
        }
    }
    identities.sort_by(|left, right| left.0.cmp(&right.0));
    let canonical = identities.iter().map(|(path, hash)| format!("{hash}  {path}\n")).collect::<String>();
    Ok((sha256_hex(canonical.as_bytes()), identities.len()))
}

fn capture_provenance(cli: &Cli, arm: &str) -> Result<Value, String> {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let git_root = crate_root.join("../..");
    let main_source = crate_root.join("examples/compaction_replay.rs");
    let projection_source = crate_root.join("examples/compaction_replay/local_parity.rs");
    let candidate_source = crate_root.join("examples/compaction_replay/candidate_review.rs");
    let adapter_diff = PathBuf::from(cli.required("adapter-diff")?);
    let projection_hash = sha256_hex(&std::fs::read(&projection_source).map_err(|error| stage_error("capture-source-identity", error.to_string()))?);
    ensure_stage(projection_hash == REVIEWED_PROJECTION_SHA256, "projection-identity", "candidate projection is not reviewed byte-identical source")?;
    let head = git_text(&git_root, &["rev-parse", "HEAD"])?;
    ensure_stage(head == REVIEWED_PRODUCTION_BASE, "production-base", format!("capture HEAD {head} is not reviewed R1 production base"))?;
    let production_dirty = git_text(&git_root, &["status", "--porcelain", "--", "crates/pi-coding-agent/src", "crates/pi-jev/src"])?;
    ensure_stage(production_dirty.is_empty(), "production-source-dirty", "production source is dirty during capture")?;
    let full_dirty = git_text(&git_root, &["status", "--porcelain"])?;
    let exe = std::env::current_exe().map_err(|error| stage_error("exe-identity", error.to_string()))?;
    let exe_bytes = std::fs::read(&exe).map_err(|error| stage_error("exe-identity", error.to_string()))?;
    let target_dir = exe.ancestors().nth(3).map(Path::to_path_buf).ok_or_else(|| stage_error("exe-identity", "cannot derive target directory"))?;
    let (production_file_set_sha256, production_file_count) = production_file_set_hash(&crate_root)?;
    let utc_ms = now_ms();
    Ok(json!({
        "arm": arm,
        "production_base_commit": REVIEWED_PRODUCTION_BASE,
        "capture_implementation_id": CAPTURE_IMPLEMENTATION_ID,
        "projection_implementation_sha256": projection_hash,
        "capture_source": {
            "example": file_identity(&main_source)?,
            "projection": file_identity(&projection_source)?,
            "candidate_adapter": file_identity(&candidate_source)?,
        },
        "complete_adapter_diff": file_identity(&adapter_diff)?,
        "production_source": {
            "file_set_sha256": production_file_set_sha256, "file_count": production_file_count,
            "git_head": head,
            "git_dirty": !full_dirty.is_empty(),
            "production_git_dirty": !production_dirty.is_empty(),
            "full_git_status": full_dirty,
        },
        "computed_exe": {"path": exe, "bytes": exe_bytes.len(), "sha256": sha256_hex(&exe_bytes), "source": "computed-from-running-exe"},
        "process": {"pid": std::process::id(), "run_id": format!("{}-{}-{}", arm, std::process::id(), utc_ms), "utc_ms": utc_ms, "argv": std::env::args().collect::<Vec<_>>()},
        "build": {"rustc": "1.98.0", "cargo": "1.98.0", "host": "x86_64-pc-windows-msvc", "profile": "release", "locked": true, "offline": true, "target_dir": target_dir},
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "non_timed": true,
        "performance_claim": false,
    }))
}

fn candidate_settings_projection(settings: &pi_coding_agent::core::compaction::compaction::CompactionSettings) -> Value {
    json!({
        "enabled": settings.enabled,
        "reserve_tokens": settings.reserve_tokens,
        "keep_recent_tokens": settings.keep_recent_tokens,
        "summary_update_policy": settings.summary_update_policy,
        "prompt": settings.prompt,
        "summary_budget_mode": settings.summary_budget_mode,
        "trigger_threshold": settings.trigger_threshold,
        "deadline_ms": settings.deadline_ms,
        "task_state": settings.task_state,
    })
}

fn candidate_fixture_settings(fixture: &LoadedFixture) -> pi_coding_agent::core::compaction::compaction::CompactionSettings {
    let mut settings = default_compaction_settings();
    settings.enabled = true;
    settings.reserve_tokens = 16384.0;
    settings.keep_recent_tokens = fixture.manifest.get("keep_recent_tokens_hint").and_then(Value::as_f64).unwrap_or(20000.0);
    settings.summary_update_policy = None;
    settings
}

fn candidate_json_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|error| stage_error("effective-config", error.to_string()))
}

fn invalid_cli(options: &[(&str, &str)]) -> Cli {
    Cli { subcommand: "candidate-control".to_string(), options: options.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect() }
}

fn effective_capture_config(cases: &[LoadedFixture], chunk_context_window: f64) -> Result<Value, String> {
    let mut effective = Vec::new();
    for fixture in cases {
        effective.push(json!({
            "case": fixture.name,
            "manifest": fixture.manifest,
            "compaction_settings": candidate_settings_projection(&candidate_fixture_settings(fixture)),
            "jev_config": {
                "keep_threshold": CompactionConfig::default().keep_threshold,
                "preserve_recent_messages": CompactionConfig::default().preserve_recent_messages,
                "max_state_tokens": CompactionConfig::default().max_state_tokens,
                "max_request_tokens": CompactionConfig::default().max_request_tokens,
                "truncate_head_chars": CompactionConfig::default().truncate_head_chars,
                "minimum_reduction_ratio": CompactionConfig::default().minimum_reduction_ratio,
            },
            "model": candidate_json_value(&fixture_model_for_estimates())?,
            "tool_output_policy": {"policy": null, "scope": null},
        }));
    }
    Ok(json!({
        "chunk_context_window": chunk_context_window,
        "case_order": effective.iter().map(|case| case["case"].clone()).collect::<Vec<_>>(),
        "effective_by_case": effective,
    }))
}

fn baseline_effective_settings(cases: &[LoadedFixture], chunk_context_window: f64) -> Result<Value, String> {
    let model = fixture_model_for_estimates();
    let mut keep_recent = Vec::new();
    let mut common = Vec::new();
    for fixture in cases {
        let settings = candidate_fixture_settings(fixture);
        keep_recent.push(json!({"case": fixture.name, "keep_recent_tokens": settings.keep_recent_tokens}));
        common.push(candidate_settings_projection(&settings));
    }
    let first = common.first().cloned().ok_or_else(|| stage_error("effective-config", "no cases"))?;
    ensure_stage(common.iter().all(|settings| {
        ["enabled", "reserve_tokens", "summary_update_policy", "prompt", "summary_budget_mode", "trigger_threshold", "deadline_ms", "task_state"]
            .iter().all(|key| settings[*key] == first[*key])
    }), "effective-config", "case compaction settings differ outside keep_recent_tokens")?;
    let jev = CompactionConfig::default();
    Ok(json!({
        "chunk_context_window": chunk_context_window,
        "compaction": {
            "enabled": first["enabled"], "reserve_tokens": first["reserve_tokens"], "keep_recent_tokens": keep_recent,
            "summary_update_policy": first["summary_update_policy"], "prompt": first["prompt"],
            "summary_budget_mode": first["summary_budget_mode"], "trigger_threshold": first["trigger_threshold"],
            "deadline_ms": first["deadline_ms"], "task_state": first["task_state"],
            "effective_per_case_projection_pointer": "/cases/*/prepare_compaction/settings",
        },
        "jev": {"keep_threshold": jev.keep_threshold, "preserve_recent_messages": jev.preserve_recent_messages, "max_state_tokens": jev.max_state_tokens, "max_request_tokens": jev.max_request_tokens, "truncate_head_chars": jev.truncate_head_chars, "minimum_reduction_ratio": jev.minimum_reduction_ratio},
        "model": {"provider": model.provider, "model_id": model.id, "api": model.api, "base_url": model.base_url, "context_window": model.context_window, "max_tokens": model.max_tokens, "input": model.input},
        "tool_output_policy": {"policy": null, "scope": null, "source": "ModelToolOutputPolicyOptions::default()"},
        "ambient_replay_provider_or_user_profile": false,
    }))
}

pub(super) fn run_capture(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let arm = cli.required("arm")?;
    ensure_stage(matches!(arm.as_str(), "reviewed-r1-baseline" | "r2-candidate"), "candidate-arm", "--arm must be reviewed-r1-baseline or r2-candidate")?;
    let expectation_path = cli.optional("expectation").map(PathBuf::from);
    if expectation_path.is_some() {
        ensure_stage(arm == "r2-candidate", "candidate-arm", "only an r2-candidate may consume a baseline expectation")?;
    }
    let chunk_context_window = cli.positive_number("chunk-context-window", 32_768.0, 1_000_000_000.0)?;
    let provenance = capture_provenance(cli, &arm)?;
    let case_dirs = discover_fixtures(&fixture_path)?;
    let mut loaded_cases = Vec::new();
    let mut fixture_inputs = Vec::new();
    for case_dir in &case_dirs {
        let transcript_path = case_dir.join("transcript.json");
        let manifest_path = case_dir.join("manifest.json");
        let before = json!({"transcript": file_identity(&transcript_path)?, "manifest": file_identity(&manifest_path)?});
        let fixture = load_fixture(case_dir)?;
        let after = json!({"transcript": file_identity(&transcript_path)?, "manifest": file_identity(&manifest_path)?});
        ensure_stage(before == after, "fixture-input-mutation", "fixture bytes changed during local capture")?;
        fixture_inputs.push(json!({"case": fixture.name, "before": before, "after": after}));
        loaded_cases.push(fixture);
    }
    ensure_stage(!loaded_cases.is_empty(), "fixture-identity", "no fixture cases were captured")?;
    let config = effective_capture_config(&loaded_cases, chunk_context_window)?;
    let effective_settings = baseline_effective_settings(&loaded_cases, chunk_context_window)?;
    let mut projections = Vec::new();
    for fixture in &loaded_cases {
        let mut projection = local_parity::case_projection(fixture, chunk_context_window)?;
        candidate_add_full_projection_hashes(&mut projection)?;
        projections.push(projection);
    }
    if let Some(ref expectation_path) = expectation_path {
        let expectation: Value = serde_json::from_slice(&std::fs::read(expectation_path).map_err(|error| stage_error("expectation-read", error.to_string()))?)
            .map_err(|error| stage_error("expectation-schema", error.to_string()))?;
        let actual = json!({"provenance": provenance, "config": config, "effective_settings": effective_settings, "fixture_inputs": fixture_inputs, "cases": projections});
        compare_baseline_envelope(&expectation, &actual)?;
    }
    let output = json!({
        "schema": 2, "harness": "compaction_replay", "subcommand": "capture-local", "status": "ok",
        "comparison": if expectation_path.is_some() { "independent-reviewed-r1-baseline-exact-full-output" } else { "capture-only-not-independent-admission" },
        "provenance": provenance, "config": config, "effective_settings": effective_settings, "fixture_inputs": fixture_inputs,
        "normalization_version": NORMALIZATION_VERSION, "non_timed": true, "performance_claim": false,
        "cases": projections,
    });
    write_new_json(&out, &output)?;
    println!("wrote {} ({} full-output local cases, arm {})", out.display(), loaded_cases.len(), arm);
    Ok(0)
}

fn same_json(expected: &Value, actual: &Value, pointer: &str, stage: &str) -> Result<(), String> {
    let left = expected.pointer(pointer).ok_or_else(|| stage_error("expectation-schema", format!("missing {pointer}")))?;
    let right = actual.pointer(pointer).ok_or_else(|| stage_error(stage, format!("candidate missing {pointer}")))?;
    ensure_stage(left == right, stage, format!("identity mismatch at {pointer}"))
}

fn compare_jev_outcome(expected_case: &Value, actual_case: &Value) -> Result<(), String> {
    same_json(expected_case, actual_case, "/jev_prepare_context/outcome", "jev-outcome-parity")
}

fn same_mapped(expected: &Value, actual: &Value, expected_pointer: &str, actual_pointer: &str, stage: &str) -> Result<(), String> {
    let left = expected.pointer(expected_pointer).ok_or_else(|| stage_error("expectation-schema", format!("baseline missing {expected_pointer}")))?;
    let right = actual.pointer(actual_pointer).ok_or_else(|| stage_error(stage, format!("candidate missing {actual_pointer}")))?;
    ensure_stage(left == right, stage, format!("identity mismatch: baseline {expected_pointer}, candidate {actual_pointer}"))
}

fn normalized_fixture_cases(value: &Value, baseline: bool) -> Result<Vec<Value>, String> {
    if baseline {
        let before = value.pointer("/run_identity/fixtures_before").and_then(Value::as_object).ok_or_else(|| stage_error("fixture-identity", "baseline fixtures_before missing"))?;
        let after = value.pointer("/run_identity/fixtures_after").and_then(Value::as_object).ok_or_else(|| stage_error("fixture-identity", "baseline fixtures_after missing"))?;
        ensure_stage(before == after, "fixture-identity", "baseline records fixture mutation")?;
        let cases = before.get("cases").and_then(Value::as_array).ok_or_else(|| stage_error("fixture-identity", "baseline fixture cases missing"))?;
        cases.iter().map(|case| {
            Ok(json!({
                "case": case["case"],
                "transcript": {"bytes": case["transcript"]["bytes"], "sha256": case["transcript"]["sha256"]},
                "manifest": {"bytes": case["manifest"]["bytes"], "sha256": case["manifest"]["sha256"]},
            }))
        }).collect()
    } else {
        let inputs = value.get("fixture_inputs").and_then(Value::as_array).ok_or_else(|| stage_error("fixture-identity", "candidate fixture inputs missing"))?;
        inputs.iter().map(|input| {
            ensure_stage(input["before"] == input["after"], "fixture-identity", "candidate records fixture mutation")?;
            Ok(json!({
                "case": input["case"],
                "transcript": {"bytes": input["before"]["transcript"]["bytes"], "sha256": input["before"]["transcript"]["sha256"]},
                "manifest": {"bytes": input["before"]["manifest"]["bytes"], "sha256": input["before"]["manifest"]["sha256"]},
            }))
        }).collect()
    }
}

fn compare_baseline_envelope(expected: &Value, actual: &Value) -> Result<(), String> {
    ensure_stage(expected.pointer("/run_identity/arm").and_then(Value::as_str) == Some("reviewed-r1-baseline"), "baseline-arm", "expectation is not a reviewed R1 baseline arm")?;
    ensure_stage(actual.pointer("/provenance/arm").and_then(Value::as_str) == Some("r2-candidate"), "candidate-arm", "candidate arm must be r2-candidate")?;
    for (expected_pointer, actual_pointer, stage) in [
        ("/run_identity/production_base_commit", "/provenance/production_base_commit", "production-base"),
        ("/run_identity/projection_implementation_sha256", "/provenance/projection_implementation_sha256", "projection-identity"),
        ("/run_identity/projection_schema", "/provenance/projection_schema_version", "projection-schema"),
        ("/run_identity/normalization_version", "/provenance/normalization_version", "projection-schema"),
        ("/run_identity/production_source_file_set/sha256", "/provenance/production_source/file_set_sha256", "production-source-parity"),
        ("/run_identity/production_source_file_set/file_count", "/provenance/production_source/file_count", "production-source-parity"),
        ("/run_identity/effective_settings", "/effective_settings", "config-parity"),
        ("/run_identity/case_names_in_order", "/config/case_order", "case-identity-parity"),
        ("/config/chunk_context_window", "/config/chunk_context_window", "config-parity"),
    ] {
        same_mapped(expected, actual, expected_pointer, actual_pointer, stage)?;
    }
    ensure_stage(actual.pointer("/provenance/production_base_commit").and_then(Value::as_str) == Some(REVIEWED_PRODUCTION_BASE), "production-base", "candidate production base is not reviewed R1")?;
    ensure_stage(actual.pointer("/provenance/projection_implementation_sha256").and_then(Value::as_str) == Some(REVIEWED_PROJECTION_SHA256), "projection-identity", "candidate shared projection hash is not reviewed")?;
    let expected_exe = expected.pointer("/run_identity/computed_executable/sha256").and_then(Value::as_str).ok_or_else(|| stage_error("expectation-schema", "baseline executable identity missing"))?;
    let actual_exe = actual.pointer("/provenance/computed_exe/sha256").and_then(Value::as_str).ok_or_else(|| stage_error("exe-identity", "candidate executable identity missing"))?;
    ensure_stage(expected_exe != actual_exe, "arm-identity", "baseline and candidate executable hashes must differ")?;
    let expected_pid = expected.pointer("/run_identity/process/pid").and_then(Value::as_u64).ok_or_else(|| stage_error("expectation-schema", "baseline capture PID missing"))?;
    let actual_pid = actual.pointer("/provenance/process/pid").and_then(Value::as_u64).ok_or_else(|| stage_error("exe-identity", "candidate capture PID missing"))?;
    ensure_stage(expected_pid != actual_pid, "arm-identity", "baseline and candidate capture PIDs must differ")?;
    let expected_fixtures = normalized_fixture_cases(expected, true)?;
    let actual_fixtures = normalized_fixture_cases(actual, false)?;
    ensure_stage(expected_fixtures == actual_fixtures, "fixture-identity", "fixture names, byte lengths, or hashes differ")?;
    let expected_cases = expected.get("cases").and_then(Value::as_array).ok_or_else(|| stage_error("expectation-schema", "baseline cases is not an array"))?;
    let actual_cases = actual.get("cases").and_then(Value::as_array).ok_or_else(|| stage_error("case-count-parity", "candidate cases is not an array"))?;
    ensure_stage(expected_cases.len() == actual_cases.len(), "case-count-parity", "baseline and candidate case counts differ")?;
    for (index, actual_case) in actual_cases.iter().enumerate() {
        let expected_case = expected_cases.get(index).ok_or_else(|| stage_error("expectation-schema", "case index missing"))?;
        compare_jev_outcome(expected_case, actual_case)?;
        for &(pointer, stage) in PARITY_SECTIONS {
            let actual_value = actual_case.pointer(pointer).ok_or_else(|| stage_error(stage, format!("candidate projection missing {pointer}")))?;
            let actual_hash = semantic_sha256(actual_value)?;
            let key = pointer.trim_start_matches('/').replace('/', "__");
            let expected_hash = expected_case.pointer(&format!("/full_projection_sha256/sections/{key}")).and_then(Value::as_str)
                .ok_or_else(|| stage_error("expectation-schema", format!("missing full-content hash for {pointer}")))?;
            ensure_stage(expected_hash == actual_hash, stage, format!("full-output digest mismatch at {pointer}"))?;
        }
    }
    Ok(())
}

fn control_error(checks: &mut Vec<Value>, name: &str, stage: &str, result: Result<(), String>) -> Result<(), String> {
    let error = result.err().ok_or_else(|| stage_error(stage, format!("{name} was unexpectedly accepted")))?;
    ensure_stage(error.starts_with(&format!("[{stage}]")), stage, format!("{name} failed at the wrong stage: {error}"))?;
    checks.push(json!({"name": name, "rejected": true, "stage": stage, "error": error}));
    Ok(())
}

pub(super) fn run_candidate_controls(cli: &Cli) -> Result<i32, String> {
    let out = PathBuf::from(cli.required("out")?);
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    if out.exists() { return Err(stage_error("out-exists", format!("{} already exists", out.display()))); }
    let mut checks = Vec::new();
    for (name, key, stage) in [
        ("iters_present_empty_rejected", "iters", "cli-iters"),
        ("warmup_present_empty_rejected", "warmup", "cli-warmup"),
        ("chunk_window_present_empty_rejected", "chunk-context-window", "cli-chunk-context-window"),
        ("timeout_present_empty_rejected", "timeout-secs", "cli-timeout-secs"),
        ("provider_delay_present_empty_rejected", "provider-delay-ms", "cli-provider-delay-ms"),
        ("context_window_present_empty_rejected", "context-window", "cli-context-window"),
        ("keep_recent_present_empty_rejected", "keep-recent", "cli-keep-recent"),
        ("summary_calls_present_empty_rejected", "expect-summary-calls", "cli-expect-summary-calls"),
    ] {
        let control = invalid_cli(&[(key, "")]);
        let result = match key {
            "iters" | "warmup" | "provider-delay-ms" | "timeout-secs" | "expect-summary-calls" => control.integer(key, 3, 1, 100_000).map(|_| ()),
            _ => control.positive_number(key, 3.0, 1_000_000_000.0).map(|_| ()),
        };
        control_error(&mut checks, name, stage, result)?;
    }

    let mut reduction_messages = vec![AgentMessage::Message(Message::User(UserMessage::new(UserContent::Text("retain the leading request".to_string()), FIXTURE_TIMESTAMP_BASE)))];
    reduction_messages.extend(tool_pairing_fixture(&["reduce"], &["reduce"]));
    for message in &mut reduction_messages {
        if let AgentMessage::Message(Message::Assistant(assistant)) = message {
            for block in &mut assistant.content {
                if let ContentBlock::ToolCall(call) = block {
                    call.arguments = json!({"path": "notes.txt"}).as_object().cloned().unwrap_or_default();
                }
            }
        }
    }
    for message in &mut reduction_messages {
        if let AgentMessage::Message(Message::ToolResult(result)) = message {
            result.content = vec![ImageOrTextContent::Text(TextContent::new("R".repeat(5000)))];
        }
    }
    let mut reduction_config = CompactionConfig::default();
    reduction_config.preserve_recent_messages = 0;
    reduction_config.minimum_reduction_ratio = 1.0;
    let skip = prepare_context(&reduction_messages, &reduction_config).err().ok_or_else(|| stage_error("selftest-control", "eligible reduction fixture unexpectedly prepared"))?;
    ensure_stage(skip == CompactionSkip::InsufficientReduction, "selftest-control", format!("reduction fixture reached {}, not InsufficientReduction", skip.as_str()))?;
    let null_lane = jev_expected_skip_value(reduction_messages.len(), skip);
    ensure_stage(null_lane["stats"].is_null() && null_lane["checksum"].is_null() && null_lane["skip_class"] == json!(CompactionSkip::InsufficientReduction.as_str()), "selftest-control", "InsufficientReduction null-timing lane is invalid")?;
    checks.push(json!({"name": "jev_insufficient_reduction_null_timing_skip_accepted", "accepted": true, "skip_class": skip.as_str(), "bench_lane": null_lane}));

    let fixture = load_fixture(&fixture_path)?;
    let baseline_case = local_parity::case_projection(&fixture, 32_768.0)?;
    let mut mismatch_case = baseline_case.clone();
    mismatch_case["jev_prepare_context"]["outcome"] = if baseline_case["jev_prepare_context"]["outcome"] == json!("prepared") { json!("expected-skip") } else { json!("prepared") };
    control_error(&mut checks, "jev_outcome_mismatch_rejected", "jev-outcome-parity", compare_jev_outcome(&baseline_case, &mismatch_case))?;

    let report = json!({"schema": 1, "self_test": "candidate-review-controls", "status": "ok", "check_count": checks.len(), "checks": checks, "performance_claim": false});
    write_new_json(&out, &report)?;
    println!("wrote {} ({} candidate review checks)", out.display(), checks.len());
    Ok(0)
}
