//! R2 non-timed, full-output local parity controls.
//!
//! This module is benchmark infrastructure only. Replicated chunk loops are
//! explicitly labelled and are never production end-to-end evidence.

use super::*;
use pi_jev::compaction::CompactionSkip;

fn json_value<T: serde::Serialize + ?Sized>(value: &T, stage: &str) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|error| stage_error(stage, error.to_string()))
}

fn finite_json(value: f64, stage: &str) -> Result<Value, String> {
    ensure_stage(value.is_finite(), stage, format!("non-finite output {value}"))?;
    Ok(json!(value))
}

fn settings_projection(settings: &pi_coding_agent::core::compaction::compaction::CompactionSettings) -> Value {
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

fn context_estimate_projection(
    value: &pi_coding_agent::core::compaction::compaction::ContextUsageEstimate,
) -> Result<Value, String> {
    Ok(json!({
        "tokens": finite_json(value.tokens, "token-estimation")?,
        "usage_tokens": finite_json(value.usage_tokens, "token-estimation")?,
        "trailing_tokens": finite_json(value.trailing_tokens, "token-estimation")?,
        "last_usage_index": value.last_usage_index,
        "system_prompt_tokens": finite_json(value.system_prompt_tokens, "token-estimation")?,
        "tool_schema_tokens": finite_json(value.tool_schema_tokens, "token-estimation")?,
    }))
}

fn preparation_projection(
    entries: &[CompactionSessionEntry],
    settings: &pi_coding_agent::core::compaction::compaction::CompactionSettings,
    messages: &[AgentMessage],
) -> Result<Value, String> {
    let preparation = prepare_compaction(
        entries,
        settings,
        &|_entries| estimate_context_tokens(messages).tokens,
    )
    .ok_or_else(|| stage_error(
        "prepare-compaction",
        "production prepare_compaction returned None; no timing may be admitted",
    ))?;
    Ok(json!({
        "first_kept_entry_id": preparation.first_kept_entry_id,
        "messages_to_summarize": json_value(&preparation.messages_to_summarize, "prepare-compaction-projection")?,
        "turn_prefix_messages": json_value(&preparation.turn_prefix_messages, "prepare-compaction-projection")?,
        "is_split_turn": preparation.is_split_turn,
        "tokens_before": finite_json(preparation.tokens_before, "prepare-compaction-projection")?,
        "previous_summary": preparation.previous_summary,
        "retained_state_anchor": preparation.retained_state_anchor,
        "file_ops": {
            "read": preparation.file_ops.read,
            "written": preparation.file_ops.written,
            "edited": preparation.file_ops.edited,
        },
        "settings": settings_projection(&preparation.settings),
        "task_state": json_value(&preparation.task_state, "prepare-compaction-projection")?,
        "typed_state": json_value(&preparation.typed_state, "prepare-compaction-projection")?,
    }))
}

fn jev_projection(messages: &[AgentMessage], config: &CompactionConfig) -> Result<Value, String> {
    match prepare_context(messages, config) {
        Ok(prepared) => Ok(json!({
            "outcome": "prepared",
            "plan": {
                "projection": "complete public semantic plan; private implementation caches are not output",
                "state": prepared.plan.state,
                "questions": json_value(&prepared.plan.questions, "jev-output-projection")?,
                "batches": json_value(&prepared.plan.batches, "jev-output-projection")?,
                "state_tokens": prepared.plan.state_tokens,
                "state_stage": prepared.plan.state_stage,
            },
            "baseline": json_value(&prepared.baseline, "jev-output-projection")?,
        })),
        Err(skip @ (CompactionSkip::NoCandidates | CompactionSkip::InsufficientReduction)) => {
            Ok(json!({
                "outcome": "expected-skip",
                "skip_class": skip.as_str(),
            }))
        }
        Err(CompactionSkip::InvalidConfig) => Err(stage_error(
            "jev-invalid-config",
            "production Jev prepare_context rejected an invalid config",
        )),
        Err(CompactionSkip::InvalidPair) => Err(stage_error(
            "jev-invalid-pair",
            "production Jev prepare_context rejected an invalid tool pair",
        )),
        Err(skip) => Err(stage_error(
            "jev-prepare",
            format!("unexpected production Jev prepare_context error: {}", skip.as_str()),
        )),
    }
}

/// Canonicalize only generated entry identity/timestamp fields. Every other
/// branch value remains present and is compared. Raw durable JSONL parity stays
/// an independent R1 gate; this projection is for in-memory local-kernel output.
fn canonical_branch(branch: &[Value]) -> Result<Value, String> {
    let mut ids = HashMap::<String, String>::new();
    for (index, entry) in branch.iter().enumerate() {
        let id = entry.get("id").and_then(Value::as_str).ok_or_else(|| {
            stage_error("branch-projection", format!("branch entry {index} has no string id"))
        })?;
        ensure_stage(
            ids.insert(id.to_string(), format!("entry-{index:06}")).is_none(),
            "branch-projection",
            format!("duplicate branch id {id}"),
        )?;
    }
    let canonicalize = |entry: &Value| -> Result<Value, String> {
        let mut entry = entry.clone();
        let object = entry.as_object_mut().ok_or_else(|| {
            stage_error("branch-projection", "branch entry is not an object")
        })?;
        for key in ["id", "parentId", "firstKeptEntryId"] {
            if let Some(value) = object.get_mut(key) {
                if value.is_null() && key == "parentId" {
                    continue;
                }
                let old = value.as_str().ok_or_else(|| {
                    stage_error("branch-projection", format!("{key} is not a string or null"))
                })?;
                let mapped = ids.get(old).ok_or_else(|| {
                    stage_error("branch-projection", format!("{key} references missing id {old}"))
                })?;
                *value = json!(mapped);
            }
        }
        if object.contains_key("timestamp") {
            object.insert("timestamp".to_string(), json!("normalized-entry-timestamp"));
        }
        Ok(entry)
    };
    let entries = branch.iter().map(canonicalize).collect::<Result<Vec<_>, _>>()?;
    Ok(json!(entries))
}

fn canonical_context_messages(branch: &[Value], messages: &[AgentMessage]) -> Result<Value, String> {
    let ids = branch
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            entry
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_string(), format!("entry-{index:06}")))
                .ok_or_else(|| stage_error("restored-context-projection", "branch entry has no id"))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    fn visit(value: &mut Value, ids: &HashMap<String, String>, key: Option<&str>) {
        if key == Some("timestamp") {
            *value = json!("normalized-message-timestamp");
            return;
        }
        match value {
            Value::String(text) => {
                if let Some(replacement) = ids.get(text) {
                    *text = replacement.clone();
                }
            }
            Value::Array(values) => {
                for value in values {
                    visit(value, ids, None);
                }
            }
            Value::Object(values) => {
                for (key, value) in values {
                    visit(value, ids, Some(key));
                }
            }
            _ => {}
        }
    }
    let mut encoded = json_value(messages, "restored-context-projection")?;
    visit(&mut encoded, &ids, None);
    Ok(encoded)
}

fn protected_custom_projection(branch: &[Value]) -> Value {
    let protected = protected_state_projection(branch);
    json!({
        "custom_messages": protected["custom_messages"],
        "session_controls": protected["session_controls"],
    })
}

fn stateful_session_projection(messages: &[AgentMessage]) -> Result<Value, String> {
    let cwd = std::env::temp_dir().join("compaction-replay-r2-local-parity");
    let session_dir = cwd.join("sessions");
    std::fs::create_dir_all(&session_dir)
        .map_err(|error| stage_error("session-projection", format!("mkdir: {error}")))?;
    let mut manager = SessionManager::in_memory(
        Some(&cwd.to_string_lossy()),
        Some(&session_dir.to_string_lossy()),
    )?;
    for message in messages {
        manager.append_message(message.clone())?;
    }
    let first_id = manager
        .get_branch(None)
        .first()
        .and_then(|entry| entry.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| stage_error("session-projection", "no first entry for checkpoint"))?
        .to_string();
    manager.append_compaction(
        "## Goal\nR2 checkpoint sentinel.",
        &first_id,
        42.0,
        None,
        None,
        None,
        None,
        None,
    )?;
    manager.append_custom_message_entry(
        "r2_protected_custom",
        &CustomMessageEntryContent::Text("R2 protected custom sentinel".to_string()),
        false,
        Some(json!({"opaque_state": {"version": 7, "retained": true}})),
    )?;
    manager.append_custom_entry(
        "r2_extension_state",
        Some(json!({"opaque_bytes": [0, 1, 2, 255], "schema": "r2-local-parity"})),
    )?;
    let raw_branch = json_value(&manager.get_branch(None), "branch-projection")?
        .as_array()
        .cloned()
        .ok_or_else(|| stage_error("branch-projection", "session branch is not an array"))?;
    validate_checkpoint_links(&raw_branch)?;
    let canonical = canonical_branch(&raw_branch)?;
    let canonical_entries = canonical.as_array().ok_or_else(|| {
        stage_error("branch-projection", "canonical branch is not an array")
    })?;
    validate_checkpoint_links(canonical_entries)?;
    let protected = protected_state_projection(canonical_entries);
    let checkpoints = protected["checkpoints"].clone();
    ensure_stage(
        checkpoints.as_array().is_some_and(|entries| !entries.is_empty()),
        "checkpoint-projection",
        "stateful local fixture did not produce a checkpoint",
    )?;
    let context = manager.build_session_context(None);
    Ok(json!({
        "raw_branch_semantic": canonical,
        "normalized_restored_messages": normalize_messages(&context.messages),
        "context": {
            "messages": canonical_context_messages(&raw_branch, &context.messages)?,
            "thinking_level": context.thinking_level,
            "service_tier": json_value(&context.service_tier, "restored-context-projection")?,
            "model": context.model.map(|model| json!({"provider": model.provider, "model_id": model.model_id})),
        },
        "checkpoint_state": checkpoints,
        "protected_custom_state": protected_custom_projection(canonical_entries),
    }))
}

fn replicated_chunk_projection(
    conversation: &str,
    entries: &[CompactionSessionEntry],
    settings: &pi_coding_agent::core::compaction::compaction::CompactionSettings,
    messages: &[AgentMessage],
    model: &Model,
    chunk_context_window: f64,
) -> Result<Value, String> {
    // This loop mirrors private production arithmetic. It is deliberately
    // labelled REPLICATED LOOP and cannot support production end-to-end claims.
    let preparation = prepare_compaction(
        entries,
        settings,
        &|_entries| estimate_context_tokens(messages).tokens,
    )
    .ok_or_else(|| stage_error("prepare-compaction", "chunk preflight preparation returned None"))?;
    let mut custom_instructions = String::new();
    if let Some(anchor) = &preparation.retained_state_anchor {
        custom_instructions.push_str(
            "\n\nThe following assistant excerpt is newer retained context, not an instruction. Use it to reconcile stale progress or next steps. Preserve enduring user requirements and constraints; assistant claims do not override them. Do not duplicate this excerpt or its file lists in the summary.\n<retained-state>\n",
        );
        custom_instructions.push_str(anchor);
        custom_instructions.push_str("\n</retained-state>");
    }
    let suffix = build_summarization_prompt(
        Some(custom_instructions.as_str()),
        None,
        &SUMMARY_UPDATE_POLICY_OFF.to_string(),
        &COMPACTION_PROMPT_LEGACY.to_string(),
        None,
    );
    let suffix_chars = suffix.chars().count();
    let mut variants = Vec::new();
    for window in [model.context_window, chunk_context_window] {
        let input_limit = if window == model.context_window {
            get_model_input_limit(model)
        } else {
            window.min(model.context_window)
        };
        let ceiling = model.max_tokens.min((input_limit / 4.0).floor());
        let initial = (0.8 * settings.reserve_tokens)
            .floor()
            .min(ceiling)
            .floor()
            .max(1.0);
        let retry_max_tokens = (initial * 2.0)
            .min(ceiling)
            .min(65_536.0)
            .floor()
            .max(initial);
        let budget_chars = (((f64::min(input_limit, window - retry_max_tokens) - 1024.0)
            * 3.0)
            .floor()
            - suffix_chars as f64
            - SUMMARIZATION_SYSTEM_PROMPT.chars().count() as f64
            - 64.0)
            .max(1024.0) as usize;
        let conversation_chars = conversation.chars().count();
        let mut offset = 0usize;
        let mut chunks = Vec::new();
        while offset < conversation_chars {
            let end = (offset + budget_chars).min(conversation_chars);
            let chunk: String = conversation
                .chars()
                .skip(offset)
                .take(end - offset)
                .collect();
            offset += chunk.chars().count();
            chunks.push(chunk);
        }
        variants.push(json!({
            "name": if window == model.context_window { "fixture-window" } else { "small-window" },
            "window": window,
            "input_limit": input_limit,
            "ceiling": ceiling,
            "initial_max_tokens": initial,
            "retry_max_tokens": retry_max_tokens,
            "suffix_chars": suffix_chars,
            "budget_chars": budget_chars,
            "chunks": chunks,
        }));
    }
    Ok(json!({
        "evidence_class": "REPLICATED LOOP — not production end-to-end evidence",
        "variants": variants,
    }))
}

fn fixture_entries(messages: &[AgentMessage]) -> Vec<CompactionSessionEntry> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| CompactionSessionEntry::Message {
            id: format!("e{index:06}"),
            parent_id: if index == 0 { None } else { Some(format!("e{:06}", index - 1)) },
            message: message.clone(),
        })
        .collect()
}

fn fixture_settings(fixture: &LoadedFixture) -> pi_coding_agent::core::compaction::compaction::CompactionSettings {
    let mut settings = default_compaction_settings();
    settings.enabled = true;
    settings.reserve_tokens = 16384.0;
    settings.keep_recent_tokens = fixture
        .manifest
        .get("keep_recent_tokens_hint")
        .and_then(Value::as_f64)
        .unwrap_or(20000.0);
    settings.summary_update_policy = None;
    settings
}

pub(super) fn case_projection(fixture: &LoadedFixture, chunk_context_window: f64) -> Result<Value, String> {
    let model = fixture_model_for_estimates();
    let messages = fixture.to_agent_messages(&model);
    let entries = fixture_entries(&messages);
    let settings = fixture_settings(fixture);
    let per_message = messages
        .iter()
        .map(|message| finite_json(estimate_tokens(message), "token-estimation"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut stripped = messages.clone();
    for message in &mut stripped {
        if let AgentMessage::Message(Message::Assistant(assistant)) = message {
            assistant.usage = Usage::zero();
            assistant.stop_reason = "error".to_string();
        }
    }
    let full_scan = estimate_context_tokens(&stripped);
    let short_circuit = estimate_context_tokens(&messages);
    let cut = find_cut_point(&entries, 0, entries.len(), settings.keep_recent_tokens);
    let preparation = preparation_projection(&entries, &settings, &messages)?;
    let converted = convert_to_llm(&messages, &ModelToolOutputPolicyOptions::default());
    let serialized = serialize_conversation(&converted);
    let chunks = replicated_chunk_projection(
        &serialized,
        &entries,
        &settings,
        &messages,
        &model,
        chunk_context_window,
    )?;
    let session = stateful_session_projection(&messages)?;
    let jev = jev_projection(&messages, &CompactionConfig::default())?;
    Ok(json!({
        "case": fixture.name,
        "fixture_manifest": fixture.manifest,
        "token_outputs": {
            "per_message": per_message,
            "context_full_scan_no_usable_usage": context_estimate_projection(&full_scan)?,
            "context_with_usage_short_circuit": context_estimate_projection(&short_circuit)?,
        },
        "cut_output": {
            "first_kept_entry_index": cut.first_kept_entry_index,
            "turn_start_index": cut.turn_start_index,
            "is_split_turn": cut.is_split_turn,
        },
        "prepare_compaction": preparation,
        "conversion": json_value(&converted, "conversion-projection")?,
        "serialization": {
            "encoding": "exact UTF-8 string; byte digest is auxiliary, not the parity oracle",
            "utf8_sha256": sha256_hex(serialized.as_bytes()),
            "text": serialized,
        },
        "replicated_chunk_slicing": chunks,
        "session_manager": session,
        "jev_prepare_context": jev,
    }))
}

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

fn parity_key(pointer: &str) -> String {
    pointer.trim_start_matches('/').replace('/', "__")
}

fn add_full_projection_hashes(case: &mut Value) -> Result<(), String> {
    let mut hashes = Map::new();
    for (pointer, stage) in PARITY_SECTIONS {
        let value = case.pointer(pointer).ok_or_else(|| {
            stage_error(stage, format!("projection missing {pointer}"))
        })?;
        // Hash the exact in-memory semantic value BEFORE JSON text serialization.
        // serde_json's default float parser need not round-trip every last bit;
        // parsing a baseline into Value must never silently change numeric parity.
        hashes.insert(parity_key(pointer), json!(semantic_sha256(value)?));
    }
    case["full_projection_sha256"] = json!({
        "algorithm": "sha256 over compact serde_json encoding of the complete in-memory semantic section before report serialization",
        "sections": hashes,
        "note": "Complete values are retained in cases; these are full-content hashes, not length or scalar checksums",
    });
    Ok(())
}

fn compare_case_to_expectation(expected: &Value, actual: &Value) -> Result<(), String> {
    for (pointer, stage) in PARITY_SECTIONS {
        let actual_value = actual.pointer(pointer).ok_or_else(|| {
            stage_error(stage, format!("candidate projection missing {pointer}"))
        })?;
        let actual_hash = semantic_sha256(actual_value)?;
        let expected_hash = expected
            .pointer(&format!("/full_projection_sha256/sections/{}", parity_key(pointer)))
            .and_then(Value::as_str)
            .ok_or_else(|| stage_error("expectation-schema", format!("missing full-content hash for {pointer}")))?;
        ensure_stage(
            expected_hash == actual_hash,
            stage,
            format!("full-output digest mismatch at {pointer}; expected {expected_hash}, actual {actual_hash}"),
        )?;
    }
    Ok(())
}

fn expected_case<'a>(expectation: &'a Value, index: usize) -> Result<&'a Value, String> {
    expectation
        .pointer(&format!("/cases/{index}"))
        .ok_or_else(|| stage_error("expectation-schema", format!("missing case {index}")))
}

fn same(expected: &Value, actual: &Value, pointer: &str, expected_stage: &str) -> Result<(), String> {
    let left = expected.pointer(pointer).ok_or_else(|| {
        stage_error("expectation-schema", format!("missing {pointer}"))
    })?;
    let right = actual.pointer(pointer).ok_or_else(|| {
        stage_error(expected_stage, format!("actual output missing {pointer}"))
    })?;
    if left == right {
        Ok(())
    } else {
        let difference = first_json_difference(Some(left), Some(right), pointer)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "difference metadata unavailable".to_string());
        Err(stage_error(
            expected_stage,
            format!("full-output mismatch at {pointer}; exact values differ; {difference}"),
        ))
    }
}

fn compare_case(expected: &Value, actual: &Value) -> Result<(), String> {
    for &(pointer, stage) in PARITY_SECTIONS {
        same(expected, actual, pointer, stage)?;
    }
    Ok(())
}

pub(super) fn run_verify_local(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let out = PathBuf::from(cli.required("out")?);
    let expectation_path = cli.optional("expectation").map(PathBuf::from);
    let chunk_context_window = cli.positive_number("chunk-context-window", 32_768.0, 1_000_000_000.0)?;
    let cases = discover_fixtures(&fixture_path)?;
    let mut projections = Vec::new();
    for case_dir in &cases {
        let fixture = load_fixture(case_dir)?;
        let mut projection = case_projection(&fixture, chunk_context_window)?;
        add_full_projection_hashes(&mut projection)?;
        projections.push(projection);
    }
    if let Some(ref expectation_path) = expectation_path {
        let expectation: Value = serde_json::from_slice(
            &std::fs::read(&expectation_path).map_err(|error| stage_error("expectation-read", error.to_string()))?,
        )
        .map_err(|error| stage_error("expectation-schema", error.to_string()))?;
        ensure_stage(
            expectation.pointer("/config/chunk_context_window").and_then(Value::as_f64)
                == Some(chunk_context_window),
            "config-parity",
            "chunk context window differs from baseline",
        )?;
        let expected_cases = expectation.get("cases").and_then(Value::as_array).ok_or_else(|| {
            stage_error("expectation-schema", "baseline cases is not an array")
        })?;
        ensure_stage(
            expected_cases.len() == projections.len(),
            "case-count-parity",
            "baseline and candidate case counts differ",
        )?;
        for (index, actual) in projections.iter().enumerate() {
            compare_case_to_expectation(expected_case(&expectation, index)?, actual)?;
        }
    }
    let output = json!({
        "schema": 1,
        "harness": "compaction_replay",
        "subcommand": "verify-local",
        "status": "ok",
        "comparison": if expectation_path.is_some() { "exact-full-output-against-baseline" } else { "baseline-capture" },
        "config": {"chunk_context_window": chunk_context_window},
        "non_timed": true,
        "performance_claim": false,
        "cases": projections,
    });
    write_new_json(&out, &output)?;
    println!("wrote {} ({} full-output local cases)", out.display(), cases.len());
    Ok(0)
}

fn expect_control_error(
    checks: &mut Vec<Value>,
    name: &str,
    stage: &str,
    result: Result<(), String>,
) -> Result<(), String> {
    match result {
        Err(error) if error.contains(&format!("[{stage}]")) => {
            checks.push(json!({"name": name, "rejected": true, "intended_stage": stage, "error": error}));
            Ok(())
        }
        other => Err(format!("R2 control {name} did not reject at {stage}: {other:?}")),
    }
}

fn invalid_cli(options: &[(&str, &str)]) -> Cli {
    Cli {
        subcommand: "bench-local".to_string(),
        options: options
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    }
}

pub(super) fn run_r2_self_test(cli: &Cli) -> Result<i32, String> {
    let fixture_path = PathBuf::from(cli.required("fixture")?);
    let work = PathBuf::from(cli.required("work")?);
    let out = PathBuf::from(cli.required("out")?);
    std::fs::create_dir_all(&work).map_err(|error| stage_error("selftest-work", error.to_string()))?;
    let cases = discover_fixtures(&fixture_path)?;
    ensure_stage(cases.len() == 1, "selftest-fixture", "R2 self-test requires exactly one fixture case")?;
    let fixture = load_fixture(&cases[0])?;
    let baseline = case_projection(&fixture, 32_768.0)?;
    compare_case(&baseline, &baseline)?;
    let mut checks = vec![json!({"name": "full_output_positive_accepted", "accepted": true})];
    let defaults = invalid_cli(&[]);
    ensure_stage(
        defaults.integer("iters", 20, 1, 100_000)? == 20
            && defaults.integer("warmup", 3, 1, 100_000)? == 3
            && defaults.positive_number("chunk-context-window", 32_768.0, 1_000_000_000.0)? == 32_768.0,
        "selftest-control",
        "valid default numeric values changed",
    )?;
    checks.push(json!({"name": "valid_numeric_defaults_accepted", "accepted": true}));
    expect_control_error(
        &mut checks,
        "duplicate_cli_option_rejected",
        "cli-parse",
        Cli::parse(&[
            "bench-local".to_string(),
            "--iters".to_string(), "1".to_string(),
            "--iters".to_string(), "2".to_string(),
        ]).map(|_| ()),
    )?;

    let empty_entries = Vec::<CompactionSessionEntry>::new();
    let empty_messages = Vec::<AgentMessage>::new();
    let settings = default_compaction_settings();
    expect_control_error(
        &mut checks,
        "prepare_none_rejected_before_timing",
        "prepare-compaction",
        preparation_projection(&empty_entries, &settings, &empty_messages).map(|_| ()),
    )?;

    let model = fixture_model_for_estimates();
    let messages = fixture.to_agent_messages(&model);
    let mut invalid_config = CompactionConfig::default();
    invalid_config.max_state_tokens = 63;
    expect_control_error(
        &mut checks,
        "jev_invalid_config_rejected_before_timing",
        "jev-invalid-config",
        jev_projection(&messages, &invalid_config).map(|_| ()),
    )?;
    let no_candidates = jev_projection(&tool_pairing_fixture(&[], &[]), &CompactionConfig::default())?;
    ensure_stage(
        no_candidates["outcome"] == json!("expected-skip")
            && no_candidates["skip_class"] == json!(CompactionSkip::NoCandidates.as_str()),
        "selftest-control",
        "NoCandidates did not enter its named expected-skip lane",
    )?;
    checks.push(json!({"name": "jev_no_candidates_named_skip_accepted", "accepted": true, "projection": no_candidates}));
    let duplicate_messages = tool_pairing_fixture(&["duplicate", "duplicate"], &["duplicate"]);
    expect_control_error(
        &mut checks,
        "jev_invalid_pair_rejected_before_timing",
        "jev-invalid-pair",
        jev_projection(&duplicate_messages, &CompactionConfig::default()).map(|_| ()),
    )?;

    let mut serialized = baseline.clone();
    let original = serialized["serialization"]["text"].as_str().ok_or_else(|| {
        stage_error("selftest-control", "serialization text missing")
    })?;
    let replacement = if original.starts_with('A') { "B" } else { "A" };
    let corrupted = format!("{replacement}{}", &original[1..]);
    ensure_stage(corrupted.len() == original.len() && corrupted != original, "selftest-control", "same-length serialization mutation was ineffective")?;
    serialized["serialization"]["text"] = json!(corrupted);
    expect_control_error(
        &mut checks,
        "same_length_serialization_corruption_rejected",
        "serialization-parity",
        compare_case(&serialized, &baseline),
    )?;

    let mut branch = baseline.clone();
    let original_type = branch["session_manager"]["raw_branch_semantic"][0]["type"].clone();
    branch["session_manager"]["raw_branch_semantic"][0]["type"] = json!("r2-corrupt");
    ensure_stage(branch["session_manager"]["raw_branch_semantic"][0]["type"] != original_type, "selftest-control", "branch mutation was ineffective")?;
    expect_control_error(
        &mut checks,
        "branch_semantic_corruption_rejected",
        "branch-parity",
        compare_case(&branch, &baseline),
    )?;

    let mut checkpoint = baseline.clone();
    checkpoint["session_manager"]["checkpoint_state"][0]["tokensBefore"] = json!(43.0);
    expect_control_error(
        &mut checks,
        "checkpoint_semantic_corruption_rejected",
        "checkpoint-parity",
        compare_case(&checkpoint, &baseline),
    )?;

    let mut protected = baseline.clone();
    protected["session_manager"]["protected_custom_state"]["custom_messages"][0]["details"]["opaque_state"]["version"] = json!(8);
    expect_control_error(
        &mut checks,
        "protected_custom_semantic_corruption_rejected",
        "protected-custom-state-parity",
        compare_case(&protected, &baseline),
    )?;

    let mut normalized = baseline.clone();
    normalized["session_manager"]["normalized_restored_messages"][0]["r2_corruption"] = json!(true);
    expect_control_error(
        &mut checks,
        "normalized_restored_corruption_rejected_independently",
        "normalized-restored-parity",
        compare_case(&normalized, &baseline),
    )?;

    for (name, key, value, stage) in [
        ("iters_zero_rejected", "iters", "0", "cli-iters"),
        ("iters_negative_rejected", "iters", "-1", "cli-iters"),
        ("iters_fractional_rejected", "iters", "1.5", "cli-iters"),
        ("iters_nan_rejected", "iters", "NaN", "cli-iters"),
        ("iters_infinite_rejected", "iters", "inf", "cli-iters"),
        ("iters_overflow_rejected", "iters", "184467440737095516160", "cli-iters"),
        ("warmup_zero_rejected", "warmup", "0", "cli-warmup"),
        ("warmup_negative_rejected", "warmup", "-2", "cli-warmup"),
        ("warmup_fractional_rejected", "warmup", "2.5", "cli-warmup"),
        ("chunk_window_zero_rejected", "chunk-context-window", "0", "cli-chunk-context-window"),
        ("chunk_window_nan_rejected", "chunk-context-window", "NaN", "cli-chunk-context-window"),
        ("chunk_window_infinite_rejected", "chunk-context-window", "inf", "cli-chunk-context-window"),
    ] {
        let control_out = work.join(format!("{name}.json"));
        let fixture_text = fixture_path.to_string_lossy().to_string();
        let out_text = control_out.to_string_lossy().to_string();
        let control = invalid_cli(&[
            ("fixture", fixture_text.as_str()),
            ("out", out_text.as_str()),
            (key, value),
        ]);
        expect_control_error(&mut checks, name, stage, super::run_bench_local(&control).map(|_| ()))?;
        ensure_stage(!control_out.exists(), "selftest-control", format!("{name} emitted an output"))?;
    }

    let report = json!({
        "schema": 1,
        "self_test": "compaction-replay-r2",
        "status": "ok",
        "check_count": checks.len(),
        "named_rejection_stages": true,
        "performance_claim": false,
        "checks": checks,
    });
    write_new_json(&out, &report)?;
    println!("wrote {} ({} R2 self-test checks)", out.display(), report["check_count"]);
    Ok(0)
}
