use super::*;
use serde_json::json;

fn usage(input: f64, output: f64, cache: f64) -> Usage {
    let mut result = empty_usage();
    result.input = input;
    result.output = output;
    result.cache_read = cache;
    result.total_tokens = input + output + cache;
    result.cost.total = 0.12;
    result
}

fn assistant(id: &str, parent: &str, model: &str, usage: Usage) -> Value {
    json!({"type":"message", "id":id, "parentId":parent, "message":{
        "role":"assistant", "provider":"fixture", "model":model, "usage":usage, "stopReason":"stop"}})
}

#[test]
fn model_switches_compaction_forks_and_child_attribution_do_not_inflate_usage() {
    let child = usage(40.0, 10.0, 0.0);
    let mut aggregate = usage(100.0, 20.0, 30.0);
    add_assistant_usage(&mut aggregate, &child);
    let entries = vec![
        json!({"id":"u", "type":"message", "parentId":null, "message":{"role":"user"}}),
        assistant("a", "u", "astra", aggregate),
        json!({"id":"compact", "type":"compaction", "parentId":"a"}),
        assistant("b", "compact", "sol", usage(10.0, 5.0, 0.0)),
        assistant("fork", "u", "excluded", usage(9000.0, 1000.0, 0.0)),
        json!({"id":"attribute", "type":"child_usage_attributed", "parentId":"fork", "targetId":"a", "childUsage":child}),
    ];
    let mut stats = account(&entries, Some("b"));
    assert_eq!(stats.total(), 165.0);
    assert_eq!(stats.models.len(), 2);
    assert_eq!(stats.compactions, 1);
    assert_eq!(stats.turns, [165.0]);
    let subagent = account(
        &[assistant("child", "", "sol", usage(40.0, 10.0, 0.0))],
        Some("child"),
    );
    stats.merge(subagent);
    assert_eq!(stats.total(), 215.0);
    assert_eq!(stats.models[&("fixture".into(), "sol".into())].calls, 2);
    assert_eq!(
        stats.turns,
        [165.0],
        "current-chat trend excludes child usage"
    );
}

#[test]
fn missing_usage_is_unknown_and_synthetic_aborts_are_not_calls() {
    let mut aborted = assistant("abort", "a", "astra", empty_usage());
    aborted["message"]["stopReason"] = json!("aborted");
    let stats = account(
        &[assistant("a", "", "astra", empty_usage()), aborted],
        Some("abort"),
    );
    let row = stats.models.values().next().unwrap();
    assert_eq!(row.calls, 1);
    assert_eq!(row.missing_usage, 1);
    assert_eq!(row.missing_cost, 1);
}

fn record(request: &str, question: &str, schema: &str, applied: bool) -> CorrelationRecord {
    serde_json::from_value(json!({
        "schema_version":schema,"request_id":request,"attempt":1,"session_id":"chat",
        "turn":1,"stage":"code_search","state_fingerprint":"fixture","state_schema_version":"1",
        "category":"code_search_relevance","question_id":question,"prompt_version":"1",
        "mode":"compare+active","applied":applied,"response_model":"jev-fixture","duration_ms":80,
        "agreement":"agree","observed_metrics":{"jev_input_tokens":100,"jev_output_tokens":10},
        "actual_action":{"candidate_count":"10","retained_count":"4","estimated_candidate_tokens_before":"1000","estimated_candidate_tokens_after":"400"}
    })).unwrap()
}

#[test]
fn batched_and_combined_jev_rows_count_usage_and_savings_once() {
    let mut records = Vec::new();
    for question in ["q1", "q2", "q3"] {
        records.push(record("request", question, "jev.active/1", true));
        records.push(record("request", question, "jev.compare/1", false));
    }
    let mut other = record("unrelated", "q", "jev.active/1", true);
    other.session_id = "another-chat".into();
    records.push(other);
    let stats = account_jev(&records, &BTreeSet::from(["chat".into()]));
    assert_eq!(stats.requests, 1);
    assert_eq!(stats.applied, 1);
    assert_eq!(stats.agrees, 3);
    assert_eq!(stats.models.values().next().unwrap().total(), 110.0);
    let saved = &stats.savings["code_search_relevance"];
    assert_eq!(
        (
            saved.before,
            saved.after,
            saved.candidates,
            saved.retained,
            saved.samples
        ),
        (1000, 400, 10, 4, 1)
    );
}

#[test]
fn compare_projections_never_count_as_applied_savings() {
    let stats = account_jev(
        &[record("compare", "q", "jev.compare/1", false)],
        &BTreeSet::from(["chat".into()]),
    );
    assert!(stats.savings.is_empty());
    assert_eq!(stats.applied, 0);
    assert_eq!(stats.requests, 1);
}

#[test]
fn compaction_reduction_stays_separate_from_candidate_savings() {
    let mut compact = record("compact", "audit", "jev.compaction/1", true);
    compact.category = "compaction".into();
    compact
        .observed_metrics
        .insert("estimated_tokens_before".into(), 2000);
    compact
        .observed_metrics
        .insert("estimated_tokens_after".into(), 500);
    let stats = account_jev(
        &[compact, record("search", "q", "jev.active/1", true)],
        &BTreeSet::from(["chat".into()]),
    );
    assert_eq!(
        (stats.compaction.before, stats.compaction.after),
        (2000, 500)
    );
    assert_eq!(stats.savings.len(), 1);
    assert_eq!(stats.savings["code_search_relevance"].before, 1000);
}

#[test]
fn loader_cache_retains_only_accounting_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut entry = assistant("a", "", "astra", usage(1.0, 2.0, 0.0));
    entry["message"]["content"] = json!([{"type":"text", "text":"private prompt contents"}]);
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            json!({"type":"session", "id":"chat", "version":3}),
            entry
        ),
    )
    .unwrap();
    let mut loader = Loader::default();
    let entries = loader.entries(&path).unwrap();
    assert!(!serde_json::to_string(&entries)
        .unwrap()
        .contains("private prompt"));
    assert_eq!(account(&entries, Some("a")).total(), 3.0);
}

#[test]
fn partial_jev_usage_preserves_known_tokens_and_marks_the_missing_part() {
    let mut partial = record("partial", "q", "jev.active/1", true);
    partial.observed_metrics.remove("jev_output_tokens");
    let stats = account_jev(&[partial], &BTreeSet::from(["chat".into()]));
    let row = stats.models.values().next().unwrap();
    assert_eq!(row.total(), 100.0);
    assert_eq!(row.missing_usage, 1);
    assert_eq!(row.calls, 1);
}

#[test]
fn saved_child_discovery_is_scoped_to_the_session_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let artifacts = crate::core::session_manager::get_session_artifact_path(
        &dir.path().to_string_lossy(),
        "chat",
    );
    let child = Path::new(&artifacts).join("sub-fixture");
    std::fs::create_dir_all(&child).unwrap();
    std::fs::create_dir_all(Path::new(&artifacts).join("other")).unwrap();
    std::fs::write(Path::new(&artifacts).join("sub-file"), "not a directory").unwrap();
    assert_eq!(child_dirs(Path::new(&artifacts)), [child]);
    let unrelated = crate::core::session_manager::get_session_artifact_path(
        &dir.path().to_string_lossy(),
        "another-chat",
    );
    assert!(child_dirs(Path::new(&unrelated)).is_empty());
}


#[test]
fn textmas_accounting_is_deduplicated_separate_and_keeps_unknown_costs() {
    use crate::core::occ::textmas::{TextMasCallStatus, TextMasRole, TextMasUsageRecord, USAGE_CUSTOM_TYPE};
    let custom = |id: &str, parent: &str, record: TextMasUsageRecord| json!({
        "id":id,"parentId":parent,"type":"custom","customType":USAGE_CUSTOM_TYPE,"data":record
    });
    let started = TextMasUsageRecord::new("task", TextMasRole::Planner, TextMasCallStatus::Started, None);
    let finished = TextMasUsageRecord::new("task", TextMasRole::Planner, TextMasCallStatus::Completed, Some(usage(11.0, 4.0, 2.0)));
    let cancelled = TextMasUsageRecord::new("task", TextMasRole::Critic, TextMasCallStatus::Cancelled, None);
    let entries = vec![
        json!({"id":"u","type":"message","message":{"role":"user"}}),
        custom("s", "u", started), custom("d", "s", finished.clone()),
        custom("dup", "d", finished), custom("c", "dup", cancelled),
        assistant("a", "c", "model", usage(3.0, 2.0, 0.0)),
        custom("offbranch", "u", TextMasUsageRecord::new("other", TextMasRole::Refiner, TextMasCallStatus::Started, None)),
    ];
    let reduced: Vec<_> = entries.iter().map(usage_entry).collect();
    let mut accounting = account(&reduced, Some("a"));
    assert_eq!(accounting.total(), 5.0, "do not forge ordinary provider calls");
    assert_eq!(accounting.turns, [5.0]);
    assert_eq!(accounting.models.len(), 1);
    let summary = accounting.textmas_usage.as_ref().unwrap();
    assert_eq!(summary.calls, 2);
    assert_eq!(summary.unknown_usage_calls, 1);
    assert_eq!(summary.unknown_cost_calls, 1);
    assert_eq!(summary.known_subtotal.total_tokens, 17.0);
    accounting.merge(account(&[custom("child", "", TextMasUsageRecord::new("task", TextMasRole::Planner, TextMasCallStatus::Started, None))], Some("child")));
    let summary = accounting.textmas_usage.unwrap();
    assert_eq!(summary.calls, 3, "distinct sessions retain separate call identities");
    assert_eq!(summary.unknown_usage_calls, 2);
    assert_eq!(summary.unknown_cost_calls, 2);
    assert_eq!(summary.known_subtotal.total_tokens, 17.0);
}

#[test]
fn stats_projection_drops_arbitrary_or_malformed_textmas_payloads() {
    let entry = json!({"id":"a","type":"custom","customType":"occ31_textmas_usage",
        "data":{"privateDraft":"DO_NOT_CACHE","usage":{"input":10}}});
    let reduced = usage_entry(&entry);
    assert!(reduced.get("data").is_none());
    assert!(!reduced.to_string().contains("DO_NOT_CACHE"));
    assert!(account(&[reduced], Some("a")).textmas_usage.is_none());
}


#[test]
fn mixed_textmas_token_reports_keep_raw_usage_and_correct_display_subtotal() {
    use crate::core::occ::textmas::{TextMasCallStatus, TextMasRole, TextMasUsageRecord, USAGE_CUSTOM_TYPE};
    let mut full = usage(10.0, 0.0, 0.0);
    full.total_tokens = 10.0;
    let mut components = usage(20.0, 0.0, 0.0);
    components.total_tokens = 0.0;
    let mut total_only = Usage::default();
    total_only.total_tokens = 30.0;
    let mut entries = Vec::new();
    let mut parent = String::new();
    for (index, (role, usage)) in TextMasRole::ALL.into_iter().zip([full, components, total_only]).enumerate() {
        let id = format!("record-{index}");
        entries.push(usage_entry(&json!({"id":id,"parentId":parent,"type":"custom","customType":USAGE_CUSTOM_TYPE,
            "data":TextMasUsageRecord::new("mixed", role, TextMasCallStatus::Completed, Some(usage))})));
        parent = id;
    }
    let mut stats = account(&entries, Some(&parent));
    let summary = stats.textmas_usage.as_ref().unwrap();
    assert_eq!(summary.known_tokens, 60.0);
    assert_eq!(summary.known_subtotal.total_tokens, 40.0, "raw provider fields remain untouched");
    assert_eq!(summary.known_subtotal.input, 30.0);
    stats.merge(account(&entries, Some(&parent)));
    assert_eq!(stats.textmas_usage.as_ref().unwrap().known_tokens, 120.0);
    assert_eq!(stats.total(), 0.0, "overhead is not a fake ordinary model reply");
}
