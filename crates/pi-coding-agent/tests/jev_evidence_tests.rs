//! ROOT-CONTRACT v6 (Evidence lane): wiring tests for the host adapter
//! (`core/jev_evidence.rs`), driven through the REAL observer with the mock
//! transport — same discipline as the search lane's tests.
//!
//! ## Landing note
//!
//! These tests need the integrator's shared hunks from
//! `reports/jev-docs-audit/EVIDENCE-SHARED-HUNKS-v1.{md,json}` (v2 semantics):
//! - `pi-jev/src/types.rs`: `DecisionCategory::{CodeRetrievalSafety, CodeCitationCheck}`.
//! - `pi-jev/src/lib.rs`: `pub mod evidence;`.
//! - `core/mod.rs`: `pub mod jev_evidence;`.
//! - `pi-jev/src/hooks.rs`: the `CodeCitationCheck` acceptance branch
//!   (`assess_citation_record`) — the citation test that asserts an accepted
//!   decision fails before it lands.
//!
//! Cross-lane acceptance discipline: the battery NEVER shares a request with
//! the filter questions (the scheduler's real per-request cap is 16; a
//! combined 8+24 request would be silently refused and kill the filter), and
//! every test asserts the ACTUAL transport-captured request —
//! `RecordedCall.question_ids` and `RecordedCall.state_fingerprint` — not
//! just the pure builders. Test discipline: `--exact --test-threads=1`,
//! `DEV_DEBUG=0 TEST_DEBUG=0`, no `PI_*` overrides.

use std::time::Duration;

use pi_coding_agent::core::jev_code_search::{
    prepare as prepare_search, recent_code_search_envelope,
};
use pi_coding_agent::core::jev_evidence::{
    annotate_citation_check, is_evidence_annotation_text, observe_safety_battery, prepare_citation,
    prepare_original_citation, run_safety_veto, BatteryInputs, CitationStageInputs,
};
use pi_jev::evidence::{battery_state, select_battery_candidates};
use pi_jev::hooks::{JevObserver, JevObserverConfig};
use pi_jev::mock::MOCK_RESPONSE_MODEL;
use pi_jev::search::SearchBudget;
use pi_jev::snapshot::fingerprint_of;
use pi_jev::{
    JevLimits, JevMode, JevStats, JevSystemOne, MockJevTransport, MockStep, SecretString,
};
use serde_json::{json, Value};
use std::sync::Arc;

fn observer(
    mode: JevMode,
    steps: Vec<MockStep>,
) -> (Arc<JevObserver>, Arc<MockJevTransport>, tempfile::TempDir) {
    observer_with_gate(mode, mode, steps)
}

/// `client_mode` builds the JevSystemOne client (production refuses an Off
/// client); `gate_mode` is what the observer's mode_gate reports — the
/// search-lane pattern for Off-mode fixtures.
fn observer_with_gate(
    client_mode: JevMode,
    gate_mode: JevMode,
    steps: Vec<MockStep>,
) -> (Arc<JevObserver>, Arc<MockJevTransport>, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let transport = Arc::new(MockJevTransport::scripted(steps));
    let client = Arc::new(
        JevSystemOne::new(
            client_mode,
            SecretString::new("synthetic"),
            transport.clone(),
            JevLimits::default(),
            Arc::new(JevStats::default()),
        )
        .unwrap(),
    );
    let observer = JevObserver::new(
        JevObserverConfig {
            mode_gate: Arc::new(move |_| (gate_mode, pi_jev::hooks::SYSTEM_ONE_MODEL.to_string())),
            // Match the policy_generation the fixture inputs declare, so the
            // observer's can_apply gate accepts the adapter's payload.
            policy_generation: Arc::new(|_, _| "gen-1".to_string()),
            ..Default::default()
        },
        client,
        temp.path().join("records.jsonl"),
    );
    (observer, transport, temp)
}

fn source_read_messages(id: &str, path: &str, content: &str) -> Vec<Value> {
    vec![
        json!({"role": "user", "content": [{"type": "text", "text": "make retries configurable"}]}),
        json!({"role": "assistant", "content": [
            {"type": "toolCall", "id": id, "name": "ipython",
             "arguments": {"code": format!("print(open(\"{path}\").read())")}}
        ]}),
        json!({
            "role": "toolResult",
            "toolCallId": id,
            "toolName": "ipython",
            "isError": false,
            "content": [{"type": "text", "text": content}],
            "details": {"status": "ok", "stdout": content, "result": null},
        }),
    ]
}

fn citation_answer_body(question_id: &str, choice: &str, confidence: f64) -> String {
    json!({
        "model": MOCK_RESPONSE_MODEL,
        "answers": {
            question_id: {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    "contradicts": if choice == "contradicts" { 0.9 } else { 0.05 },
                    "supports": if choice == "supports" { 0.9 } else { 0.05 },
                    "unclear": if choice == "unclear" { 0.9 } else { 0.05 },
                },
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 7, "output_tokens": 3}
    })
    .to_string()
}

fn inputs<'a>(
    observer: &'a Arc<JevObserver>,
    mode: JevMode,
    claim: &'a str,
    budget: &'a SearchBudget,
    cancelled: &'a (dyn Fn() -> bool + Sync),
) -> CitationStageInputs<'a> {
    CitationStageInputs {
        observer: Arc::clone(observer),
        session_id: "test-session",
        turn: 5,
        mode,
        policy_generation: "gen-1",
        max_decision_age: Duration::from_secs(60),
        budget,
        claim,
        pre_annotation_presentation: None,
        is_cancelled: cancelled,
    }
}

fn battery_inputs<'a>(
    observer: &'a Arc<JevObserver>,
    mode: JevMode,
    budget: &'a SearchBudget,
    cancelled: &'a (dyn Fn() -> bool + Sync),
) -> BatteryInputs<'a> {
    BatteryInputs {
        observer: Arc::clone(observer),
        session_id: "test-session",
        turn: 5,
        mode,
        policy_generation: "gen-1",
        max_decision_age: Duration::from_secs(60),
        budget,
        is_cancelled: cancelled,
    }
}

const CLAIM_WITH_QUOTE: &str = r#"Make the retry configurable; the source says "fn retry_with_backoff(u32) {}" is the helper."#;
const SPAN_WITH_QUOTE: &str = "// helpers\nfn retry_with_backoff(u32) {}\nfn other() {}\n";

#[tokio::test]
async fn citation_annotation_attaches_advisory_supports_when_verbatim_found() {
    let (observer, transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(citation_answer_body(
            "code_citation_check.relation",
            "supports",
            0.93,
        ))],
    );
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().expect("content");
    assert_eq!(content.len(), 2, "one additive block");
    let block: Value = serde_json::from_str(content[1]["text"].as_str().unwrap()).unwrap();
    let body = &block["jev_citation_check"];
    assert_eq!(body["advisory"], json!(true));
    assert_eq!(body["relation"], json!("supports"));
    assert_eq!(body["confidence"], json!(0.93));
    assert_eq!(body["review"], json!(false));
    assert_eq!(body["quote_presence"], json!("found_in_supplied_span"));
    assert_eq!(body["verbatim_quote_match"], json!(true));
    assert_eq!(body["source"]["path"], json!("src/retry.rs"));
    // Originals untouched: block 0 is byte-identical.
    assert_eq!(content[0]["text"].as_str().unwrap(), SPAN_WITH_QUOTE);
    assert_eq!(content[1]["type"], json!("text"));
    // Cross-lane acceptance: verify the ACTUAL outgoing request captured by
    // the native mock transport — exactly one question, and the state bytes
    // reached the transport unmodified (fingerprint equality).
    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].question_ids, vec!["code_citation_check.relation"]);
    let expected_state = json!({
        "user_text_excerpt": CLAIM_WITH_QUOTE,
        "citation_source_path": "src/retry.rs",
        "citation_span": SPAN_WITH_QUOTE,
    });
    assert_eq!(calls[0].state_fingerprint, fingerprint_of(&expected_state));
}

#[tokio::test]
async fn citation_annotation_reports_native_verbatim_failure_as_review() {
    let (observer, _transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(citation_answer_body(
            "code_citation_check.relation",
            "supports",
            0.99,
        ))],
    );
    // The claim's quote is NOT in the supplied span: the relation is still
    // advisory, but the native check forces the review flag and never
    // converts the model's "supports" into verification.
    let claim = r#"The source says "fn totally_absent_symbol() {}" exists."#;
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(&observer, JevMode::Active, claim, &budget, &cancelled);
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    let block: Value = serde_json::from_str(content[1]["text"].as_str().unwrap()).unwrap();
    assert_eq!(block["jev_citation_check"]["review"], json!(true));
    assert_eq!(
        block["jev_citation_check"]["quote_presence"],
        json!("quote_not_found_in_supplied_source_span")
    );
    assert_eq!(
        block["jev_citation_check"]["verbatim_quote_match"],
        json!(false)
    );
    assert_eq!(block["jev_citation_check"]["relation"], json!("supports"));
    assert!(block["jev_citation_check"]["note"]
        .as_str()
        .unwrap()
        .contains("not verification"));
}

#[tokio::test]
async fn citation_annotation_handles_no_quote_and_contradicts() {
    let (observer, _transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(citation_answer_body(
            "code_citation_check.relation",
            "contradicts",
            0.62,
        ))],
    );
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        "Make the retry configurable somehow.",
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    let block: Value = serde_json::from_str(content[1]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        block["jev_citation_check"]["relation"],
        json!("contradicts")
    );
    assert_eq!(
        block["jev_citation_check"]["review"],
        json!(true),
        "low confidence"
    );
    assert_eq!(
        block["jev_citation_check"]["quote_presence"],
        json!("no_quote_supplied")
    );
    assert!(block["jev_citation_check"]["verbatim_quote_match"].is_null());
}

#[tokio::test]
async fn citation_over_cap_state_refuses_transport_free() {
    // The citation state carries the claim verbatim (user_text_excerpt) plus
    // the span EXCERPT (bounded to CITATION_SPAN_EXCERPT_CHARS), so an
    // oversized claim — not the span — is what pushes the serialized state
    // over the observer's 8 KiB raw-state precondition: the adapter refuses
    // BEFORE any transport call (prepare_explicit parity, no silent drop).
    let long_claim = format!(
        r#"Claim: "fn retry_with_backoff(u32) {{}}" plus padding. {}"#,
        "x".repeat(9000)
    );
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Valid]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(&observer, JevMode::Active, &long_claim, &budget, &cancelled);
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1, "no annotation on a refused round");
    assert_eq!(
        transport.call_count(),
        0,
        "refused before any transport call"
    );
}

#[tokio::test]
async fn malformed_transport_fails_open_and_preserves_originals() {
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Timeout]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1, "no annotation on unavailable outcome");
    assert_eq!(content[0]["text"].as_str().unwrap(), SPAN_WITH_QUOTE);
    assert!(
        transport.call_count() >= 1,
        "the attempt happened, then failed open"
    );
}

#[tokio::test]
async fn skipped_answer_fails_open() {
    // A body whose answers omit the citation question -> skips non-empty.
    let body = json!({
        "model": MOCK_RESPONSE_MODEL,
        "answers": {},
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let (observer, _transport, _temp) = observer(JevMode::Active, vec![MockStep::Body(body)]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn cancelled_fails_open_before_any_transport_call() {
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Valid]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || true;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
    assert_eq!(
        transport.call_count(),
        0,
        "cancelled before any transport call"
    );
}

#[tokio::test]
async fn expired_budget_fails_open() {
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Valid]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::ZERO);
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
    assert_eq!(transport.call_count(), 0);
}

#[tokio::test]
async fn off_mode_changes_nothing() {
    // Off refuses client construction in production: build a Compare client
    // and let the OBSERVER's mode_gate report Off instead; the adapter must
    // still change nothing and call nothing.
    let (observer, transport, _temp) =
        observer_with_gate(JevMode::Compare, JevMode::Off, vec![MockStep::Valid]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Off,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
    assert_eq!(transport.call_count(), 0);
}

#[tokio::test]
async fn compare_mode_observes_without_applying() {
    let (observer, transport, _temp) = observer(JevMode::Compare, vec![MockStep::Valid]);
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Compare,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(
        result[2]["content"].as_array().unwrap().len(),
        1,
        "nothing applied"
    );
    // Compare observations may be delivered by the scheduler, but they are
    // still only the citation question; the second round short-circuits on
    // the observation cache without a second observation.
    for call in transport.calls() {
        assert_eq!(call.question_ids, vec!["code_citation_check.relation"]);
    }
    let calls_after_first = transport.calls().len();
    let second = annotate_citation_check(result, &inputs).await;
    assert_eq!(second[2]["content"].as_array().unwrap().len(), 1);
    assert_eq!(
        transport.calls().len(),
        calls_after_first,
        "cache short-circuit"
    );
}

#[tokio::test]
async fn ineligible_inputs_change_nothing() {
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Valid]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    // No ipython toolResult at all.
    let messages = vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})];
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    assert_eq!(result.len(), 1);
    // Instruction-class path (.md) is never a citation source.
    let md = source_read_messages("call-2", "docs/guide.md", "# Guide\nbe careful");
    let result = annotate_citation_check(md, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
    // Excluded message classes (pinned) never get annotations.
    let mut pinned = source_read_messages("call-3", "src/retry.rs", SPAN_WITH_QUOTE);
    pinned[2]["pinned"] = json!(true);
    let result = annotate_citation_check(pinned, &inputs).await;
    assert_eq!(result[2]["content"].as_array().unwrap().len(), 1);
    // Multi-block toolResult refuses the additive block (one-annotation rule).
    let mut multi = source_read_messages("call-4", "src/retry.rs", SPAN_WITH_QUOTE);
    multi[2]["content"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type": "text", "text": "extra"}));
    let result = annotate_citation_check(multi, &inputs).await;
    assert_eq!(
        result[2]["content"].as_array().unwrap().len(),
        2,
        "untouched"
    );
    assert_eq!(transport.call_count(), 0, "nothing eligible ever called");
}

#[tokio::test]
async fn prepare_citation_recognizes_exactly_the_read_idiom() {
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let presentations = prepare_citation(&messages, CLAIM_WITH_QUOTE);
    assert_eq!(presentations.len(), 1);
    assert_eq!(presentations[0].path, "src/retry.rs");
    assert_eq!(presentations[0].span, SPAN_WITH_QUOTE);
    assert!(!presentations[0].span_truncated);
    assert_eq!(presentations[0].message_index, 2);

    for accepted in [
        "print(open(\"src/retry.rs\", \"r\").read())",
        "print(open(\"src/retry.rs\", mode=\"r\", encoding=\"utf-8\").read())",
    ] {
        let mut messages = source_read_messages("open-read", "src/retry.rs", SPAN_WITH_QUOTE);
        messages[1]["content"][0]["arguments"]["code"] = json!(accepted);
        assert_eq!(
            prepare_citation(&messages, CLAIM_WITH_QUOTE).len(),
            1,
            "advertised read-only open form is accepted: {accepted:?}"
        );
    }
    let mut update_mode = source_read_messages("open-update", "src/retry.rs", SPAN_WITH_QUOTE);
    update_mode[1]["content"][0]["arguments"]["code"] =
        json!("print(open(\"src/retry.rs\", \"r+\").read())");
    assert!(prepare_citation(&update_mode, CLAIM_WITH_QUOTE).is_empty());

    // The imported branch matches the exact compaction-safe spelling. Its
    // aliases, open form, encoding extension, semicolon, and extra statement
    // stay outside this narrowly shared grammar.
    let imported_code = "from pathlib import Path\nprint(Path(\"src/retry.rs\").read_text())";
    let mut imported = source_read_messages("imported", "src/retry.rs", SPAN_WITH_QUOTE);
    imported[1]["content"][0]["arguments"]["code"] = json!(imported_code);
    let imported_presentations = prepare_citation(&imported, CLAIM_WITH_QUOTE);
    assert_eq!(imported_presentations.len(), 1);
    assert_eq!(imported_presentations[0].path, "src/retry.rs");
    for rejected in [
        "from pathlib import Path as P\nprint(P(\"src/retry.rs\").read_text())",
        "from pathlib import Path\nprint(open(\"src/retry.rs\").read())",
        "from pathlib import Path\nprint(Path(\"src/retry.rs\").read_text(encoding=\"utf-8\"))",
        "from pathlib import Path\nprint(Path(\"src/retry.rs\").read_text());",
        "from pathlib import Path\nprint(Path(\"src/retry.rs\").read_text())\nprint(\"extra\")",
    ] {
        let mut messages = source_read_messages("rejected", "src/retry.rs", SPAN_WITH_QUOTE);
        messages[1]["content"][0]["arguments"]["code"] = json!(rejected);
        assert!(
            prepare_citation(&messages, CLAIM_WITH_QUOTE).is_empty(),
            "imported near-miss must fail closed: {rejected:?}"
        );
    }

    // Write modes are rejected (fail closed).
    let write_call = vec![
        json!({"role": "user", "content": [{"type": "text", "text": "x"}]}),
        json!({"role": "assistant", "content": [
            {"type": "toolCall", "id": "c2", "name": "ipython",
             "arguments": {"code": "print(open(\"src/a.rs\", \"w\").write())"}}
        ]}),
        json!({"role": "toolResult", "toolCallId": "c2", "toolName": "ipython", "isError": false,
               "content": [{"type": "text", "text": "done"}],
               "details": {"status": "ok", "stdout": "done", "result": null}}),
    ];
    assert!(prepare_citation(&write_call, "claim").is_empty());
}

// ---------------------------------------------------------------------------
// Safety battery: own request, explicit subset, veto, transport-verified
// ---------------------------------------------------------------------------

fn envelope_messages(id: &str, count: usize) -> Vec<Value> {
    let candidates: Vec<Value> = (0..count)
        .map(|n| {
            json!({
                "kind": "file",
                "path": format!("src/file{n}.rs"),
                "line": n + 1,
                "snippet": format!("fn unit_{n}() {{ let value = {n}; }}"),
            })
        })
        .collect();
    let envelope = json!({"schema": "rlm.code-search/1", "candidates": candidates});
    vec![
        json!({"role": "user", "content": [{"type": "text", "text": "find the retry policy"}]}),
        json!({"role": "assistant", "content": [
            {"type": "toolCall", "id": id, "name": "ipython", "arguments": {"code": "present(results)"}}
        ]}),
        json!({
            "role": "toolResult",
            "toolCallId": id,
            "toolName": "ipython",
            "isError": false,
            "content": [{"type": "text", "text": envelope.to_string()}],
            "details": {"status": "ok", "stdout": null, "result": null},
        }),
    ]
}

/// The scripted body for one battery round over `covered` candidates with
/// per-ask Noul values (default 0.05: quiet).
fn battery_body(covered: usize, flagged_ask: Option<usize>) -> String {
    let mut answers = serde_json::Map::new();
    for ask in 0..covered * 3 {
        let value = if Some(ask) == flagged_ask { 0.93 } else { 0.05 };
        answers.insert(
            format!("code_retrieval_safety.{ask}"),
            json!({"type": "noul", "noul": value}),
        );
    }
    json!({
        "model": MOCK_RESPONSE_MODEL,
        "answers": answers,
        "usage": {"input_tokens": 7, "output_tokens": 3}
    })
    .to_string()
}

#[tokio::test]
async fn battery_round_vetoes_planned_drop_and_verifies_transport_request() {
    // Real presentation through the public search preparation API.
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    assert!(!presentations.is_empty());
    let presentation = &presentations[0];
    let envelope_indices: Vec<usize> = presentation
        .batches
        .iter()
        .flat_map(|batch| batch.candidate_indices.iter().copied())
        .collect();
    assert_eq!(envelope_indices.len(), 4);
    // Plan to drop the FIRST and SECOND candidates. The battery covers them
    // first (the veto targets), then retained candidates up to the cap.
    let removals = vec![envelope_indices[0], envelope_indices[1]];
    let covered = select_battery_candidates(
        &removals,
        &envelope_indices,
        pi_jev::evidence::SAFETY_BATTERY_CANDIDATE_CAP,
    );
    assert_eq!(covered.len(), 4);
    assert_eq!(covered[0], envelope_indices[0]);
    assert_eq!(covered[1], envelope_indices[1]);
    // The battery flags ask 0 (the first planned drop) as a high
    // premise-contradiction (ordinal = ask*3 + 1).
    let flagged_ask = Some(1usize);
    let (observer, transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(battery_body(covered.len(), flagged_ask))],
    );
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let (kept, annotation, metadata, battery_outcome) =
        run_safety_veto(&inputs, presentation, removals.clone()).await;
    // The correlated battery outcome rides to the caller so the effect site
    // can fold its freshness into still_current.
    assert!(
        battery_outcome.is_some(),
        "accepted round carries the outcome"
    );
    // The contradiction-flagged planned drop is un-dropped; the other stays.
    assert_eq!(kept, vec![envelope_indices[1]]);
    // The annotation exists, is advisory, and discloses the scope.
    let annotation = annotation.expect("annotation on flagged round");
    let body = &annotation["jev_retrieval_safety"];
    assert_eq!(body["advisory"], json!(true));
    assert_eq!(body["vetoed_removals"], json!([envelope_indices[0]]));
    assert_eq!(body["battery_scope"]["planned_removals_total"], json!(2));
    assert_eq!(body["battery_scope"]["planned_removals_covered"], json!(2));
    // Metadata is truthful about coverage.
    assert_eq!(
        metadata.get("jev_safety_battery").map(String::as_str),
        Some("on")
    );
    assert_eq!(
        metadata.get("jev_safety_drops_covered").map(String::as_str),
        Some("2/2")
    );
    // Cross-lane acceptance: the ACTUAL transport-captured request carried
    // EXACTLY the battery questions (never the filter's), and the state
    // bytes reached the transport unmodified.
    let calls = transport.calls();
    assert_eq!(calls.len(), 1, "one battery request");
    let mut expected_ids: Vec<String> = (0..covered.len() * 3)
        .map(|ask| format!("code_retrieval_safety.{ask}"))
        .collect();
    // The scheduler orders the explicit request's questions by id
    // (lexicographic); compare both sides in that same order so the
    // assertion checks the exact id set, not ask-ordinal ordering.
    let mut actual_ids = calls[0].question_ids.clone();
    actual_ids.sort();
    expected_ids.sort();
    assert_eq!(actual_ids, expected_ids);
    assert!(
        calls[0].question_ids.len() <= 16,
        "battery stays under the scheduler cap"
    );
    // Every asked candidate has its excerpt present in the state the
    // transport received (fingerprint equality over the intended state).
    // Mirror the adapter's presentation_candidates exactly: excerpts come
    // from the presentation's batch.state code_search_candidates (the search
    // lane's processed excerpts), keyed by envelope index via
    // candidate_indices — not from the raw envelope — and the task excerpt is
    // the first batch's user_text_excerpt.
    let mut task_excerpt = String::new();
    let mut envelope_excerpts: Vec<(usize, String)> = Vec::new();
    for batch in &presentation.batches {
        if task_excerpt.is_empty() {
            task_excerpt = batch
                .state
                .get("user_text_excerpt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
        let candidates = batch.state["code_search_candidates"]
            .as_array()
            .expect("batch candidates");
        for (ordinal, candidate) in candidates.iter().enumerate() {
            let envelope_index = batch.candidate_indices[ordinal];
            envelope_excerpts.push((
                envelope_index,
                candidate["excerpt"].as_str().unwrap().to_string(),
            ));
        }
    }
    let excerpts: Vec<(usize, String)> = covered
        .iter()
        .filter_map(|index| {
            envelope_excerpts
                .iter()
                .find(|(candidate, _)| candidate == index)
                .cloned()
        })
        .collect();
    let expected_state = battery_state(&task_excerpt, &excerpts);
    assert_eq!(calls[0].state_fingerprint, fingerprint_of(&expected_state));
}

#[tokio::test]
async fn battery_round_fails_open_on_transport_failure() {
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let envelope_indices: Vec<usize> = presentation
        .batches
        .iter()
        .flat_map(|batch| batch.candidate_indices.iter().copied())
        .collect();
    let removals = vec![envelope_indices[0]];
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Timeout]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let (kept, annotation, metadata, battery_outcome) =
        run_safety_veto(&inputs, presentation, removals.clone()).await;
    // Fail-open: removals unchanged, no annotation, refused telemetry.
    assert_eq!(kept, removals);
    assert!(annotation.is_none());
    assert_eq!(
        metadata.get("jev_safety_battery").map(String::as_str),
        Some("refused")
    );
    assert!(battery_outcome.is_none(), "refused rounds carry no outcome");
    assert!(
        transport.call_count() >= 1,
        "the attempt happened, then failed open"
    );
}

#[tokio::test]
async fn battery_round_fails_open_on_skipped_answers() {
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let envelope_indices: Vec<usize> = presentation
        .batches
        .iter()
        .flat_map(|batch| batch.candidate_indices.iter().copied())
        .collect();
    let removals = vec![envelope_indices[0]];
    // The body omits half the battery answers -> skips non-empty -> refused.
    let body = json!({
        "model": MOCK_RESPONSE_MODEL,
        "answers": {"code_retrieval_safety.0": {"type": "noul", "noul": 0.05}},
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Body(body)]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let (kept, annotation, metadata, battery_outcome) =
        run_safety_veto(&inputs, presentation, removals.clone()).await;
    assert_eq!(kept, removals);
    assert!(annotation.is_none());
    assert_eq!(
        metadata.get("jev_safety_battery").map(String::as_str),
        Some("refused")
    );
    assert!(battery_outcome.is_none(), "refused rounds carry no outcome");
    assert!(transport.call_count() >= 1);
}

#[tokio::test]
async fn battery_round_without_planned_removals_is_quiet_on_all_clear() {
    // No planned drops: battery covers retained candidates; all quiet values
    // mean no veto, no annotation, but truthful "on" metadata.
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let (observer, transport, _temp) =
        observer(JevMode::Active, vec![MockStep::Body(battery_body(4, None))]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let (kept, annotation, metadata, battery_outcome) =
        run_safety_veto(&inputs, presentation, Vec::new()).await;
    assert!(kept.is_empty());
    assert!(annotation.is_none(), "all-clear attaches no block");
    assert_eq!(
        metadata.get("jev_safety_battery").map(String::as_str),
        Some("on")
    );
    assert_eq!(transport.call_count(), 1);
}

#[tokio::test]
async fn battery_cancelled_fails_open_before_any_transport_call() {
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let (observer, transport, _temp) = observer(JevMode::Active, vec![MockStep::Valid]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || true;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let (_, annotation, metadata, battery_outcome) =
        run_safety_veto(&inputs, presentation, Vec::new()).await;
    assert!(annotation.is_none());
    assert_eq!(
        metadata.get("jev_safety_battery").map(String::as_str),
        Some("refused")
    );
    assert!(
        battery_outcome.is_none(),
        "cancelled rounds carry no outcome"
    );
    assert_eq!(transport.call_count(), 0);
}

#[tokio::test]
async fn battery_compare_mode_observes_only() {
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let (observer, transport, _temp) = observer(JevMode::Compare, vec![MockStep::Valid]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Compare, &budget, &cancelled);
    observe_safety_battery(&inputs, presentation);
    // Compare observations may be delivered by the scheduler, but they carry
    // only the battery questions; the second round short-circuits on the
    // observation cache.
    for call in transport.calls() {
        assert!(call
            .question_ids
            .iter()
            .all(|id| id.starts_with("code_retrieval_safety.")));
    }
    let calls_after_first = transport.calls().len();
    observe_safety_battery(&inputs, presentation);
    assert_eq!(
        transport.calls().len(),
        calls_after_first,
        "cache short-circuit"
    );
}

#[tokio::test]
async fn battery_and_filter_never_share_a_request() {
    // The battery's captured request carries ONLY code_retrieval_safety
    // questions — the filter's keep/drop questions are never in it.
    let messages = envelope_messages("env-1", 4);
    let presentations = prepare_search(&messages, "find the retry policy", &Default::default());
    let presentation = &presentations[0];
    let (observer, transport, _temp) =
        observer(JevMode::Active, vec![MockStep::Body(battery_body(4, None))]);
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = battery_inputs(&observer, JevMode::Active, &budget, &cancelled);
    let _ = run_safety_veto(&inputs, presentation, Vec::new()).await;
    let calls = transport.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0]
        .question_ids
        .iter()
        .all(|id| id.starts_with("code_retrieval_safety.")));
}

#[test]
fn recent_envelope_locator_sees_original_when_not_projected() {
    // Sanity anchor borrowed from the search lane: the locator API used by
    // the bridge is untouched by this lane.
    let messages = envelope_messages("env-9", 2);
    assert!(recent_code_search_envelope(&messages).is_some());
}

#[test]
fn evidence_annotation_texts_are_recognized_by_exact_prefix() {
    // The serialized single-key envelope prefixes are what the compaction
    // eligibility scan matches so annotated tool results stay untouched.
    let safety =
        serde_json::to_string(&json!({"jev_retrieval_safety": {"advisory": true}})).unwrap();
    let citation =
        serde_json::to_string(&json!({"jev_citation_check": {"advisory": true}})).unwrap();
    assert!(is_evidence_annotation_text(&safety));
    assert!(is_evidence_annotation_text(&citation));
    assert!(!is_evidence_annotation_text("fn retry_with_backoff() {}"));
    assert!(!is_evidence_annotation_text(""));
    assert!(!is_evidence_annotation_text(
        "{\"jev_omitted_candidates\":1}"
    ));
    // The line-find lane's envelope is a recognised advisory annotation too:
    // compaction protects it and citation reachability strips it while
    // keeping the single-original-source eligibility.
    let line_find = serde_json::to_string(&json!({"jev_line_find": {"verdict": "FOUND"}})).unwrap();
    assert!(is_evidence_annotation_text(&line_find));
}

#[test]
fn production_citation_basis_requires_one_unannotated_original_block() {
    let clean = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    assert!(prepare_original_citation(&clean, CLAIM_WITH_QUOTE).is_some());

    // A prefix-shaped block present before the request-local pipeline is not
    // trusted as a host annotation. Even if the remaining plain block would
    // look eligible after prefix skipping, strict pre-annotation capture
    // refuses the multi-block result.
    let mut prefixed_multi = clean.clone();
    prefixed_multi[2]["content"].as_array_mut().unwrap().insert(
        0,
        json!({
            "type": "text",
            "text": serde_json::to_string(&json!({
                "jev_line_find": {"verdict": "FOUND"}
            }))
            .unwrap()
        }),
    );
    assert!(prepare_citation(&prefixed_multi, CLAIM_WITH_QUOTE).len() == 1);
    assert!(prepare_original_citation(&prefixed_multi, CLAIM_WITH_QUOTE).is_none());

    let mut prefix_shaped_source = clean;
    prefix_shaped_source[2]["content"][0]["text"] = json!(serde_json::to_string(&json!({
        "jev_citation_check": {"advisory": true}
    }))
    .unwrap());
    assert!(prepare_original_citation(&prefix_shaped_source, CLAIM_WITH_QUOTE).is_none());
}

#[tokio::test]
async fn citation_reachability_refuses_multi_block_source_reads_before_any_call() {
    // Strict eligibility, recorded per the trace review: an ORIGINAL source
    // read plus an UNRECOGNISED extra block (or an image) stays ineligible —
    // no call, no record, original kept. RECOGNISED advisory prefixes
    // ({"jev_line_find":, {"jev_retrieval_safety":) are stripped for
    // eligibility instead: the composition test above covers a line-find
    // annotation composing with citation on the same source. This test pins
    // the refusal for anything outside the recognised advisory set.
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let mut messages = messages;
    messages[2]["content"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type": "text", "text": "prior note"}));
    let (observer, transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(citation_answer_body(
            "code_citation_check.relation",
            "supports",
            0.93,
        ))],
    );
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2, "nothing attached; original blocks stay");
    assert_eq!(
        transport.call_count(),
        0,
        "reachability refuses before any call"
    );
    // The defensive attach-state key in the adapter can only fire if the
    // reachability single-block rule is ever relaxed; the prefix helper that
    // compaction relies on is unit-tested above.
    assert!(!is_evidence_annotation_text("prior note"));
}

#[tokio::test]
async fn citation_composes_with_a_prior_line_find_annotation_on_the_same_source() {
    // FULL-PROFILE composition: the SAME native source-read result is
    // assessed by line-find first (its bounded advisory block is already on
    // the toolResult) and then by citation. The original source text stays
    // the single eligibility basis; both annotations compose additively; no
    // block is merged into the source text.
    let line_find_block = json!({"type": "text", "text": serde_json::to_string(&json!({
        "jev_line_find": {"verdict": "FOUND", "top_lines": ["0"]}
    })).unwrap()});
    let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
    let mut messages = messages;
    messages[2]["content"]
        .as_array_mut()
        .unwrap()
        .push(line_find_block);
    let (observer, transport, _temp) = observer(
        JevMode::Active,
        vec![MockStep::Body(citation_answer_body(
            "code_citation_check.relation",
            "supports",
            0.93,
        ))],
    );
    let budget = SearchBudget::from_now(Duration::from_secs(30));
    let cancelled = || false;
    let inputs = inputs(
        &observer,
        JevMode::Active,
        CLAIM_WITH_QUOTE,
        &budget,
        &cancelled,
    );
    let result = annotate_citation_check(messages, &inputs).await;
    let content = result[2]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3, "original + line-find + citation compose");
    assert_eq!(
        content[0]["text"].as_str().unwrap(),
        SPAN_WITH_QUOTE,
        "original source intact"
    );
    assert!(content[1]["text"]
        .as_str()
        .unwrap()
        .starts_with("{\"jev_line_find\":"));
    assert!(content[2]["text"]
        .as_str()
        .unwrap()
        .starts_with("{\"jev_citation_check\":"));
    assert_eq!(
        transport.call_count(),
        1,
        "citation assessed the same source"
    );
}

#[tokio::test]
async fn citation_refuses_unrecognised_extra_blocks_and_images() {
    // Strict eligibility: an unrecognised extra text block or an image block
    // keeps the message ineligible — no call, no annotation, original kept.
    for extra in [
        json!({"type": "text", "text": "unrecognised block"}),
        json!({"type": "image", "url": "data:image/png;base64,AAAA"}),
    ] {
        let messages = source_read_messages("call-1", "src/retry.rs", SPAN_WITH_QUOTE);
        let mut messages = messages;
        messages[2]["content"].as_array_mut().unwrap().push(extra);
        let (observer, transport, _temp) = observer(
            JevMode::Active,
            vec![MockStep::Body(citation_answer_body(
                "code_citation_check.relation",
                "supports",
                0.93,
            ))],
        );
        let budget = SearchBudget::from_now(Duration::from_secs(30));
        let cancelled = || false;
        let inputs = inputs(
            &observer,
            JevMode::Active,
            CLAIM_WITH_QUOTE,
            &budget,
            &cancelled,
        );
        let result = annotate_citation_check(messages, &inputs).await;
        assert_eq!(
            result[2]["content"].as_array().unwrap().len(),
            2,
            "original plus the extra block stay"
        );
        assert_eq!(
            transport.call_count(),
            0,
            "strict eligibility refuses before any call"
        );
    }
}
