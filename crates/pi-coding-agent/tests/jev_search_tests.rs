//! Integration tests for the search lane: code-search reranking and line-level
//! find, wired through the REAL observer with the mock transport.
//!
//! ## Landing note
//!
//! The rerank and line-find stages need the shared patches the integrator
//! applies from the search lane's handoff (`PATCHES-SEARCH-v1.md`):
//! - `pi-jev/src/types.rs`: `DecisionCategory::CodeSearchRerank`/`CodeLineFind`
//!   plus the typed `RerankAssessment`/`LineFindAssessment`.
//! - `pi-jev/src/active.rs`: `SCORED_SEARCH_CATEGORIES`.
//! - `pi-jev/src/hooks.rs`: route scored categories through typed acceptance
//!   (a Noul answer is never refused for lacking Choice confidence).
//! - `pi-jev/src/lib.rs`: `pub mod search;`.
//! - `core/mod.rs`: `pub mod jev_line_find;`.
//!
//! Before those land, Noul answers are refused (`MissingConfidence`) and the
//! tests below that assert accepted decisions fail. The integrator runs this
//! suite AFTER landing the shared patches, together with the search lane's
//! owned files. Test discipline: `--exact --test-threads=1`,
//! `DEV_DEBUG=0 TEST_DEBUG=0`, no `PI_*` overrides.

use std::time::{Duration, SystemTime};

use pi_coding_agent::core::jev_code_search::{prepare as prepare_search, recent_code_search_envelope};
use pi_coding_agent::core::jev_line_find::{
    prepare as prepare_line_find, prepare_from_snippet, narrow, LineFindSource,
};
use pi_jev::active::ActivationPolicy;
use pi_jev::hooks::{JevObserver, JevObserverConfig};
use pi_jev::search::{
    rerank_batch_questions, rerank_batch_state, reranked_order, scored_candidates_from_decisions,
    SearchBudget, LINE_WINDOW, MAX_RERANK_CANDIDATES, WHERE_QUESTION_ID,
};
use pi_jev::types::{Answer, DecisionCategory, QuestionSpec};
use pi_jev::{JevLimits, JevMode, JevStats, JevSystemOne, MockJevTransport, MockStep, SecretString};
use serde_json::{json, Value};
use std::sync::Arc;

fn observer(mode: JevMode, steps: Vec<MockStep>) -> (Arc<JevObserver>, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let client = Arc::new(
        JevSystemOne::new(
            mode,
            SecretString::new("synthetic"),
            Arc::new(MockJevTransport::scripted(steps)),
            JevLimits::default(),
            Arc::new(JevStats::default()),
        )
        .unwrap(),
    );
    let observer = JevObserver::new(
        JevObserverConfig {
            mode_gate: Arc::new(move |_| (mode, pi_jev::hooks::SYSTEM_ONE_MODEL.to_string())),
            ..Default::default()
        },
        client,
        temp.path().join("records.jsonl"),
    );
    (observer, temp)
}

fn policy(categories: &[DecisionCategory]) -> ActivationPolicy {
    ActivationPolicy {
        enabled_categories: categories.iter().copied().collect(),
        ..Default::default()
    }
}

fn envelope_message(id: &str, count: usize) -> Vec<Value> {
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

fn source_read_messages(id: &str, path: &str, content: &str) -> Vec<Value> {
    vec![
        json!({"role": "user", "content": [{"type": "text", "text": "where is the token refresh"}]}),
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

fn noul_body(answers: Value) -> MockStep {
    MockStep::Body(json!({"model": "mock", "answers": answers}).to_string())
}

#[tokio::test]
async fn rerank_reorders_candidates_with_stable_ties() {
    let (observer, _temp) = observer(
        JevMode::CompareAndActive,
        vec![noul_body(json!({
            "code_search_rerank.0": {"type": "noul", "noul": 0.10},
            "code_search_rerank.1": {"type": "noul", "noul": 0.95},
            "code_search_rerank.2": {"type": "noul", "noul": 0.30},
            "code_search_rerank.3": {"type": "noul", "noul": 0.55},
            "code_search_rerank.4": {"type": "noul", "noul": 0.80},
            "code_search_rerank.5": {"type": "noul", "noul": 0.42},
        }))],
    );
    let messages = envelope_message("call-1", 6);
    let query = "find the retry policy";
    let plan = prepare_search(&messages, query, &Default::default());
    assert_eq!(plan.len(), 1);
    let inputs = plan[0].rerank_inputs(MAX_RERANK_CANDIDATES, &[]);
    assert_eq!(inputs.len(), 6);
    let excerpts: Vec<String> = inputs.iter().map(|(_, excerpt)| excerpt.clone()).collect();
    let state = rerank_batch_state(query, &excerpts);
    let questions = rerank_batch_questions(&state).unwrap();
    assert_eq!(questions.len(), 6);
    let payload = json!({"session_id": "search-test", "turn": 1, "state": state});
    let policy = policy(&[DecisionCategory::CodeSearchRerank]);
    let outcome = observer
        .decide_prepared(&payload, "code_search_rerank", questions, &policy)
        .await;
    assert!(outcome.unavailable.is_none());
    // Post-patch: the typed scored acceptance admits all six Noul answers.
    assert_eq!(outcome.decisions.len(), 6);
    assert!(outcome.refusals.is_empty());
    assert!(observer.can_apply(&outcome));
    let indices: Vec<usize> = plan[0].scored.iter().map(|(index, _)| *index).collect();
    let scores = scored_candidates_from_decisions(
        &outcome.decisions,
        &indices,
        DecisionCategory::CodeSearchRerank,
    );
    assert_eq!(scores.len(), 6);
    let order = reranked_order(
        &scores,
        6,
        outcome.turn,
        SystemTime::now(),
        policy.max_decision_age,
    )
    .expect("complete fresh correlated set");
    // Descending noul, ties (none here) stable by original ordinal.
    assert_eq!(order, vec![1, 4, 3, 5, 2, 0]);
    let projected = plan[0]
        .project_with_order(&[], Some(&order), Some("mock"), Some("all"))
        .expect("reordered projection");
    let output: Value = serde_json::from_str(projected[0]["text"].as_str().unwrap()).unwrap();
    let paths: Vec<&str> = output["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| candidate["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["src/file1.rs", "src/file4.rs", "src/file3.rs", "src/file5.rs", "src/file2.rs", "src/file0.rs"]);
    assert_eq!(output["jev_reranked"], json!(true));
    assert_eq!(output["jev_rerank_candidates"], json!(6));
    assert_eq!(output["jev_rerank_model"], json!("mock"));
    assert_eq!(output["jev_rerank_scope"], json!("all"));
    // The original tool result in the message list is untouched.
    let original: Value = serde_json::from_str(
        messages[2]["content"][0]["text"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(original["candidates"].as_array().unwrap().len(), 6);
    assert_eq!(original["candidates"][0]["path"], "src/file0.rs");
    assert!(original.get("jev_reranked").is_none());
    // The shared deadline token type exists and carries a single deadline.
    let budget = SearchBudget::from_now(Duration::from_millis(2500));
    assert!(!budget.expired());
    observer.shutdown();
}

#[tokio::test]
async fn line_find_annotates_real_ipython_source_read_with_separate_existence() {
    let content = "pub fn refresh() {\n    let token = renew();\n    token\n}\n";
    let (observer, _temp) = observer(
        JevMode::CompareAndActive,
        vec![noul_body(json!({
            "code_line_find.0": {"type": "choice", "choice": "L0001",
                                  "probabilities": {"L0000": 0.04, "L0001": 0.88, "L0002": 0.06, "L0003": 0.02},
                                  "confidence": 0.9},
            "code_line_find.1": {"type": "noul", "noul": 0.92},
        }))],
    );
    let messages = source_read_messages("call-2", "src/session.rs", content);
    let presentations = prepare_line_find(&messages, "where is the token refresh", &Default::default());
    assert_eq!(presentations.len(), 1);
    let presentation = &presentations[0];
    assert_eq!(presentation.source, LineFindSource::IpythonRead { path: "src/session.rs".to_string() });
    assert_eq!(presentation.text.lines(), 4);
    assert!(!presentation.needs_window_pass());
    let entries = presentation.pass2_entries(None).expect("single-pass entries");
    let (state, questions) = presentation.pass2(None).expect("single-pass questions");
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[0].question_id, "code_line_find.0");
    assert_eq!(questions[1].question_id, "code_line_find.1");
    let payload = json!({"session_id": "line-find-test", "turn": 1, "state": state});
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let outcome = observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert!(outcome.unavailable.is_none());
    assert_eq!(outcome.decisions.len(), 2);
    assert!(outcome.refusals.is_empty());
    let where_decision = outcome
        .decisions
        .iter()
        .find(|decision| decision.question_id == "code_line_find.0")
        .expect("where decision");
    let exists_decision = outcome
        .decisions
        .iter()
        .find(|decision| decision.question_id == "code_line_find.1")
        .expect("existence decision");
    assert_eq!(where_decision.value, "L0001");
    // The existence answer is a SEPARATE Noul, never a Choice confidence.
    assert_eq!(exists_decision.value.parse::<f64>().unwrap(), 0.92);
    let raw = outcome.raw.as_ref().expect("raw outcome for probabilities");
    let where_record = raw
        .records
        .iter()
        .find(|record| record.question_id == "code_line_find.0")
        .expect("where record");
    let Answer::Choice { probabilities, .. } = &where_record.answer else {
        panic!("where answer must be a Choice");
    };
    let request_id = outcome.request_id.clone().unwrap();
    let annotation = presentation
        .annotate(
            where_decision,
            exists_decision,
            Some(probabilities),
            &entries,
            &request_id,
            outcome.turn,
            SystemTime::now(),
            policy.max_decision_age,
            outcome.response_model.as_deref(),
            &Default::default(),
        )
        .expect("typed annotation");
    let block = json!({"type": "text", "text": serde_json::to_string(&annotation).unwrap()});
    assert!(block["text"].as_str().unwrap().len() <= 2048);
    let line_find = &annotation["jev_line_find"];
    assert_eq!(line_find["verdict"], "present");
    assert_eq!(line_find["exists_noul"], 0.92);
    let top = line_find["top_lines"].as_array().unwrap();
    assert_eq!(top[0]["id"], "L0001");
    assert_eq!(top[0]["probability"], 0.88);
    // Ipython-read ids map to offsets in the SUPPLIED output, not file lines.
    assert_eq!(top[0]["offset"], 2);
    assert!(top[0].get("line").is_none());
    assert_eq!(line_find["scope"]["kind"], "ipython_source_read");
    assert_eq!(line_find["scope"]["path"], "src/session.rs");
    assert_eq!(line_find["scope"]["offsets"], "supplied_output");
    assert_eq!(line_find["scope"]["lines"], "L0000-L0003");
    assert_eq!(line_find["disclosures"]["windowed"], false);
    assert_eq!(line_find["disclosures"]["original_lines"], 4);
    assert_eq!(line_find["prompt_version"], "jev-search-prompts/1");
    assert!(line_find["threshold_provenance"].is_string());
    assert!(line_find["notice"].as_str().unwrap().contains("not a repository search"));
    // The annotation is ADDITIVE: the original keeps one block, the copy gets two.
    let mut requests = messages.clone();
    assert!(presentation.attach(&mut requests, block).is_some());
    let original_blocks = messages[2]["content"].as_array().unwrap().len();
    let copy_blocks = requests[2]["content"].as_array().unwrap().len();
    assert_eq!(original_blocks, 1);
    assert_eq!(copy_blocks, 2);
    assert!(requests[2]["content"][1]["text"].as_str().unwrap().contains("jev_line_find"));
    assert_eq!(messages[2]["content"][0]["text"], requests[2]["content"][0]["text"]);
    observer.shutdown();
}

#[tokio::test]
async fn window_cascade_scopes_verdict_labels_to_the_inspected_window() {
    let lines: Vec<String> = (0..600).map(|n| format!("line {n:03}")).collect();
    let content = format!("{}\n", lines.join("\n"));
    // Completeness acceptance: the pass-2 answer must carry a probability for
    // EVERY id in the inspected window (255 lines), summing to ~1.
    let mut window_probabilities = serde_json::Map::new();
    for index in 255..510 {
        let probability = if index == 300 { 0.80 } else { 0.20 / 254.0 };
        window_probabilities.insert(format!("L{index:04}"), json!(probability));
    }
    let (observer, _temp) = observer(
        JevMode::CompareAndActive,
        vec![
            noul_body(json!({"code_line_find.window": {"type": "choice", "choice": "W01",
                "probabilities": {"W00": 0.10, "W01": 0.85, "W02": 0.05}, "confidence": 0.9}})),
            MockStep::Body(
                json!({
                    "model": "mock",
                    "answers": {
                        "code_line_find.0": {"type": "choice", "choice": "L0300",
                            "probabilities": window_probabilities, "confidence": 0.9},
                        "code_line_find.1": {"type": "noul", "noul": 0.5},
                    }
                })
                .to_string(),
            ),
        ],
    );
    let messages = source_read_messages("call-3", "big.rs", &content);
    let presentations = prepare_line_find(&messages, "find line 300", &Default::default());
    assert_eq!(presentations.len(), 1);
    let presentation = &presentations[0];
    assert!(presentation.needs_window_pass());
    let (window_state, window_questions) = presentation.pass1().expect("window pass");
    assert_eq!(window_state["supplied_windows"].as_array().unwrap().len(), 3);
    let payload = json!({"session_id": "cascade-test", "turn": 1, "state": window_state});
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let first = observer
        .decide_prepared(&payload, "code_line_find", window_questions, &policy)
        .await;
    // The lone window record is a typed refusal BY DESIGN (the pair
    // acceptance covers only the pass-2 pair). The bridge consumes the
    // window Choice at set level from the RAW record, after validating a
    // complete finite normalized distribution over the supplied windows.
    assert!(first.decisions.is_empty());
    assert!(observer.can_apply(&first));
    let window_record = first
        .raw
        .as_ref()
        .expect("raw window outcome")
        .records
        .iter()
        .find(|record| record.question_id == "code_line_find.window")
        .expect("window record");
    let Answer::Choice { probabilities, .. } = &window_record.answer else {
        panic!("window answer must be a Choice");
    };
    let ranked = pi_jev::search::validate_window_distribution(probabilities, 3)
        .expect("complete finite normalized window distribution");
    assert_eq!(ranked[0].0, "W01");
    let window = narrow(&window_record.answer.selected_value()).expect("window answer");
    assert_eq!(window, 1);
    // Strict two-digit window ids only; anything else never narrows.
    assert!(narrow("W1").is_none());
    assert!(narrow("W001").is_none());
    assert!(narrow("w01").is_none());
    assert!(narrow("").is_none());
    // Pass 2 judges ONLY the inspected window.
    let entries = presentation.pass2_entries(Some(window)).expect("window entries");
    assert_eq!(entries.window_index, Some(1));
    let (state, questions) = presentation.pass2(Some(window)).expect("window questions");
    // Outgoing-payload honesty: prepare_explicit forwards payload["state"]
    // verbatim into the SystemOneRequest, so these assertions cover the ACTUAL
    // outgoing state. It must measure within the decision-state cap, survive
    // defensive bounding (snapshot::bound_json) byte-identically, show every
    // selectable id with its supplied text, and keep the annotation scope
    // exactly equal to that coverage.
    assert!(serde_json::to_vec(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    assert_eq!(pi_jev::snapshot::bound_json(state.clone(), 0), state);
    assert!(!serde_json::to_string(&state).unwrap().contains("...[truncated]"));
    let chunk_ids: Vec<String> = state["supplied_lines"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|chunk| chunk.as_str().unwrap().split('\n'))
        .map(|line| line.split(' ').next().unwrap().to_string())
        .collect();
    assert_eq!(chunk_ids.len(), LINE_WINDOW);
    assert_eq!(
        chunk_ids,
        entries
            .entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>()
    );
    let where_question = questions
        .iter()
        .find(|question| question.question_id == WHERE_QUESTION_ID)
        .expect("where question");
    let QuestionSpec::Choice { criteria, .. } = &where_question.spec else {
        panic!("where question must be a Choice");
    };
    assert_eq!(criteria.keys().cloned().collect::<Vec<_>>(), chunk_ids);
    let payload = json!({"session_id": "cascade-test", "turn": 2, "state": state});
    let second = observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert_eq!(second.decisions.len(), 2);
    let where_decision = second
        .decisions
        .iter()
        .find(|decision| decision.question_id == "code_line_find.0")
        .unwrap();
    let exists_decision = second
        .decisions
        .iter()
        .find(|decision| decision.question_id == "code_line_find.1")
        .unwrap();
    let raw = second.raw.as_ref().unwrap();
    let where_record = raw
        .records
        .iter()
        .find(|record| record.question_id == "code_line_find.0")
        .unwrap();
    let Answer::Choice { probabilities, .. } = &where_record.answer else {
        panic!("where answer must be a Choice");
    };
    let request_id = second.request_id.clone().unwrap();
    let annotation = presentation
        .annotate(
            where_decision,
            exists_decision,
            Some(probabilities),
            &entries,
            &request_id,
            second.turn,
            SystemTime::now(),
            policy.max_decision_age,
            None,
            &Default::default(),
        )
        .expect("window-scoped annotation");
    let line_find = &annotation["jev_line_find"];
    // 0.5 is the partial band: the label is accurate, not binary.
    assert_eq!(line_find["verdict"], "partially_addressed");
    assert_eq!(line_find["exists_noul"], 0.5);
    assert_eq!(line_find["scope"]["window"], "W01");
    assert_eq!(line_find["scope"]["lines"], "L0255-L0509");
    assert_eq!(line_find["disclosures"]["windowed"], true);
    assert_eq!(line_find["disclosures"]["original_lines"], 600);
    let top = line_find["top_lines"].as_array().unwrap();
    assert_eq!(top[0]["id"], "L0300");
    assert_eq!(top[0]["offset"], 301);
    observer.shutdown();
}

#[tokio::test]
async fn snippet_source_maps_line_ids_to_real_file_lines() {
    let candidate = json!({
        "kind": "file",
        "path": "src/session.rs",
        "line": 41,
        "snippet": "let alpha = 1;\nlet token = renew();\nlet omega = 3;\n",
    });
    let (observer, _temp) = observer(
        JevMode::CompareAndActive,
        vec![noul_body(json!({
            "code_line_find.0": {"type": "choice", "choice": "L0001",
                "probabilities": {"L0000": 0.2, "L0001": 0.7, "L0002": 0.1}, "confidence": 0.9},
            "code_line_find.1": {"type": "noul", "noul": 0.75},
        }))],
    );
    let presentation = prepare_from_snippet(&candidate, 2, "where is token renewed", &Default::default())
        .expect("snippet presentation");
    assert_eq!(presentation.source, LineFindSource::Snippet { path: "src/session.rs".to_string(), start_line: 41 });
    let entries = presentation.pass2_entries(None).expect("snippet entries");
    let (state, questions) = presentation.pass2(None).expect("snippet questions");
    let payload = json!({"session_id": "snippet-test", "turn": 1, "state": state});
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let outcome = observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert_eq!(outcome.decisions.len(), 2);
    let where_decision = outcome.decisions.iter().find(|d| d.question_id == "code_line_find.0").unwrap();
    let exists_decision = outcome.decisions.iter().find(|d| d.question_id == "code_line_find.1").unwrap();
    let raw = outcome.raw.as_ref().unwrap();
    let where_record = raw.records.iter().find(|r| r.question_id == "code_line_find.0").unwrap();
    let Answer::Choice { probabilities, .. } = &where_record.answer else {
        panic!("where answer must be a Choice");
    };
    let annotation = presentation
        .annotate(
            where_decision,
            exists_decision,
            Some(probabilities),
            &entries,
            outcome.request_id.as_deref().unwrap(),
            outcome.turn,
            SystemTime::now(),
            policy.max_decision_age,
            None,
            &Default::default(),
        )
        .expect("snippet annotation");
    let line_find = &annotation["jev_line_find"];
    assert_eq!(line_find["verdict"], "present");
    assert_eq!(line_find["scope"]["kind"], "snippet");
    assert_eq!(line_find["scope"]["offsets"], "file_lines");
    let top = line_find["top_lines"].as_array().unwrap();
    assert_eq!(top[0]["id"], "L0001");
    // Snippet ids map to REAL file lines: entry 1 is file line 42.
    assert_eq!(top[0]["line"], 42);
    assert!(top[0].get("offset").is_none());
    observer.shutdown();
}

#[tokio::test]
async fn stale_uncorrelated_and_cancelled_decisions_never_apply() {
    let content = "alpha\nbeta\n";
    let (observer, _temp) = observer(
        JevMode::CompareAndActive,
        vec![noul_body(json!({
            "code_line_find.0": {"type": "choice", "choice": "L0001",
                "probabilities": {"L0000": 0.3, "L0001": 0.7}, "confidence": 0.9},
            "code_line_find.1": {"type": "noul", "noul": 0.9},
        }))],
    );
    let messages = source_read_messages("call-4", "src/a.rs", content);
    let presentations = prepare_line_find(&messages, "where is beta", &Default::default());
    let presentation = &presentations[0];
    let entries = presentation.pass2_entries(None).unwrap();
    let (state, questions) = presentation.pass2(None).unwrap();
    let payload = json!({"session_id": "stale-test", "turn": 1, "state": state});
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let outcome = observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert_eq!(outcome.decisions.len(), 2);
    let where_decision = outcome.decisions.iter().find(|d| d.question_id == "code_line_find.0").unwrap();
    let exists_decision = outcome.decisions.iter().find(|d| d.question_id == "code_line_find.1").unwrap();
    let raw = outcome.raw.as_ref().unwrap();
    let where_record = raw.records.iter().find(|r| r.question_id == "code_line_find.0").unwrap();
    let Answer::Choice { probabilities, .. } = &where_record.answer else {
        panic!("where answer must be a Choice");
    };
    let request_id = outcome.request_id.clone().unwrap();
    // Stale: decided_at is beyond the age bound from a later `now`.
    let stale = presentation.annotate(
        where_decision,
        exists_decision,
        Some(probabilities),
        &entries,
        &request_id,
        outcome.turn,
        SystemTime::now() + Duration::from_secs(30),
        policy.max_decision_age,
        None,
        &Default::default(),
    );
    assert!(stale.is_none());
    // Uncorrelated: a different request id never annotates.
    let uncorrelated = presentation.annotate(
        where_decision,
        exists_decision,
        Some(probabilities),
        &entries,
        "some-other-request",
        outcome.turn,
        SystemTime::now(),
        policy.max_decision_age,
        None,
        &Default::default(),
    );
    assert!(uncorrelated.is_none());
    // Cancelled: the observer gate refuses, so the wiring must not apply.
    observer.cancel_decisions("stale-test");
    assert!(!observer.can_apply(&outcome));
    observer.shutdown();
}

#[tokio::test]
async fn deadline_and_mode_failures_leave_the_request_untouched() {
    let content = "alpha\nbeta\n";
    // Deadline exhausted before dispatch: nothing is asked, nothing changes.
    // Both bindings are renamed (deadline_observer / compare_observer): a
    // plain `observer` binding would shadow the helper fn, and the SECOND
    // helper call would then resolve to the struct, not the function.
    let (deadline_observer, _temp) = observer(JevMode::CompareAndActive, vec![noul_body(json!({}))]);
    let messages = source_read_messages("call-5", "src/a.rs", content);
    let presentations = prepare_line_find(&messages, "where is beta", &Default::default());
    let presentation = &presentations[0];
    let (state, questions) = presentation.pass2(None).unwrap();
    let mut exhausted = json!({"session_id": "deadline-test", "turn": 1, "state": state});
    exhausted["decision_timeout_ms"] = json!(0);
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let outcome = deadline_observer
        .decide_prepared(&exhausted, "code_line_find", questions, &policy)
        .await;
    assert!(!outcome.dispatched);
    assert!(outcome.decisions.is_empty());
    assert_eq!(outcome.terminal_reason.as_deref(), Some("deadline_exhausted"));
    assert!(presentation
        .pass2(None)
        .is_some());
    assert!(outcome.raw.is_none() || outcome.raw.as_ref().unwrap().records.is_empty());
    deadline_observer.shutdown();
    // Compare mode refuses everything from Jev: typed acceptance never runs,
    // and the wiring must leave the request byte-identical.
    let (compare_observer, _temp2) = observer(JevMode::Compare, vec![noul_body(json!({
        "code_line_find.0": {"type": "choice", "choice": "L0001",
            "probabilities": {"L0000": 0.3, "L0001": 0.7}, "confidence": 0.9},
        "code_line_find.1": {"type": "noul", "noul": 0.9},
    }))]);
    // Reusing `presentation` is intended: the Compare half changes only the
    // OBSERVER (Compare mode refuses), while the presentation, state and
    // questions are deterministic pure functions of the same supplied text,
    // query and options.
    let (state, questions) = presentation.pass2(None).unwrap();
    let payload = json!({"session_id": "compare-test", "turn": 1, "state": state});
    let refused = compare_observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert!(refused.decisions.is_empty());
    assert!(refused.unavailable.is_some());
    // No accepted decisions means no order and no annotation: fail open.
    let indices: Vec<usize> = vec![0, 1];
    let scores = scored_candidates_from_decisions(&refused.decisions, &indices, DecisionCategory::CodeSearchRerank);
    assert!(scores.is_empty());
    assert!(reranked_order(&scores, 2, 1, SystemTime::now(), policy.max_decision_age).is_none());
    compare_observer.shutdown();
}

#[test]
fn current_filtered_or_reranked_envelope_stays_a_reachable_snippet_source() {
    // ROOT CONTRACT v1: line-find judges the CURRENT (filtered+reranked)
    // supplied text. The projection adds disclosure keys to the request-copy
    // envelope, so the locator must not require the pristine two-key shape,
    // and the top file candidate must still feed `prepare_from_snippet`.
    let mut messages = envelope_message("call-6", 4);
    let original = messages[2]["content"][0]["text"].as_str().unwrap().to_string();
    // Pristine envelope is reachable.
    let (index, envelope) = recent_code_search_envelope(&messages).expect("pristine");
    assert_eq!(index, 2);
    assert_eq!(envelope["candidates"][0]["path"], "src/file0.rs");
    let candidate = envelope["candidates"][0].clone();
    assert!(prepare_from_snippet(&candidate, index, "find the retry policy", &Default::default()).is_some());
    // Projected (reranked) request-copy envelope is reachable too.
    let mut data: Value = serde_json::from_str(&original).unwrap();
    data["jev_reranked"] = json!(true);
    data["jev_rerank_candidates"] = json!(4);
    data["jev_rerank_scope"] = json!("all");
    data["jev_rerank_notice"] = json!("Optional candidates reordered for this request");
    data["jev_notice"] = json!("Optional candidates omitted for this request");
    messages[2]["content"][0]["text"] = json!(data.to_string());
    let (index, envelope) = recent_code_search_envelope(&messages).expect("projected");
    assert_eq!(index, 2);
    let candidate = envelope["candidates"][0].clone();
    let presentation =
        prepare_from_snippet(&candidate, index, "find the retry policy", &Default::default())
            .expect("snippet presentation from the current envelope");
    assert_eq!(
        presentation.source,
        LineFindSource::Snippet { path: "src/file0.rs".to_string(), start_line: 1 }
    );
}

#[tokio::test]
async fn stale_pre_toggle_decision_is_rejected_at_application_not_only_by_stamp() {
    // A decision accepted under one feature/overlay policy generation must
    // not apply after a toggle (e.g. full-jev off->on) changes the
    // generation: can_apply is the actual application gate and must re-reject.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let generation = Arc::new(AtomicUsize::new(0));
    let gen_for_gate = generation.clone();
    let temp = tempfile::tempdir().unwrap();
    let mode = JevMode::CompareAndActive;
    let client = Arc::new(
        JevSystemOne::new(
            mode,
            SecretString::new("synthetic"),
            Arc::new(MockJevTransport::scripted(vec![noul_body(json!({
                "code_line_find.0": {"type": "choice", "choice": "L0001",
                                      "probabilities": {"L0000": 0.3, "L0001": 0.7}, "confidence": 0.9},
                "code_line_find.1": {"type": "noul", "noul": 0.9},
            }))])),
            JevLimits::default(),
            Arc::new(JevStats::default()),
        )
        .unwrap(),
    );
    let observer = JevObserver::new(
        JevObserverConfig {
            mode_gate: Arc::new(move |_| (mode, pi_jev::hooks::SYSTEM_ONE_MODEL.to_string())),
            policy_generation: Arc::new(move |_, _| {
                format!("search-feature-gen-{}", gen_for_gate.load(Ordering::SeqCst))
            }),
            ..Default::default()
        },
        client,
        temp.path().join("records.jsonl"),
    );
    let messages = source_read_messages("call-7", "src/a.rs", "alpha\nbeta\n");
    let presentation = &prepare_line_find(&messages, "where is beta", &Default::default())[0];
    let (state, questions) = presentation.pass2(None).unwrap();
    let payload = json!({"session_id": "gen-test", "turn": 1, "state": state});
    let policy = policy(&[DecisionCategory::CodeLineFind]);
    let outcome = observer
        .decide_prepared(&payload, "code_line_find", questions, &policy)
        .await;
    assert_eq!(outcome.decisions.len(), 2);
    assert!(observer.can_apply(&outcome));
    // The toggle happens (generation moves off->on): the stamp now differs,
    // AND the held decision is rejected at the application gate itself.
    generation.store(1, Ordering::SeqCst);
    assert!(!observer.can_apply(&outcome));
    observer.shutdown();
}
