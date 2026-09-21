//! Unit tests for `pi_jev::search`: rerank scoring and line-level find.
//!
//! These tests cover the PURE module only (no transport, no host). They
//! compile and pass only after the integrator lands the shared patches:
//! `types.rs` gains `DecisionCategory::CodeSearchRerank`/`CodeLineFind`,
//! `lib.rs` gains `pub mod search;`. The tests are shipped with the search
//! lane's owned files and are part of the landed batch.
//!
//! Test discipline: `--exact --test-threads=1`, `DEV_DEBUG=0 TEST_DEBUG=0`,
//! no `PI_*` overrides. One cargo at a time (the integrator owns the build).

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use pi_jev::active::ActiveDecision;
use pi_jev::search::{
    build_line_find_text, fitting_window_state, line_find_questions, line_id, line_state,
    rerank_batch_questions, rerank_batch_state, reranked_order, scored_candidates_from_decisions,
    top_lines, validate_where_distribution, validate_window_distribution, verdict,
    window_questions, window_state, LineFindOptions, LineFindVerdict, RerankOptions, SearchBudget,
    LINE_STATE_CHUNK_CHARS, LINE_STATE_FIXED_BYTES, LINE_STATE_MAX_CHUNKS,
    EXISTS_ABSENT, EXISTS_FOUND, LINE_EXCERPT_CHARS, LINE_FIND_CATEGORY, LINE_WINDOW,
    MAX_LINE_FIND_LINES, MAX_LINE_FIND_WINDOWS, MAX_LINE_STATE_BYTES, MAX_RERANK_CANDIDATES,
    RERANK_BATCH, RERANK_CATEGORY, WHERE_QUESTION_ID, WINDOW_QUESTION_ID, PROMPT_VERSION_SEARCH,
};
use pi_jev::types::{DecisionCategory, QuestionSpec};

fn scored(
    count: usize,
    noul: f64,
    request_id: &str,
    turn: u64,
    now: SystemTime,
) -> Vec<pi_jev::search::ScoredCandidate> {
    // `decided_at` MUST be the caller's reference `now`, not a fresh
    // SystemTime::now(): a stamp later than the comparison instant is a
    // future stamp, which `reranked_order` correctly refuses (freshness is
    // `now.duration_since(decided_at)`).
    (0..count)
        .map(|candidate| pi_jev::search::ScoredCandidate {
            candidate,
            noul,
            request_id: request_id.to_string(),
            turn,
            decided_at: now,
        })
        .collect()
}

#[test]
fn rerank_batch_state_and_questions_shape() {
    let excerpts = vec![
        "file src/a.rs line 1 body".to_string(),
        "file src/b.rs line 9 body".to_string(),
    ];
    let state = rerank_batch_state("Find the session expiry", &excerpts);
    let candidates = state["code_search_candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0]["id"], "0");
    assert_eq!(candidates[1]["id"], "1");
    let questions = rerank_batch_questions(&state).unwrap();
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[0].question_id, format!("{RERANK_CATEGORY}.0"));
    assert_eq!(questions[1].question_id, format!("{RERANK_CATEGORY}.1"));
    let rendered = serde_json::to_string(&questions[0].spec).unwrap();
    assert!(rendered.contains("untrusted data"));
    assert!(rendered.contains("Find the session expiry"));
    // The 0.9 Choice-confidence threshold is never transplanted here.
    assert!(!rendered.contains("0.9"));
    // Batches above the decision bound are ineligible: empty questions, fail open.
    let oversized = rerank_batch_state("q", &vec!["e".to_string(); RERANK_BATCH + 1]);
    assert!(rerank_batch_questions(&oversized).unwrap().is_empty());
    // An empty task excerpt is ineligible too.
    let empty_task = serde_json::json!({"user_text_excerpt": "", "code_search_candidates": [{"id":"0","excerpt":"x"}]});
    assert!(rerank_batch_questions(&empty_task).unwrap().is_empty());
    // Credentials in excerpts are redacted before they reach the wire.
    let secret = rerank_batch_state("q", &["token sk-1234567890abcdef live".to_string()]);
    let questions = rerank_batch_questions(&secret).unwrap();
    let rendered = serde_json::to_string(&questions[0].spec).unwrap();
    assert!(!rendered.contains("sk-1234567890abcdef"));
}

#[test]
fn reranked_order_accepts_only_complete_fresh_correlated_sets() {
    let now = SystemTime::now();
    let age = Duration::from_secs(1);
    let mut scores = scored(3, 0.5, "req-1", 4, now);
    scores[0].noul = 0.2;
    scores[1].noul = 0.9;
    scores[2].noul = 0.2;
    let order = reranked_order(&scores, 3, 4, now, age).unwrap();
    assert_eq!(order, vec![1, 0, 2]);
    // Missing a candidate: incomplete -> fail open.
    assert!(reranked_order(&scores[..2], 3, 4, now, age).is_none());
    // Duplicate scores for one candidate -> fail open.
    let mut duplicate = scored(2, 0.5, "req-1", 4, now);
    duplicate.push(duplicate[0].clone());
    assert!(reranked_order(&duplicate, 2, 4, now, age).is_none());
    // Wrong turn: not correlated. (Per-request correlation is enforced at
    // extraction time; the set level re-verifies one shared turn.)
    assert!(reranked_order(&scores, 3, 5, now, age).is_none());
    // Stale beyond the age bound.
    let mut stale = scored(2, 0.5, "req-1", 4, now);
    stale[0].decided_at = SystemTime::now() - Duration::from_secs(30);
    assert!(reranked_order(&stale, 2, 4, now, Duration::from_secs(3)).is_none());
    // A decision stamped in the future is not usable evidence either.
    let mut future = scored(2, 0.5, "req-1", 4, now);
    future[0].decided_at = SystemTime::now() + Duration::from_secs(30);
    assert!(reranked_order(&future, 2, 4, now, age).is_none());
    // Non-finite or out-of-range scores never reorder anything.
    let mut nan = scored(2, 0.5, "req-1", 4, now);
    nan[0].noul = f64::NAN;
    assert!(reranked_order(&nan, 2, 4, now, age).is_none());
    let mut high = scored(2, 0.5, "req-1", 4, now);
    high[0].noul = 1.5;
    assert!(reranked_order(&high, 2, 4, now, age).is_none());
    // Bounds on count.
    assert!(reranked_order(&scores, 0, 4, now, age).is_none());
    assert!(reranked_order(&scored(3, 0.5, "req-1", 4, now), MAX_RERANK_CANDIDATES + 1, 4, now, age).is_none());
    // Stable ties: equal scores keep the original ordinal order.
    let tie = scored(3, 0.5, "req-1", 4, now);
    assert_eq!(reranked_order(&tie, 3, 4, now, age).unwrap(), vec![0, 1, 2]);
}

#[test]
fn scored_candidates_from_decisions_maps_local_ordinals() {
    let make = |ordinal: usize, value: &str| ActiveDecision {
        category: DecisionCategory::CodeSearchRerank,
        question_id: format!("{RERANK_CATEGORY}.{ordinal}"),
        value: value.to_string(),
        confidence: 0.0,
        response_model: None,
        request_id: "req-9".to_string(),
        turn: 2,
        decided_at: SystemTime::now(),
    };
    let decisions = vec![
        make(0, "0.87"),
        make(1, "0.12"),
        make(2, "not-a-number"),
        ActiveDecision {
            category: DecisionCategory::CodeSearchRelevance,
            question_id: "code_search_relevance.0".to_string(),
            value: "keep".to_string(),
            confidence: 0.9,
            response_model: None,
            request_id: "req-9".to_string(),
            turn: 2,
            decided_at: SystemTime::now(),
        },
    ];
    let scores = scored_candidates_from_decisions(&decisions, &[7, 11], DecisionCategory::CodeSearchRerank);
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].candidate, 7);
    assert_eq!(scores[0].noul, 0.87);
    assert_eq!(scores[1].candidate, 11);
    assert_eq!(scores[1].noul, 0.12);
}

#[test]
fn line_text_caps_truncation_and_ids() {
    let model = build_line_find_text("alpha\nbeta\ngamma\n").unwrap();
    assert_eq!(model.lines(), 3);
    assert_eq!(model.entries[0].id, "L0000");
    assert_eq!(model.entries[2].text, "gamma");
    assert!(!model.any_truncated());
    // CRLF endings and a single trailing newline artifact.
    let crlf = build_line_find_text("alpha\r\nbeta\r\n").unwrap();
    assert_eq!(crlf.entries[1].text, "beta");
    // Long lines are truncated at the excerpt bound, never silently.
    let long = "x".repeat(LINE_EXCERPT_CHARS + 50);
    let model = build_line_find_text(&format!("{long}\nshort\n")).unwrap();
    assert!(model.entries[0].truncated);
    assert!(model.any_truncated());
    assert!(!model.entries[1].truncated);
    assert_eq!(model.entries[0].text.chars().count(), LINE_EXCERPT_CHARS);
    // Empty text and over-limit texts are ineligible (never silently trimmed).
    assert!(build_line_find_text("").is_none());
    let big = "l\n".repeat(MAX_LINE_FIND_LINES + 1);
    assert!(build_line_find_text(&big).is_none());
    let max = "l\n".repeat(MAX_LINE_FIND_LINES);
    assert!(build_line_find_text(&max).is_some());
}

#[test]
fn request_entries_window_and_byte_caps() {
    // Whole text within one Choice window.
    let small = build_line_find_text(&"a\n".repeat(LINE_WINDOW)).unwrap();
    let whole = small.request_entries(None).unwrap();
    assert_eq!(whole.entries.len(), LINE_WINDOW);
    assert_eq!(whole.window_index, None);
    // Whole-text requests for multi-window texts are rejected: cascade only.
    let big = build_line_find_text(&"a\n".repeat(LINE_WINDOW + 1)).unwrap();
    assert!(big.request_entries(None).is_none());
    let window = big.request_entries(Some(0)).unwrap();
    assert_eq!(window.entries.len(), LINE_WINDOW);
    assert_eq!(window.window_index, Some(0));
    assert!(!window.bytes_capped);
    assert!(big.request_entries(Some(big.windows())).is_none());
    // The serialized state stays within the decision-state cap (enforced by
    // hooks::prepare_explicit on serde_json bytes).
    let state = line_state("q", &window.entries);
    assert!(serde_json::to_string(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    // Byte cap: measured on the EXACT JSON-escaped contribution of every line
    // plus its chunk overhead. 240 crab chars serialize to 960 bytes, so a
    // 255-line window of them exceeds the share: the maximal fitting prefix
    // is supplied and the dropped lines are disclosed (bytes_capped).
    let fat_line = "\u{1F980}".repeat(LINE_EXCERPT_CHARS);
    assert_eq!(fat_line.len(), LINE_EXCERPT_CHARS * 4);
    let fat = build_line_find_text(&format!("{fat_line}\n").repeat(LINE_WINDOW)).unwrap();
    assert_eq!(fat.lines(), LINE_WINDOW);
    let capped = fat.request_entries(Some(0)).unwrap();
    assert!(capped.bytes_capped);
    assert!(capped.entries.len() < LINE_WINDOW);
    assert!(!capped.entries.is_empty());
    // Line/chunk-boundary cap: every kept line fits the share and the next
    // dropped line would have exceeded it (a 246-char crab line needs its own
    // chunk, still within the 400-char chunk bound).
    let line_cost = |entry: &pi_jev::search::LineEntry| {
        serde_json::to_string(&entry.text).unwrap().len() - 2 + entry.id.len() + 1 + 2 + 3
    };
    let used: usize = LINE_STATE_FIXED_BYTES
        + capped.entries.iter().map(line_cost).sum::<usize>();
    assert!(used <= MAX_LINE_STATE_BYTES);
    let next = &fat.entries[capped.entries.len()];
    assert!(used + line_cost(next) > MAX_LINE_STATE_BYTES);
    // And the capped state still serializes within the decision-state cap.
    let state = line_state("q", &capped.entries);
    assert!(serde_json::to_string(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    // Measured-fit guarantee: the state actually SENT (built from the fitted
    // supply with the real query) measures within the cap; the seed only
    // chooses the supply, measurement proves the fit.
    let (fitted, state) = fat.fitting_line_request("q", Some(0)).unwrap();
    assert_eq!(fitted.entries.len(), capped.entries.len());
    assert!(fitted.bytes_capped);
    assert!(serde_json::to_vec(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    assert_eq!(pi_jev::snapshot::bound_json(state.clone(), 0), state);
}

#[test]
fn line_questions_carry_ids_and_existence_noul() {
    let model = build_line_find_text("alpha\nbeta\n").unwrap();
    let entries = model.request_entries(None).unwrap();
    let state = line_state("Where is beta?", &entries.entries);
    // The compact chunked encoding: whole tagged lines joined by newlines,
    // packed within the snapshot string bound.
    let chunks = state["supplied_lines"].as_array().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_str().unwrap(), "L0000 alpha\nL0001 beta");
    let questions = line_find_questions("Where is beta?", &entries.entries).unwrap();
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[0].question_id, WHERE_QUESTION_ID);
    assert_eq!(questions[1].question_id, format!("{LINE_FIND_CATEGORY}.1"));
    let rendered = serde_json::to_string(&questions[0].spec).unwrap();
    assert!(rendered.contains("L0000"));
    assert!(rendered.contains("L0001"));
    assert!(serde_json::to_string(&questions[1].spec).unwrap().contains("untrusted data"));
    // Empty query or too many entries: no questions.
    assert!(line_find_questions("", &entries.entries).is_none());
    let many = build_line_find_text(&"a\n".repeat(LINE_WINDOW + 1)).unwrap();
    assert!(line_find_questions("q", &many.entries).is_none());
}

#[test]
fn line_state_chunks_stay_within_snapshot_bounds() {
    // hooks defensively runs the state through snapshot::bound_json, which
    // truncates strings above MAX_TEXT_CHARS and arrays above MAX_ITEMS. The
    // chunked supplied-lines encoding must stay within BOTH bounds so no
    // supplied line is hidden while its id is still in the criteria.
    let model = build_line_find_text(&"a\n".repeat(LINE_WINDOW)).unwrap();
    let entries = model.request_entries(None).unwrap();
    assert_eq!(entries.entries.len(), LINE_WINDOW);
    assert!(!entries.bytes_capped);
    let state = line_state("q", &entries.entries);
    let chunks = state["supplied_lines"].as_array().unwrap();
    assert!(!chunks.is_empty());
    assert!(chunks.len() <= LINE_STATE_MAX_CHUNKS);
    assert!(chunks
        .iter()
        .all(|chunk| chunk.as_str().unwrap().chars().count() <= LINE_STATE_CHUNK_CHARS));
    assert!(serde_json::to_string(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    // Ids stay inline in the chunks and in the same order as the entries.
    let joined = chunks.iter().map(|chunk| chunk.as_str().unwrap()).collect::<Vec<_>>().join("\n");
    for (position, entry) in entries.entries.iter().enumerate() {
        let expected = format!("{} {}", entry.id, entry.text);
        let at = joined.find(&expected).expect("line present in order");
        if position > 0 {
            let previous = format!("{} {}", entries.entries[position - 1].id, entries.entries[position - 1].text);
            assert!(joined.find(&previous).unwrap() < at);
        }
    }
}

#[test]
fn line_state_survives_actual_enforcement_and_bounding() {
    // Max-size multibyte query: bounded_excerpt caps CHARS, so 240 four-byte
    // chars are the worst serialized query (960 bytes; multibyte never needs
    // JSON escaping).
    let query = "\u{1F980}".repeat(LINE_EXCERPT_CHARS);
    let model = build_line_find_text(&"a\n".repeat(LINE_WINDOW)).unwrap();
    let (entries, state) = model.fitting_line_request(&query, None).unwrap();
    assert_eq!(entries.entries.len(), LINE_WINDOW);
    assert!(!entries.bytes_capped);
    // The verbatim hooks::prepare_explicit check on the outgoing state.
    assert!(serde_json::to_vec(&state).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    // Bound stability: bound_json (applied defensively in some dispatch
    // paths) leaves the outgoing state unchanged — no string or array is
    // truncated, so no supplied line is ever hidden from the model.
    assert_eq!(pi_jev::snapshot::bound_json(state.clone(), 0), state);
    // Every selectable id maps to supplied text: the WHERE Choice criteria
    // keys are exactly the ids visible in the chunks, in entry order.
    let questions = line_find_questions(&query, &entries.entries).unwrap();
    let where_question = questions
        .iter()
        .find(|question| question.question_id == WHERE_QUESTION_ID)
        .unwrap();
    let criteria = match &where_question.spec {
        QuestionSpec::Choice { criteria, .. } => criteria.keys().cloned().collect::<Vec<_>>(),
        other => panic!("unexpected spec: {other:?}"),
    };
    let chunks = state["supplied_lines"].as_array().unwrap();
    let chunk_ids: Vec<String> = chunks
        .iter()
        .flat_map(|chunk| chunk.as_str().unwrap().split('\n'))
        .map(|line| line.split(' ').next().unwrap().to_string())
        .collect();
    assert_eq!(criteria, chunk_ids);
    assert_eq!(
        chunk_ids,
        entries
            .entries
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>()
    );
    // Encoder limits hold on the actual state; no truncation marker exists.
    assert!(chunks.len() <= LINE_STATE_MAX_CHUNKS);
    assert!(chunks
        .iter()
        .all(|chunk| chunk.as_str().unwrap().chars().count() <= LINE_STATE_CHUNK_CHARS));
    assert!(!serde_json::to_string(&state).unwrap().contains("...[truncated]"));
}

#[test]
fn fitting_window_state_measures_against_the_state_cap() {
    // Worst case: a full 16-window text whose leading lines are max-size
    // multibyte previews. The raw 3-preview state exceeds the decision state
    // cap; the fitted state shrinks previews UNIFORMLY and never drops a
    // window the criteria ask about.
    let preview_line = "\u{1F980}".repeat(80);
    let mut lines: Vec<String> = Vec::new();
    for _window in 0..MAX_LINE_FIND_WINDOWS {
        lines.push(preview_line.clone());
        lines.push(preview_line.clone());
        lines.push(preview_line.clone());
        for _ in 0..LINE_WINDOW - 3 {
            lines.push("filler".to_string());
        }
    }
    let model = build_line_find_text(&format!("{}\n", lines.join("\n"))).unwrap();
    assert_eq!(model.windows(), MAX_LINE_FIND_WINDOWS);
    let raw = window_state("q", &model);
    assert!(serde_json::to_vec(&raw).unwrap().len() > pi_jev::snapshot::MAX_STATE_BYTES);
    let fitted = fitting_window_state("q", &model).expect("fitted window state");
    assert!(serde_json::to_vec(&fitted).unwrap().len() <= pi_jev::snapshot::MAX_STATE_BYTES);
    assert_eq!(pi_jev::snapshot::bound_json(fitted.clone(), 0), fitted);
    // The window set is intact and matches the question criteria exactly.
    let windows = fitted["supplied_windows"].as_array().unwrap();
    assert_eq!(windows.len(), MAX_LINE_FIND_WINDOWS);
    let questions = window_questions("q", MAX_LINE_FIND_WINDOWS).unwrap();
    let criteria = match &questions[0].spec {
        QuestionSpec::Choice { criteria, .. } => criteria.keys().cloned().collect::<Vec<_>>(),
        other => panic!("unexpected spec: {other:?}"),
    };
    let ids: Vec<String> = windows
        .iter()
        .map(|window| window["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(criteria, ids);
    // Previews shrank uniformly (never per-window cherry-picking), and every
    // window keeps its id and line span.
    let preview_lengths: std::collections::BTreeSet<usize> = windows
        .iter()
        .map(|window| window["preview"].as_array().unwrap().len())
        .collect();
    assert_eq!(preview_lengths.len(), 1);
    for (index, window) in windows.iter().enumerate() {
        assert_eq!(window["id"].as_str().unwrap(), format!("W{index:02}"));
        assert!(window["lines"].as_str().unwrap().contains('-'));
    }
}

#[test]
fn window_cascade_state_and_questions() {
    let model = build_line_find_text(&"l\n".repeat(LINE_WINDOW * 2 + 5)).unwrap();
    let state = window_state("find the symbol", &model);
    let windows = state["supplied_windows"].as_array().unwrap();
    assert_eq!(windows.len(), 3);
    assert_eq!(windows[0]["id"], "W00");
    assert_eq!(windows[2]["id"], "W02");
    assert!(windows[0]["preview"].as_array().unwrap().len() <= 3);
    let questions = window_questions("find the symbol", 3).unwrap();
    assert_eq!(questions.len(), 1);
    assert_eq!(questions[0].question_id, WINDOW_QUESTION_ID);
    let rendered = serde_json::to_string(&questions[0].spec).unwrap();
    assert!(rendered.contains("W00") && rendered.contains("W02"));
    // Zero or too many windows: no questions.
    assert!(window_questions("q", 0).is_none());
    assert!(window_questions("q", 17).is_none());
    assert!(window_questions("", 3).is_none());
}

#[test]
fn where_distribution_validation() {
    let model = build_line_find_text("alpha\nbeta\n").unwrap();
    let entries = model.request_entries(None).unwrap().entries;
    let complete = BTreeMap::from([("L0000".to_string(), 0.25), ("L0001".to_string(), 0.75)]);
    let ranked = validate_where_distribution(&complete, &entries).unwrap();
    assert_eq!(ranked, vec![("L0001".to_string(), 0.75), ("L0000".to_string(), 0.25)]);
    assert_eq!(top_lines(&ranked, 1), vec![("L0001".to_string(), 0.75)]);
    // Incomplete (missing id), unknown id, and off-sum distributions fail open.
    let missing = BTreeMap::from([("L0000".to_string(), 1.0)]);
    assert!(validate_where_distribution(&missing, &entries).is_none());
    let unknown = BTreeMap::from([
        ("L0000".to_string(), 0.5),
        ("L0001".to_string(), 0.4),
        ("L0002".to_string(), 0.1),
    ]);
    assert!(validate_where_distribution(&unknown, &entries).is_none());
    let off_sum = BTreeMap::from([("L0000".to_string(), 0.5), ("L0001".to_string(), 0.2)]);
    assert!(validate_where_distribution(&off_sum, &entries).is_none());
    let nan = BTreeMap::from([("L0000".to_string(), f64::NAN), ("L0001".to_string(), 1.0)]);
    assert!(validate_where_distribution(&nan, &entries).is_none());
}

#[test]
fn verdict_thresholds_apply_to_existence_noul_only() {
    assert_eq!(verdict(0.92), Some(LineFindVerdict::Present));
    assert_eq!(verdict(EXISTS_FOUND), Some(LineFindVerdict::Present));
    assert_eq!(verdict(0.5), Some(LineFindVerdict::PartiallyAddressed));
    assert_eq!(verdict(0.30), Some(LineFindVerdict::NotPresent));
    // The absent boundary itself is the partial band (strictly below 0.35).
    assert_eq!(verdict(EXISTS_ABSENT), Some(LineFindVerdict::PartiallyAddressed));
    assert_eq!(verdict(0.349), Some(LineFindVerdict::NotPresent));
    assert!(verdict(f64::NAN).is_none());
    assert!(verdict(1.5).is_none());
    assert!(verdict(-0.1).is_none());
    assert_eq!(LineFindVerdict::Present.as_str(), "present");
    assert_eq!(LineFindVerdict::parse("partially_addressed"), Some(LineFindVerdict::PartiallyAddressed));
    assert_eq!(LineFindVerdict::parse("nope"), None);
}

#[test]
fn options_and_budget() {
    assert!(RerankOptions::default().validate().is_ok());
    assert!(RerankOptions { max_candidates: 0 }.validate().is_err());
    assert!(RerankOptions { max_candidates: MAX_RERANK_CANDIDATES + 1 }.validate().is_err());
    assert!(LineFindOptions::default().validate().is_ok());
    assert!(LineFindOptions { top_lines: 0 }.validate().is_err());
    assert!(LineFindOptions { top_lines: 9 }.validate().is_err());
    let budget = SearchBudget::from_now(Duration::from_millis(50));
    assert!(!budget.expired());
    assert!(budget.remaining_ms() <= 50);
    let gone = SearchBudget::from_now(Duration::from_millis(0));
    assert!(gone.expired());
    assert_eq!(gone.remaining_ms(), 0);
    // Prompt provenance is pinned and observable.
    assert_eq!(PROMPT_VERSION_SEARCH, "jev-search-prompts/1");
}

#[test]
fn window_distribution_validation_completeness_and_bounds() {
    // Complete, finite, normalized over exactly W00..W02.
    let complete = BTreeMap::from([
        ("W00".to_string(), 0.10),
        ("W01".to_string(), 0.85),
        ("W02".to_string(), 0.05),
    ]);
    let ranked = validate_window_distribution(&complete, 3).unwrap();
    assert_eq!(ranked, vec![("W01".to_string(), 0.85), ("W00".to_string(), 0.10), ("W02".to_string(), 0.05)]);
    // Missing a window id: incomplete -> fail open.
    let missing = BTreeMap::from([("W00".to_string(), 1.0)]);
    assert!(validate_window_distribution(&missing, 2).is_none());
    // Unknown window id: not the supplied set -> fail open.
    let unknown = BTreeMap::from([
        ("W00".to_string(), 0.5),
        ("W01".to_string(), 0.5),
        ("W09".to_string(), 0.0),
    ]);
    assert!(validate_window_distribution(&unknown, 2).is_none());
    // Off-sum and non-finite distributions fail open.
    let off_sum = BTreeMap::from([("W00".to_string(), 0.5), ("W01".to_string(), 0.2)]);
    assert!(validate_window_distribution(&off_sum, 2).is_none());
    let nan = BTreeMap::from([("W00".to_string(), f64::NAN), ("W01".to_string(), 1.0)]);
    assert!(validate_window_distribution(&nan, 2).is_none());
    // Bounds: zero windows and beyond the cascade bound.
    assert!(validate_window_distribution(&complete, 0).is_none());
    let too_many: BTreeMap<String, f64> =
        (0..17).map(|w| (pi_jev::search::window_id(w), 1.0 / 17.0)).collect();
    assert!(validate_window_distribution(&too_many, 17).is_none());
    // Stable ties: equal probabilities keep window-id order.
    let ties = BTreeMap::from([("W00".to_string(), 0.5), ("W01".to_string(), 0.5)]);
    let ranked = validate_window_distribution(&ties, 2).unwrap();
    assert_eq!(ranked, vec![("W00".to_string(), 0.5), ("W01".to_string(), 0.5)]);
}

#[test]
fn id_formats_are_pinned() {
    assert_eq!(line_id(0), "L0000");
    assert_eq!(line_id(4079), "L4079");
    assert_eq!(pi_jev::search::window_id(0), "W00");
    assert_eq!(pi_jev::search::window_id(15), "W15");
}
