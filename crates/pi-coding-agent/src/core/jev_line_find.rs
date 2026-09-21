//! Line-level semantic find over explicitly supplied text (ROOT CONTRACT v1).
//!
//! Reachability: this runtime has no `read`/`read_file` tools; source is read
//! through the ipython REPL. The primary input form is therefore the
//! deterministic ipython source-read idiom (`print(open("path").read())` and
//! `print(Path("path").read_text())`), whose stdout channel carries the raw
//! lines. The secondary form is the top candidate of an explicit
//! `rlm.code-search/1` presentation (single `file` snippet). Anything else —
//! grep snippets, multi-file forms, shell output, instruction files — is NOT
//! eligible. Line matching judges ONLY the supplied text: it is never a
//! repository index, and the annotation says so.
//!
//! Absence is scoped: for texts above the Choice limit (255 lines) a windowed
//! cascade first picks one window, and only that window is judged. Lines and
//! bytes are both capped; every truncation is disclosed in the annotation.
//! The annotation is additive (a new text block appended to the request copy
//! of the tool result): nothing is removed or rewritten, and session history
//! keeps the original.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use pi_jev::active::ActiveDecision;
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::search::{
    build_line_find_text, fitting_window_state, validate_where_distribution, verdict,
    window_questions, LineEntry, LineFindOptions, LineFindText, LineFindVerdict, WindowEntries,
    EXISTS_QUESTION_ID, EXISTS_THRESHOLD_PROVENANCE, LINE_WINDOW, PROMPT_VERSION_SEARCH,
    WHERE_QUESTION_ID, WINDOW_QUESTION_ID,
};
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};

use super::jev_code_search::MAX_PRESENTATION_BYTES;

/// Annotation bound: the additive block never exceeds 2KB.
pub const MAX_ANNOTATION_BYTES: usize = 2048;

/// Which explicitly supplied text a line-find presentation judges.
#[derive(Debug, Clone, PartialEq)]
pub enum LineFindSource {
    /// An ipython source-read tool result recognized by the deterministic
    /// paired-cell idiom. Ids map to offsets in the SUPPLIED output; real file
    /// line numbers are not claimed.
    IpythonRead { path: String },
    /// The `file` snippet of an explicit code-search presentation candidate.
    /// Ids map to real file lines starting at the candidate's 1-based `line`.
    Snippet { path: String, start_line: usize },
}

impl LineFindSource {
    fn kind(&self) -> &'static str {
        match self {
            LineFindSource::IpythonRead { .. } => "ipython_source_read",
            LineFindSource::Snippet { .. } => "snippet",
        }
    }

    fn path(&self) -> &str {
        match self {
            LineFindSource::IpythonRead { path } | LineFindSource::Snippet { path, .. } => path,
        }
    }

    /// How annotation consumers must read line references.
    fn offsets(&self) -> &'static str {
        match self {
            LineFindSource::IpythonRead { .. } => "supplied_output",
            LineFindSource::Snippet { .. } => "file_lines",
        }
    }

    fn source_line(&self, offset: usize) -> Option<usize> {
        match self {
            LineFindSource::IpythonRead { .. } => None,
            LineFindSource::Snippet { start_line, .. } => Some(start_line + offset),
        }
    }
}

/// One prepared line-find judgment over supplied text. At most one per
/// provider request (the bridge uses the first presentation only).
#[derive(Debug, Clone)]
pub struct LineFindPresentation {
    pub message_index: usize,
    pub source: LineFindSource,
    pub text: LineFindText,
    pub query: String,
    pub fingerprint: String,
}

impl LineFindPresentation {
    /// Whether the cascade window pass (pass 1) is needed first.
    pub fn needs_window_pass(&self) -> bool {
        self.text.windows() > 1
    }

    /// Pass 1 (windowed cascade): which window plausibly contains the answer.
    /// Absence is not judged here. The state is measured against the decision
    /// state cap (previews shrink; the window set never does).
    pub fn pass1(&self) -> Option<(Value, Vec<PreparedQuestion>)> {
        if !self.needs_window_pass() {
            return None;
        }
        let state = fitting_window_state(&self.query, &self.text)?;
        let questions = window_questions(&self.query, self.text.windows())?;
        Some((state, questions))
    }

    /// Pass 2: judge the supplied lines — the chosen cascade window, or the
    /// whole text when it fits one request. The state is MEASURED against the
    /// decision state cap; entries, criteria and annotation scope all come
    /// from the same fitted supply.
    pub fn pass2(&self, window: Option<usize>) -> Option<(Value, Vec<PreparedQuestion>)> {
        let (entries, state) = self.text.fitting_line_request(&self.query, window)?;
        let questions = questions_for(&self.query, &entries.entries)?;
        Some((state, questions))
    }

    /// The bounded entries a pass-2 request would supply (for fingerprints and
    /// scope disclosures) — the SAME fitted supply `pass2` sends, so scope
    /// disclosures never claim more than the outgoing state carries.
    pub fn pass2_entries(&self, window: Option<usize>) -> Option<WindowEntries> {
        self.text
            .fitting_line_request(&self.query, window)
            .map(|(entries, _state)| entries)
    }

    fn top_line_refs(
        &self,
        ranked: &[(String, f64)],
        top: usize,
    ) -> Vec<(String, f64, Option<usize>, String)> {
        let by_id: BTreeMap<&str, &LineEntry> = self
            .text
            .entries
            .iter()
            .map(|entry| (entry.id.as_str(), entry))
            .collect();
        ranked
            .iter()
            .take(top)
            .filter_map(|(id, probability)| {
                let entry = by_id.get(id.as_str())?;
                Some((
                    id.clone(),
                    *probability,
                    self.source.source_line(id[1..].parse::<usize>().ok()?),
                    entry.text.clone(),
                ))
            })
            .collect()
    }

    /// Typed set-level acceptance and annotation assembly. Requires BOTH
    /// accepted answers (where Choice + existence Noul) from the SAME
    /// request and turn, fresh, matching this presentation's category and
    /// question ids. `where_probabilities` comes from the raw decision
    /// outcome (the full Choice distribution); it is validated against the
    /// supplied entries and never trusted blindly. `None` annotates nothing
    /// (fail open).
    #[allow(clippy::too_many_arguments)]
    pub fn annotate(
        &self,
        where_decision: &ActiveDecision,
        exists_decision: &ActiveDecision,
        where_probabilities: Option<&BTreeMap<String, f64>>,
        entries: &WindowEntries,
        request_id: &str,
        turn: u64,
        now: SystemTime,
        max_age: Duration,
        model: Option<&str>,
        options: &LineFindOptions,
    ) -> Option<Value> {
        for decision in [where_decision, exists_decision] {
            if decision.category != DecisionCategory::CodeLineFind
                || decision.request_id != request_id
                || decision.turn != turn
                || !matches!(now.duration_since(decision.decided_at), Ok(age) if age <= max_age)
            {
                return None;
            }
        }
        if where_decision.question_id != WHERE_QUESTION_ID
            || exists_decision.question_id != EXISTS_QUESTION_ID
        {
            return None;
        }
        let exists_noul: f64 = exists_decision.value.trim().parse().ok()?;
        let verdict = verdict(exists_noul)?;
        let ranked = match where_probabilities {
            Some(probabilities) => validate_where_distribution(probabilities, &entries.entries)?,
            None => return None,
        };
        let top = self.top_line_refs(&ranked, options.top_lines.min(8));
        // Bound the additive block: shrink top lines, then give up (fail open).
        let mut keep = top.len();
        loop {
            let annotation = self.annotation_json(
                verdict,
                exists_noul,
                &top[..keep],
                entries,
                request_id,
                model,
            );
            match serde_json::to_string(&annotation) {
                Ok(serialized) if serialized.len() <= MAX_ANNOTATION_BYTES => {
                    return Some(annotation)
                }
                Ok(_) => {
                    if keep <= 1 {
                        return None;
                    }
                    keep -= 1;
                }
                Err(_) => return None,
            }
        }
    }

    fn annotation_json(
        &self,
        verdict: LineFindVerdict,
        exists_noul: f64,
        top: &[(String, f64, Option<usize>, String)],
        entries: &WindowEntries,
        request_id: &str,
        model: Option<&str>,
    ) -> Value {
        let ids = entries.ids();
        let top_lines: Vec<Value> = top
            .iter()
            .map(|(id, probability, source_line, excerpt)| {
                let mut entry = json!({
                    "id": id,
                    "probability": probability,
                    "excerpt": excerpt,
                });
                if let Some(source_line) = source_line {
                    entry["line"] = json!(source_line);
                } else {
                    entry["offset"] = json!(id[1..]
                        .parse::<usize>()
                        .map(|position| position + 1)
                        .unwrap_or(0));
                }
                entry
            })
            .collect();
        let mut scope = json!({
            "kind": self.source.kind(),
            "path": self.source.path(),
            "lines": format!("{}-{}", ids.first().map(String::as_str).unwrap_or_default(), ids.last().map(String::as_str).unwrap_or_default()),
            "offsets": self.source.offsets(),
        });
        if let Some(window) = entries.window_index {
            scope["window"] = json!(format!("W{window:02}"));
        }
        json!({
            "jev_line_find": {
                "query": pi_jev::redact::bounded_excerpt(&self.query, 160),
                "verdict": verdict.as_str(),
                "exists_noul": exists_noul,
                "top_lines": top_lines,
                "scope": scope,
                "disclosures": {
                    "lines_truncated": entries.any_truncated(),
                    "bytes_capped": entries.bytes_capped,
                    "windowed": entries.windowed(),
                    "original_lines": self.text.original_lines,
                },
                "model": model,
                "request_id": request_id,
                "prompt_version": PROMPT_VERSION_SEARCH,
                "threshold_provenance": EXISTS_THRESHOLD_PROVENANCE,
                "notice": "Line reference annotation for explicitly supplied text only; not a repository search or index.",
            }
        })
    }

    /// Append the annotation block to the REQUEST COPY of the tool result at
    /// `message_index`. Additive only: existing content blocks are untouched,
    /// session history keeps the original.
    pub fn attach(&self, requests: &mut [Value], block: Value) -> Option<()> {
        let message = requests.get_mut(self.message_index)?;
        if message["role"] != "toolResult" {
            return None;
        }
        let content = message.get_mut("content")?.as_array_mut()?;
        if content.len() != 1 {
            return None;
        }
        content.push(block);
        Some(())
    }

    /// Truthful telemetry for the correlation ledger.
    pub fn action_metadata(
        &self,
        verdict: LineFindVerdict,
        exists_noul: f64,
        top: Option<&str>,
        windowed: bool,
    ) -> BTreeMap<String, String> {
        let mut metadata = BTreeMap::from([
            (
                "line_find_source".to_string(),
                self.source.kind().to_string(),
            ),
            (
                "line_find_path".to_string(),
                self.source.path().chars().take(256).collect(),
            ),
            (
                "line_find_verdict".to_string(),
                verdict.as_str().to_string(),
            ),
            (
                "line_find_exists_noul".to_string(),
                format!("{exists_noul:.2}"),
            ),
            ("line_find_windowed".to_string(), windowed.to_string()),
        ]);
        if let Some(top) = top {
            metadata.insert("line_find_top".to_string(), top.to_string());
        }
        metadata
    }
}

fn questions_for(query: &str, entries: &[LineEntry]) -> Option<Vec<PreparedQuestion>> {
    pi_jev::search::line_find_questions(query, entries)
}

/// Parse a single-quoted or double-quoted string literal. Fails on escapes
/// (a path idiom never needs them) and on unterminated literals.
fn parse_string_literal(text: &str) -> Option<(&str, &str)> {
    let quote = text.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &text[1..];
    let end = rest.find(quote)?;
    let literal = &rest[..end];
    if literal.contains('\\') || literal.contains(quote) || literal.contains('\n') {
        return None;
    }
    Some((literal, &rest[end + 1..]))
}

/// Recognize the deterministic source-read idiom in a paired ipython cell:
/// `print(open(PATH).read())`, `print(open(PATH, "r").read())`,
/// `print(open(PATH, mode="r").read())`, `print(Path(PATH).read_text())`,
/// `print(Path(PATH).read_text(encoding="..."))`, optionally with a trailing
/// semicolon. The exact compaction-safe `from pathlib import Path\n` prefix is
/// also accepted only with its exact `print(Path(PATH).read_text())` spelling.
/// Everything else — aliases, write modes, readlines, bare expressions, or
/// extra statements — is rejected (fail closed).
fn parse_source_read(code: &str) -> Option<String> {
    let code = code.trim();
    let (code, imported_pathlib) = match code.strip_prefix("from pathlib import Path\n") {
        Some(expression) => (expression, true),
        None => (code.trim_end_matches(';').trim(), false),
    };
    if imported_pathlib {
        code.strip_prefix("print(Path(")?
            .strip_suffix(").read_text())")?;
    }
    let body = code.strip_prefix("print(")?.strip_suffix(')')?;
    let (path, rest) = if let Some(after) = body.strip_prefix("open(") {
        let (path, rest) = parse_string_literal(after)?;
        let rest = match rest.strip_prefix(',') {
            Some(rest) => {
                // Only the advertised read-only mode and optional encoding are
                // accepted inside open(...); update/write modes fail closed.
                let rest = rest.trim_start();
                let (mode, rest) = if rest.starts_with("mode") {
                    let rest = rest
                        .strip_prefix("mode")?
                        .trim_start()
                        .strip_prefix('=')?
                        .trim_start();
                    parse_string_literal(rest)?
                } else {
                    parse_string_literal(rest)?
                };
                if mode != "r" {
                    return None;
                }
                match rest.strip_prefix(',') {
                    Some(rest) => {
                        let rest = rest.trim_start();
                        let rest = rest
                            .strip_prefix("encoding")?
                            .trim_start()
                            .strip_prefix('=')?
                            .trim_start();
                        let (_encoding, rest) = parse_string_literal(rest)?;
                        rest
                    }
                    None => rest,
                }
            }
            None => rest,
        };
        let rest = rest.strip_prefix(')')?;
        if !rest.starts_with(".read()") {
            return None;
        }
        (path, &rest[".read()".len()..])
    } else if let Some(after) = body.strip_prefix("Path(") {
        let (path, rest) = parse_string_literal(after)?;
        let rest = rest.strip_prefix(')')?;
        let rest = rest.strip_prefix(".read_text(")?;
        let rest = match rest.strip_suffix(')') {
            Some(inner) => {
                let inner = inner.trim();
                if inner.is_empty() {
                    ""
                } else {
                    let inner = inner
                        .strip_prefix("encoding")?
                        .trim_start()
                        .strip_prefix('=')?
                        .trim_start();
                    let (_encoding, rest) = parse_string_literal(inner)?;
                    rest
                }
            }
            None => return None,
        };
        if !rest.is_empty() {
            return None;
        }
        (path, "")
    } else {
        return None;
    };
    if !rest.is_empty() {
        return None;
    }
    let path = path.trim();
    if path.is_empty() || path.chars().count() > 1024 {
        return None;
    }
    // Instruction-class paths are never line-matched (same class as pinned
    // search candidates).
    let lowered = path.to_ascii_lowercase();
    if lowered.ends_with(".md")
        || lowered.ends_with(".mdx")
        || lowered.ends_with(".mdc")
        || lowered.contains("agents")
        || lowered.contains("instruction")
        || lowered.contains("prompt")
        || lowered.contains("claude")
    {
        return None;
    }
    Some(path.to_string())
}

/// Excluded message classes, shared with the search presentation discipline.
fn excluded_message(message: &Value) -> bool {
    message["role"] != "toolResult"
        || message["toolName"] != "ipython"
        || message["isError"] != false
        || message.get("textSignature").is_some()
        || ["pinned", "mandatory", "instructions", "edited"]
            .iter()
            .any(|key| {
                message
                    .get(*key)
                    .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                    || message["details"]
                        .get(*key)
                        .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
            })
}

fn clean_details(message: &Value) -> bool {
    let details = &message["details"];
    details["status"] == "ok"
        && details["kernelRestarted"] != true
        && ["stderr", "backgroundOutput", "result"].iter().all(|key| {
            details
                .get(*key)
                .is_none_or(|value| value.is_null() || value == "")
        })
        && details
            .get("stdout")
            .and_then(Value::as_str)
            .is_some_and(|stdout| !stdout.is_empty())
}

/// Find the single paired assistant ipython call for a tool result id.
fn paired_call<'a>(messages: &'a [Value], id: &str, before: usize) -> Option<(usize, &'a Value)> {
    let mut found: Option<(usize, &Value)> = None;
    for (at, message) in messages.iter().enumerate().take(before) {
        for call in message["content"].as_array().into_iter().flatten() {
            if message["role"] == "assistant"
                && call["type"] == "toolCall"
                && call["id"] == id
                && call["name"] == "ipython"
            {
                if found.is_some() {
                    return None;
                }
                found = Some((at, call));
            }
        }
    }
    found
}

/// Recognize eligible line-find inputs in the request window, most recent
/// first: the deterministic ipython source-read idiom only. The bridge uses
/// the first presentation (at most one line-find per provider request).
pub fn prepare(
    messages: &[Value],
    query: &str,
    options: &LineFindOptions,
) -> Vec<LineFindPresentation> {
    if messages.len() > 512 || query.trim().is_empty() || options.validate().is_err() {
        return Vec::new();
    }
    if messages.iter().any(|message| {
        message["content"]
            .as_array()
            .is_some_and(|blocks| blocks.len() > 64)
    }) {
        return Vec::new();
    }
    let start = messages
        .iter()
        .rposition(|message| {
            message
                .get("providerContext")
                .is_some_and(|value| !value.is_null())
        })
        .map_or(0, |index| index + 1);
    let mut presentations = Vec::new();
    for (index, message) in messages.iter().enumerate().skip(start).rev() {
        if excluded_message(message) || !clean_details(message) {
            continue;
        }
        let Some(blocks) = message["content"]
            .as_array()
            .filter(|blocks| blocks.len() == 1)
        else {
            continue;
        };
        if blocks[0]["type"] != "text" || blocks[0].get("textSignature").is_some() {
            continue;
        }
        let Some(text) = blocks[0]["text"]
            .as_str()
            .filter(|text| text.len() <= MAX_PRESENTATION_BYTES)
        else {
            continue;
        };
        let Some(id) = message["toolCallId"].as_str().filter(|id| !id.is_empty()) else {
            continue;
        };
        let Some((call_at, call)) = paired_call(messages, id, index) else {
            continue;
        };
        if call_at < start {
            continue;
        }
        if messages
            .iter()
            .filter(|msg| msg["role"] == "toolResult" && msg["toolCallId"] == id)
            .count()
            != 1
        {
            continue;
        }
        // The real ipython toolCall block serializes its code under
        // `arguments` (pi-ai ToolCall wire shape, same as jev_retrieval);
        // `input` stays tolerated for alternate serializations. The idiom
        // itself is still strictly parsed below, so recognition stays
        // deterministic either way.
        let code = call["arguments"]
            .get("code")
            .and_then(Value::as_str)
            .or_else(|| call["input"].get("code").and_then(Value::as_str));
        let Some(code) = code else {
            continue;
        };
        let Some(path) = parse_source_read(code) else {
            continue;
        };
        let Some(text_model) = build_line_find_text(text) else {
            continue;
        };
        presentations.push(LineFindPresentation {
            message_index: index,
            source: LineFindSource::IpythonRead { path },
            text: text_model,
            query: query.to_string(),
            fingerprint: pi_jev::snapshot::fingerprint_of(&json!([id, text, query])),
        });
        break;
    }
    presentations
}

/// Secondary form: the `file` snippet of an explicit code-search candidate
/// (the bridge passes the TOP candidate of the filtered+reranked projected
/// envelope). Ids map to real file lines starting at the candidate's 1-based
/// `line`.
pub fn prepare_from_snippet(
    candidate: &Value,
    message_index: usize,
    query: &str,
    options: &LineFindOptions,
) -> Option<LineFindPresentation> {
    if query.trim().is_empty() || options.validate().is_err() {
        return None;
    }
    if candidate["kind"] != "file" {
        return None;
    }
    // Instruction-class/pinned candidates are position anchors, never
    // line-matched (same class discipline as the ipython source-read path,
    // which refuses .md/agents/instruction paths).
    if super::jev_code_search::pinned(candidate) {
        return None;
    }
    let path = candidate["path"].as_str()?.trim().to_string();
    if path.is_empty() || path.chars().count() > 1024 {
        return None;
    }
    let start_line = candidate["line"].as_u64().filter(|line| *line > 0)? as usize;
    let snippet = candidate["snippet"].as_str()?;
    let text_model = build_line_find_text(snippet)?;
    // The fingerprint runs BEFORE the struct literal: json! serializes each
    // element by reference (serde_json 1.0.151 macros.rs `to_value(&$other)`),
    // so this only borrows `path`, which the struct literal below then moves
    // into the source field as its last use.
    let fingerprint = pi_jev::snapshot::fingerprint_of(&json!([path, start_line, snippet, query]));
    Some(LineFindPresentation {
        message_index,
        source: LineFindSource::Snippet { path, start_line },
        text: text_model,
        query: query.to_string(),
        fingerprint,
    })
}

/// Parse a cascade window answer (`W03`) into a window index. Only the strict
/// two-digit form is accepted; the presentation rejects out-of-range windows
/// (its `pass2` returns `None`), so an invalid window id never reaches a
/// question about lines.
pub fn narrow(window_id: &str) -> Option<usize> {
    let rest = window_id.strip_prefix('W')?;
    let index = rest.parse::<usize>().ok()?;
    if rest.len() != 2 {
        return None;
    }
    Some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(id: &str, code: &str) -> Value {
        // Real wire shape: pi-ai ToolCall serializes `arguments`
        // (jev_retrieval reads the same key in production).
        json!({
            "type": "toolCall",
            "id": id,
            "name": "ipython",
            "arguments": {"code": code},
        })
    }

    fn source_read_message(id: &str, code: &str, content: &str) -> Vec<Value> {
        vec![
            json!({"role": "user", "content": [{"type": "text", "text": "find it"}]}),
            json!({"role": "assistant", "content": [tool_call(id, code)]}),
            json!({
                "role": "toolResult",
                "toolCallId": id,
                "toolName": "ipython",
                "isError": false,
                "content": [{"type": "text", "text": content}],
                "details": {"status": "ok", "stdout": content.to_string(), "result": null},
            }),
        ]
    }

    #[test]
    fn recognizes_print_wrapped_read_idioms_only() {
        assert_eq!(
            parse_source_read(r#"print(open("src/core/main.rs").read())"#).as_deref(),
            Some("src/core/main.rs")
        );
        assert_eq!(
            parse_source_read(r#"print(open('src/a.rs', "r").read())"#).as_deref(),
            Some("src/a.rs")
        );
        assert_eq!(
            parse_source_read(r#"print(open("src/a.rs", mode="r").read());"#).as_deref(),
            Some("src/a.rs")
        );
        assert_eq!(
            parse_source_read(r#"print(open("src/a.rs", "r", encoding="utf-8").read())"#)
                .as_deref(),
            Some("src/a.rs")
        );
        assert_eq!(
            parse_source_read(r#"print(Path("src/a.rs").read_text())"#).as_deref(),
            Some("src/a.rs")
        );
        assert_eq!(
            parse_source_read(r#"print(Path("src/a.rs").read_text(encoding="utf-8"))"#).as_deref(),
            Some("src/a.rs")
        );
        // Write modes, readlines, bare reads, extra statements, and escapes fail.
        assert!(parse_source_read(r#"print(open("src/a.rs", "w").read())"#).is_none());
        assert!(parse_source_read(r#"print(open("src/a.rs", "r+").read())"#).is_none());
        assert!(parse_source_read(r#"print(open("src/a.rs").readlines())"#).is_none());
        assert!(parse_source_read(r#"open("src/a.rs").read()"#).is_none());
        assert!(parse_source_read(r#"print(open("src/a.rs").read()); print("x")"#).is_none());
        assert!(parse_source_read(r#"print(open("src/a\").rs").read())"#).is_none());
        // Instruction-class paths are excluded.
        assert!(parse_source_read(r#"print(open("CLAUDE.md").read())"#).is_none());
        assert!(parse_source_read(r#"print(open("agents/traits.rs").read())"#).is_none());
    }

    #[test]
    fn recognizes_exact_imported_pathlib_compaction_shape() {
        assert_eq!(
            parse_source_read("from pathlib import Path\nprint(Path(\"src/a.rs\").read_text())")
                .as_deref(),
            Some("src/a.rs")
        );
        // The imported branch is exactly the already-supported compaction
        // spelling: no alias, open(), encoding, semicolon, or extra statement.
        assert!(parse_source_read(
            "from pathlib import Path as P\nprint(P(\"src/a.rs\").read_text())"
        )
        .is_none());
        assert!(
            parse_source_read("from pathlib import Path\nprint(open(\"src/a.rs\").read())")
                .is_none()
        );
        assert!(parse_source_read(
            "from pathlib import Path\nprint(Path(\"src/a.rs\").read_text(encoding=\"utf-8\"))"
        )
        .is_none());
        assert!(parse_source_read(
            "from pathlib import Path\nprint(Path(\"src/a.rs\").read_text());"
        )
        .is_none());
        assert!(parse_source_read(
            "from pathlib import Path\nprint(Path(\"src/a.rs\").read_text())\nprint(\"extra\")"
        )
        .is_none());
    }

    #[test]
    fn prepares_most_recent_eligible_source_read() {
        let content = "alpha\nbeta\ngamma\n";
        let messages =
            source_read_message("call-2", r#"print(open("src/one.rs").read())"#, content);
        let presentations = prepare(&messages, "Where is beta defined?", &Default::default());
        assert_eq!(presentations.len(), 1);
        let presentation = &presentations[0];
        assert_eq!(presentation.message_index, 2);
        assert_eq!(
            presentation.source,
            LineFindSource::IpythonRead {
                path: "src/one.rs".to_string()
            }
        );
        assert_eq!(presentation.text.lines(), 3);
        assert!(!presentation.needs_window_pass());
        // stderr or result channels disqualify the result.
        let mut noisy =
            source_read_message("call-3", r#"print(open("src/one.rs").read())"#, content);
        noisy[2]["details"]["result"] = json!(content);
        assert!(prepare(&noisy, "Where is beta?", &Default::default()).is_empty());
        // A bare (non-print) read is not recognized.
        let mut bare = source_read_message("call-4", r#"open("src/one.rs").read()"#, content);
        bare[2]["details"]["result"] = Value::Null;
        assert!(prepare(&bare, "Where is beta?", &Default::default()).is_empty());
    }

    #[test]
    fn snippet_source_maps_ids_to_real_file_lines() {
        let candidate = json!({
            "kind": "file",
            "path": "src/session.rs",
            "line": 41,
            "snippet": "let x = 1;\nlet y = 2;\nlet z = 3;\n",
        });
        let presentation = prepare_from_snippet(&candidate, 2, "Where is y?", &Default::default())
            .expect("snippet presentation");
        assert_eq!(
            presentation.source,
            LineFindSource::Snippet {
                path: "src/session.rs".to_string(),
                start_line: 41
            }
        );
        assert_eq!(presentation.text.lines(), 3);
        assert_eq!(presentation.source.source_line(1), Some(42));
        // grep-kind candidates are not single-document inputs.
        let grep = json!({"kind": "grep", "path": "src/a.rs", "line": 1, "snippet": "x"});
        assert!(prepare_from_snippet(&grep, 2, "Where is y?", &Default::default()).is_none());
    }

    #[test]
    fn paired_call_is_reachable_through_the_real_toolcall_wire_shape() {
        // The production message shape (pi-ai ToolCall `arguments`) must be
        // recognized: the deterministic source-read path is dead if only the
        // alternate `input` key is accepted. Both stay strictly idiom-checked.
        let content = "alpha\nbeta\n";
        let real = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "find it"}]}),
            json!({"role": "assistant", "content": [{"type": "toolCall", "id": "c9",
                "name": "ipython", "arguments": {"code": r#"print(Path("src/a.rs").read_text())"#}}]}),
            json!({
                "role": "toolResult",
                "toolCallId": "c9",
                "toolName": "ipython",
                "isError": false,
                "content": [{"type": "text", "text": content}],
                "details": {"status": "ok", "stdout": content, "kernelRestarted": false},
            }),
        ];
        let presentations = prepare(&real, "Where is beta?", &Default::default());
        assert_eq!(presentations.len(), 1);
        assert_eq!(
            presentations[0].source,
            LineFindSource::IpythonRead {
                path: "src/a.rs".to_string()
            }
        );
        // The alternate `input` key is still tolerated (never the only path).
        let alternate = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "find it"}]}),
            json!({"role": "assistant", "content": [{"type": "toolCall", "id": "c10",
                "name": "ipython", "input": {"code": r#"print(open("src/b.rs").read())"#}}]}),
            json!({
                "role": "toolResult",
                "toolCallId": "c10",
                "toolName": "ipython",
                "isError": false,
                "content": [{"type": "text", "text": content}],
                "details": {"status": "ok", "stdout": content, "kernelRestarted": false},
            }),
        ];
        let presentations = prepare(&alternate, "Where is beta?", &Default::default());
        assert_eq!(presentations.len(), 1);
        assert_eq!(
            presentations[0].source,
            LineFindSource::IpythonRead {
                path: "src/b.rs".to_string()
            }
        );
    }

    #[test]
    fn pinned_instruction_candidates_are_never_line_find_sources() {
        // Pinned/instruction-class candidates are position anchors, never
        // line-matched, in both source forms.
        let pinned = json!({
            "kind": "file",
            "path": "AGENTS.md",
            "line": 1,
            "snippet": "always answer politely\n",
        });
        assert!(prepare_from_snippet(&pinned, 2, "q", &Default::default()).is_none());
        let mandatory = json!({
            "kind": "file",
            "path": "src/real.rs",
            "line": 1,
            "snippet": "let x = 1;\n",
            "mandatory": true,
        });
        assert!(prepare_from_snippet(&mandatory, 2, "q", &Default::default()).is_none());
    }

    #[test]
    fn window_pass_scopes_cascade_texts() {
        let lines: Vec<String> = (0..600).map(|n| format!("line {n:03}")).collect();
        let content = lines.join("\n") + "\n";
        let messages = source_read_message("call-5", r#"print(open("big.rs").read())"#, &content);
        let presentations = prepare(&messages, "find line 500", &Default::default());
        assert_eq!(presentations.len(), 1);
        let presentation = &presentations[0];
        assert!(presentation.needs_window_pass());
        let (state, questions) = presentation.pass1().expect("window pass");
        assert_eq!(state["supplied_windows"].as_array().unwrap().len(), 3);
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].question_id, WINDOW_QUESTION_ID);
        let (state, questions) = presentation.pass2(Some(1)).expect("window slice");
        assert_eq!(questions.len(), 2);
        assert_eq!(questions[0].question_id, WHERE_QUESTION_ID);
        assert_eq!(questions[1].question_id, EXISTS_QUESTION_ID);
        // The chunked supplied-lines encoding: 255 short lines pack into a
        // handful of bounded chunks, each within the snapshot string bound
        // and the array within the snapshot item bound, so the state stays
        // reachable and no supplied line is hidden from the model.
        let chunks = state["supplied_lines"].as_array().unwrap();
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.as_str().unwrap().split("\n").count())
                .sum::<usize>(),
            LINE_WINDOW
        );
        assert!(chunks.len() <= pi_jev::snapshot::MAX_ITEMS);
        assert!(chunks.iter().all(
            |chunk| chunk.as_str().unwrap().chars().count() <= pi_jev::snapshot::MAX_TEXT_CHARS
        ));
    }
}
