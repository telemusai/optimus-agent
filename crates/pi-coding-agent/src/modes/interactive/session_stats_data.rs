//! Read-only accounting for the stats overlay. Nothing here dispatches a model request.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_ai::types::Usage;
use pi_jev::correlate::CorrelationRecord;
use serde_json::Value;

use crate::core::usage::{add_assistant_usage, empty_usage, subtract_assistant_usage};
use crate::modes::agent_connection::types as wire;

#[derive(Clone)]
pub(super) struct ModelUsage {
    pub provider: String,
    pub model: String,
    pub calls: usize,
    pub missing_usage: usize,
    pub missing_cost: usize,
    pub usage: Usage,
}

impl ModelUsage {
    fn new(provider: &str, model: &str) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            calls: 0,
            missing_usage: 0,
            missing_cost: 0,
            usage: empty_usage(),
        }
    }
    pub fn total(&self) -> f64 {
        self.usage.input + self.usage.output + self.usage.cache_read + self.usage.cache_write
    }
    fn add(&mut self, usage: Option<&Usage>) {
        self.calls += 1;
        if let Some(usage) = usage.filter(|u| valid_usage(u) && token_total(u) > 0.0) {
            add_assistant_usage(&mut self.usage, usage);
            if !usage.cost.total.is_finite() || usage.cost.total <= 0.0 {
                self.missing_cost += 1;
            }
        } else {
            self.missing_usage += 1;
            self.missing_cost += 1;
        }
    }
}

fn valid_usage(usage: &Usage) -> bool {
    [
        usage.input,
        usage.output,
        usage.cache_read,
        usage.cache_write,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value >= 0.0)
}

fn token_total(usage: &Usage) -> f64 {
    usage.input + usage.output + usage.cache_read + usage.cache_write
}

#[derive(Clone, Default)]
pub(super) struct Accounting {
    pub models: BTreeMap<(String, String), ModelUsage>,
    pub turns: Vec<f64>,
    pub compactions: usize,
}

impl Accounting {
    pub fn sorted_models(&self) -> Vec<&ModelUsage> {
        let mut rows: Vec<_> = self.models.values().collect();
        rows.sort_by(|a, b| b.total().total_cmp(&a.total()));
        rows
    }
    pub fn total(&self) -> f64 {
        self.models.values().map(ModelUsage::total).sum()
    }
    pub fn missing(&self) -> usize {
        self.models.values().map(|m| m.missing_usage).sum()
    }
    fn merge(&mut self, other: Self) {
        for (key, row) in other.models {
            let target = self
                .models
                .entry(key)
                .or_insert_with(|| ModelUsage::new(&row.provider, &row.model));
            target.calls += row.calls;
            target.missing_usage += row.missing_usage;
            target.missing_cost += row.missing_cost;
            add_assistant_usage(&mut target.usage, &row.usage);
        }
    }
}

/// Entries from SessionManager and get_session_tree have already had aggregate
/// usage applied. Subtract every child attribution, including off-branch rows.
pub(super) fn account(entries: &[Value], leaf: Option<&str>) -> Accounting {
    let by_id: HashMap<_, _> = entries
        .iter()
        .filter_map(|e| Some((e["id"].as_str()?, e)))
        .collect();
    let mut branch = Vec::new();
    let mut seen = BTreeSet::new();
    let mut cursor = leaf;
    while let Some(id) = cursor {
        if !seen.insert(id) {
            break;
        }
        let Some(entry) = by_id.get(id) else {
            break;
        };
        branch.push(*entry);
        cursor = entry["parentId"].as_str();
    }
    branch.reverse();
    let mut attributed: HashMap<&str, Usage> = HashMap::new();
    for entry in entries
        .iter()
        .filter(|e| e["type"] == "child_usage_attributed")
    {
        if let (Some(id), Ok(usage)) = (
            entry["targetId"].as_str(),
            serde_json::from_value::<Usage>(entry["childUsage"].clone()),
        ) {
            add_assistant_usage(attributed.entry(id).or_insert_with(empty_usage), &usage);
        }
    }
    let mut result = Accounting::default();
    let mut turn = None;
    for entry in branch {
        if entry["message"]["role"] == "user" {
            if let Some(tokens) = turn.take() {
                result.turns.push(tokens);
            }
            turn = Some(0.0);
        }
        let summary = matches!(
            entry["type"].as_str(),
            Some("compaction" | "branch_summary")
        );
        if entry["type"] == "compaction" {
            result.compactions += 1;
        }
        let message = &entry["message"];
        if message["role"] != "assistant" && !summary {
            continue;
        }
        let source = if summary { entry } else { message };
        let mut usage = serde_json::from_value::<Usage>(source["usage"].clone()).ok();
        if let (Some(usage), Some(child)) = (
            usage.as_mut(),
            attributed.get(entry["id"].as_str().unwrap_or("")),
        ) {
            subtract_assistant_usage(usage, child);
        }
        // Aborted/error messages with no reported usage may be synthetic.
        if !summary
            && matches!(message["stopReason"].as_str(), Some("error" | "aborted"))
            && usage.as_ref().is_none_or(|u| token_total(u) == 0.0)
        {
            continue;
        }
        // Hook-based compaction need not call a model; absent usage is not a call.
        if summary && source.get("usage").is_none() {
            continue;
        }
        let provider = source["provider"].as_str().unwrap_or("unrecorded");
        let model =
            source["model"]
                .as_str()
                .unwrap_or(if summary { "summary model" } else { "model" });
        let row = result
            .models
            .entry((provider.into(), model.into()))
            .or_insert_with(|| ModelUsage::new(provider, model));
        row.add(usage.as_ref());
        if let Some(usage) = usage.as_ref().filter(|u| valid_usage(u)) {
            *turn.get_or_insert(0.0) += token_total(usage);
        }
    }
    if let Some(tokens) = turn {
        result.turns.push(tokens);
    }
    result
}

#[derive(Clone, Default)]
pub(super) struct Savings {
    pub before: u64,
    pub after: u64,
    pub candidates: u64,
    pub retained: u64,
    pub samples: usize,
}

#[derive(Clone, Default)]
pub(super) struct JevAccounting {
    pub requests: usize,
    pub applied: usize,
    pub fallbacks: usize,
    pub agrees: usize,
    pub disagrees: usize,
    pub median_ms: Option<u64>,
    pub savings: BTreeMap<String, Savings>,
    pub compaction: Savings,
    pub models: BTreeMap<(String, String), ModelUsage>,
    pub records: usize,
}

fn count(action: &BTreeMap<String, String>, key: &str) -> Option<u64> {
    action.get(key)?.parse().ok()
}

/// A batched decision writes one row per question, sometimes in both modes.
/// Usage belongs to the request; projected candidate counts to its category.
pub(super) fn account_jev(
    records: &[CorrelationRecord],
    sessions: &BTreeSet<String>,
) -> JevAccounting {
    let mut result = JevAccounting::default();
    let mut requests: BTreeMap<(&str, &str), Vec<&CorrelationRecord>> = BTreeMap::new();
    for record in records.iter().filter(|r| sessions.contains(&r.session_id)) {
        result.records += 1;
        requests
            .entry((&record.session_id, &record.request_id))
            .or_default()
            .push(record);
    }
    let mut durations = Vec::new();
    for rows in requests.values() {
        let called = rows.iter().any(|r| r.attempt > 0);
        if called {
            result.requests += 1;
            if let Some(duration) = rows.iter().filter_map(|r| r.duration_ms).max() {
                durations.push(duration);
            }
            let model = rows
                .iter()
                .find_map(|r| r.response_model.as_deref())
                .unwrap_or("model unrecorded");
            let input = rows
                .iter()
                .filter_map(|r| r.observed_metrics.get("jev_input_tokens").copied())
                .max();
            let output = rows
                .iter()
                .filter_map(|r| r.observed_metrics.get("jev_output_tokens").copied())
                .max();
            let row = result
                .models
                .entry(("JEV".into(), model.into()))
                .or_insert_with(|| ModelUsage::new("JEV", model));
            row.calls += 1;
            row.missing_cost += 1;
            row.usage.input += input.unwrap_or(0) as f64;
            row.usage.output += output.unwrap_or(0) as f64;
            row.usage.total_tokens = row.total();
            if input.is_none() || output.is_none() {
                row.missing_usage += 1;
            }
        }
        if rows.iter().any(|r| r.applied) {
            result.applied += 1;
        }
        if rows.iter().any(|r| r.fallback_reason.is_some()) {
            result.fallbacks += 1;
        }
        let mut comparisons = BTreeSet::new();
        let mut categories = BTreeSet::new();
        for row in rows {
            if row.schema_version == pi_jev::correlate::RECORD_SCHEMA_VERSION
                && comparisons.insert((&row.category, &row.question_id))
            {
                match row.agreement.as_deref() {
                    Some("agree") => result.agrees += 1,
                    Some("disagree") => result.disagrees += 1,
                    _ => {}
                }
            }
            if !row.applied || !categories.insert(&row.category) {
                continue;
            }
            if row.category == "compaction" {
                if let (Some(before), Some(after)) = (
                    row.observed_metrics.get("estimated_tokens_before"),
                    row.observed_metrics.get("estimated_tokens_after"),
                ) {
                    result.compaction.before = result.compaction.before.saturating_add(*before);
                    result.compaction.after = result.compaction.after.saturating_add(*after);
                    result.compaction.samples += 1;
                }
            } else if let (Some(before), Some(after)) = (
                count(&row.actual_action, "estimated_candidate_tokens_before"),
                count(&row.actual_action, "estimated_candidate_tokens_after"),
            ) {
                let savings = result.savings.entry(row.category.clone()).or_default();
                savings.before = savings.before.saturating_add(before);
                savings.after = savings.after.saturating_add(after);
                savings.candidates = savings
                    .candidates
                    .saturating_add(count(&row.actual_action, "candidate_count").unwrap_or(0));
                savings.retained = savings
                    .retained
                    .saturating_add(count(&row.actual_action, "retained_count").unwrap_or(0));
                savings.samples += 1;
            }
        }
    }
    durations.sort_unstable();
    result.median_ms = durations.get(durations.len() / 2).copied();
    result
}

#[derive(Clone, Default)]
pub(super) struct Snapshot {
    pub session_id: String,
    pub name: String,
    pub include_children: bool,
    pub children: usize,
    pub accounting: Accounting,
    pub jev: JevAccounting,
    pub context: Value,
    pub jev_mode: String,
    pub full_jev: bool,
    pub jev_compaction: bool,
    pub features: Vec<(String, bool)>,
    pub pipeline: Value,
    pub notes: Vec<String>,
}

#[derive(Default)]
pub(super) struct Loader {
    files: HashMap<PathBuf, (std::time::SystemTime, u64, Vec<Value>)>,
    tree: Option<(String, Option<String>, Vec<Value>)>,
}

fn usage_entry(entry: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for key in [
        "id",
        "parentId",
        "type",
        "usage",
        "provider",
        "model",
        "targetId",
        "childUsage",
    ] {
        if let Some(value) = entry.get(key) {
            result.insert(key.into(), value.clone());
        }
    }
    if let Some(message) = entry.get("message") {
        let mut reduced = serde_json::Map::new();
        for key in ["role", "usage", "provider", "model", "stopReason"] {
            if let Some(value) = message.get(key) {
                reduced.insert(key.into(), value.clone());
            }
        }
        result.insert("message".into(), Value::Object(reduced));
    }
    Value::Object(result)
}

impl Loader {
    fn entries(&mut self, path: &Path) -> Result<Vec<Value>, String> {
        let meta = std::fs::metadata(path).map_err(|_| "Saved usage is unavailable".to_string())?;
        if meta.len() > 64 * 1024 * 1024 {
            return Err("Saved usage exceeds the 64 MiB stats read limit".into());
        }
        let modified = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if let Some((old, len, entries)) = self.files.get(path) {
            if *old == modified && *len == meta.len() {
                return Ok(entries.clone());
            }
        }
        let entries: Vec<_> =
            crate::core::session_manager::load_entries_from_file(&path.to_string_lossy())
                .into_iter()
                .map(|entry| usage_entry(&Value::Object(entry)))
                .collect();
        if entries.is_empty() {
            return Err("Saved usage has not been written yet".into());
        }
        if self.files.len() >= 256 {
            self.files.clear();
        }
        self.files
            .insert(path.into(), (modified, meta.len(), entries.clone()));
        Ok(entries)
    }

    pub async fn collect(
        &mut self,
        connection: &Arc<dyn wire::AgentConnection>,
        include_children: bool,
    ) -> Result<Snapshot, String> {
        let (state, jev_status) = tokio::join!(connection.get_state(), connection.get_jev_status());
        let state = state?;
        let mut result = Snapshot {
            session_id: state.session_id.clone(),
            name: state
                .session_name
                .clone()
                .unwrap_or_else(|| "Current chat".into()),
            include_children,
            context: state.context_usage.clone(),
            ..Default::default()
        };
        let settings = pi_jev::config::JevSettingsStore::new(crate::config::get_agent_dir()).load();
        result.jev_mode = settings.effective_mode(&state.session_id).as_str().into();
        result.full_jev = settings.full_jev_active();
        result.jev_compaction = settings.effective_compaction_enabled(&state.session_id);
        if let Ok(Value::Object(features)) =
            serde_json::to_value(settings.effective_features(&state.session_id))
        {
            result.features = features
                .into_iter()
                .filter_map(|(name, value)| value.as_bool().map(|v| (name, v)))
                .collect();
        }
        result.pipeline = jev_status
            .ok()
            .flatten()
            .and_then(|v| v.get("pipeline").cloned())
            .unwrap_or(Value::Null);
        let local = state
            .session_file
            .as_deref()
            .and_then(|path| self.entries(Path::new(path)).ok());
        let entries = if let Some(entries) = local {
            entries
        } else {
            let unchanged = self
                .tree
                .as_ref()
                .is_some_and(|(id, leaf, _)| *id == state.session_id && *leaf == state.leaf_id);
            if !unchanged {
                let tree = connection.get_session_tree().await?;
                let mut entries = Vec::new();
                let mut pending: Vec<_> = tree.tree.iter().collect();
                while let Some(node) = pending.pop() {
                    entries.push(usage_entry(
                        &serde_json::to_value(&node.entry).map_err(|e| e.to_string())?,
                    ));
                    pending.extend(node.children.iter());
                }
                self.tree = Some((state.session_id.clone(), state.leaf_id.clone(), entries));
            }
            self.tree
                .as_ref()
                .map(|(_, _, entries)| entries.clone())
                .unwrap_or_default()
        };
        result.accounting = account(&entries, state.leaf_id.as_deref());
        let mut session_ids = BTreeSet::from([state.session_id.clone()]);
        if include_children {
            let mut dirs = Vec::new();
            // Finished children may no longer be in the live worker roster.
            if let Some(dir) = state.session_dir.as_deref() {
                let artifacts =
                    crate::core::session_manager::get_session_artifact_path(dir, &state.session_id);
                dirs.extend(child_dirs(Path::new(&artifacts)));
            }
            match connection.get_rlm_child_snapshots().await {
                Ok(children) => dirs.extend(children.iter().map(|c| PathBuf::from(&c.session_dir))),
                Err(_) => result.notes.push(
                    "Live subagent roster unavailable; showing saved usage where available".into(),
                ),
            }
            let mut seen_dirs = BTreeSet::new();
            while let Some(dir) = dirs.pop() {
                if !seen_dirs.insert(dir.clone()) {
                    continue;
                }
                if seen_dirs.len() > 256 {
                    result
                        .notes
                        .push("Subagent usage truncated at 256 agents".into());
                    break;
                }
                let mut files = Vec::new();
                if let Ok(names) = std::fs::read_dir(&dir) {
                    for entry in names.flatten() {
                        let path = entry.path();
                        if path.is_dir() && entry.file_name().to_string_lossy().starts_with("sub-")
                        {
                            dirs.push(path);
                        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                            files.push((
                                entry
                                    .metadata()
                                    .and_then(|m| m.modified())
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                                path,
                            ));
                        }
                    }
                }
                files.sort_by(|a, b| b.0.cmp(&a.0));
                let entries = files.first().and_then(|(_, path)| self.entries(path).ok());
                let Some(entries) = entries else {
                    result
                        .notes
                        .push("Some subagent usage is unavailable or not persisted yet".into());
                    continue;
                };
                let Some(id) = entries.first().and_then(|e| e["id"].as_str()) else {
                    continue;
                };
                if !session_ids.insert(id.into()) {
                    continue;
                }
                result.children += 1;
                let leaf = entries
                    .iter()
                    .rev()
                    .find(|e| e["type"] != "session")
                    .and_then(|e| e["id"].as_str());
                result.accounting.merge(account(&entries, leaf));
            }
        }
        let path = PathBuf::from(crate::config::get_agent_dir()).join("jev");
        let mut records =
            pi_jev::correlate::read_records(&path.join("records.jsonl.1"), 5 * 1024 * 1024);
        records.extend(pi_jev::correlate::read_records(
            &path.join("records.jsonl"),
            5 * 1024 * 1024,
        ));
        result.jev = account_jev(&records, &session_ids);
        result.notes.sort();
        result.notes.dedup();
        Ok(result)
    }
}

fn child_dirs(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            entry.file_name().to_string_lossy().starts_with("sub-") && entry.path().is_dir()
        })
        .map(|entry| entry.path())
        .collect()
}

#[cfg(test)]
#[path = "session_stats_data_tests.rs"]
mod tests;
