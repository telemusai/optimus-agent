//! Agent-facing Jev questions. Registration and execution use the same session gate.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pi_agent_core::types::AgentToolResult;
use pi_jev::config::{JevSettings, JevSettingsStore};
use pi_jev::types::{
    Answer, DecisionBundle, DecisionCategory, DecisionOutcome, QuestionSpec, SystemOneRequest,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::core::extensions::types::{
    Extension, ExtensionHandler, ExtensionRuntime, RegisteredTool, SharedExtension, ToolDefinition,
};
use crate::core::tools::tool_definition_wrapper::text_tool_result;

pub const TOOL_NAME: &str = "jev_decide";
const SOURCE: &str = "internal:jev-dynamic";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    state: Value,
    questions: BTreeMap<String, QuestionSpec>,
    /// Choice question ids to sample; raw Jev answers are always retained.
    #[serde(default)]
    sample: Vec<String>,
}

fn enabled(settings: &JevSettings, session: &str) -> bool {
    settings.effective_mode(session).allows_active() && settings.effective_features(session).dynamic
}

impl Input {
    fn parse(value: Value) -> Result<Self, String> {
        if serde_json::to_vec(&value)
            .map_err(|_| "Invalid Jev arguments")?
            .len()
            > pi_jev::client::DEFAULT_MAX_PAYLOAD_BYTES
        {
            return Err("Jev Dynamic request exceeds the payload limit.".into());
        }
        let input: Self = serde_json::from_value(value).map_err(|_| {
            "Expected state, questions (Choice/Noul/Score), and optional sample question ids."
                .to_string()
        })?;
        pi_jev::types::validate_request_shape(&SystemOneRequest::new(
            input.state.clone(),
            input.questions.clone(),
        ))
        .map_err(|error| format!("Invalid Jev question: {error}"))?;
        if input.questions.keys().any(|id| {
            id.len() > 128
                || !id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
        }) {
            return Err("Question ids must contain at most 128 ASCII letters, digits, underscores, dots or hyphens.".into());
        }
        let mut seen = BTreeSet::new();
        for id in &input.sample {
            if !seen.insert(id)
                || !matches!(input.questions.get(id), Some(QuestionSpec::Choice { .. }))
            {
                return Err(
                    "Each sample id must name a distinct Choice question in this request.".into(),
                );
            }
        }
        Ok(input)
    }

    fn bundle(&self, session: &str, settings: &JevSettings) -> DecisionBundle {
        DecisionBundle {
            session_id: session.into(),
            turn: 0,
            stage: "dynamic".into(),
            state: self.state.clone(),
            model: settings.requested_model_or_default().into(),
            questions: self.questions.clone(),
            question_categories: self
                .questions
                .keys()
                .map(|id| (id.clone(), DecisionCategory::Dynamic))
                .collect(),
        }
    }
}

fn sample_choice(probabilities: &BTreeMap<String, f64>, draw: f64) -> Option<String> {
    if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return None;
    }
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (option, probability) in probabilities {
        if *probability > 0.0 {
            last_positive = Some(option.clone());
            cumulative += probability;
            if draw < cumulative {
                return Some(option.clone());
            }
        }
    }
    // Validation bounds the sum; absorb only final floating-point rounding.
    last_positive
}

fn result_value(
    input: &Input,
    outcome: DecisionOutcome,
    latency_ms: u64,
    mut draw: impl FnMut() -> f64,
) -> Result<Value, String> {
    if !outcome.skips.is_empty() || outcome.records.len() != input.questions.len() {
        let reasons: BTreeSet<_> = outcome.skips.iter().map(|(_, reason)| *reason).collect();
        return Err(format!(
            "Jev Dynamic returned no complete valid result ({}).",
            reasons.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let answers: BTreeMap<_, _> = outcome
        .records
        .into_iter()
        .map(|record| (record.question_id, record.answer))
        .collect();
    let mut sampled = BTreeMap::new();
    for id in &input.sample {
        let Some(Answer::Choice { probabilities, .. }) = answers.get(id) else {
            return Err("Jev Dynamic sampling requires a validated Choice answer.".into());
        };
        let choice =
            sample_choice(probabilities, draw()).ok_or("Cannot sample Jev distribution")?;
        sampled.insert(id, json!({"choice":choice,"method":"jev_distribution"}));
    }
    Ok(json!({
        "category":"dynamic", "model":outcome.response_model,
        "answers":answers, "sampled":sampled, "usage":outcome.usage,
        "latency_ms":latency_ms, "attempts":outcome.attempts,
    }))
}

async fn execute(
    value: Value,
    session: String,
    signal: Option<CancellationToken>,
) -> Result<AgentToolResult, String> {
    let input = Input::parse(value)?;
    let store = JevSettingsStore::new(super::get_agent_dir());
    let settings = store.load();
    if !enabled(&settings, &session) {
        return Err("Jev Dynamic requires /jev feature dynamic on and /jev active (or compare-active / full-jev).".into());
    }
    if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
        return Err("Jev Dynamic cancelled.".into());
    }
    let generation = super::decision_policy_generation(&settings, &session);
    let credential_stamp = super::cheap_credential_stamp(&settings);
    let (transport, credential, fingerprint) = super::build_transport(&settings);
    if matches!(fingerprint.as_str(), "no_credential" | "unavailable") {
        return Err("Jev Dynamic has no usable Jev credential. Configure /jev key.".into());
    }
    let client = pi_jev::JevSystemOne::new(
        settings.effective_mode(&session),
        credential,
        transport,
        pi_jev::JevLimits::default(),
        Arc::new(pi_jev::JevStats::default()),
    )
    .map_err(|error| format!("Jev Dynamic unavailable: {}", error.kind()))?;
    let started = Instant::now();
    let outcome = pi_jev::dynamic::decide_scoped(
        &client,
        input.bundle(&session, &settings),
        || {
            let current = store.load();
            enabled(&current, &session)
                && super::decision_policy_generation(&current, &session) == generation
                && super::cheap_credential_stamp(&current) == credential_stamp
        },
        async move {
            match signal {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending::<()>().await,
            }
        },
    )
    .await
    .map_err(|reason| format!("Jev Dynamic: {reason}. No decision was returned."))?;
    let result = result_value(
        &input,
        outcome,
        started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        rand::random::<f64>,
    )?;
    let mut details = result.clone();
    details["source"] = json!("jev");
    Ok(text_tool_result(
        serde_json::to_string_pretty(&result).unwrap(),
        details,
    ))
}

fn definition() -> ToolDefinition {
    let entry = json!({"type":["string","object","array","null"]});
    let question = |kind: &str, criteria: Value, required: Vec<&str>| {
        json!({
            "type":"object", "additionalProperties":false,
            "properties":{"type":{"const":kind},"instructions":entry,"criteria":criteria},
            "required":required,
        })
    };
    ToolDefinition {
        name: TOOL_NAME.into(), label: "Jev decision".into(),
        description: "Ask Jev bounded, typed questions. Use Choice for selecting an option, Noul for a yes/no probability, Score for an ordered rating. Supply the relevant state and complete instructions/criteria; batch independent questions. When the user asks Jev to flip a coin or make a random choice, use a Choice and put its id in sample. Sampling draws from Jev's probabilities, not a guaranteed fair distribution. Return actual results; Jev does not generate explanations. This sends supplied state/questions to the configured Jev service.".into(),
        prompt_snippet: Some("Ask Jev ad hoc Choice, Noul or Score questions".into()),
        prompt_guidelines: Some(vec!["When asked to get Jev to decide, call jev_decide with the appropriate primitive and report its returned answer. Include only relevant supplied context. For random choices use sample and report the sampled result, retaining the raw distribution. Never present a model selection as a fair random coin toss.".into()]),
        parameters: json!({
            "type":"object","additionalProperties":false,"required":["state","questions"],
            "properties":{
                "state":{"type":["string","object","array"],"description":"Relevant evidence/context for these questions."},
                "questions":{"type":"object","minProperties":1,"maxProperties":64,"additionalProperties":{"oneOf":[
                    question("choice",json!({"type":"object","minProperties":1,"maxProperties":255,"additionalProperties":entry}),vec!["type","instructions","criteria"]),
                    question("noul",json!({"type":"object","properties":{"true":entry,"false":entry},"additionalProperties":false}),vec!["type","instructions"]),
                    question("score",json!({"type":"array","minItems":2,"maxItems":10,"items":entry}),vec!["type","instructions","criteria"])
                ]}},
                "sample":{"type":"array","items":{"type":"string"},"uniqueItems":true,"maxItems":64,"description":"Choice question ids whose returned distribution should be sampled locally. Omit for ordinary decisions."}
            }
        }),
        render_shell: None, replay_built_in_tool_name: None, prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(|_, value, signal, _, ctx| {
            Box::pin(execute(value, ctx.session_manager().get_session_id(), signal.or_else(|| ctx.signal())))
        }),
        render_call: None, render_result: None,
    }
}

fn sync_tools(extension: &SharedExtension, visible: bool) -> bool {
    let mut extension = extension.lock().unwrap_or_else(|p| p.into_inner());
    if extension.tools.contains_key(TOOL_NAME) == visible {
        return false;
    }
    if visible {
        let tool = RegisteredTool {
            definition: definition(),
            source_info: extension.source_info.clone(),
        };
        extension.tools.insert(TOOL_NAME.into(), tool);
    } else {
        extension.tools.remove(TOOL_NAME);
    }
    true
}

pub(crate) fn register(extensions: &mut Vec<SharedExtension>, runtime: ExtensionRuntime) {
    register_with_store(
        extensions,
        runtime,
        Arc::new(JevSettingsStore::new(super::get_agent_dir())),
    );
}

fn register_with_store(
    extensions: &mut Vec<SharedExtension>,
    runtime: ExtensionRuntime,
    store: Arc<JevSettingsStore>,
) {
    let extension = Arc::new(Mutex::new(Extension {
        path: SOURCE.into(),
        resolved_path: SOURCE.into(),
        source_info: crate::core::source_info::create_synthetic_source_info(
            SOURCE,
            &crate::core::source_info::SyntheticSourceInfoOptions {
                source: "internal".into(),
                ..Default::default()
            },
        ),
        handlers: HashMap::new(),
        tools: HashMap::new(),
        message_renderers: HashMap::new(),
        commands: indexmap::IndexMap::new(),
        flags: HashMap::new(),
        shortcuts: HashMap::new(),
    }));
    for event in [
        "session_start",
        "session_switch",
        "before_agent_start",
        "turn_start",
    ] {
        let weak = Arc::downgrade(&extension);
        let runtime = runtime.clone();
        let store = store.clone();
        let handler: ExtensionHandler = Arc::new(move |_, ctx| {
            let weak = weak.clone();
            let runtime = runtime.clone();
            let store = store.clone();
            Box::pin(async move {
                if let Some(extension) = weak.upgrade() {
                    let settings = store.load();
                    if sync_tools(
                        &extension,
                        enabled(&settings, &ctx.session_manager().get_session_id()),
                    ) {
                        runtime.refresh_tools();
                    }
                }
                None
            })
        });
        extension
            .lock()
            .unwrap()
            .handlers
            .insert(event.into(), vec![handler]);
    }
    extensions.push(extension);
}

#[cfg(test)]
#[path = "dynamic_tests.rs"]
mod tests;
