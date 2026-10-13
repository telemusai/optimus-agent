//! Opt-in SGLang final-payload budgeting. No prompt truncation or compaction policy.
//! Text counts use SGLang's chat renderer; image counts use a zero-output prefill,
//! because /tokenize does not include downstream multimodal expansion.
use super::{abort_error, resolve_header_timeout, Model, OpenAIClient, StreamError};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::sync::Mutex;

const MAX_COUNT_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const AUDIT_ENV: &str = "OPTIMUS_K3_REQUEST_AUDIT_PATH";
static AUDIT_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub(super) struct BudgetReceipt {
    prompt_tokens: u64,
    context_window: u64,
    requested_max_tokens: u64,
    max_tokens: u64,
    count_method: &'static str,
    count_elapsed_ms: u128,
}

fn fail(message: &'static str) -> StreamError { StreamError::new(message) }

fn positive_integer(value: f64) -> Result<u64, StreamError> {
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value > (1u64 << 53) as f64 {
        return Err(StreamError::invalid_request("SGLang budget: invalid configured token limit."));
    }
    Ok(value as u64)
}

fn output_field(body: &Value) -> Result<(&'static str, u64), StreamError> {
    let field = match (body.get("max_tokens"), body.get("max_completion_tokens")) {
        (Some(_), None) => "max_tokens",
        (None, Some(_)) => "max_completion_tokens",
        _ => return Err(StreamError::invalid_request("SGLang budget: exactly one output-token limit is required.")),
    };
    let requested = body[field].as_u64().filter(|n| *n > 0)
        .ok_or_else(|| StreamError::invalid_request("SGLang budget: output-token limit must be a positive integer."))?;
    Ok((field, requested))
}

// Mutable remote media could change between count and dispatch. Native images
// are immutable data URLs; reject other media rather than assert a false guarantee.
fn image_count(body: &Value) -> Result<usize, StreamError> {
    let messages = body.get("messages").and_then(Value::as_array)
        .ok_or_else(|| StreamError::invalid_request("SGLang budget: messages must be an array."))?;
    let mut count = 0;
    for message in messages {
        if let Some(parts) = message.get("content").and_then(Value::as_array) {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("image_url") => {
                        let url = part.pointer("/image_url/url").and_then(Value::as_str).unwrap_or_default();
                        if !url.starts_with("data:image/") || !url.contains(";base64,") {
                            return Err(StreamError::invalid_request("SGLang budget: exact image counting requires an immutable data URL."));
                        }
                        count += 1;
                    }
                    Some("text") => {}
                    _ => return Err(StreamError::invalid_request("SGLang budget: unsupported multimodal content cannot be counted exactly.")),
                }
            }
        }
    }
    Ok(count)
}

fn count_payload(body: &Value, images: usize) -> Value {
    let mut count = body.clone();
    let object = count.as_object_mut().expect("validated request object");
    // TokenizeRequest permits extras; stream:true would select an unsupported
    // streaming handler. These fields do not affect the chat prompt renderer.
    object.remove("stream");
    object.remove("stream_options");
    // Chat validation also runs for /tokenize, before rendering. Do not let an
    // oversized requested output reject counting before we can clamp it. Output
    // limits are not prompt-rendering inputs in SGLang's chat processor.
    object.remove("max_completion_tokens");
    object.insert("max_tokens".into(), json!(0));
    if images > 0 {
        object.insert("stream".into(), json!(false));
        object.insert("return_prompt_token_ids".into(), json!(true));
    }
    count
}

fn parse_count(value: &Value, images: usize, context: u64) -> Result<(u64, u64), StreamError> {
    if images > 0 {
        if value.pointer("/usage/completion_tokens").and_then(Value::as_u64) != Some(0) {
            return Err(fail("SGLang budget: image count prefill did not confirm zero generated tokens."));
        }
        // Reject providers which report zero usage but still return output.
        let choices = value.get("choices").and_then(Value::as_array)
            .ok_or_else(|| fail("SGLang budget: invalid image count response."))?;
        for choice in choices {
            let message = &choice["message"];
            for key in ["content", "reasoning_content", "reasoning", "tool_calls"] {
                if let Some(v) = message.get(key) {
                    let empty = v.is_null() || v.as_str() == Some("") || v.as_array().is_some_and(Vec::is_empty);
                    if !empty { return Err(fail("SGLang budget: image count prefill unexpectedly generated output.")); }
                }
            }
        }
        let reported = value.pointer("/usage/prompt_tokens").and_then(Value::as_u64)
            .filter(|n| *n > 0).ok_or_else(|| fail("SGLang budget: missing image prompt usage."))?;
        // SGLang's K3 billed usage excludes three generation-stub input tokens.
        // Returned IDs are captured AFTER multimodal processing and retain the
        // complete model input. Never infer framing from a model alias or usage.
        let tokens = choices.first().and_then(|c| c.get("prompt_token_ids")).and_then(Value::as_array)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| fail("SGLang budget: image prefill did not return raw prompt token IDs."))?;
        if tokens.iter().any(|id| id.as_i64().is_none() && id.as_u64().is_none())
            || choices.iter().any(|c| c.get("prompt_token_ids").and_then(Value::as_array) != Some(tokens))
        {
            return Err(fail("SGLang budget: invalid or inconsistent image prompt token IDs."));
        }
        let count = tokens.len() as u64;
        if count < reported {
            return Err(fail("SGLang budget: raw image prompt count is smaller than reported usage."));
        }
        Ok((count, context))
    } else {
        let count = value.get("count").and_then(Value::as_u64)
            .ok_or_else(|| fail("SGLang budget: invalid tokenizer count."))?;
        let tokens = value.get("tokens").and_then(Value::as_array)
            .ok_or_else(|| fail("SGLang budget: tokenizer did not return token IDs."))?;
        if tokens.len() as u64 != count || tokens.iter().any(|v| v.as_u64().is_none()) {
            return Err(fail("SGLang budget: tokenizer count and token IDs disagree."));
        }
        let server_context = value.get("max_model_len").and_then(Value::as_u64)
            .filter(|n| *n > 0).ok_or_else(|| fail("SGLang budget: missing tokenizer context limit."))?;
        Ok((count, context.min(server_context)))
    }
}

async fn request_count(
    client: &OpenAIClient, body: &Value, images: usize,
    signal: Option<&tokio_util::sync::CancellationToken>, timeout_ms: Option<f64>,
    observer: Option<&crate::types::OnStreamObservation>,
) -> Result<Value, StreamError> {
    let suffix = if images > 0 { "chat/completions" } else { "tokenize" };
    let url = format!("{}/{suffix}", client.base_url.trim_end_matches('/'));
    // Same existing auth and origin. Never follow a credential-bearing redirect.
    let lease = super::http_client_pool::acquire(&url, super::http_client_pool::RedirectPolicy::Count)
        .map_err(|_| fail("SGLang budget: cannot initialize count client."))?;
    super::observe_local_phase(observer, lease.phase);
    let http = lease.client;
    let mut headers = indexmap::IndexMap::new();
    headers.insert("Authorization".to_string(), Some(format!("Bearer {}", client.api_key)));
    headers.insert("Content-Type".to_string(), Some("application/json".to_string()));
    headers.insert("Accept".to_string(), Some("application/json".to_string()));
    for (name, value) in &client.default_headers { headers.insert(name.clone(), value.clone()); }
    let mut builder = http.post(url);
    for (name, value) in &headers {
        if let Some(value) = value { builder = builder.header(name, value); }
    }
    super::observe_local_phase(observer, "count_serialize_start");
    let request = builder.json(&count_payload(body, images));
    super::observe_local_phase(observer, "count_serialize_complete");
    let operation = async {
        super::observe_local_phase(observer, "count_send_start");
        let mut response = request.send().await.map_err(|_| fail("SGLang budget: count request failed."))?;
        super::observe_local_phase(observer, "count_headers_complete");
        if !response.status().is_success() {
            // Do not reflect provider bodies, URLs, or headers into a transcript.
            return Err(StreamError::count_http_error(response.status().as_u16(), response.headers()));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| fail("SGLang budget: count response read failed."))? {
            if bytes.len().saturating_add(chunk.len()) > MAX_COUNT_RESPONSE_BYTES {
                return Err(fail("SGLang budget: count response exceeds its size limit."));
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| fail("SGLang budget: invalid count JSON."))
    };
    // Count responses are finite JSON: unlike generation SSE this deadline
    // intentionally covers the complete preflight, including its response body.
    let timed = tokio::time::timeout(resolve_header_timeout(timeout_ms), operation);
    let result = if let Some(signal) = signal {
        tokio::select! { biased;
            _ = signal.cancelled() => return Err(abort_error()),
            result = timed => result,
        }
    } else { timed.await };
    result.map_err(|_| fail("SGLang budget: count request timed out."))?
}

fn apply_budget(body: &mut Value, model: &Model, prompt: u64, context: u64) -> Result<(u64, u64), StreamError> {
    let (field, requested) = output_field(body)?;
    let ceiling = positive_integer(model.max_tokens)?;
    // Strict inequality, including one token of slack; no message rewriting.
    let remaining = context.checked_sub(prompt).and_then(|n| n.checked_sub(1))
        .filter(|n| *n > 0).ok_or_else(|| StreamError::invalid_request("SGLang budget: prompt leaves no output room within context."))?;
    let output = requested.min(ceiling).min(remaining);
    body[field] = json!(output);
    Ok((requested, output))
}

#[cfg(test)]
pub(super) async fn guard(
    model: &Model, client: &OpenAIClient, body: &mut Value,
    signal: Option<&tokio_util::sync::CancellationToken>, timeout_ms: Option<f64>,
) -> Result<Option<BudgetReceipt>, StreamError> {
    guard_observed(model, client, body, signal, timeout_ms, None).await
}

pub(super) async fn guard_observed(
    model: &Model, client: &OpenAIClient, body: &mut Value,
    signal: Option<&tokio_util::sync::CancellationToken>, timeout_ms: Option<f64>,
    observer: Option<&crate::types::OnStreamObservation>,
) -> Result<Option<BudgetReceipt>, StreamError> {
    if model.compat_completions().and_then(|c| c.sglang_token_budget) != Some(true) { return Ok(None); }
    if body.get("model").and_then(Value::as_str) != Some(model.id.as_str()) {
        return Err(StreamError::invalid_request("SGLang budget: payload hook changed the configured model identity."));
    }
    let context = positive_integer(model.context_window)?;
    positive_integer(model.max_tokens)?;
    output_field(body)?;
    let images = image_count(body)?;
    let started = std::time::Instant::now();
    super::observe_local_phase(observer, "count_start");
    let result = request_count(client, body, images, signal, timeout_ms, observer).await;
    super::observe_local_phase(observer, if result.is_ok() { "count_complete" } else { "count_failed" });
    let value = result?;
    let (prompt_tokens, context_window) = parse_count(&value, images, context)?;
    let (requested_max_tokens, max_tokens) = apply_budget(body, model, prompt_tokens, context_window)?;
    Ok(Some(BudgetReceipt {
        prompt_tokens, context_window, requested_max_tokens, max_tokens,
        count_method: if images > 0 { "sglang_image_zero_output_prefill_token_ids" } else { "sglang_chat_tokenize" },
        count_elapsed_ms: started.elapsed().as_millis(),
    }))
}

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<_> = object.keys().collect(); keys.sort();
            Value::Object(keys.into_iter().map(|key| (key.clone(), canonical(&object[key]))).collect())
        }
        Value::Array(array) => Value::Array(array.iter().map(canonical).collect()),
        other => other.clone(),
    }
}
fn summary(value: Option<&Value>) -> Value {
    let Some(value) = value else { return json!({"present":false}); };
    let bytes = if let Some(s) = value.as_str() { s.as_bytes().to_vec() } else { canonical(value).to_string().into_bytes() };
    json!({"present":true,"utf8_bytes":bytes.len(),"sha256":sha(&bytes)})
}
fn identifier(value: Option<&Value>) -> Value {
    let text = value.and_then(Value::as_str).unwrap_or_default();
    json!({"present":value.is_some(),"sha256":sha(text.as_bytes())})
}
fn safe_name(value: Option<&Value>) -> Value {
    match value.and_then(Value::as_str) {
        Some(s) if !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_-.:".contains(&b)) => json!(s),
        _ => Value::Null,
    }
}
fn kind(v: &Value) -> &'static str {
    match v { Value::Null=>"null",Value::Bool(_)=>"boolean",Value::Number(_)=>"number",Value::String(_)=>"string",Value::Array(_)=>"array",Value::Object(_)=>"object" }
}
fn tool_summary(call: &Value) -> Value {
    let args = call.pointer("/function/arguments");
    let parsed = args.and_then(Value::as_str).and_then(|s| serde_json::from_str::<Value>(s).ok());
    let semantic = parsed.as_ref().or(args);
    let keys: Vec<Value> = semantic.and_then(Value::as_object).map(|o| o.iter().map(|(key, value)|
        json!({"key_sha256":sha(key.as_bytes()),"type":kind(value)})).collect()).unwrap_or_default();
    json!({"id":identifier(call.get("id")),"name":safe_name(call.pointer("/function/name")),
        "name_sha256":identifier(call.pointer("/function/name")),"arguments":summary(semantic),"argument_types":keys})
}

fn audit_value(model: &Model, body: &Value, receipt: &BudgetReceipt, serialized: &[u8]) -> Value {
    let messages: Vec<Value> = body.get("messages").and_then(Value::as_array).into_iter().flatten().map(|m| {
        let role = match m.get("role").and_then(Value::as_str) { Some(s @ ("system"|"developer"|"user"|"assistant"|"tool")) => s, _=>"other" };
        let calls: Vec<Value> = m.get("tool_calls").and_then(Value::as_array).into_iter().flatten().map(tool_summary).collect();
        let parts: Vec<&Value> = m.get("content").and_then(Value::as_array).into_iter().flatten().collect();
        let images = parts.iter().filter(|p| p.get("type").and_then(Value::as_str)==Some("image_url")).count();
        let image_bytes: usize = parts.iter().filter_map(|p| p.pointer("/image_url/url").and_then(Value::as_str)).map(str::len).sum();
        json!({"role":role,"content":summary(m.get("content")),"reasoning_content":summary(m.get("reasoning_content")),
            "reasoning":summary(m.get("reasoning")),"reasoning_details":summary(m.get("reasoning_details")),
            "tool_calls":calls,"tool_call_id":identifier(m.get("tool_call_id")),"image_count":images,"image_url_bytes":image_bytes})
    }).collect();
    let effort = match body.get("reasoning_effort").and_then(Value::as_str) {
        Some(s @ ("none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"max")) => Some(s), _=>None,
    };
    let tools: Vec<Value> = body.get("tools").and_then(Value::as_array).into_iter().flatten().map(|t|
        json!({"name":safe_name(t.pointer("/function/name")),"definition":summary(Some(t))})).collect();
    json!({"schema":"optimus.sglang.dispatch-audit.v1","phase":"prepared_http_request","request_id":uuid::Uuid::new_v4().to_string(),
        "pid":std::process::id(),"timestamp_ms":crate::utils::now_ms(),"provider":model.provider,"model":model.id,
        "reasoning_effort":effort,"max_tokens":receipt.max_tokens,"requested_max_tokens":receipt.requested_max_tokens,
        "prompt_tokens":receipt.prompt_tokens,"context_window":receipt.context_window,"count_method":receipt.count_method,
        "count_elapsed_ms":receipt.count_elapsed_ms,"count_http_requests":1,
        "serialized_body_bytes":serialized.len(),"serialized_body_sha256":sha(serialized),"messages":messages,"tools":tools})
}

pub(super) fn audit(model: &Model, body: &Value, receipt: &BudgetReceipt, serialized: &[u8]) -> Result<(), StreamError> {
    let Some(path) = std::env::var_os(AUDIT_ENV) else { return Ok(()); };
    if path.is_empty() { return Err(fail("SGLang audit: output path is empty.")); }
    let mut bytes = serde_json::to_vec(&audit_value(model, body, receipt, serialized))
        .map_err(|_| fail("SGLang audit: metadata serialization failed."))?;
    bytes.push(b'\n');
    let _lock = AUDIT_LOCK.lock().map_err(|_| fail("SGLang audit: output lock failed."))?;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)
        .map_err(|_| fail("SGLang audit: cannot open output file."))?;
    // One append operation per record. Never write the actual payload or headers.
    let written = file.write(&bytes).map_err(|_| fail("SGLang audit: append failed."))?;
    if written != bytes.len() { return Err(fail("SGLang audit: incomplete append.")); }
    file.flush().map_err(|_| fail("SGLang audit: flush failed."))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> Model { Model { context_window:1_048_576.0,max_tokens:262_144.0,..Default::default() } }
    fn payload() -> Value { json!({"model":"kimi-k3","stream":true,"stream_options":{"include_usage":true},"max_tokens":131072,"reasoning_effort":"max","messages":[{"role":"user","content":"fixture"}]}) }

    #[test]
    fn only_proved_local_rejections_are_invalid_request() {
        use crate::utils::stream_failure::{extract_stream_failure_info, ThrownStreamError};
        let info = |error: StreamError| extract_stream_failure_info(&ThrownStreamError::Value(&error.value));
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, 1.5] {
            assert_eq!(info(positive_integer(invalid).unwrap_err()).kind, "invalid_request");
        }
        for body in [json!({}), json!({"max_tokens":0}), json!({"max_tokens":1,"max_completion_tokens":1})] {
            assert_eq!(info(output_field(&body).unwrap_err()).kind, "invalid_request");
        }
        let mut remote_media = payload();
        remote_media["messages"][0]["content"] = json!([{"type":"image_url","image_url":{"url":"https://secret.invalid/mutable"}}]);
        let error = image_count(&remote_media).unwrap_err();
        assert!(!error.value.to_string().contains("secret.invalid"));
        assert_eq!(info(error).kind, "invalid_request");
        assert_eq!(info(apply_budget(&mut payload(), &model(), 109, 110).unwrap_err()).kind, "invalid_request");
        for error in [parse_count(&json!({}), 0, 110).unwrap_err(), parse_count(&json!({}), 1, 110).unwrap_err(), fail("SGLang budget: count request failed.")] {
            let info = info(error);
            assert_eq!(info.kind, "unknown");
            assert_eq!(info.status, None);
        }
    }

    #[test]
    fn clamp_is_strict_and_preserves_history() {
        for (prompt, requested, expected) in [(100,131072,131072),(100,999999,262144),(1_000_000,131072,48575),(1_048_574,2,1)] {
            let mut body=payload();body["max_tokens"]=json!(requested);let history=body["messages"].clone();
            assert_eq!(apply_budget(&mut body,&model(),prompt,1_048_576).unwrap().1,expected);
            assert_eq!(body["messages"],history);assert!(prompt+expected<1_048_576);
        }
        for prompt in [1_048_575,1_048_576,u64::MAX] { assert!(apply_budget(&mut payload(),&model(),prompt,1_048_576).is_err()); }
    }
    #[test]
    fn count_clone_keeps_all_prompt_fields_and_never_mutates_dispatch() {
        let mut body=payload();body["tools"]=json!([{"type":"function","function":{"name":"read"}}]);
        body["messages"]=json!([{"role":"assistant","reasoning_content":"private","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"result"}]);
        let original=body.clone();let text=count_payload(&body,0);let image=count_payload(&body,1);
        assert!(text.get("stream").is_none());assert!(text.get("stream_options").is_none());assert_eq!(text["max_tokens"],0);
        assert_eq!(image["stream"],false);assert_eq!(image["max_tokens"],0);
        assert_eq!(image["return_prompt_token_ids"],true);assert!(text.get("return_prompt_token_ids").is_none());
        assert!(body.get("return_prompt_token_ids").is_none());
        for key in ["messages","tools","reasoning_effort","model"] { assert_eq!(text[key],body[key]);assert_eq!(image[key],body[key]); }
        assert_eq!(body,original);
    }
    #[test]
    fn exact_counts_reject_missing_inconsistent_and_generated_output() {
        assert_eq!(parse_count(&json!({"count":2,"tokens":[1,2],"max_model_len":1048576}),0,1048576).unwrap(),(2,1048576));
        for value in [json!({"count":2,"tokens":[1],"max_model_len":1048576}),json!({"count":2}),json!({"count":2,"tokens":[-1,2],"max_model_len":1048576})] { assert!(parse_count(&value,0,1048576).is_err()); }
        let image=json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":"","reasoning_content":null},"prompt_token_ids":vec![1;106]}]});
        assert_eq!(parse_count(&image,1,1048576).unwrap().0,106);
        let mut bad=image.clone();bad["usage"]["completion_tokens"]=json!(1);assert!(parse_count(&bad,1,1048576).is_err());
        bad=image;bad["choices"][0]["message"]["reasoning_content"]=json!("not zero");assert!(parse_count(&bad,1,1048576).is_err());
    }
    #[test]
    fn media_requires_immutable_supported_input() {
        let mut body=payload();body["messages"][0]["content"]=json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,fixture"}}]);assert_eq!(image_count(&body).unwrap(),1);
        body["messages"][0]["content"][0]["image_url"]["url"]=json!("https://mutable.invalid/image");assert!(image_count(&body).is_err());
    }
    #[test]
    fn audit_is_allowlisted_and_semantic_argument_hash_is_order_independent() {
        let secret="fixture-SECRET-never-copy";let mut body=payload();body["messages"]=json!([{"role":"assistant","content":secret,"reasoning_content":secret,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read","arguments":"{\"a\":1,\"b\":true}"}}]},{"role":"tool","tool_call_id":"call-1","content":secret}]);body["extra_secret"]=json!(secret);
        let receipt=BudgetReceipt{prompt_tokens:10,context_window:1048576,requested_max_tokens:131072,max_tokens:131072,count_method:"sglang_chat_tokenize",count_elapsed_ms:1};
        let serialized=serde_json::to_vec(&body).unwrap();let before=body.clone();let value=audit_value(&model(),&body,&receipt,&serialized);let text=value.to_string();
        assert!(!text.contains(secret));assert!(!text.contains("extra_secret"));assert_eq!(body,before);assert_eq!(value["reasoning_effort"],"max");assert_eq!(value["serialized_body_sha256"],sha(&serialized));
        let first=&value["messages"][0]["tool_calls"][0]["arguments"]["sha256"];
        body["messages"][0]["tool_calls"][0]["function"]["arguments"]=json!("{\"b\":true,\"a\":1}");
        let second=audit_value(&model(),&body,&receipt,&serde_json::to_vec(&body).unwrap());assert_eq!(*first,second["messages"][0]["tool_calls"][0]["arguments"]["sha256"]);
        assert_eq!(value["messages"][0]["tool_calls"][0]["id"]["sha256"],value["messages"][1]["tool_call_id"]["sha256"]);
    }
    #[tokio::test]
    async fn dispatch_guard_counts_hook_payload_then_clamps_actual_wire() {
        use super::super::{stream_openai_completions, OpenAICompletionsOptions};
        use crate::types::{Compat, Context, OpenAICompletionsCompat, StreamOptions};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener=tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,0)).await.unwrap();
        let address=listener.local_addr().unwrap();
        let server=tokio::spawn(async move {
            let mut observed=Vec::new();
            for index in 0..2 {
                let (mut socket,_)=listener.accept().await.unwrap();
                let mut bytes=Vec::new();
                let (head,body)=loop {
                    let mut chunk=[0;4096];let n=socket.read(&mut chunk).await.unwrap();assert_ne!(n,0);bytes.extend_from_slice(&chunk[..n]);
                    if let Some(end)=bytes.windows(4).position(|w|w==b"\r\n\r\n") {
                        let head=std::str::from_utf8(&bytes[..end]).unwrap();
                        let size:usize=head.lines().filter_map(|l|l.split_once(':')).find(|(k,_)|k.eq_ignore_ascii_case("content-length")).unwrap().1.trim().parse().unwrap();
                        if bytes.len()>=end+4+size { break (head.to_string(),serde_json::from_slice::<Value>(&bytes[end+4..end+4+size]).unwrap()); }
                    }
                };
                assert!(head.starts_with(if index==0 {"POST /v1/tokenize "} else {"POST /v1/chat/completions "}));
                assert_eq!(body["messages"][0]["content"],"after-hook fixture");
                assert_eq!(body["reasoning_effort"],"max");
                if index==0 { assert!(body.get("stream").is_none());assert_eq!(body["max_tokens"],0); }
                else { assert_eq!(body["stream"],true);assert_eq!(body["max_tokens"],23); }
                observed.push(body);
                let (content_type,response)=if index==0 {("application/json",json!({"count":1000,"tokens":vec![1;1000],"max_model_len":1024}).to_string())}
                else {("text/event-stream","data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n".to_string())};
                let reply=format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len());
                socket.write_all(reply.as_bytes()).await.unwrap();socket.shutdown().await.unwrap();
            }
            assert_eq!(observed[0]["messages"],observed[1]["messages"]);
        });
        let mut m=model();m.id="kimi-k3".into();m.provider="offline-fixture".into();m.api="openai-completions".into();m.context_window=1024.0;m.base_url=format!("http://{address}/v1");m.reasoning=true;
        m.compat=Some(Compat::Completions(OpenAICompletionsCompat{sglang_token_budget:Some(true),max_tokens_field:Some("max_tokens".into()),supports_reasoning_effort:Some(true),..Default::default()}));
        let options=OpenAICompletionsOptions{reasoning_effort:Some("max".into()),stream:StreamOptions{api_key:Some("offline-fixture-key".into()),max_tokens:Some(900.0),on_payload:Some(std::sync::Arc::new(|mut body,_|Box::pin(async move {body["messages"]=json!([{"role":"user","content":"after-hook fixture"}]);body["max_tokens"]=json!(999999);Some(body)}))),..Default::default()},..Default::default()};
        let stream=stream_openai_completions(&m,&Context::default(),Some(options));
        let result=tokio::time::timeout(std::time::Duration::from_secs(5),stream.result()).await.unwrap();
        assert_ne!(result.stop_reason,"error","{:?}",result.error_message);
        tokio::time::timeout(std::time::Duration::from_secs(5),server).await.unwrap().unwrap();
    }
    #[tokio::test]
    async fn opt_in_absent_and_invalid_body_never_need_network() {
        let client=OpenAIClient{api_key:"unused".into(),base_url:"http://127.0.0.1:1/v1".into(),default_headers:Default::default()};
        let mut m=model();let mut body=Value::Null;
        assert!(guard(&m,&client,&mut body,None,Some(1.0)).await.unwrap().is_none());
        m.compat=Some(crate::types::Compat::Completions(crate::types::OpenAICompletionsCompat{sglang_token_budget:Some(true),..Default::default()}));
        let err=guard(&m,&client,&mut body,None,Some(1.0)).await.unwrap_err();assert!(err.message.contains("model identity"));
        let info = crate::utils::stream_failure::extract_stream_failure_info(&crate::utils::stream_failure::ThrownStreamError::Value(&err.value));
        assert_eq!(info.kind, "invalid_request"); assert!(info.status.is_none());
    }

    #[test]
    fn image_raw_ids_not_billed_usage_control_strict_near_ceiling_budget() {
        let image=json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":""},"prompt_token_ids":vec![1;106]}]});
        let (raw,context)=parse_count(&image,1,110).unwrap();assert_eq!(raw,106);
        let mut body=payload();assert_eq!(apply_budget(&mut body,&model(),raw,context).unwrap().1,3);
        assert!(raw+body["max_tokens"].as_u64().unwrap()<context);
        assert!(apply_budget(&mut payload(),&model(),raw,107).is_err());
        // A usage-based count of103 would incorrectly allow6 output tokens here.
        assert_ne!(body["max_tokens"],json!(110-103-1));
    }
    #[test]
    fn image_raw_ids_missing_malformed_or_inconsistent_fail_closed() {
        let good=json!({"usage":{"prompt_tokens":103,"completion_tokens":0},"choices":[{"message":{"content":""},"prompt_token_ids":vec![1;106]}]});
        let mut bad=good.clone();bad["choices"][0].as_object_mut().unwrap().remove("prompt_token_ids");assert!(parse_count(&bad,1,110).is_err());
        bad=good.clone();bad["choices"][0]["prompt_token_ids"]=json!([]);assert!(parse_count(&bad,1,110).is_err());
        bad=good.clone();bad["choices"][0]["prompt_token_ids"][0]=json!("not-an-id");assert!(parse_count(&bad,1,110).is_err());
        bad=good.clone();bad["choices"][0]["prompt_token_ids"]=json!(vec![1;102]);assert!(parse_count(&bad,1,110).is_err());
        bad=good.clone();let mut second=bad["choices"][0].clone();second["prompt_token_ids"][0]=json!(2);bad["choices"].as_array_mut().unwrap().push(second);assert!(parse_count(&bad,1,110).is_err());
        bad=good;bad["choices"]=json!([]);assert!(parse_count(&bad,1,110).is_err());
    }

}
