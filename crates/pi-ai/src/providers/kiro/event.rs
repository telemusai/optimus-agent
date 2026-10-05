//! Decode AWS event-stream envelopes before parsing JSON (never scan binary data for braces).
use serde_json::Value;
use std::collections::HashMap;

const MAX_FRAME: usize = 16 * 1024 * 1024;

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

pub struct Frame {
    pub headers: HashMap<String, String>,
    pub payload: FramePayload,
}

/// One frame's JSON payload: the typed shape when the payload fits it, the
/// parsed `Value` otherwise (non-object payloads). Both preserve the
/// consumer's exact read semantics.
pub enum FramePayload {
    Typed(TypedKiroPayload),
    Value(Value),
}

/// A Kiro frame payload in the consumer's own shape: the members
/// `State::process` reads, parsed without building a `Value` DOM. Presence
/// booleans keep `Value::get(..).is_some()` semantics (a `null` member
/// still counts as present); wrong-typed members read as absent.
#[derive(serde::Deserialize, Default)]
pub struct TypedKiroPayload {
    #[serde(rename = "error", default, deserialize_with = "crate::utils::typed_json::any_present")]
    pub error_present: bool,
    #[serde(rename = "Error", default, deserialize_with = "crate::utils::typed_json::any_present")]
    pub error_capital_present: bool,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_string")]
    pub content: Option<String>,
    #[serde(rename = "contextUsagePercentage", default, deserialize_with = "crate::utils::typed_json::optional_f64")]
    pub context_usage_percentage: Option<f64>,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_value")]
    pub usage: Option<Value>,
    #[serde(rename = "inputTokens", default, deserialize_with = "crate::utils::typed_json::optional_u64")]
    pub input_tokens: Option<u64>,
    #[serde(rename = "outputTokens", default, deserialize_with = "crate::utils::typed_json::optional_u64")]
    pub output_tokens: Option<u64>,
    #[serde(rename = "toolUseId", default, deserialize_with = "crate::utils::typed_json::optional_string")]
    pub tool_use_id: Option<String>,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_string")]
    pub name: Option<String>,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_value")]
    pub input: Option<Value>,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_bool")]
    pub stop: Option<bool>,
    #[serde(default, deserialize_with = "crate::utils::typed_json::optional_string")]
    pub message: Option<String>,
}

fn parse_payload(payload: &[u8]) -> Result<FramePayload, String> {
    let looks_like_object = payload
        .iter()
        .find(|byte| !byte.is_ascii_whitespace())
        .map(|byte| *byte == b'{')
        .unwrap_or(false);
    if looks_like_object {
        if let Ok(typed) = serde_json::from_slice::<TypedKiroPayload>(payload) {
            return Ok(FramePayload::Typed(typed));
        }
    }
    let value = serde_json::from_slice::<Value>(payload).map_err(|_| "Invalid JSON in Kiro event")?;
    Ok(FramePayload::Value(value))
}

pub fn decode(buffer: &mut Vec<u8>) -> Result<Vec<Frame>, String> {
    let mut frames = vec![];
    let mut offset = 0;
    while buffer.len() - offset >= 12 {
        let frame = &buffer[offset..];
        let total = u32::from_be_bytes(frame[0..4].try_into().unwrap()) as usize;
        let header_len = u32::from_be_bytes(frame[4..8].try_into().unwrap()) as usize;
        if !(16..=MAX_FRAME).contains(&total) || header_len > total - 16 {
            return Err("Invalid Kiro event-stream frame length".into());
        }
        if crc32(&frame[..8]) != u32::from_be_bytes(frame[8..12].try_into().unwrap()) {
            return Err("Invalid Kiro event-stream prelude checksum".into());
        }
        if frame.len() < total {
            break;
        }
        if crc32(&frame[..total - 4])
            != u32::from_be_bytes(frame[total - 4..total].try_into().unwrap())
        {
            return Err("Invalid Kiro event-stream checksum".into());
        }
        let mut headers = HashMap::new();
        let header = &frame[12..12 + header_len];
        let mut cursor = 0;
        while cursor < header.len() {
            let size = header[cursor] as usize;
            cursor += 1;
            if cursor + size + 1 > header.len() {
                return Err("Malformed Kiro event headers".into());
            }
            let name = std::str::from_utf8(&header[cursor..cursor + size])
                .map_err(|_| "Invalid Kiro event header name")?
                .to_owned();
            cursor += size;
            let kind = header[cursor];
            cursor += 1;
            let size = match kind {
                0 | 1 => 0,
                2 => 1,
                3 => 2,
                4 => 4,
                5 | 8 => 8,
                9 => 16,
                6 | 7 => {
                    if cursor + 2 > header.len() {
                        return Err("Truncated Kiro event header".into());
                    }
                    let size =
                        u16::from_be_bytes(header[cursor..cursor + 2].try_into().unwrap()) as usize;
                    cursor += 2;
                    size
                }
                _ => return Err("Unknown Kiro event header type".into()),
            };
            if cursor + size > header.len() {
                return Err("Truncated Kiro event header value".into());
            }
            if kind == 7 {
                headers.insert(
                    name,
                    std::str::from_utf8(&header[cursor..cursor + size])
                        .map_err(|_| "Invalid Kiro event header value")?
                        .to_owned(),
                );
            }
            cursor += size;
        }
        let payload = parse_payload(&frame[12 + header_len..total - 4])?;
        frames.push(Frame { headers, payload });
        offset += total;
    }
    buffer.drain(..offset);
    Ok(frames)
}

#[cfg(test)]
mod typed_payload_tests {
	use super::{parse_payload, FramePayload};
	use serde_json::Value;

	/// The typed reads must equal the Value reads for every payload the
	/// Value consumer accepted (wrong-typed members read as absent, nulls
	/// keep presence semantics, non-objects fall back to Value).
	#[test]
	fn typed_payload_reads_match_the_value_reads() {
		let payloads: Vec<&[u8]> = vec![
			br#"{"content":"Hi"}"#,
			br#"{"content":null}"#,
			br#"{"content":42}"#,
			br#"{"usage":{"inputTokens":11,"outputTokens":7}}"#,
			br#"{"usage":null,"inputTokens":3,"outputTokens":4}"#,
			br#"{"inputTokens":3,"outputTokens":4}"#,
			br#"{"usage":"nope"}"#,
			br#"{"contextUsagePercentage":0.1}"#,
			br#"{"contextUsagePercentage":750.0}"#,
			br#"{"toolUseId":"t-1","name":"read"}"#,
			br#"{"toolUseId":5,"name":""}"#,
			br#"{"input":"{\"a\":1}"}"#,
			br#"{"input":null}"#,
			br#"{"input":{"a":1}}"#,
			br#"{"stop":true}"#,
			br#"{"stop":"true"}"#,
			br#"{"error":null}"#,
			br#"{"Error":"boom"}"#,
			br#"{"message":"exception detail"}"#,
			br#"{"unknown":"member"}"#,
			b"[1, 2]",
			b"\"just text\"",
		];
		for payload in &payloads {
			let frame = match parse_payload(payload) {
				Ok(FramePayload::Typed(typed)) => typed,
				Ok(FramePayload::Value(_)) => continue, // fallback: Value path is the old consumer
				Err(_) => panic!("payload rejected: {payload:?}"),
			};
			let value: Value = serde_json::from_slice(payload).unwrap();
			assert_eq!(
				frame.error_present || frame.error_capital_present,
				value.get("error").is_some() || value.get("Error").is_some(),
				"error presence mismatch on {payload:?}"
			);
			assert_eq!(
				frame.content.as_deref(),
				value["content"].as_str(),
				"content mismatch on {payload:?}"
			);
			assert_eq!(
				frame.context_usage_percentage,
				value["contextUsagePercentage"].as_f64(),
				"contextUsagePercentage mismatch on {payload:?}"
			);
			assert_eq!(frame.usage.as_ref(), value.get("usage"), "usage mismatch on {payload:?}");
			let usage = frame.usage.as_ref();
			let (input, out) = match usage {
				Some(usage) => (usage["inputTokens"].as_u64(), usage["outputTokens"].as_u64()),
				None => (frame.input_tokens, frame.output_tokens),
			};
			let old_usage = value.get("usage").unwrap_or(&value);
			assert_eq!(input, old_usage["inputTokens"].as_u64(), "inputTokens mismatch on {payload:?}");
			assert_eq!(out, old_usage["outputTokens"].as_u64(), "outputTokens mismatch on {payload:?}");
			assert_eq!(
				frame.tool_use_id.as_deref(),
				value["toolUseId"].as_str(),
				"toolUseId mismatch on {payload:?}"
			);
			assert_eq!(frame.name.as_deref(), value["name"].as_str(), "name mismatch on {payload:?}");
			assert_eq!(frame.input.as_ref(), value.get("input"), "input mismatch on {payload:?}");
			assert_eq!(frame.stop, value["stop"].as_bool(), "stop mismatch on {payload:?}");
			assert_eq!(
				frame.message.as_deref(),
				value["message"].as_str(),
				"message mismatch on {payload:?}"
			);
		}
	}
}
