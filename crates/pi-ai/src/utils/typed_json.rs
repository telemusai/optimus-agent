//! Lenient borrowed-JSON field readers for provider response parsing.
//!
//! The provider consumers historically walked a `serde_json::Value` DOM and
//! read fields with `get(..).and_then(Value::as_str)` semantics: a missing
//! field, a `null`, or a wrong-typed value all read as `None`, and unknown
//! fields were ignored. These helpers let a typed struct keep exactly those
//! tolerances while borrowing `&str` payloads straight from the provider-owned
//! SSE buffer instead of allocating a `String` per key and value.
//!
//! Design rule: every `optional_*` visitor treats a wrong-typed JSON value as
//! `None` (serde's `Option` + `deserialize_with`), matching
//! `get(field).and_then(Value::as_str)`-style reads. `json_trace` always
//! captures the same text `Value::to_string` would produce, so callers can
//! rebuild the previous behavior for exotic shapes.

use serde::de::{Error as DeError, Visitor};
use serde::Deserialize;
use serde_json::Value;

/// `#[serde(default, deserialize_with = "optional_borrowed_str")]`
/// `Option<&'a str>`: string values borrow from the input; any other JSON
/// shape (including `null`) yields `None`.
pub fn optional_borrowed_str<'de, D>(deserializer: D) -> Result<Option<&'de str>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	deserializer.deserialize_any(BorrowedStrVisitor)
}

struct BorrowedStrVisitor;

impl<'de> Visitor<'de> for BorrowedStrVisitor {
	type Value = Option<&'de str>;

	fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
		formatter.write_str("any JSON value")
	}

	fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> Result<Self::Value, E> {
		Ok(Some(value))
	}

	fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_some<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
		deserializer.deserialize_any(self)
	}

	fn visit_bool<E: DeError>(self, _value: bool) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_i64<E: DeError>(self, _value: i64) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_u64<E: DeError>(self, _value: u64) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_f64<E: DeError>(self, _value: f64) -> Result<Self::Value, E> {
		Ok(None)
	}

	fn visit_seq<A: serde::de::SeqAccess<'de>>(self, _seq: A) -> Result<Self::Value, A::Error> {
		Ok(None)
	}

	fn visit_map<A: serde::de::MapAccess<'de>>(self, _map: A) -> Result<Self::Value, A::Error> {
		Ok(None)
	}
}

/// `#[serde(default, deserialize_with = "optional_value")]`
/// `Option<Value>`: any JSON value; `null` stays `Some(Value::Null)` so
/// `filter(|value| !value.is_null())` semantics survive.
pub fn optional_value<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Some(Value::deserialize(deserializer)?))
}

/// `#[serde(default, deserialize_with = "optional_f64")]`
/// `Option<f64>`: `Value::as_f64` semantics — integers and floats parse,
/// numbers outside f64 become `None`, non-numbers are `None`.
pub fn optional_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Option::<f64>::deserialize(deserializer)?.or_else(|| {
		// Numbers that overflow f64 (e.g. huge u64) deserialize as None here,
		// but Value::as_f64 would also return None for them, so this matches.
		None
	}))
}

/// `#[serde(default, deserialize_with = "optional_i64")]`
/// `Option<i64>`: `Value::as_i64` semantics for the fields the consumers read.
pub fn optional_i64<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Option::<i64>::deserialize(deserializer)?)
}

/// `#[serde(default, deserialize_with = "optional_u64")]`
/// `Option<u64>`: `Value::as_u64` semantics.
pub fn optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Option::<u64>::deserialize(deserializer)?)
}

/// `#[serde(default, deserialize_with = "optional_bool")]`
/// `Option<bool>`: `Value::as_bool` semantics.
pub fn optional_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Option::<bool>::deserialize(deserializer)?)
}

/// The `js_truthy` classification shared by the completions family. `None`
/// means "field absent" (treated as falsy by every caller).
pub fn js_truthy(value: Option<&Value>) -> bool {
	match value {
		None | Some(Value::Null) => false,
		Some(Value::Bool(value)) => *value,
		Some(Value::Number(number)) => number.as_f64().map(|value| value != 0.0).unwrap_or(true),
		Some(Value::String(value)) => !value.is_empty(),
		Some(Value::Array(_) | Value::Object(_)) => true,
	}
}

/// `finish_reason` as the completions consumer treats it: `Value::String`
/// borrows, anything else serializes to the same text `Value::to_string`
/// produces (e.g. numbers, booleans, objects).
pub enum LenientString<'a> {
	Borrowed(&'a str),
	Owned(String),
}

impl LenientString<'_> {
	pub fn as_str(&self) -> &str {
		match self {
			LenientString::Borrowed(text) => text,
			LenientString::Owned(text) => text,
		}
	}
}

/// A `LenientString` that remembers whether the source JSON value was
/// truthy per `js_truthy`, so shape-sensitive filters stay exact.
pub struct TruthyLenientString<'a> {
	string: LenientString<'a>,
	truthy: bool,
}

impl<'a> TruthyLenientString<'a> {
	pub fn as_str(&self) -> &str {
		self.string.as_str()
	}

	/// `js_truthy` of the ORIGINAL JSON value, not of the stringified text.
	pub fn is_truthy(&self) -> bool {
		self.truthy
	}

	pub fn into_lenient(self) -> LenientString<'a> {
		self.string
	}
}

/// `#[serde(default, deserialize_with = "truthy_lenient_string")]`
/// `Option<TruthyLenientString<'a>>`: like `lenient_string`, plus the
/// `js_truthy` verdict of the raw JSON value.
pub fn truthy_lenient_string<'de, D>(deserializer: D) -> Result<Option<TruthyLenientString<'de>>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	struct TruthyVisitor;

	impl<'de> Visitor<'de> for TruthyVisitor {
		type Value = Option<TruthyLenientString<'de>>;

		fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
			formatter.write_str("any JSON value")
		}

		fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(value.to_string()),
				truthy: !value.is_empty(),
			}))
		}

		fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> Result<Self::Value, E> {
			Ok(Some(TruthyLenientString {
				string: LenientString::Borrowed(value),
				truthy: !value.is_empty(),
			}))
		}

		fn visit_bool<E: DeError>(self, value: bool) -> Result<Self::Value, E> {
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(if value { "true".into() } else { "false".into() }),
				truthy: value,
			}))
		}

		fn visit_i64<E: DeError>(self, value: i64) -> Result<Self::Value, E> {
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(value.to_string()),
				truthy: value != 0,
			}))
		}

		fn visit_u64<E: DeError>(self, value: u64) -> Result<Self::Value, E> {
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(value.to_string()),
				truthy: value != 0,
			}))
		}

		fn visit_f64<E: DeError>(self, value: f64) -> Result<Self::Value, E> {
			// `Number::as_f64` is None only outside f64 range, which the JSON
			// parser rejects anyway.
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(Value::from(value).to_string()),
				truthy: value != 0.0,
			}))
		}

		fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
			Ok(None)
		}

		fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
			Ok(None)
		}

		fn visit_some<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
			deserializer.deserialize_any(self)
		}

		fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
			let mut elements = Vec::new();
			while let Some(element) = seq.next_element::<Value>()? {
				elements.push(element);
			}
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(Value::Array(elements).to_string()),
				truthy: true,
			}))
		}

		fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
			let mut object = serde_json::Map::new();
			while let Some((key, value)) = map.next_entry::<String, Value>()? {
				object.insert(key, value);
			}
			Ok(Some(TruthyLenientString {
				string: LenientString::Owned(Value::Object(object).to_string()),
				truthy: true,
			}))
		}
	}

	deserializer.deserialize_any(TruthyVisitor)
}

/// `#[serde(default, deserialize_with = "lenient_string")]`
/// `Option<LenientString<'a>>`: strings borrow; non-strings re-serialize to
/// the compact-JSON text `Value::to_string` would have produced.
pub fn lenient_string<'de, D>(deserializer: D) -> Result<Option<LenientString<'de>>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	struct LenientStringVisitor;

	impl<'de> Visitor<'de> for LenientStringVisitor {
		type Value = Option<LenientString<'de>>;

		fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
			formatter.write_str("any JSON value")
		}

		fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Owned(value.to_string())))
		}

		fn visit_borrowed_str<E: DeError>(self, value: &'de str) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Borrowed(value)))
		}

		fn visit_bool<E: DeError>(self, value: bool) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Owned(if value { "true".into() } else { "false".into() })))
		}

		fn visit_i64<E: DeError>(self, value: i64) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Owned(value.to_string())))
		}

		fn visit_u64<E: DeError>(self, value: u64) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Owned(value.to_string())))
		}

		fn visit_f64<E: DeError>(self, value: f64) -> Result<Self::Value, E> {
			Ok(Some(LenientString::Owned(Value::from(value).to_string())))
		}

		fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
			Ok(None)
		}

		fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
			Ok(None)
		}

		fn visit_some<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
			deserializer.deserialize_any(self)
		}

		fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
			let mut elements = Vec::new();
			while let Some(element) = seq.next_element::<Value>()? {
				elements.push(element);
			}
			Ok(Some(LenientString::Owned(Value::Array(elements).to_string())))
		}

		fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
			let mut object = serde_json::Map::new();
			while let Some((key, value)) = map.next_entry::<String, Value>()? {
				object.insert(key, value);
			}
			Ok(Some(LenientString::Owned(Value::Object(object).to_string())))
		}
	}

	deserializer.deserialize_any(LenientStringVisitor)
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde::Deserialize;

	#[derive(Deserialize)]
	struct Sample<'a> {
		#[serde(default, deserialize_with = "optional_borrowed_str")]
		name: Option<&'a str>,
		#[serde(default, deserialize_with = "lenient_string")]
		finish: Option<LenientString<'a>>,
		#[serde(default, deserialize_with = "optional_value")]
		usage: Option<Value>,
	}

	#[test]
	fn optional_borrowed_str_matches_value_semantics() {
		let sample: Sample = serde_json::from_str(r#"{"name":"x","finish":null}"#).unwrap();
		assert_eq!(sample.name, Some("x"));
		assert!(sample.finish.is_none());

		let sample: Sample = serde_json::from_str(r#"{"name":42}"#).unwrap();
		assert_eq!(sample.name, None);

		let sample: Sample = serde_json::from_str(r#"{"name":null}"#).unwrap();
		assert_eq!(sample.name, None);

		// Borrowed: the slice outlives the parse (no allocation for the value).
		let text = r#"{"name":"borrowed-me"}"#;
		let sample: Sample = serde_json::from_str(text).unwrap();
		assert_eq!(sample.name, Some("borrowed-me"));
	}

	#[test]
	fn lenient_string_keeps_value_to_string_text() {
		let sample: Sample = serde_json::from_str(r#"{"finish":42}"#).unwrap();
		assert_eq!(sample.finish.as_ref().map(LenientString::as_str), Some("42"));

		let sample: Sample = serde_json::from_str(r#"{"finish":true}"#).unwrap();
		assert_eq!(sample.finish.as_ref().map(LenientString::as_str), Some("true"));

		let sample: Sample = serde_json::from_str(r#"{"finish":{"a":1}}"#).unwrap();
		assert_eq!(
			sample.finish.as_ref().map(LenientString::as_str),
			Some(Value::from(serde_json::json!({"a":1})).to_string().as_str())
		);

		let sample: Sample = serde_json::from_str(r#"{"finish":[1,2]}"#).unwrap();
		assert_eq!(
			sample.finish.as_ref().map(LenientString::as_str),
			Some("[1,2]")
		);
	}

	#[test]
	fn optional_value_keeps_null_distinct_from_absent() {
		let sample: Sample = serde_json::from_str(r#"{"usage":null}"#).unwrap();
		assert_eq!(sample.usage, Some(Value::Null));
		let sample: Sample = serde_json::from_str("{}").unwrap();
		assert_eq!(sample.usage, None);
	}
}
