//! Port of packages/ai/src/log.ts
//!
//! Minimal structured logger shared by pi-ai and its consumers. The library
//! never writes files: entries go to an injectable sink (set_log_sink). Without
//! a sink, warn/error fall back to console.error and debug/info are dropped.
//! Logging must never throw into the caller.

use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

pub type LogLevel = &'static str;

pub const LOG_LEVEL_DEBUG: LogLevel = "debug";
pub const LOG_LEVEL_INFO: LogLevel = "info";
pub const LOG_LEVEL_WARN: LogLevel = "warn";
pub const LOG_LEVEL_ERROR: LogLevel = "error";

pub type LogSink = Box<dyn Fn(&Value) + Send + Sync + 'static>;

static SINK: OnceLock<Mutex<Option<LogSink>>> = OnceLock::new();

fn sink_slot() -> &'static Mutex<Option<LogSink>> {
    SINK.get_or_init(|| Mutex::new(None))
}

/// Install the process-wide log sink. Pass `None` to restore the default.
pub fn set_log_sink(next: Option<LogSink>) {
    if let Ok(mut guard) = sink_slot().lock() {
        *guard = next;
    }
}

/// Serialize sink-swap test bodies. The sink is process-wide, so two tests
/// that install their own capture sinks and run in parallel can clear each
/// other's sink mid-assert. Every test that installs a sink must hold this
/// guard for its whole body (drop order with the reset guard does not
/// matter: both are released at test end).
pub fn test_sink_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_SINK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    TEST_SINK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// JSON serialization that never fails: the entry is built from JSON values, so
/// there are no circular references or BigInts to degrade.
pub fn stringify_log_entry(entry: &Value) -> String {
    match serde_json::to_string(entry) {
        Ok(text) => text,
        Err(_) => {
            let fallback = serde_json::json!({
                "ts": entry.get("ts").cloned().unwrap_or(Value::Null),
                "level": entry.get("level").cloned().unwrap_or(Value::Null),
                "component": entry.get("component").cloned().unwrap_or(Value::Null),
                "msg": entry.get("msg").map(|v| v.to_string()).unwrap_or_default(),
                "fieldsError": "unserializable fields dropped",
            });
            serde_json::to_string(&fallback).unwrap_or_default()
        }
    }
}

fn emit(level: LogLevel, component: &str, msg: &str, fields: Option<Map<String, Value>>) {
    // Reserved keys win over caller fields so entries can't be misclassified.
    let mut entry = Map::new();
    if let Some(fields) = fields {
        for (key, value) in fields {
            entry.insert(key, value);
        }
    }
    entry.insert("ts".to_string(), Value::String(now_iso_string()));
    entry.insert("level".to_string(), Value::String(level.to_string()));
    entry.insert("component".to_string(), Value::String(component.to_string()));
    entry.insert("msg".to_string(), Value::String(msg.to_string()));
    let entry = Value::Object(entry);

    let mut sink_handled = false;
    if let Ok(guard) = sink_slot().lock() {
        if let Some(sink) = guard.as_ref() {
            // Broken sink: fall through to the console fallback.
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink(&entry))).is_ok() {
                sink_handled = true;
            }
        }
    }
    if sink_handled {
        return;
    }

    if level == LOG_LEVEL_WARN || level == LOG_LEVEL_ERROR {
        eprintln!("{}", stringify_log_entry(&entry));
    }
}

/// `new Date().toISOString()` without pulling in a date crate: the format is
/// `YYYY-MM-DDTHH:MM:SS.sssZ` in UTC.
fn now_iso_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis() as i64;
    format_iso_millis(millis)
}

fn format_iso_millis(millis: i64) -> String {
    let (days, ms_of_day) = {
        let mut days = millis.div_euclid(86_400_000);
        let mut ms_of_day = millis.rem_euclid(86_400_000);
        if ms_of_day < 0 {
            days -= 1;
            ms_of_day += 86_400_000;
        }
        (days, ms_of_day)
    };
    let hours = ms_of_day / 3_600_000;
    let minutes = (ms_of_day / 60_000) % 60;
    let seconds = (ms_of_day / 1_000) % 60;
    let ms = ms_of_day % 1_000;

    // Civil-from-days (Howard Hinnant's algorithm), epoch 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, m, d, hours, minutes, seconds, ms
    )
}

pub struct Logger {
    component: String,
}

impl Logger {
    pub fn debug(&self, msg: &str, fields: Option<Map<String, Value>>) {
        emit(LOG_LEVEL_DEBUG, &self.component, msg, fields);
    }
    pub fn info(&self, msg: &str, fields: Option<Map<String, Value>>) {
        emit(LOG_LEVEL_INFO, &self.component, msg, fields);
    }
    pub fn warn(&self, msg: &str, fields: Option<Map<String, Value>>) {
        emit(LOG_LEVEL_WARN, &self.component, msg, fields);
    }
    pub fn error(&self, msg: &str, fields: Option<Map<String, Value>>) {
        emit(LOG_LEVEL_ERROR, &self.component, msg, fields);
    }
}

pub fn get_logger(component: &str) -> Logger {
    Logger {
        component: component.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_keys_win_over_caller_fields() {
        let captured: std::sync::Arc<Mutex<Vec<Value>>> = std::sync::Arc::new(Mutex::new(Vec::new()));
        let sink_target = captured.clone();
        set_log_sink(Some(Box::new(move |entry| {
            sink_target.lock().unwrap().push(entry.clone());
        })));

        let mut fields = Map::new();
        fields.insert("level".to_string(), Value::String("nonsense".to_string()));
        fields.insert("msg".to_string(), Value::String("nonsense".to_string()));
        fields.insert("extra".to_string(), Value::Number(1.into()));
        get_logger("test").warn("real message", Some(fields));

        set_log_sink(None);
        let entries = captured.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["level"], Value::String("warn".to_string()));
        assert_eq!(entries[0]["msg"], Value::String("real message".to_string()));
        assert_eq!(entries[0]["component"], Value::String("test".to_string()));
        assert_eq!(entries[0]["extra"], Value::Number(1.into()));
    }

    #[test]
    fn iso_timestamp_format() {
        assert_eq!(format_iso_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(format_iso_millis(1_700_000_000_123), "2023-11-14T22:13:20.123Z");
    }

    #[test]
    fn stringify_never_fails() {
        let entry = serde_json::json!({"ts": "t", "level": "info", "component": "c", "msg": "m"});
        assert_eq!(
            stringify_log_entry(&entry),
            "{\"ts\":\"t\",\"level\":\"info\",\"component\":\"c\",\"msg\":\"m\"}"
        );
    }
}
