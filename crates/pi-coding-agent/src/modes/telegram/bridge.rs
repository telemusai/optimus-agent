//! Port of packages/coding-agent/src/modes/telegram/bridge.ts

use std::sync::{Arc, Mutex};
use std::time::Duration;

use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use pi_agent_core::types::AgentMessage;

use crate::modes::agent_connection::types::{
    AgentConnection, AgentConnectionEvent, AgentConnectionExtensionUiRequest, AgentConnectionExtensionUiResponse,
    AgentConnectionPromptOptions, AgentConnectionSessionEvent,
};
use crate::modes::telegram::api::{
    private_message, split_telegram_text, TelegramApi, TelegramApiError, TelegramCallError, TelegramUpdate,
};
use crate::modes::telegram::commands::{parse_telegram_command, telegram_command_menu, TelegramCommands};
use crate::modes::telegram::store::{
    pairing_hash, TelegramConnectionSettings, TelegramDelivery, TelegramState, TelegramStore,
};

/// `messageText(message)`.
///
/// `if (!("content" in message)) return ""` - `bashExecution`, `branchSummary`
/// and `compactionSummary` carry no `content` key, so they read as empty.
fn message_text(message: &AgentMessage) -> String {
    let text_blocks = |blocks: &[pi_ai::types::ImageOrTextContent]| {
        blocks
            .iter()
            .filter_map(|block| match block {
                pi_ai::types::ImageOrTextContent::Text(text) => Some(text.text.clone()),
                pi_ai::types::ImageOrTextContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    match message {
        AgentMessage::Message(message) => match message {
            pi_ai::types::Message::User(user) => match &user.content {
                pi_ai::types::UserContent::Text(text) => text.clone(),
                pi_ai::types::UserContent::Blocks(blocks) => text_blocks(blocks),
            },
            pi_ai::types::Message::Assistant(assistant) => assistant
                .content
                .iter()
                .filter_map(|block| block.as_text().map(|text| text.text.clone()))
                .collect::<Vec<_>>()
                .join("\n"),
            pi_ai::types::Message::ToolResult(result) => text_blocks(&result.content),
        },
        AgentMessage::Custom(custom) => match custom {
            pi_agent_core::types::CustomAgentMessage::Custom { content, .. } => match content {
                pi_agent_core::types::CustomMessageContent::Text(text) => text.clone(),
                pi_agent_core::types::CustomMessageContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| block.as_text().map(str::to_string))
                    .collect::<Vec<_>>()
                    .join("\n"),
            },
            _ => String::new(),
        },
    }
}

/// `message.display` - only `CustomAgentMessage::Custom` declares it.
fn message_display(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Message(_) => false,
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom { display, .. }) => *display,
        AgentMessage::Custom(_) => false,
    }
}

/// `message.errorMessage` for an assistant message.
fn message_error_message(message: &AgentMessage) -> Option<String> {
    match message {
        AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => assistant.error_message.clone(),
        _ => None,
    }
}

/// `interface PendingQuestion { request; expiresAt }`.
#[derive(Debug, Clone)]
struct PendingQuestion {
    request: AgentConnectionExtensionUiRequest,
    expires_at: i64,
}

/// `acceptTelegramPairing(settings, update)`.
pub fn accept_telegram_pairing(settings: &mut TelegramConnectionSettings, update: &serde_json::Value) -> bool {
    let message = private_message(update);
    let command = parse_telegram_command(
        message.as_ref().and_then(|message| message.text.clone()).unwrap_or_default().as_str(),
        &settings.bot_username,
    );
    let blocked = message.is_none()
        || settings.paired_user_id.is_some()
        || command.as_ref().map(|command| command.name.as_str()) != Some("start")
        || settings.pairing.is_none()
        || now_ms() as f64
            >= settings
                .pairing
                .as_ref()
                .map(|pairing| pairing.expires_at)
                .unwrap_or(0.0)
        // `command.args.length > 64`: UTF-16 code units, not bytes.
        || command
            .as_ref()
            .map(|command| crate::modes::telegram::api::utf16_len(&command.args) > 64)
            .unwrap_or(false)
        || command.as_ref().map(|command| pairing_hash(&command.args)).unwrap_or_default()
            != settings
                .pairing
                .as_ref()
                .map(|pairing| pairing.hash.clone())
                .unwrap_or_default();
    if blocked {
        return false;
    }
    settings.paired_user_id = message.map(|message| message.from.id);
    settings.pairing = None;
    true
}

/// `class TelegramBridge`.
///
/// `store`, `settings`, `state` and the abort controller live in `shared` because
/// `TelegramCommands` is constructed with a reply callback in the constructor.
pub struct TelegramBridge {
    shared: Arc<BridgeShared>,
    api: Arc<TelegramApi>,
    connection: Arc<dyn AgentConnection>,
    report_error: Arc<dyn Fn(Option<String>) + Send + Sync>,
    commands: TelegramCommands,
    unsubscribe: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    draining: Mutex<Option<Arc<DrainingState>>>,
    sending: Mutex<Option<Arc<SendingState>>>,
    /// `Map` keyed by the random question id; insertion order is observable
    /// (the expiry sweep and the shutdown cancel iterate it in that order).
    questions: Mutex<IndexMap<String, PendingQuestion>>,
    working: Mutex<bool>,
    next_delivery_at: Mutex<i64>,
    delivery_failures: Mutex<f64>,
}

/// The in-flight `drainInbox()` promise (`this.draining`).
struct DrainingState {
    done: tokio::sync::Notify,
    finished: Mutex<bool>,
}

impl DrainingState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            done: tokio::sync::Notify::new(),
            finished: Mutex::new(false),
        })
    }

    fn finish(&self) {
        *self.finished.lock().unwrap() = true;
        self.done.notify_waiters();
    }
}

/// The in-flight `flushReplies()` promise (`this.sending`).
///
/// The TS promise rejects with whatever the send threw, so the stored outcome
/// carries the bridge error rather than a stringified copy.
struct SendingState {
    done: tokio::sync::Notify,
    outcome: Mutex<Option<Result<(), TelegramBridgeError>>>,
    finished: Mutex<bool>,
}

impl SendingState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            done: tokio::sync::Notify::new(),
            outcome: Mutex::new(None),
            finished: Mutex::new(false),
        })
    }

    fn finish(&self, outcome: Result<(), TelegramBridgeError>) {
        *self.outcome.lock().unwrap() = Some(outcome);
        *self.finished.lock().unwrap() = true;
        self.done.notify_waiters();
    }

    async fn wait(&self) -> Result<(), TelegramBridgeError> {
        loop {
            if let Some(outcome) = self.outcome.lock().unwrap().clone() {
                return outcome;
            }
            self.done.notified().await;
        }
    }
}

/// Bridge failures this module branches on.
///
/// The TypeScript branches on `error instanceof TelegramApiError` and reads
/// `error.retryAfter`; that distinction has to survive the port, so the error is
/// carried explicitly instead of being flattened to a string.
#[derive(Debug, Clone, PartialEq)]
pub enum TelegramBridgeError {
    /// `TelegramApiError` with its `code` and `retryAfter`.
    Api { code: f64, retry_after: f64, message: String },
    /// Any other thrown error, already stringified.
    Message(String),
}

impl TelegramBridgeError {
    fn message(&self) -> String {
        match self {
            TelegramBridgeError::Api { message, .. } => message.clone(),
            TelegramBridgeError::Message(message) => message.clone(),
        }
    }

    /// `error instanceof TelegramApiError && error.retryAfter ? ... : ...`.
    fn retry_after(&self) -> Option<f64> {
        match self {
            TelegramBridgeError::Api { retry_after, .. } if *retry_after != 0.0 => Some(*retry_after),
            _ => None,
        }
    }

    fn code(&self) -> Option<f64> {
        match self {
            TelegramBridgeError::Api { code, .. } => Some(*code),
            _ => None,
        }
    }
}

/// `error instanceof TelegramApiError` for the shared API error type.
fn call_error_to_bridge(error: TelegramCallError) -> TelegramBridgeError {
    match error {
        TelegramCallError::Api(api_error) => TelegramBridgeError::Api {
            code: api_error.code,
            retry_after: api_error.retry_after,
            message: api_error.message,
        },
        TelegramCallError::Error(message) => TelegramBridgeError::Message(message),
        TelegramCallError::Aborted => TelegramBridgeError::Message("Telegram request aborted".to_string()),
    }
}

impl From<String> for TelegramBridgeError {
    fn from(value: String) -> Self {
        TelegramBridgeError::Message(value)
    }
}

/// `error instanceof TelegramApiError` for an unwrapped `TelegramApiError`.
impl From<TelegramApiError> for TelegramBridgeError {
    fn from(value: TelegramApiError) -> Self {
        TelegramBridgeError::Api {
            code: value.code,
            retry_after: value.retry_after,
            message: value.message,
        }
    }
}

/// JavaScript `String(value)`.
fn js_string_of(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Bool(value) => if *value { "true" } else { "false" }.to_string(),
        serde_json::Value::Number(number) => number.to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| match item {
                serde_json::Value::Null => String::new(),
                other => js_string_of(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        serde_json::Value::Object(_) => "[object Object]".to_string(),
    }
}

/// `String(value ?? fallback)` - nullish coalescing keeps a `0`, `false` or `""`.
fn js_string_or(value: Option<&serde_json::Value>, fallback: &str) -> String {
    match value {
        None | Some(serde_json::Value::Null) => fallback.to_string(),
        Some(value) => js_string_of(value),
    }
}

/// `Date.now()`.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// `await delay(ms, undefined, { signal })`: rejects when the signal aborts.
async fn delay_ms(ms: u64, controller: &CancellationToken) -> Result<(), TelegramBridgeError> {
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
        _ = controller.cancelled() => Err(TelegramBridgeError::Message("Telegram request aborted".to_string())),
    }
}

/// `randomBytes(4).toString("hex")` - the pending-question id.
fn random_hex_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `randomUUID()`.
fn random_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `createHash("sha256").update(text).digest("hex")`.
fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `setInterval(callback, ms)` with the first run after one full period.
fn start_interval<F>(period_ms: u64, callback: F) -> tokio::task::JoinHandle<()>
where
    F: Fn() + Send + 'static,
{
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_millis(period_ms),
            Duration::from_millis(period_ms),
        );
        loop {
            ticker.tick().await;
            callback();
        }
    })
}

/// The mutable bridge facts `enqueueReply`/`save` touch.
///
/// The TypeScript constructor hands `this` to `TelegramCommands` before the
/// constructor returns; Rust cannot capture a half-built value, so the state
/// those two members share lives here and is captured instead.
struct BridgeShared {
    store: TelegramStore,
    settings: Mutex<TelegramConnectionSettings>,
    state: Mutex<TelegramState>,
    controller: CancellationToken,
}

impl BridgeShared {
    /// `save()`.
    fn save(&self) {
        let snapshot = {
            let state = self.state.lock().unwrap();
            serde_json::to_value(&*state).unwrap_or(serde_json::Value::Null)
        };
        let _ = self.store.write("state.json", &snapshot);
    }

    /// `enqueueReply(text)`.
    fn enqueue_reply(&self, text: &str) -> Result<(), String> {
        let paired_user_id = {
            let settings = self.settings.lock().unwrap();
            settings.paired_user_id
        };
        if self.controller.is_cancelled() || paired_user_id.is_none() || text.trim().is_empty() {
            return Ok(());
        }
        let chunks = split_telegram_text(text, 4000)?;
        {
            let mut state = self.state.lock().unwrap();
            if state.outbox.len() + chunks.len() > 1000 {
                return Err(
                    "Telegram delivery queue is full. Reconnect Telegram before sending more work.".to_string(),
                );
            }
            for chunk in chunks {
                state.outbox.push(TelegramDelivery {
                    id: random_uuid(),
                    chat_id: paired_user_id.unwrap_or(0.0),
                    text: chunk,
                });
            }
        }
        self.save();
        Ok(())
    }

    /// `this.store.write("connection.json", this.settings)`.
    fn save_connection(&self) {
        let snapshot = {
            let settings = self.settings.lock().unwrap();
            serde_json::to_value(&*settings).unwrap_or(serde_json::Value::Null)
        };
        let _ = self.store.write("connection.json", &snapshot);
    }
}

/// `privateMessage(update)` on the deserialized `TelegramUpdate` shape.
fn update_as_value(update: &TelegramUpdate) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "update_id".to_string(),
        serde_json::Number::from_f64(update.update_id)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
    );
    if let Some(message) = &update.message {
        object.insert("message".to_string(), message.clone());
    }
    serde_json::Value::Object(object)
}

impl TelegramBridge {
    /// `new TelegramBridge(store, settings, api, connection, reportError)`.
    pub fn new(
        store: TelegramStore,
        settings: TelegramConnectionSettings,
        api: Arc<TelegramApi>,
        connection: Arc<dyn AgentConnection>,
        report_error: Arc<dyn Fn(Option<String>) + Send + Sync>,
    ) -> Result<Self, String> {
        let state = store.state(settings.bot_id)?;
        let shared = Arc::new(BridgeShared {
            store,
            settings: Mutex::new(settings),
            state: Mutex::new(state),
            controller: CancellationToken::new(),
        });
        let reply_shared = shared.clone();
        let commands = TelegramCommands::new(
            connection.clone(),
            Arc::new(move |text: String| {
                // `enqueueReply` failures are the command layer's reply path: it
                // only throws when the queue is full, which the caller surfaces.
                let _ = reply_shared.enqueue_reply(&text);
            }),
        );
        Ok(Self {
            shared,
            api,
            connection,
            report_error,
            commands,
            unsubscribe: Mutex::new(None),
            draining: Mutex::new(None),
            sending: Mutex::new(None),
            questions: Mutex::new(IndexMap::new()),
            working: Mutex::new(false),
            next_delivery_at: Mutex::new(0),
            delivery_failures: Mutex::new(0.0),
        })
    }

    /// The pending extension-UI questions (read-only view for tests).
    pub fn pending_question_count(&self) -> usize {
        self.questions.lock().unwrap().len()
    }

    /// `save()`.
    fn save(&self) {
        self.shared.save();
    }

    /// `enqueueReply(text)`.
    pub fn enqueue_reply(&self, text: &str) -> Result<(), String> {
        self.shared.enqueue_reply(text)
    }

    /// `flushReplies()`.
    pub async fn flush_replies(&self, has_signal: bool) -> Result<(), TelegramBridgeError> {
        let signal = if has_signal { Some(self.shared.controller.clone()) } else { None };
        // Bind first: the `if let` scrutinee's guard would otherwise live across the await.
        let in_flight = self.sending.lock().unwrap().clone();
        if let Some(sending) = in_flight {
            return sending.wait().await;
        }
        if now_ms() < *self.next_delivery_at.lock().unwrap() {
            return Ok(());
        }
        let sending = SendingState::new();
        *self.sending.lock().unwrap() = Some(sending.clone());
        let outcome = match self.flush_replies_inner(signal).await {
            Ok(()) => Ok(()),
            Err(error) => {
                let failures = {
                    let mut failures = self.delivery_failures.lock().unwrap();
                    *failures += 1.0;
                    *failures
                };
                let backoff = match error.retry_after() {
                    Some(retry_after) => retry_after * 1000.0,
                    None => 30_000.0f64.min(1000.0 * 2f64.powf((failures - 1.0).min(5.0))),
                };
                *self.next_delivery_at.lock().unwrap() = now_ms() + backoff as i64;
                Err(error)
            }
        };
        *self.sending.lock().unwrap() = None;
        sending.finish(outcome.clone());
        outcome
    }

    /// The `(async () => { ... })()` body of `flushReplies()`.
    async fn flush_replies_inner(
        &self,
        signal: Option<CancellationToken>,
    ) -> Result<(), TelegramBridgeError> {
        let controller = self.shared.controller.clone();
        loop {
            if controller.is_cancelled() {
                return Ok(());
            }
            let next = {
                let state = self.shared.state.lock().unwrap();
                match state.outbox.first() {
                    Some(next) => (next.chat_id, next.text.clone()),
                    None => return Ok(()),
                }
            };
            let paired_user_id = self.shared.settings.lock().unwrap().paired_user_id;
            if Some(next.0) != paired_user_id {
                {
                    let mut state = self.shared.state.lock().unwrap();
                    state.outbox.remove(0);
                }
                self.save();
                continue;
            }
            self.api
                .send(next.0, &next.1, signal.clone())
                .await
                .map_err(call_error_to_bridge)?;
            if controller.is_cancelled() {
                return Ok(());
            }
            {
                let mut state = self.shared.state.lock().unwrap();
                state.outbox.remove(0);
            }
            self.save();
            *self.delivery_failures.lock().unwrap() = 0.0;
            (self.report_error)(None);
            delay_ms(1100, &controller).await?;
        }
    }

    /// `accept(update)`.
    pub async fn accept(self: &Arc<Self>, update: &TelegramUpdate) -> Result<(), TelegramBridgeError> {
        let controller = self.shared.controller.clone();
        if controller.is_cancelled() {
            return Ok(());
        }
        {
            let state = self.shared.state.lock().unwrap();
            if update.update_id < state.offset {
                return Ok(());
            }
        }
        let raw = update_as_value(update);
        let message = private_message(&raw);
        {
            let mut state = self.shared.state.lock().unwrap();
            state.offset = update.update_id + 1.0;
        }
        let Some(message) = message else {
            self.save();
            return Ok(());
        };
        let bot_username = self.shared.settings.lock().unwrap().bot_username.clone();
        let command = parse_telegram_command(message.text.as_deref().unwrap_or_default(), &bot_username);
        let paired_user_id = self.shared.settings.lock().unwrap().paired_user_id;
        if paired_user_id.is_none() {
            let paired = {
                let mut settings = self.shared.settings.lock().unwrap();
                accept_telegram_pairing(&mut settings, &raw)
            };
            if paired {
                self.shared.save_connection();
                let _ = self.enqueue_reply(
                    "Connected to Optimus. This chat controls your connected session. Send a message to begin, or /help for commands.",
                );
            }
            self.save();
            return Ok(());
        }
        if Some(message.from.id) != paired_user_id {
            self.save();
            return Ok(());
        }
        if message.text.is_none() {
            let _ = self.enqueue_reply(
                "Send a text message to Optimus. Photos, files, and voice messages are not supported by this connector yet.",
            );
            self.save();
            return Ok(());
        }
        let name = command.as_ref().map(|command| command.name.clone()).unwrap_or_default();
        let args = command.as_ref().map(|command| command.args.clone()).unwrap_or_default();
        if name == "ignore" {
            self.save();
            return Ok(());
        }
        if name == "answer" {
            self.save();
            self.answer(&args).await;
            return Ok(());
        }
        if name == "stop" || name == "cancel" {
            {
                let mut state = self.shared.state.lock().unwrap();
                state.inbox.clear();
            }
            self.save();
            if let Err(error) = self.commands.execute(&name, &args).await {
                return Err(TelegramBridgeError::Message(error));
            }
            return Ok(());
        }
        let inbox_len = self.shared.state.lock().unwrap().inbox.len();
        if inbox_len >= 100 {
            let _ = self.enqueue_reply("Optimus already has 100 messages waiting. Use /stop to clear the queue.");
        } else {
            let mut state = self.shared.state.lock().unwrap();
            state.inbox.push(crate::modes::telegram::store::TelegramInboxItem {
                id: update.update_id,
                text: message.text.clone().unwrap_or_default(),
            });
        }
        self.save();
        self.drain_inbox();
        Ok(())
    }

    /// `private drainInbox(): void`.
    fn drain_inbox(self: &Arc<Self>) {
        let controller = self.shared.controller.clone();
        if self.draining.lock().unwrap().is_some() || controller.is_cancelled() {
            return;
        }
        let draining = DrainingState::new();
        *self.draining.lock().unwrap() = Some(draining.clone());
        let bridge = self.clone();
        tokio::spawn(async move {
            let outcome = bridge.drain_inbox_body().await;
            if let Err(error) = outcome {
                (bridge.report_error)(Some(bridge.api.redact(&error.message())));
            }
            *bridge.draining.lock().unwrap() = None;
            draining.finish();
        });
    }

    /// The `(async () => { ... })()` body of `drainInbox()`.
    async fn drain_inbox_body(&self) -> Result<(), TelegramBridgeError> {
        let controller = self.shared.controller.clone();
        loop {
            if controller.is_cancelled() {
                return Ok(());
            }
            let next = {
                let state = self.shared.state.lock().unwrap();
                match state.inbox.first() {
                    Some(next) => next.clone(),
                    None => return Ok(()),
                }
            };
            {
                let mut state = self.shared.state.lock().unwrap();
                state.inbox.remove(0);
                // Commit the attempt before dispatch: never replay an uncertain
                // tool-bearing prompt after a crash.
                state.interrupted_update = Some(next.id);
            }
            self.save();
            let dispatch = self.dispatch_inbox_item(&next.text).await;
            if let Err(error) = dispatch {
                let _ = self.enqueue_reply(&self.api.redact(&error.message()));
            }
            if controller.is_cancelled() {
                return Ok(());
            }
            {
                let mut state = self.shared.state.lock().unwrap();
                state.interrupted_update = None;
            }
            self.save();
        }
    }

    /// The per-item body of `drainInbox()` including its `saveBinding()` step.
    async fn dispatch_inbox_item(&self, text: &str) -> Result<(), TelegramBridgeError> {
        let bot_username = self.shared.settings.lock().unwrap().bot_username.clone();
        let command = parse_telegram_command(text, &bot_username);
        match command {
            Some(command) => self
                .commands
                .execute(&command.name, &command.args)
                .await
                .map_err(TelegramBridgeError::Message)?,
            None => {
                // Ordinary chat traffic queues in order against updated
                // history instead of interrupting the running turn; explicit
                // steering stays available through the command surface.
                self.connection
                    .prompt(
                        text,
                        Some(AgentConnectionPromptOptions {
                            source: Some("interactive".to_string()),
                            queue_if_busy: Some(true),
                            streaming_behavior: Some("followUp".to_string()),
                            ..Default::default()
                        }),
                    )
                    .await
                    .map_err(TelegramBridgeError::Message)?
            }
        }
        self.save_binding().await;
        Ok(())
    }

    /// `saveBinding()`.
    async fn save_binding(&self) {
        let state = match self.connection.get_state().await {
            Ok(state) => state,
            Err(_) => return,
        };
        if self.shared.controller.is_cancelled() {
            return;
        }
        {
            let mut settings = self.shared.settings.lock().unwrap();
            settings.session_id = state.session_id;
            settings.session_file = state.session_file;
            settings.cwd = state.cwd;
        }
        self.shared.save_connection();
    }

    /// `onEvent(event)`.
    pub async fn on_event(self: &Arc<Self>, event: AgentConnectionEvent) -> Result<(), TelegramBridgeError> {
        if self.shared.controller.is_cancelled() {
            return Ok(());
        }
        match &event {
            AgentConnectionEvent::SessionReplaced { .. } | AgentConnectionEvent::SessionResynced { .. } => {
                self.save_binding().await;
                return Ok(());
            }
            AgentConnectionEvent::Closed { .. } => {
                (self.report_error)(Some(
                    "The Optimus session disconnected. Open Optimus and run /telegram restart.".to_string(),
                ));
                self.shared.controller.cancel();
                return Ok(());
            }
            _ => {}
        }
        if self.shared.settings.lock().unwrap().paired_user_id.is_none() {
            return Ok(());
        }
        if let AgentConnectionEvent::ExtensionUiRequest { request } = &event {
            self.question(request).await;
            return Ok(());
        }
        let AgentConnectionEvent::SessionEvent { event } = &event else {
            return Ok(());
        };
        match event {
            AgentConnectionSessionEvent::Agent(pi_agent_core::types::AgentEvent::AgentStart) => {
                *self.working.lock().unwrap() = true;
            }
            AgentConnectionSessionEvent::Agent(pi_agent_core::types::AgentEvent::AgentEnd { .. }) => {
                *self.working.lock().unwrap() = false;
            }
            _ => {}
        }
        let message = match event {
            AgentConnectionSessionEvent::MessageEnd { message } => message,
            AgentConnectionSessionEvent::Agent(pi_agent_core::types::AgentEvent::MessageEnd { message }) => message,
            _ => return Ok(()),
        };
        let is_assistant = message.role() == "assistant";
        if !is_assistant && !(message.role() == "custom" && message_display(message)) {
            return Ok(());
        }
        let text = {
            let text = message_text(message);
            if text.is_empty() && is_assistant {
                message_error_message(message).unwrap_or_default()
            } else {
                text
            }
        };
        if text.is_empty() {
            return Ok(());
        }
        let session_id = self.shared.settings.lock().unwrap().session_id.clone();
        let key = sha256_hex(&format!(
            "{session_id}:{}:{text}",
            message_timestamp(message)
        ));
        {
            let mut state = self.shared.state.lock().unwrap();
            if state.last_assistant_key.as_deref() == Some(key.as_str()) {
                return Ok(());
            }
            state.last_assistant_key = Some(key);
        }
        let _ = self.enqueue_reply(&text);
        Ok(())
    }

    /// `private async question(request)`.
    async fn question(&self, request: &AgentConnectionExtensionUiRequest) {
        // Connection management belongs to the invoking terminal, including its
        // credential dialog.
        if request
            .payload
            .get("title")
            .and_then(serde_json::Value::as_str)
            .map(|title| title.starts_with("Telegram"))
            .unwrap_or(false)
        {
            return;
        }
        if request.method == "notify" {
            if let Some(message) = request.payload.get("message").and_then(serde_json::Value::as_str) {
                let _ = self.enqueue_reply(message);
            }
            return;
        }
        if request.method == "editor" {
            let _ = self
                .connection
                .respond_to_extension_ui_request(
                    &request.id,
                    AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                )
                .await;
            let _ = self.enqueue_reply("This command needs the Optimus terminal editor and was cancelled.");
            return;
        }
        if !["select", "confirm", "input"].contains(&request.method.as_str()) {
            return;
        }
        let id = random_hex_id();
        self.questions.lock().unwrap().insert(
            id.clone(),
            PendingQuestion {
                request: request.clone(),
                expires_at: now_ms() + 5 * 60_000,
            },
        );
        let options: Vec<String> = request
            .payload
            .get("options")
            .and_then(serde_json::Value::as_array)
            .map(|options| {
                options
                    .iter()
                    .filter_map(|option| option.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let mut lines: Vec<String> = vec![
            js_string_or(request.payload.get("title"), "Optimus needs input"),
            js_string_or(request.payload.get("message"), ""),
        ];
        for (index, option) in options.iter().enumerate() {
            lines.push(format!("{}. {option}", index + 1));
        }
        let hint = if request.method == "confirm" {
            "yes|no"
        } else if request.method == "select" {
            "<number>"
        } else {
            "<text>"
        };
        lines.push(format!("Reply /answer {id} {hint}."));
        lines.push(format!("Cancel with /answer {id} cancel. Expires in 5 minutes."));
        let body = lines
            .into_iter()
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let _ = self.enqueue_reply(&body);
    }

    /// `private async answer(args)`.
    async fn answer(&self, args: &str) {
        let (question_id, raw_value) = match parse_answer_args(args) {
            Some(parsed) => parsed,
            None => {
                let _ = self.enqueue_reply("That question is missing or expired. Check Optimus for its current request.");
                return;
            }
        };
        let pending = {
            let questions = self.questions.lock().unwrap();
            match questions.get(&question_id) {
                Some(pending) if pending.expires_at >= now_ms() => pending.clone(),
                _ => {
                    drop(questions);
                    let _ =
                        self.enqueue_reply("That question is missing or expired. Check Optimus for its current request.");
                    return;
                }
            }
        };
        let value = raw_value.trim().to_string();
        let response = if value == "cancel" {
            AgentConnectionExtensionUiResponse::Cancelled { cancelled: true }
        } else if pending.request.method == "confirm" {
            let lowered = value.to_lowercase();
            if lowered != "yes" && lowered != "no" {
                let _ = self.enqueue_reply(&format!("Use /answer {question_id} yes or no."));
                return;
            }
            AgentConnectionExtensionUiResponse::Confirmed { confirmed: lowered == "yes" }
        } else if pending.request.method == "select" {
            let options = pending.request.payload.get("options");
            let selected = if is_all_digits(&value) {
                options
                    .and_then(serde_json::Value::as_array)
                    .and_then(|options| options.get(value.parse::<usize>().unwrap_or(0).wrapping_sub(1)))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            } else {
                None
            };
            match selected {
                Some(selected) => AgentConnectionExtensionUiResponse::Value { value: selected },
                None => {
                    let _ = self.enqueue_reply("Choose a number from the question's options.");
                    return;
                }
            }
        } else {
            AgentConnectionExtensionUiResponse::Value { value }
        };
        self.questions.lock().unwrap().remove(&question_id);
        let _ = self
            .connection
            .respond_to_extension_ui_request(&pending.request.id, response)
            .await;
        let _ = self.enqueue_reply("Answer sent to Optimus.");
    }

    /// `async run(signal?)`.
    pub async fn run(self: &Arc<Self>, signal: Option<CancellationToken>) -> Result<(), TelegramBridgeError> {
        let controller = self.shared.controller.clone();
        // `this.controller.signal` is the signal every API call receives.
        let request_signal = if signal.is_some() { Some(controller.clone()) } else { None };
        // `signal?.addEventListener("abort", stop, { once: true })` plus
        // `if (signal?.aborted) stop()`.
        let stop_task = signal.as_ref().map(|signal| {
            let controller = controller.clone();
            let signal = signal.clone();
            tokio::spawn(async move {
                if signal.is_cancelled() {
                    controller.cancel();
                    return;
                }
                signal.cancelled().await;
                controller.cancel();
            })
        });
        let delivery = Mutex::new(None::<tokio::task::JoinHandle<()>>);
        let typing = Mutex::new(None::<tokio::task::JoinHandle<()>>);
        let outcome = self.run_inner(request_signal, &controller, &delivery, &typing).await;
        // `finally`: stop, clearInterval, unsubscribe, removeEventListener.
        controller.cancel();
        if let Some(task) = delivery.lock().unwrap().take() {
            task.abort();
        }
        if let Some(task) = typing.lock().unwrap().take() {
            task.abort();
        }
        if let Some(unsubscribe) = self.unsubscribe.lock().unwrap().take() {
            unsubscribe();
        }
        if let Some(task) = stop_task {
            task.abort();
        }
        let pending: Vec<AgentConnectionExtensionUiRequest> = {
            let questions = self.questions.lock().unwrap();
            questions.values().map(|question| question.request.clone()).collect()
        };
        for request in pending {
            let _ = self
                .connection
                .respond_to_extension_ui_request(
                    &request.id,
                    AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                )
                .await;
        }
        self.questions.lock().unwrap().clear();
        let in_flight = self.sending.lock().unwrap().clone();
        if let Some(sending) = in_flight {
            let _ = sending.wait().await;
        }
        outcome
    }

    /// The `try { ... }` body of `run()`.
    async fn run_inner(
        self: &Arc<Self>,
        request_signal: Option<CancellationToken>,
        controller: &CancellationToken,
        delivery: &Mutex<Option<tokio::task::JoinHandle<()>>>,
        typing: &Mutex<Option<tokio::task::JoinHandle<()>>>,
    ) -> Result<(), TelegramBridgeError> {
        let bridge = self;
        {
            let weak = Arc::downgrade(bridge);
            let report = bridge.report_error.clone();
            let redact_api = bridge.api.clone();
            let unsubscribe = self.connection.subscribe(Arc::new(move |event| {
                let weak = weak.clone();
                let report = report.clone();
                let redact_api = redact_api.clone();
                Box::pin(async move {
                    if let Some(bridge) = weak.upgrade() {
                        if let Err(error) = bridge.on_event(event).await {
                            report(Some(redact_api.redact(&error.message())));
                        }
                    }
                })
            }));
            *self.unsubscribe.lock().unwrap() = Some(unsubscribe);
        }
        let interrupted = self.shared.state.lock().unwrap().interrupted_update;
        if interrupted.is_some() {
            let _ = self.enqueue_reply(
                "Optimus restarted while accepting a Telegram message. It has not been sent again automatically. Check /session and /copy before repeating it.",
            );
            {
                let mut state = self.shared.state.lock().unwrap();
                state.interrupted_update = None;
            }
            self.save();
        }
        let menu = telegram_command_menu();
        let commands: Vec<serde_json::Value> = menu
            .iter()
            .map(|command| {
                serde_json::json!({ "command": command.command, "description": command.description })
            })
            .collect();
        self.api
            .call(
                "setMyCommands",
                serde_json::json!({ "commands": commands }),
                request_signal.clone(),
            )
            .await
            .map_err(call_error_to_bridge)?;
        let state = self
            .connection
            .get_state()
            .await
            .map_err(TelegramBridgeError::Message)?;
        *self.working.lock().unwrap() = state.is_streaming;
        self.drain_inbox();
        {
            let bridge = bridge.clone();
            let handle = start_interval(1000, move || {
                let bridge = bridge.clone();
                tokio::spawn(async move {
                    if let Err(error) = bridge.flush_replies(true).await {
                        if !bridge.shared.controller.is_cancelled() {
                            (bridge.report_error)(Some(bridge.api.redact(&error.message())));
                        }
                    }
                });
            });
            *delivery.lock().unwrap() = Some(handle);
        }
        {
            let bridge = bridge.clone();
            let handle = start_interval(4000, move || {
                let bridge = bridge.clone();
                tokio::spawn(async move {
                    let paired = bridge.shared.settings.lock().unwrap().paired_user_id;
                    if *bridge.working.lock().unwrap()
                        && paired.is_some()
                        && bridge.questions.lock().unwrap().is_empty()
                    {
                        let _ = bridge
                            .api
                            .call(
                                "sendChatAction",
                                serde_json::json!({ "chat_id": paired, "action": "typing" }),
                                Some(bridge.shared.controller.clone()),
                            )
                            .await;
                    }
                    let expired: Vec<(String, String)> = {
                        let questions = bridge.questions.lock().unwrap();
                        questions
                            .iter()
                            .filter(|(_, question)| question.expires_at <= now_ms())
                            .map(|(id, question)| (id.clone(), question.request.id.clone()))
                            .collect()
                    };
                    for (id, request_id) in expired {
                        bridge.questions.lock().unwrap().remove(&id);
                        let _ = bridge
                            .connection
                            .respond_to_extension_ui_request(
                                &request_id,
                                AgentConnectionExtensionUiResponse::Cancelled { cancelled: true },
                            )
                            .await;
                    }
                });
            });
            *typing.lock().unwrap() = Some(handle);
        }
        let mut failures = 0.0f64;
        while !controller.is_cancelled() {
            // Drop the state guard before polling or accepting an update. Holding
            // it through the match blocks replies, delivery timers and accept().
            let offset = self.shared.state.lock().unwrap().offset;
            match self
                .api
                .updates(offset, request_signal.clone())
                .await
            {
                Ok(updates) => {
                    let mut failure: Option<TelegramBridgeError> = None;
                    for update in updates {
                        if controller.is_cancelled() {
                            break;
                        }
                        if let Err(error) = self.accept(&update).await {
                            failure = Some(error);
                            break;
                        }
                    }
                    if let Some(error) = failure {
                        if controller.is_cancelled() {
                            break;
                        }
                        if matches!(error.code(), Some(401.0) | Some(409.0)) {
                            return Err(error);
                        }
                        (self.report_error)(Some(self.api.redact(&error.message())));
                        failures += 1.0;
                        let exponent = (failures - 1.0).min(5.0);
                        let backoff = match error.retry_after() {
                            Some(retry_after) => retry_after * 1000.0,
                            None => 30_000.0f64.min(1000.0 * 2f64.powf(exponent)),
                        };
                        let _ = delay_ms(backoff as u64, controller).await;
                        continue;
                    }
                    failures = 0.0;
                    if *self.delivery_failures.lock().unwrap() == 0.0 {
                        (self.report_error)(None);
                    }
                }
                Err(error) => {
                    if controller.is_cancelled() {
                        break;
                    }
                    let error = call_error_to_bridge(error);
                    if matches!(error.code(), Some(401.0) | Some(409.0)) {
                        return Err(error);
                    }
                    (self.report_error)(Some(self.api.redact(&error.message())));
                    let backoff = match error.retry_after() {
                        Some(retry_after) => retry_after * 1000.0,
                        None => {
                            failures += 1.0;
                            let exponent = (failures - 1.0).min(5.0);
                            30_000.0f64.min(1000.0 * 2f64.powf(exponent))
                        }
                    };
                    let _ = delay_ms(backoff as u64, controller).await;
                }
            }
        }
        Ok(())
    }

    /// `async waitForDispatch()`.
    pub async fn wait_for_dispatch(&self) {
        let draining = self.draining.lock().unwrap().clone();
        if let Some(draining) = draining {
            loop {
                if *draining.finished.lock().unwrap() {
                    return;
                }
                draining.done.notified().await;
            }
        }
    }
}

/// JavaScript regex `\s` for the `\S`/`\s` classes this module uses.
fn is_js_space(character: char) -> bool {
    character.is_whitespace() || character == '\u{feff}'
}

/// `/^(\S+)\s+([\s\S]+)$/` on the `/answer` arguments.
///
/// Returns `(match[1], match[2])`; `None` mirrors "no match".
fn parse_answer_args(args: &str) -> Option<(String, String)> {
    let mut chars = args.char_indices().peekable();
    let mut head_end = 0usize;
    while let Some((index, character)) = chars.peek().copied() {
        if is_js_space(character) {
            break;
        }
        head_end = index + character.len_utf8();
        chars.next();
    }
    if head_end == 0 {
        return None;
    }
    let mut rest_start = args.len();
    {
        let mut seen_space = false;
        let mut offset = head_end;
        for character in args[head_end..].chars() {
            if !is_js_space(character) {
                break;
            }
            seen_space = true;
            offset += character.len_utf8();
            rest_start = offset;
        }
        if !seen_space {
            return None;
        }
    }
    if rest_start >= args.len() {
        return None;
    }
    Some((args[..head_end].to_string(), args[rest_start..].to_string()))
}

/// `/^\d+$/`.
fn is_all_digits(value: &str) -> bool {
    !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
}

/// `message.timestamp` - the epoch-millisecond timestamp of any message shape.
fn message_timestamp(message: &AgentMessage) -> i64 {
    match message {
        AgentMessage::Message(pi_ai::types::Message::User(user)) => user.timestamp,
        AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => assistant.timestamp,
        AgentMessage::Message(pi_ai::types::Message::ToolResult(result)) => result.timestamp,
        AgentMessage::Custom(custom) => match custom {
            pi_agent_core::types::CustomAgentMessage::BashExecution { timestamp, .. }
            | pi_agent_core::types::CustomAgentMessage::Custom { timestamp, .. }
            | pi_agent_core::types::CustomAgentMessage::BranchSummary { timestamp, .. }
            | pi_agent_core::types::CustomAgentMessage::CompactionSummary { timestamp, .. } => *timestamp,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::telegram::store::{pairing_hash, TelegramPairing};

    fn settings() -> TelegramConnectionSettings {
        TelegramConnectionSettings {
            version: 1.0,
            enabled: true,
            bot_token: "123456789:AAAAAAAAAAAAAAAAAAAA".to_string(),
            bot_id: 7.0,
            bot_username: "prime_bot".to_string(),
            daemon_socket: "/tmp/daemon.sock".to_string(),
            cwd: "/work".to_string(),
            session_id: "session-1".to_string(),
            session_file: None,
            paired_user_id: None,
            pairing: Some(TelegramPairing {
                hash: pairing_hash("CODE"),
                expires_at: (now_ms() + 60_000) as f64,
            }),
        }
    }

    fn pairing_update(text: &str) -> serde_json::Value {
        serde_json::json!({
            "update_id": 12,
            "message": {
                "message_id": 3,
                "date": 1000,
                "from": { "id": 42, "is_bot": false },
                "chat": { "id": 42, "type": "private" },
                "text": text,
            },
        })
    }

    #[test]
    fn pairing_accepts_only_the_hashed_code_from_a_private_chat() {
        let mut valid = settings();
        assert!(accept_telegram_pairing(&mut valid, &pairing_update("/start CODE")));
        assert_eq!(valid.paired_user_id, Some(42.0));
        assert!(valid.pairing.is_none());

        let mut wrong = settings();
        assert!(!accept_telegram_pairing(&mut wrong, &pairing_update("/start NOPE")));
        assert!(wrong.paired_user_id.is_none());

        let mut already_paired = settings();
        already_paired.paired_user_id = Some(1.0);
        assert!(!accept_telegram_pairing(&mut already_paired, &pairing_update("/start CODE")));

        let mut other_command = settings();
        assert!(!accept_telegram_pairing(&mut other_command, &pairing_update("/help CODE")));

        let mut expired = settings();
        expired.pairing = Some(TelegramPairing {
            hash: pairing_hash("CODE"),
            expires_at: (now_ms() - 1) as f64,
        });
        assert!(!accept_telegram_pairing(&mut expired, &pairing_update("/start CODE")));

        let mut long_code = settings();
        let long = "x".repeat(65);
        assert!(!accept_telegram_pairing(&mut long_code, &pairing_update(&format!("/start {long}"))));
    }

    #[test]
    fn answer_arguments_follow_the_typescript_regex() {
        assert_eq!(
            parse_answer_args("ab12 yes please"),
            Some(("ab12".to_string(), "yes please".to_string()))
        );
        assert_eq!(parse_answer_args("ab12"), None);
        assert_eq!(parse_answer_args(" ab12 yes"), None);
        assert_eq!(parse_answer_args("ab12 "), None);
        assert_eq!(parse_answer_args("ab12  1"), Some(("ab12".to_string(), "1".to_string())));
    }

    #[test]
    fn select_answers_require_digits() {
        assert!(is_all_digits("12"));
        assert!(!is_all_digits(""));
        assert!(!is_all_digits("1a"));
    }

    #[test]
    fn message_text_reads_only_content_bearing_messages() {
        let assistant = AgentMessage::Message(pi_ai::types::Message::Assistant(pi_ai::types::AssistantMessage {
            content: vec![
                pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new("first")),
                pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::default()),
                pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new("second")),
            ],
            ..Default::default()
        }));
        assert_eq!(message_text(&assistant), "first\nsecond");

        let bash = AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::BashExecution {
            command: "ls".to_string(),
            output: "file".to_string(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 0,
            exclude_from_context: None,
        });
        assert_eq!(message_text(&bash), "");
        assert!(!message_display(&bash));

        let custom = AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage::Custom {
            custom_type: "notice".to_string(),
            content: pi_agent_core::types::CustomMessageContent::Text("hello".to_string()),
            display: true,
            details: None,
            timestamp: 5,
        });
        assert_eq!(message_text(&custom), "hello");
        assert!(message_display(&custom));
        assert_eq!(message_timestamp(&custom), 5);
    }

    #[test]
    fn message_error_message_is_used_for_empty_assistant_text() {
        let mut assistant = pi_ai::types::AssistantMessage::default();
        assistant.error_message = Some("provider failed".to_string());
        let message = AgentMessage::Message(pi_ai::types::Message::Assistant(assistant));
        assert_eq!(message_text(&message), "");
        assert_eq!(message_error_message(&message).as_deref(), Some("provider failed"));
    }

    #[test]
    fn sha256_keys_match_the_typescript_dedupe_key() {
        assert_eq!(
            sha256_hex("session-1:12:hi"),
            "01fbbc5620a195e70ea5e06164dcf93dd19bed8f59aa4f39f4112cae02f89c32"
        );
    }
}
