//! Isolated native validation controller. Not a daemon wire endpoint or a
//! production setting. Uses the same authenticated-family constructor,
//! message controller trait, admission inbox and ordinary message renderer.
use crate::core::agent_messages::*;
use crate::core::agent_session::occ3_state::Occ3NativeSession;
use crate::core::occ::occ3::types::Occ3DeliveryStatus;
use pi_agent_core::agent::{Agent, PromptInput};
use pi_agent_core::types::AgentMessage;
use pi_ai::types::BoxFuture;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub struct IsolatedNativeMember {
    pub catalog: AgentFamilyCatalogEntry,
    pub endpoint: AgentSessionMessageEndpoint,
    pub agent: Arc<Agent>,
    pub native: Arc<Occ3NativeSession>,
}
/// Host-created catalog sealed before execution. Model/tool arguments cannot
/// create members, select a different sender, or install runtime state.
pub struct IsolatedNativeFamily {
    members: BTreeMap<String, IsolatedNativeMember>,
    receipts: Mutex<Vec<AgentSessionMessageReceipt>>,
}
impl IsolatedNativeFamily {
    pub fn new(members: Vec<IsolatedNativeMember>) -> Result<Arc<Self>, String> {
        if members.is_empty() || members.len() > 16 {
            return Err("Invalid isolated family".into());
        }
        let mut map = BTreeMap::new();
        for member in members {
            if member.catalog.id != member.endpoint.session_id
                || member.native.session_id() != member.catalog.id
                || map.contains_key(&member.catalog.id)
            {
                return Err("Isolated family identity mismatch".into());
            }
            map.insert(member.catalog.id.clone(), member);
        }
        Ok(Arc::new(Self {
            members: map,
            receipts: Mutex::new(Vec::new()),
        }))
    }
    pub fn controller(
        self: &Arc<Self>,
        session_id: &str,
    ) -> Result<Arc<dyn AgentSessionMessageController>, String> {
        if !self.members.contains_key(session_id) {
            return Err("Unknown isolated member".into());
        }
        Ok(Arc::new(IsolatedController {
            family: self.clone(),
            sender: session_id.into(),
        }))
    }
    pub fn install(self: &Arc<Self>) -> Result<(), String> {
        for (id, member) in &self.members {
            member
                .agent
                .set_resolve_inference_state(Some(member.native.receiver_resolver()));
            member
                .agent
                .set_source_capture_hooks(Some(member.native.source_hooks()));
            let mut state = member.agent.state();
            let mut tools = state.tools.take().unwrap_or_default();
            if tools
                .iter()
                .any(|t| t.name == crate::core::agent_session::occ3_state::NATIVE_HANDOFF_TOOL)
            {
                return Err("Native tool already installed".into());
            }
            tools.push(member.native.handoff_tool(self.controller(id)?));
            state.tools = Some(tools);
            member.agent.set_state(state);
        }
        Ok(())
    }
    pub fn receipts(&self) -> Vec<AgentSessionMessageReceipt> {
        self.receipts.lock().map(|r| r.clone()).unwrap_or_default()
    }
    pub async fn close(&self) -> Result<(), String> {
        let mut failed = false;
        for member in self.members.values() {
            member.agent.abort();
            if member.native.close().await.is_err() {
                failed = true;
            }
        }
        if failed {
            Err("Native retirement outcome uncertain".into())
        } else {
            Ok(())
        }
    }
}
struct IsolatedController {
    family: Arc<IsolatedNativeFamily>,
    sender: String,
}
impl IsolatedController {
    fn resolve(&self, selector: &str) -> Result<&IsolatedNativeMember, String> {
        let mut found = self.family.members.values().filter(|m| {
            m.endpoint.session_id == selector
                || m.endpoint.active_session_id == selector
                || m.endpoint.session_name.as_deref() == Some(selector)
        });
        let member = found.next().ok_or("Target not in isolated catalog")?;
        if found.next().is_some() {
            return Err("Ambiguous isolated target".into());
        }
        let sender = self
            .family
            .members
            .get(&self.sender)
            .ok_or("Sender unavailable")?;
        assert_agent_family_reach(&sender.catalog, &member.catalog)?;
        Ok(member)
    }
}
impl AgentSessionMessageController for IsolatedController {
    fn roster(&self) -> BoxFuture<Result<AgentFamilyRosterResult, String>> {
        let sender = &self.family.members[&self.sender];
        let entries = self
            .family
            .members
            .values()
            .filter_map(|m| {
                agent_family_relationship(&sender.catalog, &m.catalog).map(|relationship| {
                    AgentFamilyRosterEntry {
                        relationship,
                        name: m
                            .endpoint
                            .session_name
                            .clone()
                            .unwrap_or_else(|| m.catalog.id.clone()),
                        id: m.catalog.id.clone(),
                        depth: m.catalog.depth,
                        status: FAMILY_STATUS_IDLE.into(),
                        replied_since_task: None,
                    }
                })
            })
            .collect();
        let result = AgentFamilyRosterResult {
            current: AgentFamilyRosterCurrent {
                name: sender.endpoint.session_name.clone().unwrap_or_default(),
                id: self.sender.clone(),
                depth: sender.catalog.depth,
            },
            entries,
        };
        Box::pin(async move { Ok(result) })
    }
    fn await_pending_child_publication(
        &self,
        _selector: String,
    ) -> BoxFuture<Result<Option<String>, String>> {
        Box::pin(async { Ok(None) })
    }
    fn authorize_native_handoff(
        &self,
        relationship: String,
        selector: String,
    ) -> BoxFuture<Result<AuthorizedNativeHandoff, String>> {
        let result = self.resolve(&selector).and_then(|receiver| {
            let sender = &self.family.members[&self.sender];
            AuthorizedNativeHandoff::from_catalog(
                &sender.catalog,
                &receiver.catalog,
                sender.endpoint.clone(),
                receiver.endpoint.clone(),
                &relationship,
            )
        });
        Box::pin(async move { result })
    }
    fn send_prepared_native_handoff(
        &self,
        prepared: PreparedNativeHandoff,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let family = self.family.clone();
        let sender = self.sender.clone();
        Box::pin(async move {
            let from = family.members.get(&sender).ok_or("Sender unavailable")?;
            let receiver = family
                .members
                .get(prepared.authorization().target_session_id())
                .ok_or("Receiver unavailable")?;
            assert_agent_family_reach(&from.catalog, &receiver.catalog)?;
            if prepared.authorization().source_session_id() != sender {
                return Err("Native sender binding mismatch".into());
            }
            if receiver.agent.state().is_streaming {
                return Err("Isolated receiver is busy; no replay".into());
            }
            if !receiver.native.admit(&prepared)? {
                return Err("Native duplicate; no replay".into());
            }
            let payload = prepared.control_payload();
            let mut delivery = create_agent_session_message_delivery(
                &payload,
                chrono::Utc::now().timestamp_millis(),
                &Default::default(),
                &sender,
                &receiver.endpoint.session_id,
                &mut Occ2StrikeLedger::new(),
                chrono::Utc::now().timestamp_millis() as u64,
            );
            // Admitted, not yet consumed: the marker suppresses the control-only
            // body from the activity projection instead of reporting it as text.
            let pending = receiver.native.delivery_marker(&payload.id)?;
            stamp_occ3_delivery_record(&mut delivery.record, &pending)?;
            // The control-only delivery record is suppressed from
            // model-visible text by the pending delivery marker. Measured on
            // the live window (attempt-04d) and the 2026-10-30 option-B
            // diagnostic: without an explicit user turn the receiver's
            // request body carries only its developer prompt, so a live
            // model presented with the server-injected sender prefix
            // continues the SENDER's task instead of answering as itself.
            // The envelope's receiver-framed control summary is the
            // sanctioned, source-fact-free question; appending it as the
            // LAST user turn anchors the reply to the receiver's own task.
            let receiver_question = AgentMessage::Message(pi_ai::types::Message::User(
                pi_ai::types::UserMessage::new(
                    pi_ai::types::UserContent::Text(
                        prepared.envelope().control_summary.clone(),
                    ),
                    chrono::Utc::now().timestamp_millis(),
                ),
            ));
            receiver
                .agent
                .prompt(PromptInput::Messages(vec![
                    AgentMessage::Custom(delivery.record),
                    receiver_question,
                ]))
                .await
                .map_err(|_| "Native receiver failed")?;
            if receiver.native.outcome(&payload.id)? != Occ3DeliveryStatus::Consumed {
                return Err("Native receiver outcome uncertain".into());
            }
            // Completed use: one in-place rewrite, same record and message id.
            let marker = receiver.native.stamp_delivery_marker(&receiver.agent, &payload.id)?;
            if !marker.is_completed_native_latent() {
                return Err("Native receiver completed-use receipt missing".into());
            }
            let receipt = create_agent_session_message_receipt(
                &payload,
                &DELIVERY_STATUS_DELIVERED.into(),
                &chrono::Utc::now().to_rfc3339(),
            );
            let mut receipts = family
                .receipts
                .lock()
                .map_err(|_| "Receipt journal unavailable")?;
            if receipts.len() >= 256 {
                return Err("Receipt journal full".into());
            }
            receipts.push(receipt.clone());
            Ok(receipt)
        })
    }
    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let resolved = self
            .resolve(&input.target)
            .map(|member| member.endpoint.session_id.clone());
        let family = self.family.clone();
        let sender = self.sender.clone();
        Box::pin(async move {
            let id = resolved?;
            let receiver = &family.members[&id];
            let from = &family.members[&sender];
            let relationship = assert_agent_family_reach(&from.catalog, &receiver.catalog)?;
            let message =
                normalize_agent_session_message(&input.message, DEFAULT_AGENT_MESSAGE_MAX_CHARS)?;
            let payload = AgentSessionMessagePayload {
                id: create_agent_session_message_id(),
                source: AGENT_MESSAGE_SOURCE.into(),
                message,
                from: Some(AgentSessionMessageSender {
                    session_id: Some(sender.clone()),
                    session_name: from.endpoint.session_name.clone(),
                    ..Default::default()
                }),
                from_relationship: Some(relationship),
                target: receiver.endpoint.clone(),
            };
            let delivery = create_agent_session_message_delivery(
                &payload,
                chrono::Utc::now().timestamp_millis(),
                &Default::default(),
                &sender,
                &id,
                &mut Occ2StrikeLedger::new(),
                chrono::Utc::now().timestamp_millis() as u64,
            );
            receiver
                .agent
                .prompt(PromptInput::Messages(vec![AgentMessage::Custom(
                    delivery.record,
                )]))
                .await
                .map_err(|_| "Native normal delivery failed")?;
            Ok(create_agent_session_message_receipt(
                &payload,
                &DELIVERY_STATUS_DELIVERED.into(),
                &chrono::Utc::now().to_rfc3339(),
            ))
        })
    }
}
