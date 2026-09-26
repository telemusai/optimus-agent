use super::*;
use crate::core::execution_mode::ExecutionMode;

const ENTRY: &str = "execution_mode";

impl AgentSession {
    pub(super) fn saved_execution_mode(&self) -> Option<ExecutionMode> {
        self.session_manager
            .lock()
            .unwrap()
            .get_branch(None)
            .iter()
            .rev()
            .find(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("custom")
                    && entry.get("customType").and_then(Value::as_str) == Some(ENTRY)
            })
            .and_then(|entry| {
                entry
                    .get("data")?
                    .get("mode")?
                    .as_str()
                    .and_then(|mode| ExecutionMode::parse(mode).ok())
            })
    }

    /// Runs only inside the session command's commit fence, between agent runs.
    pub(super) fn change_execution_mode(&self, args: &str) -> Result<String, String> {
        let current = ExecutionMode::from_tools(&self.get_active_tool_names());
        let mode = match args.trim() {
            "" => {
                return Ok(format!(
                    "Execution mode: {}",
                    current.map(ExecutionMode::label).unwrap_or("Custom tools")
                ))
            }
            "toggle" => current
                .ok_or("Cannot toggle a custom tool set; use /mode ipython or /mode direct")?
                .toggled(),
            value => ExecutionMode::parse(value)?,
        };
        if self.base_tools_override.is_some() {
            return Err("Execution mode switching is unavailable with an SDK tool override".into());
        }
        let names = mode.tools(&self.get_active_tool_names());
        let tools: Vec<_> = {
            let registry = self.tool_registry.lock().unwrap();
            names
                .iter()
                .map(|name| {
                    registry.get(name).cloned().ok_or_else(|| {
                        format!(
                            "Cannot enable {}: tool {name} is unavailable or restricted",
                            mode.label()
                        )
                    })
                })
                .collect::<Result<_, _>>()?
        };
        self.session_manager
            .lock()
            .unwrap()
            .append_custom_entry_with_rollback(
                ENTRY,
                Some(serde_json::json!({"mode": mode.as_str()})),
            )?;
        let prompt = self.rebuild_system_prompt(&names);
        *self.base_system_prompt.lock().unwrap() = prompt.clone();
        *self.owned_skill_hint_projection.lock().unwrap() = None;
        self.agent.update_state(Box::new(move |state| {
            state.system_prompt = prompt;
            state.tools = Some(tools);
        }));
        Ok(format!("Execution mode: {}", mode.label()))
    }
}
