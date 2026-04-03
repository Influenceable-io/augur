use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::ActionType;

/// A scripted action with explicit type and arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualAction {
    pub action_type: ActionType,
    pub action_args: HashMap<String, Value>,
}

impl ManualAction {
    pub fn new(action_type: ActionType, action_args: HashMap<String, Value>) -> Self {
        Self {
            action_type,
            action_args,
        }
    }
}

/// Marker for LLM-driven autonomous action selection.
#[derive(Debug, Clone, Default)]
pub struct LLMAction;

/// An action that an agent can take during a simulation step.
#[derive(Debug, Clone)]
pub enum Action {
    /// Scripted action with explicit type and arguments.
    Manual(ManualAction),
    /// Agent observes environment and decides via LLM.
    Llm(LLMAction),
    /// Structured interview with a prompt.
    Interview { prompt: String },
    /// Multiple actions in sequence for a single step.
    Multiple(Vec<Action>),
}
