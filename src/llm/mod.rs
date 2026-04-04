pub mod openai;
pub mod tools;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Message role in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Tool call ID (required when role is Tool).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant (required for multi-turn tool use).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function_name: String,
    pub arguments: Value,
}

/// Response from an LLM completion.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Function tool definition for LLM tool calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Configuration for LLM generation parameters.
///
/// All fields are optional — only set values are sent to the API.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Sampling temperature (0.0 = deterministic, 2.0 = very random).
    pub temperature: Option<f32>,
    /// Maximum tokens in the response.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Penalize tokens that already appeared in the text.
    pub frequency_penalty: Option<f32>,
    /// Penalize tokens that appeared at all.
    pub presence_penalty: Option<f32>,
}

/// Trait for LLM backends. Implement this for different providers.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<FunctionTool>,
    ) -> Result<LlmResponse>;
}
