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

/// Trait for LLM backends. Implement this for different providers.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<FunctionTool>,
    ) -> Result<LlmResponse>;
}
