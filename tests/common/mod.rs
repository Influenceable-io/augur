use augur::llm::{ChatMessage, FunctionTool, LlmBackend, LlmResponse, ToolCall};
use anyhow::Result;
use async_trait::async_trait;

pub struct MockLlm {
    pub response_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl MockLlm {
    pub fn do_nothing() -> Self {
        Self {
            response_content: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                function_name: "do_nothing".to_string(),
                arguments: serde_json::json!({}),
            }],
        }
    }

    pub fn create_post(content: &str) -> Self {
        Self {
            response_content: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                function_name: "create_post".to_string(),
                arguments: serde_json::json!({"content": content}),
            }],
        }
    }
}

#[async_trait]
impl LlmBackend for MockLlm {
    async fn chat_completion(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Vec<FunctionTool>,
    ) -> Result<LlmResponse> {
        Ok(LlmResponse {
            content: self.response_content.clone(),
            tool_calls: self.tool_calls.clone(),
        })
    }
}
