use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionToolArgs, ChatCompletionToolType, CreateChatCompletionRequestArgs,
        FunctionObjectArgs,
    },
    Client,
};
use async_trait::async_trait;

use super::{ChatMessage, FunctionTool, LlmBackend, LlmResponse, Role, ToolCall};

/// OpenAI-compatible LLM backend using async-openai.
///
/// Works with OpenAI, OpenRouter, vLLM, or any OpenAI-compatible endpoint
/// by setting the base_url.
pub struct OpenAIBackend {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAIBackend {
    /// Create with default OpenAI configuration (reads OPENAI_API_KEY env var).
    pub fn new(model: &str) -> Self {
        Self {
            client: Client::new(),
            model: model.to_string(),
        }
    }

    /// Create with a custom base URL (for OpenRouter, vLLM, etc.).
    pub fn with_base_url(model: &str, base_url: &str, api_key: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key(api_key);
        Self {
            client: Client::with_config(config),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LlmBackend for OpenAIBackend {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<FunctionTool>,
    ) -> Result<LlmResponse> {
        // Convert messages
        let oai_messages: Vec<ChatCompletionRequestMessage> = messages
            .iter()
            .map(|msg| match msg.role {
                Role::System => ChatCompletionRequestSystemMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .unwrap()
                    .into(),
                Role::User => ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .unwrap()
                    .into(),
                Role::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .unwrap()
                    .into(),
                Role::Tool => ChatCompletionRequestUserMessageArgs::default()
                    .content(msg.content.as_str())
                    .build()
                    .unwrap()
                    .into(),
            })
            .collect();

        // Build request
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.model)
            .messages(oai_messages);

        // Add tools if any
        if !tools.is_empty() {
            let oai_tools: Vec<_> = tools
                .iter()
                .map(|tool| {
                    ChatCompletionToolArgs::default()
                        .r#type(ChatCompletionToolType::Function)
                        .function(
                            FunctionObjectArgs::default()
                                .name(&tool.name)
                                .description(&tool.description)
                                .parameters(tool.parameters.clone())
                                .build()
                                .unwrap(),
                        )
                        .build()
                        .unwrap()
                })
                .collect();
            request_builder.tools(oai_tools);
        }

        let request = request_builder.build()?;
        let response = self.client.chat().create(request).await?;

        // Parse response
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

        let content = choice.message.content.clone();

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| {
                        let args: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                        ToolCall {
                            id: tc.id.clone(),
                            function_name: tc.function.name.clone(),
                            arguments: args,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(LlmResponse {
            content,
            tool_calls,
        })
    }
}
