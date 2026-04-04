use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionToolArgs, ChatCompletionToolType, CreateChatCompletionRequestArgs,
        FunctionCall, FunctionObjectArgs,
    },
    Client,
};
use async_trait::async_trait;

use super::{ChatMessage, FunctionTool, LlmBackend, LlmResponse, ModelConfig, Role, ToolCall};

/// OpenAI-compatible LLM backend using async-openai.
///
/// Works with OpenAI, OpenRouter, vLLM, or any OpenAI-compatible endpoint
/// by setting the base_url.
pub struct OpenAIBackend {
    client: Client<OpenAIConfig>,
    model: String,
    config: ModelConfig,
}

impl OpenAIBackend {
    /// Create with default OpenAI configuration (reads OPENAI_API_KEY env var).
    pub fn new(model: &str) -> Self {
        Self {
            client: Client::new(),
            model: model.to_string(),
            config: ModelConfig::default(),
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
            config: ModelConfig::default(),
        }
    }

    /// Create for OpenRouter with model parameters.
    ///
    /// OpenRouter uses `https://openrouter.ai/api/v1` as base URL and
    /// accepts any model identifier (e.g. `anthropic/claude-sonnet-4`,
    /// `google/gemini-2.5-pro`, `meta-llama/llama-4-maverick`).
    ///
    /// Reads `OPENROUTER_API_KEY` env var if `api_key` is not provided.
    pub fn openrouter(model: &str, api_key: Option<&str>) -> Self {
        let key = api_key
            .map(String::from)
            .unwrap_or_else(|| std::env::var("OPENROUTER_API_KEY").unwrap_or_default());
        Self::with_base_url(model, "https://openrouter.ai/api/v1", &key)
    }

    /// Set generation parameters.
    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
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
                Role::Assistant => {
                    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                    builder.content(msg.content.as_str());
                    if !msg.tool_calls.is_empty() {
                        let oai_tool_calls: Vec<ChatCompletionMessageToolCall> = msg
                            .tool_calls
                            .iter()
                            .map(|tc| ChatCompletionMessageToolCall {
                                id: tc.id.clone(),
                                r#type: ChatCompletionToolType::Function,
                                function: FunctionCall {
                                    name: tc.function_name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect();
                        builder.tool_calls(oai_tool_calls);
                    }
                    builder.build().unwrap().into()
                }
                Role::Tool => ChatCompletionRequestToolMessageArgs::default()
                    .content(msg.content.as_str())
                    .tool_call_id(msg.tool_call_id.as_deref().unwrap_or(""))
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

        // Apply model config parameters
        if let Some(temp) = self.config.temperature {
            request_builder.temperature(temp);
        }
        if let Some(max_tok) = self.config.max_tokens {
            request_builder.max_tokens(max_tok);
        }
        if let Some(top_p) = self.config.top_p {
            request_builder.top_p(top_p);
        }
        if let Some(freq) = self.config.frequency_penalty {
            request_builder.frequency_penalty(freq);
        }
        if let Some(pres) = self.config.presence_penalty {
            request_builder.presence_penalty(pres);
        }

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
