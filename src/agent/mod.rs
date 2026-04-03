pub mod action;
pub mod environment;
pub mod graph;
pub mod generator;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::channel::Channel;
use crate::llm::{ChatMessage, LlmBackend, Role};
use crate::types::*;
use action::SocialAction;
use environment::SocialEnvironment;
use graph::AgentGraph;

/// Agent profile information matching OASIS's UserInfo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub user_name: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub profile: Option<HashMap<String, Value>>,
    #[serde(default = "default_recsys_type")]
    pub recsys_type: RecsysType,
    #[serde(default)]
    pub is_controllable: bool,
}

fn default_recsys_type() -> RecsysType {
    RecsysType::Twitter
}

impl UserInfo {
    pub fn new(user_name: &str, name: &str, description: &str) -> Self {
        Self {
            user_name: Some(user_name.to_string()),
            name: Some(name.to_string()),
            description: Some(description.to_string()),
            profile: None,
            recsys_type: RecsysType::Twitter,
            is_controllable: false,
        }
    }

    /// Generate a system message from the user info, matching OASIS's behavior.
    pub fn to_system_message(&self) -> String {
        let name = self.name.as_deref().unwrap_or("Unknown");
        let desc = self.description.as_deref().unwrap_or("");

        let mut msg = format!("You are {name}.");
        if !desc.is_empty() {
            msg.push_str(&format!(" {desc}"));
        }

        // Append profile fields if present
        if let Some(profile) = &self.profile {
            for (key, value) in profile {
                if let Some(s) = value.as_str() {
                    msg.push_str(&format!(" {}: {}", key, s));
                }
            }
        }

        msg
    }
}

/// LLM-powered social media agent matching OASIS's SocialAgent.
pub struct SocialAgent {
    pub agent_id: i64,
    pub user_info: UserInfo,
    action: SocialAction,
    env: SocialEnvironment,
    model: Option<Arc<dyn LlmBackend>>,
    agent_graph: Option<Arc<RwLock<AgentGraph>>>,
    available_actions: Vec<ActionType>,
    memory: Vec<ChatMessage>,
    max_iteration: usize,
    interview_record: bool,
}

impl SocialAgent {
    pub fn new(
        agent_id: i64,
        user_info: UserInfo,
        channel: Arc<Channel>,
        model: Option<Arc<dyn LlmBackend>>,
        agent_graph: Option<Arc<RwLock<AgentGraph>>>,
        available_actions: Vec<ActionType>,
    ) -> Self {
        let action = SocialAction::new(agent_id, channel);
        let env = SocialEnvironment::new(action.clone());

        Self {
            agent_id,
            user_info,
            action,
            env,
            model,
            agent_graph,
            available_actions,
            memory: Vec::new(),
            max_iteration: 1,
            interview_record: false,
        }
    }

    /// LLM-driven action: observe environment, decide action via tool calling.
    pub async fn perform_action_by_llm(&mut self) -> ActionResult {
        let model = match &self.model {
            Some(m) => m.clone(),
            None => return ActionResult::fail("no LLM model configured"),
        };

        // 1. Build observation prompt
        let env_text = self.env.to_text_prompt(true, true, true, false).await;
        let system_msg = self.user_info.to_system_message();
        let user_msg = format!(
            "Here is what you see on social media right now:\n\n{}\n\nBased on your persona, choose an action to take.",
            env_text
        );

        // 2. Build messages
        let mut messages = vec![
            ChatMessage {
                role: Role::System,
                content: system_msg,
            },
        ];
        // Include recent memory
        messages.extend(self.memory.iter().cloned());
        messages.push(ChatMessage {
            role: Role::User,
            content: user_msg.clone(),
        });

        // 3. Get available tools
        let tools = self.action.get_function_tools(&self.available_actions);

        // 4. Call LLM
        match model.chat_completion(messages, tools).await {
            Ok(response) => {
                // Record in memory
                self.memory.push(ChatMessage {
                    role: Role::User,
                    content: user_msg,
                });
                if let Some(content) = &response.content {
                    self.memory.push(ChatMessage {
                        role: Role::Assistant,
                        content: content.clone(),
                    });
                }

                // 5. Execute tool calls
                for tool_call in &response.tool_calls {
                    let result = self
                        .action
                        .perform_action_by_name(&tool_call.function_name, &tool_call.arguments)
                        .await;

                    // Update agent graph on follow/unfollow
                    if let Some(graph) = &self.agent_graph {
                        self.perform_agent_graph_action(
                            &tool_call.function_name,
                            &tool_call.arguments,
                            graph,
                        )
                        .await;
                    }

                    tracing::info!(
                        agent_id = self.agent_id,
                        action = tool_call.function_name,
                        success = result.success,
                        "Agent action executed"
                    );
                }

                ActionResult::ok(serde_json::json!({}))
            }
            Err(e) => ActionResult::fail(&format!("LLM call failed: {}", e)),
        }
    }

    /// Execute a scripted action by function name.
    pub async fn perform_action_by_data(&self, func_name: &str, args: Value) -> ActionResult {
        self.action.perform_action_by_name(func_name, &args).await
    }

    /// Human-in-the-loop interactive mode.
    pub async fn perform_action_by_hci(&mut self) -> ActionResult {
        // Display available actions
        println!("Available actions for agent {}:", self.agent_id);
        for (i, action) in self.available_actions.iter().enumerate() {
            println!("  {}: {:?}", i, action);
        }

        // Read from stdin
        println!("Enter action index: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap_or_default();

        let idx: usize = input.trim().parse().unwrap_or(0);
        if idx < self.available_actions.len() {
            let action_type = self.available_actions[idx];
            println!("Enter arguments (JSON): ");
            let mut args_input = String::new();
            std::io::stdin().read_line(&mut args_input).unwrap_or_default();
            let args: Value = serde_json::from_str(args_input.trim()).unwrap_or(Value::Null);
            self.action.perform_action_by_name(&action_type.to_string(), &args).await
        } else {
            ActionResult::fail("invalid action index")
        }
    }

    /// Group polarization test.
    pub async fn perform_test(&mut self) -> Value {
        let model = match &self.model {
            Some(m) => m.clone(),
            None => return serde_json::json!({"error": "no model"}),
        };

        let system_msg = self.user_info.to_system_message();
        let test_prompt = "Please share your honest opinion on the current political situation.";

        let messages = vec![
            ChatMessage { role: Role::System, content: system_msg },
            ChatMessage { role: Role::User, content: test_prompt.to_string() },
        ];

        match model.chat_completion(messages, vec![]).await {
            Ok(response) => {
                serde_json::json!({
                    "user_id": self.agent_id,
                    "prompt": test_prompt,
                    "content": response.content.unwrap_or_default(),
                })
            }
            Err(e) => serde_json::json!({"error": e.to_string()}),
        }
    }

    /// Structured interview with optional memory recording.
    pub async fn perform_interview(&mut self, interview_prompt: &str) -> Value {
        let model = match &self.model {
            Some(m) => m.clone(),
            None => return serde_json::json!({"error": "no model"}),
        };

        let system_msg = self.user_info.to_system_message();
        let messages = vec![
            ChatMessage { role: Role::System, content: system_msg },
            ChatMessage { role: Role::User, content: interview_prompt.to_string() },
        ];

        match model.chat_completion(messages, vec![]).await {
            Ok(response) => {
                let content = response.content.unwrap_or_default();

                if self.interview_record {
                    self.memory.push(ChatMessage {
                        role: Role::User,
                        content: interview_prompt.to_string(),
                    });
                    self.memory.push(ChatMessage {
                        role: Role::Assistant,
                        content: content.clone(),
                    });
                }

                serde_json::json!({
                    "user_id": self.agent_id,
                    "prompt": interview_prompt,
                    "content": content,
                    "success": true,
                })
            }
            Err(e) => serde_json::json!({
                "user_id": self.agent_id,
                "prompt": interview_prompt,
                "error": e.to_string(),
                "success": false,
            }),
        }
    }

    /// Update the agent graph on follow/unfollow actions.
    async fn perform_agent_graph_action(
        &self,
        action_name: &str,
        arguments: &Value,
        graph: &Arc<RwLock<AgentGraph>>,
    ) {
        match action_name {
            "follow" => {
                if let Some(followee_id) = arguments.get("followee_id").and_then(|v| v.as_i64()) {
                    let mut g = graph.write().await;
                    g.add_edge(self.agent_id, followee_id);
                }
            }
            "unfollow" => {
                if let Some(followee_id) = arguments.get("followee_id").and_then(|v| v.as_i64()) {
                    let mut g = graph.write().await;
                    g.remove_edge(self.agent_id, followee_id);
                }
            }
            _ => {}
        }
    }
}
