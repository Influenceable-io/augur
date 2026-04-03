use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

use crate::agent::action::SocialAction;
use crate::agent::graph::AgentGraph;
use crate::channel::Channel;
use crate::clock::Clock;
use crate::env_action::Action;
use crate::platform::{Platform, PlatformConfig};
use crate::types::*;
use crate::PlatformOrDefault;

/// Top-level simulation environment matching OASIS's OasisEnv.
///
/// Lifecycle: `make()` → `reset()` → `step()` (repeated) → `close()`
pub struct AugurEnv {
    pub agent_graph: Arc<RwLock<AgentGraph>>,
    pub platform: Arc<Platform>,
    pub platform_type: DefaultPlatformType,
    channel: Arc<Channel>,
    sandbox_clock: Arc<RwLock<Clock>>,
    platform_task: Option<JoinHandle<()>>,
    llm_semaphore: Arc<Semaphore>,
}

impl AugurEnv {
    pub async fn new(
        agent_graph: AgentGraph,
        platform_or_default: PlatformOrDefault,
        database_path: &str,
        semaphore: usize,
    ) -> Result<Self> {
        let channel = Arc::new(Channel::new());
        let llm_semaphore = Arc::new(Semaphore::new(semaphore));

        let (platform, platform_type, sandbox_clock) = match platform_or_default {
            PlatformOrDefault::Default(pt) => {
                let (recsys_type, clock_k) = match pt {
                    DefaultPlatformType::Twitter => (RecsysType::Twitter, 1),
                    DefaultPlatformType::Reddit => (RecsysType::Reddit, 60),
                };
                let clock = Arc::new(RwLock::new(Clock::new(clock_k)));
                let config = PlatformConfig::default();
                let platform = Platform::new(
                    database_path,
                    channel.clone(),
                    clock.clone(),
                    recsys_type,
                    config,
                )?;
                (Arc::new(platform), pt, clock)
            }
            PlatformOrDefault::Custom(p) => {
                let clock = p.sandbox_clock();
                let pt = DefaultPlatformType::Twitter;
                (Arc::new(p), pt, clock)
            }
        };

        Ok(Self {
            agent_graph: Arc::new(RwLock::new(agent_graph)),
            platform,
            platform_type,
            channel,
            sandbox_clock,
            platform_task: None,
            llm_semaphore,
        })
    }

    /// Start the platform event loop and register agents.
    ///
    /// This mirrors OASIS's reset():
    /// 1. Start the platform event loop
    /// 2. Sign up all agents on the platform
    /// 3. Insert follow relationships from the agent graph
    /// 4. Insert previous posts from agent profiles
    pub async fn reset(&mut self) -> Result<()> {
        let platform = self.platform.clone();
        self.platform_task = Some(tokio::spawn(async move {
            platform.running().await;
        }));

        let graph = self.agent_graph.read().await;
        for (agent_id, agent) in graph.get_agents() {
            let user_name = agent.user_info.user_name.as_deref().unwrap_or("unknown");
            let name = agent.user_info.name.as_deref().unwrap_or("unknown");
            let bio = agent.user_info.description.as_deref().unwrap_or("");

            let msg_id = self
                .channel
                .write_to_receive_queue(
                    agent_id,
                    serde_json::json!({"user_name": user_name, "name": name, "bio": bio}),
                    ActionType::SignUp,
                )
                .await;
            let _ = self.channel.read_from_send_queue(msg_id).await;
        }

        for (from, to) in graph.get_edges() {
            // In OASIS, followee_id is the user_id (1-indexed autoincrement)
            let followee_user_id = to + 1;
            let msg_id = self
                .channel
                .write_to_receive_queue(
                    from,
                    serde_json::json!(followee_user_id),
                    ActionType::Follow,
                )
                .await;
            let _ = self.channel.read_from_send_queue(msg_id).await;
        }

        for (agent_id, agent) in graph.get_agents() {
            if let Some(profile) = &agent.user_info.profile
                && let Some(tweets_val) = profile.get("previous_tweets")
            {
                let tweets_str = tweets_val.as_str().unwrap_or("");
                for tweet in tweets_str
                    .split(';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    let msg_id = self
                        .channel
                        .write_to_receive_queue(
                            agent_id,
                            serde_json::json!(tweet),
                            ActionType::CreatePost,
                        )
                        .await;
                    let _ = self.channel.read_from_send_queue(msg_id).await;
                }
            }
        }

        Ok(())
    }

    /// Execute one simulation step.
    ///
    /// 1. Update recommendation table
    /// 2. Process all agent actions concurrently
    /// 3. Increment clock (Twitter mode)
    pub async fn step(&mut self, actions: HashMap<i64, Action>) -> Result<()> {
        self.platform.update_rec_table().await;

        let mut tasks = Vec::new();
        for (agent_id, action) in actions {
            let semaphore = self.llm_semaphore.clone();
            let agent_graph = self.agent_graph.clone();
            let channel = self.channel.clone();

            tasks.push(tokio::spawn(async move {
                match action {
                    Action::Multiple(sub_actions) => {
                        for sub in sub_actions {
                            process_single_action(
                                agent_id,
                                sub,
                                &semaphore,
                                &agent_graph,
                                channel.clone(),
                            )
                            .await;
                        }
                    }
                    single => {
                        process_single_action(
                            agent_id,
                            single,
                            &semaphore,
                            &agent_graph,
                            channel.clone(),
                        )
                        .await;
                    }
                }
            }));
        }

        for task in tasks {
            task.await?;
        }

        if self.platform_type == DefaultPlatformType::Twitter {
            let mut clock = self.sandbox_clock.write().await;
            clock.time_step += 1;
        }

        Ok(())
    }

    /// Shut down the simulation. Sends EXIT, waits for platform to finish.
    pub async fn close(&mut self) -> Result<()> {
        self.channel
            .write_to_receive_queue(0, serde_json::Value::Null, ActionType::Exit)
            .await;

        if let Some(task) = self.platform_task.take() {
            task.await?;
        }

        tracing::info!("Simulation closed");
        Ok(())
    }
}

/// Process a single (non-Multiple) action for an agent.
///
/// Routes Manual actions through SocialAction::perform_action_by_name
/// so arg extraction logic is defined in exactly one place.
async fn process_single_action(
    agent_id: i64,
    action: Action,
    semaphore: &Semaphore,
    agent_graph: &RwLock<AgentGraph>,
    channel: Arc<Channel>,
) {
    match action {
        Action::Manual(manual) => {
            let sa = SocialAction::new(agent_id, channel);
            let args_value =
                serde_json::to_value(&manual.action_args).unwrap_or_default();
            sa.perform_action_by_name(&manual.action_type.to_string(), &args_value)
                .await;
        }
        Action::Llm(_) => {
            let _permit = semaphore.acquire().await.unwrap();
            let mut graph = agent_graph.write().await;
            if let Some(agent) = graph.get_agent_mut(agent_id) {
                agent.perform_action_by_llm().await;
            }
        }
        Action::Interview { prompt } => {
            let mut graph = agent_graph.write().await;
            if let Some(agent) = graph.get_agent_mut(agent_id) {
                agent.perform_interview(&prompt).await;
            }
        }
        Action::Multiple(_) => {
            tracing::warn!("Nested Multiple actions not supported");
        }
    }
}
