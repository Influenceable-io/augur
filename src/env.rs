use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

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
                let pt = DefaultPlatformType::Twitter; // Custom platforms default to Twitter type
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
    pub async fn reset(&mut self) -> Result<()> {
        let platform = self.platform.clone();
        self.platform_task = Some(tokio::spawn(async move {
            platform.running().await;
        }));

        // TODO: Generate custom agents (sign_up, insert follows, insert posts)

        Ok(())
    }

    /// Execute one simulation step.
    ///
    /// 1. Update recommendation table
    /// 2. Process all agent actions concurrently
    /// 3. Increment clock (Twitter mode)
    pub async fn step(&mut self, actions: HashMap<i64, Action>) -> Result<()> {
        // 1. Update recommendations
        self.platform.update_rec_table().await;

        // 2. Process actions concurrently
        let mut tasks = Vec::new();
        for (agent_id, action) in actions {
            let semaphore = self.llm_semaphore.clone();
            let agent_graph = self.agent_graph.clone();
            let _channel = self.channel.clone();

            tasks.push(tokio::spawn(async move {
                match action {
                    Action::Manual(manual) => {
                        let graph = agent_graph.read().await;
                        if let Some(agent) = graph.get_agent(agent_id) {
                            agent
                                .perform_action_by_data(&manual.action_type.to_string(), serde_json::to_value(&manual.action_args).unwrap_or_default())
                                .await;
                        }
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
                    Action::Multiple(sub_actions) => {
                        for sub in sub_actions {
                            // Process each sub-action sequentially
                            match sub {
                                Action::Manual(manual) => {
                                    let graph = agent_graph.read().await;
                                    if let Some(agent) = graph.get_agent(agent_id) {
                                        agent
                                            .perform_action_by_data(&manual.action_type.to_string(), serde_json::to_value(&manual.action_args).unwrap_or_default())
                                            .await;
                                    }
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
                    }
                }
            }));
        }

        // Wait for all actions to complete
        for task in tasks {
            task.await?;
        }

        // 3. Increment clock for Twitter mode
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
