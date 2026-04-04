use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;

use crate::agent::action::SocialAction;
use crate::agent::graph::AgentGraph;
use crate::agent::SocialAgent;
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

        // Sign up agents and capture agent_id -> user_id mapping.
        // HashMap iteration order is arbitrary, so SQLite autoincrement user_ids
        // won't necessarily equal agent_id + 1. We must use the actual user_ids.
        // Sign up agents, insert previous posts, and capture agent_id -> user_id mapping.
        // HashMap iteration order is arbitrary, so SQLite autoincrement user_ids
        // won't necessarily equal agent_id + 1. We must use the actual user_ids.
        let mut agent_to_user: HashMap<i64, i64> = HashMap::new();
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
            let (_, _, result) = self.channel.read_from_send_queue(msg_id).await;
            if let Some(uid) = result.data.get("user_id").and_then(|v| v.as_i64()) {
                agent_to_user.insert(agent_id, uid);
            }

            // Insert previous posts from agent profile (combined with sign-up pass)
            if let Some(profile) = &agent.user_info.profile
                && let Some(tweets_val) = profile.get(crate::agent::PROFILE_KEY_PREVIOUS_TWEETS)
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

        // Follow edges must be a separate pass — depends on the complete agent_to_user map.
        for (from, to) in graph.get_edges() {
            let followee_user_id = match agent_to_user.get(&to) {
                Some(&uid) => uid,
                None => {
                    tracing::warn!(from, to, "Follow edge target agent not found in user mapping");
                    continue;
                }
            };
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
            borrow_agent(agent_graph, agent_id, |mut agent: SocialAgent| async move {
                agent.perform_action_by_llm().await;
                agent
            }).await;
        }
        Action::Interview { prompt } => {
            borrow_agent(agent_graph, agent_id, |mut agent: SocialAgent| async move {
                agent.perform_interview(&prompt).await;
                agent
            }).await;
        }
        Action::Multiple(_) => {
            tracing::warn!("Nested Multiple actions not supported");
        }
    }
}

/// Take an agent out of the graph, run an async operation on it, then return it.
/// Releases the graph write lock during the operation so other agents can run concurrently.
async fn borrow_agent<F, Fut>(
    agent_graph: &RwLock<AgentGraph>,
    agent_id: i64,
    f: F,
) where
    F: FnOnce(SocialAgent) -> Fut,
    Fut: std::future::Future<Output = SocialAgent>,
{
    let agent = {
        let mut graph = agent_graph.write().await;
        match graph.take_agent(agent_id) {
            Some(a) => a,
            None => return,
        }
    };
    let agent = f(agent).await;
    let mut graph = agent_graph.write().await;
    graph.add_agent(agent_id, agent);
}
