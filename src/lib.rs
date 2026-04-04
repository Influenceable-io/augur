pub mod types;
pub mod channel;
pub mod clock;
pub mod db;
pub mod platform;
pub mod recsys;
pub mod agent;
pub mod llm;
pub mod env;
pub mod env_action;

use anyhow::Result;

pub use types::*;
pub use channel::Channel;
pub use clock::Clock;
pub use env::AugurEnv;
pub use env_action::*;
pub use agent::{SocialAgent, UserInfo};
pub use agent::graph::{AgentGraph, GraphBackend, InMemoryGraph};
pub use agent::generator::{generate_twitter_agent_graph, generate_reddit_agent_graph};
pub use platform::Platform;
pub use llm::{ModelConfig, LlmBackend, openai::OpenAIBackend};

/// Factory function matching oasis.make()
pub async fn make(
    agent_graph: AgentGraph,
    platform: PlatformOrDefault,
    database_path: &str,
    semaphore: usize,
) -> Result<AugurEnv> {
    AugurEnv::new(agent_graph, platform, database_path, semaphore).await
}

/// Platform configuration: either a default type or a custom Platform instance.
pub enum PlatformOrDefault {
    Default(DefaultPlatformType),
    Custom(Platform),
}
