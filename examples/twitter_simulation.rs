//! Twitter simulation example.
//!
//! Demonstrates setting up a Twitter-style simulation with 3 agents,
//! seeding initial posts, and running multiple simulation steps.
//!
//! Prerequisites:
//!   - Set `OPENAI_API_KEY` environment variable
//!
//! Run with:
//!   cargo run --example twitter_simulation

use std::collections::HashMap;
use std::sync::Arc;

use augur::agent::graph::AgentGraph;
use augur::agent::{SocialAgent, UserInfo};
use augur::channel::Channel;
use augur::env_action::{Action, LLMAction, ManualAction};
use augur::llm::openai::OpenAIBackend;
use augur::types::{ActionType, DefaultPlatformType};
use augur::PlatformOrDefault;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging so we can see agent actions in the console.
    tracing_subscriber::fmt::init();

    // --- Step 1: Create an LLM backend ---
    // OpenAIBackend reads OPENAI_API_KEY from the environment automatically.
    let model = Arc::new(OpenAIBackend::new("gpt-4o-mini"));

    // --- Step 2: Define agent personas ---
    // Each agent gets a unique personality that influences LLM-driven behavior.
    let personas = vec![
        (
            "policy_wonk",
            "Alex Chen",
            "Former congressional staffer turned policy analyst. Tweets about legislation, \
             government accountability, and data-driven policy.",
        ),
        (
            "grassroots_org",
            "Maria Torres",
            "Community organizer in Phoenix, AZ. Focused on local politics, voter registration, \
             and immigration reform. Bilingual English/Spanish.",
        ),
        (
            "media_critic",
            "Jordan Blake",
            "Independent journalist covering media bias and misinformation. Skeptical of both \
             major parties. Values primary sources over punditry.",
        ),
    ];

    // --- Step 3: Build the agent graph ---
    // Create agents and wire up follow relationships.
    let channel = Arc::new(Channel::new());
    let mut graph = AgentGraph::new();

    let actions = vec![
        ActionType::CreatePost,
        ActionType::LikePost,
        ActionType::Repost,
        ActionType::CreateComment,
        ActionType::Follow,
        ActionType::DoNothing,
    ];

    for (i, (username, name, bio)) in personas.iter().enumerate() {
        let info = UserInfo::new(username, name, bio);
        let agent = SocialAgent::new(
            i as i64,
            info,
            channel.clone(),
            Some(model.clone()),
            None,
            actions.clone(),
        );
        graph.add_agent(i as i64, agent);
    }

    // Alex follows Maria, Maria follows Jordan, Jordan follows Alex (a cycle).
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 0);

    // --- Step 4: Create the simulation environment ---
    // The database stores all platform state (posts, likes, follows, etc.).
    let mut env = augur::make(
        graph,
        PlatformOrDefault::Default(DefaultPlatformType::Twitter),
        "twitter_sim.db",
        4, // max concurrent LLM calls
    )
    .await?;

    // --- Step 5: Reset (start platform, sign up agents) ---
    env.reset().await?;
    println!("Platform started. Agents signed up.");

    // --- Step 6: Seed initial posts ---
    // Manual actions let us inject content without calling the LLM.
    let mut seed_actions = HashMap::new();
    seed_actions.insert(
        0i64,
        Action::Manual(ManualAction::new(ActionType::CreatePost, {
            let mut args = HashMap::new();
            args.insert(
                "content".to_string(),
                serde_json::json!(
                    "New CBO analysis shows the infrastructure bill will create 800k jobs \
                     over the next decade. Worth reading the full methodology section."
                ),
            );
            args
        })),
    );
    seed_actions.insert(
        1,
        Action::Manual(ManualAction::new(ActionType::CreatePost, {
            let mut args = HashMap::new();
            args.insert(
                "content".to_string(),
                serde_json::json!(
                    "Voter registration deadline is coming up in Maricopa County! \
                     We're hosting free registration drives this weekend. Spread the word."
                ),
            );
            args
        })),
    );

    env.step(seed_actions).await?;
    println!("Seed posts created.");

    // --- Step 7: Run LLM-driven simulation steps ---
    // Each agent observes its feed and decides what to do autonomously.
    for round in 1..=3 {
        println!("--- Round {} ---", round);
        let mut llm_actions = HashMap::new();
        for agent_id in 0..3i64 {
            llm_actions.insert(agent_id, Action::Llm(LLMAction));
        }
        env.step(llm_actions).await?;
    }

    // --- Step 8: Shut down ---
    env.close().await?;
    println!("Simulation complete.");

    Ok(())
}
