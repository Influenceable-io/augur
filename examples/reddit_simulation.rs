//! Reddit simulation example.
//!
//! Demonstrates setting up a Reddit-style simulation with 3 agents
//! using detailed persona profiles (bio, MBTI, demographics).
//!
//! Prerequisites:
//!   - Set `OPENAI_API_KEY` environment variable
//!
//! Run with:
//!   cargo run --example reddit_simulation

use std::collections::HashMap;
use std::sync::Arc;

use augur::agent::graph::AgentGraph;
use augur::agent::{SocialAgent, UserInfo};
use augur::channel::Channel;
use augur::env_action::{Action, LLMAction, ManualAction};
use augur::llm::openai::OpenAIBackend;
use augur::types::{ActionType, DefaultPlatformType, RecsysType};
use augur::PlatformOrDefault;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // --- Step 1: Create an LLM backend ---
    let model = Arc::new(OpenAIBackend::new("gpt-4o-mini"));

    // --- Step 2: Define Reddit-style personas ---
    // Reddit agents use richer profile data: bio, persona, MBTI, gender, age, country.
    #[allow(dead_code)]
    struct RedditPersona {
        username: &'static str,
        realname: &'static str,
        bio: &'static str,
        persona: &'static str,
        mbti: &'static str,
        gender: &'static str,
        age: u64,
        country: &'static str,
    }

    let personas = vec![
        RedditPersona {
            username: "fiscal_hawk_99",
            realname: "David Park",
            bio: "CPA by day, policy nerd by night. I read budget documents for fun.",
            persona: "Fiscally conservative, socially moderate. Believes in evidence-based \
                      policy and small government. Distrusts populism on both sides.",
            mbti: "ISTJ",
            gender: "male",
            age: 42,
            country: "United States",
        },
        RedditPersona {
            username: "verde_activist",
            realname: "Lucia Fernandez",
            bio: "Environmental science grad student. Organizing for climate justice.",
            persona: "Progressive environmentalist. Passionate about renewable energy, \
                      environmental justice, and holding corporations accountable. \
                      Skeptical of market-only solutions.",
            mbti: "ENFP",
            gender: "female",
            age: 26,
            country: "United States",
        },
        RedditPersona {
            username: "midwest_moderate",
            realname: "Sarah Johnson",
            bio: "High school teacher in Iowa. Two kids. Just trying to make sense of it all.",
            persona: "Pragmatic centrist. Cares about education funding, healthcare costs, \
                      and keeping civil discourse alive. Tired of partisan extremes.",
            mbti: "ISFJ",
            gender: "female",
            age: 38,
            country: "United States",
        },
    ];

    // --- Step 3: Build agent graph with rich profiles ---
    let channel = Arc::new(Channel::new());
    let mut graph = AgentGraph::new();

    let actions = vec![
        ActionType::CreatePost,
        ActionType::CreateComment,
        ActionType::LikePost,
        ActionType::DislikePost,
        ActionType::Follow,
        ActionType::DoNothing,
    ];

    for (i, p) in personas.iter().enumerate() {
        // Build a detailed profile map, matching the Reddit generator format.
        let mut profile = HashMap::new();
        profile.insert("persona".to_string(), serde_json::json!(p.persona));
        profile.insert("mbti".to_string(), serde_json::json!(p.mbti));
        profile.insert("gender".to_string(), serde_json::json!(p.gender));
        profile.insert("age".to_string(), serde_json::json!(p.age));
        profile.insert("country".to_string(), serde_json::json!(p.country));

        let description = format!(
            "{}. MBTI: {}. Gender: {}. Age: {}. Country: {}.",
            p.persona, p.mbti, p.gender, p.age, p.country,
        );

        let info = UserInfo {
            user_name: Some(p.username.to_string()),
            name: Some(p.realname.to_string()),
            description: Some(description),
            profile: Some(profile),
            recsys_type: RecsysType::Reddit,
            is_controllable: false,
        };

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

    // Wire up follows: everyone follows everyone (small community).
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 0);
    graph.add_edge(1, 2);
    graph.add_edge(2, 0);
    graph.add_edge(2, 1);

    // --- Step 4: Create Reddit simulation environment ---
    let mut env = augur::make(
        graph,
        PlatformOrDefault::Default(DefaultPlatformType::Reddit),
        "reddit_sim.db",
        4,
    )
    .await?;

    // --- Step 5: Reset (start platform, sign up agents) ---
    env.reset().await?;
    println!("Reddit platform started. Agents signed up.");

    // --- Step 6: Seed a discussion thread ---
    let mut seed = HashMap::new();
    seed.insert(
        0i64,
        Action::Manual(ManualAction::new(ActionType::CreatePost, {
            let mut args = HashMap::new();
            args.insert(
                "content".to_string(),
                serde_json::json!(
                    "The new federal budget proposal increases defense spending by 8% while \
                     cutting EPA funding by 22%. What are your thoughts on these priorities?"
                ),
            );
            args
        })),
    );
    env.step(seed).await?;
    println!("Seed post created.");

    // --- Step 7: Run LLM-driven rounds ---
    // Agents will react to the seed post according to their personas.
    for round in 1..=3 {
        println!("--- Round {} ---", round);
        let mut llm_actions = HashMap::new();
        for id in 0..3i64 {
            llm_actions.insert(id, Action::Llm(LLMAction));
        }
        env.step(llm_actions).await?;
    }

    // --- Step 8: Shut down ---
    env.close().await?;
    println!("Reddit simulation complete.");

    Ok(())
}
