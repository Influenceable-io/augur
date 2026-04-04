use std::sync::Arc;

use anyhow::Result;

use crate::channel::Channel;
use crate::llm::LlmBackend;
use crate::types::ActionType;
use super::graph::AgentGraph;
use super::{SocialAgent, UserInfo};

/// Generate an agent graph from a Twitter-format CSV file.
///
/// CSV columns: username, name, description, user_char, following_agentid_list, previous_tweets
///
/// Mirrors OASIS's generate_twitter_agent_graph.
pub fn generate_twitter_agent_graph(
    agent_info_path: &str,
    model: Arc<dyn LlmBackend>,
    available_actions: Vec<ActionType>,
) -> Result<AgentGraph> {
    let mut graph = AgentGraph::new();

    // First pass: parse agents and collect follow/tweet data
    let mut reader = csv::Reader::from_path(agent_info_path)?;
    let mut follow_edges: Vec<(i64, i64)> = Vec::new();

    for (idx, record) in reader.records().enumerate() {
        let record = record?;
        let agent_id = idx as i64;

        let username = record.get(0).unwrap_or("").to_string();
        let name = record.get(1).unwrap_or("").to_string();
        let description = record.get(2).unwrap_or("").to_string();
        // Column 3 is user_char (unused in OASIS, skipped)
        let following_list = record.get(4).unwrap_or("").to_string();
        let previous_tweets = record.get(5).unwrap_or("").to_string();

        // Parse following_agentid_list (comma-separated agent IDs in brackets)
        let cleaned = following_list.trim().trim_matches(|c| c == '[' || c == ']');
        for id_str in cleaned.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            if let Ok(followee_id) = id_str.parse::<i64>() {
                follow_edges.push((agent_id, followee_id));
            }
        }

        // Store previous_tweets in profile for reset() to pick up
        let mut profile = std::collections::HashMap::new();
        if !previous_tweets.is_empty() {
            profile.insert(super::PROFILE_KEY_PREVIOUS_TWEETS.to_string(), serde_json::json!(previous_tweets));
        }

        let user_info = UserInfo {
            user_name: Some(username),
            name: Some(name),
            description: Some(description),
            profile: if profile.is_empty() { None } else { Some(profile) },
            recsys_type: crate::types::RecsysType::Twitter,
            is_controllable: false,
        };

        let channel = Arc::new(Channel::new());
        let agent = SocialAgent::new(
            agent_id,
            user_info,
            channel,
            Some(model.clone()),
            None,
            available_actions.clone(),
        );

        graph.add_agent(agent_id, agent);
    }

    // Second pass: add follow edges
    for (from, to) in follow_edges {
        graph.add_edge(from, to);
    }

    tracing::info!("Generated Twitter agent graph with {} agents", graph.get_num_nodes());
    Ok(graph)
}

/// Generate an agent graph from a Reddit-format JSON file.
///
/// JSON fields: username, realname, bio, persona, mbti, gender, age, country
///
/// Mirrors OASIS's generate_reddit_agent_graph.
pub fn generate_reddit_agent_graph(
    agent_info_path: &str,
    model: Arc<dyn LlmBackend>,
    available_actions: Vec<ActionType>,
) -> Result<AgentGraph> {
    let mut graph = AgentGraph::new();

    let data = std::fs::read_to_string(agent_info_path)?;
    let agents: Vec<serde_json::Value> = serde_json::from_str(&data)?;

    for (idx, agent_data) in agents.iter().enumerate() {
        let agent_id = idx as i64;

        let username = agent_data.get("username").and_then(|v| v.as_str()).unwrap_or("");
        let realname = agent_data.get("realname").and_then(|v| v.as_str()).unwrap_or("");
        let _bio = agent_data.get("bio").and_then(|v| v.as_str()).unwrap_or("");
        let persona = agent_data.get("persona").and_then(|v| v.as_str()).unwrap_or("");
        let mbti = agent_data.get("mbti").and_then(|v| v.as_str()).unwrap_or("");
        let gender = agent_data.get("gender").and_then(|v| v.as_str()).unwrap_or("");
        let age = agent_data.get("age").and_then(|v| v.as_u64()).unwrap_or(0);
        let country = agent_data.get("country").and_then(|v| v.as_str()).unwrap_or("");

        let description = format!(
            "{}. MBTI: {}. Gender: {}. Age: {}. Country: {}.",
            persona, mbti, gender, age, country
        );

        let mut profile = std::collections::HashMap::new();
        profile.insert("persona".to_string(), serde_json::json!(persona));
        profile.insert("mbti".to_string(), serde_json::json!(mbti));
        profile.insert("gender".to_string(), serde_json::json!(gender));
        profile.insert("age".to_string(), serde_json::json!(age));
        profile.insert("country".to_string(), serde_json::json!(country));

        let user_info = UserInfo {
            user_name: Some(username.to_string()),
            name: Some(realname.to_string()),
            description: Some(description),
            profile: Some(profile),
            recsys_type: crate::types::RecsysType::Reddit,
            is_controllable: false,
        };

        let channel = Arc::new(Channel::new());
        let agent = SocialAgent::new(
            agent_id,
            user_info,
            channel,
            Some(model.clone()),
            None,
            available_actions.clone(),
        );

        graph.add_agent(agent_id, agent);
    }

    tracing::info!("Generated Reddit agent graph with {} agents", graph.get_num_nodes());
    Ok(graph)
}
