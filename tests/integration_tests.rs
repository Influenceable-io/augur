mod common;

use std::collections::HashMap;
use std::sync::Arc;

use augur::agent::graph::AgentGraph;
use augur::channel::Channel;
use augur::agent::{SocialAgent, UserInfo};
use augur::env_action::{Action, ManualAction};
use augur::types::{ActionType, DefaultPlatformType};
use augur::PlatformOrDefault;

use common::MockLlm;

/// Build a 5-agent graph with edges 0->1 and 1->2 for testing.
fn build_agent_graph() -> AgentGraph {
    let mut graph = AgentGraph::new();
    let model = Arc::new(MockLlm::do_nothing());
    let channel = Arc::new(Channel::new());

    for i in 0..5 {
        let info = UserInfo::new(
            &format!("user{}", i),
            &format!("User {}", i),
            &format!("I am test user {}", i),
        );
        let agent = SocialAgent::new(
            i,
            info,
            channel.clone(),
            Some(model.clone()),
            None,
            vec![
                ActionType::CreatePost,
                ActionType::LikePost,
                ActionType::Follow,
                ActionType::DoNothing,
            ],
        );
        graph.add_agent(i, agent);
    }

    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph
}

#[tokio::test]
async fn test_full_simulation_lifecycle() {
    let graph = build_agent_graph();

    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let db_path = tmp.path().to_str().unwrap();

    // Create environment
    let mut env = augur::make(
        graph,
        PlatformOrDefault::Default(DefaultPlatformType::Twitter),
        db_path,
        4,
    )
    .await
    .expect("make() failed");

    // Reset: starts platform, signs up agents
    env.reset().await.expect("reset() failed");

    // Step 1: agent 0 creates a post, agent 1 does nothing
    let mut actions = HashMap::new();
    actions.insert(
        0,
        Action::Manual(ManualAction::new(
            ActionType::CreatePost,
            {
                let mut args = HashMap::new();
                args.insert("content".to_string(), serde_json::json!("Hello from agent 0!"));
                args
            },
        )),
    );
    actions.insert(
        1,
        Action::Manual(ManualAction::new(ActionType::DoNothing, HashMap::new())),
    );
    env.step(actions).await.expect("step 1 failed");

    // Step 2: agent 2 creates a post
    let mut actions2 = HashMap::new();
    actions2.insert(
        2,
        Action::Manual(ManualAction::new(
            ActionType::CreatePost,
            {
                let mut args = HashMap::new();
                args.insert(
                    "content".to_string(),
                    serde_json::json!("Post from agent 2"),
                );
                args
            },
        )),
    );
    env.step(actions2).await.expect("step 2 failed");

    // Close
    env.close().await.expect("close() failed");
}

#[tokio::test]
async fn test_multiple_actions_per_agent() {
    let graph = build_agent_graph();

    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let db_path = tmp.path().to_str().unwrap();

    let mut env = augur::make(
        graph,
        PlatformOrDefault::Default(DefaultPlatformType::Twitter),
        db_path,
        4,
    )
    .await
    .expect("make() failed");

    env.reset().await.expect("reset() failed");

    // Agent 0 performs two CreatePost actions in one step via Multiple
    let mut actions = HashMap::new();
    actions.insert(
        0,
        Action::Multiple(vec![
            Action::Manual(ManualAction::new(
                ActionType::CreatePost,
                {
                    let mut args = HashMap::new();
                    args.insert("content".to_string(), serde_json::json!("First post"));
                    args
                },
            )),
            Action::Manual(ManualAction::new(
                ActionType::CreatePost,
                {
                    let mut args = HashMap::new();
                    args.insert("content".to_string(), serde_json::json!("Second post"));
                    args
                },
            )),
        ]),
    );
    env.step(actions).await.expect("multiple actions step failed");

    env.close().await.expect("close() failed");
}

#[tokio::test]
async fn test_empty_step() {
    let graph = build_agent_graph();

    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let db_path = tmp.path().to_str().unwrap();

    let mut env = augur::make(
        graph,
        PlatformOrDefault::Default(DefaultPlatformType::Twitter),
        db_path,
        4,
    )
    .await
    .expect("make() failed");

    env.reset().await.expect("reset() failed");

    // Empty step: no actions, should still succeed
    let actions: HashMap<i64, Action> = HashMap::new();
    env.step(actions).await.expect("empty step failed");

    env.close().await.expect("close() failed");
}
