use std::sync::Arc;

use augur::agent::graph::AgentGraph;
use augur::agent::{SocialAgent, UserInfo};
use augur::channel::Channel;
use augur::types::ActionType;

fn make_agent(id: i64) -> SocialAgent {
    let info = UserInfo::new(&format!("user{}", id), &format!("User {}", id), "bio");
    let channel = Arc::new(Channel::new());
    SocialAgent::new(id, info, channel, None, None, vec![ActionType::DoNothing])
}

#[test]
fn test_add_remove_agent() {
    let mut graph = AgentGraph::new();

    graph.add_agent(0, make_agent(0));
    graph.add_agent(1, make_agent(1));
    assert_eq!(graph.get_num_nodes(), 2);

    // Both agents should be retrievable
    assert!(graph.get_agent(0).is_some());
    assert!(graph.get_agent(1).is_some());

    // Remove one agent
    graph.remove_agent(0);
    assert_eq!(graph.get_num_nodes(), 1);
    assert!(graph.get_agent(0).is_none());
    assert!(graph.get_agent(1).is_some());
}

#[test]
fn test_add_remove_edges() {
    let mut graph = AgentGraph::new();

    graph.add_agent(0, make_agent(0));
    graph.add_agent(1, make_agent(1));
    graph.add_agent(2, make_agent(2));

    // Add edges 0->1 and 0->2
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    assert_eq!(graph.get_num_edges(), 2);

    // Duplicate edge should not increase count
    graph.add_edge(0, 1);
    assert_eq!(graph.get_num_edges(), 2);

    // Remove one edge
    graph.remove_edge(0, 1);
    assert_eq!(graph.get_num_edges(), 1);

    // Remaining edge should be 0->2
    let edges = graph.get_edges();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0], (0, 2));
}

#[test]
fn test_get_agents() {
    let mut graph = AgentGraph::new();

    graph.add_agent(0, make_agent(0));
    graph.add_agent(1, make_agent(1));

    let agents = graph.get_agents();
    assert_eq!(agents.len(), 2);

    let mut ids: Vec<i64> = agents.iter().map(|(id, _)| *id).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1]);
}

#[test]
fn test_reset() {
    let mut graph = AgentGraph::new();

    graph.add_agent(0, make_agent(0));
    graph.add_agent(1, make_agent(1));
    graph.add_edge(0, 1);

    assert_eq!(graph.get_num_nodes(), 2);
    assert_eq!(graph.get_num_edges(), 1);

    graph.reset();

    assert_eq!(graph.get_num_nodes(), 0);
    assert_eq!(graph.get_num_edges(), 0);
    assert!(graph.get_agents().is_empty());
    assert!(graph.get_edges().is_empty());
    assert!(graph.get_agent(0).is_none());
    assert!(graph.get_agent(1).is_none());
}
