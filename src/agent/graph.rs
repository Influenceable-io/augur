use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};

use super::SocialAgent;

/// Directed social graph with agent registry.
///
/// Mirrors OASIS's AgentGraph using petgraph instead of igraph.
pub struct AgentGraph {
    graph: DiGraph<i64, ()>,
    node_map: HashMap<i64, NodeIndex>,
    agent_mappings: HashMap<i64, SocialAgent>,
}

impl AgentGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            agent_mappings: HashMap::new(),
        }
    }

    pub fn add_agent(&mut self, agent_id: i64, agent: SocialAgent) {
        let node = self.graph.add_node(agent_id);
        self.node_map.insert(agent_id, node);
        self.agent_mappings.insert(agent_id, agent);
    }

    pub fn remove_agent(&mut self, agent_id: i64) {
        if let Some(node) = self.node_map.remove(&agent_id) {
            self.graph.remove_node(node);
            self.agent_mappings.remove(&agent_id);
        }
    }

    pub fn get_agent(&self, agent_id: i64) -> Option<&SocialAgent> {
        self.agent_mappings.get(&agent_id)
    }

    pub fn get_agent_mut(&mut self, agent_id: i64) -> Option<&mut SocialAgent> {
        self.agent_mappings.get_mut(&agent_id)
    }

    pub fn get_agents(&self) -> Vec<(i64, &SocialAgent)> {
        self.agent_mappings
            .iter()
            .map(|(&id, agent)| (id, agent))
            .collect()
    }

    pub fn add_edge(&mut self, from: i64, to: i64) {
        if let (Some(&from_node), Some(&to_node)) = (self.node_map.get(&from), self.node_map.get(&to)) {
            // Prevent duplicate edges
            if !self.graph.contains_edge(from_node, to_node) {
                self.graph.add_edge(from_node, to_node, ());
            }
        }
    }

    pub fn remove_edge(&mut self, from: i64, to: i64) {
        if let (Some(&from_node), Some(&to_node)) = (self.node_map.get(&from), self.node_map.get(&to))
            && let Some(edge) = self.graph.find_edge(from_node, to_node) {
                self.graph.remove_edge(edge);
            }
    }

    pub fn get_edges(&self) -> Vec<(i64, i64)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                Some((self.graph[a], self.graph[b]))
            })
            .collect()
    }

    pub fn get_num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    pub fn get_num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn reset(&mut self) {
        self.graph.clear();
        self.node_map.clear();
        self.agent_mappings.clear();
    }
}

impl Default for AgentGraph {
    fn default() -> Self {
        Self::new()
    }
}
