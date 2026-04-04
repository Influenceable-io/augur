use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};

use super::SocialAgent;

/// Trait abstracting directed graph topology.
///
/// Implementations must be Send + Sync for use behind Arc<RwLock<AgentGraph>>.
/// Only covers node/edge operations — no traversals (those are post-hoc analytics).
pub trait GraphBackend: Send + Sync {
    fn add_node(&mut self, id: i64);
    fn remove_node(&mut self, id: i64);
    fn has_node(&self, id: i64) -> bool;
    fn add_edge(&mut self, from: i64, to: i64);
    fn remove_edge(&mut self, from: i64, to: i64);
    fn has_edge(&self, from: i64, to: i64) -> bool;
    fn edges(&self) -> Vec<(i64, i64)>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn clear(&mut self);
}

/// In-memory graph backend using petgraph. Default backend for AgentGraph.
pub struct InMemoryGraph {
    graph: DiGraph<i64, ()>,
    node_map: HashMap<i64, NodeIndex>,
}

impl InMemoryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }
}

impl Default for InMemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBackend for InMemoryGraph {
    fn add_node(&mut self, id: i64) {
        if !self.node_map.contains_key(&id) {
            let node = self.graph.add_node(id);
            self.node_map.insert(id, node);
        }
    }

    fn remove_node(&mut self, id: i64) {
        if let Some(node) = self.node_map.remove(&id) {
            self.graph.remove_node(node);
        }
    }

    fn has_node(&self, id: i64) -> bool {
        self.node_map.contains_key(&id)
    }

    fn add_edge(&mut self, from: i64, to: i64) {
        if let (Some(&from_node), Some(&to_node)) = (self.node_map.get(&from), self.node_map.get(&to)) {
            if !self.graph.contains_edge(from_node, to_node) {
                self.graph.add_edge(from_node, to_node, ());
            }
        }
    }

    fn remove_edge(&mut self, from: i64, to: i64) {
        if let (Some(&from_node), Some(&to_node)) = (self.node_map.get(&from), self.node_map.get(&to))
            && let Some(edge) = self.graph.find_edge(from_node, to_node) {
                self.graph.remove_edge(edge);
            }
    }

    fn has_edge(&self, from: i64, to: i64) -> bool {
        if let (Some(&from_node), Some(&to_node)) = (self.node_map.get(&from), self.node_map.get(&to)) {
            self.graph.contains_edge(from_node, to_node)
        } else {
            false
        }
    }

    fn edges(&self) -> Vec<(i64, i64)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                Some((self.graph[a], self.graph[b]))
            })
            .collect()
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn clear(&mut self) {
        self.graph.clear();
        self.node_map.clear();
    }
}

/// Directed social graph with agent registry.
///
/// Composes a `GraphBackend` (topology) with an agent registry (HashMap).
/// The topology backend can be swapped for different scale requirements.
pub struct AgentGraph {
    topology: Box<dyn GraphBackend>,
    agent_mappings: HashMap<i64, SocialAgent>,
}

impl AgentGraph {
    /// Create with the default in-memory backend.
    pub fn new() -> Self {
        Self::with_backend(Box::new(InMemoryGraph::new()))
    }

    /// Create with a specific graph backend.
    pub fn with_backend(topology: Box<dyn GraphBackend>) -> Self {
        Self {
            topology,
            agent_mappings: HashMap::new(),
        }
    }

    pub fn add_agent(&mut self, agent_id: i64, agent: SocialAgent) {
        self.topology.add_node(agent_id);
        self.agent_mappings.insert(agent_id, agent);
    }

    pub fn remove_agent(&mut self, agent_id: i64) {
        self.topology.remove_node(agent_id);
        self.agent_mappings.remove(&agent_id);
    }

    pub fn get_agent(&self, agent_id: i64) -> Option<&SocialAgent> {
        self.agent_mappings.get(&agent_id)
    }

    pub fn get_agent_mut(&mut self, agent_id: i64) -> Option<&mut SocialAgent> {
        self.agent_mappings.get_mut(&agent_id)
    }

    /// Temporarily remove an agent from the registry (leaves graph node intact).
    /// Used to release the graph lock during async operations on the agent.
    pub fn take_agent(&mut self, agent_id: i64) -> Option<SocialAgent> {
        self.agent_mappings.remove(&agent_id)
    }

    pub fn get_agents(&self) -> Vec<(i64, &SocialAgent)> {
        self.agent_mappings
            .iter()
            .map(|(&id, agent)| (id, agent))
            .collect()
    }

    pub fn add_edge(&mut self, from: i64, to: i64) {
        self.topology.add_edge(from, to);
    }

    pub fn remove_edge(&mut self, from: i64, to: i64) {
        self.topology.remove_edge(from, to);
    }

    pub fn get_edges(&self) -> Vec<(i64, i64)> {
        self.topology.edges()
    }

    pub fn get_num_nodes(&self) -> usize {
        self.topology.node_count()
    }

    pub fn get_num_edges(&self) -> usize {
        self.topology.edge_count()
    }

    pub fn reset(&mut self) {
        self.topology.clear();
        self.agent_mappings.clear();
    }
}

impl Default for AgentGraph {
    fn default() -> Self {
        Self::new()
    }
}
