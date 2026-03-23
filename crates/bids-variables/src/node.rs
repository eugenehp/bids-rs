//! BIDS hierarchy nodes: run-level and non-run (subject/session/dataset).
//!
//! Nodes represent levels of the BIDS hierarchy that hold variables. Run-level
//! nodes carry sparse and dense time-series variables; non-run nodes carry
//! simple demographic/session variables.

use bids_core::entities::StringEntities;

use crate::collections::RunVariableCollection;
use crate::variables::{DenseRunVariable, SimpleVariable, SparseRunVariable};

/// Metadata about a single run in a BIDS dataset.
///
/// Contains the run's BIDS entities, temporal parameters (duration, TR),
/// the path to the associated image file, and the number of volumes.
/// Used by variable types to track which runs their data belongs to.
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub entities: StringEntities,
    pub duration: f64,
    pub tr: f64,
    pub image: Option<String>,
    pub n_vols: usize,
}

/// A non-run node in the BIDS hierarchy (dataset, subject, or session level).
///
/// Holds simple variables (e.g., participant demographics, session metadata)
/// at a specific level of the BIDS hierarchy.
pub struct Node {
    pub level: String,
    pub entities: StringEntities,
    pub variables: Vec<SimpleVariable>,
}

impl Node {
    pub fn new(level: &str, entities: StringEntities) -> Self {
        Self {
            level: level.to_lowercase(),
            entities,
            variables: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, var: SimpleVariable) {
        self.variables.push(var);
    }
}

/// A run-level node with timing information and both sparse and dense variables.
///
/// Represents a single functional run with its temporal parameters (duration,
/// TR, number of volumes) and holds both event-based sparse variables and
/// continuous dense variables for that run.
pub struct RunNode {
    pub level: String,
    pub entities: StringEntities,
    pub image_file: Option<String>,
    pub duration: f64,
    pub repetition_time: f64,
    pub n_vols: usize,
    pub sparse_variables: Vec<SparseRunVariable>,
    pub dense_variables: Vec<DenseRunVariable>,
}

impl RunNode {
    pub fn new(
        entities: StringEntities,
        image_file: Option<String>,
        duration: f64,
        repetition_time: f64,
        n_vols: usize,
    ) -> Self {
        Self {
            level: "run".into(),
            entities,
            image_file,
            duration,
            repetition_time,
            n_vols,
            sparse_variables: Vec::new(),
            dense_variables: Vec::new(),
        }
    }

    pub fn get_info(&self) -> RunInfo {
        RunInfo {
            entities: self.entities.clone(),
            duration: self.duration,
            tr: self.repetition_time,
            image: self.image_file.clone(),
            n_vols: self.n_vols,
        }
    }

    pub fn add_sparse_variable(&mut self, var: SparseRunVariable) {
        self.sparse_variables.push(var);
    }

    pub fn add_dense_variable(&mut self, var: DenseRunVariable) {
        self.dense_variables.push(var);
    }
}

/// Top-level index organizing all variable nodes in a BIDS dataset.
///
/// The `NodeIndex` maintains a flat list of nodes (both run-level and
/// higher-level) and provides methods to find, create, and query nodes
/// by level and entity values. Nodes are created during variable loading
/// and can be queried to extract variable collections for statistical
/// modeling.
#[derive(Default)]
pub struct NodeIndex {
    nodes: Vec<NodeEntry>,
}

enum NodeEntry {
    Run(RunNode),
    Other(Node),
}

impl NodeIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_run_node(
        &mut self,
        entities: StringEntities,
        image_file: Option<String>,
        duration: f64,
        tr: f64,
        n_vols: usize,
    ) -> usize {
        self.nodes.push(NodeEntry::Run(RunNode::new(
            entities, image_file, duration, tr, n_vols,
        )));
        self.nodes.len() - 1
    }

    pub fn create_node(&mut self, level: &str, entities: StringEntities) -> usize {
        self.nodes
            .push(NodeEntry::Other(Node::new(level, entities)));
        self.nodes.len() - 1
    }

    pub fn get_run_node_mut(&mut self, index: usize) -> Option<&mut RunNode> {
        match self.nodes.get_mut(index) {
            Some(NodeEntry::Run(n)) => Some(n),
            _ => None,
        }
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut Node> {
        match self.nodes.get_mut(index) {
            Some(NodeEntry::Other(n)) => Some(n),
            _ => None,
        }
    }

    /// Find nodes matching level and entities, sorted by subject/session/task/run.
    pub fn find_nodes(&self, level: &str, entities: &StringEntities) -> Vec<usize> {
        let sort_keys = ["subject", "session", "task", "run"];
        let mut results: Vec<(usize, Vec<String>)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                let (node_level, node_ents) = match entry {
                    NodeEntry::Run(r) => (r.level.as_str(), &r.entities),
                    NodeEntry::Other(n) => (n.level.as_str(), &n.entities),
                };
                node_level == level
                    && entities
                        .iter()
                        .all(|(k, v)| node_ents.get(k).is_none_or(|nv| nv == v))
            })
            .map(|(i, entry)| {
                let ents = match entry {
                    NodeEntry::Run(r) => &r.entities,
                    NodeEntry::Other(n) => &n.entities,
                };
                let key: Vec<String> = sort_keys
                    .iter()
                    .map(|k| ents.get(*k).cloned().unwrap_or_default())
                    .collect();
                (i, key)
            })
            .collect();
        results.sort_by(|(_, a), (_, b)| a.cmp(b));
        results.into_iter().map(|(i, _)| i).collect()
    }

    /// Find or create a node.
    pub fn get_or_create_node(&mut self, level: &str, entities: StringEntities) -> usize {
        let existing = self.find_nodes(level, &entities);
        if let Some(&idx) = existing.first() {
            idx
        } else {
            self.create_node(level, entities)
        }
    }

    /// Find or create a run node.
    pub fn get_or_create_run_node(
        &mut self,
        entities: StringEntities,
        image_file: Option<String>,
        duration: f64,
        tr: f64,
        n_vols: usize,
    ) -> usize {
        let existing = self.find_nodes("run", &entities);
        if let Some(&idx) = existing.first() {
            idx
        } else {
            self.create_run_node(entities, image_file, duration, tr, n_vols)
        }
    }

    /// Collect run-level variables into collections.
    pub fn get_run_collections(&self, entities: &StringEntities) -> Vec<RunVariableCollection> {
        let indices = self.find_nodes("run", entities);
        indices
            .iter()
            .filter_map(|&idx| {
                if let NodeEntry::Run(rn) = &self.nodes[idx] {
                    if rn.sparse_variables.is_empty() && rn.dense_variables.is_empty() {
                        return None;
                    }
                    Some(RunVariableCollection::new(
                        rn.sparse_variables.clone(),
                        rn.dense_variables.clone(),
                        None,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}
