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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// A borrowed node from a [`NodeIndex`], at whichever level it lives.
#[derive(Debug, Clone, Copy)]
pub enum NodeRef<'a> {
    /// A run-level node, carrying timing and sparse/dense variables.
    Run(&'a RunNode),
    /// A dataset, subject or session level node.
    Other(&'a Node),
}

impl NodeRef<'_> {
    /// The node's hierarchy level.
    #[must_use]
    pub fn level(&self) -> &str {
        match self {
            Self::Run(n) => &n.level,
            Self::Other(n) => &n.level,
        }
    }

    /// The node's BIDS entities.
    #[must_use]
    pub fn entities(&self) -> &StringEntities {
        match self {
            Self::Run(n) => &n.entities,
            Self::Other(n) => &n.entities,
        }
    }

    /// The run node, when this is a run-level node.
    #[must_use]
    pub fn as_run(&self) -> Option<&RunNode> {
        match self {
            Self::Run(n) => Some(n),
            Self::Other(_) => None,
        }
    }
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

    /// Collect variables at any level into [`VariableCollection`]s.
    ///
    /// Run-level nodes keep their event timing (via
    /// [`VariableCollection::from_run_collection`]); other levels hold simple
    /// variables. With `merge`, all matching nodes are combined into a single
    /// collection instead of one per node.
    #[must_use]
    pub fn get_collections(
        &self,
        level: &str,
        entities: &StringEntities,
        merge: bool,
    ) -> Vec<crate::collections::VariableCollection> {
        use crate::collections::{VariableCollection, merge_collections};

        let collections: Vec<VariableCollection> = if level == "run" {
            self.get_run_collections(entities)
                .iter()
                .map(VariableCollection::from_run_collection)
                .filter(|c| !c.variables.is_empty())
                .collect()
        } else {
            self.find_nodes(level, entities)
                .iter()
                .filter_map(|&idx| match &self.nodes[idx] {
                    NodeEntry::Other(n) if !n.variables.is_empty() => {
                        Some(VariableCollection::new(n.variables.clone()))
                    }
                    _ => None,
                })
                .collect()
        };

        if merge {
            merge_collections(&collections).into_iter().collect()
        } else {
            collections
        }
    }

    /// Nodes at `level` whose entities match `entities`.
    ///
    /// Corresponds to PyBIDS' `NodeIndex.get_nodes`. Use
    /// [`find_nodes`](Self::find_nodes) when you need the indices instead, for
    /// example to then call [`get_run_node_mut`](Self::get_run_node_mut).
    #[must_use]
    pub fn get_nodes(&self, level: &str, entities: &StringEntities) -> Vec<NodeRef<'_>> {
        self.find_nodes(level, entities)
            .into_iter()
            .map(|i| match &self.nodes[i] {
                NodeEntry::Run(n) => NodeRef::Run(n),
                NodeEntry::Other(n) => NodeRef::Other(n),
            })
            .collect()
    }

    /// The distinct levels present in this index.
    #[must_use]
    pub fn levels(&self) -> Vec<String> {
        let mut levels: Vec<String> = self
            .nodes
            .iter()
            .map(|e| match e {
                NodeEntry::Run(r) => r.level.clone(),
                NodeEntry::Other(n) => n.level.clone(),
            })
            .collect();
        levels.sort();
        levels.dedup();
        levels
    }

    /// Collect run-level variables into collections.
    #[must_use]
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
