//! BIDS-StatsModels directed acyclic graph.
//!
//! [`StatsModelsGraph`] is the top-level type for a BIDS-StatsModels
//! specification. It contains analysis nodes (run/session/subject/dataset
//! levels) connected by edges defining how data flows through the pipeline.

use crate::node::{ContrastInfo, StatsModelsEdge, StatsModelsNode, StatsModelsNodeOutput};
use crate::transformations::TransformSpec;
use bids_core::error::{BidsError, Result};
use bids_core::utils::convert_json_keys;
use std::collections::HashMap;

/// Rooted directed acyclic graph representing a BIDS-StatsModel specification.
///
/// A `StatsModelsGraph` is loaded from a JSON file conforming to the
/// [BIDS-StatsModels](https://bids-standard.github.io/stats-models/)
/// specification. It contains analysis nodes at different levels of the
/// BIDS hierarchy (run, session, subject, dataset) connected by edges
/// that define how contrasts and data flow between levels.
///
/// # Lifecycle
///
/// 1. **Load** — Parse from JSON file or value via [`from_file()`](Self::from_file)
///    or [`from_json()`](Self::from_json)
/// 2. **Validate** — Check structure via [`validate()`](Self::validate)
/// 3. **Load data** — Populate nodes with variables from a layout via
///    [`load_collections()`](Self::load_collections)
/// 4. **Execute** — Run the analysis pipeline via [`run()`](Self::run)
/// 5. **Export** — Generate DOT graph via [`write_graph()`](Self::write_graph)
///
/// Corresponds to PyBIDS' `StatsModelsGraph` class.
#[derive(Debug)]
pub struct StatsModelsGraph {
    pub name: String,
    pub description: String,
    pub nodes: Vec<StatsModelsNode>,
    node_map: HashMap<String, usize>,
    pub edges: Vec<StatsModelsEdge>,
    root_idx: usize,
}

impl StatsModelsGraph {
    /// Load from a JSON model spec (path or parsed value).
    pub fn from_json(model_json: &serde_json::Value) -> Result<Self> {
        let model = convert_json_keys(model_json);
        Self::from_parsed(&model)
    }

    /// Load from a file path.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&contents)?;
        Self::from_json(&json)
    }

    fn from_parsed(model: &serde_json::Value) -> Result<Self> {
        let name = model
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed")
            .into();
        let description = model
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .into();

        // Load nodes
        let node_specs = model
            .get("nodes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| BidsError::Validation("Model must have 'nodes' array".into()))?;

        let mut nodes = Vec::new();
        let mut node_map = HashMap::new();

        for spec in node_specs {
            let level = spec.get("level").and_then(|v| v.as_str()).unwrap_or("run");
            let node_name = spec
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed");
            let model_spec = spec
                .get("model")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let group_by: Vec<String> = spec
                .get("group_by")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let transformations = spec
                .get("transformations")
                .and_then(|v| serde_json::from_value::<TransformSpec>(v.clone()).ok());

            let contrasts: Vec<serde_json::Value> = spec
                .get("contrasts")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let dummy_contrasts = spec
                .get("dummy_contrasts")
                .cloned()
                .filter(|v| !v.is_null() && *v != serde_json::Value::Bool(false));

            node_map.insert(node_name.to_string(), nodes.len());
            nodes.push(StatsModelsNode::new(
                level,
                node_name,
                model_spec,
                group_by,
                transformations,
                contrasts,
                dummy_contrasts,
            ));
        }

        // Load edges
        let mut edges = Vec::new();
        if let Some(edge_specs) = model.get("edges").and_then(|v| v.as_array()) {
            for edge_spec in edge_specs {
                let src = edge_spec
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let dst = edge_spec
                    .get("destination")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let filter: HashMap<String, String> = edge_spec
                    .get("filter")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                edges.push(StatsModelsEdge {
                    source: src.into(),
                    destination: dst.into(),
                    filter,
                });
            }
        }

        // If no edges, create implicit pipeline order
        if edges.is_empty() && nodes.len() > 1 {
            for i in 0..nodes.len() - 1 {
                edges.push(StatsModelsEdge {
                    source: nodes[i].name.clone(),
                    destination: nodes[i + 1].name.clone(),
                    filter: HashMap::new(),
                });
            }
        }

        // Wire edges to nodes
        for edge in &edges {
            if let Some(&src_idx) = node_map.get(&edge.source) {
                nodes[src_idx].add_child(edge.clone());
            }
            if let Some(&dst_idx) = node_map.get(&edge.destination) {
                nodes[dst_idx].add_parent(edge.clone());
            }
        }

        let root_idx = model
            .get("root")
            .and_then(|v| v.as_str())
            .and_then(|name| node_map.get(name).copied())
            .unwrap_or(0);

        Ok(Self {
            name,
            description,
            nodes,
            node_map,
            edges,
            root_idx,
        })
    }

    /// Validate the model structure.
    pub fn validate(&self) -> Result<()> {
        // Check unique names
        let mut names = std::collections::HashSet::new();
        for node in &self.nodes {
            if !names.insert(&node.name) {
                return Err(BidsError::Validation(format!(
                    "Duplicate node name: '{}'",
                    node.name
                )));
            }
        }
        // Check edge references
        for edge in &self.edges {
            if !self.node_map.contains_key(&edge.source) {
                return Err(BidsError::Validation(format!(
                    "Edge references unknown source: '{}'",
                    edge.source
                )));
            }
            if !self.node_map.contains_key(&edge.destination) {
                return Err(BidsError::Validation(format!(
                    "Edge references unknown destination: '{}'",
                    edge.destination
                )));
            }
        }
        Ok(())
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<&StatsModelsNode> {
        self.node_map.get(name).map(|&i| &self.nodes[i])
    }

    /// Get the root node.
    pub fn root_node(&self) -> &StatsModelsNode {
        &self.nodes[self.root_idx]
    }

    /// Load collections from a layout into all nodes.
    pub fn load_collections(&mut self, layout: &bids_layout::BidsLayout) {
        // Use bids_variables to load, then convert to VariableCollections
        for node in &mut self.nodes {
            if let Ok(index) = bids_variables::load_variables(layout, None, Some(&node.level)) {
                let entities = bids_core::entities::StringEntities::new();
                let run_colls = index.get_run_collections(&entities);
                for rc in run_colls {
                    let vars: Vec<bids_variables::SimpleVariable> = rc
                        .sparse
                        .iter()
                        .map(|v| {
                            bids_variables::SimpleVariable::new(
                                &v.name,
                                &v.source,
                                v.str_amplitude.clone(),
                                v.index.clone(),
                            )
                        })
                        .collect();
                    if !vars.is_empty() {
                        node.add_collections(vec![bids_variables::VariableCollection::new(vars)]);
                    }
                }
            }
        }
    }

    /// Write graph structure as a DOT file (text-based graphviz).
    pub fn write_graph(&self) -> String {
        let mut dot = format!("digraph \"{}\" {{\n  node [shape=record];\n", self.name);
        for node in &self.nodes {
            dot.push_str(&format!(
                "  \"{}\" [label=\"{{name: {}|level: {}}}\"];\n",
                node.name, node.name, node.level
            ));
        }
        for edge in &self.edges {
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\";\n",
                edge.source, edge.destination
            ));
        }
        dot.push_str("}\n");
        dot
    }

    /// Render graph to a file using the `dot` command (requires graphviz installed).
    pub fn render_graph(&self, output_path: &std::path::Path, format: &str) -> std::io::Result<()> {
        let dot = self.write_graph();
        let tmp = std::env::temp_dir().join("bids_model.dot");
        std::fs::write(&tmp, &dot)?;
        let status = std::process::Command::new("dot")
            .arg(format!("-T{format}"))
            .arg("-o")
            .arg(output_path)
            .arg(&tmp)
            .status()?;
        if !status.success() {
            return Err(std::io::Error::other("dot command failed"));
        }
        Ok(())
    }

    /// Run the entire graph recursively.
    pub fn run(&self) -> Vec<StatsModelsNodeOutput> {
        let mut all_outputs = Vec::new();
        self.run_node_recursive(self.root_idx, &[], &mut all_outputs);
        all_outputs
    }

    fn run_node_recursive(
        &self,
        node_idx: usize,
        inputs: &[ContrastInfo],
        all_outputs: &mut Vec<StatsModelsNodeOutput>,
    ) {
        let node = &self.nodes[node_idx];
        let outputs = node.run(inputs, true, "TR");
        let contrasts: Vec<ContrastInfo> =
            outputs.iter().flat_map(|o| o.contrasts.clone()).collect();
        all_outputs.extend(outputs);

        for edge in &node.children {
            if let Some(&dst_idx) = self.node_map.get(&edge.destination) {
                self.run_node_recursive(dst_idx, &contrasts, all_outputs);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model() {
        let model = serde_json::json!({
            "Name": "test_model",
            "Description": "A test",
            "BIDSModelVersion": "1.0.0",
            "Nodes": [
                {
                    "Level": "Run",
                    "Name": "run",
                    "GroupBy": ["run", "subject"],
                    "Model": {"Type": "glm", "X": ["trial_type.face"]},
                    "DummyContrasts": {"Test": "t"}
                },
                {
                    "Level": "Subject",
                    "Name": "subject",
                    "GroupBy": ["subject", "contrast"],
                    "Model": {"Type": "glm", "X": [1]},
                    "DummyContrasts": {"Test": "t"}
                }
            ]
        });

        let graph = StatsModelsGraph::from_json(&model).unwrap();
        assert_eq!(graph.name, "test_model");
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        graph.validate().unwrap();
    }
}
