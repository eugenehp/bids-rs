//! Analysis nodes and edges in a BIDS-StatsModels graph.
//!
//! Each node represents a statistical analysis step at a specific level of
//! the BIDS hierarchy. Nodes contain a model specification, variable
//! transformations, contrast definitions, and grouping criteria. They
//! produce [`StatsModelsNodeOutput`]s containing design matrices and
//! contrast information.

use bids_core::entities::StringEntities;
use bids_variables::collections::VariableCollection;
use serde::{Deserialize, Serialize};

/// A directed edge between two nodes in the stats model graph.
///
/// Edges define data flow from a source node to a destination node,
/// optionally filtering which contrasts/outputs are passed through
/// based on entity values.
#[derive(Debug, Clone)]
pub struct StatsModelsEdge {
    pub source: String,
    pub destination: String,
    pub filter: StringEntities,
}

/// Information about a statistical contrast.
///
/// Defines a linear combination of model terms to be tested. Contains
/// the contrast name, the list of conditions involved, their weights,
/// the statistical test type (t, F), and the BIDS entities identifying
/// the data this contrast applies to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastInfo {
    pub name: String,
    pub conditions: Vec<String>,
    pub weights: Vec<f64>,
    pub test: Option<String>,
    pub entities: StringEntities,
}

/// A single analysis node in a BIDS-StatsModel graph.
///
/// Each node operates at a specific level (run, session, subject, dataset)
/// and defines:
/// - A statistical model specification (GLM or meta-analysis)
/// - Variable transformations to apply before modeling
/// - Explicit contrasts and/or dummy contrasts
/// - Grouping criteria for splitting data into independent analyses
///
/// Corresponds to a single entry in the `"Nodes"` array of a BIDS-StatsModels
/// JSON specification.
#[derive(Debug, Clone)]
pub struct StatsModelsNode {
    pub level: String,
    pub name: String,
    pub model: serde_json::Value,
    pub group_by: Vec<String>,
    pub transformations: Option<crate::transformations::TransformSpec>,
    pub contrasts: Vec<serde_json::Value>,
    pub dummy_contrasts: Option<serde_json::Value>,
    pub children: Vec<StatsModelsEdge>,
    pub parents: Vec<StatsModelsEdge>,
    collections: Vec<VariableCollection>,
}

impl StatsModelsNode {
    pub fn new(
        level: &str, name: &str, model: serde_json::Value,
        group_by: Vec<String>,
        transformations: Option<crate::transformations::TransformSpec>,
        contrasts: Vec<serde_json::Value>,
        dummy_contrasts: Option<serde_json::Value>,
    ) -> Self {
        Self {
            level: level.to_lowercase(), name: name.into(), model,
            group_by, transformations, contrasts, dummy_contrasts,
            children: Vec::new(), parents: Vec::new(),
            collections: Vec::new(),
        }
    }

    pub fn add_child(&mut self, edge: StatsModelsEdge) { self.children.push(edge); }
    pub fn add_parent(&mut self, edge: StatsModelsEdge) { self.parents.push(edge); }

    pub fn add_collections(&mut self, collections: Vec<VariableCollection>) {
        self.collections.extend(collections);
    }

    pub fn get_collections(&self) -> &[VariableCollection] { &self.collections }

    /// Run this node, producing outputs.
    pub fn run(
        &self,
        inputs: &[ContrastInfo],
        _force_dense: bool,
        _sampling_rate: &str,
    ) -> Vec<StatsModelsNodeOutput> {
        // Group collections and inputs by group_by entities
        let mut results = Vec::new();

        if self.collections.is_empty() && inputs.is_empty() {
            return results;
        }

        // For each collection, apply transformations and build output
        for collection in &self.collections {
            let mut coll = collection.clone();

            // Apply transformations
            if let Some(ref spec) = self.transformations {
                crate::transformations::apply_transformations(&mut coll, spec);
            }

            // Extract X variable names from model
            let x_vars: Vec<String> = self.model.get("x")
                .or_else(|| self.model.get("X"))
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| {
                    if v.is_number() { Some("intercept".into()) }
                    else { v.as_str().map(String::from) }
                }).collect())
                .unwrap_or_default();

            // Build contrasts
            let mut contrasts = Vec::new();

            // Dummy contrasts
            if let Some(ref dc) = self.dummy_contrasts {
                let test = dc.get("test").or(dc.get("Test"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("t")
                    .to_string();
                for var_name in &x_vars {
                    if var_name == "intercept" { continue; }
                    contrasts.push(ContrastInfo {
                        name: var_name.clone(),
                        conditions: vec![var_name.clone()],
                        weights: vec![1.0],
                        test: Some(test.clone()),
                        entities: collection.entities.clone(),
                    });
                }
            }

            // Explicit contrasts
            for con_spec in &self.contrasts {
                let name = con_spec.get("name").or(con_spec.get("Name"))
                    .and_then(|v| v.as_str()).unwrap_or("unnamed");
                let conditions: Vec<String> = con_spec.get("condition_list")
                    .or(con_spec.get("ConditionList"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let weights: Vec<f64> = con_spec.get("weights").or(con_spec.get("Weights"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(serde_json::Value::as_f64).collect())
                    .unwrap_or_default();
                let test = con_spec.get("test").or(con_spec.get("Test"))
                    .and_then(|v| v.as_str()).map(String::from);

                let mut entities = collection.entities.clone();
                entities.insert("contrast".into(), name.into());

                contrasts.push(ContrastInfo {
                    name: name.into(), conditions, weights, test, entities,
                });
            }

            // Build design matrix from collection
            let dm = if !x_vars.is_empty() {
                let mut cols = Vec::new();
                let mut col_names = Vec::new();
                for var_name in &x_vars {
                    if var_name == "intercept" {
                        let n = coll.variables.values().next().map_or(0, bids_variables::SimpleVariable::len);
                        cols.push(vec![1.0; n]);
                        col_names.push("intercept".into());
                    } else if let Some(var) = coll.variables.get(var_name) {
                        cols.push(var.values.clone());
                        col_names.push(var_name.clone());
                    }
                }
                if !cols.is_empty() { Some((col_names, cols)) } else { None }
            } else { None };

            results.push(StatsModelsNodeOutput {
                node_name: self.name.clone(),
                entities: collection.entities.clone(),
                x_variables: x_vars.clone(),
                contrasts,
                design_matrix: dm,
            });
        }

        results
    }
}

/// Output produced by running a stats model node.
///
/// Contains the design matrix, contrasts, and entity metadata for a single
/// group within a node's analysis. Multiple outputs may be produced per node
/// when data is split by grouping variables.
#[derive(Debug, Clone)]
pub struct StatsModelsNodeOutput {
    pub node_name: String,
    pub entities: StringEntities,
    pub x_variables: Vec<String>,
    pub contrasts: Vec<ContrastInfo>,
    /// Design matrix: (column_names, data_rows).
    pub design_matrix: Option<(Vec<String>, Vec<Vec<f64>>)>,
}

impl StatsModelsNodeOutput {
    /// Get the design matrix column names (X).
    pub fn x_columns(&self) -> &[String] {
        &self.x_variables
    }
}

/// Build groups from a list of entity maps, grouping by specified keys.
/// Returns map from group key to indices.
pub fn build_groups(
    entity_maps: &[StringEntities],
    group_by: &[String],
) -> std::collections::HashMap<Vec<(String, String)>, Vec<usize>> {
    let mut groups: std::collections::HashMap<Vec<(String, String)>, Vec<usize>> = std::collections::HashMap::new();

    if group_by.is_empty() {
        groups.insert(vec![], (0..entity_maps.len()).collect());
        return groups;
    }

    // Get unique values for each grouping variable
    let mut unique_vals: std::collections::HashMap<&str, Vec<String>> = std::collections::HashMap::new();
    for col in group_by {
        let vals: std::collections::BTreeSet<String> = entity_maps.iter()
            .filter_map(|e| e.get(col.as_str()).cloned())
            .collect();
        unique_vals.insert(col.as_str(), vals.into_iter().collect());
    }

    for (i, ents) in entity_maps.iter().enumerate() {
        let mut base: Vec<(String, String)> = Vec::new();
        let mut missing: Vec<&str> = Vec::new();

        for col in group_by {
            if let Some(val) = ents.get(col.as_str()) {
                base.push((col.clone(), val.clone()));
            } else {
                missing.push(col.as_str());
            }
        }

        if missing.is_empty() {
            base.sort();
            groups.entry(base).or_default().push(i);
        } else {
            // Cartesian product of missing values
            let mut combos = vec![base.clone()];
            for col in &missing {
                if let Some(vals) = unique_vals.get(col) {
                    let mut new_combos = Vec::new();
                    for combo in &combos {
                        for val in vals {
                            let mut c = combo.clone();
                            c.push((col.to_string(), val.clone()));
                            new_combos.push(c);
                        }
                    }
                    combos = new_combos;
                }
            }
            for mut combo in combos {
                combo.sort();
                groups.entry(combo).or_default().push(i);
            }
        }
    }

    groups
}
