#![deny(unsafe_code)]
//! BIDS Statistical Models implementation.
//!
//! This crate implements the [BIDS-StatsModels](https://bids-standard.github.io/stats-models/)
//! specification for defining reproducible neuroimaging analysis pipelines as
//! declarative JSON documents. It corresponds to PyBIDS' `bids.modeling` module.
//!
//! # Overview
//!
//! A BIDS Stats Model defines a directed acyclic graph (DAG) of analysis nodes,
//! each operating at a specific level of the BIDS hierarchy (run, session,
//! subject, dataset). Data flows from lower levels to higher levels through
//! edges, with contrasts propagating upward.
//!
//! # Components
//!
//! - [`StatsModelsGraph`] — The top-level model graph loaded from a JSON file.
//!   Validates structure, wires edges between nodes, and executes the full
//!   analysis pipeline. Can export to Graphviz DOT format.
//!
//! - [`StatsModelsNode`] — A single analysis node with a statistical model
//!   specification, variable transformations, contrasts, and dummy contrasts.
//!   Nodes group data by entity values and produce [`StatsModelsNodeOutput`]s.
//!
//! - [`TransformSpec`] and [`apply_transformations()`] — The `pybids-transforms-v1`
//!   transformer, implementing all 24 PyBIDS transformations (see
//!   [`SUPPORTED_TRANSFORMATIONS`]). Instructions that cannot be applied are
//!   reported as [`TransformError`]s rather than skipped.
//!
//! - [`HrfModel`], [`spm_hrf()`], [`glover_hrf()`] — Hemodynamic response function
//!   kernels (SPM and Glover canonical forms with optional time and dispersion
//!   derivatives). Uses a pure-Rust `gammaln` implementation matching SciPy's
//!   cephes to machine epsilon.
//!
//! - [`node_report()`] and [`generate_report()`] — Model diagnostics: contrast
//!   matrices, per-regressor and per-contrast variance inflation factors, and a
//!   self-contained HTML report with inline SVG figures.
//!
//! - [`auto_model()`] — Automatically generates a BIDS Stats Model JSON for
//!   each task in a dataset, with Factor(trial_type) at the run level and
//!   pass-through nodes at higher levels.
//!
//! - [`GlmSpec`] (aliased [`GlmmSpec`]), [`MetaAnalysisSpec`] — Statistical model
//!   specifications covering both the fixed-effect design (X) and random-effect
//!   variance components (Z, via [`VarComp`]), with [`Prior`] support, design
//!   matrix construction, VIF computation, and formatted output.
//!
//! # Example
//!
//! ```no_run
//! use bids_modeling::StatsModelsGraph;
//!
//! let mut graph = StatsModelsGraph::from_file(
//!     std::path::Path::new("model-default_smdl.json")
//! ).unwrap();
//! graph.validate().unwrap();
//! println!("DOT graph:\n{}", graph.write_graph());
//!
//! // Execute the model
//! let outputs = graph.run();
//! for output in &outputs {
//!     println!("Node: {}, Contrasts: {}", output.node_name, output.contrasts.len());
//! }
//! ```

pub mod auto_model;
pub mod graph;
pub mod hrf;
pub mod node;
pub mod report;
pub mod spec;
pub mod transformations;

pub use auto_model::auto_model;
pub use graph::StatsModelsGraph;
pub use hrf::{HrfModel, compute_regressor, glover_hrf, spm_hrf};
pub use node::{
    ContrastInfo, StatsModelsEdge, StatsModelsNode, StatsModelsNodeOutput, build_groups,
};
pub use report::{
    ContrastMatrix, FigureOptions, NodeReport, deroot, displayify, est_vif,
    generate_contrast_matrix, generate_report, get_all_contrast_vif, get_eff_reg_vif, node_report,
    plot_corr_matrix, plot_design_matrix, plot_design_timecourse, plot_vif, to_alphanum,
};
pub use spec::{
    GlmSpec, GlmmSpec, MetaAnalysisSpec, Prior, Term, VarComp, compute_vif, dummies_to_vec,
    format_correlation_matrix, format_design_matrix,
};
pub use transformations::{
    Instruction, SUPPORTED_TRANSFORMATIONS, TransformError, TransformSpec, TransformerManager,
    apply_transformations, expand_wildcards,
};

/// BIDS-StatsModels: graphs, transformations, HRFs and reports.
///
/// ```
/// use bids_modeling::prelude::*;
/// ```
pub mod prelude {
    pub use crate::auto_model;
    pub use crate::graph::StatsModelsGraph;
    pub use crate::hrf::{HrfModel, compute_regressor, glover_hrf, spm_hrf};
    pub use crate::node::{ContrastInfo, StatsModelsNode, StatsModelsNodeOutput};
    pub use crate::report::{NodeReport, generate_report, node_report};
    pub use crate::spec::{GlmSpec, GlmmSpec, Prior, Term, VarComp, compute_vif};
    pub use crate::transformations::{
        TransformError, TransformSpec, TransformerManager, apply_transformations,
    };
}
