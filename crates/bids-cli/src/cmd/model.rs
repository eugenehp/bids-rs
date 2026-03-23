//! `bids model` / `bids auto-model` subcommands: BIDS-StatsModels.

use bids_layout::BidsLayout;
use std::path::Path;

pub fn auto_model(root: &Path) -> bids_core::error::Result<()> {
    let layout = BidsLayout::new(root)?;
    let models = bids_modeling::auto_model(&layout)?;
    for model in &models {
        println!(
            "{}",
            serde_json::to_string_pretty(model).unwrap_or_default()
        );
    }
    Ok(())
}

pub fn model_report(model_path: &Path, root: &Path) -> bids_core::error::Result<()> {
    let mut graph = bids_modeling::StatsModelsGraph::from_file(model_path)?;
    graph.validate()?;
    println!("Model: {}", graph.name);
    println!("Nodes: {}", graph.nodes.len());
    println!("Edges: {}", graph.edges.len());
    println!("\nGraph (DOT):\n{}", graph.write_graph());

    let layout = BidsLayout::new(root)?;
    graph.load_collections(&layout);
    let outputs = graph.run();
    println!("Outputs: {}", outputs.len());
    for output in &outputs {
        println!(
            "  Node: {}, Entities: {:?}, Contrasts: {}",
            output.node_name,
            output.entities,
            output.contrasts.len()
        );
        for c in &output.contrasts {
            println!(
                "    Contrast: {} ({})",
                c.name,
                c.test.as_deref().unwrap_or("?")
            );
        }
    }
    Ok(())
}
