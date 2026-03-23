//! `bids upgrade` subcommand: migrate dataset_description.json to latest BIDS.

use std::path::Path;

pub fn run(root: &Path) -> bids_core::error::Result<()> {
    let desc_path = root.join("dataset_description.json");
    if !desc_path.exists() {
        return Err(bids_core::BidsError::MissingDatasetDescription);
    }
    let contents = std::fs::read_to_string(&desc_path)?;
    let mut desc: serde_json::Value = serde_json::from_str(&contents)?;

    let mut changed = false;

    let target_version = bids::SUPPORTED_BIDS_VERSION.to_string();
    let min_version = bids::MIN_COMPATIBLE_VERSION.to_string();

    // Update BIDSVersion if below minimum compatible version
    let old_ver = desc.get("BIDSVersion").and_then(|v| v.as_str()).map(String::from);
    if let Some(ver) = old_ver
        && ver.as_str() < min_version.as_str() {
            desc["BIDSVersion"] = serde_json::json!(target_version);
            changed = true;
            println!("Updated BIDSVersion: {ver} -> {target_version}");
        }

    // Add DatasetType if missing
    if desc.get("DatasetType").is_none() {
        desc["DatasetType"] = serde_json::json!("raw");
        changed = true;
        println!("Added DatasetType: raw");
    }

    // Convert PipelineDescription to GeneratedBy
    if desc.get("PipelineDescription").is_some() && desc.get("GeneratedBy").is_none() {
        let pd = desc["PipelineDescription"].take();
        desc["GeneratedBy"] = serde_json::json!([pd]);
        desc.as_object_mut().map(|m| m.remove("PipelineDescription"));
        changed = true;
        println!("Converted PipelineDescription to GeneratedBy");
    }

    if changed {
        let output = serde_json::to_string_pretty(&desc)?;
        std::fs::write(&desc_path, output)?;
        println!("Saved updated dataset_description.json");
    } else {
        println!("No changes needed.");
    }
    Ok(())
}
