//! `bids info` subcommand: show dataset summary.

use bids_layout::BidsLayout;
use std::path::PathBuf;

pub fn run(root: &PathBuf, validate: bool) -> bids_core::error::Result<()> {
    let layout = BidsLayout::builder(root).validate(validate).build()?;
    println!("{layout}");
    if let Some(desc) = layout.description() {
        println!("  Name: {}", desc.name);
        println!("  BIDS Version: {}", desc.bids_version);
        if let Some(ref license) = desc.license {
            println!("  License: {license}");
        }
        if let Some(ref authors) = desc.authors {
            println!("  Authors: {}", authors.join(", "));
        }
    }
    let datatypes = layout.get_datatypes()?;
    if !datatypes.is_empty() {
        println!("  Datatypes: {}", datatypes.join(", "));
    }
    Ok(())
}
