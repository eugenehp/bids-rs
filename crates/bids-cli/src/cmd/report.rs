//! `bids report` subcommand: generate methods sections.

use bids_layout::BidsLayout;
use std::path::PathBuf;

pub fn run(root: &PathBuf) -> bids_core::error::Result<()> {
    let layout = BidsLayout::new(root)?;
    let report = bids_reports::BidsReport::new(&layout);
    println!("{}", report.generate()?);
    Ok(())
}
