//! `bids entities` subcommand: list unique entity values.

use bids_layout::BidsLayout;
use std::path::PathBuf;

pub fn run(root: &PathBuf) -> bids_core::error::Result<()> {
    let layout = BidsLayout::new(root)?;
    for entity in &layout.get_entities()? {
        let values = layout.get_entity_values(entity)?;
        println!("{entity}: {values:?}");
    }
    Ok(())
}
