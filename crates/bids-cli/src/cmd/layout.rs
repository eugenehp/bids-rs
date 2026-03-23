//! `bids layout` subcommand: create persistent SQLite index.

use bids_layout::BidsLayout;
use std::path::Path;

pub fn run(root: &Path, db_path: &Path, reset_db: bool,
           validate: bool, index_metadata: bool) -> bids_core::error::Result<()> {
    let db_file = db_path.join("layout_index.sqlite");
    if reset_db && db_file.exists() {
        std::fs::remove_file(&db_file)?;
    }
    let layout = BidsLayout::builder(root)
        .validate(validate)
        .index_metadata(index_metadata)
        .database_path(&db_file)
        .build()?;
    println!("Database index created at {}", db_file.display());
    println!("{layout}");
    Ok(())
}
