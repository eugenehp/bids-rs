//! `bids ls` subcommand: list files matching filters.

use bids_layout::BidsLayout;
use std::path::PathBuf;

pub fn run(
    root: &PathBuf, subject: Option<String>, session: Option<String>,
    task: Option<String>, suffix: Option<String>,
    datatype: Option<String>, extension: Option<String>,
) -> bids_core::error::Result<()> {
    let layout = BidsLayout::new(root)?;
    let mut query = layout.get().invalid_filters(bids_layout::InvalidFilters::Drop);
    if let Some(ref s) = subject { query = query.subject(s); }
    if let Some(ref s) = session { query = query.session(s); }
    if let Some(ref t) = task { query = query.task(t); }
    if let Some(ref s) = suffix { query = query.suffix(s); }
    if let Some(ref d) = datatype { query = query.datatype(d); }
    if let Some(ref e) = extension { query = query.extension(e); }
    let files = query.collect()?;
    for f in &files {
        println!("{}", f.relpath(layout.root()).map_or_else(
            || f.path.display().to_string(),
            |r| r.display().to_string()));
    }
    println!("\n{} files found.", files.len());
    Ok(())
}
