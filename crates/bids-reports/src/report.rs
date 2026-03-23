//! Report generation: auto-produce methods sections from BIDS datasets.

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_layout::BidsLayout;
use crate::parsing;

/// Generate publication-quality data acquisition methods from a BIDS dataset.
///
/// Analyzes the dataset structure, metadata, and acquisition parameters to
/// produce human-readable prose suitable for a research paper's methods
/// section. Handles functional MRI, structural MRI, diffusion MRI, field
/// maps, and EEG datatypes.
///
/// Corresponds to PyBIDS' `BIDSReport` class.
///
/// # Example
///
/// ```no_run
/// # use bids_layout::BidsLayout;
/// use bids_reports::BidsReport;
///
/// # let layout = BidsLayout::new("/path").unwrap();
/// let report = BidsReport::new(&layout);
/// let methods_text = report.generate().unwrap();
/// println!("{}", methods_text);
/// ```
///
/// # Warning
///
/// Generated text should always be reviewed and verified before publication.
pub struct BidsReport<'a> {
    layout: &'a BidsLayout,
}

impl<'a> BidsReport<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }

    /// Generate a full methods section for the dataset.
    pub fn generate(&self) -> Result<String> {
        let mut sections = Vec::new();

        // Dataset info
        if let Some(desc) = self.layout.description() {
            sections.push(format!("Data from the \"{}\" dataset were used.", desc.name));
        }

        // Collect datatypes and their files
        let datatypes = self.layout.get_datatypes()?;
        let mut datatype_info = Vec::new();

        for dt in &datatypes {
            let files = self.layout.get().datatype(dt).collect()?;
            if files.is_empty() { continue; }

            // Get metadata from first file of each suffix
            let mut by_suffix: std::collections::HashMap<String, Vec<BidsFile>> =
                std::collections::HashMap::new();
            for f in &files {
                if let Some(s) = f.suffix() {
                    by_suffix.entry(s.to_string()).or_default().push(f.clone());
                }
            }

            for (suffix, suffix_files) in &by_suffix {
                let md = self.layout.get_metadata(&suffix_files[0].path)?;
                // Skip JSON sidecars
                if suffix.as_str() == "description" { continue; }
                let data_files: Vec<BidsFile> = suffix_files.iter()
                    .filter(|f| f.file_type != bids_core::file::FileType::Json)
                    .cloned().collect();
                if data_files.is_empty() { continue; }
                datatype_info.push((dt.clone(), data_files, md));
            }
        }

        let methods = parsing::generate_methods(&datatype_info);
        if !methods.is_empty() {
            sections.push(methods);
        }

        // Subjects info
        let subjects = self.layout.get_subjects()?;
        sections.push(format!("{} participants were included.", subjects.len()));

        Ok(sections.join("\n\n"))
    }

    /// Generate methods for specific files only.
    pub fn generate_from_files(&self, files: &[BidsFile]) -> Result<String> {
        let mut by_datatype: std::collections::HashMap<String, Vec<BidsFile>> =
            std::collections::HashMap::new();
        for f in files {
            let dt = f.entities.get("datatype")
                .map(|v| v.as_str_lossy().into_owned())
                .unwrap_or_else(|| "unknown".into());
            by_datatype.entry(dt).or_default().push(f.clone());
        }

        let mut datatype_info = Vec::new();
        for (dt, dt_files) in &by_datatype {
            let md = self.layout.get_metadata(&dt_files[0].path)?;
            datatype_info.push((dt.clone(), dt_files.clone(), md));
        }

        Ok(parsing::generate_methods(&datatype_info))
    }
}
