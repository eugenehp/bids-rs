//! Loading image data straight from an indexed [`BidsFile`].
//!
//! `bids-core` cannot depend on this crate — that would be a dependency cycle
//! — so PyBIDS' `BIDSImageFile.get_image()` is provided here as an extension
//! trait. Bring [`BidsImageFile`] into scope and the methods are available on
//! any `BidsFile`.

use bids_core::file::{BidsFile, FileType};

use crate::{NiftiError, NiftiHeader, NiftiImage};

/// Image access for files indexed by a BIDS layout.
///
/// Corresponds to PyBIDS' `BIDSImageFile`.
///
/// # Example
///
/// ```no_run
/// use bids_layout::BidsLayout;
/// use bids_nifti::BidsImageFile;
///
/// let layout = BidsLayout::new("/path/to/dataset").unwrap();
/// let file = layout.get_file("/path/to/sub-01_bold.nii.gz").unwrap().unwrap();
/// if file.is_image() {
///     let header = file.get_header().unwrap();
///     println!("{} volumes, TR {:?}", header.n_vols(), header.tr());
/// }
/// ```
pub trait BidsImageFile {
    /// Whether this file is a neuroimaging volume this trait can load.
    fn is_image(&self) -> bool;

    /// Read the NIfTI header without loading voxel data.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not an image, or if the header cannot
    /// be read or parsed.
    fn get_header(&self) -> Result<NiftiHeader, NiftiError>;

    /// Load the full image — header and voxel data.
    ///
    /// Corresponds to PyBIDS' `BIDSImageFile.get_image()`, which returns a
    /// nibabel image.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not an image, or if it cannot be read
    /// or parsed.
    fn get_image(&self) -> Result<NiftiImage, NiftiError>;
}

impl BidsImageFile for BidsFile {
    fn is_image(&self) -> bool {
        matches!(self.file_type, FileType::Image | FileType::Pet)
            && matches!(self.extension(), ".nii" | ".nii.gz" | ".hdr" | ".img")
    }

    fn get_header(&self) -> Result<NiftiHeader, NiftiError> {
        self.require_image()?;
        NiftiHeader::from_file(&self.path)
    }

    fn get_image(&self) -> Result<NiftiImage, NiftiError> {
        self.require_image()?;
        NiftiImage::from_file(&self.path)
    }
}

/// Private helper shared by the trait methods.
trait RequireImage {
    fn require_image(&self) -> Result<(), NiftiError>;
}

impl RequireImage for BidsFile {
    fn require_image(&self) -> Result<(), NiftiError> {
        if BidsImageFile::is_image(self) {
            return Ok(());
        }
        Err(NiftiError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "'{}' is not a NIfTI image (extension '{}')",
                self.filename,
                self.extension()
            ),
        )))
    }
}
