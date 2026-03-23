//! File writing utilities with conflict resolution for BIDS datasets.
//!
//! Supports writing file contents, creating symlinks, or copying from source
//! files, with configurable behavior when the target path already exists.

use bids_core::error::{BidsError, Result};
use std::path::{Path, PathBuf};

/// Conflict resolution strategy when the output path already exists.
///
/// Used by [`write_to_file()`] to determine behavior when the target file
/// is already present on disk.
///
/// | Strategy | Behavior |
/// |----------|----------|
/// | `Fail` | Return an error (default — safest) |
/// | `Skip` | Silently do nothing |
/// | `Overwrite` | Delete existing file and write new one |
/// | `Append` | Write to `name_1.ext`, `name_2.ext`, etc. |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictStrategy {
    /// Return an error if the file exists.
    #[default]
    Fail,
    /// Silently skip writing if the file exists.
    Skip,
    /// Delete and replace the existing file.
    Overwrite,
    /// Write to a numbered variant (`file_1.ext`, `file_2.ext`, …).
    Append,
}

impl std::fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fail => write!(f, "fail"),
            Self::Skip => write!(f, "skip"),
            Self::Overwrite => write!(f, "overwrite"),
            Self::Append => write!(f, "append"),
        }
    }
}

/// Write contents to a file, optionally creating a symlink or copying from another file.
///
/// Corresponds to PyBIDS' `write_to_file()`.
///
/// # Errors
///
/// Returns an error if the file already exists (with `ConflictStrategy::Fail`),
/// the source file doesn't exist (for `copy_from`), no data source is
/// provided, or any I/O operation fails.
pub fn write_to_file(
    path: &Path,
    contents: Option<&[u8]>,
    link_to: Option<&Path>,
    copy_from: Option<&Path>,
    root: Option<&Path>,
    conflicts: ConflictStrategy,
) -> Result<()> {
    let mut full_path = match root {
        Some(r) if !path.is_absolute() => r.join(path),
        _ => path.to_path_buf(),
    };

    if full_path.exists() || full_path.is_symlink() {
        match conflicts {
            ConflictStrategy::Fail => {
                return Err(BidsError::Io(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("A file at path {} already exists", full_path.display()),
                )));
            }
            ConflictStrategy::Skip => return Ok(()),
            ConflictStrategy::Overwrite => {
                if full_path.is_dir() {
                    return Ok(()); // Don't overwrite directories
                }
                std::fs::remove_file(&full_path)?;
            }
            ConflictStrategy::Append => {
                full_path = find_append_path(&full_path);
            }
        }
    }

    // Create parent dirs
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if let Some(link_target) = link_to {
        #[cfg(unix)]
        std::os::unix::fs::symlink(link_target, &full_path)?;
        #[cfg(not(unix))]
        std::fs::copy(link_target, &full_path)?;
    } else if let Some(src) = copy_from {
        if !src.exists() {
            return Err(BidsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Source file '{}' does not exist", src.display()),
            )));
        }
        std::fs::copy(src, &full_path)?;
    } else if let Some(data) = contents {
        std::fs::write(&full_path, data)?;
    } else {
        return Err(BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "One of contents, copy_from or link_to must be provided",
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_contents() {
        let dir = std::env::temp_dir().join("bids_writer_test_contents");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        write_to_file(
            &path,
            Some(b"hello"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_conflict_fail() {
        let dir = std::env::temp_dir().join("bids_writer_test_fail");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        write_to_file(
            &path,
            Some(b"first"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        let result = write_to_file(
            &path,
            Some(b"second"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        );
        assert!(result.is_err());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_conflict_skip() {
        let dir = std::env::temp_dir().join("bids_writer_test_skip");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        write_to_file(
            &path,
            Some(b"first"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        write_to_file(
            &path,
            Some(b"second"),
            None,
            None,
            None,
            ConflictStrategy::Skip,
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first"); // unchanged

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_conflict_overwrite() {
        let dir = std::env::temp_dir().join("bids_writer_test_overwrite");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        write_to_file(
            &path,
            Some(b"first"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        write_to_file(
            &path,
            Some(b"second"),
            None,
            None,
            None,
            ConflictStrategy::Overwrite,
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_conflict_append() {
        let dir = std::env::temp_dir().join("bids_writer_test_append");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        write_to_file(
            &path,
            Some(b"first"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        write_to_file(
            &path,
            Some(b"second"),
            None,
            None,
            None,
            ConflictStrategy::Append,
        )
        .unwrap();

        // Original should be unchanged, new file should exist as test_1.txt
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");
        assert_eq!(
            fs::read_to_string(dir.join("test_1.txt")).unwrap(),
            "second"
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("bids_writer_test_parents");
        let path = dir.join("a").join("b").join("c").join("test.txt");

        write_to_file(
            &path,
            Some(b"deep"),
            None,
            None,
            None,
            ConflictStrategy::Fail,
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "deep");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_copy_from() {
        let dir = std::env::temp_dir().join("bids_writer_test_copy");
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("source.txt");
        let dst = dir.join("dest.txt");

        fs::write(&src, b"source content").unwrap();
        write_to_file(&dst, None, None, Some(&src), None, ConflictStrategy::Fail).unwrap();
        assert_eq!(fs::read_to_string(&dst).unwrap(), "source content");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_no_source_errors() {
        let dir = std::env::temp_dir().join("bids_writer_test_nosrc");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");

        let result = write_to_file(&path, None, None, None, None, ConflictStrategy::Fail);
        assert!(result.is_err());

        fs::remove_dir_all(&dir).unwrap();
    }
}

fn find_append_path(path: &Path) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let parent = path.parent().unwrap_or(Path::new("."));

    for i in 1..i32::MAX {
        let new_name = if ext.is_empty() {
            format!("{stem}_{i}")
        } else {
            format!("{stem}_{i}.{ext}")
        };
        let candidate = parent.join(new_name);
        if !candidate.exists() {
            return candidate;
        }
    }
    path.to_path_buf() // Fallback
}
