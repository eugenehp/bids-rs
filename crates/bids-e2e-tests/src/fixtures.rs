//! Automatic test fixture management for bids-examples.
//!
//! Provides [`BidsExamples`], which locates or auto-downloads the official
//! [bids-examples](https://github.com/bids-standard/bids-examples) repository
//! to a local cache directory.  Tests call [`BidsExamples::require()`] and get
//! back a path — no manual `git clone` step required.
//!
//! # Resolution order
//!
//! 1. `BIDS_EXAMPLES_DIR` environment variable (explicit override)
//! 2. Sibling directory `../bids-examples` relative to the workspace root
//! 3. Platform cache: `~/.cache/bids-rs/bids-examples` (Linux/macOS)
//!    or `%LOCALAPPDATA%/bids-rs/bids-examples` (Windows)
//!
//! If none of the above exist, [`BidsExamples::require()`] clones the repo
//! automatically (requires `git` on `PATH`).
//!
//! # Offline mode
//!
//! Set `BIDS_EXAMPLES_OFFLINE=1` to skip the auto-download and only use
//! already-present directories.  Tests will be skipped if no fixture is found.
//!
//! # Example
//!
//! ```rust,ignore
//! use bids_e2e_tests::fixtures::BidsExamples;
//!
//! let examples = BidsExamples::require();
//! let ds001 = examples.dataset("ds001").unwrap();
//! let layout = bids_layout::BidsLayout::new(&ds001).unwrap();
//! ```

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Repository URL for the official BIDS examples.
const REPO_URL: &str = "https://github.com/bids-standard/bids-examples.git";

/// Sentinel file we check to confirm a valid checkout.
const SENTINEL: &str = "ds001";

/// Manages access to the bids-examples test fixture directory.
#[derive(Debug, Clone)]
pub struct BidsExamples {
    root: PathBuf,
}

/// Global singleton — download happens at most once per test run.
static INSTANCE: OnceLock<Option<BidsExamples>> = OnceLock::new();

impl BidsExamples {
    /// Get the bids-examples fixture, downloading if necessary.
    ///
    /// Returns `None` if offline mode is active and no local copy exists,
    /// or if `git` is not available.
    pub fn require() -> Option<&'static BidsExamples> {
        INSTANCE
            .get_or_init(|| Self::resolve().or_else(Self::auto_download))
            .as_ref()
    }

    /// Root directory containing all example datasets.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to a specific example dataset (e.g., `"ds001"`).
    ///
    /// Returns `None` if the dataset directory doesn't exist.
    pub fn dataset(&self, name: &str) -> Option<PathBuf> {
        let p = self.root.join(name);
        if p.is_dir() && p.join("dataset_description.json").exists() {
            Some(p)
        } else {
            None
        }
    }

    /// List all available example dataset names.
    pub fn list_datasets(&self) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.join("dataset_description.json").exists() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names.sort();
        names
    }

    // ── Resolution ──────────────────────────────────────────────────────

    /// Try to find an existing bids-examples directory.
    fn resolve() -> Option<BidsExamples> {
        // 1. Env var override
        if let Ok(dir) = std::env::var("BIDS_EXAMPLES_DIR") {
            let p = PathBuf::from(&dir);
            if is_valid_checkout(&p) {
                return Some(BidsExamples { root: p });
            }
        }

        // 2. Sibling directories (workspace-relative)
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_default();

        for relative in &["../bids-examples", "bids-examples"] {
            let candidate = workspace_root.join(relative);
            if is_valid_checkout(&candidate) {
                return Some(BidsExamples {
                    root: candidate.canonicalize().unwrap_or(candidate),
                });
            }
        }

        // 3. Platform cache directory
        let cache = cache_dir();
        if is_valid_checkout(&cache) {
            return Some(BidsExamples { root: cache });
        }

        None
    }

    /// Clone the repository into the cache directory.
    fn auto_download() -> Option<BidsExamples> {
        if std::env::var("BIDS_EXAMPLES_OFFLINE").is_ok() {
            eprintln!("bids-examples: offline mode — skipping download");
            return None;
        }

        let dest = cache_dir();
        if is_valid_checkout(&dest) {
            return Some(BidsExamples { root: dest });
        }

        eprintln!("bids-examples: auto-downloading to {} ...", dest.display());

        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let status = std::process::Command::new("git")
            .args(["clone", "--depth", "1", REPO_URL])
            .arg(&dest)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .status();

        match status {
            Ok(s) if s.success() && is_valid_checkout(&dest) => {
                eprintln!("bids-examples: downloaded successfully");
                Some(BidsExamples { root: dest })
            }
            Ok(s) => {
                eprintln!("bids-examples: git clone failed (exit {})", s);
                None
            }
            Err(e) => {
                eprintln!("bids-examples: git not found ({e}), skipping download");
                None
            }
        }
    }
}

/// Platform-specific cache directory.
fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("bids-rs").join("bids-examples");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("bids-rs")
            .join("bids-examples");
    }
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("bids-rs").join("bids-examples");
    }
    PathBuf::from("/tmp/bids-rs/bids-examples")
}

fn is_valid_checkout(path: &Path) -> bool {
    path.is_dir() && path.join(SENTINEL).is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir_is_absolute() {
        let dir = cache_dir();
        assert!(
            dir.is_absolute() || dir.starts_with("/tmp"),
            "cache dir should be absolute: {}",
            dir.display()
        );
    }

    #[test]
    fn test_require_returns_valid_root() {
        // This will either find an existing checkout or download one
        if let Some(examples) = BidsExamples::require() {
            assert!(examples.root().is_dir());
            assert!(examples.root().join("ds001").is_dir());
            assert!(!examples.list_datasets().is_empty());
            assert!(examples.dataset("ds001").is_some());
            assert!(examples.dataset("nonexistent_dataset_xyz").is_none());
        } else {
            eprintln!("SKIP: bids-examples not available and auto-download failed");
        }
    }
}
