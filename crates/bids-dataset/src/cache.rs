//! HuggingFace-style local cache for BIDS datasets.
//!
//! Cache layout (mirrors HuggingFace Hub cache structure):
//!
//! ```text
//! ~/.cache/bids-rs/
//!   datasets/
//!     ds004362/
//!       refs/
//!         1.0.0          → file containing snapshot hash (commit sha)
//!         latest         → symlink to latest version ref
//!       snapshots/
//!         {hash}/        → actual dataset files (hardlinked from blobs)
//!           dataset_description.json
//!           sub-001/eeg/sub-001_task-rest_eeg.edf
//!       blobs/
//!         {sha256}-{size} → content-addressed blob storage
//!   catalog.json          → cached search results (TTL-based)
//! ```
//!
//! Blobs are content-addressed by SHA-256 + size. Snapshot directories
//! contain hardlinks to blobs, so multiple versions of the same file
//! don't waste disk space.
//!
//! On browser wasm the cache lives in the Origin Private File System (the
//! `opfs` module there), its root an OPFS path, and
//! [`OpenNeuro::download_to_cache`](crate::OpenNeuro::download_to_cache) is
//! async. OPFS has no links, so a blob would be a second copy of each file:
//! files go straight into their snapshot, as large files do natively, with
//! the same `.complete` marker and `refs/`. The methods that touch the cache
//! are async there; read its files with `opfs::read`.

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Default cache directory (~/.cache/bids-rs).
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    dirs_cache().join("bids-rs")
}

/// Platform-appropriate cache base directory.
fn dirs_cache() -> PathBuf {
    // $BIDS_CACHE_DIR > $XDG_CACHE_HOME > ~/.cache
    if let Ok(d) = std::env::var("BIDS_CACHE_DIR") {
        return PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(d);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache");
    }
    PathBuf::from(".cache")
}

/// Manages the local BIDS dataset cache.
pub struct Cache {
    root: PathBuf,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            root: default_cache_dir(),
        }
    }
}

impl Cache {
    /// Open a cache at a specific path.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Root path of the cache.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path to a dataset's directory in the cache.
    #[must_use]
    pub fn dataset_dir(&self, dataset_id: &str) -> PathBuf {
        self.root.join("datasets").join(dataset_id)
    }

    /// Path to a specific snapshot of a dataset.
    #[must_use]
    pub fn snapshot_dir(&self, dataset_id: &str, version: &str) -> PathBuf {
        self.dataset_dir(dataset_id).join("snapshots").join(version)
    }

    /// Path to the blobs directory for a dataset.
    #[must_use]
    pub fn blobs_dir(&self, dataset_id: &str) -> PathBuf {
        self.dataset_dir(dataset_id).join("blobs")
    }
}

/// The cache on disk, with `std::fs` (every target but browser wasm, which
/// has the async OPFS versions below).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
impl Cache {
    /// Check if a specific dataset version is fully cached.
    #[must_use]
    pub fn is_cached(&self, dataset_id: &str, version: &str) -> bool {
        let snap = self.snapshot_dir(dataset_id, version);
        snap.exists() && snap.join(".complete").exists()
    }

    /// Resolve a dataset to its local path. Returns `None` if not cached.
    ///
    /// If `version` is `None`, returns the latest cached version.
    #[must_use]
    pub fn resolve(&self, dataset_id: &str, version: Option<&str>) -> Option<PathBuf> {
        if let Some(v) = version {
            let snap = self.snapshot_dir(dataset_id, v);
            if snap.exists() { Some(snap) } else { None }
        } else {
            // Check refs/latest
            let latest_ref = self.dataset_dir(dataset_id).join("refs").join("latest");
            if let Ok(v) = std::fs::read_to_string(&latest_ref) {
                let v = v.trim();
                let snap = self.snapshot_dir(dataset_id, v);
                if snap.exists() { Some(snap) } else { None }
            } else {
                None
            }
        }
    }

    /// List all cached datasets.
    #[must_use]
    pub fn list_datasets(&self) -> Vec<CachedDataset> {
        let ds_dir = self.root.join("datasets");
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&ds_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    let versions = self.list_versions(&name);
                    let size = dir_size(&entry.path());
                    out.push(CachedDataset {
                        id: name,
                        versions,
                        size_bytes: size,
                    });
                }
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// List cached versions of a dataset.
    #[must_use]
    pub fn list_versions(&self, dataset_id: &str) -> Vec<String> {
        let snap_dir = self.dataset_dir(dataset_id).join("snapshots");
        let mut versions = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&snap_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    versions.push(name);
                }
            }
        }
        versions.sort();
        versions
    }

    /// Store a downloaded file into the blob cache and create a hardlink
    /// (or copy) in the snapshot directory.
    ///
    /// Returns the snapshot path of the file.
    pub fn store_file(
        &self,
        dataset_id: &str,
        version: &str,
        relative_path: &str,
        content: &[u8],
    ) -> Result<PathBuf, std::io::Error> {
        let blob_name = blob_name(content);

        // Write blob
        let blob_dir = self.blobs_dir(dataset_id);
        std::fs::create_dir_all(&blob_dir)?;
        let blob_path = blob_dir.join(&blob_name);
        if !blob_path.exists() {
            std::fs::write(&blob_path, content)?;
        }

        // Create snapshot entry (hardlink or copy)
        let snap_path = self.snapshot_dir(dataset_id, version).join(relative_path);
        if let Some(parent) = snap_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !snap_path.exists() {
            // Try hardlink first, fall back to copy
            if std::fs::hard_link(&blob_path, &snap_path).is_err() {
                std::fs::copy(&blob_path, &snap_path)?;
            }
        }

        Ok(snap_path)
    }

    /// Store a file from disk (already downloaded) into the cache.
    pub fn store_file_from_disk(
        &self,
        dataset_id: &str,
        version: &str,
        relative_path: &str,
        source: &Path,
    ) -> Result<PathBuf, std::io::Error> {
        let snap_path = self.snapshot_dir(dataset_id, version).join(relative_path);
        if snap_path.exists() {
            return Ok(snap_path);
        }

        if let Some(parent) = snap_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // For large files, just hardlink/copy directly (skip blob hashing)
        let meta = std::fs::metadata(source)?;
        if meta.len() > 10 * 1024 * 1024 {
            // Large file: hardlink or copy
            if std::fs::hard_link(source, &snap_path).is_err() {
                std::fs::copy(source, &snap_path)?;
            }
        } else {
            // Small file: read, hash, store as blob
            let content = std::fs::read(source)?;
            return self.store_file(dataset_id, version, relative_path, &content);
        }

        Ok(snap_path)
    }

    /// Mark a snapshot as complete.
    pub fn mark_complete(&self, dataset_id: &str, version: &str) -> Result<(), std::io::Error> {
        let snap = self.snapshot_dir(dataset_id, version);
        std::fs::create_dir_all(&snap)?;
        std::fs::write(snap.join(".complete"), version)?;

        // Update refs/latest
        let refs = self.dataset_dir(dataset_id).join("refs");
        std::fs::create_dir_all(&refs)?;
        std::fs::write(refs.join("latest"), version)?;
        std::fs::write(refs.join(version), version)?;

        Ok(())
    }

    /// Remove a specific dataset version from the cache.
    pub fn evict(&self, dataset_id: &str, version: &str) -> Result<(), std::io::Error> {
        let snap = self.snapshot_dir(dataset_id, version);
        if snap.exists() {
            std::fs::remove_dir_all(&snap)?;
        }
        Ok(())
    }

    /// Remove an entire dataset from the cache.
    pub fn evict_dataset(&self, dataset_id: &str) -> Result<(), std::io::Error> {
        let dir = self.dataset_dir(dataset_id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Total cache size in bytes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        dir_size(&self.root)
    }

    /// Clean the entire cache.
    pub fn clean(&self) -> Result<(), std::io::Error> {
        if self.root.exists() {
            std::fs::remove_dir_all(&self.root)?;
        }
        Ok(())
    }
}

/// The cache in OPFS (browser wasm): the same methods, async, the paths
/// OPFS paths. Reading, listing and removing work anywhere; writing
/// (`store_file`, `mark_complete`) only in a dedicated Web Worker (see
/// [`crate::opfs`]). There's no `store_file_from_disk`, as there's no disk.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
impl Cache {
    /// Check if a specific dataset version is fully cached.
    pub async fn is_cached(&self, dataset_id: &str, version: &str) -> bool {
        let complete = self.snapshot_dir(dataset_id, version).join(".complete");
        match (crate::opfs::root().await, crate::opfs::path_str(&complete)) {
            (Ok(root), Ok(path)) => crate::opfs::size(&root, path).await.is_some(),
            _ => false,
        }
    }

    /// Resolve a dataset to its OPFS path. Returns `None` if not cached.
    ///
    /// If `version` is `None`, returns the latest cached version.
    pub async fn resolve(&self, dataset_id: &str, version: Option<&str>) -> Option<PathBuf> {
        let root = crate::opfs::root().await.ok()?;
        let version = match version {
            Some(v) => v.to_string(),
            None => {
                let latest = self.dataset_dir(dataset_id).join("refs").join("latest");
                let latest = crate::opfs::read(crate::opfs::path_str(&latest).ok()?)
                    .await
                    .ok()?;
                String::from_utf8(latest).ok()?.trim().to_string()
            }
        };
        let snap = self.snapshot_dir(dataset_id, &version);
        crate::opfs::try_dir(&root, crate::opfs::path_str(&snap).ok()?)
            .await
            .map(|_| snap)
    }

    /// List all cached datasets.
    pub async fn list_datasets(&self) -> Vec<CachedDataset> {
        let mut out = Vec::new();
        let Ok(root) = crate::opfs::root().await else {
            return out;
        };
        let datasets = self.root.join("datasets");
        let Some(datasets) = (match crate::opfs::path_str(&datasets) {
            Ok(path) => crate::opfs::try_dir(&root, path).await,
            Err(_) => None,
        }) else {
            return out;
        };
        for (name, is_dir) in crate::opfs::list(&datasets).await.unwrap_or_default() {
            if is_dir && !name.starts_with('.') {
                let size_bytes = match crate::opfs::try_dir(&datasets, &name).await {
                    Some(dir) => crate::opfs::tree_size(&dir).await,
                    None => 0,
                };
                out.push(CachedDataset {
                    versions: self.list_versions(&name).await,
                    id: name,
                    size_bytes,
                });
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// List cached versions of a dataset.
    pub async fn list_versions(&self, dataset_id: &str) -> Vec<String> {
        let snapshots = self.dataset_dir(dataset_id).join("snapshots");
        let (Ok(root), Ok(path)) = (crate::opfs::root().await, crate::opfs::path_str(&snapshots))
        else {
            return Vec::new();
        };
        let Some(snapshots) = crate::opfs::try_dir(&root, path).await else {
            return Vec::new();
        };
        let mut versions: Vec<String> = crate::opfs::list(&snapshots)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|(name, is_dir)| *is_dir && !name.starts_with('.'))
            .map(|(name, _)| name)
            .collect();
        versions.sort();
        versions
    }

    /// Store a downloaded file in the snapshot directory (no blob: OPFS has
    /// no links, so one would be a second copy).
    ///
    /// Returns the snapshot path of the file.
    pub async fn store_file(
        &self,
        dataset_id: &str,
        version: &str,
        relative_path: &str,
        content: &[u8],
    ) -> Result<PathBuf, std::io::Error> {
        let snap_path = self.snapshot_dir(dataset_id, version).join(relative_path);
        let root = crate::opfs::root().await.map_err(into_io)?;
        crate::opfs::write(
            &root,
            crate::opfs::path_str(&snap_path).map_err(into_io)?,
            content,
        )
        .await
        .map_err(into_io)?;
        Ok(snap_path)
    }

    /// Mark a snapshot as complete.
    pub async fn mark_complete(
        &self,
        dataset_id: &str,
        version: &str,
    ) -> Result<(), std::io::Error> {
        self.opfs_mark_complete(dataset_id, version)
            .await
            .map_err(into_io)
    }

    /// Remove a specific dataset version from the cache.
    pub async fn evict(&self, dataset_id: &str, version: &str) -> Result<(), std::io::Error> {
        self.opfs_remove(&self.snapshot_dir(dataset_id, version))
            .await
    }

    /// Remove an entire dataset from the cache.
    pub async fn evict_dataset(&self, dataset_id: &str) -> Result<(), std::io::Error> {
        self.opfs_remove(&self.dataset_dir(dataset_id)).await
    }

    /// Total cache size in bytes.
    pub async fn total_size(&self) -> u64 {
        let (Ok(root), Ok(path)) = (crate::opfs::root().await, crate::opfs::path_str(&self.root))
        else {
            return 0;
        };
        match crate::opfs::try_dir(&root, path).await {
            Some(dir) => crate::opfs::tree_size(&dir).await,
            None => 0,
        }
    }

    /// Clean the entire cache.
    pub async fn clean(&self) -> Result<(), std::io::Error> {
        self.opfs_remove(&self.root).await
    }

    async fn opfs_remove(&self, path: &Path) -> Result<(), std::io::Error> {
        let root = crate::opfs::root().await.map_err(into_io)?;
        crate::opfs::remove(&root, crate::opfs::path_str(path).map_err(into_io)?)
            .await
            .map_err(into_io)
    }

    /// `mark_complete`, with this crate's error, for `download_to_cache`.
    pub(crate) async fn opfs_mark_complete(
        &self,
        dataset_id: &str,
        version: &str,
    ) -> crate::Result<()> {
        let root = crate::opfs::root().await?;
        let snap = self.snapshot_dir(dataset_id, version);
        let snap = crate::opfs::dir(&root, crate::opfs::path_str(&snap)?, true).await?;
        crate::opfs::write(&snap, ".complete", version.as_bytes()).await?;

        // Update refs/latest
        let refs = self.dataset_dir(dataset_id).join("refs");
        let refs = crate::opfs::dir(&root, crate::opfs::path_str(&refs)?, true).await?;
        crate::opfs::write(&refs, "latest", version.as_bytes()).await?;
        crate::opfs::write(&refs, version, version.as_bytes()).await
    }
}

/// This crate's error as the `std::io::Error` the native methods return.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
fn into_io(e: crate::Error) -> std::io::Error {
    match e {
        crate::Error::Io(e) => e,
        other => std::io::Error::other(other.to_string()),
    }
}

/// Info about a cached dataset.
#[derive(Debug, Clone)]
pub struct CachedDataset {
    pub id: String,
    pub versions: Vec<String>,
    pub size_bytes: u64,
}

impl std::fmt::Display for CachedDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({} version(s), {:.1} MB)",
            self.id,
            self.versions.len(),
            self.size_bytes as f64 / 1e6
        )
    }
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = std::fs::metadata(&p) {
                total += m.len();
            }
        }
    }
    total
}

/// A blob's file name in the cache: `{sha256 hex}-{size}`. Existing caches
/// are keyed by it, so it must not change.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
fn blob_name(content: &[u8]) -> String {
    let hash: String = Sha256::digest(content)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{hash}-{}", content.len())
}

// the std::fs cache's tests; the OPFS one's are in tests/browser.rs
#[cfg(all(test, not(all(target_family = "wasm", target_os = "unknown"))))]
mod tests {
    use super::*;

    #[test]
    fn blob_name_is_sha256_hex_and_size() {
        assert_eq!(
            blob_name(b"hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824-5"
        );
        assert_eq!(
            blob_name(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855-0"
        );
    }

    #[test]
    fn test_cache_store_and_resolve() {
        let dir = std::env::temp_dir().join("bids_cache_test");
        let cache = Cache::new(dir.clone());

        // Store a file
        let path = cache
            .store_file(
                "ds001",
                "1.0.0",
                "dataset_description.json",
                b"{\"Name\": \"test\"}",
            )
            .unwrap();
        assert!(path.exists());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"Name\": \"test\"}"
        );

        // Not complete yet
        assert!(!cache.is_cached("ds001", "1.0.0"));

        // Mark complete
        cache.mark_complete("ds001", "1.0.0").unwrap();
        assert!(cache.is_cached("ds001", "1.0.0"));

        // Resolve
        let resolved = cache.resolve("ds001", Some("1.0.0")).unwrap();
        assert!(resolved.exists());

        // Resolve latest
        let latest = cache.resolve("ds001", None).unwrap();
        assert!(latest.exists());

        // List
        let datasets = cache.list_datasets();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].id, "ds001");

        // Evict
        cache.evict("ds001", "1.0.0").unwrap();
        assert!(!cache.is_cached("ds001", "1.0.0"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_blob_dedup() {
        let dir = std::env::temp_dir().join("bids_cache_dedup");
        let cache = Cache::new(dir.clone());

        // Store same content in two versions
        cache
            .store_file("ds001", "1.0.0", "file.txt", b"hello")
            .unwrap();
        cache
            .store_file("ds001", "2.0.0", "file.txt", b"hello")
            .unwrap();

        // Should only have one blob
        let blobs: Vec<_> = std::fs::read_dir(cache.blobs_dir("ds001"))
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(blobs.len(), 1);

        // Both snapshots should have the file
        assert!(
            cache
                .snapshot_dir("ds001", "1.0.0")
                .join("file.txt")
                .exists()
        );
        assert!(
            cache
                .snapshot_dir("ds001", "2.0.0")
                .join("file.txt")
                .exists()
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
