//! OpenNeuro dataset catalog — search, list, and download BIDS datasets.
//!
//! Uses the OpenNeuro GraphQL API for searching and the public S3 bucket
//! (`s3://openneuro.org`) for listing and downloading files.

use crate::http;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata about an OpenNeuro dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub modalities: Vec<String>,
    pub species: Option<String>,
    pub study_domain: Option<String>,
    pub latest_version: Option<String>,
    pub size_bytes: Option<u64>,
    pub doi: Option<String>,
}

/// A file listed from the OpenNeuro S3 bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteFile {
    /// Path relative to dataset root (e.g., "sub-01/eeg/sub-01_task-rest_eeg.edf").
    pub path: String,
    /// File size in bytes.
    pub size: u64,
}

/// Client for searching and downloading from OpenNeuro.
///
/// Rate limits are configurable via [`RateLimitConfig`] or environment variables:
///
/// | Env var | Default | Description |
/// |---|---|---|
/// | `BIDS_S3_DOWNLOAD_RPS` | 50 | S3 download requests/second |
/// | `BIDS_S3_LISTING_RPS` | 10 | S3 listing requests/second |
/// | `BIDS_GRAPHQL_RPS` | 5 | GraphQL requests/second |
/// | `BIDS_MAX_RETRIES` | 3 | Retry attempts per request |
/// | `BIDS_429_COOLDOWN_SECS` | 30 | Default 429 cooldown |
/// | `BIDS_DOWNLOAD_THREADS` | 8 | Parallel download threads |
///
/// [`RateLimitConfig`]: crate::ratelimit::RateLimitConfig
pub struct OpenNeuro {
    agent: ureq::Agent,
    graphql_endpoint: String,
    s3_bucket: String,
    graphql_limiter: crate::ratelimit::RateLimiter,
    s3_listing_limiter: crate::ratelimit::RateLimiter,
    s3_download_limiter: crate::ratelimit::RateLimiter,
    config: crate::ratelimit::RateLimitConfig,
}

impl OpenNeuro {
    /// Create with default config (reads from env vars, falls back to safe defaults).
    pub fn new() -> Self {
        Self::with_config(crate::ratelimit::RateLimitConfig::from_env())
    }

    /// Create with a custom rate limit configuration.
    ///
    /// ```
    /// use bids_dataset::{OpenNeuro, ratelimit::RateLimitConfig};
    ///
    /// let on = OpenNeuro::with_config(
    ///     RateLimitConfig::from_env()
    ///         .s3_download_rps(20.0)  // slower downloads
    ///         .download_threads(4)    // fewer threads
    ///         .max_retries(5)         // more retries
    /// );
    /// ```
    pub fn with_config(config: crate::ratelimit::RateLimitConfig) -> Self {
        Self {
            agent: http::make_agent(),
            graphql_endpoint: "https://openneuro.org/crn/graphql".to_string(),
            s3_bucket: "https://s3.amazonaws.com/openneuro.org".to_string(),
            graphql_limiter: config.graphql_limiter(),
            s3_listing_limiter: config.s3_listing_limiter(),
            s3_download_limiter: config.s3_download_limiter(),
            config,
        }
    }

    /// Search the OpenNeuro catalog.
    pub fn search(&self) -> SearchBuilder<'_> {
        SearchBuilder::new(self)
    }

    /// List all files in a dataset on S3.
    ///
    /// Optionally filter to a subtree with `prefix` (e.g. `Some("sub-01/eeg")`).
    pub fn list_files(
        &self,
        dataset_id: &str,
        prefix: Option<&str>,
    ) -> crate::Result<Vec<RemoteFile>> {
        let mut all = Vec::new();
        let mut marker = String::new();
        let base = match prefix {
            Some(p) => format!("{}/{}/", dataset_id, p.trim_end_matches('/')),
            None => format!("{dataset_id}/"),
        };

        loop {
            let mut url = format!("{}?prefix={}&max-keys=1000", self.s3_bucket, base);
            if !marker.is_empty() {
                url.push_str(&format!("&marker={marker}"));
            }

            let mut resp = http::get_with_retry_limited(
                &self.agent,
                &url,
                self.config.max_retries,
                Some(&self.s3_listing_limiter),
            )?;

            let body = resp
                .body_mut()
                .read_to_string()
                .map_err(|e| crate::Error::Network(e.to_string()))?;

            let (files, truncated, next) = parse_s3_listing(&body, dataset_id)?;
            all.extend(files);

            if truncated && !next.is_empty() {
                marker = next;
            } else {
                break;
            }
        }

        Ok(all)
    }

    /// Download a single file from S3 to a local path.
    ///
    /// Skips if the file already exists with the expected size.
    pub fn download_file(
        &self,
        dataset_id: &str,
        remote_path: &str,
        local_dir: &Path,
    ) -> crate::Result<std::path::PathBuf> {
        let local_path = local_dir.join(dataset_id).join(remote_path);
        let url = format!("{}/{}/{}", self.s3_bucket, dataset_id, remote_path);

        if local_path.exists() {
            return Ok(local_path);
        }
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let resp = http::get_with_retry_limited(
            &self.agent,
            &url,
            self.config.max_retries,
            Some(&self.s3_download_limiter),
        )?;

        let mut file = std::fs::File::create(&local_path)?;
        std::io::copy(&mut resp.into_body().into_reader(), &mut file)?;

        Ok(local_path)
    }

    /// Download a full dataset (or filtered subset) to a local directory.
    ///
    /// - `filter`: optional closure — return `true` for files to download.
    /// - Existing files with matching size are skipped (resume support).
    /// - Uses 8 parallel download threads for throughput.
    pub fn download_dataset<F>(
        &self,
        dataset_id: &str,
        local_dir: &Path,
        filter: Option<F>,
    ) -> crate::Result<DownloadReport>
    where
        F: Fn(&RemoteFile) -> bool,
    {
        let files = self.list_files(dataset_id, None)?;
        let targets: Vec<RemoteFile> = match &filter {
            Some(f) => files.into_iter().filter(|rf| f(rf)).collect(),
            None => files,
        };

        // Separate into already-cached and need-download
        let mut to_download = Vec::new();
        let mut skipped = 0usize;
        for rf in &targets {
            let local = local_dir.join(dataset_id).join(&rf.path);
            if local.exists() {
                if let Ok(m) = std::fs::metadata(&local) {
                    if m.len() == rf.size {
                        skipped += 1;
                        continue;
                    }
                }
            }
            to_download.push(rf.clone());
        }

        // Parallel download with N threads
        let n_threads = self.config.download_threads.min(to_download.len().max(1));
        let downloaded = std::sync::atomic::AtomicUsize::new(0);
        let downloaded_bytes = std::sync::atomic::AtomicU64::new(0);
        let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let rate_limiter = &self.s3_download_limiter;
        let retries = self.config.max_retries;

        let chunk_size = to_download.len().div_ceil(n_threads);
        std::thread::scope(|scope| {
            for chunk in to_download.chunks(chunk_size) {
                let agent = http::make_agent();
                let s3 = &self.s3_bucket;
                let dl = &downloaded;
                let dl_bytes = &downloaded_bytes;
                let errs = &errors;
                let ds_id = dataset_id;
                let dir = local_dir;
                let rl = rate_limiter;

                scope.spawn(move || {
                    for rf in chunk {
                        let url = format!("{}/{}/{}", s3, ds_id, rf.path);
                        let local_path = dir.join(ds_id).join(&rf.path);
                        if let Some(parent) = local_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        match http::get_with_retry_limited(&agent, &url, retries, Some(rl)) {
                            Ok(resp) => {
                                match std::fs::File::create(&local_path) {
                                    Ok(mut file) => {
                                        match std::io::copy(
                                            &mut resp.into_body().into_reader(),
                                            &mut file,
                                        ) {
                                            Ok(_) => {
                                                dl.fetch_add(
                                                    1,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );
                                                dl_bytes.fetch_add(
                                                    rf.size,
                                                    std::sync::atomic::Ordering::Relaxed,
                                                );
                                            }
                                            Err(_) => {
                                                // Remove partial file to prevent stale cache hits
                                                drop(file);
                                                let _ = std::fs::remove_file(&local_path);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        errs.lock().unwrap().push(format!("{}: {}", rf.path, e));
                                    }
                                }
                            }
                            Err(e) => {
                                errs.lock().unwrap().push(format!("{}: {}", rf.path, e));
                            }
                        }
                    }
                });
            }
        });

        Ok(DownloadReport {
            dataset_id: dataset_id.into(),
            local_root: local_dir.join(dataset_id),
            total_files: targets.len(),
            downloaded: downloaded.load(std::sync::atomic::Ordering::Relaxed),
            skipped,
            total_bytes: targets.iter().map(|f| f.size).sum(),
            downloaded_bytes: downloaded_bytes.load(std::sync::atomic::Ordering::Relaxed),
            errors: errors.into_inner().unwrap(),
        })
    }

    /// Download a dataset into the HuggingFace-style local cache.
    ///
    /// Returns the path to the cached snapshot directory. If the dataset is
    /// already fully cached, returns immediately.
    ///
    /// Uses content-addressed blob storage for deduplication across versions.
    /// Downloads are parallelized across N threads (configured by
    /// `BIDS_DOWNLOAD_THREADS`, default 8).
    pub fn download_to_cache<F>(
        &self,
        dataset_id: &str,
        version: &str,
        cache: &crate::Cache,
        filter: Option<F>,
    ) -> crate::Result<std::path::PathBuf>
    where
        F: Fn(&RemoteFile) -> bool + Send,
    {
        // Check if already cached
        if cache.is_cached(dataset_id, version) {
            return Ok(cache.snapshot_dir(dataset_id, version));
        }

        let files = self.list_files(dataset_id, None)?;
        let targets: Vec<RemoteFile> = match &filter {
            Some(f) => files.into_iter().filter(|rf| f(rf)).collect(),
            None => files,
        };

        // Filter out already-cached files with correct size
        let to_download: Vec<&RemoteFile> = targets
            .iter()
            .filter(|rf| {
                let snap_path = cache.snapshot_dir(dataset_id, version).join(&rf.path);
                if snap_path.exists() {
                    if let Ok(meta) = std::fs::metadata(&snap_path) {
                        if meta.len() == rf.size {
                            return false;
                        }
                        // Size mismatch — re-download
                        let _ = std::fs::remove_file(&snap_path);
                    }
                }
                true
            })
            .collect();

        // Parallel download with N threads
        let n_threads = self.config.download_threads.min(to_download.len().max(1));
        let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let rate_limiter = &self.s3_download_limiter;
        let retries = self.config.max_retries;

        let tmp_dir = cache.root().join("tmp");
        std::fs::create_dir_all(&tmp_dir).map_err(crate::Error::Io)?;

        let chunk_size = (to_download.len() + n_threads - 1) / n_threads.max(1);
        std::thread::scope(|scope| {
            for chunk in to_download.chunks(chunk_size) {
                let agent = http::make_agent();
                let s3 = &self.s3_bucket;
                let errs = &errors;
                let ds_id = dataset_id;
                let ver = version;
                let rl = rate_limiter;
                let tmp = &tmp_dir;

                scope.spawn(move || {
                    for rf in chunk {
                        let url = format!("{}/{}/{}", s3, ds_id, rf.path);
                        let tmp_file = tmp.join(format!("{}-{}", ds_id, rf.path.replace('/', "_")));

                        if let Some(parent) = tmp_file.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }

                        match http::get_with_retry_limited(&agent, &url, retries, Some(rl)) {
                            Ok(resp) => {
                                match std::fs::File::create(&tmp_file) {
                                    Ok(mut file) => {
                                        let copy_ok = std::io::copy(
                                            &mut resp.into_body().into_reader(),
                                            &mut file,
                                        )
                                        .is_ok();
                                        drop(file); // close before store/remove
                                        if copy_ok {
                                            if let Err(e) = cache.store_file_from_disk(
                                                ds_id, ver, &rf.path, &tmp_file,
                                            ) {
                                                errs.lock()
                                                    .unwrap()
                                                    .push(format!("{}: {}", rf.path, e));
                                            }
                                        }
                                        let _ = std::fs::remove_file(&tmp_file);
                                    }
                                    Err(e) => {
                                        errs.lock().unwrap().push(format!("{}: {}", rf.path, e));
                                    }
                                }
                            }
                            Err(e) => {
                                errs.lock().unwrap().push(format!("{}: {}", rf.path, e));
                            }
                        }
                    }
                });
            }
        });

        let errs = errors.into_inner().unwrap();
        if !errs.is_empty() {
            return Err(crate::Error::Network(format!(
                "{} download errors: {}",
                errs.len(),
                errs[0]
            )));
        }

        // Mark complete
        cache
            .mark_complete(dataset_id, version)
            .map_err(crate::Error::Io)?;

        Ok(cache.snapshot_dir(dataset_id, version))
    }

    /// Raw GraphQL query (rate-limited to 5 req/s).
    pub fn graphql(&self, query: &str) -> crate::Result<serde_json::Value> {
        let body = serde_json::json!({ "query": query });
        let json = http::post_json_limited(
            &self.agent,
            &self.graphql_endpoint,
            &body,
            Some(&self.graphql_limiter),
        )?;

        if let Some(errors) = json["errors"].as_array() {
            if !errors.is_empty() {
                let msg = errors[0]["message"].as_str().unwrap_or("Unknown");
                return Err(crate::Error::Api(msg.into()));
            }
        }
        Ok(json)
    }
}

impl Default for OpenNeuro {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from a download operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadReport {
    pub dataset_id: String,
    pub local_root: std::path::PathBuf,
    pub total_files: usize,
    pub downloaded: usize,
    pub skipped: usize,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub errors: Vec<String>,
}

impl std::fmt::Display for DownloadReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}/{} files ({} skipped, {:.1} MB)",
            self.dataset_id,
            self.downloaded,
            self.total_files,
            self.skipped,
            self.downloaded_bytes as f64 / 1e6
        )?;
        if !self.errors.is_empty() {
            write!(f, ", {} errors", self.errors.len())?;
        }
        Ok(())
    }
}

/// Builder for searching the OpenNeuro catalog.
pub struct SearchBuilder<'a> {
    client: &'a OpenNeuro,
    modality: Option<String>,
    species: Option<String>,
    keyword: Option<String>,
    limit: u32,
}

impl<'a> SearchBuilder<'a> {
    fn new(client: &'a OpenNeuro) -> Self {
        Self {
            client,
            modality: None,
            species: None,
            keyword: None,
            limit: 100,
        }
    }

    #[must_use]
    pub fn modality(mut self, m: &str) -> Self {
        self.modality = Some(m.into());
        self
    }
    #[must_use]
    pub fn species(mut self, s: &str) -> Self {
        self.species = Some(s.into());
        self
    }
    #[must_use]
    pub fn keyword(mut self, k: &str) -> Self {
        self.keyword = Some(k.into());
        self
    }
    #[must_use]
    pub fn limit(mut self, n: u32) -> Self {
        self.limit = n;
        self
    }

    pub fn execute(self) -> crate::Result<Vec<DatasetInfo>> {
        // Server-side modality filter (the only filter the API supports)
        let modality_arg = self
            .modality
            .as_deref()
            .map(|m| format!(r#", modality: "{m}""#))
            .unwrap_or_default();

        let q = format!(
            r#"{{ datasets(first: {}{}, orderBy: {{created: descending}}) {{ edges {{ node {{ id name metadata {{ species studyDomain modalities openneuroPaperDOI }} latestSnapshot {{ tag size }} }} }} }} }}"#,
            self.limit, modality_arg
        );

        let resp = self.client.graphql(&q)?;
        let edges = resp["data"]["datasets"]["edges"]
            .as_array()
            .ok_or_else(|| crate::Error::Api("No datasets".into()))?;

        let mut out = Vec::new();
        for edge in edges {
            let n = &edge["node"];
            let md = &n["metadata"];
            let sn = &n["latestSnapshot"];

            let modalities: Vec<String> = md["modalities"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let info = DatasetInfo {
                id: n["id"].as_str().unwrap_or("").into(),
                name: n["name"].as_str().unwrap_or("").into(),
                modalities,
                species: md["species"].as_str().map(Into::into),
                study_domain: md["studyDomain"]
                    .as_str()
                    .filter(|s| !s.is_empty() && *s != "#")
                    .map(Into::into),
                latest_version: sn["tag"].as_str().map(Into::into),
                size_bytes: sn["size"].as_u64(),
                doi: md["openneuroPaperDOI"]
                    .as_str()
                    .filter(|s| !s.is_empty() && *s != "#")
                    .map(Into::into),
            };

            if let Some(ref m) = self.modality {
                if !info.modalities.iter().any(|mm| mm.eq_ignore_ascii_case(m)) {
                    continue;
                }
            }
            if let Some(ref s) = self.species {
                if !info
                    .species
                    .as_ref()
                    .is_some_and(|sp| sp.eq_ignore_ascii_case(s))
                {
                    continue;
                }
            }
            // Client-side keyword filter (API doesn't support text search)
            if let Some(ref kw) = self.keyword {
                let kw_lower = kw.to_lowercase();
                let name_match = info.name.to_lowercase().contains(&kw_lower);
                let domain_match = info
                    .study_domain
                    .as_ref()
                    .is_some_and(|d| d.to_lowercase().contains(&kw_lower));
                if !name_match && !domain_match {
                    continue;
                }
            }

            out.push(info);
        }
        Ok(out)
    }
}

// ─── S3 listing parser ─────────────────────────────────────────────────────────
//
// This is a minimal XML parser for the S3 ListBucketResult format.
// It only handles the simple tag structure returned by S3 and does NOT
// support CDATA, attributes, namespaces, or other XML features.
// If the S3 response format ever changes significantly, consider
// switching to `quick-xml`.

fn parse_s3_listing(
    xml: &str,
    dataset_id: &str,
) -> Result<(Vec<RemoteFile>, bool, String), crate::Error> {
    let mut files = Vec::new();
    let prefix = format!("{dataset_id}/");
    let truncated = xml_tag_value(xml, "IsTruncated").as_deref() == Some("true");
    let mut last_key = String::new();

    let mut pos = 0;
    while let Some(start) = xml[pos..].find("<Contents>") {
        let abs = pos + start;
        if let Some(end) = xml[abs..].find("</Contents>") {
            let entry = &xml[abs..abs + end + 11];
            let key = xml_tag_value(entry, "Key").unwrap_or_default();
            let size: u64 = xml_tag_value(entry, "Size")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            if key.starts_with(&prefix) && size > 0 {
                files.push(RemoteFile {
                    path: key[prefix.len()..].into(),
                    size,
                });
            }
            if !key.is_empty() {
                last_key = key;
            }
            pos = abs + end + 11;
        } else {
            break;
        }
    }
    Ok((files, truncated, last_key))
}

fn xml_tag_value(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let s = xml.find(&open)? + open.len();
    let e = xml[s..].find(&close)? + s;
    Some(xml[s..e].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_listing_parser() {
        let xml = r#"<ListBucketResult>
<IsTruncated>false</IsTruncated>
<Contents><Key>ds001/dataset_description.json</Key><Size>1102</Size></Contents>
<Contents><Key>ds001/sub-01/anat/sub-01_T1w.nii.gz</Key><Size>5000000</Size></Contents>
<Contents><Key>ds001/sub-01/</Key><Size>0</Size></Contents>
</ListBucketResult>"#;
        let (files, trunc, _) = parse_s3_listing(xml, "ds001").unwrap();
        assert!(!trunc);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "dataset_description.json");
        assert_eq!(files[1].path, "sub-01/anat/sub-01_T1w.nii.gz");
        assert_eq!(files[1].size, 5_000_000);
    }

    #[test]
    fn test_dataset_info_serde() {
        let info = DatasetInfo {
            id: "ds000117".into(),
            name: "Test".into(),
            modalities: vec!["eeg".into()],
            species: Some("Human".into()),
            study_domain: None,
            latest_version: Some("1.0.0".into()),
            size_bytes: Some(1000),
            doi: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: DatasetInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "ds000117");
    }
}
