//! `bids dataset` subcommand: search, download, and manage OpenNeuro datasets.

use bids_dataset::{OpenNeuro, Cache, DatasetFilter, Aggregator, Split};
use clap::Subcommand;
use std::path::PathBuf;
use super::humanize_bytes;

#[derive(Subcommand)]
pub enum DatasetCommands {
    /// Search OpenNeuro for datasets
    Search {
        #[arg(short, long)]
        modality: Option<String>,
        #[arg(short, long)]
        species: Option<String>,
        #[arg(short, long)]
        keyword: Option<String>,
        #[arg(short, long, default_value = "20")]
        limit: u32,
    },
    /// Download files from OpenNeuro (like `huggingface-cli download`)
    Download {
        dataset_id: String,
        #[arg(trailing_var_arg = true)]
        filenames: Vec<String>,
        #[arg(short, long)]
        version: Option<String>,
        #[arg(long)]
        include: Option<Vec<String>>,
        #[arg(long)]
        exclude: Option<Vec<String>>,
        #[arg(long)]
        local_dir: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        force: bool,
        #[arg(short, long)]
        quiet: bool,
        #[arg(long, default_value = "8")]
        max_workers: usize,
        #[arg(long)]
        rps: Option<f64>,
        #[arg(long)]
        retries: Option<u32>,
    },
    /// List files in a remote dataset
    #[command(alias = "repo-files")]
    Ls {
        dataset_id: String,
        #[arg(short, long)]
        prefix: Option<String>,
        #[arg(long)]
        include: Option<String>,
        #[arg(short, long)]
        verbose: bool,
    },
    /// Manage the local cache
    #[command(alias = "scan-cache")]
    Cache {
        #[command(subcommand)]
        subcmd: CacheCommands,
    },
    /// Aggregate datasets and export ML-ready manifests
    Aggregate {
        #[arg(required = true)]
        datasets: Vec<String>,
        #[arg(short, long)]
        modality: Option<String>,
        #[arg(short, long)]
        extension: Option<String>,
        #[arg(short, long, default_value = "manifest.csv")]
        output: String,
        #[arg(long)]
        split_dir: Option<String>,
        #[arg(long, default_value = "0.8")]
        train_ratio: f64,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Print environment and version info
    Env,
}

#[derive(Subcommand)]
pub enum CacheCommands {
    #[command(alias = "scan")]
    List {
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(short, long)]
        verbose: bool,
    },
    Path {
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    #[command(alias = "delete")]
    Evict {
        dataset_id: String,
        #[arg(short, long)]
        version: Option<String>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    Clean {
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

fn resolve_cache(dir: Option<PathBuf>) -> Cache {
    dir.map(Cache::new).unwrap_or_default()
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    let (pat, haystack) = if !pattern.contains('/') {
        (pattern, path.rsplit('/').next().unwrap_or(path))
    } else {
        (pattern, path)
    };
    globset::GlobBuilder::new(pat)
        .literal_separator(false)
        .build()
        .ok()
        .and_then(|g| g.compile_matcher().is_match(haystack).then_some(true))
        .unwrap_or(false)
}

fn file_matches_filters(
    path: &str, specific: &[String], include: &[String], exclude: &[String],
) -> bool {
    if !specific.is_empty() {
        return specific.iter().any(|f| path == f || path.ends_with(&format!("/{f}")));
    }
    if !include.is_empty() && !include.iter().any(|p| glob_matches(p, path)) {
        return false;
    }
    if exclude.iter().any(|p| glob_matches(p, path)) {
        return false;
    }
    true
}

pub fn run(subcmd: DatasetCommands) -> bids_core::error::Result<()> {
    match subcmd {
        DatasetCommands::Search { modality, species, keyword, limit } => {
            cmd_search(modality, species, keyword, limit)
        }
        DatasetCommands::Download {
            dataset_id, filenames, version, include, exclude,
            local_dir, cache_dir, force, quiet, max_workers, rps, retries,
        } => {
            cmd_download(dataset_id, filenames, version, include, exclude,
                         local_dir, cache_dir, force, quiet, max_workers, rps, retries)
        }
        DatasetCommands::Ls { dataset_id, prefix, include, verbose } => {
            cmd_ls(dataset_id, prefix, include, verbose)
        }
        DatasetCommands::Cache { subcmd } => cmd_cache(subcmd),
        DatasetCommands::Aggregate { datasets, modality, extension, output, split_dir, train_ratio, cache_dir } => {
            cmd_aggregate(datasets, modality, extension, output, split_dir, train_ratio, cache_dir)
        }
        DatasetCommands::Env => cmd_env(),
    }
}

fn cmd_search(modality: Option<String>, species: Option<String>,
              keyword: Option<String>, limit: u32) -> bids_core::error::Result<()> {
    let on = OpenNeuro::new();
    let mut search = on.search().limit(limit);
    if let Some(ref m) = modality { search = search.modality(m); }
    if let Some(ref s) = species { search = search.species(s); }
    if let Some(ref k) = keyword { search = search.keyword(k); }
    let hits = search.execute()?;

    println!("{:<12} {:<50} {:>10} {:>12}", "ID", "Name", "Modalities", "Size");
    println!("{}", "─".repeat(86));
    for ds in &hits {
        let mods = ds.modalities.join(",");
        let size = ds.size_bytes.map(|b| format!("{:.0} MB", b as f64 / 1e6))
            .unwrap_or_else(|| "?".into());
        let name = if ds.name.len() > 48 { format!("{}…", &ds.name[..47]) } else { ds.name.clone() };
        println!("{:<12} {:<50} {:>10} {:>12}", ds.id, name, mods, size);
    }
    println!("\n{} datasets found.", hits.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_download(
    dataset_id: String, filenames: Vec<String>, version: Option<String>,
    include: Option<Vec<String>>, exclude: Option<Vec<String>>,
    local_dir: Option<PathBuf>, cache_dir: Option<PathBuf>,
    force: bool, quiet: bool, max_workers: usize,
    rps: Option<f64>, retries: Option<u32>,
) -> bids_core::error::Result<()> {
    let mut config = bids_dataset::RateLimitConfig::from_env();
    config.download_threads = max_workers;
    if let Some(r) = rps { config.s3_download_rps = r; }
    if let Some(n) = retries { config.max_retries = n; }
    let on = OpenNeuro::with_config(config);
    let cache = resolve_cache(cache_dir.clone());

    let ver = if let Some(v) = version { v } else {
        if !quiet { eprintln!("Fetching latest version..."); }
        let resp = on.graphql(&format!(
            r#"{{ dataset(id: "{dataset_id}") {{ latestSnapshot {{ tag }} }} }}"#
        ))?;
        resp["data"]["dataset"]["latestSnapshot"]["tag"]
            .as_str().unwrap_or("latest").to_string()
    };

    if !force && local_dir.is_none() && cache.is_cached(&dataset_id, &ver) {
        println!("{}", cache.snapshot_dir(&dataset_id, &ver).display());
        return Ok(());
    }
    if !force {
        if let Some(ref dir) = local_dir {
            let target = dir.join(&dataset_id);
            if target.exists() && std::fs::read_dir(&target).map(|mut d| d.next().is_some()).unwrap_or(false) {
                println!("{}", target.display());
                return Ok(());
            }
        }
    }

    if force { let _ = cache.evict(&dataset_id, &ver); }

    let include_patterns = include.unwrap_or_default();
    let exclude_patterns = exclude.unwrap_or_default();
    let specific_files: Vec<String> = filenames;

    let pb = if quiet {
        indicatif::ProgressBar::hidden()
    } else {
        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {msg}"
        ).expect("hard-coded progress template is valid"));
        pb.set_message(format!("Downloading {dataset_id} v{ver}"));
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    };

    if let Some(ref dir) = local_dir {
        let report = on.download_dataset(&dataset_id, dir, Some(move |f: &bids_dataset::RemoteFile| {
            file_matches_filters(&f.path, &specific_files, &include_patterns, &exclude_patterns)
        }))?;
        pb.finish_and_clear();
        if !quiet {
            eprintln!("{report}");
        }
        println!("{}", dir.join(&dataset_id).display());
    } else {
        let snap = on.download_to_cache(&dataset_id, &ver, &cache, Some(move |f: &bids_dataset::RemoteFile| {
            file_matches_filters(&f.path, &specific_files, &include_patterns, &exclude_patterns)
        }))?;
        pb.finish_and_clear();
        println!("{}", snap.display());
    }
    Ok(())
}

fn cmd_ls(dataset_id: String, prefix: Option<String>,
          include: Option<String>, verbose: bool) -> bids_core::error::Result<()> {
    let on = OpenNeuro::new();
    let files = on.list_files(&dataset_id, prefix.as_deref())
        ?;

    let filtered: Vec<_> = if let Some(ref pat) = include {
        files.iter().filter(|f| glob_matches(pat, &f.path)).collect()
    } else {
        files.iter().collect()
    };

    if verbose {
        for f in &filtered {
            println!("{:>12}  {}", humanize_bytes(f.size), f.path);
        }
    } else {
        for f in &filtered {
            println!("{}", f.path);
        }
    }
    let total: u64 = filtered.iter().map(|f| f.size).sum();
    eprintln!("{} files, {} total", filtered.len(), humanize_bytes(total));
    Ok(())
}

fn cmd_cache(subcmd: CacheCommands) -> bids_core::error::Result<()> {
    match subcmd {
        CacheCommands::List { cache_dir, verbose } => {
            let cache = resolve_cache(cache_dir);
            let datasets = cache.list_datasets();
            if datasets.is_empty() {
                println!("No datasets cached.");
            } else {
                println!("DATASET         VERSIONS    SIZE       PATH");
                println!("{}", "─".repeat(70));
                for ds in &datasets {
                    println!("{:<15} {:>8}    {:>8}   {}",
                        ds.id, ds.versions.len(),
                        humanize_bytes(ds.size_bytes),
                        cache.dataset_dir(&ds.id).display());
                    if verbose {
                        for ver in &ds.versions {
                            let snap = cache.snapshot_dir(&ds.id, ver);
                            let complete = if cache.is_cached(&ds.id, ver) { "✓" } else { "…" };
                            println!("  {} {} {}", complete, ver, snap.display());
                        }
                    }
                }
                println!("\nTotal: {}", humanize_bytes(cache.total_size()));
            }
            println!("Cache dir: {}", cache.root().display());
            Ok(())
        }
        CacheCommands::Path { cache_dir } => {
            println!("{}", resolve_cache(cache_dir).root().display());
            Ok(())
        }
        CacheCommands::Evict { dataset_id, version, cache_dir } => {
            let cache = resolve_cache(cache_dir);
            if let Some(ver) = version {
                cache.evict(&dataset_id, &ver)?;
                println!("Evicted {dataset_id} v{ver}");
            } else {
                cache.evict_dataset(&dataset_id)?;
                println!("Evicted {dataset_id} (all versions)");
            }
            Ok(())
        }
        CacheCommands::Clean { cache_dir, yes } => {
            let cache = resolve_cache(cache_dir);
            let size = cache.total_size();
            if size == 0 {
                println!("Cache is already empty.");
                return Ok(());
            }
            if !yes {
                eprint!("Delete {} of cached data? [y/N] ", humanize_bytes(size));
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Aborted.");
                    return Ok(());
                }
            }
            cache.clean()?;
            println!("Cleaned {}", humanize_bytes(size));
            Ok(())
        }
    }
}

fn cmd_aggregate(
    datasets: Vec<String>, modality: Option<String>, extension: Option<String>,
    output: String, split_dir: Option<String>, train_ratio: f64,
    cache_dir: Option<PathBuf>,
) -> bids_core::error::Result<()> {
    let cache = resolve_cache(cache_dir);
    let mut agg = Aggregator::new();

    for ds_id in &datasets {
        let root = cache.resolve(ds_id, None)
            .ok_or_else(|| bids_core::BidsError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("{ds_id} not found in cache. Run: bids dataset download {ds_id}"))))?;

        let mut filter = DatasetFilter::new();
        if let Some(ref m) = modality { filter = filter.modality(m); }
        if let Some(ref e) = extension { filter = filter.extension(e); }

        let count = agg.add_dataset(&root, filter)
            ?;
        eprintln!("  {ds_id} — {count} files");
    }

    agg.export_manifest(&output)
        ?;
    println!("Manifest: {} ({} files, {} subjects)", output, agg.len(), agg.subjects().len());

    if let Some(ref dir) = split_dir {
        let val_ratio = (1.0 - train_ratio) / 2.0;
        let split = Split::ratio(train_ratio, val_ratio, val_ratio);
        let report = agg.export_split(dir, split)
            ?;
        println!("Splits: {report}");
    }
    Ok(())
}

fn cmd_env() -> bids_core::error::Result<()> {
    println!("bids-rs environment:");
    println!("  bids-cli version:    {}", env!("CARGO_PKG_VERSION"));
    println!("  cache directory:     {}", Cache::default().root().display());
    println!("  rust version:        {}", super::rustc_version());
    println!("  os:                  {} {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("\nRate limit settings:");
    let cfg = bids_dataset::RateLimitConfig::from_env();
    println!("  BIDS_S3_DOWNLOAD_RPS:   {}", cfg.s3_download_rps);
    println!("  BIDS_S3_LISTING_RPS:    {}", cfg.s3_listing_rps);
    println!("  BIDS_GRAPHQL_RPS:       {}", cfg.graphql_rps);
    println!("  BIDS_MAX_RETRIES:       {}", cfg.max_retries);
    println!("  BIDS_RETRY_BASE_MS:     {}", cfg.retry_base_ms);
    println!("  BIDS_429_COOLDOWN_SECS: {}", cfg.default_429_cooldown_secs);
    println!("  BIDS_DOWNLOAD_THREADS:  {}", cfg.download_threads);
    Ok(())
}
