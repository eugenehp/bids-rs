//! Command-line interface for BIDS dataset tools.
//!
//! Provides subcommands for inspecting, querying, validating, and transforming
//! BIDS datasets from the terminal. Run `bids --help` for usage information.
//!
//! # Subcommands
//!
//! - `info` — Show dataset summary (name, version, subjects, datatypes)
//! - `ls` — List files matching entity filters
//! - `eeg` — EEG-specific queries (channels, events, summary)
//! - `entities` — List all entities and their unique values
//! - `layout` — Create/load a persistent SQLite index
//! - `report` — Generate a publication-quality methods section
//! - `auto-model` — Auto-generate BIDS Stats Model JSON
//! - `upgrade` — Upgrade dataset_description.json to latest BIDS version
//! - `model-report` — Load and execute a BIDS Stats Model
//! - `dataset` — Search, download, and manage OpenNeuro datasets (search, download, ls, cache, aggregate, env)

mod cmd;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bids", about = "BIDS dataset tools", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show dataset summary
    Info {
        #[arg(default_value = ".")]
        root: PathBuf,
        #[arg(long)]
        no_validate: bool,
    },
    /// List files matching filters
    Ls {
        root: PathBuf,
        #[arg(short, long)]
        subject: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(short, long)]
        task: Option<String>,
        #[arg(long)]
        suffix: Option<String>,
        #[arg(short, long)]
        datatype: Option<String>,
        #[arg(short, long)]
        extension: Option<String>,
    },
    /// Show EEG-specific information
    Eeg {
        root: PathBuf,
        #[command(subcommand)]
        subcmd: cmd::eeg::EegCommands,
    },
    /// List entities and their values
    Entities { root: PathBuf },
    /// Create/load a persistent SQLite index
    Layout {
        root: PathBuf,
        db_path: PathBuf,
        #[arg(long)]
        reset_db: bool,
        #[arg(long)]
        no_validate: bool,
        #[arg(long)]
        no_index_metadata: bool,
    },
    /// Generate a methods section report
    Report { root: PathBuf },
    /// Auto-generate a BIDS Stats Model
    AutoModel { root: PathBuf },
    /// Upgrade BIDS dataset metadata
    Upgrade { root: PathBuf },
    /// Generate a model report from a BIDS Stats Model JSON
    ModelReport { model: PathBuf, root: PathBuf },
    /// Search, download, and manage OpenNeuro datasets
    Dataset {
        #[command(subcommand)]
        subcmd: cmd::dataset::DatasetCommands,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Info { root, no_validate } => cmd::info::run(&root, !no_validate),
        Commands::Ls {
            root,
            subject,
            session,
            task,
            suffix,
            datatype,
            extension,
        } => cmd::ls::run(&root, subject, session, task, suffix, datatype, extension),
        Commands::Eeg { root, subcmd } => cmd::eeg::run(&root, subcmd),
        Commands::Entities { root } => cmd::entities::run(&root),
        Commands::Layout {
            root,
            db_path,
            reset_db,
            no_validate,
            no_index_metadata,
        } => cmd::layout::run(&root, &db_path, reset_db, !no_validate, !no_index_metadata),
        Commands::Report { root } => cmd::report::run(&root),
        Commands::AutoModel { root } => cmd::model::auto_model(&root),
        Commands::Upgrade { root } => cmd::upgrade::run(&root),
        Commands::ModelReport { model, root } => cmd::model::model_report(&model, &root),
        Commands::Dataset { subcmd } => cmd::dataset::run(subcmd),
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
