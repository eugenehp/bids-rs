//! `bids eeg` subcommand: EEG-specific dataset inspection.

use bids_eeg::eeg_layout::EegLayout;
use bids_layout::BidsLayout;
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum EegCommands {
    Summary,
    Channels {
        #[arg(short, long)]
        subject: String,
        #[arg(short, long)]
        task: Option<String>,
    },
    Events {
        #[arg(short, long)]
        subject: String,
        #[arg(short, long)]
        task: Option<String>,
    },
}

pub fn run(root: &PathBuf, subcmd: EegCommands) -> bids_core::error::Result<()> {
    let layout = BidsLayout::new(root)?;
    let eeg = EegLayout::new(&layout);
    match subcmd {
        EegCommands::Summary => {
            println!("{}", eeg.summary()?);
        }
        EegCommands::Channels { subject, task } => {
            let mut query = layout.get().suffix("eeg").subject(&subject);
            if let Some(ref t) = task {
                query = query.task(t);
            }
            for f in &query.collect()? {
                println!("Channels for: {}", f.filename);
                if let Some(channels) = eeg.get_channels(f)? {
                    println!(
                        "  {:>4}  {:>6}  {:>8}  {:>6}",
                        "Name", "Type", "Units", "SFreq"
                    );
                    for ch in &channels {
                        println!(
                            "  {:>4}  {:>6}  {:>8}  {:>6}",
                            ch.name,
                            ch.channel_type.to_string(),
                            ch.units,
                            ch.sampling_frequency
                                .map_or("n/a".into(), |f| f.to_string())
                        );
                    }
                    println!("  Total: {} channels", channels.len());
                } else {
                    println!("  No channels file found.");
                }
                println!();
            }
        }
        EegCommands::Events { subject, task } => {
            let mut query = layout.get().suffix("eeg").subject(&subject);
            if let Some(ref t) = task {
                query = query.task(t);
            }
            for f in &query.collect()? {
                println!("Events for: {}", f.filename);
                if let Some(events) = eeg.get_events(f)? {
                    println!(
                        "  {:>8}  {:>8}  {:>15}  {:>8}",
                        "Onset", "Duration", "Trial Type", "Value"
                    );
                    for ev in &events {
                        println!(
                            "  {:>8.3}  {:>8.3}  {:>15}  {:>8}",
                            ev.onset,
                            ev.duration,
                            ev.trial_type.as_deref().unwrap_or("n/a"),
                            ev.value.as_deref().unwrap_or("n/a")
                        );
                    }
                    println!("  Total: {} events", events.len());
                } else {
                    println!("  No events file found.");
                }
                println!();
            }
        }
    }
    Ok(())
}
