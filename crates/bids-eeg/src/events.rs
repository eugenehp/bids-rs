//! EEG events from `_events.tsv`.
//!
//! Provides [`EegEvent`] with onset, duration, trial type, value, sample
//! number, and response time, plus [`read_events_tsv()`] for parsing.

use bids_core::error::{BidsError, Result};
use bids_io::tsv::read_tsv;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// An event from a BIDS _events.tsv file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EegEvent {
    /// Onset time in seconds from the beginning of the recording.
    pub onset: f64,
    /// Duration of the event in seconds.
    pub duration: f64,
    /// Type/category of the event.
    pub trial_type: Option<String>,
    /// Numeric value (e.g., trigger code).
    pub value: Option<String>,
    /// Sample number (0-indexed).
    pub sample: Option<i64>,
    /// Response time in seconds.
    pub response_time: Option<f64>,
    /// Any additional columns.
    pub extra: HashMap<String, String>,
}

/// Read events from a BIDS _events.tsv file.
pub fn read_events_tsv(path: &Path) -> Result<Vec<EegEvent>> {
    let rows = read_tsv(path)?;
    let mut events = Vec::with_capacity(rows.len());

    let known_cols = ["onset", "duration", "trial_type", "value", "sample", "response_time"];

    for row in &rows {
        let onset: f64 = row.get("onset")
            .ok_or_else(|| BidsError::Csv("Missing 'onset' column in events.tsv".into()))?
            .parse()
            .map_err(|_| BidsError::Csv("Invalid onset value".into()))?;

        let duration: f64 = row.get("duration")
            .and_then(|s| if s.is_empty() { None } else { s.parse().ok() })
            .unwrap_or(0.0);

        let trial_type = row.get("trial_type")
            .filter(|s| !s.is_empty())
            .cloned();

        let value = row.get("value")
            .filter(|s| !s.is_empty())
            .cloned();

        let sample = row.get("sample")
            .and_then(|s| s.parse().ok());

        let response_time = row.get("response_time")
            .and_then(|s| s.parse().ok());

        let extra: HashMap<String, String> = row.iter()
            .filter(|(k, v)| !known_cols.contains(&k.as_str()) && !v.is_empty())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        events.push(EegEvent {
            onset,
            duration,
            trial_type,
            value,
            sample,
            response_time,
            extra,
        });
    }

    Ok(events)
}

/// Get unique trial types from events.
pub fn unique_trial_types(events: &[EegEvent]) -> Vec<String> {
    let mut types: Vec<String> = events.iter()
        .filter_map(|e| e.trial_type.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    types.sort();
    types
}

/// Filter events by trial type.
pub fn filter_by_trial_type<'a>(events: &'a [EegEvent], trial_type: &str) -> Vec<&'a EegEvent> {
    events.iter()
        .filter(|e| e.trial_type.as_deref() == Some(trial_type))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_events_tsv() {
        let dir = std::env::temp_dir().join("bids_eeg_events_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "onset\tduration\ttrial_type\tvalue\tsample").unwrap();
        writeln!(f, "0.5\t0.0\tstimulus\t1\t128").unwrap();
        writeln!(f, "1.5\t0.0\tresponse\t2\t384").unwrap();
        writeln!(f, "3.0\t0.5\tstimulus\t1\t768").unwrap();

        let events = read_events_tsv(&path).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].onset, 0.5);
        assert_eq!(events[0].trial_type.as_deref(), Some("stimulus"));
        assert_eq!(events[0].sample, Some(128));
        assert_eq!(events[1].trial_type.as_deref(), Some("response"));

        let types = unique_trial_types(&events);
        assert_eq!(types, vec!["response", "stimulus"]);

        let stim_events = filter_by_trial_type(&events, "stimulus");
        assert_eq!(stim_events.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
