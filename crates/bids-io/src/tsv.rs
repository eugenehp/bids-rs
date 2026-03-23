//! TSV (Tab-Separated Values) file reading and writing for BIDS datasets.
//!
//! BIDS uses TSV files extensively for tabular data: events, channels, electrodes,
//! participants, sessions, scans, and more. This module handles reading both
//! plain `.tsv` and gzip-compressed `.tsv.gz` files, converting `n/a` values
//! to empty strings per BIDS convention.

use bids_core::error::{BidsError, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A single row from a TSV file, mapping column names to string values.
///
/// The `n/a` sentinel value used in BIDS is automatically converted to an
/// empty string during parsing.
pub type TsvRow = HashMap<String, String>;

/// Read a TSV file and return rows as a Vec of HashMaps.
///
/// The first line is treated as the header. `n/a` values are stored as empty strings.
///
/// # Errors
///
/// Returns an error if the file can't be opened or is empty.
pub fn read_tsv(path: &Path) -> Result<Vec<TsvRow>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    parse_tsv_reader(reader)
}

/// Read a gzipped TSV file (`.tsv.gz`) and return rows.
///
/// # Errors
///
/// Returns an error if the file can't be opened, decompressed, or is empty.
pub fn read_tsv_gz(path: &Path) -> Result<Vec<TsvRow>> {
    let file = std::fs::File::open(path)?;
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    parse_tsv_reader(reader)
}

/// Read a TSV file and return `(headers, rows)` where rows are `Vec<Vec<String>>`.
///
/// # Errors
///
/// Returns an error if the file can't be opened or is empty.
pub fn read_tsv_raw(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    parse_tsv_raw_reader(reader)
}

/// Read a gzipped TSV file and return `(headers, rows)`.
///
/// # Errors
///
/// Returns an error if the file can't be opened, decompressed, or is empty.
pub fn read_tsv_gz_raw(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let file = std::fs::File::open(path)?;
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    parse_tsv_raw_reader(reader)
}

fn parse_tsv_reader<R: Read>(reader: BufReader<R>) -> Result<Vec<TsvRow>> {
    let mut lines = reader.lines();

    let header_line = lines
        .next()
        .ok_or_else(|| BidsError::Csv("Empty TSV file".to_string()))??;
    let headers: Vec<String> = header_line.split('\t').map(|s| s.trim().to_string()).collect();

    let mut rows = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<&str> = line.split('\t').collect();
        let mut row = TsvRow::new();
        for (i, header) in headers.iter().enumerate() {
            let val = values.get(i).unwrap_or(&"").trim();
            let val = if val == "n/a" { "" } else { val };
            row.insert(header.clone(), val.to_string());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn parse_tsv_raw_reader<R: Read>(reader: BufReader<R>) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut lines = reader.lines();

    let header_line = lines
        .next()
        .ok_or_else(|| BidsError::Csv("Empty TSV file".to_string()))??;
    let headers: Vec<String> = header_line.split('\t').map(|s| s.trim().to_string()).collect();

    let mut rows = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<String> = line.split('\t').map(|s| {
            let v = s.trim();
            if v == "n/a" { String::new() } else { v.to_string() }
        }).collect();
        rows.push(values);
    }
    Ok((headers, rows))
}

/// Write rows to a TSV file.
///
/// # Errors
///
/// Returns an I/O error if the file can't be created or written.
pub fn write_tsv(path: &Path, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{}", headers.join("\t"))?;
    for row in rows {
        writeln!(file, "{}", row.join("\t"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_tsv() {
        let dir = std::env::temp_dir().join("bids_io_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "onset\tduration\ttrial_type").unwrap();
        writeln!(f, "1.0\t0.5\tgo").unwrap();
        writeln!(f, "2.0\tn/a\tstop").unwrap();

        let rows = read_tsv(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["onset"], "1.0");
        assert_eq!(rows[0]["trial_type"], "go");
        assert_eq!(rows[1]["duration"], ""); // n/a -> empty
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
