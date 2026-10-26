//! Writing BIDS-EEG recordings.
//!
//! The read side of this crate understands EDF, BDF and BrainVision; this
//! module is the write side for the formats a raw BIDS-EEG dataset needs:
//!
//! - [`BrainVisionWriter`] — streams samples into a BrainVision triplet
//!   (`_eeg.vhdr` / `_eeg.vmrk` / `_eeg.eeg`, `IEEE_FLOAT_32`, multiplexed).
//!   Samples go straight to disk, so recordings of any length are written in
//!   constant memory.
//! - [`write_eeg_sidecar()`] — `_eeg.json` from an [`EegMetadata`].
//! - [`write_channels_tsv()`] / [`write_events_tsv()`] — the TSV companions.
//! - [`PhysioWriter`] — headerless `_physio.tsv.gz` plus its JSON sidecar.
//!
//! Everything written here reads back through [`read_brainvision()`],
//! [`read_channels_tsv()`], [`read_events_tsv()`] and
//! [`EegLayout::get_physio()`].
//!
//! [`read_brainvision()`]: crate::read_brainvision
//! [`read_channels_tsv()`]: crate::read_channels_tsv
//! [`read_events_tsv()`]: crate::read_events_tsv
//! [`EegLayout::get_physio()`]: crate::EegLayout::get_physio

use crate::channels::Channel;
use crate::events::EegEvent;
use crate::metadata::EegMetadata;
use bids_core::error::{BidsError, Result};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Paths and sample count of a finished BrainVision recording.
#[derive(Debug, Clone)]
pub struct BrainVisionFiles {
    pub vhdr: PathBuf,
    pub vmrk: PathBuf,
    pub eeg: PathBuf,
    /// Samples written per channel.
    pub n_samples: u64,
}

#[derive(Debug, Clone)]
struct Marker {
    /// 0-based sample index.
    sample: u64,
    /// Length in samples (BrainVision `size`; 1 for instantaneous).
    size: u64,
    description: String,
}

/// Streams a multi-channel recording into BrainVision format.
///
/// ```no_run
/// use bids_eeg::{Channel, ChannelType};
/// use bids_eeg::writer::BrainVisionWriter;
///
/// let ch = |n: &str| Channel {
///     name: n.into(), channel_type: ChannelType::EEG, units: "µV".into(),
///     description: None, sampling_frequency: None, low_cutoff: None,
///     high_cutoff: None, notch: None, reference: None, status: None,
///     status_description: None,
/// };
/// let mut w = BrainVisionWriter::create(
///     "sub-01_task-rest_eeg.vhdr".as_ref(), &[ch("Fz"), ch("Cz")], 256.0,
/// ).unwrap();
/// w.write_frame(&[1.0, 2.0]).unwrap();
/// w.add_marker(0.0, 0.0, "start");
/// let files = w.finish().unwrap();
/// assert_eq!(files.n_samples, 1);
/// ```
pub struct BrainVisionWriter {
    data: BufWriter<std::fs::File>,
    vhdr: PathBuf,
    vmrk: PathBuf,
    eeg: PathBuf,
    channels: Vec<(String, String)>,
    sampling_frequency: f64,
    n_samples: u64,
    markers: Vec<Marker>,
    recording_start: Option<f64>,
}

impl BrainVisionWriter {
    /// Create the data file next to `vhdr_path` and start a recording.
    ///
    /// `vhdr_path` must end in `.vhdr`; the `.eeg` and `.vmrk` files share its
    /// stem. The header and marker files are written by [`finish()`](Self::finish).
    ///
    /// # Errors
    ///
    /// Returns an error if the path does not end in `.vhdr`, there are no
    /// channels, the sampling frequency is not positive, or the data file
    /// cannot be created.
    pub fn create(vhdr_path: &Path, channels: &[Channel], sampling_frequency: f64) -> Result<Self> {
        if vhdr_path.extension().and_then(|e| e.to_str()) != Some("vhdr") {
            return Err(invalid(format!(
                "BrainVision header path must end in .vhdr: {}",
                vhdr_path.display()
            )));
        }
        if channels.is_empty() {
            return Err(invalid("BrainVision recording needs at least one channel"));
        }
        if !(sampling_frequency.is_finite() && sampling_frequency > 0.0) {
            return Err(invalid(format!(
                "invalid sampling frequency {sampling_frequency}"
            )));
        }
        if let Some(parent) = vhdr_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let eeg = vhdr_path.with_extension("eeg");
        let vmrk = vhdr_path.with_extension("vmrk");
        let data = BufWriter::with_capacity(1 << 20, std::fs::File::create(&eeg)?);
        Ok(Self {
            data,
            vhdr: vhdr_path.to_path_buf(),
            vmrk,
            eeg,
            channels: channels
                .iter()
                .map(|c| (c.name.clone(), c.units.clone()))
                .collect(),
            sampling_frequency,
            n_samples: 0,
            markers: Vec::new(),
            recording_start: None,
        })
    }

    /// Record the wall-clock start (Unix seconds) in the `New Segment` marker.
    pub fn set_recording_start(&mut self, unix_secs: f64) {
        self.recording_start = Some(unix_secs);
    }

    /// Number of channels per frame.
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Samples written per channel so far.
    #[must_use]
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }

    /// Write one sample for every channel, in channel order.
    ///
    /// # Errors
    ///
    /// Returns an error if `frame` has the wrong length or the write fails.
    pub fn write_frame(&mut self, frame: &[f32]) -> Result<()> {
        if frame.len() != self.channels.len() {
            return Err(invalid(format!(
                "frame has {} values, recording has {} channels",
                frame.len(),
                self.channels.len()
            )));
        }
        for v in frame {
            self.data.write_all(&v.to_le_bytes())?;
        }
        self.n_samples += 1;
        Ok(())
    }

    /// Write many frames at once from a multiplexed buffer
    /// (`[ch0_s0, ch1_s0, …, ch0_s1, …]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a whole number of frames or the
    /// write fails.
    pub fn write_frames(&mut self, interleaved: &[f32]) -> Result<()> {
        let n_ch = self.channels.len();
        if interleaved.len() % n_ch != 0 {
            return Err(invalid(format!(
                "buffer of {} values is not a multiple of {n_ch} channels",
                interleaved.len()
            )));
        }
        for v in interleaved {
            self.data.write_all(&v.to_le_bytes())?;
        }
        self.n_samples += (interleaved.len() / n_ch) as u64;
        Ok(())
    }

    /// Add a `Comment` marker at `onset_s` seconds from the first sample.
    ///
    /// Markers are clamped to the recording when the file is finished, and
    /// dropped if they fall entirely outside it.
    pub fn add_marker(&mut self, onset_s: f64, duration_s: f64, description: &str) {
        let sample = (onset_s.max(0.0) * self.sampling_frequency).round() as u64;
        let size = ((duration_s.max(0.0) * self.sampling_frequency).round() as u64).max(1);
        self.markers.push(Marker {
            sample,
            size,
            description: description.to_string(),
        });
    }

    /// Flush the data file and write the `.vhdr` and `.vmrk` files.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be written.
    pub fn finish(mut self) -> Result<BrainVisionFiles> {
        self.data.flush()?;
        drop(self.data);

        let file_name = |p: &Path| -> String {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let eeg_name = file_name(&self.eeg);
        let vmrk_name = file_name(&self.vmrk);

        let mut vhdr = String::new();
        vhdr.push_str("Brain Vision Data Exchange Header File Version 1.0\r\n");
        vhdr.push_str("; Data written by bids-eeg\r\n\r\n");
        vhdr.push_str("[Common Infos]\r\nCodepage=UTF-8\r\n");
        vhdr.push_str(&format!(
            "DataFile={eeg_name}\r\nMarkerFile={vmrk_name}\r\n"
        ));
        vhdr.push_str("DataFormat=BINARY\r\nDataOrientation=MULTIPLEXED\r\n");
        vhdr.push_str(&format!("NumberOfChannels={}\r\n", self.channels.len()));
        vhdr.push_str("; Sampling interval in microseconds\r\n");
        vhdr.push_str(&format!(
            "SamplingInterval={}\r\n\r\n",
            1_000_000.0 / self.sampling_frequency
        ));
        vhdr.push_str("[Binary Infos]\r\nBinaryFormat=IEEE_FLOAT_32\r\n\r\n");
        vhdr.push_str("[Channel Infos]\r\n");
        vhdr.push_str("; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,\r\n");
        vhdr.push_str("; <Resolution in \"Unit\">,<Unit>\r\n");
        for (i, (name, unit)) in self.channels.iter().enumerate() {
            vhdr.push_str(&format!(
                "Ch{}={},,1,{}\r\n",
                i + 1,
                escape_bv(name),
                escape_bv(unit)
            ));
        }
        std::fs::write(&self.vhdr, vhdr)?;

        let mut vmrk = String::new();
        vmrk.push_str("Brain Vision Data Exchange Marker File, Version 1.0\r\n\r\n");
        vmrk.push_str(&format!(
            "[Common Infos]\r\nCodepage=UTF-8\r\nDataFile={eeg_name}\r\n\r\n"
        ));
        vmrk.push_str("[Marker Infos]\r\n");
        vmrk.push_str(
            "; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\r\n",
        );
        vmrk.push_str(
            "; <Size in data points>,<Channel number (0 = marker is related to all channels)>\r\n",
        );
        let segment_date = self
            .recording_start
            .map(|t| format!(",{}", brainvision_timestamp(t)))
            .unwrap_or_default();
        vmrk.push_str(&format!("Mk1=New Segment,,1,1,0{segment_date}\r\n"));
        let mut markers = self.markers;
        markers.sort_by_key(|m| m.sample);
        let mut k = 2;
        for m in markers {
            if self.n_samples == 0 || m.sample >= self.n_samples {
                continue;
            }
            let size = m.size.min(self.n_samples - m.sample).max(1);
            vmrk.push_str(&format!(
                "Mk{k}=Comment,{},{},{size},0\r\n",
                escape_bv(&m.description),
                m.sample + 1
            ));
            k += 1;
        }
        std::fs::write(&self.vmrk, vmrk)?;

        Ok(BrainVisionFiles {
            vhdr: self.vhdr,
            vmrk: self.vmrk,
            eeg: self.eeg,
            n_samples: self.n_samples,
        })
    }
}

/// Write an `_eeg.json` sidecar.
///
/// `None` fields are omitted rather than written as `null`. `extra` entries
/// (e.g. `"SoftwareVersions"`) are merged in and win over typed fields.
///
/// # Errors
///
/// Returns an error if serialization or the write fails.
pub fn write_eeg_sidecar(
    path: &Path,
    metadata: &EegMetadata,
    extra: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<()> {
    let mut value = serde_json::to_value(metadata)?;
    if let serde_json::Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
        if let Some(extra) = extra {
            for (k, v) in extra {
                map.insert(k.clone(), v.clone());
            }
        }
    }
    write_json(path, &value)
}

/// Write a `_channels.tsv` file.
///
/// Columns: `name`, `type`, `units`, then any optional column that at least
/// one channel fills in (`description`, `sampling_frequency`, `low_cutoff`,
/// `high_cutoff`, `notch`, `reference`, `status`, `status_description`).
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn write_channels_tsv(path: &Path, channels: &[Channel]) -> Result<()> {
    type Col = (&'static str, fn(&Channel) -> Option<String>);
    let optional: [Col; 8] = [
        ("description", |c| c.description.clone()),
        ("sampling_frequency", |c| c.sampling_frequency.map(fmt_num)),
        ("low_cutoff", |c| c.low_cutoff.map(fmt_num)),
        ("high_cutoff", |c| c.high_cutoff.map(fmt_num)),
        ("notch", |c| c.notch.map(fmt_num)),
        ("reference", |c| c.reference.clone()),
        ("status", |c| c.status.clone()),
        ("status_description", |c| c.status_description.clone()),
    ];
    let used: Vec<&Col> = optional
        .iter()
        .filter(|(_, get)| channels.iter().any(|c| get(c).is_some()))
        .collect();

    let mut headers = vec!["name", "type", "units"];
    headers.extend(used.iter().map(|(h, _)| *h));
    let rows: Vec<Vec<String>> = channels
        .iter()
        .map(|c| {
            let mut row = vec![
                tsv_cell(Some(&c.name)),
                tsv_cell(Some(&c.channel_type.to_string())),
                tsv_cell(Some(&c.units)),
            ];
            row.extend(used.iter().map(|(_, get)| tsv_cell(get(c).as_deref())));
            row
        })
        .collect();
    ensure_parent(path)?;
    bids_io::tsv::write_tsv(path, &headers, &rows)
}

/// Write an `_events.tsv` file.
///
/// Columns: `onset`, `duration`, then `trial_type` / `value` / `sample` /
/// `response_time` when any event sets them, then every `extra` key (sorted).
///
/// # Errors
///
/// Returns an error if the write fails.
pub fn write_events_tsv(path: &Path, events: &[EegEvent]) -> Result<()> {
    let has_trial = events.iter().any(|e| e.trial_type.is_some());
    let has_value = events.iter().any(|e| e.value.is_some());
    let has_sample = events.iter().any(|e| e.sample.is_some());
    let has_rt = events.iter().any(|e| e.response_time.is_some());
    let mut extra_keys: Vec<&String> = events.iter().flat_map(|e| e.extra.keys()).collect();
    extra_keys.sort();
    extra_keys.dedup();

    let mut headers: Vec<&str> = vec!["onset", "duration"];
    if has_trial {
        headers.push("trial_type");
    }
    if has_value {
        headers.push("value");
    }
    if has_sample {
        headers.push("sample");
    }
    if has_rt {
        headers.push("response_time");
    }
    headers.extend(extra_keys.iter().map(|k| k.as_str()));

    let rows: Vec<Vec<String>> = events
        .iter()
        .map(|e| {
            let mut row = vec![fmt_num(e.onset), fmt_num(e.duration)];
            if has_trial {
                row.push(tsv_cell(e.trial_type.as_deref()));
            }
            if has_value {
                row.push(tsv_cell(e.value.as_deref()));
            }
            if has_sample {
                row.push(tsv_cell(e.sample.map(|s| s.to_string()).as_deref()));
            }
            if has_rt {
                row.push(tsv_cell(e.response_time.map(fmt_num).as_deref()));
            }
            for k in &extra_keys {
                row.push(tsv_cell(e.extra.get(*k).map(String::as_str)));
            }
            row
        })
        .collect();
    ensure_parent(path)?;
    bids_io::tsv::write_tsv(path, &headers, &rows)
}

/// Streams a continuous recording into `_physio.tsv.gz` + `_physio.json`.
///
/// BIDS physio files carry no header row; column names, sampling frequency
/// and the offset from the parent recording live in the JSON sidecar, which
/// [`finish()`](Self::finish) writes next to the data file.
pub struct PhysioWriter {
    out: flate2::write::GzEncoder<BufWriter<std::fs::File>>,
    path: PathBuf,
    columns: Vec<String>,
    sampling_frequency: f64,
    n_rows: u64,
    line: String,
}

impl PhysioWriter {
    /// Create a writer for `path`, which must end in `_physio.tsv.gz`.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is wrong, there are no columns, the
    /// sampling frequency is not positive, or the file cannot be created.
    pub fn create(path: &Path, columns: &[&str], sampling_frequency: f64) -> Result<Self> {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".tsv.gz") {
            return Err(invalid(format!(
                "physio path must end in .tsv.gz: {}",
                path.display()
            )));
        }
        if columns.is_empty() {
            return Err(invalid("physio recording needs at least one column"));
        }
        if !(sampling_frequency.is_finite() && sampling_frequency > 0.0) {
            return Err(invalid(format!(
                "invalid sampling frequency {sampling_frequency}"
            )));
        }
        ensure_parent(path)?;
        let file = BufWriter::new(std::fs::File::create(path)?);
        Ok(Self {
            out: flate2::write::GzEncoder::new(file, flate2::Compression::default()),
            path: path.to_path_buf(),
            columns: columns.iter().map(|c| (*c).to_string()).collect(),
            sampling_frequency,
            n_rows: 0,
            line: String::new(),
        })
    }

    /// Rows written so far.
    #[must_use]
    pub fn n_rows(&self) -> u64 {
        self.n_rows
    }

    /// Write one row. `None` becomes `n/a`.
    ///
    /// # Errors
    ///
    /// Returns an error if the row has the wrong length or the write fails.
    pub fn write_row(&mut self, row: &[Option<f64>]) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(invalid(format!(
                "row has {} values, physio has {} columns",
                row.len(),
                self.columns.len()
            )));
        }
        self.line.clear();
        for (i, v) in row.iter().enumerate() {
            if i > 0 {
                self.line.push('\t');
            }
            match v {
                Some(x) if x.is_finite() => self.line.push_str(&x.to_string()),
                _ => self.line.push_str("n/a"),
            }
        }
        self.line.push('\n');
        self.out.write_all(self.line.as_bytes())?;
        self.n_rows += 1;
        Ok(())
    }

    /// Finish the gzip stream and write the JSON sidecar.
    ///
    /// `start_time` is the offset in seconds of the first row relative to the
    /// parent recording (BIDS `StartTime`). `extra` is merged into the sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if either file cannot be written.
    pub fn finish(
        self,
        start_time: f64,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<PathBuf> {
        let mut file = self.out.finish()?;
        file.flush()?;
        let mut map = serde_json::Map::new();
        map.insert(
            "SamplingFrequency".into(),
            serde_json::json!(self.sampling_frequency),
        );
        map.insert("StartTime".into(), serde_json::json!(start_time));
        map.insert("Columns".into(), serde_json::json!(self.columns));
        if let Some(extra) = extra {
            for (k, v) in extra {
                map.insert(k.clone(), v.clone());
            }
        }
        let json_path = sidecar_for_tsv_gz(&self.path);
        write_json(&json_path, &serde_json::Value::Object(map))?;
        Ok(json_path)
    }
}

/// `x_physio.tsv.gz` → `x_physio.json`.
#[must_use]
pub fn sidecar_for_tsv_gz(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name.strip_suffix(".tsv.gz").unwrap_or(&name);
    path.with_file_name(format!("{stem}.json"))
}

/// Format Unix seconds as the 20-digit BrainVision marker timestamp
/// (`YYYYMMDDhhmmssuuuuuu`, UTC).
#[must_use]
pub fn brainvision_timestamp(unix_secs: f64) -> String {
    let micros_total = (unix_secs * 1_000_000.0).round() as i64;
    let secs = micros_total.div_euclid(1_000_000);
    let micros = micros_total.rem_euclid(1_000_000);
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}{mo:02}{d:02}{h:02}{mi:02}{s:02}{micros:06}")
}

/// Format Unix seconds as an ISO 8601 UTC timestamp with milliseconds
/// (`YYYY-MM-DDThh:mm:ss.sss`), the form BIDS uses for `acq_time`.
#[must_use]
pub fn iso8601_utc(unix_secs: f64) -> String {
    let millis_total = (unix_secs * 1000.0).round() as i64;
    let secs = millis_total.div_euclid(1000);
    let millis = millis_total.rem_euclid(1000);
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}")
}

/// Unix seconds → (year, month, day, hour, minute, second) in UTC.
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Howard Hinnant's days-to-civil.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (
        y,
        m,
        d,
        (sod / 3600) as u32,
        ((sod % 3600) / 60) as u32,
        (sod % 60) as u32,
    )
}

/// BrainVision fields are comma-separated; the spec escapes commas as `\1`.
fn escape_bv(s: &str) -> String {
    s.replace(['\r', '\n'], " ").replace(',', "\\1")
}

/// A TSV cell: tabs/newlines flattened to spaces, empty → `n/a`.
fn tsv_cell(s: Option<&str>) -> String {
    match s {
        Some(v) if !v.trim().is_empty() => v.replace(['\t', '\r', '\n'], " "),
        _ => "n/a".to_string(),
    }
}

fn fmt_num(x: f64) -> String {
    if x.is_finite() {
        x.to_string()
    } else {
        "n/a".to_string()
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    ensure_parent(path)?;
    let mut s = serde_json::to_string_pretty(value)?;
    s.push('\n');
    std::fs::write(path, s)?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn invalid(msg: impl Into<String>) -> BidsError {
    BidsError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        msg.into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ChannelType;
    use crate::data::{ReadOptions, read_brainvision};

    fn ch(name: &str) -> Channel {
        Channel {
            name: name.into(),
            channel_type: ChannelType::EEG,
            units: "µV".into(),
            description: None,
            sampling_frequency: None,
            low_cutoff: None,
            high_cutoff: None,
            notch: None,
            reference: None,
            status: Some("good".into()),
            status_description: None,
        }
    }

    #[test]
    fn brainvision_round_trip() {
        let dir = tempdir("bv_round_trip");
        let vhdr = dir.join("sub-01_task-rest_eeg.vhdr");
        let channels = [ch("TP9"), ch("AF7"), ch("AF8"), ch("TP10")];
        let mut w = BrainVisionWriter::create(&vhdr, &channels, 256.0).unwrap();
        w.set_recording_start(1_793_008_760.905_163);
        for i in 0..512 {
            let x = i as f32;
            w.write_frame(&[x, -x, x * 0.5, 1000.25]).unwrap();
        }
        w.write_frames(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        w.add_marker(1.0, 0.5, "eyes, closed");
        w.add_marker(99.0, 0.0, "past the end");
        let files = w.finish().unwrap();
        assert_eq!(files.n_samples, 514);

        let data = read_brainvision(&vhdr, &ReadOptions::default()).unwrap();
        assert_eq!(data.channel_labels, ["TP9", "AF7", "AF8", "TP10"]);
        assert!((data.sampling_rates[0] - 256.0).abs() < 1e-9);
        assert_eq!(data.data[0].len(), 514);
        assert_eq!(data.data[1][10], -10.0);
        assert_eq!(data.data[2][11], 5.5);
        assert_eq!(data.data[3][0], 1000.25);
        assert_eq!(data.data[3][513], 8.0);

        let comments: Vec<_> = data
            .annotations
            .iter()
            .filter(|a| a.description != "New Segment/")
            .collect();
        assert_eq!(comments.len(), 1, "{:?}", data.annotations);
        assert!((comments[0].onset - 1.0).abs() < 1e-9);
        assert!((comments[0].duration - 0.5).abs() < 1e-9);

        let vmrk = std::fs::read_to_string(&files.vmrk).unwrap();
        assert!(vmrk.contains("eyes\\1 closed"));
        assert!(vmrk.contains("Mk1=New Segment,,1,1,0,20261026095920905163"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_bad_input() {
        let dir = tempdir("bv_rejects");
        assert!(BrainVisionWriter::create(&dir.join("x.eeg"), &[ch("A")], 256.0).is_err());
        assert!(BrainVisionWriter::create(&dir.join("x.vhdr"), &[], 256.0).is_err());
        assert!(BrainVisionWriter::create(&dir.join("x.vhdr"), &[ch("A")], 0.0).is_err());
        let mut w =
            BrainVisionWriter::create(&dir.join("x.vhdr"), &[ch("A"), ch("B")], 10.0).unwrap();
        assert!(w.write_frame(&[1.0]).is_err());
        assert!(w.write_frames(&[1.0, 2.0, 3.0]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn channels_and_events_round_trip() {
        let dir = tempdir("bv_tsv");
        let chans = dir.join("sub-01_task-rest_channels.tsv");
        let mut aux = ch("AUX");
        aux.channel_type = ChannelType::MISC;
        aux.status = None;
        write_channels_tsv(&chans, &[ch("Fz"), aux]).unwrap();
        let back = crate::read_channels_tsv(&chans).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].channel_type, ChannelType::MISC);
        assert_eq!(back[0].units, "µV");
        assert_eq!(back[0].status.as_deref(), Some("good"));

        let events_path = dir.join("sub-01_task-rest_events.tsv");
        let mut extra = std::collections::HashMap::new();
        extra.insert("context".to_string(), "tab\there".to_string());
        let events = [
            EegEvent {
                onset: 1.5,
                duration: 2.0,
                trial_type: Some("focus".into()),
                value: None,
                sample: Some(384),
                response_time: None,
                extra,
            },
            EegEvent {
                onset: 4.0,
                duration: 0.0,
                trial_type: None,
                value: None,
                sample: None,
                response_time: None,
                extra: Default::default(),
            },
        ];
        write_events_tsv(&events_path, &events).unwrap();
        let back = crate::read_events_tsv(&events_path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].trial_type.as_deref(), Some("focus"));
        assert_eq!(back[0].sample, Some(384));
        assert_eq!(
            back[0].extra.get("context").map(String::as_str),
            Some("tab here")
        );
        assert!((back[1].onset - 4.0).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn eeg_sidecar_omits_nulls() {
        let dir = tempdir("bv_sidecar");
        let path = dir.join("sub-01_task-rest_eeg.json");
        let md: EegMetadata = serde_json::from_str(
            r#"{"TaskName":"rest","SamplingFrequency":256,"EEGReference":"Fpz"}"#,
        )
        .unwrap();
        let mut extra = serde_json::Map::new();
        extra.insert("PowerLineFrequency".into(), serde_json::json!("n/a"));
        write_eeg_sidecar(&path, &md, Some(&extra)).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.values().all(|v| !v.is_null()), "{obj:?}");
        assert_eq!(obj["EEGReference"], "Fpz");
        assert_eq!(obj["PowerLineFrequency"], "n/a");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn physio_is_headerless_with_sidecar() {
        let dir = tempdir("bv_physio");
        let path = dir.join("sub-01_task-rest_recording-ppg_physio.tsv.gz");
        let mut w = PhysioWriter::create(&path, &["red", "infrared"], 64.0).unwrap();
        w.write_row(&[Some(1.0), Some(2.0)]).unwrap();
        w.write_row(&[None, Some(f64::NAN)]).unwrap();
        assert!(w.write_row(&[Some(1.0)]).is_err());
        let json = w.finish(0.25, None).unwrap();
        assert_eq!(json, dir.join("sub-01_task-rest_recording-ppg_physio.json"));

        let mut text = String::new();
        use std::io::Read;
        flate2::read::GzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(text, "1\t2\nn/a\tn/a\n");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(v["Columns"], serde_json::json!(["red", "infrared"]));
        assert_eq!(v["StartTime"], 0.25);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn timestamps() {
        assert_eq!(brainvision_timestamp(0.0), "19700101000000000000");
        assert_eq!(iso8601_utc(1_772_323_200.5), "2026-03-01T00:00:00.500");
        assert_eq!(iso8601_utc(951_782_400.0), "2000-02-29T00:00:00.000");
    }

    fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bids_eeg_writer_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
