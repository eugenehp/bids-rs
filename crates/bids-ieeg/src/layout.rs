//! High-level iEEG interface on top of [`BidsLayout`].

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_eeg::data::{Annotation, EegData, ReadOptions, read_eeg_data, read_brainvision_markers};
use bids_layout::BidsLayout;

pub struct IeegLayout<'a> { layout: &'a BidsLayout }

impl<'a> IeegLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self { Self { layout } }

    pub fn get_ieeg_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().suffix("ieeg").collect() }
    pub fn get_ieeg_files_for_subject(&self, sub: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("ieeg").subject(sub).collect()
    }
    pub fn get_ieeg_files_for_task(&self, task: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("ieeg").task(task).collect()
    }

    pub fn get_channels(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::Channel>>> {
        bids_core::try_read_companion(&f.companion("channels", "tsv"), bids_eeg::read_channels_tsv)
    }
    pub fn get_electrodes(&self, f: &BidsFile) -> Result<Option<Vec<super::IeegElectrode>>> {
        bids_core::try_read_companion(&f.companion("electrodes", "tsv"), super::read_ieeg_electrodes)
    }
    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::EegEvent>>> {
        bids_core::try_read_companion(&f.companion("events", "tsv"), bids_eeg::read_events_tsv)
    }
    pub fn get_coordsystem(&self, f: &BidsFile) -> Result<Option<super::IeegCoordSystem>> {
        let p = f.companion("coordsystem", "json");
        if p.exists() { Ok(Some(super::IeegCoordSystem::from_file(&p)?)) } else { Ok(None) }
    }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<super::IeegMetadata>> {
        Ok(self.layout.get_metadata(&f.path)?.deserialize_as())
    }

    /// Read the actual signal data from an iEEG data file.
    ///
    /// Supports the same formats as EEG: `.edf`, `.bdf`, `.vhdr` (BrainVision).
    /// Returns multichannel time-series data in physical units.
    pub fn read_data(&self, f: &BidsFile, opts: &ReadOptions) -> Result<EegData> {
        read_eeg_data(&f.path, opts)
    }

    /// Read signal data selecting specific channels.
    pub fn read_data_channels(&self, f: &BidsFile, channels: &[&str]) -> Result<EegData> {
        let opts = ReadOptions::new()
            .with_channels(channels.iter().map(std::string::ToString::to_string).collect());
        self.read_data(f, &opts)
    }

    /// Read a time window from an iEEG data file.
    pub fn read_data_time_range(&self, f: &BidsFile, start: f64, end: f64) -> Result<EegData> {
        let opts = ReadOptions::new().with_time_range(start, end);
        self.read_data(f, &opts)
    }

    /// Read BrainVision markers from a .vmrk file associated with the given file.
    pub fn get_brainvision_markers(&self, f: &BidsFile, sampling_rate: f64) -> Result<Vec<Annotation>> {
        let vmrk_path = f.path.with_extension("vmrk");
        if vmrk_path.exists() {
            read_brainvision_markers(&vmrk_path, sampling_rate)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_all_channels_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("channels").datatype("ieeg").extension("tsv").collect()
    }
    pub fn get_all_electrodes_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("electrodes").datatype("ieeg").extension("tsv").collect()
    }
    pub fn get_all_events_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("events").datatype("ieeg").extension("tsv").collect()
    }
    pub fn get_ieeg_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("ieeg").return_unique("subject")
    }
    pub fn get_ieeg_tasks(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("ieeg").return_unique("task")
    }

    pub fn summary(&self) -> Result<super::IeegSummary> {
        let files = self.get_ieeg_files()?;
        let subjects = self.get_ieeg_subjects()?;
        let tasks = self.get_ieeg_tasks()?;
        let sf = files.first().and_then(|f| self.get_metadata(f).ok().flatten()).map(|m| m.sampling_frequency);
        let ch = files.first().and_then(|f| self.get_channels(f).ok().flatten()).map(|c| c.len());
        Ok(super::IeegSummary { n_subjects: subjects.len(), n_recordings: files.len(), subjects, tasks, sampling_frequency: sf, channel_count: ch })
    }
}


