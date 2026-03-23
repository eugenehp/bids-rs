//! Common trait for multichannel time-series data across modalities.
//!
//! Implemented by `EegData`, `MegData`, `NirsData`, and any other type that
//! holds channels × samples arrays. Enables generic processing pipelines
//! that work across EEG, MEG, NIRS, etc.

/// Trait for read-only access to multichannel time-series data.
///
/// All electrophysiology data types (`EegData`, `MegData`, `NirsData`)
/// implement this trait, enabling modality-agnostic processing.
pub trait TimeSeries {
    /// Number of channels.
    fn n_channels(&self) -> usize;
    /// Number of time samples (for channel 0; channels may differ for multi-rate).
    fn n_samples(&self) -> usize;
    /// Channel names / labels.
    fn channel_names(&self) -> &[String];
    /// Primary sampling rate in Hz.
    fn sampling_rate(&self) -> f64;
    /// Get one channel's data by index.
    fn channel_data(&self, index: usize) -> Option<&[f64]>;
    /// Total duration in seconds.
    fn duration(&self) -> f64;

    // ── Provided methods ────────────────────────────────────────────────

    /// Get one channel's data by name.
    fn channel_data_by_name(&self, name: &str) -> Option<&[f64]> {
        let idx = self.channel_names().iter().position(|n| n == name)?;
        self.channel_data(idx)
    }

    /// Time array for channel 0: `[0, 1/sr, 2/sr, ...]`.
    fn times(&self) -> Vec<f64> {
        let sr = self.sampling_rate();
        (0..self.n_samples()).map(|i| i as f64 / sr).collect()
    }

    /// Mean value per channel.
    fn channel_means(&self) -> Vec<f64> {
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 }
            })
            .collect()
    }

    /// Standard deviation per channel (using pre-computed means).
    fn channel_stds_with_means(&self, means: &[f64]) -> Vec<f64> {
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                if d.len() < 2 { return 0.0; }
                let m = means[ch];
                let var = d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64;
                var.sqrt()
            })
            .collect()
    }

    /// Standard deviation per channel.
    fn channel_stds(&self) -> Vec<f64> {
        let means = self.channel_means();
        self.channel_stds_with_means(&means)
    }

    /// Z-score normalize all channels (zero mean, unit variance).
    /// Returns a new `Vec<Vec<f64>>`.
    #[must_use]
    fn z_score(&self) -> Vec<Vec<f64>> {
        let means = self.channel_means();
        let stds = self.channel_stds_with_means(&means);
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                let m = means[ch];
                let s = if stds[ch] > f64::EPSILON { stds[ch] } else { 1.0 };
                d.iter().map(|v| (v - m) / s).collect()
            })
            .collect()
    }

    /// Min-max normalize all channels to [0, 1].
    #[must_use]
    fn min_max_normalize(&self) -> Vec<Vec<f64>> {
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                let min = d.iter().copied().fold(f64::INFINITY, f64::min);
                let max = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                let range = if range > f64::EPSILON { range } else { 1.0 };
                d.iter().map(|v| (v - min) / range).collect()
            })
            .collect()
    }

    // ── ML-oriented methods ─────────────────────────────────────────────

    /// Extract a time window as a channels × samples `Vec<Vec<f64>>`.
    ///
    /// Useful for cutting epochs from continuous data for ML training.
    /// `start_sec` and `end_sec` are in seconds.
    #[must_use]
    fn window(&self, start_sec: f64, end_sec: f64) -> Vec<Vec<f64>> {
        let sr = self.sampling_rate();
        let start = (start_sec * sr).round() as usize;
        let end = (end_sec * sr).round() as usize;
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                let s = start.min(d.len());
                let e = end.min(d.len());
                d[s..e].to_vec()
            })
            .collect()
    }

    /// Extract non-overlapping fixed-length epochs.
    ///
    /// Returns a Vec of epochs, each epoch is channels × window_samples.
    /// Drops the last partial epoch if it's shorter than `window_sec`.
    #[must_use]
    fn epochs(&self, window_sec: f64) -> Vec<Vec<Vec<f64>>> {
        self.epochs_with_stride(window_sec, window_sec)
    }

    /// Extract epochs with a given stride (allows overlap when stride < window).
    ///
    /// Returns `Vec<epoch>` where each epoch is `Vec<channel_data>`.
    #[must_use]
    fn epochs_with_stride(&self, window_sec: f64, stride_sec: f64) -> Vec<Vec<Vec<f64>>> {
        let dur = self.duration();
        if dur < window_sec { return vec![]; }
        let n = ((dur - window_sec) / stride_sec).floor() as usize + 1;
        (0..n).map(|i| {
            let start = i as f64 * stride_sec;
            self.window(start, start + window_sec)
        }).collect()
    }

    /// Flatten channels × samples into a single contiguous `Vec<f64>` (row-major).
    ///
    /// Layout: `[ch0_s0, ch0_s1, ..., ch0_sN, ch1_s0, ..., chM_sN]`.
    /// This is the format expected by most ML frameworks (batch × features).
    #[must_use]
    fn to_flat_vec(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_channels() * self.n_samples());
        for ch in 0..self.n_channels() {
            if let Some(d) = self.channel_data(ch) {
                out.extend_from_slice(d);
            }
        }
        out
    }

    /// Get data as a contiguous `Vec<f64>` in column-major order (samples × channels).
    ///
    /// Layout: `[ch0_s0, ch1_s0, ..., chM_s0, ch0_s1, ch1_s1, ..., chM_sN]`.
    /// This matches the layout expected by many time-series models (T × C).
    #[must_use]
    fn to_column_major(&self) -> Vec<f64> {
        let nc = self.n_channels();
        let ns = self.n_samples();
        let mut out = Vec::with_capacity(nc * ns);
        for s in 0..ns {
            for ch in 0..nc {
                let val = self.channel_data(ch)
                    .and_then(|d| d.get(s))
                    .copied()
                    .unwrap_or(0.0);
                out.push(val);
            }
        }
        out
    }

    /// Shape as (n_channels, n_samples) — matches tensor dimension conventions.
    #[must_use]
    fn shape(&self) -> (usize, usize) {
        (self.n_channels(), self.n_samples())
    }

    // ── Feature extraction (inspired by MOABB pipelines) ────────────────

    /// Log-variance per channel — a simple but effective BCI feature.
    ///
    /// Equivalent to MOABB's `LogVariance` transformer.
    /// Returns one value per channel: `ln(var(channel_data))`.
    #[must_use]
    fn log_variance(&self) -> Vec<f64> {
        let means = self.channel_means();
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                if d.len() < 2 { return f64::NEG_INFINITY; }
                let m = means[ch];
                let var = d.iter().map(|v| (v - m).powi(2)).sum::<f64>() / d.len() as f64;
                if var > 0.0 { var.ln() } else { f64::NEG_INFINITY }
            })
            .collect()
    }

    /// Band power per channel — average power in the signal.
    ///
    /// Returns one value per channel: `mean(x²)`.
    #[must_use]
    fn band_power(&self) -> Vec<f64> {
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                if d.is_empty() { return 0.0; }
                d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64
            })
            .collect()
    }

    /// RMS (root-mean-square) per channel.
    #[must_use]
    fn rms(&self) -> Vec<f64> {
        self.band_power().iter().map(|p| p.sqrt()).collect()
    }

    /// Peak-to-peak amplitude per channel: `max - min`.
    #[must_use]
    fn peak_to_peak(&self) -> Vec<f64> {
        (0..self.n_channels())
            .map(|ch| {
                let d = self.channel_data(ch).unwrap_or(&[]);
                if d.is_empty() { return 0.0; }
                let min = d.iter().copied().fold(f64::INFINITY, f64::min);
                let max = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                max - min
            })
            .collect()
    }

    /// Compute the covariance matrix (channels × channels).
    ///
    /// Returns a flat Vec in row-major order (length = n_channels²).
    /// Used for Riemannian geometry BCI methods (CSP, MDM, etc.).
    #[must_use]
    fn covariance_matrix(&self) -> Vec<f64> {
        let nc = self.n_channels();
        let ns = self.n_samples();
        let means = self.channel_means();
        let mut cov = vec![0.0; nc * nc];

        if ns < 2 { return cov; }

        for i in 0..nc {
            let di = self.channel_data(i).unwrap_or(&[]);
            for j in i..nc {
                let dj = self.channel_data(j).unwrap_or(&[]);
                let sum: f64 = di.iter().zip(dj.iter())
                    .map(|(a, b)| (a - means[i]) * (b - means[j]))
                    .sum();
                let val = sum / (ns - 1) as f64;
                cov[i * nc + j] = val;
                cov[j * nc + i] = val;
            }
        }
        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal TimeSeries implementation for testing.
    struct TestSeries {
        channels: Vec<Vec<f64>>,
        names: Vec<String>,
        sr: f64,
    }

    impl TimeSeries for TestSeries {
        fn n_channels(&self) -> usize { self.channels.len() }
        fn n_samples(&self) -> usize { self.channels.first().map_or(0, |v| v.len()) }
        fn channel_names(&self) -> &[String] { &self.names }
        fn sampling_rate(&self) -> f64 { self.sr }
        fn channel_data(&self, index: usize) -> Option<&[f64]> {
            self.channels.get(index).map(|v| v.as_slice())
        }
        fn duration(&self) -> f64 { self.n_samples() as f64 / self.sr }
    }

    fn make_test_series() -> TestSeries {
        TestSeries {
            channels: vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![10.0, 20.0, 30.0, 40.0, 50.0],
            ],
            names: vec!["Ch1".into(), "Ch2".into()],
            sr: 100.0,
        }
    }

    #[test]
    fn test_times() {
        let ts = make_test_series();
        let times = ts.times();
        assert_eq!(times.len(), 5);
        assert!((times[0] - 0.0).abs() < 1e-10);
        assert!((times[1] - 0.01).abs() < 1e-10);
        assert!((times[4] - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_channel_means() {
        let ts = make_test_series();
        let means = ts.channel_means();
        assert!((means[0] - 3.0).abs() < 1e-10);
        assert!((means[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_stds() {
        let ts = make_test_series();
        let stds = ts.channel_stds();
        // std of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811
        assert!((stds[0] - 1.5811388300841898).abs() < 1e-10);
        assert!((stds[1] - 15.811388300841896).abs() < 1e-10);
    }

    #[test]
    fn test_z_score() {
        let ts = make_test_series();
        let z = ts.z_score();
        assert_eq!(z.len(), 2);
        assert_eq!(z[0].len(), 5);
        // After z-score: mean ≈ 0, std ≈ 1
        let mean: f64 = z[0].iter().sum::<f64>() / z[0].len() as f64;
        assert!(mean.abs() < 1e-10, "z-score mean = {}", mean);
        let var: f64 = z[0].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (z[0].len() - 1) as f64;
        assert!((var - 1.0).abs() < 1e-10, "z-score variance = {}", var);
    }

    #[test]
    fn test_min_max_normalize() {
        let ts = make_test_series();
        let norm = ts.min_max_normalize();
        assert!((norm[0][0] - 0.0).abs() < 1e-10); // min → 0
        assert!((norm[0][4] - 1.0).abs() < 1e-10); // max → 1
        assert!((norm[0][2] - 0.5).abs() < 1e-10); // middle → 0.5
    }

    #[test]
    fn test_channel_data_by_name() {
        let ts = make_test_series();
        assert_eq!(ts.channel_data_by_name("Ch1"), Some(&[1.0, 2.0, 3.0, 4.0, 5.0][..]));
        assert_eq!(ts.channel_data_by_name("Ch2"), Some(&[10.0, 20.0, 30.0, 40.0, 50.0][..]));
        assert_eq!(ts.channel_data_by_name("Missing"), None);
    }

    #[test]
    fn test_duration() {
        let ts = make_test_series();
        assert!((ts.duration() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_window() {
        let ts = make_long_series();
        let w = ts.window(0.0, 0.5);
        assert_eq!(w.len(), 2); // 2 channels
        assert_eq!(w[0].len(), 50); // 0.5s × 100Hz
    }

    #[test]
    fn test_epochs() {
        let ts = make_long_series();
        let epochs = ts.epochs(0.5);
        // 1.0s duration / 0.5s window = 2 epochs
        assert_eq!(epochs.len(), 2);
        assert_eq!(epochs[0].len(), 2); // 2 channels per epoch
        assert_eq!(epochs[0][0].len(), 50); // 50 samples per window
    }

    #[test]
    fn test_epochs_with_stride() {
        let ts = make_long_series();
        let epochs = ts.epochs_with_stride(0.5, 0.25);
        // 1.0s, window=0.5, stride=0.25 → floor((1.0-0.5)/0.25)+1 = 3
        assert_eq!(epochs.len(), 3);
    }

    #[test]
    fn test_to_flat_vec() {
        let ts = make_test_series();
        let flat = ts.to_flat_vec();
        assert_eq!(flat.len(), 10); // 2 channels × 5 samples
        assert_eq!(&flat[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]); // ch0
        assert_eq!(&flat[5..], &[10.0, 20.0, 30.0, 40.0, 50.0]); // ch1
    }

    #[test]
    fn test_to_column_major() {
        let ts = make_test_series();
        let col = ts.to_column_major();
        assert_eq!(col.len(), 10);
        // First two elements: ch0_s0, ch1_s0
        assert_eq!(col[0], 1.0);
        assert_eq!(col[1], 10.0);
        // Next: ch0_s1, ch1_s1
        assert_eq!(col[2], 2.0);
        assert_eq!(col[3], 20.0);
    }

    #[test]
    fn test_shape() {
        let ts = make_test_series();
        assert_eq!(ts.shape(), (2, 5));
    }

    #[test]
    fn test_log_variance() {
        let ts = make_test_series();
        let lv = ts.log_variance();
        assert_eq!(lv.len(), 2);
        // var([1,2,3,4,5]) = 2.0, ln(2.0) ≈ 0.693
        assert!((lv[0] - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_band_power() {
        let ts = make_test_series();
        let bp = ts.band_power();
        assert_eq!(bp.len(), 2);
        // mean([1,4,9,16,25]) = 11.0
        assert!((bp[0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_to_peak() {
        let ts = make_test_series();
        let ptp = ts.peak_to_peak();
        assert!((ptp[0] - 4.0).abs() < 1e-10); // 5-1
        assert!((ptp[1] - 40.0).abs() < 1e-10); // 50-10
    }

    #[test]
    fn test_covariance_matrix() {
        let ts = make_test_series();
        let cov = ts.covariance_matrix();
        assert_eq!(cov.len(), 4); // 2×2
        // Diagonal should be variance with n-1 denominator
        // var([1,2,3,4,5], ddof=1) = 2.5
        assert!((cov[0] - 2.5).abs() < 1e-10);
        // cov(ch0, ch1) should be positive (both increasing)
        assert!(cov[1] > 0.0);
        // Symmetric
        assert!((cov[1] - cov[2]).abs() < 1e-10);
    }

    fn make_long_series() -> TestSeries {
        let n = 100; // 100 samples @ 100Hz = 1.0s
        TestSeries {
            channels: vec![
                (0..n).map(|i| i as f64).collect(),
                (0..n).map(|i| (i * 2) as f64).collect(),
            ],
            names: vec!["Ch1".into(), "Ch2".into()],
            sr: 100.0,
        }
    }
}
