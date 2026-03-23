#![deny(unsafe_code)]
//! Butterworth IIR filter with zero-phase filtfilt.
//!
//! Implements `scipy.signal.butter` + `scipy.signal.filtfilt` for anti-alias
//! filtering before downsampling, as used by PyBIDS.

use std::f64::consts::PI;

/// Butterworth low-pass filter coefficients.
///
/// Returns `(b, a)` — numerator and denominator of the IIR transfer function,
/// matching `scipy.signal.butter(order, cutoff, btype='low', output='ba')`.
///
/// `cutoff` is normalized frequency: cutoff_hz / (fs/2), must be in (0, 1).
#[must_use]
pub fn butter_lowpass(order: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(cutoff > 0.0 && cutoff < 1.0, "cutoff must be in (0,1), got {cutoff}");

    // Step 1: Analog Butterworth poles on the unit circle (left half-plane)
    let mut poles_s: Vec<(f64, f64)> = Vec::with_capacity(order);
    for k in 0..order {
        let theta = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
        poles_s.push((theta.cos(), theta.sin()));
    }

    // Step 2: Pre-warp cutoff for bilinear transform
    let fs = 2.0; // normalized sampling rate
    let warped = 2.0 * fs * (PI * cutoff / fs).tan();

    // Step 3: Scale analog poles by warped cutoff
    let poles_a: Vec<(f64, f64)> = poles_s.iter()
        .map(|(re, im)| (re * warped, im * warped))
        .collect();

    // Step 4: Bilinear transform: s -> (2*fs*(z-1))/(z+1)
    // z-domain pole = (1 + s/(2*fs)) / (1 - s/(2*fs))
    let mut poles_z: Vec<(f64, f64)> = Vec::new();
    let c = 2.0 * fs;
    for &(re, im) in &poles_a {
        let denom_re = 1.0 - re / c;
        let denom_im = -im / c;
        let num_re = 1.0 + re / c;
        let num_im = im / c;
        let d2 = denom_re * denom_re + denom_im * denom_im;
        poles_z.push((
            (num_re * denom_re + num_im * denom_im) / d2,
            (num_im * denom_re - num_re * denom_im) / d2,
        ));
    }

    // Step 5: All zeros at z = -1 for low-pass
    let zeros_z: Vec<(f64, f64)> = vec![(-1.0, 0.0); order];

    // Step 6: Convert poles/zeros to polynomial coefficients
    let a = poly_from_roots(&poles_z);
    let b_unnorm = poly_from_roots(&zeros_z);

    // Step 7: Normalize gain at DC (z=1)
    let gain_a: f64 = a.iter().sum();
    let gain_b: f64 = b_unnorm.iter().sum();
    let gain = gain_a / gain_b;

    let b: Vec<f64> = b_unnorm.iter().map(|&x| x * gain).collect();

    (b, a)
}

/// Direct-form II transposed IIR filter (single-pass).
#[must_use]
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let nb = b.len();
    let na = a.len();
    let nfilt = nb.max(na);
    let mut y = vec![0.0; n];
    let mut d = vec![0.0; nfilt]; // delay line

    let a0 = a[0];
    for i in 0..n {
        let mut out = b.first().copied().unwrap_or(0.0) * x[i] + d[0];
        out /= a0;
        y[i] = out;
        // Shift delay line
        for j in 0..nfilt - 1 {
            d[j] = b.get(j + 1).copied().unwrap_or(0.0) * x[i]
                - a.get(j + 1).copied().unwrap_or(0.0) * out
                + d.get(j + 1).copied().unwrap_or(0.0);
        }
        if nfilt > 0 { d[nfilt - 1] = 0.0; }
    }
    y
}

/// Zero-phase filtering: apply filter forward, then backward.
///
/// Matches `scipy.signal.filtfilt(b, a, x)`.
#[must_use]
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    // Pad signal to reduce edge effects (3 * max(len(a), len(b)) samples)
    let npad = 3 * b.len().max(a.len());
    let mut padded = Vec::with_capacity(x.len() + 2 * npad);
    // Reflect-pad start
    for i in (1..=npad.min(x.len() - 1)).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    // Reflect-pad end
    let last = x[x.len() - 1];
    for i in 1..=npad.min(x.len() - 1) {
        padded.push(2.0 * last - x[x.len() - 1 - i]);
    }

    // Forward pass
    let fwd = lfilter(b, a, &padded);
    // Reverse
    let rev_input: Vec<f64> = fwd.into_iter().rev().collect();
    // Backward pass
    let bwd = lfilter(b, a, &rev_input);
    // Reverse again and extract original-length portion
    let result: Vec<f64> = bwd.into_iter().rev().collect();
    let start = npad.min(result.len());
    let end = (start + x.len()).min(result.len());
    result[start..end].to_vec()
}

/// Butterworth high-pass filter coefficients.
///
/// Returns `(b, a)` matching `scipy.signal.butter(order, cutoff, btype='high')`.
/// `cutoff` is normalized frequency in (0, 1).
#[must_use]
pub fn butter_highpass(order: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(cutoff > 0.0 && cutoff < 1.0, "cutoff must be in (0,1), got {cutoff}");
    // High-pass = transform low-pass: s → 1/s before bilinear.
    // Equivalently: zeros at z=+1 (not -1), and gain at Nyquist=1.
    let (b_lp, a_lp) = butter_lowpass(order, cutoff);
    // Spectral inversion: negate every other coefficient of b, flipping the response
    let b_hp: Vec<f64> = b_lp.iter().enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .collect();
    let a_hp: Vec<f64> = a_lp.iter().enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .collect();
    // Re-normalize: gain at Nyquist (z = -1) should be 1
    let gain_a: f64 = a_hp.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
    let gain_b: f64 = b_hp.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
    let gain = gain_a / gain_b;
    let b: Vec<f64> = b_hp.iter().map(|&v| v * gain).collect();
    (b, a_hp)
}

/// Butterworth band-pass filter coefficients.
///
/// Returns `(b, a)` matching `scipy.signal.butter(order, [low, high], btype='band')`.
/// `low` and `high` are normalized frequencies in (0, 1).
#[must_use]
pub fn butter_bandpass(order: usize, low: f64, high: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(low > 0.0 && low < 1.0 && high > 0.0 && high < 1.0 && low < high,
        "frequencies must be 0 < low < high < 1, got low={low}, high={high}");
    // Cascade: highpass at `low` then lowpass at `high`
    let (b_hp, a_hp) = butter_highpass(order, low);
    let (b_lp, a_lp) = butter_lowpass(order, high);
    // Convolve the transfer functions: B(z) = B_hp(z) * B_lp(z), A(z) = A_hp(z) * A_lp(z)
    let b = convolve(&b_hp, &b_lp);
    let a = convolve(&a_hp, &a_lp);
    (b, a)
}

/// Notch (band-stop) filter using second-order IIR sections.
///
/// Removes a narrow frequency band around `freq_hz` (e.g., 50 or 60 Hz power line noise).
/// `quality` controls the notch width (default ~30). Higher = narrower.
///
/// Like MNE's `raw.notch_filter()` or `scipy.signal.iirnotch`.
#[must_use]
pub fn notch_filter(x: &[f64], freq_hz: f64, fs: f64, quality: f64) -> Vec<f64> {
    let w0 = 2.0 * PI * freq_hz / fs;
    let bw = w0 / quality;
    let r = 1.0 - bw / 2.0; // pole radius

    // Second-order IIR notch: zeros on unit circle at ±w0, poles just inside
    let b: &[f64] = &[1.0, -2.0 * w0.cos(), 1.0];
    let a: &[f64] = &[1.0, -2.0 * r * w0.cos(), r * r];

    // Normalize gain at DC to 1
    let dc_b: f64 = b.iter().sum();
    let dc_a: f64 = a.iter().sum();
    let gain = dc_a / dc_b;
    let b_norm: Vec<f64> = b.iter().map(|&v| v * gain).collect();

    filtfilt(&b_norm, a, x)
}

/// Resample a signal from `fs_old` to `fs_new` using linear interpolation.
///
/// For anti-aliasing when downsampling, applies a lowpass filter at the new
/// Nyquist frequency before decimation (like MNE's `raw.resample()`).
#[must_use]
pub fn resample(x: &[f64], fs_old: f64, fs_new: f64) -> Vec<f64> {
    if x.is_empty() || fs_old <= 0.0 || fs_new <= 0.0 { return vec![]; }
    if (fs_old - fs_new).abs() < 1e-10 { return x.to_vec(); }

    let ratio = fs_new / fs_old;
    let n_out = (x.len() as f64 * ratio).round() as usize;
    if n_out == 0 { return vec![]; }

    // Anti-alias filter when downsampling
    let src = if fs_new < fs_old {
        let cutoff = fs_new / fs_old; // normalized Nyquist of new rate
        let cutoff = cutoff.clamp(0.01, 0.99);
        let (b, a) = butter_lowpass(8, cutoff);
        filtfilt(&b, &a, x)
    } else {
        x.to_vec()
    };

    // Linear interpolation
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t = i as f64 / ratio;
        let idx = t.floor() as usize;
        let frac = t - idx as f64;
        if idx + 1 < src.len() {
            out.push(src[idx] * (1.0 - frac) + src[idx + 1] * frac);
        } else if idx < src.len() {
            out.push(src[idx]);
        }
    }
    out
}

/// Convolve two polynomial coefficient vectors.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut out = vec![0.0; n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// Compute polynomial coefficients from complex roots.
/// Returns real coefficients [1, c1, c2, ...] for (z-r1)(z-r2)...
fn poly_from_roots(roots: &[(f64, f64)]) -> Vec<f64> {
    let mut coeffs: Vec<(f64, f64)> = vec![(1.0, 0.0)];
    for &(rr, ri) in roots {
        let mut new_coeffs = vec![(0.0, 0.0); coeffs.len() + 1];
        for (i, &(cr, ci)) in coeffs.iter().enumerate() {
            // Multiply by (z - root): shift + subtract root*current
            new_coeffs[i].0 += cr;
            new_coeffs[i].1 += ci;
            new_coeffs[i + 1].0 -= cr * rr - ci * ri;
            new_coeffs[i + 1].1 -= cr * ri + ci * rr;
        }
        coeffs = new_coeffs;
    }
    // Extract real parts (imaginary should be ~0 for conjugate pairs)
    coeffs.iter().map(|(r, _)| *r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butter_lowpass_order1() {
        let (b, a) = butter_lowpass(1, 0.5);
        // For order 1 at Nyquist/2, known result
        assert_eq!(b.len(), 2);
        assert_eq!(a.len(), 2);
        assert!((a[0] - 1.0).abs() < 1e-10);
        // DC gain should be 1.0
        let dc_gain: f64 = b.iter().sum::<f64>() / a.iter().sum::<f64>();
        assert!((dc_gain - 1.0).abs() < 1e-10, "DC gain = {}", dc_gain);
    }

    #[test]
    fn test_butter_lowpass_order5() {
        let (b, a) = butter_lowpass(5, 0.25);
        assert_eq!(b.len(), 6);
        assert_eq!(a.len(), 6);
        let dc_gain: f64 = b.iter().sum::<f64>() / a.iter().sum::<f64>();
        assert!((dc_gain - 1.0).abs() < 1e-10, "DC gain = {}", dc_gain);
    }

    #[test]
    fn test_lfilter_passthrough() {
        // With b=[1], a=[1], output = input
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = lfilter(&[1.0], &[1.0], &x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_butter_highpass() {
        let (b, a) = butter_highpass(3, 0.1);
        assert_eq!(b.len(), 4);
        assert_eq!(a.len(), 4);
        // DC gain should be ~0 (high-pass blocks DC)
        let dc_gain: f64 = b.iter().sum::<f64>() / a.iter().sum::<f64>();
        assert!(dc_gain.abs() < 0.01, "HP DC gain should be ~0, got {}", dc_gain);
        // Nyquist gain should be ~1
        let ny_gain_b: f64 = b.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
        let ny_gain_a: f64 = a.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
        let ny_gain = ny_gain_b / ny_gain_a;
        assert!((ny_gain - 1.0).abs() < 0.01, "HP Nyquist gain should be ~1, got {}", ny_gain);
    }

    #[test]
    fn test_butter_bandpass() {
        let (b, a) = butter_bandpass(2, 0.1, 0.4);
        // DC gain should be ~0
        let dc_b: f64 = b.iter().sum();
        let dc_a: f64 = a.iter().sum();
        assert!((dc_b / dc_a).abs() < 0.01, "BP DC gain should be ~0");
        // Nyquist gain should be ~0
        let ny_b: f64 = b.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
        let ny_a: f64 = a.iter().enumerate().map(|(i, &v)| v * (-1.0f64).powi(i as i32)).sum();
        assert!((ny_b / ny_a).abs() < 0.01, "BP Nyquist gain should be ~0");
    }

    #[test]
    fn test_notch_filter() {
        // Signal with 50 Hz hum + 10 Hz signal
        let fs = 500.0;
        let n = 1000;
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
        }).collect();

        let filtered = notch_filter(&signal, 50.0, fs, 30.0);
        assert_eq!(filtered.len(), n);

        // 50 Hz energy should be reduced; 10 Hz preserved
        // Compare energy in second half (after transient)
        let half = n / 2;
        let orig_energy: f64 = signal[half..].iter().map(|v| v * v).sum::<f64>();
        let filt_energy: f64 = filtered[half..].iter().map(|v| v * v).sum::<f64>();
        // Original has ~1.25 (1.0 + 0.25), filtered should have ~1.0
        assert!(filt_energy < orig_energy * 0.9,
            "Notch should reduce energy: orig={:.3}, filt={:.3}", orig_energy, filt_energy);
    }

    #[test]
    fn test_resample_downsample() {
        let fs = 1000.0;
        let n = 1000;
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 10.0 * t).sin()
        }).collect();

        let resampled = resample(&signal, fs, 250.0);
        assert_eq!(resampled.len(), 250); // 1000 * 250/1000

        // The 10 Hz signal should be preserved at 250 Hz sampling
        // Check approximate peak
        let max = resampled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 0.8, "10 Hz should be preserved, peak={:.3}", max);
    }

    #[test]
    fn test_resample_upsample() {
        let signal = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let up = resample(&signal, 100.0, 200.0);
        assert_eq!(up.len(), 10);
        // First and last should be close to original
        assert!((up[0] - 0.0).abs() < 0.1);
        assert!((up[2] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_resample_identity() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let out = resample(&signal, 256.0, 256.0);
        assert_eq!(out.len(), signal.len());
        for (a, b) in signal.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_filtfilt_removes_high_freq() {
        // Generate signal: low freq + high freq
        let n = 200;
        let fs = 100.0;
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 40.0 * t).sin()
        }).collect();

        // Low-pass at 10 Hz (cutoff = 10 / (100/2) = 0.2)
        let (b, a) = butter_lowpass(5, 0.2);
        let filtered = filtfilt(&b, &a, &signal);

        assert_eq!(filtered.len(), signal.len());
        // The 40Hz component should be attenuated significantly
        // Check that the filtered signal has lower energy than the original
        let orig_energy: f64 = signal.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let filt_energy: f64 = filtered.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!(filt_energy < orig_energy * 0.7,
            "Filtered energy {} should be much less than original {}", filt_energy, orig_energy);
    }
}
