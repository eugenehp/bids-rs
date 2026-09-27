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
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0,1), got {cutoff}"
    );

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
    let poles_a: Vec<(f64, f64)> = poles_s
        .iter()
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
        if nfilt > 0 {
            d[nfilt - 1] = 0.0;
        }
    }
    y
}

/// Steady-state initial filter state, matching `scipy.signal.lfilter_zi`.
///
/// Returns the delay-line contents that hold a filter's output constant when
/// its input is constant 1. Scaling this by the first input sample and using it
/// as `lfilter`'s starting state is what stops the filter from ringing as it
/// charges up from rest — which is exactly what [`filtfilt`] needs at both
/// ends of the signal.
///
/// Solves `(I - companion(a)ᵀ) · zi = b[1..] - a[1..]·b[0]`, as scipy does.
///
/// # Panics
///
/// Panics if `a` is empty or `a[0]` is zero, i.e. the filter is not realizable.
#[must_use]
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    assert!(!a.is_empty(), "denominator `a` must not be empty");
    let a0 = a[0];
    assert!(a0 != 0.0, "a[0] must be non-zero");

    let n = a.len().max(b.len());
    if n < 2 {
        // A pure gain has no state to carry.
        return Vec::new();
    }

    // Normalise by a[0] and zero-pad both to length n, as scipy does.
    let mut an = vec![0.0; n];
    let mut bn = vec![0.0; n];
    for (i, slot) in an.iter_mut().enumerate() {
        *slot = a.get(i).copied().unwrap_or(0.0) / a0;
    }
    for (i, slot) in bn.iter_mut().enumerate() {
        *slot = b.get(i).copied().unwrap_or(0.0) / a0;
    }

    // m = I - companion(a)ᵀ. companion(a)ᵀ has -a[1..] down its first column
    // and ones on its superdiagonal.
    let k = n - 1;
    let mut m = vec![vec![0.0; k]; k];
    for (i, row) in m.iter_mut().enumerate() {
        row[0] += an[i + 1];
        if i + 1 < k {
            row[i + 1] -= 1.0;
        }
        row[i] += 1.0;
    }

    let rhs: Vec<f64> = (0..k).map(|i| bn[i + 1] - an[i + 1] * bn[0]).collect();
    solve_linear(m, rhs)
}

/// Gaussian elimination with partial pivoting.
///
/// Sized by filter order — a 5th-order Butterworth gives a 5x5 — so the cubic
/// cost is irrelevant and a dependency on a linear-algebra crate is not worth
/// it here.
fn solve_linear(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for col in 0..n {
        // Pivot on the largest magnitude available, for numerical stability.
        let pivot = (col..n)
            .max_by(|&i, &j| {
                m[i][col]
                    .abs()
                    .partial_cmp(&m[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        m.swap(col, pivot);
        rhs.swap(col, pivot);

        let diag = m[col][col];
        if diag == 0.0 {
            continue; // Singular; leave this variable at whatever falls out.
        }
        for row in (col + 1)..n {
            let factor = m[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            // `col < row` always, so these two rows are disjoint.
            let (above, below) = m.split_at_mut(row);
            for (dst, src) in below[0][col..].iter_mut().zip(&above[col][col..]) {
                *dst -= factor * src;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut out = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for c in (row + 1)..n {
            acc -= m[row][c] * out[c];
        }
        out[row] = if m[row][row] == 0.0 {
            0.0
        } else {
            acc / m[row][row]
        };
    }
    out
}

/// `lfilter` started from a given delay-line state rather than from rest.
fn lfilter_from(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> Vec<f64> {
    let nfilt = b.len().max(a.len());
    let mut y = vec![0.0; x.len()];
    let mut d = vec![0.0; nfilt];
    for (slot, v) in d.iter_mut().zip(zi) {
        *slot = *v;
    }

    let a0 = a[0];
    for i in 0..x.len() {
        let out = (b.first().copied().unwrap_or(0.0) * x[i] + d[0]) / a0;
        y[i] = out;
        for j in 0..nfilt - 1 {
            d[j] = b.get(j + 1).copied().unwrap_or(0.0) * x[i]
                - a.get(j + 1).copied().unwrap_or(0.0) * out
                + d.get(j + 1).copied().unwrap_or(0.0);
        }
        d[nfilt - 1] = 0.0;
    }
    y
}

/// Zero-phase filtering: apply filter forward, then backward.
///
/// Matches `scipy.signal.filtfilt(b, a, x)` with its default `method="pad"`:
/// an odd extension of `3 * max(len(a), len(b))` samples at each end, and both
/// passes started from [`lfilter_zi`] scaled by the first sample they see.
///
/// That `zi` initialisation is not cosmetic. Without it each pass starts from
/// rest and charges up over the first few samples, which for a 5th-order
/// Butterworth leaves the output wrong by ~1.4e-2 near the edges — far above
/// anything attributable to rounding, and not removed by the padding, since
/// the transient is re-introduced at the start of the backward pass too.
#[must_use]
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    // scipy requires len(x) > padlen; we clamp instead so short signals still
    // filter rather than panicking. `npad` is what actually got applied, and is
    // what the final slice has to use.
    let npad = (3 * b.len().max(a.len())).min(x.len() - 1);

    let mut padded = Vec::with_capacity(x.len() + 2 * npad);
    for i in (1..=npad).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    let last = x[x.len() - 1];
    for i in 1..=npad {
        padded.push(2.0 * last - x[x.len() - 1 - i]);
    }

    let zi = lfilter_zi(b, a);

    // Forward pass, started from the steady state for padded[0].
    let zi_fwd: Vec<f64> = zi.iter().map(|z| z * padded[0]).collect();
    let fwd = lfilter_from(b, a, &padded, &zi_fwd);

    // Backward pass over the reversed result, started from its own first sample.
    let rev_input: Vec<f64> = fwd.into_iter().rev().collect();
    let zi_bwd: Vec<f64> = zi.iter().map(|z| z * rev_input[0]).collect();
    let bwd = lfilter_from(b, a, &rev_input, &zi_bwd);

    let result: Vec<f64> = bwd.into_iter().rev().collect();
    result[npad..npad + x.len()].to_vec()
}

/// Butterworth high-pass filter coefficients.
///
/// Returns `(b, a)` matching `scipy.signal.butter(order, cutoff, btype='high')`.
/// Design a Butterworth high-pass filter.
///
/// `cutoff` is normalised to the Nyquist frequency, in `(0, 1)`. Returns the
/// `(b, a)` coefficients, matching `scipy.signal.butter(order, cutoff,
/// btype='high')`.
///
/// # Panics
///
/// Panics if `cutoff` is not in `(0, 1)` or `order` is zero.
///
/// # Example
///
/// ```
/// let (b, a) = bids_filter::butter_highpass(2, 0.2);
/// assert_eq!(b.len(), 3);
/// assert!((a[0] - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn butter_highpass(order: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0,1), got {cutoff}"
    );
    assert!(order > 0, "order must be positive");

    // Analog prototype poles, pre-warped cutoff, then the low-pass-to-high-pass
    // substitution s -> wc/s: each pole maps to wc/p and the filter gains
    // `order` zeros at the origin. Reflecting the low-pass coefficients instead
    // — "spectral inversion" — yields a high-pass at 1 - cutoff, not cutoff.
    let poles_s = prototype_poles(order);
    let warped = prewarp(cutoff);

    let poles_a: Vec<Complex> = poles_s
        .iter()
        .map(|p| Complex::new(warped, 0.0).div(*p))
        .collect();
    let zeros_a: Vec<Complex> = vec![Complex::new(0.0, 0.0); order];

    // The prototype has unit gain; s -> wc/s scales it by prod(-p).
    let k_analog = poles_s
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, p| acc.mul(p.scale(-1.0)))
        .re;

    let (zeros_z, poles_z) = bilinear(&zeros_a, &poles_a, order);
    let gain = k_analog * bilinear_gain(&zeros_a, &poles_a);
    let a = poly_from_roots_c(&poles_z);
    let b: Vec<f64> = poly_from_roots_c(&zeros_z)
        .iter()
        .map(|v| v * gain)
        .collect();
    (b, a)
}

/// Butterworth band-pass filter coefficients.
///
/// Returns `(b, a)` matching `scipy.signal.butter(order, [low, high], btype='band')`.
/// Design a Butterworth band-pass filter.
///
/// `low` and `high` are normalised to the Nyquist frequency, with
/// `0 < low < high < 1`. The result has order `2 * order`. Returns the
/// `(b, a)` coefficients, matching `scipy.signal.butter(order, [low, high],
/// btype='band')`.
///
/// # Panics
///
/// Panics unless `0 < low < high < 1` and `order` is positive.
///
/// # Example
///
/// ```
/// let (b, a) = bids_filter::butter_bandpass(2, 0.1, 0.4);
/// assert_eq!(b.len(), 5);
/// assert_eq!(a.len(), 5);
/// ```
#[must_use]
pub fn butter_bandpass(order: usize, low: f64, high: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        low > 0.0 && low < 1.0 && high > 0.0 && high < 1.0 && low < high,
        "frequencies must be 0 < low < high < 1, got low={low}, high={high}"
    );
    assert!(order > 0, "order must be positive");

    // Low-pass-to-band-pass: each prototype pole splits into a conjugate pair
    // straddling the geometric centre frequency. Cascading a separate
    // high-pass and low-pass produces a filter of the same order but different
    // poles, so it does not agree with SciPy.
    let poles_s = prototype_poles(order);
    let wl = prewarp(low);
    let wh = prewarp(high);
    let bw = wh - wl;
    let w0 = (wl * wh).sqrt();

    let mut poles_a: Vec<Complex> = Vec::with_capacity(2 * order);
    for p in &poles_s {
        let half = p.scale(bw / 2.0);
        let disc = half.mul(half).sub(Complex::new(w0 * w0, 0.0)).sqrt();
        poles_a.push(half.add(disc));
        poles_a.push(half.sub(disc));
    }
    // `order` zeros at the origin; the other `order` sit at infinity and the
    // bilinear transform places them at z = -1.
    let zeros_a: Vec<Complex> = vec![Complex::new(0.0, 0.0); order];

    // s -> (s^2 + w0^2)/(s·bw) scales the prototype's unit gain by bw^order.
    let k_analog = bw.powi(order as i32);

    let (zeros_z, poles_z) = bilinear(&zeros_a, &poles_a, 2 * order);
    let gain = k_analog * bilinear_gain(&zeros_a, &poles_a);
    let a = poly_from_roots_c(&poles_z);
    let b: Vec<f64> = poly_from_roots_c(&zeros_z)
        .iter()
        .map(|v| v * gain)
        .collect();
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
    if x.is_empty() || fs_old <= 0.0 || fs_new <= 0.0 {
        return vec![];
    }
    if (fs_old - fs_new).abs() < 1e-10 {
        return x.to_vec();
    }

    let ratio = fs_new / fs_old;
    let n_out = (x.len() as f64 * ratio).round() as usize;
    if n_out == 0 {
        return vec![];
    }

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
        assert!(
            dc_gain.abs() < 0.01,
            "HP DC gain should be ~0, got {}",
            dc_gain
        );
        // Nyquist gain should be ~1
        let ny_gain_b: f64 = b
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0f64).powi(i as i32))
            .sum();
        let ny_gain_a: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0f64).powi(i as i32))
            .sum();
        let ny_gain = ny_gain_b / ny_gain_a;
        assert!(
            (ny_gain - 1.0).abs() < 0.01,
            "HP Nyquist gain should be ~1, got {}",
            ny_gain
        );
    }

    #[test]
    fn test_butter_bandpass() {
        let (b, a) = butter_bandpass(2, 0.1, 0.4);
        // DC gain should be ~0
        let dc_b: f64 = b.iter().sum();
        let dc_a: f64 = a.iter().sum();
        assert!((dc_b / dc_a).abs() < 0.01, "BP DC gain should be ~0");
        // Nyquist gain should be ~0
        let ny_b: f64 = b
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0f64).powi(i as i32))
            .sum();
        let ny_a: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0f64).powi(i as i32))
            .sum();
        assert!((ny_b / ny_a).abs() < 0.01, "BP Nyquist gain should be ~0");
    }

    #[test]
    fn test_notch_filter() {
        // Signal with 50 Hz hum + 10 Hz signal
        let fs = 500.0;
        let n = 1000;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let filtered = notch_filter(&signal, 50.0, fs, 30.0);
        assert_eq!(filtered.len(), n);

        // 50 Hz energy should be reduced; 10 Hz preserved
        // Compare energy in second half (after transient)
        let half = n / 2;
        let orig_energy: f64 = signal[half..].iter().map(|v| v * v).sum::<f64>();
        let filt_energy: f64 = filtered[half..].iter().map(|v| v * v).sum::<f64>();
        // Original has ~1.25 (1.0 + 0.25), filtered should have ~1.0
        assert!(
            filt_energy < orig_energy * 0.9,
            "Notch should reduce energy: orig={:.3}, filt={:.3}",
            orig_energy,
            filt_energy
        );
    }

    #[test]
    fn test_resample_downsample() {
        let fs = 1000.0;
        let n = 1000;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

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
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 40.0 * t).sin()
            })
            .collect();

        // Low-pass at 10 Hz (cutoff = 10 / (100/2) = 0.2)
        let (b, a) = butter_lowpass(5, 0.2);
        let filtered = filtfilt(&b, &a, &signal);

        assert_eq!(filtered.len(), signal.len());
        // The 40Hz component should be attenuated significantly
        // Check that the filtered signal has lower energy than the original
        let orig_energy: f64 = signal.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let filt_energy: f64 = filtered.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!(
            filt_energy < orig_energy * 0.7,
            "Filtered energy {} should be much less than original {}",
            filt_energy,
            orig_energy
        );
    }

    /// Pinned against `scipy.signal.lfilter_zi(*butter(order, cutoff))`.
    #[test]
    fn lfilter_zi_matches_scipy() {
        let cases: [(usize, f64, &[f64]); 3] = [
            (1, 0.5, &[0.499_999_999_999_999_94]),
            (
                3,
                0.3,
                &[
                    0.950_467_003_642_746_6,
                    -0.360_049_469_100_745_03,
                    0.187_294_297_617_145_97,
                ],
            ),
            (
                5,
                0.2,
                &[
                    0.998_717_418_921_026_7,
                    -1.983_117_596_219_422,
                    1.810_074_712_311_333_7,
                    -0.748_003_966_808_707_3,
                    0.126_713_203_234_314_73,
                ],
            ),
        ];

        for (order, cutoff, expected) in cases {
            let (b, a) = butter_lowpass(order, cutoff);
            let zi = lfilter_zi(&b, &a);
            assert_eq!(zi.len(), expected.len(), "order {order}");
            for (got, want) in zi.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-12,
                    "order {order} cutoff {cutoff}: got {got}, scipy {want}"
                );
            }
        }
    }

    /// A constant input must come out constant. Without `zi` the filter charges
    /// up from rest and the first samples sag — the bug this guards.
    #[test]
    fn filtfilt_preserves_a_constant() {
        let (b, a) = butter_lowpass(5, 0.2);
        let x = vec![3.25_f64; 100];
        for (i, v) in filtfilt(&b, &a, &x).iter().enumerate() {
            assert!(
                (v - 3.25).abs() < 1e-9,
                "sample {i} drifted to {v} from a constant 3.25"
            );
        }
    }
}

/// Butterworth filter design and zero-phase filtering.
///
/// ```
/// use bids_filter::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        butter_bandpass, butter_highpass, butter_lowpass, filtfilt, lfilter, notch_filter, resample,
    };
}

// ───────────────────── Filter design internals ─────────────────────
//
// Enough complex arithmetic to run SciPy's design pipeline: build the analog
// Butterworth prototype, apply the low-pass-to-X frequency transform in the
// analog domain, then bilinear-transform to z and expand the roots into
// polynomial coefficients.

/// A complex number, in the small amount this module needs.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The point `e^{i·theta}` on the unit circle.
    fn from_polar(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }

    fn scale(self, k: f64) -> Self {
        Self::new(self.re * k, self.im * k)
    }

    fn modulus(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The principal square root.
    fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let m = self.modulus();
        let re = ((m + self.re) / 2.0).sqrt();
        let im = ((m - self.re) / 2.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
}

/// Poles of the normalised analog Butterworth prototype.
///
/// `order` poles evenly spaced on the left half of the unit circle.
fn prototype_poles(order: usize) -> Vec<Complex> {
    (0..order)
        .map(|k| {
            let theta = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
            Complex::from_polar(theta)
        })
        .collect()
}

/// Pre-warp a normalised frequency for the bilinear transform.
///
/// Uses `fs = 2`, the convention SciPy's `butter` works in, so a cutoff of 1
/// is Nyquist.
fn prewarp(freq: f64) -> f64 {
    let fs = 2.0;
    2.0 * fs * (PI * freq / fs).tan()
}

/// Bilinear-transform analog zeros and poles into the z domain.
///
/// Zeros at infinity — there are `degree - zeros.len()` of them — map to
/// `z = -1`, which is what pads the numerator to the filter's order.
fn bilinear(zeros: &[Complex], poles: &[Complex], degree: usize) -> (Vec<Complex>, Vec<Complex>) {
    let fs2 = 4.0; // 2 * fs with fs = 2
    let map = |s: Complex| {
        let num = Complex::new(fs2 + s.re, s.im);
        let den = Complex::new(fs2 - s.re, -s.im);
        num.div(den)
    };

    let mut zeros_z: Vec<Complex> = zeros.iter().map(|z| map(*z)).collect();
    let poles_z: Vec<Complex> = poles.iter().map(|p| map(*p)).collect();
    zeros_z.resize(degree.max(zeros_z.len()), Complex::new(-1.0, 0.0));
    (zeros_z, poles_z)
}

/// The gain factor the bilinear transform contributes.
///
/// `prod(2·fs - z_s) / prod(2·fs - p_s)`, which is what carries the analog
/// gain through to the digital filter.
fn bilinear_gain(zeros: &[Complex], poles: &[Complex]) -> f64 {
    let fs2 = 4.0; // 2 * fs with fs = 2
    let term = |acc: Complex, r: &Complex| acc.mul(Complex::new(fs2 - r.re, -r.im));
    let num = zeros.iter().fold(Complex::new(1.0, 0.0), term);
    let den = poles.iter().fold(Complex::new(1.0, 0.0), term);
    num.div(den).re
}

/// Expand complex roots into real polynomial coefficients.
///
/// The roots arrive in conjugate pairs, so the imaginary parts cancel; they
/// are dropped after the expansion.
fn poly_from_roots_c(roots: &[Complex]) -> Vec<f64> {
    let mut coeffs = vec![Complex::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![Complex::new(0.0, 0.0); coeffs.len() + 1];
        for (i, c) in coeffs.iter().enumerate() {
            next[i] = next[i].add(*c);
            next[i + 1] = next[i + 1].sub(c.mul(*r));
        }
        coeffs = next;
    }
    coeffs.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod scipy_parity {
    //! Regression tests against `scipy.signal.butter`.
    //!
    //! The committed golden data only ever covered `btype='low'`, which let a
    //! high-pass design that produced a filter at `1 - cutoff` — and a
    //! band-pass built by cascading it — pass unnoticed. These pin all three
    //! response types to SciPy's coefficients.

    use super::{butter_bandpass, butter_highpass, butter_lowpass};

    /// Assert two coefficient vectors agree to near machine precision.
    fn assert_close(label: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-12 + 1e-9 * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "{label}[{i}]: got {g:.17e}, want {w:.17e} (diff {:.3e})",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn highpass_matches_scipy() {
        // scipy.signal.butter(2, 0.2, btype="high")
        let (b, a) = butter_highpass(2, 0.2);
        assert_close(
            "highpass(2, 0.2) b",
            &b,
            &[
                0.638_945_525_159_022_5,
                -1.277_891_050_318_045,
                0.638_945_525_159_022_5,
            ],
        );
        assert_close(
            "highpass(2, 0.2) a",
            &a,
            &[1.0, -1.142_980_502_539_901_1, 0.412_801_598_096_188_7],
        );

        // scipy.signal.butter(4, 0.35, btype="high")
        let (b, a) = butter_highpass(4, 0.35);
        assert_close(
            "highpass(4, 0.35) b",
            &b,
            &[
                0.216_178_641_293_292_5,
                -0.864_714_565_173_17,
                1.297_071_847_759_755_1,
                -0.864_714_565_173_17,
                0.216_178_641_293_292_5,
            ],
        );
        assert_close(
            "highpass(4, 0.35) a",
            &a,
            &[
                1.0,
                -1.175_125_637_613_149_8,
                0.925_648_732_510_383_7,
                -0.310_421_333_939_956_75,
                0.047_662_556_629_189_79,
            ],
        );
    }

    #[test]
    fn highpass_is_not_the_lowpass_mirror() {
        // The old implementation returned scipy's butter(order, 1 - cutoff).
        // Guard the specific regression: the two must differ.
        let (_, a_at_c) = butter_highpass(3, 0.3);
        let (_, a_at_mirror) = butter_highpass(3, 0.7);
        assert!(
            (a_at_c[1] - a_at_mirror[1]).abs() > 1e-6,
            "high-pass at 0.3 and 0.7 should not coincide"
        );
        // A high-pass passes Nyquist and stops DC.
        let dc_gain: f64 = {
            let (b, a) = butter_highpass(3, 0.3);
            b.iter().sum::<f64>() / a.iter().sum::<f64>()
        };
        assert!(
            dc_gain.abs() < 1e-12,
            "DC gain should vanish, got {dc_gain}"
        );
    }

    #[test]
    fn bandpass_matches_scipy() {
        // scipy.signal.butter(2, [0.1, 0.4], btype="band")
        let (b, a) = butter_bandpass(2, 0.1, 0.4);
        assert_close(
            "bandpass(2, 0.1, 0.4) b",
            &b,
            &[
                0.131_106_439_916_626_02,
                0.0,
                -0.262_212_879_833_252_03,
                0.0,
                0.131_106_439_916_626_02,
            ],
        );
        assert_close(
            "bandpass(2, 0.1, 0.4) a",
            &a,
            &[
                1.0,
                -2.180_657_838_602_799_3,
                2.020_004_116_183_511,
                -1.025_510_847_713_418_3,
                0.272_214_937_925_007_3,
            ],
        );

        // scipy.signal.butter(3, [0.05, 0.45], btype="band")
        let (b, a) = butter_bandpass(3, 0.05, 0.45);
        assert_close(
            "bandpass(3, 0.05, 0.45) b",
            &b,
            &[
                0.098_531_160_923_927_04,
                0.0,
                -0.295_593_482_771_781_1,
                0.0,
                0.295_593_482_771_781_1,
                0.0,
                -0.098_531_160_923_927_04,
            ],
        );
        assert_close(
            "bandpass(3, 0.05, 0.45) a",
            &a,
            &[
                1.0,
                -3.126_622_865_296_128_8,
                4.073_198_268_645_467,
                -3.118_706_810_497_47,
                1.636_214_621_816_703_7,
                -0.516_272_165_239_347_8,
                0.056_297_236_491_842_59,
            ],
        );
    }

    #[test]
    fn lowpass_still_matches_scipy() {
        // scipy.signal.butter(4, 0.25, btype="low")
        let (b, a) = butter_lowpass(4, 0.25);
        assert_close(
            "lowpass(4, 0.25) a",
            &a,
            &[
                1.0,
                -1.968_427_786_938_518,
                1.735_860_709_208_886,
                -0.724_470_829_507_362_3,
                0.120_389_599_896_244_4,
            ],
        );
        // A low-pass passes DC with unit gain.
        let dc_gain: f64 = b.iter().sum::<f64>() / a.iter().sum::<f64>();
        assert!((dc_gain - 1.0).abs() < 1e-12, "DC gain {dc_gain}");
    }
}
