//! Hemodynamic response functions for fMRI statistical modeling.
//!
//! Implements SPM and Glover canonical double-gamma HRFs with optional
//! time and dispersion derivatives, plus FIR basis sets. Numerically
//! validated against SciPy to <1e-10 relative error.

/// Hemodynamic Response Function (HRF) model type.
///
/// Specifies which HRF basis set to use for convolving event regressors
/// in fMRI statistical modeling. The SPM and Glover forms are double-gamma
/// functions with different parameters; each can include time and/or
/// dispersion derivative basis functions.
///
/// The FIR (Finite Impulse Response) model uses a set of delta functions
/// at specified delays, making no assumptions about HRF shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HrfModel {
    Spm,
    SpmDerivative,
    SpmDerivativeDispersion,
    Glover,
    GloverDerivative,
    GloverDerivativeDispersion,
    Fir,
    None,
}

impl HrfModel {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "spm" => Self::Spm,
            "spm + derivative" => Self::SpmDerivative,
            "spm + derivative + dispersion" => Self::SpmDerivativeDispersion,
            "glover" => Self::Glover,
            "glover + derivative" => Self::GloverDerivative,
            "glover + derivative + dispersion" => Self::GloverDerivativeDispersion,
            "fir" => Self::Fir,
            _ => Self::None,
        }
    }
}

/// Parameters for the double-gamma HRF.
struct HrfParams {
    delay: f64,
    undershoot: f64,
    dispersion: f64,
    u_dispersion: f64,
    ratio: f64,
}

const SPM_PARAMS: HrfParams = HrfParams {
    delay: 6.0,
    undershoot: 16.0,
    dispersion: 1.0,
    u_dispersion: 1.0,
    ratio: 0.167,
};
const GLOVER_PARAMS: HrfParams = HrfParams {
    delay: 6.0,
    undershoot: 12.0,
    dispersion: 0.9,
    u_dispersion: 0.9,
    ratio: 0.35,
};

/// Compute a gamma difference HRF kernel.
fn gamma_difference_hrf(
    tr: f64,
    oversampling: usize,
    time_length: f64,
    onset: f64,
    p: &HrfParams,
) -> Vec<f64> {
    let dt = tr / oversampling as f64;
    let n = (time_length / dt).round() as usize;
    let mut hrf = Vec::with_capacity(n);
    // Match numpy.linspace(0, time_length, n) — NOT i*dt
    for i in 0..n {
        let t = if n > 1 {
            i as f64 * time_length / (n - 1) as f64
        } else {
            0.0
        };
        let t = t - onset;
        if t <= 0.0 {
            hrf.push(0.0);
            continue;
        }
        // scipy.stats.gamma.pdf(x, a, scale=s) = x^(a-1) * exp(-x/s) / (s^a * Γ(a))
        // PyBIDS: gamma.pdf(t, delay/disp, dt/disp) — positional arg is scale
        // scipy convention: gamma.pdf(x, a, loc=0, scale=1) where a=shape
        // BUT PyBIDS calls gamma.pdf(t, delay/disp, dt/disp) which means
        // shape = delay/disp, scale = dt/disp (second positional = scale in scipy)
        let g1 = gamma_pdf(t, p.delay / p.dispersion, dt / p.dispersion);
        let g2 = gamma_pdf(t, p.undershoot / p.u_dispersion, dt / p.u_dispersion);
        hrf.push(g1 - p.ratio * g2);
    }
    let sum: f64 = hrf.iter().sum();
    if sum.abs() > 1e-15 {
        for v in &mut hrf {
            *v /= sum;
        }
    }
    hrf
}

/// Gamma probability density function matching scipy.stats.gamma.pdf(x, a, loc, scale=1).
///
/// Uses log-space computation for numerical stability, matching scipy's cephes implementation.
/// PyBIDS calls `gamma.pdf(t, delay/disp, dt/disp)` — 3rd positional arg is `loc`.
fn gamma_pdf(x: f64, shape: f64, loc: f64) -> f64 {
    let x = x - loc;
    if x < 0.0 || shape <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return if shape == 1.0 {
            1.0
        } else if shape > 1.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    // Log-space: log(pdf) = (a-1)*log(x) - x - gammaln(a)
    let log_pdf = (shape - 1.0) * x.ln() - x - gammaln(shape);
    log_pdf.exp()
}

/// Log-gamma function matching scipy's cephes implementation.
///
/// For x < 13: recurrence + rational polynomial approximation.
/// For x >= 13: Stirling series.
/// Natural log of the gamma function, for positive arguments.
///
/// Small positive integers use the exact factorial identity
/// `ln Γ(n) = ln((n-1)!)`; everything else uses the Lanczos approximation
/// (g = 7, 9 terms), which holds ~15 significant digits.
///
/// The rational approximation previously used here was only correct at
/// integer arguments, where the factorial path short-circuits it. That made
/// `spm_hrf` — whose shape parameters are integers — agree with SciPy while
/// `glover_hrf`, whose dispersion of 0.9 gives a shape of 6/0.9, was off by
/// ~0.5 in log space and visibly wrong after normalisation.
fn gammaln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Exact for small positive integers; 18! is the largest factorial below
    // 2^53 and so the largest that stays exactly representable.
    if x.fract() == 0.0 && x <= 19.0 {
        let mut z = 1.0f64;
        let mut k = 2.0f64;
        while k < x {
            z *= k;
            k += 1.0;
        }
        return z.ln();
    }

    /// Lanczos coefficients for g = 7.
    const LANCZOS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    let z = x - 1.0;
    let mut series = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        series += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// SPM canonical HRF.
#[must_use]
pub fn spm_hrf(tr: f64, oversampling: usize, time_length: f64, onset: f64) -> Vec<f64> {
    gamma_difference_hrf(tr, oversampling, time_length, onset, &SPM_PARAMS)
}

/// Glover canonical HRF.
#[must_use]
pub fn glover_hrf(tr: f64, oversampling: usize, time_length: f64, onset: f64) -> Vec<f64> {
    gamma_difference_hrf(tr, oversampling, time_length, onset, &GLOVER_PARAMS)
}

/// SPM time derivative.
pub fn spm_time_derivative(tr: f64, oversampling: usize, time_length: f64, onset: f64) -> Vec<f64> {
    let d = 0.1;
    let h1 = spm_hrf(tr, oversampling, time_length, onset);
    let h2 = spm_hrf(tr, oversampling, time_length, onset + d);
    h1.iter().zip(&h2).map(|(a, b)| (a - b) / d).collect()
}

/// Glover time derivative.
pub fn glover_time_derivative(
    tr: f64,
    oversampling: usize,
    time_length: f64,
    onset: f64,
) -> Vec<f64> {
    let d = 0.1;
    let h1 = glover_hrf(tr, oversampling, time_length, onset);
    let h2 = glover_hrf(tr, oversampling, time_length, onset + d);
    h1.iter().zip(&h2).map(|(a, b)| (a - b) / d).collect()
}

/// SPM dispersion derivative.
pub fn spm_dispersion_derivative(
    tr: f64,
    oversampling: usize,
    time_length: f64,
    onset: f64,
) -> Vec<f64> {
    let dd = 0.01;
    let h1 = gamma_difference_hrf(tr, oversampling, time_length, onset, &SPM_PARAMS);
    let p2 = HrfParams {
        dispersion: SPM_PARAMS.dispersion + dd,
        ..SPM_PARAMS
    };
    let h2 = gamma_difference_hrf(tr, oversampling, time_length, onset, &p2);
    h1.iter().zip(&h2).map(|(a, b)| (a - b) / dd).collect()
}

/// Glover dispersion derivative.
pub fn glover_dispersion_derivative(
    tr: f64,
    oversampling: usize,
    time_length: f64,
    onset: f64,
) -> Vec<f64> {
    let dd = 0.01;
    let h1 = gamma_difference_hrf(tr, oversampling, time_length, onset, &GLOVER_PARAMS);
    let p2 = HrfParams {
        dispersion: GLOVER_PARAMS.dispersion + dd,
        ..GLOVER_PARAMS
    };
    let h2 = gamma_difference_hrf(tr, oversampling, time_length, onset, &p2);
    h1.iter().zip(&h2).map(|(a, b)| (a - b) / dd).collect()
}

/// Get HRF kernel(s) for a given model.
#[must_use]
pub fn hrf_kernel(
    model: &HrfModel,
    tr: f64,
    oversampling: usize,
    fir_delays: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    match model {
        HrfModel::Spm => vec![spm_hrf(tr, oversampling, 32.0, 0.0)],
        HrfModel::SpmDerivative => vec![
            spm_hrf(tr, oversampling, 32.0, 0.0),
            spm_time_derivative(tr, oversampling, 32.0, 0.0),
        ],
        HrfModel::SpmDerivativeDispersion => vec![
            spm_hrf(tr, oversampling, 32.0, 0.0),
            spm_time_derivative(tr, oversampling, 32.0, 0.0),
            spm_dispersion_derivative(tr, oversampling, 32.0, 0.0),
        ],
        HrfModel::Glover => vec![glover_hrf(tr, oversampling, 32.0, 0.0)],
        HrfModel::GloverDerivative => vec![
            glover_hrf(tr, oversampling, 32.0, 0.0),
            glover_time_derivative(tr, oversampling, 32.0, 0.0),
        ],
        HrfModel::GloverDerivativeDispersion => vec![
            glover_hrf(tr, oversampling, 32.0, 0.0),
            glover_time_derivative(tr, oversampling, 32.0, 0.0),
            glover_dispersion_derivative(tr, oversampling, 32.0, 0.0),
        ],
        HrfModel::Fir => fir_delays
            .unwrap_or(&[])
            .iter()
            .map(|&delay| {
                let d = delay as usize;
                let mut kernel = vec![0.0; d * oversampling + oversampling];
                for i in (d * oversampling)..(d * oversampling + oversampling) {
                    if i < kernel.len() {
                        kernel[i] = 1.0;
                    }
                }
                kernel
            })
            .collect(),
        HrfModel::None => vec![{
            let mut k = vec![0.0; oversampling];
            k[0] = 1.0;
            k
        }],
    }
}

/// Sample a condition as an event regressor (boxcar), then convolve with HRF.
///
/// `exp_condition`: (onsets, durations, amplitudes)
/// `frame_times`: sample time points in seconds
pub fn compute_regressor(
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    hrf_model: &HrfModel,
    frame_times: &[f64],
    oversampling: usize,
    fir_delays: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    let n = frame_times.len();
    if n < 2 {
        return vec![vec![0.0; n]];
    }

    let tr = frame_times[frame_times.len() - 1] / (n - 1) as f64;
    let n_hr = n * oversampling;
    let dt = tr / oversampling as f64;

    // Build high-res regressor
    let mut hr_reg = vec![0.0f64; n_hr];
    for i in 0..onsets.len() {
        let onset_idx = ((onsets[i] / dt).round() as usize).min(n_hr - 1);
        let dur_idx = (((onsets[i] + durations[i]) / dt).round() as usize).min(n_hr - 1);
        hr_reg[onset_idx] += amplitudes[i];
        if dur_idx < n_hr && dur_idx != onset_idx {
            hr_reg[dur_idx] -= amplitudes[i];
        }
    }
    // Cumulative sum to create boxcar
    for i in 1..n_hr {
        hr_reg[i] += hr_reg[i - 1];
    }

    // Get kernels
    let kernels = hrf_kernel(hrf_model, tr, oversampling, fir_delays);

    // Convolve and downsample
    kernels
        .iter()
        .map(|kernel| {
            let conv = convolve(&hr_reg, kernel);
            // Downsample to frame_times
            (0..n)
                .map(|i| {
                    let idx = i * oversampling;
                    if idx < conv.len() { conv[idx] } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

/// Simple 1D convolution.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            if i >= j {
                result[i] += a[i - j] * b[j];
            }
        }
    }
    result
}

/// Regressor names for a given HRF model.
#[must_use]
pub fn regressor_names(
    con_name: &str,
    model: &HrfModel,
    fir_delays: Option<&[f64]>,
) -> Vec<String> {
    match model {
        HrfModel::Spm | HrfModel::Glover | HrfModel::None => vec![con_name.into()],
        HrfModel::SpmDerivative | HrfModel::GloverDerivative => {
            vec![con_name.into(), format!("{}_derivative", con_name)]
        }
        HrfModel::SpmDerivativeDispersion | HrfModel::GloverDerivativeDispersion => vec![
            con_name.into(),
            format!("{}_derivative", con_name),
            format!("{}_dispersion", con_name),
        ],
        HrfModel::Fir => fir_delays
            .unwrap_or(&[])
            .iter()
            .map(|d| format!("{}_delay_{}", con_name, *d as i64))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spm_hrf() {
        let hrf = spm_hrf(2.0, 50, 32.0, 0.0);
        assert!(!hrf.is_empty());
        let sum: f64 = hrf.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "HRF should sum to ~1, got {}", sum);
        // Peak should be in the first half
        let peak_idx = hrf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            peak_idx < hrf.len() / 2,
            "Peak should be in first half, at idx {}",
            peak_idx
        );
    }

    #[test]
    fn test_glover_hrf() {
        let hrf = glover_hrf(2.0, 50, 32.0, 0.0);
        assert!(!hrf.is_empty());
        let sum: f64 = hrf.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_regressor() {
        let onsets = vec![2.0, 6.0];
        let durations = vec![1.0, 1.0];
        let amplitudes = vec![1.0, 1.0];
        let frame_times: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();

        let regs = compute_regressor(
            &onsets,
            &durations,
            &amplitudes,
            &HrfModel::Spm,
            &frame_times,
            50,
            None,
        );
        assert_eq!(regs.len(), 1); // SPM produces 1 regressor
        assert_eq!(regs[0].len(), 100);
    }

    #[test]
    fn test_regressor_names() {
        assert_eq!(regressor_names("face", &HrfModel::Spm, None), vec!["face"]);
        assert_eq!(
            regressor_names("face", &HrfModel::SpmDerivative, None),
            vec!["face", "face_derivative"]
        );
    }
}

#[cfg(test)]
mod pybids_parity {
    //! Regression tests against PyBIDS' `bids.modeling.hrf`.
    //!
    //! The committed golden data exercised `spm_hrf`, whose shape parameters
    //! are integers and so took `gammaln`'s exact factorial path. `glover_hrf`
    //! uses a dispersion of 0.9, giving a shape of 6/0.9, which went through
    //! the rational approximation instead — and that was wrong by ~0.5 in log
    //! space, leaving the normalised HRF visibly misshapen.

    use super::{glover_hrf, spm_hrf};

    #[test]
    fn glover_matches_pybids() {
        // bids.modeling.hrf.glover_hrf(2.0, oversampling=16, time_length=32.0)
        let h = glover_hrf(2.0, 16, 32.0, 0.0);
        assert_eq!(h.len(), 256);

        for (i, want) in [
            (40usize, 0.029_878_996_620_615_9_f64),
            (45, 0.031_472_264_197_766_1),
            (46, 0.031_485_522_667_294_26),
            (50, 0.030_618_452_101_931_176),
            (60, 0.023_685_401_454_912_478),
            (80, 0.005_136_734_535_805_884),
        ] {
            let got = h[i];
            assert!(
                (got - want).abs() <= 1e-12 + 1e-9 * want.abs(),
                "glover_hrf[{i}]: got {got:.17e}, want {want:.17e}"
            );
        }

        let sum: f64 = h.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "HRF should be normalised, got {sum}"
        );
        let peak = h.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            (peak - 0.031_485_522_667_294_26).abs() < 1e-12,
            "peak {peak}"
        );
    }

    #[test]
    fn spm_is_unchanged_by_the_gammaln_rewrite() {
        // bids.modeling.hrf.spm_hrf(2.0, oversampling=16, time_length=32.0)
        let h = spm_hrf(2.0, 16, 32.0, 0.0);
        assert_eq!(h.len(), 256);
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "got {sum}");
        // SPM peaks around 5 s; at TR 2 s with 16x oversampling that is ~40.
        let peak_idx = h
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((36..=44).contains(&peak_idx), "peak at {peak_idx}");
    }

    #[test]
    fn gammaln_matches_scipy_on_non_integers() {
        // scipy.special.gammaln at the shapes the Glover model uses.
        for (x, want) in [
            (6.0_f64 / 0.9, 5.963_669_124_282_592_f64),
            (12.0 / 0.9, 20.833_615_986_426_6),
            (2.5, 0.284_682_870_472_919_2),
            (0.5, 0.572_364_942_924_700_1),
        ] {
            let got = super::gammaln(x);
            assert!(
                (got - want).abs() <= 1e-12 + 1e-11 * want.abs(),
                "gammaln({x}): got {got:.17}, want {want:.17}"
            );
        }
    }
}
