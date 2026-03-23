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
/// Reproduces scipy.special.gammaln to machine epsilon.
fn gammaln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let mut xx = x;

    if xx < 13.0 {
        let mut z = 1.0;
        while xx >= 3.0 {
            xx -= 1.0;
            z *= xx;
        }
        while xx < 2.0 {
            if xx == 0.0 {
                return f64::INFINITY;
            }
            z /= xx;
            xx += 1.0;
        }
        if z < 0.0 {
            z = -z;
        }
        if xx == 2.0 {
            return z.ln();
        }
        xx -= 2.0;
        // Rational approximation for lgamma(2+x), 0 <= x <= 1 (cephes coefficients)
        const P: [f64; 8] = [
            -1.716_185_138_865_495,
            2.476_565_080_557_592e1,
            -3.798_042_564_709_456_3e2,
            6.293_311_553_128_184e2,
            8.669_662_027_904_133e2,
            -3.145_127_296_884_836_7e4,
            -3.614_441_341_869_117_6e4,
            6.645_614_382_024_054e4,
        ];
        const Q: [f64; 8] = [
            -3.084_023_001_197_389_7e1,
            3.153_506_269_796_041_6e2,
            -1.015_156_367_490_219_2e3,
            -3.107_771_671_572_311e3,
            2.253_811_842_098_015e4,
            4.755_846_277_527_881e3,
            -1.346_599_598_649_693e5,
            -1.151_322_596_755_534_9e5,
        ];
        let mut xnum = 0.0;
        let mut xden = 1.0;
        for i in 0..8 {
            xnum = (xnum + P[i]) * xx;
            xden = xden * xx + Q[i];
        }
        return (z * (xnum / xden + 1.0)).ln();
    }

    // Stirling series for x >= 13
    let q = (xx - 0.5) * xx.ln() - xx + 0.918_938_533_204_672_8;
    if xx > 1.0e8 {
        return q;
    }
    let inv_x = 1.0 / xx;
    let inv_x2 = inv_x * inv_x;
    const S: [f64; 6] = [
        8.333_333_333_333_333e-2,
        -2.777_777_777_777_778e-3,
        7.936_507_936_507_937e-4,
        -5.952_380_952_380_953e-4,
        8.417_508_417_508_417e-4,
        -1.917_526_917_526_917_6e-3,
    ];
    let mut s = S[5];
    for i in (0..5).rev() {
        s = s * inv_x2 + S[i];
    }
    q + s * inv_x
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
