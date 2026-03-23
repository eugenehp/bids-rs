#!/usr/bin/env python3
"""Generate high-precision golden data for numerical comparison."""

import json
import os
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.stats import gamma
from bids.modeling.hrf import spm_hrf, glover_hrf, spm_time_derivative, glover_time_derivative

OUT = os.path.join(os.path.dirname(__file__), 'golden', 'precision.json')
results = {}

# ── 1. Butterworth coefficients at multiple orders/cutoffs ──
for order in [1, 2, 3, 4, 5]:
    for cutoff in [0.1, 0.2, 0.3, 0.5, 0.8]:
        b, a = butter(order, cutoff, btype='low')
        key = f"butter_o{order}_c{cutoff}"
        results[key] = {
            'b': [float(x) for x in b],
            'a': [float(x) for x in a],
        }

# ── 2. lfilter output ──
np.random.seed(0)
x = np.random.randn(50).tolist()
b5, a5 = butter(3, 0.3)
y_lfilter = lfilter(b5, a5, x)
results['lfilter_output'] = [float(v) for v in y_lfilter]

# ── 3. filtfilt output ──
np.random.seed(0)
sig = np.random.randn(200).tolist()
b5, a5 = butter(5, 0.2)
y_ff = filtfilt(b5, a5, sig)
results['filtfilt_output_first20'] = [float(v) for v in y_ff[:20]]
results['filtfilt_output_last20'] = [float(v) for v in y_ff[-20:]]
results['filtfilt_rms'] = float(np.sqrt(np.mean(y_ff**2)))

# ── 4. HRF values at specific time points ──
for tr in [1.0, 2.0, 0.5]:
    for ov in [16, 50]:
        h = spm_hrf(tr, oversampling=ov, time_length=32.0)
        key = f"spm_hrf_tr{tr}_ov{ov}"
        results[key] = {
            'len': len(h),
            'values': [float(v) for v in h[:50]],  # first 50 samples
            'sum': float(np.sum(h)),
            'peak_idx': int(np.argmax(h)),
            'peak_val': float(np.max(h)),
        }

        h2 = glover_hrf(tr, oversampling=ov, time_length=32.0)
        key2 = f"glover_hrf_tr{tr}_ov{ov}"
        results[key2] = {
            'len': len(h2),
            'values': [float(v) for v in h2[:50]],
            'sum': float(np.sum(h2)),
            'peak_idx': int(np.argmax(h2)),
            'peak_val': float(np.max(h2)),
        }

# ── 5. Time derivative ──
td = spm_time_derivative(2.0, 50, 32.0)
results['spm_time_deriv'] = {
    'len': len(td),
    'first20': [float(v) for v in td[:20]],
    'peak_idx': int(np.argmax(np.abs(td))),
}
td2 = glover_time_derivative(2.0, 50, 32.0)
results['glover_time_deriv'] = {
    'len': len(td2),
    'first20': [float(v) for v in td2[:20]],
    'peak_idx': int(np.argmax(np.abs(td2))),
}

# ── 6. Raw gamma PDF values ──
from scipy.stats import gamma as gamma_dist
test_points = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 16.0, 20.0]
for shape in [3.0, 6.0, 10.0]:
    for loc in [0.0, 0.04]:
        vals = [float(gamma_dist.pdf(x, shape, loc=loc, scale=1.0)) for x in test_points]
        results[f"gamma_pdf_shape{shape}_loc{loc}"] = {
            'x': test_points,
            'values': vals,
        }

# ── 7. DC gain of filters ──
for order in [1, 3, 5]:
    for cutoff in [0.1, 0.3, 0.5]:
        b, a = butter(order, cutoff)
        dc = sum(b) / sum(a)
        results[f"dc_gain_o{order}_c{cutoff}"] = float(dc)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Written {len(results)} precision tests to {OUT}")
