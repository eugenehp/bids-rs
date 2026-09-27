"""Haemodynamic response functions mirroring ``bids.modeling.hrf``."""

import numpy as np
import pybids_rs

__all__ = ['glover_hrf', 'spm_hrf']


def spm_hrf(tr, oversampling=50, time_length=32.0, onset=0.0):
    """SPM canonical HRF."""
    return np.asarray(pybids_rs.spm_hrf(tr, oversampling, time_length, onset))


def glover_hrf(tr, oversampling=50, time_length=32.0, onset=0.0):
    """Glover canonical HRF."""
    return np.asarray(pybids_rs.glover_hrf(tr, oversampling, time_length, onset))
