"""Writes minimal.snirf, the fixture for tests/snirf.rs, with libhdf5 (h5py).

dataTimeSeries is chunked with gzip and shuffle, time is contiguous, so the
test reads both storage layouts. Values: sample s, channel c = (4s + c) / 7.
"""

import h5py
import numpy as np

with h5py.File("minimal.snirf", "w") as f:
    f["formatVersion"] = "1.0"
    d = f.create_group("nirs").create_group("data1")
    ts = np.arange(400, dtype="f8").reshape(100, 4) / 7.0
    d.create_dataset("dataTimeSeries", data=ts, chunks=(25, 4), compression="gzip", shuffle=True)
    d.create_dataset("time", data=np.arange(100, dtype="f8") / 10.0)
    for i in range(1, 5):
        m = d.create_group(f"measurementList{i}")
        m["sourceIndex"] = np.int32(i)
        m["detectorIndex"] = np.int32(10 + i)
        m["wavelengthIndex"] = np.int32(1 + i % 2)
