#!/usr/bin/env python3
"""Generate golden reference data from PyBIDS for end-to-end comparison with bids-rs.

Outputs JSON files that the Rust integration tests can load and compare against.
"""

import json
import os
import sys
import time
import numpy as np

# ── Paths ──
EXAMPLES = os.path.join(os.path.dirname(__file__), '..', '..', 'pybids', 'bids-examples')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'golden')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Layout tests ──
def test_layout(dataset_name):
    from bids import BIDSLayout
    t0 = time.time()
    layout = BIDSLayout(os.path.join(EXAMPLES, dataset_name))
    t_index = time.time() - t0

    result = {
        'dataset': dataset_name,
        'timing_index_ms': round(t_index * 1000, 2),
        'subjects': sorted(layout.get_subjects()),
        'sessions': sorted(layout.get_sessions()),
        'tasks': sorted(layout.get_tasks()),
        'runs': sorted(layout.get_runs()),
        'datatypes': sorted(layout.get_datatypes()),
        'suffixes': sorted(layout.get_suffixes()),
        'n_files': len(layout.get(return_type='file')),
    }

    # Query tests
    t0 = time.time()
    eeg_files = sorted([f.relpath for f in layout.get(suffix='eeg')])
    result['timing_query_eeg_ms'] = round((time.time() - t0) * 1000, 2)
    result['eeg_files'] = eeg_files

    events_files = sorted([f.relpath for f in layout.get(suffix='events', extension='.tsv')])
    result['events_files'] = events_files

    channels_files = sorted([f.relpath for f in layout.get(suffix='channels', extension='.tsv')])
    result['channels_files'] = channels_files

    # Metadata test
    if eeg_files:
        first = layout.get(suffix='eeg')[0]
        md = layout.get_metadata(first.path)
        result['first_eeg_metadata'] = {k: v for k, v in md.items()
                                         if isinstance(v, (str, int, float, bool))}

    # get_subjects with filter
    if result['tasks']:
        task = result['tasks'][0]
        subs = sorted(layout.get(suffix='eeg', task=task, return_type='id', target='subject'))
        result[f'subjects_task_{task}'] = subs

    return result


# ── 2. Entity parsing tests ──
def test_entity_parsing():
    from bids.layout.utils import parse_file_entities
    test_paths = [
        'sub-01/eeg/sub-01_task-rest_eeg.edf',
        'sub-02/ses-01/func/sub-02_ses-01_task-motor_run-02_bold.nii.gz',
        'sub-03/anat/sub-03_T1w.nii.gz',
        'sub-04/dwi/sub-04_acq-multiband_dwi.nii.gz',
        'sub-05/ses-02/eeg/sub-05_ses-02_task-faces_run-01_channels.tsv',
    ]
    results = {}
    for path in test_paths:
        ents = parse_file_entities(path)
        results[path] = {k: str(v) for k, v in ents.items()}
    return results


# ── 3. Path building tests ──
def test_path_building():
    from bids.layout.writing import build_path
    tests = [
        {
            'entities': {'subject': '001', 'suffix': 'T1w', 'extension': '.nii'},
            'patterns': [
                'sub-{subject}[/ses-{session}]/anat/sub-{subject}[_ses-{session}]_{suffix<T1w|T2w>}{extension<.nii|.nii.gz>|.nii.gz}',
            ],
            'strict': False,
        },
        {
            'entities': {'subject': '001', 'extension': '.bvec'},
            'patterns': [
                'sub-{subject}[/ses-{session}]/{datatype|dwi}/sub-{subject}[_ses-{session}]_{suffix|dwi}{extension<.bval|.bvec|.json|.nii.gz|.nii>|.nii.gz}',
            ],
            'strict': True,
        },
    ]
    results = []
    for t in tests:
        result = build_path(t['entities'], t['patterns'], strict=t['strict'])
        results.append({'entities': t['entities'], 'result': result})
    return results


# ── 4. HRF tests ──
def test_hrf():
    from bids.modeling.hrf import spm_hrf, glover_hrf, compute_regressor
    results = {}

    spm = spm_hrf(2.0, oversampling=50, time_length=32.0)
    results['spm_hrf_len'] = len(spm)
    results['spm_hrf_sum'] = float(np.sum(spm))
    results['spm_hrf_peak_idx'] = int(np.argmax(spm))
    results['spm_hrf_first10'] = [float(x) for x in spm[:10]]

    glover = glover_hrf(2.0, oversampling=50, time_length=32.0)
    results['glover_hrf_len'] = len(glover)
    results['glover_hrf_sum'] = float(np.sum(glover))
    results['glover_hrf_peak_idx'] = int(np.argmax(glover))

    # Compute regressor
    frame_times = np.arange(0, 50, 0.5)
    onsets = np.array([2.0, 6.0, 12.0])
    durations = np.array([1.0, 1.0, 2.0])
    amplitudes = np.array([1.0, 1.0, 1.0])
    reg, names = compute_regressor(
        np.array([onsets, durations, amplitudes]),
        'spm', frame_times, con_id='stim', oversampling=50
    )
    results['regressor_shape'] = list(reg.shape)
    results['regressor_names'] = names
    results['regressor_first10'] = [float(x) for x in reg[:10, 0]]

    return results


# ── 5. NIfTI header tests ──
def test_nifti():
    import nibabel as nib
    import tempfile
    results = {}

    # Create a synthetic NIfTI file
    data = np.zeros((64, 64, 32, 100), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header['pixdim'][4] = 2.0  # TR = 2s

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        nib.save(img, f.name)
        results['nifti_path'] = f.name

    hdr = nib.load(results['nifti_path']).header
    results['ndim'] = int(hdr['dim'][0])
    results['dim'] = [int(x) for x in hdr['dim'][1:5]]
    results['pixdim'] = [float(x) for x in hdr['pixdim'][1:5]]
    results['n_vols'] = int(hdr['dim'][4])
    results['datatype'] = int(hdr['datatype'])

    return results


# ── 6. Inflect tests ──
def test_inflect():
    from bids.external import inflect
    p = inflect.engine()
    words = ['subjects', 'sessions', 'runs', 'tasks', 'vertices',
             'analyses', 'atlases', 'categories', 'echoes']
    results = {}
    for w in words:
        singular = p.singular_noun(w)
        results[w] = singular if singular else w
    return results


# ── 7. Butterworth filter tests ──
def test_filter():
    from scipy.signal import butter, filtfilt
    results = {}

    # Order 5, cutoff 0.2
    b, a = butter(5, 0.2, btype='low')
    results['butter5_b'] = [float(x) for x in b]
    results['butter5_a'] = [float(x) for x in a]

    # Order 1, cutoff 0.5
    b1, a1 = butter(1, 0.5, btype='low')
    results['butter1_b'] = [float(x) for x in b1]
    results['butter1_a'] = [float(x) for x in a1]

    # filtfilt test
    np.random.seed(42)
    t = np.arange(200) / 100.0
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 40 * t)
    b, a = butter(5, 0.2, btype='low')
    filtered = filtfilt(b, a, signal)
    results['filtfilt_first10'] = [float(x) for x in filtered[:10]]
    results['filtfilt_energy'] = float(np.mean(filtered**2))
    results['original_energy'] = float(np.mean(signal**2))

    return results


# ── Run all ──
if __name__ == '__main__':
    all_results = {}

    print('Testing layouts...')
    for ds in ['eeg_cbm', 'eeg_rishikesh']:
        ds_path = os.path.join(EXAMPLES, ds)
        if os.path.exists(ds_path):
            all_results[f'layout_{ds}'] = test_layout(ds)

    print('Testing entity parsing...')
    all_results['entity_parsing'] = test_entity_parsing()

    print('Testing path building...')
    all_results['path_building'] = test_path_building()

    print('Testing HRF...')
    all_results['hrf'] = test_hrf()

    print('Testing NIfTI...')
    all_results['nifti'] = test_nifti()

    print('Testing inflect...')
    all_results['inflect'] = test_inflect()

    print('Testing filter...')
    all_results['filter'] = test_filter()

    # Write results
    out_path = os.path.join(OUT_DIR, 'golden.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Written golden data to {out_path}')
    print(f'Total test groups: {len(all_results)}')
