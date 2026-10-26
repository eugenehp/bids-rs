"""A drop-in ``bids`` package backed by the Rust ``pybids_rs`` extension.

Mirrors PyBIDS' public module structure one-to-one, so code written against
PyBIDS imports and runs unchanged::

    from bids import BIDSLayout
    layout = BIDSLayout('/path/to/dataset')
    layout.get(suffix='bold', extension='.nii.gz')

Put this directory ahead of PyBIDS on ``sys.path`` and ``import bids`` resolves
here instead. See ``BACKEND`` to tell which implementation is loaded.
"""

from .layout import BIDSLayout, BIDSLayoutIndexer, BIDSValidator

__all__ = [
    'BIDSLayout',
    'BIDSLayoutIndexer',
    'BIDSValidator',
    'config',
    'layout',
    'modeling',
    'reports',
    'utils',
    'variables',
]

#: Identifies which implementation backs this ``bids`` package.
BACKEND = 'bids-rs'

__version__ = '0.0.4+bids-rs'
