"""Statistical modeling mirroring ``bids.modeling``.

Graph execution lives in the Rust ``bids-modeling`` crate. The Python binding
currently surfaces the HRF functions; the graph types are declared here so the
module shape matches PyBIDS, and they raise if used.
"""

from . import hrf

__all__ = [
    'BIDSStatsModelsGraph',
    'BIDSStatsModelsNode',
    'BIDSStatsModelsNodeOutput',
    'TransformerManager',
    'auto_model',
    'hrf',
]


def _unbound(name):
    def _raise(*args, **kwargs):  # noqa: ARG001
        raise NotImplementedError(
            f'{name} is implemented in the bids-modeling Rust crate but is not yet '
            'exposed through the pybids_rs extension module'
        )

    return _raise


BIDSStatsModelsGraph = _unbound('BIDSStatsModelsGraph')
BIDSStatsModelsNode = _unbound('BIDSStatsModelsNode')
BIDSStatsModelsNodeOutput = _unbound('BIDSStatsModelsNodeOutput')
TransformerManager = _unbound('TransformerManager')
auto_model = _unbound('auto_model')
