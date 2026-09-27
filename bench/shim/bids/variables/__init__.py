"""Variable loading mirroring ``bids.variables``.

Collections are returned by the Rust engine as plain dicts of
``{variable_name: values}`` rather than pandas-backed objects, so the
container types here are thin views over that.
"""

__all__ = [
    'BIDSRunVariableCollection',
    'BIDSVariableCollection',
    'load_variables',
    'merge_collections',
]


class BIDSVariableCollection:
    """A group of variables at one level of the BIDS hierarchy."""

    def __init__(self, payload):
        self.level = payload.get('level', '')
        self.variables = payload.get('variables', {})
        self.entities = payload.get('entities', {})
        self.sampling_rate = payload.get('sampling_rate')

    def __repr__(self):
        return f'<BIDSVariableCollection level={self.level!r} n={len(self.variables)}>'

    def __getitem__(self, name):
        return self.variables[name]

    def __contains__(self, name):
        return name in self.variables

    def keys(self):
        """Variable names."""
        return self.variables.keys()

    def to_df(self, **kwargs):  # noqa: ARG002
        """Variables as a wide pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.variables)


class BIDSRunVariableCollection(BIDSVariableCollection):
    """A group of run-level variables."""


def load_variables(layout, types=None, levels=None, skip_empty=True, dataset=None, scan_length=None, **kwargs):  # noqa: ARG001
    """Load variables from a layout, keyed by level."""
    wanted = levels or ['run', 'session', 'subject', 'dataset']
    if isinstance(wanted, str):
        wanted = [wanted]
    out = {}
    for level in wanted:
        colls = [BIDSVariableCollection(c) for c in layout.get_collections(level)]
        if colls or not skip_empty:
            out[level] = colls
    return out


def merge_collections(collections, **kwargs):  # noqa: ARG001
    """Merge several collections into one."""
    if not collections:
        return None
    merged = {}
    for c in collections:
        merged.update(c.variables)
    first = collections[0]
    return BIDSVariableCollection(
        {'level': first.level, 'variables': merged, 'entities': first.entities}
    )
