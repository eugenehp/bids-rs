"""File and entity models mirroring ``bids.layout.models``."""

import json
import os

__all__ = [
    'BIDSDataFile',
    'BIDSFile',
    'BIDSImageFile',
    'BIDSJSONFile',
    'Config',
    'Entity',
    'Tag',
]


class BIDSFile(os.PathLike):
    """A file in a BIDS dataset.

    Wraps the Rust ``pybids_rs.BIDSFile`` and adds the layout-dependent
    accessors PyBIDS puts on this class.
    """

    def __init__(self, inner, layout=None):
        self._inner = inner
        self._layout = layout

    # ── Identity ──

    @property
    def path(self):
        """Absolute path to the file."""
        return self._inner.path

    @property
    def filename(self):
        """Basename of the file."""
        return self._inner.filename

    @property
    def dirname(self):
        """Directory containing the file."""
        return os.path.dirname(self._inner.path)

    @property
    def entities(self):
        """Entity dict parsed from the filename and path."""
        return dict(self._inner.entities)

    def __fspath__(self):
        return self._inner.path

    def __repr__(self):
        return f'<{type(self).__name__} filename={self.filename!r}>'

    def __eq__(self, other):
        return isinstance(other, BIDSFile) and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    # ── Accessors ──

    @property
    def relpath(self):
        """Path relative to the dataset root."""
        return self._inner.relpath()

    def get_entities(self, metadata=False, values='tags'):  # noqa: ARG002
        """Entities for this file, optionally including sidecar metadata."""
        ents = self.entities
        if metadata:
            ents = {**ents, **self.get_metadata()}
        return ents

    def get_metadata(self):
        """Sidecar metadata for this file, with BIDS inheritance applied."""
        if self._layout is None:
            return {}
        return self._layout.get_metadata(self.path)

    def get_associations(self, kind=None, include_parents=False):  # noqa: ARG002
        """Files associated with this one."""
        if self._layout is None:
            return []
        return self._layout.get_associations(self.path, kind=kind)

    def copy(self, path_patterns, symbolic_link=False, root=None, conflicts='fail'):  # noqa: ARG002
        """Copy this file to a path built from ``path_patterns``."""
        raise NotImplementedError('BIDSFile.copy is not implemented in the bids-rs shim')


class BIDSDataFile(BIDSFile):
    """A tabular BIDS file (``.tsv``, ``.tsv.gz``)."""

    def get_df(self, include_timing=True, adjust_onset=False, enforce_dtypes=True, **kwargs):  # noqa: ARG002
        """Read the file into a pandas DataFrame."""
        import pandas as pd

        return pd.read_csv(self.path, sep='\t', na_values='n/a')


class BIDSImageFile(BIDSFile):
    """A neuroimaging volume (``.nii``, ``.nii.gz``)."""

    def get_image(self, **kwargs):  # noqa: ARG002
        """Load the image with nibabel."""
        import nibabel as nb

        return nb.load(self.path)


class BIDSJSONFile(BIDSFile):
    """A JSON sidecar."""

    def get_dict(self):
        """Parse the file into a dict."""
        with open(self.path, encoding='utf-8') as fh:
            return json.load(fh)

    def get_json(self):
        """Raw file contents."""
        with open(self.path, encoding='utf-8') as fh:
            return fh.read()


def make_bidsfile(inner, layout=None):
    """Build the ``BIDSFile`` subclass matching a file's extension."""
    ext = os.path.basename(inner.path)
    if ext.endswith(('.nii', '.nii.gz')):
        return BIDSImageFile(inner, layout)
    if ext.endswith(('.tsv', '.tsv.gz')):
        return BIDSDataFile(inner, layout)
    if ext.endswith('.json'):
        return BIDSJSONFile(inner, layout)
    return BIDSFile(inner, layout)


class Entity:
    """A BIDS entity and the values it takes in a dataset."""

    def __init__(self, name, layout=None, pattern=None, dtype='str'):
        self.name = name
        self.pattern = pattern
        self.dtype = dtype
        self._layout = layout

    def __repr__(self):
        return f'<Entity name={self.name!r}>'

    def unique(self):
        """Distinct values this entity takes."""
        if self._layout is None:
            return []
        return self._layout.get_entity_values(self.name)

    def count(self, files=False):  # noqa: ARG002
        """Number of distinct values."""
        return len(self.unique())


class Config:
    """A named entity configuration."""

    def __init__(self, name, entities=None, default_path_patterns=None):
        self.name = name
        self.entities = entities or {}
        self.default_path_patterns = default_path_patterns or []

    def __repr__(self):
        return f'<Config name={self.name!r}>'


class Tag:
    """An (entity, value) pair attached to a file."""

    def __init__(self, file_path, entity_name, value, dtype='str'):
        self.file_path = file_path
        self.entity_name = entity_name
        self.value = value
        self.dtype = dtype

    def __repr__(self):
        return f'<Tag {self.entity_name}={self.value!r}>'
