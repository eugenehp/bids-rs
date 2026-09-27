"""``BIDSLayout`` mirroring ``bids.layout.layout``, backed by ``pybids_rs``."""

import enum
import os

import pybids_rs

from ..exceptions import NoMatchError, TargetError
from .models import Entity, make_bidsfile
from .utils import BIDSMetadata

__all__ = ['BIDSLayout', 'Query']


class Query(enum.Enum):
    """Sentinels for entity presence/absence in a query."""

    NONE = 1
    REQUIRED = ANY = 2
    OPTIONAL = 3


class BIDSLayout:
    """Index and query a BIDS dataset.

    The same constructor signature and query surface as PyBIDS'
    ``BIDSLayout``, with indexing and querying performed by the Rust
    ``bids-rs`` engine.
    """

    def __init__(
        self,
        root=None,
        validate=True,
        absolute_paths=True,  # noqa: ARG002
        derivatives=False,
        config=None,  # noqa: ARG002
        sources=None,  # noqa: ARG002
        database_path=None,
        reset_database=False,  # noqa: ARG002
        indexer=None,
        **kwargs,  # noqa: ARG002
    ):
        index_metadata = True
        if indexer is not None:
            validate = indexer.validate
            index_metadata = indexer.index_metadata

        self._root = str(root)
        self._inner = pybids_rs.BIDSLayout(
            self._root,
            validate=validate,
            derivatives=bool(derivatives),
            index_metadata=index_metadata,
            database_path=str(database_path) if database_path else None,
        )
        if isinstance(derivatives, (str, os.PathLike)):
            self._inner.add_derivatives(str(derivatives))
        elif isinstance(derivatives, (list, tuple)):
            for d in derivatives:
                self._inner.add_derivatives(str(d))

    # ── Construction ──

    @classmethod
    def load(cls, database_path):
        """Load a previously saved index."""
        obj = cls.__new__(cls)
        obj._inner = pybids_rs.BIDSLayout.load(str(database_path))
        obj._root = obj._inner.root()
        return obj

    def save(self, database_path, replace_connection=True):  # noqa: ARG002
        """Persist the index to disk."""
        self._inner.save(str(database_path))

    # ── Identity ──

    @property
    def root(self):
        """Dataset root."""
        return self._root

    def __repr__(self):
        n_subjects = len(self.get_subjects())
        return f'BIDSLayout: root={self._root} | Subjects: {n_subjects}'

    def __len__(self):
        return len(self.get_files())

    @property
    def files(self):
        """Mapping of path to :class:`BIDSFile` for every indexed file."""
        return {p: make_bidsfile(_Stub(p), self) for p in self._inner.get_files()}

    @property
    def entities(self):
        """Mapping of entity name to :class:`Entity`."""
        return self.get_entities()

    @property
    def description(self):
        """Parsed ``dataset_description.json``."""
        return self._inner.get_dataset_description()

    # ── Querying ──

    def get(self, return_type='object', target=None, scope='all', regex_search=False, **filters):  # noqa: ARG002
        """Query files by entity values."""
        kwargs = dict(filters)
        kwargs['return_type'] = return_type
        if target is not None:
            kwargs['target'] = target
        if scope is not None:
            kwargs['scope'] = scope

        if return_type in ('id', 'dir') and target is None:
            raise TargetError(f"return_type={return_type!r} requires a 'target' argument")

        results = self._inner.get(**kwargs)
        if return_type in ('object', None):
            return [make_bidsfile(f, self) for f in results]
        return results

    def get_files(self, scope='all'):  # noqa: ARG002
        """Every indexed file path."""
        return self._inner.get_files()

    def get_file(self, filename, scope='all'):  # noqa: ARG002
        """A single file by path."""
        inner = self._inner.get_file(str(filename))
        return None if inner is None else make_bidsfile(inner, self)

    def get_subjects(self, **filters):
        """Distinct subject labels."""
        return self._inner.get_subjects() if not filters else self.get(
            return_type='id', target='subject', **filters
        )

    def get_sessions(self, **filters):
        """Distinct session labels."""
        return self._inner.get_sessions() if not filters else self.get(
            return_type='id', target='session', **filters
        )

    def get_tasks(self, **filters):
        """Distinct task labels."""
        return self._inner.get_tasks() if not filters else self.get(
            return_type='id', target='task', **filters
        )

    def get_runs(self, **filters):
        """Distinct run labels."""
        return self._inner.get_runs() if not filters else self.get(
            return_type='id', target='run', **filters
        )

    def get_datatypes(self, **filters):
        """Distinct datatypes."""
        return self._inner.get_datatypes() if not filters else self.get(
            return_type='id', target='datatype', **filters
        )

    def get_suffixes(self, **filters):
        """Distinct suffixes."""
        return self._inner.get_suffixes() if not filters else self.get(
            return_type='id', target='suffix', **filters
        )

    def get_entities(self, scope='all', metadata=None):  # noqa: ARG002
        """Mapping of entity name to :class:`Entity`.

        Follows PyBIDS' ``metadata`` semantics: ``None`` returns every
        available entity — the filename vocabulary plus any keys indexed from
        JSON sidecars; ``True`` returns only the metadata-derived keys;
        ``False`` returns only the filename entities.
        """
        configured = set(self._inner.get_configured_entities())
        present = set(self._inner.get_entities())
        if metadata is True:
            names = present - configured
        elif metadata is False:
            names = configured
        else:
            names = configured | present
        return {name: Entity(name, self) for name in sorted(names)}

    def get_entity_values(self, entity):
        """Distinct values for one entity."""
        return self._inner.get_entity_values(entity)

    # ── Metadata and associations ──

    def get_metadata(self, path, include_entities=False, scope='all'):  # noqa: ARG002
        """Sidecar metadata for a file, with BIDS inheritance applied."""
        md = BIDSMetadata(str(path))
        md.update(self._inner.get_metadata(str(path)))
        return md

    def get_dataset_description(self, scope='self', all_=False):  # noqa: ARG002
        """Parsed ``dataset_description.json``."""
        return self._inner.get_dataset_description()

    def get_associations(self, path, kind=None, include_parents=False):  # noqa: ARG002
        """Files associated with a given file."""
        return [make_bidsfile(f, self) for f in self._inner.get_associations(str(path), kind)]

    def get_bval(self, path, **filters):  # noqa: ARG002
        """The ``.bval`` for a DWI image."""
        f = self._inner.get_bval(str(path))
        return None if f is None else f.path

    def get_bvec(self, path, **filters):  # noqa: ARG002
        """The ``.bvec`` for a DWI image."""
        f = self._inner.get_bvec(str(path))
        return None if f is None else f.path

    def get_fieldmap(self, path, return_list=False):
        """Fieldmaps whose ``IntendedFor`` covers a file."""
        fmaps = self._inner.get_fieldmap(str(path))
        if return_list:
            return fmaps
        return fmaps[0] if fmaps else None

    def get_nearest(self, path, return_type='filename', strict=True, all_=False, ignore_strict_entities='extension', full_search=False, **filters):  # noqa: ARG002
        """The file closest to ``path`` in entity space."""
        f = self._inner.get_nearest(str(path), **filters)
        if f is None:
            return None
        return f.path if return_type in ('filename', 'file') else make_bidsfile(f, self)

    def get_tr(self, derivatives=False, **filters):  # noqa: ARG002
        """Repetition time for matching functional runs."""
        try:
            return self._inner.get_tr(**filters)
        except Exception as exc:  # noqa: BLE001
            raise NoMatchError(str(exc)) from exc

    def get_collections(self, level, types=None, variables=None, merge=False, sampling_rate=None, skip_empty=False, **kwargs):  # noqa: ARG002
        """Variable collections at a level of the BIDS hierarchy."""
        return self._inner.get_collections(level=level, merge=merge)

    # ── Paths ──

    def parse_file_entities(self, filename, scope='all', entities=None, config=None, include_unmatched=False):  # noqa: ARG002
        """Parse BIDS entities out of a filename."""
        return self._inner.parse_file_entities(str(filename))

    def build_path(self, source, path_patterns=None, strict=False, scope='all', validate=True, absolute_paths=None):  # noqa: ARG002
        """Build a BIDS path from entities."""
        entities = source if isinstance(source, dict) else source.entities
        return self._inner.build_path(
            {k: str(v) for k, v in entities.items()},
            path_patterns=list(path_patterns) if path_patterns else None,
            strict=strict,
        )

    def copy_files(self, files=None, path_patterns=None, symbolic_links=True, root=None, conflicts='fail', **kwargs):  # noqa: ARG002
        """Copy matching files into ``root`` using ``path_patterns``."""
        return self._inner.copy_files(
            list(path_patterns or []),
            root=str(root) if root else None,
            symbolic_links=symbolic_links,
            **kwargs,
        )

    def write_to_file(self, entities, path_patterns=None, contents=None, link_to=None, copy_from=None, content_mode='text', conflicts='fail', strict=False, validate=True):  # noqa: ARG002
        """Write contents to a path built from entities."""
        return self._inner.write_to_file(
            {k: str(v) for k, v in entities.items()},
            contents or '',
            path_patterns=list(path_patterns) if path_patterns else None,
            strict=strict,
        )

    # ── Derivatives ──

    def add_derivatives(self, path, parent_database_path=None, **kwargs):  # noqa: ARG002
        """Index a derivatives dataset alongside this one."""
        for p in (path if isinstance(path, (list, tuple)) else [path]):
            self._inner.add_derivatives(str(p))

    @property
    def derivatives(self):
        """Registered derivative pipeline names."""
        return {name: None for name in self._inner.get_derivatives()}

    # ── Export ──

    def to_df(self, metadata=False, scope='all', **filters):  # noqa: ARG002
        """Index as a pandas DataFrame of (path, entity, value) rows."""
        import pandas as pd

        rows = self._inner.to_df(metadata)
        return pd.DataFrame(rows, columns=['path', 'entity', 'value'])

    def clone(self):
        """An independent copy of this layout."""
        obj = type(self).__new__(type(self))
        obj._inner = self._inner.clone_layout()
        obj._root = self._root
        return obj


class _Stub:
    """Minimal stand-in so ``files`` can hand back path-only file objects."""

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.entities = {}

    def relpath(self):
        return self.path


class BIDSValidator:
    """Filename-level BIDS validation."""

    def __init__(self, index_associated=True):
        self.index_associated = index_associated

    def is_bids(self, path):
        """Whether a path is a valid BIDS filename."""
        return pybids_rs.validate_bids_path(str(path))
