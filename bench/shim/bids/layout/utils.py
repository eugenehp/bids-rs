"""Helpers mirroring ``bids.layout.utils``."""

from ..exceptions import ConfigError

__all__ = ['BIDSMetadata', 'PaddedInt', 'add_config_paths', 'parse_file_entities']


class BIDSMetadata(dict):
    """A metadata dict that reports the file it came from when a key is missing."""

    def __init__(self, source_file=None, *args, **kwargs):
        self.source_file = source_file
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            raise KeyError(
                f"Metadata term {key!r} unavailable for file {self.source_file}."
            ) from exc


class PaddedInt(int):
    """An integer that remembers its zero-padded string form."""

    def __new__(cls, val):
        if isinstance(val, PaddedInt):
            return val
        s = str(val)
        int_val = super().__new__(cls, s)
        int_val._str = s
        return int_val

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        return super().__eq__(other) or str(self) == other

    def __hash__(self):
        return super().__hash__()


def parse_file_entities(filename, entities=None, config=None, include_unmatched=False):  # noqa: ARG001
    """Parse BIDS entities out of a filename.

    The Rust binding exposes entity parsing as a layout method rather than a
    free function, so this needs a layout: use
    ``BIDSLayout.parse_file_entities`` instead.
    """
    raise NotImplementedError(
        'the bids-rs backend parses entities through BIDSLayout.parse_file_entities; '
        'the free function has no layout to read the entity config from'
    )


def add_config_paths(**kwargs):
    """Register additional named entity-configuration files."""
    from pathlib import Path

    from .. import config as cf

    paths = cf.get_option('config_paths')
    for k, path in kwargs.items():
        if not Path(path).exists():
            raise ConfigError(f'Configuration file "{k}" does not exist')
        if k in paths:
            raise ConfigError(f'Configuration {k!r} already exists')
    cf.set_option('config_paths', {**paths, **kwargs})
