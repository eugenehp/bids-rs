"""Path construction mirroring ``bids.layout.writing``."""

__all__ = ['build_path', 'write_to_file']


def build_path(entities, path_patterns, strict=False):
    """Build a BIDS path from entities and a list of patterns."""
    raise NotImplementedError(
        'Use BIDSLayout.build_path; the free function is not wired in the bids-rs shim'
    )


def write_to_file(path, contents=None, link_to=None, copy_from=None, conflicts='fail'):  # noqa: ARG001
    """Write contents to a path."""
    import os

    if os.path.exists(path) and conflicts == 'fail':
        raise ValueError(f'A file at path {path} already exists.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'wb' if isinstance(contents, bytes) else 'w'
    with open(path, mode) as fh:
        fh.write(contents)
