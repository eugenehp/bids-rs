"""Package-wide options mirroring ``bids.config``."""

import json
from pathlib import Path

__all__ = ['from_file', 'get_option', 'reset_options', 'set_option', 'set_options']

_default_settings = {'config_paths': {}, 'extension_initial_dot': True}
_settings = dict(_default_settings)


def set_option(key, value):
    """Set a package-wide option."""
    if key not in _settings:
        raise ValueError(f"Invalid setting: '{key}'")
    _settings[key] = value


def set_options(**kwargs):
    """Set several package-wide options."""
    for k, v in kwargs.items():
        set_option(k, v)


def get_option(key):
    """Read a package-wide option."""
    if key not in _settings:
        raise ValueError(f"Invalid setting: '{key}'")
    return _settings[key]


def from_file(filenames, error_on_missing=True):
    """Load settings from one or more JSON files."""
    from .utils import listify

    for f in listify(filenames):
        path = Path(f)
        if path.exists():
            _settings.update(json.loads(path.read_text(encoding='utf-8')))
        elif error_on_missing:
            raise ValueError(f"Config file '{f}' does not exist.")


def reset_options(update_from_file=False):  # noqa: ARG001
    """Reset every option to its default."""
    global _settings  # noqa: PLW0603
    _settings = dict(_default_settings)
