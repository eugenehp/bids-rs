"""Utilities mirroring ``bids.utils``."""

import re

__all__ = ['convert_JSON', 'listify', 'matches_entities', 'natural_sort', 'splitext']


def listify(obj):
    """Wrap a scalar in a list; pass lists and tuples through."""
    return obj if isinstance(obj, (list, tuple)) else [obj]


def matches_entities(obj, entities, strict=False):
    """Whether an object's entities match the supplied criteria."""
    obj_entities = getattr(obj, 'entities', {})
    if strict and set(obj_entities.keys()) != set(entities.keys()):
        return False
    for name, val in entities.items():
        if name not in obj_entities:
            return False
        if str(obj_entities[name]) not in {str(v) for v in listify(val)}:
            return False
    return True


def natural_sort(seq, field=None):
    """Sort strings with embedded numbers in numeric order."""

    def key(item):
        text = getattr(item, field) if field is not None else item
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split('([0-9]+)', str(text))
        ]

    return sorted(seq, key=key)


def splitext(path):
    """Split a path into its stem followed by each extension segment."""
    import os

    base = os.path.basename(path)
    parts = base.split(os.extsep)
    return [os.path.join(os.path.dirname(path), parts[0]), *parts[1:]]


def convert_JSON(j):  # noqa: N802 — name mirrors PyBIDS
    """Recursively convert CamelCase JSON keys to snake_case."""

    def to_snake(name):
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()

    if isinstance(j, dict):
        return {to_snake(k): convert_JSON(v) for k, v in j.items()}
    if isinstance(j, list):
        return [convert_JSON(v) for v in j]
    return j
