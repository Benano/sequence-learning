#!/usr/bin/env python3
from types import SimpleNamespace


def dict_to_namespace(data):
    """Recursively converts a dict to a SimpleNamespace."""
    if isinstance(data, dict):
        # Convert all values in the dict first, then wrap in SimpleNamespace
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # Handle lists of dicts if your config has them
        return [dict_to_namespace(i) for i in data]
    else:
        return data
