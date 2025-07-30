#!/usr/bin/env python3
import ast

import numpy as np

from elise.data import MultiHotPattern, Pattern


def load_multi_hot_pattern(filename):
    with open(filename, "r") as file:
        content = file.read().strip()

    pat = ast.literal_eval(f"[{content}]")
    return pat


def load_one_hot_pattern(pattern_file):
    pat = np.loadtxt(pattern_file, delimiter=" ").astype(int).T

    return pat


def load_pattern_flexible(pattern_file, pattern_duration, pattern_dt=None):
    """
    Try to load a pattern file as multi-hot; if that fails, load as one-hot.
    Returns a numpy array.
    """
    try:
        # Try multi-hot
        pat = load_multi_hot_pattern(pattern_file)
        pattern = MultiHotPattern(
            pattern=pat,
            duration=pattern_duration,
        )

    except (ValueError, SyntaxError):
        # Try one-hot
        pat = load_one_hot_pattern(pattern_file)

        if len(pat) > pattern_duration:
            pat = pat[: int(pattern_duration), :]
        pat = pat[:, pat.any(axis=0)]

        pattern = Pattern(
            pattern=pat,
            dt=pattern_dt,
        )

    return pattern
