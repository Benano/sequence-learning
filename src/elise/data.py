#!/usr/bin/env python3
"""
Refactored pattern and dataloader system.

Architecture:
1. Sequence: Raw data representation (no time)
2. Pattern: Adds temporal information
3. Dataloader: Handles sampling and transformations
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt

# ============================================================================
# LAYER 1: SEQUENCES (no temporal information)
# ============================================================================


class Sequence(ABC):
    """Base class for sequences without temporal information."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of elements in sequence."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> npt.NDArray:
        """Return element at index."""
        pass

    def slice(self, start: int, stop: Optional[int] = None) -> "Sequence":
        """Return a sliced version of the sequence."""
        return SlicedSequence(self, start, stop)


class SlicedSequence(Sequence):
    """Wrapper for sliced sequences."""

    def __init__(self, parent: Sequence, start: int, stop: Optional[int] = None):
        self.parent = parent
        self.start = start
        self.stop = stop if stop is not None else len(parent)

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, idx: int) -> npt.NDArray:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for sequence of length {len(self)}"
            )
        return self.parent[self.start + idx]


class MultiHotSequence(Sequence):
    """
    Multi-hot encoded sequence.

    Example: [2, 4, [0, 3], [0, 3], -1] represents a melody where
    multiple notes can be active simultaneously.
    """

    def __init__(self, elements: List[Union[int, List[int]]], width: int):
        """
        Initialize multi-hot sequence.

        :param elements: List of indices or lists of indices (-1 for silence)
        :param width: Width of the one-hot encoding
        """
        self._elements = elements
        self.width = width
        self._validate()
        self._cache = {}  # Cache converted elements

    def _validate(self):
        """Validate that all indices are within width."""
        max_val = -1
        for elem in self._elements:
            if isinstance(elem, int) and elem > max_val and elem != -1:
                max_val = elem
            elif isinstance(elem, list):
                max_val = max(max_val, max(elem))

        if max_val >= self.width:
            raise ValueError(f"Maximum index {max_val} exceeds width {self.width}")

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, idx: int) -> npt.NDArray:
        """Return one-hot encoded vector at index."""
        if idx in self._cache:
            return self._cache[idx]

        vec = np.zeros(self.width, dtype=np.float64)
        elem = self._elements[idx]

        if isinstance(elem, int):
            if elem != -1:  # -1 represents silence
                vec[elem] = 1.0
        elif isinstance(elem, list):
            vec[elem] = 1.0

        self._cache[idx] = vec
        return vec


class OneHotSequence(Sequence):
    """One-hot encoded sequence (only one element active at a time)."""

    def __init__(self, elements: List[int], width: int):
        """
        Initialize one-hot sequence.

        :param elements: List of indices
        :param width: Width of the one-hot encoding
        """
        self._elements = np.array(elements)
        self.width = width

        if np.max(self._elements) >= width:
            raise ValueError(
                f"Maximum index {np.max(self._elements)} exceeds width {width}"
            )

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, idx: int) -> npt.NDArray:
        """Return one-hot encoded vector at index."""
        vec = np.zeros(self.width, dtype=np.float64)
        vec[self._elements[idx]] = 1.0
        return vec


class FunctionSequence(Sequence):
    """
    Sequence defined by a continuous function.

    This bridges discrete and continuous patterns.
    """

    def __init__(
        self,
        func: Callable[[float], npt.NDArray],
        duration: float,
        num_samples: Optional[int] = None,
    ):
        """
        Initialize function-based sequence.

        :param func: Function that takes time (0 to 1) and returns array
        :param duration: Total duration for normalization purposes
        :param num_samples: If provided, precompute this many samples
        """
        self.func = func
        self.duration = duration
        self._num_samples = num_samples
        self._cache = None

        if num_samples is not None:
            self._precompute()

    def _precompute(self):
        """Precompute function values."""
        times = np.linspace(0, 1, self._num_samples)
        self._cache = np.array([self.func(t) for t in times])

    def __len__(self) -> int:
        if self._num_samples is None:
            raise ValueError("FunctionSequence without num_samples has no fixed length")
        return self._num_samples

    def __getitem__(self, idx: int) -> npt.NDArray:
        if self._cache is not None:
            return self._cache[idx]

        # On-the-fly computation
        if self._num_samples is None:
            raise ValueError("Cannot index FunctionSequence without num_samples")

        t = idx / self._num_samples
        return self.func(t)


# ============================================================================
# LAYER 2: PATTERNS (adds temporal information)
# ============================================================================


class Pattern(ABC):
    """Base class for patterns with temporal information."""

    @abstractmethod
    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern value at time t."""
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """Total duration of pattern."""
        pass


class DiscretePattern(Pattern):
    """Pattern based on a discrete sequence with temporal sampling."""

    def __init__(self, sequence: Sequence, duration: float):
        """
        Initialize discrete pattern.

        :param sequence: The underlying sequence
        :param duration: Total duration in ms
        """
        self.sequence = sequence
        self._duration = duration
        self.dt = duration / len(sequence)

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern at time t (with wrapping)."""
        t_wrapped = t % self._duration
        idx = int(t_wrapped / self.dt)
        idx = min(idx, len(self.sequence) - 1)  # Clamp to valid range
        return self.sequence[idx]


class ContinuousPattern(Pattern):
    """Pattern based on a continuous function."""

    def __init__(self, func: Callable[[float], npt.NDArray], duration: float):
        """
        Initialize continuous pattern.

        :param func: Function that takes time and returns array
        :param duration: Total duration (for wrapping)
        """
        self.func = func
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern at time t (with wrapping)."""
        t_wrapped = t % self._duration
        return self.func(t_wrapped)


# ============================================================================
# LAYER 3: DATALOADER (handles sampling and transformations)
# ============================================================================


class Dataloader:
    """
    Unified dataloader for both discrete and continuous patterns.

    Handles transformation pipeline and iteration.
    """

    def __init__(
        self,
        pattern: Pattern,
        transforms: Optional[List[Callable]] = None,
        online_transforms: Optional[List[Callable]] = None,
        output_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataloader.

        :param pattern: Pattern object to load from
        :param transforms: Pre-computed transforms (applied once at init)
        :param online_transforms: Online transforms (applied each call)
        :param output_transform: Final transform (e.g., to_biounits)
        """
        self.pattern = pattern
        self.duration = pattern.duration

        # Apply pre-transforms if pattern is discrete (can be precomputed)
        if isinstance(pattern, DiscretePattern) and transforms:
            self._apply_pretransforms(transforms)

        self.online_transforms = online_transforms or []
        self.output_transform = output_transform

    def _apply_pretransforms(self, transforms: List[Callable]):
        """Apply transforms to discrete pattern (modifies in place)."""
        # This could be optimized by caching the entire pattern
        # For now, transforms will be applied on-the-fly
        pass

    def __call__(self, t: float) -> npt.NDArray:
        """
        Get pattern at time t with all transforms applied.

        :param t: Time in ms
        :return: Transformed pattern array
        """
        # Get base pattern
        output = self.pattern(t)

        # Apply online transforms
        for transform in self.online_transforms:
            output = transform(output)

        # Apply output transform (e.g., to_biounits)
        if self.output_transform is not None:
            output = self.output_transform(output)

        return output

    def iter(self, t_start: float, t_stop: float, dt: float):
        """
        Iterate over pattern in time range.

        :param t_start: Start time
        :param t_stop: Stop time
        :param dt: Time step
        :yield: (time, pattern) tuples
        """
        t = t_start
        while t < t_stop:
            yield t, self(t)
            t += dt

    def get_full_pattern(self, dt: float) -> npt.NDArray:
        """
        Get full pattern as array.

        :param dt: Sampling time step
        :return: Array of pattern over full duration
        """
        samples = []
        for _, pattern in self.iter(0, self.duration, dt):
            samples.append(pattern)
        return np.array(samples)


# ============================================================================
# LAYER 4: PATTERN COMPOSITORS (compose multiple patterns)
# ============================================================================


class ConcatenatedPattern(Pattern):
    """Pattern that concatenates multiple patterns in sequence."""

    def __init__(self, patterns: List[Pattern]):
        """
        Initialize concatenated pattern.

        :param patterns: List of patterns to concatenate
        """
        self.patterns = patterns
        self._duration = sum(p.duration for p in patterns)

        # Precompute boundaries
        self._boundaries = [0]
        for p in patterns:
            self._boundaries.append(self._boundaries[-1] + p.duration)

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern at time t."""
        t_wrapped = t % self._duration

        # Find which pattern we're in
        for i, (start, end) in enumerate(
            zip(self._boundaries[:-1], self._boundaries[1:])
        ):
            if start <= t_wrapped < end:
                # Time relative to this pattern's start
                t_local = t_wrapped - start
                return self.patterns[i](t_local)

        # If we're exactly at the end, return last pattern's last value
        return self.patterns[-1](self.patterns[-1].duration - 1e-6)


class ShuffledPattern(Pattern):
    """Pattern that randomly concatenates patterns to fill a duration."""

    def __init__(
        self, patterns: List[Pattern], total_duration: float, seed: Optional[int] = None
    ):
        """
        Initialize shuffled pattern.

        :param patterns: List of patterns to shuffle and concatenate
        :param total_duration: Total duration to fill
        :param seed: Random seed for reproducibility
        """
        self.patterns = patterns
        self._duration = total_duration
        self._rng = np.random.default_rng(seed)

        # Build the shuffled sequence
        self._sequence = []  # List of (pattern_idx, start_time)
        self._boundaries = [0]

        current_time = 0
        while current_time < total_duration:
            # Pick random pattern
            idx = self._rng.choice(len(patterns))
            pattern = patterns[idx]

            self._sequence.append(idx)
            current_time += pattern.duration
            self._boundaries.append(current_time)

        # Trim last boundary to exact duration
        self._boundaries[-1] = total_duration

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern at time t."""
        t_wrapped = t % self._duration

        # Find which pattern segment we're in
        for i, (start, end) in enumerate(
            zip(self._boundaries[:-1], self._boundaries[1:])
        ):
            if start <= t_wrapped < end:
                pattern_idx = self._sequence[i]
                pattern = self.patterns[pattern_idx]

                # Time relative to this pattern's start
                t_local = t_wrapped - start

                # Wrap if we exceed pattern duration (for last truncated pattern)
                if t_local >= pattern.duration:
                    t_local = pattern.duration - 1e-6

                return pattern(t_local)

        # Fallback
        return self.patterns[self._sequence[-1]](0)


class RepeatedPattern(Pattern):
    """Pattern that repeats a single pattern N times."""

    def __init__(self, pattern: Pattern, num_repeats: int):
        """
        Initialize repeated pattern.

        :param pattern: Pattern to repeat
        :param num_repeats: Number of times to repeat
        """
        self.pattern = pattern
        self.num_repeats = num_repeats
        self._duration = pattern.duration * num_repeats

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return pattern at time t."""
        t_wrapped = t % self._duration
        t_in_pattern = t_wrapped % self.pattern.duration
        return self.pattern(t_in_pattern)


class RandomPattern(Pattern):
    """Pattern that randomly samples from multiple patterns at each time."""

    def __init__(
        self, patterns: List[Pattern], duration: float, seed: Optional[int] = None
    ):
        """
        Initialize random pattern.

        :param patterns: List of patterns to sample from
        :param duration: Total duration
        :param seed: Random seed
        """
        self.patterns = patterns
        self._duration = duration
        self._rng = np.random.default_rng(seed)

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> npt.NDArray:
        """Return random pattern at time t."""
        # Pick random pattern based on time (for some consistency)
        idx = int(t * 1000) % len(self.patterns)  # Change every ms
        return self.patterns[idx](t % self.patterns[idx].duration)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_multihot_from_file(
    filepath: str, width: int, delimiter: str = ",", skiprows: int = 1
) -> MultiHotSequence:
    """
    Load multi-hot sequence from file.

    :param filepath: Path to file
    :param width: Width of encoding
    :param delimiter: CSV delimiter
    :param skiprows: Rows to skip
    :return: MultiHotSequence
    """
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows).astype(int)
    # Convert to list format expected by MultiHotSequence
    elements = data.tolist()
    return MultiHotSequence(elements, width)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Multi-hot pattern with transforms
    melody = MultiHotSequence([2, 4, [0, 3], [0, 3], -1], width=13)
    pattern = DiscretePattern(melody, duration=250.0)

    def to_biounits(x, E_l=-70.0):
        return E_l + x * 20.0

    def add_noise(x):
        return x + np.random.normal(0, 0.01, x.shape)

    dataloader = Dataloader(
        pattern,
        online_transforms=[add_noise],
        output_transform=lambda x: to_biounits(x, E_l=-70.0),
    )

    # Use in simulation
    for t, target in dataloader.iter(0, 500.0, dt=0.01):
        # network(u_inp=target)
        pass

    # Example 2: Sliced pattern (for long preludes)
    prelude = OneHotSequence(list(range(300)), width=300)
    excerpt = prelude.slice(0, 250)  # First 250 notes
    pattern = DiscretePattern(excerpt, duration=500.0)
    dataloader = Dataloader(pattern)

    # Example 3: Continuous pattern
    def circle(t):
        return np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])

    pattern = ContinuousPattern(circle, duration=1000.0)
    dataloader = Dataloader(pattern)

    breakpoint()

    # Example 4: Concatenated patterns
    melody1 = DiscretePattern(MultiHotSequence([1, 2, 3], width=13), duration=100)
    melody2 = DiscretePattern(MultiHotSequence([4, 5, 6], width=13), duration=150)
    melody3 = DiscretePattern(MultiHotSequence([7, 8, 9], width=13), duration=200)

    concatenated = ConcatenatedPattern([melody1, melody2, melody3])
    # Total duration = 100 + 150 + 200 = 450ms

    dataloader = Dataloader(concatenated, output_transform=lambda x: to_biounits(x))

    # Example 5: Shuffled patterns (like your ShuffleDataloader)
    shuffled = ShuffledPattern(
        patterns=[melody1, melody2, melody3],
        total_duration=5000,  # Fill 5 seconds with random patterns
        seed=42,
    )

    dataloader = Dataloader(shuffled)

    # Example 6: Repeated pattern
    short_melody = DiscretePattern(MultiHotSequence([1, 2, 3], width=13), duration=100)
    repeated = RepeatedPattern(short_melody, num_repeats=10)
    # Total duration = 100 * 10 = 1000ms

    dataloader = Dataloader(repeated)

    # Example 7: Complex composition
    # Create a training set with shuffled patterns
    patterns = []
    for i in range(10):
        seq = MultiHotSequence([i, i + 1, i + 2], width=13)
        pat = DiscretePattern(seq, duration=200)
        patterns.append(pat)

    # Shuffle them to create varied training data
    training_pattern = ShuffledPattern(patterns, total_duration=10000, seed=42)

    # Apply transforms
    dataloader = Dataloader(
        training_pattern,
        online_transforms=[add_noise],
        output_transform=lambda x: to_biounits(x, E_l=-70.0),
    )

    # Use in training loop
    for epoch in range(10):
        for t, target in dataloader.iter(0, training_pattern.duration, dt=0.01):
            # network(u_inp=target)
            pass
