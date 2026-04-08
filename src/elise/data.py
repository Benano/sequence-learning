"""Module for creating datasets and loading them into the model."""

# /usr/bin/env python3

import copy
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class BasePattern(ABC):
    """
    Base Class for all pattern types.

    More specific patterns should inherit from this.
    Defines the interface for all other Pattern classes.

    :ivar _pattern: The original input pattern array
    :vartype _pattern: npt.NDArray
    :ivar pattern: The converted pattern array
    :vartype pattern: npt.NDArray
    :ivar dt: Time step
    :vartype dt: float
    :ivar dur: Total duration of the pattern
    :vartype dur: float
    :ivar shape: Shape of the pattern array
    :vartype shape: tuple
    """

    def __init__(
        self,
        pattern: npt.NDArray,
        dt: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Initialize the Pattern object.

        :param pattern: The input pattern array.
        :type pattern: npt.NDArray
        :param dt: Time step for the pattern (mutually exclusive with 'duration').
        :type dt: Optional[float]
        :param duration: Total duration of the pattern (mutually exclusive with 'dt').
        :type duration: Optional[float]
        :raises ValueError: If neither or both 'dt' and 'duration' are provided.

        Calculates either duration or time step based on provided arguments.

        Attributes:
            _pattern (npt.NDArray): Original input pattern array.
            pattern (npt.NDArray): Converted pattern array.
            dt (float): Time step if specified.
            duration (float): Total duration if specified.
            shape (tuple): Shape of the converted pattern array.
        """
        self._pattern = pattern
        self.pattern = self._convert(pattern)
        if (dt is None and duration is None) or (
            dt is not None and duration is not None
        ):
            raise ValueError("Exactly one of 'dt' or 'duration' must be provided")

        # Use isinstance, and handle int case as desired
        if isinstance(dt, float):
            self.dt = dt
            self.duration = self.dt * self.__len__()
        elif isinstance(duration, float):
            self.duration = duration
            self.dt = self.duration / self.__len__()
        else:
            # Neither dt nor duration is a float
            raise TypeError("dt/duration must be of type float")

        self.shape = self.pattern.shape
        self.width = self.pattern.shape[-1]

    @abstractmethod
    def _convert(self, pattern: npt.NDArray) -> npt.NDArray:
        """
        Convert the input pattern.

        Basically one of the core functionalities of the Pattern-Classes.
        Every Pattern class needs to implement this.

        :param pattern: The input pattern array
        :type pattern: npt.NDArray
        :return: The converted pattern
        :rtype: npt.NDArray
        """
        pass

    def transform(self, transformation: Callable[[npt.NDArray], npt.NDArray]) -> None:
        """Apply a transformation to the whole pattern.

        At the moment, the transformation takes only a pattern, no other arguments.
        """
        self.pattern = transformation(self.pattern)
        self.shape = self.pattern.shape
        self.width = self.pattern.shape[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the Pattern object.

        :return: String representation
        :rtype: str
        """
        return f"{self.__class__.__name__}(pattern={self._pattern}, dt={self.dt})"

    def __str__(self) -> str:
        """
        Return a string representation.
        """
        return "\n".join(["pattern=", str(self.pattern), f"dt={self.dt}"])

    def __len__(self) -> int:
        """
        Return the length of the pattern.

        :return: Length of the pattern
        :rtype: int
        """
        return self.pattern.shape[0]

    def __getitem__(self, idx) -> Any:
        """
        Get item at the specified index.

        :param idx: Index
        :type idx: int
        :return: Item at the specified index
        :rtype: Any
        """
        return self.pattern[idx]


class Pattern(BasePattern):
    """Simplest Pattern class.

    No conversion happens.
    """

    def _convert(self, pattern: npt.NDArray):
        """No conversion."""
        return pattern


class RandomPattern(BasePattern):
    def __init__(self, duration, width, rng, dt: float = 1.0):
        self.nr_notes = int(duration)
        self.width = width
        self.dt = dt
        self.rng = rng
        self.duration = self.nr_notes * self.dt

        pattern = np.zeros((self.nr_notes, self.width), dtype=int)
        pattern[0] = rng.integers(0, 2, self.width)
        self.trans_p = np.zeros((2, self.width), dtype=float)
        self.trans_p[0] = self.rng.uniform(0.1, 0.8, self.width)
        self.trans_p[1] = self.rng.uniform(0.1, 0.2, self.width)
        for i in range(self.nr_notes - 1):
            c_probs = np.where(pattern[i], self.trans_p[0], self.trans_p[1])
            random_vals = self.rng.uniform(size=self.width)
            result = (random_vals < c_probs).astype(int)
            pattern[i + 1] = result

        super().__init__(pattern=pattern, dt=dt)

    def _convert(self, pattern: np.ndarray) -> np.ndarray:
        # No conversion in generator; just return the input
        return pattern


class RandomSampledNonMarkovianPattern(BasePattern):
    def __init__(self, duration, width, rng, dt: float = 1.0, nmk=2):
        self.nr_notes = int(duration)
        self.width = width
        self.dt = dt
        self.rng = rng
        self.duration = self.nr_notes * self.dt
        self.nmk = nmk

        pattern = np.zeros((self.nr_notes + self.nmk, self.width), dtype=int)
        pattern[0 : self.nmk] = rng.integers(0, 2, (self.nmk, self.width))

        self.trans_p = np.zeros((2, self.width), dtype=float)
        self.trans_p[0] = self.rng.uniform(0.1, 0.8, self.width)
        self.trans_p[1] = self.rng.uniform(0.1, 0.2, self.width)
        for i in range(self.nr_notes - 1):
            c_probs = np.where(pattern[i - self.nmk], self.trans_p[0], self.trans_p[1])
            random_vals = self.rng.uniform(size=self.width)
            result = (random_vals < c_probs).astype(int)
            pattern[i + 1 + self.nmk] = result

        pattern = pattern[self.nmk :, :]

        super().__init__(pattern=pattern, dt=dt)

    def _convert(self, pattern: np.ndarray) -> np.ndarray:
        # No conversion in generator; just return the input
        return pattern


class RandomCopiedNonMarkovianPattern(BasePattern):
    def __init__(self, duration, width, rng, dt: float = 1.0, nmk=2):
        self.nr_notes = int(duration)
        self.width = width
        self.dt = dt
        self.rng = rng
        self.duration = self.nr_notes * self.dt

        pattern = np.zeros((self.nr_notes, self.width), dtype=int)
        pattern[0] = rng.integers(0, 2, self.width)
        self.trans_p = np.zeros((2, self.width), dtype=float)
        self.trans_p[0] = self.rng.uniform(0.1, 0.8, self.width)
        self.trans_p[1] = self.rng.uniform(0.1, 0.2, self.width)
        for i in range(self.nr_notes - 1):
            c_probs = np.where(pattern[i], self.trans_p[0], self.trans_p[1])
            random_vals = self.rng.uniform(size=self.width)
            result = (random_vals < c_probs).astype(int)
            pattern[i + 1] = result

        index_first_proportion = int(len(pattern) * 0.0)
        non_markov_chunk = pattern[
            index_first_proportion : index_first_proportion + nmk, :
        ]

        index_last_proportion = int(len(pattern) * 0.5)
        pattern[
            index_last_proportion : index_last_proportion + nmk, :
        ] = non_markov_chunk

        super().__init__(pattern=pattern, dt=dt)

    def _convert(self, pattern: np.ndarray) -> np.ndarray:
        # No conversion in generator; just return the input
        return pattern


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el


class OneHotPattern(BasePattern):
    """
    Turn a sequential pattern into a one-hot encoded pattern.

    This class extends the base Pattern class to create one-hot encoded patterns.
    """

    def __init__(self, pattern: npt.NDArray, dt: float) -> None:
        """
        Initialize the OneHotPattern object.

        :param pattern: The input sequential pattern array
        :type pattern: npt.NDArray
        :param dt: Time step
        :type dt: float
        """

        self.width = np.max(pattern) + 1
        super().__init__(pattern, dt)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self.width))
        for i, j in enumerate(pattern):
            res[i, j] = 1.0
        return res


def flatten_nested_tuples(lst):
    for el in lst:
        if isinstance(el, (tuple, list)):
            yield from flatten_nested_tuples(el)
        else:
            yield el


class MultiHotPattern(BasePattern):
    """
    Same as OneHot, but more then one can be active at a time.

    Encode it as:
    [2, 4, [0, 3], [0, 3], -1]

    If an entry is -1, there's no, the whole vector is zero (= a pause in the pattern)
    """

    def __init__(
        self,
        pattern: List[int | List[int]],
        duration: float,
    ) -> None:
        """DOCSTRING."""
        self.width = max(flatten_nested_tuples(pattern)) + 1
        super().__init__(pattern, duration=duration)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self.width))
        for i, pat in enumerate(pattern):
            if isinstance(pat, int) and pat == -1:
                continue
            else:
                res[i, pat] = 1.0
        return res


class BaseContinuousPattern(ABC):
    """DOCSTRING."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass

    @abstractmethod
    def __call__(self, t: float) -> npt.NDArray:
        """DOCSTRING."""
        pass


class CirclePattern(BaseContinuousPattern):
    def __init__(self, radius: float, center_x: float, center_y: float, period: float):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self._period = period
        self.width = 2

    @property
    def duration(self):
        return self._period

    def __call__(self, t):
        x = self.center_x + self.radius * np.cos(2 * np.pi * t / self._period)
        y = self.center_y + self.radius * np.sin(2 * np.pi * t / self._period)

        return np.array([x, y])


class LorenzAttractor:
    def __init__(
        self,
        sigma: float = 10,
        rho: float = 28,
        beta: float = 8 / 3,
        x0: float = 0,
        y0: float = 0,
        z0: float = 0,
        t0: float = 0.0,
        duration: float = 10.0,
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x = x0
        self.y = y0
        self.z = z0
        self.last_t = t0
        self._duration = duration
        self.width = 3

    @property
    def duration(self):
        return self._duration

    def __call__(self, t: float):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z

        dt = t - self.last_t
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        self.last_t = t

        return np.array([self.x, self.y, self.z])


class BaseDataloader(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, t: float, offset: float = 0.0) -> npt.NDArray:
        pass

    @abstractmethod
    def iter(self, t_start: float, t_stop: float, dt: float) -> npt.NDArray:
        pass

    @abstractmethod
    def get_full_pattern(self, dt) -> npt.NDArray:
        pass

    def save(self, path: str) -> None:
        """
        Save the dataloader.

        :param path: File path to save the loader.
        :type path: str
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a tracker object from a file.

        :param path: File path to load the loader from.
        :type path: str
        :return: The loaded loader object.
        :rtype: Tracker
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class DiscreteDataloader(BaseDataloader):
    """
    Dataloader for pattern data.

    Has a call-method for passing the correct pattern at time t.
    And an iter-method that acts as an iterator.
    Handles pre-transforms and online-transforms for pattern data.

    :ivar pat: The pattern object
    :vartype pat: Pattern
    :ivar dur: Duration of the pattern
    :vartype dur: float
    :ivar dt: Time step of the pattern
    :vartype dt: float
    :ivar online_transforms: List of online transforms to be applied
    :vartype online_transforms: List[Callable]
    """

    def __init__(
        self,
        pattern: Pattern,
        pre_transforms: List[Callable] = [],
        online_transforms: List[Callable] = [],
    ) -> None:
        """
        Initialize the Dataloader object.

        :param pattern: The pattern object
        :type pattern: Pattern
        :param pre_transforms: List of pre-transforms to be applied once, defaults to []
        :type pre_transforms: List[Callable], optional
        :param online_transforms: List of online transforms to be applied on each call, defaults to []  # noqa
        :type online_transforms: List[Callable], optional
        :raises ValueError: If a pre-transform changes the shape of the pattern
        """

        self.pattern = pattern
        # apply pre-transforms directly once
        for transform in pre_transforms:
            self.pattern.transform(transform)

        self.duration = self.pattern.duration
        self.dt = self.pattern.dt
        self.width = self.pattern.width
        self.online_transforms = online_transforms
        self.apply_online_transforms = 1

    def _time_to_idx(self, t: float) -> int:
        return int((t % self.duration) / self.dt)

    def _apply_online_transforms(self, pattern_1d):
        for transform in self.online_transforms:
            pattern_1d = transform(pattern_1d)
        return pattern_1d

    def __call__(self, t: float, offset: float = 1e-6):
        """
        Return the correct pattern at time t.

        The offset is a small value added to t to make sure that the pattern is
        read "in the middle". This is to prevent that floating point glitches read
        from the wrong pattern.

        :param t: Time
        :type t: float
        :param offset: Time offset, defaults to 1e-6
        :type offset: float, optional
        :return: Pattern at time t
        :rtype: npt.NDArray

        Example:
            for t in np.arange(0, 100, 0.1):
                pattern = dataloader(t)
                my_simulation.step(t, u_inp=pattern,...)
        """
        idx = self._time_to_idx(t + offset)
        pattern_t = self.pattern[idx]

        if self.apply_online_transforms:
            pattern_t = self._apply_online_transforms(pattern_t)

        return pattern_t

    def iter(self, t_start, t_stop, dt):
        """
        Use dataloader as an iterator/iterable.

        :param t_start: Start time
        :type t_start: float
        :param t_stop: Stop time
        :type t_stop: float
        :param dt: Time step
        :type dt: float
        :yield: Tuple of time and pattern
        :rtype: Tuple[float, npt.NDArray]

        Example:
            for t, pattern in dataloader.iter(t_start, t_stop, dt):
                my_simulation.step(t, u_inp=pattern,...)
        """
        t = t_start
        while t < t_stop:
            yield t, self.__call__(t, offset=dt * 0.01)
            t += dt

    def get_full_pattern(self, dt, num=1, online_transforms=True):
        """Return the full pattern."""
        """
        :param dt: Simulation time step
        :type dt: float
        :param num: Number of patterns repetitions
        :type dt: int
        :param online_transforms: Simulation time step
        :type dt: bool
        :return: Full pattern
        :rtype: npt.NDArray
        """

        if not online_transforms:
            self.apply_online_transforms = 0

        full_pattern = []
        for _, pattern in self.iter(0, self.duration - 0.5 * dt, dt):
            full_pattern.append(pattern)

        self.apply_online_transforms = 1  # reset to default

        return np.array(full_pattern)


class ShuffleDataloader(BaseDataloader):
    """
    A dataloader that shuffles patterns and applies transformations.

    This class extends `BaseDataloader` and is designed to load data patterns,
    shuffle them, and apply transformations both preemptively and online.

    :param pattern: List of patterns to shuffle and iterate over.
    :type pattern: List[Pattern]
    :param t_max: Maximum time for the sequence.
    :type t_max: float
    :param t_start: Starting time for the sequence, defaults to 0.0.
    :type t_start: float, optional
    :param seed: Seed for random number generation, defaults to None.
    :type seed: Optional[int], optional
    :param pre_transforms: List of preprocessing transformations to apply once.
    :type pre_transforms: List[Callable]
    :param online_transforms: List of online transformations to apply during iteration,
    defaults to an empty list.
    :type online_transforms: List[Callable]
    """

    def __init__(
        self,
        pattern: List[Pattern],
        t_max: float,
        t_start: float = 0.0,
        seed: Optional[int] = None,
        pre_transforms: List[Callable] = [],
        online_transforms: List[Callable] = [],
    ) -> None:
        """
        Initialize the ShuffleDataloader instance.

        Creates a shuffled sequence of patterns and applies pre-transforms.

        :param pattern: List of patterns to shuffle and iterate over.
        :type pattern: List[Pattern]
        :param t_max: Maximum time for the sequence.
        :type t_max: float
        :param t_start: Starting time for the sequence, defaults to 0.0.
        :type t_start: float, optional
        :param seed: Seed for random number generation, defaults to None.
        :type seed: Optional[int], optional
        :param pre_transforms: List of preprocessing transformations to apply once.
        :type pre_transforms: List[Callable]
        :param online_transforms: List of online transformations to apply during
        iteration, defaults to an empty list.
        :type online_transforms: List[Callable]
        """
        self.pattern = copy.deepcopy(pattern)
        self.num_pattern = len(pattern)
        self.dts = [pat.dt for pat in self.pattern]
        self.durations = [pat.duration for pat in self.pattern]
        self.t_max = t_max
        self.online_transforms = online_transforms

        # apply pre-transforms directly once
        for i in range(self.num_pattern):
            for transform in pre_transforms:
                self.pattern[i].transform(transform)

        # Interface attributes expected by the training pipeline
        self.width = self.pattern[0].width
        self.dt = self.pattern[0].dt
        self.duration = self.durations[
            0
        ]  # single-pattern duration, used for cycle counting

        self._rng = np.random.default_rng(seed)

        # store the indexes of the randomly drawn pattern
        self._pat_ids = [self._rng.choice(self.num_pattern)]
        # store the starting times of the pattern
        self._starting_times = [t_start]

        # now, fill the arrays until t_max
        # we do this, because the pattern can have arbitrary lengths:
        while self._starting_times[-1] < t_max:
            current_pat_idx = self._rng.choice(self.num_pattern)
            self._pat_ids.append(current_pat_idx)
            self._starting_times.append(
                self._starting_times[-1] + self.durations[self._pat_ids[-2]]
            )

    def _apply_online_transforms(self, pattern_1d):
        """
        Apply online transformations to a pattern.

        This method applies all specified online transformations in sequence
        to the given 1-dimensional pattern.

        :param pattern_1d: The input pattern to transform.
        :type pattern_1d: Any
        :return: Transformed pattern.
        :rtype: Any
        """
        for transform in self.online_transforms:
            pattern_1d = transform(pattern_1d)
        return pattern_1d

    def __call__(self, t: float, offset: float = 1e-6):
        """
        Retrieve the pattern at a specific time `t`.

        This method finds the appropriate pattern based on the given time `t`,
        applies online transformations, and returns the transformed result.

        :param t: Time at which to retrieve the pattern.
        :type t: float
        :param offset: Small offset added to `t` for numerical stability,
        defaults to 1e-6.
        :type offset: float, optional
        :return: Transformed pattern at time `t`.
        :rtype: Any

        :raises ValueError: If `t` exceeds `t_max`.
        """
        if t > self.t_max:
            raise ValueError("t must be small smaller than t_max.")
        # find index of the pattern that is at t:
        t += offset
        idx_in_sequence = np.searchsorted(self._starting_times, t) - 1
        pat_idx = self._pat_ids[idx_in_sequence]
        time_in_pattern = t - self._starting_times[idx_in_sequence]

        # get the pattern at the time t:
        idx_in_pattern = int((time_in_pattern) / self.dts[pat_idx])
        pattern_t = self.pattern[pat_idx][idx_in_pattern]

        pattern_t = self._apply_online_transforms(pattern_t)

        return pattern_t

    def iter(self, t_start, t_stop, dt):
        """
        Use dataloader as an iterator/iterable.

        Generates tuples of time and corresponding patterns between `t_start`
        and `t_stop` with a step size of `dt`.

        :param t_start: Start time of iteration.
        :type t_start: float
        :param t_stop: Stop time of iteration.
        :type t_stop: float
        :param dt: Time step between iterations.
        :type dt: float
        :yield: Tuple containing time and corresponding pattern.
        :rtype: Iterator[Tuple[float, np.ndarray]]

        Example:
            >>> for t, pattern in dataloader.iter(t_start=0.0, t_stop=10.0, dt=0.1):
            ...     print(t, pattern)
            ...
            0.0 [pattern data]
            0.1 [pattern data]
            ...
            9.9 [pattern data]
        """
        t = t_start
        while t < t_stop:
            yield t, self.__call__(t, offset=dt * 0.01)
            t += dt

    def get_full_pattern(
        self,
        dt: float,
        idx: Optional[List[int]] = None,
        online_transforms: bool = True,
        num: int = 1,
    ):
        """
        Retrieve the full concatenated pattern sequence (patterns concatenated in time).

        :param dt: Time step used for sampling patterns.
        :type dt: float
        :param idx: Indices of patterns to include. Defaults to all patterns.
        :type idx: Optional[List[int]]
        :param online_transforms: Whether to apply online transforms.
        :type online_transforms: bool
        :param num: Unused — kept for interface compatibility with other dataloaders.
        :type num: int
        :return: The concatenated array of patterns (time × width).
        :rtype: np.ndarray
        """
        full_pattern = []
        if idx is None:
            idx = list(range(self.num_pattern))
        transforms = self.online_transforms if online_transforms else []
        for i in idx:
            dl = Dataloader(self.pattern[i], online_transforms=transforms)
            full_pattern.append(dl.get_full_pattern(dt))

        return np.concatenate(full_pattern, axis=0)


class ContinuousDataloader(BaseDataloader):
    """
    A continuous data loader that applies transformations to patterns.

    This class extends BaseDataloader to provide functionality for loading
    and transforming continuous data patterns.

    :param pattern: The pattern to be loaded and transformed.
    :type pattern: Any
    :param pre_transforms: List of transformations to apply before loading.
    :type pre_transforms: List[Callable]
    :param online_transforms: List of transformations to apply during loading.
    :type online_transforms: List[Callable]
    """

    def __init__(
        self,
        pattern: BaseContinuousPattern,
        pre_transforms: List[Callable] = [],
        online_transforms: List[Callable] = [],
    ):
        self.pattern = pattern
        self.duration = self.pattern.duration
        self.width = pattern.width
        self.pre_transforms = pre_transforms
        self.online_transforms = online_transforms
        self.apply_online_transforms = 1

    @staticmethod
    def _apply_transforms(transforms: Callable, pattern: npt.NDArray) -> npt.NDArray:
        for transform in transforms:
            pattern = transform(pattern)
        return pattern

    def __call__(self, t: float, offset: float = 0.0) -> npt.NDArray:
        """
        Call the dataloader to get a transformed pattern at a specific time.

        :param t: The time at which to get the pattern.
        :type t: float
        :param offset: Time offset to apply.
        :type offset: float
        :return: The transformed pattern at the specified time.
        :rtype: Any
        """
        pat = self.pattern(t + offset)
        pat = self._apply_transforms(self.pre_transforms, pat)
        pat = self._apply_transforms(self.online_transforms, pat)
        return pat

    def iter(
        self, t_start: float, t_stop: float, dt: float
    ) -> Tuple[float, npt.NDArray]:
        """
        Iterate over the pattern within a time range.

        :param t_start: Start time of the iteration.
        :type t_start: float
        :param t_stop: End time of the iteration.
        :type t_stop: float
        :param dt: Time step for the iteration.
        :type dt: float
        :yield: A tuple of (time, pattern) for each time step.
        :rtype: Tuple[float, Any]
        """
        t = t_start
        while t < t_stop:
            yield t, self.__call__(t)
            t += dt

    def get_full_pattern(self, dt: float) -> npt.NDArray:
        """
        Get the full pattern as a numpy array.

        :param dt: Time step for sampling the pattern.
        :type dt: float
        :return: The full pattern as a numpy array.
        :rtype: np.ndarray
        """
        full_pattern = []
        for _, pattern in self.iter(0, self.duration, dt):
            full_pattern.append(pattern)

        return np.array(full_pattern)


def Dataloader(
    pattern: Union[BasePattern, BaseContinuousPattern],
    pre_transforms: List[Callable] = [],
    online_transforms: List[Callable] = [],
) -> Union[DiscreteDataloader, ContinuousDataloader]:
    """
    Factory function to create an appropriate Dataloader based on the pattern type.

    This function determines whether to create a DiscreteDataloader or a
    ContinuousDataloader based on the type of the input pattern.
    It also applies the specified pre-transforms and online-transforms to the
    created dataloader.

    :param pattern: The pattern object to be loaded.
    :type pattern: Union[BasePattern, BaseContinuousPattern]
    :param pre_transforms: List of transformations to apply before loading the pattern.
    :type pre_transforms: List[Callable]
    :param online_transforms: List of transformations to apply during pattern loading.
    :type online_transforms: List[Callable]
    :return: An instance of either DiscreteDataloader or ContinuousDataloader.
    :rtype: Union[DiscreteDataloader, ContinuousDataloader]
    :raises TypeError: If the pattern is neither a BasePattern nor a
    BaseContinuousPattern.

    :Example:

    >>> discrete_pattern = BasePattern()
    >>> discrete_loader = Dataloader(discrete_pattern)
    >>> continuous_pattern = BaseContinuousPattern()
    >>> continuous_loader = Dataloader(continuous_pattern)
    """
    if isinstance(pattern, BasePattern):
        return DiscreteDataloader(
            pattern, pre_transforms=pre_transforms, online_transforms=online_transforms
        )
    if isinstance(pattern, BaseContinuousPattern):
        return ContinuousDataloader(
            pattern, pre_transforms=pre_transforms, online_transforms=online_transforms
        )
    else:
        raise TypeError(
            f"pattern should inherit from BasePattern or BaseContinuousPatter."  # noqa
        )


class MultiPatternDataloader(BaseDataloader):
    """
    Dataloader that concatenates the outputs of multiple patterns.

    :param patterns: List of tuples with:
                     (pattern, [pre_transforms], [online_transforms])
                     Transforms are optional.
    """

    def __init__(
        self,
        patterns: List[Tuple[BasePattern, List[Callable], List[Callable]]],
        pre_transform: List[Callable] = [],
        online_transform: List[Callable] = [],
    ):
        self.dataloaders = []
        self.durations = []
        self.width = 0
        self.widths = []

        for pattern in patterns:
            dl = Dataloader(pattern, pre_transform, online_transform)
            self.dataloaders.append(dl)
            self.durations.append(dl.duration)
            self.widths.append(dl.width)
            self.width += dl.width

        self.duration = np.max(self.durations)

    def __call__(self, t: float, offset: float = 0.0):
        pats = [np.asarray(dl(t, offset)) for dl in self.dataloaders]
        return np.concatenate(pats, axis=-1)

    def iter(self, t_start: float, t_stop: float, dt: float):
        t = t_start
        while t < t_stop:
            yield t, self.__call__(t)
            t += dt

    def get_individual_patterns(self, dt: float):
        """
        Get individual patterns as a list of numpy arrays.

        :param dt: Time step for sampling the patterns.
        :type dt: float
        :return: List of individual patterns as numpy arrays.
        :rtype: List[npt.NDArray]
        """
        return [dl.get_full_pattern(dt) for dl in self.dataloaders]

    def get_full_pattern(self, dt: float, concat=True, num=1, online_transforms=False):
        if concat:
            full_pattern = np.array(
                [pattern for _, pattern in self.iter(0, self.duration * num, dt)]
            )
        else:
            full_pattern = [dl.get_full_pattern(dt) for dl in self.dataloaders]

        return full_pattern

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    pattern = RandomSampledNonMarkovianPattern(
        duration=50, width=6, rng=rng, dt=0.1, nmk=6
    )

    plt.imshow(pattern.pattern.T, aspect="auto", origin="lower")
    plt.xlabel("Time step")
    plt.ylabel("Pattern dimension")
    plt.title("Random Pattern over Time")
    plt.colorbar(label="Value")
    plt.show()
