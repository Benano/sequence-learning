"""Module for creating datasets and loading them into the model."""

# /usr/bin/env python3


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
        if type(dt) is float:
            self.dt = dt
            self.duration = self.dt * self.__len__()
        if type(duration) is float:
            self.duration = duration
            self.dt = self.duration / self.__len__()
        self.shape = self.pattern.shape

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
        if self.shape != self.pattern.shape:
            raise ValueError(
                "The transformation {transform} may not change the shape of the pattern!"  # noqa
            )

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


class Pattern2:
    """TODO: rewrite Pattern with properties:

    - pattern should be a property
    - and t_max also which is changed whenever pattern is changed
    - compare also in terms of speed
    """

    pass


class OneHotPattern(BasePattern):
    """
    Turn a sequential pattern into a one-hot encoded pattern.

    This class extends the base Pattern class to create one-hot encoded patterns.

    :ivar _width: The width of the one-hot encoded pattern
    :vartype _width: int
    """

    def __init__(self, pattern: npt.NDArray, dt: float, width: int) -> None:
        """
        Initialize the OneHotPattern object.

        :param pattern: The input sequential pattern array
        :type pattern: npt.NDArray
        :param dt: Time step
        :type dt: float
        :param width: Width of the one-hot encoded pattern
        :type width: int
        """
        if len(pattern.shape) != 1:
            raise ValueError("pattern must be one dimensional")
        if np.max(pattern) > width - 1:
            raise ValueError(
                "width must be greater then or equal to the maximum value in pattern"
            )
        self._width = width
        super().__init__(pattern, dt)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self._width))
        for i, j in enumerate(pattern):
            res[i, j] = 1.0
        return res


class MultiHotPattern(BasePattern):
    """
    Same as OneHot, but more then one can be active at a time.

    Encode it as:
    [2, 4, [0, 3], [0, 3], -1]

    If an entry is -1, there's no, the whole vector is zero (= a pause in the pattern)
    """

    def __init__(
        self, pattern: List[int | List[int]], duration: float, width: int
    ) -> None:
        """DOCSTRING."""
        # max in list:
        max_val = -1000000
        for i in pattern:
            if isinstance(i, int) and i > max_val:
                max_val = i
            elif isinstance(i, list):
                for j in i:
                    if j > max_val:
                        max_val = j
        if max_val > width - 1:
            raise ValueError(
                "width must be greater then or equal to the maximum value in pattern"
            )
        self._width = width
        super().__init__(pattern, duration=duration)

    def _convert(self, pattern):
        res = np.zeros((len(pattern), self._width))
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
        self.duration = self.pattern.duration
        self.dt = self.pattern.dt

        self.online_transforms = online_transforms

        # apply pre-transforms directly once
        for transform in pre_transforms:
            self.pattern.transform(transform)

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

    def get_full_pattern(self, dt):
        """Return the full pattern."""
        """
        :param dt: Simulation time step
        :type dt: float
        :return: Full pattern
        :rtype: npt.NDArray
        """

        full_pattern = []
        for _, pattern in self.iter(0, self.duration, dt):
            full_pattern.append(pattern)

        return np.array(full_pattern)


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

        self.pre_transforms = pre_transforms
        self.online_transforms = online_transforms

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
            f"pattern is {type(pattern)}, but should inherit from BasePattern or BaseContinuousPatter."  # noqa
        )
