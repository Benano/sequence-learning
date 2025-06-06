import copy
import pickle
import warnings
from collections import defaultdict
from typing import Any, List, Tuple, Union

import numpy as np

from .model import Network


class Tracker:
    """
    A class for tracking and storing network variable values over time.

    This class allows for periodic tracking of specified network variables,
    storing their values, and saving/loading the tracker state.
    """

    def __init__(
        self,
        network: Network,
        variables: List[Tuple[str, str]],
        track_step: int,
    ) -> None:
        """
        Initialize the Tracker.

        :param network: The network object to track variables from.
        :type network: Network
        :param variables: List of tuples containing variable names and views.
        :type variables: List[Tuple[str, str]]
        :param track_step: Number of steps between each tracking event.
        :type track_step: int
        :raises ValueError: If variables are not in the correct format.
        """
        if not all(
            isinstance(var, (list, tuple)) and len(var) == 2 for var in variables
        ):
            raise ValueError(
                "Variables must be list of lists with two elements (name, view)."
            )
        self.network: Network = network
        self.variables: List[Tuple[str, str]] = variables
        self.track_step: int = track_step
        self.track_counter: int = 0
        self.track_dict: defaultdict = defaultdict(list)

    def _get_variable_value(self, name: str, view: str) -> Any:
        """
        Retrieve the value of a variable from the network.

        :param name: Name of the variable.
        :type name: str
        :param view: View of the variable.
        :type view: str
        :return: The value of the variable or None if an error occurs.
        :rtype: Any
        """

        try:
            return self.network.get_val(name, view)
        except (AttributeError, TypeError) as e:
            warnings.warn(f"Error retrieving {name} ({view}): {e}")
            return None

    def track(self, time: Union[int, float]) -> None:
        """
        Track variables at the current time step.

        :param time: The current time value to be tracked.
        :type time: Union[int, float]
        """
        self.track_dict["time"].append(time)
        self.track_dict["step"].append(copy.copy(self.track_counter))
        self.track_counter += 1
        if self.track_counter % self.track_step == 0:
            for name, view in self.variables:
                key = f"{name}_{view}"
                val = self._get_variable_value(name, view)
                if val is not None:  # Only store valid results
                    self.track_dict[key].append(val)

    def store(self, value_name: str, val: Any) -> None:
        """
        Store a custom value in the tracker.

        :param value_name: Name of the value to store.
        :type value_name: str
        :param val: The value to be stored.
        :type val: Any
        """
        self.track_dict[value_name].append(val)

    def save(self, path: str) -> None:
        """
        Save the tracker object to a file.

        :param path: File path to save the tracker.
        :type path: str
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Tracker":
        """
        Load a tracker object from a file.

        :param path: File path to load the tracker from.
        :type path: str
        :return: The loaded tracker object.
        :rtype: Tracker
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Get tracked values by key.

        :param item: Key of the tracked values.
        :type item: str
        :return: Array of tracked values for the given key.
        :rtype: np.ndarray
        """
        return np.array(self.track_dict[item])

    def __repr__(self) -> str:
        """
        Return a string representation of the Tracker object.

        :return: A string representation of the Tracker.
        :rtype: str
        """
        return (
            f"Tracker({list(self.track_dict.keys())}, "
            f"length={self.track_counter}, "
            f"track_step={self.track_step})"
        )
