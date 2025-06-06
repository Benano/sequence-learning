#!/usr/bin/env python3

"""
Functions for evaluating the model performance.
"""
from typing import Callable

import numba
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view


def window_slider(
    data: npt.NDArray, target: npt.NDArray, func: Callable[[npt.NDArray], float]
):
    """
    Apply a sliding window operation on data using a given function.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array defining the window shape
    :type target: npt.NDArray
    :param func: Function to apply on each window
    :type func: Callable[[npt.NDArray, npt.NDArray], float]
    :return: Array of results from applying func on each window
    :rtype: npt.NDArray
    """
    windows = sliding_window_view(data, target.shape).squeeze()
    return np.array([func(chunk, target) for chunk in windows])


@numba.njit()
def pearson_coef(data: npt.NDArray, target: npt.NDArray):
    """
    Calculate the mean Pearson correlation coefficient between data and target.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array
    :type target: npt.NDArray
    :return: Mean Pearson correlation coefficient
    :rtype: float
    """
    t, n = data.shape
    # apperently, transposing the array beforehand speeds up the
    dataT = data.T
    targetT = target.T
    mean_coef = 0.0
    for i in range(n):
        mean_coef += np.corrcoef(dataT[i], targetT[i])[0, 1]
    return mean_coef / n


def mse(data: npt.NDArray, target: npt.NDArray):
    """
    Calculate the Mean Squared Error between data and target.

    :param data: Input data array
    :type data: npt.NDArray
    :param target: Target array
    :type target: npt.NDArray
    :return: Mean Squared Error
    :rtype: float
    """
    return np.mean((data - target) ** 2)
