"""
pygama convenience functions for linearly fitting data
"""

from typing import Optional, Union

import numpy as np


def linear_fit_by_sums(
    x: np.ndarray, y: np.ndarray, var: Optional[Union[float, np.ndarray]] = 1
) -> tuple[float, float]:
    """
    Fast computation of weighted linear least squares fit to a linear model

    Note: doesn't compute covariances. If you want covariances, just use polyfit

    Parameters
    ----------
    x
        x values for the fit
    y
        y values for the fit
    var
        The variances for each y-value

    Returns
    -------
    (m, b)
        The slope (m) and y-intercept (b) of the best fit (in the least-squares
        sense) of the data to y = mx + b
    """
    y = y / var
    x = x / var
    sum_wts = len(y) / var if np.isscalar(var) else np.sum(1 / var)
    sum_x = np.sum(x)
    sum_xx = np.sum(x * x)
    sum_y = np.sum(y)
    sum_yx = np.sum(y * x)
    m = (sum_wts * sum_yx - sum_y * sum_x) / (sum_wts * sum_xx - sum_x**2)
    b = (sum_y - m * sum_x) / sum_wts
    return m, b


def fit_simple_scaling(
    x: np.ndarray, y: np.ndarray, var: Optional[Union[float, np.ndarray]] = 1
) -> tuple[float, float]:
    """
    Fast computation of weighted linear least squares fit to a simple scaling

    I.e. y = scale * x. Returns the best fit scale parameter and its variance.

    Parameters
    ----------
    x
        x values for the fit
    y
        y values for the fit
    var
        The variances for each y-value

    Returns
    -------
    scale, scale_var
        The scale parameter and its variance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    scale_var = 1 / np.sum(x * x / var)
    scale = np.sum(y * x / var) * scale_var
    return scale, scale_var
