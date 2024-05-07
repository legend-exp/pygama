"""
pygama utility functions.
"""

import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """
    given a file size in bytes, output a human-readable form.
    Parameters
    ----------
    num
        File size, in bytes
    suffix
        Desired file size suffix
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1024.0
    return "{:.1f} {} {}".format(num, "Y", suffix)


def get_par_names(func: Callable) -> tuple[str, ...]:
    """
    Return a list containing the names of the arguments of "func" other than the
    first argument. In pygamaland, those are the function's "parameters."

    Parameters
    ----------
    func
        A function whose parameters we want to return
    """
    import inspect

    par = inspect.getfullargspec(func)
    return par[0][1:]


def get_formatted_stats(mean: float, sigma: float, ndigs: int = 2) -> str:
    """
    convenience function for formatting mean +/- sigma to the right number of
    significant figures.

    Parameters
    ----------
    mean
        The mean value we want to format
    sigma
        The sigma value we want to format
    ndigs
        The number of significant digits we want to display
    """
    if sigma == 0:
        fmt = "%d" % ndigs
        fmt = "%#." + fmt + "g"
        return fmt % mean, fmt % sigma
    sig_pos = int(np.floor(np.log10(abs(sigma))))
    sig_fmt = "%d" % ndigs
    sig_fmt = "%#." + sig_fmt + "g"
    mean_pos = int(np.floor(np.log10(abs(mean))))
    mdigs = mean_pos - sig_pos + ndigs
    if mdigs < ndigs - 1:
        mdigs = ndigs - 1
    mean_fmt = "%d" % mdigs
    mean_fmt = "%#." + mean_fmt + "g"
    return mean_fmt % mean, sig_fmt % sigma


def print_fit_results(
    pars: np.ndarray,
    cov: np.ndarray,
    func: Optional[Callable] = None,
    title: Optional[str] = None,
    pad: Optional[bool] = True,
) -> None:
    """
    Convenience function to write scipy.optimize.curve_fit results to the log

    Parameters
    ----------
    pars
        The parameter values of the function func
    func
        A function, if passed then the function's parameters' names are logged
    title
        A title to log
    pad
        If True, adds spaces to the log messages

    Returns
    -------
    None
        Writes the curve_fit results to the log
    """
    if title is not None:
        log.info(f"{title}:")
    par_names = []
    if func is None:
        for i in range(len(pars)):
            par_names.append("p" + str(i))
    else:
        par_names = get_par_names(func)
    for i in range(len(pars)):
        mean, sigma = get_formatted_stats(pars[i], np.sqrt(cov[i][i]))
        log.info(f"{par_names[i]} = {mean} +/- {sigma}")
    if pad:
        log.info("")
