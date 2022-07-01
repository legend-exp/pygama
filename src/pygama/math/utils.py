"""
pygama utility functions.
"""
import sys

import numpy as np
import tqdm

import logging

log = logging.getLogger(__name__)


def tqdm_range(start, stop, step=1, verbose=False, text=None, bar_length=20, unit=None):
    """
    Uses tqdm.trange which wraps around the python range and also has the option
    to display a progress
    For example:

    .. code-block:: python

        for start_row in range(0, tot_n_rows, buffer_len):
        ...

    Can be converted to the following
    
    .. code-block:: python

        for start_row in tqdm_range(0, tot_n_rows, buffer_len, verbose):
        ...

    Parameters
    ----------
    start : int
        starting iteration value
    stop : int
        ending iteration value
    step : int
        step size in between each iteration
    verbose : int
        verbose = 0 hides progress bar verbose > 0 displays progress bar
    text : str
        text to display in front of the progress bar
    bar_length : str
        horizontal length of the bar in cursor spaces
    unit : str
        physical units to be displayed
    Returns
    -------
    iterable : tqdm.trange
        object that can be iterated over in a for loop
    """
    hide_bar = True
    if isinstance(verbose, int):
        if verbose > 0:
            hide_bar = False
    elif isinstance(verbose, bool):
        if verbose is True:
            hide_bar = False

    if text is None:
        text = "Processing"

    if unit is None:
        unit = "it"

    bar_format = f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:{-bar_length}b}}"

    return tqdm.trange(start, stop, step,
                       disable=hide_bar, desc=text,
                       bar_format=bar_format, unit=unit, unit_scale=True)


def sizeof_fmt(num, suffix='B'):
    """
    given a file size in bytes, output a human-readable form.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1024.0
    return "{:.1f} {} {}".format(num, 'Y', suffix)


def get_par_names(func):
    """
    Return a list containing the names of the arguments of "func" other than the
    first argument. In pygamaland, those are the function's "parameters."
    """
    from scipy._lib._util import getargspec_no_self
    args, varargs, varkw, defaults = getargspec_no_self(func)
    return args[1:]


def get_formatted_stats(mean, sigma, ndigs=2):
    """
    convenience function for formatting mean +/- sigma to the right number of
    significant figures.
    """
    sig_pos = int(np.floor(np.log10(abs(sigma))))
    sig_fmt = '%d' % ndigs
    sig_fmt = '%#.' + sig_fmt + 'g'
    mean_pos = int(np.floor(np.log10(abs(mean))))
    mdigs = mean_pos-sig_pos+ndigs
    if mdigs < ndigs-1: mdigs = ndigs - 1
    mean_fmt = '%d' % mdigs
    mean_fmt = '%#.' + mean_fmt + 'g'
    return mean_fmt % mean, sig_fmt % sigma


def print_fit_results(pars, cov, func=None, title=None, pad=True):
    """
    convenience function for scipy.optimize.curve_fit results
    """
    if title is not None:
        log.info(f"{title}:")
    if func is None:
        for i in range(len(pars)): par_names.append("p"+str(i))
    else:
        par_names = get_par_names(func)
    for i in range(len(pars)):
        mean, sigma = get_formatted_stats(pars[i], np.sqrt(cov[i][i]))
        log.info(f"{par_names[i]} = {mean} +/- {sigma}")
    if pad:
        log.info("")
