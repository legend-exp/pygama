"""
pygama utility functions.
"""
import logging
import sys

import numpy as np

log = logging.getLogger(__name__)


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
