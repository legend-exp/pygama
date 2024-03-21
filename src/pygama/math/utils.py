"""
pygama utility functions.
"""
import logging
import sys
import os
from typing import Optional, Union, Callable, Any, Iterator
from collections.abc import MutableMapping

import numpy as np

log = logging.getLogger(__name__)


def sizeof_fmt(num: float, suffix: str='B') -> str:
    """
    given a file size in bytes, output a human-readable form.
    Parameters 
    ----------
    num 
        File size, in bytes 
    suffix
        Desired file size suffix
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1024.0
    return "{:.1f} {} {}".format(num, 'Y', suffix)


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


def get_formatted_stats(mean: float, sigma: float, ndigs: int =2) -> str:
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
        fmt = '%d' % ndigs
        fmt = '%#.' + fmt + 'g'
        return fmt % mean, fmt % sigma
    sig_pos = int(np.floor(np.log10(abs(sigma))))
    sig_fmt = '%d' % ndigs
    sig_fmt = '%#.' + sig_fmt + 'g'
    mean_pos = int(np.floor(np.log10(abs(mean))))
    mdigs = mean_pos-sig_pos+ndigs
    if mdigs < ndigs-1: mdigs = ndigs - 1
    mean_fmt = '%d' % mdigs
    mean_fmt = '%#.' + mean_fmt + 'g'
    return mean_fmt % mean, sig_fmt % sigma


def print_fit_results(pars: np.ndarray, cov: np.ndarray, func: Optional[Callable]=None, title: Optional[str]=None, pad: Optional[bool]=True) -> None:
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
        for i in range(len(pars)): par_names.append("p"+str(i))
    else:
        par_names = get_par_names(func)
    for i in range(len(pars)):
        mean, sigma = get_formatted_stats(pars[i], np.sqrt(cov[i][i]))
        log.info(f"{par_names[i]} = {mean} +/- {sigma}")
    if pad:
        log.info("")


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get environment value as a boolean, returning True for 1, t and true
    (caps-insensitive), and False for any other value and default if undefined.
    """
    val = os.getenv(name)
    if not val:
        return default
    elif val.lower() in ("1", "t", "true"):
        return True
    else:
        return False

class NumbaMathDefaults(MutableMapping):
    """Bare-bones class to store some Numba default options. Defaults values
    are set from environment variables

    Examples
    --------
    Set all default option values for a processor at once by expanding the
    provided dictionary:

    >>> from numba import guvectorize
    >>> from pygama.math.utils import numba_defaults_kwargs as nb_kwargs
    >>> @guvectorize([], "", **nb_kwargs, nopython=True) # def proc(...): ...

    Customize one argument but still set defaults for the others:

    >>> from pygama.math.utils import numba_defaults as nb_defaults
    >>> @guvectorize([], "", **nb_defaults(cache=False) # def proc(...): ...

    Override global options at runtime:

    >>> from pygama.math.utils import numba_defaults
    >>> # must set options before explicitly importing pygama.math.distributions!
    >>> numba_defaults.cache = False
    """

    def __init__(self) -> None:
        self.parallel: bool = getenv_bool("MATH_PARALLEL", default=True)
        self.fastmath: bool = getenv_bool("MATH_FAST", default=True)

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, val: Any) -> None:
        self.__dict__[item] = val

    def __delitem__(self, item: str) -> None:
        del self.__dict__[item]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __call__(self, **kwargs) -> dict:
        mapping = self.__dict__.copy()
        mapping.update(**kwargs)
        return mapping

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)


numba_math_defaults = NumbaMathDefaults()
numba_math_defaults_kwargs = numba_math_defaults