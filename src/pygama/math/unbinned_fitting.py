"""
pygama convenience functions for fitting ubinned data
"""
import logging
import numpy as np
from typing import Callable

from iminuit import Minuit, cost

log = logging.getLogger(__name__)


def fit_unbinned(func: Callable, data: np.ndarray, guess:np.ndarray =None,
             Extended: bool = True, cost_func:str = 'LL', simplex: bool = False,
             bounds: list[tuple[float, float], ...]=None, fixed: list[bool, ...]=None) -> tuple[np.ndarray, ...]:
    """Do a unbinned fit to data.
    Default is Extended Log Likelihood fit, with option for other cost functions.

    Parameters
    ----------
    func
        the function to fit
    data
        the data values to be fit
    guess
        initial guess parameters
    Extended
        run extended or non extended fit
    cost_func
        cost function to use. LL is ExtendedUnbinnedNLL, None is for just UnbinnedNLL
    simplex
        whether to include a round of simpson minimisation before main minimisation
    bounds
        Each tuple is (min, max) for the corresponding parameters.
        Bounds can be None, e.g. [(0,None), (0,10)]
    fixed
        list of parameter indices to fix

    Returns
    -------
    pars, errs, cov
        the best-fit parameters and their errors / covariance
    """
    if guess is None:
        raise NotImplementedError("auto-guessing not yet implemented, you must supply a guess.")

    if cost_func =='LL':
        if Extended == True:
            cost_func = cost.ExtendedUnbinnedNLL(data, func)

        else:
            cost_func = cost.UnbinnedNLL(data, func)
    if isinstance(guess, dict):
        m = Minuit(cost_func, **guess)
    else:
        m = Minuit(cost_func, *guess)
    if bounds is not None:
        if isinstance(bounds, dict):
            for arg, key in bounds.items():
                m.limits[arg] = key
        else:
            m.limits = bounds
    if fixed is not None:
        for fix in fixed:
            m.fixed[fix] = True
    if simplex == True:
        m.simplex().migrad()
    else:
        m.migrad()
    m.hesse()
    return m.values, m.errors, m.covariance
