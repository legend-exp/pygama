"""
pygama convenience functions for fitting ubinned data
"""
from iminuit import Minuit, cost


def fit_unbinned(func, data, guess=None,
             Extended=True, cost_func = 'LL',simplex=False,
             bounds=None, fixed=None):
    """Do a unbinned fit to data.
    Default is Extended Log Likelihood fit, with option for other cost functions.

    Parameters
    ----------
    func : the function to fit
    data : the data
    guess : initial guess parameters
    Extended : run extended or non extended fit
    cost_func : cost function to use
    simplex : whether to include a round of simpson minimisation before main minimisation
    bounds : list of tuples with bounds can be None, e.g. [(0,None), (0,10)]
    fixed : list of parameter indices to fix
    coeff, cov_matrix : tuple(array, matrix)
    """
    if guess is None:
        print("auto-guessing not yet implemented, you must supply a guess.")
        return None, None

    if cost_func =='LL':
        if Extended ==True:
            cost_func = cost.ExtendedUnbinnedNLL(data, func)

        else:
            cost_func = cost.UnbinnedNLL(data, func)

    m = Minuit(cost_func, *guess)
    if bounds is not None:
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
