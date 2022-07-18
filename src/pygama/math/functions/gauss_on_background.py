"""
Gaussian distributions on linear and uniform backgrounds for pygama
"""
import numba as nb
import numpy as np

from pygama.math.functions.gauss import nb_gauss_norm

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_gauss_uniform(x, n_sig, mu, sigma, n_bkg, components = False):
    """
    A Gaussian signal on a uniform background
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    n_sig : float
        Number of counts in the signal
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_bkg : float
        The number of counts in the background
    components : bool
        If true, returns the signal and background components separately
    """

    if components==False:
        return 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg + n_sig * nb_gauss_norm(x,mu,sigma)
    else:
        return n_sig * nb_gauss_norm(x,mu,sigma), 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg


@nb.njit(**kwd)
def nb_gauss_linear(x, n_sig, mu, sigma, n_bkg, b, m, components=False):
    """
    Gaussian signal on top of a linear background function
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    n_sig : float
        Number of counts in the signal
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_bkg : float
        The number of counts in the background
    b : float
        The y-intercept of the linear background
    m : float
        Slope of the linear background
    components : bool
        If true, returns the signal and background components separately
    """

    norm = (m/2 *np.nanmax(x)**2 + b*np.nanmax(x)) - (m/2 *np.nanmin(x)**2 + b*np.nanmin(x))

    if components==False:
        return n_bkg/norm * (m * x + b) + n_sig * nb_gauss_norm(x, mu, sigma)
    else:
        return  n_sig * nb_gauss_norm(x, mu, sigma), n_bkg/norm * (m * x + b)
