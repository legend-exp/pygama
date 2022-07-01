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
    define a gaussian signal on a uniform background,
    args: n_sig mu, sigma for the signal and n_bkg for the background
    TO DO: candidate for replacement by gauss_poly
    """
    if components==False:
        return 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg + n_sig * nb_gauss_norm(x,mu,sigma)
    else:
        return n_sig * nb_gauss_norm(x,mu,sigma), 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg


@nb.njit(**kwd)
def nb_gauss_linear(x, n_sig, mu, sigma, n_bkg, b, m, components=False):
    """
    gaussian signal + linear background function
    args: n_sig mu, sigma for the signal and n_bkg,b,m for the background
    TO DO: candidate for replacement by gauss_poly
    """
    norm = (m/2 *np.nanmax(x)**2 + b*np.nanmax(x)) - (m/2 *np.nanmin(x)**2 + b*np.nanmin(x))

    if components==False:
        return n_bkg/norm * (m * x + b) + n_sig * nb_gauss_norm(x, mu, sigma)
    else:
        return  n_sig * nb_gauss_norm(x, mu, sigma), n_bkg/norm * (m * x + b)
