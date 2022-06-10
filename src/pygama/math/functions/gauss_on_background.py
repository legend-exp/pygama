"""
Gaussian distributions on linear and uniform backgrounds for pygama
"""
import numpy as np

from pygama.math.functions.gauss import gauss_norm


def gauss_uniform(x, n_sig, mu, sigma, n_bkg, components = False):
    """
    define a gaussian signal on a uniform background,
    args: n_sig mu, sigma for the signal and n_bkg for the background
    TO DO: candidate for replacement by gauss_poly
    """
    if components==False:
        return 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg + n_sig * gauss_norm(x,mu,sigma)
    else:
        return n_sig * gauss_norm(x,mu,sigma), 1/(np.nanmax(x)-np.nanmin(x)) * n_bkg


def gauss_linear(x, n_sig, mu, sigma, n_bkg, b, m, components=False):
    """
    gaussian signal + linear background function
    args: n_sig mu, sigma for the signal and n_bkg,b,m for the background
    TO DO: candidate for replacement by gauss_poly
    """
    norm = (m/2 *np.nanmax(x)**2 + b*np.nanmax(x)) - (m/2 *np.nanmin(x)**2 + b*np.nanmin(x))

    if components==False:
        return n_bkg/norm * (m * x + b) + n_sig * gauss_norm(x, mu, sigma)
    else:
        return  n_sig * gauss_norm(x, mu, sigma), n_bkg/norm * (m * x + b)
