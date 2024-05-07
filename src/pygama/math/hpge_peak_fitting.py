"""
pygama convenience functions for fitting hpge peak shape data
"""

import logging
import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from pygama.math.functions.exgauss import nb_exgauss_pdf
from pygama.math.functions.gauss import nb_gauss_pdf
from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss
from pygama.math.functions.step import nb_unnorm_step_pdf

log = logging.getLogger(__name__)


def hpge_peak_fwhm(
    sigma: float, htail: float, tau: float, cov: Optional[float] = None
) -> tuple[float, float]:
    """
    Return the FWHM of the hpge_peak function, ignoring background and step
    components. If calculating error also need the normalisation for the step
    function.

    Parameters
    ----------
    sigma
        The width of the hpge_peak
    htail
        The height of the tail in the hpge_peak
    tau
        The characteristic scale in the extended Gaussian in the hpge_peak
    cov
        The covariant matrix of the previous parameters

    Returns
        FWHM, FWHM_uncertainty
            The FWHM of the hpge_peak and its uncertainty
    """

    # optimize this to find max value
    def neg_hpge_peak_peak_bgfree(e, sigma, htail, tau):
        return -gauss_on_exgauss.get_pdf(
            np.array([e]), *np.array([0, sigma, htail, tau])
        )[0]

    if htail < 0 or htail > 1:
        raise ValueError("htail outside allowed limits of 0 and 1")

    res = minimize_scalar(
        neg_hpge_peak_peak_bgfree,
        args=(sigma, htail, tau),
        bounds=(-sigma - htail, sigma + htail),
    )
    e_max = res.x
    half_max = -neg_hpge_peak_peak_bgfree(e_max, sigma, htail, tau) / 2.0

    # root find this to find the half-max energies
    def hpge_peak_peak_bgfree_halfmax(e, sigma, htail, tau, half_max):
        return (
            gauss_on_exgauss.get_pdf(np.array([e]), *np.array([0, sigma, htail, tau]))[
                0
            ]
            - half_max
        )

    try:
        lower_hm = brentq(
            hpge_peak_peak_bgfree_halfmax,
            -(2.5 * sigma / 2 + htail * tau),
            e_max,
            args=(sigma, htail, tau, half_max),
        )
    except Exception:
        lower_hm = brentq(
            hpge_peak_peak_bgfree_halfmax,
            -(5 * sigma + htail * tau),
            e_max,
            args=(sigma, htail, tau, half_max),
        )
    try:
        upper_hm = brentq(
            hpge_peak_peak_bgfree_halfmax,
            e_max,
            2.5 * sigma / 2,
            args=(sigma, htail, tau, half_max),
        )
    except Exception:
        upper_hm = brentq(
            hpge_peak_peak_bgfree_halfmax,
            e_max,
            5 * sigma,
            args=(sigma, htail, tau, half_max),
        )

    if cov is None:
        return upper_hm - lower_hm

    # calculate uncertainty
    # nsig set to 1, mu to 0, hstep+nbkg set to 0
    pars = [1, 0, sigma, htail, tau, 0, 0]
    step_norm = 1
    gradmax = hpge_peak_parameter_gradient(e_max, pars, step_norm)
    gradmax *= 0.5
    grad1 = hpge_peak_parameter_gradient(lower_hm, pars, step_norm)
    grad1 -= gradmax
    grad1 /= hpge_peak_peakshape_derivative(lower_hm, pars, step_norm)
    grad2 = hpge_peak_parameter_gradient(upper_hm, pars, step_norm)
    grad2 -= gradmax
    grad2 /= hpge_peak_peakshape_derivative(upper_hm, pars, step_norm)
    grad2 -= grad1

    fwfm_unc = np.sqrt(np.dot(grad2, np.dot(cov, grad2)))

    return upper_hm - lower_hm, fwfm_unc


def hpge_peak_fwfm(sigma, htail, tau, frac_max=0.5, cov=None):
    """
    Return the FWHM of the radford_peak function, ignoring background and step
    components. If calculating error also need the normalisation for the step
    function.
    """

    # optimize this to find max value
    def neg_radford_peak_bgfree(e, sigma, htail, tau):
        return -gauss_on_exgauss.get_pdf(np.array([e]), 0, sigma, htail, tau)[0]

    if htail < 0 or htail > 1:
        raise ValueError("htail outside allowed limits of 0 and 1")

    res = minimize_scalar(
        neg_radford_peak_bgfree,
        args=(sigma, htail, tau),
        bounds=(-sigma - htail, sigma + htail),
    )
    e_max = res.x
    val_frac_max = -neg_radford_peak_bgfree(e_max, sigma, htail, tau) * frac_max

    # root find this to find the half-max energies
    def radford_peak_bgfree_fracmax(e, sigma, htail, tau, val_frac_max):
        return (
            gauss_on_exgauss.get_pdf(np.array([e]), 0, sigma, htail, tau)[0]
            - val_frac_max
        )

    try:
        lower_hm = brentq(
            radford_peak_bgfree_fracmax,
            -(2.5 * sigma / 2 + htail * tau),
            e_max,
            args=(sigma, htail, tau, val_frac_max),
        )
    except Exception:
        lower_hm = brentq(
            radford_peak_bgfree_fracmax,
            -(5 * sigma + htail * tau),
            e_max,
            args=(sigma, htail, tau, val_frac_max),
        )
    try:
        upper_hm = brentq(
            radford_peak_bgfree_fracmax,
            e_max,
            2.5 * sigma / 2,
            args=(sigma, htail, tau, val_frac_max),
        )
    except Exception:
        upper_hm = brentq(
            radford_peak_bgfree_fracmax,
            e_max,
            5 * sigma,
            args=(sigma, htail, tau, val_frac_max),
        )

    if cov is None:
        return upper_hm - lower_hm
    # calculate uncertainty
    # nsig set to 1, mu to 0, hstep+nbkg set to 0
    pars = [1, 0, sigma, htail, tau, 0, 0]

    rng = np.random.default_rng(1)
    par_b = rng.multivariate_normal(pars, cov, size=100)
    y_b = np.zeros(len(par_b))
    for i, p in enumerate(par_b):
        try:
            y_b[i] = hpge_peak_fwfm(p[2], p[3], p[4], frac_max=frac_max)
        except Exception:
            y_b[i] = np.nan
    yerr_boot = np.nanstd(y_b, axis=0)

    return upper_hm - lower_hm, yerr_boot


def hpge_peak_mode(mu, sigma, htail, tau, cov=None):

    if htail < 0 or htail > 1:
        if cov is not None:
            return np.nan, np.nan
        else:
            return np.nan

    try:
        mode = brentq(
            hpge_peak_peakshape_derivative,
            mu - 2 * sigma - htail * tau,
            mu + 2 * sigma + htail * tau,
            args=([1, mu, sigma, htail, tau, 0, 0], 1),
        )
    except ValueError:
        try:
            mode = brentq(
                hpge_peak_peakshape_derivative,
                mu - 4 * sigma - htail * tau,
                mu + 4 * sigma + htail * tau,
                args=([1, mu, sigma, htail, tau, 0, 0], 1),
            )
        except ValueError:
            mode = np.nan

    if cov is None:
        return mode
    else:
        # nsig set to 1, hstep+nbkg set to 0
        pars = np.array([1, mu, sigma, htail, tau, 0, 0])
        rng = np.random.default_rng(1)
        par_b = rng.multivariate_normal(pars, cov, size=10000)
        modes = np.array([hpge_peak_mode(p[1], p[2], p[3], p[4]) for p in par_b])
        mode_err_boot = np.nanstd(modes, axis=0)

        return mode, mode_err_boot


def hpge_peak_peakshape_derivative(
    e: np.ndarray, pars: np.ndarray, step_norm: float
) -> np.ndarray:
    """
    Computes the derivative of the hpge_peak peak shape

    Parameters
    ----------
    e
        The array of energies of the hpge_peak
    pars
        The parameters of the hpge_peak fit
    step_norm
        The normalization of the background step function in the hpge_peak

    Returns
    -------
    derivative
        the derivative of the hpge_peak
    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    sigma = abs(sigma)
    gaus = nb_gauss_pdf(e, mu, sigma)
    y = (e - mu) / sigma
    ret = -(1 - htail) * (y / sigma) * gaus
    ret -= (
        htail / tau * (-nb_exgauss_pdf(np.array([e, e - 1]), mu, sigma, tau)[0] + gaus)
    )

    return n_sig * ret - n_bkg * hstep * gaus / step_norm  # need norm factor for bkg


def hpge_peak_parameter_gradient(
    e: float, pars: np.ndarray, step_norm: float
) -> np.ndarray:
    """
    Computes the gradient of the hpge_peak parameters

    Parameters
    ----------
    e
        The energy of the hpge_peak
    pars
        The parameters of the hpge_peak fit
    step_norm
        The normalization of the background step function in the hpge_peak

    Returns
    -------
    gradient
        gradient of the n_sig, mu, sigma, h_tail, tau, n_bkg, and hstep parameters of the
        HPGe peak

    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    gaus = nb_gauss_pdf(np.array([e, e - 1]), mu, sigma)[0]

    tail_l = nb_exgauss_pdf(np.array([e, e - 1]), mu, sigma, tau)[0]
    if n_bkg == 0:
        step_f = 0
    else:
        step_f = (
            nb_unnorm_step_pdf(np.array([e, e - 1]), mu, sigma, hstep)[0] / step_norm
        )

    # some unitless numbers that show up a bunch
    y = (e - mu) / sigma
    sig_tau_l = sigma / tau

    g_n_sig = 0.5 * (htail * tail_l + (1 - htail) * gaus)
    g_n_bkg = step_f

    g_hs = n_bkg * math.erfc(y / np.sqrt(2)) / step_norm

    g_ht = (n_sig / 2) * (tail_l - gaus)

    # gradient of gaussian part
    g_mu = (1 - htail) * y / sigma * gaus
    g_sigma = (1 - htail) * (y * y + -1) / sigma * gaus

    # gradient of low tail, use approximation if necessary
    g_mu += htail / tau * (-tail_l + gaus)
    g_sigma += htail / tau * (sig_tau_l * tail_l - (sig_tau_l - y) * gaus)
    g_tau = (
        -htail
        / tau
        * (
            (1.0 + sig_tau_l * y + sig_tau_l * sig_tau_l) * tail_l
            - sig_tau_l * sig_tau_l * gaus
        )
        * n_sig
    )

    g_mu = n_sig * g_mu + (2 * n_bkg * hstep * gaus) / step_norm
    g_sigma = n_sig * g_sigma + (2 * n_bkg * hstep * gaus * y) / (
        step_norm * np.sqrt(sigma)
    )

    gradient = g_n_sig, g_mu, g_sigma, g_ht, g_tau, g_n_bkg, g_hs
    return np.array(gradient)
