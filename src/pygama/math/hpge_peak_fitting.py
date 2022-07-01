"""
pygama convenience functions for fitting hpge peak shape data
"""
import logging
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from pygama.math.distributions import (
    nb_exgauss,
    nb_gauss_norm,
    nb_gauss_with_tail_pdf,
    nb_unnorm_step_pdf,
)

log = logging.getLogger(__name__)


def hpge_peak_fwhm(sigma, htail, tau,  cov = None):
    """
    Return the FWHM of the hpge_peak_peak function, ignoring background and step
    components. If calculating error also need the normalisation for the step
    function.
    """
    # optimize this to find max value
    def neg_hpge_peak_peak_bgfree(E, sigma, htail, tau):
        return -nb_gauss_with_tail_pdf(np.array([E]), 0, sigma, htail, tau)[0]

    if htail<0 or htail>1:
        log.warning("htail outside allowed limits of 0 and 1")
        raise ValueError

    res = minimize_scalar( neg_hpge_peak_peak_bgfree,
                           args=(sigma, htail, tau),
                           bounds=(-sigma-htail, sigma+htail) )
    Emax = res.x
    half_max = -neg_hpge_peak_peak_bgfree(Emax, sigma, htail, tau)/2.

    # root find this to find the half-max energies
    def hpge_peak_peak_bgfree_halfmax(E, sigma, htail, tau, half_max):
        return nb_gauss_with_tail_pdf(np.array([E]), 0, sigma, htail, tau)[0] - half_max

    try:
        lower_hm = brentq( hpge_peak_peak_bgfree_halfmax,
                       -(2.5*sigma/2 + htail*tau), Emax,
                       args = (sigma, htail, tau, half_max) )
    except:
        lower_hm = brentq( hpge_peak_peak_bgfree_halfmax,
               -(5*sigma + htail*tau), Emax,
               args = (sigma, htail, tau, half_max) )
    try:
        upper_hm = brentq( hpge_peak_peak_bgfree_halfmax,
                       Emax, 2.5*sigma/2,
                       args = (sigma, htail, tau, half_max) )
    except:
        upper_hm = brentq( hpge_peak_peak_bgfree_halfmax,
                   Emax, 5*sigma,
                   args = (sigma, htail, tau, half_max) )

    if cov is None: return upper_hm - lower_hm

    #calculate uncertainty
    #amp set to 1, mu to 0, hstep+bg set to 0
    pars = [1,0, sigma, htail, tau,0,0]
    step_norm = 1
    gradmax = hpge_peak_parameter_gradient(Emax, pars, step_norm)
    gradmax *= 0.5
    grad1 = hpge_peak_parameter_gradient(lower_hm, pars,step_norm)
    grad1 -= gradmax
    grad1 /= hpge_peak_peakshape_derivative(lower_hm, pars,step_norm)
    grad2 = hpge_peak_parameter_gradient(upper_hm, pars,step_norm)
    grad2 -= gradmax
    grad2 /= hpge_peak_peakshape_derivative(upper_hm, pars,step_norm)
    grad2 -= grad1

    fwfm_unc = np.sqrt(np.dot(grad2, np.dot(cov, grad2)))

    return upper_hm - lower_hm, fwfm_unc


def hpge_peak_peakshape_derivative(E, pars, step_norm):
    """
    Computes the derivative of the hpge_peak peak shape
    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    sigma = abs(sigma)
    gaus = nb_gauss_norm(E, mu, sigma)
    y = (E-mu)/sigma
    ret = -(1-htail)*(y/sigma)*gaus
    ret -= htail/tau*(-nb_exgauss(np.array([E,E-1]), mu, sigma, tau)[0]+gaus)

    return n_sig*ret - n_bkg*hstep*gaus/step_norm #need norm factor for bkg


def hpge_peak_parameter_gradient(E, pars, step_norm):
    """
    Computes the gradient of the hpge_peak parameter
    """
    n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars

    gaus = nb_gauss_norm(np.array([E, E-1]), mu, sigma)[0]
    tailL = nb_exgauss(np.array([E, E-1]), mu, sigma, tau)[0]
    if n_bkg ==0:
        step_f = 0
    else:
        step_f = nb_unnorm_step_pdf(np.array([E, E-1]), mu, sigma, hstep)[0] /step_norm

    #some unitless numbers that show up a bunch
    y = (E-mu)/sigma
    sigtauL = sigma/tau

    g_n_sig = 0.5*(htail*tailL + (1-htail)*gaus)
    g_n_bkg = step_f

    g_hs = n_bkg*math.erfc(y/np.sqrt(2))/step_norm

    g_ht = (n_sig/2)*(tailL-gaus)

    #gradient of gaussian part
    g_mu = (1-htail)*y/sigma*gaus
    g_sigma = (1-htail)*(y*y +-1)/sigma*gaus

    #gradient of low tail, use approximation if necessary
    g_mu += htail/tau*(-tailL+gaus)
    g_sigma += htail/tau*(sigtauL*tailL-(sigtauL-y)*gaus)
    g_tau = -htail/tau*( (1.+sigtauL*y+sigtauL*sigtauL)*tailL - sigtauL*sigtauL*gaus) * n_sig

    g_mu = n_sig*g_mu + (2*n_bkg*hstep*gaus)/step_norm
    g_sigma = n_sig*g_sigma + (2*n_bkg*hstep*gaus*y)/(step_norm*np.sqrt(sigma))

    gradient = g_n_sig, g_mu, g_sigma,g_ht, g_tau, g_n_bkg, g_hs
    return np.array(gradient)
