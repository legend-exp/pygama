"""
Convenience functions to select distributions
"""
import logging

import numpy as np

from pygama.math.hpge_peak_fitting import hpge_peak_fwhm
from pygama.math.distributions import (
    nb_extended_hpge_peak_pdf,
    nb_gauss_step,
    nb_gauss_step_cdf,
    nb_gauss_step_pdf,
    nb_hpge_peak_cdf,
    nb_hpge_peak_pdf,
)

log = logging.getLogger(__name__)


def get_mu_func(func, pars, cov = None, errors=None):
    """
    Function to select which distribution to get the centroid from
    """
    if  func == nb_gauss_step_cdf or func == nb_gauss_step_pdf or func == nb_gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    elif  func == nb_hpge_peak_cdf or func == nb_hpge_peak_pdf or func == nb_extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    else:
        log.warning(f'get_mu_func not implemented for {func.__name__}')
        raise NameError


def get_fwhm_func(func, pars, cov = None):
    """
    Function to select which distribution to get the FWHM of
    """
    if  func == nb_gauss_step_cdf or func == nb_gauss_step_pdf or func == nb_gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if cov is None:
            return sigma*2*np.sqrt(2*np.log(2))
        else:
            return sigma*2*np.sqrt(2*np.log(2)), np.sqrt(cov[2][2])*2*np.sqrt(2*np.log(2))

    elif  func == nb_hpge_peak_cdf or func == nb_hpge_peak_pdf or func == nb_extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars

        return hpge_peak_fwhm(sigma, htail, tau, cov)
    else:
        log.warning(f'get_fwhm_func not implemented for {func.__name__}')
        raise NameError


def get_total_events_func(func, pars, cov = None, errors=None):
    """
    Function to select which distribution to get the total number of events from
    """
    if  func == nb_gauss_step_cdf or func == nb_gauss_step_pdf or func == nb_gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[3]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[3][3]**2)
        else:
            return n_sig+n_bkg

    elif  func == nb_hpge_peak_cdf or func == nb_hpge_peak_pdf or func == nb_extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[5]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[5][5]**2)
        else:
            return n_sig+n_bkg
    else:
        log.warning(f'get_total_events_func not implemented for {func.__name__}')
        raise NameError
