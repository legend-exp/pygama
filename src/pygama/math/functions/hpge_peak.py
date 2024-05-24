r"""
Provide a convenience function for the HPGe peak shape.

A HPGe peak consists of a Gaussian
on an Exgauss on a step function.

.. math::

    PDF = n_sig*((1-htail)*gauss + htail*exgauss) + n_bkg*step


Called with

hpge_peak.get_pdf(x, x_lo, x_hi, n_sig, mu, sigma, htail, tau, n_bkg, hstep)

Parameters
----------
x_lo
    Lower bound of the step function
x_hi
    Upper bound of the step function
n_sig
    The area of the gauss on exgauss
mu
    The centroid of the Gaussian
sigma
    The standard deviation of the Gaussian
htail
    The height of the Gaussian tail
tau
    The characteristic scale of the Gaussian tail
n_bkg
    The area of the step background
hstep
    The height of the step function background

Returns
-------
hpge_peak
    A subclass of SumDists and rv_continuous, has methods of pdf, cdf, etc.

Notes
-----
The extended Gaussian distribution and the step distribution share the mu, sigma with the Gaussian
"""

import numpy as np

from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss
from pygama.math.functions.step import step
from pygama.math.functions.sum_dists import SumDists
from pygama.math.hpge_peak_fitting import hpge_peak_fwfm, hpge_peak_fwhm, hpge_peak_mode

(x_lo, x_hi, n_sig, mu, sigma, frac1, tau, n_bkg, hstep) = range(9)
par_array = [
    (gauss_on_exgauss, [mu, sigma, frac1, tau]),
    (step, [x_lo, x_hi, mu, sigma, hstep]),
]

hpge_peak = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=[
        "x_lo",
        "x_hi",
        "n_sig",
        "mu",
        "sigma",
        "htail",
        "tau",
        "n_bkg",
        "hstep",
    ],
    name="hpge_peak",
)


# This is defined here as to avoid a circular import inside `SumDists`
def hpge_get_fwhm(self, pars: np.ndarray, cov: np.ndarray = None) -> tuple:
    r"""
    Get the fwhm value from the output of a fit quickly
    Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak,
    and otherwise returns 2sqrt(2log(2))*sigma

    Parameters
    ----------
    pars
        Array of fit parameters
    cov
        Optional, array of covariances for calculating error on the fwhm


    Returns
    -------
    fwhm, error
        the value of the fwhm and its error
    """
    req_args = np.array(self.required_args())
    sigma_idx = np.where(req_args == "sigma")[0][0]

    if ("htail" in req_args) and (
        "hstep" in req_args
    ):  # having both the htail and hstep means it is an exgauss on a step
        htail_idx = np.where(req_args == "htail")[0][0]
        tau_idx = np.where(req_args == "tau")[0][0]
        # We need to ditch the x_lo and x_hi columns and rows
        if cov is not None:
            cov = np.array(cov)
            dropped_cov = cov[:, 2:][2:, :]
            return hpge_peak_fwhm(
                pars[sigma_idx], pars[htail_idx], pars[tau_idx], dropped_cov
            )
        else:
            return hpge_peak_fwhm(pars[sigma_idx], pars[htail_idx], pars[tau_idx])

    else:
        if cov is None:
            return pars[sigma_idx] * 2 * np.sqrt(2 * np.log(2))
        else:
            return pars[sigma_idx] * 2 * np.sqrt(2 * np.log(2)), np.sqrt(
                cov[sigma_idx][sigma_idx]
            ) * 2 * np.sqrt(2 * np.log(2))


# This is defined here as to avoid a circular import inside `SumDists`
def hpge_get_fwfm(
    self, pars: np.ndarray, frac_max=0.5, cov: np.ndarray = None
) -> tuple:
    r"""
    Get the fwhm value from the output of a fit quickly
    Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak,
    and otherwise returns 2sqrt(-2log(frac_max))*sigma

    Parameters
    ----------
    pars
        Array of fit parameters
    cov
        Optional, array of covariances for calculating error on the fwhm


    Returns
    -------
    fwhm, error
        the value of the fwhm and its error
    """
    req_args = np.array(self.required_args())
    sigma_idx = np.where(req_args == "sigma")[0][0]

    if ("htail" in req_args) and (
        "hstep" in req_args
    ):  # having both the htail and hstep means it is an exgauss on a step
        htail_idx = np.where(req_args == "htail")[0][0]
        tau_idx = np.where(req_args == "tau")[0][0]
        # We need to ditch the x_lo and x_hi columns and rows
        if cov is not None:
            cov = np.array(cov)
            dropped_cov = cov[:, 2:][2:, :]
            return hpge_peak_fwfm(
                pars[sigma_idx],
                pars[htail_idx],
                pars[tau_idx],
                frac_max=frac_max,
                cov=dropped_cov,
            )
        else:
            return hpge_peak_fwfm(
                pars[sigma_idx], pars[htail_idx], pars[tau_idx], frac_max=frac_max
            )

    else:
        if cov is None:
            return pars[sigma_idx] * 2 * np.sqrt(2 * np.log(2))
        else:
            return pars[sigma_idx] * 2 * np.sqrt(-2 * np.log(frac_max)), np.sqrt(
                cov[sigma_idx][sigma_idx]
            ) * 2 * np.sqrt(-2 * np.log(frac_max))


# This is defined here as to avoid a circular import inside `SumDists`
def hpge_get_mode(self, pars: np.ndarray, cov: np.ndarray = None) -> tuple:
    r"""
    Get the fwhm value from the output of a fit quickly
    Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak,
    and otherwise returns 2sqrt(2log(2))*sigma

    Parameters
    ----------
    pars
        Array of fit parameters
    cov
        Optional, array of covariances for calculating error on the fwhm


    Returns
    -------
    fwhm, error
        the value of the fwhm and its error
    """
    req_args = np.array(self.required_args())
    sigma_idx = np.where(req_args == "sigma")[0][0]
    mu_idx = np.where(req_args == "mu")[0][0]

    if ("htail" in req_args) and (
        "hstep" in req_args
    ):  # having both the htail and hstep means it is an exgauss on a step
        htail_idx = np.where(req_args == "htail")[0][0]
        tau_idx = np.where(req_args == "tau")[0][0]
        # We need to ditch the x_lo and x_hi columns and rows
        if cov is not None:
            cov = np.array(cov)
            dropped_cov = cov[2:, 2:]

            return hpge_peak_mode(
                pars[mu_idx],
                pars[sigma_idx],
                pars[htail_idx],
                pars[tau_idx],
                dropped_cov,
            )
        else:
            return hpge_peak_mode(
                pars[mu_idx], pars[sigma_idx], pars[htail_idx], pars[tau_idx]
            )

    else:
        if cov is None:
            return pars[mu_idx]
        else:
            return pars[mu_idx], np.sqrt(cov[mu_idx][mu_idx])


# hpge_peak.get_fwhm = hpge_get_fwhm
hpge_peak.get_fwfm = hpge_get_fwfm.__get__(hpge_peak)
hpge_peak.get_mode = hpge_get_mode.__get__(hpge_peak)
hpge_peak.get_fwhm = hpge_get_fwhm.__get__(hpge_peak)
