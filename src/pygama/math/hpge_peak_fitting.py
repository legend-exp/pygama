"""
pygama convenience functions for fitting hpge peak shape data
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from pygama.math.functions.exgauss import nb_exgauss_pdf
from pygama.math.functions.gauss import nb_gauss_pdf
from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss
from pygama.math.functions.step import nb_unnorm_step_pdf

log = logging.getLogger(__name__)


def _within_bounds(p, bounds):
    """Check a parameter draw against a ``{index: (lo, hi)}`` support map."""
    for idx, limits in bounds.items():
        lo, hi = limits
        if lo is not None and p[idx] < lo:
            return False
        if hi is not None and p[idx] > hi:
            return False
    return True


def bootstrap_valid_pars(
    rng, pars, cov, evaluate, size=100, max_draws=None, bounds=None
):
    """Draw ``size`` parameter vectors from ``N(pars, cov)`` that *evaluate* accepts.

    Parameters whose physical range is bounded — ``htail`` must lie in ``[0, 1]``
    — make a plain multivariate-normal draw produce unusable samples whenever the
    fitted value sits near a limit.  Dropping those samples shrinks the set the
    spread is computed from, and does so one-sidedly; this redraws instead, so
    the returned values are a full-size sample of the *constrained* distribution.

    Drawing is capped at *max_draws* so a fit whose parameters are almost never
    valid terminates instead of looping forever.

    Parameters
    ----------
    rng
        Random generator used for the draws.
    pars
        Mean of the multivariate normal, i.e. the best-fit parameters.
    cov
        Covariance matrix of the best-fit parameters.
    evaluate
        Callable applied to each draw.  A draw is rejected if this raises or
        returns a non-finite value.
    size
        Number of valid samples to collect.
    max_draws
        Maximum number of draws to attempt.  Defaults to ``20 * size``.
    bounds
        Optional ``{index: (lo, hi)}`` map of the support to draw from, with
        ``None`` for an open side.  A draw falling outside it is rejected.
        Pass the limits the fit itself used, so the sample comes from the model
        that was actually fitted rather than from a wider region the fit could
        never have reached.

    Returns
    -------
    accepted
        Array of the accepted parameter vectors, ``(n_valid, len(pars))``.
    values
        Array of the corresponding *evaluate* results, ``(n_valid,)``.
    n_drawn
        Total number of draws attempted, including rejected ones.
    last_error
        The most recent rejection cause, or ``None`` if nothing was rejected.
    """
    if max_draws is None:
        max_draws = 20 * size

    accepted = []
    values = []
    n_drawn = 0
    last_error = None

    while len(values) < size and n_drawn < max_draws:
        batch = rng.multivariate_normal(pars, cov, size=size)
        for p in batch:
            if n_drawn >= max_draws:
                break
            n_drawn += 1
            if bounds is not None and not _within_bounds(p, bounds):
                last_error = "outside fit bounds"
                continue
            try:
                value = evaluate(p)
            except Exception as e:
                # a draw that violates a bounded parameter is simply rejected
                last_error = e
                continue
            if not np.isfinite(value):
                last_error = "non-finite value"
                continue
            accepted.append(p)
            values.append(value)
            if len(values) == size:
                break

    if len(values) < size:
        log.warning(
            "bootstrap: only %s/%s draws valid after %s attempts: %s",
            len(values),
            size,
            n_drawn,
            last_error,
        )
    elif n_drawn > size:
        log.debug(
            "bootstrap: %s draws redrawn to reach %s valid samples: %s",
            n_drawn - size,
            size,
            last_error,
        )

    accepted = (
        np.asarray(accepted)
        if accepted
        else np.empty((0, np.size(pars)))  # keep the 2D shape when all rejected
    )

    return accepted, np.asarray(values), n_drawn, last_error


def hpge_peak_fwhm(
    sigma: float, htail: float, tau: float, cov: float | None = None
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
        msg = "htail outside allowed limits of 0 and 1"
        raise ValueError(msg)

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
        msg = "htail outside allowed limits of 0 and 1"
        raise ValueError(msg)

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
    _, y_b, _, _ = bootstrap_valid_pars(
        rng,
        pars,
        cov,
        lambda p: hpge_peak_fwfm(p[2], p[3], p[4], frac_max=frac_max),
    )
    yerr_boot = np.nanstd(y_b, axis=0) if len(y_b) > 0 else np.nan

    return upper_hm - lower_hm, yerr_boot


def hpge_peak_mode(mu, sigma, htail, tau, cov=None):
    if htail < 0 or htail > 1:
        if cov is not None:
            return np.nan, np.nan
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
    # nsig set to 1, hstep+nbkg set to 0
    pars = np.array([1, mu, sigma, htail, tau, 0, 0])
    rng = np.random.default_rng(1)
    # draws with an out-of-range htail come back as nan and are dropped rather
    # than redrawn: at this sample size the spread is already precise, and
    # redrawing costs up to 2x the runtime on the energy-calibration hot path
    # for a sub-percent change in the result
    n_boot = 10000
    par_b = rng.multivariate_normal(pars, cov, size=n_boot)
    modes = np.array([hpge_peak_mode(p[1], p[2], p[3], p[4]) for p in par_b])
    n_bad = int(np.count_nonzero(~np.isfinite(modes)))
    if n_bad:
        log.debug(
            "hpge_peak_mode: %s/%s bootstrap draws unusable (htail outside [0, 1])",
            n_bad,
            n_boot,
        )
    mode_err_boot = np.nanstd(modes, axis=0) if n_bad < n_boot else np.nan

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
