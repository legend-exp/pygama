"""
This module provides functions for correcting the a/e energy dependence, determining the cut level and calculating survival fractions.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from matplotlib.colors import LogNorm
from scipy.stats import chi2

import pygama.math.binned_fitting as pgf
import pygama.math.histogram as pgh
import pygama.pargen.energy_cal as pgc
from pygama.math.distributions import exgauss, gauss_on_exgauss, gaussian, nb_erfc
from pygama.math.functions.gauss import nb_gauss_amp
from pygama.math.functions.hpge_peak import hpge_get_fwfm, hpge_get_fwhm, hpge_get_mode
from pygama.math.functions.sum_dists import SumDists
from pygama.pargen.survival_fractions import (
    compton_sf,
    compton_sf_sweep,
    get_sf_sweep,
    get_survival_fraction,
)
from pygama.pargen.utils import convert_to_minuit, return_nans

log = logging.getLogger(__name__)

(x_lo, x_hi, n_sig, mu, sigma, n_bkg, tau) = range(7)
par_array = [(gaussian, [mu, sigma]), (exgauss, [mu, sigma, tau])]
aoe_peak = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg", "tau"],
    name="aoe_peak",
)


(x_lo, x_hi, n_sig, mu, sigma, frac1, tau, n_bkg, tau_bkg) = range(9)
par_array = [
    (gauss_on_exgauss, [mu, sigma, frac1, tau]),
    (exgauss, [mu, sigma, tau_bkg]),
]
aoe_peak_with_high_tail = SumDists(
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
        "tau_sig",
        "n_bkg",
        "tau",
    ],
    name="aoe_peak_with_high_tail",
)
aoe_peak_with_high_tail.get_fwfm = hpge_get_fwfm.__get__(aoe_peak_with_high_tail)
aoe_peak_with_high_tail.get_mode = hpge_get_mode.__get__(aoe_peak_with_high_tail)
aoe_peak_with_high_tail.get_fwhm = hpge_get_fwhm.__get__(aoe_peak_with_high_tail)


def aoe_peak_guess(func, hist, bins, var, **kwargs):
    """
    Build initial parameter guesses for an A/E peak fit.

    Estimates the peak centroid and width from the histogram using
    :func:`pgf.gauss_mode_width_max`, falling back to a simple Gaussian guess
    when that fails.  The returned values are packaged via
    :func:`~pygama.pargen.utils.convert_to_minuit` so they can be passed
    directly to :class:`~iminuit.Minuit`.

    Parameters
    ----------
    func
        PDF to guess parameters for; one of ``aoe_peak``,
        ``aoe_peak_with_high_tail``, ``exgauss``, or ``gaussian``.
    hist
        Histogram counts.
    bins
        Histogram bin edges.
    var
        Histogram bin variances.
    **kwargs
        Override specific guess values by name (e.g. ``mu=0.95``).

    Returns
    -------
    values
        :class:`~iminuit.util.ValueView` of initial parameter guesses for ``func``.
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mu = bin_centers[np.argmax(hist)]
    pars, _ = pgf.gauss_mode_width_max(hist, bins, var, mode_guess=mu, n_bins=5)
    bin_centres = pgh.get_bin_centers(bins)
    _, fallback_sigma, _ = pgh.get_gaussian_guess(hist, bins)

    if pars is None or (
        pars[0] > bins[-1]
        or pars[0] < bins[0]
        or pars[-1] < (0.5 * np.nanmax(hist))
        or pars[-1] > (2 * np.nanmax(hist))
        or pars[1] > 2 * fallback_sigma
        or pars[1] < 0.5 * fallback_sigma
    ):
        i_0 = np.argmax(hist)
        mu = bin_centres[i_0]
        sigma = fallback_sigma
    else:
        mu = pars[0]
        sigma = pars[1]

    ls_guess = 2 * np.sum(hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))])

    if func == aoe_peak:
        guess_dict = {
            "x_lo": bins[0],
            "x_hi": bins[-1],
            "n_sig": ls_guess,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": np.sum(hist) - ls_guess,
            "tau": 0.1,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

    elif func == aoe_peak_with_high_tail:
        guess_dict = {
            "x_lo": bins[0],
            "x_hi": bins[-1],
            "n_sig": ls_guess,
            "mu": mu,
            "sigma": sigma,
            "htail": 0.1,
            "tau_sig": -0.1,
            "n_bkg": np.sum(hist) - ls_guess,
            "tau": 0.1,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

    elif func == exgauss:
        guess_dict = {
            "x_lo": bins[0],
            "x_hi": bins[-1],
            "area": np.sum(hist) - ls_guess,
            "mu": mu,
            "sigma": sigma,
            "tau": 0.1,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

    elif func == gaussian:
        guess_dict = {
            "x_lo": bins[0],
            "x_hi": bins[-1],
            "area": ls_guess,
            "mu": mu,
            "sigma": sigma,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

    for item, value in kwargs.items():
        guess_dict[item] = value

    return convert_to_minuit(guess_dict, func).values  # noqa: PD011


def aoe_peak_bounds(func, guess, **kwargs):
    """
    Build parameter bounds for an A/E peak fit.

    parameters
    ----------
    func
        PDF to bound; one of ``aoe_peak``, ``aoe_peak_with_high_tail``,
        ``exgauss``, or ``gaussian``.
    guess
        Current parameter guess dict (used to derive relative bounds for
        ``mu`` and ``sigma``).
    **kwargs
        Override specific bounds by name (e.g. ``tau=(0, 0.5)``).

    Returns
    -------
    bounds_dict
        Dict mapping each parameter name to a ``(lower, upper)`` tuple
        suitable for passing to :attr:`~iminuit.Minuit.limits`.
    """
    if func == aoe_peak:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "n_sig": (0, None),
            "mu": ((guess["x_lo"] + guess["x_hi"]) / 2, guess["x_hi"]),
            "sigma": (
                (guess["x_hi"] - guess["x_lo"]) / 50,
                (guess["x_hi"] - guess["x_lo"]) / 4,
            ),
            "n_bkg": (0, None),
            "tau": (0, 1),
        }
    elif func == aoe_peak_with_high_tail:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "n_sig": (0, None),
            "mu": ((guess["x_lo"] + guess["x_hi"]) / 2, guess["x_hi"]),
            "sigma": (
                (guess["x_hi"] - guess["x_lo"]) / 50,
                (guess["x_hi"] - guess["x_lo"]) / 4,
            ),
            "htail": (0, 0.5),
            "tau_sig": (None, 0),
            "n_bkg": (0, None),
            "tau": (0, 1),
        }
    elif func == exgauss:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "area": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, (guess["x_hi"] - guess["x_lo"]) / 4),
            "tau": (0, 1),
        }
    elif func == gaussian:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "area": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, (guess["x_hi"] - guess["x_lo"]) / 4),
        }

    for item, value in kwargs.items():
        bounds_dict[item] = value
    return bounds_dict


def aoe_peak_fixed(func, **_kwargs):
    """
    Return the fixed parameters and free-parameter mask for an A/E peak fit.

    parameters
    ----------
    func
        PDF to query; one of ``aoe_peak``, ``aoe_peak_with_high_tail``,
        ``exgauss``, or ``gaussian``.
    **kwargs
        Unused; reserved for future overrides.

    Returns
    -------
    fixed
        List of parameter names to hold fixed (typically the fit-range
        boundaries ``x_lo`` and ``x_hi``).
    mask
        Boolean array aligned with ``func.required_args()``; ``True`` for
        free parameters, ``False`` for fixed ones.
    """
    if func in (aoe_peak, aoe_peak_with_high_tail):
        fixed = ["x_lo", "x_hi"]
    elif func == exgauss:
        fixed = ["x_lo", "x_hi", "mu", "sigma"]
    elif func == gaussian:
        fixed = ["x_lo", "x_hi"]
    mask = ~np.isin(func.required_args(), fixed)
    return fixed, mask


class Pol1:
    """Linear model for the A/E mean as a function of energy: mean(E) = a·E + b."""

    @staticmethod
    def func(x, a, b):
        """Evaluate ``a*x + b``."""
        return x * a + b

    @staticmethod
    def string_func(input_param):
        """Return the expression string for the hit-dict format."""
        return f"{input_param}*a+b"

    @staticmethod
    def guess(_bands, _means, _mean_errs):
        """Return initial parameter guess ``[a, b]``."""
        return [-1e-06, 5e-01]


class SigmaFit:
    """Energy-dependent A/E sigma model: sigma(E) = sqrt(a + (b/E)^c)."""

    @staticmethod
    def func(x, a, b, c):
        """Evaluate ``sqrt(a + (b/x)^c)``, returning NaN for non-positive arguments."""
        return np.where(
            (x > 0) & ((a + (b / (x + 10**-99)) ** c) > 0),
            np.sqrt(a + (b / (x + 10**-99)) ** c),
            np.nan,
        )

    @staticmethod
    def string_func(input_param):
        """Return the expression string for the hit-dict format."""
        return f"(a+(b/({input_param}+10**-99))**c)**(0.5)"

    @staticmethod
    def guess(_bands, sigmas, _sigma_errs):
        """Return initial parameter guess ``[a, b, c]``."""
        return [np.nanpercentile(sigmas, 50) ** 2, 2, 2]


class SigmoidFit:
    """Sigmoid model for A/E cut survival: sf(E) = (a + b·E) · erfc(c·E + d)."""

    @staticmethod
    def func(x, a, b, c, d):
        """Evaluate ``(a + b*x) * erfc(c*x + d)``."""
        return (a + b * x) * nb_erfc(c * x + d)

    @staticmethod
    def guess(_xs, ys, _y_errs):
        """Return initial parameter guess ``[a, b, c, d]``."""
        return [np.nanmax(ys) / 2, 0, 1, 1.5]


def unbinned_aoe_fit(
    aoe: np.array,
    pdf=aoe_peak,
    display: int = 0,
) -> tuple(np.array, np.array):
    """
    Fitting function for A/E, first fits just a Gaussian before using the full pdf to fit
    if fails will return NaN values

    parameters
    ----------
    aoe
        A/E values.
    pdf
        PDF to fit to.
    display
        Verbosity level; 0 = silent, >1 = diagnostic plots.

    Returns
    -------
    values
        Fitted parameter values.
    errors
        Fitted parameter errors.
    covariance
        Covariance matrix.
    fit
        Full fit info tuple ``(values, errors, covariance, gof, pdf, mask, valid, m)``.
    """
    if not isinstance(aoe, np.ndarray):
        aoe = np.array(aoe)

    # bin_width = (
    #     2
    #     * (np.nanpercentile(aoe, 75) - np.nanpercentile(aoe, 25))
    #     * len(aoe) ** (-1 / 3)
    # )
    # nbins = int(np.ceil((np.nanmax(aoe) - np.nanmin(aoe)) / bin_width))
    hist, bins, var = pgh.get_hist(
        aoe[(aoe < np.nanpercentile(aoe, 99)) & (aoe > np.nanpercentile(aoe, 10))],
        bins=500,
    )

    gpars = aoe_peak_guess(gaussian, hist, bins, var)
    c1_min = gpars["mu"] - 0.5 * gpars["sigma"]
    c1_max = gpars["mu"] + 3 * gpars["sigma"]
    gpars["x_lo"] = c1_min
    gpars["x_hi"] = c1_max
    # Initial fit just using Gaussian
    csig = cost.ExtendedUnbinnedNLL(
        aoe[(aoe < c1_max) & (aoe > c1_min)], gaussian.pdf_ext
    )

    msig = Minuit(csig, *gpars)

    bounds = aoe_peak_bounds(gaussian, gpars)
    for arg, val in bounds.items():
        msig.limits[arg] = val
    for fix in aoe_peak_fixed(gaussian)[0]:
        msig.fixed[fix] = True
    msig.migrad()

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = msig.values["mu"] - 15 * msig.values["sigma"]  # noqa: PD011
    fmin = max(fmin, np.nanmin(aoe))
    fmax_bkg = msig.values["mu"] - 5 * msig.values["sigma"]  # noqa: PD011
    fmax = msig.values["mu"] + 5 * msig.values["sigma"]  # noqa: PD011
    n_bkg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - msig.values["area"]  # noqa: PD011

    bkg_guess = aoe_peak_guess(
        exgauss,
        hist,
        bins,
        var,
        area=n_bkg_guess,
        mu=msig.values["mu"],  # noqa: PD011
        sigma=msig.values["sigma"] * 0.9,  # noqa: PD011
        x_lo=fmin,
        x_hi=fmax_bkg,
    )

    cbkg = cost.ExtendedUnbinnedNLL(
        aoe[(aoe < fmax_bkg) & (aoe > fmin)], exgauss.pdf_ext
    )
    mbkg = Minuit(cbkg, *bkg_guess)

    bounds = aoe_peak_bounds(exgauss, bkg_guess)

    for arg, val in bounds.items():
        mbkg.limits[arg] = val
    for fix in aoe_peak_fixed(exgauss)[0]:
        mbkg.fixed[fix] = True
    mbkg.simplex().migrad()
    mbkg.hesse()

    nsig_guess = (
        msig.values["area"]  # noqa: PD011
        - np.diff(
            exgauss.cdf_ext(
                np.array([msig.values["mu"] - 2 * msig.values["sigma"], fmax]),  # noqa: PD011
                mbkg.values["area"],  # noqa: PD011
                msig.values["mu"],  # noqa: PD011
                msig.values["sigma"],  # noqa: PD011
                mbkg.values["tau"],  # noqa: PD011
            )
        )[0]
    )
    nsig_guess = max(nsig_guess, 0)

    x0 = aoe_peak_guess(
        pdf,
        hist,
        bins,
        var,
        n_sig=nsig_guess,
        mu=msig.values["mu"],  # noqa: PD011
        sigma=msig.values["sigma"],  # noqa: PD011
        n_bkg=mbkg.values["area"],  # noqa: PD011
        tau=mbkg.values["tau"],  # noqa: PD011
        x_lo=fmin,
        x_hi=fmax,
    )

    bounds = aoe_peak_bounds(pdf, x0)

    gof_hist, gof_bins, gof_var = pgh.get_hist(
        aoe[(aoe < fmax) & (aoe > fmin)], bins=100, range=(fmin, fmax)
    )

    # Full fit using Gaussian signal with Gaussian tail background
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax) & (aoe > fmin)], pdf.pdf_ext)
    m = Minuit(c, *x0)
    for arg, val in bounds.items():
        m.limits[arg] = val
    fixed, mask = aoe_peak_fixed(pdf)
    for fix in fixed:
        m.fixed[fix] = True
    m.migrad()
    m.hesse()

    valid1 = (
        m.valid
        & (~np.isnan(np.array(m.errors)[mask]).any())
        & (~(np.array(m.errors)[mask] == 0).all())
    )

    cs = pgf.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        pdf.get_pdf,
        m.values,
        method="Pearson",
        scale_bins=True,
    )
    cs = (cs[0], cs[1] + len(np.where(mask)[0]))

    fit1 = (m.values, m.errors, m.covariance, cs, pdf, mask, valid1, m)  # noqa: PD011

    m2 = Minuit(c, *x0)
    for arg, val in bounds.items():
        m2.limits[arg] = val
    fixed, mask = aoe_peak_fixed(pdf)
    for fix in fixed:
        m2.fixed[fix] = True
    m2.simplex().migrad()
    m2.hesse()

    valid2 = (
        m2.valid
        & (~np.isnan(np.array(m2.errors)[mask]).any())
        & (~(np.array(m2.errors)[mask] == 0).all())
    )
    cs2 = pgf.goodness_of_fit(
        gof_hist,
        gof_bins,
        gof_var,
        pdf.get_pdf,
        m2.values,
        method="Pearson",
        scale_bins=True,
    )
    cs2 = (cs2[0], cs2[1] + len(np.where(mask)[0]))

    fit2 = (m2.values, m2.errors, m2.covariance, cs2, pdf, mask, valid2, m2)  # noqa: PD011

    frac_errors1 = np.sum(
        np.abs(
            np.array(m.errors)[mask & (np.array(m.values) > 0)]
            / np.array(m.values)[mask & (np.array(m.values) > 0)]
        )
    )
    frac_errors2 = np.sum(
        np.abs(
            np.array(m2.errors)[mask & (np.array(m2.values) > 0)]
            / np.array(m2.values)[mask & (np.array(m2.values) > 0)]
        )
    )

    if valid2 is False:
        fit = fit1
    elif valid1 is False:
        fit = fit2
    elif cs[0] * 1.05 < cs2[0]:
        fit = fit1

    elif cs2[0] * 1.05 < cs[0]:
        fit = fit2

    elif frac_errors1 <= frac_errors2:
        fit = fit1

    elif frac_errors1 > frac_errors2:
        fit = fit2
    else:
        fit = fit1

    if display > 1:
        aoe = aoe[(aoe < fmax) & (aoe > fmin)]
        nbins = 100

        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, _bars = plt.hist(aoe, bins=nbins, histtype="step", label="Data")
        dx = np.diff(bins)
        plt.plot(xs, pdf.get_pdf(xs, *x0) * dx[0], label="Guess")
        plt.plot(xs, pdf.get_pdf(xs, *fit[0]) * dx[0], label="Full Fit")
        pdf.components = True
        sig, bkg = pdf.get_pdf(xs, *fit[0])
        pdf.components = False
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(
            xs,
            gaussian.pdf_ext(xs, *msig.values)[1] * dx[0],  # noqa: PD011
            label="Initial Gaussian",
        )
        plt.plot(xs, exgauss.pdf_ext(xs, *mbkg.values)[1] * dx[0], label="Bkg guess")  # noqa: PD011
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        bin_centers = (bins[1:] + bins[:-1]) / 2
        res = (pdf.pdf(bin_centers, *fit[0]) * dx[0]) - counts
        plt.plot(
            bin_centers,
            [
                re / count if count != 0 else re
                for re, count in zip(res, counts, strict=False)
            ],
            label="Normalised Residuals",
        )
        plt.legend(loc="upper left")
        plt.show()
        return fit[0], fit[1], fit[2], fit
    return fit[0], fit[1], fit[2], fit


def fit_time_means(tstamps, means, sigmas):
    """
    Fit the time dependence of the means of the A/E distribution

    parameters
    ----------
    tstamps
        Timestamps of the data.
    means
        Means of the A/E distribution.
    sigmas
        Sigmas of the A/E distribution.

    Returns
    -------
    out_dict
        Dictionary mapping each timestamp to its corrected mean value.
    """
    out_dict = {}
    current_tstamps = []
    current_means = []
    current_sigmas = []
    rolling_mean = means[
        np.where(
            (np.abs(np.diff(means)) < (0.4 * np.array(sigmas)[1:]))
            & (~np.isnan(np.abs(np.diff(means)) < (0.4 * np.array(sigmas)[1:])))
        )[0][0]
    ]
    for i, tstamp in enumerate(tstamps):
        if (
            (
                np.abs(means[i] - rolling_mean) > 0.4 * sigmas[i]
                and np.abs(means[i] - rolling_mean) > rolling_mean * 0.01
            )
            or np.isnan(means[i])
            or np.isnan(sigmas[i])
        ):
            if i + 1 == len(means):
                out_dict[tstamp] = np.nan
            elif (np.abs(means[i + 1] - means[i]) < 0.4 * sigmas[i + 1]) and not (
                np.isnan(means[i])
                or np.isnan(means[i + 1])
                or np.isnan(sigmas[i])
                or np.isnan(sigmas[i + 1])
            ):
                for ts in current_tstamps:
                    out_dict[ts] = rolling_mean
                rolling_mean = means[i]
                current_means = [means[i]]
                current_tstamps = [tstamp]
                current_sigmas = [sigmas[i]]
            else:
                out_dict[tstamp] = np.nan
        else:
            current_tstamps.append(tstamp)
            current_means.append(means[i])
            current_sigmas.append(sigmas[i])
            rolling_mean = np.average(
                current_means, weights=1 / np.array(current_sigmas)
            )
    for tstamp in current_tstamps:
        out_dict[tstamp] = rolling_mean
    return out_dict


def average_consecutive(tstamps, means):
    """
    Fit the time dependence of the means of the A/E distribution by average consecutive entries

    parameters
    ----------
    tstamps
        Timestamps of the data.
    means
        Means of the A/E distribution.

    Returns
    -------
    out_dict
        Dictionary mapping each timestamp to the average of its mean and the next.
    """
    out_dict = {}
    for i, tstamp in enumerate(tstamps):
        if i + 1 == len(means):
            out_dict[tstamp] = means[i]
        else:
            out_dict[tstamp] = np.mean([means[i], means[i + 1]])
    return out_dict


def interpolate_consecutive(tstamps, means, times, aoe_param, output_name):
    """
    Fit the time dependence of the means of the A/E distribution by average consecutive entries

    parameters
    ----------
    tstamps
        Timestamps of the data.
    means
        Means of the A/E distribution.
    times
        Unix timestamps of each mean sample.
    aoe_param
        Name of the A/E parameter in the dataframe.
    output_name
        Key to use for the correction expression in the output dict.

    Returns
    -------
    out_dict
        Dictionary mapping each timestamp to a per-output-name correction expression.
    """
    out_dict = {}
    for i, tstamp in enumerate(tstamps):
        if i + 1 == len(means):
            out_dict[tstamp] = {
                output_name: {
                    "expression": f"{aoe_param}/a",
                    "parameters": {"a": means[i]},
                }
            }
        else:
            out_dict[tstamp] = {
                output_name: {
                    "expression": f"{aoe_param} / ((timestamp - a ) * b + c) ",
                    "parameters": {
                        "a": times[i],
                        "b": (means[i + 1] - means[i]) / (times[i + 1] - times[i]),
                        "c": means[i],
                    },
                }
            }

    return out_dict


class CalAoE:
    """
    Class for calibrating the A/E,
    this follow 5 steps:
    the time correction, drift time correction, energy correction,
    the cut level calculation and the survival fraction calculation.
    """

    def __init__(
        self,
        cal_dicts: dict | None = None,
        cal_energy_param: str = "cuspEmax_ctc_cal",
        eres_func: callable = lambda _x: 1,
        pdf=aoe_peak,
        selection_string: str = "index==index",
        dt_corr: bool = False,
        dep_correct: bool = False,
        dt_cut: dict | None = None,
        dt_param: str = "dt_eff",
        high_cut_val: float = 3,
        mean_func: Callable = Pol1,
        sigma_func: Callable = SigmaFit,
        compt_bands_width: float = 20,
        debug_mode: bool = False,
    ):
        """
        parameters
        ----------

        cal_dicts
            Dictionary of calibration parameters; can be empty/None for a single run, or keyed by
            timestamp in the format ``YYYYMMDDTHHMMSSZ`` for multiple runs.
        cal_energy_param
            Calibrated energy parameter to use for A/E calibrations and for determining peak events.
        eres_func
            Function to determine the energy resolution; should accept a single calibrated energy value.
        pdf
            PDF to fit to the A/E distribution.
        selection_string
            Selection string passed as a query to the data DataFrame.
        dt_corr
            Whether to apply drift-time correction.
        dep_correct
            Whether to correct the double escape peak into the single-site band before cut determination.
        dt_cut
            Dictionary of the drift time cut parameters in the form::

                {"out_param": "dt_cut", "hard": False}

            where ``out_param`` is the name of the parameter to cut on in the dataframe (should have
            been precalculated) and ``"hard"`` controls whether these events are removed completely
            for survival fraction calculations, or only for cut determination / A/E calibration steps.
        dt_param
            Name of the drift-time parameter in the dataframe.
        high_cut_val
            Value to cut the A/E distribution at on the high side.
        mean_func
            Class implementing the energy-dependence model for the A/E mean (e.g. :class:`Pol1`).
        sigma_func
            Class implementing the energy-dependence model for the A/E sigma (e.g. :class:`SigmaFit`).
        compt_bands_width
            Width of the Compton bands in keV used for the energy correction.
        debug_mode
            If ``True``, re-raise exceptions from the A/E calibration steps instead of returning NaN.

        """
        self.cal_dicts = cal_dicts if cal_dicts is not None else {}
        self.cal_energy_param = cal_energy_param
        self.eres_func = eres_func
        self.pdf = pdf
        self.selection_string = selection_string
        self.dt_corr = dt_corr
        self.dt_param = dt_param
        self.dep_correct = dep_correct
        self.dt_cut = dt_cut
        if self.dt_cut is not None:
            self.dt_cut_param = dt_cut["out_param"]
            self.fit_selection = f"{self.selection_string} & {self.dt_cut_param}"
            self.dt_cut_hard = dt_cut["hard"]
        else:
            self.dt_cut_param = None
            self.dt_cut_hard = False
            self.fit_selection = self.selection_string
        self.high_cut_val = high_cut_val
        self.mean_func = mean_func
        self.sigma_func = sigma_func
        self.compt_bands_width = compt_bands_width
        self.debug_mode = debug_mode

    def update_cal_dicts(self, update_dict):
        """
        Merge new entries into the A/E calibration dictionary.

        If ``cal_dicts`` is keyed by run timestamps, each timestamp's
        sub-dict is updated individually, using *update_dict* directly as a
        fallback when a timestamp is absent.  Otherwise *update_dict* is
        merged directly.

        parameters
        ----------
        update_dict
            Dictionary of new calibration entries to merge.
        """
        if len(self.cal_dicts) > 0 and re.match(
            r"(\d{8})T(\d{6})Z", next(iter(self.cal_dicts))
        ):
            for tstamp in self.cal_dicts:
                if tstamp in update_dict:
                    self.cal_dicts[tstamp].update(update_dict[tstamp])
                else:
                    self.cal_dicts[tstamp].update(update_dict)
        else:
            self.cal_dicts.update(update_dict)

    def time_correction(
        self,
        df: pd.DataFrame,
        aoe_param: str,
        mode: str = "full",
        output_name: str = "AoE_Timecorr",
        display: int = 0,
    ):
        """
        Calculates the time correction for the A/E parameter by fitting the 1-1.3 MeV
        Compton continuum and extracting the centroid. If only a single run is passed will
        just perform a shift of the A/E parameter to 1 using the centroid otherwise for multiple
        runs the shift will be determined on the mode given in.

        parameters
        ----------

        df
            DataFrame containing the data.
        aoe_param
            Name of the A/E parameter to use.
        mode
            Mode to use for the time correction, can be ``"full"``, ``"partial"``, or ``"none"``:

            none: just use the mean of the a/e centroids to shift all the data
            partial: iterate through the centroids if vary by less than 0.4 sigma
            then group and take mean otherwise when a run higher than 0.4 sigma
            is found if it is a single run set to nan otherwise start a new block
            full : each run will be corrected individually
            average_consecutive: average the consecutive centroids
            interpolate_consecutive: interpolate between the consecutive centroids
        output_name
            Name of the output parameter for the time-corrected A/E in the dataframe and to
            be added to the calibration dictionary.
        display
            Plot verbosity level.

        """
        log.info("Starting A/E time correction")
        self.timecorr_df = pd.DataFrame()
        try:
            if "run_timestamp" in df:
                for tstamp, time_df in df.groupby("run_timestamp", sort=True):
                    try:
                        pars, errs, _, _ = unbinned_aoe_fit(
                            time_df.query(
                                f"{self.fit_selection} & ({self.cal_energy_param}>1000) & ({self.cal_energy_param}<1300)"
                            )[aoe_param],
                            pdf=self.pdf,
                            display=display,
                        )
                        self.timecorr_df = pd.concat(
                            [
                                self.timecorr_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "run_timestamp": tstamp,
                                            "mean": pars["mu"],
                                            "mean_err": errs["mu"],
                                            "sigma": pars["sigma"],
                                            "sigma_err": errs["sigma"],
                                            "res": pars["sigma"] / pars["mu"],
                                            "res_err": (pars["sigma"] / pars["mu"])
                                            * np.sqrt(
                                                errs["sigma"] / pars["sigma"]
                                                + errs["mu"] / pars["mu"]
                                            ),
                                        }
                                    ]
                                ),
                            ]
                        )
                    except BaseException as e:
                        if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                            raise (e)
                        self.timecorr_df = pd.concat(
                            [
                                self.timecorr_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "run_timestamp": tstamp,
                                            "mean": np.nan,
                                            "mean_err": np.nan,
                                            "sigma": np.nan,
                                            "sigma_err": np.nan,
                                            "res": np.nan,
                                            "res_err": np.nan,
                                        }
                                    ]
                                ),
                            ]
                        )
                self.timecorr_df = self.timecorr_df.set_index("run_timestamp")
                if len(self.timecorr_df) > 1:
                    if mode == "partial":
                        time_dict = fit_time_means(
                            np.array(self.timecorr_df.index),
                            np.array(self.timecorr_df["mean"]),
                            np.array(self.timecorr_df["sigma"]),
                        )
                        final_time_dict = {
                            tstamp: {
                                output_name: {
                                    "expression": f"{aoe_param}/a",
                                    "parameters": {"a": t_dict},
                                }
                            }
                            for tstamp, t_dict in time_dict.items()
                        }

                    elif mode == "full":
                        time_dict = dict(
                            zip(
                                np.array(self.timecorr_df.index),
                                np.array(self.timecorr_df["mean"]),
                                strict=False,
                            )
                        )
                        final_time_dict = {
                            tstamp: {
                                output_name: {
                                    "expression": f"{aoe_param}/a",
                                    "parameters": {"a": t_dict},
                                }
                            }
                            for tstamp, t_dict in time_dict.items()
                        }

                    elif mode == "none":
                        time_dict = {
                            time: np.nanmean(self.timecorr_df["mean"])
                            for time in np.array(self.timecorr_df.index)
                        }
                        final_time_dict = {
                            tstamp: {
                                output_name: {
                                    "expression": f"{aoe_param}/a",
                                    "parameters": {"a": t_dict},
                                }
                            }
                            for tstamp, t_dict in time_dict.items()
                        }

                    elif mode == "average_consecutive":
                        time_dict = average_consecutive(
                            np.array(self.timecorr_df.index),
                            np.array(self.timecorr_df["mean"]),
                        )
                        final_time_dict = {
                            tstamp: {
                                output_name: {
                                    "expression": f"{aoe_param}/a",
                                    "parameters": {"a": t_dict},
                                }
                            }
                            for tstamp, t_dict in time_dict.items()
                        }

                    elif mode == "interpolate_consecutive":
                        if "timestamp" in df:
                            times = []
                            for tstamp in np.array(self.timecorr_df.index):
                                times.append(
                                    np.nanmedian(
                                        df.query(f"run_timestamp=='{tstamp}'")[
                                            "timestamp"
                                        ]
                                    )
                                )
                            final_time_dict = interpolate_consecutive(
                                np.array(self.timecorr_df.index),
                                np.array(self.timecorr_df["mean"]),
                                times,
                                aoe_param,
                                output_name,
                            )
                            time_dict = dict(
                                zip(
                                    np.array(self.timecorr_df.index),
                                    np.array(self.timecorr_df["mean"]),
                                    strict=False,
                                )
                            )
                        else:
                            msg = "need timestamp column in dataframe for interpolation"
                            raise ValueError(msg)

                    else:
                        msg = "unknown mode"
                        raise ValueError(msg)

                    df[output_name] = df[aoe_param] / np.array(
                        [time_dict[tstamp] for tstamp in df["run_timestamp"]]
                    )
                    self.update_cal_dicts(final_time_dict)
                else:
                    df[output_name] = (
                        df[aoe_param] / np.array(self.timecorr_df["mean"])[0]
                    )
                    self.update_cal_dicts(
                        {
                            output_name: {
                                "expression": f"{aoe_param}/a",
                                "parameters": {
                                    "a": np.array(self.timecorr_df["mean"])[0]
                                },
                            }
                        }
                    )
                log.info("A/E time correction finished")
            else:
                try:
                    pars, errs, _, _ = unbinned_aoe_fit(
                        df.query(
                            f"{self.fit_selection} & {self.cal_energy_param}>1000 & {self.cal_energy_param}<1300"
                        )[aoe_param],
                        pdf=self.pdf,
                        display=display,
                    )
                    self.timecorr_df = pd.concat(
                        [
                            self.timecorr_df,
                            pd.DataFrame(
                                [
                                    {
                                        "run_timestamp": np.nan,
                                        "mean": pars["mu"],
                                        "mean_err": errs["mu"],
                                        "sigma": pars["sigma"],
                                        "sigma_err": errs["sigma"],
                                        "res": pars["sigma"] / pars["mu"],
                                        "res_err": (pars["sigma"] / pars["mu"])
                                        * np.sqrt(
                                            errs["sigma"] / pars["sigma"]
                                            + errs["mu"] / pars["mu"]
                                        ),
                                    }
                                ]
                            ),
                        ]
                    )
                except BaseException as e:
                    if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                        raise (e)

                    self.timecorr_df = pd.concat(
                        [
                            self.timecorr_df,
                            pd.DataFrame(
                                [
                                    {
                                        "run_timestamp": np.nan,
                                        "mean": np.nan,
                                        "mean_err": np.nan,
                                        "sigma": np.nan,
                                        "sigma_err": np.nan,
                                        "res": np.nan,
                                        "res_err": np.nan,
                                    }
                                ]
                            ),
                        ]
                    )
                df[output_name] = df[aoe_param] / pars["mu"]
                self.update_cal_dicts(
                    {
                        output_name: {
                            "expression": f"{aoe_param}/a",
                            "parameters": {"a": pars["mu"]},
                        }
                    }
                )
                log.info("Finished A/E time correction")
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                raise (e)
            log.error("A/E time correction failed")
            df[output_name] = df[aoe_param] / np.nan
            self.update_cal_dicts(
                {
                    output_name: {
                        "expression": f"{aoe_param}/a",
                        "parameters": {"a": np.nan},
                    }
                }
            )

    def drift_time_correction(
        self,
        data: pd.DataFrame,
        aoe_param,
        out_param="AoE_DTcorr",
        display: int = 0,
    ):
        """
        Calculates the correction needed to align the two drift time regions for ICPC detectors.
        This is done by fitting the two drift time peaks in the drift time spectrum and then
        fitting the A/E peaks in each of these regions. A simple linear correction is then applied
        to align these regions.

        parameters
        ----------

        data
            DataFrame containing the data.
        aoe_param
            Name of the A/E parameter to use as starting point.
        out_param
            Name of the output parameter for the drift-time-corrected A/E in the dataframe and to
            be added to the calibration dictionary.
        display
            Plot verbosity level.

        """
        log.info("Starting A/E drift time correction")
        self.dt_res_dict = {}
        try:
            dep_events = data.query(
                f"{self.fit_selection}&{self.cal_energy_param}>1582&{self.cal_energy_param}<1602&{self.cal_energy_param}=={self.cal_energy_param}&{aoe_param}=={aoe_param}"
            )

            hist, bins, var = pgh.get_hist(
                dep_events[aoe_param],
                bins=500,
            )
            bin_cs = (bins[1:] + bins[:-1]) / 2
            mu = bin_cs[np.argmax(hist)]
            aoe_range = [mu * 0.9, mu * 1.1]

            dt_range = [
                np.nanpercentile(dep_events[self.dt_param], 1),
                np.nanpercentile(dep_events[self.dt_param], 99),
            ]

            self.dt_res_dict["final_selection"] = (
                f"{aoe_param}>{aoe_range[0]}&{aoe_param}<{aoe_range[1]}&{self.dt_param}>{dt_range[0]}&{self.dt_param}<{dt_range[1]}&{self.dt_param}=={self.dt_param}"
            )

            final_df = dep_events.query(self.dt_res_dict["final_selection"])

            hist, bins, var = pgh.get_hist(
                final_df[self.dt_param],
                dx=32,
                range=(
                    np.nanmin(final_df[self.dt_param]),
                    np.nanmax(final_df[self.dt_param]),
                ),
            )

            bcs = pgh.get_bin_centers(bins)
            mus = bcs[pgc.get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 2)]
            pk_pars, _pk_covs = pgc.hpge_fit_energy_peak_tops(
                hist,
                bins,
                var=var,
                peak_locs=mus,
                n_to_fit=5,
            )

            mus = pk_pars[:, 0]
            sigmas = pk_pars[:, 1]
            amps = pk_pars[:, 2]

            if len(mus) > 2:
                ids = np.array(
                    sorted([np.argmax(amps), np.argmax(amps[amps != np.argmax(amps)])])
                )
            else:
                ids = np.full(len(mus), True, dtype=bool)
            mus = mus[ids]
            sigmas = sigmas[ids]
            amps = amps[ids]

            self.dt_res_dict["dt_fit"] = {"mus": mus, "sigmas": sigmas, "amps": amps}

            if len(mus) < 2:
                log.info("Only 1 drift time peak found, no correction needed")
                self.alpha = 0

            else:
                aoe_grp1 = self.dt_res_dict["aoe_grp1"] = (
                    f"{self.dt_param}>{mus[0] - 2 * sigmas[0]} & {self.dt_param}<{mus[0] + 2 * sigmas[0]}"
                )
                aoe_grp2 = self.dt_res_dict["aoe_grp2"] = (
                    f"{self.dt_param}>{mus[1] - 2 * sigmas[1]} & {self.dt_param}<{mus[1] + 2 * sigmas[1]}"
                )

                aoe_pars, aoe_errs, _, _ = unbinned_aoe_fit(
                    final_df.query(aoe_grp1)[aoe_param], pdf=self.pdf, display=display
                )

                self.dt_res_dict["aoe_fit1"] = {
                    "pars": aoe_pars.to_dict(),
                    "errs": aoe_errs.to_dict(),
                }

                aoe_pars2, aoe_errs2, _, _ = unbinned_aoe_fit(
                    final_df.query(aoe_grp2)[aoe_param], pdf=self.pdf, display=display
                )

                self.dt_res_dict["aoe_fit2"] = {
                    "pars": aoe_pars2.to_dict(),
                    "errs": aoe_errs2.to_dict(),
                }

                try:
                    self.alpha = (aoe_pars2["mu"] - aoe_pars["mu"]) / (
                        (mus[0] * aoe_pars["mu"]) - (mus[1] * aoe_pars2["mu"])
                    )
                except ZeroDivisionError:
                    self.alpha = 0
                self.dt_res_dict["alpha"] = self.alpha
                log.info("dtcorr successful alpha:%s", self.alpha)

        except BaseException as e:
            if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                raise (e)
            log.error("Drift time correction failed")
            self.alpha = 0

        data[out_param] = data[aoe_param] * (1 + self.alpha * data[self.dt_param])
        self.update_cal_dicts(
            {
                out_param: {
                    "expression": f"{aoe_param}*(1+a*{self.dt_param})",
                    "parameters": {"a": self.alpha},
                }
            }
        )

    def energy_correction(
        self,
        data: pd.DataFrame,
        aoe_param: str,
        corrected_param: str = "AoE_Corrected",
        classifier_param: str = "AoE_Classifier",
        display: int = 0,
    ):
        """
        Calculates the corrections needed for the energy dependence of the A/E.
        Does this by fitting the compton continuum in slices and then applies fits
        to the centroid and variance.

        parameters
        ----------

        data
            DataFrame containing the data.
        aoe_param
            Name of the A/E parameter to use as starting point.
        corrected_param
            Name of the output parameter for the energy-mean-corrected A/E to be added in the
            dataframe and to the calibration dictionary.
        classifier_param
            Name of the output parameter for the full mean- and sigma-corrected A/E classifier
            to be added in the dataframe and to the calibration dictionary.
        display
            Plot verbosity level.

        """

        log.info("Starting A/E energy correction")
        self.energy_corr_res_dict = {}

        compt_bands = np.arange(900, 2350, self.compt_bands_width)
        peaks = np.array(
            [1080, 1094, 1459, 1512, 1552, 1592, 1620, 1650, 1670, 1830, 2105]
        )
        allowed = np.array([], dtype=bool)
        for band in compt_bands:
            allow = True
            for peak in peaks:
                if (
                    (peak - 5) > band and (peak - 5) < (band + self.compt_bands_width)
                ) or (
                    (peak + 5 > band) and (peak + 5) < (band + self.compt_bands_width)
                ):
                    allow = False
            allowed = np.append(allowed, allow)
        compt_bands = compt_bands[allowed]

        self.energy_corr_fits = pd.DataFrame(
            columns=[
                "compt_bands",
                "mean",
                "mean_err",
                "sigma",
                "sigma_err",
                "ratio",
                "ratio_err",
            ],
            dtype=float,
        )
        try:
            select_df = data.query(f"{self.fit_selection} & {aoe_param}>0")

            # Fit each compton band
            for band in compt_bands:
                try:
                    pars, errs, cov, _ = unbinned_aoe_fit(
                        select_df.query(
                            f"{self.cal_energy_param}>{band}&{self.cal_energy_param}< {self.compt_bands_width + band}"
                        )[aoe_param],
                        pdf=self.pdf,
                        display=display,
                    )

                    mean, mean_err = self.pdf.get_mu(pars, cov)
                    sigma, sigma_err = self.pdf.get_fwhm(pars, cov)
                    sigma = sigma / 2.355
                    sigma_err = sigma_err / 2.355

                    self.energy_corr_fits = pd.concat(
                        [
                            self.energy_corr_fits,
                            pd.DataFrame(
                                [
                                    {
                                        "compt_bands": band
                                        + self.compt_bands_width / 2,
                                        "mean": mean,
                                        "mean_err": mean_err,
                                        "sigma": sigma,
                                        "sigma_err": sigma_err,
                                        "ratio": pars["n_sig"] / pars["n_bkg"],
                                        "ratio_err": (pars["n_sig"] / pars["n_bkg"])
                                        * np.sqrt(
                                            (errs["n_sig"] / pars["n_sig"]) ** 2
                                            + (errs["n_bkg"] / pars["n_bkg"]) ** 2
                                        ),
                                    }
                                ]
                            ),
                        ]
                    )

                except BaseException as e:
                    if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                        raise (e)
                    self.energy_corr_fits = pd.concat(
                        [
                            self.energy_corr_fits,
                            pd.DataFrame(
                                [
                                    {
                                        "compt_bands": band,
                                        "mean": np.nan,
                                        "mean_err": np.nan,
                                        "sigma": np.nan,
                                        "sigma_err": np.nan,
                                        "ratio": np.nan,
                                        "ratio_err": np.nan,
                                    }
                                ]
                            ),
                        ]
                    )
            self.energy_corr_fits = self.energy_corr_fits.set_index("compt_bands")
            valid_fits = self.energy_corr_fits.query(
                "mean_err==mean_err&sigma_err==sigma_err & sigma_err!=0 & mean_err!=0"
            )
            self.energy_corr_res_dict["n_of_valid_fits"] = len(valid_fits)
            log.info("%s compton bands fit successfully", len(valid_fits))
            # Fit mus against energy
            p0_mu = self.mean_func.guess(
                valid_fits.index, valid_fits["mean"], valid_fits["mean_err"]
            )
            c_mu = cost.LeastSquares(
                valid_fits.index,
                valid_fits["mean"],
                valid_fits["mean_err"],
                self.mean_func.func,
            )
            c_mu.loss = "soft_l1"
            m_mu = Minuit(c_mu, *p0_mu)
            m_mu.simplex()
            m_mu.migrad()
            m_mu.hesse()

            mu_pars = m_mu.values  # noqa: PD011
            mu_errs = m_mu.errors

            csqr_mu = np.sum(
                (
                    (
                        valid_fits["mean"]
                        - self.mean_func.func(valid_fits.index, *mu_pars)
                    )
                    ** 2
                )
                / valid_fits["mean_err"]
            )
            dof_mu = len(valid_fits["mean"]) - len(mu_pars)
            p_val_mu = chi2.sf(csqr_mu, dof_mu)
            self.mean_fit_obj = m_mu

            # Fit sigma against energy
            p0_sig = self.sigma_func.guess(
                valid_fits.index, valid_fits["sigma"], valid_fits["sigma_err"]
            )
            c_sig = cost.LeastSquares(
                valid_fits.index,
                valid_fits["sigma"],
                valid_fits["sigma_err"],
                self.sigma_func.func,
            )
            c_sig.loss = "soft_l1"
            m_sig = Minuit(c_sig, *p0_sig)
            m_sig.simplex()
            m_sig.migrad()
            m_sig.hesse()

            sig_pars = m_sig.values  # noqa: PD011
            sig_errs = m_sig.errors

            csqr_sig = np.sum(
                (
                    (
                        valid_fits["sigma"]
                        - self.sigma_func.func(valid_fits.index, *sig_pars)
                    )
                    ** 2
                )
                / valid_fits["sigma_err"]
            )
            dof_sig = len(valid_fits["sigma"]) - len(sig_pars)
            p_val_sig = chi2.sf(csqr_sig, dof_sig)

            self.SigmaFit_obj = m_sig

            # Get DEP fit
            n_sigma = 4
            peak = 1592
            sigma = self.eres_func(peak) / 2.355
            emin = peak - n_sigma * sigma
            emax = peak + n_sigma * sigma
            try:
                dep_pars, dep_err, _, _ = unbinned_aoe_fit(
                    select_df.query(
                        f"{self.cal_energy_param}>{emin}&{self.cal_energy_param}<{emax}"
                    )[aoe_param],
                    pdf=self.pdf,
                    display=display,
                )
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                    raise (e)

                dep_pars, dep_err, _ = return_nans(self.pdf)

            data[corrected_param] = data[aoe_param] / self.mean_func.func(
                data[self.cal_energy_param], *mu_pars
            )
            data[classifier_param] = (data[corrected_param] - 1) / self.sigma_func.func(
                data[self.cal_energy_param], *sig_pars
            )
            log.info("Finished A/E energy successful")
            log.info("mean pars are %s", mu_pars.to_dict())
            log.info("sigma pars are %s", sig_pars.to_dict())

        except BaseException as e:
            if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                raise (e)
            log.error("A/E energy correction failed")
            mu_pars, mu_errs, _mu_cov = return_nans(self.mean_func.func)
            csqr_mu, dof_mu, p_val_mu = (np.nan, np.nan, np.nan)
            csqr_sig, dof_sig, p_val_sig = (np.nan, np.nan, np.nan)
            sig_pars, sig_errs, _sig_cov = return_nans(self.sigma_func.func)
            dep_pars, dep_err, _dep_cov = return_nans(self.pdf)
            data[corrected_param] = data[aoe_param] * np.nan
            data[classifier_param] = data[aoe_param] * np.nan

        self.energy_corr_res_dict["mean_fits"] = {
            "func": self.mean_func.__name__,
            "module": self.mean_func.__module__,
            "expression": self.mean_func.string_func("x"),
            "pars": mu_pars.to_dict(),
            "errs": mu_errs.to_dict(),
            "p_val_mu": p_val_mu,
            "csqr_mu": (csqr_mu, dof_mu),
        }

        self.energy_corr_res_dict["SigmaFits"] = {
            "func": self.sigma_func.__name__,
            "module": self.sigma_func.__module__,
            "expression": self.sigma_func.string_func("x"),
            "pars": sig_pars.to_dict(),
            "errs": sig_errs.to_dict(),
            "p_val_mu": p_val_sig,
            "csqr_mu": (csqr_sig, dof_sig),
        }

        self.energy_corr_res_dict["dep_fit"] = {
            "func": self.pdf.name,
            "pars": dep_pars.to_dict(),
            "errs": dep_err.to_dict(),
        }

        self.update_cal_dicts(
            {
                corrected_param: {
                    "expression": f"{aoe_param}/({self.mean_func.string_func(self.cal_energy_param)})",
                    "parameters": mu_pars.to_dict(),
                },
                f"_{classifier_param}_intermediate": {
                    "expression": f"({aoe_param})-({self.mean_func.string_func(self.cal_energy_param)})",
                    "parameters": mu_pars.to_dict(),
                },
                classifier_param: {
                    "expression": f"(_{classifier_param}_intermediate)/({self.sigma_func.string_func(self.cal_energy_param)})",
                    "parameters": sig_pars.to_dict(),
                },
            }
        )

    def get_aoe_cut_fit(
        self,
        data: pd.DataFrame,
        aoe_param: str,
        peak: float,
        ranges: tuple,
        dep_acc: float = 0.9,
        output_cut_param: str = "AoE_Low_Cut",
        _display: int = 1,
    ):
        """
        Determines A/E cut by sweeping through values and for each one fitting the
        DEP to determine how many events survive.
        Fits the resulting distribution and
        interpolates to get cut value at desired DEP survival fraction (typically 90%)

        parameters
        ----------

        data
            DataFrame containing the data.
        aoe_param
            Name of the A/E parameter to use as starting point.
        peak
            Energy in keV of the peak to use for cut determination, e.g. ``1592.5``.
        ranges
            Tuple of ``(below, above)`` in keV defining the fit range around ``peak``, e.g. ``(20, 40)``.
        dep_acc
            Desired DEP survival fraction for the final cut.
        output_cut_param
            Name of the output boolean parameter for events passing the A/E cut in the dataframe
            and to be added to the calibration dictionary.
        display
            Plot verbosity level.

        """

        log.info("Starting A/E low cut determination")
        self.low_cut_res_dict = {}
        self.cut_fits = pd.DataFrame()

        min_range, max_range = ranges
        erange = (peak - min_range, peak + max_range)
        try:
            select_df = data.query(
                f"{self.fit_selection}&({self.cal_energy_param} > {erange[0]}) & ({self.cal_energy_param} < {erange[1]})"
            )

            # if dep_correct is True:
            #     peak_aoe = (select_df[aoe_param] / dep_mu(select_df[self.cal_energy_param])) - 1
            #     peak_aoe = select_df[aoe_param] / sig_func(select_df[self.cal_energy_param])

            self.cut_fits, _, _ = get_sf_sweep(
                select_df[self.cal_energy_param],
                select_df[aoe_param],
                None,
                peak,
                self.eres_func(peak),
                fit_range=erange,
                data_mask=None,
                cut_range=(-8, 0),
                n_samples=40,
                mode="greater",
                debug_mode=self.debug_mode,
            )

            valid_fits = self.cut_fits.query(
                f"sf_err<{(1.5 * np.nanpercentile(self.cut_fits['sf_err'], 85))}&sf_err==sf_err"
            )

            c = cost.LeastSquares(
                valid_fits.index,
                valid_fits["sf"],
                valid_fits["sf_err"],
                SigmoidFit.func,
            )
            c.loss = "soft_l1"
            m1 = Minuit(
                c,
                *SigmoidFit.guess(
                    valid_fits.index, valid_fits["sf"], valid_fits["sf_err"]
                ),
            )
            m1.simplex().migrad()
            xs = np.arange(
                np.nanmin(valid_fits.index), np.nanmax(valid_fits.index), 0.01
            )
            p = SigmoidFit.func(xs, *m1.values)  # noqa: PD011
            self.cut_fit = {
                "function": SigmoidFit.__name__,
                "pars": m1.values.to_dict(),  # noqa: PD011
                "errs": m1.errors.to_dict(),
            }
            self.low_cut_val = round(xs[np.argmin(np.abs(p - (100 * dep_acc)))], 3)
            log.info("Cut found at %s", self.low_cut_val)

            data[output_cut_param] = data[aoe_param] > self.low_cut_val
            if self.dt_cut_param is not None:
                data[output_cut_param] = (
                    data[output_cut_param] & (data[self.dt_cut_param])
                )
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                raise (e)
            log.error("A/E cut determination failed")
            self.low_cut_val = np.nan
            data[output_cut_param] = False

        self.update_cal_dicts(
            {
                output_cut_param: {
                    "expression": f"({aoe_param}>a)",
                    "parameters": {"a": self.low_cut_val},
                }
            }
        )

    def calculate_survival_fractions_sweep(
        self,
        data: pd.DataFrame,
        aoe_param: str,
        peaks: list,
        fit_widths: list[tuple],
        n_samples=26,
        cut_range=(-5, 5),
        mode="greater",
    ):
        """
        Calculate survival fractions for the A/E cut for a list of peaks by sweeping through values
        of the A/E cut to show how this varies

        parameters
        ----------

        data
            DataFrame containing the data.
        aoe_param
            Name of the parameter in the dataframe for the final A/E classifier.
        peaks
            List of peak energies in keV to calculate survival fractions for.
        fit_widths
            List of ``(below, above)`` tuples in keV defining the fit range around each peak.
        n_samples
            Number of cut values to sample in the sweep.
        cut_range
            Range of A/E cut values to sweep through.
        mode
            Whether to keep events with A/E ``"greater"`` or ``"less"`` than the cut value.

        """
        sfs = pd.DataFrame()
        peak_dfs = {}

        for i, peak in enumerate(peaks):
            try:
                select_df = data.query(
                    f"{self.selection_string}&{aoe_param}=={aoe_param}"
                )
                fwhm = self.eres_func(peak)
                if peak == 2039:
                    emin = 2 * fwhm
                    emax = 2 * fwhm
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{peak - emin})&({self.cal_energy_param}<{peak + emax})"
                    )

                    cut_df, sf, sf_err = compton_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        cut_range=cut_range,
                        n_samples=n_samples,
                        mode=mode,
                        data_mask=(
                            peak_df[self.dt_cut_param].to_numpy()
                            if self.dt_cut_param is not None
                            else None
                        ),
                    )
                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    peak_dfs[peak] = cut_df
                else:
                    emin, emax = fit_widths[i]
                    fit_range = (peak - emin, peak + emax)
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{peak - emin})&({self.cal_energy_param}<{peak + emax})"
                    )
                    cut_df, sf, sf_err = get_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        fit_range=fit_range,
                        cut_range=cut_range,
                        n_samples=n_samples,
                        mode=mode,
                        data_mask=(
                            peak_df[self.dt_cut_param].to_numpy()
                            if self.dt_cut_param is not None
                            else None
                        ),
                        debug_mode=self.debug_mode,
                    )

                    cut_df = cut_df.query(
                        f"sf_err<5*{np.nanpercentile(cut_df['sf_err'], 50)}& sf_err==sf_err & sf<=100"
                    )

                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    peak_dfs[peak] = cut_df
                log.info("%skeV: %.1f +/- %.1f %%", peak, sf, sf_err)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                    raise (e)
                sfs = pd.concat(
                    [
                        sfs,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(
                    "A/E Survival fraction sweep determination failed for %s peak",
                    peak,
                )
        sfs = sfs.set_index("peak")
        return sfs, peak_dfs

    def calculate_survival_fractions(
        self,
        data: pd.DataFrame,
        aoe_param: str,
        peaks: list,
        fit_widths: list[tuple],
        mode: str = "greater",
    ):
        """
        Calculate survival fractions for the A/E cut for a list of peaks for the final
        A/E cut value

        parameters
        ----------

        data
            DataFrame containing the data.
        aoe_param
            Name of the parameter in the dataframe for the final A/E classifier.
        peaks
            List of peak energies in keV to calculate survival fractions for.
        fit_widths
            List of ``(below, above)`` tuples in keV defining the fit range around each peak.
        mode
            Whether to keep events with A/E ``"greater"`` or ``"less"`` than the cut value.

        """
        sfs = pd.DataFrame()
        for i, peak in enumerate(peaks):
            try:
                fwhm = self.eres_func(peak)
                if peak == 2039:
                    emin = 2 * fwhm
                    emax = 2 * fwhm
                    peak_df = data.query(
                        f"({self.cal_energy_param}>{peak - emin})&({self.cal_energy_param}<{peak + emax})"
                    )

                    sf_dict = compton_sf(
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        self.high_cut_val,
                        mode=mode,
                        data_mask=(
                            peak_df[self.dt_cut_param].to_numpy()
                            if self.dt_cut_param is not None
                            else None
                        ),
                    )
                    sf = sf_dict["sf"]
                    sf_err = sf_dict["sf_err"]
                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                else:
                    emin, emax = fit_widths[i]
                    fit_range = (peak - emin, peak + emax)
                    peak_df = data.query(
                        f"({self.cal_energy_param}>{peak - emin})&({self.cal_energy_param}<{peak + emax})"
                    )
                    sf, sf_err, _, _ = get_survival_fraction(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        fit_range=fit_range,
                        mode=mode,
                        high_cut=self.high_cut_val,
                        data_mask=(
                            peak_df[self.dt_cut_param].to_numpy()
                            if self.dt_cut_param is not None
                            else None
                        ),
                    )
                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                log.info("%skeV: %.1f +/- %.1f %%", peak, sf, sf_err)

            except BaseException as e:
                if isinstance(e, KeyboardInterrupt) or self.debug_mode:
                    raise (e)
                sfs = pd.concat(
                    [
                        sfs,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(
                    "A/E survival fraction determination failed for %s peak", peak
                )
        return sfs.set_index("peak")

    def calibrate(
        self,
        df: pd.DataFrame,
        initial_aoe_param: str,
        peaks_of_interest: list | None = None,
        fit_widths: list[tuple] | None = None,
        cut_peak_idx: int = 0,
        dep_acc: float = 0.9,
        sf_nsamples: int = 11,
        sf_cut_range: tuple = (-5, 5),
        timecorr_mode: str = "full",
        override_dict: dict | None = None,
    ):
        """
        Main function to run a full A/E calibration with all steps i.e. time correction, drift time correction,
        energy correction, A/E cut determination and survival fraction calculation

        parameters
        ----------

        df
            DataFrame containing the data.
        initial_aoe_param
            Name of the A/E parameter in the dataframe to use as starting point.
        peaks_of_interest
            List of peak energies in keV to calculate survival fractions for.
        fit_widths
            List of ``(below, above)`` tuples in keV for the fit range around each peak.
        cut_peak_idx
            Index into ``peaks_of_interest`` of the peak used for cut determination.
        dep_acc
            Desired survival fraction in the DEP for the final cut value.
        sf_nsamples
            Number of samples to take in the survival fraction sweep.
        sf_cut_range
            Range of A/E cut values to use for the survival fraction sweep.
        timecorr_mode
            Mode to use for the time correction; see :meth:`time_correction` for details.
        override_dict
            Optional dictionary of pre-computed calibration entries that bypass one or more
            fit steps. Each key should map to a sub-dict with ``"expression"`` (a string
            expression) and ``"parameters"`` (a dict of named constants used by the expression).
            Supported keys and the fit step they replace:

            - ``"AoE_Timecorr"`` – skips :meth:`time_correction`; the expression is evaluated
              with columns from *df* plus the stored parameters to populate ``df["AoE_Timecorr"]``.
            - ``"AoE_DTcorr"`` – skips :meth:`drift_time_correction` (only relevant when
              ``self.dt_corr`` is ``True``); populates ``df["AoE_DTcorr"]``. Silently ignored
              if ``self.dt_corr`` is ``False``.
            - ``"AoE_Corrected"``, ``"_AoE_Classifier_intermediate"``, and ``"AoE_Classifier"``
              – together skip :meth:`energy_correction`. All three keys must be present to
              take the override path; if only one of ``"AoE_Corrected"`` or ``"AoE_Classifier"``
              is given (regardless of whether ``"_AoE_Classifier_intermediate"`` is present) a
              warning is logged and normal energy correction runs instead.
            - ``"AoE_Low_Cut"`` – skips :meth:`get_aoe_cut_fit`; the expression should evaluate
              to a boolean array. The numeric cut threshold is read from ``parameters["a"]``
              (or the sole parameter value if ``"a"`` is absent) and stored as
              ``self.low_cut_val``.

        """
        if peaks_of_interest is None:
            peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        if fit_widths is None:
            fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]

        if override_dict is None or "AoE_Timecorr" not in override_dict:
            self.time_correction(
                df, initial_aoe_param, mode=timecorr_mode, output_name="AoE_Timecorr"
            )
        else:
            self.update_cal_dicts({"AoE_Timecorr": override_dict["AoE_Timecorr"]})
            df["AoE_Timecorr"] = ne.evaluate(
                override_dict["AoE_Timecorr"]["expression"],
                local_dict={col: df[col].to_numpy() for col in df.columns}
                | override_dict["AoE_Timecorr"]["parameters"],
            )

        if self.dt_corr is True:
            aoe_param = "AoE_DTcorr"
            if override_dict is None or "AoE_DTcorr" not in override_dict:
                self.drift_time_correction(df, "AoE_Timecorr", out_param=aoe_param)
            else:
                self.update_cal_dicts({"AoE_DTcorr": override_dict["AoE_DTcorr"]})
                df["AoE_DTcorr"] = ne.evaluate(
                    override_dict["AoE_DTcorr"]["expression"],
                    local_dict={col: df[col].to_numpy() for col in df.columns}
                    | override_dict["AoE_DTcorr"]["parameters"],
                )
        else:
            aoe_param = "AoE_Timecorr"

        if override_dict is not None and (
            ("AoE_Corrected" in override_dict) != ("AoE_Classifier" in override_dict)
        ):
            log.warning(
                "override_dict must contain both 'AoE_Corrected' and 'AoE_Classifier' to "
                "override energy correction; ignoring partial override and running energy correction."
            )

        if override_dict is None or (
            "AoE_Corrected" not in override_dict
            or "AoE_Classifier" not in override_dict
        ):
            self.energy_correction(
                df,
                aoe_param,
                corrected_param="AoE_Corrected",
                classifier_param="AoE_Classifier",
            )
        else:
            self.update_cal_dicts({"AoE_Corrected": override_dict["AoE_Corrected"]})
            self.update_cal_dicts(
                {
                    "_AoE_Classifier_intermediate": override_dict[
                        "_AoE_Classifier_intermediate"
                    ]
                }
            )
            self.update_cal_dicts({"AoE_Classifier": override_dict["AoE_Classifier"]})
            df["AoE_Corrected"] = ne.evaluate(
                override_dict["AoE_Corrected"]["expression"],
                local_dict={col: df[col].to_numpy() for col in df.columns}
                | override_dict["AoE_Corrected"]["parameters"],
            )
            df["_AoE_Classifier_intermediate"] = ne.evaluate(
                override_dict["_AoE_Classifier_intermediate"]["expression"],
                local_dict={col: df[col].to_numpy() for col in df.columns}
                | override_dict["_AoE_Classifier_intermediate"]["parameters"],
            )
            df["AoE_Classifier"] = ne.evaluate(
                override_dict["AoE_Classifier"]["expression"],
                local_dict={col: df[col].to_numpy() for col in df.columns}
                | override_dict["AoE_Classifier"]["parameters"],
            )

        if override_dict is None or ("AoE_Low_Cut" not in override_dict):
            self.get_aoe_cut_fit(
                df,
                "AoE_Classifier",
                peaks_of_interest[cut_peak_idx],
                fit_widths[cut_peak_idx],
                dep_acc,
                output_cut_param="AoE_Low_Cut",
            )
        else:
            self.update_cal_dicts({"AoE_Low_Cut": override_dict["AoE_Low_Cut"]})
            df["AoE_Low_Cut"] = ne.evaluate(
                override_dict["AoE_Low_Cut"]["expression"],
                local_dict={col: df[col].to_numpy() for col in df.columns}
                | override_dict["AoE_Low_Cut"]["parameters"],
            )

            # Ensure the numeric low-cut threshold is stored for downstream use
            params = override_dict["AoE_Low_Cut"].get("parameters", {})
            low_cut_val = None
            if "a" in params:
                low_cut_val = params["a"]
            elif len(params) == 1:
                # Fall back to the single provided parameter if its name is not "a"
                low_cut_val = next(iter(params.values()))
            if low_cut_val is not None:
                self.low_cut_val = low_cut_val
        df["AoE_Double_Sided_Cut"] = df["AoE_Low_Cut"] & (
            df["AoE_Classifier"] < self.high_cut_val
        )

        self.update_cal_dicts(
            {
                "AoE_High_Side_Cut": {
                    "expression": "(a>AoE_Classifier)",
                    "parameters": {"a": self.high_cut_val},
                }
            }
        )

        self.update_cal_dicts(
            {
                "AoE_Double_Sided_Cut": {
                    "expression": "(AoE_High_Side_Cut) & (AoE_Low_Cut)",
                    "parameters": {},
                }
            }
        )

        log.info("Compute low side survival fractions: ")
        (
            self.low_side_sfs,
            self.low_side_peak_dfs,
        ) = self.calculate_survival_fractions_sweep(
            df,
            "AoE_Classifier",
            peaks_of_interest,
            fit_widths,
            n_samples=sf_nsamples,
            cut_range=sf_cut_range,
            mode="greater",
        )

        log.info("Compute 2 side survival fractions: ")
        self.two_side_sfs = self.calculate_survival_fractions(
            df, "AoE_Classifier", peaks_of_interest, fit_widths, mode="greater"
        )

        if re.match(r"(\d{8})T(\d{6})Z", next(iter(self.cal_dicts))):
            self.low_side_sfs_by_run = {}
            self.two_side_sfs_by_run = {}
            for tstamp in self.cal_dicts:
                log.info("Compute survival fractions for %s: ", tstamp)
                self.low_side_sfs_by_run[tstamp] = self.calculate_survival_fractions(
                    df.query(f"run_timestamp == '{tstamp}'"),
                    "AoE_Classifier",
                    peaks_of_interest,
                    fit_widths,
                    mode="greater",
                )

                self.two_side_sfs_by_run[tstamp] = self.calculate_survival_fractions(
                    df.query(f"run_timestamp == '{tstamp}'"),
                    "AoE_Classifier",
                    peaks_of_interest,
                    fit_widths,
                    mode="greater",
                )
        else:
            self.low_side_sfs_by_run = None
            self.two_side_sfs_by_run = None


# below are a few plotting functions that can be used to plot the results of the A/E calibration


def plot_aoe_mean_time(
    aoe_class, _data, time_param="AoE_Timecorr", figsize=(12, 8), fontsize=12
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(1, 1)
    try:
        ax.errorbar(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.timecorr_df.index
            ],
            aoe_class.timecorr_df["mean"],
            yerr=aoe_class.timecorr_df["mean_err"],
            linestyle=" ",
            marker="x",
        )

        grouped_means = [
            cal_dict[time_param]["parameters"]["a"]
            for tstamp, cal_dict in aoe_class.cal_dicts.items()
        ]
        ax.step(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.cal_dicts
            ],
            grouped_means,
            where="post",
        )
        ax.fill_between(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.cal_dicts
            ],
            y1=np.array(grouped_means) - 0.2 * np.array(aoe_class.timecorr_df["sigma"]),
            y2=np.array(grouped_means) + 0.2 * np.array(aoe_class.timecorr_df["sigma"]),
            color="green",
            alpha=0.2,
        )
        ax.fill_between(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.cal_dicts
            ],
            y1=np.array(grouped_means) - 0.4 * np.array(aoe_class.timecorr_df["sigma"]),
            y2=np.array(grouped_means) + 0.4 * np.array(aoe_class.timecorr_df["sigma"]),
            color="yellow",
            alpha=0.2,
        )
    except Exception:
        pass
    ax.set_xlabel("time")
    ax.set_ylabel("A/E mean")
    myfmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(myfmt)
    plt.close()
    return fig


def plot_aoe_res_time(
    aoe_class, _data, _time_param="AoE_Timecorr", figsize=(12, 8), fontsize=12
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(1, 1)
    with contextlib.suppress(Exception):
        ax.errorbar(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.timecorr_df.index
            ],
            aoe_class.timecorr_df["res"],
            yerr=aoe_class.timecorr_df["res_err"],
            linestyle=" ",
            marker="x",
        )
    ax.set_xlabel("time")
    ax.set_ylabel("A/E res")
    myfmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(myfmt)
    plt.close()
    return fig


def drifttime_corr_plot(
    aoe_class,
    data,
    aoe_param="AoE_Timecorr",
    aoe_param_corr="AoE_DTcorr",
    figsize=(12, 8),
    fontsize=12,
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()

    try:
        dep_events = data.query(
            f"{aoe_class.fit_selection}&{aoe_class.cal_energy_param}>1582&{aoe_class.cal_energy_param}<1602&{aoe_class.cal_energy_param}=={aoe_class.cal_energy_param}&{aoe_param}=={aoe_param}"
        )
        final_df = dep_events.query(aoe_class.dt_res_dict["final_selection"])

        plt.subplot(2, 2, 1)
        aoe_pars = aoe_class.dt_res_dict["aoe_fit1"]["pars"]

        xs = np.linspace(aoe_pars["x_lo"], aoe_pars["x_hi"], 100)
        _counts, aoe_bins, _bars = plt.hist(
            final_df.query(
                f"{aoe_class.dt_res_dict['aoe_grp1']}&({aoe_param}<{aoe_pars['x_hi']})&({aoe_param}>{aoe_pars['x_lo']})"
            )[aoe_param],
            bins=400,
            histtype="step",
            label="data",
        )
        dx = np.diff(aoe_bins)
        aoe_class.pdf.components = False
        plt.plot(
            xs, aoe_class.pdf.get_pdf(xs, *aoe_pars.values()) * dx[0], label="full fit"
        )
        aoe_class.pdf.components = True
        sig, bkg = aoe_class.pdf.get_pdf(xs, *aoe_pars.values())
        aoe_class.pdf.components = False
        plt.plot(xs, sig * dx[0], label="peak fit")
        plt.plot(xs, bkg * dx[0], label="bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("counts")

        aoe_pars2 = aoe_class.dt_res_dict["aoe_fit2"]["pars"]
        plt.subplot(2, 2, 2)
        xs = np.linspace(aoe_pars2["x_lo"], aoe_pars2["x_hi"], 100)
        _counts, aoe_bins2, _bars = plt.hist(
            final_df.query(
                f"{aoe_class.dt_res_dict['aoe_grp2']}&({aoe_param}<{aoe_pars2['x_hi']})&({aoe_param}>{aoe_pars2['x_lo']})"
            )[aoe_param],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins2)
        plt.plot(
            xs, aoe_class.pdf.get_pdf(xs, *aoe_pars2.values()) * dx[0], label="full fit"
        )
        aoe_class.pdf.components = True
        sig, bkg = aoe_class.pdf.get_pdf(xs, *aoe_pars2.values())
        aoe_class.pdf.components = False
        plt.plot(xs, sig * dx[0], label="peak fit")
        plt.plot(xs, bkg * dx[0], label="bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("counts")

        hist, bins, _var = pgh.get_hist(
            final_df[aoe_class.dt_param],
            dx=32,
            range=(
                np.nanmin(final_df[aoe_class.dt_param]),
                np.nanmax(final_df[aoe_class.dt_param]),
            ),
        )

        plt.subplot(2, 2, 3)
        plt.step(pgh.get_bin_centers(bins), hist, label="data")

        mus = aoe_class.dt_res_dict["dt_fit"]["mus"]
        sigmas = aoe_class.dt_res_dict["dt_fit"]["sigmas"]
        amps = aoe_class.dt_res_dict["dt_fit"]["amps"]

        for mu, sigma, amp in zip(mus, sigmas, amps, strict=False):
            plt.plot(
                pgh.get_bin_centers(bins),
                nb_gauss_amp(pgh.get_bin_centers(bins), mu, sigma, amp),
            )
        plt.xlabel("drift time (ns)")
        plt.ylabel("Counts")

        plt.subplot(2, 2, 4)
        bins = np.linspace(
            np.nanpercentile(final_df[aoe_param], 1),
            np.nanpercentile(final_df[aoe_param_corr], 99),
            200,
        )
        plt.hist(final_df[aoe_param], bins=bins, histtype="step", label="uncorrected")
        plt.hist(
            final_df[aoe_param_corr], bins=bins, histtype="step", label="corrected"
        )
        plt.xlabel("A/E")
        plt.ylabel("counts")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlim(bins[0], bins[-1])
    except Exception:
        pass
    plt.close()
    return fig


def plot_compt_bands_overlayed(
    aoe_class,
    data,
    eranges: list[tuple],
    aoe_param="AoE_Timecorr",
    aoe_range: list[float] | None = None,
    title="Compton Bands",
    density=True,
    n_bins=50,
    figsize=(12, 8),
    fontsize=12,
) -> None:
    """
    Function to plot various compton bands to check energy dependence and corrections
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()

    for erange in eranges:
        try:
            select_df = data.query(
                f"{aoe_class.selection_string}&{aoe_class.cal_energy_param}>{erange[0]}&{aoe_class.cal_energy_param}<{erange[1]}&{aoe_param}=={aoe_param}"
            )
            if aoe_range is not None:
                select_df = select_df.query(
                    f"{aoe_param}>{aoe_range[0]}&{aoe_param}<{aoe_range[1]}"
                )
                bins = np.linspace(aoe_range[0], aoe_range[1], n_bins)
            else:
                bins = np.linspace(0.85, 1.05, n_bins)
            plt.hist(
                select_df[aoe_param],
                bins=bins,
                histtype="step",
                label=f"{erange[0]}-{erange[1]}",
                density=density,
            )
        except Exception:
            pass
    plt.ylabel("counts")
    plt.xlabel(aoe_param)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.close()
    return fig


def plot_dt_dep(
    aoe_class,
    data,
    eranges: list[tuple],
    titles: list | None = None,
    aoe_param="AoE_Timecorr",
    bins=(200, 100),
    dt_max=2000,
    figsize=(12, 8),
    fontsize=12,
) -> None:
    """
    Function to produce 2d histograms of A/E against drift time to check dependencies
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    for i, erange in enumerate(eranges):
        try:
            plt.subplot(3, 2, i + 1)
            select_df = data.query(
                f"{aoe_class.selection_string}&{aoe_class.cal_energy_param}<{erange[1]}&{aoe_class.cal_energy_param}>{erange[0]}&{aoe_param}=={aoe_param}"
            )

            hist, bs, _var = pgh.get_hist(select_df[aoe_param], bins=500)
            bin_cs = (bs[1:] + bs[:-1]) / 2
            mu = bin_cs[np.argmax(hist)]
            aoe_range = [mu * 0.9, mu * 1.1]

            final_df = select_df.query(
                f"{aoe_param}<{aoe_range[1]}&{aoe_param}>{aoe_range[0]}&{aoe_class.dt_param}<{dt_max}"
            )
            plt.hist2d(
                final_df[aoe_param],
                final_df[aoe_class.dt_param],
                bins=bins,
                norm=LogNorm(),
            )
            plt.ylabel("drift time (ns)")
            plt.xlabel("A/E")
            if titles is None:
                plt.title(f"{erange[0]}-{erange[1]}")
            else:
                plt.title(titles[i])
        except Exception:
            pass
    plt.tight_layout()
    plt.close()
    return fig


def plot_mean_fit(aoe_class, _data, figsize=(12, 8), fontsize=12) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    try:
        ax1.errorbar(
            aoe_class.energy_corr_fits.index,
            aoe_class.energy_corr_fits["mean"],
            yerr=aoe_class.energy_corr_fits["mean_err"],
            xerr=aoe_class.compt_bands_width / 2,
            label="data",
            linestyle=" ",
        )

        ax1.plot(
            aoe_class.energy_corr_fits.index.to_numpy(),
            aoe_class.mean_func.func(
                aoe_class.energy_corr_fits.index.to_numpy(),
                **aoe_class.energy_corr_res_dict["mean_fits"]["pars"],
            ),
            label="linear model",
        )
        ax1.errorbar(
            1592,
            aoe_class.energy_corr_res_dict["dep_fit"]["pars"]["mu"],
            yerr=aoe_class.energy_corr_res_dict["dep_fit"]["errs"]["mu"],
            label="DEP",
            color="green",
            linestyle=" ",
        )

        ax1.legend(title="A/E mu energy dependence", frameon=False)

        ax1.set_ylabel("raw A/E (a.u.)", ha="right", y=1)
        ax2.scatter(
            aoe_class.energy_corr_fits.index,
            100
            * (
                aoe_class.energy_corr_fits["mean"]
                - aoe_class.mean_func.func(
                    aoe_class.energy_corr_fits.index,
                    **aoe_class.energy_corr_res_dict["mean_fits"]["pars"],
                )
            )
            / aoe_class.mean_func.func(
                aoe_class.energy_corr_fits.index,
                **aoe_class.energy_corr_res_dict["mean_fits"]["pars"],
            ),
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100
            * (
                aoe_class.energy_corr_res_dict["dep_fit"]["pars"]["mu"]
                - aoe_class.mean_func.func(
                    1592, **aoe_class.energy_corr_res_dict["mean_fits"]["pars"]
                )
            )
            / aoe_class.mean_func.func(
                1592, **aoe_class.energy_corr_res_dict["mean_fits"]["pars"]
            ),
            lw=1,
            c="g",
        )
    except Exception:
        pass
    ax2.set_ylabel("residuals %", ha="right", y=1)
    ax2.set_xlabel("energy (keV)", ha="right", x=1)
    plt.tight_layout()
    plt.close()
    return fig


def plot_sigma_fit(aoe_class, _data, figsize=(12, 8), fontsize=12) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    try:
        ax1.errorbar(
            aoe_class.energy_corr_fits.index,
            aoe_class.energy_corr_fits["sigma"],
            yerr=aoe_class.energy_corr_fits["sigma_err"],
            xerr=aoe_class.compt_bands_width / 2,
            label="data",
            linestyle=" ",
        )
        sig_pars = aoe_class.energy_corr_res_dict["SigmaFits"]["pars"]
        if aoe_class.sigma_func == SigmaFit:
            label = f"sqrt model: \nsqrt({sig_pars['a']:1.4f}+({sig_pars['b']:1.1f}/E)^{sig_pars['c']:1.1f})"
        else:
            msg = "unknown sigma function"
            raise ValueError(msg)
        ax1.plot(
            aoe_class.energy_corr_fits.index.to_numpy(),
            aoe_class.sigma_func.func(
                aoe_class.energy_corr_fits.index.to_numpy(), **sig_pars
            ),
            label=label,
        )
        ax1.errorbar(
            1592,
            aoe_class.energy_corr_res_dict["dep_fit"]["pars"]["sigma"],
            yerr=aoe_class.energy_corr_res_dict["dep_fit"]["errs"]["sigma"],
            label="DEP",
            color="green",
            linestyle=" ",
        )
        ax1.set_ylabel("A/E stdev (a.u.)", ha="right", y=1)
        ax1.legend(title="A/E stdev energy dependence", frameon=False)
        ax2.scatter(
            aoe_class.energy_corr_fits.index,
            100
            * (
                aoe_class.energy_corr_fits["sigma"]
                - aoe_class.sigma_func.func(
                    aoe_class.energy_corr_fits.index, **sig_pars
                )
            )
            / aoe_class.sigma_func.func(aoe_class.energy_corr_fits.index, **sig_pars),
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100
            * (
                aoe_class.energy_corr_res_dict["dep_fit"]["pars"]["sigma"]
                - aoe_class.sigma_func.func(1592, **sig_pars)
            )
            / aoe_class.sigma_func.func(1592, **sig_pars),
            lw=1,
            c="g",
        )
    except Exception:
        pass
    ax2.set_ylabel("residuals", ha="right", y=1)
    ax2.set_xlabel("energy (keV)", ha="right", x=1)
    plt.tight_layout()
    plt.close()
    return fig


def plot_cut_fit(
    aoe_class, _data, dep_acc=0.9, figsize=(12, 8), fontsize=12
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig = plt.figure()
    try:
        plt.errorbar(
            aoe_class.cut_fits.index,
            aoe_class.cut_fits["sf"],
            yerr=aoe_class.cut_fits["sf_err"],
            linestyle=" ",
        )

        plt.plot(
            aoe_class.cut_fits.index.to_numpy(),
            SigmoidFit.func(
                aoe_class.cut_fits.index.to_numpy(), **aoe_class.cut_fit["pars"]
            ),
        )
        plt.hlines(
            (100 * dep_acc),
            -8.1,
            aoe_class.low_cut_val,
            color="red",
            linestyle="--",
        )
        plt.vlines(
            aoe_class.low_cut_val,
            np.nanmin(aoe_class.cut_fits["sf"]) * 0.9,
            (100 * dep_acc),
            color="red",
            linestyle="--",
        )
        plt.xlim([-8.1, 0.1])
        vals, _labels = plt.yticks()
        plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
        plt.ylim([np.nanmin(aoe_class.cut_fits["sf"]) * 0.9, 102])
    except Exception:
        pass
    plt.xlabel("cut value")
    plt.ylabel("survival percentage")
    plt.close()
    return fig


def get_peak_label(peak: float) -> str:
    if peak == 2039:
        return "CC @"
    if peak == 1592.5:
        return "Tl DEP @"
    if peak == 1620.5:
        return "Bi FEP @"
    if peak == 2103.53:
        return "Tl SEP @"
    if peak == 2614.5:
        return "Tl FEP @"
    return ""


def plot_survival_fraction_curves(
    aoe_class, _data, figsize=(12, 8), fontsize=12
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    try:
        plt.vlines(
            aoe_class.low_cut_val,
            0,
            100,
            label=f"cut value: {aoe_class.low_cut_val:1.2f}",
            color="black",
        )

        for peak, survival_df in aoe_class.low_side_peak_dfs.items():
            with contextlib.suppress(Exception):
                plt.errorbar(
                    survival_df.index,
                    survival_df["sf"],
                    yerr=survival_df["sf_err"],
                    label=f"{get_peak_label(peak)} {peak} keV: {aoe_class.low_side_sfs.loc[peak]['sf']:2.1f} +/- {aoe_class.low_side_sfs.loc[peak]['sf_err']:2.1f} %",
                )
    except Exception:
        pass
    vals, _labels = plt.yticks()
    plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
    plt.legend(loc="upper right")
    plt.xlabel("cut value")
    plt.ylabel("survival percentage")
    plt.ylim([0, 105])
    plt.close()
    return fig


def plot_spectra(
    aoe_class,
    data,
    xrange=(900, 3000),
    n_bins=2101,
    xrange_inset=(1580, 1640),
    n_bins_inset=200,
    figsize=(12, 8),
    fontsize=12,
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots()
    try:
        bins = np.linspace(xrange[0], xrange[1], n_bins)
        ax.hist(
            data.query(aoe_class.selection_string)[aoe_class.cal_energy_param],
            bins=bins,
            histtype="step",
            label="before PSD",
        )
        ax.hist(
            data.query(f"{aoe_class.selection_string}&AoE_Low_Cut")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
            label="low side PSD cut",
        )
        ax.hist(
            data.query(f"{aoe_class.selection_string}&AoE_Double_Sided_Cut")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
            label="double sided PSD cut",
        )
        ax.hist(
            data.query(f"{aoe_class.selection_string} & (~AoE_Double_Sided_Cut)")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
            label="rejected by PSD cut",
        )

        axins = ax.inset_axes([0.25, 0.07, 0.4, 0.3])
        bins = np.linspace(xrange_inset[0], xrange_inset[1], n_bins_inset)
        select_df = data.query(
            f"{aoe_class.cal_energy_param}<{xrange_inset[1]}&{aoe_class.cal_energy_param}>{xrange_inset[0]}"
        )
        axins.hist(
            select_df.query(aoe_class.selection_string)[aoe_class.cal_energy_param],
            bins=bins,
            histtype="step",
        )
        axins.hist(
            select_df.query(f"{aoe_class.selection_string}&AoE_Low_Cut")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
        )
        axins.hist(
            select_df.query(f"{aoe_class.selection_string}&AoE_Double_Sided_Cut")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
        )
        axins.hist(
            select_df.query(f"{aoe_class.selection_string} & (~AoE_Double_Sided_Cut)")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
        )
    except Exception:
        pass
    ax.set_xlim(xrange)
    ax.set_yscale("log")
    plt.xlabel("energy (keV)")
    plt.ylabel("counts")
    plt.legend(loc="upper left")
    plt.close()
    return fig


def plot_sf_vs_energy(
    aoe_class, data, xrange=(900, 3000), n_bins=701, figsize=(12, 8), fontsize=12
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    try:
        bins = np.linspace(xrange[0], xrange[1], n_bins)
        counts_pass, bins_pass, _ = pgh.get_hist(
            data.query(f"{aoe_class.selection_string}&AoE_Double_Sided_Cut")[
                aoe_class.cal_energy_param
            ],
            bins=bins,
        )
        counts, bins, _ = pgh.get_hist(
            data.query(aoe_class.selection_string)[aoe_class.cal_energy_param],
            bins=bins,
        )
        survival_fracs = counts_pass / (counts + 10**-99)

        plt.step(pgh.get_bin_centers(bins_pass), 100 * survival_fracs)
    except Exception:
        pass
    plt.ylim([0, 100])
    vals, _labels = plt.yticks()
    plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
    plt.xlabel("energy (keV)")
    plt.ylabel("survival percentage")
    plt.close()
    return fig


def plot_classifier(
    aoe_class,
    data,
    aoe_param="AoE_Classifier",
    xrange=(900, 3000),
    yrange=(-50, 10),
    xn_bins=700,
    yn_bins=500,
    figsize=(12, 8),
    fontsize=12,
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    with contextlib.suppress(Exception):
        plt.hist2d(
            data.query(aoe_class.selection_string)[aoe_class.cal_energy_param],
            data.query(aoe_class.selection_string)[aoe_param],
            bins=[
                np.linspace(xrange[0], xrange[1], xn_bins),
                np.linspace(yrange[0], yrange[1], yn_bins),
            ],
            norm=LogNorm(),
        )
    plt.xlabel("energy (keV)")
    plt.ylabel(aoe_param)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.close()
    return fig
