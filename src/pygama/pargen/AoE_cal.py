"""
This module provides functions for correcting the a/e energy dependence, determining the cut level and calculating survival fractions.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from matplotlib.colors import LogNorm
from scipy.stats import chi2

import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.energy_cal as pgc
from pygama.math.peak_fitting import nb_erfc
from pygama.pargen.utils import return_nans

log = logging.getLogger(__name__)


class PDF:

    """
    Base class for A/E pdfs.
    """

    @staticmethod
    def pdf(x):
        return

    @staticmethod
    def _replace_values(dic, **kwargs):
        for item, value in kwargs.items():
            dic[item] = value
        return dic


class StandardAoE(PDF):
    @staticmethod
    def pdf(
        x: np.array,
        n_sig: float,
        mu: float,
        sigma: float,
        n_bkg: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
        components: bool = False,
    ) -> np.array:
        """
        PDF for A/E consists of a Gaussian signal with Gaussian tail background
        """
        try:
            sig = n_sig * pgf.gauss_norm(x, mu, sigma)
            bkg = n_bkg * pgf.gauss_tail_norm(
                x, mu, sigma, tau_bkg, lower_range, upper_range
            )
        except Exception:
            sig = np.full_like(x, np.nan)
            bkg = np.full_like(x, np.nan)

        if components is False:
            return sig + bkg
        else:
            return sig, bkg

    @staticmethod
    def extended_pdf(
        x: np.array,
        n_sig: float,
        mu: float,
        sigma: float,
        n_bkg: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
        components: bool = False,
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a Gaussian signal with Gaussian tail background
        """
        if components is True:
            sig, bkg = StandardAoE.pdf(
                x,
                n_sig,
                mu,
                sigma,
                n_bkg,
                tau_bkg,
                lower_range,
                upper_range,
                components,
            )
            return n_sig + n_bkg, sig, bkg
        else:
            return n_sig + n_bkg, StandardAoE.pdf(
                x,
                n_sig,
                mu,
                sigma,
                n_bkg,
                tau_bkg,
                lower_range,
                upper_range,
                components,
            )

    @staticmethod
    def guess(hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)
        except Exception:
            pars, cov = pgf.gauss_mode_width_max(
                hist, bins, var, mode_guess=mu, n_bins=20
            )
            _, sigma, _ = pars
        ls_guess = 2 * np.sum(
            hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))]
        )

        guess_dict = {
            "n_sig": ls_guess,
            "mu": mu,
            "sigma": sigma,
            "n_bkg": np.sum(hist) - ls_guess,
            "tau_bkg": 0.1,
            "lower_range": np.nanmin(bins),
            "upper_range": np.nanmax(bins),
            "components": 0,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

        return StandardAoE._replace_values(guess_dict, **kwargs)

    @staticmethod
    def bounds(guess, **kwargs):
        bounds_dict = {
            "n_sig": (0, None),
            "mu": (None, None),
            "sigma": (0, None),
            "n_bkg": (0, None),
            "tau_bkg": (0, None),
            "lower_range": (None, None),
            "upper_range": (None, None),
            "components": (None, None),
        }

        return [
            bound
            for field, bound in StandardAoE._replace_values(
                bounds_dict, **kwargs
            ).items()
        ]

    def fixed(**kwargs):
        fixed_dict = {
            "n_sig": False,
            "mu": False,
            "sigma": False,
            "n_bkg": False,
            "tau_bkg": False,
            "lower_range": True,
            "upper_range": True,
            "components": True,
        }

        return [
            fixed
            for field, fixed in StandardAoE._replace_values(
                fixed_dict, **kwargs
            ).items()
        ]

    @staticmethod
    def width(pars, errs, cov):
        return pars["sigma"], errs["sigma"]

    @staticmethod
    def centroid(pars, errs, cov):
        return pars["mu"], errs["mu"]


class StandardAoEplusHighTail(PDF):
    @staticmethod
    def pdf(
        x: np.array,
        n_sig: float,
        mu: float,
        sigma: float,
        htail: float,
        tau_sig: float,
        n_bkg: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
        components: bool = False,
    ) -> np.array:
        """
        PDF for A/E consists of a Gaussian signal with tail with Gaussian tail background
        """
        try:
            sig = n_sig * (
                (1 - htail) * pgf.gauss_norm(x, mu, sigma)
                + htail
                * pgf.gauss_tail_norm(x, mu, sigma, tau_sig, lower_range, upper_range)
            )
            bkg = n_bkg * pgf.gauss_tail_norm(
                x, mu, sigma, tau_bkg, lower_range, upper_range
            )
        except Exception:
            sig = np.full_like(x, np.nan)
            bkg = np.full_like(x, np.nan)

        if components is False:
            return sig + bkg
        else:
            return sig, bkg

    @staticmethod
    def extended_pdf(
        x: np.array,
        n_sig: float,
        mu: float,
        sigma: float,
        htail: float,
        tau_sig: float,
        n_bkg: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
        components: bool = False,
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a Gaussian signal with Gaussian tail background
        """
        if components is True:
            sig, bkg = StandardAoEplusHighTail.pdf(
                x,
                n_sig,
                mu,
                sigma,
                htail,
                tau_sig,
                n_bkg,
                tau_bkg,
                lower_range,
                upper_range,
                components,
            )
            return n_sig + n_bkg, sig, bkg
        else:
            return n_sig + n_bkg, StandardAoEplusHighTail.pdf(
                x,
                n_sig,
                mu,
                sigma,
                htail,
                tau_sig,
                n_bkg,
                tau_bkg,
                lower_range,
                upper_range,
                components,
            )

    @staticmethod
    def guess(hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)
        except Exception:
            pars, cov = pgf.gauss_mode_width_max(
                hist, bins, var, mode_guess=mu, n_bins=20
            )
            _, sigma, _ = pars
        ls_guess = 2 * np.sum(
            hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))]
        )

        guess_dict = {
            "n_sig": ls_guess,
            "mu": mu,
            "sigma": sigma,
            "htail": 0.1,
            "tau_sig": -0.1,
            "n_bkg": np.sum(hist) - ls_guess,
            "tau_bkg": 0.1,
            "lower_range": np.nanmin(bins),
            "upper_range": np.nanmax(bins),
            "components": 0,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

        return StandardAoEplusHighTail._replace_values(guess_dict, **kwargs)

    @staticmethod
    def bounds(guess, **kwargs):
        bounds_dict = {
            "n_sig": (0, None),
            "mu": (None, None),
            "sigma": (0, None),
            "htail": (0, 1),
            "tau_sig": (None, 0),
            "n_bkg": (0, None),
            "tau_bkg": (0, None),
            "lower_range": (None, None),
            "upper_range": (None, None),
            "components": (None, None),
        }

        return [
            bound
            for field, bound in StandardAoEplusHighTail._replace_values(
                bounds_dict, **kwargs
            ).items()
        ]

    @staticmethod
    def fixed(**kwargs):
        fixed_dict = {
            "n_sig": False,
            "mu": False,
            "sigma": False,
            "htail": False,
            "tau_sig": False,
            "n_bkg": False,
            "tau_bkg": False,
            "lower_range": True,
            "upper_range": True,
            "components": True,
        }

        return [
            fixed
            for field, fixed in StandardAoEplusHighTail._replace_values(
                fixed_dict, **kwargs
            ).items()
        ]

    @staticmethod
    def width(pars, errs, cov):
        fwhm, fwhm_err = pgf.radford_fwhm(
            pars[2], pars[3], np.abs(pars[4]), cov=cov[:7, :7]
        )
        return fwhm / 2.355, fwhm_err / 2.355

    @staticmethod
    def centroid(pars, errs, cov):
        return pars["mu"], errs["mu"]


class StandardAoEbkg(PDF):
    @staticmethod
    def pdf(
        x: np.array,
        n_events: float,
        mu: float,
        sigma: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
    ) -> np.array:
        """
        PDF for A/E consists of a Gaussian signal with tail with Gaussian tail background
        """
        try:
            sig = n_events * pgf.gauss_tail_norm(
                x, mu, sigma, tau_bkg, lower_range, upper_range
            )
        except Exception:
            sig = np.full_like(x, np.nan)

        return sig

    @staticmethod
    def extended_pdf(
        x: np.array,
        n_events: float,
        mu: float,
        sigma: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a Gaussian signal with Gaussian tail background
        """
        return n_events, StandardAoEbkg.pdf(
            x, n_events, mu, sigma, tau_bkg, lower_range, upper_range
        )

    @staticmethod
    def guess(hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)
        except Exception:
            pars, cov = pgf.gauss_mode_width_max(
                hist, bins, var, mode_guess=mu, n_bins=20
            )
            _, sigma, _ = pars
        ls_guess = 2 * np.sum(
            hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))]
        )

        guess_dict = {
            "n_events": np.sum(hist) - ls_guess,
            "mu": mu,
            "sigma": sigma,
            "tau_bkg": 0.1,
            "lower_range": np.nanmin(bins),
            "upper_range": np.nanmax(bins),
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

        return StandardAoEbkg._replace_values(guess_dict, **kwargs)

    @staticmethod
    def bounds(guess, **kwargs):
        bounds_dict = {
            "n_events": (0, None),
            "mu": (None, None),
            "sigma": (0, None),
            "tau_bkg": (0, None),
            "lower_range": (None, None),
            "upper_range": (None, None),
        }

        return [
            bound
            for field, bound in StandardAoEbkg._replace_values(
                bounds_dict, **kwargs
            ).items()
        ]

    @staticmethod
    def fixed(**kwargs):
        fixed_dict = {
            "n_bkg": False,
            "mu": False,
            "sigma": False,
            "tau_bkg": False,
            "lower_range": True,
            "upper_range": True,
        }

        return [
            fixed
            for field, fixed in StandardAoEbkg._replace_values(
                fixed_dict, **kwargs
            ).items()
        ]


class Gaussian(PDF):
    @staticmethod
    def pdf(x: np.array, n_events: float, mu: float, sigma: float) -> np.array:
        """
        PDF for A/E consists of a Gaussian signal with tail with Gaussian tail background
        """
        try:
            sig = n_events * pgf.gauss_norm(x, mu, sigma)
        except Exception:
            sig = np.full_like(x, np.nan)

        return sig

    @staticmethod
    def extended_pdf(
        x: np.array, n_events: float, mu: float, sigma: float
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a Gaussian signal with Gaussian tail background
        """
        return n_events, Gaussian.pdf(x, n_events, mu, sigma)

    @staticmethod
    def guess(hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)
        except Exception:
            pars, cov = pgf.gauss_mode_width_max(
                hist, bins, var, mode_guess=mu, n_bins=20
            )
            _, sigma, _ = pars
        ls_guess = 2 * np.sum(
            hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))]
        )

        guess_dict = {"n_events": ls_guess, "mu": mu, "sigma": sigma}
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

        return Gaussian._replace_values(guess_dict, **kwargs)

    @staticmethod
    def bounds(gpars, **kwargs):
        bounds_dict = {"n_events": (0, None), "mu": (None, None), "sigma": (0, None)}

        return [
            bound
            for field, bound in Gaussian._replace_values(bounds_dict, **kwargs).items()
        ]

    @staticmethod
    def fixed(**kwargs):
        fixed_dict = {
            "n_events": False,
            "mu": False,
            "sigma": False,
        }

        return [
            fixed
            for field, fixed in Gaussian._replace_values(fixed_dict, **kwargs).items()
        ]


class DriftTimeDistribution(PDF):
    @staticmethod
    def pdf(
        x,
        n_sig1,
        mu1,
        sigma1,
        htail1,
        tau1,
        n_sig2,
        mu2,
        sigma2,
        htail2,
        tau2,
        components,
    ):
        gauss1 = n_sig1 * pgf.gauss_with_tail_pdf(x, mu1, sigma1, htail1, tau1)
        gauss2 = n_sig2 * pgf.gauss_with_tail_pdf(x, mu2, sigma2, tau2, htail2)
        if components is True:
            return gauss1, gauss2
        else:
            return gauss1 + gauss2

    @staticmethod
    def extended_pdf(
        x,
        n_sig1,
        mu1,
        sigma1,
        htail1,
        tau1,
        n_sig2,
        mu2,
        sigma2,
        htail2,
        tau2,
        components,
    ):
        if components is True:
            gauss1, gauss2 = DriftTimeDistribution.pdf(
                x,
                n_sig1,
                mu1,
                sigma1,
                htail1,
                tau1,
                n_sig2,
                mu2,
                sigma2,
                htail2,
                tau2,
                components,
            )
            return n_sig1 + n_sig2, gauss1, gauss2

        else:
            return n_sig1 + n_sig2, DriftTimeDistribution.pdf(
                x,
                n_sig1,
                mu1,
                sigma1,
                htail1,
                tau1,
                n_sig2,
                mu2,
                sigma2,
                htail2,
                tau2,
                components,
            )

    @staticmethod
    def guess(hist: np.array, bins: np.array, var: np.array, **kwargs) -> list:
        """
        Guess for fitting dt spectrum
        """
        bcs = pgh.get_bin_centers(bins)
        mus = pgc.get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 5)
        if len(mus) > 2:
            mus = pgc.get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 8)
        elif len(mus) < 2:
            mus = pgc.get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 3)
        mu1 = bcs[mus[0]]
        mu2 = bcs[mus[-1]]

        pars, cov = pgf.gauss_mode_width_max(
            hist,
            bins,
            var=None,
            mode_guess=mu1,
            n_bins=10,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )
        mu1, sigma1, amp = pars
        ix = np.where(bcs < mu1 + 3 * sigma1)[0][-1]
        n_sig1 = np.sum(hist[:ix])
        pars2, cov2 = pgf.gauss_mode_width_max(
            hist,
            bins,
            var=None,
            mode_guess=mu2,
            n_bins=10,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )
        mu2, sigma2, amp2 = pars2

        guess_dict = {
            "n_sig1": n_sig1,
            "mu1": mu1,
            "sigma1": sigma1,
            "htail1": 0.5,
            "tau1": 0.1,
            "n_sig2": np.sum(hist) - n_sig1,
            "mu2": mu2,
            "sigma2": sigma2,
            "htail2": 0.5,
            "tau2": 0.1,
            "components": 0,
        }
        for key, guess in guess_dict.items():
            if np.isnan(guess):
                guess_dict[key] = 0

        return DriftTimeDistribution._replace_values(guess_dict, **kwargs)

    @staticmethod
    def bounds(guess, **kwargs):
        bounds_dict = {
            "n_sig1": (0, None),
            "mu1": (None, None),
            "sigma1": (0, None),
            "htail1": (0, 1),
            "tau1": (None, None),
            "n_sig2": (0, None),
            "mu2": (None, None),
            "sigma2": (0, None),
            "htail2": (0, 1),
            "tau2": (None, None),
            "components": (None, None),
        }

        return [
            bound
            for field, bound in DriftTimeDistribution._replace_values(
                bounds_dict, **kwargs
            ).items()
        ]

    @staticmethod
    def fixed(**kwargs):
        fixed_dict = {
            "n_sig1": False,
            "mu1": False,
            "sigma1": False,
            "htail1": False,
            "tau1": False,
            "n_sig2": False,
            "mu2": False,
            "sigma2": False,
            "htail2": False,
            "tau2": False,
            "components": True,
        }

        return [
            fixed
            for field, fixed in DriftTimeDistribution._replace_values(
                fixed_dict, **kwargs
            ).items()
        ]


class Pol1:
    @staticmethod
    def func(x, a, b):
        return x * a + b

    @staticmethod
    def string_func(input_param):
        return f"{input_param}*a+b"

    @staticmethod
    def guess(bands, means, mean_errs):
        return [-1e-06, 5e-01]


class SigmaFit:
    @staticmethod
    def func(x, a, b, c):
        return np.sqrt(a + (b / (x + 10**-99)) ** c)

    @staticmethod
    def string_func(input_param):
        return f"(a+(b/({input_param}+10**-99))**c)**(0.5)"

    @staticmethod
    def guess(bands, sigmas, sigma_errs):
        return [np.nanpercentile(sigmas, 50) ** 2, 2, 2]


class SigmoidFit:
    @staticmethod
    def func(x, a, b, c, d):
        return (a + b * x) * nb_erfc(c * x + d)

    @staticmethod
    def guess(xs, ys, y_errs):
        return [np.nanmax(ys) / 2, 0, 1, 1.5]


def unbinned_aoe_fit(
    aoe: np.array,
    pdf=StandardAoE,
    display: int = 0,
) -> tuple(np.array, np.array):
    """
    Fitting function for A/E, first fits just a Gaussian before using the full pdf to fit
    if fails will return NaN values

    Args:
    aoe: np.array
        A/E values
    pdf: PDF
        PDF to fit to
    display: int
        Level of display

    Returns: tuple(np.array, np.array)
        Tuple of fit values and errors
    """
    hist, bins, var = pgh.get_hist(aoe, bins=500)

    gpars = Gaussian.guess(hist, bins, var)
    c1_min = gpars["mu"] - 2 * gpars["sigma"]
    c1_max = gpars["mu"] + 3 * gpars["sigma"]

    # Initial fit just using Gaussian
    c1 = cost.UnbinnedNLL(aoe[(aoe < c1_max) & (aoe > c1_min)], Gaussian.pdf)

    m1 = Minuit(c1, **gpars)
    m1.limits = [
        (0, len(aoe[(aoe < c1_max) & (aoe > c1_min)])),
        (gpars["mu"] * 0.8, gpars["mu"] * 1.2),
        (0.8 * gpars["sigma"], gpars["sigma"] * 1.2),
    ]
    m1.fixed = Gaussian.fixed()
    m1.migrad()

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = m1.values["mu"] - 15 * m1.values["sigma"]
    if fmin < np.nanmin(aoe):
        fmin = np.nanmin(aoe)
    fmax_bkg = m1.values["mu"] - 5 * m1.values["sigma"]
    fmax = m1.values["mu"] + 5 * m1.values["sigma"]

    n_bkg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - m1.values["n_events"]

    bkg_guess = StandardAoEbkg.guess(
        hist,
        bins,
        var,
        n_events=n_bkg_guess,
        mu=m1.values["mu"],
        sigma=m1.values["sigma"],
        lower_range=fmin,
        upper_range=fmax_bkg,
    )

    c2 = cost.ExtendedUnbinnedNLL(
        aoe[(aoe < fmax_bkg) & (aoe > fmin)], StandardAoEbkg.extended_pdf
    )
    m2 = Minuit(c2, **bkg_guess)
    m2.fixed = StandardAoEbkg.fixed(mu=True)
    m2.limits = StandardAoEbkg.bounds(
        bkg_guess, n_events=(0, 2 * len(aoe[(aoe < fmax_bkg) & (aoe > fmin)]))
    )
    m2.simplex().migrad()
    m2.hesse()

    x0 = pdf.guess(
        hist,
        bins,
        var,
        n_sig=m1.values["n_events"],
        mu=m1.values["mu"],
        sigma=m1.values["sigma"],
        n_bkg=m2.values["n_events"],
        tau_bkg=m2.values["tau_bkg"],
        lower_range=fmin,
        upper_range=fmax,
    )

    # Full fit using Gaussian signal with Gaussian tail background
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax) & (aoe > fmin)], pdf.extended_pdf)
    m = Minuit(c, **x0)
    m.limits = pdf.bounds(
        x0,
        n_sig=(0, 2 * len(aoe[(aoe < fmax) & (aoe > fmin)])),
        n_bkg=(0, 2 * len(aoe[(aoe < fmax) & (aoe > fmin)])),
    )
    m.fixed = pdf.fixed()
    m.migrad()
    m.hesse()

    if np.isnan(m.errors).all():
        try:
            m.simplex.migrad()
            m.hesse()
        except Exception:
            return return_nans(pdf)

    if display > 1:
        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, bars = plt.hist(
            aoe[(aoe < fmax) & (aoe > fmin)], bins=200, histtype="step", label="Data"
        )
        dx = np.diff(bins)
        plt.plot(xs, pdf.pdf(xs, *m.values) * dx[0], label="Full fit")
        sig, bkg = pdf.pdf(xs, *m.values[:-1], True)
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(xs, Gaussian.pdf(xs, *m1.values) * dx[0], label="Initial Gaussian")
        plt.plot(xs, StandardAoEbkg.pdf(xs, *m2.values) * dx[0], label="Bkg guess")
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        bin_centers = (bins[1:] + bins[:-1]) / 2
        res = (pdf.pdf(bin_centers, *m.values) * dx[0]) - counts
        plt.plot(
            bin_centers,
            [re / count if count != 0 else re for re, count in zip(res, counts)],
            label="Normalised Residuals",
        )
        plt.legend(loc="upper left")
        plt.show()
        return m.values, m.errors, m.covariance

    else:
        return m.values, m.errors, m.covariance


def fit_time_means(tstamps, means, reses):
    """
    Fit the time dependence of the means of the A/E distribution

    Args:

    tstamps: np.array
        Timestamps of the data
    means: np.array
        Means of the A/E distribution
    reses: np.array
        Resolutions of the A/E distribution

    Returns: dict
        Dictionary of the time dependence of the means
    """
    out_dict = {}
    current_tstamps = []
    current_means = []
    current_reses = []
    rolling_mean = means[
        np.where(
            (np.abs(np.diff(means)) < (0.4 * np.array(reses)[1:]))
            & (~np.isnan(np.abs(np.diff(means)) < (0.4 * np.array(reses)[1:])))
        )[0][0]
    ]
    for i, tstamp in enumerate(tstamps):
        if (
            (
                np.abs(means[i] - rolling_mean) > 0.4 * reses[i]
                and np.abs(means[i] - rolling_mean) > rolling_mean * 0.01
            )
            or np.isnan(means[i])
            or np.isnan(reses[i])
        ):
            if i + 1 == len(means):
                out_dict[tstamp] = np.nan
            else:
                if (np.abs(means[i + 1] - means[i]) < 0.4 * reses[i + 1]) and not (
                    np.isnan(means[i])
                    or np.isnan(means[i + 1])
                    or np.isnan(reses[i])
                    or np.isnan(reses[i + 1])
                ):
                    for ts in current_tstamps:
                        out_dict[ts] = rolling_mean
                    rolling_mean = means[i]
                    current_means = [means[i]]
                    current_tstamps = [tstamp]
                    current_reses = [reses[i]]
                else:
                    out_dict[tstamp] = np.nan
        else:
            current_tstamps.append(tstamp)
            current_means.append(means[i])
            current_reses.append(reses[i])
            rolling_mean = np.average(
                current_means, weights=1 / np.array(current_reses)
            )
    for tstamp in current_tstamps:
        out_dict[tstamp] = rolling_mean
    return out_dict


def energy_guess(energy, func_i, fit_range=None, bin_width=0.5, peak=None, eres=None):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    hist, bins, var = pgh.get_hist(energy, dx=bin_width, range=fit_range)
    if (func_i == pgf.extended_radford_pdf) or (func_i == pgf.radford_pdf):
        bin_cs = (bins[1:] + bins[:-1]) / 2
        i_0 = np.nanargmax(hist)

        if peak is not None:
            mu = peak
        else:
            mu = bin_cs[np.nanmax(hist)]

        if eres is not None:
            sigma = eres / 2.355
        else:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)

        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range]) - (
            (n_bins_range * 2) * (bg0 - step / 2)
        )
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0
        parguess = [
            nsig_guess,
            mu,
            sigma,
            htail,
            tau,
            nbkg_guess,
            hstep,
            fit_range[0],
            fit_range[1],
            0,
        ]
        for i, guess in enumerate(parguess):
            if np.isnan(guess):
                parguess[i] = 0
        return parguess

    elif (func_i == pgf.extended_gauss_step_pdf) or (func_i == pgf.gauss_step_pdf):
        if peak is not None:
            mu = peak
        else:
            mu = bin_cs[np.nanmax(hist)]

        if eres is not None:
            sigma = eres / 2.355
        else:
            _, sigma, _ = pgh.get_Gaussian_guess(hist, bins)

        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((3 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
        if nbkg_guess < 0:
            nbkg_guess = 0
        if nsig_guess < 0:
            nsig_guess = 0

        parguess = [
            nsig_guess,
            mu,
            sigma,
            nbkg_guess,
            hstep,
            fit_range[0],
            fit_range[1],
            0,
        ]
        for i, guess in enumerate(parguess):
            if np.isnan(guess):
                parguess[i] = 0
        return parguess


def fix_all_but_nevents(func):
    """
    Returns: Sequence list of fixed indexes for fitting and mask for parameters
    """

    if (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        # pars are: n_sig, mu, sigma, n_bkg, hstep, lower, upper, components
        return [1, 2, 4, 5, 6, 7], np.array(
            [True, False, False, True, False, False, False, False]
        )

    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):
        # pars are: n_sig, mu, sigma, htail,tau, n_bkg, hstep, components
        return [1, 2, 3, 4, 6, 7, 8, 9], np.array(
            [True, False, False, False, False, True, False, False, False, False]
        )

    else:
        log.error(f"get_hpge_E_fixed not implemented for {func.__name__}")
        return None
    return None


def get_bounds(func, parguess):
    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):
        bounds = [
            (0, 2 * (parguess[0] + parguess[5])),
            (parguess[1] - 1, parguess[1] + 1),
            (0, None),
            (0, 0.5),
            (0, None),
            (0, 2 * (parguess[0] + parguess[5])),
            (-1, 1),
            (None, None),
            (None, None),
            (None, None),
        ]
        return bounds

    elif (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        bounds = [
            (0, 2 * (parguess[0] + parguess[5])),
            (parguess[1] - 1, parguess[1] + 1),
            (0, None),
            (0, 2 * (parguess[0] + parguess[5])),
            (-1, 1),
            (None, None),
            (None, None),
            (None, None),
        ]
        return bounds

    else:
        log.error(f"get_bounds not implemented for {func.__name__}")
        return []


def get_peak_label(peak: float) -> str:
    if peak == 2039:
        return "CC @"
    elif peak == 1592.5:
        return "Tl DEP @"
    elif peak == 1620.5:
        return "Bi FEP @"
    elif peak == 2103.53:
        return "Tl SEP @"
    elif peak == 2614.5:
        return "Tl FEP @"


def update_guess(func, parguess, energies):
    if (
        func == pgf.gauss_step_cdf
        or func == pgf.gauss_step_pdf
        or func == pgf.extended_gauss_step_pdf
    ):
        total_events = len(energies)
        parguess[0] = len(
            energies[
                (energies > parguess[1] - 2 * parguess[2])
                & (energies < parguess[1] + 2 * parguess[2])
            ]
        )
        parguess[3] = total_events - parguess[0]
        return parguess

    if (
        func == pgf.radford_cdf
        or func == pgf.radford_pdf
        or func == pgf.extended_radford_pdf
    ):
        total_events = len(energies)
        parguess[0] = len(
            energies[
                (energies > parguess[1] - 2 * parguess[2])
                & (energies < parguess[1] + 2 * parguess[2])
            ]
        )
        parguess[5] = total_events - parguess[0]
        return parguess

    else:
        log.error(f"update_guess not implemented for {func.__name__}")
        return parguess


def get_survival_fraction(
    energy,
    cut_param,
    cut_val,
    peak,
    eres_pars,
    high_cut=None,
    guess_pars_cut=None,
    guess_pars_surv=None,
    dt_mask=None,
    mode="greater",
    func=pgf.extended_radford_pdf,
    gof_func=pgf.radford_pdf,
    display=0,
):
    if dt_mask is None:
        dt_mask = np.full(len(cut_param), True, dtype=bool)

    nan_idxs = np.isnan(cut_param)
    if high_cut is not None:
        idxs = (cut_param > cut_val) & (cut_param < high_cut) & dt_mask
    else:
        if mode == "greater":
            idxs = (cut_param > cut_val) & dt_mask
        elif mode == "less":
            idxs = (cut_param < cut_val) & dt_mask
        else:
            raise ValueError("mode not recognised")

    if guess_pars_cut is None or guess_pars_surv is None:
        (pars, errs, cov, _, func, gof_func, _, _, _) = pgc.unbinned_staged_energy_fit(
            energy,
            func,
            gof_func,
            guess_func=energy_guess,
            bounds_func=get_bounds,
            guess_kwargs={"peak": peak, "eres": eres_pars},
        )

        guess_pars_cut = pars
        guess_pars_surv = pars
    # add update guess here for n_sig and n_bkg
    guess_pars_cut = update_guess(func, guess_pars_cut, energy[(~nan_idxs) & (~idxs)])
    (cut_pars, cut_errs, cut_cov, _, _, _, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy[(~nan_idxs) & (~idxs)],
        func,
        gof_func,
        guess=guess_pars_cut,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        fixed_func=fix_all_but_nevents,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        lock_guess=True,
        allow_tail_drop=False,
    )
    guess_pars_surv = update_guess(func, guess_pars_cut, energy[(~nan_idxs) & (idxs)])
    (surv_pars, surv_errs, surv_cov, _, _, _, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy[(~nan_idxs) & (idxs)],
        func,
        gof_func,
        guess=guess_pars_surv,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        fixed_func=fix_all_but_nevents,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        lock_guess=True,
        allow_tail_drop=False,
    )

    ct_n = cut_pars["n_sig"]
    ct_err = cut_errs["n_sig"]
    surv_n = surv_pars["n_sig"]
    surv_err = surv_errs["n_sig"]

    pc_n = ct_n + surv_n
    pc_err = np.sqrt(surv_err**2 + ct_err**2)

    sf = (surv_n / pc_n) * 100
    err = sf * np.sqrt((pc_err / pc_n) ** 2 + (surv_err / surv_n) ** 2)
    return sf, err, cut_pars, surv_pars


def get_sf_sweep(
    energy: np.array,
    cut_param: np.array,
    final_cut_value: float = None,
    peak: float = 1592.5,
    eres_pars: list = None,
    dt_mask=None,
    cut_range=(-5, 5),
    n_samples=26,
    mode="greater",
) -> tuple(pd.DataFrame, float, float):
    """
    Calculates survival fraction for gamma lines using fitting method as in cut determination
    """

    if dt_mask is None:
        dt_mask = np.full(len(cut_param), True, dtype=bool)

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame(columns=["cut_val", "sf", "sf_err"])

    (pars, _, _, _, func, gof_func, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy,
        pgf.extended_radford_pdf,
        pgf.radford_pdf,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        guess_kwargs={"peak": peak, "eres": eres_pars},
    )
    guess_pars_cut = pars
    guess_pars_surv = pars

    for cut_val in cut_vals:
        try:
            sf, err, _, _ = get_survival_fraction(
                energy,
                cut_param,
                cut_val,
                peak,
                eres_pars,
                dt_mask=dt_mask,
                mode=mode,
                guess_pars_cut=guess_pars_cut,
                guess_pars_surv=guess_pars_surv,
                func=func,
                gof_func=gof_func,
            )
            out_df = pd.concat(
                [out_df, pd.DataFrame([{"cut_val": cut_val, "sf": sf, "sf_err": err}])]
            )
        except Exception:
            pass
    out_df.set_index("cut_val", inplace=True)
    if final_cut_value is not None:
        sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
            energy,
            cut_param,
            final_cut_value,
            peak,
            eres_pars,
            dt_mask=dt_mask,
            mode=mode,
            guess_pars_cut=guess_pars_cut,
            guess_pars_surv=guess_pars_surv,
            func=func,
            gof_func=gof_func,
        )
    else:
        sf = None
        sf_err = None
    return (
        out_df,
        sf,
        sf_err,
    )


def compton_sf(cut_param, low_cut_val, high_cut_val=None, mode="greater", dt_mask=None):
    if dt_mask is None:
        dt_mask = np.full(len(cut_param), True, dtype=bool)

    if high_cut_val is not None:
        mask = (cut_param > low_cut_val) & (cut_param < high_cut_val) & dt_mask
    else:
        if mode == "greater":
            mask = (cut_param > low_cut_val) & dt_mask
        elif mode == "less":
            mask = (cut_param < low_cut_val) & dt_mask
        else:
            raise ValueError("mode not recognised")

    sf = 100 * len(cut_param[mask]) / len(cut_param)
    sf_err = sf * np.sqrt((1 / len(cut_param)) + 1 / (len(cut_param[mask]) + 10**-99))
    return {
        "low_cut": low_cut_val,
        "sf": sf,
        "sf_err": sf_err,
        "high_cut": high_cut_val,
    }


def compton_sf_sweep(
    energy: np.array,
    cut_param: np.array,
    final_cut_value: float,
    peak: float,
    eres: list[float, float] = None,
    dt_mask: np.array = None,
    cut_range=(-5, 5),
    n_samples=51,
    mode="greater",
) -> tuple(float, np.array, list):
    """
    Determines survival fraction for compton continuum by basic counting
    """

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame(columns=["cut_val", "sf", "sf_err"])

    for cut_val in cut_vals:
        ct_dict = compton_sf(cut_param, cut_val, mode=mode, dt_mask=dt_mask)
        df = pd.DataFrame(
            [
                {
                    "cut_val": ct_dict["low_cut"],
                    "sf": ct_dict["sf"],
                    "sf_err": ct_dict["sf_err"],
                }
            ]
        )
        out_df = pd.concat([out_df, df])
    out_df.set_index("cut_val", inplace=True)

    sf_dict = compton_sf(cut_param, final_cut_value, mode=mode, dt_mask=dt_mask)

    return out_df, sf_dict["sf"], sf_dict["sf_err"]


class CalAoE:
    def __init__(
        self,
        cal_dicts: dict = None,
        cal_energy_param: str = "cuspEmax_ctc_cal",
        eres_func: callable = lambda x: 1,
        pdf=StandardAoE,
        selection_string: str = "",
        dt_corr: bool = False,
        dep_acc: float = 0.9,
        dep_correct: bool = False,
        dt_cut: dict = None,
        dt_param: str = "dt_eff",
        high_cut_val: int = 3,
        mean_func: Callable = Pol1,
        sigma_func: Callable = SigmaFit,
        compt_bands_width: int = 20,
        plot_options: dict = None,
    ):
        self.cal_dicts = cal_dicts if cal_dicts is not None else {}
        self.cal_energy_param = cal_energy_param
        self.eres_func = eres_func
        self.pdf = pdf
        self.selection_string = selection_string
        self.dt_corr = dt_corr
        self.dt_param = "dt_eff"
        self.dep_correct = dep_correct
        self.dt_cut = dt_cut
        self.dep_acc = dep_acc
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
        self.plot_options = plot_options if plot_options is not None else {}

    def update_cal_dicts(self, update_dict):
        if re.match(r"(\d{8})T(\d{6})Z", list(self.cal_dicts)[0]):
            for tstamp in self.cal_dicts:
                if tstamp in update_dict:
                    self.cal_dicts[tstamp].update(update_dict[tstamp])
                else:
                    self.cal_dicts[tstamp].update(update_dict)
        else:
            self.cal_dicts.update(update_dict)

    def time_correction(self, df, aoe_param, output_name="AoE_Timecorr", display=0):
        log.info("Starting A/E time correction")
        self.timecorr_df = pd.DataFrame(
            columns=["run_timestamp", "mean", "mean_err", "res", "res_err"]
        )
        try:
            if "run_timestamp" in df:
                for tstamp, time_df in df.groupby("run_timestamp", sort=True):
                    try:
                        pars, errs, cov = unbinned_aoe_fit(
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
                    except Exception:
                        self.timecorr_df = pd.concat(
                            [
                                self.timecorr_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "run_timestamp": tstamp,
                                            "mean": np.nan,
                                            "mean_err": np.nan,
                                            "res": np.nan,
                                            "res_err": np.nan,
                                        }
                                    ]
                                ),
                            ]
                        )
                self.timecorr_df.set_index("run_timestamp", inplace=True)
                time_dict = fit_time_means(
                    np.array(self.timecorr_df.index),
                    np.array(self.timecorr_df["mean"]),
                    np.array(self.timecorr_df["res"]),
                )

                df[output_name] = df[aoe_param] / np.array(
                    [time_dict[tstamp] for tstamp in df["run_timestamp"]]
                )
                self.update_cal_dicts(
                    {
                        tstamp: {
                            output_name: {
                                "expression": f"{aoe_param}/a",
                                "parameters": {"a": t_dict},
                            }
                        }
                        for tstamp, t_dict in time_dict.items()
                    }
                )
                log.info("A/E time correction finished")
            else:
                try:
                    pars, errs, cov = unbinned_aoe_fit(
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
                                        "mean": pars["mu"],
                                        "mean_err": errs["mu"],
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
                except Exception:
                    self.timecorr_df = pd.concat(
                        [
                            self.timecorr_df,
                            pd.DataFrame(
                                [
                                    {
                                        "mean": np.nan,
                                        "mean_err": np.nan,
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
                log.info("A/E time correction finished")
        except Exception:
            log.error("A/E time correction failed")
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
        Calculates the correction needed to align the two drift time regions for ICPC detectors
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

            self.dt_res_dict[
                "final_selection"
            ] = f"{aoe_param}>{aoe_range[0]}&{aoe_param}<{aoe_range[1]}&{self.dt_param}>{dt_range[0]}&{self.dt_param}<{dt_range[1]}&{self.dt_param}=={self.dt_param}"

            final_df = dep_events.query(self.dt_res_dict["final_selection"])

            hist, bins, var = pgh.get_hist(
                final_df[self.dt_param],
                dx=10,
                range=(
                    np.nanmin(final_df[self.dt_param]),
                    np.nanmax(final_df[self.dt_param]),
                ),
            )

            gpars = self.dt_res_dict["dt_guess"] = DriftTimeDistribution.guess(
                hist, bins, var
            )
            cost_func = cost.ExtendedUnbinnedNLL(
                final_df[self.dt_param], DriftTimeDistribution.extended_pdf
            )
            m = Minuit(cost_func, **gpars)
            m.limits = DriftTimeDistribution.bounds(gpars)
            m.fixed = DriftTimeDistribution.fixed()
            m.simplex().migrad()
            m.hesse()

            self.dt_res_dict["dt_fit"] = {
                "pars": m.values,
                "errs": m.errors,
                "object": m,
            }
            aoe_grp1 = self.dt_res_dict[
                "aoe_grp1"
            ] = f'{self.dt_param}>{m.values["mu1"] - 2 * m.values["sigma1"]} & {self.dt_param}<{m.values["mu1"] + 2 * m.values["sigma1"]}'
            aoe_grp2 = self.dt_res_dict[
                "aoe_grp2"
            ] = f'{self.dt_param}>{m.values["mu2"] - 2 * m.values["sigma2"]} & {self.dt_param}<{m.values["mu2"] + 2 * m.values["sigma2"]}'

            aoe_pars, aoe_errs, _ = unbinned_aoe_fit(
                final_df.query(aoe_grp1)[aoe_param], pdf=self.pdf, display=display
            )

            self.dt_res_dict["aoe_fit1"] = {"pars": aoe_pars, "errs": aoe_errs}

            aoe_pars2, aoe_errs2, _ = unbinned_aoe_fit(
                final_df.query(aoe_grp2)[aoe_param], pdf=self.pdf, display=display
            )

            self.dt_res_dict["aoe_fit2"] = {"pars": aoe_pars2, "errs": aoe_errs2}

            try:
                self.alpha = (aoe_pars["mu"] - aoe_pars2["mu"]) / (
                    (m.values["mu2"] * aoe_pars2["mu"])
                    - (m.values["mu1"] * aoe_pars["mu"])
                )
            except ZeroDivisionError:
                self.alpha = 0
            self.dt_res_dict["alpha"] = self.alpha
            log.info(f"dtcorr successful alpha:{self.alpha}")
            data[out_param] = data[aoe_param] * (1 + self.alpha * data[self.dt_param])
        except Exception:
            log.error("Drift time correction failed")
            self.alpha = np.nan

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
        corrected_param="AoE_Corrected",
        classifier_param="AoE_Classifier",
        display: int = 0,
    ):
        """
        Calculates the corrections needed for the energy dependence of the A/E.
        Does this by fitting the compton continuum in slices and then applies fits to the centroid and variance.
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
                if (peak - 5) > band and (peak - 5) < (band + self.compt_bands_width):
                    allow = False
                elif (peak + 5 > band) and (peak + 5) < (band + self.compt_bands_width):
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
                    pars, errs, cov = unbinned_aoe_fit(
                        select_df.query(
                            f"{self.cal_energy_param}>{band}&{self.cal_energy_param}< {self.compt_bands_width+band}"
                        )[aoe_param],
                        pdf=self.pdf,
                        display=display,
                    )

                    mean, mean_err = self.pdf.centroid(pars, errs, cov)
                    sigma, sigma_err = self.pdf.width(pars, errs, cov)

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

                except Exception:
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
            self.energy_corr_fits.set_index("compt_bands", inplace=True)
            valid_fits = self.energy_corr_fits.query(
                "mean_err==mean_err&sigma_err==sigma_err & sigma_err!=0 & mean_err!=0"
            )
            self.energy_corr_res_dict["n_of_valid_fits"] = len(valid_fits)
            log.info(f"{len(valid_fits)} compton bands fit successfully")
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

            mu_pars = m_mu.values
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
            dof_mu = len(valid_fits["mean"]) - len(pars)
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

            sig_pars = m_sig.values
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
                dep_pars, dep_err, _ = unbinned_aoe_fit(
                    select_df.query(
                        f"{self.cal_energy_param}>{emin}&{self.cal_energy_param}<{emax}"
                    )[aoe_param],
                    pdf=self.pdf,
                    display=display,
                )
            except Exception:
                dep_pars, dep_err, _ = return_nans(self.pdf)

            data[corrected_param] = data[aoe_param] / self.mean_func.func(
                data[self.cal_energy_param], *mu_pars
            )
            data[classifier_param] = (data[corrected_param] - 1) / self.sigma_func.func(
                data[self.cal_energy_param], *sig_pars
            )
            log.info("Finished A/E energy successful")
            log.info(f"mean pars are {mu_pars.to_dict()}")
            log.info(f"sigma pars are {sig_pars.to_dict()}")
        except Exception:
            log.error("A/E energy correction failed")
            mu_pars, mu_errs, mu_cov = return_nans(self.mean_func.func)
            csqr_mu, dof_mu, p_val_mu = (np.nan, np.nan, np.nan)
            csqr_sig, dof_sig, p_val_sig = (np.nan, np.nan, np.nan)
            sig_pars, sig_errs, sig_cov = return_nans(self.sigma_func.func)
            dep_pars, dep_err, dep_cov = return_nans(self.pdf)

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
            "func": self.pdf.__name__,
            "module": self.pdf.__module__,
            "pars": dep_pars.to_dict(),
            "errs": dep_err.to_dict(),
        }

        self.update_cal_dicts(
            {
                corrected_param: {
                    "expression": f"{aoe_param}/({self.mean_func.string_func(self.cal_energy_param)})",
                    "parameters": mu_pars.to_dict(),
                },
                classifier_param: {
                    "expression": f"({corrected_param}-1)/({self.sigma_func.string_func(self.cal_energy_param)})",
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
        dep_acc: float,
        output_cut_param: str = "AoE_Low_Cut",
        display: int = 1,
    ):
        """
        Determines A/E cut by sweeping through values and for each one fitting the DEP to determine how many events survive.
        Then interpolates to get cut value at desired DEP survival fraction (typically 90%)
        """

        log.info("Starting A/E low cut determination")
        self.low_cut_res_dict = {}
        self.cut_fits = pd.DataFrame(columns=["cut_val", "sf", "sf_err"])

        min_range, max_range = ranges

        try:
            select_df = data.query(
                f"{self.fit_selection}&({self.cal_energy_param} > {peak - min_range}) & ({self.cal_energy_param} < {peak + max_range})"
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
                dt_mask=None,
                cut_range=(-8, 0),
                n_samples=40,
                mode="greater",
            )

            valid_fits = self.cut_fits.query(
                f'sf_err<{(1.5 * np.nanpercentile(self.cut_fits["sf_err"],85))}&sf_err==sf_err'
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
            p = SigmoidFit.func(xs, *m1.values)
            self.cut_fit = {
                "function": SigmoidFit.__name__,
                "pars": m1.values.to_dict(),
                "errs": m1.errors.to_dict(),
            }
            self.low_cut_val = round(xs[np.argmin(np.abs(p - (100 * self.dep_acc)))], 3)
            log.info(f"Cut found at {self.low_cut_val}")

            data[output_cut_param] = data[aoe_param] > self.low_cut_val
            if self.dt_cut_param is not None:
                data[output_cut_param] = data[output_cut_param] & (
                    data[self.dt_cut_param]
                )
        except Exception:
            log.error("A/E cut determination failed")
            self.low_cut_val = np.nan
        if self.dt_cut_param is not None and self.dt_cut_hard is True:
            self.update_cal_dicts(
                {
                    output_cut_param: {
                        "expression": f"({aoe_param}>a) & ({self.dt_cut_param})",
                        "parameters": {"a": self.low_cut_val},
                    }
                }
            )
        else:
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
        data,
        aoe_param,
        peaks,
        fit_widths,
        nsamples=26,
        cut_range=(-5, 5),
        mode="greater",
    ):
        sfs = pd.DataFrame(columns=["peak", "sf", "sf_err"])
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
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )

                    cut_df, sf, sf_err = compton_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        cut_range=cut_range,
                        nsamples=nsamples,
                        mode=mode,
                        dt_mask=peak_df[self.dt_cut_param].to_numpy()
                        if self.dt_cut_param is not None
                        else None,
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
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )
                    cut_df, sf, sf_err = get_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        cut_range=cut_range,
                        nsamples=nsamples,
                        mode=mode,
                        dt_mask=peak_df[self.dt_cut_param].to_numpy()
                        if self.dt_cut_param is not None
                        else None,
                    )

                    cut_df = cut_df.query(
                        f'sf_err<5*{np.nanpercentile(cut_df["sf_err"], 50)}& sf_err==sf_err & sf<=100'
                    )

                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    peak_dfs[peak] = cut_df
                log.info(f"{peak}keV: {sf:2.1f} +/- {sf_err:2.1f} %")
            except Exception:
                sfs = pd.concat(
                    [
                        sfs,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(
                    f"A/E Survival fraction sweep determination failed for {peak} peak"
                )
        sfs.set_index("peak", inplace=True)
        return sfs, peak_dfs

    def calculate_survival_fractions(
        self, data, aoe_param, peaks, fit_widths, mode="greater"
    ):
        sfs = pd.DataFrame(columns=["peak", "sf", "sf_err"])
        for i, peak in enumerate(peaks):
            fwhm = self.eres_func(peak)
            try:
                if peak == 2039:
                    emin = 2 * fwhm
                    emax = 2 * fwhm
                    peak_df = data.query(
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )

                    sf_dict = compton_sf(
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        self.high_cut_val,
                        mode=mode,
                        dt_mask=peak_df[self.dt_cut_param].to_numpy()
                        if self.dt_cut_param is not None
                        else None,
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
                    peak_df = data.query(
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )
                    sf, sf_err, _, _ = get_survival_fraction(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        mode=mode,
                        high_cut=self.high_cut_val,
                        dt_mask=peak_df[self.dt_cut_param].to_numpy()
                        if self.dt_cut_param is not None
                        else None,
                    )
                    sfs = pd.concat(
                        [
                            sfs,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                log.info(f"{peak}keV: {sf:2.1f} +/- {sf_err:2.1f} %")

            except Exception:
                sfs = pd.concat(
                    [
                        sfs,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(f"A/E survival fraction determination failed for {peak} peak")
        sfs.set_index("peak", inplace=True)
        return sfs

    def get_results_dict(self):
        return {
            "cal_energy_param": self.cal_energy_param,
            "dt_param": self.dt_param,
            "rt_correction": self.dt_corr,
            "pdf": self.pdf.__name__,
            "1000-1300keV": self.timecorr_df.to_dict("index"),
            "correction_fit_results": self.energy_corr_res_dict,
            "low_cut": self.low_cut_val,
            "high_cut": self.high_cut_val,
            "low_side_sfs": self.low_side_sf.to_dict("index"),
            "2_side_sfs": self.two_side_sf.to_dict("index"),
        }

    def fill_plot_dict(self, data, plot_dict=None):
        if plot_dict is not None:
            for key, item in self.plot_options.items():
                if item["options"] is not None:
                    plot_dict[key] = item["function"](self, data, **item["options"])
                else:
                    plot_dict[key] = item["function"](self, data)
        else:
            plot_dict = {}
        return plot_dict

    def calibrate(
        self,
        df,
        initial_aoe_param,
        peaks_of_interest=None,
        fit_widths=None,
        cut_peak_idx=0,
        dep_acc=0.9,
        sf_nsamples=26,
        sf_cut_range=(-5, 5),
    ):
        if peaks_of_interest is None:
            peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        if fit_widths is None:
            fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]

        self.time_correction(df, initial_aoe_param, output_name="AoE_Timecorr")
        log.info("Finished A/E time correction")

        if self.dt_corr is True:
            aoe_param = "AoE_DTcorr"
            self.drift_time_correction(df, "AoE_Timecorr", out_param=aoe_param)
        else:
            aoe_param = "AoE_Timecorr"

        self.energy_correction(
            df,
            aoe_param,
            corrected_param="AoE_Corrected",
            classifier_param="AoE_Classifier",
        )

        self.get_aoe_cut_fit(
            df,
            "AoE_Classifier",
            peaks_of_interest[cut_peak_idx],
            fit_widths[cut_peak_idx],
            dep_acc,
            output_cut_param="AoE_Low_Cut",
        )

        df["AoE_Double_Sided_Cut"] = df["AoE_Low_Cut"] & (
            df["AoE_Classifier"] < self.high_cut_val
        )

        if self.dt_cut_param is not None and self.dt_cut_hard is True:
            self.update_cal_dicts(
                {
                    "AoE_High_Side_Cut": {
                        "expression": f"(a<AoE_Classifier)& ({self.dt_cut_param}",
                        "parameters": {"a": self.high_cut_val},
                    }
                }
            )
        else:
            self.update_cal_dicts(
                {
                    "AoE_High_Side_Cut": {
                        "expression": "(a<AoE_Classifier)",
                        "parameters": {"a": self.high_cut_val},
                    }
                }
            )

        self.update_cal_dicts(
            {
                "AoE_Double_Sided_Cut": {
                    "expression": "(a>AoE_Classifier) & (AoE_Low_Cut)",
                    "parameters": {"a": self.high_cut_val},
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
            nsamples=sf_nsamples,
            cut_range=sf_cut_range,
            mode="greater",
        )

        log.info("Compute 2 side survival fractions: ")
        self.two_side_sf = self.calculate_survival_fractions(
            df, "AoE_Classifier", peaks_of_interest, fit_widths, mode="greater"
        )


def plot_aoe_mean_time(
    aoe_class, data, time_param="AoE_Timecorr", figsize=(12, 8), fontsize=12
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
            y1=np.array(grouped_means) - 0.2 * np.array(aoe_class.timecorr_df["res"]),
            y2=np.array(grouped_means) + 0.2 * np.array(aoe_class.timecorr_df["res"]),
            color="green",
            alpha=0.2,
        )
        ax.fill_between(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in aoe_class.cal_dicts
            ],
            y1=np.array(grouped_means) - 0.4 * np.array(aoe_class.timecorr_df["res"]),
            y2=np.array(grouped_means) + 0.4 * np.array(aoe_class.timecorr_df["res"]),
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
    aoe_class, data, time_param="AoE_Timecorr", figsize=(12, 8), fontsize=12
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
            aoe_class.timecorr_df["res"],
            yerr=aoe_class.timecorr_df["res_err"],
            linestyle=" ",
        )
    except Exception:
        pass
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

        xs = np.linspace(aoe_pars["lower_range"], aoe_pars["upper_range"], 100)
        counts, aoe_bins, bars = plt.hist(
            final_df.query(
                f'{aoe_class.dt_res_dict["aoe_grp1"]}&({aoe_param}<{aoe_pars["upper_range"]})&({aoe_param}>{aoe_pars["lower_range"]})'
            )[aoe_param],
            bins=400,
            histtype="step",
            label="data",
        )
        dx = np.diff(aoe_bins)
        plt.plot(xs, aoe_class.pdf.pdf(xs, *aoe_pars) * dx[0], label="full fit")
        sig, bkg = aoe_class.pdf.pdf(xs, *aoe_pars[:-1], True)
        plt.plot(xs, sig * dx[0], label="peak fit")
        plt.plot(xs, bkg * dx[0], label="bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("counts")

        aoe_pars2 = aoe_class.dt_res_dict["aoe_fit2"]["pars"]
        plt.subplot(2, 2, 2)
        xs = np.linspace(aoe_pars2["lower_range"], aoe_pars2["upper_range"], 100)
        counts, aoe_bins2, bars = plt.hist(
            final_df.query(
                f'{aoe_class.dt_res_dict["aoe_grp2"]}&({aoe_param}<{aoe_pars2["upper_range"]})&({aoe_param}>{aoe_pars2["lower_range"]})'
            )[aoe_param],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins2)
        plt.plot(xs, aoe_class.pdf.pdf(xs, *aoe_pars2) * dx[0], label="full fit")
        sig, bkg = aoe_class.pdf.pdf(xs, *aoe_pars2[:-1], True)
        plt.plot(xs, sig * dx[0], label="peak fit")
        plt.plot(xs, bkg * dx[0], label="bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("counts")

        hist, bins, var = pgh.get_hist(
            final_df[aoe_class.dt_param],
            dx=10,
            range=(
                np.nanmin(final_df[aoe_class.dt_param]),
                np.nanmax(final_df[aoe_class.dt_param]),
            ),
        )

        plt.subplot(2, 2, 3)
        plt.step(pgh.get_bin_centers(bins), hist, label="data")
        plt.plot(
            pgh.get_bin_centers(bins),
            DriftTimeDistribution.pdf(
                pgh.get_bin_centers(bins), **aoe_class.dt_res_dict["dt_guess"]
            )
            * np.diff(bins)[0],
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins),
            DriftTimeDistribution.pdf(
                pgh.get_bin_centers(bins), *aoe_class.dt_res_dict["dt_fit"]["pars"]
            )
            * np.diff(bins)[0],
            label="fit",
        )
        plt.xlabel("drift time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

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
    aoe_range: list[float] = None,
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
    titles: list = None,
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

            hist, bs, var = pgh.get_hist(select_df[aoe_param], bins=500)
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


def plot_mean_fit(aoe_class, data, figsize=(12, 8), fontsize=12) -> plt.figure:
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


def plot_sigmafit(aoe_class, data, figsize=(12, 8), fontsize=12) -> plt.figure:
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
            label = f'sqrt model: \nsqrt({sig_pars["a"]:1.4f}+({sig_pars["b"]:1.1f}/E)^{sig_pars["c"]:1.1f})'
        else:
            raise ValueError("unknown sigma function")
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


def plot_cut_fit(aoe_class, data, figsize=(12, 8), fontsize=12) -> plt.figure:
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
            (100 * aoe_class.dep_acc),
            -8.1,
            aoe_class.low_cut_val,
            color="red",
            linestyle="--",
        )
        plt.vlines(
            aoe_class.low_cut_val,
            np.nanmin(aoe_class.cut_fits["sf"]) * 0.9,
            (100 * aoe_class.dep_acc),
            color="red",
            linestyle="--",
        )
        plt.xlim([-8.1, 0.1])
        vals, labels = plt.yticks()
        plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
        plt.ylim([np.nanmin(aoe_class.cut_fits["sf"]) * 0.9, 102])
    except Exception:
        pass
    plt.xlabel("cut value")
    plt.ylabel("survival percentage")
    plt.close()
    return fig


def plot_survival_fraction_curves(
    aoe_class, data, figsize=(12, 8), fontsize=12
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
            try:
                plt.errorbar(
                    survival_df.index,
                    survival_df["sf"],
                    yerr=survival_df["sf_err"],
                    label=f'{get_peak_label(peak)} {peak} keV: {aoe_class.low_side_sf.loc[peak]["sf"]:2.1f} +/- {aoe_class.low_side_sf.loc[peak]["sf_err"]:2.1f} %',
                )
            except Exception:
                pass
    except Exception:
        pass
    vals, labels = plt.yticks()
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
    vals, labels = plt.yticks()
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
    try:
        plt.hist2d(
            data.query(aoe_class.selection_string)[aoe_class.cal_energy_param],
            data.query(aoe_class.selection_string)[aoe_param],
            bins=[
                np.linspace(xrange[0], xrange[1], xn_bins),
                np.linspace(yrange[0], yrange[1], yn_bins),
            ],
            norm=LogNorm(),
        )
    except Exception:
        pass
    plt.xlabel("energy (keV)")
    plt.ylabel(aoe_param)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.close()
    return fig
