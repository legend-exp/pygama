"""
This module provides functions for correcting the a/e energy dependence, determining the cut level and calculating survival fractions.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
from datetime import datetime
from typing import Callable

import matplotlib as mpl

mpl.use("agg")
import lgdo.lh5_store as lh5
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.stats import chi2

import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.ecal_th as thc
import pygama.pargen.energy_cal as pgc
from pygama.math.peak_fitting import nb_erfc
from pygama.pargen.energy_cal import get_i_local_maxima

log = logging.getLogger(__name__)


def return_nans(func):
    args = func.__code__.co_varnames[: func.__code__.co_argcount][1:]
    c = cost.UnbinnedNLL(np.array([0]), func)
    m = Minuit(c, *[np.nan for arg in args])
    return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)


class PDF:

    """
    Base class for A/E pdfs.
    """

    def pdf(x):
        return

    def return_nans(self):
        args = self.pdf.__code__.co_varnames[: self.pdf.__code__.co_argcount][2:]
        c = cost.UnbinnedNLL(np.array([0]), self.pdf)
        m = Minuit(c, *[np.nan for arg in args])
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)

    def _replace_values(self, dic, **kwargs):
        for item, value in kwargs.items():
            dic[item] = value
        return dic


class standard_aoe(PDF):
    def pdf(
        self,
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
        PDF for A/E consists of a gaussian signal with gaussian tail background
        """
        try:
            sig = n_sig * pgf.gauss_norm(x, mu, sigma)
            bkg = n_bkg * pgf.gauss_tail_norm(
                x, mu, sigma, tau_bkg, lower_range, upper_range
            )
        except:
            sig = np.full_like(x, np.nan)
            bkg = np.full_like(x, np.nan)

        if components == False:
            return sig + bkg
        else:
            return sig, bkg

    def extended_pdf(
        self,
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
        Extended PDF for A/E consists of a gaussian signal with gaussian tail background
        """
        if components == True:
            sig, bkg = self.pdf(
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
            return n_sig + n_bkg, self.pdf(
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

    def guess(self, hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
        except:
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

        return self._replace_values(guess_dict, **kwargs)

    def bounds(self, guess, **kwargs):
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
            for field, bound in self._replace_values(bounds_dict, **kwargs).items()
        ]

    def fixed(self, **kwargs):
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
            fixed for field, fixed in self._replace_values(fixed_dict, **kwargs).items()
        ]

    def width(self, pars, errs, cov):
        return pars["sigma"], errs["sigma"]

    def centroid(self, pars, errs, cov):
        return pars["mu"], errs["mu"]


class standard_aoe_with_high_tail(PDF):
    def pdf(
        self,
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
        PDF for A/E consists of a gaussian signal with tail with gaussian tail background
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
        except:
            sig = np.full_like(x, np.nan)
            bkg = np.full_like(x, np.nan)

        if components == False:
            return sig + bkg
        else:
            return sig, bkg

    def extended_pdf(
        self,
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
        Extended PDF for A/E consists of a gaussian signal with gaussian tail background
        """
        if components == True:
            sig, bkg = self.pdf(
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
            return n_sig + n_bkg, self.pdf(
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

    def guess(self, hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
        except:
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

        return self._replace_values(guess_dict, **kwargs)

    def bounds(self, guess, **kwargs):
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
            for field, bound in self._replace_values(bounds_dict, **kwargs).items()
        ]

    def fixed(self, **kwargs):
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
            fixed for field, fixed in self._replace_values(fixed_dict, **kwargs).items()
        ]

    def width(self, pars, errs, cov):
        fwhm, fwhm_err = pgf.radford_fwhm(
            pars[2], pars[3], np.abs(pars[4]), cov=cov[:7, :7]
        )
        return fwhm / 2.355, fwhm_err / 2.355

    def centroid(self, pars, errs, cov):
        return pars["mu"], errs["mu"]


class standard_aoe_bkg(PDF):
    def pdf(
        self,
        x: np.array,
        n_events: float,
        mu: float,
        sigma: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
    ) -> np.array:
        """
        PDF for A/E consists of a gaussian signal with tail with gaussian tail background
        """
        try:
            sig = n_events * pgf.gauss_tail_norm(
                x, mu, sigma, tau_bkg, lower_range, upper_range
            )
        except:
            sig = np.full_like(x, np.nan)

        return sig

    def extended_pdf(
        self,
        x: np.array,
        n_events: float,
        mu: float,
        sigma: float,
        tau_bkg: float,
        lower_range: float = np.inf,
        upper_range: float = np.inf,
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a gaussian signal with gaussian tail background
        """
        return n_events, self.pdf(
            x, n_events, mu, sigma, tau_bkg, lower_range, upper_range
        )

    def guess(self, hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
        except:
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

        return self._replace_values(guess_dict, **kwargs)

    def bounds(self, guess, **kwargs):
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
            for field, bound in self._replace_values(bounds_dict, **kwargs).items()
        ]

    def fixed(self, **kwargs):
        fixed_dict = {
            "n_bkg": False,
            "mu": False,
            "sigma": False,
            "tau_bkg": False,
            "lower_range": True,
            "upper_range": True,
        }

        return [
            fixed for field, fixed in self._replace_values(fixed_dict, **kwargs).items()
        ]


class gaussian(PDF):
    def pdf(self, x: np.array, n_events: float, mu: float, sigma: float) -> np.array:
        """
        PDF for A/E consists of a gaussian signal with tail with gaussian tail background
        """
        try:
            sig = n_events * pgf.gauss_norm(x, mu, sigma)
        except:
            sig = np.full_like(x, np.nan)

        return sig

    def extended_pdf(
        self, x: np.array, n_events: float, mu: float, sigma: float
    ) -> tuple(float, np.array):
        """
        Extended PDF for A/E consists of a gaussian signal with gaussian tail background
        """
        return n_events, self.pdf(
            x, n_events, mu, sigma, tau_bkg, lower_range, upper_range
        )

    def guess(self, hist, bins, var, **kwargs):
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mu = bin_centers[np.argmax(hist)]
        try:
            _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
        except:
            pars, cov = pgf.gauss_mode_width_max(
                hist, bins, var, mode_guess=mu, n_bins=20
            )
            _, sigma, _ = pars
        ls_guess = 2 * np.sum(
            hist[(bin_centers > mu) & (bin_centers < (mu + 2.5 * sigma))]
        )

        guess_dict = {"n_events": ls_guess, "mu": mu, "sigma": sigma}

        return self._replace_values(guess_dict, **kwargs)

    def bounds(self, gpars, **kwargs):
        bounds_dict = {"n_events": (0, None), "mu": (None, None), "sigma": (0, None)}

        return [
            bound
            for field, bound in self._replace_values(bounds_dict, **kwargs).items()
        ]

    def fixed(self, **kwargs):
        fixed_dict = {
            "n_events": False,
            "mu": False,
            "sigma": False,
        }

        return [
            fixed for field, fixed in self._replace_values(fixed_dict, **kwargs).items()
        ]


class drift_time_distribution(PDF):
    def pdf(
        self,
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

    def extended_pdf(
        self,
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
            gauss1, gauss2 = self.pdf(
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
            return n_sig1 + n_sig2, self.pdf(
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

    def guess(self, hist: np.array, bins: np.array, var: np.array, **kwargs) -> list:
        """
        Guess for fitting dt spectrum
        """
        bcs = pgh.get_bin_centers(bins)
        mus = get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 5)
        if len(mus) > 2:
            mus = get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 8)
        elif len(mus) < 2:
            mus = get_i_local_maxima(hist / (np.sqrt(var) + 10**-99), 3)
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

        return self._replace_values(guess_dict, **kwargs)

    def bounds(self, guess, **kwargs):
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
            for field, bound in self._replace_values(bounds_dict, **kwargs).items()
        ]

    def fixed(self, **kwargs):
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
            fixed for field, fixed in self._replace_values(fixed_dict, **kwargs).items()
        ]


def tag_pulser(files, lh5_path):
    pulser_df = lh5.load_dfs(files, ["timestamp", "trapTmax"], lh5_path)
    pulser_props = cts.find_pulser_properties(pulser_df, energy="trapTmax")
    if len(pulser_props) > 0:
        final_mask = None
        for entry in pulser_props:
            e_cut = (pulser_df.trapTmax.values < entry[0] + entry[1]) & (
                pulser_df.trapTmax.values > entry[0] - entry[1]
            )
            if final_mask is None:
                final_mask = e_cut
            else:
                final_mask = final_mask | e_cut
        ids = ~(final_mask)
        log.debug(f"pulser found: {pulser_props}")
    else:
        ids = np.ones(len(pulser_df), dtype=bool)
        log.debug(f"no pulser found")
    return ids


def load_aoe(
    files: list,
    lh5_path: str,
    cal_dict: dict,
    params: [
        A_max,
        tp_0_est,
        tp_99,
        dt_eff,
        A_max_tri,
        cuspEmax,
        cuspEmax_ctc_cal,
        is_valid_cal,
    ],
    energy_param: str,
    current_param: str,
) -> tuple(np.array, np.array, np.array, np.array):
    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    # switch this to dataframes, include timestamp

    sto = lh5.LH5Store()

    if isinstance(files, dict):
        df = []
        all_files = []
        for tstamp, tfiles in files.items():
            table = sto.read_object(lh5_path, tfiles)[0]
            if tstamp in cal_dict:
                file_df = table.eval(cal_dict[tstamp]).get_dataframe()
            else:
                file_df = table.eval(cal_dict).get_dataframe()
            file_df["timestamp"] = np.full(len(file_df), tstamp, dtype=object)
            params.append("timestamp")
            df.append(file_df)
            all_files += tfiles

        df = pd.concat(df)

    elif isinstance(files, list):
        table = sto.read_object(lh5_path, files)[0]
        df = table.eval(cal_dict).get_dataframe()
        all_files = files

    ids = tag_pulser(all_files, lh5_path)
    df["is_not_pulser"] = ids
    params.append("is_not_pulser")

    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True, axis=1)

    param_dict = {}
    for param in params:
        # add cuts in here
        if param not in df:
            df[param] = lh5.load_nda(all_files, [param], lh5_path)[param]

    df["AoE_uncorr"] = np.divide(df[current_param], df[energy_param])
    return df


def unbinned_aoe_fit(
    aoe: np.array, pdf=standard_aoe, display: int = 0, verbose: bool = False
) -> tuple(np.array, np.array):
    """
    Fitting function for A/E, first fits just a gaussian before using the full pdf to fit
    if fails will return NaN values
    """
    hist, bins, var = pgh.get_hist(aoe, bins=500)

    gauss = gaussian()
    gpars = gauss.guess(hist, bins, var)
    c1_min = gpars["mu"] - 2 * gpars["sigma"]
    c1_max = gpars["mu"] + 3 * gpars["sigma"]

    # Initial fit just using Gaussian
    c1 = cost.UnbinnedNLL(aoe[(aoe < c1_max) & (aoe > c1_min)], gauss.pdf)

    m1 = Minuit(c1, **gpars)
    m1.limits = [
        (0, len(aoe[(aoe < c1_max) & (aoe > c1_min)])),
        (gpars["mu"] * 0.8, gpars["mu"] * 1.2),
        (0.8 * gpars["sigma"], gpars["sigma"] * 1.2),
    ]
    m1.fixed = gauss.fixed()
    m1.migrad()

    if verbose:
        print(m1)

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = m1.values["mu"] - 15 * m1.values["sigma"]
    if fmin < np.nanmin(aoe):
        fmin = np.nanmin(aoe)
    fmax_bkg = m1.values["mu"] - 5 * m1.values["sigma"]
    fmax = m1.values["mu"] + 5 * m1.values["sigma"]

    n_bkg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - m1.values["n_events"]

    aoe_bkg = standard_aoe_bkg()
    bkg_guess = aoe_bkg.guess(
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
        aoe[(aoe < fmax_bkg) & (aoe > fmin)], aoe_bkg.extended_pdf
    )
    m2 = Minuit(c2, **bkg_guess)
    m2.fixed = aoe_bkg.fixed(mu=True)
    m2.limits = aoe_bkg.bounds(
        bkg_guess, n_events=(0, 2 * len(aoe[(aoe < fmax_bkg) & (aoe > fmin)]))
    )
    m2.simplex().migrad()
    m2.hesse()

    aoe_pdf = pdf()

    x0 = aoe_pdf.guess(
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
    if verbose:
        print(x0)

    # Full fit using gaussian signal with gaussian tail background
    c = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax) & (aoe > fmin)], aoe_pdf.extended_pdf)
    m = Minuit(c, **x0)
    m.limits = aoe_pdf.bounds(
        x0,
        n_sig=(0, 2 * len(aoe[(aoe < fmax) & (aoe > fmin)])),
        n_bkg=(0, 2 * len(aoe[(aoe < fmax) & (aoe > fmin)])),
    )
    m.fixed = aoe_pdf.fixed()
    m.migrad()
    m.hesse()

    if verbose:
        print(m)

    if np.isnan(m.errors).all():
        try:
            m.simplex.migrad()
            m.hesse()
        except:
            return aoe_pdf.return_nans()

    if display > 1:
        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, bars = plt.hist(
            aoe[(aoe < fmax) & (aoe > fmin)], bins=200, histtype="step", label="Data"
        )
        dx = np.diff(bins)
        plt.plot(xs, aoe_pdf.pdf(xs, *m.values) * dx[0], label="Full fit")
        sig, bkg = aoe_pdf.pdf(xs, *m.values[:-1], True)
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(xs, gauss.pdf(xs, *m1.values) * dx[0], label="Initial Gaussian")
        plt.plot(xs, aoe_bkg.pdf(xs, *m2.values) * dx[0], label="Bkg guess")
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.show()

        plt.figure()
        bin_centers = (bins[1:] + bins[:-1]) / 2
        res = (aoe_pdf.pdf(bin_centers, *m.values) * dx[0]) - counts
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


def aoe_timecorr(
    df, energy_param, current_param, pdf=standard_aoe, plot_dict={}, display=0
):
    if "timestamp" in df:
        tstamps = sorted(np.unique(df["timestamp"]))
        if len(tstamps) > 1:
            means = []
            errors = []
            reses = []
            res_errs = []
            final_tstamps = []
            for tstamp, time_df in df.groupby("timestamp", sort=True):
                pars, errs, cov = unbinned_aoe_fit(
                    time_df.query(
                        f"is_usable_fits & cuspEmax_ctc_cal>1000 & cuspEmax_ctc_cal<1300"
                    )["AoE_uncorr"],
                    pdf=pdf,
                    display=display,
                )
                final_tstamps.append(tstamp)
                means.append(pars["mu"])
                errors.append(errs["mu"])
                reses.append(pars["sigma"] / pars["mu"])
                res_errs.append(
                    reses[-1]
                    * np.sqrt(errs["sigma"] / pars["sigma"] + errs["mu"] / pars["mu"])
                )
            mean_dict = fit_time_means(tstamps, means, reses)

            df["AoE_timecorr"] = df["AoE_uncorr"] / np.array(
                [mean_dict[tstamp] for tstamp in df["timestamp"]]
            )
            out_dict = {
                tstamp: {
                    "AoE_Timecorr": {
                        "expression": f"({current_param}/{energy_param})/a",
                        "parameters": {"a": mean_dict[tstamp]},
                    }
                }
                for tstamp in mean_dict
            }
            res_dict = {
                "times": tstamps,
                "mean": means,
                "mean_errs": errors,
                "res": reses,
                "res_errs": res_errs,
            }
            if display > 0:
                fig1, ax = plt.subplots(1, 1)
                ax.errorbar(
                    [datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ") for tstamp in tstamps],
                    means,
                    yerr=errors,
                    linestyle=" ",
                )
                ax.step(
                    [
                        datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                        for tstamp in list(mean_dict)
                    ],
                    [mean_dict[tstamp] for tstamp in mean_dict],
                    where="post",
                )
                ax.fill_between(
                    [
                        datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                        for tstamp in list(mean_dict)
                    ],
                    y1=np.array([mean_dict[tstamp] for tstamp in mean_dict])
                    - 0.2 * np.array(reses),
                    y2=np.array([mean_dict[tstamp] for tstamp in mean_dict])
                    + 0.2 * np.array(reses),
                    color="green",
                    alpha=0.2,
                )
                ax.fill_between(
                    [
                        datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                        for tstamp in list(mean_dict)
                    ],
                    y1=np.array([mean_dict[tstamp] for tstamp in mean_dict])
                    - 0.4 * np.array(reses),
                    y2=np.array([mean_dict[tstamp] for tstamp in mean_dict])
                    + 0.4 * np.array(reses),
                    color="yellow",
                    alpha=0.2,
                )
                ax.set_xlabel("time")
                ax.set_ylabel("A/E mean")
                myFmt = mdates.DateFormatter("%b %d")
                ax.xaxis.set_major_formatter(myFmt)
                plot_dict["aoe_time"] = fig1
                if display > 1:
                    plt.show()
                else:
                    plt.close()
                fig2, ax = plt.subplots(1, 1)
                ax.errorbar(
                    [datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ") for tstamp in tstamps],
                    reses,
                    yerr=res_errs,
                    linestyle=" ",
                )
                ax.set_xlabel("time")
                ax.set_ylabel("A/E res")
                myFmt = mdates.DateFormatter("%b %d")
                ax.xaxis.set_major_formatter(myFmt)
                plot_dict["aoe_res"] = fig2
                if display > 1:
                    plt.show()
                else:
                    plt.close()
                return df, out_dict, res_dict, plot_dict
            else:
                return df, out_dict, res_dict
    else:
        pars, errs, cov = unbinned_aoe_fit(
            df.query("is_usable_fits & cuspEmax_ctc_cal>1000 & cuspEmax_ctc_cal<1300")[
                "AoE_uncorr"
            ]
        )
        df["AoE_timecorr"] = df["AoE_uncorr"] / pars["mu"]
        out_dict = {
            "AoE_Timecorr": {
                "expression": f"({current_param}/{energy_param})/a",
                "parameters": {"a": pars["mu"]},
            }
        }
        res_err = (pars["sigma"] / pars["mu"]) * np.sqrt(
            errs["sigma"] / pars["sigma"] + errs["mu"] / pars["mu"]
        )
        fit_result = {
            "times": [np.nan],
            "mean": [pars["mu"]],
            "mean_errs": [errs["mu"]],
            "res": [pars["sigma"] / pars["mu"]],
            "res_errs": [res_err],
        }
        if display > 0:
            return df, out_dict, fit_result, plot_dict
        else:
            return df, out_dict, fit_result


def pol1(x: np.array, a: float, b: float) -> np.array:
    """Basic Polynomial for fitting A/E centroid against energy"""
    return a * x + b


def sigma_fit(x: np.array, a: float, b: float, c: float) -> np.array:
    """Function definition for fitting A/E sigma against energy"""
    return np.sqrt(a + (b / (x + 10**-99)) ** c)


def AoEcorrection(
    energy: np.array,
    aoe: np.array,
    eres: list,
    pdf=standard_aoe,
    plot_dict: dict = {},
    display: int = 0,
    comptBands_width=20,
    sigma_func=sigma_fit,
) -> tuple(np.array, np.array):
    """
    Calculates the corrections needed for the energy dependence of the A/E.
    Does this by fitting the compton continuum in slices and then applies fits to the centroid and variance.
    """

    comptBands = np.arange(900, 2350, comptBands_width)
    peaks = np.array([1080, 1094, 1459, 1512, 1552, 1592, 1620, 1650, 1670, 1830, 2105])
    allowed = np.array([], dtype=bool)
    for i, band in enumerate(comptBands):
        allow = True
        for peak in peaks:
            if (peak - 5) > band and (peak - 5) < (band + comptBands_width):
                allow = False
            elif (peak + 5 > band) and (peak + 5) < (band + comptBands_width):
                allow = False
        allowed = np.append(allowed, allow)
    comptBands = comptBands[allowed]

    results_dict = {}
    comptBands = comptBands[::-1]  # Flip so color gets darker when plotting
    compt_aoe = np.zeros(len(comptBands))
    aoe_sigmas = np.zeros(len(comptBands))
    compt_aoe_err = np.zeros(len(comptBands))
    aoe_sigmas_err = np.zeros(len(comptBands))
    ratio = np.zeros(len(comptBands))
    ratio_err = np.zeros(len(comptBands))

    copper = cm = plt.get_cmap("copper")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(comptBands))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=copper)

    if display > 0:
        fits_fig = plt.figure()

    # Fit each compton band
    for i, band in enumerate(comptBands):
        aoe_tmp = aoe[
            (energy > band) & (energy < band + comptBands_width) & (aoe > 0)
        ]  # [:20000]
        try:
            aoe_pdf = pdf()
            pars, errs, cov = unbinned_aoe_fit(aoe_tmp, pdf=pdf, display=display)
            compt_aoe[i], compt_aoe_err[i] = aoe_pdf.centroid(pars, errs, cov)
            aoe_sigmas[i], aoe_sigmas_err[i] = aoe_pdf.width(pars, errs, cov)

            ratio[i] = pars["n_sig"] / pars["n_bkg"]
            ratio_err[i] = ratio[i] * np.sqrt(
                (errs["n_sig"] / pars["n_sig"]) ** 2
                + (errs["n_bkg"] / pars["n_bkg"]) ** 2
            )

        except:
            compt_aoe[i] = np.nan
            aoe_sigmas[i] = np.nan
            compt_aoe_err[i] = np.nan
            aoe_sigmas_err[i] = np.nan
            ratio[i] = np.nan
            ratio_err[i] = np.nan

        if display > 0:
            if (
                np.isnan(errs["mu"])
                | np.isnan(errs["sigma"])
                | (errs["mu"] == 0)
                | (errs["sigma"] == 0)
            ):
                pass
            else:
                xs = np.arange(
                    pars["mu"] - 4 * pars["sigma"],
                    pars["mu"] + 3 * pars["sigma"],
                    pars["sigma"] / 10,
                )
                colorVal = scalarMap.to_rgba(i)
                aoe_pdf = pdf()
                plt.plot(xs, aoe_pdf.pdf(xs, *pars), color=colorVal)

    if display > 0:
        plt.xlabel("A/E")
        plt.ylabel("Expected Counts")
        plt.title("Compton Band Fits")
        cbar = plt.colorbar(
            cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("copper_r")),
            orientation="horizontal",
            label="Compton Band Energy",
            ticks=[0, 16, 32, len(comptBands)],
        )  # cax=ax,
        cbar.ax.set_xticklabels(
            [
                comptBands[::-1][0],
                comptBands[::-1][16],
                comptBands[::-1][32],
                comptBands[::-1][-1],
            ]
        )
        plot_dict["band_fits"] = fits_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

    ids = (
        np.isnan(compt_aoe_err)
        | np.isnan(aoe_sigmas_err)
        | (aoe_sigmas_err == 0)
        | (compt_aoe_err == 0)
    )
    results_dict["n_of_valid_fits"] = len(np.where(~ids)[0])
    # Fit mus against energy
    p0_mu = [-1e-06, 5e-01]
    c_mu = cost.LeastSquares(
        comptBands[~ids], compt_aoe[~ids], compt_aoe_err[~ids], pol1
    )
    c_mu.loss = "soft_l1"
    m_mu = Minuit(c_mu, *p0_mu)
    m_mu.simplex()
    m_mu.migrad()
    m_mu.hesse()

    pars = m_mu.values
    errs = m_mu.errors

    csqr_mu = np.sum(
        ((compt_aoe[~ids] - pol1(comptBands[~ids], *pars)) ** 2) / compt_aoe_err[~ids]
    )
    dof_mu = len(compt_aoe[~ids]) - len(pars)
    results_dict["p_val_mu"] = chi2.sf(csqr_mu, dof_mu)
    results_dict["csqr_mu"] = (csqr_mu, dof_mu)

    # Fit sigma against energy
    p0_sig = [np.nanpercentile(aoe_sigmas[~ids], 50) ** 2, 2, 2]
    c_sig = cost.LeastSquares(
        comptBands[~ids], aoe_sigmas[~ids], aoe_sigmas_err[~ids], sigma_func
    )
    c_sig.loss = "soft_l1"
    m_sig = Minuit(c_sig, *p0_sig)
    m_sig.simplex()
    m_sig.migrad()
    m_sig.hesse()

    sig_pars = m_sig.values
    sig_errs = m_sig.errors

    csqr_sig = np.sum(
        ((aoe_sigmas[~ids] - sigma_func(comptBands[~ids], *sig_pars)) ** 2)
        / aoe_sigmas_err[~ids]
    )
    dof_sig = len(aoe_sigmas[~ids]) - len(sig_pars)
    results_dict["p_val_sig"] = chi2.sf(csqr_sig, dof_sig)
    results_dict["csqr_sig"] = (csqr_sig, dof_sig)

    model = pol1(comptBands, *pars)
    sig_model = sigma_func(comptBands, *sig_pars)

    # Get DEP fit
    sigma = np.sqrt(eres[0] + 1592 * eres[1]) / 2.355
    n_sigma = 4
    peak = 1592
    emin = peak - n_sigma * sigma
    emax = peak + n_sigma * sigma
    try:
        dep_pars, dep_err, dep_cov = unbinned_aoe_fit(
            aoe[(energy > emin) & (energy < emax) & (aoe > 0)], pdf=pdf
        )
    except:
        dep_pars, dep_err, dep_cov = return_nans(pdf)

    if display > 0:
        mean_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            compt_aoe[~ids],
            yerr=compt_aoe_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        ax1.plot(comptBands[~ids] + 10, model[~ids], label="linear model")
        ax1.errorbar(
            1592,
            dep_pars["mu"],
            xerr=n_sigma * sigma,
            yerr=dep_err["mu"],
            label="DEP",
            color="green",
            linestyle=" ",
        )

        ax1.legend(title="A/E mu energy dependence", frameon=False)

        ax1.set_ylabel("raw A/E (a.u.)", ha="right", y=1)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (compt_aoe[~ids] - model[~ids]) / model[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100 * (dep_pars["mu"] - pol1(1592, *pars)) / pol1(1592, *pars),
            lw=1,
            c="g",
        )
        ax2.set_ylabel("Residuals %", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["mean_fit"] = mean_fig
        if display > 1:
            plt.show()
        else:
            plt.close()

        sig_fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.errorbar(
            comptBands[~ids] + 10,
            aoe_sigmas[~ids],
            yerr=aoe_sigmas_err[~ids],
            xerr=10,
            label="data",
            linestyle=" ",
        )
        if sigma_func == sigma_fit:
            label = f"sqrt model: \nsqrt({sig_pars[0]:1.4f}+({sig_pars[1]:1.1f}/E)^{sig_pars[2]:1.1f})"
        elif sigma_func == sigma_fit_quadratic:
            label = f"quad model: \n({sig_pars[0]:1.4f}+({sig_pars[1]:1.6f}*E)+\n({sig_pars[2]:1.6f}*E)^2)"
        else:
            raise ValueError("unknown sigma function")
        ax1.plot(
            comptBands[~ids],
            sig_model[~ids],
            label=label,
        )
        ax1.errorbar(
            1592,
            dep_pars["sigma"],
            xerr=n_sigma * sigma,
            yerr=dep_err["sigma"],
            label="DEP",
            color="green",
        )
        ax1.set_ylabel("A/E stdev (a.u.)", ha="right", y=1)
        ax1.legend(title="A/E stdev energy dependence", frameon=False)
        ax2.scatter(
            comptBands[~ids] + 10,
            100 * (aoe_sigmas[~ids] - sig_model[~ids]) / sig_model[~ids],
            lw=1,
            c="b",
        )
        ax2.scatter(
            1592,
            100
            * (dep_pars["sigma"] - sigma_func(1592, *sig_pars))
            / sigma_func(1592, *sig_pars),
            lw=1,
            c="g",
        )
        ax2.set_ylabel("Residuals", ha="right", y=1)
        ax2.set_xlabel("Energy (keV)", ha="right", x=1)
        plt.tight_layout()
        plot_dict["sigma_fit"] = sig_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return pars, sig_pars, results_dict, dep_pars, plot_dict
    else:
        return pars, sig_pars, results_dict, dep_pars


def plot_compt_bands_overlayed(
    aoe: np.array,
    energy: np.array,
    eranges: list[tuple],
    aoe_range: list[float] = None,
    density=True,
) -> None:
    """
    Function to plot various compton bands to check energy dependence and corrections
    """

    for erange in eranges:
        range_idxs = (energy > erange - 10) & (energy < erange + 10)
        hist, bins, var = pgh.get_hist(
            aoe[range_idxs][
                (~np.isnan(aoe[range_idxs]))
                & (aoe[range_idxs] > np.nanpercentile(aoe[range_idxs], 1))
                & (aoe[range_idxs] < np.nanpercentile(aoe[range_idxs], 99))
            ],
            bins=100,
        )
        bin_cs = (bins[1:] + bins[:-1]) / 2
        mu = bin_cs[np.argmax(hist)]
        if aoe_range is not None:
            idxs = (
                (energy > erange - 10)
                & (energy < erange + 10)
                & (aoe > aoe_range[0])
                & (aoe < aoe_range[1])
                & (~np.isnan(aoe))
            )
            bins = np.linspace(aoe_range[0], aoe_range[1], 50)
        else:
            idxs = (energy > erange - 10) & (energy < erange + 10) & (~np.isnan(aoe))
            bins = np.linspace(0.85, 1.05, 200)
        plt.hist(
            aoe[idxs],
            bins=bins,
            histtype="step",
            label=f"{erange-10}-{erange+10}",
            density=density,
        )


def plot_dt_dep(
    aoe: np.array, energy: np.array, dt: np.array, erange: list[tuple], title: str
) -> None:
    """
    Function to produce 2d histograms of A/E against drift time to check dependencies
    """

    hist, bins, var = pgh.get_hist(
        aoe[(energy > erange[0]) & (energy < erange[1]) & (~np.isnan(aoe))], bins=500
    )
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > erange[0])
        & (energy < erange[1])
        & (aoe > aoe_range[0])
        & (aoe < aoe_range[1])
        & (dt < 2000)
    )

    plt.hist2d(aoe[idxs], dt[idxs], bins=[200, 100], norm=LogNorm())
    plt.ylabel("Drift Time (ns)")
    plt.xlabel("A/E")
    plt.title(title)


def energy_guess(hist, bins, var, func_i, peak, eres_pars, fit_range):
    """
    Simple guess for peak fitting
    """
    if func_i == pgf.extended_radford_pdf:
        bin_cs = (bins[1:] + bins[:-1]) / 2
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
        i_0 = np.nanargmax(hist)
        mu = peak
        height = hist[i_0]
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
        ]  #
        return parguess

    elif func_i == pgf.extended_gauss_step_pdf:
        mu = peak
        sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
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
        return [nsig_guess, mu, sigma, nbkg_guess, hstep, fit_range[0], fit_range[1], 0]


def unbinned_energy_fit(
    energy: np.array,
    peak: float,
    eres_pars: list = None,
    simplex=False,
    guess=None,
    display=0,
    verbose: bool = False,
) -> tuple(np.array, np.array):
    """
    Fitting function for energy peaks used to calculate survival fractions
    """
    try:
        hist, bins, var = pgh.get_hist(
            energy, dx=0.5, range=(np.nanmin(energy), np.nanmax(energy))
        )
    except ValueError:
        pars, errs, cov = return_nans(pgf.radford_pdf)
        return pars, errs
    sigma = thc.fwhm_slope(peak, *eres_pars) / 2.355
    if guess is None:
        x0 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_gauss_step_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_gauss_step_pdf)
        m = Minuit(c, *x0)
        m.limits = [
            (0, 2 * np.sum(hist)),
            (peak - 1, peak + 1),
            (0, None),
            (0, 2 * np.sum(hist)),
            (-1, 1),
            (None, None),
            (None, None),
            (None, None),
        ]
        m.fixed[-3:] = True
        m.simplex().migrad()
        m.hesse()
        x0 = m.values[:3]
        x0 += [0.2, 0.2 * m.values[2]]
        x0 += m.values[3:]
        if verbose:
            print(m)
        bounds = [
            (0, 2 * np.sum(hist)),
            (peak - 1, peak + 1),
            (0, None),
            (0, 1),
            (0, None),
            (0, 2 * np.sum(hist)),
            (-1, 1),
            (None, None),
            (None, None),
            (None, None),
        ]
        fixed = [7, 8, 9]
    else:
        x0 = guess
        x1 = energy_guess(
            hist,
            bins,
            var,
            pgf.extended_radford_pdf,
            peak,
            eres_pars,
            (np.nanmin(energy), np.nanmax(energy)),
        )
        x0[0] = x1[0]
        x0[5] = x1[5]
        bounds = [
            (0, 2 * np.sum(hist)),
            (guess[1] - 0.5, guess[1] + 0.5),
            sorted((0.8 * guess[2], 1.2 * guess[2])),
            sorted((0.8 * guess[3], 1.2 * guess[3])),
            sorted((0.8 * guess[4], 1.2 * guess[4])),
            (0, 2 * np.sum(hist)),
            sorted((0.8 * guess[6], 1.2 * guess[6])),
            (None, None),
            (None, None),
            (None, None),
        ]
        fixed = [1, 2, 3, 4, 6, 7, 8, 9]
    if len(x0) == 0:
        pars, errs, cov = return_nans(pgf.extended_radford_pdf)
        return pars, errs

    if verbose:
        print(x0)
    c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_radford_pdf)
    m = Minuit(c, *x0)
    m.limits = bounds
    for fix in fixed:
        m.fixed[fix] = True
    if simplex == True:
        m.simplex().migrad()
    else:
        m.migrad()

    m.hesse()
    if verbose:
        print(m)
    if display > 1:
        plt.figure()
        bcs = (bins[1:] + bins[:-1]) / 2
        plt.step(bcs, hist, where="mid")
        plt.plot(bcs, pgf.radford_pdf(bcs, *x0) * np.diff(bcs)[0])
        plt.plot(bcs, pgf.radford_pdf(bcs, *m.values) * np.diff(bcs)[0])
        plt.show()

    if not np.isnan(m.errors[:-3]).all():
        return m.values, m.errors
    else:
        try:
            m.simplex().migrad()
            m.minos()
            if not np.isnan(m.errors[:-3]).all():
                return m.values, m.errors
        except:
            pars, errs, cov = return_nans(pgf.extended_radford_pdf)
            return pars, errs


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


def get_survival_fraction(
    energy,
    aoe,
    cut_val,
    peak,
    eres_pars,
    high_cut=None,
    guess_pars_cut=None,
    guess_pars_surv=None,
    dt_mask=None,
    display=0,
):
    if dt_mask is None:
        dt_mask = np.full(len(aoe), True, dtype=bool)

    nan_idxs = np.isnan(aoe)
    if high_cut is not None:
        idxs = (aoe > cut_val) & (aoe < high_cut) & dt_mask
    else:
        idxs = (aoe > cut_val) & dt_mask

    if guess_pars_cut is None or guess_pars_surv is None:
        pars, errs = unbinned_energy_fit(energy, peak, eres_pars, simplex=True)
        guess_pars_cut = pars
        guess_pars_surv = pars

    cut_pars, ct_errs = unbinned_energy_fit(
        energy[(~nan_idxs) & (~idxs)],
        peak,
        eres_pars,
        guess=guess_pars_cut,
        simplex=False,
        display=display,
        verbose=False,
    )

    surv_pars, surv_errs = unbinned_energy_fit(
        energy[(~nan_idxs) & (idxs)],
        peak,
        eres_pars,
        guess=guess_pars_surv,
        simplex=False,
        display=display,
    )

    ct_n = cut_pars["n_sig"]
    ct_err = ct_errs["n_sig"]
    surv_n = surv_pars["n_sig"]
    surv_err = surv_errs["n_sig"]

    pc_n = ct_n + surv_n
    pc_err = np.sqrt(surv_err**2 + ct_err**2)

    sf = (surv_n / pc_n) * 100
    err = sf * np.sqrt((pc_err / pc_n) ** 2 + (surv_err / surv_n) ** 2)
    return sf, err, cut_pars, surv_pars


def get_aoe_cut_fit(
    energy: np.array,
    aoe: np.array,
    peak: float,
    ranges: tuple(int, int),
    dep_acc: float,
    eres_pars: list,
    display: int = 1,
    dep_correct: bool = False,
    dep_mu: Callable = None,
    sig_func: Callable = None,
    plot_dict={},
) -> float:
    """
    Determines A/E cut by sweeping through values and for each one fitting the DEP to determine how many events survive.
    Then interpolates to get cut value at desired DEP survival fraction (typically 90%)
    """

    min_range, max_range = ranges

    peak_energy = energy[
        (energy > peak - min_range) & (energy < peak + max_range)
    ]  # [:20000]
    peak_aoe = aoe[
        (energy > peak - min_range) & (energy < peak + max_range)
    ]  # [:20000]

    if dep_correct is True:
        peak_aoe = (peak_aoe / dep_mu(peak_energy)) - 1
        peak_aoe = peak_aoe / sig_func(peak_energy)

    cut_vals = np.arange(-8, 0, 0.2)
    sfs = []
    sf_errs = []
    for cut_val in cut_vals:
        sf, err, cut_pars, surv_pars = get_survival_fraction(
            peak_energy,
            peak_aoe,
            cut_val,
            peak,
            eres_pars,
            guess_pars_cut=None,
            guess_pars_surv=None,
        )
        sfs.append(sf)
        sf_errs.append(err)

    # return cut_vals, sfs, sf_errs
    ids = (sf_errs < (1.5 * np.nanpercentile(sf_errs, 85))) & (~np.isnan(sf_errs))

    def fit_func(x, a, b, c, d):
        return (a + b * x) * nb_erfc(c * x + d)

    c = cost.LeastSquares(
        cut_vals[ids], np.array(sfs)[ids], np.array(sf_errs)[ids], fit_func
    )
    c.loss = "soft_l1"
    m1 = Minuit(c, np.nanmax(sfs) / 2, 0, 1, 1.5)
    m1.simplex().migrad()
    xs = np.arange(np.nanmin(cut_vals[ids]), np.nanmax(cut_vals[ids]), 0.01)
    p = fit_func(xs, *m1.values)
    cut_val = round(xs[np.argmin(np.abs(p - (100 * 0.9)))], 3)

    if display > 0:
        fig = plt.figure()
        plt.errorbar(
            cut_vals[ids],
            np.array(sfs)[ids],
            yerr=np.array(sf_errs)[ids],
            linestyle=" ",
        )

        plt.plot(xs, p)
        plt.hlines((100 * dep_acc), -8.1, cut_val, color="red", linestyle="--")
        plt.vlines(
            cut_val,
            np.nanmin(np.array(sfs)[ids]) * 0.9,
            (100 * dep_acc),
            color="red",
            linestyle="--",
        )
        plt.xlabel("cut value")
        plt.ylabel("survival percentage")
        plt.xlim([-8.1, 0.1])
        plt.ylim([np.nanmin(np.array(sfs)[ids]) * 0.9, 102])
        plot_dict["cut_determination"] = fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return cut_val, plot_dict
    else:
        return cut_val


def get_sf(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    aoe_cut_val: float,
    eres_pars: list,
    dt_mask: np.array = None,
    display: int = 0,
) -> tuple(np.array, np.array, np.array, float, float):
    """
    Calculates survival fraction for gamma lines using fitting method as in cut determination
    """

    if dt_mask is None:
        dt_mask = np.full(len(aoe), True, dtype=bool)

    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range) & (~np.isnan(aoe))]
        peak_dt_mask = dt_mask[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
    else:
        peak_energy = energy[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range) & (~np.isnan(aoe))]
        peak_dt_mask = dt_mask[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        # if len(peak_aoe)>50000:
        #     rng = np.random.default_rng(10)
        #     rands = rng.choice(len(peak_aoe),50000,replace=False)
        #     rands.sort()
        #     peak_energy = peak_energy[rands]
        #     peak_aoe = peak_aoe[rands]
        #     peak_dt_mask = peak_dt_mask[rands]

    pars, errors = unbinned_energy_fit(peak_energy, peak, eres_pars, simplex=False)
    pc_n = pars["n_sig"]
    pc_err = errors["n_sig"]
    sfs = []
    sf_errs = []

    cut_vals = np.arange(-5, 5, 0.2)
    final_cut_vals = []
    for cut_val in cut_vals:
        try:
            sf, err, cut_pars, surv_pars = get_survival_fraction(
                peak_energy, peak_aoe, cut_val, peak, eres_pars, dt_mask=peak_dt_mask
            )
            if np.isnan(cut_pars).all() == False and np.isnan(surv_pars).all() == False:
                guess_pars_cut = cut_pars
                guess_pars_surv = surv_pars
        except:
            sf = np.nan
            err = np.nan
        sfs.append(sf)
        sf_errs.append(err)
        final_cut_vals.append(cut_val)
    ids = (
        (sf_errs < (5 * np.nanpercentile(sf_errs, 50)))
        & (~np.isnan(sf_errs))
        & (np.array(sfs) < 100)
    )
    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy, peak_aoe, aoe_cut_val, peak, eres_pars, dt_mask=peak_dt_mask
    )

    if display > 0:
        plt.figure()
        plt.errorbar(cut_vals, sfs, sf_errs)
        plt.show()

    return (
        np.array(final_cut_vals)[ids],
        np.array(sfs)[ids],
        np.array(sf_errs)[ids],
        sf,
        sf_err,
    )


def compton_sf(
    energy: np.array,
    aoe: np.array,
    cut: float,
    peak: float,
    eres: list[float, float],
    dt_mask: np.array = None,
    display: int = 1,
) -> tuple(float, np.array, list):
    """
    Determines survival fraction for compton continuum by basic counting
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    sf_errs = []
    ids = (energy > emin) & (energy < emax) & (~np.isnan(aoe))
    aoe = aoe[ids]
    if dt_mask is None:
        dt_mask = np.full(len(aoe), True, dtype=bool)
    else:
        dt_mask = dt_mask[ids]
    cut_vals = np.arange(-5, 5, 0.1)
    for cut_val in cut_vals:
        sfs.append(100 * len(aoe[(aoe > cut_val) & dt_mask]) / len(aoe))
        sf_errs.append(
            sfs[-1]
            * np.sqrt(
                (1 / len(aoe)) + 1 / (len(aoe[(aoe > cut_val) & dt_mask]) + 10**-99)
            )
        )
    sf = 100 * len(aoe[(aoe > cut) & dt_mask]) / len(aoe)
    sf_err = sf * np.sqrt(
        1 / len(aoe) + 1 / (len(aoe[(aoe > cut) & dt_mask]) + 10**-99)
    )
    return cut_vals, sfs, sf_errs, sf, sf_err


def get_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    fit_width: tuple(int, int),
    eres_pars: list,
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    dt_mask: np.array = None,
    display: int = 1,
) -> tuple(float, float):
    """
    Calculates survival fraction for gamma line without sweeping through values
    """

    if dt_mask is None:
        dt_mask = np.full(len(aoe), True, dtype=bool)

    min_range = peak - fit_width[0]
    max_range = peak + fit_width[1]
    if peak == "1592.5":
        peak_energy = energy[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range) & (~np.isnan(aoe))]
    else:
        peak_energy = energy[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        peak_aoe = aoe[(energy > min_range) & (energy < max_range) & (~np.isnan(aoe))]
        peak_dt_mask = dt_mask[
            (energy > min_range) & (energy < max_range) & (~np.isnan(aoe))
        ]
        # if len(peak_aoe)>50000:
        #     rng = np.random.default_rng(10)
        #     rands = rng.choice(len(peak_aoe),50000,replace=False)
        #     rands.sort()
        #     peak_energy = peak_energy[rands]
        #     peak_aoe = peak_aoe[rands]
        #     peak_dt_mask = peak_dt_mask[rands]

    sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
        peak_energy,
        peak_aoe,
        aoe_low_cut_val,
        peak,
        eres_pars,
        high_cut=aoe_high_cut_val,
        dt_mask=peak_dt_mask,
    )
    return sf, sf_err


def compton_sf_no_sweep(
    energy: np.array,
    aoe: np.array,
    peak: float,
    eres: list[float, float],
    aoe_low_cut_val: float,
    aoe_high_cut_val: float = None,
    dt_mask: np.array = None,
    display: int = 1,
) -> float:
    """
    Calculates survival fraction for compton contiuum without sweeping through values
    """

    fwhm = np.sqrt(eres[0] + peak * eres[1])

    emin = peak - 2 * fwhm
    emax = peak + 2 * fwhm
    sfs = []
    ids = (energy > emin) & (energy < emax) & (~np.isnan(aoe))
    aoe = aoe[ids]
    if dt_mask is None:
        dt_mask = np.full(len(aoe), True, dtype=bool)
    else:
        dt_mask = dt_mask[ids]
    if aoe_high_cut_val is None:
        sf = 100 * len(aoe[(aoe > aoe_low_cut_val)]) / len(aoe)
        sf_err = sf * np.sqrt(
            1 / len(aoe) + 1 / len(aoe[(aoe > aoe_low_cut_val) & dt_mask])
        )
    else:
        sf = (
            100
            * len(aoe[(aoe > aoe_low_cut_val) & (aoe < aoe_high_cut_val) & dt_mask])
            / len(aoe)
        )
        sf_err = sf * np.sqrt(
            1 / len(aoe)
            + 1 / len(aoe[(aoe > aoe_low_cut_val) & (aoe < aoe_high_cut_val) & dt_mask])
        )
    return sf, sf_err


def apply_dtcorr(aoe: np.array, dt: np.array, alpha: float) -> np.array:
    """Aligns dt regions"""
    return aoe * (1 + alpha * dt)


def drift_time_correction(
    aoe: np.array,
    energy: np.array,
    dt: np.array,
    pdf=standard_aoe,
    display: int = 0,
    plot_dict: dict = {},
) -> tuple(np.array, float):
    """
    Calculates the correction needed to align the two drift time regions for ICPC detectors
    """
    hist, bins, var = pgh.get_hist(
        aoe[(energy > 1582) & (energy < 1602) & (~np.isnan(energy)) & (~np.isnan(aoe))],
        bins=500,
    )
    bin_cs = (bins[1:] + bins[:-1]) / 2
    mu = bin_cs[np.argmax(hist)]
    aoe_range = [mu * 0.9, mu * 1.1]

    idxs = (
        (energy > 1582)
        & (energy < 1602)
        & (aoe > aoe_range[0])
        & (aoe < aoe_range[1])
        & (dt > np.nanpercentile(dt, 1))
        & (dt < np.nanpercentile(dt, 99))
        & (~np.isnan(dt))
        & (~np.isnan(aoe))
        & (~np.isnan(energy))
    )

    hist, bins, var = pgh.get_hist(
        dt[idxs], dx=10, range=(np.nanmin(dt[idxs]), np.nanmax(dt[idxs]))
    )
    dt_distrib = drift_time_distribution()

    gpars = dt_distrib.guess(hist, bins, var)
    cost_func = cost.ExtendedUnbinnedNLL(dt[idxs], dt_distrib.extended_pdf)
    m = Minuit(cost_func, **gpars)
    m.limits = dt_distrib.bounds(gpars)
    m.fixed = dt_distrib.fixed()
    m.simplex().migrad()
    m.hesse()

    aoe_mask = (
        (idxs)
        & (dt > m.values["mu1"] - 2 * m.values["sigma1"])
        & (dt < m.values["mu1"] + 2 * m.values["sigma1"])
    )
    aoe_pars, aoe_errs, _ = unbinned_aoe_fit(aoe[aoe_mask], pdf=pdf, display=display)

    aoe_mask2 = (
        (idxs)
        & (dt > m.values["mu2"] - 2 * m.values["sigma2"])
        & (dt < m.values["mu2"] + 2 * m.values["sigma2"])
    )

    aoe_pars2, aoe_errs2, _ = unbinned_aoe_fit(aoe[aoe_mask2], pdf=pdf, display=display)

    try:
        alpha = (aoe_pars["mu"] - aoe_pars2["mu"]) / (
            (m.values["mu2"] * aoe_pars2["mu"]) - (m.values["mu1"] * aoe_pars["mu"])
        )
    except ZeroDivisionError:
        alpha = 0
    aoe_corrected = apply_dtcorr(aoe, dt, alpha)

    if display > 0:
        aoe_pdf = pdf()
        dt_fig = plt.figure()

        plt.subplot(2, 2, 1)
        xs = np.linspace(aoe_pars["lower_range"], aoe_pars["upper_range"], 1000)
        counts, aoe_bins, bars = plt.hist(
            aoe[
                (aoe < aoe_pars["upper_range"])
                & (aoe > aoe_pars["lower_range"])
                & aoe_mask
            ],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins)
        plt.plot(xs, aoe_pdf.pdf(xs, *aoe_pars) * dx[0], label="Full fit")
        sig, bkg = aoe_pdf.pdf(xs, *aoe_pars[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(2, 2, 2)
        xs = np.linspace(aoe_pars2["lower_range"], aoe_pars2["upper_range"], 1000)
        counts, aoe_bins2, bars = plt.hist(
            aoe[
                (aoe < aoe_pars2["upper_range"])
                & (aoe > aoe_pars2["lower_range"])
                & aoe_mask2
            ],
            bins=400,
            histtype="step",
            label="Data",
        )
        dx = np.diff(aoe_bins2)
        plt.plot(xs, aoe_pdf.pdf(xs, *aoe_pars2) * dx[0], label="Full fit")
        sig, bkg = aoe_pdf.pdf(xs, *aoe_pars2[:-1], True)
        plt.plot(xs, sig * dx[0], label="Peak fit")
        plt.plot(xs, bkg * dx[0], label="Bkg fit")
        plt.legend(loc="upper left")
        plt.xlabel("A/E")
        plt.ylabel("Counts")

        plt.subplot(2, 2, 3)
        plt.step(pgh.get_bin_centers(bins), hist, label="Data")
        plt.plot(
            pgh.get_bin_centers(bins),
            dt_distrib.pdf(pgh.get_bin_centers(bins), **gpars) * np.diff(bins)[0],
            label="Guess",
        )
        plt.plot(
            pgh.get_bin_centers(bins),
            dt_distrib.pdf(pgh.get_bin_centers(bins), *m.values) * np.diff(bins)[0],
            label="Fit",
        )
        plt.xlabel("Drift Time (ns)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 4)
        bins = np.linspace(
            np.nanpercentile(aoe[idxs], 1),
            np.nanpercentile(aoe_corrected[idxs], 99),
            200,
        )
        plt.hist(aoe[idxs], bins=bins, histtype="step", label="Uncorrected")
        plt.hist(aoe_corrected[idxs], bins=bins, histtype="step", label="Corrected")
        plt.xlabel("A/E")
        plt.ylabel("Counts")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.xlim(
            np.nanpercentile(aoe[idxs], 1), np.nanpercentile(aoe_corrected[idxs], 99)
        )

        plot_dict["dt_corr"] = dt_fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return alpha, plot_dict
    else:
        return alpha


def cal_aoe(
    files: list,
    lh5_path,
    cal_dict: dict,
    current_param: str,
    energy_param: str,
    cal_energy_param: str,
    eres_pars: list,
    pdf=standard_aoe,
    cut_field: str = "is_valid_cal",
    dt_corr: bool = False,
    dep_correct: bool = False,
    dt_cut: dict = None,
    aoe_high_cut: int = 4,
    sigma_func=sigma_fit,
    display: int = 0,
) -> tuple(dict, dict):
    """
    Main function for running the a/e correction and cut determination.

    dt_cut: dictionary should contain two fields "cut" containing a dictionary of the form required by the hit_config and
            hard specifying whether this is a hard cut so these events should be removed (e.g. tail to high A/E) or soft cut
            where these events are just not used for the A/E fits and cut determination (e.g. tail to low A/E)
    """
    params = [
        current_param,
        "tp_0_est",
        "tp_99",
        "dt_eff",
        energy_param,
        cal_energy_param,
        cut_field,
    ]
    if dt_cut is not None:
        if re.match(r"(\d{8})T(\d{6})Z", list(cal_dict)[0]):
            for tstamp in cal_dict:
                cal_dict[tstamp].update(dt_cut["cut"])
        else:
            cal_dict.update(dt_cut["cut"])
        dt_cut_field = list(dt_cut["cut"])[0]
        params.append(dt_cut_field)
    else:
        dt_cut_field = None
    df = load_aoe(
        files,
        lh5_path,
        cal_dict,
        params,
        energy_param=energy_param,
        current_param=current_param,
    )
    if dt_cut is not None:
        df["dt_cut"] = df[list(dt_cut["cut"])[0]]
    else:
        df["dt_cut"] = np.full(len(df), True, dtype=bool)

    df["is_usable_fits"] = df[cut_field] & df["is_not_pulser"] & df["dt_cut"]
    try:
        df, timecorr_dict, res_dict = aoe_timecorr(
            df, energy_param, current_param, pdf=pdf
        )
        log.info("Finished A/E time correction")
    except:
        log.info("A/E time correction failed")
        res_dict = {}
        timecorr_dict = {
            "AoE_Timecorr": {
                "expression": f"({current_param}/{energy_param})/a",
                "parameters": {"a": np.nan},
            }
        }

    if re.match(r"(\d{8})T(\d{6})Z", list(cal_dict)[0]):
        for tstamp in cal_dict:
            if tstamp in timecorr_dict:
                cal_dict[tstamp].update(timecorr_dict[tstamp])
            else:
                cal_dict[tstamp].update(timecorr_dict)
    else:
        cal_dict.update(timecorr_dict)

    if dt_corr == True:
        aoe_param = "AoE_dtcorr"
        try:
            if np.isnan(df.query("is_usable_fits")["AoE_timecorr"]).all():
                raise ValueError
            alpha = drift_time_correction(
                df.query("is_usable_fits")["AoE_timecorr"],
                df.query("is_usable_fits")[cal_energy_param],
                df.query("is_usable_fits")["dt_eff"],
                pdf=pdf,
            )
            df["AoE_dtcorr"] = apply_dtcorr(df["AoE_timecorr"], df["dt_eff"], alpha)
            log.info(f"dtcorr successful alpha:{alpha}")
        except:
            log.error("A/E dtcorr failed")
            alpha = np.nan
    else:
        aoe_param = "AoE_timecorr"

    try:
        log.info("Starting A/E energy correction")
        mu_pars, sigma_pars, results_dict, dep_pars = AoEcorrection(
            df.query("is_usable_fits")[cal_energy_param],
            df.query("is_usable_fits")[aoe_param],
            eres_pars,
            pdf=pdf,
            sigma_func=sigma_func,
        )
        dep_mu = dep_pars["mu"]
        log.info("Finished A/E energy correction")
        df["AoE_corrected"] = df[aoe_param] / pol1(df[cal_energy_param], *mu_pars)
        df["AoE_classifier"] = (df["AoE_corrected"] - 1) / sigma_func(
            df[cal_energy_param], *sigma_pars
        )
    except:
        log.error("A/E energy correction failed")
        args = pol1.__code__.co_varnames[: pol1.__code__.co_argcount][1:]
        c = cost.UnbinnedNLL(np.array([0]), pol1)
        m = Minuit(c, *[np.nan for arg in args])
        mu_pars = m.values
        args = sigma_func.__code__.co_varnames[: sigma_func.__code__.co_argcount][1:]
        c = cost.UnbinnedNLL(np.array([0]), sigma_func)
        m = Minuit(c, *[np.nan for arg in args])
        sigma_pars = m.values
        dep_mu = np.nan
        results_dict = {}

    try:
        if dep_correct is True:
            cut = get_aoe_cut_fit(
                df.query("is_usable_fits")[cal_energy_param],
                df.query("is_usable_fits")["AoE_corrected"],
                1592,
                (40, 20),
                0.9,
                eres_pars,
                dep_correct=True,
                dep_mu=lambda x: dep_mu / pol1(1592.5, *mu_pars),
                sig_func=lambda x: sigma_func(x, *sig_pars),
                display=0,
            )
        else:
            cut = get_aoe_cut_fit(
                df.query("is_usable_fits")[cal_energy_param],
                df.query("is_usable_fits")["AoE_classifier"],
                1592,
                (40, 20),
                0.9,
                eres_pars,
                display=0,
            )

        log.info(f"Cut found at {cut}")
    except:
        log.error("A/E cut determination failed")
        cut = np.nan

    aoe_cal_dict = {}
    if dt_corr == False:
        aoe_uncorr_param = "AoE_Timecorr"
    else:
        aoe_cal_dict.update(
            {
                "AoE_DTcorr": {
                    "expression": f"AoE_Timecorr*(1+a*dt_eff)",
                    "parameters": {"a": alpha},
                }
            }
        )
        aoe_uncorr_param = "AoE_DTcorr"

    aoe_cal_dict.update(
        {
            "AoE_Corrected": {
                "expression": f"((({aoe_uncorr_param})/(a*{cal_energy_param} +b))-1)",
                "parameters": mu_pars.to_dict(),
            }
        }
    )
    if sigma_func == sigma_fit:
        aoe_cal_dict.update(
            {
                "AoE_Classifier": {
                    "expression": f"AoE_Corrected/(sqrt(a+(b/{cal_energy_param})**c))",
                    "parameters": sigma_pars.to_dict(),
                }
            }
        )
    else:
        raise ValueError("Unknown sigma func")

    if dt_cut is not None:
        if dt_cut["hard"] is True:
            aoe_cal_dict.update(
                {
                    "AoE_Low_Cut": {
                        "expression": f"(AoE_Classifier>a) & ({list(dt_cut['cut'])[0]})",
                        "parameters": {"a": cut},
                    },
                    "AoE_Double_Sided_Cut": {
                        "expression": "(a>AoE_Classifier) & (AoE_Low_Cut)",
                        "parameters": {"a": aoe_high_cut},
                    },
                }
            )
        else:
            aoe_cal_dict.update(
                {
                    "AoE_Low_Cut": {
                        "expression": "AoE_Classifier>a",
                        "parameters": {"a": cut},
                    },
                    "AoE_Double_Sided_Cut": {
                        "expression": "(a>AoE_Classifier) & (AoE_Low_Cut)",
                        "parameters": {"a": aoe_high_cut},
                    },
                }
            )
    else:
        aoe_cal_dict.update(
            {
                "AoE_Low_Cut": {
                    "expression": "AoE_Classifier>a",
                    "parameters": {"a": cut},
                },
                "AoE_Double_Sided_Cut": {
                    "expression": "(a>AoE_Classifier)&(AoE_Low_Cut)",
                    "parameters": {"a": aoe_high_cut},
                },
            }
        )

    if re.match(r"(\d{8})T(\d{6})Z", list(cal_dict)[0]):
        for tstamp in cal_dict:
            cal_dict[tstamp].update(aoe_cal_dict)
    else:
        cal_dict.update(aoe_cal_dict)

    try:
        log.info("  Compute low side survival fractions: ")

        peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        sf = np.zeros(len(peaks_of_interest))
        sferr = np.zeros(len(peaks_of_interest))
        fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]
        full_sfs = []
        full_sf_errs = []
        full_cut_vals = []

        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                cut_vals, sfs, sf_errs, sf[i], sferr[i] = compton_sf(
                    df.query(f"{cut_field}& is_not_pulser")[
                        cal_energy_param
                    ].to_numpy(),
                    df.query(f"{cut_field}& is_not_pulser")[
                        "AoE_classifier"
                    ].to_numpy(),
                    cut,
                    peak,
                    eres_pars,
                    dt_mask=df.query(f"{cut_field}& is_not_pulser")[
                        "dt_cut"
                    ].to_numpy(),
                )

                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(sf_errs)
            else:
                cut_vals, sfs, sf_errs, sf[i], sferr[i] = get_sf(
                    df.query(f"{cut_field}& is_not_pulser")[
                        cal_energy_param
                    ].to_numpy(),
                    df.query(f"{cut_field}& is_not_pulser")[
                        "AoE_classifier"
                    ].to_numpy(),
                    peak,
                    fit_widths[i],
                    cut,
                    eres_pars,
                    dt_mask=df.query(f"{cut_field}& is_not_pulser")[
                        "dt_cut"
                    ].to_numpy(),
                )
                full_cut_vals.append(cut_vals)
                full_sfs.append(sfs)
                full_sf_errs.append(sf_errs)

            log.info(f"{peak}keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %")

        sf_2side = np.zeros(len(peaks_of_interest))
        sferr_2side = np.zeros(len(peaks_of_interest))
        log.info("Calculating 2 sided cut sfs")
        for i, peak in enumerate(peaks_of_interest):
            if peak == 2039:
                sf_2side[i], sferr_2side[i] = compton_sf_no_sweep(
                    df.query(f"{cut_field}& is_not_pulser")[
                        cal_energy_param
                    ].to_numpy(),
                    df.query(f"{cut_field}& is_not_pulser")[
                        "AoE_classifier"
                    ].to_numpy(),
                    peak,
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                    dt_mask=df.query(f"{cut_field}& is_not_pulser")[
                        "dt_cut"
                    ].to_numpy(),
                )
            else:
                sf_2side[i], sferr_2side[i] = get_sf_no_sweep(
                    df.query(f"{cut_field}& is_not_pulser")[
                        cal_energy_param
                    ].to_numpy(),
                    df.query(f"{cut_field}& is_not_pulser")[
                        "AoE_classifier"
                    ].to_numpy(),
                    peak,
                    fit_widths[i],
                    eres_pars,
                    cut,
                    aoe_high_cut_val=aoe_high_cut,
                    dt_mask=df.query(f"{cut_field}& is_not_pulser")[
                        "dt_cut"
                    ].to_numpy(),
                )

            log.info(f"{peak}keV: {sf_2side[i]:2.1f} +/- {sferr_2side[i]:2.1f} %")

        def convert_sfs_to_dict(peaks_of_interest, sfs, sf_errs):
            out_dict = {}
            for i, peak in enumerate(peaks_of_interest):
                out_dict[str(peak)] = {
                    "sf": f"{sfs[i]:2f}",
                    "sf_err": f"{sf_errs[i]:2f}",
                }
            return out_dict

        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": energy_param,
            "Cal_energy_param": cal_energy_param,
            "dt_param": "dt_eff",
            "rt_correction": dt_corr,
            "1000-1300keV": res_dict,
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
            "Low_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf, sferr),
            "2_side_sfs": convert_sfs_to_dict(peaks_of_interest, sf_2side, sferr_2side),
        }
        log.info("Done")
        log.info(f"Results are {out_dict}")

    except:
        log.error("A/E Survival fraction determination failed")
        out_dict = {
            "correction_fit_results": results_dict,
            "A/E_Energy_param": energy_param,
            "Cal_energy_param": cal_energy_param,
            "dt_param": "dt_eff",
            "rt_correction": False,
            "1000-1300keV_mean": res_dict,
            "Mean_pars": list(mu_pars),
            "Sigma_pars": list(sigma_pars),
            "Low_cut": cut,
            "High_cut": aoe_high_cut,
        }
    if display <= 0:
        return cal_dict, out_dict
    else:
        plot_dict = {}
        try:
            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 16

            fig1 = plt.figure()
            plt.subplot(3, 2, 1)
            plot_dt_dep(
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                [1582, 1602],
                f"Tl DEP",
            )
            plt.subplot(3, 2, 2)
            plot_dt_dep(
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                [1510, 1630],
                f"Bi FEP",
            )
            plt.subplot(3, 2, 3)
            plot_dt_dep(
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                [2030, 2050],
                "Qbb",
            )
            plt.subplot(3, 2, 4)
            plot_dt_dep(
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                [2080, 2120],
                f"Tl SEP",
            )
            plt.subplot(3, 2, 5)
            plot_dt_dep(
                df.query("is_valid_cal& is_not_pulser")["AoE_timecorr"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                [2584, 2638],
                f"Tl FEP",
            )
            plt.tight_layout()
            plot_dict["dt_deps"] = fig1
            if display > 1:
                plt.show()
            else:
                plt.close()

            if dt_corr == True:
                alpha, plot_dict = drift_time_correction(
                    df.query("is_usable_fits")["AoE_timecorr"],
                    df.query("is_usable_fits")[cal_energy_param],
                    df.query("is_usable_fits")["dt_eff"],
                    display=display,
                    plot_dict=plot_dict,
                )

                fig_dt = plt.figure()
                plt.subplot(3, 2, 1)
                plot_dt_dep(
                    df.query("is_valid_cal& is_not_pulser")["AoE_dtcorr"],
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                    df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                    [1582, 1602],
                    f"Tl DEP",
                )
                plt.subplot(3, 2, 2)
                plot_dt_dep(
                    df.query("is_valid_cal& is_not_pulser")["AoE_dtcorr"],
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                    df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                    [1510, 1630],
                    f"Bi FEP",
                )
                plt.subplot(3, 2, 3)
                plot_dt_dep(
                    df.query("is_valid_cal& is_not_pulser")["AoE_dtcorr"],
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                    df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                    [2030, 2050],
                    "Qbb",
                )
                plt.subplot(3, 2, 4)
                plot_dt_dep(
                    df.query("is_valid_cal& is_not_pulser")["AoE_dtcorr"],
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                    df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                    [2080, 2120],
                    f"Tl SEP",
                )
                plt.subplot(3, 2, 5)
                plot_dt_dep(
                    df.query("is_valid_cal& is_not_pulser")["AoE_dtcorr"],
                    df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                    df.query("is_valid_cal& is_not_pulser")["dt_eff"],
                    [2584, 2638],
                    f"Tl FEP",
                )
                plt.tight_layout()
                plot_dict["dt_deps_dtcorr"] = fig_dt
                if display > 1:
                    plt.show()
                else:
                    plt.close()

            fig2 = plt.figure()
            plot_compt_bands_overlayed(
                df.query("is_valid_cal& is_not_pulser")[aoe_param],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                [950, 1250, 1460, 1660, 1860, 2060, 2270],
            )
            plt.ylabel("Counts")
            plt.xlabel("Raw A/E")
            plt.title(f"Compton Bands before Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_nocorr"] = fig2
            if display > 1:
                plt.show()
            else:
                plt.close()

            _, _, _, plot_dict = aoe_timecorr(
                df,
                energy_param,
                current_param,
                pdf=pdf,
                plot_dict=plot_dict,
                display=display,
            )

            _, _, _, _, plot_dict = AoEcorrection(
                df.query("is_usable_fits")[cal_energy_param],
                df.query("is_usable_fits")[aoe_param],
                eres_pars,
                pdf=pdf,
                sigma_func=sigma_func,
                plot_dict=plot_dict,
                display=display,
            )

            fig3 = plt.figure()
            plot_compt_bands_overlayed(
                df.query("is_valid_cal& is_not_pulser")["AoE_classifier"],
                df.query("is_valid_cal& is_not_pulser")[cal_energy_param],
                [950, 1250, 1460, 1660, 1860, 2060, 2270],
                [-5, 5],
            )
            plt.ylabel("Counts")
            plt.xlabel("Corrected A/E")
            plt.title(f"Compton Bands after Correction")
            plt.legend(loc="upper left")
            plot_dict["compt_bands_corr"] = fig3
            if display > 1:
                plt.show()
            else:
                plt.close()

            if dep_correct is True:
                _, plot_dict = get_aoe_cut_fit(
                    df.query("is_usable_fits")[cal_energy_param],
                    df.query("is_usable_fits")["AoE_corrected"],
                    1592,
                    (40, 20),
                    0.9,
                    eres_pars,
                    dep_correct=True,
                    dep_mu=lambda x: dep_mu / pol1(1592.5, *mu_pars),
                    sig_func=lambda x: sigma_func(x, *sig_pars),
                    display=display,
                    plot_dict=plot_dict,
                )
            else:
                _, plot_dict = get_aoe_cut_fit(
                    df.query("is_usable_fits")[cal_energy_param],
                    df.query("is_usable_fits")["AoE_classifier"],
                    1592,
                    (40, 20),
                    0.9,
                    eres_pars,
                    display=display,
                    plot_dict=plot_dict,
                )

            fig4 = plt.figure()
            plt.vlines(cut, 0, 100, label=f"Cut Value: {cut:1.2f}", color="black")

            for i, peak in enumerate(peaks_of_interest):
                plt.errorbar(
                    full_cut_vals[i],
                    full_sfs[i],
                    yerr=full_sf_errs[i],
                    label=f"{get_peak_label(peak)} {peak} keV: {sf[i]:2.1f} +/- {sferr[i]:2.1f} %",
                )

            handles, labels = plt.gca().get_legend_handles_labels()
            # order = [1, 2, 3, 0, 4, 5]
            plt.legend(
                # [handles[idx] for idx in order],
                # [labels[idx] for idx in order],
                loc="upper right",
            )
            plt.xlabel("Cut Value")
            plt.ylabel("Survival Fraction %")
            plt.ylim([0, 105])
            plot_dict["surv_fracs"] = fig4
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig5, ax = plt.subplots()
            bins = np.arange(900, 3000, 1)
            ax.hist(
                df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Before PSD",
            )
            ax.hist(
                df.query(f"is_usable_fits & AoE_classifier > {cut}")[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Low side PSD cut",
            )
            ax.hist(
                df.query(
                    f"is_usable_fits & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}"
                )[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Double sided PSD cut",
            )
            ax.hist(
                df.query(
                    f"is_valid_cal& is_not_pulser & (AoE_classifier < {cut} | AoE_classifier > {aoe_high_cut} | (~is_usable_fits))"
                )[cal_energy_param],
                bins=bins,
                histtype="step",
                label="Rejected by PSD cut",
            )

            axins = ax.inset_axes([0.25, 0.07, 0.4, 0.3])
            bins = np.linspace(1580, 1640, 200)
            axins.hist(
                df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                df.query(f"is_usable_fits & AoE_classifier > {cut}")[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                df.query(
                    f"is_usable_fits & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}"
                )[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                df.query(
                    f"is_valid_cal& is_not_pulser & (AoE_classifier < {cut} | AoE_classifier > {aoe_high_cut}| (~is_usable_fits))"
                )[cal_energy_param],
                bins=bins,
                histtype="step",
            )
            ax.set_xlim([900, 3000])
            ax.set_yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper left")
            plot_dict["PSD_spectrum"] = fig5
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig6 = plt.figure()
            bins = np.arange(900, 3000, 3)
            counts_pass, bins_pass, _ = pgh.get_hist(
                df.query(
                    f"is_usable_fits & AoE_classifier > {cut} & AoE_classifier < {aoe_high_cut}"
                )[cal_energy_param],
                bins=bins,
            )
            counts, bins, _ = pgh.get_hist(
                df.query(f"is_valid_cal& is_not_pulser")[cal_energy_param], bins=bins
            )
            survival_fracs = counts_pass / (counts + 10**-99)

            plt.step(pgh.get_bin_centers(bins_pass), survival_fracs)
            plt.xlabel("Energy (keV)")
            plt.ylabel("Survival Fraction")
            plt.ylim([0, 1])
            plot_dict["psd_sf"] = fig6
            if display > 1:
                plt.show()
            else:
                plt.close()

            return cal_dict, out_dict, plot_dict
        except:
            return cal_dict, out_dict, plot_dict
