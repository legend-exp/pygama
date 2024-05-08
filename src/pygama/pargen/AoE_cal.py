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

import pygama.math.binned_fitting as pgf
import pygama.math.histogram as pgh
import pygama.pargen.energy_cal as pgc
from pygama.math.distributions import (
    exgauss,
    gauss_on_exgauss,
    gauss_on_step,
    gaussian,
    hpge_peak,
    nb_erfc,
)
from pygama.math.functions.gauss import nb_gauss_amp
from pygama.math.functions.hpge_peak import hpge_get_fwfm, hpge_get_fwhm, hpge_get_mode
from pygama.math.functions.sum_dists import SumDists
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
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mu = bin_centers[np.argmax(hist)]
    try:
        _, sigma, _ = pgh.get_gaussian_guess(hist, bins)
    except Exception:
        pars, cov = pgf.gauss_mode_width_max(hist, bins, var, mode_guess=mu, n_bins=20)
        _, sigma, _ = pars
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

    return convert_to_minuit(guess_dict, func).values


def aoe_peak_bounds(func, guess, **kwargs):
    if func == aoe_peak:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "n_sig": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, None),
            "n_bkg": (0, None),
            "tau": (0, None),
        }
    elif func == aoe_peak_with_high_tail:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "n_sig": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, None),
            "htail": (0, 1),
            "tau_sig": (None, 0),
            "n_bkg": (0, None),
            "tau": (0, None),
        }
    elif func == exgauss:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "area": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, None),
            "tau": (0, None),
        }
    elif func == gaussian:
        bounds_dict = {
            "x_lo": (None, None),
            "x_hi": (None, None),
            "area": (0, None),
            "mu": (guess["x_lo"], guess["x_hi"]),
            "sigma": (0, None),
        }

    for item, value in kwargs.items():
        bounds_dict[item] = value
    return bounds_dict


def aoe_peak_fixed(func, **kwargs):
    if func == aoe_peak:
        fixed = ["x_lo", "x_hi"]
    elif func == aoe_peak_with_high_tail:
        fixed = ["x_lo", "x_hi"]
    elif func == exgauss:
        fixed = ["x_lo", "x_hi"]
    elif func == gaussian:
        fixed = ["x_lo", "x_hi"]
    mask = ~np.in1d(func.required_args(), fixed)
    return fixed, mask


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
    pdf=aoe_peak,
    display: int = 0,
) -> tuple(np.array, np.array):
    """
    Fitting function for A/E, first fits just a Gaussian before using the full pdf to fit
    if fails will return NaN values

    Parameters
    ----------
    aoe: np.array
        A/E values
    pdf: PDF
        PDF to fit to
    display: int
        Level of display

    Returns
    -------
    tuple(np.array, np.array)
        Tuple of fit values and errors
    """
    if not isinstance(aoe, np.ndarray):
        aoe = np.array(aoe)

    bin_width = (
        2
        * (np.nanpercentile(aoe, 75) - np.nanpercentile(aoe, 25))
        * len(aoe) ** (-1 / 3)
    )
    nbins = int(np.ceil((np.nanmax(aoe) - np.nanmin(aoe)) / bin_width))
    hist, bins, var = pgh.get_hist(aoe, bins=500)

    gpars = aoe_peak_guess(gaussian, hist, bins, var)
    c1_min = gpars["mu"] - 2 * gpars["sigma"]
    c1_max = gpars["mu"] + 3 * gpars["sigma"]

    # Initial fit just using Gaussian
    c1 = cost.ExtendedUnbinnedNLL(
        aoe[(aoe < c1_max) & (aoe > c1_min)], gaussian.pdf_ext
    )

    m1 = Minuit(c1, *gpars)

    bounds = aoe_peak_bounds(gaussian, gpars)
    for arg, val in bounds.items():
        m1.limits[arg] = val
    for fix in aoe_peak_fixed(gaussian)[0]:
        m1.fixed[fix] = True
    m1.migrad()

    # Range to fit over, below this tail behaviour more exponential, few events above
    fmin = m1.values["mu"] - 15 * m1.values["sigma"]
    if fmin < np.nanmin(aoe):
        fmin = np.nanmin(aoe)
    fmax_bkg = m1.values["mu"] - 5 * m1.values["sigma"]
    fmax = m1.values["mu"] + 5 * m1.values["sigma"]

    n_bkg_guess = len(aoe[(aoe < fmax) & (aoe > fmin)]) - m1.values["area"]

    bkg_guess = aoe_peak_guess(
        exgauss,
        hist,
        bins,
        var,
        area=n_bkg_guess,
        mu=m1.values["mu"],
        sigma=m1.values["sigma"],
        x_lo=fmin,
        x_hi=fmax_bkg,
    )

    c2 = cost.ExtendedUnbinnedNLL(aoe[(aoe < fmax_bkg) & (aoe > fmin)], exgauss.pdf_ext)
    m2 = Minuit(c2, *bkg_guess)

    bounds = aoe_peak_bounds(exgauss, bkg_guess)

    for arg, val in bounds.items():
        m2.limits[arg] = val
    for fix in aoe_peak_fixed(exgauss)[0]:
        m2.fixed[fix] = True
    m2.simplex().migrad()
    m2.hesse()

    x0 = aoe_peak_guess(
        pdf,
        hist,
        bins,
        var,
        n_sig=m1.values["area"],
        mu=m1.values["mu"],
        sigma=m1.values["sigma"],
        n_bkg=m2.values["area"],
        tau=m2.values["tau"],
        x_lo=fmin,
        x_hi=fmax,
    )

    bounds = aoe_peak_bounds(pdf, x0)

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

    if np.isnan(m.errors).all():
        try:
            m.simplex.migrad()
            m.hesse()
        except Exception:
            return return_nans(pdf)

    if display > 1:
        aoe = aoe[(aoe < fmax) & (aoe > fmin)]
        bin_width = (
            2
            * (np.nanpercentile(aoe, 75) - np.nanpercentile(aoe, 25))
            * len(aoe) ** (-1 / 3)
        )
        nbins = int(np.ceil((np.nanmax(aoe) - np.nanmin(aoe)) / bin_width))  # *5

        plt.figure()
        xs = np.linspace(fmin, fmax, 1000)
        counts, bins, bars = plt.hist(aoe, bins=nbins, histtype="step", label="Data")
        dx = np.diff(bins)
        plt.plot(xs, pdf.get_pdf(xs, *m.values) * dx[0], label="Full fit")
        pdf.components = True
        sig, bkg = pdf.get_pdf(xs, *m.values)
        pdf.components = False
        plt.plot(xs, sig * dx[0], label="Signal")
        plt.plot(xs, bkg * dx[0], label="Background")
        plt.plot(
            xs, gaussian.pdf_ext(xs, *m1.values)[1] * dx[0], label="Initial Gaussian"
        )
        plt.plot(xs, exgauss.pdf_ext(xs, *m2.values)[1] * dx[0], label="Bkg guess")
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


def fit_time_means(tstamps, means, sigmas):
    """
    Fit the time dependence of the means of the A/E distribution

    Args:

    tstamps: np.array
        Timestamps of the data
    means: np.array
        Means of the A/E distribution
    sigmas: np.array
        Sigmas of the A/E distribution

    Returns: dict
        Dictionary of the time dependence of the means
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
            else:
                if (np.abs(means[i + 1] - means[i]) < 0.4 * sigmas[i + 1]) and not (
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


def energy_guess(energy, func_i, fit_range=None, bin_width=1, peak=None, eres=None):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    if func_i == hpge_peak or func_i == gauss_on_step:
        parguess = pgc.get_hpge_energy_peak_par_guess(
            energy, func_i, fit_range=fit_range
        )

        if peak is not None:
            parguess["mu"] = peak

        if eres is not None:
            parguess["sigma"] = eres / 2.355

        for i, guess in enumerate(parguess):
            if np.isnan(guess):
                parguess[i] = 0

    else:
        log.error(f"energy_guess not implemented for {func_i}")
        return None
    return parguess


def fix_all_but_nevents(func):
    """
    Returns: Sequence list of fixed indexes for fitting and mask for parameters
    """

    if func == gauss_on_step:
        # pars are: n_sig, mu, sigma, n_bkg, hstep, lower, upper, components
        fixed = ["x_lo", "x_hi", "mu", "sigma", "hstep"]

    elif func == hpge_peak:
        # pars are: , components
        fixed = ["x_lo", "x_hi", "mu", "sigma", "htail", "tau", "hstep"]

    else:
        log.error(f"get_hpge_E_fixed not implemented for {func}")
        return None, None
    mask = ~np.in1d(func.required_args(), fixed)
    return fixed, mask


def get_bounds(func, parguess):
    if func == hpge_peak or func == gauss_on_step:
        bounds = pgc.get_hpge_energy_bounds(func, parguess)

        bounds["mu"] = (parguess["mu"] - 1, parguess["mu"] + 1)
        bounds["n_sig"] = (0, 2 * (parguess["n_sig"] + parguess["n_bkg"]))
        bounds["n_bkg"] = (0, 2 * (parguess["n_sig"] + parguess["n_bkg"]))

    else:
        log.error(f"get_bounds not implemented for {func}")
        return None
    return bounds


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
    if func == gauss_on_step:
        total_events = len(energies)
        parguess["n_sig"] = len(
            energies[
                (energies > parguess["mu"] - 2 * parguess["sigma"])
                & (energies < parguess["mu"] + 2 * parguess["sigma"])
            ]
        )
        parguess["n_bkg"] = total_events - parguess["n_sig"]
        return parguess

    if func == hpge_peak:
        total_events = len(energies)
        parguess["n_sig"] = len(
            energies[
                (energies > parguess["mu"] - 2 * parguess["sigma"])
                & (energies < parguess["mu"] + 2 * parguess["sigma"])
            ]
        )
        parguess["n_bkg"] = total_events - parguess["n_sig"]
        return parguess

    else:
        log.error(f"update_guess not implemented for {func}")
        return parguess


def get_survival_fraction(
    energy,
    cut_param,
    cut_val,
    peak,
    eres_pars,
    fit_range=None,
    high_cut=None,
    guess_pars_cut=None,
    guess_pars_surv=None,
    dt_mask=None,
    mode="greater",
    func=hpge_peak,
    display=0,
):
    if dt_mask is None:
        dt_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))

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
        (pars, errs, cov, _, func, _, _, _) = pgc.unbinned_staged_energy_fit(
            energy,
            func,
            guess_func=energy_guess,
            bounds_func=get_bounds,
            guess_kwargs={"peak": peak, "eres": eres_pars},
            fit_range=fit_range,
        )

        guess_pars_cut = pars
        guess_pars_surv = pars
    # add update guess here for n_sig and n_bkg
    guess_pars_cut = update_guess(func, guess_pars_cut, energy[(~nan_idxs) & (~idxs)])
    (cut_pars, cut_errs, cut_cov, _, _, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy[(~nan_idxs) & (~idxs)],
        func,
        guess=guess_pars_cut,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        fixed_func=fix_all_but_nevents,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        lock_guess=True,
        allow_tail_drop=False,
        fit_range=fit_range,
    )
    guess_pars_surv = update_guess(func, guess_pars_cut, energy[(~nan_idxs) & (idxs)])
    (surv_pars, surv_errs, surv_cov, _, _, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy[(~nan_idxs) & (idxs)],
        func,
        guess=guess_pars_surv,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        fixed_func=fix_all_but_nevents,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        lock_guess=True,
        allow_tail_drop=False,
        fit_range=fit_range,
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
    fit_range=None,
    debug_mode=False,
) -> tuple(pd.DataFrame, float, float):
    """
    Calculates survival fraction for gamma lines using fitting method as in cut determination
    """

    if dt_mask is None:
        dt_mask = np.full(len(cut_param), True, dtype=bool)

    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame()

    (pars, _, _, _, func, _, _, _) = pgc.unbinned_staged_energy_fit(
        energy,
        hpge_peak,
        guess_func=energy_guess,
        bounds_func=get_bounds,
        guess_kwargs={"peak": peak, "eres": eres_pars},
        fit_range=fit_range,
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
                fit_range=fit_range,
                dt_mask=dt_mask,
                mode=mode,
                guess_pars_cut=guess_pars_cut,
                guess_pars_surv=guess_pars_surv,
                func=func,
            )
            out_df = pd.concat(
                [out_df, pd.DataFrame([{"cut_val": cut_val, "sf": sf, "sf_err": err}])]
            )
        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif debug_mode:
                raise (e)
    out_df.set_index("cut_val", inplace=True)
    if final_cut_value is not None:
        sf, sf_err, cut_pars, surv_pars = get_survival_fraction(
            energy,
            cut_param,
            final_cut_value,
            peak,
            eres_pars,
            fit_range=fit_range,
            dt_mask=dt_mask,
            mode=mode,
            guess_pars_cut=guess_pars_cut,
            guess_pars_surv=guess_pars_surv,
            func=func,
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

    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    if high_cut_val is not None:
        mask = (cut_param > low_cut_val) & (cut_param < high_cut_val) & dt_mask
    else:
        if mode == "greater":
            mask = (cut_param > low_cut_val) & dt_mask
        elif mode == "less":
            mask = (cut_param < low_cut_val) & dt_mask
        else:
            raise ValueError("mode not recognised")

    ct_n = len(cut_param[~mask])
    ct_err = np.sqrt(len(cut_param[~mask]))
    surv_n = len(cut_param[mask])
    surv_err = np.sqrt(len(cut_param[mask]))

    pc_n = ct_n + surv_n
    pc_err = np.sqrt(surv_err**2 + ct_err**2)

    sf = (surv_n / pc_n) * 100
    err = sf * np.sqrt((pc_err / pc_n) ** 2 + (surv_err / surv_n) ** 2)

    return {
        "low_cut": low_cut_val,
        "sf": sf,
        "sf_err": err,
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
    if not isinstance(energy, np.ndarray):
        energy = np.array(energy)
    if not isinstance(cut_param, np.ndarray):
        cut_param = np.array(cut_param)

    cut_vals = np.linspace(cut_range[0], cut_range[1], n_samples)
    out_df = pd.DataFrame()

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
        pdf=aoe_peak,
        selection_string: str = "index==index",
        dt_corr: bool = False,
        dep_correct: bool = False,
        dt_cut: dict = None,
        dt_param: str = "dt_eff",
        high_cut_val: int = 3,
        mean_func: Callable = Pol1,
        sigma_func: Callable = SigmaFit,
        compt_bands_width: int = 20,
        debug_mode: bool = False,
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
        if len(self.cal_dicts) > 0 and re.match(
            r"(\d{8})T(\d{6})Z", list(self.cal_dicts)[0]
        ):
            for tstamp in self.cal_dicts:
                if tstamp in update_dict:
                    self.cal_dicts[tstamp].update(update_dict[tstamp])
                else:
                    self.cal_dicts[tstamp].update(update_dict)
        else:
            self.cal_dicts.update(update_dict)

    def time_correction(
        self, df, aoe_param, mode="full", output_name="AoE_Timecorr", display=0
    ):
        log.info("Starting A/E time correction")
        self.timecorr_df = pd.DataFrame()
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
                        if e == KeyboardInterrupt:
                            raise (e)
                        elif self.debug_mode:
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
                self.timecorr_df.set_index("run_timestamp", inplace=True)
                if len(self.timecorr_df) > 1:
                    if mode == "partial":
                        time_dict = fit_time_means(
                            np.array(self.timecorr_df.index),
                            np.array(self.timecorr_df["mean"]),
                            np.array(self.timecorr_df["sigma"]),
                        )

                    elif mode == "full":
                        time_dict = {
                            time: mean
                            for time, mean in zip(
                                np.array(self.timecorr_df.index),
                                np.array(self.timecorr_df["mean"]),
                            )
                        }

                    else:
                        raise ValueError("unknown mode")

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
                    if e == KeyboardInterrupt:
                        raise (e)
                    elif self.debug_mode:
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
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
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
            pk_pars, pk_covs = pgc.hpge_fit_energy_peak_tops(
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

                aoe_pars, aoe_errs, _ = unbinned_aoe_fit(
                    final_df.query(aoe_grp1)[aoe_param], pdf=self.pdf, display=display
                )

                self.dt_res_dict["aoe_fit1"] = {
                    "pars": aoe_pars.to_dict(),
                    "errs": aoe_errs.to_dict(),
                }

                aoe_pars2, aoe_errs2, _ = unbinned_aoe_fit(
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
                log.info(f"dtcorr successful alpha:{self.alpha}")

        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
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
                    if e == KeyboardInterrupt:
                        raise (e)
                    elif self.debug_mode:
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
            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)

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

        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
                raise (e)
            log.error("A/E energy correction failed")
            mu_pars, mu_errs, mu_cov = return_nans(self.mean_func.func)
            csqr_mu, dof_mu, p_val_mu = (np.nan, np.nan, np.nan)
            csqr_sig, dof_sig, p_val_sig = (np.nan, np.nan, np.nan)
            sig_pars, sig_errs, sig_cov = return_nans(self.sigma_func.func)
            dep_pars, dep_err, dep_cov = return_nans(self.pdf)
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
                dt_mask=None,
                cut_range=(-8, 0),
                n_samples=40,
                mode="greater",
                debug_mode=self.debug_mode,
            )

            valid_fits = self.cut_fits.query(
                f'sf_err<{(1.5 * np.nanpercentile(self.cut_fits["sf_err"], 85))}&sf_err==sf_err'
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
            self.low_cut_val = round(xs[np.argmin(np.abs(p - (100 * dep_acc)))], 3)
            log.info(f"Cut found at {self.low_cut_val}")

            data[output_cut_param] = data[aoe_param] > self.low_cut_val
            if self.dt_cut_param is not None:
                data[output_cut_param] = data[output_cut_param] & (
                    data[self.dt_cut_param]
                )
        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
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
        data,
        aoe_param,
        peaks,
        fit_widths,
        n_samples=26,
        cut_range=(-5, 5),
        mode="greater",
    ):
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
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )

                    cut_df, sf, sf_err = compton_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[aoe_param].to_numpy(),
                        self.low_cut_val,
                        peak,
                        fwhm,
                        cut_range=cut_range,
                        n_samples=n_samples,
                        mode=mode,
                        dt_mask=(
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
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
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
                        dt_mask=(
                            peak_df[self.dt_cut_param].to_numpy()
                            if self.dt_cut_param is not None
                            else None
                        ),
                        debug_mode=self.debug_mode,
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
            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)
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
        sfs = pd.DataFrame()
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
                        dt_mask=(
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
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
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
                        dt_mask=(
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
                log.info(f"{peak}keV: {sf:2.1f} +/- {sf_err:2.1f} %")

            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)
                sfs = pd.concat(
                    [
                        sfs,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(f"A/E survival fraction determination failed for {peak} peak")
        sfs.set_index("peak", inplace=True)
        return sfs

    def calibrate(
        self,
        df,
        initial_aoe_param,
        peaks_of_interest=None,
        fit_widths=None,
        cut_peak_idx=0,
        dep_acc=0.9,
        sf_nsamples=11,
        sf_cut_range=(-5, 5),
        timecorr_mode="full",
    ):
        if peaks_of_interest is None:
            peaks_of_interest = [1592.5, 1620.5, 2039, 2103.53, 2614.50]
        if fit_widths is None:
            fit_widths = [(40, 25), (25, 40), (0, 0), (25, 40), (50, 50)]

        self.time_correction(
            df, initial_aoe_param, mode=timecorr_mode, output_name="AoE_Timecorr"
        )

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

        if re.match(r"(\d{8})T(\d{6})Z", list(self.cal_dicts)[0]):
            self.low_side_sfs_by_run = {}
            self.two_side_sfs_by_run = {}
            for tstamp in self.cal_dicts:
                log.info(f"Compute survival fractions for {tstamp}: ")
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

        xs = np.linspace(aoe_pars["x_lo"], aoe_pars["x_hi"], 100)
        counts, aoe_bins, bars = plt.hist(
            final_df.query(
                f'{aoe_class.dt_res_dict["aoe_grp1"]}&({aoe_param}<{aoe_pars["x_hi"]})&({aoe_param}>{aoe_pars["x_lo"]})'
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
        counts, aoe_bins2, bars = plt.hist(
            final_df.query(
                f'{aoe_class.dt_res_dict["aoe_grp2"]}&({aoe_param}<{aoe_pars2["x_hi"]})&({aoe_param}>{aoe_pars2["x_lo"]})'
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

        mus = aoe_class.dt_res_dict["dt_fit"]["mus"]
        sigmas = aoe_class.dt_res_dict["dt_fit"]["sigmas"]
        amps = aoe_class.dt_res_dict["dt_fit"]["amps"]

        for mu, sigma, amp in zip(mus, sigmas, amps):
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


def plot_sigma_fit(aoe_class, data, figsize=(12, 8), fontsize=12) -> plt.figure:
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


def plot_cut_fit(
    aoe_class, data, dep_acc=0.9, figsize=(12, 8), fontsize=12
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
                    label=f'{get_peak_label(peak)} {peak} keV: {aoe_class.low_side_sfs.loc[peak]["sf"]:2.1f} +/- {aoe_class.low_side_sfs.loc[peak]["sf_err"]:2.1f} %',
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
