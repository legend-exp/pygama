"""
This module contains the functions for performing the energy optimisation.
This happens in 2 steps, firstly a grid search is performed on each peak
separately using the optimiser, then the resulting grids are interpolated
to provide the best energy resolution at Qbb
"""

import json
import logging
import os
import pathlib
import sys
from collections import namedtuple

import lgdo.lh5 as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from dspeed.units import unit_registry as ureg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings

import pygama.math.histogram as pgh
import pygama.math.distributions as pgd
import pygama.math.binned_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def simple_guess(energy, func, fit_range=None, bin_width=1):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    hist, bins, var = pgh.get_hist(energy, range=fit_range, dx=bin_width)

    if func == pgd.hpge_peak:
        bin_cs = (bins[1:] + bins[:-1]) / 2
        _, sigma, amp = pgh.get_gaussian_guess(hist, bins)
        i_0 = np.nanargmax(hist)
        mu = bin_cs[i_0]
        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((4 * sigma) // dx)
        nsig = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg = np.sum(hist) - nsig
        parguess = {
            "n_sig": nsig,
            "mu": mu,
            "sigma": sigma,
            "htail": htail,
            "tau": tau,
            "n_bkg": n_bkg,
            "hstep": hstep,
            "x_lo": fit_range[0],
            "x_hi": fit_range[1],
        } 

    elif func == pgd.gauss_on_step:
        mu, sigma, amp = pgh.get_gaussian_guess(hist, bins)
        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((4 * sigma) // dx)
        nsig = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg = np.sum(hist) - nsig
        parguess = {
            "n_sig": n_sig,
            "mu": mu,
            "sigma": sigma,
            "htail": htail,
            "tau": tau,
            "n_bkg": n_bkg,
            "hstep": hstep,
            "x_lo": fit_range[0],
            "x_hi": fit_range[1],
        }
    else:
        log.error(f"simple_guess not implemented for {func.__name__}")
        return return_nans(func)

    return convert_to_minuit(parguess, func).values


def get_peak_fwhm_with_dt_corr(
    energies,
    alpha,
    dt,
    func,
    gof_func,
    peak,
    kev_width,
    guess=None,
    kev=False,
    frac_max=0.5,
    allow_tail_drop=False,
    display=0,
):
    """
    Applies the drift time correction and fits the peak returns the fwhm, fwhm/max and associated errors,
    along with the number of signal events and the reduced chi square of the fit. Can return result in ADC or keV.
    """

    correction = np.multiply(
        np.multiply(alpha, dt, dtype="float64"), energies, dtype="float64"
    )
    ct_energy = np.add(correction, energies)

    bin_width = 1
    lower_bound = (np.nanmin(ct_energy) // bin_width) * bin_width
    upper_bound = ((np.nanmax(ct_energy) // bin_width) + 1) * bin_width
    hist, bins, var = pgh.get_hist(
        ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
    )
    mu = bins[np.nanargmax(hist)]
    adc_to_kev = mu / peak
    # Making the window slightly smaller removes effects where as mu moves edge can be outside bin width
    lower_bound = mu - ((kev_width[0] - 2) * adc_to_kev)
    upper_bound = mu + ((kev_width[1] - 2) * adc_to_kev)
    win_idxs = (ct_energy > lower_bound) & (ct_energy < upper_bound)
    fit_range = (lower_bound, upper_bound)
    if peak > 1500:
        gof_range = (mu - (7 * adc_to_kev), mu + (7 * adc_to_kev))
    else:
        gof_range = (mu - (5 * adc_to_kev), mu + (5 * adc_to_kev))
    # if kev==True:
    # else:
    #    tol=0.01
    tol = None
    try:
        if display > 0:
            (
                energy_pars,
                energy_err,
                cov,
                chisqr,
                func,
                gof_func,
                _,
                _,
                _,
            ) = pgc.unbinned_staged_energy_fit(
                ct_energy[win_idxs],
                func=func,
                gof_range=gof_range,
                fit_range=fit_range,
                guess_func=simple_guess,
                tol=tol,
                guess=guess,
                allow_tail_drop=allow_tail_drop,
                display=display,
            )
            plt.figure()
            xs = np.arange(lower_bound, upper_bound, bin_width)
            hist, bins, var = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.step((bins[1:] + bins[:-1]) / 2, hist)
            plt.plot(xs, gof_func(xs, *energy_pars))
            plt.show()
        else:
            (
                energy_pars,
                energy_err,
                cov,
                chisqr,
                func,
                gof_func,
                _,
                _,
                _,
            ) = pgc.unbinned_staged_energy_fit(
                ct_energy[win_idxs],
                func=func,
                gof_func=gof_func,
                gof_range=gof_range,
                fit_range=fit_range,
                guess_func=simple_guess,
                tol=tol,
                guess=guess,
                allow_tail_drop=allow_tail_drop,
            )

        fwhm = func.get_fwfm(energy_pars, frac_max)

        xs = np.arange(lower_bound, upper_bound, 0.1)
        y = func(xs, *energy_pars)[1]
        max_val = np.amax(y)

        fwhm_o_max = fwhm / max_val

        rng = np.random.default_rng(1)
        # generate set of bootstrapped parameters
        par_b = rng.multivariate_normal(energy_pars, cov, size=100)
        y_max = np.array([func(xs, *p)[1] for p in par_b])
        maxs = np.nanmax(y_max, axis=1)

        yerr_boot = np.nanstd(y_max, axis=0)

        if func == pgd.hpge_peak and not (
            energy_pars["htail"] < 1e-6 and energy_err["htail"] < 1e-6
        ):
            y_b = np.zeros(len(par_b))
            for i, p in enumerate(par_b):
                try:
                    y_b[i] = func.get_fwfm(energy_pars, frac_max)
                except Exception:
                    y_b[i] = np.nan
            fwhm_err = np.nanstd(y_b, axis=0)
            fwhm_o_max_err = np.nanstd(y_b / maxs, axis=0)
        else:
            max_err = np.nanstd(maxs)
            fwhm_o_max_err = fwhm_o_max * np.sqrt(
                (np.array(fwhm_err) / np.array(fwhm)) ** 2
                + (np.array(max_err) / np.array(max_val)) ** 2
            )

        if display > 1:
            plt.figure()
            plt.step((bins[1:] + bins[:-1]) / 2, hist)
            for i in range(100):
                plt.plot(xs, y_max[i, :])
            plt.show()

        if display > 0:
            plt.figure()
            hist, bins, var = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.step((bins[1:] + bins[:-1]) / 2, hist)
            plt.plot(xs, y, color="orange")
            plt.fill_between(
                xs, y - yerr_boot, y + yerr_boot, facecolor="C1", alpha=0.5
            )
            plt.show()

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, None

    if kev is True:
        fwhm *= peak / energy_pars[1]
        fwhm_err *= peak / energy_pars[1]

    return (
        fwhm,
        fwhm_o_max,
        fwhm_err,
        fwhm_o_max_err,
        chisqr,
        energy_pars[0],
        energy_err[0],
        energy_pars,
    )


def fom_fwhm_with_dt_corr_fit(
    tb_in, kwarg_dict, ctc_parameter, nsteps=29, idxs=None, frac_max=0.2, display=0
):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm with its error,
    the corresponding alpha value and the number of events in the fitted peak, also the reduced chisquare of the
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    gof_func = kwarg_dict["gof_func"]
    energies = tb_in[parameter].nda
    energies = energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    min_alpha = 0
    max_alpha = 3.50e-06
    alphas = np.linspace(min_alpha, max_alpha, nsteps, dtype="float64")
    if ctc_parameter == "QDrift":
        dt = tb_in["dt_eff"].nda
    elif ctc_parameter == "dt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    elif ctc_parameter == "rt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")

    if idxs is not None:
        energies = energies[idxs]
        dt = dt[idxs]

    if np.isnan(energies).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }
    if np.isnan(dt).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }
    fwhms = np.array([])
    final_alphas = np.array([])
    fwhm_errs = np.array([])
    best_fwhm = np.inf
    for alpha in alphas:
        (
            _,
            fwhm_o_max,
            _,
            fwhm_o_max_err,
            _,
            _,
            _,
            fit_pars,
        ) = get_peak_fwhm_with_dt_corr(
            energies,
            alpha,
            dt,
            func,
            gof_func,
            peak,
            kev_width,
            guess=None,
            frac_max=0.5,
            allow_tail_drop=False,
        )
        if not np.isnan(fwhm_o_max):
            fwhms = np.append(fwhms, fwhm_o_max)
            final_alphas = np.append(final_alphas, alpha)
            fwhm_errs = np.append(fwhm_errs, fwhm_o_max_err)
            if fwhms[-1] < best_fwhm:
                best_fwhm = fwhms[-1]
        log.info(f"alpha: {alpha}, fwhm/max:{fwhm_o_max:.4f}+-{fwhm_o_max_err:.4f}")

    # Make sure fit isn't based on only a few points
    if len(fwhms) < nsteps * 0.2:
        log.debug("less than 20% fits successful")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    ids = (fwhm_errs < 2 * np.nanpercentile(fwhm_errs, 50)) & (fwhm_errs > 1e-10)
    # Fit alpha curve to get best alpha

    try:
        alphas = np.linspace(
            final_alphas[ids][0], final_alphas[ids][-1], nsteps * 20, dtype="float64"
        )
        alpha_fit, cov = np.polyfit(
            final_alphas[ids], fwhms[ids], w=1 / fwhm_errs[ids], deg=4, cov=True
        )
        fit_vals = np.polynomial.polynomial.polyval(alphas, alpha_fit[::-1])
        alpha = alphas[np.nanargmin(fit_vals)]

        rng = np.random.default_rng(1)
        alpha_pars_b = rng.multivariate_normal(alpha_fit, cov, size=1000)
        fits = np.array(
            [
                np.polynomial.polynomial.polyval(alphas, pars[::-1])
                for pars in alpha_pars_b
            ]
        )
        min_alphas = np.array([alphas[np.nanargmin(fit)] for fit in fits])
        alpha_err = np.nanstd(min_alphas)
        if display > 0:
            plt.figure()
            yerr_boot = np.std(fits, axis=0)
            plt.errorbar(final_alphas, fwhms, yerr=fwhm_errs, linestyle=" ")
            plt.plot(alphas, fit_vals)
            plt.fill_between(
                alphas,
                fit_vals - yerr_boot,
                fit_vals + yerr_boot,
                facecolor="C1",
                alpha=0.5,
            )
            plt.show()

    except Exception:
        log.debug("alpha fit failed")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if np.isnan(fit_vals).all():
        log.debug("alpha fit all nan")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": 0,
            "alpha_err": np.nan,
            "chisquare": (np.nan, np.nan),
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    else:
        # Return fwhm of optimal alpha in kev with error
        (
            final_fwhm,
            _,
            final_err,
            _,
            csqr,
            n_sig,
            n_sig_err,
            _,
        ) = get_peak_fwhm_with_dt_corr(
            energies,
            alpha,
            dt,
            func,
            gof_func,
            peak,
            kev_width,
            guess=None,
            kev=True,
            frac_max=frac_max,
            allow_tail_drop=True,
            display=display,
        )
        if np.isnan(final_fwhm) or np.isnan(final_err):
            log.debug(f"final fit failed, alpha was {alpha}")
        return {
            "fwhm": final_fwhm,
            "fwhm_err": final_err,
            "alpha": alpha,
            "alpha_err": alpha_err,
            "chisquare": csqr,
            "n_sig": n_sig,
            "n_sig_err": n_sig_err,
        }


def fom_fwhm_fit(tb_in, kwarg_dict):
    """
    FOM with no ctc sweep, used for optimising ftp.
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    gof_func = kwarg_dict["gof_func"]
    energies = tb_in[parameter].nda
    energies = energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    try:
        alpha = kwarg_dict["alpha"]
        if isinstance(alpha, dict):
            alpha = alpha[parameter]
    except KeyError:
        alpha = 0
    try:
        ctc_param = kwarg_dict["ctc_param"]
        dt = tb_in[ctc_param].nda
    except KeyError:
        dt = 0

    if np.isnan(energies).any():
        return {
            "fwhm_o_max": np.nan,
            "max_o_fwhm": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    (
        _,
        final_fwhm_o_max,
        _,
        final_fwhm_o_max_err,
        csqr,
        n_sig,
        n_sig_err,
    ) = get_peak_fwhm_with_dt_corr(
        energies, alpha, dt, func, gof_func, peak=peak, kev_width=kev_width, kev=True
    )
    return {
        "fwhm_o_max": final_fwhm_o_max,
        "max_o_fwhm": final_fwhm_o_max_err,
        "chisquare": csqr,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }

# use energy cal classes?
def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1 * x + m2 * (x**2))


def interpolate_energy(peak_energies, points, err_points, energy):
    nan_mask = np.isnan(points) | (points < 0)
    if len(points[~nan_mask]) < 3:
        return np.nan, np.nan, np.nan
    else:
        param_guess = [2, 0.001, 0.000001]  #
        # param_bounds = (0, [10., 1. ])#
        try:
            # switch this to iminuit
            fit_pars, fit_covs = curve_fit(
                fwhm_slope,
                peak_energies[~nan_mask],
                points[~nan_mask],
                sigma=err_points[~nan_mask],
                p0=param_guess,
                absolute_sigma=True,
            )  # bounds=param_bounds,
            fit_qbb = fwhm_slope(energy, *fit_pars)

            rng = np.random.default_rng(1)

            # generate set of bootstrapped parameters
            par_b = rng.multivariate_normal(fit_pars, fit_covs, size=1000)
            qbb_vals = np.array([fwhm_slope(energy, *p) for p in par_b])
            qbb_err = np.nanstd(qbb_vals)
        except Exception:
            return np.nan, np.nan, np.nan

        if nan_mask[-1] is True or nan_mask[-2] is True:
            qbb_err = np.nan
        if qbb_err / fit_qbb > 0.1:
            qbb_err = np.nan

    return fit_qbb, qbb_err, fit_pars


def fom_fwhm(tb_in, kwarg_dict, ctc_parameter, alpha, idxs=None, display=0):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    cs_func = kwarg_dict["gof_func"]
    energies = tb_in[parameter].nda
    energies = energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]

    if ctc_parameter == "QDrift":
        dt = tb_in["dt_eff"].nda
    elif ctc_parameter == "dt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    elif ctc_parameter == "rt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")
    if np.isnan(energies).any() or np.isnan(dt).any():
        if np.isnan(energies).any():
            log.debug(f"nan energy values for peak {peak}")
        else:
            log.debug(f"nan dt values for peak {peak}")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if idxs is not None:
        energies = energies[idxs]
        dt = dt[idxs]

    # Return fwhm of optimal alpha in kev with error
    try:
        (
            final_fwhm,
            _,
            final_err,
            _,
            csqr,
            n_sig,
            n_sig_err,
            _,
        ) = get_peak_fwhm_with_dt_corr(
            energies,
            alpha,
            dt,
            func,
            cs_func,
            peak,
            kev_width,
            kev=True,
            display=display,
        )
    except Exception:
        final_fwhm = np.nan
        final_err = np.nan
        csqr = np.nan
        n_sig = np.nan
        n_sig_err = np.nan
    return {
        "fwhm": final_fwhm,
        "fwhm_err": final_err,
        "alpha": alpha,
        "chisquare": csqr,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }


def single_peak_fom(data, kwarg_dict):
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]
    peak_dicts = kwarg_dict["peak_dicts"]
    if "frac_max" in kwarg_dict:
        frac_max = kwarg_dict["frac_max"]
    else:
        frac_max = 0.2
    out_dict = fom_fwhm_with_dt_corr_fit(
        data, peak_dicts[0], ctc_param, idxs=idx_list[0], frac_max=frac_max, display=0
    )

    out_dict["y_val"] = out_dict["fwhm"]
    out_dict["y_err"] = out_dict["fwhm_err"]
    return out_dict


def new_fom(data, kwarg_dict):
    peaks = kwarg_dict["peaks_kev"]
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]

    peak_dicts = kwarg_dict["peak_dicts"]

    if "frac_max" in kwarg_dict:
        frac_max = kwarg_dict["frac_max"]
    else:
        frac_max = 0.2

    out_dict = fom_fwhm_with_dt_corr_fit(
        data, peak_dicts[-1], ctc_param, idxs=idx_list[-1], frac_max=frac_max, display=0
    )
    alpha = out_dict["alpha"]
    log.info(alpha)
    fwhms = []
    fwhm_errs = []
    n_sig = []
    n_sig_err = []
    for i, _ in enumerate(peaks[:-1]):
        out_peak_dict = fom_fwhm(
            data,
            peak_dicts[i],
            ctc_param,
            alpha,
            idxs=idx_list[i],
            frac_max=frac_max,
            display=0,
        )
        fwhms.append(out_peak_dict["fwhm"])
        fwhm_errs.append(out_peak_dict["fwhm_err"])
        n_sig.append(out_peak_dict["n_sig"])
        n_sig_err.append(out_peak_dict["n_sig_err"])
    fwhms.append(out_dict["fwhm"])
    fwhm_errs.append(out_dict["fwhm_err"])
    n_sig.append(out_dict["n_sig"])
    n_sig_err.append(out_dict["n_sig_err"])
    log.info(f"fwhms are {fwhms}keV +- {fwhm_errs}")
    qbb, qbb_err, fit_pars = interpolate_energy(
        np.array(peaks), np.array(fwhms), np.array(fwhm_errs), 2039
    )

    log.info(f"Qbb fwhm is {qbb} keV +- {qbb_err}")

    return {
        "y_val": qbb,
        "y_err": qbb_err,
        "qbb_fwhm": qbb,
        "qbb_fwhm_err": qbb_err,
        "alpha": alpha,
        "peaks": peaks.tolist(),
        "fwhms": fwhms,
        "fwhm_errs": fwhm_errs,
        "n_events": n_sig,
        "n_sig_err": n_sig_err,
    }