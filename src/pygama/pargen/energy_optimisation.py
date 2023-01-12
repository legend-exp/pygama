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
import pickle as pkl
import sys
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost, util
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, minimize
from scipy.stats import chisquare, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def run_optimisation(
    file,
    opt_config,
    dsp_config,
    cuts,
    fom,
    db_dict=None,
    n_events=8000,
    wf_field="waveform",
    **fom_kwargs,
):
    """
    Runs optimisation on .lh5 file

    Parameters
    ----------
    file: string
        path to raw .lh5 file
    opt_config: str
        path to JSON dictionary to configure optimisation
    dsp_config: str
        path to JSON dictionary specifying dsp configuration
    fom: function
        When given the output lh5 table of a DSP iteration, the
        fom_function must return a scalar figure-of-merit value upon which the
        optimization will be based. Should accept verbosity as a second argument
    db_dict: dict
        Dictionary specifying any values to put in processing chain e.g. pz constant
    n_events : int
        Number of events to run over
    """
    grid = set_par_space(opt_config)
    waveforms = sto.read_object(f"/raw/{wf_field}", file, idx=cuts, n_rows=n_events)[0]
    baseline = sto.read_object("/raw/baseline", file, idx=cuts, n_rows=n_events)[0]
    tb_data = lh5.Table(col_dict={f"{wf_field}": waveforms, "baseline": baseline})
    return opt.run_grid(tb_data, dsp_config, grid, fom, db_dict, **fom_kwargs)


def run_optimisation_multiprocessed(
    file,
    opt_config,
    dsp_config,
    cuts,
    lh5_path,
    fom=None,
    db_dict=None,
    processes=5,
    n_events=8000,
    **fom_kwargs,
):
    """
    Runs optimisation on .lh5 file, this version multiprocesses the grid points, it also can handle multiple grids being passed
    as long as they are the same dimensions.

    Parameters
    ----------
    file: string
        path to raw .lh5 file
    opt_config: str
        path to JSON dictionary to configure optimisation
    dsp_config: str
        path to JSON dictionary specifying dsp configuration
    fom: function
        When given the output lh5 table of a DSP iteration, the
        fom_function must return a scalar figure-of-merit value upon which the
        optimization will be based. Should accept verbosity as a second argument
    n_events : int
        Number of events to run over
    db_dict: dict
        Dictionary specifying any values to put in processing chain e.g. pz constant
    processes : int
        Number of separate processes to run for the multiprocessing
    """

    def form_dict(in_dict, length):
        keys = list(in_dict.keys())
        out_list = []
        for i in range(length):
            out_list.append({keys[0]: 0})
        for key in keys:
            if isinstance(in_dict[key], list):
                if len(in_dict[key]) == length:
                    for i in range(length):
                        out_list[i][key] = in_dict[key][i]
                else:
                    for i in range(length):
                        out_list[i][key] = in_dict[key]
            else:
                for i in range(length):
                    out_list[i][key] = in_dict[key]
        return out_list

    if not isinstance(opt_config, list):
        opt_config = [opt_config]
    grid = []
    for i, opt_conf in enumerate(opt_config):
        grid.append(set_par_space(opt_conf))
    if fom_kwargs:
        if "fom_kwargs" in fom_kwargs:
            fom_kwargs = fom_kwargs["fom_kwargs"]
        fom_kwargs = form_dict(fom_kwargs, len(grid))
    sto = lh5.LH5Store()
    waveforms = sto.read_object(
        f"{lh5_path}/{wf_field}", file, idx=cuts, n_rows=n_events
    )[0]
    baseline = sto.read_object(f"{lh5_path}/baseline", file, idx=cuts, n_rows=n_events)[
        0
    ]
    tb_data = lh5.Table(col_dict={f"{wf_field}": waveforms, "baseline": baseline})
    return opt.run_grid_multiprocess_parallel(
        tb_data,
        dsp_config,
        grid,
        fom,
        db_dict=db_dict,
        processes=processes,
        fom_kwargs=fom_kwargs,
    )


def set_par_space(opt_config):
    """
    Generates grid for optimizer from dictionary of form {param : {start: , end: , spacing: }}
    """
    par_space = opt.ParGrid()
    for name in opt_config.keys():
        p_values = opt_config[name]
        for param in p_values.keys():
            str_vals = set_values(p_values[param])
            par_space.add_dimension(name, param, str_vals)
    return par_space


def set_values(par_values):
    """
    Finds values for grid
    """
    string_values = np.arange(
        par_values["start"], par_values["end"], par_values["spacing"]
    )
    try:
        string_values = [f'{val:.4f}*{par_values["unit"]}' for val in string_values]
    except:
        string_values = [f"{val:.4f}" for val in string_values]
    return string_values


def simple_guess(hist, bins, var, func_i, fit_range):
    """
    Simple guess for peak fitting
    """
    if func_i == pgf.extended_radford_pdf:
        bin_cs = (bins[1:] + bins[:-1]) / 2
        _, sigma, amp = pgh.get_gaussian_guess(hist, bins)
        i_0 = np.nanargmax(hist)
        mu = bin_cs[i_0]
        height = hist[i_0]
        bg0 = np.mean(hist[-10:])
        step = np.mean(hist[:10]) - bg0
        htail = 1.0 / 5
        tau = 0.5 * sigma

        hstep = step / (bg0 + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((4 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
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
        mu, sigma, amp = pgh.get_gaussian_guess(hist, bins)
        i_0 = np.argmax(hist)
        bg = np.mean(hist[-10:])
        step = bg - np.mean(hist[:10])
        hstep = step / (bg + np.mean(hist[:10]))
        dx = np.diff(bins)[0]
        n_bins_range = int((4 * sigma) // dx)
        nsig_guess = np.sum(hist[i_0 - n_bins_range : i_0 + n_bins_range])
        nbkg_guess = np.sum(hist) - nsig_guess
        return [nsig_guess, mu, sigma, nbkg_guess, hstep, fit_range[0], fit_range[1], 0]


def unbinned_energy_fit(
    energy,
    func,
    gof_func,
    gof_range,
    fit_range=(np.inf, np.inf),
    guess=None,
    tol=None,
    verbose=False,
    display=0,
):

    """
    Unbinned fit to energy. This is different to the default fitting as
    it will try different fitting methods and choose the best. This is necessary for the lower statistics.
    """

    bin_width = 1
    lower_bound = (np.nanmin(energy) // bin_width) * bin_width
    upper_bound = ((np.nanmax(energy) // bin_width) + 1) * bin_width
    hist1, bins, var = pgh.get_hist(
        energy, dx=bin_width, range=(lower_bound, upper_bound)
    )
    bin_cs1 = (bins[:-1] + bins[1:]) / 2
    if guess is not None:
        x0 = [*guess[:-2], fit_range[0], fit_range[1], False]
    else:
        if func == pgf.extended_radford_pdf:
            x0 = simple_guess(hist1, bins, var, pgf.extended_gauss_step_pdf, fit_range)
            if verbose:
                print(x0)
            c = cost.ExtendedUnbinnedNLL(energy, pgf.extended_gauss_step_pdf)
            m = Minuit(c, *x0)
            m.fixed[-3:] = True
            m.simplex().migrad()
            m.hesse()
            if guess is not None:
                x0_rad = [*guess[:-2], fit_range[0], fit_range[1], False]
            else:
                x0_rad = simple_guess(hist1, bins, var, func, fit_range)
            x0 = m.values[:3]
            x0 += x0_rad[3:5]
            x0 += m.values[3:]
        else:
            x0 = simple_guess(hist1, bins, var, func, fit_range)
    if verbose:
        print(x0)
    c = cost.ExtendedUnbinnedNLL(energy, func)
    m = Minuit(c, *x0)
    if tol is not None:
        m.tol = tol
    m.fixed[-3:] = True
    m.migrad()
    m.hesse()

    hist, bins, var = pgh.get_hist(energy, dx=1, range=gof_range)
    bin_cs = (bins[:-1] + bins[1:]) / 2
    m_fit = func(bin_cs1, *m.values)[1]

    valid1 = (
        m.valid
        # & m.accurate
        & (~np.isnan(m.errors).any())
        & (~(np.array(m.errors[:-3]) == 0).all())
    )

    cs = pgf.goodness_of_fit(
        hist, bins, None, gof_func, m.values[:-3], method="Pearson"
    )
    cs = cs[0] / cs[1]
    m2 = Minuit(c, *x0)
    if tol is not None:
        m2.tol = tol
    m2.fixed[-3:] = True
    m2.simplex().migrad()
    m2.hesse()
    m2_fit = func(bin_cs1, *m2.values)[1]
    valid2 = (
        m2.valid
        # & m2.accurate
        & (~np.isnan(m.errors).any())
        & (~(np.array(m2.errors[:-3]) == 0).all())
    )

    cs2 = pgf.goodness_of_fit(
        hist, bins, None, gof_func, m2.values[:-3], method="Pearson"
    )
    cs2 = cs2[0] / cs2[1]

    frac_errors1 = np.sum(np.abs(np.array(m.errors)[:-3] / np.array(m.values)[:-3]))
    frac_errors2 = np.sum(np.abs(np.array(m2.errors)[:-3] / np.array(m2.values)[:-3]))

    if verbose:
        print(m)
        print(m2)
        print(frac_errors1, frac_errors2)

    if display > 1:
        m_fit = gof_func(bin_cs1, *m.values)
        m2_fit = gof_func(bin_cs1, *m2.values)
        plt.figure()
        plt.plot(bin_cs1, hist1, label=f"hist")
        plt.plot(bin_cs1, func(bin_cs1, *x0)[1], label=f"Guess")
        plt.plot(bin_cs1, m_fit, label=f"Fit 1: {cs}")
        plt.plot(bin_cs1, m2_fit, label=f"Fit 2: {cs2}")
        plt.legend()
        plt.show()

    if valid1 == False and valid2 == False:
        log.debug("Extra simplex needed")
        m = Minuit(c, *x0)
        if tol is not None:
            m.tol = tol
        m.fixed[-3:] = True
        m.limits = pgc.get_hpge_E_bounds(func)
        m.simplex().simplex().migrad()
        m.hesse()
        if verbose:
            print(m)
        cs = pgf.goodness_of_fit(
            hist, bins, None, gof_func, m.values[:-3], method="Pearson"
        )
        cs = cs[0] / cs[1]
        valid3 = (
            m.valid
            # & m.accurate
            & (~np.isnan(m.errors).any())
            & (~(np.array(m.errors[:-3]) == 0).all())
        )
        if valid3 == False:
            raise RuntimeError

        pars = np.array(m.values)[:-1]
        errs = np.array(m.errors)[:-1]
        cov = np.array(m.covariance)[:-1, :-1]
        csqr = cs

    elif valid2 == False or cs * 1.05 < cs2:
        pars = np.array(m.values)[:-1]
        errs = np.array(m.errors)[:-3]
        cov = np.array(m.covariance)[:-1, :-1]
        csqr = cs

    elif valid1 == False or cs2 * 1.05 < cs:
        pars = np.array(m2.values)[:-1]
        errs = np.array(m2.errors)[:-3]
        cov = np.array(m2.covariance)[:-1, :-1]
        csqr = cs2

    elif frac_errors1 < frac_errors2:
        pars = np.array(m.values)[:-1]
        errs = np.array(m.errors)[:-3]
        cov = np.array(m.covariance)[:-1, :-1]
        csqr = cs

    elif frac_errors1 > frac_errors2:
        pars = np.array(m2.values)[:-1]
        errs = np.array(m2.errors)[:-3]
        cov = np.array(m2.covariance)[:-1, :-1]
        csqr = cs2

    else:
        raise RuntimeError

    return pars, errs, cov, csqr


def get_peak_fwhm_with_dt_corr(
    Energies,
    alpha,
    dt,
    func,
    gof_func,
    peak,
    kev_width,
    guess=None,
    kev=False,
    display=0,
):

    """
    Applies the drift time correction and fits the peak returns the fwhm, fwhm/max and associated errors,
    along with the number of signal events and the reduced chi square of the fit. Can return result in ADC or keV.
    """

    correction = np.multiply(
        np.multiply(alpha, dt, dtype="float64"), Energies, dtype="float64"
    )
    ct_energy = np.add(correction, Energies)

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
            energy_pars, energy_err, cov, chisqr = unbinned_energy_fit(
                ct_energy[win_idxs],
                func,
                gof_func,
                gof_range,
                fit_range,
                tol=tol,
                guess=guess,
                verbose=True,
                display=display,
            )
            print(energy_pars)
            print(energy_err)
            print(cov)
            plt.figure()
            xs = np.arange(lower_bound, upper_bound, bin_width)
            hist, bins, var = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.plot((bins[1:] + bins[:-1]) / 2, hist)
            plt.plot(xs, gof_func(xs, *energy_pars))
            plt.show()
        else:
            energy_pars, energy_err, cov, chisqr = unbinned_energy_fit(
                ct_energy[win_idxs],
                func,
                gof_func,
                gof_range,
                fit_range,
                guess=guess,
                tol=tol,
            )
        if func == pgf.extended_radford_pdf:
            if energy_pars[3] < 1e-6 and energy_err[3] < 1e-6:
                fwhm = energy_pars[2] * 2 * np.sqrt(2 * np.log(2))
                fwhm_err = np.sqrt(cov[2][2]) * 2 * np.sqrt(2 * np.log(2))
            else:
                fwhm = pgf.radford_fwhm(energy_pars[2], energy_pars[3], energy_pars[4])

        elif func == pgf.extended_gauss_step_pdf:
            fwhm = energy_pars[2] * 2 * np.sqrt(2 * np.log(2))
            fwhm_err = np.sqrt(cov[2][2]) * 2 * np.sqrt(2 * np.log(2))

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

        if func == pgf.extended_radford_pdf and not (
            energy_pars[3] < 1e-6 and energy_err[3] < 1e-6
        ):
            y_b = np.zeros(len(par_b))
            for i, p in enumerate(par_b):
                try:
                    y_b[i] = pgf.radford_fwhm(p[2], p[3], p[4])  #
                except:
                    y_b[i] = np.nan
            fwhm_err = np.nanstd(y_b, axis=0)
            if fwhm_err == 0:
                fwhm, fwhm_err = pgf.radford_fwhm(
                    energy_pars[2],
                    energy_pars[3],
                    energy_pars[4],
                    cov=cov[:, :-2][:-2, :],
                )
            fwhm_o_max_err = np.nanstd(y_b / maxs, axis=0)
        else:
            max_err = np.nanstd(maxs)
            fwhm_o_max_err = fwhm_o_max * np.sqrt(
                (np.array(fwhm_err) / np.array(fwhm)) ** 2
                + (np.array(max_err) / np.array(max_val)) ** 2
            )

        if display > 1:
            plt.figure()
            plt.plot((bins[1:] + bins[:-1]) / 2, hist)
            for i in range(100):
                plt.plot(xs, y_max[i, :])
            plt.show()

        if display > 0:
            plt.figure()
            hist, bins, var = pgh.get_hist(
                ct_energy, dx=bin_width, range=(lower_bound, upper_bound)
            )
            plt.plot((bins[1:] + bins[:-1]) / 2, hist)
            plt.plot(xs, gof_func(xs, *energy_pars))
            plt.fill_between(
                xs, y - yerr_boot, y + yerr_boot, facecolor="C1", alpha=0.5
            )
            plt.show()

    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None

    if kev == True:
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


def fom_FWHM_with_dt_corr_fit(tb_in, kwarg_dict, ctc_parameter, idxs=None, display=0):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm with its error,
    the corresponding alpha value and the number of events in the fitted peak, also the reduced chisquare of the
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    gof_func = kwarg_dict["gof_func"]
    Energies = tb_in[parameter].nda
    Energies = Energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]
    min_alpha = 0
    max_alpha = 3.50e-06
    astep = 1.250e-07
    if ctc_parameter == "QDrift":
        dt = tb_in["dt_eff"].nda
    elif ctc_parameter == "dt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    elif ctc_parameter == "rt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")

    if idxs is not None:
        Energies = Energies[idxs]
        dt = dt[idxs]

    if np.isnan(Energies).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }
    if np.isnan(dt).any():
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    alphas = np.array(
        [
            0.000e00,
            1.250e-07,
            2.500e-07,
            3.750e-07,
            5.000e-07,
            6.250e-07,
            7.500e-07,
            8.750e-07,
            1.000e-06,
            1.125e-06,
            1.250e-06,
            1.375e-06,
            1.500e-06,
            1.625e-06,
            1.750e-06,
            1.875e-06,
            2.000e-06,
            2.125e-06,
            2.250e-06,
            2.375e-06,
            2.500e-06,
            2.625e-06,
            2.750e-06,
            2.875e-06,
            3.000e-06,
            3.125e-06,
            3.250e-06,
            3.375e-06,
            3.500e-06,
        ],
        dtype="float64",
    )
    fwhms = np.array([])
    final_alphas = np.array([])
    fwhm_errs = np.array([])
    guess = None
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
            Energies, alpha, dt, func, gof_func, peak, kev_width, guess=guess
        )
        if not np.isnan(fwhm_o_max):
            fwhms = np.append(fwhms, fwhm_o_max)
            final_alphas = np.append(final_alphas, alpha)
            fwhm_errs = np.append(fwhm_errs, fwhm_o_max_err)
            guess = fit_pars
            if fwhms[-1] < best_fwhm:
                best_fwhm = fwhms[-1]
                best_fit = fit_pars
        log.info(f"alpha: {alpha}, fwhm/max:{fwhm_o_max}+-{fwhm_o_max_err}")

    # Make sure fit isn't based on only a few points
    if len(fwhms) < 10:
        log.error("less than 10 fits successful")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    ids = (fwhm_errs < 2 * np.nanpercentile(fwhm_errs, 50)) & (fwhm_errs > 0)
    # Fit alpha curve to get best alpha

    try:
        alphas = np.arange(
            final_alphas[ids][0], final_alphas[ids][-1], astep / 20, dtype="float64"
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

    except:
        log.error("alpha fit failed")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if np.isnan(fit_vals).all():
        log.error("alpha fit all nan")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "chisquare": np.nan,
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
            Energies,
            alpha,
            dt,
            func,
            gof_func,
            peak,
            kev_width,
            guess=best_fit,
            kev=True,
            display=display,
        )
        if np.isnan(final_fwhm) or np.isnan(final_err):
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
                Energies,
                alpha,
                dt,
                func,
                gof_func,
                peak,
                kev_width,
                kev=True,
                display=display,
            )
        if np.isnan(final_fwhm) or np.isnan(final_err):
            log.error(f"final fit failed, alpha was {alpha}")
        return {
            "fwhm": final_fwhm,
            "fwhm_err": final_err,
            "alpha": alpha,
            "alpha_err": alpha_err,
            "chisquare": csqr,
            "n_sig": n_sig,
            "n_sig_err": n_sig_err,
        }


def fom_all_fit(tb_in, kwarg_dict):
    """
    FOM to run over different ctc parameters
    """
    ctc_parameters = ["QDrift"]  #'dt',
    output_dict = {}
    for param in ctc_parameters:
        out = fom_FWHM_with_dt_corr_fit(tb_in, kwarg_dict, param)
        output_dict[param] = out
    return output_dict


def fom_FWHM_fit(tb_in, kwarg_dict):
    """
    FOM with no ctc sweep, used for optimising ftp.
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    gof_func = kwarg_dict["gof_func"]
    Energies = tb_in[parameter].nda
    Energies = Energies.astype("float64")
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

    if np.isnan(Energies).any():
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
        Energies, alpha, dt, func, gof_func, peak=peak, kev_width=kev_width, kev=True
    )
    return {
        "fwhm_o_max": final_fwhm_o_max,
        "max_o_fwhm": final_fwhm_o_max_err,
        "chisquare": csqr,
        "n_sig": n_sig,
        "n_sig_err": n_sig_err,
    }


def get_wf_indexes(sorted_indexs, n_events):
    out_list = []
    if isinstance(n_events, list):
        for i in range(len(n_events)):
            new_list = []
            for idx, entry in enumerate(sorted_indexs):
                if (entry >= np.sum(n_events[:i])) and (
                    entry < np.sum(n_events[: i + 1])
                ):
                    new_list.append(idx)
            out_list.append(new_list)
    else:
        for i in range(int(len(sorted_indexs) / n_events)):
            new_list = []
            for idx, entry in enumerate(sorted_indexs):
                if (entry >= i * n_events) and (entry < (i + 1) * n_events):
                    new_list.append(idx)
            out_list.append(new_list)
    return out_list


def index_data(data, indexes):
    new_baselines = lh5.Array(data["baseline"].nda[indexes])
    new_waveform_values = data["waveform"]["values"].nda[indexes]
    new_waveform_dts = data["waveform"]["dt"].nda[indexes]
    new_waveform_t0 = data["waveform"]["t0"].nda[indexes]
    new_waveform = lh5.WaveformTable(
        None, new_waveform_t0, "ns", new_waveform_dts, "ns", new_waveform_values
    )
    new_data = lh5.Table(col_dict={"waveform": new_waveform, "baseline": new_baselines})
    return new_data


def event_selection(
    raw_files,
    lh5_path,
    dsp_config,
    db_dict,
    peaks_keV,
    peak_idxs,
    kev_widths,
    cut_parameters={"bl_mean": 4, "bl_std": 4, "pz_std": 4},
    energy_parameter="trapTmax",
    wf_field: str = "waveform",
    n_events=10000,
    threshold=1000,
):
    if not isinstance(peak_idxs, list):
        peak_idxs = [peak_idxs]
    if not isinstance(kev_widths, list):
        kev_widths = [kev_widths]

    sto = lh5.LH5Store()
    df = lh5.load_dfs(raw_files, ["daqenergy", "timestamp"], lh5_path)

    pulser_props = cts.find_pulser_properties(df, energy="daqenergy")
    if len(pulser_props) > 0:
        out_df = cts.tag_pulsers(df, pulser_props, window=0.001)
        ids = out_df.isPulser == 1
        log.debug(f"pulser found: {pulser_props}")
    else:
        log.debug("no_pulser")
        ids = np.zeros(len(df.daqenergy.values), dtype=bool)
    # Get events around peak using raw file values
    initial_mask = (df.daqenergy.values > threshold) & (~ids)
    rough_energy = df.daqenergy.values[initial_mask]
    initial_idxs = np.where(initial_mask)[0]

    guess_keV = 2620 / np.nanpercentile(rough_energy, 99)
    Euc_min = threshold / guess_keV * 0.6
    Euc_max = 2620 / guess_keV * 1.1
    dEuc = 1 / guess_keV
    hist, bins, var = pgh.get_hist(rough_energy, range=(Euc_min, Euc_max), dx=dEuc)
    detected_peaks_locs, detected_peaks_keV, roughpars = pgc.hpge_find_E_peaks(
        hist,
        bins,
        var,
        np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2614.553]),
    )
    log.debug(f"detected {detected_peaks_keV} keV peaks at {detected_peaks_locs}")

    masks = []
    for peak_idx in peak_idxs:
        peak = peaks_keV[peak_idx]
        kev_width = kev_widths[peak_idx]
        try:
            if peak not in detected_peaks_keV:
                raise ValueError
            detected_peak_idx = np.where(detected_peaks_keV == peak)[0]
            peak_loc = detected_peaks_locs[detected_peak_idx]
            log.info(f"{peak} peak found at {peak_loc}")
            rough_adc_to_kev = roughpars[0]
            e_lower_lim = peak_loc - (1.1 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.1 * kev_width[1]) / rough_adc_to_kev
        except:
            log.debug(f"{peak} peak not found attempting to use rough parameters")
            peak_loc = (peak - roughpars[1]) / roughpars[0]
            rough_adc_to_kev = roughpars[0]
            e_lower_lim = peak_loc - (1.5 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.5 * kev_width[1]) / rough_adc_to_kev
        log.debug(f"lower_lim:{e_lower_lim}, upper_lim:{e_upper_lim}")
        e_mask = (rough_energy > e_lower_lim) & (rough_energy < e_upper_lim)
        e_idxs = initial_idxs[e_mask][: int(2.5 * n_events)]
        masks.append(e_idxs)
        log.debug(f"{len(e_idxs)} events found in energy range for {peak}")

    idx_list_lens = [len(masks[peak_idx]) for peak_idx in peak_idxs]

    sort_index = np.argsort(np.concatenate(masks))
    idx_list = get_wf_indexes(sort_index, idx_list_lens)
    idxs = np.array(sorted(np.concatenate(masks)))

    waveforms = sto.read_object(
        f"{lh5_path}/{wf_field}", raw_files, idx=idxs, n_rows=len(idxs)
    )[0]
    baseline = sto.read_object(
        f"{lh5_path}/baseline", raw_files, idx=idxs, n_rows=len(idxs)
    )[0]
    input_data = lh5.Table(col_dict={f"{wf_field}": waveforms, "baseline": baseline})

    if isinstance(dsp_config, str):
        with open(dsp_config) as r:
            dsp_config = json.load(r)

    dsp_config["outputs"] = cts.get_keys(
        dsp_config["outputs"], list(cut_parameters)
    ) + ["energy_parameter"]

    log.debug("Processing data")
    tb_data = opt.run_one_dsp(input_data, dsp_config, db_dict=db_dict)

    cut_dict = cts.generate_cuts(tb_data, cut_parameters)
    log.debug(f"Cuts are: {cut_dict}")
    log.debug("Loaded Cuts")
    ct_mask = cts.get_cut_indexes(tb_data, cut_dict, "raw")

    final_events = []
    for peak_idx in peak_idxs:
        peak = peaks_keV[peak_idx]
        kev_width = kev_widths[peak_idx]

        peak_ids = np.array(idx_list[peak_idx])
        peak_ct_mask = ct_mask[peak_ids]
        peak_ids = peak_ids[peak_ct_mask]

        energy = tb_data[energy_parameter].nda[peak_ids]

        hist, bins, var = pgh.get_hist(
            energy, range=(int(threshold), int(np.nanmax(energy))), dx=1
        )
        peak_loc = pgh.get_bin_centers(bins)[np.nanargmax(hist)]
        rough_adc_to_kev = peak / peak_loc

        e_lower_lim = peak_loc - (1.5 * kev_width[0]) / rough_adc_to_kev
        e_upper_lim = peak_loc + (1.5 * kev_width[1]) / rough_adc_to_kev

        e_ranges = (int(peak_loc - e_lower_lim), int(e_upper_lim - peak_loc))
        params, errors, covs, bins, ranges, p_val = pgc.hpge_fit_E_peaks(
            energy,
            [peak_loc],
            [e_ranges],
            n_bins=(np.nanmax(energy) - np.nanmin(energy)) // 1,
        )
        if params[0] is None:
            log.debug("Fit failed, using max guess")
            hist, bins, var = pgh.get_hist(
                energy, range=(int(e_lower_lim), int(e_upper_lim)), dx=1
            )
            params = [[0, pgh.get_bin_centers(bins)[np.nanargmax(hist)], 0, 0, 0, 0]]
        updated_adc_to_kev = peak / params[0][1]
        e_lower_lim = params[0][1] - (kev_width[0]) / updated_adc_to_kev
        e_upper_lim = params[0][1] + (kev_width[1]) / updated_adc_to_kev
        log.info(f"lower lim is :{e_lower_lim}, upper lim is {e_upper_lim}")
        final_mask = (energy > e_lower_lim) & (energy < e_upper_lim)
        final_events.append(peak_ids[final_mask][:n_events])
        log.info(f"{len(peak_ids[final_mask][:n_events])} passed selections for {peak}")

    sort_index = np.argsort(np.concatenate(final_events))
    idx_list = get_wf_indexes(sort_index, [len(mask) for mask in final_events])
    idxs = np.array(sorted(np.concatenate(final_events)))

    final_data = index_data(input_data, idxs)
    return final_data, idx_list


def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1 * x + m2 * (x**2))


def interpolate_energy(peak_energies, points, err_points, energy):

    nan_mask = np.isnan(points) | (points < 0)
    if len(points[~nan_mask]) < 3:
        return np.nan, np.nan, np.nan
    elif nan_mask[-1] == True or nan_mask[-2] == True:
        return np.nan, np.nan, np.nan
    else:
        param_guess = [0.2, 0.001, 0.000001]  #
        # param_bounds = (0, [10., 1. ])#
        try:
            fit_pars, fit_covs = curve_fit(
                fwhm_slope,
                peak_energies[~nan_mask],
                points[~nan_mask],
                sigma=err_points[~nan_mask],
                p0=param_guess,
                absolute_sigma=True,
            )  # bounds=param_bounds,
            fit_qbb = fwhm_slope(energy, *fit_pars)

            xs = np.arange(peak_energies[0], peak_energies[-1], 0.1)

            rng = np.random.default_rng(1)

            # generate set of bootstrapped parameters
            par_b = rng.multivariate_normal(fit_pars, fit_covs, size=1000)
            qbb_vals = np.array([fwhm_slope(energy, *p) for p in par_b])
            qbb_err = np.nanstd(qbb_vals)
        except:
            return np.nan, np.nan, np.nan

        if nan_mask[-2] == True:
            qbb_vals += qbb_err

    return fit_qbb, qbb_err, fit_pars


def fom_FWHM(tb_in, kwarg_dict, ctc_parameter, alpha, idxs=None, display=0):
    """
    FOM for sweeping over ctc values to find the best value, returns the best found fwhm
    """
    parameter = kwarg_dict["parameter"]
    func = kwarg_dict["func"]
    cs_func = kwarg_dict["gof_func"]
    Energies = tb_in[parameter].nda
    Energies = Energies.astype("float64")
    peak = kwarg_dict["peak"]
    kev_width = kwarg_dict["kev_width"]

    if ctc_parameter == "QDrift":
        dt = tb_in["dt_eff"].nda
    elif ctc_parameter == "dt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_0_est"].nda, dtype="float64")
    elif ctc_parameter == "rt":
        dt = np.subtract(tb_in["tp_99"].nda, tb_in["tp_01"].nda, dtype="float64")
    if np.isnan(Energies).any() or np.isnan(dt).any():
        if np.isnan(Energies).any():
            log.warning(f"nan energy values for peak {peak}")
        else:
            log.warning(f"nan dt values for peak {peak}")
        return {
            "fwhm": np.nan,
            "fwhm_err": np.nan,
            "alpha": np.nan,
            "chisquare": np.nan,
            "n_sig": np.nan,
            "n_sig_err": np.nan,
        }

    if idxs is not None:
        Energies = Energies[idxs]
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
            Energies,
            alpha,
            dt,
            func,
            cs_func,
            peak,
            kev_width,
            kev=True,
            display=display,
        )
    except:
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
    peaks = kwarg_dict["peaks_keV"]
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]
    peak_dicts = kwarg_dict["peak_dicts"]

    out_dict = fom_FWHM_with_dt_corr_fit(
        data, peak_dicts[0], ctc_param, idxs=idx_list[0], display=0
    )
    out_dict["y_val"] = out_dict["fwhm"]
    return out_dict


def new_fom(data, kwarg_dict):

    peaks = kwarg_dict["peaks_keV"]
    idx_list = kwarg_dict["idx_list"]
    ctc_param = kwarg_dict["ctc_param"]

    peak_dicts = kwarg_dict["peak_dicts"]

    out_dict = fom_FWHM_with_dt_corr_fit(
        data, peak_dicts[-1], ctc_param, idxs=idx_list[-1], display=0
    )
    alpha = out_dict["alpha"]
    log.info(alpha)
    fwhms = []
    fwhm_errs = []
    n_sig = []
    n_sig_err = []
    for i, peak in enumerate(peaks[:-1]):
        out_peak_dict = fom_FWHM(
            data, peak_dicts[i], ctc_param, alpha, idxs=idx_list[i], display=0
        )
        # n_sig_minimum = peak_dicts[i]["n_sig_minimum"]
        # if peak_dict["n_sig"]<n_sig_minimum:
        #    out_peak_dict['fwhm'] = np.nan
        #   out_peak_dict['fwhm_err'] = np.nan
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
        "qbb_fwhm": qbb,
        "qbb_fwhm_err": qbb_err,
        "alpha": alpha,
        "peaks": peaks.tolist(),
        "fwhms": fwhms,
        "fwhm_errs": fwhm_errs,
        "n_events": n_sig,
        "n_sig_err": n_sig_err,
    }


OptimiserDimension = namedtuple(
    "OptimiserDimension", "name parameter min_val max_val rounding unit"
)


class BayesianOptimizer:
    np.random.seed(55)
    lambda_param = 0.01
    eta_param = 0
    kernel = None
    # FIXME: the following throws a TypeError
    # kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1, length_scale_bounds="fixed") #+ WhiteKernel(noise_level=0.0111)

    def __init__(self, acq_func, batch_size):

        self.dims = []
        self.current_iter = 0

        self.batch_size = batch_size
        self.iters = 0

        self.gauss_pr = GaussianProcessRegressor(kernel=self.kernel)
        self.best_samples_ = pd.DataFrame(columns=["x", "y", "ei"])
        self.distances_ = []

        if acq_func == "ei":
            self.acq_function = self._get_expected_improvement
        elif acq_func == "ucb":
            self.acq_function = self._get_ucb
        elif acq_func == "lcb":
            self.acq_function = self._get_lcb

    def add_dimension(self, name, parameter, min_val, max_val, rounding=2, unit=None):
        self.dims.append(
            OptimiserDimension(name, parameter, min_val, max_val, rounding, unit)
        )

    def get_n_dimensions(self):
        return len(self.dims)

    def add_initial_values(self, x_init, y_init):
        self.x_init = x_init
        self.y_init = y_init

    def _get_expected_improvement(self, x_new):

        # Using estimate from Gaussian surrogate instead of actual function for
        # a new trial data point to avoid cost

        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        sigma_y_new = sigma_y_new.reshape(-1, 1)[0]

        # Using estimates from Gaussian surrogate instead of actual function for
        # entire prior distribution to avoid cost

        mean_y = self.gauss_pr.predict(self.x_init)
        min_mean_y = np.min(mean_y)
        z = (mean_y_new - min_mean_y - self.eta_param) / (sigma_y_new + 1e-9)  #
        exp_imp = (mean_y_new - min_mean_y - self.eta_param) * norm.cdf(
            z
        ) + sigma_y_new * norm.pdf(z)
        return exp_imp

    def _get_ucb(self, x_new):

        # Using estimate from Gaussian surrogate instead of actual function for
        # a new trial data point to avoid cost

        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        sigma_y_new = sigma_y_new.reshape(-1, 1)[0]

        return mean_y_new + self.lambda_param * sigma_y_new

    def _get_lcb(self, x_new):

        # Using estimate from Gaussian surrogate instead of actual function for
        # a new trial data point to avoid cost

        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        sigma_y_new = sigma_y_new.reshape(-1, 1)[0]

        return mean_y_new - self.lambda_param * sigma_y_new

    def _acquisition_function(self, x):
        return self.acq_function(x)

    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None
        # Trial with an array of random data points
        rands = np.random.uniform(
            np.array([dim.min_val for dim in self.dims]),
            np.array([dim.max_val for dim in self.dims]),
            (self.batch_size, self.get_n_dimensions()),
        )
        for x_start in rands:
            response = minimize(
                fun=self._acquisition_function,
                x0=x_start,
                bounds=[(dim.min_val, dim.max_val) for dim in self.dims],
                method="L-BFGS-B",
            )
            if response.fun[0] < min_ei:
                min_ei = response.fun[0]
                x_optimal = [
                    y.round(dim.rounding) for y, dim in zip(response.x, self.dims)
                ]
        if x_optimal in self.x_init and self.iters < 5:
            if self.iters < 5:
                self.iters += 1
                x_optimal, min_ei = self._get_next_probable_point()
            else:
                perturb = np.random.uniform(
                    np.array([(dim.max_val - dim.min_val) / 100 for dim in self.dims]),
                    np.array([(dim.max_val - dim.min_val) / 10 for dim in self.dims]),
                    (1, len(self.dims)),
                )
                x_optimal += perturb
                x_optimal = [
                    y.round(dim.rounding) for y, dim in zip(x_optimal[0], self.dims)
                ]
                for i, y in enumerate(x_optimal):
                    if y > self.dims[i].max_val:
                        x_optimal[i] = self.dims[i].max_val
                    elif y < self.dims[i].min_val:
                        x_optimal[i] = self.dims[i].min_val

        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y):
        self.x_init = np.append(self.x_init, np.array([x]), axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)

    def get_first_point(self):
        y_min_ind = np.nanargmin(self.y_init)
        self.y_min = self.y_init[y_min_ind]
        self.optimal_x = self.x_init[y_min_ind]
        self.optimal_ei = None
        return self.optimal_x, self.optimal_ei

    def iterate_values(self):
        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        x_next, ei = self._get_next_probable_point()
        return x_next, ei

    def update_db_dict(self, db_dict):
        if self.current_iter == 0:
            x_new, ei = self.get_first_point()
        x_new, ei = self.iterate_values()
        self.current_x = x_new
        self.current_ei = ei
        for i, val in enumerate(x_new):
            name, parameter, min_val, max_val, rounding, unit = self.dims[i]
            if unit is not None:
                value_str = f"{val}*{unit}"
            else:
                value_str = f"{val}"
            if name not in db_dict.keys():
                db_dict[name] = {parameter: value_str}
            else:
                db_dict[name][parameter] = value_str
        self.current_iter += 1
        return db_dict

    def update(self, results):
        y_val = results["y_val"]
        self._extend_prior_with_posterior_data(self.current_x, np.array([y_val]))

        if np.isnan(y_val):
            pass
        else:
            if y_val < self.y_min:
                self.y_min = y_val
                self.optimal_x = self.current_x
                self.optimal_ei = self.current_ei
                self.optimal_results = results

        if self.current_iter == 1:
            self.prev_x = self.current_x
        else:
            self.distances_.append(
                np.linalg.norm(np.array(self.prev_x) - np.array(self.current_x))
            )
            self.prev_x = self.current_x

        self.best_samples_ = self.best_samples_.append(
            {"y": self.y_min, "ei": self.optimal_ei}, ignore_index=True
        )

    def get_best_vals(self):
        out_dict = {}
        for i, val in enumerate(self.optimal_x):
            name, parameter, min_val, max_val, rounding, unit = self.dims[i]
            value_str = f"{val}*{unit}"
            if name not in out_dict.keys():
                out_dict[name] = {parameter: value_str}
            else:
                out_dict[name][parameter] = value_str
        return out_dict

    def plot(self, init_samples=None):
        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        elif len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)
            ys = np.zeros_like(points)
            for i, point in enumerate(points):
                ys[i] = self.gauss_pr.predict(
                    np.array([point]).reshape(1, -1), return_std=False
                )
            fig = plt.figure()
            plt.plot(points, ys)
            plt.scatter(np.array(self.x_init), np.array(self.y_init))
            if init_samples is not None:
                init_ys = np.array(
                    [
                        np.where(init_sample == self.x_init)[0][0]
                        for init_sample in init_samples
                    ]
                )
                plt.scatter(
                    np.array(init_samples)[:, 0],
                    np.array(self.y_init)[init_ys],
                    color="red",
                )
            plt.scatter(
                self.optimal_x[0],
                self.y_min,
                color="orange",
            )

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel(f"Kernel Value")
        elif len(self.dims) == 2:
            x, y = np.mgrid[
                self.dims[0].min_val : self.dims[0].max_val : 0.1,
                self.dims[1].min_val : self.dims[1].max_val : 0.1,
            ]
            points = np.vstack((x.flatten(), y.flatten())).T
            out_grid = np.zeros(
                (
                    int((self.dims[0].max_val - self.dims[0].min_val) * 10),
                    int((self.dims[1].max_val - self.dims[1].min_val) * 10),
                )
            )

            j = 0
            for i, _ in np.ndenumerate(out_grid):
                out_grid[i] = self.gauss_pr.predict(
                    points[j].reshape(1, -1), return_std=False
                )
                j += 1

            fig = plt.figure()
            plt.imshow(
                out_grid,
                norm=LogNorm(),
                origin="lower",
                aspect="auto",
                extent=(0, out_grid.shape[1], 0, out_grid.shape[0]),
            )
            plt.scatter(
                np.array(self.x_init - self.dims[1].min_val)[:, 1] * 10,
                np.array(self.x_init - self.dims[0].min_val)[:, 0] * 10,
            )
            if init_samples is not None:
                plt.scatter(
                    (init_samples[:, 1] - self.dims[1].min_val) * 10,
                    (init_samples[:, 0] - self.dims[0].min_val) * 10,
                    color="red",
                )
            plt.scatter(
                (self.optimal_x[1] - self.dims[1].min_val) * 10,
                (self.optimal_x[0] - self.dims[0].min_val) * 10,
                color="orange",
            )
            ticks, labels = plt.xticks()
            labels = np.linspace(self.dims[1].min_val, self.dims[1].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[1], 5)
            plt.xticks(ticks=ticks, labels=labels, rotation=45)
            ticks, labels = plt.yticks()
            labels = np.linspace(self.dims[0].min_val, self.dims[0].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[0], 5)
            plt.yticks(ticks=ticks, labels=labels, rotation=45)
            plt.xlabel(
                f"{self.dims[1].name}-{self.dims[1].parameter}({self.dims[1].unit})"
            )
            plt.ylabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
        plt.title(f"{self.dims[0].name} Kernel Prediction")
        plt.tight_layout()
        plt.close()
        return fig

    def plot_acq(self, init_samples=None):
        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        elif len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)
            ys = np.zeros_like(points)
            for i, point in enumerate(points):
                ys[i] = self._acquisition_function(np.array([point]).reshape(1, -1)[0])
            fig = plt.figure()
            plt.plot(points, ys)
            plt.scatter(np.array(self.x_init), np.array(self.y_init))
            if init_samples is not None:
                init_ys = np.array(
                    [
                        np.where(init_sample == self.x_init)[0][0]
                        for init_sample in init_samples
                    ]
                )
                plt.scatter(
                    np.array(init_samples)[:, 0],
                    np.array(self.y_init)[init_ys],
                    color="red",
                )
            plt.scatter(
                self.optimal_x[0],
                self.y_min,
                color="orange",
            )

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel(f"Acquisition Function Value")

        elif len(self.dims) == 2:
            x, y = np.mgrid[
                self.dims[0].min_val : self.dims[0].max_val : 0.1,
                self.dims[1].min_val : self.dims[1].max_val : 0.1,
            ]
            points = np.vstack((x.flatten(), y.flatten())).T
            out_grid = np.zeros(
                (
                    int((self.dims[0].max_val - self.dims[0].min_val) * 10),
                    int((self.dims[1].max_val - self.dims[1].min_val) * 10),
                )
            )

            j = 0
            for i, _ in np.ndenumerate(out_grid):
                out_grid[i] = self._acquisition_function(points[j])
                j += 1

            fig = plt.figure()
            plt.imshow(
                out_grid,
                norm=LogNorm(),
                origin="lower",
                aspect="auto",
                extent=(0, out_grid.shape[1], 0, out_grid.shape[0]),
            )
            plt.scatter(
                np.array(self.x_init - self.dims[1].min_val)[:, 1] * 10,
                np.array(self.x_init - self.dims[0].min_val)[:, 0] * 10,
            )
            if init_samples is not None:
                plt.scatter(
                    (init_samples[:, 1] - self.dims[1].min_val) * 10,
                    (init_samples[:, 0] - self.dims[0].min_val) * 10,
                    color="red",
                )
            plt.scatter(
                (self.optimal_x[1] - self.dims[1].min_val) * 10,
                (self.optimal_x[0] - self.dims[0].min_val) * 10,
                color="orange",
            )
            ticks, labels = plt.xticks()
            labels = np.linspace(self.dims[1].min_val, self.dims[1].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[1], 5)
            plt.xticks(ticks=ticks, labels=labels, rotation=45)
            ticks, labels = plt.yticks()
            labels = np.linspace(self.dims[0].min_val, self.dims[0].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[0], 5)
            plt.yticks(ticks=ticks, labels=labels, rotation=45)
            plt.xlabel(
                f"{self.dims[1].name}-{self.dims[1].parameter}({self.dims[1].unit})"
            )
            plt.ylabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
        plt.title(f"{self.dims[0].name} Acquisition Space")
        plt.tight_layout()
        plt.close()
        return fig


def run_optimisation(
    tb_data,
    dsp_config,
    fom_function,
    optimisers,
    fom_kwargs=None,
    db_dict=None,
    nan_val=10,
    n_iter=10,
):
    if not isinstance(optimisers, list):
        optimisers = [optimisers]
    if not isinstance(fom_kwargs, list):
        fom_kwargs = [fom_kwargs]
    if not isinstance(fom_function, list):
        fom_function = [fom_function]

    for j in range(n_iter):
        for optimiser in optimisers:
            db_dict = optimiser.update_db_dict(db_dict)

        log.info(f"Iteration number: {j+1}")
        log.info(f"Processing with {db_dict}")

        tb_out = opt.run_one_dsp(tb_data, dsp_config, db_dict=db_dict)

        res = np.ndarray(shape=len(optimisers), dtype="O")

        for i in range(len(optimisers)):
            if fom_kwargs[i] is not None:
                if len(fom_function) > 1:
                    res[i] = fom_function[i](tb_out, fom_kwargs[i])
                else:
                    res[i] = fom_function[0](tb_out, fom_kwargs[i])
            else:
                if len(fom_function) > 1:
                    res[i] = fom_function[i](tb_out)
                else:
                    res[i] = fom_function[0](tb_out)

        log.info(f"Results of iteration {j+1} are {res}")

        for i, optimiser in enumerate(optimisers):
            if np.isnan(res[i]["y_val"]):
                if isinstance(nan_val, list):
                    res[i]["y_val"] = nan_val[i]
                else:
                    res[i]["y_val"] = nan_val

            optimiser.update(res[i])

    out_param_dict = {}
    out_results_list = []
    for optimiser in optimisers:
        param_dict = optimiser.get_best_vals()
        out_param_dict.update(param_dict)
        results_dict = optimiser.optimal_results
        out_results_list.append(results_dict)

    return out_param_dict, out_results_list


def get_ctc_grid(grids, ctc_param):
    """
    Reshapes optimizer grids to be in easier form
    """
    error_grids = []
    dt_grids = []
    alpha_grids = []
    alpha_error_grids = []
    nevents_grids = []
    for grid in grids:
        shape = grid.shape
        dt_grid = np.ndarray(shape=shape)
        alpha_grid = np.ndarray(shape=shape)
        error_grid = np.ndarray(shape=shape)
        alpha_error_grid = np.ndarray(shape=shape)
        nevents_grid = np.ndarray(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dt_grid[i, j] = grid[i, j][ctc_param]["fwhm"]
                error_grid[i, j] = grid[i, j][ctc_param]["fwhm_err"]
                nevents_grid[i, j] = grid[i, j][ctc_param]["n_sig"]
                try:
                    alpha_grid[i, j] = grid[i, j][ctc_param]["alpha"]
                except:
                    pass
                try:
                    alpha_error_grid[i, j] = grid[i, j][ctc_param]["alpha_err"]
                except:
                    pass
        dt_grids.append(dt_grid)
        alpha_grids.append(alpha_grid)
        error_grids.append(error_grid)
        alpha_error_grids.append(alpha_error_grid)
        nevents_grids.append(nevents_grid)
    return dt_grids, error_grids, alpha_grids, alpha_error_grids, nevents_grids


def interpolate_energy_old(peak_energies, grids, error_grids, energy, nevents_grids):
    """
    Interpolates fwhm vs energy for every grid point
    """

    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    out_grid_err = np.empty(grid_shape)
    n_event_lim = np.array(
        [0.98 * np.nanpercentile(nevents_grid, 50) for nevents_grid in nevents_grids]
    )
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        err_points = np.array([error_grids[i][index] for i in range(len(grids))])
        n_sigs = np.array([nevents_grids[i][index] for i in range(len(grids))])
        nan_mask = (
            np.isnan(points)
            | (points < 0)
            | (0.1 * points < err_points)
            | (n_sigs < n_event_lim)
        )
        try:
            if len(points[nan_mask]) > 2:
                raise ValueError
            elif nan_mask[-1] == True or nan_mask[-2] == True:
                raise ValueError
            param_guess = [0.2, 0.001, 0.000001]
            param_bounds = param_bounds = (0, [1, np.inf, np.inf])  # ,0.1
            fit_pars, fit_covs = curve_fit(
                fwhm_slope,
                peak_energies[~nan_mask],
                points[~nan_mask],
                sigma=err_points[~nan_mask],
                p0=param_guess,
                bounds=param_bounds,
                absolute_sigma=True,
            )  #
            fit_qbb = fwhm_slope(energy, *fit_pars)
            sderrs = np.sqrt(np.diag(fit_covs))
            qbb_err = fwhm_slope(energy, *(fit_pars + sderrs)) - fwhm_slope(
                energy, *fit_pars
            )
            out_grid[index] = fit_qbb
            out_grid_err[index] = qbb_err
        except:
            out_grid[index] = np.nan
            out_grid_err[index] = np.nan
    return out_grid, out_grid_err


def find_lowest_grid_point_save(grid, err_grid, opt_dict):
    """
    Finds the lowest grid point, if more than one with same value returns shortest filter.
    """
    opt_name = list(opt_dict.keys())[0]
    print(opt_name)
    keys = list(opt_dict[opt_name].keys())
    param_list = []
    shape = []
    db_dict = {}
    for key in keys:
        param_dict = opt_dict[opt_name][key]
        grid_axis = np.arange(
            param_dict["start"], param_dict["end"], param_dict["spacing"]
        )
        unit = param_dict.get("unit")
        param_list.append(grid_axis)
        shape.append(len(grid_axis))

    total_lengths = np.zeros(shape)

    for index, x in np.ndenumerate(total_lengths):
        for i, param in enumerate(param_list):
            total_lengths[index] += param[index[i]]
    min_val = np.nanmin(grid)
    lowest_ixs = np.where(grid == min_val)
    try:
        fwhm_dict = {"fwhm": min_val, "fwhm_err": err_grid[lowest_ixs][0]}
    except:
        print(lowest_ixs)
    if len(lowest_ixs[0]) == 1:
        for i, key in enumerate(keys):
            if i == 0:
                if unit is not None:
                    db_dict[opt_name] = {
                        key: f"{param_list[i][lowest_ixs[i]][0]}*{unit}"
                    }
                else:
                    db_dict[opt_name] = {key: f"{param_list[i][lowest_ixs[i]][0]}"}
            else:
                if unit is not None:
                    db_dict[opt_name][key] = f"{param_list[i][lowest_ixs[i]][0]}*{unit}"
                else:
                    db_dict[opt_name][key] = f"{param_list[i][lowest_ixs[i]][0]}"
    else:
        shortest_length = np.argmin(total_lengths[lowest_ixs])
        final_idxs = [lowest_ix[shortest_length] for lowest_ix in lowest_ixs]
        for i, key in enumerate(keys):
            if unit is not None:
                db_dict[opt_name] = {key: f"{param_list[i][lowest_ixs[i]][0]}*{unit}"}
            else:
                db_dict[opt_name] = {key: f"{param_list[i][lowest_ixs[i]][0]}"}
    return lowest_ixs, fwhm_dict, db_dict


def interpolate_grid(energies, grids, int_energy, deg, nevents_grids):
    """
    Interpolates energy vs parameter for every grid point using polynomial.
    """
    grid_no = len(grids)
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    n_event_lim = np.array(
        [0.98 * np.nanpercentile(nevents_grid, 50) for nevents_grid in nevents_grids]
    )
    for index, x in np.ndenumerate(grids[0]):
        points = np.array([grids[i][index] for i in range(len(grids))])
        n_sigs = np.array([nevents_grids[i][index] for i in range(len(grids))])
        nan_mask = np.isnan(points) | (points < 0) | (n_sigs < n_event_lim)
        try:
            if len(points[~nan_mask]) < 3:
                raise IndexError
            fit_point = np.polynomial.polynomial.polyfit(
                energies[~nan_mask], points[~nan_mask], deg=deg
            )
            out_grid[index] = np.polynomial.polynomial.polyval(int_energy, fit_point)
        except:
            out_grid[index] = np.nan
    return out_grid


def get_best_vals(peak_grids, peak_energies, param, opt_dict, save_path=None):
    """
    Finds best filter parameters
    """
    dt_grids, error_grids, alpha_grids, alpha_error_grids, nevents_grids = get_ctc_grid(
        peak_grids, param
    )
    qbb_grid, qbb_errs = interpolate_energy(
        peak_energies, dt_grids, error_grids, 2039.061, nevents_grids
    )
    qbb_alphas = interpolate_grid(
        peak_energies[2:], alpha_grids[2:], 2039.061, 1, nevents_grids[2:]
    )
    ixs, fwhm_dict, db_dict = find_lowest_grid_point_save(qbb_grid, qbb_errs, opt_dict)
    out_grid = {"fwhm": qbb_grid, "fwhm_err": qbb_errs, "alphas": qbb_alphas}

    if isinstance(save_path, str):
        mpl.use("pdf")
        e_param = list(opt_dict.keys())[0]
        opt_dict = opt_dict[e_param]

        detector = save_path.split("/")[-1]
        save_path = os.path.join(save_path, f"{e_param}-{param}.pdf")
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        with PdfPages(save_path) as pdf:

            keys = list(opt_dict.keys())
            print(keys)
            x_dict = opt_dict[keys[1]]
            xvals = np.arange(x_dict["start"], x_dict["end"], x_dict["spacing"])
            xs = (
                np.arange(0, len(xvals), 1),
                np.arange(x_dict["start"], x_dict["end"], x_dict["spacing"]),
            )
            y_dict = opt_dict[keys[0]]
            yvals = np.arange(y_dict["start"], y_dict["end"], y_dict["spacing"])
            ys = (
                np.arange(0, len(yvals), 1),
                np.arange(y_dict["start"], y_dict["end"], y_dict["spacing"]),
            )
            for i, x in enumerate(xs[1]):
                xs[1][i] = round(x, 1)
            for i, y in enumerate(ys[1]):
                ys[1][i] = round(y, 1)
            print(ixs)
            points = np.array(
                [dt_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(dt_grids))]
            )
            err_points = np.array(
                [error_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(error_grids))]
            )
            alpha_points = np.array(
                [alpha_grids[i][ixs[0][0], ixs[1][0]] for i in range(len(alpha_grids))]
            )
            alpha_error_points = np.array(
                [
                    alpha_error_grids[i][ixs[0][0], ixs[1][0]]
                    for i in range(len(alpha_error_grids))
                ]
            )
            param_guess = [0.2, 0.001, 0.000001]
            param_bounds = (0, [1, np.inf, np.inf])  # ,0.1
            nan_mask = np.isnan(points)
            nan_mask = nan_mask | (points < 0) | (0.1 * points < err_points)
            fit_pars, fit_covs = curve_fit(
                fwhm_slope,
                peak_energies[~nan_mask],
                points[~nan_mask],
                sigma=err_points[~nan_mask],
                p0=param_guess,
                bounds=param_bounds,
                absolute_sigma=True,
            )  #
            energy_x = np.arange(200, 2600, 10)
            plt.rcParams["figure.figsize"] = (12, 18)
            plt.rcParams["font.size"] = 12
            plt.figure()
            for i, dt_grid in enumerate(dt_grids):
                plt.subplot(3, 2, i + 1)
                v_min = np.nanmin(np.abs(dt_grid))
                if v_min == 0:
                    for j in range(10):
                        v_min = np.nanpercentile(np.abs(dt_grid), j + 1)
                        if v_min > 0.1:
                            break
                plt.imshow(
                    dt_grid,
                    norm=LogNorm(vmin=v_min, vmax=np.nanpercentile(dt_grid, 98)),
                    cmap="viridis",
                )

                plt.xticks(xs[0], xs[1])
                plt.yticks(ys[0], ys[1])

                plt.xlabel(f"{keys[1]} (us)")
                plt.ylabel(f"{keys[0]} (us)")
                plt.title(f"{peak_energies[i]:.1f} kev")
                plt.xticks(rotation=45)
                cbar = plt.colorbar()
                cbar.set_label("FWHM (keV)")
            plt.tight_layout()
            plt.suptitle(f"{detector}-{e_param}-{param}")
            pdf.savefig()
            plt.close()

            plt.figure()

            plt.imshow(
                qbb_grid,
                norm=LogNorm(
                    vmin=np.nanmin(qbb_grid), vmax=np.nanpercentile(dt_grid, 98)
                ),
                cmap="viridis",
            )
            plt.xlabel(f"{keys[1]} (us)")
            plt.ylabel(f"{keys[0]} (us)")
            plt.title(f"Qbb")
            plt.xticks(rotation=45)
            cbar = plt.colorbar()
            cbar.set_label("FWHM (keV)")
            plt.tight_layout()
            plt.suptitle(f"{detector}-{e_param}-{param}")
            pdf.savefig()
            plt.close()

            fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
            ax1.errorbar(peak_energies, points, yerr=err_points, fmt=" ")
            ax1.plot(energy_x, fwhm_slope(energy_x, *fit_pars))
            ax1.errorbar(
                [2039], qbb_grid[ixs[0], ixs[1]], yerr=qbb_errs[ixs[0], ixs[1]], fmt=" "
            )
            ax1.set_ylabel("FWHM energy resolution (keV)", ha="right", y=1)
            ax2.scatter(
                peak_energies,
                (points - fwhm_slope(peak_energies, *fit_pars)) / err_points,
                lw=1,
                c="b",
            )
            ax2.set_xlabel("Energy (keV)", ha="right", x=1)
            ax2.set_ylabel("Standardised Residuals", ha="right", y=1)
            fig.suptitle(f"{detector}-{e_param}-{param}")
            pdf.savefig()
            plt.close()

            try:
                alphas = qbb_alphas[ixs[0], ixs[1]][0]
                if isinstance(save_path, str):
                    alpha_fit = np.polynomial.polynomial.polyfit(
                        peak_energies[2:], alpha_points[2:], deg=1
                    )
                    fig, (ax1, ax2) = plt.subplots(
                        2, 1, constrained_layout=True, sharex=True
                    )
                    ax1.errorbar(
                        peak_energies[:],
                        alpha_points[:],
                        yerr=alpha_error_points[:],
                        linestyle=" ",
                    )
                    ax1.plot(
                        peak_energies[2:],
                        np.polynomial.polynomial.polyval(peak_energies[2:], alpha_fit),
                    )
                    ax1.scatter([2039], qbb_alphas[ixs[0], ixs[1]])
                    ax1.set_ylabel("Charge Trapping Value", ha="right", y=1)
                    ax2.scatter(
                        peak_energies[2:],
                        (
                            alpha_points[2:]
                            - np.polynomial.polynomial.polyval(
                                peak_energies[2:], alpha_fit
                            )
                        )
                        / alpha_points[2:],
                        lw=1,
                        c="b",
                    )
                    ax2.set_xlabel("Energy (keV)", ha="right", x=1)
                    ax2.set_ylabel("Residuals (%)", ha="right", y=1)
                    fig.suptitle(f"{detector}-{param}")
                    pdf.savefig()
                    plt.close()
            except:
                alphas = np.nan
    else:
        try:
            alphas = qbb_alphas[ixs[0], ixs[1]][0]
        except:
            alphas = np.nan
    return alphas, fwhm_dict, db_dict, out_grid


def get_filter_params(
    grids, matched_configs, peak_energies, parameters, save_path=None
):
    """
    Finds best parameters for filter
    """

    full_db_dict = {}
    full_fwhm_dict = {}
    full_grids = {}

    for param in parameters:
        opt_dict = matched_configs[param]
        peak_grids = grids[param]
        ctc_params = list(peak_grids[0][0, 0].keys())
        ctc_dict = {}

        for ctc_param in ctc_params:
            if ctc_param == "QDrift":
                alpha, fwhm, db_dict, output_grid = get_best_vals(
                    peak_grids, peak_energies, ctc_param, opt_dict, save_path=save_path
                )
                opt_name = list(opt_dict.keys())[0]
                db_dict[opt_name].update({"alpha": alpha})

            else:
                alpha, fwhm, _, output_grid = get_best_vals(
                    peak_grids, peak_energies, ctc_param, opt_dict, save_path=save_path
                )
            try:
                full_grids[param][ctc_param] = output_grid
            except:
                full_grids[param] = {ctc_param: output_grid}
            fwhm.update({"alpha": alpha})
            ctc_dict[ctc_param] = fwhm
        full_fwhm_dict[param] = ctc_dict
        full_db_dict.update(db_dict)
    return full_db_dict, full_fwhm_dict, full_grids
