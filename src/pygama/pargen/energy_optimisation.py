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
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def run_optimisation_grid(
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
    waveforms = sto.read(f"/raw/{wf_field}", file, idx=cuts, n_rows=n_events)[0]
    baseline = sto.read("/raw/baseline", file, idx=cuts, n_rows=n_events)[0]
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
    wf_field="waveform",
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
        for _ in range(length):
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
    for opt_conf in opt_config:
        grid.append(set_par_space(opt_conf))
    if fom_kwargs:
        if "fom_kwargs" in fom_kwargs:
            fom_kwargs = fom_kwargs["fom_kwargs"]
        fom_kwargs = form_dict(fom_kwargs, len(grid))
    sto = lh5.LH5Store()
    waveforms = sto.read(f"{lh5_path}/{wf_field}", file, idx=cuts, n_rows=n_events)[0]
    baseline = sto.read(f"{lh5_path}/baseline", file, idx=cuts, n_rows=n_events)[0]
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
    except Exception:
        string_values = [f"{val:.4f}" for val in string_values]
    return string_values


def simple_guess(energy, func_i, fit_range=None, bin_width=1):
    """
    Simple guess for peak fitting
    """
    if fit_range is None:
        fit_range = (np.nanmin(energy), np.nanmax(energy))
    hist, bins, var = pgh.get_hist(energy, range=fit_range, dx=bin_width)

    if func_i == pgf.extended_radford_pdf:
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
                gof_func=gof_func,
                gof_range=gof_range,
                fit_range=fit_range,
                guess_func=simple_guess,
                tol=tol,
                guess=guess,
                allow_tail_drop=allow_tail_drop,
                verbose=True,
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
        if func == pgf.extended_radford_pdf:
            if energy_pars[3] < 1e-6 and energy_err[3] < 1e-6:
                fwhm = energy_pars[2] * 2 * np.sqrt(2 * np.log(1 / frac_max))
                fwhm_err = np.sqrt(cov[2][2]) * 2 * np.sqrt(2 * np.log(1 / frac_max))
            else:
                fwhm = pgf.radford_full_width_at_frac_max(
                    energy_pars[2], energy_pars[3], energy_pars[4], frac_max
                )

        elif func == pgf.extended_gauss_step_pdf:
            fwhm = energy_pars[2] * 2 * np.sqrt(2 * np.log(1 / frac_max))
            fwhm_err = np.sqrt(cov[2][2]) * 2 * np.sqrt(2 * np.log(1 / frac_max))

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
                    y_b[i] = pgf.radford_full_width_at_frac_max(
                        p[2], p[3], p[4], frac_max
                    )  #
                except Exception:
                    y_b[i] = np.nan
            fwhm_err = np.nanstd(y_b, axis=0)
            if fwhm_err == 0:
                fwhm, fwhm_err = pgf.radford_full_width_at_frac_max(
                    energy_pars[2],
                    energy_pars[3],
                    energy_pars[4],
                    frac_max,
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


def index_data(data, indexes, wf_field="waveform"):
    new_baselines = lh5.Array(data["baseline"].nda[indexes])
    new_waveform_values = data[wf_field]["values"].nda[indexes]
    new_waveform_dts = data[wf_field]["dt"].nda[indexes]
    new_waveform_t0 = data[wf_field]["t0"].nda[indexes]
    new_waveform = lh5.WaveformTable(
        None, new_waveform_t0, "ns", new_waveform_dts, "ns", new_waveform_values
    )
    new_data = lh5.Table(col_dict={wf_field: new_waveform, "baseline": new_baselines})
    return new_data


def event_selection(
    raw_files,
    lh5_path,
    dsp_config,
    db_dict,
    peaks_kev,
    peak_idxs,
    kev_widths,
    cut_parameters=None,
    pulser_mask=None,
    energy_parameter="trapTmax",
    wf_field: str = "waveform",
    n_events=10000,
    threshold=1000,
    initial_energy="daqenergy",
    check_pulser=True,
):
    """
    Function for selecting events in peaks using raw files,
    to do this it uses the daqenergy to get a first rough selection
    then runs 1 dsp to get a more accurate energy estimate and apply cuts
    returns the indexes of the final events and the peak to which each index corresponds
    """

    if not isinstance(peak_idxs, list):
        peak_idxs = [peak_idxs]
    if not isinstance(kev_widths, list):
        kev_widths = [kev_widths]

    if lh5_path[-1] != "/":
        lh5_path += "/"

    raw_fields = [
        field.replace(lh5_path, "") for field in lh5.ls(raw_files[0], lh5_path)
    ]
    initial_fields = cts.get_keys(raw_fields, [initial_energy])
    initial_fields += ["timestamp"]

    df = lh5.read_as(lh5_path, raw_files, "pd", field_mask=initial_fields)
    df["initial_energy"] = df.eval(initial_energy)

    if pulser_mask is None and check_pulser is True:
        pulser_props = cts.find_pulser_properties(df, energy="initial_energy")
        if len(pulser_props) > 0:
            final_mask = None
            for entry in pulser_props:
                e_cut = (df.initial_energy.values < entry[0] + entry[1]) & (
                    df.initial_energy.values > entry[0] - entry[1]
                )
                if final_mask is None:
                    final_mask = e_cut
                else:
                    final_mask = final_mask | e_cut
            ids = final_mask
            log.debug(f"pulser found: {pulser_props}")
        else:
            log.debug("no_pulser")
            ids = np.zeros(len(df.initial_energy.values), dtype=bool)
        # Get events around peak using raw file values
    elif pulser_mask is not None:
        ids = pulser_mask
    else:
        ids = np.zeros(len(df.initial_energy.values), dtype=bool)

    initial_mask = (df["initial_energy"] > threshold) & (~ids)
    rough_energy = np.array(df["initial_energy"])[initial_mask]
    initial_idxs = np.where(initial_mask)[0]

    guess_kev = 2620 / np.nanpercentile(rough_energy, 99)
    euc_min = threshold / guess_kev * 0.6
    euc_max = 2620 / guess_kev * 1.1
    deuc = 1  # / guess_kev
    hist, bins, var = pgh.get_hist(rough_energy, range=(euc_min, euc_max), dx=deuc)
    detected_peaks_locs, detected_peaks_kev, roughpars = pgc.hpge_find_E_peaks(
        hist,
        bins,
        var,
        np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2103.53, 2614.553]),
    )
    log.debug(f"detected {detected_peaks_kev} keV peaks at {detected_peaks_locs}")

    masks = []
    for peak_idx in peak_idxs:
        peak = peaks_kev[peak_idx]
        kev_width = kev_widths[peak_idx]
        try:
            if peak not in detected_peaks_kev:
                raise ValueError
            detected_peak_idx = np.where(detected_peaks_kev == peak)[0]
            peak_loc = detected_peaks_locs[detected_peak_idx]
            log.info(f"{peak} peak found at {peak_loc}")
            rough_adc_to_kev = roughpars[0]
            e_lower_lim = peak_loc - (1.1 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.1 * kev_width[1]) / rough_adc_to_kev
        except Exception:
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

    if len(idxs) == 0:
        raise ValueError("No events found in energy range")

    input_data = sto.read(f"{lh5_path}", raw_files, idx=idxs, n_rows=len(idxs))[0]

    if isinstance(dsp_config, str):
        with open(dsp_config) as r:
            dsp_config = json.load(r)

    dsp_config["outputs"] = cts.get_keys(
        dsp_config["outputs"], list(cut_parameters)
    ) + [energy_parameter]

    log.debug("Processing data")
    tb_data = opt.run_one_dsp(input_data, dsp_config, db_dict=db_dict)

    if cut_parameters is not None:
        cut_dict = cts.generate_cuts(tb_data, cut_parameters)
        log.debug(f"Cuts are: {cut_dict}")
        log.debug("Loaded Cuts")
        ct_mask = cts.get_cut_indexes(tb_data, cut_dict)
    else:
        ct_mask = np.full(len(tb_data), True, dtype=bool)

    final_events = []
    out_events = []
    for peak_idx in peak_idxs:
        peak = peaks_kev[peak_idx]
        kev_width = kev_widths[peak_idx]

        peak_ids = np.array(idx_list[peak_idx])
        peak_ct_mask = ct_mask[peak_ids]
        peak_ids = peak_ids[peak_ct_mask]

        energy = tb_data[energy_parameter].nda[peak_ids]

        hist, bins, var = pgh.get_hist(
            energy,
            range=(np.floor(np.nanmin(energy)), np.ceil(np.nanmax(energy))),
            dx=peak / (np.nanpercentile(energy, 50)),
        )
        peak_loc = pgh.get_bin_centers(bins)[np.nanargmax(hist)]

        mu, _, _ = pgc.hpge_fit_E_peak_tops(
            hist,
            bins,
            var,
            [peak_loc],
            n_to_fit=7,
        )[
            0
        ][0]

        if mu is None or np.isnan(mu):
            log.debug("Fit failed, using max guess")
            rough_adc_to_kev = peak / peak_loc
            e_lower_lim = peak_loc - (1.5 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.5 * kev_width[1]) / rough_adc_to_kev
            hist, bins, var = pgh.get_hist(
                energy, range=(int(e_lower_lim), int(e_upper_lim)), dx=1
            )
            mu = pgh.get_bin_centers(bins)[np.nanargmax(hist)]

        updated_adc_to_kev = peak / mu
        e_lower_lim = mu - (kev_width[0]) / updated_adc_to_kev
        e_upper_lim = mu + (kev_width[1]) / updated_adc_to_kev
        log.info(f"lower lim is :{e_lower_lim}, upper lim is {e_upper_lim}")

        final_mask = (energy > e_lower_lim) & (energy < e_upper_lim)
        final_events.append(peak_ids[final_mask][:n_events])
        out_events.append(idxs[final_events[-1]])

        log.info(f"{len(peak_ids[final_mask][:n_events])} passed selections for {peak}")
        if len(peak_ids[final_mask]) < 0.5 * n_events:
            log.warning("Less than half number of specified events found")
        elif len(peak_ids[final_mask]) < 0.1 * n_events:
            log.error("Less than 10% number of specified events found")

    out_events = np.unique(np.concatenate(out_events))
    sort_index = np.argsort(np.concatenate(final_events))
    idx_list = get_wf_indexes(sort_index, [len(mask) for mask in final_events])
    return out_events, idx_list


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


OptimiserDimension = namedtuple(
    "OptimiserDimension", "name parameter min_val max_val round unit"
)


class BayesianOptimizer:

    """
    Bayesian optimiser uses Gaussian Process Regressor from sklearn to fit kernel
    to data, takes in a series of init samples for this fit and then calculates
    the next point using the acquisition function specified.
    """

    np.random.seed(55)
    lambda_param = 0.01
    eta_param = 0

    def __init__(self, acq_func, batch_size, kernel=None, sampling_rate=None):
        self.dims = []
        self.current_iter = 0

        self.batch_size = batch_size
        self.iters = 0

        if isinstance(sampling_rate, str):
            self.sampling_rate = ureg.Quantity(sampling_rate)
        elif isinstance(sampling_rate, pint.Quantity):
            self.sampling_rate = sampling_rate
        else:
            if sampling_rate is not None:
                raise TypeError("Unknown type for sampling rate")

        self.gauss_pr = GaussianProcessRegressor(kernel=kernel)
        self.best_samples_ = pd.DataFrame(columns=["x", "y", "ei"])
        self.distances_ = []

        if acq_func == "ei":
            self.acq_function = self._get_expected_improvement
        elif acq_func == "ucb":
            self.acq_function = self._get_ucb
        elif acq_func == "lcb":
            self.acq_function = self._get_lcb

    def add_dimension(
        self, name, parameter, min_val, max_val, round_to_samples=False, unit=None
    ):
        if round_to_samples is True and self.sampling_rate is None:
            raise ValueError("Must provide sampling rate to round to samples")
        if unit is not None:
            unit = ureg.Quantity(unit)
        self.dims.append(
            OptimiserDimension(
                name, parameter, min_val, max_val, round_to_samples, unit
            )
        )

    def get_n_dimensions(self):
        return len(self.dims)

    def add_initial_values(self, x_init, y_init, yerr_init):
        self.x_init = x_init
        self.y_init = y_init
        self.yerr_init = yerr_init

    def _get_expected_improvement(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )

        mean_y = self.gauss_pr.predict(self.x_init)
        min_mean_y = np.min(mean_y)
        z = (mean_y_new[0] - min_mean_y - 1) / (sigma_y_new[0] + 1e-9)
        exp_imp = (mean_y_new[0] - min_mean_y - 1) * norm.cdf(z) + sigma_y_new[
            0
        ] * norm.pdf(z)
        return exp_imp

    def _get_ucb(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        return mean_y_new[0] + self.lambda_param * sigma_y_new[0]

    def _get_lcb(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        return mean_y_new[0] - self.lambda_param * sigma_y_new[0]

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
                fun=self.acq_function,
                x0=x_start,
                bounds=[(dim.min_val, dim.max_val) for dim in self.dims],
                method="L-BFGS-B",
            )
            if response.fun < min_ei:
                min_ei = response.fun
                x_optimal = []
                for y, dim in zip(response.x, self.dims):
                    if dim.round is True and dim.unit is not None:
                        # round so samples is integer

                        x_optimal.append(
                            float(
                                round(
                                    (y * (dim.unit / self.sampling_rate)).to(
                                        "dimensionless"
                                    ),
                                    0,
                                )
                                * (self.sampling_rate / dim.unit)
                            )
                        )
                    else:
                        x_optimal.append(y)
        if x_optimal in self.x_init:
            perturb = np.random.uniform(
                -np.array([(dim.max_val - dim.min_val) / 10 for dim in self.dims]),
                np.array([(dim.max_val - dim.min_val) / 10 for dim in self.dims]),
                (1, len(self.dims)),
            )
            x_optimal += perturb
            new_x_optimal = []
            for y, dim in zip(x_optimal[0], self.dims):
                if dim.round is True and dim.unit is not None:
                    # round so samples is integer
                    new_x_optimal.append(
                        float(
                            round(
                                (y * (dim.unit / self.sampling_rate)).to(
                                    "dimensionless"
                                ),
                                0,
                            )
                            * (self.sampling_rate / dim.unit)
                        )
                    )
                else:
                    new_x_optimal.append(y)
            x_optimal = new_x_optimal
            for i, y in enumerate(x_optimal):
                if y > self.dims[i].max_val:
                    x_optimal[i] = self.dims[i].max_val
                elif y < self.dims[i].min_val:
                    x_optimal[i] = self.dims[i].min_val
        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y, yerr):
        self.x_init = np.append(self.x_init, np.array([x]), axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)
        self.yerr_init = np.append(self.yerr_init, np.array(yerr), axis=0)

    def get_first_point(self):
        y_min_ind = np.nanargmin(self.y_init)
        self.y_min = self.y_init[y_min_ind]
        self.optimal_x = self.x_init[y_min_ind]
        self.optimal_ei = None
        return self.optimal_x, self.optimal_ei

    @ignore_warnings(category=ConvergenceWarning)
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
                value_str = f"{val}*{unit.units:~}"
                if "µ" in value_str:
                    value_str = value_str.replace("µ", "u")
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
        y_err = results["y_err"]
        self._extend_prior_with_posterior_data(
            self.current_x, np.array([y_val]), np.array([y_err])
        )

        if np.isnan(y_val) | np.isnan(y_err):
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

        self.best_samples_ = pd.concat(
            [
                self.best_samples_,
                pd.DataFrame(
                    {"x": self.optimal_x, "y": self.y_min, "ei": self.optimal_ei}
                ),
            ],
            ignore_index=True,
        )

    def get_best_vals(self):
        out_dict = {}
        for i, val in enumerate(self.optimal_x):
            name, parameter, min_val, max_val, rounding, unit = self.dims[i]
            if unit is not None:
                value_str = f"{val}*{unit.units:~}"
                if "µ" in value_str:
                    value_str = value_str.replace("µ", "u")
            else:
                value_str = f"{val}"
            if name not in out_dict.keys():
                out_dict[name] = {parameter: value_str}
            else:
                out_dict[name][parameter] = value_str
        return out_dict

    @ignore_warnings(category=ConvergenceWarning)
    def plot(self, init_samples=None):
        nan_idxs = np.isnan(self.y_init)
        fail_idxs = np.isnan(self.yerr_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        elif len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)
            ys = np.zeros_like(points)
            ys_err = np.zeros_like(points)
            for i, point in enumerate(points):
                ys[i], ys_err[i] = self.gauss_pr.predict(
                    np.array([point]).reshape(1, -1), return_std=True
                )
            fig = plt.figure()

            plt.scatter(np.array(self.x_init), np.array(self.y_init), label="Samples")
            plt.scatter(
                np.array(self.x_init)[fail_idxs],
                np.array(self.y_init)[fail_idxs],
                color="green",
                label="Failed samples",
            )
            plt.fill_between(points, ys - ys_err, ys + ys_err, alpha=0.1)
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
                    label="Init Samples",
                )
            plt.scatter(self.optimal_x[0], self.y_min, color="orange", label="Optimal")

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel("Kernel Value")
            plt.legend()
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

    @ignore_warnings(category=ConvergenceWarning)
    def plot_acq(self, init_samples=None):
        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        elif len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)
            ys = np.zeros_like(points)
            for i, point in enumerate(points):
                ys[i] = self.acq_function(np.array([point]).reshape(1, -1)[0])
            fig = plt.figure()
            plt.plot(points, ys)
            plt.scatter(np.array(self.x_init), np.array(self.y_init), label="Samples")
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
                    label="Init Samples",
                )
            plt.scatter(self.optimal_x[0], self.y_min, color="orange", label="Optimal")

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel("Acquisition Function Value")
            plt.legend()

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
                out_grid[i] = self.acq_function(points[j])
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


def run_multiple_optimisation(
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
        if np.isnan(results_dict["y_val"]):
            log.error(f"Energy optimisation failed for {optimiser.dims[0][0]}")
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
                except Exception:
                    pass
                try:
                    alpha_error_grid[i, j] = grid[i, j][ctc_param]["alpha_err"]
                except Exception:
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

    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    out_grid_err = np.empty(grid_shape)
    n_event_lim = np.array(
        [0.98 * np.nanpercentile(nevents_grid, 50) for nevents_grid in nevents_grids]
    )
    for index, _ in np.ndenumerate(grids[0]):
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
            elif nan_mask[-1] is True or nan_mask[-2] is True:
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
        except Exception:
            out_grid[index] = np.nan
            out_grid_err[index] = np.nan
    return out_grid, out_grid_err


def find_lowest_grid_point_save(grid, err_grid, opt_dict):
    """
    Finds the lowest grid point, if more than one with same value returns shortest filter.
    """
    opt_name = list(opt_dict.keys())[0]
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

    for index, _ in np.ndenumerate(total_lengths):
        for i, param in enumerate(param_list):
            total_lengths[index] += param[index[i]]
    min_val = np.nanmin(grid)
    lowest_ixs = np.where(grid == min_val)
    try:
        fwhm_dict = {"fwhm": min_val, "fwhm_err": err_grid[lowest_ixs][0]}
    except Exception:
        pass
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
    grid_shape = grids[0].shape
    out_grid = np.empty(grid_shape)
    n_event_lim = np.array(
        [0.98 * np.nanpercentile(nevents_grid, 50) for nevents_grid in nevents_grids]
    )
    for index, _ in np.ndenumerate(grids[0]):
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
        except Exception:
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
            plt.title("Qbb")
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
            except Exception:
                alphas = np.nan
    else:
        try:
            alphas = qbb_alphas[ixs[0], ixs[1]][0]
        except Exception:
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
            except Exception:
                full_grids[param] = {ctc_param: output_grid}
            fwhm.update({"alpha": alpha})
            ctc_dict[ctc_param] = fwhm
        full_fwhm_dict[param] = ctc_dict
        full_db_dict.update(db_dict)
    return full_db_dict, full_fwhm_dict, full_grids
