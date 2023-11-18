"""
This module contains the functions for performing the filter optimisation.
This happens with a grid search performed on ENC peak.
"""

import inspect
import json
import logging
import os
import pathlib
import pickle as pkl
import sys
import time
from collections import namedtuple

import lgdo
import lgdo.lh5_store as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from iminuit import Minuit, cost, util
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.interpolate import splev, splrep
from scipy.optimize import minimize

import pygama.math.peak_fitting as pgf
from pygama.math.histogram import get_hist
from pygama.pargen.cuts import generate_cuts, get_cut_indexes
from pygama.pargen.dsp_optimize import run_one_dsp
from pygama.pargen.energy_optimisation import index_data

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def noise_optimization(
    raw_list: list[str],
    dsp_proc_chain: dict,
    par_dsp: dict,
    opt_dict: dict,
    lh5_path: str,
    verbose: bool = False,
    display: int = 0,
) -> dict:
    """
    This function calculates the optimal filter par.
    Parameters
    ----------
    raw_list : str
        raw files to run the macro on
    dsp_proc_chain: str
        Path to minimal dsp config file
    par_dsp: str
        Dictionary with default dsp parameters
    opt_dict: str
        Dictionary with parameters for optimization
    lh5_path:  str
        Name of channel to process, should be name of lh5 group in raw files
    Returns
    -------
    res_dict : dict
    """

    t0 = time.time()
    tb_data = load_data(raw_list, lh5_path, n_events=opt_dict["n_events"])
    t1 = time.time()
    log.info(f"Time to open raw files {t1-t0:.2f} s, n. baselines {len(tb_data)}")
    if verbose:
        print(f"Time to open raw files {t1-t0:.2f} s, n. baselines {len(tb_data)}")

    with open(dsp_proc_chain) as r:
        dsp_proc_chain = json.load(r)

    dsp_data = run_one_dsp(tb_data, dsp_proc_chain)
    cut_dict = generate_cuts(dsp_data, parameters=opt_dict["cut_pars"])
    idxs = get_cut_indexes(dsp_data, cut_dict)
    tb_data = index_data(tb_data, idxs)
    log.info(f"... {len(tb_data)} baselines after cuts")
    if verbose:
        print(f"... {len(tb_data)} baselines after cuts")

    samples = np.arange(opt_dict["start"], opt_dict["stop"], opt_dict["step"])
    samples_val = np.arange(opt_dict["start"], opt_dict["stop"], opt_dict["step_val"])

    opt_dict_par = opt_dict["optimization"]

    res_dict = {}
    if display > 0:
        freq, pow_spectrum = calculate_fft(tb_data)
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        ax.plot(freq, pow_spectrum)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (MHz)", ha="right", x=1)
        ax.set_ylabel(f"power spectral density", ha="right", y=1)

        plot_dict = {}
        plot_dict["nopt"] = {}
        plot_dict["nopt"]["fft"] = {}
        plot_dict["nopt"]["fft"]["frequency"] = freq
        plot_dict["nopt"]["fft"]["pow_spectrum"] = pow_spectrum
        plot_dict["nopt"]["fft"]["fig"] = fig

    ene_pars = [par for par in opt_dict_par.keys()]
    for ene_par in ene_pars:
        log.info(f"\nRunning optimization for {ene_par} filter")
        if verbose:
            print(f"\nRunning optimization for {ene_par} filter")
        wf_par = opt_dict_par[ene_par]["waveform_out"]
        dict_str = opt_dict_par[ene_par]["dict_str"]
        filter_par = opt_dict_par[ene_par]["filter_par"]
        ene_str = opt_dict_par[ene_par]["ene_str"]
        if display > 0:
            plot_dict["nopt"][dict_str] = {}
            par_dict_plot = plot_dict["nopt"][dict_str]

        dsp_proc_chain["outputs"] = [ene_str]
        sample_list, fom_list, fom_err_list = [], [], []
        for i, x in enumerate(samples):
            x = f"{x:.1f}"
            log.info(f"\nCase {i}, par = {x} us")
            if verbose:
                print(f"\nCase {i}, par = {x} us")
            par_dsp[lh5_path][dict_str][filter_par] = f"{x}*us"

            t2 = time.time()
            dsp_data = run_one_dsp(tb_data, dsp_proc_chain, db_dict=par_dsp[lh5_path])
            log.info(f"Time to process dsp data {time.time()-t2:.2f} s")
            if verbose:
                print(f"Time to process dsp data {time.time()-t2:.2f} s")
            energies = dsp_data[ene_str].nda

            if opt_dict["perform_fit"]:
                fom_results = simple_gaussian_fit(energies, dx=opt_dict["dx"])
            else:
                fom_results = calculate_spread(
                    energies,
                    opt_dict["percentile_low"],
                    opt_dict["percentile_high"],
                    opt_dict["n_bootstrap_samples"],
                )
            sample_list.append(float(x))
            fom_list.append(fom_results["fom"])
            fom_err_list.append(fom_results["fom_err"])
            if display > 0:
                par_dict_plot[x] = {}
                par_dict_plot[x]["energies"] = energies
                par_dict_plot[x]["fom"] = fom_results["fom"]
                par_dict_plot[x]["fom_err"] = fom_results["fom_err"]
        sample_list = np.array(sample_list)
        fom_list = np.array(fom_list)
        fom_err_list = np.array(fom_err_list)

        guess_par = sample_list[np.nanargmin(fom_list)]
        if verbose:
            print(f"guess par: {guess_par:.2f} us")

        tck = splrep(sample_list, fom_list, k=opt_dict["fit_deg"])

        def spl_func(x_val):
            return splev(x_val, tck)

        result = minimize(spl_func, guess_par)
        best_par = result.x[0]
        if (best_par < np.min(sample_list)) or (best_par > np.max(sample_list)):
            log.info(
                f"Par from minimization not accepted {best_par:.2f}, setting par to guess"
            )
            if verbose:
                print(
                    f"Par from minimization not accepted {best_par:.2f}, setting par to guess"
                )
            best_par = guess_par

        best_val = spl_func(best_par)

        b_best_pars = np.zeros(opt_dict["n_bootstrap_samples"])
        for i in range(opt_dict["n_bootstrap_samples"]):
            indices = np.random.choice(len(sample_list), len(sample_list), replace=True)
            b_sample_list = sample_list[indices]
            b_fom_list = fom_list[indices]
            b_best_pars[i] = b_sample_list[np.nanargmin(b_fom_list)]
        best_par_err = np.std(b_best_pars)
        log.info(f"best par: {best_par:.2f} ± {best_par_err:.2f} us")
        if verbose:
            print(f"best par: {best_par:.2f} ± {best_par_err:.2f} us")

        par_dict_plot["best_par"] = best_par
        par_dict_plot["best_par_err"] = best_par_err
        par_dict_plot["best_val"] = best_val

        res_dict[dict_str] = {
            filter_par: f"{best_par:.2f}*us",
            f"{filter_par}_err": f"{best_par_err:.2f}*us",
        }

        if display > 0:
            plot_range = opt_dict["plot_range"]
            fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
            for i, x in enumerate(sample_list):
                x = f"{x:.1f}"
                energies = par_dict_plot[x]["energies"]
                par_dict_plot[x].pop("energies")
                hist, bins, var = get_hist(
                    energies, range=plot_range, dx=opt_dict["dx"]
                )
                bc = (bins[:-1] + bins[1:]) / 2.0
                string_res = (
                    f"par = {x} us, FOM = {fom_list[i]:.3f} ± {fom_err_list[i]:.3f} ADC"
                )
                ax.plot(bc, hist, ds="steps", label=string_res)
                log.info(string_res)
                if verbose:
                    print(string_res)
            ax.set_xlabel("energy (ADC)")
            ax.set_ylabel("counts")
            ax.legend(loc="upper right")
            par_dict_plot["distribution"] = fig
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
            ax.errorbar(
                sample_list,
                fom_list,
                yerr=fom_err_list,
                color="b",
                fmt="x",
                ms=4,
                ls="",
                capsize=4,
                label="samples",
            )
            ax.plot(samples_val, spl_func(samples_val), "k:", label="fit")
            ax.errorbar(
                best_par,
                best_val,
                xerr=best_par_err,
                color="r",
                fmt="o",
                ms=6,
                ls="",
                capsize=4,
                label=rf"best par: {best_par:.2f} ± {best_par_err:.2f} $\mu$s",
            )
            ax.set_xlabel(rf"{ene_par} parameter ($\mu$s)")
            ax.set_ylabel("FOM (ADC)")
            ax.legend()
            if display > 1:
                plt.show()
            else:
                plt.close()
            par_dict_plot["optimization"] = fig

    log.info(f"Time to complete the optimization {time.time()-t0:.2f} s")
    if verbose:
        print(f"Time to complete the optimization {time.time()-t0:.2f} s")
    if display > 0:
        return res_dict, plot_dict
    else:
        return res_dict


def load_data(
    raw_list: list[str],
    lh5_path: str,
    bls: bool = True,
    n_events: int = 10000,
    threshold: int = 200,
) -> lgdo.Table:
    sto = lh5.LH5Store()

    energies = sto.read_object(f"{lh5_path}/raw/daqenergy", raw_list)[0]

    if bls:
        idxs = np.where(energies.nda == 0)[0]
    else:
        idxs = np.where(energies.nda > threshold)[0]

    waveforms = sto.read_object(
        f"{lh5_path}/raw/waveform", raw_list, n_rows=n_events, idx=idxs
    )[0]
    daqenergy = sto.read_object(
        f"{lh5_path}/raw/daqenergy", raw_list, n_rows=n_events, idx=idxs
    )[0]
    baseline = sto.read_object(
        f"{lh5_path}/raw/baseline", raw_list, n_rows=n_events, idx=idxs
    )[0]

    tb_data = lh5.Table(
        col_dict={"waveform": waveforms, "daqenergy": daqenergy, "baseline": baseline}
    )

    return tb_data


def calculate_spread(energies, percentile_low, percentile_high, n_samples):
    spreads = np.zeros(n_samples)
    for i in range(n_samples):
        resampled = np.random.choice(energies, size=len(energies), replace=True)
        spread = np.percentile(resampled, percentile_high) - np.percentile(
            resampled, percentile_low
        )
        spreads[i] = spread

    mean_spread = np.mean(spreads)
    std_spread = np.std(spreads, ddof=1) / np.sqrt(n_samples)

    results = {}
    results["fom"] = mean_spread
    results["fom_err"] = std_spread
    return results


def simple_gaussian_fit(energies, dx=1, sigma_thr=4, allowed_p_val=1e-20):
    fit_range = [np.percentile(energies, 0.2), np.percentile(energies, 99.8)]

    hist, bins, var = get_hist(energies, range=fit_range, dx=dx)
    guess, bounds = simple_gaussian_guess(hist, bins, pgf.extended_gauss_pdf)
    fit_range = [guess[0] - sigma_thr * guess[1], guess[0] + sigma_thr * guess[1]]

    energies_fit = energies[(energies > fit_range[0]) & (energies < fit_range[1])]
    pars, errs, cov = pgf.fit_unbinned(
        pgf.extended_gauss_pdf,
        energies_fit,
        guess=guess,
        bounds=bounds,
    )

    mu, mu_err = pars[0], errs[0]
    fwhm = pars[1] * 2 * np.sqrt(2 * np.log(2))
    fwhm_err = errs[1] * 2 * np.sqrt(2 * np.log(2))

    hist, bins, var = get_hist(energies_fit, range=fit_range, dx=dx)
    gof_pars = pars
    gof_pars[2] *= dx
    chisq, dof = pgf.goodness_of_fit(
        hist, bins, None, pgf.gauss_pdf, gof_pars, method="Pearson"
    )
    p_val = scipy.stats.chi2.sf(chisq, dof + len(gof_pars))

    if (
        sum(sum(c) if c is not None else 0 for c in cov[:3, :][:, :3]) == np.inf
        or sum(sum(c) if c is not None else 0 for c in cov[:3, :][:, :3]) == 0
        or np.isnan(sum(sum(c) if c is not None else 0 for c in cov[:3, :][:, :3]))
    ):
        log.debug("fit failed, cov estimation failed")
        fit_failed = True
    elif (np.abs(np.array(errs)[:3] / np.array(pars)[:3]) < 1e-7).any() or np.isnan(
        np.array(errs)[:3]
    ).any():
        log.debug("fit failed, parameter error too low")
        fit_failed = True
    elif p_val < allowed_p_val or np.isnan(p_val):
        log.debug("fit failed, parameter error too low")
        fit_failed = True
    else:
        fit_failed = False

    if fit_failed:
        log.debug(f"Returning values from guess")
        mu = guess[0]
        mu_err = 0
        fwhm = guess[1] * 2 * np.sqrt(2 * np.log(2))
        fwhm_err = 0

    results = {
        "pars": pars[:3],
        "errors": errs[:3],
        "covariance": cov[:3],
        "mu": mu,
        "mu_err": mu_err,
        "fom": fwhm,
        "fom_err": fwhm_err,
        "chisq": chisq / dof,
        "p_val": p_val,
    }
    return results


def simple_gaussian_guess(hist, bins, func, toll=0.2):
    max_idx = np.argmax(hist)
    mu = bins[max_idx]
    max_amp = np.max(hist)

    idx = np.where(hist > max_amp / 2)
    ilo, ihi = idx[0][0], idx[0][-1]

    sigma = (bins[ihi] - bins[ilo]) / 2.355

    if sigma == 0:
        log.debug("error in sigma evaluation, using 2*(bin width) as sigma")
        sigma = 2 * (bins[1] - bins[0])

    dx = np.diff(bins)[0]
    n_bins_range = int((4 * sigma) // dx)

    min_idx = max_idx - n_bins_range
    max_idx = max_idx + n_bins_range
    min_idx = max(0, min_idx)
    max_idx = min(len(hist), max_idx)

    n_sig = np.sum(hist[min_idx:max_idx])

    guess = [mu, sigma, n_sig]
    bounds = [
        (mu - sigma, mu + sigma),
        (sigma - sigma * toll, sigma + sigma * toll),
        (n_sig + n_sig * toll, n_sig + n_sig * toll),
    ]

    for i, par in enumerate(inspect.getfullargspec(func)[0][1:]):
        if par == "lower_range" or par == "upper_range":
            guess.append(np.inf)
            bounds.append(None)
        elif par == "n_bkg" or par == "hstep" or par == "components":
            guess.append(0)
            bounds.append(None)
    return guess, bounds


def calculate_fft(tb_data, cut=1):
    bls = tb_data["waveform"].values.nda
    nev, size = bls.shape

    sample_time_us = float(tb_data["waveform"].dt.nda[0]) / 1000
    sampling_rate = 1 / sample_time_us
    fft_size = size // 2 + 1

    frequency = np.linspace(0, sampling_rate / 2, fft_size)
    power_spectrum = np.zeros(fft_size, dtype=np.float64)

    for bl in bls:
        fft = np.fft.rfft(bl)
        abs_fft = np.abs(fft)
        power_spectrum += np.square(abs_fft)
    power_spectrum /= nev

    return frequency[cut:], power_spectrum[cut:]
