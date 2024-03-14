"""
This module contains the functions for performing the filter optimisation.
This happens with a grid search performed on ENC peak.
"""

import logging
import time

import lgdo
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import BSpline, splev, splrep
from scipy.optimize import minimize

from pygama.math.binned_fitting import goodness_of_fit
from pygama.math.distributions import gauss_on_uniform
from pygama.math.histogram import get_hist
from pygama.math.unbinned_fitting import fit_unbinned
from pygama.pargen.dsp_optimize import run_one_dsp

log = logging.getLogger(__name__)


def noise_optimization(
    tb_data: lgdo.Table,
    dsp_proc_chain: dict,
    par_dsp: dict,
    opt_dict: dict,
    lh5_path: str,
    display: int = 0,
) -> dict:
    """
    This function calculates the optimal filter par.
    Parameters
    ----------
    tb_data : str
        raw table to run the macro on
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

    samples = np.arange(opt_dict["start"], opt_dict["stop"], opt_dict["step"])
    samples_val = np.arange(opt_dict["start"], opt_dict["stop"], opt_dict["step_val"])

    opt_dict_par = opt_dict["optimization"]

    res_dict = {}
    if display > 0:
        dsp_data = run_one_dsp(tb_data, dsp_proc_chain, db_dict=par_dsp)
        psd = np.mean(dsp_data["wf_psd"].values.nda, axis=0)
        sample_us = float(dsp_data["wf_presum"].dt.nda[0]) / 1000
        freq = np.linspace(0, (1 / sample_us) / 2, len(psd))
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
        ax.plot(freq, psd)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("power spectral density")

        plot_dict = {}
        plot_dict["nopt"] = {"fft": {"frequency": freq, "psd": psd, "fig": fig}}
        plt.close()

    result_dict = {}
    ene_pars = [par for par in opt_dict_par.keys()]
    log.info(f"\nRunning optimization for {ene_pars}")
    for i, x in enumerate(samples):
        x = f"{x:.1f}"
        log.info(f"\nCase {i}, par = {x} us")
        for ene_par in ene_pars:
            dict_str = opt_dict_par[ene_par]["dict_str"]
            filter_par = opt_dict_par[ene_par]["filter_par"]
            if dict_str in par_dsp:
                par_dsp[dict_str].update({filter_par: f"{x}*us"})
            else:
                par_dsp[dict_str] = {filter_par: f"{x}*us"}

        t1 = time.time()
        dsp_data = run_one_dsp(tb_data, dsp_proc_chain, db_dict=par_dsp)
        log.info(f"Time to process dsp data {time.time()-t1:.2f} s")

        for ene_par in ene_pars:
            dict_str = opt_dict_par[ene_par]["dict_str"]
            ene_str = opt_dict_par[ene_par]["ene_str"]
            if dict_str not in result_dict:
                result_dict[dict_str] = {}
            par_dict_res = result_dict[dict_str]

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

            par_dict_res[x] = {}
            par_dict_res[x]["energies"] = energies
            par_dict_res[x]["fom"] = fom_results["fom"]
            par_dict_res[x]["fom_err"] = fom_results["fom_err"]

    for ene_par in ene_pars:
        log.info(f"\nOptimization for {ene_par}")
        dict_str = opt_dict_par[ene_par]["dict_str"]
        par_dict_res = result_dict[dict_str]
        sample_list = np.array([float(x) for x in result_dict[dict_str].keys()])
        fom_list = np.array(
            [result_dict[dict_str][x]["fom"] for x in result_dict[dict_str].keys()]
        )
        fom_err_list = np.array(
            [result_dict[dict_str][x]["fom_err"] for x in result_dict[dict_str].keys()]
        )

        guess_par = sample_list[np.nanargmin(fom_list)]

        tck = splrep(sample_list, fom_list, k=opt_dict["fit_deg"])
        tck = BSpline(tck[0], tck[1], tck[2])

        result = minimize(splev, guess_par, args=(tck))
        best_par = result.x[0]
        if (best_par < np.min(sample_list)) or (best_par > np.max(sample_list)):
            log.info(
                f"Par from minimization not accepted {best_par:.2f}, setting par to guess"
            )
            best_par = guess_par

        best_val = splev(best_par, tck)

        b_best_pars = np.zeros(opt_dict["n_bootstrap_samples"])
        for i in range(opt_dict["n_bootstrap_samples"]):
            indices = np.random.choice(len(sample_list), len(sample_list), replace=True)
            b_sample_list = sample_list[indices]
            b_fom_list = fom_list[indices]
            b_best_pars[i] = b_sample_list[np.nanargmin(b_fom_list)]
        best_par_err = np.std(b_best_pars)
        log.info(f"best par: {best_par:.2f} ± {best_par_err:.2f} us")

        par_dict_res["best_par"] = best_par
        par_dict_res["best_par_err"] = best_par_err
        par_dict_res["best_val"] = best_val

        filter_par = opt_dict_par[ene_par]["filter_par"]
        res_dict[dict_str] = {
            filter_par: f"{best_par:.2f}*us",
            f"{filter_par}_err": f"{best_par_err:.2f}*us",
        }

        if display > 0:
            plot_range = opt_dict["plot_range"]
            fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
            for i, x in enumerate(sample_list):
                x = f"{x:.1f}"
                energies = par_dict_res[x]["energies"]
                par_dict_res[x].pop("energies")
                hist, bins, var = get_hist(
                    energies, range=plot_range, dx=opt_dict["dx"]
                )
                bc = (bins[:-1] + bins[1:]) / 2.0
                string_res = (
                    f"par = {x} us, FOM = {fom_list[i]:.3f} ± {fom_err_list[i]:.3f} ADC"
                )
                ax.plot(bc, hist, ds="steps", label=string_res)
                log.info(string_res)
            ax.set_xlabel("energy (ADC)")
            ax.set_ylabel("counts")
            ax.legend(loc="upper right")
            par_dict_res["distribution"] = fig
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
            ax.plot(samples_val, splev(samples_val, tck), "k:", label="fit")
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
            par_dict_res["optimization"] = fig
            plot_dict["nopt"][dict_str] = par_dict_res

    log.info(f"Time to complete the optimization {time.time()-t0:.2f} s")
    if display > 0:
        return res_dict, plot_dict
    else:
        return res_dict


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
    guess, bounds = simple_gaussian_guess(hist, bins, gauss_on_uniform)
    fit_range = [guess[0] - sigma_thr * guess[1], guess[0] + sigma_thr * guess[1]]

    energies_fit = energies[(energies > fit_range[0]) & (energies < fit_range[1])]
    pars, errs, cov = fit_unbinned(
        gauss_on_uniform.pdf_ext,
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
    chisq, dof = goodness_of_fit(
        hist, bins, None, gauss_on_uniform.pdf_norm, gof_pars, method="Pearson"
    )
    p_val = scipy.stats.chi2.sf(chisq, dof + len(gof_pars))

    if (
        sum(sum(c) if c is not None else 0 for c in cov[2:, :][:, 2:]) == np.inf
        or sum(sum(c) if c is not None else 0 for c in cov[2:, :][:, 2:]) == 0
        or np.isnan(sum(sum(c) if c is not None else 0 for c in cov[2:, :][:, 2:]))
    ):
        log.debug("fit failed, cov estimation failed")
        fit_failed = True
    elif (np.abs(np.array(errs)[2:] / np.array(pars)[2:]) < 1e-7).any() or np.isnan(
        np.array(errs)[2:]
    ).any():
        log.debug("fit failed, parameter error too low")
        fit_failed = True
    elif p_val < allowed_p_val or np.isnan(p_val):
        log.debug("fit failed, parameter error too low")
        fit_failed = True
    else:
        fit_failed = False

    if fit_failed:
        log.debug("Returning values from guess")
        mu = guess[0]
        mu_err = 0
        fwhm = guess[1] * 2 * np.sqrt(2 * np.log(2))
        fwhm_err = 0

    results = {
        "pars": pars,
        "errors": errs,
        "covariance": cov,
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

    guess = {"mu": mu, "sigma": sigma, "n_sig": n_sig}
    bounds = {
        "mu": (mu - sigma, mu + sigma),
        "sigma": (sigma - sigma * toll, sigma + sigma * toll),
        "n_sig": (n_sig + n_sig * toll, n_sig + n_sig * toll),
    }

    for par in func.required_args():
        if par == "x_lo" or par == "x_hi":
            guess[par] = np.inf
            bounds[par] = None
        elif par == "n_bkg" or par == "hstep":
            guess[par] = 0
            bounds[par] = None
    return guess, bounds
