"""
This module provides a routine for running the energy calibration on Th data
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from datetime import datetime

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.energy_cal as cal

log = logging.getLogger(__name__)


def fwhm_slope(x: np.array, m0: float, m1: float, m2: float = None) -> np.array:
    """
    Fit the energy resolution curve
    """
    if m2 is None:
        return np.sqrt(m0 + m1 * x)
    else:
        return np.sqrt(m0 + m1 * x + m2 * x**2)


def load_data(
    files: list[str],
    lh5_path: str,
    energy_params: list[str],
    hit_dict: dict = {},
    cut_parameters: list[str] = ["bl_mean", "bl_std", "pz_std"],
    display=0,
) -> dict[str, np.ndarray]:

    df = lh5.load_dfs(files, ["timestamp", "trapTmax"], lh5_path)
    pulser_props = cts.find_pulser_properties(df, energy="trapTmax")
    if display > 0:
        plot_dict = {}
    if len(pulser_props) > 0:
        out_df = cts.tag_pulsers(df, pulser_props, window=0.001)
        ids = ~(out_df.isPulser == 1)
        log.debug(f"pulser found: {pulser_props}")

        if display > 0:
            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 10

            fig = plt.figure()
            plt.hist(df["trapTmax"], bins=10000, histtype="step", label="With Pulser")
            plt.hist(
                df["trapTmax"][ids], bins=10000, histtype="step", label="Pulser Removed"
            )
            plt.xlim([0, np.nanpercentile(df["trapTmax"], 99)])
            plt.xlabel("Energy (ADC)")
            plt.ylabel("Counts")
            plt.legend(loc="upper right")
            plot_dict["pulser"] = fig
            if display > 1:
                plt.show()
            else:
                plt.close()

    else:
        ids = np.ones(len(df), dtype=bool)
        log.debug(f"no pulser found")

    if len(hit_dict.keys()) == 0:
        try:
            energy_dict = lh5.load_nda(files, energy_params, lh5_path)
        except RuntimeError:
            energy_params = [
                energy_param.split("_")[0] for energy_param in energy_params
            ]
            energy_dict = lh5.load_nda(files, energy_params, lh5_path)
            energy_dict["timestamp"] = df["timestamp"][ids].to_numpy()

    else:
        sto = lh5.LH5Store()
        energy_dict = {"timestamp": df["timestamp"][ids].to_numpy()}
        table = sto.read_object(lh5_path, files)[0]
        df = table.eval(hit_dict).get_dataframe()
        cut_parameters = cts.get_keys(table, cut_parameters)

        for param in energy_params:
            if param in df:
                energy_dict[param] = df[param][ids].to_numpy()
            else:
                dat = lh5.load_nda(files, [param], lh5_path)[param]
                energy_dict.update({param: dat[ids]})
        if cut_parameters is not None:
            for param in cut_parameters:
                if param in df:
                    energy_dict[param] = df[param][ids].to_numpy()
                else:
                    dat = lh5.load_nda(files, [param], lh5_path)[param]
                    energy_dict.update({param: dat[ids]})
    if display > 0:
        return energy_dict, plot_dict
    else:
        return energy_dict


def energy_cal_th(
    files: list[str],
    energy_params: list[str],
    hit_dict: dict = {},
    cut_parameters: dict[str, int] = {"bl_mean": 4, "bl_std": 4, "pz_std": 4},
    lh5_path: str = "dsp",
    display: int = 0,
    guess_keV: float | None = None,
    threshold: int = 0,
    p_val: float = 0,
    n_events: int = 15000,
    final_cut_field: str = "is_valid_cal",
    deg: int = 1,
) -> tuple(dict, dict):

    """
    This is an example script for calibrating Th data.
    """

    if isinstance(energy_params, str):
        energy_params = [energy_params]

    ####################
    # Start the analysis
    ####################

    log.debug(f"{len(files)} files found")
    log.debug("Loading and applying charge trapping corr...")
    if display > 0:
        uncal_pass, plot_dict = load_data(
            files,
            lh5_path,
            energy_params,
            hit_dict,
            cut_parameters=list(cut_parameters) if cut_parameters is not None else None,
            display=display,
        )
    else:
        uncal_pass = load_data(
            files,
            lh5_path,
            energy_params,
            hit_dict,
            cut_parameters=list(cut_parameters) if cut_parameters is not None else None,
        )

    total_events = len(uncal_pass[energy_params[0]])
    log.debug("Done")
    if cut_parameters is not None:
        cut_dict = cts.generate_cuts(uncal_pass, cut_parameters)
        hit_dict.update(
            cts.cut_dict_to_hit_dict(cut_dict, final_cut_field=final_cut_field)
        )
        mask = cts.get_cut_indexes(uncal_pass, cut_dict)
        uncal_fail = {}
        for param in uncal_pass:
            uncal_fail[param] = uncal_pass[param][~mask]
            uncal_pass[param] = uncal_pass[param][mask]
        events_pqc = len(uncal_pass[energy_params[0]])
        log.debug(f"{events_pqc} events pass")

    glines = [
        583.191,
        727.330,
        860.564,
        1592.53,
        1620.50,
        2103.53,
        2614.50,
    ]  # gamma lines used for calibration
    range_keV = [
        (20, 20),
        (30, 30),
        (30, 30),
        (40, 25),
        (25, 40),
        (40, 40),
        (60, 60),
    ]  # side bands width
    funcs = [
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
        pgf.extended_radford_pdf,
    ]
    gof_funcs = [
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
        pgf.radford_pdf,
    ]
    output_dict = {}
    for energy_param in energy_params:
        kev_ranges = range_keV.copy()
        if guess_keV is None:
            guess_keV = 2620 / np.nanpercentile(
                uncal_pass[energy_param][uncal_pass[energy_param] > threshold], 99
            )

        log.debug(f"Find peaks and compute calibration curve for {energy_param}")
        log.debug(f"Guess is {guess_keV:.3f}")

        try:
            pars, cov, results = cal.hpge_E_calibration(
                uncal_pass[energy_param],
                glines,
                guess_keV,
                deg=deg,
                range_keV=range_keV,
                funcs=funcs,
                gof_funcs=gof_funcs,
                n_events=n_events,
                allowed_p_val=p_val,
                simplex=True,
                verbose=False,
            )
            pk_pars = results["pk_pars"]
            found_peaks = results["got_peaks_locs"]
            fitted_peaks = results["fitted_keV"]
        except np.linalg.LinAlgError:
            found_peaks = np.array([])
            fitted_peaks = np.array([])

        for i, peak in enumerate(glines):
            if peak not in fitted_peaks:
                kev_ranges[i] = (kev_ranges[i][0] - 5, kev_ranges[i][1] - 5)
        for i, peak in enumerate(glines):
            if peak not in fitted_peaks:
                kev_ranges[i] = (kev_ranges[i][0] - 5, kev_ranges[i][1] - 5)
        for i, peak in enumerate(fitted_peaks):
            try:
                if results["pk_fwhms"][:, 1][i] / results["pk_fwhms"][:, 0][i] > 0.05:
                    index = np.where(glines == peak)[0][0]
                    kev_ranges[i] = (kev_ranges[index][0] - 5, kev_ranges[index][1] - 5)
            except:
                pass

        try:
            pars, cov, results = cal.hpge_E_calibration(
                uncal_pass[energy_param],
                glines,
                guess_keV,
                deg=deg,
                range_keV=kev_ranges,
                funcs=funcs,
                gof_funcs=gof_funcs,
                n_events=n_events,
                allowed_p_val=p_val,
                simplex=True,
                verbose=False,
            )
        except np.linalg.LinAlgError:
            pars = None
        log.debug("done")
        if pars is None:
            log.info("Calibration failed")
            pars, cov, results = cal.hpge_E_calibration(
                uncal_pass[energy_param],
                glines,
                guess_keV,
                deg=deg,
                range_keV=kev_ranges,
                funcs=funcs,
                gof_funcs=gof_funcs,
                n_events=n_events,
                allowed_p_val=0,
                simplex=True,
                verbose=False,
            )
            if deg == 1:
                hit_dict[f"{energy_param}_cal"] = {
                    "expression": f"a*{energy_param}+b",
                    "parameters": {"a": pars[0], "b": pars[1]},
                }
            elif deg == 0:
                hit_dict[f"{energy_param}_cal"] = {
                    "expression": f"a*{energy_param}",
                    "parameters": {"a": pars[0]},
                }
            elif deg == 2:
                hit_dict[f"{energy_param}_cal"] = {
                    "expression": f"a*{energy_param}**2 +b*{energy_param}+c",
                    "parameters": {"a": pars[0], "b": pars[1], "c": pars[2]},
                }
            else:
                hit_dict[f"{energy_param}_cal"] = {}
                log.warning(f"hit_dict not implemented for deg = {deg}")

            fitted_peaks = results["fitted_keV"]
            fwhms = results["pk_fwhms"][:, 0]
            dfwhms = results["pk_fwhms"][:, 1]

            #####
            # Remove the Tl SEP and DEP from calibration if found
            fwhm_peaks = np.array([], dtype=np.float32)
            all_peaks = np.array([], dtype=np.float32)
            indexes = []
            for i, peak in enumerate(fitted_peaks):
                all_peaks = np.append(all_peaks, peak)
                if peak == 2103.53:
                    log.info(f"Tl SEP found at index {i}")
                    indexes.append(i)
                    continue
                elif peak == 1592.53:
                    log.info(f"Tl DEP found at index {i}")
                    indexes.append(i)
                    continue
                elif np.isnan(dfwhms[i]):
                    log.info(f"{peak} failed")
                    indexes.append(i)
                    continue
                else:
                    fwhm_peaks = np.append(fwhm_peaks, peak)
            fit_fwhms = np.delete(fwhms, [indexes])
            fit_dfwhms = np.delete(dfwhms, [indexes])
            #####
            param_guess = [0.2, 0.001]
            param_bounds = (0, [10.0, 1.0])
            fit_pars, fit_covs = curve_fit(
                fwhm_slope,
                fwhm_peaks,
                fit_fwhms,
                sigma=fit_dfwhms,
                p0=param_guess,
                bounds=param_bounds,
                absolute_sigma=True,
            )

            output_dict[f"{energy_param}_cal"] = {
                "Qbb_fwhm": np.nan,
                "Qbb_fwhm_err": np.nan,
                "2.6_fwhm": np.nan,
                "2.6_fwhm_err": np.nan,
                "eres_pars": fit_pars.tolist(),
                "fitted_peaks": np.nan,
                "fwhms": np.nan,
                "peak_fit_pars": np.nan,
            }
            continue
        log.info(f"Calibration pars are {pars}")
        if deg == 1:
            hit_dict[f"{energy_param}_cal"] = {
                "expression": f"a*{energy_param}+b",
                "parameters": {"a": pars[0], "b": pars[1]},
            }
        elif deg == 0:
            hit_dict[f"{energy_param}_cal"] = {
                "expression": f"a*{energy_param}",
                "parameters": {"a": pars[0]},
            }
        elif deg == 2:
            hit_dict[f"{energy_param}_cal"] = {
                "expression": f"a*{energy_param}**2 +b*{energy_param}+c",
                "parameters": {"a": pars[0], "b": pars[1], "c": pars[2]},
            }
        else:
            hit_dict[f"{energy_param}_cal"] = {}
            log.warning(f"hit_dict not implemented for deg = {deg}")
        fitted_peaks = results["fitted_keV"]
        fitted_funcs = []
        fitted_gof_funcs = []
        for i, peak in enumerate(glines):
            if peak in fitted_peaks:
                fitted_funcs.append(funcs[i])
                fitted_gof_funcs.append(gof_funcs[i])

        ecal_pass = pgf.poly(uncal_pass[energy_param], pars)
        if cut_parameters is not None:
            ecal_fail = pgf.poly(uncal_fail[energy_param], pars)

        pk_pars = results["pk_pars"]
        pk_covs = results["pk_covs"]

        pk_rs_dict = {peak: pk_pars[i].tolist() for i, peak in enumerate(fitted_peaks)}

        peaks_kev = results["got_peaks_keV"]

        pk_ranges = results["pk_ranges"]
        p_vals = results["pk_pvals"]
        mus = [
            pgf.get_mu_func(func_i, pars_i)
            for func_i, pars_i in zip(fitted_funcs, pk_pars)
        ]

        fwhms = results["pk_fwhms"][:, 0]
        dfwhms = results["pk_fwhms"][:, 1]

        #####
        # Remove the Tl SEP and DEP from calibration if found
        fwhm_peaks = np.array([], dtype=np.float32)
        all_peaks = np.array([], dtype=np.float32)
        indexes = []
        for i, peak in enumerate(fitted_peaks):
            all_peaks = np.append(all_peaks, peak)
            if peak == 2103.53:
                log.info(f"Tl SEP found at index {i}")
                indexes.append(i)
                continue
            elif peak == 1592.53:
                log.info(f"Tl DEP found at index {i}")
                indexes.append(i)
                continue
            elif np.isnan(dfwhms[i]):
                log.info(f"{peak} failed")
                indexes.append(i)
                continue
            else:
                fwhm_peaks = np.append(fwhm_peaks, peak)
        fit_fwhms = np.delete(fwhms, [indexes])
        fit_dfwhms = np.delete(dfwhms, [indexes])
        fit_mus = np.delete(mus, [indexes])
        #####

        for i, peak in enumerate(fwhm_peaks):
            log.info(
                f"FWHM of {peak} keV peak is: {fit_fwhms[i]:1.2f} +- {fit_dfwhms[i]:1.2f} keV"
            )
        param_guess = [0.2, 0.001]
        param_bounds = (0, [10.0, 1.0])
        fit_pars, fit_covs = curve_fit(
            fwhm_slope,
            fwhm_peaks,
            fit_fwhms,
            sigma=fit_dfwhms,
            p0=param_guess,
            bounds=param_bounds,
            absolute_sigma=True,
        )

        rng = np.random.default_rng(1)
        pars_b = rng.multivariate_normal(fit_pars, fit_covs, size=1000)
        fits = np.array([fwhm_slope(fwhm_peaks, *par_b) for par_b in pars_b])
        qbb_vals = np.array([fwhm_slope(2039.0, *par_b) for par_b in pars_b])
        qbb_err = np.nanstd(qbb_vals)

        log.info(f"FWHM curve fit: {fit_pars}")
        predicted_fwhms = fwhm_slope(fwhm_peaks, *fit_pars)
        log.info(f"FWHM fit values: {predicted_fwhms}")
        fit_qbb = fwhm_slope(2039.0, *fit_pars)
        log.info(f"FWHM energy resolution at Qbb: {fit_qbb:1.2f} +- {qbb_err:1.2f} keV")

        if display > 0:
            plot_dict_param = {}
            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 12
            selection = (ecal_pass > 2560) & (ecal_pass < 2660)
            fig_sta = plt.figure()
            plt.hist2d(
                uncal_pass["timestamp"][selection],
                ecal_pass[selection],
                bins=[100, np.arange(2560, 2660, 1)],
                norm=LogNorm(),
            )

            ticks, labels = plt.xticks()
            plt.xlabel(
                f"Time starting : {datetime.utcfromtimestamp(ticks[0]).strftime('%d/%m/%y %H:%M')}"
            )
            plt.ylabel("Energy(keV)")
            plt.ylim([2600, 2630])

            plt.xticks(
                ticks,
                [datetime.utcfromtimestamp(tick).strftime("%H:%M") for tick in ticks],
            )
            plot_dict_param["cal_stability"] = fig_sta
            if display > 1:
                plt.show()
            else:
                plt.close()

            time_slice = 500
            utime_array = uncal_pass["timestamp"][selection]
            par_array = ecal_pass[selection]
            bins = np.arange(
                (np.amin(utime_array) // time_slice) * time_slice,
                ((np.amax(utime_array) // time_slice) + 2) * time_slice,
                time_slice,
            )
            # bin time values
            times_average = (bins[:-1] + bins[1:]) / 2

            binned = np.digitize(utime_array, bins)
            bin_nos = np.unique(binned)

            par_average = np.zeros(len(bins) - 1)
            par_average[:] = np.nan
            for i in bin_nos:
                if len(par_array[np.where(binned == i)[0]]) > 50:
                    par_average[i - 1] = np.percentile(
                        par_array[np.where(binned == i)[0]], 50
                    )

            plot_dict_param["mean_stability"] = {
                "energy": par_average,
                "time": times_average,
            }

            plt.rcParams["figure.figsize"] = (12, 20)
            plt.rcParams["font.size"] = 12

            fig1 = plt.figure()
            range_adu = 5 / pars[0]  # 10keV window around peak in adu
            for i, peak in enumerate(mus):
                plt.subplot(math.ceil((len(mus)) / 2), 2, i + 1)
                binning = np.arange(pk_ranges[i][0], pk_ranges[i][1], 1)
                bin_cs = (binning[1:] + binning[:-1]) / 2
                energies = uncal_pass[energy_param][
                    (uncal_pass[energy_param] > pk_ranges[i][0])
                    & (uncal_pass[energy_param] < pk_ranges[i][1])
                ][:n_events]

                counts, bs, bars = plt.hist(energies, bins=binning, histtype="step")
                fit_vals = fitted_gof_funcs[i](bin_cs, *pk_pars[i]) * np.diff(bs)
                plt.plot(bin_cs, fit_vals)
                plt.step(
                    bin_cs,
                    [
                        (fval - count) / count if count != 0 else (fval - count)
                        for count, fval in zip(counts, fit_vals)
                    ],
                )
                plt.plot(
                    [bin_cs[10]],
                    [0],
                    label=get_peak_label(fitted_peaks[i]),
                    linestyle="None",
                )
                plt.plot(
                    [bin_cs[10]],
                    [0],
                    label=f"{fitted_peaks[i]:.1f} keV",
                    linestyle="None",
                )
                plt.plot(
                    [bin_cs[10]],
                    [0],
                    label=f"{fwhms[i]:.2f} +- {dfwhms[i]:.2f} keV",
                    linestyle="None",
                )
                plt.plot(
                    [bin_cs[10]],
                    [0],
                    label=f"p-value : {p_vals[i]:.2f}",
                    linestyle="None",
                )

                plt.xlabel("Energy (keV)")
                plt.ylabel("Counts")
                plt.legend(loc="upper left", frameon=False)
                plt.xlim([peak - range_adu, peak + range_adu])
                locs, labels = plt.xticks()
                new_locs, new_labels = get_peak_labels(locs, pars)
                plt.xticks(ticks=new_locs, labels=new_labels)

            plt.tight_layout()
            plot_dict_param["peak_fits"] = fig1
            if display > 1:
                plt.show()
            else:
                plt.close()

            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 12

            fig2, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )

            cal_bins = np.arange(0, np.nanmax(mus) * 1.1, 10)

            ax1.scatter(all_peaks, mus, marker="x", c="b")

            ax1.plot(pgf.poly(cal_bins, pars), cal_bins, lw=1, c="g")

            ax1.grid()
            ax1.set_xlim([200, 2700])
            ax1.set_ylabel("Energy (ADC)")
            ax2.plot(
                all_peaks,
                pgf.poly(np.array(mus), pars) - all_peaks,
                lw=0,
                marker="x",
                c="b",
            )
            ax2.grid()
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel("Residuals (keV)")
            plot_dict_param["cal_fit"] = fig2
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig3, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )
            ax1.errorbar(
                fwhm_peaks, fit_fwhms, yerr=fit_dfwhms, marker="x", lw=0, c="b"
            )

            fwhm_slope_bins = np.arange(200, 2700, 10)

            qbb_line_vx = [2039.0, 2039.0]
            qbb_line_vy = [
                0.9 * np.nanmin(fwhm_slope(fwhm_slope_bins, *fit_pars)),
                fit_qbb,
            ]
            qbb_line_hx = [200, 2039.0]
            qbb_line_hy = [fit_qbb, fit_qbb]

            ax1.plot(
                fwhm_slope_bins, fwhm_slope(fwhm_slope_bins, *fit_pars), lw=1, c="g"
            )
            ax1.plot(qbb_line_hx, qbb_line_hy, lw=1, c="r")
            ax1.plot(qbb_line_vx, qbb_line_vy, lw=1, c="r")
            ax1.plot(
                np.nan,
                np.nan,
                "-",
                color="none",
                label=f"Qbb fwhm: {fit_qbb:1.2f} +- {qbb_err:1.2f} keV",
            )
            ax1.legend(loc="upper left", frameon=False)
            ax1.set_ylim(
                [
                    0.9 * np.nanmin(fwhm_slope(fwhm_slope_bins, *fit_pars)),
                    1.1 * np.nanmax(fwhm_slope(fwhm_slope_bins, *fit_pars)),
                ]
            )
            ax1.set_xlim([200, 2700])
            ax1.grid()
            ax1.set_ylabel("FWHM energy resolution (keV)")
            ax2.plot(
                fwhm_peaks,
                (fit_fwhms - fwhm_slope(fwhm_peaks, *fit_pars)) / fit_dfwhms,
                lw=0,
                marker="x",
                c="b",
            )
            ax2.set_xlabel("Energy (keV)")
            ax2.set_ylabel("Normalised Residuals")
            ax2.grid()
            plt.tight_layout()
            plot_dict_param["fwhm_fit"] = fig3
            if display > 1:
                plt.show()
            else:
                plt.close()

            fig4 = plt.figure()
            bins = np.linspace(0, 3000, 1000)
            plot_dict_param["spectrum"] = {
                "bins": bins,
                "counts": np.histogram(ecal_pass, bins)[0],
            }
            plt.hist(
                ecal_pass,
                bins=bins,
                histtype="step",
                label=f"{len(ecal_pass)} events passed quality cuts",
            )
            if cut_parameters is not None:
                plt.hist(
                    ecal_fail,
                    bins=bins,
                    histtype="step",
                    label=f"{len(ecal_fail)} events failed quality cuts",
                )
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper right")
            plot_dict_param["spectrum_plot"] = fig4
            if display > 1:
                plt.show()
            else:
                plt.close()

            if cut_parameters is not None:
                fig5 = plt.figure()
                n_bins = 500
                counts_pass, bins_pass, _ = pgh.get_hist(
                    ecal_pass, bins=n_bins, range=(0, 3000)
                )
                counts_fail, bins_fail, _ = pgh.get_hist(
                    ecal_fail, bins=n_bins, range=(0, 3000)
                )
                sf = (
                    100
                    * (counts_pass + 10 ** (-6))
                    / (counts_pass + counts_fail + 10 ** (-6))
                )

                plt.step(pgh.get_bin_centers(bins_pass), sf, where="post")
                plt.xlabel("Energy (keV)")
                plt.ylabel("Survival Fraction (%)")
                plt.ylim([50, 100])
                plt.xlim([0, 3000])
                plot_dict_param["survival_frac"] = fig5
                if display > 1:
                    plt.show()
                else:
                    plt.close()

            plot_dict[energy_param] = plot_dict_param

        if fitted_peaks[-1] == 2614.50:
            fep_fwhm = round(fwhms[-1], 2)
            fep_dwhm = round(dfwhms[-1], 2)
        else:
            fep_fwhm = np.nan
            fep_dwhm = np.nan

        output_dict[f"{energy_param}_cal"] = {
            "Qbb_fwhm": round(fit_qbb, 2),
            "Qbb_fwhm_err": round(qbb_err, 2),
            "2.6_fwhm": fep_fwhm,
            "2.6_fwhm_err": fep_dwhm,
            "eres_pars": fit_pars.tolist(),
            "fitted_peaks": results["fitted_keV"].tolist(),
            "fwhms": results["pk_fwhms"].tolist(),
            "peak_fit_pars": pk_rs_dict,
        }
        log.info(
            f"Results {energy_param}: {json.dumps(output_dict[f'{energy_param}_cal'], indent=2)}"
        )

    log.info(f"Finished all calibrations")
    if display > 0:
        return hit_dict, output_dict, plot_dict
    else:
        return hit_dict, output_dict


def get_peak_labels(
    labels: list[str], pars: list[float]
) -> tuple(list[float], list[float]):
    out = []
    out_labels = []
    for i, label in enumerate(labels):
        if i % 2 == 1:
            continue
        else:
            out.append(f"{pgf.poly(label, pars):.1f}")
            out_labels.append(label)
    return out_labels, out


def get_peak_label(peak: float) -> str:
    if peak == 583.191:
        return "Tl 583"
    elif peak == 727.33:
        return "Bi 727"
    elif peak == 860.564:
        return "Tl 860"
    elif peak == 1592.53:
        return "Tl DEP"
    elif peak == 1620.5:
        return "Bi FEP"
    elif peak == 2103.53:
        return "Tl SEP"
    elif peak == 2614.5:
        return "Tl FEP"
