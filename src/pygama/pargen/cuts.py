"""
This module provides routines for calculating and applying quality cuts
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)


def get_keys(in_data, parameters):
    out_params = []
    if isinstance(in_data, dict):
        possible_keys = in_data.keys()
    elif isinstance(in_data, list):
        possible_keys = in_data
    for param in parameters:
        for key in possible_keys:
            if key in param:
                out_params.append(key)
    return np.unique(out_params).tolist()


def generate_cuts(
    data: dict[str, np.ndarray], parameters: dict[str, int], rounding: int = 2
) -> dict:
    """
    Finds double sided cut boundaries for a file for the parameters specified

    Parameters
    ----------
    data : lh5 table or dictionary of arrays
                data to calculate cuts on
    parameters : dict
                 dictionary with the parameter to be cut and the number of sigmas to cut at
    """

    output_dict = {}
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, lh5.Table):
        data = {entry: data[entry].nda for entry in get_keys(data, parameters)}
        data = pd.DataFrame.from_dict(data)
    elif isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    for par in parameters.keys():
        if isinstance(parameters[par], dict):
            if "Lower Boundary" in list(parameters[par]) or "Upper Boundary" in list(
                parameters[par]
            ):
                output_dict[par] = parameters[par].copy()
                if "Lower Boundary" not in parameters[par]:
                    output_dict[par]["Lower Boundary"] = -np.inf
                if "Upper Boundary" not in parameters[par]:
                    output_dict[par]["Upper Boundary"] = np.inf
                continue
        num_sigmas = parameters[par]
        try:
            all_par_array = data[par].to_numpy()
        except KeyError:
            all_par_array = data.eval(par).to_numpy()
        idxs = (all_par_array > np.nanpercentile(all_par_array, 1)) & (
            all_par_array < np.nanpercentile(all_par_array, 99)
        )
        par_array = all_par_array[idxs]
        counts, start_bins, var = pgh.get_hist(par_array, 1000)
        max_idx = np.argmax(counts)
        mu = start_bins[max_idx]
        try:
            pars, cov = pgf.gauss_mode_width_max(
                counts,
                start_bins,
                mode_guess=mu,
                n_bins=50,
                cost_func="Least Squares",
                inflate_errors=False,
                gof_method="var",
            )

            guess_mu, guess_sig, guess_amp = pars

            lower_bound = guess_mu - 10 * guess_sig

            upper_bound = guess_mu + 10 * guess_sig

            if lower_bound < np.nanmin(par_array) or upper_bound > np.nanmax(par_array):
                pars, cov = pgf.gauss_mode_width_max(
                    counts,
                    start_bins,
                    mode_guess=mu,
                    n_bins=5,
                    cost_func="Least Squares",
                    inflate_errors=False,
                    gof_method="var",
                )

            guess_mu, guess_sig, guess_amp = pars

            lower_bound = guess_mu - 10 * guess_sig

            upper_bound = guess_mu + 10 * guess_sig

        except:
            bin_range = 1000

            if max_idx < bin_range:
                lower_bound_idx = 0
            else:
                lower_bound_idx = max_idx - bin_range
            lower_bound = start_bins[lower_bound_idx]

            if max_idx > len(start_bins) - bin_range:
                upper_bound_idx = -1
            else:
                upper_bound_idx = max_idx + bin_range

            upper_bound = start_bins[upper_bound_idx]

        if lower_bound < np.nanmin(par_array):
            lower_bound = np.nanmin(par_array)
        if upper_bound > np.nanmax(par_array):
            upper_bound = np.nanmax(par_array)

        try:
            counts, bins, var = pgh.get_hist(
                par_array, bins=200, range=(lower_bound, upper_bound)
            )

            bin_centres = pgh.get_bin_centers(bins)

            fwhm = pgh.get_fwhm(counts, bins)[0]
            mean = float(bin_centres[np.argmax(counts)])
            pars, cov = pgf.gauss_mode_width_max(
                counts,
                bins,
                mode_guess=mean,
                n_bins=20,
                cost_func="Least Squares",
                inflate_errors=False,
                gof_method="var",
            )
            mean - pars[0]
            std = fwhm / 2.355
        except IndexError:
            bin_range = 5000

            if max_idx < bin_range:
                lower_bound_idx = 0
            else:
                lower_bound_idx = max_idx - bin_range
            lower_bound = start_bins[lower_bound_idx]

            if max_idx > len(start_bins) - bin_range:
                upper_bound_idx = -1
            else:
                upper_bound_idx = max_idx + bin_range
            upper_bound = start_bins[upper_bound_idx]
            counts, bins, var = pgh.get_hist(
                par_array, bins=1000, range=(lower_bound, upper_bound)
            )

            bin_centres = pgh.get_bin_centers(bins)

            fwhm = pgh.get_fwhm(counts, bins)[0]
            mean = float(bin_centres[np.argmax(counts)])
            std = fwhm / 2.355

        if isinstance(num_sigmas, (int, float)):
            num_sigmas_left = num_sigmas
            num_sigmas_right = num_sigmas
        elif isinstance(num_sigmas, dict):
            if "left" in num_sigmas:
                num_sigmas_left = num_sigmas["left"]
            else:
                num_sigmas["left"] = -np.inf
                num_sigmas_left = -np.inf
            if "right" in num_sigmas:
                num_sigmas_right = num_sigmas["right"]
            else:
                num_sigmas["right"] = np.inf
                num_sigmas_right = np.inf
        upper = float((num_sigmas_right * std) + mean)
        lower = float((-num_sigmas_left * std) + mean)
        output_dict[par] = {
            "Mean Value": round(mean, rounding),
            "Sigmas Cut": num_sigmas,
            "Upper Boundary": round(upper, rounding),
            "Lower Boundary": round(lower, rounding),
        }
    return output_dict


def get_cut_indexes(
    all_data: dict[str, np.ndarray], cut_dict: dict, energy_param: str = "trapTmax"
) -> list[int]:

    """
    Returns a mask of the data, for a single file, that passes cuts based on dictionary of cuts
    in form of cut boundaries above
    Parameters
    ----------
    File : dict or lh5_table
           dictionary of parameters + array such as load_nda or lh5 table of params
    Cut_dict : string
                Dictionary file with cuts
    """

    indexes = None
    keys = cut_dict.keys()
    if isinstance(all_data, pd.DataFrame):
        pass
    elif isinstance(all_data, lh5.Table):
        cut_keys = list(cut_dict)
        cut_keys.append(energy_param)
        all_data = {
            entry: all_data[entry].nda for entry in get_keys(all_data, cut_keys)
        }
        all_data = pd.DataFrame.from_dict(all_data)
    elif isinstance(all_data, dict):
        all_data = pd.DataFrame.from_dict(all_data)
    for cut in keys:
        try:
            data = all_data[cut]
        except KeyError:
            data = all_data.eval(cut).to_numpy()
        upper = cut_dict[cut]["Upper Boundary"]
        lower = cut_dict[cut]["Lower Boundary"]
        idxs = (data < upper) & (data > lower) & (~np.isnan(data))
        percent = 100 * len(np.where(idxs)[0]) / len(idxs)
        log.info(f"{percent:.2f}% passed {cut} cut")

        # Combine masks
        if indexes is not None:
            indexes = indexes & idxs

        else:
            indexes = idxs
        log.debug(f"{cut} loaded")
    percent = 100 * len(np.where(indexes)[0]) / len(indexes)
    log.info(f"{percent:.2f}% passed all cuts")
    return indexes


def cut_dict_to_hit_dict(cut_dict, final_cut_field="is_valid_cal"):
    out_dict = {}
    symbols = "/-+*"
    replacewith = "_"
    for i, param in enumerate(cut_dict):

        out_dict[
            f"{''.join(replacewith  if c in symbols else c for c in param).replace('(','').replace(')','')}_cut"
        ] = {
            "expression": f"(a<({param}))&(({param})<b)",
            "parameters": {
                "a": cut_dict[param]["Lower Boundary"],
                "b": cut_dict[param]["Upper Boundary"],
            },
        }
    quality_cut_exp = ""
    for par in list(cut_dict)[:-1]:
        quality_cut_exp += f"({''.join(replacewith  if c in symbols else c for c in par).replace('(','').replace(')','')}_cut)&"
    quality_cut_exp += f"({''.join(replacewith  if c in symbols else c for c in list(cut_dict)[-1]).replace('(','').replace(')','')}_cut)"
    out_dict[final_cut_field] = {"expression": quality_cut_exp, "parameters": {}}
    return out_dict


def find_pulser_properties(df, energy="daqenergy"):

    hist, bins, var = pgh.get_hist(df[energy], dx=1, range=(100, np.nanmax(df[energy])))
    if np.any(var == 0):
        var[np.where(var == 0)] = 1
    imaxes = pgc.get_i_local_maxima(hist / np.sqrt(var), 5)
    peak_energies = pgh.get_bin_centers(bins)[imaxes]
    pt_pars, pt_covs = pgc.hpge_fit_E_peak_tops(
        hist, bins, var, peak_energies, n_to_fit=15
    )
    peak_e_err = pt_pars[:, 1] * 4

    out_pulsers = []
    for i, e in enumerate(peak_energies):
        if peak_e_err[i] > 200:
            continue
        else:
            try:
                e_cut = (df[energy] > e - peak_e_err[i]) & (
                    df[energy] < e + peak_e_err[i]
                )
                df_peak = df[e_cut]

                time_since_last = (
                    df_peak.timestamp.values[1:] - df_peak.timestamp.values[:-1]
                )

                tsl = time_since_last[
                    (time_since_last >= 0)
                    & (time_since_last < np.percentile(time_since_last, 99.9))
                ]

                bins = np.arange(0.1, 5, 0.0001)
                bcs = pgh.get_bin_centers(bins)
                hist, bins, var = pgh.get_hist(tsl, bins=bins)

                maxs = pgc.get_i_local_maxima(hist, 40)
                if len(maxs) < 2:
                    continue
                else:

                    max_locs = np.array([0.0])
                    max_locs = np.append(max_locs, bcs[np.array(maxs)])
                    if (
                        len(np.where(np.abs(np.diff(np.diff(max_locs))) <= 0.001)[0])
                        > 1
                        or (np.abs(np.diff(np.diff(max_locs))) <= 0.001).all()
                    ):
                        pulser_e = e
                        period = stats.mode(tsl).mode[0]
                        out_pulsers.append((pulser_e, peak_e_err[i], period, energy))

                    else:
                        continue
            except:
                continue
    return out_pulsers


def tag_pulsers(df, chan_info, window=0.01):
    df["isPulser"] = 0

    if isinstance(chan_info, tuple):
        chan_info = [chan_info]
    final_mask = None
    for chan_i in chan_info:
        pulser_energy, peak_e_err, period, energy_name = chan_i

        e_cut = (df[energy_name] < pulser_energy + peak_e_err) & (
            df[energy_name] > pulser_energy - peak_e_err
        )
        df_pulser = df[e_cut]

        time_since_last = np.zeros(len(df_pulser))
        time_since_last[1:] = (
            df_pulser.timestamp.values[1:] - df_pulser.timestamp.values[:-1]
        )

        mode_idxs = (time_since_last > period - window) & (
            time_since_last < period + window
        )

        pulser_events = np.count_nonzero(mode_idxs)
        # print(f"pulser events: {pulser_events}")
        if pulser_events < 3:
            return df
        df_pulser = df_pulser[mode_idxs]

        ts = df_pulser.timestamp.values
        diff_zero = np.zeros(len(ts))
        diff_zero[1:] = np.around(np.divide(np.subtract(ts[1:], ts[:-1]), period))
        diff_cum = np.cumsum(diff_zero)
        z = np.polyfit(diff_cum, ts, 1)
        p = np.poly1d(z)

        period = z[0]
        phase = z[1]
        pulser_mod = np.abs(df_pulser.timestamp - phase) % period
        mod = np.abs(df.timestamp - phase) % period

        period_cut = (mod < 0.1) | ((period - mod) < 0.1)  # 0.1)

        if final_mask is None:
            final_mask = e_cut & period_cut
        else:
            final_mask = final_mask | (e_cut & period_cut)

    df.loc[final_mask, "isPulser"] = 1

    return df
