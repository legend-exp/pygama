"""
This module provides routines for calculating and applying quality cuts
"""

from __future__ import annotations

import logging

import lgdo.lh5 as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lgdo.types import Table
from scipy import stats

import pygama.math.binned_fitting as pgf
import pygama.math.histogram as pgh
import pygama.pargen.energy_cal as pgc

log = logging.getLogger(__name__)
sto = lh5.LH5Store()
mpl.use("agg")


def get_keys(in_data, cut_dict):
    """
    Get the keys of the data that are used in the cut dictionary
    """
    parameters = []
    for _, entry in cut_dict.items():
        if "cut_parameter" in entry:
            parameters.append(entry["cut_parameter"])
        else:
            parameters.append(entry["expression"])

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
    data: dict[str, np.ndarray],
    cut_dict: dict[str, int],
    rounding: int = 4,
    display: int = 0,
) -> dict:
    """
    Finds double sided cut boundaries for a file for the parameters specified

    Parameters
    ----------
    data : lh5 table, dictionary of arrays or pandas dataframe
                data to calculate cuts on
    parameters : dict
                dictionary of the form:
                {
                    "output_parameter_name": {
                        "cut_parameter": "parameter_to_cut_on",
                        "cut_level": number_of_sigmas,
                        "mode": "inclusive" or "exclusive"
                    }
                }
                number of sigmas can instead be a dictionary to specify different cut levels for low and high side
                or to only have a one sided cut only specify one of the low or high side
                e.g.
                {
                    "output_parameter_name": {
                        "cut_parameter": "parameter_to_cut_on",
                        "cut_level": {"low_side": 3, "high_side": 2},
                        "mode": "inclusive" or "exclusive"
                    }
                }
                alternatively can specify hit dict fields to just copy dict into output dict e.g.
                {
                    "is_valid_t0":{
                        "expression":"(tp_0_est>a)&(tp_0_est<b)",
                        "parameters":{"a":46000, "b":52000}
                    }
                }
                or
                {
                    "is_valid_cal":{
                        "expression":"(~is_pileup_tail)&(~is_pileup_baseline)"
                    }
                }
    rounding : int
                number of decimal places to round to
    display : int
                if 1 will display plots of the cuts
                if 0 will not display plots

    Returns
    -------
    dict
        dictionary of the form (same as hit dicts):
        {
            "output_parameter_name": {
                "expression": "cut_expression",
                "parameters": {"a": lower_bound, "b": upper_bound}
            }
        }
    plot_dict
        dictionary of plots

    """

    output_dict = {}
    plot_dict = {}
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, Table):
        data = {entry: data[entry].nda for entry in get_keys(data, cut_dict)}
        data = pd.DataFrame.from_dict(data)
    elif isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
    for out_par, cut in cut_dict.items():
        if "expression" in cut:
            output_dict[out_par] = {"expression": cut["expression"]}
            if "parameters" in cut:
                output_dict[out_par].update({"parameters": cut["parameters"]})
        else:
            par = cut["cut_parameter"]
            num_sigmas = cut["cut_level"]
            mode = cut["mode"]
            try:
                all_par_array = data[par].to_numpy()
            except KeyError:
                all_par_array = data.eval(par).to_numpy()
            idxs = (all_par_array > np.nanpercentile(all_par_array, 1)) & (
                all_par_array < np.nanpercentile(all_par_array, 99)
            )
            par_array = all_par_array[idxs]
            bin_width = np.nanpercentile(par_array, 55) - np.nanpercentile(
                par_array, 50
            )

            counts, start_bins, var = pgh.get_hist(
                par_array,
                range=(np.nanmin(par_array), np.nanmax(par_array)),
                dx=bin_width,
            )
            max_idx = np.argmax(counts)
            mu = start_bins[max_idx]
            try:
                fwhm = pgh.get_fwhm(counts, start_bins)[0]
                guess_sig = fwhm / 2.355

                lower_bound = mu - 10 * guess_sig

                upper_bound = mu + 10 * guess_sig

            except Exception:
                lower_bound = np.nanpercentile(par_array, 5)
                upper_bound = np.nanpercentile(par_array, 95)

            if (lower_bound < np.nanmin(par_array)) or (
                lower_bound > np.nanmax(par_array)
            ):
                lower_bound = np.nanmin(par_array)
            if (upper_bound > np.nanmax(par_array)) or (
                upper_bound < np.nanmin(par_array)
            ):
                upper_bound = np.nanmax(par_array)

            try:
                counts, bins, var = pgh.get_hist(
                    par_array,
                    dx=(
                        np.nanpercentile(par_array, 52)
                        - np.nanpercentile(par_array, 50)
                    ),
                    range=(lower_bound, upper_bound),
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
                mean = pars[0]
                std = fwhm / 2.355

                if (
                    mean < np.nanmin(bins)
                    or mean > np.nanmax(bins)
                    or (mean + std) < mu
                    or (mean - std) > mu
                ):
                    raise IndexError
            except IndexError:
                try:
                    fwhm = pgh.get_fwhm(counts, bins)[0]
                    mean = float(bin_centres[np.argmax(counts)])
                    std = fwhm / 2.355
                except Exception:
                    lower_bound = np.nanpercentile(par_array, 5)
                    upper_bound = np.nanpercentile(par_array, 95)

                    counts, bins, var = pgh.get_hist(
                        par_array,
                        dx=np.nanpercentile(par_array, 51)
                        - np.nanpercentile(par_array, 50),
                        range=(lower_bound, upper_bound),
                    )

                    bin_centres = pgh.get_bin_centers(bins)

                    fwhm = pgh.get_fwhm(counts, bins)[0]
                    mean = float(bin_centres[np.argmax(counts)])
                    std = fwhm / 2.355

            if isinstance(num_sigmas, (int, float)):
                num_sigmas_left = num_sigmas
                num_sigmas_right = num_sigmas
            elif isinstance(num_sigmas, dict):
                if "low_side" in num_sigmas:
                    num_sigmas_left = num_sigmas["low_side"]
                else:
                    num_sigmas_left = None
                if "high_side" in num_sigmas:
                    num_sigmas_right = num_sigmas["high_side"]
                else:
                    num_sigmas_right = None
            upper = round(float((num_sigmas_right * std) + mean), rounding)
            lower = round(float((-num_sigmas_left * std) + mean), rounding)
            if mode == "inclusive":
                if upper is not None and lower is not None:
                    cut_string = f"({par}>a) & ({par}<b)"
                    par_dict = {"a": lower, "b": upper}
                elif upper is None:
                    cut_string = f"{par}>a"
                    par_dict = {"a": lower}
                elif lower is None:
                    cut_string = f"{par}<a"
                    par_dict = {"a": upper}
            elif mode == "exclusive":
                if upper is not None and lower is not None:
                    cut_string = f"({par}<a) | ({par}>b)"
                    par_dict = {"a": lower, "b": upper}
                elif upper is None:
                    cut_string = f"{par}<a"
                    par_dict = {"a": lower}
                elif lower is None:
                    cut_string = f"{par}>a"
                    par_dict = {"a": upper}

            output_dict[out_par] = {"expression": cut_string, "parameters": par_dict}

            if display > 0:
                fig = plt.figure()
                plt.hist(
                    all_par_array,
                    bins=np.linspace(
                        np.nanpercentile(all_par_array, 1),
                        np.nanpercentile(all_par_array, 99),
                        100,
                    ),
                    histtype="step",
                )
                if upper is not None:
                    plt.axvline(upper)
                if lower is not None:
                    plt.axvline(lower)
                plt.ylabel("counts")
                plt.xlabel(out_par)
                plot_dict[out_par] = fig
                plt.close()
    if display > 0:
        return output_dict, plot_dict
    else:
        return output_dict


def get_cut_indexes(data, cut_parameters):
    """
    Get the indexes of the data that pass the cuts in
    """
    if data is not isinstance(Table):
        try:
            data = Table(data)
        except Exception:
            raise ValueError("Data must be a Table")

    cut_dict = generate_cuts(data, parameters=cut_parameters)
    log.debug(f"Cuts are {cut_dict}")
    ct_mask = np.full(len(data), True, dtype=bool)
    for outname, info in cut_dict.items():
        outcol = data.eval(info["expression"], info.get("parameters", None))
        data.add_column(outname, outcol)
    log.debug("Applied Cuts")

    for cut in cut_dict:
        ct_mask = data[cut].nda & ct_mask
    return ct_mask


def find_pulser_properties(df, energy="daqenergy"):
    """
    Searches for pulser in the energy spectrum using time between events in peaks
    """
    if np.nanmax(df[energy]) > 8000:
        hist, bins, var = pgh.get_hist(
            df[energy], dx=1, range=(1000, np.nanmax(df[energy]))
        )
        allowed_err = 200
    else:
        hist, bins, var = pgh.get_hist(
            df[energy], dx=0.2, range=(500, np.nanmax(df[energy]))
        )
        allowed_err = 50
    if np.any(var == 0):
        var[np.where(var == 0)] = 1
    imaxes = pgc.get_i_local_maxima(hist / np.sqrt(var), 3)
    peak_energies = pgh.get_bin_centers(bins)[imaxes]
    pt_pars, pt_covs = pgc.hpge_fit_E_peak_tops(
        hist, bins, var, peak_energies, n_to_fit=10
    )
    peak_e_err = pt_pars[:, 1] * 4

    allowed_mask = np.ones(len(peak_energies), dtype=bool)
    for i, e in enumerate(peak_energies[1:-1]):
        i += 1
        if peak_e_err[i] > allowed_err:
            continue
        if i == 1:
            if (
                e - peak_e_err[i] < peak_energies[i - 1] + peak_e_err[i - 1]
                and peak_e_err[i - 1] < allowed_err
            ):
                overlap = (
                    peak_energies[i - 1]
                    + peak_e_err[i - 1]
                    - (peak_energies[i] - peak_e_err[i])
                )
                peak_e_err[i] -= overlap * (
                    peak_e_err[i] / (peak_e_err[i] + peak_e_err[i - 1])
                )
                peak_e_err[i - 1] -= overlap * (
                    peak_e_err[i - 1] / (peak_e_err[i] + peak_e_err[i - 1])
                )

        if (
            e + peak_e_err[i] > peak_energies[i + 1] - peak_e_err[i + 1]
            and peak_e_err[i + 1] < allowed_err
        ):
            overlap = (e + peak_e_err[i]) - (peak_energies[i + 1] - peak_e_err[i + 1])
            total = peak_e_err[i] + peak_e_err[i + 1]
            peak_e_err[i] -= (overlap) * (peak_e_err[i] / total)
            peak_e_err[i + 1] -= (overlap) * (peak_e_err[i + 1] / total)

    out_pulsers = []
    for i, e in enumerate(peak_energies[allowed_mask]):
        if peak_e_err[i] > allowed_err:
            continue

        try:
            e_cut = (df[energy] > e - peak_e_err[i]) & (df[energy] < e + peak_e_err[i])
            df_peak = df[e_cut]

            time_since_last = (
                df_peak.timestamp.values[1:] - df_peak.timestamp.values[:-1]
            )

            tsl = time_since_last[
                (time_since_last >= 0)
                & (time_since_last < np.percentile(time_since_last, 99.9))
            ]

            bins = np.arange(0.1, 5, 0.001)
            bcs = pgh.get_bin_centers(bins)
            hist, bins, var = pgh.get_hist(tsl, bins=bins)

            maxs = pgh.get_i_local_maxima(hist, 45)
            maxs = maxs[maxs > 20]

            super_max = pgh.get_i_local_maxima(hist, 500)
            super_max = super_max[super_max > 20]
            if len(maxs) < 2:
                continue
            else:
                max_locs = np.array([0.0])
                max_locs = np.append(max_locs, bcs[np.array(maxs)])
                if (
                    len(np.where(np.abs(np.diff(np.diff(max_locs))) <= 0.001)[0]) > 1
                    or (np.abs(np.diff(np.diff(max_locs))) <= 0.001).all()
                    or len(super_max) > 0
                ):
                    pulser_e = e
                    period = stats.mode(tsl).mode[0]
                    if period > 0.1:
                        out_pulsers.append((pulser_e, peak_e_err[i], period, energy))

                else:
                    continue
        except Exception:
            continue
    return out_pulsers


def get_tcm_pulser_ids(tcm_file, channel, multiplicity_threshold):
    if isinstance(channel, str):
        if channel[:2] == "ch":
            chan = int(channel[2:])
        else:
            chan = int(channel)
    else:
        chan = channel
    if isinstance(tcm_file, list):
        mask = np.array([], dtype=bool)
        for file in tcm_file:
            _, file_mask = get_tcm_pulser_ids(file, chan, multiplicity_threshold)
            mask = np.append(mask, file_mask)
        ids = np.where(mask)[0]
    else:
        data = pd.DataFrame(
            {
                "array_id": sto.read("hardware_tcm_1/array_id", tcm_file)[0].view_as(
                    "np"
                ),
                "array_idx": sto.read("hardware_tcm_1/array_idx", tcm_file)[0].view_as(
                    "np"
                ),
            }
        )
        cumulength = sto.read("hardware_tcm_1/cumulative_length", tcm_file)[0].view_as(
            "np"
        )
        cumulength = np.append(np.array([0]), cumulength)
        n_channels = np.diff(cumulength)
        evt_numbers = np.repeat(np.arange(0, len(cumulength) - 1), np.diff(cumulength))
        evt_mult = np.repeat(np.diff(cumulength), np.diff(cumulength))
        data["evt_number"] = evt_numbers
        data["evt_mult"] = evt_mult
        high_mult_events = np.where(n_channels > multiplicity_threshold)[  # noqa: F841
            0
        ]

        ids = data.query(f"array_id=={channel} and evt_number in @high_mult_events")[
            "array_idx"
        ].to_numpy()
        mask = np.zeros(len(data.query(f"array_id=={channel}")), dtype="bool")
        mask[ids] = True
    return ids, mask


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

        period = z[0]
        phase = z[1]
        mod = np.abs(df.timestamp - phase) % period

        period_cut = (mod < 0.1) | ((period - mod) < 0.1)  # 0.1)

        if final_mask is None:
            final_mask = e_cut & period_cut
        else:
            final_mask = final_mask | (e_cut & period_cut)

    df.loc[final_mask, "isPulser"] = 1

    return df
