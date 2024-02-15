"""
This module is for extracting a single pole zero constant from the decay tail
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle as pkl
from collections import OrderedDict

import matplotlib as mpl

mpl.use("agg")
from typing import Tuple

import lgdo
import lgdo.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from scipy.stats import linregress

import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_cal as pgc
import pygama.pargen.energy_optimisation as om
from pygama.math.histogram import better_int_binning

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def get_decay_constant(
    slopes: np.array, wfs: lgdo.WaveformTable, display: int = 0
) -> dict:
    """
    Finds the decay constant from the modal value of the tail slope after cuts
    and saves it to the specified json.

    Parameters
    ----------
    slopes : array
        tail slope array

    dict_file : str
        path to json file to save decay constant value to.
        It will be saved as a dictionary of form {'pz': {'tau': decay_constant}}

    Returns
    -------
    tau_dict : dict
    """
    tau_dict = {}

    pz = tau_dict.get("pz")

    counts, bins, var = pgh.get_hist(slopes, bins=100000, range=(-0.01, 0))
    bin_centres = pgh.get_bin_centers(bins)
    high_bin = bin_centres[np.argmax(counts)]
    try:
        pars, cov = pgf.gauss_mode_width_max(
            counts,
            bins,
            n_bins=10,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )
        if np.abs(np.abs(pars[0] - high_bin) / high_bin) > 0.05:
            raise ValueError
        high_bin = pars[0]
    except:
        pass
    tau = round(-1 / (high_bin), 1)

    sampling_rate = wfs["dt"].nda[0]
    units = wfs["dt"].attrs["units"]
    tau = f"{tau*sampling_rate}*{units}"

    tau_dict["pz"] = {"tau": tau}
    if display > 0:
        out_plot_dict = {}
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 8
        fig, ax = plt.subplots()
        bins = np.linspace(-0.01, 0, 100000)  # change if needed
        counts, bins, bars = ax.hist(slopes, bins=bins, histtype="step")
        plot_max = np.argmax(counts)
        in_min = plot_max - 20
        if in_min < 0:
            in_min = 0
        in_max = plot_max + 21
        if in_max >= len(bins):
            in_min = len(bins) - 1
        plt.xlabel("Slope")
        plt.ylabel("Counts")
        plt.yscale("log")
        axins = ax.inset_axes([0.5, 0.45, 0.47, 0.47])
        axins.hist(
            slopes[(slopes > bins[in_min]) & (slopes < bins[in_max])],
            bins=200,
            histtype="step",
        )
        axins.axvline(high_bin, color="red")
        axins.set_xlim(bins[in_min], bins[in_max])
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
        out_plot_dict["slope"] = fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return tau_dict, out_plot_dict
    else:
        return tau_dict


def get_dpz_decay_constants(
    tb_data: lgdo.Table,
    cut_idxs: list,
    wf_field: str,
    dpz_opt_dsp_dict: dict,
    percent_tau1_fit: float,
    percent_tau2_fit: float,
    offset_from_wf_max: int,
    superpulse_bl_idx: int,
    superpulse_window_width: int,
    display: int = 0,
) -> tuple[dict, dict]:
    """
    Gets values for the DPZ time constants in 3 stages:

    1. Perform a linear fit to the start and end of the decaying tail of a superpulse
    2. Use those initial guesses to seed a LSQ fit to a DPZ model of the sum of two decaying exponentials
    3. Use the results of the model fit as initial guesses in a DSP routine that optimizes the flatness of the decaying tail

    The first step in this process is to generate a superpulse from high-energy waveforms.


    Parameters
    ----------
    tb_data
        Table containing high-energy event waveforms and daqenergy. Pulser events must have already been removed from this table
    cut_idxs
        A list of indices to cut generated from `cut_parameters` in :func:`dsp_preprocess_decay_const`
    wf_field
        The key in the waveform table corresponding to the waveforms to analyze.
    dpz_opt_dsp_dict
        A dsp dictionary containing the routines to optimize the DPZ parameters.
    percent_tau1_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the start of the tail.
    percent_tau2_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the end of the tail.
    offset_from_wf_max
        The number of indices off the maximum of the waveform to start the fit of the tail. Usually ~100 samples works fine.
    superpulse_bl_idx
        The index at which to stop when computing the mean value of a baseline, used for DPZ
    superpulse_window_width
        The window of acceptance for tp100s while selecting waveforms to make a superpulse, with a center set at the median tp100 value. If a tp100 falls outside this window, ignore this waveform for the superpulse
    display
        An integer. If greater than 1, plots and shows the attempts to fit the long and short time constants. If greater than 0, saves the plot to a dictionary.


    Returns
    -------
    tau_dict
        dictionary of form {'dpz': {'tau1': decay_constant1, 'tau2': decay_constant2, 'frac': fraction}}
    out_plot_dict
        A dictionary containing monitoring plots


    Notes
    -----
    tau1 is the shorter time constant, tau2 is the longer, and frac is the amount of the larger time constant present in the sum of the two exponentials
    """
    n_events = 10000
    tau_dict = {}
    out_plot_dict = {}

    # Get high energy waveforms to create a superpulse. Eventually allow user which peak to select? For now, use the 2615 keV peak
    df = tb_data.view_as("pd")
    threshold = 200
    if cut_idxs is not None:
        initial_mask = (df.daqenergy.values > threshold) & (cut_idxs)
    else:
        initial_mask = df.daqenergy.values > threshold
    E_uncal = df.daqenergy.values[initial_mask]
    initial_idxs = np.where(initial_mask)[0]

    guess_keV = 2620 / np.nanpercentile(E_uncal, 99)
    Euc_min = threshold / guess_keV * 0.6
    Euc_max = 2620 / guess_keV * 1.1
    dEuc = 1 / guess_keV
    # daqenergy is an int so use integer binning (dx used to be bugged as output so switched to nbins)
    Euc_min, Euc_max, nbins = better_int_binning(
        x_lo=Euc_min, x_hi=Euc_max, n_bins=(Euc_max - Euc_min) / dEuc
    )
    hist, bins, var = pgh.get_hist(E_uncal, range=(Euc_min, Euc_max), bins=nbins)
    detected_peaks_locs, detected_peaks_keV, roughpars = pgc.hpge_find_E_peaks(
        hist,
        bins,
        var,
        np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2103.53, 2614.553]),
    )

    peak = 2614.553
    kev_width = [70, 70]
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
    e_mask = (E_uncal > e_lower_lim) & (E_uncal < e_upper_lim)
    e_idxs = initial_idxs[e_mask][: int(n_events)]
    log.debug(f"{len(e_idxs)} events found in energy range for {peak} for superpulse")

    # Stuff this all into a new table to eventually run dsp on
    high_E_wf_tb = lgdo.WaveformTable(
        t0=tb_data[wf_field]["t0"][e_idxs],
        t0_units=tb_data[wf_field]["t0"].attrs["units"],
        dt=tb_data[wf_field]["dt"][e_idxs],
        dt_units=tb_data[wf_field]["dt"].attrs["units"],
        values=tb_data[wf_field]["values"][e_idxs],
        values_units=tb_data[wf_field]["values"].attrs["units"],
    )

    high_E_wfs = high_E_wf_tb["values"].nda[:]

    high_E_tb = lgdo.Table(size=high_E_wf_tb.size)
    high_E_tb.add_field(wf_field, high_E_wf_tb)

    # Time align the waveforms to their maximum
    tp100s = []
    for wf in high_E_wfs:
        tp100s.append(np.argmax(wf))

    time_aligned_wfs = tp100_align(high_E_wfs, superpulse_window_width, tp100s)

    # Baseline subtract the time aligned waveforms
    bl_sub_time_aligned_wfs = []

    for i in range(len(time_aligned_wfs)):
        bl_sub_time_aligned_wfs.append(
            time_aligned_wfs[i] - np.mean(time_aligned_wfs[i][:superpulse_bl_idx])
        )

    # Create a superpulse
    superpulse = np.mean(bl_sub_time_aligned_wfs, axis=0)

    # Fit the superpulse and get rough DPZ constants
    tau1s_fit, tau2s_fit, f2s_fit, dpz_out_plot_dict = dpz_model_fit(
        superpulse,
        percent_tau1_fit=percent_tau1_fit,
        percent_tau2_fit=percent_tau2_fit,
        idx_shift=offset_from_wf_max,
        plot=display,
    )
    out_plot_dict |= dpz_out_plot_dict  # merge the plotting dictionaries

    # Optimize the flatness of high energy waveforms to get optimal DPZ constants
    dpz_opt_tb_out = opt.run_one_dsp(
        high_E_tb,
        dpz_opt_dsp_dict,
        db_dict=dict({"dpz": {"tau1": tau1s_fit, "tau2": tau2s_fit, "frac": f2s_fit}}),
    )

    # Update tau_dict with the dpz constants
    tau1 = np.nanmedian(dpz_opt_tb_out["tau1"].nda)
    tau2 = np.nanmedian(dpz_opt_tb_out["tau2"].nda)
    frac = np.nanmedian(dpz_opt_tb_out["frac"].nda)

    sampling_rate = high_E_wf_tb["dt"].nda[0]
    units = high_E_wf_tb["dt"].attrs["units"]
    tau1 = f"{tau1*sampling_rate}*{units}"
    tau2 = f"{tau2*sampling_rate}*{units}"

    tau_dict["dpz"] = {"tau1": tau1, "tau2": tau2, "frac": frac}

    return tau_dict, out_plot_dict


def dsp_preprocess_decay_const(
    tb_data,
    dsp_config: dict,
    display: int = 0,
    wf_field: str = "waveform_presummed",
    wf_plot: str = "wf_pz",
    norm_param: str = "pz_mean",
    cut_parameters: dict = {"bl_mean": 4, "bl_std": 4, "bl_slope": 4},
    double_pz: bool = False,
    dpz_opt_dsp_dict: dict = None,
    percent_tau1_fit: float = 0.1,
    percent_tau2_fit: float = 0.2,
    offset_from_wf_max: int = 10,
    superpulse_bl_idx: int = 25,
    superpulse_window_width: int = 13,
) -> dict:
    """
    This function calculates the pole zero constant for the input data by calling :func:`get_decay_constant`. It can also calculate the double pole zero constants by calling :func:`get_dpz_decay_constants`.

    Parameters
    ----------
    tb_data
        A table containing waveform data.
    dsp_config
        A dsp config dictionary, this is a stripped down version which just includes cuts and slope of decay tail for the 1PZ computation
    display
        An integer that determines whether or not to plot fit results. If greater than 1, shows the plots. If greater than 0, saves the plots to an output dictionary
    wf_field
        The key for the waveforms to analyze in `tb_data`
    wf_plot
        The key for the waveforms to plot from dsp_config for the 1PZ optimization
    norm_param
        The key from dsp_config to normalize the waveforms that are plotted
    cut_parameters
        A dictionary specifying the cuts to apply from outputs from the dsp_config
    double_pz
        A boolean that decided whether or not to calculate DPZ parameters using :func:`get_dpz_decay_constants`
    dpz_opt_dsp_dict
        A dictionary that provides dsp routines to optimize the DPZ parameters
    percent_tau1_fit
        The fractional percent of the length of the tail of the waveform to fit for DPZ, used to fit the start of the tail.
    percent_tau2_fit
        The fractional percent of the length of the tail of the waveform to fit for DPZ, used to fit the end of the tail.
    offset_from_wf_max
        The number of indices off the maximum of the waveform to start the fit of the tail for dpzDPZ. Usually ~100 samples works fine.
    superpulse_bl_idx
        The index at which to stop when computing the mean value of a baseline, used for DPZ
    superpulse_window_width
        The window of acceptance for tp100s while selecting waveforms to make a superpulse, with a center set at the median tp100 value. If a tp100 falls outside this window, ignore this waveform for the superpulse

    Returns
    -------
    tau_dict
        A dictionary containing the 1PZ constant, as well as the DPZ constants if requested
    plot_dict
        A dictionary containing plots if the display option is set > 0
    """
    plot_dict = {}
    tb_out = opt.run_one_dsp(tb_data, dsp_config)
    log.debug("Processed Data")
    cut_dict = cts.generate_cuts(tb_out, parameters=cut_parameters)
    log.debug("Generated Cuts:", cut_dict)
    idxs = cts.get_cut_indexes(tb_out, cut_dict)
    log.debug("Applied cuts")
    log.debug(f"{len(idxs)} events passed cuts")
    slopes = tb_out["tail_slope"].nda
    log.debug("Calculating pz constant")

    # Get the 1PZ constant from the average tail slope
    tau_dict, pz_plot_dict = get_decay_constant(
        slopes[idxs], tb_data[wf_field], display
    )
    plot_dict |= pz_plot_dict  # merge the dictionaries
    # Do DPZ correction if requested, see :func:`get_dpz_decay_constants` for more details
    if double_pz == True:
        log.debug("Calculating double pz constants")
        tau_dict_dpz, dpz_plot_dict = get_dpz_decay_constants(
            tb_data,
            idxs,
            wf_field,
            dpz_opt_dsp_dict,
            percent_tau1_fit,
            percent_tau2_fit,
            offset_from_wf_max,
            superpulse_bl_idx,
            display,
        )
        tau_dict["dpz"] = tau_dict_dpz["dpz"]
        tau1 = tau_dict_dpz["dpz"]["tau1"]
        tau2 = tau_dict_dpz["dpz"]["tau2"]
        frac = tau_dict_dpz["dpz"]["frac"]
        plot_dict |= dpz_plot_dict  # merge the dpz plot dict into the default one

    # Generate some plots to see the quality of the PZ/DPZ correction
    if display > 0:
        if double_pz:
            # Use the DPZ parameters in the same config for the 1PZ DSP
            tb_out = opt.run_one_dsp(
                tb_data,
                dsp_config,
                db_dict={"pz": {"tau": tau1, "tau2": tau2, "frac": frac}},
            )
        else:
            tb_out = opt.run_one_dsp(tb_data, dsp_config, db_dict=tau_dict)
        wfs = tb_out[wf_plot]["values"].nda[idxs]
        wf_idxs = np.random.choice(len(wfs), 100)
        if norm_param is not None:
            means = tb_out[norm_param].nda[idxs][wf_idxs]
            wfs = np.divide(wfs[wf_idxs], np.reshape(means, (len(wf_idxs), 1)))
        else:
            wfs = wfs[wf_idxs]
        fig2 = plt.figure()
        for wf in wfs:
            plt.plot(np.arange(0, len(wf), 1), wf)
        plt.axhline(1, color="black")
        plt.axhline(0, color="black")
        plt.xlabel("Samples")
        plt.ylabel("ADU")
        plot_dict["waveforms"] = fig2
        if display > 1:
            plt.show()
        else:
            plt.close()

    return tau_dict, plot_dict


def one_exp(ts: np.array, tau2: float, f: float) -> np.array:
    """
    This function computes a decaying exponential. Used to subtract off the estimated long-time constant in :func:`linear_dpz_fit`

    Parameters
    ----------
    ts
        Array of time point values, in samples
    tau2
        A guess for the long time constant in HPGe waveforms
    f
        A guess for the fraction of the long time constant exponential present in an HPGe waveform
    """
    return f * np.exp(-ts / tau2)


def line(ts: np.array, m: float, b: float) -> np.array:
    """
    Computes a line. Used to subtract off the estimated long-time constant in :func:`linear_dpz_fit`

    Parameters
    ----------
    ts
        Array of time point values, in samples
    m
        The slope
    b
        The y-intercept
    """
    return m * ts + b


def dpz_model(ts: np.array, A: float, tau1: float, tau2: float, f2: float) -> np.array:
    """
    Models the double pole zero function as A*[(1-f2)*exp(-t/tau1) + f2*exp(-t/tau2)],
    this is the expected shape of an HPGe waveform. Used in performing fits in :func:`dpz_model_fit`

    Parameters
    ----------
    ts
        Array of time point values, in samples
    A
        The overall amplitude of the waveform
    tau1
        The short time constant present
    tau2
        The long time constant present
    f2
        The fraction of the long time constant present in the overall waveform
    """
    return A * (1 - f2) * np.exp(-ts / tau1) + A * f2 * np.exp(-ts / tau2)


def linear_dpz_fit(
    waveform: np.array,
    percent_tau1_fit: float,
    percent_tau2_fit: float,
    idx_shift: int,
    plot: int,
) -> tuple[float, float, float, dict]:
    """
    Parameters
    ----------
    waveform
        An array containing waveform data
    percent_tau1_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the start of the tail.
    percent_tau2_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the end of the tail.
    idx_shift
        The number of indices off the maximum of the waveform to start the fit of the tail. Usually ~100 samples works fine.
    plot
        An integer, if greater than 1 plots and shows the fit results, if greater than 0 saves the plot to a dictionary.

    Returns
    -------
    tau1
        A guess of the shorter double pole zero time constant
    tau2
        A guess of the longer double pole zero time constant
    frac
        A guess of the fraction in the double pole zero of the longer time constant
    out_plot_dict
        A dictionary containing the output plots if requested

    Notes
    -----
    Extracts an initial guess of the double pole zero parameters by performing two linear fits to the log of the waveform.
    The first log-fit is to long time scales, and fits the long time constant tau2. This piece is subtracted off the waveform,
    and then the shorter time constant is fit. The average of the fractions returned by the intercepts of these fits is returned.

    The waveform must be baseline subtracted to provide accurate fit results!
    """
    out_plot_dict = {}
    # Get the indices to start the fit from
    wf_max_idx = np.argmax(waveform)
    tau2_idx = -int((len(waveform) - wf_max_idx) * percent_tau2_fit)
    tau1_idx = int((len(waveform) - wf_max_idx) * percent_tau1_fit)

    # Rescale the waveform to ensure fitting converges
    fit_start_idx = (
        np.argmax(waveform) + idx_shift
    )  # shifting from the max of the waveform helps the convervence
    scaled_pulse = waveform / np.amax(waveform)
    scaled_pulse = scaled_pulse[fit_start_idx:]

    # Fit the long time constant
    ts = np.arange(0, len(scaled_pulse))
    slope_2, intercept_2, *_ = linregress(
        ts[tau2_idx:], np.log(scaled_pulse[tau2_idx:])
    )

    if plot > 0:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(ts[tau2_idx:], np.log(scaled_pulse[tau2_idx:]))
        plt.plot(ts[tau2_idx:], line(ts[tau2_idx:], slope_2, intercept_2))
        plt.ylabel("ADC")
        plt.xlabel("Time [Samples]")
        plt.title("Long Time Constant Fitted")
        plt.grid(True)
        out_plot_dict["long_tau_linear"] = fig
        if plot > 1:
            plt.show()
        else:
            plt.close()

    # Subtract off the long time constant
    sub_pulse = np.abs(scaled_pulse - one_exp(ts, -1 / slope_2, np.exp(intercept_2)))

    # Fit the short time constant
    slope_1, intercept_1, *_ = linregress(ts[:tau1_idx], np.log(sub_pulse[:tau1_idx]))
    # If the fit fails, just return an arbitrary good-enough guess. All detectors have a roughly ~130 sample short time constant for presumming of 8.
    if slope_1 > 0:
        slope_1 = -1 / 130
        intercept_1 = np.log(1 - np.exp(intercept_2))

    if plot > 0:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(ts[:tau1_idx], np.log(sub_pulse[:tau1_idx]))
        plt.plot(ts[:tau1_idx], line(ts[:tau1_idx], slope_1, intercept_1))
        plt.ylabel("ADC")
        plt.xlabel("Time [Samples]")
        plt.title("Short Time Constant Fitted")
        plt.grid(True)
        out_plot_dict["short_tau_linear"] = fig
        if plot > 1:
            plt.show()
        else:
            plt.close()

    return (
        -1 / slope_1,
        -1 / slope_2,
        np.mean([np.exp(intercept_2), 1 - np.exp(intercept_1)]),
        out_plot_dict,
    )


def dpz_model_fit(
    waveform: np.array,
    percent_tau1_fit: float,
    percent_tau2_fit: float,
    idx_shift: int,
    plot: int,
) -> tuple[float, float, float, dict]:
    """
    Parameters
    ----------
    waveform
        An array containing waveform data that has been baseline subtracted
    percent_tau1_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the start of the tail.
    percent_tau2_fit
        The fractional percent of the length of the tail of the waveform to fit, used to fit the end of the tail.
    idx_shift
        The number of indices off the maximum of the waveform to start the fit of the tail. Usually ~100 samples works fine.
    plot
        An integer, that if greater than 1 plots and shows the fit results, if greater than 0 saves the plots to a dictionary

    Returns
    -------
    tau1
        A guess of the shorter double pole zero time constant
    tau2
        A guess of the longer double pole zero time constant
    frac2
        A guess of the fraction in the double pole zero of the longer time constant
    out_plot_dict
        A dictionary containing the output plots if requested

    Notes
    -----
    Extracts the double pole zero parameters by fitting to an analytic model, utilizing the best-guess from linear fitting.
    Fits to the function
    f(t) = A*[(1-f2)*exp(-t/tau1)+f2*exp(-t/tau2)]
    It fits from the maximum of the waveform onwards to the end of the tail.
    The waveform must be baseline subtracted in order to get the best fit results
    """
    out_plot_dict = {}
    # Get the initial guess of the DPZ constants using a quick linear fit to the start and end of the tail
    tau1_guess, tau2_guess, f2_guess, linear_plot_dict = linear_dpz_fit(
        waveform, percent_tau1_fit, percent_tau2_fit, idx_shift, plot
    )
    out_plot_dict |= linear_plot_dict  # merge dictionary in place

    # Select the data to fit, from the maximum plus some offset
    fit_start_idx = np.argmax(waveform) + idx_shift
    waveform = waveform[fit_start_idx:]

    # Create the Iminuit cost function using least squares
    ts = np.arange(0, len(waveform))

    def cost_function(A, tau1, tau2, f2):
        output = dpz_model(ts, A, tau1, tau2, f2)
        res = 0
        for i in range(len(output)):
            res += (output[i] - waveform[i]) ** 2

        return res

    # Perform the fit
    m = Minuit(
        cost_function,
        A=np.amax(waveform),
        tau1=tau1_guess,
        tau2=tau2_guess,
        f2=f2_guess,
    )
    m.errordef = Minuit.LEAST_SQUARES
    m.limits[0] = (0, None)
    m.limits[1] = (0, None)
    m.limits[2] = (0, None)
    m.limits[3] = (0, 1)  # the fraction MUST be between 0 and 1
    m.migrad()

    if plot > 0:
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(ts, waveform)
        plt.plot(ts, dpz_model(ts, *m.values), c="r")
        plt.ylabel("ADC")
        plt.xlabel("Time [Samples]")
        plt.title("DPZ Model Fit to Waveform")
        plt.grid(True)
        out_plot_dict["dpz_model_fit"] = fig
        if plot > 1:
            plt.show()
        else:
            plt.close()

        fig = plt.figure(figsize=(12, 8))
        plt.scatter(ts, waveform - dpz_model(ts, *m.values), c="r")
        plt.ylabel("ADC")
        plt.xlabel("Time [Samples]")
        plt.title("DPZ Model Waveform Fit Residuals")
        plt.grid(True)
        out_plot_dict["dpz_model_fit_residuals"] = fig
        if plot > 1:
            plt.show()
        else:
            plt.close()

    return m.values[1], m.values[2], m.values[3], out_plot_dict


def tp100_align(wfs: np.array, tp100_window_width: int, tp100s: np.array) -> np.array:
    """
    Align provided waveforms at their maximum, so that an average over all the waveforms creates a valid superpulse to fit using :func:`dpz_model_fit`.
    Do this without sacrificing too much of the length of the decaying tail

    Parameters
    ----------
    wfs
        An array of arrays containing waveforms
    tp100_window_width
        The window of acceptance, with a center set at the median tp100 value. If a tp100 falls outside this window, ignore this waveform for the superpulse
    tp100s
        An array containing the indices of the maximums of the waveforms, in samples

    Returns
    -------
    time_aligned_wfs
        An array of waveforms that are all aligned at their maximal values
    """
    tp100_window_width = (
        13  # If tp100 isn't in +/- this window of the median, forget about it
    )
    median_tp100 = int(np.nanmedian(tp100s))  # in samples
    wf_len = len(wfs[0])
    time_aligned_wfs = []

    for i, wf in enumerate(wfs):
        if np.isnan(tp100s[i]):
            pass
        elif (
            median_tp100 - tp100_window_width
            <= tp100s[i]
            <= median_tp100 + tp100_window_width
        ):
            wf_win = wf[
                tp100s[i]
                - (median_tp100 - tp100_window_width) : wf_len
                - (-1 * tp100s[i] + median_tp100 + tp100_window_width)
            ]
            time_aligned_wfs.append(wf_win)

    return time_aligned_wfs
