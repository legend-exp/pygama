"""
This module is for extracting the pole zero constants from the decay tail
"""

from __future__ import annotations

import logging

import lgdo
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from scipy.stats import linregress

import pygama.pargen.dsp_optimize as opt
from pygama.pargen.data_cleaning import get_mode_stdev

log = logging.getLogger(__name__)


def dpz_model(
    ts: np.array, amp: float, tau1: float, tau2: float, f2: float
) -> np.array:
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
    return amp * (1 - f2) * np.exp(-ts / tau1) + amp * f2 * np.exp(-ts / tau2)


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
        plt.plot(ts[tau2_idx:], slope_2 * ts[tau2_idx:] + intercept_2)
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
    sub_pulse = np.abs(scaled_pulse - np.exp(intercept_2) * np.exp(ts * slope_2))

    # Fit the short time constant
    slope_1, intercept_1, *_ = linregress(ts[:tau1_idx], np.log(sub_pulse[:tau1_idx]))
    # If the fit fails, just return an arbitrary good-enough guess. All detectors have a roughly ~130 sample short time constant for presumming of 8.
    if slope_1 > 0:
        slope_1 = -1 / 130
        intercept_1 = np.log(1 - np.exp(intercept_2))

    if plot > 0:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(ts[:tau1_idx], np.log(sub_pulse[:tau1_idx]))
        plt.plot(ts[:tau1_idx], slope_1 * ts[:tau1_idx] + intercept_1)
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

    def cost_function(amp, tau1, tau2, f2):
        output = dpz_model(ts, amp, tau1, tau2, f2)
        res = 0
        for i in range(len(output)):
            res += (output[i] - waveform[i]) ** 2

        return res

    # Perform the fit
    m = Minuit(
        cost_function,
        amp=np.amax(waveform),
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

    if plot > 0:
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

    # order so tau1 largest
    if m.values[1] > m.values[2]:
        return m.values[1], m.values[2], m.values[3], out_plot_dict
    else:
        return m.values[2], m.values[1], 1 - m.values[3], out_plot_dict


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


class PZCorrect:
    def __init__(self, dsp_config, wf_field, debug_mode=False):
        self.dsp_config = dsp_config
        self.wf_field = wf_field
        self.output_dict = {}
        self.results_dict = {}
        self.debug_mode = debug_mode

    def get_single_decay_constant(
        self, tb_data: lgdo.Table, slope_param="tail_slope", display=0
    ):
        """
        Finds the decay constant from the modal value of the tail slope after cuts
        and saves it to the specified json. Updates self.output_dict with tau value

        Parameters
        ----------
        - slopes: numpy array of tail slopes
        - wfs: WaveformTable object containing waveform data

        """
        tb_out = opt.run_one_dsp(tb_data, self.dsp_config)
        slopes = tb_out[slope_param].nda
        wfs = tb_data[self.wf_field]

        mode, stdev = get_mode_stdev(slopes)
        tau = round(-1 / (mode), 1)
        err = round((-1 / (mode + (stdev / np.sqrt(len(slopes))))) - tau, 1)

        sampling_rate = wfs["dt"].nda[0]
        units = wfs["dt"].attrs["units"]
        tau = f"{tau*sampling_rate}*{units}"
        err = f"{err*sampling_rate}*{units}"

        if "pz" in self.output_dict:
            self.output_dict["pz"].update({"tau1": tau, "tau1_err": err})
        else:
            self.output_dict["pz"] = {"tau1": tau, "tau1_err": err}

        self.results_dict.update(
            {"single_decay_constant": {"slope_pars": {"mode": mode, "stdev": stdev}}}
        )

        if display <= 0:
            return
        else:
            out_plot_dict = {}
            return out_plot_dict

    def get_dpz_decay_constants(
        self,
        tb_data: lgdo.Table,
        percent_tau1_fit: float = 0.1,
        percent_tau2_fit: float = 0.2,
        offset_from_wf_max: int = 10,
        superpulse_bl_idx: int = 25,
        superpulse_window_width: int = 13,
        display: int = 0,
    ) -> dict:
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

        # Get high energy waveforms to create a superpulse. Eventually allow user which peak to select? For now, use the 2615 keV peak
        sampling_rate = tb_data[self.wf_field]["dt"].nda[0]
        units = tb_data[self.wf_field]["dt"].attrs["units"]

        high_e_wfs = tb_data[self.wf_field]["values"].nda[:]

        # Time align the waveforms to their maximum
        tp100s = []
        for wf in high_e_wfs:
            tp100s.append(np.argmax(wf))

        time_aligned_wfs = tp100_align(high_e_wfs, superpulse_window_width, tp100s)

        # Baseline subtract the time aligned waveforms
        bl_sub_time_aligned_wfs = []

        for i in range(len(time_aligned_wfs)):
            bl_sub_time_aligned_wfs.append(
                time_aligned_wfs[i] - np.mean(time_aligned_wfs[i][:superpulse_bl_idx])
            )

        # Create a superpulse
        superpulse = np.mean(bl_sub_time_aligned_wfs, axis=0)

        # Fit the superpulse and get rough DPZ constants
        tau1s_fit, tau2s_fit, f2s_fit, out_plot_dict = dpz_model_fit(
            superpulse,
            percent_tau1_fit=percent_tau1_fit,
            percent_tau2_fit=percent_tau2_fit,
            idx_shift=offset_from_wf_max,
            plot=display,
        )
        log.debug("Found initial guesses for DPZ constants:")
        for item, value in {
            "tau1": f"{tau1s_fit * sampling_rate}*{units}",
            "tau2": f"{tau2s_fit * sampling_rate}*{units}",
            "frac": f2s_fit,
        }.items():
            log.debug(f"{item}: {value}")

        self.results_dict.update(
            {
                "double_pole_zero": {
                    "guesses": {"tau1": tau1s_fit, "tau2": tau2s_fit, "frac": f2s_fit}
                }
            }
        )

        # Optimize the flatness of high energy waveforms to get optimal DPZ constants
        dpz_opt_tb_out = opt.run_one_dsp(
            tb_data,
            self.dsp_config,
            db_dict=dict(
                {"pz": {"tau1": tau1s_fit, "tau2": tau2s_fit, "frac": f2s_fit}}
            ),
        )

        # Update tau_dict with the dpz constants
        tau1 = np.nanmedian(dpz_opt_tb_out["tau1"].nda)
        tau2 = np.nanmedian(dpz_opt_tb_out["tau2"].nda)
        frac = np.nanmedian(dpz_opt_tb_out["frac"].nda)
        tau1_err = (
            np.nanpercentile(dpz_opt_tb_out["tau1"].nda, 68.27)
            - np.nanpercentile(dpz_opt_tb_out["tau1"].nda, 31.73)
        ) / 2
        tau2_err = (
            np.nanpercentile(dpz_opt_tb_out["tau2"].nda, 68.27)
            - np.nanpercentile(dpz_opt_tb_out["tau2"].nda, 31.73)
        ) / 2
        frac_err = (
            np.nanpercentile(dpz_opt_tb_out["frac"].nda, 68.27)
            - np.nanpercentile(dpz_opt_tb_out["frac"].nda, 31.73)
        ) / 2

        if "units" in dpz_opt_tb_out["tau1"].attrs and dpz_opt_tb_out["tau1"].attrs[
            "units"
        ] not in ["ADC", "sample"]:
            tau1 = f'{tau1}*{dpz_opt_tb_out["tau1"].attrs["units"]}'
            tau1_err = f'{tau1_err}*{dpz_opt_tb_out["tau1"].attrs["units"]}'
        else:
            tau1 = f"{tau1*sampling_rate}*{units}"
            tau1_err = f"{tau1_err*sampling_rate}*{units}"

        if "units" in dpz_opt_tb_out["tau2"].attrs and dpz_opt_tb_out["tau2"].attrs[
            "units"
        ] not in ["ADC", "sample"]:
            tau2 = f'{tau2}*{dpz_opt_tb_out["tau2"].attrs["units"]}'
            tau2_err = f'{tau2_err}*{dpz_opt_tb_out["tau2"].attrs["units"]}'
        else:
            tau2 = f"{tau2*sampling_rate}*{units}"
            tau2_err = f"{tau2_err*sampling_rate}*{units}"

        output_dict = {
            "tau1": tau1,
            "tau2": tau2,
            "frac": frac,
            "tau1_err": tau1_err,
            "tau2_err": tau2_err,
            "frac_err": frac_err,
        }
        if "pz" in self.output_dict:
            self.output_dict["pz"].update(output_dict)
        else:
            self.output_dict["pz"] = output_dict

        if display <= 0:
            return
        else:
            return out_plot_dict

    def plot_waveforms_after_correction(
        self,
        tb_data,
        wf_field,
        xlim=(0, 1024),
        ylim=None,
        norm_param=None,
        display=0,
        figsize=(8, 6),
        fontsize=8,
    ):
        tb_out = opt.run_one_dsp(tb_data, self.dsp_config, db_dict=self.output_dict)
        wfs = tb_out[wf_field]["values"].nda
        wf_idxs = np.random.choice(len(wfs), 100)
        if norm_param is not None:
            means = tb_out[norm_param].nda[wf_idxs]
            wfs = np.divide(wfs[wf_idxs], np.reshape(means, (len(wf_idxs), 1)))
        else:
            wfs = wfs[wf_idxs]
        plt.rcParams["font.size"] = fontsize
        fig = plt.figure(figsize=figsize)
        for wf in wfs:
            plt.plot(np.arange(0, len(wf), 1), wf)
        plt.axhline(1, color="black")
        plt.axhline(0, color="black")
        plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel("Samples")
        plt.ylabel("ADU")
        plot_dict = {"waveforms": fig}
        if display > 1:
            plt.show()
        else:
            plt.close()
        return plot_dict

    def plot_slopes(
        self,
        tb_data,
        slope_field,
        with_correction=False,
        display=0,
        figsize=(8, 6),
        fontsize=8,
    ):

        tb_out = opt.run_one_dsp(
            tb_data,
            self.dsp_config,
            db_dict=self.output_dict if with_correction else {},
        )
        slopes = tb_out[slope_field].nda

        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots()
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
        bins = np.arange(
            np.nanpercentile(slopes, 1),
            np.nanpercentile(slopes, 99),
            np.nanpercentile(slopes, 51) - np.nanpercentile(slopes, 50),
        )
        counts, bins, bars = ax.hist(slopes, bins=bins, histtype="step")
        plt.xlabel("Slope")
        plt.ylabel("Counts")
        if "single_decay_constant" in self.results_dict:
            high_bin = self.results_dict["single_decay_constant"]["slope_pars"]["mode"]
            sigma = self.results_dict["single_decay_constant"]["slope_pars"]["stdev"]
            ax.axvline(high_bin, color="red")
            in_min = high_bin - 4 * sigma
            in_max = high_bin + 4 * sigma
            axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])
            axins.hist(
                slopes[(slopes > in_min) & (slopes < in_max)],
                bins=50,
                histtype="step",
            )
            axins.axvline(high_bin, color="red")
            axins.set_xlim(in_min, in_max)
            ax.set_xlim(np.nanpercentile(slopes, 1), np.nanpercentile(slopes, 99))
        if with_correction:
            out_plot_dict = {"corrected_slope": fig}
        else:
            out_plot_dict = {"slope": fig}
        if display > 1:
            plt.show()
        else:
            plt.close()
        return out_plot_dict
