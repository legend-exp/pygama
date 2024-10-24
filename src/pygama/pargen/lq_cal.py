from __future__ import annotations

import logging
import re
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from matplotlib.colors import LogNorm
from scipy.stats import linregress

import pygama.math.histogram as pgh
import pygama.pargen.AoE_cal as AoE
from pygama.math.distributions import gaussian
from pygama.pargen.survival_fractions import compton_sf_sweep, get_sf_sweep

log = logging.getLogger(__name__)


def get_fit_range(lq: np.array) -> tuple(float, float):
    """
    Function for determining the fit range for a given distribution of lq values
    """

    # Get an initial guess of mu and sigma, use these values to determine our final fit range
    left_guess = np.nanpercentile(lq, 1)
    right_guess = np.nanpercentile(lq, 95)
    test_range = (left_guess, right_guess)

    hist, bins, _ = pgh.get_hist(lq, bins=100, range=test_range)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)

    left_edge = mu - 2.5 * sigma
    right_edge = mu + 2.5 * sigma
    fit_range = (left_edge, right_edge)

    return fit_range


def get_lq_hist(
    df: pd.DataFrame(),
    lq_param: str,
    cal_energy_param: str,
    peak: float,
    sidebands: bool = True,
):
    """
    Function for getting a distribution of LQ values for a given peak. Returns a histogram of the
    LQ distribution as well as an array of bin edges
    """

    if sidebands:
        # Get a histogram of events in the peak using sideband subtraction
        # Uses a 13 keV window, and the sideband is to the right of the peak
        # Default option

        df_peak = df.query(
            f"{cal_energy_param} < ({peak} + 5) & {cal_energy_param} > ({peak} - 8)"
        )
        df_sideband = df.query(
            f"{cal_energy_param} < ({peak} + 20) & {cal_energy_param} > ({peak} + 7)"
        )

        fit_range = get_fit_range(df_peak[lq_param].to_numpy())

        sideband_hist, bins, _ = pgh.get_hist(
            df_sideband[lq_param].to_numpy(), bins=100, range=fit_range
        )
        dep_hist, _, _ = pgh.get_hist(
            df_peak[lq_param].to_numpy(), bins=100, range=fit_range
        )
        final_hist = dep_hist - sideband_hist
        var = np.sqrt(np.add(sideband_hist, dep_hist))

        return final_hist, bins, var

    else:
        # Return a histogram in a 13 keV range surrounding the specified peak

        df_peak = df.query(
            f"{cal_energy_param} < ({peak} + 5) & {cal_energy_param} > ({peak} - 8)"
        )

        fit_range = get_fit_range(df_peak[lq_param].to_numpy())
        dep_hist, bins, var = pgh.get_hist(
            df_peak[lq_param].to_numpy(), bins=100, range=fit_range
        )

        return dep_hist, bins, var


def binned_lq_fit(
    df: pd.DataFrame,
    lq_param: str,
    cal_energy_param: str,
    peak: float,
    cdf=gaussian,
    sidebands: bool = True,
):
    """Function for fitting a distribution of LQ values within a specified
        energy peak. Fits a gaussian to the distribution

    Parameters
    ----------
    df: pd.DataFrame()
        Dataframe containing the data for fitting. Data must
        contain the desired lq parameter and the calibrated
        energy
    lq_param: string
        Name of the LQ parameter to fit
    cal_energy_param: string
        Name of the calibrated energy parameter of choice
    peak: float
        Energy value, in keV, of the peak who's LQ
        distribution will be fit
    cdf: callable
        Function to be used for the binned fit
    sidebands: bool
        Whether or not to perform a sideband subtraction when
        fitting the LQ distribution

    Returns
    -------
    m1.values: array-like object
        Resulting parameter values from the peak fit
    m1.errors: array-like object
        Resulting parameter errors from the peak fit
    hist: array
        Histogram that was used for the binned fit
    bins: array
        Array of bin edges used for the binned fit
    """

    hist, bins, var = get_lq_hist(df, lq_param, cal_energy_param, peak, sidebands)

    # Temporary fix for negative bin counts
    # TODO: Adjust fitting to handle negative bin counts
    hist[hist < 0] = 0

    bin_centers = (bins[:-1] + bins[1:]) / 2

    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)

    c1 = cost.BinnedNLL(hist, bins, gaussian.get_cdf, verbose=0)
    m1 = Minuit(c1, mu=mu, sigma=sigma)
    m1.simplex().migrad()
    m1.hesse()

    return m1.values, m1.errors, hist, bins


def calculate_time_means(
    df: pd.DataFrame,
    lq_param: str,
    cal_energy_param: str,
    peak: float,
    sidebands: bool = True,
):
    """
    Function for calculating the arithmetic mean and sigma for LQ events in a
    specified peak
    """

    df_peak = df.query(
        f"{cal_energy_param} < ({peak} + 5) & {cal_energy_param} > ({peak} - 8)"
    )
    fit_range = get_fit_range(df_peak[lq_param].to_numpy())

    lq_peak_vals = df_peak.query(
        f"{lq_param} < {fit_range[1]} & {lq_param} > {fit_range[0]}"
    )[lq_param].to_numpy()

    mean = np.mean(lq_peak_vals)
    sigma = np.std(lq_peak_vals)
    mean_err = sigma / np.sqrt(len(lq_peak_vals))
    sig_err = 2 * sigma**4 / (len(lq_peak_vals) - 1)

    pars = {"mu": mean, "sigma": sigma}
    errors = {"mu": mean_err, "sigma": sig_err}

    return pars, errors


def fit_time_means(tstamps, means, reses):
    out_dict = {}
    current_tstamps = []
    current_means = []
    current_reses = []

    rolling_mean = means[np.where(~np.isnan(means))[0][0]]
    for i, tstamp in enumerate(tstamps):
        if np.isnan(means[i]) or np.isnan(reses[i]):
            out_dict[tstamp] = np.nan

        else:
            current_tstamps.append(tstamp)
            current_means.append(means[i])
            current_reses.append(reses[i])
            rolling_mean = np.average(
                current_means, weights=1 / np.array(current_reses)
            )
    for tstamp in current_tstamps:
        out_dict[tstamp] = rolling_mean
    return out_dict


class LQCal:
    """A class for calibrating the LQ parameter and determining the LQ cut value"""

    def __init__(
        self,
        cal_dicts: dict,
        cal_energy_param: str,
        dt_param: str,
        eres_func: callable,
        cdf: callable = gaussian,
        selection_string: str = "is_valid_cal&is_not_pulser",
        debug_mode=False,
    ):
        """
        Parameters
        ----------
        cal_dicts: dict
            A dictionary containing the hit-level operations to apply
            to the data.
        cal_energy_param: string
            The calibrated energy parameter of choice
        dt_param: string
            The drift-time parameter of choice
        eres_function: callable
            The energy resolutions function
        cdf: callable
            The CDF used for the binned fits
        selection_string: string
            A string of flags to apply the data when running the calibration
        debug_mode: boolean
            Determines if the class is initialized in debug mode
        """

        self.cal_dicts = cal_dicts
        self.cal_energy_param = cal_energy_param
        self.dt_param = dt_param
        self.eres_func = eres_func
        self.cdf = cdf
        self.selection_string = selection_string
        self.debug_mode = debug_mode

    def update_cal_dicts(self, update_dict):
        if re.match(r"(\d{8})T(\d{6})Z", list(self.cal_dicts)[0]):
            for tstamp in self.cal_dicts:
                if tstamp in update_dict:
                    self.cal_dicts[tstamp].update(update_dict[tstamp])
                else:
                    self.cal_dicts[tstamp].update(update_dict)
        else:
            self.cal_dicts.update(update_dict)

    def lq_timecorr(self, df, lq_param, output_name="LQ_Timecorr", display=0):
        """
        Calculates the average LQ value for DEP events for each specified
        run_timestamp. Applies a time normalization based on the average LQ value
        in the DEP across all run_timestamps.
        """

        log.info("Starting LQ time correction")
        self.timecorr_df = pd.DataFrame()
        try:
            if "run_timestamp" in df:
                for tstamp, time_df in df.groupby("run_timestamp", sort=True):
                    try:
                        pars, errs = calculate_time_means(
                            time_df.query(f"{self.selection_string}"),
                            lq_param,
                            self.cal_energy_param,
                            peak=1592.5,
                        )
                        self.timecorr_df = pd.concat(
                            [
                                self.timecorr_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "run_timestamp": tstamp,
                                            "mean": pars["mu"],
                                            "mean_err": errs["mu"],
                                            "res": pars["sigma"] / pars["mu"],
                                            "res_err": (pars["sigma"] / pars["mu"])
                                            * np.sqrt(
                                                errs["sigma"] / pars["sigma"]
                                                + errs["mu"] / pars["mu"]
                                            ),
                                        }
                                    ]
                                ),
                            ]
                        )
                    except BaseException as e:
                        if e == KeyboardInterrupt:
                            raise (e)
                        elif self.debug_mode:
                            raise (e)

                        self.timecorr_df = pd.concat(
                            [
                                self.timecorr_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "run_timestamp": tstamp,
                                            "mean": np.nan,
                                            "mean_err": np.nan,
                                            "res": np.nan,
                                            "res_err": np.nan,
                                        }
                                    ]
                                ),
                            ]
                        )

                self.timecorr_df.set_index("run_timestamp", inplace=True)
                time_dict = fit_time_means(
                    np.array(self.timecorr_df.index),
                    np.array(self.timecorr_df["mean"]),
                    np.array(self.timecorr_df["res"]),
                )

                df[output_name] = df[lq_param] / np.array(
                    [time_dict[tstamp] for tstamp in df["run_timestamp"]]
                )
                self.update_cal_dicts(
                    {
                        tstamp: {
                            output_name: {
                                "expression": f"{lq_param}/a",
                                "parameters": {"a": t_dict},
                            }
                        }
                        for tstamp, t_dict in time_dict.items()
                    }
                )
                log.info("LQ time correction finished")
            else:
                try:
                    pars, errs = calculate_time_means(
                        df.query(f"{self.selection_string}"),
                        lq_param,
                        self.cal_energy_param,
                        peak=1592.5,
                    )
                    self.timecorr_df = pd.concat(
                        [
                            self.timecorr_df,
                            pd.DataFrame(
                                [
                                    {
                                        "run_timestamp": np.nan,
                                        "mean": pars["mu"],
                                        "mean_err": errs["mu"],
                                        "res": pars["sigma"] / pars["mu"],
                                        "res_err": (pars["sigma"] / pars["mu"])
                                        * np.sqrt(
                                            errs["sigma"] / pars["sigma"]
                                            + errs["mu"] / pars["mu"]
                                        ),
                                    }
                                ]
                            ),
                        ]
                    )
                except BaseException as e:
                    if e == KeyboardInterrupt:
                        raise (e)
                    elif self.debug_mode:
                        raise (e)
                    self.timecorr_df = pd.concat(
                        [
                            self.timecorr_df,
                            pd.DataFrame(
                                [
                                    {
                                        "run_timestamp": np.nan,
                                        "mean": np.nan,
                                        "mean_err": np.nan,
                                        "res": np.nan,
                                        "res_err": np.nan,
                                    }
                                ]
                            ),
                        ]
                    )
                df[output_name] = df[lq_param] / pars["mu"]
                self.update_cal_dicts(
                    {
                        output_name: {
                            "expression": f"{lq_param}/a",
                            "parameters": {"a": pars["mu"]},
                        }
                    }
                )
                log.info("LQ time correction finished")
        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
                raise (e)
            log.error("LQ time correction failed")
            self.update_cal_dicts(
                {
                    output_name: {
                        "expression": f"{lq_param}/a",
                        "parameters": {"a": np.nan},
                    }
                }
            )

    def drift_time_correction(
        self, df: pd.DataFrame(), lq_param, cal_energy_param: str, display: int = 0
    ):
        """
        Deterimines the drift time correction parameters for LQ by fitting a degree 1 polynomial to
        the LQ vs drift time distribution for DEP events. Corrects for any linear dependence and
        centers the final LQ distribution to a mean of 0.
        """

        log.info("Starting LQ drift time correction")
        try:
            pars = binned_lq_fit(df, lq_param, self.cal_energy_param, peak=1592.5)[0]
            mean = pars[0]
            sigma = pars[1]

            dep_events = df.query(
                f"{self.cal_energy_param} > 1589.5 & {self.cal_energy_param} < 1595.5 & {self.cal_energy_param}=={self.cal_energy_param}&{lq_param}=={lq_param}"
            )

            dt_range = [
                np.nanpercentile(dep_events[self.dt_param], 10),
                np.nanpercentile(dep_events[self.dt_param], 95),
            ]

            lq_range = [mean - 2 * sigma, mean + 2 * sigma]

            self.lq_range = lq_range
            self.dt_range = dt_range

            final_df = dep_events.query(
                f"{lq_param} > {lq_range[0]} & {lq_param} < {lq_range[1]} & {self.dt_param} > {dt_range[0]} & {self.dt_param} < {dt_range[1]}"
            )

            result = linregress(
                final_df[self.dt_param],
                final_df[lq_param],
                alternative="greater",
            )
            self.dt_fit_pars = result

            df["LQ_Corrected"] = (
                df[lq_param]
                - df[self.dt_param] * self.dt_fit_pars[0]
                - self.dt_fit_pars[1]
            )

        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
                raise (e)
            log.error("LQ drift time correction failed")
            self.dt_fit_pars = (np.nan, np.nan)

        self.update_cal_dicts(
            {
                "LQ_Corrected": {
                    "expression": f"{lq_param} - dt_eff*a - b",
                    "parameters": {"a": self.dt_fit_pars[0], "b": self.dt_fit_pars[1]},
                }
            }
        )

    def get_cut_lq_dep(self, df: pd.DataFrame(), lq_param: str, cal_energy_param: str):
        """
        Determines the cut value for LQ. Value is calculated by fitting the LQ distribution
        for events in the DEP to a gaussian. The LQ values are normalized by the fitted
        sigma, and the cut value is set to a value of 3. Sideband subtraction is used to
        determine the LQ distribution for DEP events. Events greater than the cut value
        fail the cut.
        """

        log.info("Starting LQ Cut calculation")
        try:

            pars, errs, hist, bins = binned_lq_fit(
                df, lq_param, cal_energy_param, peak=1592.5
            )

            self.cut_fit_pars = pars
            self.cut_fit_errs = errs
            self.fit_hist = (hist, bins)
            self.cut_val = 3

            df["LQ_Classifier"] = np.divide(df[lq_param].to_numpy(), pars[1])
            df["LQ_Cut"] = df["LQ_Classifier"] < self.cut_val

        except BaseException as e:
            if e == KeyboardInterrupt:
                raise (e)
            elif self.debug_mode:
                raise (e)
            log.error("LQ cut determination failed")
            self.cut_val = np.nan
            c = cost.UnbinnedNLL(np.array([0]), gaussian.pdf)
            m = Minuit(c, np.full(2, np.nan))
            self.cut_fit_pars = pars = m.values

        self.update_cal_dicts(
            {
                "LQ_Classifier": {
                    "expression": f"({lq_param} / a)",
                    "parameters": {"a": pars[1]},
                },
                "LQ_Cut": {
                    "expression": "(LQ_Classifier < a)",
                    "parameters": {"a": self.cut_val},
                },
            }
        )

    def calibrate(self, df, initial_lq_param):
        """Run the LQ calibration and calculate the cut value"""

        self.lq_timecorr(df, initial_lq_param)
        log.info("Finished LQ Time Correction")

        self.drift_time_correction(
            df, lq_param="LQ_Timecorr", cal_energy_param=self.cal_energy_param
        )
        log.info("Finished LQ Drift Time Correction")

        self.get_cut_lq_dep(
            df, lq_param="LQ_Corrected", cal_energy_param=self.cal_energy_param
        )
        log.info("Finished Calculating the LQ Cut Value")

        final_lq_param = "LQ_Classifier"
        peaks_of_interest = [1592.5, 2039, 2103.53, 2614.50]
        self.low_side_sf = pd.DataFrame()
        fit_widths = [(40, 25), (0, 0), (25, 40), (50, 50)]
        self.low_side_peak_dfs = {}

        log.info("Calculating peak survival fractions")
        for i, peak in enumerate(peaks_of_interest):
            try:
                select_df = df.query(f"{self.selection_string}")
                fwhm = self.eres_func(peak)
                if peak == 2039:
                    emin = 2 * fwhm
                    emax = 2 * fwhm
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{peak-emin})&({self.cal_energy_param}<{peak+emax})"
                    )

                    cut_df, sf, sf_err = compton_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[final_lq_param].to_numpy(),
                        self.cut_val,
                        cut_range=(0, 5),
                        n_samples=30,
                        mode="less",
                    )
                    self.low_side_sf = pd.concat(
                        [
                            self.low_side_sf,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    self.low_side_peak_dfs[peak] = cut_df
                else:
                    emin, emax = fit_widths[i]
                    fit_range = (peak - emin, peak + emax)
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{fit_range[0]})&({self.cal_energy_param}<{fit_range[1]})"
                    )
                    cut_df, sf, sf_err = get_sf_sweep(
                        peak_df[self.cal_energy_param].to_numpy(),
                        peak_df[final_lq_param].to_numpy(),
                        self.cut_val,
                        peak,
                        fwhm,
                        fit_range=fit_range,
                        cut_range=(0, 5),
                        n_samples=30,
                        mode="less",
                    )
                    self.low_side_sf = pd.concat(
                        [
                            self.low_side_sf,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    self.low_side_peak_dfs[peak] = cut_df
                log.info(f"{peak}keV: {sf:2.1f} +/- {sf_err:2.1f} %")
            except BaseException as e:
                if e == KeyboardInterrupt:
                    raise (e)
                elif self.debug_mode:
                    raise (e)
                self.low_side_sf = pd.concat(
                    [
                        self.low_side_sf,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(f"LQ Survival fraction determination failed for {peak} peak")
        self.low_side_sf.set_index("peak", inplace=True)


def plot_lq_mean_time(
    lq_class, data, lq_param="LQ_Timecorr", figsize=(12, 8), fontsize=12
) -> plt.figure:
    """Plots the mean LQ value calculated for each given timestamp"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(1, 1)
    try:
        ax.errorbar(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in lq_class.timecorr_df.index
            ],
            lq_class.timecorr_df["mean"],
            yerr=lq_class.timecorr_df["mean_err"],
            linestyle=" ",
        )

        grouped_means = [
            cal_dict["LQ_Timecorr"]["parameters"]["a"]
            for tstamp, cal_dict in lq_class.cal_dicts.items()
        ]
        ax.step(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in lq_class.cal_dicts
            ],
            grouped_means,
            where="post",
        )
        ax.fill_between(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in lq_class.cal_dicts
            ],
            y1=np.array(grouped_means) - 0.2 * np.array(lq_class.timecorr_df["res"]),
            y2=np.array(grouped_means) + 0.2 * np.array(lq_class.timecorr_df["res"]),
            color="green",
            alpha=0.2,
        )
        ax.fill_between(
            [
                datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                for tstamp in lq_class.cal_dicts
            ],
            y1=np.array(grouped_means) - 0.4 * np.array(lq_class.timecorr_df["res"]),
            y2=np.array(grouped_means) + 0.4 * np.array(lq_class.timecorr_df["res"]),
            color="yellow",
            alpha=0.2,
        )
    except Exception:
        pass
    ax.set_xlabel("time")
    ax.set_ylabel("LQ mean")
    myfmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(myfmt)
    plt.close()
    return fig


def plot_drift_time_correction(
    lq_class, data, lq_param="LQ_Timecorr", figsize=(12, 8), fontsize=12
) -> plt.figure:
    """Plots a 2D histogram of LQ versus effective drift time in a 6 keV
    window around the DEP. Additionally plots the fit results for the
    drift time correction."""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(1, 1)

    try:
        dep_range = (1590, 1595)

        initial_df = data[
            (data[lq_class.cal_energy_param] > dep_range[0])
            & (data[lq_class.cal_energy_param] < dep_range[1])
        ]
        max_dt = 1500
        max_lq = 2.5

        plt.hist2d(
            initial_df["dt_eff"],
            initial_df[lq_param],
            bins=100,
            range=((0, max_dt), (0, max_lq)),
            norm=mcolors.LogNorm(),
        )

        x = np.linspace(0, max_dt, 100)
        model = lq_class.dt_fit_pars[0] * x + lq_class.dt_fit_pars[1]

        plt.plot(x, model, color="r")

        plt.axvline(lq_class.dt_range[0], color="k")
        plt.axvline(lq_class.dt_range[1], color="k")
        plt.axhline(lq_class.lq_range[0], color="k")
        plt.axhline(lq_class.lq_range[1], color="k")

        plt.xlabel("Drift Time (ns)")
        plt.ylabel("LQ")

        plt.title("LQ versus Drift Time for DEP")

    except Exception:
        pass

    plt.tight_layout()
    plt.close()
    return fig


def plot_lq_cut_fit(lq_class, data, figsize=(12, 8), fontsize=12) -> plt.figure:
    """Plots the final histogram of LQ values for events in the
    DEP, and the fit results used for determining the cut
    value"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize
    fig, (ax1, ax2) = plt.subplots(2, 1)
    try:
        hist, bins = lq_class.fit_hist
        fit_pars = lq_class.cut_fit_pars

        x_low = bins[0]
        x_high = bins[-1]

        ax1.stairs(hist, bins, label="data")
        xs = np.linspace(round(bins[0], 3), round(bins[-1], 3), len(bins) - 1)
        ls = np.sum(hist)
        dx = np.diff(bins)
        ax1.plot(
            xs,
            gaussian.pdf_norm(xs, x_low, x_high, fit_pars[0], fit_pars[1]) * dx * ls,
            label="Gaussian Fit",
        )

        # ax1.set_xlabel('LQ')
        ax1.set_title("Fit of LQ events in DEP")
        ax1.legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        reses = (
            hist
            - (
                gaussian.pdf_norm(bin_centers, x_low, x_high, fit_pars[0], fit_pars[1])
                * dx
                * ls
            )
        ) / (
            gaussian.pdf_norm(bin_centers, x_low, x_high, fit_pars[0], fit_pars[1])
            * dx
            * ls
        )
        ax2.plot(bin_centers, reses, marker="s", linestyle="")
        ax2.set_xlabel("LQ")
        ax2.set_ylabel("residuals")

    except Exception:
        pass

    plt.tight_layout()
    plt.close()
    return fig


def plot_survival_fraction_curves(
    lq_class, data, figsize=(12, 8), fontsize=12
) -> plt.figure:
    """Plots the survival fraction curves as a function of
    LQ cut values for every peak of interest"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    try:
        plt.vlines(
            lq_class.cut_val,
            0,
            100,
            label=f"cut value: {lq_class.cut_val:1.2f}",
            color="black",
        )

        for peak, survival_df in lq_class.low_side_peak_dfs.items():
            try:
                plt.errorbar(
                    survival_df.index,
                    survival_df["sf"],
                    yerr=survival_df["sf_err"],
                    label=f'{AoE.get_peak_label(peak)} {peak} keV: {lq_class.low_side_sf.loc[peak]["sf"]:2.1f} +/- {lq_class.low_side_sf.loc[peak]["sf_err"]:2.1f} %',
                )
            except Exception:
                pass
    except Exception:
        pass
    vals, labels = plt.yticks()
    plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
    plt.legend(loc="lower right")
    plt.xlabel("cut value")
    plt.ylabel("survival percentage")
    plt.ylim([0, 105])
    plt.close()
    return fig


def plot_sf_vs_energy(
    lq_class, data, xrange=(900, 3000), n_bins=701, figsize=(12, 8), fontsize=12
) -> plt.figure:
    """Plots the survival fraction as a function of energy"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    try:
        bins = np.linspace(xrange[0], xrange[1], n_bins)
        counts_pass, bins_pass, _ = pgh.get_hist(
            data.query(f"{lq_class.selection_string}&LQ_Cut")[
                lq_class.cal_energy_param
            ],
            bins=bins,
        )
        counts, bins, _ = pgh.get_hist(
            data.query(lq_class.selection_string)[lq_class.cal_energy_param],
            bins=bins,
        )
        survival_fracs = counts_pass / (counts + 10**-99)

        plt.step(pgh.get_bin_centers(bins_pass), 100 * survival_fracs)
    except Exception:
        pass
    plt.ylim([0, 100])
    vals, labels = plt.yticks()
    plt.yticks(vals, [f"{x:,.0f} %" for x in vals])
    plt.xlabel("energy (keV)")
    plt.ylabel("survival percentage")
    plt.close()
    return fig


def plot_spectra(
    lq_class,
    data,
    xrange=(900, 3000),
    n_bins=2101,
    xrange_inset=(1580, 1640),
    n_bins_inset=200,
    figsize=(12, 8),
    fontsize=12,
) -> plt.figure:
    """Plots a 2D histogram of the LQ classifier vs calibrated energy"""

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots()
    try:
        bins = np.linspace(xrange[0], xrange[1], n_bins)
        ax.hist(
            data.query(lq_class.selection_string)[lq_class.cal_energy_param],
            bins=bins,
            histtype="step",
            label="before PSD",
        )
        # ax.hist(
        #     data.query(f"{lq_class.selection_string}&AoE_Double_Sided_Cut")[
        #         lq_class.cal_energy_param
        #     ],
        #     bins=bins,
        #     histtype="step",
        #     label="after double sided A/E cut",
        # )
        ax.hist(
            data.query(f"{lq_class.selection_string}&LQ_Cut")[
                lq_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
            label="after LQ cut",
        )
        ax.hist(
            data.query(f"{lq_class.selection_string} & (~LQ_Cut)")[
                lq_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
            label="rejected by LQ cut",
        )

        axins = ax.inset_axes([0.25, 0.07, 0.4, 0.3])
        bins = np.linspace(xrange_inset[0], xrange_inset[1], n_bins_inset)
        select_df = data.query(
            f"{lq_class.cal_energy_param}<{xrange_inset[1]}&{lq_class.cal_energy_param}>{xrange_inset[0]}"
        )
        axins.hist(
            select_df.query(lq_class.selection_string)[lq_class.cal_energy_param],
            bins=bins,
            histtype="step",
        )
        # axins.hist(
        #     select_df.query(f"{lq_class.selection_string}&AoE_Double_Sided_Cut")[
        #         lq_class.cal_energy_param
        #     ],
        #     bins=bins,
        #     histtype="step",
        # )
        axins.hist(
            select_df.query(f"{lq_class.selection_string}&LQ_Cut")[
                lq_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
        )
        axins.hist(
            select_df.query(f"{lq_class.selection_string} & (~LQ_Cut)")[
                lq_class.cal_energy_param
            ],
            bins=bins,
            histtype="step",
        )
    except Exception:
        pass
    ax.set_xlim(xrange)
    ax.set_yscale("log")
    plt.xlabel("energy (keV)")
    plt.ylabel("counts")
    plt.legend(loc="upper left")
    plt.close()
    return fig


def plot_classifier(
    lq_class,
    data,
    lq_param="LQ_Classifier",
    xrange=(800, 3000),
    yrange=(-10, 30),
    xn_bins=700,
    yn_bins=500,
    figsize=(12, 8),
    fontsize=12,
) -> plt.figure:
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure()
    try:
        plt.hist2d(
            data.query(lq_class.selection_string)[lq_class.cal_energy_param],
            data.query(lq_class.selection_string)[lq_param],
            bins=[
                np.linspace(xrange[0], xrange[1], xn_bins),
                np.linspace(yrange[0], yrange[1], yn_bins),
            ],
            norm=LogNorm(),
        )
    except Exception:
        pass
    plt.xlabel("energy (keV)")
    plt.ylabel(lq_param)
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.close()
    return fig
