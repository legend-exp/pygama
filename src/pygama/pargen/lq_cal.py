"""
Calibration routines for the Late Charge (LQ) parameter in HPGe detectors.

LQ is computed from the ratio of the late-charge integral to the total charge
and is sensitive to multi-site events.  This module provides functions to
* determine the fit range and histogram the LQ distribution,
* fit the distribution with a Gaussian model,
* perform a drift-time correction for the time-dependent LQ shift,
* calculate the LQ cut value from the DEP (double-escape peak) survival fraction,
* compute survival fractions as a function of LQ cut position,
* produce diagnostic plots.

The main entry point is the :class:`LQCal` class.
"""

from __future__ import annotations

import contextlib
import logging
import re
import traceback
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LinearRegression

import pygama.math.histogram as pgh
import pygama.pargen.AoE_cal as AoE
from pygama.math.distributions import gaussian
from pygama.pargen.survival_fractions import compton_sf_sweep, get_sf_sweep

log = logging.getLogger(__name__)

from hist import Hist
from hist.axis import Regular
from scipy.stats import exponnorm, norm


def fd_bin_width(data):
    """Freedman-Diaconis bin width: best for general use and skewed data."""
    data = np.asarray(data)
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    n = len(data)
    if iqr == 0 or n == 0:
        return None
    return 2 * iqr / np.cbrt(n)


def fit_gaussian_tail_to_histogram(
    hist,
    tail_side="auto",
):
    """
    Fit histogram with Gaussian + smooth one-sided EMG tail.

    Model:
        Gaussian core + smooth right/left exponentially modified Gaussian tail.

    Parameters
    ----------
    y_values : array-like
        Values to histogram and fit.

    tail_side : {"auto", "right", "left"}, optional
        Which side gets the smooth EMG tail.
        If "auto", the function detects the stronger tail automatically.

    Returns
    -------
    mu_fit, sigma_fit, mu_err, sigma_err
        Core Gaussian mean/sigma and errors.
    """

    N = np.sum(hist.values())
    if N < 10:
        return np.nan, np.nan, np.nan, np.nan, None, None, None, None, None

    # get 10%/50%/90% quantile estimates
    cdf = np.cumsum(hist.values())
    cdf = cdf / N
    quant_bins = np.searchsorted(cdf, [0.1, 0.5, 0.9])
    q10, q50, q90 = hist.axes.centers[0][quant_bins]

    # Automatically detect tail side
    if tail_side == "auto":
        tail_side = "right" if (q90 - q50) > (q50 - q10) else "left"
        log.debug(
            f"Using {tail_side} tail: q90-q50 = {q90 - q50}, q50-q10 = {q50 - q10}"
        )

    if tail_side not in ["right", "left"]:
        raise ValueError("tail_side must be 'auto', 'right', or 'left'.")

    # Model: Gaussian + smooth EMG tail
    def model_right(x, A_g, mu, sigma, A_t, tau):
        gaussian = A_g * norm.pdf(x, mu, sigma)

        K = tau / sigma
        tail = A_t * exponnorm.pdf(x, K, loc=mu, scale=sigma)

        return gaussian + tail

    def model_left(x, A_g, mu, sigma, A_t, tau):
        gaussian = A_g * norm.pdf(x, mu, sigma)

        K = tau / sigma
        tail = A_t * exponnorm.pdf(-x, K, loc=-mu, scale=sigma)

        return gaussian + tail

    if tail_side == "right":
        model = model_right
    else:
        model = model_left

    # -------------------------------
    # Initial guesses
    # -------------------------------
    mu0 = np.searchsorted(cdf, 0.5)
    sigma0 = (q90 - q10) / 2.5

    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0

    # Since the model returns expected bin counts,
    # A_g and A_t are approximately event-count amplitudes.
    # As a starting point, assume about 80% of the events belong to the Gaussian core and 20% belong to the tail
    A_g0 = 0.8 * N
    A_t0 = 0.2 * N
    tau0 = sigma0

    try:
        nll = cost.ExtendedBinnedNLL(
            hist.values(),
            hist.axes.edges[0],
            model,
            use_pdf="approximate",
        )
        fitter = Minuit(nll, A_g0, mu0, sigma0, A_t0, tau0)
        fitter.limits[0] = (0, N)
        fitter.limits[1] = (hist.axes.edges[0][0], hist.axes.edges[0][-1])
        fitter.limits[2] = (0, hist.axes.edges[0][-1] - hist.axes.edges[0][0])
        fitter.limits[3] = (0, N)
        fitter.limits[4] = (0, 5 * (hist.axes.edges[0][-1] - hist.axes.edges[0][0]))
        result = fitter.migrad()

        popt = np.array(result.values)
        perr = np.array(result.errors)
        A_g_fit, mu_fit, sigma_fit, A_t_fit, tau_fit = popt
        A_g_err, mu_err, sigma_err, A_t_err, tau_err = perr

        ##########
        if not result.accurate or not result.valid:
            log.warning(f"Fit failed {result!r}")

        log.debug(f"Fit tail_side = {tail_side}")
        log.debug(f"mu    = {mu_fit:.6g} ± {mu_err:.3g}")
        log.debug(f"sigma = {sigma_fit:.6g} ± {sigma_err:.3g}")
        log.debug(f"A_g   = {A_g_fit:.6g} ± {A_g_err:.3g}")
        log.debug(f"A_t   = {A_t_fit:.6g} ± {A_t_err:.3g}")
        log.debug(f"tau   = {tau_fit:.6g} ± {tau_err:.3g}")

        # TO DO: return information about the fit quality
        return (
            mu_fit,
            sigma_fit,
            mu_err,
            sigma_err,
            popt,
            perr,
            model,
        )

    except Exception:
        log.error(f"Gaussian + smooth {tail_side} EMG tail fit failed", exc_info=True)

        return np.nan, np.nan, np.nan, np.nan, None, None, model


def extract_gaussian_trends(
    lq_over_e,
    energy,
    energy_windows,
    #    lq_filter,
    #    det,
    tail_side="auto",
    max_cols=3,
):
    means, mean_errors, sigmas, sigma_errs, bin_centers = [], [], [], [], []

    hist_panels = []

    for i in range(len(energy_windows) - 1):
        lo, hi = energy_windows[i], energy_windows[i + 1]

        idx = (energy > lo) & (energy < hi)
        values = np.asarray(lq_over_e[idx], dtype=float)
        values = values[np.isfinite(values)]

        # this determins what values we are fitting
        if len(values) > 10:
            # -------------------------------
            # Estimate binning
            # -------------------------------
            fit_min, fit_max = get_fit_range(values)
            # fit_max += (fit_max-fit_min)/4 # increase high side range by 50% to help tail fit
            if (
                np.sum((values >= fit_min) & (values <= fit_max) & np.isfinite(values))
                <= 10
            ):
                continue
            bin_width = fd_bin_width(values)
            if bin_width is None or bin_width <= 0:
                bins = 30
            else:
                bins = max(10, int(np.ceil((fit_max - fit_min) / bin_width)))

            hist = Hist(Regular(bins, fit_min, fit_max))
            hist.fill(values)

            result = fit_gaussian_tail_to_histogram(
                hist,
                tail_side=tail_side,
            )

            (
                mean,
                sigma,
                mean_err,
                sigma_err,
                popt,
                perr,
                model,
            ) = result

            if np.isfinite(mean) and np.isfinite(sigma):
                means.append(mean)
                mean_errors.append(mean_err)
                sigmas.append(sigma)
                sigma_errs.append(sigma_err)
                bin_centers.append(0.5 * (lo + hi))

                hist_panels.append(
                    {
                        "hist": hist,
                        "lo": lo,
                        "hi": hi,
                        "popt": popt,
                        "model": model,
                        "mean": mean,
                        "sigma": sigma,
                        "mean_err": mean_err,
                        "sigma_err": sigma_err,
                    }
                )

    return (
        np.array(bin_centers),
        np.array(means),
        np.array(mean_errors),
        np.array(sigmas),
        np.array(sigma_errs),
        hist_panels,
    )


class MeanModel:
    @staticmethod
    def __call__(E, mu_slope, mu_offset):
        return mu_slope * E + mu_offset

    @staticmethod
    def grad(E, mu_slope, mu_offset):
        return np.array([E, np.ones_like(E)])

    @staticmethod
    def name(energy_par):
        return f"mu_slope*{energy_par} + mu_offset"

    @staticmethod
    def parnames():
        return ["mu_slope", "mu_offset"]


def fit_mean_model(bin_centers, means, mean_errs):
    # Linear model: mean(E) = mE + b

    model = MeanModel()

    # Clean input data
    # again added the mean errers being less than 1 bit....
    mask = (
        ~np.isnan(bin_centers)
        & ~np.isnan(means)
        & ~np.isnan(mean_errs)
        & (mean_errs > 1e-8)
    )  # & (mean_errs <=1)
    x = np.asarray(bin_centers[mask], dtype=float)
    y = np.asarray(means[mask], dtype=float)
    s = np.asarray(mean_errs[mask], dtype=float)

    # Avoid zero / negative uncertainties
    # s = np.where(s <= 0, 1e-8, s)

    # to avoid zero and negative uncertainties, but without making them to small
    valid_s = s[np.isfinite(s) & (s > 0)]
    s_median = np.median(s)
    if len(valid_s) == 0:
        # No usable uncertainties
        s = np.ones_like(y)
        absolute_sigma = False
    else:
        s_median = np.median(valid_s)
        s = np.where(np.isfinite(s) & (s > 0), s, s_median)
        absolute_sigma = True

    # Initial guess: slope and intercept
    p0 = [0.0, np.nanmean(y)]

    try:
        nll = cost.LeastSquares(x, y, s, model, grad=model.grad)
        fitter = Minuit(nll, *p0)
        result = fitter.migrad()
        popt = np.array(result.values)
        perr = np.array(result.errors)
        return popt, perr, model

    except Exception:
        log.error("Mean linear fit failed", exc_info=True)
        return [np.nan, np.nan], [np.nan, np.nan], model


class SigmaModel:
    @staticmethod
    def __call__(E, A, B):
        return np.sqrt(A**2 / (E**2) + B**2)

    @staticmethod
    def grad(E, A, B):
        sig = np.sqrt(A**2 / (E**2) + B**2)
        return np.array([A / E**2 / sig, B / sig])

    @staticmethod
    def name(energy_par):
        return f"np.sqrt((a_sigma**2/{energy_par}**2)+b_sigma**2)"

    @staticmethod
    def parnames():
        return ["a_sigma", "b_sigma"]


def fit_sigma_model(bin_centers, sigmas, sigma_errs):
    # 1. Fixed model: Ensure all inputs are float64 and handle potential negative inside sqrt
    model = SigmaModel()

    mask = (
        ~np.isnan(bin_centers)
        & ~np.isnan(sigmas)
        & ~np.isnan(sigma_errs)
        & (sigma_errs > 1e-8)
    )
    x = bin_centers[mask]
    y = sigmas[mask]
    s = sigma_errs[mask]

    try:
        initial_guess = [np.nanmean(x * y), np.nanmean(y)]

        nll = cost.LeastSquares(x, y, s, model, grad=model.grad)
        fitter = Minuit(nll, *initial_guess)
        fitter.limits[0] = (0, np.max(x * y))
        fitter.limits[1] = (0, np.max(y))
        result = fitter.migrad()
        popt = np.array(result.values)
        perr = np.array(result.errors)
        return np.abs(popt), perr, model
    except Exception:
        log.error("Sigma model fit failed", exc_info=True)
        return [np.nan, np.nan], [np.nan, np.nan], model


##################################


def get_fit_range(lq: np.array) -> tuple(float, float):
    """
    Determine a ±2.5σ fit range around the mode of an LQ distribution.

    Uses a 100-bin histogram over the 1st–95th percentile range to locate
    the peak centroid and estimate the width, then returns a ±2.5σ window
    suitable for a subsequent Gaussian fit.

    Parameters
    ----------
    lq
        1-D array of LQ parameter values.

    Returns
    -------
    fit_range
        ``(left_edge, right_edge)`` tuple defining the ±2.5σ window
        around the estimated peak centroid.
    """  # noqa: RUF002

    # Get an initial guess of mu and sigma, use these values to determine our final fit range
    left_guess = np.nanpercentile(lq, 1)
    right_guess = np.nanpercentile(lq, 95)
    test_range = (left_guess, right_guess)

    hist, bins, _ = pgh.get_hist(lq, bins=50, range=test_range)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)

    left_edge = mu - 2.5 * sigma
    right_edge = mu + 2.5 * sigma
    return (left_edge, right_edge)


def get_lq_hist(
    df: pd.DataFrame(),
    lq_param: str,
    cal_energy_param: str,
    peak: float,
    sidebands: bool = True,
):
    """
    Histogram the LQ distribution for events near a calibration peak.

    Optionally performs sideband subtraction to isolate peak-associated
    events from the Compton continuum.  The peak window is
    ``(peak - 8, peak + 5)`` keV and the sideband window is
    ``(peak + 7, peak + 20)`` keV.

    Parameters
    ----------
    df
        DataFrame containing both the LQ parameter and calibrated energy.
    lq_param
        Name of the LQ parameter column.
    cal_energy_param
        Name of the calibrated energy column.
    peak
        Peak energy in keV around which to select events.
    sidebands
        If ``True`` (default), subtract a sideband histogram to remove
        the Compton background contribution.

    Returns
    -------
    hist
        Counts array (sideband-subtracted when *sidebands* is ``True``).
    bins
        Bin-edge array.
    var
        Per-bin variance (combined Poisson uncertainty).
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
    _cdf=gaussian,
    sidebands: bool = True,
):
    """Function for fitting a distribution of LQ values within a specified
        energy peak. Fits a gaussian to the distribution

    Parameters
    ----------
    df
        DataFrame containing the LQ parameter and calibrated energy.
    lq_param
        Name of the LQ parameter column.
    cal_energy_param
        Name of the calibrated energy column.
    peak
        Energy value in keV of the peak whose LQ distribution will be fit.
    cdf
        CDF callable used for the binned fit.
    sidebands
        Whether to apply sideband subtraction when building the histogram.

    Returns
    -------
    values
        Best-fit parameter values ``(mu, sigma)`` from the Gaussian fit.
    errors
        Parameter uncertainties corresponding to *values*.
    hist
        Histogram counts used for the binned fit.
    bins
        Bin-edge array used for the binned fit.
    """

    hist, bins, _var = get_lq_hist(df, lq_param, cal_energy_param, peak, sidebands)

    # Temporary fix for negative bin counts
    # TODO: Adjust fitting to handle negative bin counts
    hist[hist < 0] = 0

    bin_centers = (bins[:-1] + bins[1:]) / 2

    mu = bin_centers[np.argmax(hist)]
    _, sigma, _ = pgh.get_gaussian_guess(hist, bins)

    c1 = cost.BinnedNLL(hist, bins, gaussian.get_cdf)
    m1 = Minuit(c1, mu=mu, sigma=sigma)
    m1.simplex().migrad()
    m1.hesse()

    return m1.values, m1.errors, hist, bins  # noqa: PD011


def calculate_time_means(
    df: pd.DataFrame,
    lq_param: str,
    cal_energy_param: str,
    peak: float,
    sidebands: bool = True,  # noqa: ARG001
):
    """
    Compute the arithmetic mean and standard deviation of LQ at a peak.

    Selects events in a ±8/+5 keV window around *peak*, restricts to the
    ±2.5σ fit range returned by :func:`get_fit_range`, and returns the
    sample mean and standard deviation together with their statistical
    uncertainties.

    Parameters
    ----------
    df
        DataFrame containing the LQ and calibrated energy columns.
    lq_param
        Name of the LQ parameter column.
    cal_energy_param
        Name of the calibrated energy column.
    peak
        Peak energy in keV.
    sidebands
        Unused; kept for signature consistency with :func:`binned_lq_fit`.

    Returns
    -------
    pars
        Dictionary ``{"mu": mean, "sigma": std}`` of the LQ distribution.
    errors
        Dictionary ``{"mu": mean_err, "sigma": sigma_err}`` of the
        statistical uncertainties.
    """  # noqa: RUF002

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
    """
    Compute a running weighted-average LQ mean across run timestamps.

    Accumulates valid (non-NaN) mean/resolution pairs and stores the
    weighted-average mean (weighted by inverse resolution) for every
    accumulated timestamp.  Timestamps with NaN values in either *means*
    or *reses* are assigned NaN in the output.

    Parameters
    ----------
    tstamps
        Ordered sequence of run-timestamp strings.
    means
        Array of per-timestamp LQ mean values; may contain NaN.
    reses
        Array of per-timestamp LQ resolutions (sigma/mu) used as inverse
        weights; may contain NaN.

    Returns
    -------
    out_dict
        Dictionary mapping each timestamp to the cumulative weighted-
        average LQ mean up to that point, or ``np.nan`` for invalid
        entries.
    """
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
        use_log_pdf: bool = False,
    ):
        """
        Parameters
        ----------
        cal_dicts
            Hit-level calibration dictionary of expressions/parameters to
            apply to the data.  May be keyed by run timestamp.
        cal_energy_param
            Name of the calibrated energy parameter.
        dt_param
            Name of the drift-time parameter.
        eres_func
            Callable that returns the energy resolution (FWHM) in keV for
            a given energy.
        cdf
            CDF callable used for the binned Gaussian fits.
        selection_string
            Boolean expression selecting valid, non-pulser events.
        debug_mode
            If ``True``, exceptions are re-raised instead of being caught
            and logged.
        use_log_pdf
            Build the survival-fraction unbinned NLL fits from the models'
            log-densities (``iminuit`` ``log=True`` mode) — faster on large
            samples, results differ at machine-precision level.  The LQ
            calibration's own fits are binned and unaffected.
        """

        self.cal_dicts = cal_dicts
        self.cal_energy_param = cal_energy_param
        self.dt_param = dt_param
        self.eres_func = eres_func
        self.cdf = cdf
        self.selection_string = selection_string
        self.debug_mode = debug_mode
        self.use_log_pdf = use_log_pdf

    def update_cal_dicts(self, update_dict):
        """
        Merge new entries into the calibration dictionary.

        If ``cal_dicts`` is keyed by run timestamps, each timestamp's
        sub-dict is updated individually, using *update_dict* directly as a
        fallback when a timestamp is absent.  Otherwise *update_dict* is
        merged directly.

        Parameters
        ----------
        update_dict
            Dictionary of new calibration entries to merge.
        """
        if len(self.cal_dicts) > 0 and re.match(
            r"(\d{8})T(\d{6})Z", next(iter(self.cal_dicts))
        ):
            for tstamp in self.cal_dicts:
                if tstamp in update_dict:
                    self.cal_dicts[tstamp].update(update_dict[tstamp])
                else:
                    self.cal_dicts[tstamp].update(update_dict)
        else:
            self.cal_dicts.update(update_dict)

    ######################
    def get_lq_over_e(
        self,
        df: pd.DataFrame(),
        lq_param,
        cal_energy_param: str,
        out_param: str = "LQ_over_E",
    ):
        try:
            df[out_param] = df[lq_param].to_numpy() / df[cal_energy_param].to_numpy()
            self.update_cal_dicts(
                {out_param: {"expression": f"{lq_param}/{cal_energy_param}"}}
            )

        except Exception:
            if self.debug_mode:
                raise
            log.error("LQ over correction failed", exc_info=True)

    def energy_width_correction(
        self,
        df: pd.DataFrame(),
        lq_param,
        cal_energy_param: str,
        out_param: str = "LQ_Corrected",
    ):
        """
        performs an energy width correction to the LQ/E distribution. Fits across the energy range 250keV to 2650keV in 25keV
        """
        log.info("Starting LQ energy width correction")

        try:
            bin_centers, means, mean_errs, sigmas, sigma_errs, hist_panels = (
                extract_gaussian_trends(
                    df[lq_param].to_numpy(),
                    df[cal_energy_param].to_numpy(),
                    np.linspace(250, 2650, 25),
                )
            )
            # fit a linear model to mu vs E. outputs are used only in plots?
            popt_mean, perr_mean, mean_model = fit_mean_model(
                bin_centers, means, mean_errs
            )

            # Fit sqrt(A/E^2+B) analytical model to sigma vs E
            popt_sigma, perr_sigma, sigma_model = fit_sigma_model(
                bin_centers, sigmas, sigma_errs
            )

            # for plotting later...
            self.gauss_trends_hist = hist_panels
            self.energy_bin_centers = bin_centers

            # update the dataframe...
            mean = mean_model(df[cal_energy_param], *popt_mean)
            width_sigma = sigma_model(df[cal_energy_param], *popt_sigma)
            df[out_param] = (df[lq_param] - mean) / width_sigma

        except Exception:
            if self.debug_mode:
                raise
            log.error("LQ energy width correction failed", exc_info=True)
            means = []
            mean_errs = []
            popt_mean = [np.nan, np.nan]
            perr_mean = [np.nan, np.nan]
            mean_model = MeanModel
            sigmas = []
            sigma_errs = []
            popt_sigma = [np.nan, np.nan]
            perr_sigma = [np.nan, np.nan]
            sigma_model = SigmaModel
            df[out_param] = np.nan

        self.mean_cor_res = {
            "means": means,
            "mean_errs": mean_errs,
            "model": mean_model,
            "popt": popt_mean,
            "perr": perr_mean,
        }
        self.sig_cor_res = {
            "sigmas": sigmas,
            "sigma_errs": sigma_errs,
            "model": sigma_model,
            "popt": popt_sigma,
            "perr": perr_sigma,
        }

        mean_model_str = mean_model.name(cal_energy_param)
        sig_model_str = sigma_model.name(cal_energy_param)
        # save the fit parameters
        self.update_cal_dicts(
            {
                out_param: {
                    "expression": f"({lq_param} - ({mean_model_str}))/({sig_model_str})",
                    "parameters": {
                        mean_model.parnames()[0]: popt_mean[0],
                        mean_model.parnames()[1]: popt_mean[1],
                        sigma_model.parnames()[0]: popt_sigma[0],
                        sigma_model.parnames()[1]: popt_sigma[1],
                    },
                }
            }
        )

    ######################

    def lq_timecorr(
        self,
        df,
        lq_param,
        output_name="LQ_Timecorr",
    ):
        """
        Normalise LQ by the time-varying DEP mean.

        Fits the LQ distribution at the DEP (1592.5 keV) for each
        ``run_timestamp`` group (or globally when no timestamp column is
        present) and divides *lq_param* by the resulting weighted-average
        mean.  The normalised column is stored as *output_name* in *df* and
        the corresponding calibration expression is written into
        :attr:`cal_dicts`.

        Parameters
        ----------
        df
            DataFrame modified in-place; must contain *lq_param* and the
            calibrated energy column.  If a ``run_timestamp`` column is
            present, the correction is applied per timestamp.
        lq_param
            Name of the raw LQ parameter column.
        output_name
            Name of the output time-corrected LQ column.
        """

        log.info("Starting LQ time correction")
        self.timecorr_df = pd.DataFrame()

        try:
            df_sel = df.query(self.selection_string) if self.selection_string else df
            if "run_timestamp" in df:
                for tstamp, time_df in df_sel.groupby("run_timestamp", sort=True):
                    try:
                        pars, errs = calculate_time_means(
                            time_df,
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
                    except Exception as e:
                        if self.debug_mode:
                            raise
                        log.debug(
                            "LQ time correction: fit failed for run_timestamp %s, filling nan: %s",
                            tstamp,
                            e,
                            exc_info=True,
                        )
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

                self.timecorr_df = self.timecorr_df.set_index("run_timestamp")
                time_dict = fit_time_means(
                    np.array(self.timecorr_df.index),
                    np.array(self.timecorr_df["mean"]),
                    np.array(self.timecorr_df["res"]),
                )

                try:
                    df[output_name] = df[lq_param] / np.array(
                        [time_dict[tstamp] for tstamp in df["run_timestamp"]]
                    )
                except KeyError as e:
                    msg = f"no time-correction entry for run_timestamp {e.args[0]!r}"
                    raise KeyError(msg) from e
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
                        df_sel,
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
                except Exception as e:
                    if self.debug_mode:
                        raise
                    log.debug(
                        "LQ time correction: single-run fit failed, filling nan: %s",
                        e,
                        exc_info=True,
                    )
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
                try:
                    df[output_name] = df[lq_param] / pars["mu"]
                except Exception as e:
                    msg = "applying single-run LQ time correction failed"
                    raise RuntimeError(msg) from e
                self.update_cal_dicts(
                    {
                        output_name: {
                            "expression": f"{lq_param}/a",
                            "parameters": {"a": pars["mu"]},
                        }
                    }
                )
                log.info("LQ time correction finished")
        except Exception:
            if self.debug_mode:
                raise
            log.error("LQ time correction failed", exc_info=True)
            self.update_cal_dicts(
                {
                    output_name: {
                        "expression": f"{lq_param}/a",
                        "parameters": {"a": np.nan},
                    }
                }
            )

    def drift_time_correction(
        self,
        df: pd.DataFrame,
        lq_param,
        cal_energy_param: str,  # noqa: ARG002
        out_param: str = "LQ_DT_Corrected",
        mode: str = "linear",
    ):
        """
        Remove the linear drift-time dependence from the LQ distribution.

        Fits a degree-1 polynomial to LQ vs. drift time for DEP events in
        a 6 keV window around 1592.5 keV.  Subtracts the fitted slope and
        intercept to produce a corrected column (*out_param*) centred at
        zero, and writes the calibration expression into :attr:`cal_dicts`.

        Parameters
        ----------
        df
            DataFrame modified in-place; must contain *lq_param*, the
            calibrated energy column, and the drift-time column.
        lq_param
            Name of the LQ parameter column to correct.
        cal_energy_param
            Name of the calibrated energy column.
        out_param
            Name of the output drift-time-corrected LQ column.
        mode
            Method used to correct drift time. Options are "linear" and "LR-MCD".
        """

        log.info("Starting LQ drift time correction")
        log.info(f"Drift time correction mode: {mode}")

        # perform a check on the range of values seen in the DEP and use that to maybe edit the mode.
        try:
            dep_events = df.query(
                f"{self.cal_energy_param} > 1589.5 & {self.cal_energy_param} < 1595.5 & {self.cal_energy_param}=={self.cal_energy_param}&{lq_param}=={lq_param}"
            )

            dt_range = [
                np.nanpercentile(dep_events[self.dt_param], 10),
                np.nanpercentile(dep_events[self.dt_param], 95),
            ]

            if (dt_range[1] - dt_range[0]) > 200:
                mode = "LR-MCD"
            elif (dt_range[1] - dt_range[0]) <= 200:
                mode = "linear"

            log.info(f"Drift time correction mode after range checks: {mode}")

        except Exception as e:
            msg = "DEP event selection for mode determination failed"
            raise RuntimeError(msg) from e

        self.dt_correction_mode = mode

        if mode == "linear":
            try:
                pars = binned_lq_fit(df, lq_param, self.cal_energy_param, peak=1592.5)[
                    0
                ]
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

            except Exception:
                if self.debug_mode:
                    raise
                log.error(
                    "LQ drift time correction (linear mode) failed", exc_info=True
                )
                self.dt_fit_pars = (np.nan, np.nan)

        elif mode == "LR-MCD":
            # want to try with keeping the same dataselection as for the "linear" method
            try:
                pars = binned_lq_fit(df, lq_param, self.cal_energy_param, peak=1592.5)[
                    0
                ]
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

                ###########here is the difference between linear and LR-MCD #############
                data_dep = final_df[[self.dt_param, lq_param]]
                mcd = MinCovDet().fit(data_dep.to_numpy())
                inliers = data_dep[mcd.support_]

                # ------------------------------------------------------------------
                # LR-MCD
                # ------------------------------------------------------------------
                inlier_dt = inliers.to_numpy()[:, 0]
                inlier_lq = inliers.to_numpy()[:, 1]

                lr_mcd = LinearRegression().fit(inlier_dt.reshape(-1, 1), inlier_lq)
                self.dt_fit_pars = (lr_mcd.coef_[0], lr_mcd.intercept_)

            except Exception:
                if self.debug_mode:
                    raise
                log.error("LQ drift time correction (MCD mode) failed", exc_info=True)
                self.dt_fit_pars = (np.nan, np.nan)

        # update dataframe
        try:
            df[out_param] = (
                df[lq_param]
                - df[self.dt_param] * self.dt_fit_pars[0]
                - self.dt_fit_pars[1]
            )
        except Exception as e:
            msg = "applying LQ drift-time correction failed"
            raise RuntimeError(msg) from e

        # add parameters to
        self.update_cal_dicts(
            {
                out_param: {
                    "expression": f"{lq_param} - {self.dt_param}*a - b",
                    "parameters": {"a": self.dt_fit_pars[0], "b": self.dt_fit_pars[1]},
                }
            }
        )

    ##################################

    def get_cut_lq_dep(
        self,
        df: pd.DataFrame,
        lq_param: str,
        cal_energy_param: str,
        classifier_param: str = "LQ_Classifier",
        cut_param: str = "LQ_Cut",
    ):
        """
        Determine the LQ cut value from the DEP distribution.

        Fits a Gaussian to the sideband-subtracted LQ distribution at the
        DEP (1592.5 keV).  The LQ values are normalised by the fitted σ to
        produce *classifier_param*, and a fixed cut at 3 σ is applied
        (*cut_param*).  Both columns are added to *df* and the corresponding
        calibration expressions are written into :attr:`cal_dicts`.

        Parameters
        ----------
        df
            DataFrame modified in-place; must contain *lq_param* and
            *cal_energy_param*.
        lq_param
            Name of the (drift-time-corrected) LQ parameter column.
        cal_energy_param
            Name of the calibrated energy column.
        classifier_param
            Name of the output normalised LQ classifier column.
        cut_param
            Name of the output boolean cut column.
        """  # noqa: RUF002

        log.info("Starting LQ Cut calculation")
        try:
            try:
                pars, errs, hist, bins = binned_lq_fit(
                    df, lq_param, cal_energy_param, peak=1592.5
                )
            except Exception as e:
                msg = "binned LQ fit at DEP failed"
                raise RuntimeError(msg) from e

            self.cut_fit_pars = pars
            self.cut_fit_errs = errs
            self.fit_hist = (hist, bins)
            self.cut_val = 3

            try:
                df[classifier_param] = np.divide(df[lq_param].to_numpy(), pars[1])
                df[cut_param] = df[classifier_param] < self.cut_val
            except Exception as e:
                msg = "applying LQ classifier/cut to data failed"
                raise RuntimeError(msg) from e

        except Exception:
            if self.debug_mode:
                raise
            log.error("LQ cut determination failed", exc_info=True)
            self.cut_val = np.nan
            c = cost.UnbinnedNLL(np.array([0]), gaussian.pdf)
            m = Minuit(c, np.full(2, np.nan))
            self.cut_fit_pars = pars = m.values  # noqa: PD011

        self.update_cal_dicts(
            {
                classifier_param: {
                    "expression": f"({lq_param} / a)",
                    "parameters": {"a": pars[1]},
                },
                cut_param: {
                    "expression": f"({classifier_param} < a)",
                    "parameters": {"a": self.cut_val},
                },
            }
        )

    def calibrate(self, df, initial_lq_param, suffix: str | None = None):
        """
        Run the full LQ calibration pipeline.

        Executes the three calibration steps in sequence:

        1. :meth:`lq_timecorr` — normalise by the time-varying DEP mean.
        2. :meth:`drift_time_correction` — remove the linear drift-time
           dependence.
        3. :meth:`get_cut_lq_dep` — fit the normalised DEP distribution
           and set the cut value.

        Survival fractions at the DEP (1592.5 keV), Qββ (2039 keV),
        2103.53 keV, and 2614.5 keV are then computed and stored in
        :attr:`low_side_sf` and :attr:`low_side_peak_dfs`.

        Parameters
        ----------
        df
            DataFrame modified in-place; must contain *initial_lq_param*
            and all columns referenced by :attr:`cal_energy_param`,
            :attr:`dt_param`, and :attr:`selection_string`.
        initial_lq_param
            Name of the raw LQ parameter column in *df*.
        suffix
            Optional suffix appended to all output parameter names, separated
            by an underscore (e.g. ``suffix="foo"`` produces
            ``"LQ_Timecorr_foo"``, ``"LQ_Classifier_foo"``, etc.).  When
            ``None`` (the default) the original names are used unchanged.
        """

        _n = (lambda base: f"{base}_{suffix}") if suffix else (lambda base: base)

        lq_over_e_name = _n("LQ_over_E")
        timecorr_name = _n("LQ_Timecorr")
        # corrected_name = _n("LQ_Corrected")
        dt_corrected_name = _n("LQ_DT_Corrected")
        # energy_corrected_name = _n("LQ_Energy_Corrected")
        corrected_name = _n("LQ_Corrected")
        classifier_name = _n("LQ_Classifier")
        cut_name = _n("LQ_Cut")

        self.update_cal_dicts(
            {
                lq_over_e_name: {"expression": "np.nan"},
                timecorr_name: {"expression": "np.nan"},
                dt_corrected_name: {"expression": "np.nan"},
                corrected_name: {"expression": "np.nan"},
                classifier_name: {"expression": "np.nan"},
                cut_name: {"expression": "np.nan"},
            }
        )

        self.get_lq_over_e(
            df,
            initial_lq_param,
            cal_energy_param=self.cal_energy_param,
            out_param=lq_over_e_name,
        )

        self.lq_timecorr(
            df,
            lq_over_e_name,
            output_name=timecorr_name,
        )
        log.info("Finished LQ Time Correction")

        self.drift_time_correction(
            df,
            lq_param=timecorr_name,
            cal_energy_param=self.cal_energy_param,
            out_param=dt_corrected_name,
            mode="LR-MCD",
        )
        log.info("Finished LQ Drift Time Correction")

        self.energy_width_correction(
            df,
            lq_param=dt_corrected_name,
            cal_energy_param=self.cal_energy_param,
            out_param=corrected_name,
        )
        log.info("Finished LQ E width Correction")

        self.get_cut_lq_dep(
            df,
            lq_param=corrected_name,
            cal_energy_param=self.cal_energy_param,
            classifier_param=classifier_name,
            cut_param=cut_name,
        )
        log.info("Finished Calculating the LQ Cut Value")

        final_lq_param = classifier_name
        peaks_of_interest = [1592.5, 2039, 2103.53, 2614.50]
        self.low_side_sf = pd.DataFrame()
        fit_widths = [(40, 25), (0, 0), (25, 40), (50, 50)]
        self.low_side_peak_dfs = {}

        log.info("Calculating peak survival fractions")
        for i, peak in enumerate(peaks_of_interest):
            try:
                try:
                    select_df = (
                        df.query(f"{self.selection_string}")
                        if self.selection_string
                        else df
                    )
                except Exception as e:
                    msg = "event selection failed"
                    raise RuntimeError(msg) from e
                fwhm = self.eres_func(peak)
                if peak == 2039:
                    emin = 2 * fwhm
                    emax = 2 * fwhm
                    peak_df = select_df.query(
                        f"({self.cal_energy_param}>{peak - emin})&({self.cal_energy_param}<{peak + emax})"
                    )

                    try:
                        cut_df, sf, sf_err = compton_sf_sweep(
                            peak_df[self.cal_energy_param].to_numpy(),
                            peak_df[final_lq_param].to_numpy(),
                            self.cut_val,
                            cut_range=(0, 5),
                            n_samples=30,
                            mode="less",
                        )
                    except Exception as e:
                        msg = "survival-fraction fit failed"
                        raise RuntimeError(msg) from e
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
                    try:
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
                            use_log_pdf=self.use_log_pdf,
                        )
                    except Exception as e:
                        msg = "survival-fraction fit failed"
                        raise RuntimeError(msg) from e
                    self.low_side_sf = pd.concat(
                        [
                            self.low_side_sf,
                            pd.DataFrame([{"peak": peak, "sf": sf, "sf_err": sf_err}]),
                        ]
                    )
                    self.low_side_peak_dfs[peak] = cut_df
                log.info("%skeV: %2.1f +/- %2.1f %%", peak, sf, sf_err)
            except Exception:
                if self.debug_mode:
                    raise
                self.low_side_sf = pd.concat(
                    [
                        self.low_side_sf,
                        pd.DataFrame([{"peak": peak, "sf": np.nan, "sf_err": np.nan}]),
                    ]
                )
                log.error(
                    f"LQ survival fraction determination failed for {peak} peak",
                    exc_info=True,
                )
        self.low_side_sf = self.low_side_sf.set_index("peak")


def plot_lq_mean_time(
    lq_class,
    data=None,  # noqa: ARG001
    lq_param="LQ_Timecorr",
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    """Plots the mean LQ value calculated for each given timestamp"""

    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "lower right",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

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
                cal_dict[lq_param]["parameters"]["a"]
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
                y1=np.array(grouped_means)
                - 0.2 * np.array(lq_class.timecorr_df["res"]),
                y2=np.array(grouped_means)
                + 0.2 * np.array(lq_class.timecorr_df["res"]),
                color="green",
                alpha=0.2,
            )
            ax.fill_between(
                [
                    datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ")
                    for tstamp in lq_class.cal_dicts
                ],
                y1=np.array(grouped_means)
                - 0.4 * np.array(lq_class.timecorr_df["res"]),
                y2=np.array(grouped_means)
                + 0.4 * np.array(lq_class.timecorr_df["res"]),
                color="yellow",
                alpha=0.2,
            )

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    ax.set_xlabel("time")
    ax.set_ylabel("LQ mean")
    myfmt = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(myfmt)

    return fig


def plot_drift_time_correction(
    lq_class,
    data: pd.DataFrame,
    lq_param="LQ_Timecorr",
    lq_corr_param="LQ_DT_Corrected",
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    """Plots a 2D histogram of LQ versus effective drift time in a 6 keV
    window around the DEP. Additionally plots the fit results for the
    drift time correction."""

    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "lower right",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        else:
            ax1, ax2 = fig.subplots(2, 1, sharex=True)

        try:
            dep_range = (1590, 1595)

            initial_df = data.query(
                f"{lq_class.cal_energy_param} > {dep_range[0]} & {lq_class.cal_energy_param} < {dep_range[1]}"
            )
            dt_range = (0, lq_class.dt_range[1] * 1.5)
            # take MCD lq_range and double width while keeping same center
            lq_range = (
                lq_class.lq_range[0] * 1.5 - lq_class.lq_range[1] * 0.5,
                lq_class.lq_range[1] * 1.5 - lq_class.lq_range[0] * 0.5,
            )

            ax1.hist2d(
                initial_df[lq_class.dt_param],
                initial_df[lq_param],
                bins=100,
                range=(dt_range, lq_range),
                norm=mcolors.LogNorm(),
            )

            x = np.linspace(*dt_range, 100)
            model = lq_class.dt_fit_pars[0] * x + lq_class.dt_fit_pars[1]

            ax1.plot(x, model, color="r")

            ax1.axvline(lq_class.dt_range[0], color="k")
            ax1.axvline(lq_class.dt_range[1], color="k")
            ax1.axhline(lq_class.lq_range[0], color="k")
            ax1.axhline(lq_class.lq_range[1], color="k")

            ax1.set_ylabel("LQ (no DT corr)")

            lq_range2 = (
                lq_class.lq_range[0] - lq_class.lq_range[1],
                lq_class.lq_range[1] - lq_class.lq_range[0],
            )
            ax2.hist2d(
                initial_df[lq_class.dt_param],
                initial_df[lq_corr_param],
                bins=100,
                range=(dt_range, lq_range2),
                norm=mcolors.LogNorm(),
            )
            ax2.axhline(0, color="k", ls=":")
            ax2.set_ylabel("LQ (DT corr)")
            ax2.set_xlabel("Drift Time (ns)")

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig


def plot_lq_cut_fit(
    lq_class,
    data: pd.DataFrame | None = None,  # noqa: ARG001
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    """Plots the final histogram of LQ values for events in the
    DEP, and the fit results used for determining the cut
    value"""

    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "upper left",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            (ax1, ax2) = fig.subplots(2, 1)

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
                gaussian.pdf_norm(xs, x_low, x_high, fit_pars[0], fit_pars[1])
                * dx
                * ls,
                label="Gaussian Fit",
            )

            # ax1.set_xlabel('LQ')
            # ax1.set_title("Fit of LQ events in DEP")
            ax1.legend()

            bin_centers = (bins[:-1] + bins[1:]) / 2
            reses = (
                hist
                - (
                    gaussian.pdf_norm(
                        bin_centers, x_low, x_high, fit_pars[0], fit_pars[1]
                    )
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

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig


def plot_survival_fraction_curves(
    lq_class,
    data: pd.DataFrame | None = None,  # noqa: ARG001
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    """Plots the survival fraction curves as a function of
    LQ cut values for every peak of interest"""
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "lower right",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            ax.vlines(
                lq_class.cut_val,
                0,
                100,
                label=f"cut value: {lq_class.cut_val:1.2f}",
                color="black",
            )

            for peak, survival_df in lq_class.low_side_peak_dfs.items():
                with contextlib.suppress(Exception):
                    ax.errorbar(
                        survival_df.index,
                        survival_df["sf"],
                        yerr=survival_df["sf_err"],
                        label=f"{AoE.get_peak_label(peak)} {peak} keV: {lq_class.low_side_sf.loc[peak]['sf']:2.1f} +/- {lq_class.low_side_sf.loc[peak]['sf_err']:2.1f} %",
                    )

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    vals = ax.get_yticks()
    ax.set_yticks(vals, [f"{x:,.0f} %" for x in vals])
    ax.legend()
    ax.set_xlabel("cut value")
    ax.set_ylabel("survival percentage")
    ax.set_ylim([0, 105])

    return fig


def plot_sf_vs_energy(
    lq_class,
    data: pd.DataFrame,
    xrange=(900, 3000),
    n_bins=701,
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
    cut_param="LQ_Cut",
) -> plt.Figure:
    """Plots the survival fraction as a function of energy"""
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "upper left",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            if lq_class.selection_string:
                data = data.query(lq_class.selection_string)

            bins = np.linspace(xrange[0], xrange[1], n_bins)
            counts_pass, bins_pass, _ = pgh.get_hist(
                data.query(f"{cut_param}")[lq_class.cal_energy_param],
                bins=bins,
            )
            counts, bins, _ = pgh.get_hist(
                data[lq_class.cal_energy_param],
                bins=bins,
            )
            survival_fracs = counts_pass / (counts + 10**-99)

            ax.step(pgh.get_bin_centers(bins_pass), 100 * survival_fracs)

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    ax.set_ylim([0, 100])
    vals = ax.get_yticks()
    ax.set_yticks(vals, [f"{x:,.0f} %" for x in vals])
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel("survival percentage")

    return fig


def plot_spectra(
    lq_class,
    data: pd.DataFrame,
    xrange=(900, 3000),
    n_bins=2101,
    xrange_inset=(1580, 1640),
    n_bins_inset=200,
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
    cut_param="LQ_Cut",
) -> plt.Figure:
    """Plots a 2D histogram of the LQ classifier vs calibrated energy"""
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "upper left",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            if lq_class.selection_string:
                data = data.query(lq_class.selection_string)

            bins = np.linspace(xrange[0], xrange[1], n_bins)
            ax.hist(
                data[lq_class.cal_energy_param],
                bins=bins,
                histtype="step",
                label="before PSD",
            )
            ax.hist(
                data.query(f"{cut_param}")[lq_class.cal_energy_param],
                bins=bins,
                histtype="step",
                label="after LQ cut",
            )
            ax.hist(
                data.query(f"~{cut_param}")[lq_class.cal_energy_param],
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
                select_df[lq_class.cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                select_df.query(f"{cut_param}")[lq_class.cal_energy_param],
                bins=bins,
                histtype="step",
            )
            axins.hist(
                select_df.query(f"(~{cut_param})")[lq_class.cal_energy_param],
                bins=bins,
                histtype="step",
            )

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    ax.set_xlim(xrange)
    ax.set_yscale("log")
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel("counts")
    ax.legend()

    return fig


def plot_classifier(
    lq_class,
    data: pd.DataFrame,
    lq_param="LQ_Classifier",
    xrange=(800, 3000),
    yrange=(-10, 30),
    xn_bins=700,
    yn_bins=500,
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "upper left",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            if lq_class.selection_string:
                data = data.query(lq_class.selection_string)
            h = Hist(
                Regular(
                    xn_bins, xrange[0], xrange[1], name="energy", label="Energy (keV)"
                ),
                Regular(yn_bins, yrange[0], yrange[1], name=lq_param, label=lq_param),
            )
            h.fill(data[lq_class.cal_energy_param], data[lq_param])
            h.plot2d(ax=ax, norm=LogNorm(), cbar=False)
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig


# ------------------------------------------------------------------------------------


def plot_gaussian_trends_hist(
    lq_class,
    data: pd.DataFrame | None = None,
    lq_param="LQ_classifier",
    max_cols=3,
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
) -> plt.Figure:
    rc = {
        "figure.titlesize": "large",
        "figure.constrained_layout.use": True,
        "axes.titlesize": "medium",
        "axes.labelsize": "small",
        "xtick.labelsize": "x-small",
        "ytick.labelsize": "x-small",
        "legend.fontsize": "x-small",
        "legend.loc": "lower center",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        adjust_fig_height = False
        if fig is None:
            fig = plt.figure()
            adjust_fig_height = True

        try:
            hist_panels = lq_class.gauss_trends_hist

            if len(hist_panels) > 0:
                n_panels = len(hist_panels)
                ncols = min(max_cols, n_panels)
                nrows = int(np.ceil(n_panels / ncols))

                if adjust_fig_height:
                    fig.set_figheight(fig.get_figwidth() * 0.8 * nrows / ncols)
                axs = fig.subplots(
                    nrows,
                    ncols,
                    squeeze=False,
                )

                axs_flat = axs.ravel()

                for ax, panel in zip(axs_flat, hist_panels):
                    h = panel["hist"]
                    lo = panel["lo"]
                    hi = panel["hi"]
                    popt = panel["popt"]
                    model = panel["model"]

                    h.plot(ax=ax, histtype="step", label=f"{lo:.0f}–{hi:.0f} keV")

                    if popt is not None and model is not None:
                        x_fit = np.linspace(
                            h.axes.edges[0][0], h.axes.edges[0][-1], 1000
                        )
                        y_fit = model(x_fit, *popt)

                        ax.plot(
                            x_fit,
                            y_fit * np.median(h.axes.widths[0]),
                            color="red",
                            lw=2,
                            label=(
                                rf"$\mu={panel['mean']:.3g}$"
                                + "\n"
                                + rf"$\sigma={panel['sigma']:.3g}$"
                            ),
                        )

                    ax.set_xlabel("")
                    ax.grid(True)

                    ax.legend()

                # only add axis titles at edges
                fig.supxlabel(f"{lq_param}/E")
                fig.supylabel("Counts")

                # Turn off unused subplots
                for ax in axs_flat[len(hist_panels) :]:
                    ax.axis("off")

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig


def plot_mean_vs_energy(
    lq_class,
    data: pd.DataFrame | None = None,  # noqa: ARG001
    lq_param="LQ_classifier",
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
):
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "best",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            bin_centers = lq_class.energy_bin_centers
            means = lq_class.mean_cor_res["means"]
            mean_errs = lq_class.mean_cor_res["mean_errs"]
            model = lq_class.mean_cor_res["model"]
            popt = lq_class.mean_cor_res["popt"]
            perr = lq_class.mean_cor_res["perr"]

            # -----------------------------
            # Clean data
            # -----------------------------
            mask = (
                np.isfinite(bin_centers) & np.isfinite(means) & np.isfinite(mean_errs)
            )
            x = np.asarray(bin_centers[mask], dtype=float)
            y = np.asarray(means[mask], dtype=float)
            yerr = np.asarray(mean_errs[mask], dtype=float)

            if len(x) == 0:
                log.warning("No valid fit results")
                return None

            # Add error bars manually
            ax.errorbar(x, y, yerr=yerr, fmt="o", color="black", capsize=2)

            # -----------------------------
            # Fit overlay
            # -----------------------------
            if not np.isnan(popt).any():
                x_fit = np.linspace(np.min(x), np.max(x), 1000)
                y_fit = model(x_fit, *popt)

                m_fit, b_fit = popt
                m_err, b_err = perr

                # Chi-square
                s_data = np.where(yerr <= 0, 1e-8, yerr)
                y_model = model(x, *popt)
                chi2 = np.sum(((y - y_model) / s_data) ** 2)
                dof = len(x) - len(popt)
                chi2_red = chi2 / dof if dof > 0 else np.nan

                label = (
                    r"Fit: $mE + b$"
                    + "\n"
                    + rf"$m = {m_fit:.3g} \pm {m_err:.3g}$"
                    + "\n"
                    + rf"$b = {b_fit:.3g} \pm {b_err:.3g}$"
                    + "\n"
                    + rf"$\chi^2_{{\mathrm{{red}}}} = {chi2_red:.3f}$"
                )

                ax.plot(x_fit, y_fit, "-", color="red", label=label)

            # -----------------------------
            # Formatting
            # -----------------------------
            ax.set_xlabel("Energy [keV]")
            ax.set_ylabel(f"Mean ({lq_param}/E)")
            ax.grid()
            ax.legend()

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig


def plot_sigma_vs_energy(
    lq_class,
    data: pd.DataFrame | None = None,  # noqa: ARG001
    lq_param="LQ_classifier",
    fig: plt.Figure | None = None,
    rcParams: dict | None = None,
):
    rc = {
        "figure.constrained_layout.use": True,
        "legend.loc": "best",
    }
    if rcParams is not None:
        rc |= rcParams

    with plt.rc_context(rc=rc):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.subplots()

        try:
            bin_centers = lq_class.energy_bin_centers
            sigmas = lq_class.sig_cor_res["sigmas"]
            sigma_errs = lq_class.sig_cor_res["sigma_errs"]
            model = lq_class.sig_cor_res["model"]
            popt = lq_class.sig_cor_res["popt"]
            perr = lq_class.sig_cor_res["perr"]

            # -----------------------------
            # Clean data
            # -----------------------------
            mask = (
                np.isfinite(bin_centers) & np.isfinite(sigmas) & np.isfinite(sigma_errs)
            )
            x = np.asarray(bin_centers[mask], dtype=float)
            y = np.asarray(sigmas[mask], dtype=float)
            yerr = np.asarray(sigma_errs[mask], dtype=float)

            if len(x) == 0:
                log.warning("No valid sigma data")
                return None

            ax.errorbar(x, y, yerr=yerr, fmt="o", color="black", capsize=2)

            # -----------------------------
            # Fit overlay
            # -----------------------------
            if not np.isnan(popt).any():
                x_fit = np.linspace(np.min(x), np.max(x), 1000)
                y_fit = model(x_fit, *popt)

                A_fit, B_fit = popt
                A_err, B_err = perr

                s_data = np.where(yerr <= 0, 1e-8, yerr)

                y_model = model(x, *popt)
                residuals = y - y_model
                chi2 = np.sum((residuals / s_data) ** 2)
                dof = len(x) - len(popt)
                chi2_red = chi2 / dof if dof > 0 else np.nan

                label = (
                    r"Fit: $\sqrt{\left| \frac{A^2}{E^2} + B^2 \right|}$"
                    + "\n"
                    + rf"$A = {A_fit:.3f} \pm {A_err:.3f}$"
                    + "\n"
                    + rf"$B = {B_fit:.3f} \pm {B_err:.3f}$"
                    + "\n"
                    + rf"$\chi^2_{{\mathrm{{red}}}} = {chi2_red:.3f}$"
                )

                ax.plot(x_fit, y_fit, "-", color="red", label=label)

            ax.set_xlabel("Energy [keV]")
            ax.set_ylabel(f"σ ({lq_param}/E)")
            ax.grid()
            ax.legend()

        except Exception as e:
            tb_list = traceback.format_tb(e.__traceback__, limit=2)
            compact_tb = "".join(tb_list)

            msg = f"\n{compact_tb}{type(e).__name__}: {e}"
            log.warning(msg)

    return fig
