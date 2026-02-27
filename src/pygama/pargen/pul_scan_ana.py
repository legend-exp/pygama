# pul_scan_ana.py
"""
Module for analyzing pulser scan data using Bayesian fitting with Stan models.
This module provides functions to detect and fit peaks in energy spectra,
perform linear fits, and analyze non-linearity in pulser scan data.
It utilizes cmdstanpy for interfacing with Stan models.
- find_and_fit_peaks (detects and fits peaks using a Gaussian Stan model)
- linear_fit (performs linear fitting using a linear Stan model)
- PulserScanAnalyzer (class to encapsulate the analysis workflow)
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
from scipy.signal import find_peaks

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ===========================
# Dataclasses
# ===========================
@dataclass
class PulserScanResult:
    peaks_x: np.ndarray
    peaks_y: np.ndarray
    peaks_y_unc: np.ndarray
    peaks_sigmas: np.ndarray
    linear_pars: tuple[float, float]
    linear_errs: tuple[float, float]
    linear_cov_ab: float


@dataclass
class LinearStanData:
    """
    Structure of the data expected by the linear Stan model.

    Attributes:
        N: Number of observations
        x: Array of x values
        x_std: Array of uncertainties on x
        y: Array of y values
        y_std: Array of uncertainties on y
    """

    N: int
    x: list[float]
    x_std: list[float]
    y: list[float]
    y_std: list[float]
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float


@dataclass
class GaussStanData:
    """
    Structure of the data expected by the gauss Stan model.

    Attributes:
        N: Number of observations
        energies: Array of energies
        mu_lower: inferior bound for mu
        mu_upper: upper bound for mu
        sigma_lower: lower bound for sigma
        sigma_upper: upper bound for sigma
    """

    N: int
    energies: list[float]
    mu_lower: float
    mu_upper: float
    sigma_lower: float
    sigma_upper: float


# ===========================
# Peak detection & fitting
# ===========================


def find_and_fit_peaks(
    energies: np.ndarray,
    pulser_pos: np.ndarray,
    gauss_model: cmdstanpy.CmdStanModel,
    peak_spacing: float,  # << expected distance between peaks (same `energies` units)
    peak_range_factor: float = 0.33,  # << each peaks is fitted in peak ± peak_spacing * peak_range_factor
    max_peaks: int = 155,
    prominence: float = 10.0,
    bins: int = 10000,
    missing_gap_factor: float = 0.33,  # gap threshold factor: missing peak if more than missing_gap_factor * peak_spacing
) -> dict[str, Any]:
    """
    - Detects peaks using hist_and_peaks.
    - For each detected peak, performs a local fit within a continuous window:
        [center - peak_range_factor*peak_spacing, center + peak_range_factor*peak_spacing].
    - Also returns posterior samples of μ to allow uncertainty propagation.

    Parameters
    ----------
    energies : np.ndarray
        Input energy (or ADC) values.
    pulser_pos : np.ndarray
        Known or reference pulser positions used for calibration or validation.
    gauss_model : cmdstanpy.CmdStanModel
        Compiled Stan Gaussian model used for Bayesian fitting of each peak (es: ./stan_models/gauss/gauss.stan)..
    peak_spacing : float
        Expected distance between peaks (in the same units as `energies`).
    peak_range_factor : float, optional
        Each peak is fitted within ± peak_spacing * peak_range_factor (default is 1/3).
    max_peaks : int, optional
        Maximum number of peaks to search for (default is 155).
    prominence : float, optional
        Minimum prominence required to identify a peak (default is 10.0).
    bins : int, optional
        Number of histogram bins used to build the energy spectrum (default is 10000) (used only to find peaks).
    missing_gap_factor : float, optional
        If the distance between two consecutive peaks exceeds
        `missing_gap_factor * peak_spacing`, a "missing" peak is flagged.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing, for each fitted peak:
        - Pulser positions
        - Fitted peak centers (mu)
        - Uncertainties on fitted peak centers (mu_std)
        - Fitted peak widths (sigma)
    """
    gap_threshold = missing_gap_factor * peak_spacing

    _, b, peaks_idx, _ = hist_and_peaks(energies, prominence=prominence, bins=bins)
    bin_centers = b[:-1] + (b[1] - b[0]) / 2
    log.debug(
        f"Found peaks: {len(bin_centers[peaks_idx])}, with values: {bin_centers[peaks_idx][:20]}..."
    )

    peaks_idx = np.sort(np.asarray(peaks_idx, dtype=int))[:max_peaks]
    pulser_pos = np.sort(np.asarray(pulser_pos, dtype=float)[:max_peaks])
    log.debug(
        f"Using up to max_peaks: {max_peaks}, found peaks count: {len(peaks_idx)}"
    )

    pk_vals = bin_centers[peaks_idx]

    # Filter peaks using greedy algorithm to handle missing peaks
    pk_vals, pulser_pos = select_valid_peaks(
        pk_vals=pk_vals, pul_pos=pulser_pos, x=peak_spacing, tol=gap_threshold
    )
    log.debug(f"After selecting valid peaks, peaks count: {len(pk_vals)}")

    pos, peaks, pks_unc, sigmas = [], [], [], []

    half_window = peak_range_factor * peak_spacing

    for i, pval in enumerate(pk_vals):
        # Select data within the fitting window
        lo = pval - half_window
        hi = pval + half_window
        # log.debug(f"Fitting peak at approx {pval} in range [{lo}, {hi}]")

        sel = (energies >= lo) & (energies <= hi)
        ene_tmp = energies[sel]
        # log.debug(f"Number of values in fitting window: {len(ene_tmp)}")
        # fallback: if too narrow or empty try to widen range (20%)
        if len(ene_tmp) < 1000:
            lo2 = pval - 1.20 * half_window
            hi2 = pval + 1.20 * half_window
            sel = (energies >= lo2) & (energies <= hi2)
            ene_tmp = energies[sel]

        if len(ene_tmp) < 1000:
            continue  # skip peak if still too narrow or empty

        ensure_cmdstan_installed()
        # Fit Gaussian peak using Stan model
        try:
            fit = fit_gaussian_stan(gauss_model, ene_tmp)
            df = fit.draws_pd()
            mu = df["mu"].mean()
            mu_unc = df["mu"].std()
            sigma = df["sigma"].mean()
            pos.append(pulser_pos[i])
            peaks.append(mu)
            pks_unc.append(mu_unc)
            sigmas.append(sigma)
        except Exception:
            # if a fit fails, skip this peak
            continue

        log.debug(f"Completed fitting peaks: {i} / {len(pk_vals)}")

    log.debug(f"Total fitted peaks: {len(peaks)}")
    return {
        "pulser_pos": np.asarray(pos, dtype=float),
        "mus": np.asarray(peaks, dtype=float),
        "mu_stds": np.asarray(pks_unc, dtype=float),
        "sigmas": np.asarray(sigmas, dtype=float),
    }


def select_valid_peaks(
    pk_vals: np.ndarray, pul_pos: np.ndarray, x: float, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm to filter peak values based on expected distance x and tolerance tol.
    It starts from the first peak and iteratively searches for the next peak
    within the range [last_pos + x - tol, last_pos + x + tol].
    If multiple peaks are found in this range, the one closest to last_pos + x is selected.
    The process continues until no more peaks can be found within the specified range.

    Parameters:
    -----------
    pk_vals : np.ndarray
        Array of peak values (must be sorted).
    pul_pos : np.ndarray
        Array of pulser positions corresponding to pk_vals.
    x : float
        Expected distance between peaks.
    tol : float
        Tolerance around expected distance.
    """

    if len(pk_vals) == 0:
        return np.array([]), np.array([])

    filtered_idx = [0]  # first peak is always selected
    last_pos = pk_vals[0]

    # position to start searching for the next peak
    search_start = 1

    while True:
        target = last_pos + x
        # search range
        lo = target - tol
        hi = target + tol

        # find indices of peaks within [lo, hi], pk_vals is sorted
        i0 = np.searchsorted(pk_vals, lo, side="left")
        i1 = np.searchsorted(pk_vals, hi, side="right")

        i0 = max(i0, search_start)
        if i0 >= i1 or i0 >= len(pk_vals):
            break  # no candidate => end

        candidates = np.arange(i0, i1)

        dists = np.abs(pk_vals[candidates] - target)
        best_rel = candidates[np.argmin(dists)]

        filtered_idx.append(best_rel)
        last_pos = pk_vals[best_rel]
        search_start = best_rel + 1

        if search_start >= len(pk_vals):
            break

    filtered_pk = pk_vals[filtered_idx]
    filtered_pul = pul_pos[filtered_idx]
    return filtered_pk, filtered_pul


def fit_gaussian_stan(
    gauss_model: cmdstanpy.CmdStanModel, energies: np.ndarray
) -> cmdstanpy.CmdStanMCMC:
    """
    Fit a Gaussian peak using the provided Stan model.

    Parameters
    ----------
    gauss_model : cmdstanpy.CmdStanModel
        Compiled Stan Gaussian model used for Bayesian fitting of the peak.
    energies : np.ndarray
        energies values within the fitting window.

    Returns
    -------
    fit_gauss : float
        Fit MCMC for the gaussian peak.
    """
    # create data dict for stan model
    x_lim = (energies.min(), energies.max())

    # log.debug(f"Fitting Gaussian with energy range: {x_lim}")
    # log.debug(f"Number of energies for fitting: {len(energies)}")

    data_dict = GaussStanData(
        N=len(energies),
        energies=list(energies),
        mu_lower=x_lim[0],
        mu_upper=x_lim[1],
        sigma_lower=1e-3,
        sigma_upper=(x_lim[1] - x_lim[0]) * 4,
    ).__dict__

    fit_gauss = gauss_model.sample(
        data=data_dict,
        chains=4,
        parallel_chains=4,
        iter_sampling=3000,
        show_console=False,
        show_progress=False,
    )

    return fit_gauss


# ===========================
# Linear fit
# ===========================


def linear_fit(
    linear_model: cmdstanpy.CmdStanModel,
    pulser_pos: np.ndarray,
    pulser_pos_stds: np.ndarray,
    mus: np.ndarray,
    mu_stds: np.ndarray,
    k: float = 10.0,
) -> tuple[list[float], list[float], float]:
    """
    Fit linear function using the provided Stan model.
    Uses np.polyfit to get initial estimates for parameters and uses them to set priors,
    then performs Bayesian fitting with Stan.

    Parameters
    ----------
    linear_model : CmdStanModel
        Stan linear Model (es. ./stan_models/linear/linear.stan).
    pulser_pos : array
        x. (known pulser positions).
    pulser_pos_stds : array
        pulser_pos stds.
    mus : array
        y (peaks centers).
    mu_stds : array
        mu stds.
    k : float
        multiplier factor to construct parameters priors.

    Returns
    -------
    pars : [a, b]
    errs : [a_std, b_std]
    cov_ab : float
        Covariance between a and b computed from draws.
    """

    log.debug("Performing initial linear fit with np.polyfit to estimate priors.")
    log.debug(f"pulser_pos length: {len(pulser_pos)}, mus length: {len(mus)}")
    log.debug(f"pulser_pos: {pulser_pos[:10]}..., mus: {mus[:10]}...")

    coeff, cov = np.polyfit(pulser_pos, mus, 1, cov=True)
    a, b = coeff  # np.polyfit(deg=1) -> [slope, intercept]
    # log.debug(f"Initial fit results: a={a}, b={b}")
    a_std, b_std = np.sqrt(np.diag(cov))
    # log.debug(f"Initial fit uncertainties: a_std={a_std}, b_std={b_std}")
    a_sd = a_std * k
    b_sd = b_std * k
    # I prefer using gaussian priors on a and b than uniform in a range
    start_pars = {"a": a, "b": b, "a_std": a_sd, "b_std": b_sd}

    ensure_cmdstan_installed()

    fit = fit_linear_stan(
        linear_model, pulser_pos, pulser_pos_stds, mus, mu_stds, start_pars
    )

    df = fit.draws_pd()

    a_fit = df["a"].mean()
    b_fit = df["b"].mean()
    a_fit_std = df["a"].std()
    b_fit_std = df["b"].std()

    cov_matrix = np.cov(df[["a", "b"]].values.T)
    cov_ab = float(cov_matrix[0, 1])

    log.debug(f"Final fit results: a={a_fit}, b={b_fit}")
    log.debug(f"Final fit uncertainties: a_std={a_fit_std}, b_std={b_fit_std}")
    log.debug(f"Covariance between a and b: cov_ab={cov_ab}")
    return [a_fit, b_fit], [a_fit_std, b_fit_std], cov_ab


def fit_linear_stan(
    linear_model: cmdstanpy.CmdStanModel,
    x: np.ndarray,
    x_std: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    starting_pars: dict[str, float],
) -> cmdstanpy.CmdStanMCMC:
    """
    Fit linear function using the provided Stan model.
    The data structure of the Stan model is defined in LinearStanData

    Parameters
    ----------
    linear_model : CmdStanModel
        Stan linear Model (es. ./stan_models/linear/linear.stan).
    x : array
        x.
    x_std : array
        x_std.
    y : array
        y.
    y_std : array
        y_std.
    starting_pars : dict
        Dictionary with starting parameters for a and b and their stds:
        {'a': value, 'a_std': value, 'b': value, 'b_std': value}
    """
    # Create data dict for stan model
    data = LinearStanData(
        N=len(x),
        x=x,
        x_std=x_std,
        y=y,
        y_std=y_std,
        a_mean=starting_pars["a"],
        a_std=starting_pars["a_std"],
        b_mean=starting_pars["b"],
        b_std=starting_pars["b_std"],
    ).__dict__

    fit = linear_model.sample(
        data=data,
        chains=4,
        parallel_chains=4,
        iter_sampling=3000,
        show_console=False,
        show_progress=False,
    )

    return fit


# ===========================
# Analyzer class
# ===========================


class PulserScanAnalyzer:
    def __init__(
        self,
        gauss_model: cmdstanpy.CmdStanModel | None = None,
        linear_model: cmdstanpy.CmdStanModel | None = None,
    ):
        """
        If no models are provided, load the default ones from:
          - <__file__path>/stan_models/gauss/gauss.stan
          - <__file__path>/stan_models/linear/linear.stan
        """
        if gauss_model is None or linear_model is None:
            stan_dir = Path(__file__).parent
            gauss_file = stan_dir / "stan-models" / "gauss" / "gauss.stan"
            linear_file = stan_dir / "stan-models" / "linear" / "linear.stan"
            if gauss_model is None:
                gauss_model = cmdstanpy.CmdStanModel(stan_file=gauss_file)
            if linear_model is None:
                linear_model = cmdstanpy.CmdStanModel(stan_file=linear_file)

        self.gauss_model = gauss_model
        self.linear_model = linear_model
        log.debug(
            f"Created PulserScanAnalyzer with: gauss_model={gauss_file}, linear_model={linear_file}"
        )

    def run(
        self, energies: np.ndarray, pulser_positions: np.ndarray, **kwargs
    ) -> PulserScanResult:
        """
        - find and fit peaks using Stan gaussian model
        - linear fit of peak positions vs known pulser positions using Stan linear model
        - compute linearity deviations

        Parameters
        ----------
        energies : np.ndarray
            Input energy (or ADC) values.
        pulser_positions : np.ndarray
            Known or reference pulser positions used for calibration or validation.
        **kwargs : dict
            Additional keyword arguments passed to `find_and_fit_peaks` and `linear_fit`.
        """

        # Obtain default arguments of find_and_fit_peaks
        kwargs_peaks = filter_kwargs_for(find_and_fit_peaks, kwargs)
        log.debug(f"Filtered kwargs for find_and_fit_peaks: {kwargs_peaks.keys()}")
        # Obtain default arguments of linear_fit
        kwargs_linear = filter_kwargs_for(linear_fit, kwargs)
        log.debug(f"Filtered kwargs for find_and_fit_peaks: {kwargs_linear.keys()}")

        peaks = find_and_fit_peaks(
            energies=energies,
            pulser_pos=pulser_positions,
            gauss_model=self.gauss_model,
            **kwargs_peaks,
        )

        pars, errs, cov_ab = linear_fit(
            linear_model=self.linear_model,
            pulser_pos=peaks["pulser_pos"],
            pulser_pos_stds=np.full_like(
                peaks["pulser_pos"], fill_value=0.0001
            ),  # << small stds for pulser positions --> use uncertainties of pulser?
            mus=peaks["mus"],
            mu_stds=peaks["sigmas"],
            **kwargs_linear,
        )

        self.results = PulserScanResult(
            peaks_x=peaks["pulser_pos"],
            peaks_y=peaks["mus"],
            peaks_y_unc=peaks["mu_stds"],
            peaks_sigmas=peaks["sigmas"],
            linear_pars=(pars[0], pars[1]),
            linear_errs=(errs[0], errs[1]),
            linear_cov_ab=cov_ab,
        )

    def get_results(self) -> PulserScanResult:
        """Return the results of the analysis."""
        return self.results


# ===========================
# Utilities
# ===========================


def filter_kwargs_for(func: callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Return a dictionary containing only the keyword arguments that are valid
    parameters of the given function.

    Parameters
    ----------
    func : callable
        The function whose signature will be inspected.
    kwargs : dict[str, Any]
        A dictionary of keyword arguments to be filtered.

    Returns
    -------
    dict[str, Any]
        A dictionary containing only the key-value pairs from `kwargs` whose keys
        match parameter names in `func`’s signature.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def hist_and_peaks(
    data: np.ndarray, prominence: int = 10, bins: int = 10000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Create histogram and find peaks using scipy find_peaks.
    Parameters
    ----------
    data : array
        Input data to histogram and find peaks.
    prominence : float
        Minimum prominence required to identify a peak.
    bins : int
        Number of histogram bins.
    Returns
    -------
    h : array
        Histogram counts.
    b : array
        Histogram bin edges.
    p : array
        Indices of the peaks in the histogram.
    prop : dict
        Properties of the peaks returned by find_peaks.
    """
    h, b = np.histogram(data, bins=bins)
    p, prop = find_peaks(h, prominence=prominence)

    return h, b, p, prop


def ensure_cmdstan_installed():
    """
    Check whether CmdStan is installed.
    If not, install it automatically.

    This function does NOT compile any Stan model;
    it only ensures that the CmdStan backend is available.
    """
    try:
        path = cmdstanpy.cmdstan_path()
        log.info(f"[INFO] CmdStan already installed at: {path}")
        return path
    except Exception:
        log.info(
            "[WARNING] CmdStan not detected. Starting installation "
            "(this may require a C++ compiler and may take a few minutes)..."
        )
        cmdstanpy.install_cmdstan()
        path = cmdstanpy.mdstan_path()
        log.info(f"[INFO] CmdStan successfully installed at: {path}")
