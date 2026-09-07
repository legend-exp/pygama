"""
This module provides routines for measuring cross-talk (XTC) between
germanium channels and for building the resulting cross-talk matrix.

The four main functions, in order of execution, are:
prepare_baseline, xtalk_column, xtalk_histogram_fitter, and build_xtalk_matrix.

:func:`plot_xtalk_matrix` draws what the last of them returns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import lgdo
import lh5
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import pygama.math.histogram as pgh
from pygama.math.functions.gauss import nb_gauss_amp
from pygama.pargen.xtc_utils import EventSelector

log = logging.getLogger(__name__)

DEFAULT_ENERGY_PARAM = "cuspEmax_ctc_cal"
DEFAULT_POSITIVE_PARAM = "trapTmax"
DEFAULT_NEGATIVE_PARAM = "trapTmin"
DEFAULT_TRIGGER_PARAM = "trapTmax"
DEFAULT_TRIGGER_ENERGY_RANGE = (1500, 99999)
DEFAULT_RESPONSE_ENERGY_RANGE = (-99999, 100)
DEFAULT_NBINS = 700
DEFAULT_RANGE_MULTIPLIER = 3
DEFAULT_LOW_STATS_THRESHOLD = 100
DEFAULT_Y_MASK_THRESHOLD = 0.05
DEFAULT_SHARP_FIT_MIN_POINTS = 5

#: Outcome of fitting one histogram, ordered from the most to the least
#: trustworthy.  Written into the lh5 file as ``fit_status_codes`` so a
#: reader never has to hard-code these numbers.
FIT_STATUS = {
    "ok": 0,
    "ok_few_points": 1,
    "low_stats": 2,
    "fit_failed": 3,
    "no_stats": 4,
    "not_filled": 5,
}

#: Statuses whose ``mu`` and ``sigma`` come from a converged fit.
FIT_STATUS_SUCCESS = (FIT_STATUS["ok"], FIT_STATUS["ok_few_points"])

#: Column each polarity is stored under in an xtc lh5 file.  
XTC_LH5_FIELD = {"neg": "xtalk_matrix_negative", "pos": "xtalk_matrix_positive"}

XTC_PLOT_RANGE = {"neg": (-0.003, 0.001), "pos": (-0.0007, 0.003)}


def prepare_baseline(
    hit_files: str | list,
    dsp_files: str | list,
    chn_id: str | int,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Measure the positive and negative baselines of a single channel.

    Selects baseline events in the hit tier -- those whose flag fields match
    ``config["baseline_conditions"]`` -- then averages the corresponding
    positive- and negative-going DSP amplitudes over exactly those events.

    Parameters
    ----------
    hit_files
        Hit-tier file, or list of files, to read the selection flags from.
    dsp_files
        DSP-tier file, or list of files, holding the amplitudes.  Must cover
        the same events, in the same order, as *hit_files*.
    chn_id
        Channel identifier (rawid) of the detector, without the ``ch``
        prefix.  Tables are read from ``ch{chn_id}/hit/`` and
        ``ch{chn_id}/dsp/``.
    config
        Selection configuration.  Recognised keys, all optional:

        ``baseline_conditions``
            Mapping of hit-tier flag field to the value it must equal for an
            event to count as baseline, e.g. ``{"is_empty_candidate": 63}``.
            Defaults to ``{}``, i.e. no flag cut.
        ``energy_param``
            Hit-tier field the selection is applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``positive_param``, ``negative_param``
            DSP-tier fields averaged to give the positive and negative
            baselines.  Default ``"trapTmax"`` and ``"trapTmin"``.
    debug_mode
        If True, re-raise instead of falling back to a null result.

    Returns
    -------
    dict
        Keys ``detector_id``, ``positive_baseline``, ``negative_baseline``
        (``None`` when the measurement failed), ``success``, ``processed_at``
        and ``parameters``.
    """
    config = config or {}
    conditions = config.get("baseline_conditions", {})
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)

    success = True
    positive_baseline = None
    negative_baseline = None

    try:
        try:
            baseline_selection = EventSelector(
                table_path=f"ch{chn_id}/hit/",
                files=hit_files,
                ene_dataset=energy_param,
                conditions=conditions,
            )
        except Exception as e:
            msg = (
                f"baseline selection on {energy_param} failed: "
                f"{type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e

        if len(baseline_selection.selected_idxs) == 0:
            msg = "no events passed the baseline selection"
            raise RuntimeError(msg)

        try:
            positive_selection = EventSelector(
                table_path=f"ch{chn_id}/dsp/",
                files=dsp_files,
                ene_dataset=positive_param,
                idx=baseline_selection.selected_idxs,
            )
            negative_selection = EventSelector(
                table_path=f"ch{chn_id}/dsp/",
                files=dsp_files,
                ene_dataset=negative_param,
                idx=baseline_selection.selected_idxs,
            )
        except Exception as e:
            msg = (
                f"reading {positive_param}/{negative_param} on the selected "
                f"events failed: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e

        if (
            len(positive_selection.selected_energies) == 0
            or len(negative_selection.selected_energies) == 0
        ):
            msg = "no baseline events survived the dsp-tier nan cut"
            raise RuntimeError(msg)

        positive_baseline = float(np.mean(positive_selection.selected_energies))
        negative_baseline = float(np.mean(negative_selection.selected_energies))

    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "baseline preparation failed for channel %s: %s",
            chn_id,
            e
        )
        success = False

    return {
        "detector_id": chn_id,
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "success": success,
        "processed_at": datetime.now().isoformat(),
        "parameters": {
            "baseline_conditions": conditions,
            "energy_param": energy_param,
            "positive_param": positive_param,
            "negative_param": negative_param,
        },
    }


def _resolve_baseline(baseline: dict, chn_id: str | int) -> tuple[float, float] | None:
    """
    Return ``(positive, negative)`` for *chn_id*, or None if unusable.
    """
    entry = baseline.get(chn_id)
    if entry is None:
        entry = baseline.get(str(chn_id))
    if not isinstance(entry, dict):
        return None

    positive = entry.get("positive_baseline")
    negative = entry.get("negative_baseline")
    if positive is None or negative is None:
        return None

    positive = float(positive)
    negative = float(negative)
    if np.isnan(positive) or np.isnan(negative):
        return None

    return positive, negative


def _build_hist(
    vals: np.ndarray, nbins: int, range_multiplier: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Histogram *vals* over ``mean +/- range_multiplier * stdev``.

    Returns ``None`` when the sample is empty or has no usable spread.
    """
    if vals.size == 0:
        return None

    mean = np.mean(vals)
    stdev = np.std(vals)
    if np.isnan(mean) or np.isnan(stdev) or stdev <= 0:
        return None

    return np.histogram(
        vals,
        bins=nbins,
        range=(mean - range_multiplier * stdev, mean + range_multiplier * stdev),
    )


def xtalk_column(
    hit_files: str | list,
    dsp_files: str | list,
    trigger_detector_id: str | int,
    baseline: dict,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Fill the histograms for one column of the cross-talk matrix.

    Selects the events in which *trigger_detector_id* fired with 
    high enough energy (determined by *trigger_energy_range* in *config*). 

    Then, for each detector in the keys of *baseline*, among these events, it 
    further selects the events in which the detector did *not* 
    fire with high energy (otherwise it's multiplicity event).

    Finally, calculates the per-event cross talk value for each of these events
    and fills them into a histogram. 

    Detector pairs skipped are recorded with ``valid = False`` and an empty
    histogram. This happens when the response channel is the trigger itself, or when
    either channel has no usable baseline (missing, ``None`` or NaN).  If the
    *trigger* channel has no usable baseline the whole column is skipped.

    Parameters
    ----------
    hit_files
        Hit-tier file, or list of files, holding the selection flags.
    dsp_files
        DSP-tier file, or list of files, holding the amplitudes.  Must cover
        the same events, in the same order, as *hit_files*.
    trigger_detector_id
        Channel id of the trigger detector, without the ``ch`` prefix.
    baseline
        Per-channel baselines, as produced by :func:`prepare_baseline` and
        collected by channel id. It should have the following structure: 
        {chn_id: {"positive_baseline": float, "negative_baseline": float}, ...}

        for example:
            {
                1104000: {"positive_baseline": 5.0, "negative_baseline": -3.0},
                1104001: {"positive_baseline": None, "negative_baseline": None},
                ...
            }
    config
        Selection and histogram configuration.  Recognised keys, all
        optional:

        ``energy_param``
            Hit-tier field both selections are applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``trigger_param``
            DSP-tier field giving the trigger energy in the denominator of
            the cross-talk ratio.  Default ``"trapTmax"``.
        ``positive_param``, ``negative_param``
            DSP-tier response fields histogrammed against the positive and
            negative baselines.  Default ``"trapTmax"`` and ``"trapTmin"``.
        ``trigger_conditions``, ``response_conditions``
            Mappings of hit-tier flag field to the value it must equal, for
            the trigger and response selections respectively, e.g.
            ``{"is_highly_positive_polarity_candidate": 511}``.  Default
            ``{}``, i.e. no flag cut.
        ``trigger_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting real triggers.
            Default ``(1500, 99999)``.
        ``response_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting channels that did
            *not* see a real hit.  Default ``(-99999, 100)``.
        ``nbins``
            Bins per histogram.  Default 700.
        ``range_multiplier``
            Histogram half-width in standard deviations about the mean.
            Default 3.
    debug_mode
        If True, re-raise instead of falling back to an empty column or an
        empty element.

    Returns
    -------
    dict
        ``trigger_id``, ``response_ids`` ``(N,)``, ``valid`` ``(N,)`` bool,
        ``n_events`` ``(N,)``, ``neg_counts``/``pos_counts`` ``(N, nbins)``,
        ``neg_bins``/``pos_bins`` ``(N, nbins + 1)``, ``parameters`` and
        ``processed_at``.  Bins are NaN wherever the histogram is empty.
    """

    config = config or {}
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    trigger_param = config.get("trigger_param", DEFAULT_TRIGGER_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    trigger_conditions = config.get("trigger_conditions", {})
    response_conditions = config.get("response_conditions", {})
    trigger_energy_range = tuple(
        config.get("trigger_energy_range", DEFAULT_TRIGGER_ENERGY_RANGE)
    )
    response_energy_range = tuple(
        config.get("response_energy_range", DEFAULT_RESPONSE_ENERGY_RANGE)
    )
    nbins = int(config.get("nbins", DEFAULT_NBINS))
    range_multiplier = float(config.get("range_multiplier", DEFAULT_RANGE_MULTIPLIER))

    chn_id_list = list(baseline.keys())
    n_response = len(chn_id_list)
    neg_counts = np.zeros((n_response, nbins), dtype=np.int64)
    pos_counts = np.zeros((n_response, nbins), dtype=np.int64)
    neg_bins = np.full((n_response, nbins + 1), np.nan)
    pos_bins = np.full((n_response, nbins + 1), np.nan)
    valid = np.zeros(n_response, dtype=bool)
    n_events = np.zeros(n_response, dtype=np.int64)

    trigger_selection = None
    trigger_all = None

    # trigger selection. Only need to be done once per column. 
    try:
        if _resolve_baseline(baseline, trigger_detector_id) is None:
            msg = f"trigger channel {trigger_detector_id} has no usable baseline"
            raise RuntimeError(msg)

        try:
            trigger_selection = EventSelector(
                table_path=f"ch{trigger_detector_id}/hit/",
                files=hit_files,
                ene_dataset=energy_param,
                conditions=trigger_conditions,
                energy_range=trigger_energy_range,
            )
        except Exception as e:
            msg = f"trigger event selection failed: {type(e).__name__}: {e}"
            raise RuntimeError(msg) from e

        if len(trigger_selection.selected_idxs) == 0:
            msg = "no events passed the trigger selection"
            raise RuntimeError(msg)

        # read once for the whole column: this is the work the column shares
        try:
            trigger_all = lh5.read(
                f"ch{trigger_detector_id}/dsp/{trigger_param}", dsp_files
            ).nda
        except Exception as e:
            msg = f"reading trigger {trigger_param} failed: {type(e).__name__}: {e}"
            raise RuntimeError(msg) from e

    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "xtalk column for trigger %s failed, writing an empty column: %s",
            trigger_detector_id,
            e,
        )
        trigger_selection = None

    # loop over response detectors starts here 
    # If trigger selection failed or the trigger baseline is None, 
    # skip the whole loop to saving an empty column. 
    if trigger_selection is not None:
        for k, response_id in enumerate(chn_id_list):
            if str(response_id) == str(trigger_detector_id):
                log.debug("self-interaction at channel %s ignored", response_id)
                continue

            baselines = _resolve_baseline(baseline, response_id)
            if baselines is None:
                log.debug(
                    "response channel %s has no usable baseline, skipping", response_id
                )
                continue
            positive_baseline, negative_baseline = baselines

            try:
                response_selection = EventSelector(
                    table_path=f"ch{response_id}/hit/",
                    files=hit_files,
                    ene_dataset=energy_param,
                    conditions=response_conditions,
                    energy_range=response_energy_range,
                    idx=trigger_selection.selected_idxs,
                )

                # selected_idxs index the original array, not the trigger
                # subset, so they address trigger_all directly
                coincident_idxs = response_selection.selected_idxs
                trigger_energies = trigger_all[coincident_idxs]

                response_table = lh5.read(
                    f"ch{response_id}/dsp/",
                    dsp_files,
                    field_mask=[positive_param, negative_param],
                    idx=coincident_idxs,
                )

                neg_vals = (
                    response_table[negative_param].nda - negative_baseline
                ) / trigger_energies
                pos_vals = (
                    response_table[positive_param].nda - positive_baseline
                ) / trigger_energies
            except Exception as e:
                if debug_mode:
                    raise
                log.error(
                    "xtalk element (%s, %s) failed: %s",
                    trigger_detector_id,
                    response_id,
                    e,
                )
                continue

            valid[k] = True
            n_events[k] = len(trigger_energies)

            neg_hist = _build_hist(neg_vals, nbins, range_multiplier)
            if neg_hist is not None:
                neg_counts[k], neg_bins[k] = neg_hist
            else:
                log.debug(
                    "negative histogram for element (%s, %s) is empty",
                    trigger_detector_id,
                    response_id,
                )

            pos_hist = _build_hist(pos_vals, nbins, range_multiplier)
            if pos_hist is not None:
                pos_counts[k], pos_bins[k] = pos_hist
            else:
                log.debug(
                    "positive histogram for element (%s, %s) is empty",
                    trigger_detector_id,
                    response_id,
                )

    parameters = {
        "energy_param": energy_param,
        "trigger_param": trigger_param,
        "positive_param": positive_param,
        "negative_param": negative_param,
        "trigger_conditions": trigger_conditions,
        "response_conditions": response_conditions,
        "trigger_energy_range": list(trigger_energy_range),
        "response_energy_range": list(response_energy_range),
        "nbins": nbins,
        "range_multiplier": range_multiplier,
    }

    log.info(
        "xtalk column of trigger %s filled, %s/%s elements",
        trigger_detector_id,
        int(valid.sum()),
        n_response,
    )

    return {
        "trigger_id": trigger_detector_id,
        "response_ids": np.asarray(chn_id_list),
        "valid": valid,
        "n_events": n_events,
        "neg_counts": neg_counts,
        "neg_bins": neg_bins,
        "pos_counts": pos_counts,
        "pos_bins": pos_bins,
        "parameters": parameters,
        "processed_at": datetime.now().isoformat(),
    }


def _fit_gaussian_with_fallbacks(
    counts: np.ndarray,
    bins: np.ndarray,
    low_stats_threshold: float,
    y_mask_threshold: float,
    sharp_fit_min_points: int,
) -> tuple[float, float, float, int, int]:
    """Fit a gaussian to one histogram.

    Returns ``(A, mu, sigma, total_counts, status)``, where *status* is one of
    the values of :data:`FIT_STATUS` and the three fit parameters are NaN
    wherever that status says they are not available.
    """
    y = np.asarray(counts, dtype=float)
    total_counts = int(y.sum())

    if total_counts == 0:
        return np.nan, np.nan, np.nan, 0, FIT_STATUS["no_stats"]

    x = pgh.get_bin_centers(bins)

    # too few counts, fallback to histogram arithmetic mean 
    if total_counts < low_stats_threshold:
        mu = float(np.sum(x * y) / total_counts)
        sigma = float(np.sqrt(np.sum(y * (x - mu) ** 2) / total_counts))
        return np.nan, mu, sigma, total_counts, FIT_STATUS["low_stats"]

    # fit the peak rather than the tails
    mask = y > y_mask_threshold * np.max(y)
    if int(mask.sum()) < sharp_fit_min_points:
        # peak too sharp, fallback to no mask
        mask = np.ones_like(y, dtype=bool)
        status = FIT_STATUS["ok_few_points"]
    else:
        status = FIT_STATUS["ok"]

    x_fit = x[mask]
    y_fit = y[mask]
    amplitude_0 = float(np.max(y_fit))
    mu_0 = float(np.average(x_fit, weights=y_fit))
    sigma_0 = float(np.sqrt(np.average((x_fit - mu_0) ** 2, weights=y_fit)))
    if sigma_0 <= 0:
        sigma_0 = float(x[1] - x[0]) if len(x) > 1 else 1.0 # Prevent ZeroDivisionError

    try:
        popt, _ = curve_fit(nb_gauss_amp, x_fit, y_fit, p0=[mu_0, sigma_0, amplitude_0])
    except (RuntimeError, ValueError) as e:
        log.debug("gaussian fit did not converge: %s", e)
        return np.nan, np.nan, np.nan, total_counts, FIT_STATUS["fit_failed"]

    mu, sigma, amplitude = (float(v) for v in popt)
    return amplitude, mu, abs(sigma), total_counts, status


def xtalk_histogram_fitter(
    histogram_data: dict,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Fit a gaussian to every histogram of one cross-talk column.

    *histogram_data* is exactly the dict :func:`xtalk_column` returns.

    Every element is fitted twice, once against the negative and once against
    the positive response, and each fit lands in one of the outcomes of
    :data:`FIT_STATUS`:

    ``ok``
        the fit converged on the bins above ``y_mask_threshold`` of the peak.
    ``ok_few_points``
        that mask left fewer than ``sharp_fit_min_points`` bins, so the fit
        converged on all of them instead.
    ``low_stats``
        fewer than ``low_stats_threshold`` counts, so ``mu`` and ``sigma``
        are the moments of the histogram rather than a fit, and ``A`` is NaN.
    ``fit_failed``
        the fit did not converge; all three parameters are NaN.
    ``no_stats``
        the histogram is empty.
    ``not_filled``
        The fitting is skipped if "valid" is False, leaving FIT_STATUS "not_filled".

    Parameters
    ----------
    histogram_data
        Result dict of :func:`xtalk_column`.
    config
        Fit configuration, read straight off the top level.  Recognised
        keys, all optional:

        ``low_stats_threshold``
            Counts below which the moments replace the fit.  Default 100.
        ``y_mask_threshold``
            Bins below this fraction of the tallest one are dropped before
            fitting.  Default 0.05.
        ``sharp_fit_min_points``
            Bins that mask must leave for the fit to use it.  Default 5.
    debug_mode
        If True, re-raise instead of recording an element as ``fit_failed``.

    Returns
    -------
    dict
        Every key of *histogram_data*, unchanged, plus ``{neg,pos}_A``,
        ``{neg,pos}_mu``, ``{neg,pos}_sigma``, ``{neg,pos}_total_counts``,
        ``{neg,pos}_status`` and ``{neg,pos}_success``, all ``(N,)``, and
        ``fit_parameters``, ``fit_status_codes`` and ``fitted_at``.
        ``total_counts`` is the integral of the histogram, which is at most
        ``n_events`` -- events outside the histogram range are not in it.
    """

    config = config or {}
    low_stats_threshold = float(
        config.get("low_stats_threshold", DEFAULT_LOW_STATS_THRESHOLD)
    )
    y_mask_threshold = float(config.get("y_mask_threshold", DEFAULT_Y_MASK_THRESHOLD))
    sharp_fit_min_points = int(
        config.get("sharp_fit_min_points", DEFAULT_SHARP_FIT_MIN_POINTS)
    )

    trigger_id = histogram_data["trigger_id"]
    response_ids = np.asarray(histogram_data["response_ids"])
    valid = np.asarray(histogram_data["valid"], dtype=bool)
    n_response = len(response_ids)

    result = dict(histogram_data)

    for polarity in ("neg", "pos"):
        counts = np.asarray(histogram_data[f"{polarity}_counts"])
        bins = np.asarray(histogram_data[f"{polarity}_bins"])

        amplitude = np.full(n_response, np.nan)
        mu = np.full(n_response, np.nan)
        sigma = np.full(n_response, np.nan)
        total_counts = np.zeros(n_response, dtype=np.int64)
        status = np.full(n_response, FIT_STATUS["not_filled"], dtype=np.int8)

        for k in range(n_response):
            if not valid[k]:
                continue

            try:
                fit = _fit_gaussian_with_fallbacks(
                    counts[k],
                    bins[k],
                    low_stats_threshold,
                    y_mask_threshold,
                    sharp_fit_min_points,
                )
            except Exception as e:
                if debug_mode:
                    raise
                log.error(
                    "%s fit of element (%s, %s) failed: %s",
                    polarity,
                    trigger_id,
                    response_ids[k],
                    e,
                )
                status[k] = FIT_STATUS["fit_failed"]
                continue

            amplitude[k], mu[k], sigma[k], total_counts[k], status[k] = fit

        result[f"{polarity}_A"] = amplitude
        result[f"{polarity}_mu"] = mu
        result[f"{polarity}_sigma"] = sigma
        result[f"{polarity}_total_counts"] = total_counts
        result[f"{polarity}_status"] = status
        result[f"{polarity}_success"] = np.isin(status, FIT_STATUS_SUCCESS)

    result["fit_parameters"] = {
        "low_stats_threshold": low_stats_threshold,
        "y_mask_threshold": y_mask_threshold,
        "sharp_fit_min_points": sharp_fit_min_points,
    }
    result["fit_status_codes"] = FIT_STATUS
    result["fitted_at"] = datetime.now().isoformat()

    log.info(
        "xtalk fits of trigger %s: %s negative and %s positive of %s elements "
        "converged",
        trigger_id,
        int(result["neg_success"].sum()),
        int(result["pos_success"].sum()),
        n_response,
    )

    return result


def build_xtalk_matrix(
    fitted_columns: dict,
    config: dict | None = None,
) -> lgdo.Table:
    """Assemble the fitted columns of a cross-talk matrix into the matrix.

    Element ``[j1, j2]`` of a matrix:  ``rawid_index[j1]`` represents triggered 
    detector, while ``rawid_index[j2]`` represents the responding detector.  

    Parameters
    ----------
    fitted_columns
        Result dicts of :func:`xtalk_histogram_fitter`, keyed by trigger
        channel id.
    config
        Recognised keys, all optional:

        ``max_status``
            Highest :data:`FIT_STATUS` code to accept into the matrix. Default
            ``FIT_STATUS["low_stats"]``.

    Returns
    -------
    lgdo.Table
        A table with the following fields. Values are sorted in the order of 
        ``rawid_index``:

        ``rawid_index`` ``(N,)``
            Detector ids: ``rawid_index[j]`` is the detector at row and
            column ``j`` of every matrix.
        ``xtalk_matrix_negative``, ``xtalk_matrix_positive`` ``(N, N)``
            The fitted peak positions, as **fractions**, which is the unit
            the production files store.
        ``..._sigma`` ``(N, N)``
            The width of each of those fits, also as fractions.
        ``..._status`` ``(N, N)``
            The :data:`FIT_STATUS` code of each element, meaning explained in 
            :func:`xtalk_histogram_fitter`. 
    """
    config = config or {}
    max_status = int(config.get("max_status", FIT_STATUS["low_stats"]))

    columns = {int(trigger_id): column for trigger_id, column in fitted_columns.items()}
    rawids = None

    # Check whether the response id lists are identical across all columns
    for trigger_id, column in columns.items():
        response_ids = np.asarray(column["response_ids"], dtype=np.int64)
        if rawids is None:
            rawids = response_ids
        elif not np.array_equal(rawids, response_ids):
            msg = (
                f"column {trigger_id} covers different detectors, or covers "
                f"them in a different order, than the columns before it"
            )
            raise ValueError(msg)

        if int(column["trigger_id"]) != trigger_id:
            msg = (
                f"column filed under trigger {trigger_id} reports "
                f"trigger_id {column['trigger_id']}"
            )
            raise ValueError(msg)

    index_of = {int(rawid): j for j, rawid in enumerate(rawids)}
    unknown = sorted(set(columns) - set(index_of))
    if unknown:
        msg = (
            f"columns {unknown} trigger on detectors that are not among the "
            f"responses, so they have no row in the matrix"
        )
        raise ValueError(msg)

    n_detectors = len(rawids)
    shape = (n_detectors, n_detectors)
    mu = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    sigma = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    status = {
        p: np.full(shape, FIT_STATUS["not_filled"], dtype=np.int8)
        for p in ("neg", "pos")
    }

    for trigger_id, column in columns.items():
        row = index_of[trigger_id]
        for polarity in ("neg", "pos"):
            column_status = np.asarray(column[f"{polarity}_status"], dtype=np.int8)
            accepted = column_status <= max_status

            status[polarity][row] = column_status
            mu[polarity][row] = np.where(
                accepted, np.asarray(column[f"{polarity}_mu"], dtype=float), np.nan
            )
            sigma[polarity][row] = np.where(
                accepted, np.asarray(column[f"{polarity}_sigma"], dtype=float), np.nan
            )

    col_dict = {"rawid_index": lgdo.Array(np.asarray(rawids, dtype=np.int64))}
    for polarity in ("neg", "pos"):
        field = XTC_LH5_FIELD[polarity]
        col_dict[field] = lgdo.Array(mu[polarity])
        col_dict[f"{field}_sigma"] = lgdo.Array(sigma[polarity])
        col_dict[f"{field}_status"] = lgdo.Array(status[polarity])

    missing = n_detectors - len(columns)
    if missing:
        log.warning(
            "%s of %s detectors have no fitted column, their rows stay NaN",
            missing,
            n_detectors,
        )

    return lgdo.Table(
        col_dict=col_dict, attrs={"fit_status_codes": json.dumps(FIT_STATUS)}
    )


def plot_xtalk_matrix(
    matrix: lgdo.Table,
    polarity: str = "neg",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
    title: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Draw a heatmap of one polarity of a cross-talk matrix.

    Parameters
    ----------
    matrix
        Table :func:`build_xtalk_matrix` returned, or one read back from an
        xtc lh5 file with :func:`lh5.read`.  Only the polarity's own column
        is used, so a production file that carries nothing but the two
        matrices plots as well as one this module wrote.
    polarity
        ``"neg"`` or ``"pos"``, naming the column through
        :data:`XTC_LH5_FIELD`.
    vmin, vmax
        Colour-scale limits, as fractions.  ``None`` takes the polarity's
        entry in :data:`XTC_PLOT_RANGE`.
    cmap
        Colormap, defaulting to reversed jet as in the original analysis.
    title
        Figure title.  ``None`` names the polarity.
    figsize
        Figure size, in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, for the caller to show or to ``savefig``.
    """
    if polarity not in XTC_LH5_FIELD:
        msg = f"polarity must be one of {tuple(XTC_LH5_FIELD)}, got {polarity!r}"
        raise ValueError(msg)

    values = matrix[XTC_LH5_FIELD[polarity]]
    values = np.asarray(values.nda if hasattr(values, "nda") else values)

    default_vmin, default_vmax = XTC_PLOT_RANGE[polarity]

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        values,
        origin="lower",
        vmin=default_vmin if vmin is None else vmin,
        vmax=default_vmax if vmax is None else vmax,
        cmap=plt.cm.jet_r if cmap is None else cmap,
    )
    fig.colorbar(image, ax=ax, label="Cross-talk (fraction)")
    ax.set_xlabel("Response channel index")
    ax.set_ylabel("Trigger channel index")
    ax.set_title(title if title is not None else f"{polarity} cross-talk matrix")
    fig.tight_layout()

    return fig
