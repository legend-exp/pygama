"""
This module provides routines for measuring cross-talk (XTC) between
germanium detectors and for building the resulting cross-talk matrix.

The four main functions, in order of execution, are:
prepare_detector, attach_response_amps, xtalk_element, and build_xtalk_matrix.
The others are just helpers.
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

log = logging.getLogger(__name__)

DEFAULT_ENERGY_PARAM = "cuspEmax_ctc_cal"
DEFAULT_BASELINE_CONDITIONS = {"is_empty_candidate": 63}
DEFAULT_TRIGGER_CONDITIONS = {"is_highly_positive_polarity_candidate": 511}
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
#: trustworthy. Written into the lh5 file as ``fit_status_codes``.
FIT_STATUS = {
    "ok": 0,
    "ok_few_points": 1,
    "low_stats": 2,
    "fit_failed": 3,
    "no_stats": 4,
    "not_filled": 5,
}
FIT_STATUS_SUCCESS = (FIT_STATUS["ok"], FIT_STATUS["ok_few_points"])

XTC_LH5_FIELD = {"neg": "xtalk_matrix_negative", "pos": "xtalk_matrix_positive"}
XTC_PLOT_RANGE = {"neg": (-0.003, 0.001), "pos": (-0.0007, 0.003)}

# Keys that will be copied into the xtalk table as attributes to record
# the settings during production.
XTC_SELECTION_KEYS = {
    "trigger": (
        "energy_param",
        "trigger_param",
        "trigger_conditions",
        "trigger_energy_range",
    ),
    "response": (
        "energy_param",
        "positive_param",
        "negative_param",
        "baseline_conditions",
        "response_conditions",
        "response_energy_range",
    ),
}
XTC_SETTING_KEYS = (
    "nbins",
    "range_multiplier",
    "low_stats_threshold",
    "y_mask_threshold",
    "sharp_fit_min_points",
    "trigger_selection_parameters",
    "response_selection_parameters",
)


def _selection_mask(
    table,
    ene_field: str,
    conditions: dict | None = None,
    energy_range: tuple | None = None,
) -> np.ndarray:
    """
    Rows of *table* that survive the event cuts, as a boolean mask.
    """
    energies = table[ene_field].nda
    mask = np.isfinite(energies)

    for flag, value in (conditions or {}).items():
        mask &= table[flag].nda == value

    if energy_range is not None:
        emin, emax = energy_range
        mask &= (energies >= emin) & (energies <= emax)

    return mask


def prepare_detector(
    hit_files: str | list,
    dsp_files: str | list,
    chn_id: str | int,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Read one detector once and keep the selections the matrix needs from it.
    The primary information that is retrieved:

    *trigger*
        the events in which it fired, as global entry numbers, together with
        the DSP amplitude each of them fired with -- the denominator of the
        cross-talk ratio.
    *response*
        its baselines (pos & neg) and a boolean mask of which events are usable
        as cross-talk responses.  The response amplitudes themselves are left to
        :func:`attach_response_amps`, so that what this returns stays small.

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
            event to count as a baseline.  Default ``{"is_empty_candidate": 63}``.
        ``trigger_conditions``
            Mapping of hit-tier flag field to the value it must equal for an
            event to count as a trigger.  Default
            ``{"is_highly_positive_polarity_candidate": 511}``.
        ``trigger_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting real triggers.
            Default ``(1500, 99999)``. The upper bound filters out unrealistic events.
        ``response_conditions``
            Mapping of hit-tier flag field to the value it must equal for the
            detector to be measurable as a response.  Default ``{}``.
        ``response_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting the events in which
            this detector did *not* see a real hit -- an event in which it did
            is a multiplicity event, not cross-talk.  Default
            ``(-99999, 100)``.
        ``energy_param``
            Hit-tier field all three selections are applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``positive_param``, ``negative_param``
            DSP-tier fields holding the response amplitudes, averaged over the
            baseline events to give the baselines.
            Defaults ``"trapTmax"`` and ``"trapTmin"``.
        ``trigger_param``
            DSP-tier field giving the amplitude this detector triggered with.
            Default ``"trapTmax"``.
    debug_mode
        If True, re-raise instead of falling back to a null result.

    Returns
    -------
    dict
        Metadata -- ``detector_id``, ``n_rows``, ``processed_at`` and
        ``parameters`` -- plus:

        ``negative_baseline``, ``positive_baseline``: the baselines to subtract
        from the response amplitudes. They are always both finite or both ``None``.
        ``response_keep``: which events are usable as a response. It is a boolean
        mask whose length is the total number of events.
        ``trigger_idxs``, ``trigger_amplitudes``: the events in which this
        detector fired, and the corresponding DSP amplitudes.
        ``read_success``, ``baseline_success``, ``trigger_success``: booleans
        indicating whether the detector was read and the selections were made.
    """
    config = config or {}
    baseline_conditions = dict(
        config.get("baseline_conditions", DEFAULT_BASELINE_CONDITIONS)
    )
    trigger_conditions = dict(
        config.get("trigger_conditions", DEFAULT_TRIGGER_CONDITIONS)
    )
    response_conditions = dict(config.get("response_conditions", {}))
    trigger_energy_range = tuple(
        config.get("trigger_energy_range", DEFAULT_TRIGGER_ENERGY_RANGE)
    )
    response_energy_range = tuple(
        config.get("response_energy_range", DEFAULT_RESPONSE_ENERGY_RANGE)
    )
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    trigger_param = config.get("trigger_param", DEFAULT_TRIGGER_PARAM)

    dsp_fields = list(dict.fromkeys([positive_param, negative_param, trigger_param]))
    hit_fields = list(
        dict.fromkeys(
            [
                energy_param,
                *baseline_conditions,
                *trigger_conditions,
                *response_conditions,
            ]
        )
    )

    read_success = True
    baseline_success = True
    trigger_success = True
    positive_baseline = None
    negative_baseline = None
    n_rows = 0
    response_keep = np.empty(0, dtype=bool)
    trigger_idxs = np.empty(0, dtype=np.int64)
    trigger_amplitudes = np.empty(0, dtype=np.float32)

    baseline_mask = None
    trigger_mask = None

    try:
        hit_table = lh5.read(f"ch{chn_id}/hit/", hit_files, field_mask=hit_fields)
        n_rows = len(hit_table[energy_param].nda)

        baseline_mask = _selection_mask(hit_table, energy_param, baseline_conditions)
        trigger_mask = _selection_mask(
            hit_table, energy_param, trigger_conditions, trigger_energy_range
        )
        response_keep = _selection_mask(
            hit_table, energy_param, response_conditions, response_energy_range
        )
        del hit_table

        dsp_table = lh5.read(f"ch{chn_id}/dsp/", dsp_files, field_mask=dsp_fields)
        positive_amps = dsp_table[positive_param].nda
        negative_amps = dsp_table[negative_param].nda
        trigger_amps_all = dsp_table[trigger_param].nda

        if len(positive_amps) != n_rows:
            msg = (
                f"the hit tier holds {n_rows} events and the dsp tier "
                f"{len(positive_amps)}, so they do not describe the same "
                f"events"
            )
            raise RuntimeError(msg)
    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "reading channel %s failed, none of its selections were made: %s: %s",
            chn_id,
            type(e).__name__,
            e,
        )
        read_success = False
        baseline_success = False
        trigger_success = False
        n_rows = 0
        response_keep = np.empty(0, dtype=bool)

    if read_success:
        try:
            positive_baseline_vals = positive_amps[baseline_mask]
            negative_baseline_vals = negative_amps[baseline_mask]
            positive_baseline_vals = positive_baseline_vals[
                np.isfinite(positive_baseline_vals)
            ]
            negative_baseline_vals = negative_baseline_vals[
                np.isfinite(negative_baseline_vals)
            ]
            if len(positive_baseline_vals) == 0 or len(negative_baseline_vals) == 0:
                msg = "no events passed the baseline selection"
                raise RuntimeError(msg)

            positive_baseline = float(np.mean(positive_baseline_vals))
            negative_baseline = float(np.mean(negative_baseline_vals))
            if not np.isfinite(positive_baseline) or not np.isfinite(negative_baseline):
                msg = (
                    f"the baseline average came out as "
                    f"({positive_baseline}, {negative_baseline}), which is not "
                    f"a pair of numbers to subtract"
                )
                raise RuntimeError(msg)
        except Exception as e:
            if debug_mode:
                raise
            log.error("baseline preparation failed for channel %s: %s", chn_id, e)
            positive_baseline = None
            negative_baseline = None
            baseline_success = False

        try:
            idxs = np.flatnonzero(trigger_mask).astype(np.int64)
            amplitudes = trigger_amps_all[trigger_mask]

            # a zero or non-finite trigger amplitude cannot be divided by
            usable = np.isfinite(amplitudes) & (amplitudes != 0)
            if not usable.all():
                log.debug(
                    "trigger %s: dropping %d of %d events with a non-finite or zero %s",
                    chn_id,
                    int((~usable).sum()),
                    len(usable),
                    trigger_param,
                )
                idxs = idxs[usable]
                amplitudes = amplitudes[usable]

            if len(idxs) == 0:
                msg = "no events passed the trigger selection"
                raise RuntimeError(msg)

            trigger_idxs = idxs
            trigger_amplitudes = amplitudes
        except Exception as e:
            if debug_mode:
                raise
            log.error("trigger selection failed for channel %s: %s", chn_id, e)
            trigger_idxs = np.empty(0, dtype=np.int64)
            trigger_amplitudes = np.empty(0, dtype=np.float32)
            trigger_success = False

    log.info(
        "channel %s: %s events, baseline %s, %s of them measurable as a "
        "response, %s trigger events",
        chn_id,
        n_rows,
        "measured" if baseline_success else "not measured",
        int(response_keep.sum()),
        len(trigger_idxs),
    )

    return {
        "detector_id": chn_id,
        "n_rows": n_rows,
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "response_keep": response_keep,
        "trigger_idxs": trigger_idxs,
        "trigger_amplitudes": trigger_amplitudes,
        "read_success": read_success,
        "baseline_success": baseline_success,
        "trigger_success": trigger_success,
        "processed_at": datetime.now().isoformat(),
        "parameters": {
            "baseline_conditions": baseline_conditions,
            "trigger_conditions": trigger_conditions,
            "response_conditions": response_conditions,
            "trigger_energy_range": list(trigger_energy_range),
            "response_energy_range": list(response_energy_range),
            "energy_param": energy_param,
            "positive_param": positive_param,
            "negative_param": negative_param,
            "trigger_param": trigger_param,
            "hit_files": [hit_files] if isinstance(hit_files, str) else list(hit_files),
            "dsp_files": [dsp_files] if isinstance(dsp_files, str) else list(dsp_files),
        },
    }


def attach_response_amps(
    detector_info: dict,
    debug_mode: bool = False,
) -> dict:
    """Add per-event response amplitudes to ``detector_info``.

    The amplitudes are needed by :func:`xtalk_element` but are omitted from
    :func:`prepare_detector` because they are copied directly from the DSP
    fields and do not need to be written to disk. Amplitudes are attached here
    rather than read inside :func:`xtalk_element` to avoid reading the DSP
    files separately for each cross-talk element.

    Parameters
    ----------
    detector_info
        The dict returned by :func:`prepare_detector`.
    debug_mode
        If True, re-raise instead of falling back to empty arrays.

    Returns
    -------
    dict
        A shallow copy of *detector_info* with ``positive_response_amps`` and
        ``negative_response_amps`` ``(n_rows,)`` added, which is what
        :func:`xtalk_element` wants of a responding detector.
    """
    detector_id = detector_info.get("detector_id")
    parameters = detector_info.get("parameters") or {}
    n_rows = int(detector_info.get("n_rows", 0))
    positive_param = parameters.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = parameters.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    dsp_files = parameters.get("dsp_files")

    positive_response_amps = np.empty(0, dtype=np.float32)
    negative_response_amps = np.empty(0, dtype=np.float32)

    if not detector_info.get("read_success", True):
        log.debug(
            "channel %s was never read, so it has no amplitudes to attach",
            detector_id,
        )
    else:
        try:
            if dsp_files is None:
                msg = (
                    "parameters records no dsp_files, so this detector was not "
                    "prepared by a version of prepare_detector that records them"
                )
                raise RuntimeError(msg)

            dsp_fields = list(dict.fromkeys([positive_param, negative_param]))
            dsp_table = lh5.read(
                f"ch{detector_id}/dsp/", dsp_files, field_mask=dsp_fields
            )
            positive = dsp_table[positive_param].nda
            negative = dsp_table[negative_param].nda

            if len(positive) != n_rows or len(negative) != n_rows:
                msg = (
                    f"channel {detector_id} was prepared over {n_rows} events "
                    f"but these files hold {len(positive)}, so they are not "
                    f"the files it was prepared from"
                )
                raise RuntimeError(msg)

            positive_response_amps = positive
            negative_response_amps = negative
        except Exception as e:
            if debug_mode:
                raise
            log.error(
                "attaching response amplitudes to channel %s failed, so every "
                "element measured against it will be unfilled: %s: %s",
                detector_id,
                type(e).__name__,
                e,
            )
            positive_response_amps = np.empty(0, dtype=np.float32)
            negative_response_amps = np.empty(0, dtype=np.float32)

    attached = dict(detector_info)
    attached["positive_response_amps"] = positive_response_amps
    attached["negative_response_amps"] = negative_response_amps
    return attached


def _resolve_trigger(detector_info: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return ``(idxs, amplitudes)`` of *detector_info*, or None if unusable.
    """
    idxs = detector_info.get("trigger_idxs")
    amplitudes = detector_info.get("trigger_amplitudes")
    if idxs is None or amplitudes is None:
        return None

    idxs = np.asarray(idxs, dtype=np.int64)
    # not cast to float64: the cross-talk ratio is computed in whatever
    # precision the dsp tier stored the amplitude in
    amplitudes = np.asarray(amplitudes)
    if idxs.size != amplitudes.size:
        msg = (
            f"channel {detector_info.get('detector_id')} has {idxs.size} trigger "
            f"indices but {amplitudes.size} trigger amplitudes"
        )
        raise ValueError(msg)
    if idxs.size == 0:
        return None

    return idxs, amplitudes


def _build_hist(
    vals: np.ndarray, nbins: int, range_multiplier: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Histogram *vals* over ``mean +/- range_multiplier * stdev``.

    Returns ``None`` when the sample is empty or has no usable spread.
    """
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    mean = np.mean(vals)
    stdev = np.std(vals)
    if not np.isfinite(mean) or not np.isfinite(stdev) or stdev <= 0:
        return None

    counts, edges = np.histogram(
        vals,
        bins=nbins,
        range=(mean - range_multiplier * stdev, mean + range_multiplier * stdev),
    )
    # numpy hands back the edges in the dtype of the values it binned, which is
    # single precision for a dsp amplitude; widening them keeps the bin centres
    # the fit is run on -- and so the fitted peak position -- out of float32
    return counts, edges.astype(np.float64)


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
        sigma_0 = float(x[1] - x[0]) if len(x) > 1 else 1.0  # Prevent ZeroDivisionError

    try:
        popt, _ = curve_fit(nb_gauss_amp, x_fit, y_fit, p0=[mu_0, sigma_0, amplitude_0])
    except (RuntimeError, ValueError) as e:
        log.debug("gaussian fit did not converge: %s", e)
        return np.nan, np.nan, np.nan, total_counts, FIT_STATUS["fit_failed"]

    mu, sigma, amplitude = (float(v) for v in popt)
    return amplitude, mu, abs(sigma), total_counts, status


def xtalk_element(
    trigger_detector_info: dict,
    response_detector_info: dict,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Measure one element of the cross-talk matrix.

    Retrieve the computed masks and amplitudes from the two detectors,
    build the cross-talk ratio histogram for events that pass both the trigger
    and the response selections, and fit it.

    Parameters
    ----------
    trigger_detector_info
        The :func:`prepare_detector` result of the triggering detector.
    response_detector_info
        The :func:`prepare_detector` result of the responding detector, with
        the amplitudes added by :func:`attach_response_amps`.
    config
        Histogram and fit configuration.  Recognised keys, all optional:

        ``nbins``
            Bins in the histogram.  Default 700.
        ``range_multiplier``
            Histogram half-width in standard deviations about the mean.
            Default 3.
        ``low_stats_threshold``
            Counts below which the histogram moments replace the fit.
            Default 100.
        ``y_mask_threshold``
            Bins below this fraction of the tallest one are dropped before
            fitting.  Default 0.05.
        ``sharp_fit_min_points``
            The ``y_mask_threshold`` mask is given up when it leaves fewer
            bins than this.  Default 5.
    debug_mode
        If True, re-raise instead of recording an unfilled or failed element.

    Returns
    -------
    dict
        ``trigger_id``, ``response_id``, ``valid``, ``n_events``, then per
        polarity ``{neg,pos}_counts`` ``(nbins,)`` and ``{neg,pos}_bins``
        ``(nbins + 1,)`` holding the histogram, and ``{neg,pos}_A``,
        ``{neg,pos}_mu``, ``{neg,pos}_sigma``, ``{neg,pos}_total_counts``,
        ``{neg,pos}_status`` and ``{neg,pos}_success`` holding the fit.
        Finally ``parameters``, ``fit_status_codes`` and ``processed_at``.

        ``parameters`` holds the settings used to produce the element, and
        will be copied into the cross-talk matrix table as attributes.

        ``fit_status_codes`` defines the mapping of each of the following fit outcomes
        to an integer, which is what is stored in the ``{neg,pos}_status`` fields:

        ``ok``
            the fit converged on the bins above ``y_mask_threshold`` of the
            peak.
        ``ok_few_points``
            that mask left fewer than ``sharp_fit_min_points`` bins, so the
            fit converged on all of them instead.
        ``low_stats``
            fewer than ``low_stats_threshold`` counts, so ``mu`` and ``sigma``
            are the moments of the histogram rather than a fit, and ``A`` is
            NaN.
        ``fit_failed``
            the fit did not converge; all three parameters are NaN.
        ``no_stats``
            the histogram is empty.
        ``not_filled``
            the pair had nothing to measure, so no histogram was built.
    """
    config = config or {}
    nbins = int(config.get("nbins", DEFAULT_NBINS))
    range_multiplier = float(config.get("range_multiplier", DEFAULT_RANGE_MULTIPLIER))
    low_stats_threshold = float(
        config.get("low_stats_threshold", DEFAULT_LOW_STATS_THRESHOLD)
    )
    y_mask_threshold = float(config.get("y_mask_threshold", DEFAULT_Y_MASK_THRESHOLD))
    sharp_fit_min_points = int(
        config.get("sharp_fit_min_points", DEFAULT_SHARP_FIT_MIN_POINTS)
    )

    trigger_id = trigger_detector_info.get("detector_id")
    response_id = response_detector_info.get("detector_id")

    positive_baseline = response_detector_info.get("positive_baseline")
    negative_baseline = response_detector_info.get("negative_baseline")

    valid = False
    n_events = 0
    histograms = {"neg": None, "pos": None}

    self_interaction = str(trigger_id) == str(response_id)
    if self_interaction:
        log.debug("self-interaction at channel %s ignored", trigger_id)

    if not self_interaction:
        try:
            if positive_baseline is None or negative_baseline is None:
                msg = f"response channel {response_id} has no usable baseline"
                raise RuntimeError(msg)

            positive_baseline = float(positive_baseline)
            negative_baseline = float(negative_baseline)

            missing = [
                key
                for key in ("positive_response_amps", "negative_response_amps")
                if key not in response_detector_info
            ]
            if missing:
                msg = (
                    f"response channel {response_id} is missing {', '.join(missing)}: "
                    f"a responding channel has to go through attach_response_amps "
                    f"before it can be measured against"
                )
                raise KeyError(msg)

            response_keep = np.asarray(
                response_detector_info["response_keep"], dtype=bool
            )
            positive_response_amps = np.asarray(
                response_detector_info["positive_response_amps"]
            )
            negative_response_amps = np.asarray(
                response_detector_info["negative_response_amps"]
            )

            n_total = len(response_keep)
            if n_total == 0:
                msg = f"response channel {response_id} holds no events"
                raise RuntimeError(msg)
            if (
                len(positive_response_amps) != n_total
                or len(negative_response_amps) != n_total
            ):
                msg = (
                    f"response channel {response_id} has {n_total} selected "
                    f"events but {len(positive_response_amps)} positive and "
                    f"{len(negative_response_amps)} negative amplitudes"
                )
                raise RuntimeError(msg)

            resolved = _resolve_trigger(trigger_detector_info)
            if resolved is None:
                msg = f"trigger channel {trigger_id} selected no usable events"
                raise RuntimeError(msg)
            trigger_idxs, trigger_amplitudes_all = resolved

            if trigger_idxs.max() >= n_total:
                msg = (
                    f"trigger index {trigger_idxs.max()} is past the {n_total} "
                    f"events of channel {response_id}, so the two channels do "
                    f"not cover the same events"
                )
                raise IndexError(msg)

            keep = response_keep[trigger_idxs]
            coincident_idxs = trigger_idxs[keep]
            trigger_amplitudes = trigger_amplitudes_all[keep]

            neg_vals = (
                negative_response_amps[coincident_idxs] - negative_baseline
            ) / trigger_amplitudes
            pos_vals = (
                positive_response_amps[coincident_idxs] - positive_baseline
            ) / trigger_amplitudes

            valid = True
            n_events = len(trigger_amplitudes)
            histograms["neg"] = _build_hist(neg_vals, nbins, range_multiplier)
            histograms["pos"] = _build_hist(pos_vals, nbins, range_multiplier)

        except Exception as e:
            if debug_mode:
                raise
            log.error("xtalk element (%s, %s) failed: %s", trigger_id, response_id, e)
            valid = False
            n_events = 0
            histograms = {"neg": None, "pos": None}

    result = {
        "trigger_id": trigger_id,
        "response_id": response_id,
        "valid": valid,
        "n_events": int(n_events),
    }

    for polarity in ("neg", "pos"):
        counts = np.zeros(nbins, dtype=np.int64)
        bins = np.full(nbins + 1, np.nan)
        amplitude = np.nan
        mu = np.nan
        sigma = np.nan
        total_counts = 0
        status = FIT_STATUS["not_filled"]

        if valid:
            histogram = histograms[polarity]
            if histogram is not None:
                counts, bins = histogram
            else:
                log.debug(
                    "%s histogram for element (%s, %s) is empty",
                    polarity,
                    trigger_id,
                    response_id,
                )

            try:
                amplitude, mu, sigma, total_counts, status = (
                    _fit_gaussian_with_fallbacks(
                        counts,
                        bins,
                        low_stats_threshold,
                        y_mask_threshold,
                        sharp_fit_min_points,
                    )
                )
            except Exception as e:
                if debug_mode:
                    raise
                log.error(
                    "%s fit of element (%s, %s) failed: %s",
                    polarity,
                    trigger_id,
                    response_id,
                    e,
                )
                status = FIT_STATUS["fit_failed"]

        result[f"{polarity}_counts"] = counts
        result[f"{polarity}_bins"] = bins
        result[f"{polarity}_A"] = amplitude
        result[f"{polarity}_mu"] = mu
        result[f"{polarity}_sigma"] = sigma
        result[f"{polarity}_total_counts"] = int(total_counts)
        result[f"{polarity}_status"] = int(status)
        result[f"{polarity}_success"] = int(status) in FIT_STATUS_SUCCESS

    trigger_parameters = trigger_detector_info.get("parameters", {})
    response_parameters = response_detector_info.get("parameters", {})

    result["parameters"] = {
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "nbins": nbins,
        "range_multiplier": range_multiplier,
        "low_stats_threshold": low_stats_threshold,
        "y_mask_threshold": y_mask_threshold,
        "sharp_fit_min_points": sharp_fit_min_points,
        "trigger_selection_parameters": {
            key: trigger_parameters[key]
            for key in XTC_SELECTION_KEYS["trigger"]
            if key in trigger_parameters
        },
        "response_selection_parameters": {
            key: response_parameters[key]
            for key in XTC_SELECTION_KEYS["response"]
            if key in response_parameters
        },
    }
    result["fit_status_codes"] = FIT_STATUS
    result["processed_at"] = datetime.now().isoformat()

    log.debug(
        "xtalk element (%s, %s): %s events, neg %s, pos %s",
        trigger_id,
        response_id,
        n_events,
        result["neg_status"],
        result["pos_status"],
    )

    return result


def _differing_keys(left: dict, right: dict, prefix: str = "") -> list[str]:
    """Which settings two elements disagree on, named down to the leaf."""
    differing = []
    for key in sorted(set(left) | set(right)):
        left_value, right_value = left.get(key), right.get(key)
        if left_value == right_value:
            continue
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            differing += _differing_keys(left_value, right_value, f"{prefix}{key}.")
        else:
            differing.append(f"{prefix}{key}")
    return differing


def build_xtalk_matrix(
    fitted_elements: list[dict],
    rawids: list[int] | None = None,
    config: dict | None = None,
) -> lgdo.Table:
    """Assemble measured cross-talk elements into the matrix.

    Given an element ``[j1, j2]`` of a matrix, one could use ``rawid_index[j1]``
    to find the channel id of the triggering detector and ``rawid_index[j2]``
    the channel id of the responding detector.

    Parameters
    ----------
    fitted_elements
        List of :func:`xtalk_element` results.
    rawids
        Order to index the matrix in.  Must hold exactly the channel ids seen
        in ``fitted_elements``.  ``None`` takes them in first-seen order.
    config
        Recognised keys, all optional:

        ``max_status``
            Highest :data:`FIT_STATUS` code to accept into the matrix. Default
            ``FIT_STATUS["low_stats"]``.
        ``require_same_parameters``
            If True, raise when the elements were not all produced with
            the same settings. Default True.

    Returns
    -------
    lgdo.Table
        A table with the following fields. Values are sorted in the order of
        ``rawid_index``:

        ``rawid_index`` ``(N,)``
            ``rawid_index[i]`` is the channel id of the detector corresponding to
            row or column *i* of the matrix.
        ``xtalk_matrix_negative``, ``xtalk_matrix_positive`` ``(N, N)``
            The fitted peak positions, i.e., the xtalk values, as **fractions**.
        ``..._sigma`` ``(N, N)``
            The width of each of those fits, also as fractions.
        ``..._status`` ``(N, N)``
            The :data:`FIT_STATUS` code of each element, meaning explained in
            :func:`xtalk_element`.

        ``fit_status_codes`` and the keys from ``parameters`` that are within
        :data:`XTC_SETTING_KEYS` are propagated from the elements and saved as
        attributes of the table.
    """
    config = config or {}
    max_status = int(config.get("max_status", FIT_STATUS["low_stats"]))
    require_same_parameters = bool(config.get("require_same_parameters", True))

    # index the rawids by first seen order by default
    index_of: dict[int, int] = {}
    for element in fitted_elements:
        for key in ("trigger_id", "response_id"):
            index_of.setdefault(int(element[key]), len(index_of))

    # if rawids is given, use it to index the matrix instead
    if rawids is not None:
        rawids = [int(rawid) for rawid in rawids]
        if len(set(rawids)) != len(rawids):
            msg = f"rawids {rawids} hold repeated channel ids"
            raise ValueError(msg)
        if set(rawids) != set(index_of):
            msg = (
                f"rawids {sorted(rawids)} are not the channel ids seen in the "
                f"elements {sorted(index_of)}"
            )
            raise ValueError(msg)
        index_of = {rawid: i for i, rawid in enumerate(rawids)}

    rawids = list(index_of)
    n_detectors = len(rawids)
    shape = (n_detectors, n_detectors)
    mu = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    sigma = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    status = {
        p: np.full(shape, FIT_STATUS["not_filled"], dtype=np.int8)
        for p in ("neg", "pos")
    }

    placed = set()
    settings = None
    disagreeing = []
    for element in fitted_elements:
        row = index_of[int(element["trigger_id"])]
        col = index_of[int(element["response_id"])]
        if (row, col) in placed:
            msg = (
                f"the pair (trigger {element['trigger_id']}, response "
                f"{element['response_id']}) is measured by more than one element"
            )
            raise ValueError(msg)
        placed.add((row, col))

        parameters = element.get("parameters") or {}
        element_settings = {
            key: parameters[key] for key in XTC_SETTING_KEYS if key in parameters
        }

        if settings is None:
            settings = element_settings
        elif element_settings != settings:
            disagreeing.append(
                (
                    element["trigger_id"],
                    element["response_id"],
                    _differing_keys(settings, element_settings),
                )
            )

        for polarity in ("neg", "pos"):
            element_status = int(element[f"{polarity}_status"])
            status[polarity][row, col] = element_status

            if element_status <= max_status:
                mu[polarity][row, col] = float(element[f"{polarity}_mu"])
                sigma[polarity][row, col] = float(element[f"{polarity}_sigma"])

    col_dict = {"rawid_index": lgdo.Array(np.asarray(rawids, dtype=np.int64))}
    for polarity in ("neg", "pos"):
        field = XTC_LH5_FIELD[polarity]
        col_dict[field] = lgdo.Array(mu[polarity])
        col_dict[f"{field}_sigma"] = lgdo.Array(sigma[polarity])
        col_dict[f"{field}_status"] = lgdo.Array(status[polarity])

    missing = n_detectors**2 - len(placed)
    if missing:
        log.warning(
            "%s of %s matrix elements were not measured, they stay NaN",
            missing,
            n_detectors**2,
        )

    parameters = {"max_status": max_status}
    if disagreeing:
        trigger_id, response_id, differing = disagreeing[0]
        msg = (
            f"{len(disagreeing)} of {len(fitted_elements)} elements were not "
            f"produced with the settings of the first one: the pair (trigger "
            f"{trigger_id}, response {response_id}) differs in {differing}.  "
            f"They are likely from different productions"
        )
        if require_same_parameters:
            raise ValueError(msg)
        log.warning("%s; the matrix records no settings of its own", msg)
        parameters["mixed_parameters"] = True
    elif settings:
        parameters.update(settings)

    return lgdo.Table(
        col_dict=col_dict,
        attrs={
            "fit_status_codes": json.dumps(FIT_STATUS),
            "parameters": json.dumps(parameters, default=str),
        },
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
