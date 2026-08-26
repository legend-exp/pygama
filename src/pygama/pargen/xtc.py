"""
This module provides routines for measuring cross-talk (XTC) between
germanium channels and for building the resulting cross-talk matrix.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import lh5
import numpy as np

from pygama.pargen.xtc_utils import EventSelector, xtalk_element

log = logging.getLogger(__name__)

DEFAULT_ENERGY_PARAM = "cuspEmax_ctc_cal"
DEFAULT_POSITIVE_PARAM = "trapTmax"
DEFAULT_NEGATIVE_PARAM = "trapTmin"
DEFAULT_TRIGGER_PARAM = "trapTmax"
DEFAULT_TRIGGER_ENERGY_RANGE = (1500, 99999)
DEFAULT_RESPONSE_ENERGY_RANGE = (-99999, 100)
DEFAULT_NBINS = 700
DEFAULT_RANGE_MULTIPLIER = 3


def prepare_baseline(
    hit_files: str | list,
    dsp_files: str | list,
    chn_id: str | int,
    out_path: str | Path | None = None,
    config: dict | None = None,
    display: int = 0,
    debug_mode: bool = False,
) -> dict:
    """Measure the positive and negative baselines of a single channel.

    Selects baseline events in the hit tier -- those whose flag fields match
    ``config["baseline_conditions"]`` -- then averages the corresponding
    positive- and negative-going DSP amplitudes over exactly those events.

    The result is written to *out_path* as JSON and also returned, so a
    caller that keeps the value in memory need not read it back.

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
    out_path
        JSON file to write the result to; parent directories are created.
        ``None`` computes the result and returns it without writing.
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
    display
        If greater than zero, write a before/after selection histogram for
        each DSP parameter next to *out_path*.  Ignored when *out_path* is
        ``None``.
    debug_mode
        If True, re-raise instead of falling back to a nan result.

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

    out_path = Path(out_path) if out_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    success = True
    positive_baseline = np.nan
    negative_baseline = np.nan
    positive_selection = None
    negative_selection = None

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
        positive_baseline = np.nan
        negative_baseline = np.nan

    # display plots run outside the guarding path so a plotting failure cannot
    # turn a good measurement into a nan result
    if display > 0 and success:
        if out_path is None:
            log.warning(
                "prepare_baseline: display=%s ignored, no out_path to write plots next to",
                display,
            )
        else:
            for selection, param in (
                (positive_selection, positive_param),
                (negative_selection, negative_param),
            ):
                try:
                    selection.draw(out_path.with_name(f"{out_path.stem}_{param}.png"))
                except Exception as e:
                    log.debug("prepare_baseline: %s display plot failed: %s", param, e)

    result = {
        "detector_id": chn_id,
        "positive_baseline": (
            None if np.isnan(positive_baseline) else positive_baseline
        ),
        "negative_baseline": (
            None if np.isnan(negative_baseline) else negative_baseline
        ),
        "success": success,
        "processed_at": datetime.now().isoformat(),
        "parameters": {
            "baseline_conditions": conditions,
            "energy_param": energy_param,
            "positive_param": positive_param,
            "negative_param": negative_param,
            "n_hit_files": 1 if isinstance(hit_files, str) else len(hit_files),
            "n_dsp_files": 1 if isinstance(dsp_files, str) else len(dsp_files),
        },
    }

    if out_path is not None:
        out_path.write_text(json.dumps(result, indent=2))
        log.info("baseline of channel %s written to %s", chn_id, out_path)

    return result


# merge baseline?


def _resolve_baseline(baseline: dict, chn_id: str | int) -> tuple[float, float] | None:
    """Return ``(positive, negative)`` for *chn_id*, or None if unusable.

    A channel is unusable when it is absent from *baseline* or either of its
    two values is ``None`` or NaN -- the replacement for the old
    ``skipped_channels.npy``.  Both ``int`` and ``str`` keys are accepted,
    since a *baseline* round-tripped through JSON has string keys.
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

    Returns ``None`` when the sample is empty or has no usable spread, in
    which case the caller records an empty histogram for the element.
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
    out_path: str | Path | None = None,
    debug_mode: bool = False,
) -> dict:
    """Fill one column of the cross-talk matrix.

    Selects the events in which *trigger_detector_id* fired, then, for every
    channel in *chn_id_list*, measures the amplitude that channel shows in
    exactly those events and expresses it as a percentage of the trigger
    energy (see :func:`.xtc_utils.xtalk_element`).  The distribution of that
    percentage over the selected events is histogrammed, once against the
    positive-going and once against the negative-going response parameter.

    The trigger selection and the trigger energies are read once and reused
    for the whole column; this is why the column, rather than the single
    matrix element, is the unit of work.

    Elements are skipped -- recorded with ``valid = False`` and an empty
    histogram -- when the response channel is the trigger itself, or when
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
        Channel identifier (rawid) of the trigger detector, without the
        ``ch`` prefix.
    chn_id_list
        Response channel identifiers, in the order their histograms are
        written.  Pass the full detector list for a whole column, or a slice
        of it to split a column across several jobs.
    baseline
        Per-channel baselines, as produced by :func:`prepare_baseline` and
        collected by channel id::

            {
                1104000: {"positive_baseline": 5.0, "negative_baseline": -3.0},
                1104001: {"positive_baseline": None, "negative_baseline": None},
                ...
            }

        Extra keys in each entry (``success``, ``processed_at``, ...) are
        ignored, so the merged output of :func:`prepare_baseline` can be
        passed straight through.  Keys may be ``int`` or ``str``.
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
    out_path
        ``.npz`` file to write the column to; parent directories are
        created.  ``None`` computes the result and returns it without
        writing.
    debug_mode
        If True, re-raise instead of falling back to an empty column or an
        empty element.

    Returns
    -------
    dict
        ``trigger_id``, ``response_ids`` ``(N,)``, ``valid`` ``(N,)`` bool,
        ``n_events`` ``(N,)``, ``neg_counts``/``pos_counts`` ``(N, nbins)``,
        ``neg_bins``/``pos_bins`` ``(N, nbins + 1)`` and ``parameters``.
        Bins are NaN wherever the histogram is empty.  In the ``.npz`` file
        ``parameters`` is stored as a JSON string.
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

    out_path = Path(out_path) if out_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

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
    # skip to saving an empty column. 
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

                neg_vals = np.asarray(
                    xtalk_element(
                        trigger_energies,
                        response_table[negative_param].nda,
                        negative_baseline,
                    )
                )
                pos_vals = np.asarray(
                    xtalk_element(
                        trigger_energies,
                        response_table[positive_param].nda,
                        positive_baseline,
                    )
                )
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
        "n_hit_files": 1 if isinstance(hit_files, str) else len(hit_files),
        "n_dsp_files": 1 if isinstance(dsp_files, str) else len(dsp_files),
    }

    result = {
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

    if out_path is not None:
        np.savez_compressed(
            out_path,
            **{k: v for k, v in result.items() if k != "parameters"},
            parameters=json.dumps(parameters),
        )
        log.info(
            "xtalk column of trigger %s (%s/%s elements filled) written to %s",
            trigger_detector_id,
            int(valid.sum()),
            n_response,
            out_path,
        )

    return result


# Or maybe baseline should be two separate arguments?
# Or maybe baseline is a file path?


def xtalk_histogram_fitter(
    histogram_file: str | Path, config: dict | None, out_path: str | Path
) -> None:
    raise NotImplementedError()


def build_xtalk_matrix(
    chn_id_list: list, fitted_files: list, out_path: str | Path, config: dict | None
) -> None:
    raise NotImplementedError()
