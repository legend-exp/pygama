"""
This module provides routines for measuring cross-talk (XTC) between
germanium channels and for building the resulting cross-talk matrix.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from pygama.pargen.xtc_utils import EventSelector

log = logging.getLogger(__name__)

DEFAULT_ENERGY_PARAM = "cuspEmax_ctc_cal"
DEFAULT_POSITIVE_PARAM = "trapTmax"
DEFAULT_NEGATIVE_PARAM = "trapTmin"


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
            msg = f"baseline selection on {energy_param} failed"
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
            msg = f"reading {positive_param}/{negative_param} on the selected events failed"
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
        log.error("baseline preparation failed for channel %s: %s", chn_id, e)
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


def xtalk_column(
    hit_files: str | list,
    dsp_files: str | list,
    trigger_detector_id: str,
    chn_id_list: list,
    baseline: dict,
    out_path: str | Path,
) -> None:
    raise NotImplementedError()


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
