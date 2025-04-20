"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

import awkward as ak
import numpy as np
from lgdo import lh5, types

from .. import utils
from . import xtalk


def apply_recovery_cut(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    channel_mapping: dict,
    *,
    timestamps: types.Array,
    flag: types.Array,
    time_window: float,
) -> types.Array:

    discharge_timestamps = timestamps.nda[flag.nda == 1]
    is_recovering = np.full(len(timestamps.nda), False)
    for tstamp in discharge_timestamps:
        is_recovering = is_recovering | np.where(
            (
                ((timestamps.nda - tstamp) < time_window)
                & ((timestamps.nda - tstamp) >= 0)
            ),
            True,
            False,
        )

    # return the result as LGDO
    return types.Array(is_recovering)


def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    channel_mapping: dict,
    *,
    return_mode: str,
    uncal_energy_expr: str,
    cal_energy_expr: str,
    multiplicity_expr: str,
    xtalk_threshold: float = None,
    xtalk_matrix_filename: str = "",
    xtalk_rawid_obj: str = "xtc/rawid_index",
    xtalk_matrix_obj: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_obj: str = "xtc/xtalk_matrix_positive",
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is applied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    return_mode
        string which can be either energy to return corrected energy or tcm_index
    uncal_energy_expr
        expression for the pulse parameter to be gathered for the uncalibrated energy (used for correction),
        can be a combination of different fields.
    cal_energy_expr
        expression for the pulse parameter to be gathered for the calibrated energy, used for the xtalk threshold,
        can be a combination of different fields.
    xtalk_threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    xtalk_matrix_obj
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_obj
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_rawids_obj
        name of the lh5 object containing the name of the rawids
    """

    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_obj, xtalk_matrix_filename, "np")
    tcm_index_array = xtalk.build_tcm_index_array(tcm, datainfo, xtalk_matrix_rawids)

    energy_corr = xtalk.get_xtalk_correction(
        tcm,
        datainfo,
        uncal_energy_expr,
        cal_energy_expr,
        xtalk_threshold,
        xtalk_matrix_filename,
        xtalk_rawid_obj,
        xtalk_matrix_obj,
        positive_xtalk_matrix_obj,
    )

    multiplicity_mask = xtalk.filter_hits(
        datainfo,
        tcm,
        multiplicity_expr,
        energy_corr,
        xtalk_matrix_rawids,
    )
    energy_corr = ak.from_regular(energy_corr)
    multiplicity_mask = ak.from_regular(multiplicity_mask)
    tcm_index_array = ak.from_regular(tcm_index_array)

    if return_mode == "energy":
        return types.VectorOfVectors(energy_corr[multiplicity_mask])
    elif return_mode == "tcm_index":
        return types.VectorOfVectors(tcm_index_array[multiplicity_mask])
    else:
        raise ValueError(f"Unknown mode: {return_mode}")


def apply_xtalk_correction_and_calibrate(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    channel_mapping: dict,
    *,
    return_mode: str,
    uncal_energy_expr: str,
    cal_energy_expr: str,
    cal_par_files: str | Sequence[str],
    multiplicity_expr: str,
    xtalk_matrix_filename: str,
    xtalk_threshold: float = None,
    xtalk_rawid_obj: str = "xtc/rawid_index",
    xtalk_matrix_obj: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_obj: str = "xtc/xtalk_matrix_positive",
    uncal_var: str = "dsp.cuspEmax",
    recal_var: str = "hit.cuspEmax_ctc_cal",
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.

    The correction is applied using matrix algebra for all triggers above the
    xalk threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    return_mode
        string which can be either ``energy`` to return corrected energy or
        ``tcm_index``.
    uncal_energy_expr
        expression for the pulse parameter to be gathered for the uncalibrated
        energy (used for correction), can be a combination of different fields.
    cal_energy_expr
        expression for the pulse parameter to be gathered for the calibrated
        energy, used for the xtalk threshold, can be a combination of different
        fields.
    cal_par_files
        path to the generated hit tier par-files defining the calibration
        curves. Used to recalibrate the data after xtalk correction.
    multiplicity_expr:
        expression defining the logic used to compute the event multiplicity.
    xtalk_threshold
        threshold used for xtalk correction, hits below this energy will not be
        used to correct the other hits.
    xtalk_matrix_filename
        path to the file containing the xtalk matrices.
    xtalk_matrix_obj
        name of the lh5 object containing the xtalk matrix.
    positive_xtalk_matrix_obj
        name of the lh5 object containing the positive polarity xtalk matrix.
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids.
    recal_var
        name of the energy variable to use for re-calibration.
    """

    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_obj, xtalk_matrix_filename, "np")
    tcm_index_array = xtalk.build_tcm_index_array(tcm, datainfo, xtalk_matrix_rawids)

    energy_corr = xtalk.get_xtalk_correction(
        tcm,
        datainfo,
        uncal_energy_expr,
        cal_energy_expr,
        xtalk_threshold,
        xtalk_matrix_filename,
        xtalk_rawid_obj,
        xtalk_matrix_obj,
        positive_xtalk_matrix_obj,
    )

    calibrated_corr = xtalk.calibrate_energy(
        datainfo,
        tcm,
        energy_corr,
        xtalk_matrix_rawids,
        cal_par_files,
        uncal_var,
        recal_var,
        channel_mapping,
    )

    multiplicity_mask = xtalk.filter_hits(
        datainfo,
        tcm,
        multiplicity_expr,
        calibrated_corr,
        xtalk_matrix_rawids,
    )

    calibrated_corr = ak.from_regular(calibrated_corr)
    multiplicity_mask = ak.from_regular(multiplicity_mask)
    tcm_index_array = ak.from_regular(tcm_index_array)

    if return_mode == "energy":
        return types.VectorOfVectors(calibrated_corr[multiplicity_mask])
    elif return_mode == "tcm_index":
        return types.VectorOfVectors(tcm_index_array[multiplicity_mask])
    else:
        raise ValueError(f"Unknown mode: {return_mode}")
