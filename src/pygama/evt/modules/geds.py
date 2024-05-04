"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

import awkward as ak
import numpy as np
from lgdo import lh5, types

from .. import utils
from . import xtalk

sto = lh5.LH5Store()


def apply_recovery_cut(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
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
    *,
    mode: str,
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    multiplicity_logic: str,
    threshold: float = None,
    xtalk_matrix_filename: str = "",
    xtalk_rawid_name: str = "xtc/rawid_index",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is applied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    uncalibrated_energy_name
        expression for the pulse parameter to be gathered, can be a combination of different fields.
    calibrated_energy_name
        name of the pulse parameter for calibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    xtalk_matrix_name
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_name
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids
    """

    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_name, xtalk_matrix_filename, "np")
    tcm_id_array = xtalk.build_tcm_id_array(tcm, datainfo, xtalk_matrix_rawids)

    energies_corr = xtalk.get_xtalk_correction(
        tcm,
        datainfo,
        uncalibrated_energy_name,
        calibrated_energy_name,
        threshold,
        xtalk_matrix_filename,
        xtalk_rawid_name,
        xtalk_matrix_name,
        positive_xtalk_matrix_name,
    )

    multiplicity_mask = xtalk.filter_hits(
        datainfo,
        tcm,
        multiplicity_logic,
        energies_corr,
        xtalk_matrix_rawids,
    )
    energies_corr = ak.from_regular(energies_corr)
    multiplicity_mask = ak.from_regular(multiplicity_mask)

    if mode == "energy":
        return types.VectorOfVectors(energies_corr[multiplicity_mask])
    elif mode == "tcm_id":
        return types.VectorOfVectors(tcm_id_array[multiplicity_mask])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_xtalk_correction_and_calibrate(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    mode: str,
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    par_files: str | list[str],
    multiplicity_logic: str,
    threshold: float = None,
    xtalk_matrix_filename: str = "",
    xtalk_rawid_name: str = "xtc/rawid_index",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
    out_param: str = None,
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is applied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    uncalibrated_energy_name
        expression for the pulse parameter to be gathered, can be a combination of different fields.
    calibrated_energy_name
        name of the pulse parameter for calibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
    threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    xtalk_matrix_name
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_name
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids
    """

    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_name, xtalk_matrix_filename, "np")
    tcm_id_array = xtalk.build_tcm_id_array(tcm, datainfo, xtalk_matrix_rawids)

    energies_corr = xtalk.get_xtalk_correction(
        tcm,
        datainfo,
        uncalibrated_energy_name,
        calibrated_energy_name,
        threshold,
        xtalk_matrix_filename,
        xtalk_rawid_name,
        xtalk_matrix_name,
        positive_xtalk_matrix_name,
    )

    if out_param is None:
        out_param = calibrated_energy_name.split(".")[-1]

    calibrated_corr = xtalk.calibrate_energy(
        datainfo,
        tcm,
        energies_corr,
        xtalk_matrix_rawids,
        par_files,
        uncalibrated_energy_name,
        out_param,
    )

    multiplicity_mask = xtalk.filter_hits(
        datainfo,
        tcm,
        multiplicity_logic,
        calibrated_corr,
        xtalk_matrix_rawids,
    )

    calibrated_corr = ak.from_regular(calibrated_corr)
    multiplicity_mask = ak.from_regular(multiplicity_mask)

    if mode == "energy":
        return types.VectorOfVectors(calibrated_corr[multiplicity_mask])
    elif mode == "tcm_id":
        return types.VectorOfVectors(tcm_id_array[multiplicity_mask])
    else:
        raise ValueError(f"Unknown mode: {mode}")
