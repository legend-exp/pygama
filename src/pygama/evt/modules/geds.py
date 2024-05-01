"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from lgdo import lh5, types

from .. import utils
from . import xtalk


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
                & ((timestamps.nda - tstamp) > 0)
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
    uncalibrated_energy_name: str,
    calibrated_energy_name: str,
    threshold: float = None,
    xtalk_matrix_filename: str,
    xtalk_rawid_name: str = "xtc/xtalk_rawids",
    xtalk_matrix_name: str = "xtc/xtalk_matrix_negative",
    positive_xtalk_matrix_name: str = "xtc/xtalk_matrix_positive",
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a lh5 file.

    The correction is appplied using matrix algebra for all triggers above the threshold.

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    uncalibrated_energy_name
        name of the pulse parameter for uncalibrated energy to be gathered, optionally prefixed by tier
        name (e.g. ``hit.cusp_Emax``). If no tier is specified, it defaults
        to ``hit``.
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

    # read lh5 files to numpy
    xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_name, xtalk_matrix_filename, "np")
    xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_name, xtalk_matrix_filename, "np")
    positive_xtalk_matrix_numpy = lh5.read_as(
        positive_xtalk_matrix_name, xtalk_matrix_filename, "np"
    )

    # Combine positive and negative matrixs
    # Now the matrix should have negative values corresponding to negative cross talk
    # and positive values corresponding to positive cross talk .
    # we also set nan to 0 and we transpose so that the row corresponds to response and column trigger
    xtalk_matrix = np.nan_to_num(
        np.where(
            abs(xtalk_matrix_numpy) > abs(positive_xtalk_matrix_numpy),
            xtalk_matrix_numpy,
            positive_xtalk_matrix_numpy,
        ),
        0,
    ).T

    uncalibrated_energy_array = xtalk.build_energy_array(
        uncalibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )
    calibrated_energy_array = xtalk.build_energy_array(
        calibrated_energy_name, tcm, datainfo, xtalk_matrix_rawids
    )

    energies_corr = xtalk.xtalk_corrected_energy(
        uncalibrated_energy_array, calibrated_energy_array, xtalk_matrix, threshold
    )

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(uncalibrated_energy_name)
    )
