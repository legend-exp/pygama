"""Event processors for HPGe data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from lgdo import lh5, types

from .. import utils


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
    energy_observable: types.VectorOfVectors,
    rawids: types.VectorOfVectors,
    xtalk_matrix_filename: str,
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.

    The format of `xtalk_matrix_filename` should be...

    Parameters
    ----------
    datainfo, tcm, table_names
        positional arguments automatically supplied by :func:`.build_evt`.
    energy_observable
        array of energy values to correct, one event per row. The detector
        identifier is stored in `rawids`, which has the same layout.
    rawids
        array of detector identifiers for each energy in `energy_observable`.
    xtalk_matrix_filename
        name of the file containing the cross-talk matrices.
    """
    # read in xtalk matrices
    lh5.read_as("", xtalk_matrix_filename, "ak")

    # do the correction
    energies_corr = ...

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(energy_observable)
    )
