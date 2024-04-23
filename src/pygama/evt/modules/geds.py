"""Event processors for HPGe data."""

from __future__ import annotations

import json
from collections.abc import Sequence

from lgdo import types

from .. import utils
from . import xtalk


def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    energy_observable: types.VectorOfVectors,
    rawids: types.VectorOfVectors,
    xtalk_matrix_filename: str,
    threshold: float,
    det_names: bool = False,
    energy_observable_negative: types.VectorOfVectors = None,
    positive_xtalk_matrix_filename: str = None,
) -> types.VectorOfVectors:
    """Applies the cross-talk correction to the energy observable.
    The format of `xtalk_matrix_filename` should be currently be a path to a JSON file.

    The correction is appplied recursively, the energies are sorted and then the correction
    is applied from the largest to smallest (above the threshold).

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
    threshold
        threshold used for cross talk correction, hits below this energy will not
        be used to correct the other hits.
    det_names
        bool to say the x-talk matrices use the detector names (default False)
    energy_observable_negative
        array of negative energy values to correct, one event per row. The detector
        identifier is stored in `rawids`, which has the same layout. Default None
    positive_xtalk_matrix_filename
        name of the file containing the positive cross-talk matrices.

    """
    try:
        with open(xtalk_matrix_filename) as file:
            xtalk_matrix = json.load(file)
    except FileNotFoundError:
        raise ValueError(
            f"path to x-talk matrix {xtalk_matrix_filename} does not exist"
        )

    positive_xtalk_matrix = None
    if positive_xtalk_matrix_filename is not None:
        try:
            with open(xtalk_matrix_filename) as file:
                positive_xtalk_matrix = json.load(file)
        except FileNotFoundError:
            raise ValueError(
                f"path to x-talk matrix {positive_xtalk_matrix_filename} does not exist"
            )

    xtalk_matrix = xtalk.manipulate_xtalk_matrix(
        xtalk_matrix, positive_xtalk_matrix, det_names
    )

    # do the correction
    energies_corr = xtalk.xtalk_corrected_energy_awkard_slow(
        energies=energy_observable.view_as("ak"),
        rawids=rawids.view_as("ak"),
        matrix=xtalk_matrix,
        allow_non_existing=False,
        threshold=threshold,
    )

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(energy_observable)
    )
