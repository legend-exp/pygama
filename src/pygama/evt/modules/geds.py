"""Event processors for HPGe data."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from lgdo import lh5, types

from .. import utils
from . import xtalk


def apply_xtalk_correction(
    datainfo: utils.DataInfo,
    tcm: utils.TCMData,
    table_names: Sequence[str],
    *,
    energy_observable: types.VectorOfVectors,
    rawids: types.VectorOfVectors,
    threshold: float = None,
    xtalk_matrix_filename: str,
    positive_xtalk_matrix_filename: str = None,
    xtalk_rawid_name: str = "xtalk_rawids",
    xtalk_matrix_name: str = "xtalk_matrix",
    positive_xtalk_matrix_name: str = None,
    is_json_input_file: bool = True,
    det_names: bool = False,
    energy_observable_negative: types.VectorOfVectors = None,
    use_slow_correction: bool = True,
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
    threshold
        threshold used for xtalk correction, hits below this energy will not
        be used to correct the other hits.
    xtalk_matrix_filename
        name of the file containing the xtalk matrices.
    positive_xtalk_matrix_filename
        name of the file containing the positive xtalk matrices.
    xtalk_matrix_name
        name of the lh5 object containing the xtalk matrix
    positive_xtalk_matrix_name
        name of the lh5 object containing the positive polarity xtalk matrix
    xtalk_matrix_rawids
        name of the lh5 object containing the name of the rawids
    is_json_input_file
        boolean to say the xtalk matrix is stored as json file
    det_names
        bool to say the xtalk matrices use the detector names (default False)
    energy_observable_negative
        array of negative energy values to correct, one event per row. The detector
        identifier is stored in `rawids`, which has the same layout. Default None
    use_slow_correction
        boolean flag to use the slow xtalk correction (with loops)
    """
    # read json file to a dict
    if is_json_input_file:
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

    else:
        # read lh5 files to numpy
        xtalk_matrix_numpy = lh5.read_as(xtalk_matrix_name, xtalk_matrix_filename, "np")
        xtalk_matrix_rawids = lh5.read_as(xtalk_rawid_name, xtalk_matrix_filename, "np")

        if positive_xtalk_matrix_filename is not None:
            positive_xtalk_matrix_numpy = lh5.read_as(
                positive_xtalk_matrix_name, positive_xtalk_matrix_filename, "np"
            )
            positive_xtalk_matrix_rawids = lh5.read_as(
                xtalk_rawid_name, positive_xtalk_matrix_filename, "np"
            )

            if np.any(xtalk_matrix_rawids != positive_xtalk_matrix_rawids):
                raise ValueError(
                    "Positive and Negative cross talk matrix must have the same rawids"
                )

    if use_slow_correction:
        xtalk_matrix = xtalk.numpy_to_dict(xtalk_matrix_numpy, xtalk_matrix_rawids)

        if positive_xtalk_matrix_filename is not None:
            positive_xtalk_matrix = xtalk.numpy_to_dict(
                positive_xtalk_matrix_numpy, positive_xtalk_matrix_rawids
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
    else:
        raise NotImplementedError(
            "Error fast (matrix based ) xtalk correction isnt implemented yet"
        )

    # return the result as LGDO
    return types.VectorOfVectors(
        energies_corr, attrs=utils.copy_lgdo_attrs(energy_observable)
    )
